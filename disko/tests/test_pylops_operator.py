#
# Copyright Tim Molteno 2017 tim@elec.ac.nz
#

import unittest
import logging
import json

import numpy as np

import pylops

from disko import DiSkO
import disko
from disko import HealpixFoV, HealpixSubFoV, AdaptiveMeshFoV, Resolution
import astropy.constants as const

from tart.operation import settings
from tart_tools import api_imaging

logger = logging.getLogger(__name__)
# Add a null handler so logs can go somewhere
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


def dottest(Op, nr, nc, tol):

    pylops.utils.dottest(Op, nr, nc, rtol=1e-06,
                         complexflag=0, raiseerror=True, verb=True)

    u = np.random.randn(nc)  # random sky
    v = np.random.randn(nr)  # random vis

    logger.info("u = {}".format(u))
    logger.info("v = {}".format(v))

    y = Op.matvec(u)   # Op * u
    x = Op.rmatvec(v)  # Op'* v

    logger.info("x = {}".format(x))
    logger.info("y = {}".format(y))

    yy = np.dot(y, v)  # (Op  * u)' * v
    xx = np.dot(u, x)  # u' * (Op' * v)

    err = np.abs((yy-xx)/((yy+xx+1e-15)/2))
    if err < tol:
        logger.info(
            'Dot test passed, v^T(Opu)={} - u^T(Op^Tv)={}, err={}'.format(yy, xx, err))
        return True
    else:
        raise ValueError(
            'Dot test failed, v^T(Opu)={} - u^T(Op^Tv)={}, err={}'.format(yy, xx, err))


class TestPylopsOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load data from a JSON file
        fname = 'test_data/test_data.json'
        logger.info("Getting Data from file: {}".format(fname))
        with open(fname, 'r') as json_file:
            calib_info = json.load(json_file)

        info = calib_info['info']
        cls.ant_pos = np.array(calib_info['ant_pos'])
        config = settings.from_api_json(info['info'], cls.ant_pos)

        flag_list = []

        gains_json = calib_info['gains']
        gains = np.asarray(gains_json['gain'])
        phase_offsets = np.asarray(gains_json['phase_offset'])

        for d in calib_info['data']:
            vis_json, source_json = d
            cv, _timestamp = api_imaging.vis_calibrated(
                vis_json, config, gains, phase_offsets, flag_list)

        cls.disko = DiSkO.from_cal_vis(cv)
        cls.nside = 16
        cls.sphere = HealpixFoV(cls.nside)
        res_deg = 4.0
        cls.subsphere = HealpixSubFoV.from_resolution(res_arcmin=res_deg*60.0,
                                                         theta=np.radians(0.0),
                                                         phi=0.0,
                                                         radius_rad=np.radians(89))

        cls.adaptive_sphere = AdaptiveMeshFoV.from_resolution(res_min=Resolution.from_arcmin(20),
                                                                 res_max=Resolution.from_deg(
                                                                     res_deg),
                                                                 theta=np.radians(0.0), phi=0.0,
                                                                 fov=Resolution.from_deg(10))

        cls.gamma = cls.disko.make_gamma(cls.sphere)
        cls.subgamma = cls.disko.make_gamma(cls.subsphere)

    def test_pylops_dot(self):
        r'''
            Test using the build-in pylops tester for new operators
        '''
        data = self.disko.vis_to_data()
        frequencies = [self.disko.frequency]

        Op = disko.DiSkOOperator(self.disko.u_arr,
                                 self.disko.v_arr,
                                 self.disko.w_arr,
                                 data, frequencies, self.sphere)
        # Test that we have the same effect as matrix vector multiply

        sky = np.random.normal(0, 1, self.sphere.npix)

        vis1 = self.gamma @ sky

        vis2 = Op @ sky  # Op.matvec(sky)
        logger.info(f"vis1: {vis1[0:10]}")
        logger.info(f"vis2: {vis2[0:10]}")

        self.assertEqual(vis1.shape, vis2.shape)
        self.assertTrue(np.allclose(vis1, vis2))

        dottest(Op, self.disko.n_v*2, self.sphere.npix, tol=1e-04)

        Op = disko.DirectImagingOperator(self.disko.u_arr,
                                         self.disko.v_arr,
                                         self.disko.w_arr,
                                         data, frequencies, self.sphere)
        pylops.utils.dottest(Op, self.sphere.npix, self.disko.n_v*2,
                             rtol=1e-06, complexflag=0,
                             raiseerror=True, verb=True)

    def test_pylops_tiny(self):
        r'''
            Test such a small gamma that we can inspect every element and
            check that the matrix is what we expect it to be.
        '''
        tiny_subsphere = HealpixSubFoV.from_resolution(res_arcmin=3600,
                                                          theta=np.radians(
                                                              0.0),
                                                          phi=0.0,
                                                          radius_rad=np.radians(80))
        self.assertEqual(tiny_subsphere.npix, 4)

        frequencies = [1.5e9]
        wavelength = const.c.value / frequencies[0]

        n_vis = 3
        u = np.random.uniform(0, 1, n_vis)
        v = np.random.uniform(0, 1, n_vis)
        w = np.random.uniform(0, 1, n_vis)
        tiny_disko = DiSkO(u, v, w, frequencies[0])

        tiny_gamma = tiny_disko.make_gamma(tiny_subsphere)
        logger.info("Gamma={}".format(tiny_gamma))

        data = tiny_disko.vis_to_data(np.random.normal(0, 1, tiny_disko.n_v) +
                                      1.0j*np.random.normal(0, 1, tiny_disko.n_v))
        p2j = 2*np.pi*1.0j / wavelength

        Op = disko.DiSkOOperator(tiny_disko.u_arr, tiny_disko.v_arr,
                                 tiny_disko.w_arr, data, frequencies, tiny_subsphere)

        logger.info("Op Matrix")
        for i in range(Op.M):
            col = [Op.A(i, j, p2j) for j in range(Op.N)]
            logger.info(col)

        logger.info("Op Matrix Ajoint")
        for i in range(Op.N):
            col = [Op.Ah(i, j, p2j) for j in range(Op.M)]
            logger.info(col)

        for i in range(Op.M):
            for j in range(Op.N):
                self.assertAlmostEqual(Op.A(i, j, p2j), tiny_gamma[i, j])

        for i in range(Op.N):
            for j in range(Op.M):
                self.assertAlmostEqual(Op.Ah(i, j, p2j), tiny_gamma[j, i])

        dottest(Op, Op.M, Op.N, 1e-6)

        sky = np.random.normal(0, 1, tiny_subsphere.npix)
        logger.info("sky={}".format(sky))
        vis1 = tiny_gamma @ sky
        vis2 = Op.matvec(sky)
        logger.info(vis1)
        logger.info(vis2)

        self.assertEqual(vis1.shape, vis2.shape)
        self.assertTrue(np.allclose(vis1, vis2))
