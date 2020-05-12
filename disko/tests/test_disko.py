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
from disko import HealpixSphere, HealpixSubSphere, AdaptiveMeshSphere

from tart.operation import settings
from tart_tools import api_imaging
from tart.imaging import elaz
from tart.util import constants

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)

class TestDiSkO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load data from a JSON file
        fname = 'test_data/test_data.json'
        logger.info("Getting Data from file: {}".format(fname))
        with open(fname, 'r') as json_file:
            calib_info = json.load(json_file)

        info = calib_info['info']
        cls.ant_pos = calib_info['ant_pos']
        config = settings.from_api_json(info['info'], cls.ant_pos)

        flag_list = []

        gains_json = calib_info['gains']
        gains = np.asarray(gains_json['gain'])
        phase_offsets = np.asarray(gains_json['phase_offset'])
        #config = settings.from_api_json(info['info'], cls.ant_pos)

        measurements = []
        for d in calib_info['data']:
            vis_json, source_json = d
            cv, _timestamp = api_imaging.vis_calibrated(vis_json, config, gains, phase_offsets, flag_list)

        cls.disko = DiSkO.from_cal_vis(cv)
        cls.nside = 16
        cls.sphere = HealpixSphere(cls.nside)
        res_deg = 4.0
        cls.subsphere = HealpixSubSphere.from_resolution(resolution=res_deg*60.0, 
                                      theta = np.radians(0.0), phi=0.0, radius=np.radians(89))
        
        cls.adaptive_sphere = AdaptiveMeshSphere.from_resolution(res_arcmin=20, res_arcmax=res_deg*60, 
                                                         theta=np.radians(0.0), 
                                                         phi=0.0, radius=np.radians(10))

        cls.gamma = cls.disko.make_gamma(cls.sphere)
        cls.subgamma = cls.disko.make_gamma(cls.subsphere)


    def get_harmonic_sky(self, h_index):
        harmonics = self.disko.get_harmonics(self.sphere)
        sky = np.zeros_like(self.sphere.pixels, dtype=np.complex128)
        sky = sky + harmonics[h_index]
        sky = sky.reshape([-1, 1])
        return sky, harmonics

    def get_point_sky(self):
        sky = np.zeros_like(self.sphere.pixels, dtype=np.complex128)
        sky[-1] = 1.0
        sky = sky.reshape([-1, 1])
        return sky

    def test_harmonics_normalized(self):
        ### Check the harmonics are normalized.

        harmonics = self.disko.get_harmonics(self.sphere)
        for h_i in harmonics:
            dot = h_i @ h_i.conj().T
            self.assertAlmostEqual(dot, 1.0)
  
    @unittest.skip("Should Fail as the adaptive mesh harmonics dont work")
    def test_adaptive_harmonics_normalized(self):
        ### Check the harmonics are normalized.
        harmonics = self.disko.get_harmonics(self.adaptive_sphere)
        for h_i in harmonics:
            dot = h_i @ h_i.conj().T
            self.assertAlmostEqual(dot, 1.0)
  
    def test_vis_from_harmonic_sky(self):
        sky, harmonics = self.get_harmonic_sky(0)

        vis = harmonics[0] @ sky.conj()
        vis2 = self.gamma @ sky
        logger.info("vis = {}".format(vis))
        logger.info("vis2 = {}".format(vis2.shape))

        self.assertAlmostEqual(np.real(vis[0]), 1.0)
        self.assertAlmostEqual(np.imag(vis[0]), 0.0)

        self.assertAlmostEqual(np.real(vis2[0, 0]), 1.0)
        self.assertAlmostEqual(np.imag(vis2[0, 0]), 0.0)

    def test_vis(self):
        ''' Check that the effect of multiplication by gamma is the same as 
            inner product with harmonics
        '''
        sky, harmonics = self.get_harmonic_sky(10)
        vis = [h @ sky.conj() for h in harmonics]
        vis2 = self.gamma @ sky
        
        self.assertEqual(harmonics[0].shape[0], self.sphere.npix)

        for a, b in zip(vis, vis2):
            self.assertAlmostEqual(a[0], b[0])

    def test_from_pos(self):
        ''' Check that the DiSkO calculated from ant_pos only agrees with that from the 
            calibrated vis
        '''
        dut = DiSkO.from_ant_pos(self.ant_pos, wavelength=constants.L1_WAVELENGTH)
        self.assertTrue(dut.n_v == self.disko.n_v)
        self.assertTrue(np.allclose(dut.u_arr, self.disko.u_arr))

        harmonics = self.disko.get_harmonics(self.sphere)
        harmonics1 = dut.get_harmonics(self.sphere)
        for a, b in zip(harmonics, harmonics1):
            self.assertTrue(np.allclose(a, b))

    def test_solve_vis(self):
        sky1 = self.disko.solve_vis(self.disko.vis_arr, self.sphere, scale=True)
        sky2 = self.disko.solve_vis(self.disko.vis_arr, self.subsphere, scale=True)
        self.assertEqual(sky1.shape[0], 3072)
        self.assertEqual(sky2.shape[0], 1504)

    def test_matrix_free(self):
        
        ## Generate fake data with a frequency axis and an npol axis.
        data = np.zeros((self.disko.n_v, 1, 1), dtype=np.complex128)
        data[:,0,0] = self.disko.vis_arr
        sky = self.disko.solve_matrix_free(data, self.subsphere, alpha=0.0, scale=False)
        self.assertEqual(sky.shape[0], 1504)
        
        # Check that sky is a solution
        vis = self.subgamma @ sky
        
        self.assertEqual(vis[:,0].shape, self.disko.vis_arr.shape)
        self.assertTrue(np.allclose(vis[:,0], self.disko.vis_arr, atol=1e-6))

    def test_dot_matrix_free(self):
        r'''
            Test using the build-in pylops tester for new operators
        '''
        data = np.zeros((self.disko.n_v, 1, 1), dtype=np.complex128)
        data[:,0,0] = self.disko.vis_arr
        frequencies = [self.disko.frequency]

        Op = disko.DiSkOOperator(self.disko.u_arr, 
                                 self.disko.v_arr,
                                 self.disko.w_arr, 
                                 data, frequencies, self.sphere)
        # Test that we have the same effect as matrix vector multiply
        
        sky = np.random.normal(0,1, self.sphere.npix)

        vis1 = self.gamma @ sky

        vis2 = Op.matvec(sky)
        logger.info(vis1[0:10])
        logger.info(vis2[0:10])
        
        self.assertEqual(vis1.shape, vis2.shape)
        self.assertTrue(np.allclose(vis1, vis2))

        pylops.utils.dottest(Op, self.disko.n_v, self.sphere.npix, tol=1e-06, complexflag=2, raiseerror=True, verb=True)
        pylops.utils.dottest(Op, self.disko.n_v, self.sphere.npix, tol=1e-06, complexflag=3, raiseerror=True, verb=True)
    
    def test_tiny_gamma(self):
        r'''
            Test such a small gamma that we can inspect every element and 
            check that the matrix is what we expect it to be.
            
            
        '''
        tiny_subsphere = HealpixSubSphere.from_resolution(resolution=60*60.0, 
                                      theta = np.radians(0.0), phi=0.0, radius=np.radians(80))
        self.assertEqual(tiny_subsphere.npix, 4)

        frequencies = [1.5e9]
        wavelength = 2.99793e8 / frequencies[0]
        
        n_vis = 3
        u = np.random.uniform(0,1, n_vis) 
        v = np.random.uniform(0,1, n_vis)
        w = np.random.uniform(0,1, n_vis)
        tiny_disko = DiSkO(u,v,w, frequencies[0])
        
        tiny_gamma = tiny_disko.make_gamma(tiny_subsphere)
        logger.info("Gamma={}".format(tiny_gamma))
        
        data = np.zeros((tiny_disko.n_v, 1, 1), dtype=np.complex128)
        data[:,0,0] = np.random.normal(0,1,tiny_disko.n_v) + 1.0j*np.random.normal(0,1,tiny_disko.n_v)
        p2j = 2*np.pi*1.0j / wavelength

        Op = disko.DiSkOOperator(tiny_disko.u_arr, tiny_disko.v_arr, tiny_disko.w_arr, data, frequencies, tiny_subsphere)


        logger.info("Op Matrix")
        for i in range(Op.M):
            col = [Op.A(j, i, p2j ) for j in range(Op.N)]
            logger.info(col)
                
        sky = np.random.normal(0,1, tiny_subsphere.npix)
        logger.info("sky={}".format(sky))
        vis1 = tiny_gamma @ sky
        vis2 = Op.matvec(sky)
        logger.info(vis1)
        logger.info(vis2)
        
        self.assertEqual(vis1.shape, vis2.shape)
        self.assertTrue(np.allclose(vis1, vis2))

        
    def test_gamma_size(self):
        dut = DiSkO.from_ant_pos(self.ant_pos, wavelength=constants.L1_WAVELENGTH)
        gamma = dut.make_gamma(self.sphere)
        gamma_sub = dut.make_gamma(self.subsphere)
        self.assertEqual(gamma.shape[1], self.sphere.npix)
        self.assertEqual(gamma_sub.shape[1], self.subsphere.npix)

    def test_ml_sky(self):
        ### Create an image of a sky composed of multiples of the harmonics.
        ### Image that sky, and check that it products the correct visibilities.
        
        sky, harmonics = self.get_harmonic_sky(1)
        vis = self.gamma @ sky
        self.assertAlmostEqual(np.abs(vis[1, 0]), 1.0)
        imaged_sky = self.disko.solve_vis(vis, self.sphere)
        vis2 = self.gamma @ imaged_sky
        self.assertTrue(np.allclose(vis[:, 0], vis2[:, 0]))

    @unittest.skip("Should Fail as the point sky is not entirely in the range space")
    def test_imaging(self):
        ### Create an image of a sky. Calculate visibilities from that sky.
        ### Image those visabilities. Check that the skies are close.
        sky = self.get_point_sky()

        vis = self.gamma @ sky
        #logger.info("vis = {}".format(vis))

        #sphere, sky = self.disko.image_visibilities(vis, self.nside)
        sky2 = self.disko.solve_vis(vis, self.sphere)
        #sky_r = np.real(sky).reshape(sphere.pixels.shape)

        for i in range(self.sphere.npix):
            p2 = sky2[i]
            p1 = sky[i]
            d = np.abs(p2 - p1)
            if p1 > 0 or d > 3.0e-2:
                logger.info("{}: d={} {}, {}".format(i, d, p1, p2))
        self.assertTrue(np.allclose(sky, sky2))

    @unittest.skip("Should Fail as Direct DiSkO sucks")
    def test_solve_vs_direct(self):
        ### Create an image of a sky. Calculate visibilities from that sky.

        sky, _ = self.get_harmonic_sky(0)

        vis = self.gamma @ sky
        logger.info("vis = {}".format(vis))

        sky = sky.reshape([-1,])
        pixels1 = self.disko.image_visibilities(vis, self.sphere)
        pixels2 = self.disko.solve_vis(vis, self.sphere)

        #print(sphere1.pixels.shape)
        #print(sphere2.pixels.shape)
        #print(sky.shape)
        #print(np.where(np.abs(sky) > 0.1))
        #print(np.where(np.abs(sky - sphere1.pixels) > 0.1))
        #logger.info("image_vis {}".format(np.sum(np.abs(sky - sphere1.pixels) > 0.1)))
        #logger.info("solve_vis {}".format(np.sum(np.abs(sky - sphere2.pixels) > 0.1)))

        for p1, p2, s in zip(pixels1, pixels2, sky):
            logger.info("{} {}. {}".format(s, p1, p2))
            
        self.assertTrue(np.allclose(sky, pixels2))
