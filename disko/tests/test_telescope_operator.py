#
# Copyright Tim Molteno 2017-2019 tim@elec.ac.nz
#

import unittest
import logging
import json

import numpy as np

#from spotless import sphere
from disko import TelescopeOperator, HealpixSphere, DiSkO, normal_svd, dask_svd

from tart.operation import settings
from tart_tools import api_imaging
from tart.imaging import elaz


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)


class TestTelescopeOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load data from a JSON file
        np.seterr(all='raise')
        fname = 'test_data/test_data.json'
        logger.info("Getting Data from file: {}".format(fname))
        with open(fname, 'r') as json_file:
            calib_info = json.load(json_file)

        info = calib_info['info']
        ant_pos = calib_info['ant_pos']
        config = settings.from_api_json(info['info'], ant_pos)

        flag_list = []

        gains_json = calib_info['gains']
        gains = np.asarray(gains_json['gain'])
        phase_offsets = np.asarray(gains_json['phase_offset'])
        config = settings.from_api_json(info['info'], ant_pos)
    
        measurements = []
        for d in calib_info['data']:
            vis_json, source_json = d
            cv, timestamp = api_imaging.vis_calibrated(vis_json, config, gains, phase_offsets, flag_list)
            src_list = elaz.from_json(source_json, 0.0)

        cls.disko = DiSkO.from_cal_vis(cv)
        cls.nside = 16
        cls.sphere = HealpixSphere(cls.nside)
        cls.to = TelescopeOperator(cls.disko, cls.sphere)


    def get_point_sky(self):
        sky = np.zeros((self.to.n_s, 1))
        sky[1] = 1.0
        sky = sky.reshape([-1,1])
        return sky

    def test_svd(self):
        test = np.dot(self.to.U, np.dot(self.to.sigma, self.to.Vh))
        self.assertTrue(np.allclose(test, self.to.gamma))
        self.assertTrue(np.allclose(np.identity(self.to.n_s), np.dot(self.to.V, self.to.Vh)))

    def test_harmonics(self):
        ### Check the harmonics are normalized.
        n_h = self.to.n_v // 2
        for i in range(0,n_h):
            h_re = self.to.harmonic(i) 
            h_im = self.to.harmonic(i+n_h)
        
            h_i = h_re + 1.0j*h_im
            dot = h_i @ h_i.conj().T
            #dot = np.dot(h_re, h_re) + np.dot(h_im, h_im)
            self.assertAlmostEqual(dot, 1.0)
  

    def test_null_harmonics(self):
        ### Check the harmonics are normalized and are in the null space of Gamma.
        for i in range(0,self.to.n_n()):
            h_i = self.to.null_harmonic(i)
            dot = h_i @ h_i.conj().T
            self.assertAlmostEqual(dot, 1.0)
            
            # Check that it is in the null space. I.e. multiplication by Gamma returns zero.
            dut = self.to.gamma @ h_i
            self.assertTrue(np.allclose(np.abs(dut), np.zeros(self.to.n_v)))

    def test_range_harmonics(self):
        ### Check orthogonality of the harmonics.
        N = self.to.n_r()
        for i in range(0,N):
            h_i = self.to.range_harmonic(i)
            for j in range(0, N):
                h_j = self.to.range_harmonic(j)
                
                dot = h_i @ h_j.conj().T
                logger.info("natural dot = {}".format(dot))
                
                if (i == j):
                    self.assertAlmostEqual(dot, 1.0)
                else:
                    self.assertAlmostEqual(dot, 0.0)
  
    def test_sky_conversion(self):
        ### Check orthogonality of the harmonics.
        s = np.random.rand(self.to.n_s)
        x = self.to.sky_to_natural(s)
        s2 = self.to.natural_to_sky(x)
        self.assertTrue(np.allclose(s, s2))

    #def test_dask_vs_numpy_svd(self):
        #### Check orthogonality of the harmonics.
        #rows = 4 # 276
        #columns = 16 # 3072
        #x = np.random.rand(rows, 1)
        #y = np.random.rand(1, columns)
        #A = np.outer(x,y)
        #A = np.random.rand(rows, columns)
        
        #[u, s, vt], r = normal_svd(A)
        #[u1, s1, vt1], r1 = dask_svd(A)
        
        #self.assertEqual(r, r1)
        #logger.info("u = {}".format(u))
        #logger.info("u_dask = {}".format(u1))
        #logger.info("vT = {}".format(vt))
        #logger.info("vT_dask = {}".format(vt1))
        #self.assertTrue(np.allclose(s, s1))
        #self.assertTrue(np.allclose(np.abs(u), np.abs(u1)))
        #self.assertTrue(np.allclose(np.abs(vt), np.abs(vt1), atol=0.01))

    def test_vis(self):
        ### Check that v = A_r x_r is the same as Gamma s
        sky = self.get_point_sky()
        
        vis = self.to.gamma @ sky 
        #logger.info("vis = {}".format(vis[:,0]))

        x = self.to.sky_to_natural(sky)
        
        A = self.to.U @ self.to.sigma
        vis2 = A @ x
        #logger.info("vis2 = {}".format(vis2[:,0]))
        
        x_r = x[0:self.to.rank]
        vis3 = self.to.A_r @ x_r
        #logger.info("vis3 = {}".format(vis3[:,0]))
        
        for v1, v2, v3 in zip(vis[:,0], vis2[:,0], vis3[:,0]):
            logger.info("{},{},{}".format(v1, v2, v3))
            self.assertAlmostEqual(v1, v2, 6)
            
        self.assertTrue(np.allclose(vis[:,0], vis2[:,0]))
        self.assertTrue(np.allclose(vis2[:,0], vis3[:,0]))
       
    def test_vis_in_range(self):
        # Project a sky into the range space of Gamma^H and check that image_visibilities
        # Don't change        sky = np.zeros((self.to.n_s, 1))
        sky = self.get_point_sky()
        
        vis = self.to.gamma @ sky 
        #logger.info("vis = {}".format(vis))

        sky_r = self.to.P_r() @ sky 
        vis2 = self.to.gamma @ sky_r 
        #logger.info("vis2 = {}".format(vis2))

        self.assertTrue(np.allclose(vis, vis2))



    def test_imaging(self):
        ### Check that v = A_r x_r is the same as Gamma s
        sky = self.get_point_sky()
        # This sky contains null space components, so lets  project those out.
        vis_orig = self.to.gamma @ sky
        
        sky_r = self.to.P_r() @ sky
        
        vis = self.to.gamma @ sky_r
        logger.info("vis = {}".format(np.real(vis)[0:10]))
        
        # Check that the vis from the sky is the same as the vis from the range-space sky.
        self.assertTrue(np.allclose(vis_orig, vis))
        
        
        # Now image the range-space vis 
        imaged_sky = self.to.image_visibilities(vis, self.sphere, scale=False)
            
        vis3 = self.to.gamma @ imaged_sky
        # Now check that the visibilities from the imaged sky match the original visibiities
        
        for v1,v2 in zip(vis, vis3):
            logger.info("a,b = {} {}".format(v1, v2))
            self.assertAlmostEqual(v1[0], v2[0])
            

    def test_A(self):
    
        Ar = self.to.U_1 @ self.to.sigma[0:self.to.rank, 0:self.to.rank] # The new telescope operator.
        self.assertEqual(Ar.shape[0], self.to.A_r.shape[0])
        self.assertEqual(Ar.shape[1], self.to.A_r.shape[1])
        

    def test_imaging_vs_natural(self):
        ### Check that v = A_r x_r is the same as Gamma s
        sky = self.get_point_sky()

        # Get the visibilities
        vis = self.to.gamma @ sky
                
        # Now image the range-space vis 
        imaged_sky = self.to.image_natural(vis, self.sphere, scale=False)
                
        vis3 = self.to.gamma @ imaged_sky
        # Now check that the visibilities from the imaged sky match the original visibiities
        
        for v1,v2 in zip(vis, vis3):
            logger.info("a,b = {} {}".format(v1, v2))
            self.assertAlmostEqual(v1[0], v2[0])
            

    def test_bayes(self):
        sky = self.get_point_sky()
        vis = self.to.gamma @ sky
        
        prior = to.get_prior() # in the image space.
    
        prior_r =  prior.linear_transform(self.to.Vh)

        sigma_vis = 1e-6*np.identity(self.to.n_v)

        posterior_r = self.to.sequential_inference(prior_r, vis.flatten(), sigma_vis)

        
        
