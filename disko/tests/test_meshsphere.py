#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

import unittest
import logging
import os

import numpy as np

from disko import AdaptiveMeshSphere, area, HealpixSubSphere

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
LOGGER.setLevel(logging.INFO)

class TestMeshsphere(unittest.TestCase):

    def setUp(self):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]
        self.sphere = AdaptiveMeshSphere.from_resolution(res_arcmin=60, res_arcmax=60, 
                                                         theta=np.radians(0.0), 
                                                         phi=0.0, radius_deg=10)

    
    def test_sizes(self):
       self.assertEqual(self.sphere.npix, self.sphere.el_r.shape[0])
       self.assertEqual(self.sphere.npix, self.sphere.l.shape[0])

    def test_areas(self):
        points = np.array([[0,0],
                  [1,0],
                  [1,1]])
        cells = [[0,1,2]]
        self.assertAlmostEqual(area(cells[0], points), 0.5)
        for a in self.sphere.pixel_areas:
            self.assertTrue(a > 0.0)
             
    def test_lmn(self):
        hp_sphere = HealpixSubSphere.from_resolution(res_arcmin=60.0, 
                                              theta=np.radians(0.0), 
                                              phi=0.0, radius_rad=np.radians(10))
        
        self.assertAlmostEqual(self.sphere.fov, hp_sphere.fov)

        print(np.max(self.sphere.l), np.max(self.sphere.m), np.max(self.sphere.n_minus_1))
        print(np.max(hp_sphere.l), np.max(hp_sphere.m), np.max(hp_sphere.n_minus_1))
        
        print("el_r {} {}".format(np.max(self.sphere.el_r), np.max(hp_sphere.el_r)))
        self.assertAlmostEqual(np.max(self.sphere.el_r), np.max(hp_sphere.el_r), 1)

        self.assertAlmostEqual(np.max(self.sphere.m), np.max(hp_sphere.m), 2)
        self.assertAlmostEqual(np.max(self.sphere.l), np.max(hp_sphere.l), 2)
        self.assertAlmostEqual(np.min(self.sphere.n_minus_1), np.min(hp_sphere.n_minus_1), 2)

    #def test_harmonics(self):
        
        
    @unittest.skip("We don't have svg write going yet")        
    def test_svg(self):
        res_deg = 10
        fname='test.svg'

        self.sphere.to_svg(fname=fname, pixels_only=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    def test_fits(self):
        res_deg = 10
        fname='test.fits'

        self.sphere.to_fits(fname=fname, fov=90)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
