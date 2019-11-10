#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

import unittest
import logging
import sys

import numpy as np

from disko import sphere

logger = logging.getLogger(__name__)
#logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

class TestSubsphere(unittest.TestCase):

    def setUp(self):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]
        self.sphere = sphere.HealpixSubSphere(resolution=60.0, 
                                              theta = np.radians(10.0), phi=0.0, radius=np.radians(1))
    

    def test_big_subsphere(self):
        # Check that a full subsphere is the same as the sphere.
        res_deg = 3.0
        big = sphere.HealpixSubSphere(resolution=res_deg*60.0, 
                                      theta = np.radians(0.0), phi=0.0, radius=np.radians(180))
        old = sphere.HealpixSphere(32)
        
        self.assertEqual(big.nside, 32)
        self.assertEqual(big.npix, old.npix)
        
    def test_tiny_subsphere(self):
        # Check that a full subsphere is the same as the sphere.
        res_deg = 0.5
        big = sphere.HealpixSubSphere(resolution=res_deg*60.0, 
                                      theta = np.radians(0.0), phi=0.0, radius=np.radians(5))
        
        self.assertEqual(big.nside, 128)
        self.assertEqual(big.npix, 364)
    
    def test_sizes(self):
       self.assertEqual(self.sphere.npix, self.sphere.el_r.shape[0])
       self.assertEqual(self.sphere.npix, self.sphere.l.shape[0])


    def test_svg(self):
        res_deg = 10
        big = sphere.HealpixSubSphere(resolution=res_deg*60.0, 
                                      theta = np.radians(0.0), phi=0.0, radius=np.radians(45))

        big.to_svg(fname='test.svg', pixels_only=True)
        #self.assertEqual(big.nside, 32)
