#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

import unittest
import logging
import os

import numpy as np

from disko import sphere

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
LOGGER.setLevel(logging.INFO)

class TestSubsphere(unittest.TestCase):

    def setUp(self):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]
        self.sphere = sphere.HealpixSubSphere.from_resolution(resolution=60.0, 
                                              theta=np.radians(10.0), 
                                              phi=0.0, radius=np.radians(1))

    def test_big_subsphere(self):
        # Check that a full subsphere is the same as the sphere.
        res_deg = 3.0
        big = sphere.HealpixSubSphere.from_resolution(resolution=res_deg*60.0, 
                                      theta=np.radians(0.0), phi=0.0, 
                                      radius=np.radians(180))
        old = sphere.HealpixSphere(32)

        self.assertEqual(big.nside, 32)
        self.assertEqual(big.npix, old.npix)

    def test_tiny_subsphere(self):
        # Check that a full subsphere is the same as the sphere.
        res_deg = 0.5
        big = sphere.HealpixSubSphere.from_resolution(resolution=res_deg*60.0, 
                                      theta=np.radians(0.0), 
                                      phi=0.0, radius=np.radians(5))

        self.assertEqual(big.nside, 128)
        self.assertEqual(big.npix, 364)
    
    def test_sizes(self):
       self.assertEqual(self.sphere.npix, self.sphere.el_r.shape[0])
       self.assertEqual(self.sphere.npix, self.sphere.l.shape[0])

    def test_svg(self):
        res_deg = 10
        fname='test.svg'
        big = sphere.HealpixSubSphere.from_resolution(resolution=res_deg*60.0, 
                                      theta=np.radians(0.0), phi=0.0, 
                                      radius=np.radians(45))

        big.to_svg(fname=fname, pixels_only=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
