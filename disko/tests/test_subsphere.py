#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

import unittest
import logging
import os

import numpy as np

from disko import HealpixSubSphere, HealpixSphere
from disko import fov

LOGGER = logging.getLogger(__name__)
# Add a null handler so logs can go somewhere
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.INFO)


class TestSubsphere(unittest.TestCase):

    def setUp(self):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]
        self.sphere = HealpixSubSphere(res_arcmin=60.0,
                                                       theta=np.radians(10.0),
                                                       phi=0.0, radius_rad=np.radians(1))
    def test_area(self):
        sky = HealpixSphere(nside=128)
        
        self.assertAlmostEqual(sky.get_area(), 4*np.pi)

        hemisphere = HealpixSubSphere(res_arcmin=60.0,
                                                       theta=np.radians(0.0),
                                                       phi=0.0, radius_rad=np.radians(90))
        self.assertAlmostEqual(hemisphere.get_area(), 2*np.pi, 1)

    def test_copy(self):
        sky = HealpixSphere(nside=128)
        sky2 = sky.copy()
        sky.pixels += 1
        self.assertFalse(np.allclose(sky.pixels, sky2.pixels))
        self.assertTrue(np.allclose(sky.pixel_areas, sky2.pixel_areas))
        self.assertEqual(sky.nside, sky2.nside)
        sph3 = self.sphere.copy()
        sph3.pixels += 1
        self.assertFalse(np.allclose(self.sphere.pixels, sph3.pixels))
        self.assertTrue(np.allclose(self.sphere.pixel_areas, sph3.pixel_areas))
        
    def test_big_subsphere(self):
        # Check that a full subsphere is the same as the sphere.
        res_deg = 3.0
        big = HealpixSubSphere(res_arcmin=res_deg*60.0,
                                               theta=np.radians(0.0), phi=0.0,
                                               radius_rad=np.radians(180))
        old = HealpixSphere(32)

        self.assertEqual(big.nside, 32)
        self.assertEqual(big.npix, old.npix)

    def test_tiny_subsphere(self):
        # Check that a full subsphere is the same as the sphere.
        res_deg = 0.5
        tiny = HealpixSubSphere(res_arcmin=res_deg*60.0,
                                                theta=np.radians(0.0),
                                                phi=0.0, radius_rad=np.radians(5))

        self.assertEqual(tiny.nside, 128)
        self.assertEqual(tiny.npix, 364)

    def test_sizes(self):
        self.assertEqual(self.sphere.npix, self.sphere.el_r.shape[0])
        self.assertEqual(self.sphere.npix, self.sphere.l.shape[0])

    def test_svg(self):
        res_deg = 10
        fname = 'test.svg'
        big = HealpixSubSphere(res_arcmin=res_deg*60.0,
                                               theta=np.radians(0.0), phi=0.0,
                                               radius_rad=np.radians(45))

        big.to_svg(fname=fname, pixels_only=True, show_cbar=False)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    def test_fits(self):
        res_deg = 10
        fname = 'test.fits'
        big = HealpixSubSphere(res_arcmin=res_deg*60.0,
                                               theta=np.radians(0.0), phi=0.0,
                                               radius_rad=np.radians(45))

        big.to_fits(fname=fname)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
        
        
    def test_load_save(self):
        res_deg = 10
        sph = HealpixSubSphere(res_arcmin=res_deg*60.0,
                                               theta=np.radians(0.0), phi=0.0,
                                               radius_rad=np.radians(45))
        sph.to_hdf('test.h5')
        
        sph2 = fov.from_hdf('test.h5')
        
        self.assertTrue(np.allclose(sph.pixels, sph2.pixels))
        self.assertTrue(np.allclose(sph.pixel_areas, sph2.pixel_areas))
        self.assertTrue(np.allclose(sph.pixel_indices, sph2.pixel_indices))

