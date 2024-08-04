#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

import unittest
import logging
import os
import datetime
import numpy as np

from disko import HealpixSubFoV, HealpixFoV
from disko import fov

LOGGER = logging.getLogger(__name__)
# Add a null handler so logs can go somewhere
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.INFO)


class TestSubsphere(unittest.TestCase):

    def setUp(self):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]
        self.sphere = HealpixSubFoV(res_arcmin=60.0,
                                       theta=np.radians(10.0),
                                       phi=0.0, radius_rad=np.radians(1))
        self.sphere.set_info(timestamp=datetime.datetime.now(),
                             lon=170.5, lat=-45.5, height=42)

    def test_area(self):
        sky = HealpixFoV(nside=128)

        self.assertAlmostEqual(sky.get_area(), 4*np.pi)

        hemisphere = HealpixSubFoV(res_arcmin=60.0,
                                      theta=np.radians(0.0),
                                      phi=0.0, radius_rad=np.radians(90))
        self.assertAlmostEqual(hemisphere.get_area(), 2*np.pi, 1)

    def test_copy(self):
        sky = HealpixFoV(nside=128)
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
        big = HealpixSubFoV(res_arcmin=res_deg*60.0,
                               theta=np.radians(0.0), phi=0.0,
                               radius_rad=np.radians(180))
        old = HealpixFoV(32)

        self.assertEqual(big.nside, 32)
        self.assertEqual(big.npix, old.npix)

    def test_tiny_subsphere(self):
        # Check that a full subsphere is the same as the sphere.
        res_deg = 0.5
        tiny = HealpixSubFoV(res_arcmin=res_deg*60.0,
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
        big = HealpixSubFoV(res_arcmin=res_deg*60.0,
                               theta=np.radians(0.0), phi=0.0,
                               radius_rad=np.radians(45))

        big.to_svg(fname=fname, pixels_only=True, show_cbar=False)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    def test_fits(self):
        res_deg = 10
        fname = 'test.fits'
        big = HealpixSubFoV(res_arcmin=res_deg*60.0,
                               theta=np.radians(0.0), phi=0.0,
                               radius_rad=np.radians(45))

        big.to_fits(fname=fname)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    def test_load_save(self):
        res_deg = 10
        sph = HealpixSubFoV(res_arcmin=res_deg*60.0,
                               theta=np.radians(0.0), phi=0.0,
                               radius_rad=np.radians(45))

        sph.set_info(timestamp=datetime.datetime.now(),
                     lon=170.5, lat=-45.5, height=42)

        sph.to_hdf('test.h5')

        sph2 = fov.from_hdf('test.h5')

        self.assertTrue(np.allclose(sph.pixels, sph2.pixels))
        self.assertTrue(np.allclose(sph.pixel_areas, sph2.pixel_areas))
        self.assertTrue(np.allclose(sph.pixel_indices, sph2.pixel_indices))

    def test_indexing(self):
        sph = HealpixSubFoV(res_arcmin=60.0,
                               theta=np.radians(0.0), phi=0.0,
                               radius_rad=np.radians(90))

        for i in range(500):
            el = np.random.uniform(np.radians(1), np.radians(90))
            az = np.random.uniform(np.radians(-180), np.radians(180))
            ind = sph.index_of(el, az)
            self.assertTrue(ind < sph.npix)
