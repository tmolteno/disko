#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
#

import unittest

import numpy as np

from disko import HealpixFoV, fov, sphere


class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def test_hp_elaz_hp(self):
        theta = np.random.rand(100) * np.pi / 2
        phi = np.random.rand(100) * np.pi * 2

        el, az = sphere.hp2elaz(theta, phi)

        theta2, phi2 = sphere.elaz2hp(el, az)

        for i in range(100):
            self.assertAlmostEqual(theta2[i], theta[i])
            self.assertAlmostEqual(phi2[i], phi[i])

    def test_elaz(self):
        # Round-trip: el,az -> theta,phi -> el2,az2 should recover original
        for i in range(100):
            el = np.random.uniform(0, np.pi / 2)
            az = np.random.uniform(-np.pi, np.pi)

            theta, phi = sphere.elaz2hp(el, az)
            el2, az2 = sphere.hp2elaz(theta, phi)
            self.assertAlmostEqual(el, el2)
            self.assertAlmostEqual(az, az2)

    def test_load_save(self):

        sph = HealpixFoV(nside=64)
        sph.to_hdf("test.h5")

        sph2 = fov.from_hdf("test.h5")

        self.assertTrue(np.allclose(sph.pixels, sph2.pixels))
        self.assertTrue(np.allclose(sph.pixel_areas, sph2.pixel_areas))
