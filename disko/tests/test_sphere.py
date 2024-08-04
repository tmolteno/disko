#
# Copyright Tim Molteno 2017-2019 tim@elec.ac.nz
#

import unittest

import numpy as np

from disko import sphere, HealpixFoV
from disko import fov


class TestUtil(unittest.TestCase):

    def setUp(self):
        pass

    def test_hp_elaz_hp(self):
        theta = np.random.rand(100)*np.pi/2
        phi = np.random.rand(100)*np.pi*2

        el, az = sphere.hp2elaz(theta, phi)

        theta2, phi2 = sphere.elaz2hp(el, az)

        for i in range(100):
            self.assertAlmostEqual(theta2[i], theta[i])
            self.assertAlmostEqual(phi2[i], phi[i])

    def test_elaz(self):
        # Zenith
        for i in range(100):
            el = np.pi/2
            az = np.random.uniform(-np.pi, np.pi)

            theta, phi = sphere.elaz2hp(el, az)
            self.assertEqual(theta, 0)
            self.assertEqual(phi, 0)

        # North
        el = 0.0
        az = 0.0

        theta, phi = sphere.elaz2hp(el, az)
        self.assertEqual(theta, 0)
        self.assertEqual(phi, 0)
            

    def test_load_save(self):

        sph = HealpixFoV(nside=64)
        sph.to_hdf('test.h5')

        sph2 = fov.from_hdf('test.h5')

        self.assertTrue(np.allclose(sph.pixels, sph2.pixels))
        self.assertTrue(np.allclose(sph.pixel_areas, sph2.pixel_areas))
