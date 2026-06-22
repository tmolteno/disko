#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
#

import datetime
import logging
import os
import unittest

import numpy as np

from disko import AdaptiveMeshFoV, HealpixSubFoV, Resolution, area, fov

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.INFO)

# gmsh 4.15+ is incompatible with pygmsh 7.x;
# the pygmsh.geo.Geometry context manager crashes on exit.
_GMSH_OK = False
try:
    from disko.sphere_mesh import get_mesh

    _r = Resolution.from_deg(10)
    _e = Resolution.from_arcmin(60)
    get_mesh(_r.radians() / 2, _e.radians())
    _GMSH_OK = True
except Exception:
    pass

_skip_gmsh = unittest.skipUnless(_GMSH_OK, "gmsh/pygmsh not compatible")


class TestMeshArea(unittest.TestCase):
    """Tests for the standalone area() function (no gmsh needed)."""

    def test_areas(self):
        points = np.array([[0, 0], [1, 0], [1, 1]])
        cells = [[0, 1, 2]]
        self.assertAlmostEqual(area(cells[0], points), 0.5)


class TestMeshsphere(unittest.TestCase):
    def setUp(self):
        if not _GMSH_OK:
            self.skipTest("gmsh/pygmsh not compatible")
        self.sphere = AdaptiveMeshFoV(
            res_min=Resolution.from_arcmin(60),
            res_max=Resolution.from_arcmin(60),
            theta=np.radians(0.0),
            phi=0.0,
            fov=Resolution.from_deg(20),
        )
        self.sphere.set_info(
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            lon=170.5,
            lat=-45.5,
            height=42,
        )

    @_skip_gmsh
    def test_copy(self):
        sph3 = self.sphere.copy()
        sph3.pixels += 1
        self.assertFalse(np.allclose(self.sphere.pixels, sph3.pixels))
        self.assertTrue(np.allclose(self.sphere.pixel_areas, sph3.pixel_areas))

    @_skip_gmsh
    def test_area(self):
        self.assertAlmostEqual(self.sphere.get_area(), 1.0)

    @_skip_gmsh
    def test_sizes(self):
        self.assertEqual(self.sphere.npix, self.sphere.el_r.shape[0])
        self.assertEqual(self.sphere.npix, self.sphere.l.shape[0])

    @_skip_gmsh
    def test_lmn(self):
        hp_sphere = HealpixSubFoV(
            res_arcmin=60.0,
            theta=np.radians(0.0),
            phi=0.0,
            radius_rad=np.radians(10),
        )
        self.assertAlmostEqual(self.sphere.fov.degrees(), hp_sphere.fov.degrees())
        self.assertAlmostEqual(np.max(self.sphere.el_r), np.max(hp_sphere.el_r), 1)
        self.assertAlmostEqual(np.max(self.sphere.m), np.max(hp_sphere.m), 2)
        self.assertAlmostEqual(np.max(self.sphere.l), np.max(hp_sphere.l), 2)
        self.assertAlmostEqual(
            np.min(self.sphere.n_minus_1), np.min(hp_sphere.n_minus_1), 2
        )

    @unittest.skip("Qhull Delaunay precision issue with 2D mesh points")
    def test_adaptive(self):
        grad, cell_pairs = self.sphere.gradient()

    @unittest.skip("We don't have svg write going yet")
    def test_svg(self):
        fname = "test.svg"
        self.sphere.to_svg(fname=fname, pixels_only=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    @_skip_gmsh
    def test_fits(self):
        fname = "test.fits"
        self.sphere.to_fits(fname=fname)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    @_skip_gmsh
    def test_load_save(self):
        self.sphere.to_hdf("test.h5")
        sph2 = fov.from_hdf("test.h5")
        self.assertTrue(np.allclose(self.sphere.pixels, sph2.pixels))
        self.assertTrue(np.allclose(self.sphere.pixel_areas, sph2.pixel_areas))
        self.assertTrue(np.allclose(self.sphere.l, sph2.l))
        self.assertTrue(np.allclose(self.sphere.m, sph2.m))
        self.assertTrue(np.allclose(self.sphere.n_minus_1, sph2.n_minus_1))
