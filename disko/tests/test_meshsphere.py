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
        from astropy.io import fits
        from astropy.wcs import WCS

        fname = "test.fits"
        try:
            self.sphere.to_fits(fname=fname)
            self.assertTrue(os.path.isfile(fname))

            with fits.open(fname) as hdul:
                hdr = hdul[0].header

            # Issue #14: FITS written from the non-MS path must carry a real
            # world coordinate system, not just CRPIX/CDELT.
            for key in [
                "RADESYS",
                "CTYPE1", "CRVAL1", "CUNIT1", "CRPIX1", "CDELT1",
                "CTYPE2", "CRVAL2", "CUNIT2", "CRPIX2", "CDELT2",
            ]:
                self.assertIn(key, hdr)

            self.assertEqual(hdr["RADESYS"].strip(), "ICRS")
            self.assertEqual(hdr["CTYPE1"].strip(), "RA---SIN")
            self.assertEqual(hdr["CUNIT1"].strip(), "deg")
            self.assertEqual(hdr["CTYPE2"].strip(), "DEC--SIN")
            self.assertEqual(hdr["CUNIT2"].strip(), "deg")

            # The header must parse as a celestial WCS...
            wcs = WCS(hdr)
            self.assertTrue(wcs.has_celestial)

            # ... CRVAL is the world coordinate of the reference pixel, which
            # is the phase center in the middle of the image...
            crval = np.array([hdr["CRVAL1"], hdr["CRVAL2"]])
            np.testing.assert_allclose(
                wcs.all_pix2world([[hdr["CRPIX1"], hdr["CRPIX2"]]], 1)[0],
                crval,
                atol=1e-6,
            )

            # ... so CRVAL lies inside the coordinate range the image covers.
            # Use the midpoints of the four edges: for a wide field of view
            # the image corners fall outside the SIN projection disk and are
            # not on the sky at all, the edge midpoints always are.
            cx = int(round(hdr["CRPIX1"]))
            cy = int(round(hdr["CRPIX2"]))
            self.assertTrue(1 <= cx <= hdr["NAXIS1"])
            self.assertTrue(1 <= cy <= hdr["NAXIS2"])
            ra_lo = wcs.all_pix2world([[1, cy]], 1)[0][0]
            ra_hi = wcs.all_pix2world([[hdr["NAXIS1"], cy]], 1)[0][0]
            dec_lo = wcs.all_pix2world([[cx, 1]], 1)[0][1]
            dec_hi = wcs.all_pix2world([[cx, hdr["NAXIS2"]]], 1)[0][1]

            self.assertTrue(
                min(dec_lo, dec_hi) <= crval[1] <= max(dec_lo, dec_hi),
                f"CRVAL2 {crval[1]} not in [{dec_lo}, {dec_hi}]",
            )
            # Right ascension wraps at 360 degrees, so compare offsets from
            # CRVAL: the two edges must be on opposite sides of it.
            d1 = (ra_lo - crval[0] + 180.0) % 360.0 - 180.0
            d2 = (ra_hi - crval[0] + 180.0) % 360.0 - 180.0
            self.assertLessEqual(
                d1 * d2,
                0.0,
                f"CRVAL1 {crval[0]} not between {ra_lo} and {ra_hi}",
            )

            # The WCS comes from the sphere itself: the phase center is the
            # zenith seen from the sphere's location at its timestamp...
            ra, dec = self.sphere.phase_center_radec()
            self.assertAlmostEqual(hdr["CRVAL1"], ra, delta=1e-9)
            self.assertAlmostEqual(hdr["CRVAL2"], dec, delta=1e-9)
            # ... whose declination is the observer's latitude (to better
            # than a degree).
            self.assertAlmostEqual(hdr["CRVAL2"], -45.5, delta=1.0)
        finally:
            if os.path.isfile(fname):
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
