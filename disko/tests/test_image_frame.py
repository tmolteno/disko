#
# Copyright Tim Molteno 2026 tim@elec.ac.nz
# License: GPLv3
#
"""The image path's frame and sampling (issues #24 and #26).

Issue #24: ``FoV.to_fits`` built a grid uniform in the direction cosine
l (the SIN plane coordinate) but wrote a CDELT derived as if the plane
were linear in angle, and with a reference pixel half a pixel off the
grid centre. On a wide field (tart.hdf's 155 deg) the header's plane
coordinate ran past the projection at the edges and astropy returned
NaN in every outermost row/column.

Issue #26: the grid was decomposed in the geolocated frame
(``elaz2lmn``) while the header claimed ICRS, so every world coordinate
sat ~0.13 deg (the pole angle) off the pixel's true direction — and the
SVG drew its polygons in that geolocated frame while markers came from
the ICRS phase-centre frame, a disagreement the issues measured as the
same pole rotation (it was in fact worse: the marker transform was also
missing the half-turn the grid's projection applies). ``to_fits`` and
``to_svg`` now share one decomposition, ``icrs_lmn_from_elaz``.

These tests pin the two observable consequences:

- a pixel's world coordinate IS its true celestial direction (to
  round-trip precision, not just "within the pole angle"), world
  coordinates are finite where the data exists, and the header maps
  pixels exactly along the sin-space sampling;
- an overplotted marker lands on the pixel that holds the source, in
  the same drawn frame as the grid polygons.
"""

import datetime
import os
import re
import tempfile
import unittest

import numpy as np

from disko import coords
from disko.healpix_sphere import HealpixSubFoV

LON, LAT, HEIGHT = 170.5, -45.5, 42.0
OBSTIME = datetime.datetime(2021, 3, 25, 20, 49, 23, tzinfo=datetime.timezone.utc)

# A known MS pointing (test_data/test.ms field 0), used to build a
# genuinely phase-steered FoV like test_overplot does.
MS_RA0, MS_DEC0 = 306.0584, -45.9215

WIDTH = 2000  # to_fits grid width/height


def make_fov(radius_deg, res_arcmin=60.0, phase_center=None):
    fov = HealpixSubFoV(
        res_arcmin=res_arcmin, theta=0.0, phi=0.0,
        radius_rad=np.radians(radius_deg),
    )
    fov.set_info(timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT,
                 phase_center=phase_center)
    return fov


def write_fits(fov, path):
    fov.to_fits(path, title="image frame test")


def sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation (degrees) of two ICRS directions."""
    ra1, dec1, ra2, dec2 = np.radians([ra1, dec1, ra2, dec2])
    sin_half = np.sin((dec2 - dec1) / 2) ** 2 + (
        np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    )
    return float(np.degrees(2 * np.arcsin(np.sqrt(sin_half))))


class TestSinSpaceSampling(unittest.TestCase):
    """Issue #24: CDELT/CRPIX derive from the actual sin-space grid."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fits = os.path.join(self._tmp.name, "out.fits")

    def test_edges_finite_on_wide_field(self):
        # The issue's reproduction: on tart.hdf's ~155 deg field the old
        # angle-based CDELT put the edges outside the SIN projection and
        # every outermost row/column was NaN. Edge midpoints of the
        # image must always be on the sky (corners of a field wider
        # than 90 deg legitimately are not: l^2 + m^2 > 1, off the
        # hemisphere, and cannot be represented by RA---SIN at all).
        from astropy.io import fits
        from astropy.wcs import WCS

        fov = make_fov(77.5, res_arcmin=120.0)  # 155 deg field, coarse
        write_fits(fov, self.fits)
        hdr = fits.getheader(self.fits)
        wcs = WCS(hdr)

        cx = int(hdr["CRPIX1"])  # a column/row near the centre, 1-based
        cy = int(hdr["CRPIX2"])
        edge_pixels = [
            (1, cy), (hdr["NAXIS1"], cy),
            (cx, 1), (cx, hdr["NAXIS2"]),
        ]
        for px, py in edge_pixels:
            world = wcs.all_pix2world([[px, py]], 1)[0]
            self.assertTrue(
                np.all(np.isfinite(world)),
                f"pixel ({px}, {py}) is NaN: the wide-field edges must "
                "map onto the sky (issue #24)",
            )

    def test_corners_finite_on_narrow_field(self):
        # A 60 deg field fits well inside the SIN hemisphere
        # (2 * sin(30 deg)^2 = 0.5 < 1), so every corner is on the sky.
        from astropy.io import fits
        from astropy.wcs import WCS

        fov = make_fov(30.0)
        write_fits(fov, self.fits)
        hdr = fits.getheader(self.fits)
        wcs = WCS(hdr)

        for px, py in [(1, 1), (hdr["NAXIS1"], 1),
                       (1, hdr["NAXIS2"]), (hdr["NAXIS1"], hdr["NAXIS2"])]:
            world = wcs.all_pix2world([[px, py]], 1)[0]
            self.assertTrue(np.all(np.isfinite(world)),
                            f"corner ({px}, {py}) is NaN")

    def test_header_maps_pixels_along_the_sampling_grid(self):
        # The grid samples l, m uniformly (one step dl per pixel), and
        # for RA---SIN the plane coordinate x (deg) is exactly
        # l = radians(x). So the header's world coordinate at a grid
        # sample must be precisely the celestial direction of lmn_to_radec
        # at that sample — this pins CDELT, CRPIX and the projection
        # together (an angle-based CDELT, a sign flip, or the old
        # half-pixel-off CRPIX all fail it).
        from astropy.io import fits
        from astropy.wcs import WCS

        fov = make_fov(30.0)
        write_fits(fov, self.fits)
        hdr = fits.getheader(self.fits)
        wcs = WCS(hdr)
        pc = fov.get_phase_center()

        # The reference pixel is exactly the l = m = 0 sample: for
        # linspace(-l0, l0, N) that lies half a sample past N//2.
        self.assertEqual(hdr["CRPIX1"], (WIDTH + 1) / 2.0)
        self.assertEqual(hdr["CRPIX2"], (WIDTH + 1) / 2.0)

        # l grows with the column (east) and with the row (north): both
        # CDELTs positive, so the header follows the data instead of
        # mirroring it.
        self.assertGreater(hdr["CDELT1"], 0.0)
        self.assertGreater(hdr["CDELT2"], 0.0)

        l0 = np.sin(fov.fov.radians() / 2)
        dl = 2 * l0 / (WIDTH - 1)
        self.assertAlmostEqual(hdr["CDELT1"], np.degrees(dl), places=12)
        self.assertAlmostEqual(hdr["CDELT2"], np.degrees(dl), places=12)

        for col, row in [(1, 1000), (500, 1300), (1001, 1001),
                         (1500, 700), (2000, 2000)]:
            l_val = (col - hdr["CRPIX1"]) * np.radians(hdr["CDELT1"])
            m_val = (row - hdr["CRPIX2"]) * np.radians(hdr["CDELT2"])
            expected = coords.lmn_to_radec(l_val, m_val, pc)
            got = wcs.all_pix2world([[col, row]], 1)[0]
            self.assertLess(
                sep_deg(got[0], got[1], expected[0], expected[1]), 1e-6,
                f"pixel ({col}, {row}) does not follow the sampling grid",
            )


class TestPixelsAreTheirOwnDirections(unittest.TestCase):
    """Issue #26: pixel -> world is the pixel's true ICRS direction."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _check(self, phase_center):
        from astropy.io import fits
        from astropy.wcs import WCS

        fov = make_fov(30.0, phase_center=phase_center)
        path = os.path.join(self._tmp.name, "out.fits")
        write_fits(fov, path)
        hdr = fits.getheader(path)
        wcs = WCS(hdr)
        os.remove(path)

        l0 = np.sin(fov.fov.radians() / 2)
        dl = 2 * l0 / (WIDTH - 1)

        rng = np.random.default_rng(27)
        for _ in range(12):
            # Directions inside the 30 deg disc, so every sample sits
            # inside the grid (the centre pixel is the zenith here, el=90).
            el = rng.uniform(np.radians(62.0), np.radians(86.0))
            az = rng.uniform(-np.pi, np.pi)

            # Truth: where this direction actually is on the sky.
            ra_true, dec_true = coords.elaz_to_radec(
                el, az, LON, LAT, HEIGHT, obstime=OBSTIME
            )
            # Where the grid puts it: ICRS l,m about the phase centre...
            l, m, _n = fov.icrs_lmn_from_elaz(el, az)
            # ... and the pixel that carries that sample.
            col = (float(l) + l0) / dl + 1.0  # 1-based FITS pixel coord
            row = (float(m) + l0) / dl + 1.0

            world = wcs.all_pix2world([[col, row]], 1)[0]
            err = sep_deg(world[0], world[1], ra_true, dec_true)
            self.assertLess(
                err, 1e-4,
                "the header placed a pixel pole-angle/phase-steering away "
                f"from its true direction ({err:.6f} deg, issue #26)",
            )

    def test_zenith_centred_grid(self):
        self._check(None)

    def test_phase_steered_grid(self):
        pc = coords.PhaseCenter.from_phase_dir(
            MS_RA0, MS_DEC0, field_id=0, n_fields=60
        )
        self._check(pc)


class TestMarkerLandsOnItsPixel(unittest.TestCase):
    """Issue #26, end to end through to_svg: marker on the grid."""

    def test_marker_on_its_own_grid_pixel(self):
        fov = make_fov(30.0)

        class _Source(object):
            def __init__(self, el, az):
                self.el_r = el
                self.az_r = az

        el, az = np.radians(70.0), np.radians(30.0)
        hp_index = fov.index_of(el, az)
        label_index = int(np.where(fov.pixel_indices == hp_index)[0][0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.svg")
            fov.to_svg(path, src_list=[_Source(el, az)], show_grid=False,
                       title=None, show_cbar=False, pixels_only=True)
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()

        # The grid polygon of the pixel containing the source (its
        # label is drawn at the polygon's mean position)...
        match = re.search(r'<text[^>]*>\s*%d\s*</text>' % label_index, data)
        self.assertIsNotNone(match, "the source's grid pixel is not labelled")
        gx, gy = map(
            int, re.search(r'x="(-?\d+)" y="(-?\d+)"', match.group(0)).groups()
        )

        # ... and the marker ellipse.
        ellipse = re.search(r'<ellipse cx="(-?\d+)" cy="(-?\d+)"', data)
        self.assertIsNotNone(ellipse, "the source marker was not drawn")
        mx, my = map(int, ellipse.groups())

        distance = np.hypot(mx - gx, my - gy)
        # A pixel is ~55 arcmin across here (~60 px on the 4000 px
        # canvas); the marker sits inside its pixel, up to the offset
        # between the source direction and its pixel's mean position.
        self.assertLess(
            distance, 90.0,
            f"marker is {distance:.0f} px from its own grid pixel "
            "(issue #26)",
        )

        # And explicitly not the old half-turn-away placement: the
        # mirrored position of the pixel label is across the image.
        mirror = np.hypot(mx - (2 * 2000 - gx), my - (2 * 2000 - gy))
        self.assertGreater(mirror, 1000.0)


if __name__ == "__main__":
    unittest.main()
