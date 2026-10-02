#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
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
        self.sphere.set_info(timestamp=datetime.datetime.now(datetime.timezone.utc),
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
        from astropy.io import fits
        from astropy.wcs import WCS

        res_deg = 10
        fname = 'test.fits'
        big = HealpixSubFoV(res_arcmin=res_deg*60.0,
                               theta=np.radians(0.0), phi=0.0,
                               radius_rad=np.radians(45))
        big.set_info(timestamp=datetime.datetime.now(datetime.timezone.utc),
                     lon=170.5, lat=-45.5, height=42)

        try:
            big.to_fits(fname=fname)
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
            ra, dec = big.phase_center_radec()
            self.assertAlmostEqual(hdr["CRVAL1"], ra, delta=1e-9)
            self.assertAlmostEqual(hdr["CRVAL2"], dec, delta=1e-9)
            # ... whose declination is the observer's latitude (to better
            # than a degree).
            self.assertAlmostEqual(hdr["CRVAL2"], -45.5, delta=1.0)
        finally:
            if os.path.isfile(fname):
                os.remove(fname)

    def test_load_save(self):
        res_deg = 10
        sph = HealpixSubFoV(res_arcmin=res_deg*60.0,
                               theta=np.radians(0.0), phi=0.0,
                               radius_rad=np.radians(45))

        sph.set_info(timestamp=datetime.datetime.now(datetime.timezone.utc),
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
