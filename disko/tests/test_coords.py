#
# Copyright Tim Molteno 2026 tim@elec.ac.nz
# License: GPLv3
#
"""Unit tests for disko.coords — issue #10 Phase 1 coordinate transforms.

Known values are pinned to a fixed site and epoch (the TART array, during
the 2021-03-25 observation that test_data/test.ms was made from) so the
expected numbers are stable.
"""

import datetime
import unittest

import numpy as np

from disko import coords, sphere
from disko.healpix_sphere import HealpixSubFoV

LON, LAT, HEIGHT = 170.5, -45.5, 42.0
OBSTIME = datetime.datetime(2021, 3, 25, 20, 49, 23, tzinfo=datetime.timezone.utc)


def make_fov():
    fov = HealpixSubFoV(
        res_arcmin=60.0, theta=0.0, phi=0.0, radius_rad=np.radians(30)
    )
    fov.set_info(timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT)
    return fov


class TestZenithDerivation(unittest.TestCase):
    """The zenith-derived phase centre of the #14 fix, now in coords."""

    def test_zenith_matches_phase_center_radec(self):
        # Gate: the coords transform agrees with FoV.phase_center_radec().
        fov = make_fov()
        self.assertIsNone(fov.phase_center)  # nothing stored -> derived

        ra, dec = fov.phase_center_radec()
        ra2, dec2 = coords.zenith_radec(LON, LAT, HEIGHT, obstime=OBSTIME)
        self.assertAlmostEqual(ra, ra2, delta=1e-9)
        self.assertAlmostEqual(dec, dec2, delta=1e-9)

        # The derived centre is explicitly provenanced as 'zenith'.
        pc = fov.get_phase_center()
        self.assertEqual(pc.provenance, coords.PROVENANCE_ZENITH)
        self.assertEqual(pc.obstime, OBSTIME)
        # GeoLocation round-trips through geocentric coordinates, so the
        # site comes back with ~1e-11 of float noise: compare, don't eq.
        np.testing.assert_allclose(pc.site, (LON, LAT, HEIGHT), atol=1e-6)

        # The zenith's ICRS declination is the site latitude (within the
        # ~0.15 deg pole-of-date/ICRS offset, the tolerance the #14 tests
        # already use).
        self.assertAlmostEqual(dec, LAT, delta=1.0)

    def test_elaz_zenith_maps_to_phase_center(self):
        ra_z, dec_z = coords.zenith_radec(LON, LAT, HEIGHT, obstime=OBSTIME)
        ra, dec = coords.elaz_to_radec(
            np.pi / 2, 0.0, LON, LAT, HEIGHT, obstime=OBSTIME
        )
        self.assertAlmostEqual(ra, ra_z, delta=1e-9)
        self.assertAlmostEqual(dec, dec_z, delta=1e-9)


class TestElAzCelestialRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            el = rng.uniform(np.radians(1.0), np.radians(89.0))
            az = rng.uniform(-np.pi, np.pi)
            ra, dec = coords.elaz_to_radec(
                el, az, LON, LAT, HEIGHT, obstime=OBSTIME
            )
            el2, az2 = coords.radec_to_elaz(
                ra, dec, LON, LAT, HEIGHT, obstime=OBSTIME
            )
            self.assertAlmostEqual(el, el2, delta=1e-9)
            # az wraps at +/-pi: compare the wrapped difference.
            d_az = (az2 - az + np.pi) % (2 * np.pi) - np.pi
            self.assertAlmostEqual(d_az, 0.0, delta=1e-9)

    def test_azimuth_convention(self):
        # Azimuth is clockwise from north (astropy and disko agree), so a
        # source east of the zenith (az=90 deg) has larger RA than the
        # zenith, and one to the north (az=0) has larger declination.
        ra_z, dec_z = coords.zenith_radec(LON, LAT, HEIGHT, obstime=OBSTIME)
        ra_e, dec_e = coords.elaz_to_radec(
            np.radians(45), np.radians(90), LON, LAT, HEIGHT, obstime=OBSTIME
        )
        ra_n, dec_n = coords.elaz_to_radec(
            np.radians(45), 0.0, LON, LAT, HEIGHT, obstime=OBSTIME
        )
        self.assertGreater((ra_e - ra_z + 180) % 360 - 180, 0.0)  # east
        self.assertGreater(dec_n, dec_z)  # north


class TestLmnCelestial(unittest.TestCase):
    def setUp(self):
        self.ra0, self.dec0 = 306.0584, -45.9215  # a known MS pointing
        self.pc = coords.PhaseCenter.from_phase_dir(
            self.ra0, self.dec0, field_id=0, n_fields=60
        )

    def test_phase_center_is_origin(self):
        l, m, n = coords.radec_to_lmn(self.ra0, self.dec0, self.pc)
        self.assertAlmostEqual(float(l), 0.0, delta=1e-12)
        self.assertAlmostEqual(float(m), 0.0, delta=1e-12)
        self.assertAlmostEqual(float(n), 1.0, delta=1e-12)

    def test_round_trip(self):
        rng = np.random.default_rng(3)
        ra = self.ra0 + rng.uniform(-10.0, 10.0, 100)
        dec = np.clip(self.dec0 + rng.uniform(-10.0, 10.0, 100), -89, 89)
        l, m, n = coords.radec_to_lmn(ra, dec, self.pc)
        ra2, dec2 = coords.lmn_to_radec(l, m, self.pc)
        # Vectorised: whole grids convert at once.
        self.assertEqual(np.asarray(ra).shape, ra.shape)
        np.testing.assert_allclose(ra2, ra % 360.0, atol=1e-9)
        np.testing.assert_allclose(dec2, dec, atol=1e-9)

    def test_lmn_matches_elaz2lmn(self):
        # elaz2lmn decomposes in the geolocated (of-date) frame while
        # radec_to_lmn decomposes in ICRS, so the two differ only by the
        # position angle between the pole of date and the ICRS pole
        # (~0.15 deg in 2026; measured max over random directions is
        # 0.0023 in direction cosine). Both put l toward east, m toward
        # north, n toward the zenith.
        ra_z, dec_z = coords.zenith_radec(LON, LAT, HEIGHT, obstime=OBSTIME)
        pc = coords.PhaseCenter.from_zenith(ra_z, dec_z, obstime=OBSTIME)
        rng = np.random.default_rng(42)
        worst = 0.0
        for _ in range(100):
            el = rng.uniform(np.radians(5.0), np.radians(88.0))
            az = rng.uniform(-np.pi, np.pi)
            ra, dec = coords.elaz_to_radec(
                el, az, LON, LAT, HEIGHT, obstime=OBSTIME
            )
            l_c, m_c, n_c = coords.radec_to_lmn(ra, dec, pc)
            l_t, m_t, n_t = sphere.elaz2lmn(el, az)
            worst = max(
                worst,
                abs(l_c - l_t), abs(m_c - m_t), abs(n_c - n_t),
            )
        self.assertLess(worst, 0.005)


class TestIersOfflinePolicy(unittest.TestCase):
    """The #14 IERS policy now lives in exactly one place: offline_iers()."""

    def test_policy_scoped_and_restored(self):
        from astropy.utils import iers

        saved = (iers.conf.auto_download, iers.conf.auto_max_age)
        iers.conf.auto_download = True
        iers.conf.auto_max_age = 10
        try:
            with coords.offline_iers():
                self.assertFalse(iers.conf.auto_download)
                self.assertIsNone(iers.conf.auto_max_age)
            self.assertEqual(
                (iers.conf.auto_download, iers.conf.auto_max_age), (True, 10)
            )

            # Restored even when the body raises...
            with self.assertRaises(ValueError):
                with coords.offline_iers():
                    raise ValueError("boom")
            self.assertEqual(
                (iers.conf.auto_download, iers.conf.auto_max_age), (True, 10)
            )

            # ... and transforms do not leak a policy change either.
            coords.zenith_radec(LON, LAT, HEIGHT, obstime=OBSTIME)
            coords.elaz_to_radec(1.0, 2.0, LON, LAT, HEIGHT, obstime=OBSTIME)
            coords.radec_to_elaz(10.0, -20.0, LON, LAT, HEIGHT, obstime=OBSTIME)
            self.assertEqual(
                (iers.conf.auto_download, iers.conf.auto_max_age), (True, 10)
            )
        finally:
            iers.conf.auto_download, iers.conf.auto_max_age = saved


if __name__ == "__main__":
    unittest.main()
