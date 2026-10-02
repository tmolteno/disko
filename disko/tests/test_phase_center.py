#
# Copyright Tim Molteno 2026 tim@elec.ac.nz
# License: GPLv3
#
"""Tests for the first-class phase centre of a FoV (issue #10, Phase 1):

- provenance: MS PHASE_DIR vs zenith-derived, and which field it came from,
- HDF persistence in both directions (old files load in new code, new files
  load in the old code path),
- frame-aware source placement / index_of,
- the validation oracle: on a zenith-pointed TART MS the stored PHASE_DIR
  must agree with the zenith-derived centre to well under a pixel.
"""

import datetime
import json
import os
import tempfile
import unittest

import h5py
import numpy as np

from disko import coords, fov
from disko.healpix_sphere import HealpixFoV, HealpixSubFoV
from disko.ms_helper import casa_read_ms, get_array_location, phase_center_from_hdr
from disko.resolution import Resolution
from disko.sphere import GeoLocation
from disko.sphere_mesh import AdaptiveMeshFoV

LON, LAT, HEIGHT = 170.5, -45.5, 42.0
OBSTIME = datetime.datetime(2021, 3, 25, 20, 49, 23, tzinfo=datetime.timezone.utc)

# A known pointing: field 0 of test_data/test.ms (source J202414-455517).
MS_RA, MS_DEC = 306.0584, -45.9215

MS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "test.ms")


def make_fov():
    return HealpixSubFoV(
        res_arcmin=60.0, theta=0.0, phi=0.0, radius_rad=np.radians(30)
    )


class _ElazSource(object):
    """Stands in for a tart.imaging.elaz source (celestial-free)."""

    def __init__(self, el_r, az_r):
        self.el_r = el_r
        self.az_r = az_r


class _CelestialSource(object):
    """A source given only in celestial coordinates (ICRS degrees)."""

    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec


class TestPhaseCentreProvenance(unittest.TestCase):
    def test_ms_phase_dir_stored_with_provenance(self):
        fov1 = make_fov()
        pc = coords.PhaseCenter.from_phase_dir(
            MS_RA, MS_DEC, field_id=0, n_fields=60
        )
        fov1.set_info(
            timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT, phase_center=pc
        )

        self.assertEqual(pc.provenance, coords.PROVENANCE_PHASE_DIR)
        self.assertEqual(pc.field_id, 0)
        self.assertEqual(pc.n_fields, 60)
        # set_info() binds the pointing to this sphere's site and time.
        self.assertEqual(pc.obstime, OBSTIME)
        # GeoLocation round-trips through geocentric coordinates (~1e-11
        # of float noise), so compare rather than require equality.
        np.testing.assert_allclose(pc.site, (LON, LAT, HEIGHT), atol=1e-6)

        # The stored pointing wins over the zenith derivation.
        self.assertEqual(fov1.phase_center_radec(), (MS_RA, MS_DEC))
        self.assertIs(fov1.get_phase_center(), pc)

        # ... which differs from what the zenith derivation says here
        # (0.045 deg: this test's rounded site/time are not test.ms's).
        ra_z, dec_z = coords.zenith_radec(LON, LAT, HEIGHT, obstime=OBSTIME)
        self.assertGreater(abs(ra_z - MS_RA), 0.01)

    def test_zenith_derived_when_nothing_stored(self):
        fov1 = make_fov()
        fov1.set_info(timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT)
        self.assertIsNone(fov1.phase_center)
        pc = fov1.get_phase_center()
        self.assertEqual(pc.provenance, coords.PROVENANCE_ZENITH)
        self.assertIsNone(pc.field_id)

    def test_set_info_without_phase_center_keeps_one(self):
        fov1 = make_fov()
        fov1.set_info(
            timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT,
            phase_center=coords.PhaseCenter.from_phase_dir(MS_RA, MS_DEC),
        )
        fov1.set_info(
            timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT,
        )
        self.assertIsNotNone(fov1.phase_center)
        self.assertEqual(fov1.phase_center_radec(), (MS_RA, MS_DEC))

    def test_phase_center_from_hdr(self):
        # The MS entry point: PHASE_DIR rides in the casa_read_ms header.
        self.assertIsNone(phase_center_from_hdr({}))
        self.assertIsNone(phase_center_from_hdr(None))
        self.assertIsNone(phase_center_from_hdr({"CRVAL1": MS_RA}))

        pc = phase_center_from_hdr(
            {"CRVAL1": MS_RA, "CRVAL2": MS_DEC},
            field_info={"field_id": 3, "n_fields": 60, "field_name": "x"},
        )
        self.assertEqual(pc.ra, MS_RA)
        self.assertEqual(pc.dec, MS_DEC)
        self.assertEqual(pc.provenance, coords.PROVENANCE_PHASE_DIR)
        self.assertEqual(pc.field_id, 3)
        self.assertEqual(pc.n_fields, 60)

        # No field context (e.g. TART .h5 input has no header at all): the
        # centre is still usable, just without the per-field detail.
        pc = phase_center_from_hdr({"CRVAL1": MS_RA, "CRVAL2": MS_DEC})
        self.assertIsNone(pc.field_id)
        self.assertIsNone(pc.n_fields)

    def test_unknown_provenance_rejected(self):
        with self.assertRaises(ValueError):
            coords.PhaseCenter(MS_RA, MS_DEC, provenance="guessing")


def _legacy_from_hdf(filename):
    """The pre-Phase-1 reader, verbatim from disko/fov/factory.py @ 1d2d7c9.

    It reads only fov_type/timestamp/geolocation/center from the
    'information' dataset, so a file written by the new code (which adds an
    optional 'phase_center' key) must load here: unknown keys are ignored.
    """
    from tart.util import utc

    ret = None
    with h5py.File(filename, "r") as h5f:
        info_string = np.bytes_(h5f['information'][0]).decode('UTF-8')
        info_json = json.loads(info_string)

        fov_type = info_json['fov_type']
        timestamp = utc.to_utc(
            datetime.datetime.fromisoformat(info_json['timestamp'])
        )
        geolocation = GeoLocation.from_json(info_json['geolocation']).loc
        centre = info_json['center']

        if fov_type == 'HealpixFoV':
            ret = HealpixFoV.from_hdf(h5f)
        elif fov_type == 'HealpixSubFoV':
            ret = HealpixSubFoV.from_hdf(h5f)
        elif fov_type == 'AdaptiveMeshFoV':
            ret = AdaptiveMeshFoV.from_hdf(h5f)
        else:
            raise RuntimeError(f"Unknown field of view class: {fov_type}.")

        ret.timestamp = timestamp
        ret.geolocation = geolocation
        ret.centre = centre
    return ret


def _write_legacy_hdf(path, fov1):
    """Write an HDF file exactly as disko did before Phase 1 of #10: the
    'information' JSON carries only the original four keys."""
    info_json = {
        'fov_type': type(fov1).__name__,
        'timestamp': fov1.timestamp.isoformat(),
        'geolocation': fov1.geolocation.to_json(),
        'center': 2,
        # deliberately no 'phase_center' key
    }
    dt = h5py.special_dtype(vlen=bytes)
    with h5py.File(path, "w") as h5f:
        dset = h5f.create_dataset('information', (1,), dtype=dt)
        dset[0] = json.dumps(info_json)
        h5f.create_dataset("nside", data=[fov1.nside])
        h5f.create_dataset("res_arcmin", data=[fov1.res_arcmin])
        h5f.create_dataset("theta", data=[fov1.theta])
        h5f.create_dataset("phi", data=[fov1.phi])
        h5f.create_dataset("radius_rad", data=[fov1.radius_rad])
        h5f.create_dataset("pixels", data=fov1.pixels)
        h5f.create_dataset("pixel_indices", data=fov1.pixel_indices)


class TestPhaseCentreHdf(unittest.TestCase):
    def _round_trip(self, fov1):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fov.h5")
            fov1.to_hdf(path)
            return fov.from_hdf(path)

    def test_round_trip_with_phase_center(self):
        fov1 = make_fov()
        fov1.set_info(
            timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT,
            phase_center=coords.PhaseCenter.from_phase_dir(
                MS_RA, MS_DEC, field_id=7, n_fields=60
            ),
        )
        fov2 = self._round_trip(fov1)

        pc1, pc2 = fov1.phase_center, fov2.phase_center
        self.assertIsNotNone(pc2)
        self.assertEqual(pc2.ra, pc1.ra)
        self.assertEqual(pc2.dec, pc1.dec)
        self.assertEqual(pc2.provenance, coords.PROVENANCE_PHASE_DIR)
        self.assertEqual(pc2.field_id, 7)
        self.assertEqual(pc2.n_fields, 60)
        self.assertEqual(pc2.obstime, pc1.obstime)
        np.testing.assert_allclose(pc2.site, pc1.site, atol=1e-6)
        # and the sphere answers with the stored value, not the zenith
        self.assertEqual(fov2.phase_center_radec(), (MS_RA, MS_DEC))
        self.assertTrue(np.allclose(fov1.pixels, fov2.pixels))

    def test_round_trip_without_phase_center(self):
        fov1 = make_fov()
        fov1.set_info(timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT)
        ra0, dec0 = fov1.phase_center_radec()

        fov2 = self._round_trip(fov1)
        self.assertIsNone(fov2.phase_center)
        ra1, dec1 = fov2.phase_center_radec()
        self.assertAlmostEqual(ra1, ra0, delta=1e-9)
        self.assertAlmostEqual(dec1, dec0, delta=1e-9)
        self.assertEqual(
            fov2.get_phase_center().provenance, coords.PROVENANCE_ZENITH
        )

    def test_round_trip_mesh_fov(self):
        # AdaptiveMeshFoV.to_hdf writes the same shared header, so it must
        # persist the phase centre too (recompute=False needs no gmsh).
        fov1 = AdaptiveMeshFoV(
            Resolution.from_arcmin(60.0), Resolution.from_arcmin(120.0),
            Resolution.from_rad(np.radians(45.0)), 0.0, 0.0, recompute=False,
        )
        fov1.npix = 4
        fov1.pixels = np.zeros(4)
        fov1.points = np.zeros((4, 2))
        fov1.simplices = np.array([[0, 1, 2], [1, 2, 3]])
        fov1.pixel_areas = np.ones(4) / 4
        fov1.l = np.zeros(4)  # noqa: E741
        fov1.m = np.zeros(4)
        fov1.n_minus_1 = np.zeros(4)
        fov1.el_r = np.zeros(4)
        fov1.az_r = np.zeros(4)
        fov1.set_info(
            timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT,
            phase_center=coords.PhaseCenter.from_phase_dir(
                MS_RA, MS_DEC, field_id=0, n_fields=1
            ),
        )
        fov2 = self._round_trip(fov1)
        self.assertIsNotNone(fov2.phase_center)
        self.assertEqual(
            (fov2.phase_center.ra, fov2.phase_center.dec), (MS_RA, MS_DEC)
        )
        self.assertEqual(fov2.phase_center.provenance,
                         coords.PROVENANCE_PHASE_DIR)

    def test_old_file_without_phase_center_key_loads(self):
        # Back-compat direction 1: a file written by the previous version
        # (no 'phase_center' key) loads and falls back to the zenith.
        fov1 = make_fov()
        fov1.set_info(timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old.h5")
            _write_legacy_hdf(path, fov1)
            with h5py.File(path, "r") as h5f:
                info = json.loads(np.bytes_(h5f['information'][0]).decode('UTF-8'))
                self.assertNotIn("phase_center", info)
                self.assertEqual(
                    sorted(info.keys()),
                    ["center", "fov_type", "geolocation", "timestamp"],
                )
            fov2 = fov.from_hdf(path)
        self.assertIsNone(fov2.phase_center)
        self.assertTrue(np.allclose(fov1.pixels, fov2.pixels))
        self.assertEqual(
            fov2.get_phase_center().provenance, coords.PROVENANCE_ZENITH
        )

    def test_new_file_loads_with_legacy_reader(self):
        # Back-compat direction 2: a file written now still loads through
        # the old factory code, which ignores keys it does not know.
        fov1 = make_fov()
        fov1.set_info(
            timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT,
            phase_center=coords.PhaseCenter.from_phase_dir(
                MS_RA, MS_DEC, field_id=0, n_fields=60
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "new.h5")
            fov1.to_hdf(path)
            with h5py.File(path, "r") as h5f:
                info = json.loads(np.bytes_(h5f['information'][0]).decode('UTF-8'))
                self.assertIn("phase_center", info)  # the file is new-format
                for key in ("fov_type", "timestamp", "geolocation", "center"):
                    self.assertIn(key, info)  # ... and all old keys remain
            legacy = _legacy_from_hdf(path)
        self.assertTrue(np.allclose(legacy.pixels, fov1.pixels))
        # The old reader has no concept of a phase centre attribute set
        # from the file, but nothing about the load failed.


@unittest.skipUnless(os.path.isdir(MS_PATH), "test_data/test.ms fixture missing")
class TestMsPhaseDirOracle(unittest.TestCase):
    """End-to-end MS -> PHASE_DIR -> FoV, with the zenith as the oracle.

    test_data/test.ms holds 60 fields with 60 distinct PHASE_DIRs; these
    are zenith-pointed TART observations, so for each field the stored
    PHASE_DIR must land almost exactly on the centre derived from that
    field's own timestamp and the array location. Measured separations are
    0.000007 deg (see the phase-1 report); the tolerance below has >100x
    margin while still proving the two agree far better than one image
    pixel (~0.08 deg).
    """

    def _read_field(self, field):
        geo = get_array_location(MS_PATH)
        (_, _, _, _, _, hdr, tstamp, _, _, field_info) = casa_read_ms(
            ms_file=MS_PATH, num_vis=30, ms_column="DATA",
            angular_resolution=3.0, channel=0, field_id=field,
            rng=np.random.default_rng(7),
        )
        fov1 = make_fov()
        fov1.set_info(
            timestamp=tstamp, lon=float(geo["lon"]), lat=float(geo["lat"]),
            height=float(geo["height"]),
            phase_center=phase_center_from_hdr(hdr, field_info=field_info),
        )
        return fov1

    def test_phase_dir_agrees_with_zenith_derived(self):
        seps = []
        ras = []
        for field in (0, 59):
            fov1 = self._read_field(field)
            pc = fov1.get_phase_center()
            self.assertEqual(pc.provenance, coords.PROVENANCE_PHASE_DIR)
            self.assertEqual(pc.field_id, field)
            self.assertEqual(pc.n_fields, 60)
            # The stored centre is what the sphere reports.
            self.assertEqual(fov1.phase_center_radec(), (pc.ra, pc.dec))

            lon, lat, height, obstime = fov1.site_obstime()
            ra_z, dec_z = coords.zenith_radec(lon, lat, height, obstime=obstime)
            dra = (ra_z - pc.ra + 180) % 360 - 180
            sep = np.hypot(
                dra * np.cos(np.radians(pc.dec)), dec_z - pc.dec
            )
            seps.append(sep)
            ras.append(pc.ra)

        # Oracle: zenith-pointed observation -> the two agree closely.
        for sep in seps:
            self.assertLess(sep, 0.001, f"separation {sep} deg too large")

        # But the fields genuinely point elsewhere: phase centres are
        # per-field, not one centre for the whole MS (field 59 is
        # ~0.25 deg in RA from field 0).
        self.assertGreater(ras[1] - ras[0], 0.1)


class TestFrameAwarePlacement(unittest.TestCase):
    def setUp(self):
        self.fov1 = make_fov()
        self.fov1.set_info(
            timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT,
            phase_center=coords.PhaseCenter.from_phase_dir(MS_RA, MS_DEC),
        )

    def test_index_of_same_pixel_in_both_frames(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            el = rng.uniform(np.radians(65.0), np.radians(88.0))
            az = rng.uniform(-np.pi, np.pi)
            ra, dec = coords.elaz_to_radec(
                el, az, LON, LAT, HEIGHT, obstime=OBSTIME
            )
            self.assertEqual(
                self.fov1.index_of(el, az),
                self.fov1.index_of(ra, dec, frame="icrs"),
            )

    def test_index_of_defaults_to_elaz(self):
        # The pre-existing signature and behaviour are unchanged.
        el = np.radians(80.0)
        az = np.radians(10.0)
        self.assertEqual(
            self.fov1.index_of(el, az), self.fov1.index_of(el, az, frame="elaz")
        )

    def test_index_of_unknown_frame(self):
        with self.assertRaises(ValueError):
            self.fov1.index_of(1.0, 2.0, frame="galactic")

    def test_source_elaz_in_both_frames(self):
        el_r, az_r = np.radians(60.0), np.radians(30.0)
        elaz_src = _ElazSource(el_r, az_r)
        ra, dec = coords.elaz_to_radec(el_r, az_r, LON, LAT, HEIGHT, obstime=OBSTIME)
        cel_src = _CelestialSource(ra, dec)

        self.assertEqual(self.fov1.source_elaz(elaz_src), (el_r, az_r))
        el2, az2 = self.fov1.source_elaz(cel_src)
        self.assertAlmostEqual(el2, el_r, delta=1e-9)
        d_az = (az2 - az_r + np.pi) % (2 * np.pi) - np.pi
        self.assertAlmostEqual(d_az, 0.0, delta=1e-9)

    def test_source_without_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            self.fov1.source_elaz(object())

    def test_svg_placement_identical_in_both_frames(self):
        # healpix_sphere to_svg:439 places sources through source_elaz(),
        # so the same source given in elaz or celestial coordinates must
        # draw the same picture. The celestial round trip is only accurate
        # to ~1e-10 rad, which shows up in the rotation angle's last
        # digits, so compare the drawing with the angle rounded.
        import re

        def canonical(data):
            return re.sub(
                rb'rotate\(-?\d+(?:\.\d+)?,', b'rotate(<ANGLE>,', data
            )

        el_r, az_r = np.radians(60.0), np.radians(30.0)
        ra, dec = coords.elaz_to_radec(el_r, az_r, LON, LAT, HEIGHT, obstime=OBSTIME)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "elaz.svg")
            path_b = os.path.join(tmpdir, "celestial.svg")
            self.fov1.to_svg(path_a, src_list=[_ElazSource(el_r, az_r)],
                             show_grid=True, title=None)
            self.fov1.to_svg(path_b, src_list=[_CelestialSource(ra, dec)],
                             show_grid=True, title=None)
            with open(path_a, "rb") as f:
                svg_a = f.read()
            with open(path_b, "rb") as f:
                svg_b = f.read()
        self.assertIn(b"ellipse", svg_a)  # the source was actually drawn
        self.assertEqual(canonical(svg_a), canonical(svg_b))
        angle_a = float(re.search(rb'rotate\((-?\d+(?:\.\d+)?)', svg_a).group(1))
        angle_b = float(re.search(rb'rotate\((-?\d+(?:\.\d+)?)', svg_b).group(1))
        self.assertAlmostEqual(angle_a, angle_b, delta=1e-6)
        self.assertAlmostEqual(angle_a, -30.0, delta=1e-6)

    def test_to_fits_uses_stored_phase_center(self):
        from astropy.io import fits

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.fits")
            self.fov1.to_fits(path, title="phase centre test")
            hdr = fits.getheader(path)
        self.assertEqual(hdr["CRVAL1"], MS_RA)
        self.assertEqual(hdr["CRVAL2"], MS_DEC)
        self.assertEqual(hdr["RADESYS"], "ICRS")
        self.assertEqual(hdr["CTYPE1"], "RA---SIN")
        self.assertEqual(hdr["CTYPE2"], "DEC--SIN")


if __name__ == "__main__":
    unittest.main()
