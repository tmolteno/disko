#
# Copyright Tim Molteno 2026 tim@elec.ac.nz
# License: GPLv3
"""
Coordinate-aware overplotting (issue #7, issue #10 Phase 2).

The symptom (#7): sources were placed purely by el/az, so a
phase-steered MS overplotted as if the grid were zenith-centred. The
grid and its FITS WCS are centred on the FoV's phase centre (l=m=0 at
the centre pixel, CRVAL = the phase centre), so a source's counterpart
appears on the image at its l,m about THAT direction. These tests pin:

- the phase-steering case built from real MS data: test_data/test.ms
  holds 60 fields with 60 distinct PHASE_DIRs, and field 59's pointing
  is ~0.253 deg in RA from field 0's — so a FoV carrying field 59's
  PHASE_DIR at an epoch where the zenith sits at field 0's pointing is
  genuinely off-zenith, and a source AT that phase centre must land on
  the image centre while the same source lands elsewhere on a
  zenith-centred grid. Pre-Phase-2 code (placement via source_elaz only)
  puts the marker 0.25 deg off centre, so it fails this test.

- the frame bound (the "frame-consistency check" of #10 Phase 2). The
  overplot is ICRS l,m about the FoV's phase centre (the frame source
  catalogues live in and the frame the FITS WCS claims, RADESYS=ICRS).
  Phase 2 left the grid in its geolocated decomposition, so the two
  differed by the pole angle (ITRS z-axis 0.151 deg from the ICRS pole
  in 2026; Phase 1 measured max direction-cosine disagreement 0.0023)
  and asserted that as a bound. Issue #26 has since fixed it — the
  image path (to_fits grid, to_svg grid and markers) now decomposes
  everything through FoV.icrs_lmn_from_elaz, see
  disko/tests/test_image_frame.py — so what remains asserted here is
  the frame difference itself: the image frame and the raw geolocated
  el/az of the same direction still differ by the pole rotation alone
  (< 0.25 deg), while the overplot's radial geometry (angle from the
  phase centre) and its elaz<->celestial round trip agree to far
  better than 0.01 deg.
"""

import datetime
import os
import re
import tempfile
import unittest

import numpy as np

from disko import coords
from disko.healpix_sphere import HealpixSubFoV
from disko.ms_helper import casa_read_ms, get_array_location, phase_center_from_hdr
from disko.sphere import elaz2lmn

LON, LAT, HEIGHT = 170.5, -45.5, 42.0
OBSTIME = datetime.datetime(2021, 3, 25, 20, 49, 23, tzinfo=datetime.timezone.utc)

MS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "test.ms")

# Field 0's pointing of test_data/test.ms (source J202414-455517); also
# its own epoch's zenith, to within 0.001 deg (Phase 1's MS oracle).
MS_RA0, MS_DEC0 = 306.0584, -45.9215


def make_fov(res_arcmin=60.0):
    return HealpixSubFoV(
        res_arcmin=res_arcmin, theta=0.0, phi=0.0, radius_rad=np.radians(30)
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


def sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation (degrees) of two ICRS directions."""
    ra1, dec1, ra2, dec2 = np.radians([ra1, dec1, ra2, dec2])
    sin_half = np.sin((dec2 - dec1) / 2) ** 2 + (
        np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    )
    return float(np.degrees(2 * np.arcsin(np.sqrt(sin_half))))


def angle_between_elaz(el1, az1, el2, az2):
    """Angular separation (degrees) of two geolocated directions."""
    dot = (
        np.sin(el1) * np.sin(el2)
        + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    )
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


@unittest.skipUnless(os.path.isdir(MS_PATH), "test_data/test.ms fixture missing")
class TestPhaseSteeredOverplot(unittest.TestCase):
    """The #7 symptom, built from the phase-steering case in test.ms.

    Field 59 of test.ms points at RA 306.3114 deg while field 0 (and so
    field 0's epoch's zenith) sits at RA 306.0584 deg — 0.253 deg apart.
    A FoV that carries field 59's PHASE_DIR but is observed at field 0's
    epoch is therefore a genuinely phase-steered pointing, exactly the
    case #7's second screenshot shows.
    """

    @classmethod
    def setUpClass(cls):
        geo = get_array_location(MS_PATH)

        def read_field(field):
            (_, _, _, _, _, hdr, tstamp, _, _, field_info) = casa_read_ms(
                ms_file=MS_PATH, num_vis=10, ms_column="DATA",
                angular_resolution=3.0, channel=0, field_id=field,
                rng=np.random.default_rng(7),
            )
            return tstamp, phase_center_from_hdr(hdr, field_info=field_info)

        cls.lon = float(geo["lon"])
        cls.lat = float(geo["lat"])
        cls.height = float(geo["height"])
        cls.t0, _ = read_field(0)          # field 0's epoch: zenith ~ RA0
        _, pc59 = read_field(59)           # the non-default field's PHASE_DIR
        cls.pc59 = pc59

    def _steered_and_zenith_fovs(self):
        # The phase-steered grid: field 59's pointing applied at field 0's
        # epoch (a PhaseCenter read from a non-default MS field, genuinely
        # off the zenith here). set_info() binds obstime/site into the
        # stored centre.
        steered = make_fov()
        steered.set_info(
            timestamp=self.t0, lon=self.lon, lat=self.lat,
            height=self.height, phase_center=self.pc59,
        )
        # The same site and epoch, but no stored centre: zenith-derived.
        zenith = make_fov()
        zenith.set_info(
            timestamp=self.t0, lon=self.lon, lat=self.lat, height=self.height,
        )
        return steered, zenith

    def test_phase_dir_is_genuinely_off_zenith_here(self):
        # Guard for the construction itself: field 59's PHASE_DIR is a
        # quarter of a degree from field 0's epoch's zenith, so the
        # assertions below are discriminating (well over three image
        # pixels at any render scale used here).
        steered, _ = self._steered_and_zenith_fovs()
        ra_z, dec_z = coords.zenith_radec(
            self.lon, self.lat, self.height, obstime=self.t0
        )
        self.assertEqual(steered.phase_center_radec(), (self.pc59.ra, self.pc59.dec))
        self.assertEqual(self.pc59.field_id, 59)
        self.assertEqual(self.pc59.n_fields, 60)
        # (0.253 deg of RA at dec -45.92 -> 0.176 deg on the sky.)
        off = sep_deg(self.pc59.ra, self.pc59.dec, ra_z, dec_z)
        self.assertGreater(off, 0.15)
        self.assertLess(off, 0.2)

    def test_source_at_phase_centre_lands_on_image_centre(self):
        # A source AT the phase centre direction is what a phase-steered
        # image shows at its centre pixel (l=m=0): its l,m about the
        # stored centre are (0, 0, 1) -> drawn at the grid centre.
        steered, zenith_fov = self._steered_and_zenith_fovs()
        src = _CelestialSource(self.pc59.ra, self.pc59.dec)

        el_a, az_a = steered.source_draw_elaz(src)
        self.assertAlmostEqual(el_a, np.pi / 2, delta=1e-9)
        self.assertAlmostEqual(az_a, 0.0, delta=1e-9)
        # ... and that is literally the centre pixel of the grid.
        self.assertEqual(
            steered.index_of(el_a, az_a), steered.index_of(np.pi / 2, 0.0)
        )

        # Pre-Phase-2 placement (pure el/az) would put the same source
        # 0.25 deg away from the centre: this is an assertion that fails
        # on pre-Phase-2 code.
        old_el, old_az = steered.source_elaz(src)
        self.assertGreater(angle_between_elaz(el_a, az_a, old_el, old_az), 0.15)

        # The same source on a zenith-centred grid lands elsewhere: off
        # centre by exactly the phase-steering offset.
        el_z, az_z = zenith_fov.source_draw_elaz(src)
        landing = angle_between_elaz(el_a, az_a, el_z, az_z)
        self.assertGreater(landing, 0.15)
        ra_z, dec_z = coords.zenith_radec(
            self.lon, self.lat, self.height, obstime=self.t0
        )
        steering = sep_deg(self.pc59.ra, self.pc59.dec, ra_z, dec_z)
        self.assertAlmostEqual(landing, steering, delta=0.001)

    def test_svg_marker_position(self):
        # End-to-end through the drawing itself: to_svg places the marker
        # at the image centre on the phase-steered grid, and off centre
        # on the zenith-centred one. (On pre-Phase-2 code the marker
        # sits ~12 px from the centre on the phase-steered grid.)
        steered, zenith_fov = self._steered_and_zenith_fovs()
        src = _CelestialSource(self.pc59.ra, self.pc59.dec)
        centre = 2000  # PlotCoords centre: h=4000 -> int(h/2)

        def first_ellipse(fov):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "out.svg")
                fov.to_svg(path, src_list=[src], show_grid=True, title=None)
                with open(path, "r", encoding="utf-8") as f:
                    data = f.read()
            match = re.search(r'<ellipse cx="(-?\d+)" cy="(-?\d+)"', data)
            self.assertIsNotNone(match, "the source marker was not drawn")
            return int(match.group(1)), int(match.group(2))

        self.assertEqual(first_ellipse(steered), (centre, centre))
        zx, zy = first_ellipse(zenith_fov)
        offset_px = np.hypot(zx - centre, zy - centre)
        # sin(0.176 deg) * scale(4000/sin(30 deg)/2.1) ~ 11.7 px.
        self.assertGreater(offset_px, 8)


class TestOverplotFrameBound(unittest.TestCase):
    """The frame-consistency check of #10 Phase 2, post issue #26.

    Chosen frame: the overplot is ICRS l,m about the FoV's phase centre
    (the frame source catalogues live in and the frame the FITS WCS
    claims, RADESYS=ICRS). Since issue #26 the drawn grid shares that
    frame too (to_svg places both grid polygons and markers from
    icrs_lmn_from_elaz / source_lmn, so they agree exactly — asserted
    in disko/tests/test_image_frame.py), so what is left to bound here
    is the relationship between that image frame and the raw
    geolocated el/az of the same direction: they differ by the pole
    angle alone.
    """

    def _fov(self, phase_center=None):
        fov = make_fov()
        fov.set_info(timestamp=OBSTIME, lon=LON, lat=LAT, height=HEIGHT,
                     phase_center=phase_center)
        return fov

    def test_elaz_and_celestial_markers_agree_sub_001deg(self):
        # The part of the overplot that CAN be exact: the same source
        # given in el/az or in the equivalent celestial coordinates
        # lands on the same spot of the drawn grid to far better than
        # 0.01 deg.
        fov = self._fov(coords.PhaseCenter.from_phase_dir(
            MS_RA0, MS_DEC0, field_id=0, n_fields=60))
        rng = np.random.default_rng(11)
        worst = 0.0
        for _ in range(30):
            el = rng.uniform(np.radians(20.0), np.radians(88.0))
            az = rng.uniform(-np.pi, np.pi)
            ra, dec = coords.elaz_to_radec(el, az, LON, LAT, HEIGHT,
                                           obstime=OBSTIME)
            marker_a = fov.source_draw_elaz(_ElazSource(el, az))
            marker_b = fov.source_draw_elaz(_CelestialSource(ra, dec))
            worst = max(worst, angle_between_elaz(*marker_a, *marker_b))
        self.assertLess(worst, 0.01)

    def test_radial_offset_matches_celestial_offset_sub_001deg(self):
        # The overplot's radial geometry is exact: the marker's angular
        # distance from the grid centre equals the great-circle distance
        # from the source to the phase centre (radec_to_lmn's n is that
        # cosine, and lmn_to_elaz preserves n), to far better than
        # 0.01 deg — regardless of the pole rotation, which only swings
        # the marker's position angle about the centre.
        fov = self._fov(coords.PhaseCenter.from_phase_dir(
            MS_RA0, MS_DEC0, field_id=0, n_fields=60))
        src = _CelestialSource(
            MS_RA0 + 0.4 / np.cos(np.radians(MS_DEC0)), MS_DEC0 + 0.3
        )
        el, az = fov.source_draw_elaz(src)
        off_grid = angle_between_elaz(np.pi / 2, 0.0, el, az)
        off_sky = sep_deg(src.ra, src.dec, MS_RA0, MS_DEC0)
        self.assertLess(abs(off_grid - off_sky), 0.01)

    def test_image_frame_and_geolocated_differ_only_by_pole_angle(self):
        # The image frame (ICRS l,m about the phase centre, what the
        # FITS WCS and the drawn grid now both use) and the raw
        # geolocated el/az of the same direction differ only by the
        # pole-angle rotation: Phase 1 measured max 0.0023 in direction
        # cosine; here the same quantity is asserted as an angle. This
        # is a property of the two FRAMES, not a mixing of them inside
        # the image path — since issue #26 grid and marker share one
        # frame (asserted in test_image_frame.py), so the remaining
        # bound is image-frame placement vs physical el/az. The bound
        # is asserted with the rationale rather than removed: a
        # physically placed marker (source_elaz) is what the healpy
        # renderer uses, and it must land a pole angle away from the
        # image-frame placement.
        fov = self._fov()
        pc = fov.get_phase_center()
        self.assertEqual(pc.provenance, coords.PROVENANCE_ZENITH)
        rng = np.random.default_rng(42)
        worst_deg = 0.0
        worst_lmn = 0.0
        for _ in range(100):
            el = rng.uniform(np.radians(5.0), np.radians(88.0))
            az = rng.uniform(-np.pi, np.pi)
            ra, dec = coords.elaz_to_radec(el, az, LON, LAT, HEIGHT,
                                           obstime=OBSTIME)
            # The image-frame placement (ICRS l,m about the phase
            # centre, expressed as an el/az direction) ...
            marker = fov.source_draw_elaz(_CelestialSource(ra, dec))
            # ... vs the raw geolocated el/az of the same direction.
            truth = fov.source_elaz(_CelestialSource(ra, dec))
            worst_deg = max(worst_deg, angle_between_elaz(*marker, *truth))
            # The same disagreement in direction cosines (Phase 1's
            # 0.0023 measurement, recomputed through the marker path).
            l_c, m_c, n_c = coords.radec_to_lmn(ra, dec, pc)
            l_t, m_t, n_t = elaz2lmn(truth[0], truth[1])
            worst_lmn = max(worst_lmn, abs(l_c - l_t), abs(m_c - m_t),
                            abs(n_c - n_t))
        # The asserted bound: the pole angle (0.151 deg in 2026, so
        # <= 0.152 deg of angular swing for any direction) ...
        self.assertLess(worst_deg, 0.25)
        self.assertLess(worst_lmn, 0.005)
        # ... and it is a real, measurable rotation (not accidental
        # agreement): over 100 random directions it shows up clearly.
        self.assertGreater(worst_lmn, 0.001)


if __name__ == "__main__":
    unittest.main()
