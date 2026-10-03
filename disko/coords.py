# Copyright Tim Molteno 2026 tim@elec.ac.nz
# License: GPLv3
"""
Celestial coordinate support for DiSkO fields of view (issue #10, Phase 1).

This module owns the transforms between the three frames a FoV lives in:

- the geolocated frame: elevation / azimuth about the observer's zenith
  (``el``/``az``, radians, azimuth clockwise from north),
- the celestial frame: ICRS right ascension / declination (degrees),
- the image grid: direction cosines ``l, m`` about the phase centre
  (the convention of :func:`disko.sphere.elaz2lmn`, so ``l`` is positive
  toward east and ``m`` toward north).

:class:`PhaseCenter` is the first-class record of a FoV's celestial
pointing: RA/Dec + obstime + geolocation + an explicit provenance telling
whether the value was read from an MS ``PHASE_DIR`` or derived from the
observer's zenith.

Design rules:

- astropy is imported lazily inside each function so that merely importing
  disko stays cheap.
- The IERS offline policy introduced by the #14 FITS fix lives exactly
  once, in :func:`offline_iers`; every transform that may consult the IERS
  tables runs inside it, so an offline machine never blocks on a download
  and the settings are always restored.
"""

import contextlib
import datetime
import logging

import numpy as np

logger = logging.getLogger(__name__)

DEG = np.pi / 180.0

# Where a phase centre came from. These strings are persisted in HDF files
# (see FoV.to_hdf_header), so keep them stable.
PROVENANCE_PHASE_DIR = "ms_phase_dir"  # read from an MS FIELD PHASE_DIR column
PROVENANCE_ZENITH = "zenith"  # derived from the observer's zenith (#14 fix)


@contextlib.contextmanager
def offline_iers():
    '''
        Shared IERS policy for every transform that may consult the IERS
        Earth-orientation tables.

        Earth orientation moves a phase centre by a small fraction of an
        image pixel, and the tables may not be downloadable (offline CI),
        so never let a stale table or a network fetch stop coordinate work.
        Introduced by the #14 FITS fix as a scoped override; centralised
        here so there is exactly one such override, always restored.
    '''
    from astropy.utils import iers

    saved = (iers.conf.auto_download, iers.conf.auto_max_age)
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    try:
        yield
    finally:
        iers.conf.auto_download, iers.conf.auto_max_age = saved


def as_datetime(obstime):
    '''
        Normalise an obstime to a timezone-aware UTC datetime (None passes
        through). Accepts datetime (naive is taken as UTC) or anything
        astropy can parse (string, astropy Time).
    '''
    if obstime is None:
        return None
    if isinstance(obstime, datetime.datetime):
        if obstime.tzinfo is None:
            return obstime.replace(tzinfo=datetime.timezone.utc)
        return obstime.astimezone(datetime.timezone.utc)
    from astropy.time import Time

    return Time(obstime).to_datetime(timezone=datetime.timezone.utc)


def _as_float(value, unit):
    '''Read a plain float from an astropy Quantity (of `unit`) or a number.'''
    if hasattr(value, "to_value"):
        return float(value.to_value(unit))
    return float(value)


def site_from_geolocation(geolocation):
    '''
        (lon_deg, lat_deg, height_m) from whatever a FoV carries as its
        geolocation: the GeoLocation wrapper set by FoV.set_info(), the
        astropy EarthLocation stored by fov.from_hdf(), a plain dict, or
        None (which means the origin, matching FoV's constructor default).
    '''
    if geolocation is None:
        return 0.0, 0.0, 0.0
    geo = getattr(geolocation, "loc", geolocation)  # unwrap GeoLocation
    if isinstance(geo, dict):
        return (
            float(geo.get("lon", 0.0)),
            float(geo.get("lat", 0.0)),
            float(geo.get("height", 0.0)),
        )
    return (
        _as_float(getattr(geo, "lon", 0.0), "deg"),
        _as_float(getattr(geo, "lat", 0.0), "deg"),
        _as_float(getattr(geo, "height", 0.0), "m"),
    )


def _altaz_frame(lon, lat, height=0.0, obstime=None):
    '''The astropy AltAz frame of an observer at this site and time.'''
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation
    from astropy.time import Time

    if obstime is None:
        obstime = datetime.datetime.now(datetime.timezone.utc)
    location = EarthLocation.from_geodetic(
        lon=lon * u.deg, lat=lat * u.deg, height=height * u.m
    )
    return AltAz(obstime=Time(as_datetime(obstime)), location=location)


def zenith_radec(lon, lat, height=0.0, obstime=None):
    '''
        ICRS (ra_deg, dec_deg) of the zenith seen from this site at this
        time — the #14 derivation, shared by FoV.phase_center_radec().
    '''
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    frame = _altaz_frame(lon, lat, height=height, obstime=obstime)
    with offline_iers():
        zenith = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame).transform_to(
            "icrs"
        )
    return float(zenith.ra.deg), float(zenith.dec.deg)


def elaz_to_radec(el, az, lon, lat, height=0.0, obstime=None):
    '''
        ICRS (ra_deg, dec_deg) of a direction given as el/az (radians) at
        this site and time. Azimuth is clockwise from north, matching
        :func:`disko.sphere.elaz2hp`.

        Vectorised: el/az may be arrays (whole pixel grids or grid
        corners, issue #26), in which case two arrays come back; scalar
        input still returns two floats.
    '''
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    frame = _altaz_frame(lon, lat, height=height, obstime=obstime)
    with offline_iers():
        sky = SkyCoord(
            alt=np.asarray(el, dtype=float) / DEG * u.deg,
            az=np.asarray(az, dtype=float) / DEG * u.deg,
            frame=frame,
        ).transform_to("icrs")
    ra = np.asarray(sky.ra.deg, dtype=float)
    dec = np.asarray(sky.dec.deg, dtype=float)
    if ra.ndim == 0:
        return float(ra), float(dec)
    return ra, dec


def radec_to_elaz(ra, dec, lon, lat, height=0.0, obstime=None):
    '''
        (el, az) in radians of an ICRS direction (degrees) seen from this
        site at this time. Azimuth is wrapped to [-pi, pi). (Scalar.)
    '''
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    frame = _altaz_frame(lon, lat, height=height, obstime=obstime)
    with offline_iers():
        sky = SkyCoord(
            ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs"
        ).transform_to(frame)
    el = float(sky.alt.deg) * DEG
    az = float(sky.az.deg) * DEG
    az = (az + np.pi) % (2 * np.pi) - np.pi
    return el, az


def radec_to_lmn(ra, dec, phase_center):
    '''
        Direction cosines (l, m, n) of celestial directions (degrees)
        relative to a phase centre: the gnomonic projection about
        ``phase_center``, with ``l`` positive toward east and ``m`` toward
        north (the convention of :func:`disko.sphere.elaz2lmn`, so (0, 0, 1)
        is the phase centre). Vectorised: ra/dec may be arrays.
    '''
    ra0 = phase_center.ra * DEG
    dec0 = phase_center.dec * DEG
    ra = np.asarray(ra, dtype=float) * DEG
    dec = np.asarray(dec, dtype=float) * DEG

    cos_dec = np.cos(dec)
    sin_dec = np.sin(dec)
    cos_dec0 = np.cos(dec0)
    sin_dec0 = np.sin(dec0)
    cos_dra = np.cos(ra - ra0)

    l = cos_dec * np.sin(ra - ra0)  # noqa: E741
    m = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra
    n = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra
    return l, m, n


def lmn_to_radec(l, m, phase_center):  # noqa: E741
    '''
        Inverse of :func:`radec_to_lmn`: celestial (ra_deg, dec_deg) of grid
        positions (l, m) about a phase centre. ``n`` is taken positive (the
        hemisphere facing the phase centre). Vectorised.
    '''
    ra0 = phase_center.ra * DEG
    dec0 = phase_center.dec * DEG
    l = np.asarray(l, dtype=float)  # noqa: E741
    m = np.asarray(m, dtype=float)

    n = np.sqrt(np.clip(1.0 - l * l - m * m, 0.0, None))
    sin_dec = m * np.cos(dec0) + n * np.sin(dec0)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    ra = ra0 + np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def lmn_to_elaz(l, m, n):  # noqa: E741
    '''
        The geolocated (el_r, az_r) whose direction cosines — in the sense
        of :func:`disko.sphere.elaz2lmn` — are (l, m, n): the exact
        inverse of elaz2lmn (el = asin(n), az = atan2(l, m), valid for the
        visible hemisphere |el| <= pi/2).

        This is how a direction computed in the celestial frame (see
        :func:`radec_to_lmn`) is drawn on the grid, whose pixel positions
        come from the geolocated decomposition (issue #7, issue #10 Phase
        2): marker and grid pixels then share one drawing transform.
    '''
    el = np.arcsin(np.clip(float(n), -1.0, 1.0))
    az = np.arctan2(float(l), float(m))
    return el, az


def source_radec(source, lon, lat, height=0.0, obstime=None):
    '''
        ICRS (ra_deg, dec_deg) of a source placed on a FoV, in either
        frame.

        Sources that carry celestial coordinates (``ra``/``dec`` in
        degrees) are used as they are; sources given as el/az
        (``el_r``/``az_r``, e.g. the TART catalog objects) are converted
        with this site and time — the inverse of
        :func:`source_elaz_rad`, so a source can be evaluated in the
        celestial frame where the overplot lives (issue #7, issue #10
        Phase 2: "TART sources arrive as elaz -> convert with
        obstime/site").
    '''
    el = getattr(source, "el_r", None)
    az = getattr(source, "az_r", None)
    if el is not None and az is not None:
        return elaz_to_radec(el, az, lon, lat, height=height, obstime=obstime)

    ra = getattr(source, "ra", None)
    dec = getattr(source, "dec", None)
    if ra is None or dec is None:
        raise ValueError(
            f"Source {source!r} has neither elaz (el_r/az_r) nor celestial "
            "(ra/dec) coordinates"
        )
    return float(ra), float(dec)


def source_elaz_rad(source, lon, lat, height=0.0, obstime=None):
    '''
        (el_r, az_r) of a source placed on a FoV, in either frame.

        Sources that carry their own el/az (``el_r``/``az_r``, e.g. the
        TART catalog objects) are used as they are; sources given in
        celestial coordinates (``ra``/``dec`` in degrees) are converted
        with this site and time so they land on the right pixel of a
        phase-steered grid.
    '''
    el = getattr(source, "el_r", None)
    az = getattr(source, "az_r", None)
    if el is not None and az is not None:
        return float(el), float(az)

    ra = getattr(source, "ra", None)
    dec = getattr(source, "dec", None)
    if ra is None or dec is None:
        raise ValueError(
            f"Source {source!r} has neither elaz (el_r/az_r) nor celestial "
            "(ra/dec) coordinates"
        )
    el, az = radec_to_elaz(ra, dec, lon, lat, height=height, obstime=obstime)
    return float(el), float(az)


class PhaseCenter(object):
    '''
        The celestial pointing of a FoV: ICRS RA/Dec (degrees) plus the
        obstime and geolocation it applies at, and an explicit provenance:

        - PROVENANCE_PHASE_DIR: read from an MS FIELD PHASE_DIR column,
        - PROVENANCE_ZENITH: derived from the observer's zenith (#14 fix).

        The provenance survives HDF round trips so a consumer can always
        tell a pointing that was observed from one that was inferred.
    '''

    PROVENANCE_PHASE_DIR = PROVENANCE_PHASE_DIR
    PROVENANCE_ZENITH = PROVENANCE_ZENITH

    def __init__(self, ra, dec, provenance, obstime=None, geolocation=None,
                 field_id=None, n_fields=None):
        if provenance not in (PROVENANCE_PHASE_DIR, PROVENANCE_ZENITH):
            raise ValueError(f"Unknown phase centre provenance: {provenance!r}")
        self.ra = float(ra) % 360.0
        self.dec = float(dec)
        self.provenance = provenance
        self.obstime = as_datetime(obstime)
        # GeoLocation / EarthLocation / None; None means "not bound yet"
        # (FoV.set_info() fills it in from the sphere it is attached to).
        self.geolocation = geolocation
        # MS context (issue #10, Phase 1): a phase centre read from an MS is
        # the PHASE_DIR of ONE FIELD, not of "the" MS — an MS can hold many
        # fields with different pointings (test_data/test.ms has 60). These
        # record which field this pointing came from; None outside MS input.
        self.field_id = field_id
        self.n_fields = n_fields

    @classmethod
    def from_phase_dir(cls, ra, dec, obstime=None, geolocation=None,
                       field_id=None, n_fields=None):
        '''
            A pointing read from an MS FIELD PHASE_DIR column (degrees,
            treated as ICRS, matching what to_fits already writes into the
            FITS header for MS input). `field_id`/`n_fields` say which of
            the MS's fields this centre belongs to (field 0 by default).
        '''
        return cls(
            ra, dec, PROVENANCE_PHASE_DIR, obstime=obstime,
            geolocation=geolocation, field_id=field_id, n_fields=n_fields,
        )

    @classmethod
    def from_zenith(cls, ra, dec, obstime=None, geolocation=None):
        '''A pointing derived from the observer's zenith (the #14 fix).'''
        return cls(
            ra, dec, PROVENANCE_ZENITH, obstime=obstime, geolocation=geolocation
        )

    @property
    def site(self):
        '''(lon_deg, lat_deg, height_m) this pointing applies at.'''
        return site_from_geolocation(self.geolocation)

    def to_dict(self):
        '''JSON-ready form, stored in the FoV HDF 'information' dataset.'''
        geo = self.geolocation
        geo_json = None
        if geo is not None:
            lon, lat, height = site_from_geolocation(geo)
            geo_json = {"lon": lon, "lat": lat, "height": height}
        return {
            "ra": self.ra,
            "dec": self.dec,
            "provenance": self.provenance,
            "obstime": self.obstime.isoformat() if self.obstime is not None else None,
            "geolocation": geo_json,
            "field_id": self.field_id,
            "n_fields": self.n_fields,
        }

    @classmethod
    def from_dict(cls, data):
        '''Inverse of to_dict (geolocation comes back as an EarthLocation).'''
        obstime = data.get("obstime")
        if obstime is not None:
            obstime = datetime.datetime.fromisoformat(obstime)
        geolocation = None
        geo_json = data.get("geolocation")
        if geo_json is not None:
            import astropy.units as u
            from astropy.coordinates import EarthLocation

            geolocation = EarthLocation.from_geodetic(
                lon=geo_json["lon"] * u.deg,
                lat=geo_json["lat"] * u.deg,
                height=geo_json.get("height", 0.0) * u.m,
            )
        return cls(
            ra=data["ra"],
            dec=data["dec"],
            provenance=data["provenance"],
            obstime=obstime,
            geolocation=geolocation,
            field_id=data.get("field_id"),
            n_fields=data.get("n_fields"),
        )

    def __repr__(self):
        field = ""
        if self.field_id is not None:
            field = f", field={self.field_id}"
            if self.n_fields is not None:
                field += f"/{self.n_fields}"
        return (
            f"PhaseCenter(ra={self.ra:.6f}, dec={self.dec:.6f}, "
            f"provenance={self.provenance}{field}, obstime={self.obstime})"
        )
