# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3

# Classes to hold pixelated spheres
# Tim Molteno tim@elec.ac.nz 2019-2023
#

import logging
import json
import copy
import h5py
import datetime

import numpy as np
import healpy as hp

from tart.util import utc

from astropy.coordinates import EarthLocation
from . import coords
from .resolution import Resolution

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


PI_OVER_2 = np.pi / 2


class GeoLocation(object):
    '''
        Utility class to serialize astropy EarthLocation
    '''

    def __init__(self, lon, lat, height):
        self.loc = EarthLocation.from_geodetic(lon=lon, lat=lat, height=height).to_geodetic()

    @classmethod
    def from_json(cls, json_string):
        data = json.loads(json_string)
        logger.info(f"GeoLocation.from_json({data})")
        return cls(lon=data['lon'],
                   lat=data['lat'],
                   height=data['height'])

    def to_json(self):
        ret = {'lat': self.loc.lat.value,
               'lon': self.loc.lon.value,
               'height': self.loc.height.value}
        return json.dumps(ret)


class LonLat(object):
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    @classmethod
    def from_hp(class_object, hpang):
        return class_object(hpang.phi, PI_OVER_2 - hpang.theta)

    @classmethod
    def from_pix(class_object, nside, pix):
        (lon, lat) = hp.center(nside, pix)
        return class_object(lon, lat)


class HpAngle(object):
    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

    @classmethod
    def from_lonlat(class_object, lonlat):
        return class_object(PI_OVER_2 - lonlat.lat, lonlat.lon)

    @classmethod
    def from_elaz(class_object, el, az):
        theta = PI_OVER_2 - el
        phi = -az
        return class_object(theta, phi)

    def proj(self):
        # viewpoint is from straight up, projected down.
        r = np.sin(self.theta)

        x = r * np.sin(self.phi)
        y = -r * np.cos(self.phi)

        return (x, y)


class ElAz(object):
    def __init__(self, el, az):
        self.el = el
        self.az = az

    @classmethod
    def from_hp(cls, hp):
        el = PI_OVER_2 - hp.theta
        az = -hp.phi
        return cls(el, az)

    def to_hp(self):
        return HpAngle.from_elaz(self.el, self.az)

    # def to_lmn(self):
    # l = self.az.sin()*self.el.cos()
    # m = self.az.cos()*self.el.cos()
    # n = self.el.sin() // Often written in this weird way... np.sqrt(1.0 - l**2 - m**2)
    # return (l, m, n)
    # }


class PlotCoords(object):
    """
    The image is w x h, with the center at h/2 and h/2.
    The virtual coordinates start at the center, and the disk as a radius of 1.0

    The color bar exists on the left

    """

    def __init__(self, h, fov_rad):
        # w is width in pixels
        angular_scale = 1.0 / np.sin(fov_rad)
        self.scale = float(h) * angular_scale / 2.1
        self.center_x = int(round(float(h) / 2.0))
        self.center_y = int(round(float(h) / 2.0))
        self.line_size = int(float(h) / 400)

    def from_d(self, d):
        """
        Scale a distance. to the pixel
        """
        return int(d * self.scale)

    def from_x(self, x):
        return int(x * self.scale) + self.center_x

    def from_y(self, y):
        ret = int(y * self.scale) + self.center_y
        # logger.info("from_y({})->{}".format(y, ret))
        return ret

    def from_elaz(self, elaz):
        hp = elaz.to_hp()
        (x, y) = hp.proj()
        return (self.from_x(x), self.from_y(y))


def elaz2lmn(el_r, az_r):
    l = np.sin(az_r) * np.cos(el_r)  # noqa: E741
    m = np.cos(az_r) * np.cos(el_r)
    # Often written in this weird way... np.sqrt(1.0 - l**2 - m**2)
    n = np.sin(el_r)
    return l, m, n


def hp2elaz(theta, phi):
    el = np.pi / 2 - theta
    az = -phi
    return el, az


def elaz2hp(el, az):
    theta = np.pi / 2 - el
    phi = -az
    return theta, phi


def lonlat(theta, phi):
    """Converts theta and phi to longitude and latitude
    From colatitude to latitude and from astro longitude to geo longitude"""

    longitude = -1 * np.asarray(phi)
    latitude = np.pi / 2 - np.asarray(theta)
    return longitude, latitude


def image_stats(sky):

    rsky = sky.flatten()

    ret = {}

    ret["N_s"] = sky.shape[0]
    ret["sdev"] = np.std(rsky)
    ret["mean"] = np.mean(rsky)
    ret["min"], ret["med"], ret["max"] = np.percentile(rsky, [0, 50, 100])

    abs_deviation = np.array(np.abs(ret["med"] - rsky))
    ret["mad"] = np.median(abs_deviation, axis=0)

    if ret["sdev"] > 0:
        ret["S/N"] = (ret["max"] - ret["mean"]) / ret["sdev"]
    else:
        ret["S/N"] = ret["max"] - ret["mean"]

    if ret["mad"] > 0:
        ret["R_mad"] = (ret["max"] - ret["med"]) / ret["mad"]
    else:
        ret["R_mad"] = ret["max"] - ret["med"]

    return ret


def factors(n):
    result = set()
    for i in range(1, int(n ** 0.5) + 1):
        div, mod = divmod(n, i)
        if mod == 0:
            result |= {i, div}
    ret = sorted(list(result))
    return ret[len(ret) // 2]


class FoV(object):
    """
    A base class for all sphere's including grids. The sphere must be aware of coordinates.

    A sphere has a phase-center, and optionally a domain (field-of-view for circular skies)

    The base coordinates are SkyCoordinates, so the
    The two coordinate systems are elevation and azimuth in the geolocated frame (the phase center is
    straight up)
    """

    def __init__(self):
        self.pixels = None
        self.pixmin = None
        self.pixmax = None
        self.pixel_areas = None
        self.fov = None
        # The celestial pointing (coords.PhaseCenter) is first-class data:
        # None until set_info() is given one (e.g. from an MS PHASE_DIR), in
        # which case the phase centre is derived from the zenith (#14).
        self.phase_center = None
        self.set_info(timestamp=datetime.datetime.now(datetime.timezone.utc),
                      lon=0, lat=0, height=0)

    def set_info(self, timestamp, lon, lat, height, phase_center=None):
        '''
            Set the timestamp and geographic information about this sphere.

            phase_center: optional coords.PhaseCenter giving this sphere's
            celestial pointing (e.g. from_phase_dir() on an MS PHASE_DIR
            read). Its obstime and geolocation are filled in from these
            arguments when it does not carry them, so the pointing always
            applies at this sphere's site and time. When omitted, any
            previously set phase centre is left alone; with none set at
            all the phase centre falls back to the zenith derivation in
            phase_center_radec().
        '''
        self.timestamp = utc.to_utc(timestamp)
        self.geolocation = GeoLocation(lon=lon, lat=lat, height=height)
        if phase_center is not None:
            if phase_center.obstime is None:
                phase_center.obstime = coords.as_datetime(self.timestamp)
            if phase_center.geolocation is None:
                phase_center.geolocation = self.geolocation
            self.phase_center = phase_center

    def site_obstime(self):
        '''
            (lon_deg, lat_deg, height_m, obstime) of this sphere: the
            geolocation set by set_info() (a GeoLocation wrapper) or by
            fov.from_hdf() (an astropy EarthLocation). Both expose
            lon/lat/height.
        '''
        lon, lat, height = coords.site_from_geolocation(
            getattr(self, "geolocation", None)
        )
        return lon, lat, height, getattr(self, "timestamp", None)

    def as_elaz(self, x, y, frame="elaz"):
        '''
            Interpret (x, y) as coordinates in `frame` and return the
            equivalent (el, az) in radians for this sphere.

            frame="elaz" (the default) means x, y are elevation and azimuth
            in radians in the geolocated frame. frame="icrs" means x, y are
            right ascension and declination in degrees, converted using
            this sphere's site and time — so a celestial source can be
            placed on a phase-steered grid.
        '''
        if frame == "elaz":
            return x, y
        if frame in ("icrs", "celestial"):
            lon, lat, height, obstime = self.site_obstime()
            return coords.radec_to_elaz(
                x, y, lon, lat, height=height, obstime=obstime
            )
        raise ValueError(
            f"Unknown coordinate frame {frame!r} (expected 'elaz' or 'icrs')"
        )

    def source_elaz(self, source):
        '''
            (el_r, az_r) of a source placed on this sphere. Sources that
            carry elaz (e.g. the TART catalog objects) are used as they
            are; sources given in celestial coordinates (ra/dec degrees)
            are converted with this sphere's site and time.
        '''
        lon, lat, height, obstime = self.site_obstime()
        return coords.source_elaz_rad(
            source, lon, lat, height=height, obstime=obstime
        )

    def callback(self, x, i):
        fname = f"callback_{i:05d}.hdf"
        self.set_visible_pixels(x)
        self.to_hdf(fname)

    def copy(self):
        ret = copy.deepcopy(self)
        ret.pixels = np.array(self.pixels)
        ret.pixel_areas = np.array(self.pixel_areas)
        return ret

    def index_of(self, el, az, frame="elaz"):
        raise RuntimeError("index_of() not implemented for this sphere")

    def min_res(self):
        raise Exception("min_res() not implemented for this sphere")

    def get_area(self):
        return np.sum(self.pixel_areas)

    def get_power(self):
        # Total flux of this image.
        # This is the sum of the pixel intensities
        return np.sum((self.pixels/self.pixel_areas)**2)/self.npix

    def rms(self):
        return np.sqrt(np.mean(self.pixels**2))

    '''
        This will become the plot range when the sphere is plotted
    '''
    def set_plot_range(self, lower, upper):
        self.pixmin = lower
        self.pixmax = upper

    def get_plot_range(self, stats):
        if self.pixmin is not None:
            pixmin = self.pixmin
        else:
            pixmin = stats["min"]

        if self.pixmax is not None:
            pixmax = self.pixmax
        else:
            pixmax = stats["max"]
        return pixmin, pixmax

    def normalize_pixel(self, pixel, stats):

        pixmin, pixmax = self.get_plot_range(stats)

        return (pixel - pixmin) / (1e-14 + pixmax - pixmin)

    def to_hdf_header(self, h5f):
        dt = h5py.special_dtype(vlen=bytes)

        info_json = {}
        info_json['fov_type'] = type(self).__name__
        info_json['timestamp'] = self.timestamp.isoformat()
        info_json['geolocation'] = self.geolocation.to_json()
        info_json['center'] = 2
        # Optional (issue #10, Phase 1): only written when a phase centre is
        # stored, and readers ignore keys they do not know, so files written
        # by older versions still load (and files written here still load in
        # older code paths).
        if getattr(self, "phase_center", None) is not None:
            info_json['phase_center'] = self.phase_center.to_dict()

        conf_dset = h5f.create_dataset('information', (1,), dtype=dt)
        conf_dset[0] = json.dumps(info_json)

    def to_hdf(self, filename):
        raise Exception("to_hdf() not implemented for this sphere")

    def to_svg(
        self,
        fname,
        pixels_only=False,
        show_grid=False,
        src_list=None,
        title=None,
        show_cbar=True,
    ):
        raise Exception("SVG not implemented for this sphere")

    def set_visible_pixels(self, pix, scale=False):
        # This discards the imaginary part.
        rpix = np.asarray(np.real(pix))

        stats = image_stats(rpix)
        if scale:
            rpix = (rpix - stats["min"]) / stats["sdev"]
            # n_s = rpix.shape[0]
            # fact = factors(n_s)
            # rpix = exposure.equalize_adapthist(rpix.reshape((n_s//fact, -1)), clip_limit=0.03)
        self.pixels = rpix.reshape((len(pix),))
        logger.info(
            f"Pixels Set {self.pixels.shape}, Image stats: {json.dumps(stats, sort_keys=True)}")
        return stats

    def phase_center_radec(self):
        '''
            Right ascension and declination (degrees, ICRS) of this sphere's
            phase center.

            The stored phase center wins: when the sphere was given one
            (issue #10 Phase 1 — e.g. read from an MS PHASE_DIR via
            set_info(..., phase_center=...)), its RA/Dec is returned.

            Otherwise the phase center is derived: to_fits() grids the
            image on direction cosines (l, m) about l=0, m=0, which is the
            direction straight up in the geolocated frame of the sphere
            (the phase center, see the class docstring). So the phase
            center is the zenith as seen from this sphere's geolocation at
            its timestamp, and its celestial coordinates come from exactly
            those two pieces of sphere data (the #14 fix).
        '''
        pc = getattr(self, "phase_center", None)
        if pc is not None:
            return pc.ra, pc.dec
        lon, lat, height, obstime = self.site_obstime()
        return coords.zenith_radec(lon, lat, height=height, obstime=obstime)

    def get_phase_center(self):
        '''
            This sphere's phase center as a coords.PhaseCenter, with its
            provenance: the stored one when there is one ("ms_phase_dir"),
            else a zenith-derived one ("zenith") built from this sphere's
            geolocation and timestamp.
        '''
        pc = getattr(self, "phase_center", None)
        if pc is not None:
            return pc
        lon, lat, height, obstime = self.site_obstime()
        ra, dec = coords.zenith_radec(lon, lat, height=height, obstime=obstime)
        return coords.PhaseCenter.from_zenith(
            ra, dec, obstime=obstime,
            geolocation=getattr(self, "geolocation", None),
        )

    def to_fits(self, fname, title=None, info={}):
        from astropy.io import fits
        from scipy.interpolate import griddata

        # Make a grid on the plane, at the width of the narrowest pixel
        # f = scipy.interpolate.interp2d(self.el_r, self.az_r, self.pixels, fill_value=-1)
        l = np.sin(self.az_r) * np.cos(self.el_r)  # noqa: E741
        m = -np.cos(self.az_r) * np.cos(self.el_r)

        points = (l, m)
        values = self.pixels

        l0 = np.sin(self.fov.radians() / 2)

        width = 2000
        height = 2000
        x = np.linspace(-l0, l0, width)
        y = np.linspace(-l0, l0, height)
        xx, yy = np.meshgrid(x, y)

        grid = griddata(points, values, (xx, yy), method="cubic")

        hdr = fits.Header()
        hdr["COMMENT"] = "DiSkO: {}".format(title)

        hdr["ORIGIN"] = ("DiSkO ",)
        hdr.comments["ORIGIN"] = "L-2 Regularizing imager written by Tim Molteno"

        hdr["CRPIX1"] = width // 2 + 1.0
        hdr["CDELT1"] = -self.fov.degrees() / width # note: RA increases East
        hdr["CRPIX2"] = height // 2 + 1.0
        hdr["CDELT2"] = self.fov.degrees() / height

        # Issue #14: the non-MS paths pass no world coordinate system in
        # `info`, so CRPIX/CDELT alone left an invalid WCS. Derive the rest
        # from the sphere itself (the reference pixel is the phase center).
        # Anything the caller supplies (casa_read_ms passes a full WCS)
        # still wins, see the loop below.
        wcs_keys = {
            "RADESYS": ("ICRS", "Celestial coordinate reference frame"),
            "CTYPE1": ("RA---SIN", "Right ascension angle cosine"),
            "CUNIT1": ("deg", "Units of CDELT1 and CRVAL1"),
            "CTYPE2": ("DEC--SIN", "Declination angle cosine"),
            "CUNIT2": ("deg", "Units of CDELT2 and CRVAL2"),
        }
        if "CRVAL1" not in info or "CRVAL2" not in info:
            ra, dec = self.phase_center_radec()
            wcs_keys["CRVAL1"] = (ra, "Right ascension of the phase center")
            wcs_keys["CRVAL2"] = (dec, "Declination of the phase center")

        for key in wcs_keys:
            if key not in info:
                hdr[key] = wcs_keys[key]
        for key in info:
            hdr[key] = info[key]
        # https://archive.stsci.edu/fuse/DH_Final/FITS_File_Headers.html
        logger.info("Writing FITS image: {}".format(fname))

        hdu = fits.PrimaryHDU(np.array(grid, dtype=np.float32), header=hdr)
        hdu.writeto(fname, overwrite=True)


class SquareFoV(FoV):

    def __init__(self, res_arcmin=None, theta=0.0, phi=0.0, width_rad=0.0):
        super().__init__()
        logger.info(r"SquareFoV:")
        logger.info(f"    res={res_arcmin} arcmin,")
        logger.info(f"    theta={theta}, phi={phi}, width={width_rad} rad)")
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]

        self.width_rad = width_rad

        self.theta = theta
        self.phi = phi

        # Actual width and resolution
        self.width_pix = 2*int(np.ceil(np.degrees(width_rad)*60 / res_arcmin))
        self.res_arcmin = np.degrees(width_rad)*60 / self.width_pix

        self.npix = self.width_pix*self.width_pix

        logger.info(
            f"New SubFoV, width_pix={self.width_pix} npix={self.npix}, res={self.res_arcmin} arcmin")

        self.fov = Resolution.from_rad(width_rad)
        self.pixels = np.zeros(self.npix)

        self.pixel_areas = np.ones(self.npix)/self.npix

        # Assume flat sky
        # Create the direction cosines
        el_r = []
        az_r = []
        center = self.width_pix // 2


        arcmin2rad = np.radians(1.0/60)
        def pixels_to_rad(pix):
           return (pix*self.res_arcmin*arcmin2rad) / 2  # Nyquist

        # TODO add self.el_r and self.az_r
        for i in range(self.width_pix):
            dx = (i - center)
            for j in range(self.width_pix):
                dy = j - center

                dr = np.sqrt(dx*dx + dy*dy)
                el = np.pi/2 - pixels_to_rad(dr)

                az = np.arctan2(dy,dx)

                el_r.append(el)
                az_r.append(az)

        self.el_r = np.array(el_r)
        self.az_r = np.array(az_r)

        self.l, self.m, self.n = elaz2lmn(self.el_r, self.az_r)
        self.n_minus_1 = self.n - 1


    def __repr__(self):
        return f"SquareFoV fov={self.fov}, width={self.width_pix}, res={self.res_arcmin}"

    def to_hdf(self, filename):
        with h5py.File(filename, "w") as h5f:
            self.to_hdf_header(h5f)

            h5f.create_dataset('res_arcmin', data=[self.res_arcmin])
            h5f.create_dataset('theta', data=[self.theta])
            h5f.create_dataset('phi', data=[self.phi])
            h5f.create_dataset('width_rad', data=[self.width_rad])

            h5f.create_dataset('pixels', data=self.pixels)
            h5f.create_dataset('pixel_indices', data=self.pixel_indices)

    def get_image(self):
        return self.pixels.reshape(self.width_pix, self.width_pix)


'''
class HexagonGenerator(object):
    """
    Returns a hexagon generator for hexagons of the specified size.

    https://variable-scope.com/posts/hexagon-tilings-with-python
    """

    def __init__(self, edge_length):
        self.edge_length = edge_length

    @property
    def col_width(self):
        return self.edge_length * 3

    @property
    def row_height(self):
        return np.sin(np.pi / 3) * self.edge_length

    def __call__(self, row, col):
        x = (col + 0.5 * (row % 2)) * self.col_width
        y = row * self.row_height
        for angle in range(0, 360, 60):
            x += np.cos(np.radians(angle)) * self.edge_length
            y += np.sin(np.radians(angle)) * self.edge_length
        yield x
        yield y


class HexagonSubFoV(FoV):
    def __init__(self):
        image = Image.new("RGB", (250, 250), "white")
        draw = Draw(image)
        hexagon_generator = HexagonGenerator(40)
        for row in range(7):
            color = row * 10, row * 20, row * 30
            for col in range(2):
                hexagon = hexagon_generator(row, col)
                draw.polygon(list(hexagon), Brush(color))
        draw.flush()
        image.show()
'''
