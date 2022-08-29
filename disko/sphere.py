# Classes to hold pixelated spheres
# Tim Molteno tim@elec.ac.nz 2019
#

import logging
import json
import svgwrite

import numpy as np
import healpy as hp


logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)

from .resolution import Resolution

def create_fov(nside, fov, res, theta=0.0, phi=0.0):
    """
    Create an appropriate Sphere object based on:

    - fov : The field of view as a Resolution object
s
    """
    
    if nside is not None and fov is None:
        sphere = HealpixSphere(nside)
    elif nside is not None and fov is not None:
        radius_rad = fov.radians() / 2
        sphere = HealpixSubSphere.from_resolution(
            nside=nside, theta=theta, phi=phi, radius_rad=radius_rad)
    elif res is not None and fov is not None:
        radius_rad = fov.radians() / 2
        res_arcmin = res.arcmin()
        sphere = HealpixSubSphere.from_resolution(
            res_arcmin=res_arcmin, theta=theta, phi=phi, radius_rad=radius_rad)
    else:
        raise RuntimeError("Either nside, or res_arcmin must be specified")
    
    logger.info(f"create_fov -> {sphere}")
    return sphere


PI_OVER_2 = np.pi / 2


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


def cmap(fract):
    start = 1.0
    rot = -1.5
    sat = 1.5
    _gamma = 1.0

    pi = 3.14159265

    angle = 2.0 * pi * (start / 3.0 + rot * fract + 1.0)

    amp = sat * fract * (1.0 - fract) / 2.0

    # compute the RGB vectors according to main equations
    red = fract + amp * (-0.14861 * np.cos(angle) + 1.78277 * np.sin(angle))
    grn = fract + amp * (-0.29227 * np.cos(angle) - 0.90649 * np.sin(angle))
    blu = fract + amp * (1.97294 * np.cos(angle))

    # find where RBB are outside the range [0,1], clip
    red = np.clip(red, 0.0, 1.0)
    grn = np.clip(grn, 0.0, 1.0)
    blu = np.clip(blu, 0.0, 1.0)

    return (red * 255.0, grn * 255.0, blu * 255.0)


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
    l = np.sin(az_r) * np.cos(el_r)
    m = np.cos(az_r) * np.cos(el_r)
    n = np.sin(el_r)  # Often written in this weird way... np.sqrt(1.0 - l**2 - m**2)
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


class Sphere(object):
    """
    A base class for all sphere's including grids.
    """
    
    def __init__(self):
        self.pixels = None
        self.fov = None
        
    def callback(self, x, i):
        fname = f"callback_{i:05d}.svg"
        stats = self.set_visible_pixels(x)
        self.to_svg(fname, title=f"Iteration {i}")
    
    def min_res(self):
        raise Exception("min_res not implemented for this sphere")

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
        logger.info(f"Pixels Set {self.pixels.shape}, Image stats: {json.dumps(stats, sort_keys=True)}")
        return stats

    def to_fits(self, fname, title=None, info={}):
        from astropy.io import fits
        from scipy.interpolate import griddata

        # Make a grid on the plane, at the width of the narrowest pixel
        # f = scipy.interpolate.interp2d(self.el_r, self.az_r, self.pixels, fill_value=-1)
        l = np.sin(self.az_r) * np.cos(self.el_r)
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
        hdr["COMMENT"] = "POINTLESS: {}".format(title)

        hdr["ORIGIN"] = ("POINTLESS ",)
        hdr.comments["ORIGIN"] = "L-2 Regularizing imager written by Tim Molteno"

        hdr["CRPIX1"] = width // 2 + 1.0
        hdr["CDELT1"] = self.fov.degrees() / width
        hdr["CRPIX2"] = height // 2 + 1.0
        hdr["CDELT2"] = self.fov.degrees() / height
        for key in info:
            hdr[key] = info[key]
        # https://archive.stsci.edu/fuse/DH_Final/FITS_File_Headers.html
        logger.info("Writing FITS image: {}".format(fname))

        hdu = fits.PrimaryHDU(np.array(grid, dtype=np.float32), header=hdr)
        hdu.writeto(fname, overwrite=True)


class HealpixSphere(Sphere):
    """
    A healpix Sphere
    """

    def __init__(self, nside):
        super().__init__()
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        res = hp.nside2resol(nside, arcmin=True)
        self._min_res = Resolution.from_arcmin(res)

        logger.info(f"New Sphere, nside={nside}. npix={self.npix}, res={self.min_res()} arcmin")

        self.pixel_indices = np.arange(self.npix)
        theta, phi = hp.pix2ang(nside, self.pixel_indices)

        # For limited fields of view
        # healpy.query_polygon

        self.pixels = np.zeros(self.npix)  # + hp.UNSEEN
        self.pixel_areas = np.ones(self.npix)/self.npix

        el_r, az_r = hp2elaz(theta, phi)

        self.el_r = el_r
        self.az_r = az_r

        self.l, self.m, self.n = elaz2lmn(self.el_r, self.az_r)
        self.n_minus_1 = self.n - 1

    def __repr__(self):
        return f"HealpixSphere nside={self.nside}"
    
    def min_res(self):
        return self._min_res
    
    def get_lmn(self):
        return self.l, self.m, self.n

    def index_of(self, el, az):
        theta, phi = elaz2hp(el, az)
        return hp.ang2pix(self.nside, theta, phi)

    def plot_dot(self, el, az):
        theta, phi = elaz2hp(el, az)
        hp.projplot(theta, phi, "k.", rot=(0, 90, 0))  #

    def plot_x(self, plt, el, az):
        theta, phi = elaz2hp(el, az)
        hp.projplot(theta, phi, "ro", rot=(0, 90, 180))  #


    def corners(self, pixel):
        bounds = hp.boundaries(self.nside, pixel, step=1)
        bounds = np.array(bounds).T
        return bounds
        # lat, lon = hp.vec2ang(bounds)
        # ret = []
        # for a,b in zip(lat, lon):
        # ret.append([b,a])

        # return ret


    def to_svg(
        self,
        fname,
        pixels_only=False,
        show_grid=False,
        src_list=None,
        title=None,
        show_cbar=True,
    ):

        if self.fov is None:
            raise Exception("Field of view is required for SVG generation. Use PDF instead")
        
        h = 4000
        w = 4200
        # dwg = svgwrite.Drawing(filename=fname, size=(w,w), profile='tiny')
        dwg = svgwrite.Drawing(filename=fname, size=(w, h))

        # dwg.add(dwg.line((0, 0), (10, 0), stroke=svgwrite.rgb(10, 10, 16, '%')))
        # dwg.add(dwg.text('Test', insert=(0, 0.2), fill='red'))

        pc = PlotCoords(h, self.fov.radians()/2)
        line_size = pc.line_size

        # dwg.desc("Gridless imaging from visibilities.")
        if True:  # info is not None:
            rad = self.fov.radians() / 2
            width = np.sin(rad)
            font_size = pc.from_d(0.05 * width)

            x = pc.from_x(-width)

            if title is not None:
                y = font_size
                dwg.add(
                    dwg.text(
                        title,
                        (x, y),
                        text_anchor="start",
                        font_size="{}px".format(font_size),
                    )
                )

            y = font_size * 2
            dwg.add(
                dwg.text(
                    f"Res: {self.min_res()}",
                    (x, y),
                    text_anchor="start",
                    font_size="{}px".format(font_size),
                )
            )

            y = font_size * 3
            dwg.add(
                dwg.text(
                    f"FOV: {self.fov}",
                    (x, y),
                    text_anchor="start",
                    font_size="{}px".format(font_size),
                )
            )

            y = font_size * 4
            dwg.add(
                dwg.text(
                    "N_s: {}".format(self.pixels.shape[0]),
                    (x, y),
                    text_anchor="start",
                    font_size="{}px".format(font_size),
                )
            )

        stats = image_stats(self.pixels)

        if not pixels_only:
            svg_pixels = dwg.g(
                stroke_width=2, stroke_linejoin="round", stroke_opacity=1.0
            )

        for i in range(self.npix):
            idx = self.pixel_indices[i]
            corners = self.corners(idx)  # x,y,z points on boundary
            value = self.pixels[i]

            max_lat = -1e99
            min_lat = 1e99
            x_mean = 0.0
            y_mean = 0.0

            cnr_lat, cnr_lon = hp.vec2ang(corners, lonlat=False)

            poly = []
            for p, phi, theta in zip(corners, cnr_lon, cnr_lat):
                # logger.info("p = {} lat={}, lon={}".format(p, lat, lon))
                lat = np.pi / 2 - theta

                max_lat = max(max_lat, lat)
                min_lat = min(min_lat, lat)

                # if theta > np.pi / 2:
                # logger.info("colatitude {} > PI_OVER_2 ll={}".format(theta, p))

                hpang = HpAngle(theta, np.pi + phi)
                (x, y) = hpang.proj()

                poly.append((pc.from_x(x), pc.from_y(y)))

                x_mean = x_mean + x
                y_mean = y_mean + y

            x_mean = x_mean / 4.0
            y_mean = y_mean / 4.0

            if max_lat > np.pi / 2:  # Ignore points on the horizon
                continue

            if pixels_only:
                if min_lat >= 0.00:  # Ignore points on, or below the horizon
                    dlat = max_lat - min_lat
                    # logger.info(poly)
                    font_size = pc.from_d(dlat * np.sin(max_lat)) // 5

                    dwg.add(
                        dwg.polygon(
                            points=poly,
                            fill="none",
                            stroke="black",
                            stroke_linejoin="round",
                            stroke_width=font_size // 10,
                        )
                    )

                    x = pc.from_x(x_mean)
                    y = pc.from_y(y_mean) + (font_size // 2)
                    dwg.add(
                        dwg.text(
                            "{}".format(i),
                            (x, y),
                            text_anchor="middle",
                            font_size="{}px".format(font_size),
                        )
                    )
            else:
                (r, g, b) = cmap((value - stats["min"]) / (1e-14 + stats["max"] - stats["min"]))
                colour = svgwrite.rgb(r, g, b)
                if min_lat > 0.07:  # Ignore points on, or below the horizon
                    svg_pixels.add(dwg.polygon(points=poly, fill=colour, stroke=colour))

        grid_color = "black"
        if not pixels_only:
            dwg.add(svg_pixels)
            grid_color = "grey"

        if show_cbar:
            N = 255
            bar_boxes = dwg.g(
                stroke_width=2, stroke_linejoin="round", stroke_opacity=1.0
            )
            values = np.linspace(stats["min"], stats["max"], N)

            rvals = (values - stats["min"]) / (1e-14 + stats["max"] - stats["min"])

            start_y = 0.05 * width
            stop_y = 2.05 * width
            x0 = pc.from_d(2.1 * width)
            x1 = pc.from_d(2.15 * width)

            def y_scale(y):
                dy = stop_y - start_y
                return stop_y - y * dy

            for i in range(N - 1):
                v0 = rvals[i]
                v1 = rvals[i + 1]
                (r, g, b) = cmap(v0)

                y0 = pc.from_d(y_scale(v0))
                y1 = pc.from_d(y_scale(v1))

                poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                colour = svgwrite.rgb(r, g, b)
                bar_boxes.add(dwg.polygon(points=poly, fill=colour, stroke=colour))
                # logger.info("box {}".format(poly))

                if (i == np.argmin(np.abs(values))) and (i != 0):
                    bar_boxes.add(
                        dwg.text(
                            "0",
                            (x0 - font_size / 2, y1 + font_size / 2),
                            text_anchor="end",
                            font_size="{}px".format(font_size),
                        )
                    )
                    bar_boxes.add(
                        dwg.line(
                            start=(x0 - font_size / 3, y0),
                            end=(x0, y0),
                            stroke="black",
                            stroke_width="{}".format(line_size),
                        )
                    )

                if i == 0:
                    bar_boxes.add(
                        dwg.text(
                            "{:5.3f}".format(stats["min"]),
                            (x0 - font_size / 2, y1),
                            text_anchor="end",
                            font_size="{}px".format(font_size),
                        )
                    )

                if i == N - 2:
                    bar_boxes.add(
                        dwg.text(
                            "{:5.3f}".format(stats["max"]),
                            (x0 - font_size / 2, y1 + font_size),
                            text_anchor="end",
                            font_size="{}px".format(font_size),
                        )
                    )

            dwg.add(bar_boxes)

            y0 = pc.from_d(start_y)
            y1 = pc.from_d(stop_y)
            box_border = dwg.g(
                fill="none",
                stroke=grid_color,
                stroke_width="{}".format(line_size / 2),
                stroke_linejoin="round",
            )
            box_border.add(dwg.polygon(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
            dwg.add(box_border)

        if show_grid:
            grid_lines = dwg.g(
                fill="none",
                stroke=grid_color,
                stroke_width="{}".format(line_size),
                stroke_linejoin="round",
                stroke_dasharray="{},{}".format(5 * line_size, 10 * line_size),
            )
            
            fov_rad = self.fov.radians()

            for rad in np.linspace(0, fov_rad / 2, 4)[1:]:  # three circles
                radius = pc.from_d(np.sin(rad))
                grid_lines.add(
                    dwg.circle(center=(pc.from_x(0.0), pc.from_y(0.0)), r=radius)
                )

            for angle in range(0, 360, 30):
                rad = np.radians(angle)
                radius = np.sin(fov_rad / 6)
                x0 = radius * np.sin(rad)
                y0 = radius * np.cos(rad)

                radius = np.sin(fov_rad / 2)
                x = radius * np.sin(rad)
                y = radius * np.cos(rad)
                grid_lines.add(
                    dwg.line(
                        start=(pc.from_x(x0), pc.from_y(y0)),
                        end=(pc.from_x(x), pc.from_y(y)),
                    )
                )

            dwg.add(grid_lines)

        if src_list is not None:
            angular_size = np.radians(2.0)
            source_circles = dwg.g(
                fill="none", stroke="red", stroke_width="{}".format(line_size)
            )
            for s in src_list:
                if s.el_r > np.radians(10.0):
                    elaz = ElAz(s.el_r, s.az_r)
                    (x, y) = pc.from_elaz(elaz)

                    radial_size = angular_size * np.sin(s.el_r)

                    radiusx = pc.from_d(angular_size)
                    radiusy = pc.from_d(radial_size)
                    # Project the source circle onto an ellipse.
                    circ = dwg.ellipse(center=(x, y), r=(radiusx, radiusy))
                    circ.rotate(-np.degrees(s.az_r), center=(x, y))

                    source_circles.add(circ)
            dwg.add(source_circles)

        with open(fname, "w", encoding="utf-8") as outfile:
            dwg.write(outfile)

    def plot(self, plt, src_list):
        rot = (0, 90, 0)
        plt.figure()  # (figsize=(6,6))
        logger.info("self.pixels: {}".format(self.pixels.shape))
        if True:
            hp.orthview(
                self.pixels, rot=rot, xsize=1000, cbar=True, half_sky=False, hold=False
            )
            hp.graticule(verbose=False)
        else:
            hp.mollview(self.pixels, rot=rot, xsize=1000, cbar=True)
            hp.graticule(verbose=True)

        if src_list is not None:
            for s in src_list:
                self.plot_x(plt, s.el_r, s.az_r)


def my_query_disk(nside, x0, radius):
    # Avoid the stupid nside limit of powers of two.
    ret = []
    npix = hp.nside2npix(nside)
    all_pixels = np.array(range(npix))
    all_vec = hp.pix2vec(nside, all_pixels)

    for i in all_pixels:
        x1 = hp.pix2vec(nside, i)
        angle = np.arccos(np.dot(x1, x0))
        if angle <= radius:
            ret.append(i)

    return np.array(ret)


class HealpixSubSphere(HealpixSphere):
    """
    A healpix subset of a sphere bounded by a range in theta and phi
    """

    def __init__(self, nside):
        res = hp.nside2resol(nside, arcmin=True)
        self.nside = nside
        self._min_res = Resolution.from_arcmin(res)
        logger.info(f"New SubSphere, nside={self.nside}, res={self._min_res}")

    @classmethod
    def from_resolution(
        cls, res_arcmin=None, nside=None, theta=0.0, phi=0.0, radius_rad=0.0
    ):
        logger.info(f"HealpixSubSphere.from_resolution(res={res_arcmin} arcmin, nside={nside}, theta={theta}, phi={phi}, radius_rad={radius_rad})")
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]

        if nside is None:  # Calculate nside to the appropriate resolution
            nside = 1
            while hp.nside2resol(nside, arcmin=True) > res_arcmin:
                logger.info(f"nside={nside} res={hp.nside2resol(nside, arcmin=True)}")
                nside *= 2

        ret = cls(nside)

        # The coordinates of the unit vector defining the center
        x0 = hp.ang2vec(theta, phi)

        # https://healpy.readthedocs.io/en/latest/generated/healpy.query_polygon.html
        ret.pixel_indices = hp.query_disc(
            nside=nside, vec=x0, radius=radius_rad, inclusive=False, nest=False
        ).astype(int)

        ret.npix = ret.pixel_indices.shape[0]
        
        ret.pixel_areas = np.ones(ret.npix)/ret.npix

        logger.info("New SubSphere, nside={}. npix={}".format(ret.nside, ret.npix))

        theta, phi = hp.pix2ang(nside, ret.pixel_indices)

        ret.fov = Resolution.from_rad(radius_rad * 2)
        ret.pixels = np.zeros(ret.npix)  # + hp.UNSEEN

        el_r, az_r = hp2elaz(theta, phi)

        ret.el_r = el_r
        ret.az_r = az_r

        ret.l, ret.m, ret.n = elaz2lmn(ret.el_r, ret.az_r)
        ret.n_minus_1 = ret.n - 1

        if False:
            import matplotlib.pyplot as plt

            # plt.plot(ret.l, ret.m, 'x')
            plt.plot(el_r, az_r, "x")
            plt.show()
        return ret

    def __repr__(self):
        return f"HealpixSubSphere fov={self.fov}, nside={self.nside}"

    def split(self, n):
        ret = []
        pixel_list = np.array_split(self.pixels, n)
        pixel_indices = np.array(range(self.npix))
        pixel_indices_list = np.array_split(pixel_indices, n)
        el_list = np.array_split(self.el_r, n)
        az_list = np.array_split(self.az_r, n)

        for pix, idx, el, az in zip(pixel_list, pixel_indices_list, el_list, az_list):
            subs = HealpixSubSphere(self.nside)
            subs.pixels = np.zeros_like(pix)
            subs.parent_indices = idx
            subs.npix = pix.shape[0]
            subs.el_r = el
            subs.az_r = az
            subs.l, subs.m, subs.n = elaz2lmn(subs.el_r, subs.az_r)
            ret.append(subs)
        return ret

    def plot(self, plt, src_list):
        """
        Modified plot to deal with the reduced number of pixels
        """
        all_npix = hp.nside2npix(self.nside)
        all_pixels = np.zeros(all_npix) + hp.UNSEEN
        all_pixels[self.pixel_indices] = self.pixels

        rot = (0, 90, 0)
        plt.figure()  # (figsize=(6,6))
        logger.info("self.pixels: {}".format(self.pixels.shape))
        if True:
            hp.orthview(
                all_pixels, rot=rot, xsize=1000, cbar=True, half_sky=True, hold=False
            )
            hp.graticule(verbose=False)
        else:
            hp.mollview(all_pixels, rot=rot, xsize=1000, cbar=True)
            hp.graticule(verbose=True)

        if src_list is not None:
            for s in src_list:
                self.plot_x(plt, s.el_r, s.az_r)


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


class HexagonSubSphere(HealpixSphere):
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


if __name__ == "__main__":
    sph = HealpixSphere(2)
    sph.to_svg(fname="sph.svg", pixels_only=True)

    res_deg = 5
    print("Done")
    big = HealpixSubSphere(
        resolution=res_deg * 60.0,
        theta=np.radians(20.0),
        phi=0.0,
        radius=np.radians(35),
    )
    print(big.nside)
    big.to_svg(fname="test.svg", pixels_only=True, show_grid=True)
    big.set_visible_pixels(np.random.rand(big.npix))
    big.to_svg(fname="test_map.svg", show_grid=True)
