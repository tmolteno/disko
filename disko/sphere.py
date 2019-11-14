# Classes to hold pixelated spheres
# Tim Molteno tim@elec.ac.nz 2019
#

import logging

import svgwrite

import numpy as np
import healpy as hp


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


PI_OVER_2 = np.pi / 2

class LonLat(object):
    def __init__(self, lon,  lat):
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
    def __init__(self, theta,  phi):
        self.theta = theta
        self.phi = phi
    
    @classmethod
    def from_lonlat(class_object, lonlat):
        return class_object(PI_OVER_2 - lonlat.lat, lonlat.lon)
    
    @classmethod
    def from_elaz(class_object, el,  az):
        theta = PI_OVER_2 - el
        phi = -az
        return class_object(theta, phi)
    
    def proj(self):
        # viewpoint is from straight up, projected down.
        r = np.sin(self.theta)
        
        x = r*np.sin(self.phi)
        y = -r*np.cos(self.phi)
        
        return (x,y)

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
    
    #def to_lmn(self):
        #l = self.az.sin()*self.el.cos()
        #m = self.az.cos()*self.el.cos()
        #n = self.el.sin() // Often written in this weird way... np.sqrt(1.0 - l**2 - m**2)
        #return (l, m, n)
    #}



def cmap(fract):
    start = 1.0
    rot = -1.5
    sat = 1.5
    _gamma = 1.0
    
    pi = 3.14159265
    
    angle = 2.0 * pi * (start / 3.0 + rot * fract + 1.)

    amp = sat * fract * (1. - fract) / 2.

    # compute the RGB vectors according to main equations
    red = fract + amp * (-0.14861 * np.cos(angle) + 1.78277 * np.sin(angle))
    grn = fract + amp * (-0.29227 * np.cos(angle) - 0.90649 * np.sin(angle))
    blu = fract + amp * (1.97294 * np.cos(angle))

    # find where RBB are outside the range [0,1], clip
    red = np.clip(red, 0.0, 1.0)
    grn = np.clip(grn, 0.0, 1.0)
    blu = np.clip(blu, 0.0, 1.0)

    return (red*255.0, grn*255.0, blu*255.0)



class PlotCoords(object):

    def __init__(self, w):
        self.scale = float(w)/2.1
        self.center = int(round(float(w)/2.0))
        self.line_size = int(float(w) / 400)
    
    def from_d(self, d):
        return int(d*self.scale)
    
    def from_x(self, x):
        return int(x*self.scale) + self.center
    
    def from_y(self, y):
        return int(y*self.scale) + self.center
    
    def from_elaz(self, elaz):
        hp = elaz.to_hp()
        (x,y) = hp.proj()
        return (self.from_x(x), self.from_y(y))

def elaz2lmn(el_r, az_r):
    l = np.sin(az_r)*np.cos(el_r)
    m = np.cos(az_r)*np.cos(el_r)
    n = np.sin(el_r) # Often written in this weird way... np.sqrt(1.0 - l**2 - m**2)
    return l, m, n


def hp2elaz(theta, phi):
    el = np.pi/2 - theta
    az = -phi
    return el, az

def elaz2hp(el, az):
    theta = np.pi/2 - el
    phi = -az
    return theta, phi

def lonlat(theta, phi):
    """Converts theta and phi to longitude and latitude
    From colatitude to latitude and from astro longitude to geo longitude"""

    longitude = -1 * np.asarray(phi)
    latitude = np.pi / 2 - np.asarray(theta)
    return longitude, latitude

def image_stats(sky):
    rsky = np.real(sky)
    
    npix = sky.shape[0]
    max_p = np.max(rsky)
    sdev_p = np.std(rsky)
    min_p = np.min(rsky)
    mean_p = np.mean(rsky)
    med_p = np.median(rsky)
    deviation = np.abs(med_p - rsky)
    mad_p = np.median(deviation)
    
    if (mad_p < 1e-9):
        mad_p = sdev_p
        
    logger.info("{{'N_s':{}, 'S/N': {}, 'min': {}, 'max': {}, 'mean': {}, 'sdev': {}, 'R_mad': {}, 'MAD': {}, 'median': {}}}".format(npix, (max_p/sdev_p), min_p, max_p, mean_p, sdev_p, (max_p/mad_p), mad_p, med_p))
    
    return max_p, min_p, mad_p

class HealpixSphere(object):
    ''' 
        A healpix Sphere 
    '''
    
    def __init__(self, nside):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        logger.info("New Sphere, nside={}. npix={}".format(self.nside, self.npix))
        
        self.pixel_indices = np.arange(self.npix)
        theta, phi = hp.pix2ang(nside, self.pixel_indices)
        
        # For limited fields of view
        #healpy.query_polygon

        self.pixels = np.zeros(self.npix) # + hp.UNSEEN
        
        el_r, az_r = hp2elaz(theta, phi)
        
        self.el_r = el_r
        self.az_r = az_r

        self.l, self.m, self.n = elaz2lmn(self.el_r, self.az_r)

    def get_lmn(self):
        return self.l, self.m, self.n

    def index_of(self, el, az):
        theta, phi = elaz2hp(el, az)
        return hp.ang2pix(self.nside, theta, phi)

    def plot_dot(self, el, az):
        theta, phi = elaz2hp(el, az)
        hp.projplot(theta, phi, 'k.', rot=(0,90,0))  #

    def plot_x(self, plt, el, az):
        theta, phi = elaz2hp(el, az)
        hp.projplot(theta, phi, 'ro', rot=(0,90,180))  #

    def set_visible_pixels(self, pix, scale=True):
        # This discards the imaginary part.
        rpix = np.real(pix)
        if scale:
            max_p, min_p, mad_p = image_stats(rpix)
            rpix = (rpix - min_p) / mad_p
        self.pixels = rpix.reshape((len(self.pixels),))
        
    def corners(self, pixel):
        bounds = hp.boundaries(self.nside, pixel, step=1)
        bounds = np.array(bounds).T
        return bounds
        #lat, lon = hp.vec2ang(bounds)
        #ret = []
        #for a,b in zip(lat, lon):
            #ret.append([b,a])
            
        #return ret
        
    
        
    def to_svg(self, fname, pixels_only=False, show_grid=False, src_list=None):

        w = 4000
        dwg = svgwrite.Drawing(filename=fname, size=(w,w), profile='tiny')
        
        #dwg.add(dwg.line((0, 0), (10, 0), stroke=svgwrite.rgb(10, 10, 16, '%')))
        #dwg.add(dwg.text('Test', insert=(0, 0.2), fill='red'))
        
        pc = PlotCoords(w)
        line_size = pc.line_size
        
        #dwg.desc("Gridless imaging from visibilities.")
        
        max_p = np.max(self.pixels)
        min_p = np.min(self.pixels)
        mean_p = np.mean(self.pixels)
        
        sdev_p = np.std(self.pixels)

        if sdev_p > 0:
            logger.info("'N_s':{}, 'S/N': {}, 'min': {}, 'max': {}, 'mean': {}, 'sdev': {}".format(self.npix, (max_p/sdev_p), min_p, max_p, mean_p, sdev_p))
        else:
            logger.info("'N_s':{}, 'S/N': {}, 'min': {}, 'max': {}, 'mean': {}, 'sdev': {}".format(self.npix, 'Inf', min_p, max_p, mean_p, sdev_p))

        med = np.median(self.pixels)

        if not pixels_only:
            svg_pixels = dwg.g(stroke_width=2, stroke_linejoin="round", stroke_opacity=1.0)

        for i in range(self.npix):
            idx = self.pixel_indices[i]
            corners = self.corners(idx) # x,y,z points on boundary
            value = self.pixels[i]
            
            max_lat = -1e99
            min_lat = 1e99
            x_mean = 0.0
            y_mean = 0.0
            
            cnr_lat, cnr_lon = hp.vec2ang(corners, lonlat=False)

            poly = []
            for p, phi, theta in zip(corners, cnr_lon, cnr_lat):
                #logger.info("p = {} lat={}, lon={}".format(p, lat, lon))
                lat = np.pi/2 - theta
                
                max_lat = max(max_lat, lat)
                min_lat = min(min_lat,  lat)
                
                #if theta > np.pi / 2:
                    #logger.info("colatitude {} > PI_OVER_2 ll={}".format(theta, p))
                
                hpang = HpAngle(theta, np.pi + phi)
                (x,y) = hpang.proj()

                poly.append((pc.from_x(x), pc.from_y(y)))
                
                x_mean = x_mean + x
                y_mean = y_mean + y

            x_mean = x_mean/4.0
            y_mean = y_mean/4.0
            
            if max_lat > np.pi/2:  # Ignore points on the horizon
                continue
            
            if pixels_only:
                if min_lat >= 0.00:  # Ignore points on, or below the horizon
                    dlat = max_lat - min_lat
                    logger.info(poly)
                    font_size = pc.from_d(dlat*np.sin(max_lat))//5

                    dwg.add(dwg.polygon(points=poly, fill='none', stroke='black', stroke_linejoin="round", stroke_width=font_size//10))

                    x = pc.from_x(x_mean)
                    y = pc.from_y(y_mean) + (font_size//2)
                    dwg.add(dwg.text("{}".format(i), (x, y), text_anchor='middle', font_size="{}px".format(font_size)))
            else:
                (r, g, b) = cmap((value - min_p)/(max_p - min_p))
                colour = svgwrite.rgb(r, g, b)
                if min_lat > 0.07:  # Ignore points on, or below the horizon
                    svg_pixels.add(dwg.polygon(points=poly, fill=colour, stroke=colour))

        grid_color = 'black'
        if not pixels_only:
            dwg.add(svg_pixels)
            grid_color = 'grey'
            
        if show_grid:
            grid_lines = dwg.g(fill='none', stroke=grid_color, stroke_width="{}".format(line_size), stroke_linejoin="round", stroke_dasharray="{},{}".format(5*line_size, 10*line_size))
            for angle in [30, 60, 90]:
                rad = np.radians(angle)
                radius = pc.from_d(np.sin(rad))
                grid_lines.add(dwg.circle(center=(pc.from_x(0.0), pc.from_y(0.0)), r=radius ))

            for angle in range(0, 360, 30):
                rad = np.radians(angle)
                x = np.sin(rad)
                y = np.cos(rad)
                grid_lines.add(dwg.line(start=(pc.from_x(0.0), pc.from_y(0.0)), end=(pc.from_x(x), pc.from_y(y))))
                
            dwg.add(grid_lines)
        
        if src_list is not None:
            angular_size = np.radians(2.0)
            source_circles = dwg.g(fill='none', stroke='red', stroke_width="{}".format(line_size))
            for s in src_list:
                if s.el_r > np.radians(10.0):
                    elaz = ElAz(s.el_r, s.az_r)
                    (x,y) = pc.from_elaz(elaz)

                    radial_size = angular_size*np.sin(s.el_r)

                    radiusx = pc.from_d(angular_size)
                    radiusy = pc.from_d(radial_size)
                    # Project the source circle onto an ellipse.
                    circ = dwg.ellipse(center=(x,y),
                                      r=(radiusx, radiusy))
                    circ.rotate(-np.degrees(s.az_r), center=(x,y))

                    source_circles.add(circ)
            dwg.add(source_circles)

        dwg.save()



    def plot(self, plt, src_list):
        rot = (0, 90, 0)
        plt.figure() # (figsize=(6,6))
        logger.info('self.pixels: {}'.format(self.pixels.shape))
        if True:
            hp.orthview(self.pixels, rot=rot, xsize=1000, cbar=True, half_sky=True, hold=True)
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
    ''' 
        A healpix subset of a sphere bounded by a range in theta and phi
    '''
    def __init__(self, resolution=None, nside=None, theta=0.0, phi=0.0, radius=0.0):
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]

        if nside is None: # Calculate nside to the appropriate resolution
            nside = 1
            while True:
                nside = nside * 2
                res = hp.nside2resol(nside, arcmin = True)
                logger.info("nside={} res={} arcmin".format(nside, res))
                if res < resolution:
                    break
                
            self.nside = nside
        else:
            self.nside = nside
    
        logger.info("New SubSphere, nside={}".format(self.nside))
        
        x0 = hp.ang2vec(theta, phi)
        
        
        # https://healpy.readthedocs.io/en/latest/generated/healpy.query_polygon.html
        self.pixel_indices = hp.query_disc(nside, x0, radius, inclusive=False, nest=False)
        #self.pixel_indices = my_query_disk(nside, x0, radius)
                
        self.npix = self.pixel_indices.shape[0]
        
        logger.info("New SubSphere, nside={}. npix={}".format(self.nside, self.npix))

        theta, phi = hp.pix2ang(nside, self.pixel_indices)
        

        self.pixels = np.zeros(self.npix) # + hp.UNSEEN
        
        el_r, az_r = hp2elaz(theta, phi)
        
        self.el_r = el_r
        self.az_r = az_r

        self.l, self.m, self.n = elaz2lmn(self.el_r, self.az_r)

    def plot(self, plt, src_list):
        '''
            Modified plot to deal with the reduced number of pixels
        '''
        all_npix = hp.nside2npix(self.nside)
        all_pixels = np.zeros(all_npix) + hp.UNSEEN
        all_pixels[self.pixel_indices] = self.pixels

        rot = (0, 90, 0)
        plt.figure() # (figsize=(6,6))
        logger.info('self.pixels: {}'.format(self.pixels.shape))
        if True:
            hp.orthview(all_pixels, rot=rot, xsize=1000, cbar=True, half_sky=True, hold=True)
            hp.graticule(verbose=False)
        else:
            hp.mollview(all_pixels, rot=rot, xsize=1000, cbar=True)
            hp.graticule(verbose=True)
        
        if src_list is not None:
            for s in src_list:
                self.plot_x(plt, s.el_r, s.az_r)


if __name__=="__main__":
        sph = HealpixSphere(2)
        sph.to_svg(fname='sph.svg', pixels_only=True)

        res_deg = 5
        print("Done")
        big = HealpixSubSphere(resolution=res_deg*60.0, 
                               theta = np.radians(20.0), phi=0.0, radius=np.radians(35))
        print(big.nside)
        big.to_svg(fname='test.svg', pixels_only=True, show_grid=True)
        big.set_visible_pixels(np.random.rand(big.npix))
        big.to_svg(fname='test_map.svg', show_grid=True)
        
