#
# Copyright Tim Molteno 2022 tim@elec.ac.nz
#

import numpy as np

import astropy.constants as const


def parse_ending(x_str, ending):
    if x_str.endswith(ending):
        return float(x_str.split(ending)[0])
    return None


DEGREES = "deg"
MINUTES = 'arcmin'
SECONDS = 'arcsec'
MILLIARCSECONDS = 'mas'
MICROARCSECONDS = 'uas'


def rayleigh_criterion(bl, frequency):
    '''
        The accepted criterion for determining the diffraction limit to resolution
        developed by Lord Rayleigh in the 19th century.

        approx resolution given by first order Bessel functions
        assuming array is a flat disk of length bl
    '''
    min_wl = const.c.value / frequency
    return (1.220 * min_wl / bl)


class Resolution:
    '''
        Degrees (°), minutes ('), seconds (")
    '''

    def __init__(self, x_rad):
        self.x_rad = x_rad

    @classmethod
    def from_deg(cls, x_deg):
        return cls(np.radians(x_deg))

    @classmethod
    def from_rad(cls, x_rad):
        return cls(x_rad)

    @classmethod
    def from_arcmin(cls, x_arcmin):
        return cls(np.radians(x_arcmin/60))

    @classmethod
    def from_arcsec(cls, x_arcsec):
        return cls(np.radians(x_arcsec/3600))

    @classmethod
    def from_string(cls, x_str):
        endings = [MICROARCSECONDS, MILLIARCSECONDS, SECONDS, MINUTES, DEGREES]
        deg_factors = [3600*1000000, 3600*1000, 3600, 60, 1]
        parsed = [parse_ending(x_str, e) for e in endings]
        for p, f in zip(parsed, deg_factors):
            if p is not None:
                return cls.from_deg(p/f)

        return cls.from_deg(float(x_str))

    def radians(self):
        return self.x_rad

    def degrees(self):
        return np.degrees(self.x_rad)

    def arcmin(self):
        return self.degrees() * 60

    def arcsec(self):
        return self.degrees() * 3600

    def mas(self):
        return np.de

    def __repr__(self):
        d = self.degrees()
        if np.abs(d) > 1:
            return f"{d:4.2f}deg"

        if np.abs(self.arcmin()) > 1:
            return f"{self.arcmin():4.2f}{MINUTES}"

        arcsec = self.arcsec()
        if np.abs(arcsec) > 1:
            return f"{arcsec:4.2f}{SECONDS}"

        mas = arcsec * 1000
        if np.abs(mas) >= 1:
            return f"{mas:4.2f}{MILLIARCSECONDS}"

        uas = mas * 1000
        return f"{uas:4.2f}{MICROARCSECONDS}"

    def get_min_baseline(self, frequency):
        '''
            Get the shortest baseline length (in meters) that will resolve
            this resolution, at the specified frequency.

            Double-slit interferometer (spacing d). Fringe maxima
            occur at angles where
                d * sin(theta) = n * wavelength

            n = 1: sin(theta) = wavelength/d
            n = 2: sin(theta) = 2*wavelength/d

            angular spacing = wavelength / d

            so d = spacing / theta
        '''
        wavelength = const.c.value / frequency
        spacing = wavelength / self.x_rad
        return spacing*2    # Nyquist requires twice this...

    @classmethod
    def from_baseline(cls, bl, frequency):
        res_limit = rayleigh_criterion(bl, frequency)
        return cls(res_limit / 2)  # Nyquist requires twice the resolution
