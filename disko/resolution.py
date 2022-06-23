import numpy as np

class Resolution:
    
    def __init__(self, x_rad):
        self.x_rad = x_rad
        self.x_deg = np.degrees(x_rad)
        
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
    
    def arcmin(self):
        return self.x_deg * 60

    def arcsec(self):
        return self.x_deg * 3600

    def mas(self):
        return np.de
    def __repr__(self):
        d = np.degrees(self.x_rad)
        if d > 1:
            return f"{np.d:4.2f} deg"
        
        if self.arcmin() > 1:
            return f"{self.arcmin():4.2f}\""
        
        arcsec = self.arcsec()
        if arcsec > 1:
            return f"{arcsec:4.2f}'"
        
        mas = arcsec * 1000
        if mas > 1:
            return f"{mas:4.2f} mas"

        uas = mas * 1000
        if uas > 1:
            return f"{uas:4.2f} uas"
