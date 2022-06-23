import numpy as np

class Resolution:
    
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
    
    def __repr__(self):
        d = np.degrees(self.x_rad)
        if d > 1:
            return f"{np.d:4.2f} deg"
        if d > 1/60:
            return f"{d*60:4.2f}\""
        
        return f"{d*3600:4.2f}'"
