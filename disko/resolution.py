import numpy as np

def parse_ending(x_str, ending):
    if x_str.endswith(ending):
        return float(x_str.split(ending)[0])
    return None

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

    @classmethod
    def from_string(cls, x_str):
        endings = ['uas', 'mas', '"', '\'', 'deg']
        deg_factors = [3600*1000000, 3600*1000, 3600, 60, 1]
        parsed = [ parse_ending(x_str, e) for e in endings]
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
        d = np.degrees(self.x_rad)
        if np.abs(d) > 1:
            return f"{d:4.2f}deg"
        
        if np.abs(self.arcmin()) > 1:
            return f"{self.arcmin():4.2f}\""
        
        arcsec = self.arcsec()
        if np.abs(arcsec) > 1:
            return f"{arcsec:4.2f}'"
        
        mas = arcsec * 1000
        if np.abs(mas) >= 1:
            return f"{mas:4.2f}mas"

        uas = mas * 1000
        return f"{uas:4.2f}uas"
