import json
import h5py

import numpy as np

from ..healpix_sphere import HealpixSphere, HealpixSubSphere
from ..sphere import GeoLocation

def from_hdf(filename):
    ret = None
    with h5py.File(filename, "r") as h5f:
        info_string = np.string_(h5f['information'][0]).decode('UTF-8')
        info_json = json.loads(info_string)
        
        fov_type = info_json['fov_type']
        timestamp = info_json['timestamp']
        geolocation = GeoLocation.from_json(info_json['geolocation'])
        centre = info_json['center']

        if fov_type == 'HealpixSphere':
            ret = HealpixSphere.from_hdf(h5f)
        elif fov_type == 'HealpixSubSphere':
            ret = HealpixSubSphere.from_hdf(h5f)
        else:
            raise RuntimeError(f"Unknown field of view class: {fov_type}.")
        
        ret.timestamp = timestamp
        ret.geolocation = geolocation
        ret.centre = centre

    return ret