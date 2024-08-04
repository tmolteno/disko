import json
import h5py
import datetime
import logging

import numpy as np
from tart.util import utc


from ..healpix_sphere import HealpixFoV, HealpixSubFoV
from ..sphere_mesh import AdaptiveMeshFoV
from ..sphere import GeoLocation

logger = logging.getLogger(__name__)

def from_hdf(filename):
    ret = None
    with h5py.File(filename, "r") as h5f:
        info_string = np.string_(h5f['information'][0]).decode('UTF-8')
        info_json = json.loads(info_string)
        
        fov_type = info_json['fov_type']
        timestamp = utc.to_utc(datetime.datetime.fromisoformat(info_json['timestamp']))
        geolocation = GeoLocation.from_json(info_json['geolocation']).loc
        centre = info_json['center']

        logger.info(f"FoV timestamp: {timestamp.isoformat()}")
        logger.info(f"FoV location: {info_json['geolocation']}")

        if fov_type == 'HealpixFoV':
            ret = HealpixFoV.from_hdf(h5f)
        elif fov_type == 'HealpixSubFoV':
            ret = HealpixSubFoV.from_hdf(h5f)
        elif fov_type == 'AdaptiveMeshFoV':
            ret = AdaptiveMeshFoV.from_hdf(h5f)
        else:
            raise RuntimeError(f"Unknown field of view class: {fov_type}.")
        
        ret.timestamp = timestamp
        ret.geolocation = geolocation
        ret.centre = centre

    return ret
