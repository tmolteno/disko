import json
import h5py
import datetime
import logging

import numpy as np
from tart.util import utc


from ..healpix_sphere import HealpixSphere, HealpixSubSphere
from ..sphere_mesh import AdaptiveMeshSphere
from ..sphere import GeoLocation

logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def from_hdf(filename):
    ret = None
    with h5py.File(filename, "r") as h5f:
        info_string = np.string_(h5f['information'][0]).decode('UTF-8')
        info_json = json.loads(info_string)
        
        fov_type = info_json['fov_type']
        timestamp = utc.to_utc(datetime.datetime.fromisoformat(info_json['timestamp']))
        geolocation = GeoLocation.from_json(info_json['geolocation']).loc
        centre = info_json['center']

        logger.info(f"Sphere timestamp: {timestamp.isoformat()}")
        logger.info(f"Sphere location: {info_json['geolocation']}")

        if fov_type == 'HealpixSphere':
            ret = HealpixSphere.from_hdf(h5f)
        elif fov_type == 'HealpixSubSphere':
            ret = HealpixSubSphere.from_hdf(h5f)
        elif fov_type == 'AdaptiveMeshSphere':
            ret = AdaptiveMeshSphere.from_hdf(h5f)
        else:
            raise RuntimeError(f"Unknown field of view class: {fov_type}.")
        
        ret.timestamp = timestamp
        ret.geolocation = geolocation
        ret.centre = centre

    return ret
