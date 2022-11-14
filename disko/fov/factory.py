import json
import h5py

import numpy as np

from ..healpix_sphere import HealpixSphere, HealpixSubSphere


def from_hdf(filename):
    ret = None
    with h5py.File(filename, "r") as h5f:
        info_string = np.string_(h5f['information'][0]).decode('UTF-8')
        info_json = json.loads(info_string)
        
        fov_type = info_json['fov_type']
        
        if fov_type == 'HealpixSphere':
            ret = HealpixSphere.from_hdf(h5f)
        elif fov_type == 'HealpixSubSphere':
            ret = HealpixSubSphere.from_hdf(h5f)
        else:
            raise RuntimeError(f"Unknown field of view class: {fov_type}.")
    return ret
