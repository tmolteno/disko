# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3

import json
import h5py
import datetime
import logging

import numpy as np
from tart.util import utc

from ..coords import PhaseCenter
from .fov import GeoLocation
from .healpix import HealpixFoV, HealpixSubFoV
from .mesh import AdaptiveMeshFoV

logger = logging.getLogger(__name__)


def from_hdf(filename):
    ret = None
    with h5py.File(filename, "r") as h5f:
        info_string = np.bytes_(h5f['information'][0]).decode('UTF-8')
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

        # Optional (issue #10, Phase 1): files written before the phase
        # centre became first-class data have no 'phase_center' key and
        # keep loading with the zenith-derived fallback; readers of older
        # versions ignore this key, so files written here still load there.
        phase_center_json = info_json.get('phase_center')
        if phase_center_json is not None:
            ret.phase_center = PhaseCenter.from_dict(phase_center_json)
            # Fill anything the file did not carry from the file's own
            # timestamp/location, the same way set_info() would.
            if ret.phase_center.obstime is None:
                ret.phase_center.obstime = timestamp
            if ret.phase_center.geolocation is None:
                ret.phase_center.geolocation = geolocation

    return ret
