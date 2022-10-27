#
# Copyright Tim Molteno 2017 tim@elec.ac.nz
#

import unittest
import logging
import json

import numpy as np

from disko import DiSkO
from disko import HealpixSphere, HealpixSubSphere, Resolution

#from tart.operation import settings
#from tart_tools import api_imaging
#from tart.imaging import elaz
#from tart.util import constants

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
logger.setLevel(logging.INFO)

class TestDiSkOMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load data from a MS file
        fname = 'test_data/test.ms'
        logger.info("Getting Data from MS file: {}".format(fname))

        res = Resolution.from_deg(180)
        cls.disko = DiSkO.from_ms(fname, res=res, num_vis=500)
        cls.nside = 16
        cls.sphere = HealpixSphere(cls.nside)
        res_deg = 4.0
        cls.subsphere = HealpixSubSphere.from_resolution(res_arcmin=res_deg*60.0, 
                                      theta = np.radians(0.0), phi=0.0, radius_rad=np.radians(89))

        cls.gamma = cls.disko.make_gamma(cls.sphere)

    def test_load(self):
        self.assertTrue(self.gamma is not None)
