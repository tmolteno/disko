#
# Copyright Tim Molteno 2017-2019 tim@elec.ac.nz
#

import os
import unittest

import numpy as np

from disko import plsqr


class TestProjectionLSQR(unittest.TestCase):

    def setUp(self):
        pass

    @unittest.skip("Not even started going")
    def test_random(self):
        A = np.random.random((10, 8))
        v = np.random.random(10)

        x = plsqr(A, v, alpha=0.01)
