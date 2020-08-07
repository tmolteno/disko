#
# Copyright Tim Molteno 2019 tim@elec.ac.nz
#

import unittest
import logging
import os

import numpy as np

from disko import multivariate_gaussian as mg

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler()) # Add a null handler so logs can go somewhere
LOGGER.setLevel(logging.INFO)

class TestMultivariate(unittest.TestCase):


    def test_linear(self):
        # Test a really silly example of a linear transformation
        
        mu = np.zeros((1))
        sigma = np.zeros((1,1))
        sigma[0,0] = 1
        x = mg.MultivariateGaussian(mu+1, sigma)

        y = x.linear_transform(sigma*3, mu+2)
        
        self.assertAlmostEqual(y.sigma[0,0], 3)
        self.assertAlmostEqual(y.mu[0], 5)

