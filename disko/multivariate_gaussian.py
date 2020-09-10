import scipy
import logging
import h5py
import json

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)

class MultivariateGaussian:
    '''
        Handle a multivariate gaussian with D dimensions, and real entries.
        
        The basic idea is to take :
            * measurements v_0, with covariance sigma_v
            * model A, such that v = A x
        
        lh = MultivariateGaussian(v_0, sigma_v) % model for the measurement process, PDF for a measurmeent v
        
        prior = MultivariateGaussian(mu_x, sigma_x)   % distribution over vectors x
        
        posterior = bayes_update(lh, prior)
        
    '''
    def __init__(self, mu, sigma=None, sigma_inv=None):
        '''
            Create a D-dimensional multivariate Gaussian with known mean and standard deviation
        '''
        self.dtype = np.float32
        try:
            self.D = mu.shape[0]
        except:
            raise ValueError('Mean mu {} must be a vector'.format(mu.shape))
        
        if (sigma is None) and (sigma_inv is None):
            raise ValueError('Either sigma or sigma_inv must be provided')
            
        self.mu = np.array(mu.flatten(), dtype=self.dtype)
        self._sigma = sigma
        self._sigma_inv = sigma_inv
        
        if sigma is not None:
            d = sigma.shape
            self._sigma = np.array(sigma, dtype=self.dtype)
        else:
            self._sigma_inv = np.array(sigma_inv, dtype=self.dtype)
            d = sigma_inv.shape
            
        if (d[0] != self.D) or (d[1] != self.D):
            raise ValueError('Covariance {} must be a {}x{} square matrix'.format(d, self.D, self.D))
    
        logger.info("MultivariateGaussian({}, {})".format(mu.shape, d))
        
        self._A = None
    
    def sigma_inv(self):
        if self._sigma_inv is None:
            self._sigma_inv = np.linalg.inv(self._sigma)
        return self._sigma_inv

    def sigma(self):
        if self._sigma is None:
            self._sigma = np.linalg.inv(self._sigma_inv)
        return self._sigma


    def bayes_update(self, precision_y, y, A):
        '''
            Return a new MultivariateGaussian, after update by measurements, 
            
            @param likelihood, measurements
            
            The self variable is the prior.
            
            See https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf p92
        '''
        logger.info("bayes_update({}, {}, {})".format(precision_y.shape, y.shape, A.shape))
        
        L = precision_y
        atl = A.T @ L
        
        sigma_1_inv = self.sigma_inv() + atl @ A
        sigma_1 = np.linalg.inv(sigma_1_inv)
        mu_1 = sigma_1 @ (self.sigma_inv() @ self.mu + atl @ y)
        return MultivariateGaussian(mu_1, sigma=sigma_1, sigma_inv = sigma_1_inv)
    
    
    def linear_transform(self, A, b=None):
        '''
            Linear transform
            y = A x + b
        '''
        sigma_1 = A @ self.sigma() @ A.T
        if b is None:
            mu_1 = A @ self.mu
        else:
            mu_1 = A @ self.mu + b

        return MultivariateGaussian(mu_1, sigma_1)
    
    def block(self, start, stop):
        sig = self.sigma()
        return MultivariateGaussian(self.mu[start:stop], sigma=sig[start:stop, start:stop])
    
    @classmethod
    def outer(self, a, b):
        logger.info("outer({}, {})".format(a.mu.shape, b.mu.shape))
        mu = np.block([a.mu.flatten(), b.mu.flatten()])
        sigma = scipy.linalg.block_diag(a.sigma(), b.sigma())
        return MultivariateGaussian(mu, sigma=sigma)

    def sample(self):
        '''
            Return a sample from this multivariate distribution
        '''
        z = np.random.normal(0, 1, self.D)
        if self._A is None:
            self._A = np.linalg.cholesky(self.sigma())
        
        return self.mu + self._A @ z
        

    def variance(self):
        var = self.sigma().diagonal()
        return np.sqrt(var)


    def to_hdf5(self, filename, json_info="{}"):
        ''' Save the MultivariateGaussian object,
            to a portable HDF5 format
        '''
        logger.info("Writing MultivariateGaussian to HDF5 {} {}".format(filename, self.sigma().dtype))
        with h5py.File(filename, "w") as h5f:
            conftype = h5py.special_dtype(vlen=bytes)
            
            conf_dset = h5f.create_dataset('info', (1,), dtype=conftype)
            conf_dset[0] = json_info

            h5f.create_dataset('sigma', data=self.sigma(), compression="gzip", compression_opts=9)
            h5f.create_dataset('sigma_inv', data=self.sigma_inv(), compression="gzip", compression_opts=9)
            h5f.create_dataset('mu', data=self.mu, compression="gzip", compression_opts=9)


    @classmethod
    def from_hdf5(cls, filename):
        logger.info("Loading MultivariateGaussian from HDF5 {}".format(filename))
        
        with h5py.File(filename, "r") as h5f:

            mu = h5f['mu'][:]
            sigma = h5f['sigma'][:]
            sigma_inv = h5f['sigma_inv'][:]

        return MultivariateGaussian(mu=mu, sigma=sigma, sigma_inv=sigma_inv)
