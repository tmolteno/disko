import scipy
import logging
import h5py
import json

import numpy as np
import dask.array as da

from .util import log_array, da_identity, da_block_diag

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def factors(n):
    return np.sort(list(set(
        factor for i in range(1, int(n**0.5) + 1) if n % i == 0
        for factor in (i, n//i)
    )))

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
        self.dtype = np.float64
        try:
            self.D = mu.shape[0]
        except:
            raise ValueError('Mean mu {} must be a vector'.format(mu.shape))
        
        if (sigma is None) and (sigma_inv is None):
            raise ValueError('Either sigma or sigma_inv must be provided')
            
        self.mu = mu.flatten()
        self._sigma = sigma
        self._sigma_inv = sigma_inv
        
        log_array("mu", self.mu)
        
        storage = self.mu.nbytes
        
        if sigma is not None:
            d = sigma.shape
            
            self._sigma = sigma
            log_array("sigma", sigma)
            storage += self._sigma.nbytes
        else:
            self._sigma_inv = sigma_inv
            d = sigma_inv.shape
            log_array("sigma_inv", sigma_inv)
            storage += self._sigma_inv.nbytes
            
        if (d[0] != self.D) or (d[1] != self.D):
            raise ValueError('Covariance {} must be a {}x{} square matrix'.format(d, self.D, self.D))
    
        logger.info("MultivariateGaussian({}, {}): {:.2f} GB".format(mu.shape, d, storage/1e9))
        
        self._A = None

    def sp_inv(self, A):
        '''
            Find the inverse of a postitive definite matrix
        '''
        logger.info("Inverting {} matrix".format(A.shape))
        log_array("A", A)
        D = A.shape[0]
        
        b = da_identity(D).rechunk(A.chunks)
        Ainv = da.linalg.solve(A, b, sym_pos=True)
        return Ainv

    def sigma_inv(self):
        if self._sigma_inv is None:
            self.rechunk()
            self._sigma_inv = self.sp_inv(self._sigma)
        return self._sigma_inv

    def sigma(self):
        if self._sigma is None:
            self.rechunk()
            self._sigma = self.sp_inv(self._sigma_inv)
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
        sigma_1 = self.sp_inv(sigma_1_inv)
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

        return MultivariateGaussian(mu_1, sigma=sigma_1)
    
    def block(self, start, stop):
        sig = self.sigma()
        return MultivariateGaussian(self.mu[start:stop], sigma=sig[start:stop, start:stop])
    
    @classmethod
    def outer(self, a, b):
        logger.info("outer({}, {})".format(a.mu.shape, b.mu.shape))
        mu = da.block([a.mu.flatten(), b.mu.flatten()])
        
        a_s = a.sigma().shape
        b_s = b.sigma().shape
        
        data = [
                [a.sigma(), da.zeros((a_s[0], b_s[1]))],
                [da.zeros((b_s[0], a_s[1])), b.sigma()]
            ]
        sigma = da.block(data)
        return MultivariateGaussian(mu, sigma=sigma)

    @classmethod
    def diagonal(self, mu, diag):
        logger.info("diagonal({}, {})".format(mu.shape, diag.shape))
        sigma = da.diag(diag)
        return MultivariateGaussian(mu, sigma=sigma)

    def rechunk(self, mem_limit=1e8):
        fact = factors(self.D)
        mem = 16 * (fact**2)
        reason = fact[np.argwhere(mem < 1e8)].flatten()
    
        if self._sigma is not None:
            self._sigma = self._sigma.rechunk(reason[-1])
            log_array("sigma", self._sigma)

        if self._sigma_inv is not None:
            self._sigma_inv = self._sigma_inv.rechunk(reason[-1])
            log_array("sigma_inv", self._sigma_inv)

    def sample(self):
        '''
            Return a sample from this multivariate distribution
        '''
        z = da.random.normal(0, 1, self.D)
        if self._A is None:
            self.rechunk()
            logger.info("Cholesky factoring...")

            sigma = self.sigma() + da_identity(self.D)*1e-9
            
            self._A = da.linalg.cholesky(sigma)
            logger.info("          ...done")
            self._A.persist()
            
        return np.array(self.mu + self._A @ z)
        


    def cg_sample(self, epsilon=1e-9):
        '''
            Return a sample from this multivariate distribution using the algorithm of Fox and Parker
            
            http://www.physics.otago.ac.nz/data/fox/publications/ParkerFox_CG_Sampling.pdf
        '''

        self.rechunk()
        A = self.sigma()
        
        logger.info("CG sampling...")

        while True:
            

            self._A = da.linalg.cholesky()
            logger.info("          ...done")
            self._A.persist()
            
        return np.array(self.mu + self._A @ z)
        

    def variance(self):
        var = np.diagonal(self.sigma())
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
