import numpy as np

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
    def __init__(self, mu, sigma):
        '''
            Create a D-dimensional multivariate Gaussian with known mean and standard deviation
        '''
        try:
            self.D = mu.shape[0]
        except:
            raise ValueError('Mean mu {} must be a vector'.format(mu.shape))
        self.mu = mu
        self.sigma = sigma
        self.dtype = np.float64
        
        if (sigma.shape[0] != self.D) or (sigma.shape[1] != self.D):
            raise ValueError('Covariance sigma {} must be a {}x{} square matrix'.format(sigma.shape, self.D))
    
        self._sigma_inv = None
        self._A = None
    
    def sigma_inv(self):
        if self._sigma_inv is None:
            self._sigma_inv = np.linalg.inv(self.sigma)
        return self._sigma_inv


    def bayes_update(self, likelihood, measurements):
        '''
            Return a new MultivariateGaussian, after update by measurements, 
            
            @param likelihood, measurements
            
            The self variable is the prior.
            
            See section 3.1 of the documentation
        '''
        sigma_1 = np.linalg.inv(likelihood.sigma_inv() + self.sigma_inv())
        mu_1 = sigma_1 @ (self.sigma_inv() @ self.mu + likelihood.sigma_inv() @ measurements)
        return MultivariateGaussian(mu_1, sigma_1)
    
    
    def linear_transform(self, A, b):
        '''
            Linear transform
            y = A x + b
        '''
        sigma_1 = A @ self.sigma @ self.sigma.T
        mu_1 = A @ self.mu + b
        return MultivariateGaussian(mu_1, sigma_1)
    
    
    def sample(self):
        '''
            Return a sample from this multivariate distribution
        '''
        z = np.random.normal(0, 1, self.D)
        if self._A is None:
            self._A = np.linalg.cholesky(self.sigma)
        
        return self.mu + self._A @ z
        
