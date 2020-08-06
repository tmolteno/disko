class MultivariateGaussian:
    '''
        Handle a multivariate gaussian with D dimensions, and real entries.
        
    '''
    def __init__(self, D, mu, sigma):
        '''
            Create a D-dimensional multivariate Gaussian with known mean and standard deviation
        '''
        self.D = D
        self.mu = mu
        self.sigma = sigma
        
        if (mu.shape[0] != D):
            raise ValueError('Mean mu {} must be a {}-vector'.format(mu.shape, D)
        if (sigma.shape[0] != (D,D)):
            raise ValueError('Covariance sigma {} must be a {}x{} square matrix'.format(sigma.shape, D)
        
    def bayes_update(self, lh, prior):
        '''
            Return a new MultivariateGaussian, after update by likelihood.
            
            @param lh, prior
        '''
        pass
    
