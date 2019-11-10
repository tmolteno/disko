#
# The Telescope Operator Class
#
# Does an SVD to get an explicit basis for the null space and range space
# of the telescope
#
#
import logging
import time
import scipy
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from .sphere import HealpixSphere

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def plot_spectrum(s, n_s, n_v, rank, name):
    plt.figure(num=None, figsize=(3, 2), dpi=300, facecolor='w', edgecolor='k')
    plt.plot(s)
    plt.grid(True)
    plt.title('Singular Value Spectrum $N_s={}$,  $N_v={}$, $r={}$'.format(n_s, n_v, rank))
    plt.xlabel('Rank ($i$)')
    plt.ylabel('Singular value $\sigma_i$')
    plt.savefig('{}_Singular_Ns_{}_Nv_{}.pdf'.format(name, n_s, n_v))
    plt.close()

def plot_uv(to, name):
    uv = []
    
    plt.figure(num=None, figsize=(3, 2), dpi=300, facecolor='w', edgecolor='k')
    for u, v, w in zip(to.grid.u_arr, to.grid.v_arr, to.grid.w_arr):
        plt.plot(u,v, '.', color='black')
    
    plt.grid(True)
    plt.title('{} U-V Coverage'.format(name))
    plt.xlabel('u (wavelengths)')
    plt.ylabel('v (wavelengths)')
    plt.savefig('{}_UV.pdf'.format(name))
    plt.close()


def tf_svd(x, tol=1e-4):
    import tensorflow as tf
    s, u, v = tf.linalg.svd(x, full_matrices=True, compute_uv=True)

    with tf.Session().as_default():
        U = u.eval()
        s = s.eval()
        Vh = np.conj(v.eval()).T
        
    try:
        rank = np.min(np.argwhere(s < tol))
    except:
        rank = x.shape[0]
        
    return  [U, s, Vh], rank

def normal_svd(x, tol=1e-4):
    n_v = x.shape[0]
    n_s = x.shape[1]

    [U, s, Vh] = scipy.linalg.svd(np.array(x), full_matrices=True)
    logger.info("s = {}".format(s))
    try:
        rank = np.min(np.argwhere(s < tol))
    except:
        rank = n_v
        
    logger.info("rank = {}".format(rank))

    range_ranks = s >= tol  # Where values are significant
    null_ranks = s < tol  # Where values are top be truncated
    
    s[null_ranks] = 0  # All low values set to 0
        
    return [U, s, Vh], rank


def dask_svd(x, tol=1e-4):
    # Try and use a tall and thin svd computation. Will need to be done on the hermitian transpose of the telescope operator.
    import dask.array as da
    import dask
    n_v = x.shape[0]
    T = np.array(x, dtype=np.complex128)
    rows, cols = T.shape

    A = da.from_array(T, chunks=(cols, rows))

    # Use the transpose uT ST V
    v, s, uT = da.linalg.svd(A.conj().T)
    
    s = s.compute()
    u = uT.conj().T.compute()
    v = v.compute()
    
    logger.info("dask u = {}".format(u.shape))
    logger.info("dask s = {}".format(s))
    logger.info("dask v = {}".format(v.shape))

    try:
        rank = np.min(np.argwhere(s < tol))
    except:
        logger.exception("Error in rank")
        rank = n_v
    logger.info("rank = {}".format(rank))
        
    # Now do QR decomposition to get the full U matrix (it is the u from svd with Q2 added)
    # (1) Add zeros to the sigma matrix
    # (2) Complete V to the full V matrix by Gram-Schmidt Process
    
    q, r = np.linalg.qr(T.conj().T, mode='complete')
    logger.info("dask q = {}".format(q.shape))
    logger.info("dask r = {}".format(r.shape))
    
    return [u, s, q], rank
            

def to_column(x):
    return x.reshape([-1,1])
'''
    The TelescopeOperator class contains all the methods to convert to and from the natural basis.
    
    
'''
class TelescopeOperator:
    '''
        Do an SVD to allow conversion into the natural basis.
        
        Gamma = N_v x N_s (m x n) where n > m (harmonics are row vectors of Gamma)
        
        C^n = R(Gamma^H) \oplus N(Gamma)
        C^m = R(Gamma) \oplus N(Gamma^H)
        
        https://math.stackexchange.com/questions/1771013/null-space-from-svd
        
        The sky is C^n, in other words, we can represent the sky as the direct sum of the null-space of Gamma, and the range of (Gamma^H).
        
        
        Use SVD. Gamma = U Sigma V^h
        ma
        The represent the U and V as block matrices.
        
        Gamma = [U_1 U_2] [Sigma_1 0] [ V_1^H ]
                          [ 0      0] [ V_2^H ]
                          
        The R(Gamma^H) is the column vectors of V_1
        The N(Gamma) is the column vectors of V_2

        The projection of a sky vector into the R(Gamma^H) is V_1 V_1^H
        The projection of a sky vector into the N(Gamma) is V_2 V_2^H
        
        The projection of a visibility vector into the R(Gamma) is U_1 U_1^H
        The projection of a visibility vector into the N(Gamma^H) is U_2 U_2^H

        U_1 are the basis vectors (columns) for the range space of A
        
        Let A = U Sigma = [ U_1 Sigma_1     0 ] = [ A_r 0 ]
                          [       0         0 ]   [  0  0 ]
        
        The v = A x = [ A_r  0 ] [x_r] = A_r x_r = U_1 Sigma_1 V_1^H s
                      [  0   0 ] [x_n]
                      
        where x_r = V_1^H s
              x_n = V_2^H s
        So:
        
        (1) The range space of the telescope consists of vectors U_1 (in the natural basis). These can be transformed back to the sky space by:
                padding with zeros, operating on by V


        (2) The null space of the telescope consists of vectors V_2 (in the sky space). 
        (3) The range space of the telescope (in the sky space) consists of vectors V_1
        (4) Imaging v = U_1 \SIgma_1 V_1^H s, implies
                 Sigma_1^{-1} U_1^H v = V_1^H s
                 V_1 Sigma_1^{-1} U_1^H v = s
                 
            Since A_r = U_1 Sigma_1, this is the same as
            
                 s = V_1 A_r^{-1} v
    '''
    def __init__(self, grid, sphere, use_cache=True):
        self.grid = grid
        self.sphere = sphere
        self.gamma = grid.make_gamma(sphere)
        
        logger.info("Gamma = {}".format(self.gamma.shape))
        self.n_v = self.gamma.shape[0]
        self.n_s = self.gamma.shape[1]
        logger.info("n_v = {}".format(self.n_v))
        logger.info("n_s = {}".format(self.n_s))


        
        fname = 'svd_{}_{}.npz'.format(self.n_s, self.n_v)
        cache = Path(fname)
        
        if use_cache and cache.is_file():
            logger.info("Loading Cache file {}".format(fname))
            npzfile = np.load(fname)
            self.U = npzfile['U']
            self.Vh = npzfile['Vh']
            self.s = npzfile['s']
            self.sigma = npzfile['sigma']
            self.rank = npzfile['rank']
            self.P_r = npzfile['P_r']
            self.V = self.Vh.conj().T
            self.V_1 = self.V[:, 0:self.rank]
            self.V_2 = self.V[:, self.rank:]
            self.V_1h = self.V_1.conj().T
            
        else:
            logger.info("Performing SVD. Writing to cache: {}".format(fname))
            
            ### Take the SVD of the gamma matrix.
            [self.U, self.s, self.Vh], rank = normal_svd(np.array(self.gamma))
            self.rank = rank
            
            logger.info("S = {}".format(self.s))
            self.sigma = scipy.linalg.diagsvd(self.s, self.n_v, self.n_s)

            self.V = self.Vh.conj().T
            self.V_1 = self.V[:, 0:self.rank]
            self.V_2 = self.V[:, self.rank:]
            self.V_1h = self.V_1.conj().T
            logger.info("Calculating orthogonal projections")

            self.P_r = self.V_1 @ self.V_1h # Projection onto the range space of A

            np.savez_compressed(fname, U=self.U, Vh=self.Vh, s=self.s, sigma=self.sigma, rank=rank, P_r=self.P_r)
            logger.info("Cache file {} saved".format(fname))

        logger.info("    U  {} bytes".format(self.U.nbytes))
        logger.info("    Sigma {} bytes".format(self.sigma.nbytes))
        logger.info("    Vh {} bytes".format(self.Vh.nbytes))
        logger.info("U = {}".format(self.U.shape))
        logger.info("V = {}".format(self.Vh.shape))
        logger.info("rank = {}".format(self.rank))

    
        self.A = self.U @ self.sigma # The new telescope operator.
        self.A_r = self.A[0:self.rank, 0:self.rank]

        logger.info("V_1 = {}".format(self.V_1.shape))
        logger.info("V_2 = {}".format(self.V_2.shape))


        logger.info("A = {}".format(self.A.shape))
        logger.info("A_r = {}".format(self.A_r.shape))
        
        logger.info("P_r = {}".format(self.P_r.shape)) # Projection onto the range space of A
        #self.P_n = self.V_2 @ self.V_2.conj().T  # Projection onto the null-space of AA^H
        #logger.info("P_n = {}".format(self.P_n.shape))


    
    def n_s(self):
        return self.n_s
    
    def n_v(self):
        return self.n_v
    
    def n_r(self):  # Dimension of the range space
        return self.rank 
    
    def n_n(self):  # Dimension of the null space
        return self.n_s - self.rank
    
    def harmonic(self, h):
        # Row vectors of gamma
        return self.gamma[h,:]



    def range_harmonic(self, h):
        # The column vector of V_. These are the basis vectors of the Measurable Sky (in the sky space)
        return self.V_1[:, h]
    
    def null_harmonic(self, h):
        # The right singular vectors of the SVD (The columns of V)
        return self.V_2[:, h]
    
        
    def natural_A_row(self, h):
        # The row vector of the natural-basis telescope operator A = U @ Sigma 
        return self.A[h, :]
        

    
    '''
        Convert the natural-basis vector x into the sky basis
    '''
    def natural_to_sky(self, x):
        return self.V @ x
    
    
    '''
        Convert the sky vector to the natural basis. This is the inverse of Vh
        as it is unitary.
    '''
    def sky_to_natural(self, s):
        return self.Vh @ s


    # Project the sky into the null space.
    def sky_to_null(self, s):
        # Storing P_n is very slow as it's a huge projection matrix.
        # self.P_n = self.V_2 @ self.V_2.conj().T  # Projection onto the null-space of A
        ret = np.dot(self.V_2, np.dot(self.V_2.conj().T, s))
        return ret #self.P_n @ s
    

    def null_to_sky(self, x_n):
        x = np.zeros(self.n_s)
        x[self.rank:-1] = x_n
        return natural_to_sky(x)
    

    def image_visibilities(self, vis_arr, sphere, scale=True):
        """Create a gridless image from visibilities

        v = T s, so just use a solver to find s,
        
        Args:

            vis_arr (np.array): An array of visibilities
            nside (int):        The healpix nside parameter.
        """

        logger.info("Imaging Direct nside={}".format(sphere.nside))
        t0 = time.time()

        sky, residuals, rank, s = np.linalg.lstsq(self.gamma, to_column(vis_arr), rcond=None)
        
        t1 = time.time()
        logger.info("Elapsed {}s".format(time.time() - t0))

        sphere.set_visible_pixels(sky, scale)
        return sky


    def image_natural(self, vis_arr, sphere, scale=True):
        """Create a gridless image from visibilities in the natural basis

        v = A_r x_r, so just use a solver to find x_r,
        
        then sky = to_sky(x_r)
        
        Args:

            vis_arr (np.array): An array of visibilities
            nside (int):        The healpix nside parameter.
        """

        logger.info("Imaging Natural nside={}".format(sphere.nside))
        t0 = time.time()

        x_r = np.linalg.solve(self.A_r, to_column(vis_arr))
        #x_n = to_column(np.zeros(self.n_n()))
        #logging.info("x_r = {}".format(x_r.shape))
        #logging.info("x_n = {}".format(x_n.shape))

        #x = np.concatenate((x_r, x_n)).reshape([-1,1])
        #logging.info("x = {}".format(x.shape))
        
        #sphere.set_visible_pixels(self.natural_to_sky(x))
        sky = self.V_1 @ x_r
        sphere.set_visible_pixels(sky, scale)
        
        logger.info("Elapsed {}s".format(time.time() - t0))
        
        return sky


    def image_tikhonov(self, vis_arr, sphere, alpha, scale=True):
        ''' Do a Tikhonov regularization solution
            using the power of the SVD!
        '''
        D = np.diag(self.s / (self.s**2 + alpha**2))
        logger.info("D = {}".format(D.shape))
        logger.info("vis_arr = {}".format(vis_arr.shape))

        sky = self.V_1 @ D @ self.U.conj().T @ vis_arr
        sphere.set_visible_pixels(sky, scale)
        return sky


