#
# The Telescope Operator Class
#
# Does an SVD to get an explicit basis for the null space and range space
# of the telescope
#
#
from .util import log_array, da_identity, da_diagsvd
import logging
import time
import scipy
import numpy as np
import matplotlib.pyplot as plt

import dask.array as da
import dask


import h5py

from pathlib import Path

from .disko import vis_to_real
from .multivariate_gaussian import MultivariateGaussian

logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)

SVD_TOL = 1e-3
USE_DASK = True


def plot_spectrum(s, n_s, n_v, rank, name):
    plt.figure(num=None, figsize=(6, 4), dpi=300, facecolor="w", edgecolor="k")
    plt.plot(s)
    plt.grid(True)
    plt.title(
        "Singular Value Spectrum $N_s={}$,  $N_v={}$, $r={}$".format(
            n_s, n_v, rank)
    )
    plt.xlabel("Rank ($i$)")
    plt.ylabel("Singular value $\\sigma_i$")
    plt.savefig("{}_Singular_Ns_{}_Nv_{}.pdf".format(name, n_s, n_v))
    plt.close()


def plot_uv(to, name):
    to.plot_uv(name)


def tf_svd(x, tol=SVD_TOL):
    import tensorflow as tf

    s, u, v = tf.linalg.svd(x, full_matrices=True, compute_uv=True)

    with tf.Session().as_default():
        U = u.eval()
        s = s.eval()
        Vh = np.conj(v.eval()).T

    try:
        rank = np.min(np.argwhere(s < tol))
    except:
        rank = s.shape[0]

    return [U, s, Vh], rank


MAX_COND = 10000.0


def normal_svd(x, tol=SVD_TOL):
    n_v = x.shape[0]
    n_s = x.shape[1]

    [U, s, Vh] = scipy.linalg.svd(np.array(x), full_matrices=True)
    logger.info("Cond(A) = {}".format(s[0] / s[-1]))

    tol = s[0] / MAX_COND
    try:
        rank = np.min(np.argwhere(s < tol))
    except:
        # OK its full rank.
        rank = s.shape[0]

    logger.info("rank = {}".format(rank))

    range_ranks = s >= tol  # Where values are significant
    null_ranks = s < tol  # Where values are top be truncated

    s[null_ranks] = 0  # All low values set to 0

    sigma = scipy.linalg.diagsvd(s, n_v, n_s)

    return [U, sigma, Vh], s, rank


def dask_svd(x, tol=SVD_TOL):
    # Try and use a tall and thin svd computation. Will need to be done on the hermitian transpose of the telescope operator.

    n_v = x.shape[0]
    n_s = x.shape[1]

    log_array("x", x)

    # if False:
    # A = x.rechunk('auto', n_v)
    # log_array("A", A)
    # ns = min(n_v, n_s)
    # U, s, Vh = da.linalg.svd_compressed(A, k=ns, n_power_iter=4)
    # else:

    A = x.rechunk((n_v, "auto"))
    log_array("A", A)

    # Do the SVD
    U, s, Vh = scipy.linalg.svd(A, full_matrices=True)

    #sigma = da_diagsvd(s, n_v, n_s)
    sigma = scipy.linalg.diagsvd(s, n_v, n_s)

    U = da.from_array(U, chunks="auto")
    Vh = da.from_array(Vh, chunks="auto")

    # Clean up by condition number
    s_0 = np.amax(s)
    tol = s_0 / MAX_COND
    rank = np.sum(s > tol, dtype=int)

    logger.info("tol = {}".format(tol))
    logger.info("Cond(A) = {}".format(s_0 / np.amin(s)))

    s[s < tol] = 0  # All low values set to 0

    log_array("U", U)
    log_array("s", s)
    log_array("Vh", Vh)

    return [U, sigma, Vh], s, rank


def to_column(x):
    return x.reshape([-1, 1])


"""
    The TelescopeOperator class contains all the methods to convert to and from the natural basis.
"""


class TelescopeOperator:
    r"""
    Do an SVD to allow conversion into the natural basis.

    Gamma = N_v x N_s (m x n) where n > m (harmonics are row vectors of Gamma)

    C^n = R(Gamma^H) \\oplus N(Gamma)
    C^m = R(Gamma) \\oplus N(Gamma^H)

    https://math.stackexchange.com/questions/1771013/null-space-from-svd

    The sky is C^n, in other words, we can represent the sky as the direct sum of the null-space of Gamma, and the range of (Gamma^H).


    Use SVD. Gamma = U Sigma V^H
    ma
    The represent the U and V as block matrices.

    Gamma = [U_1 U_2] [Sigma_1 0] [ V_1^H ]
                      [ 0      0] [ V_2^H ]

    The R(Gamma^H) is the column vectors of V_1
    The N(Gamma) is the column vectors of V_2

    The projection of a sky vector into the R(Gamma^H) is V_1 V_1^H
    The projection of a sky vector into the N(Gamma) is V_2 V_2^H

    The projection of a visibility vector into the R(Gamma) is U_1 U_1^H
    The projection of a sky vector into the N(Gamma^H) is U_2 U_2^H

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
    (3) The range space of the telescope Adjoint (in the sky space) consists of vectors V_1
    (4) Imaging v = U_1 \SIgma_1 V_1^H s, implies
             Sigma_1^{-1} U_1^H v = V_1^H s
             V_1 Sigma_1^{-1} U_1^H v = s

        Since A_r = U_1 Sigma_1, this is the same as

             s = V_1 A_r^{-1} v
    """

    def __init__(self, grid, sphere, use_cache=False):
        self.grid = grid
        self.sphere = sphere

        _gamma = grid.make_gamma(sphere)  #, makecomplex=True)
        self.n_v = _gamma.shape[0]
        self.n_s = _gamma.shape[1]

        # Create temporary file for gamma
        # with h5py.File('gamma.hdf', "w") as h5f:
        # log_array("_gamma", _gamma)
        # h5f.create_dataset('gamma',data=_gamma, chunks=(10, self.n_v))

        # f = h5py.File('gamma.hdf')
        if USE_DASK:
            self.gamma = da.from_array(_gamma)
        else:
            self.gamma = _gamma

        log_array("Gamma", self.gamma)

        logger.info("n_v = {}".format(self.n_v))
        logger.info("n_s = {}".format(self.n_s))

        self._P_r = None

        fname = "svd_{}_{}.npz".format(self.n_s, self.n_v)
        cache = Path(fname)

        if use_cache and cache.is_file():
            logger.info("Loading Cache file {}".format(fname))
            npzfile = np.load(fname)
            self.U = npzfile["U"]
            self.Vh = npzfile["Vh"]
            self.s = npzfile["s"]
            self.sigma = npzfile["sigma"]
            self.rank = npzfile["rank"]
            self.V = self.Vh.conj().T
            self.V_1 = self.V[:, 0: self.rank]
            self.V_2 = self.V[:, self.rank:]

        else:
            logger.info("Performing SVD.")

            # Take the SVD of the gamma matrix.
            if USE_DASK:
                [self.U, self.sigma, self.Vh], self.s, self.rank = dask_svd(
                    self.gamma)
            else:
                [self.U, self.sigma, self.Vh], self.s, self.rank = normal_svd(
                    np.array(self.gamma)
                )

            self.V = self.Vh.T
            self.V_1 = self.V[:, 0: self.rank]
            self.V_2 = self.V[:, self.rank:]
            # logger.info("Calculating orthogonal projections")

            # self._P_r = self.V_1 @ self.V_1h # Projection onto the range space of A

            if use_cache:
                logger.info("Writing to cache: {}".format(fname))
                np.savez_compressed(
                    fname, U=self.U, Vh=self.Vh, s=self.s, sigma=self.sigma, rank=self.rank
                )
                logger.info("Cache file {} saved".format(fname))

        log_array("U", self.U)
        log_array("Sigma", self.sigma)
        log_array("Vh", self.Vh)

        logger.info("rank = {}".format(self.rank))

        self.U_1 = self.U[:, 0: self.rank]
        self.U_2 = self.U[:, self.rank:]

        self.sigma_1 = self.sigma[0: self.rank, 0: self.rank]

        self.A_r = self.U_1 @ self.sigma_1  #

        log_array("V_1", self.V_1)
        log_array("V_2", self.V_2)
        log_array("A_r", self.A_r)

        # logger.info("P_r = {}".format(self._P_r.shape)) # Projection onto the range space of A
        # self.P_n = self.V_2 @ self.V_2.conj().T  # Projection onto the null-space of AA^H
        # logger.info("P_n = {}".format(self.P_n.shape))

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
        return self.gamma[h, :]

    def P_r(self):
        if self._P_r is None:
            V_1h = self.V_1.conj().T
            self._P_r = self.V_1 @ V_1h  # Projection onto the range space of A
            logger.info(
                "P_r = {}".format(self._P_r.shape)
            )  # Projection onto the range space of A
        return self._P_r

    def range_harmonic(self, h):
        # The column vector of V_. These are the basis vectors of the Measurable Sky (in the sky space)
        return self.V_1[:, h]

    def null_harmonic(self, h):
        # The right singular vectors of the SVD (The columns of V)
        return self.V_2[:, h]

    def natural_A_row(self, h):
        # The row vector of the natural-basis telescope operator A = U @ Sigma
        A = self.U @ self.sigma  # The new telescope operator.

        return A[h, :]

    """
        Convert the natural-basis vector x into the sky basis
    """
    def natural_to_sky(self, x):
        return self.V @ x

    """
        Convert the sky vector to the natural basis. This is the inverse of Vh
        as it is unitary.
    """

    def sky_to_natural(self, s):
        return self.Vh @ s

    # Project the sky into the null space.
    def sky_to_null(self, s):
        # Storing P_n is very slow as it's a huge projection matrix.
        # self.P_n = self.V_2 @ self.V_2.conj().T  # Projection onto the null-space of A
        ret = np.dot(self.V_2, np.dot(self.V_2.conj().T, s))
        return ret  # self.P_n @ s

    def null_to_sky(self, x_n):
        x = np.zeros(self.n_s)
        x[self.rank: -1] = x_n
        return self.natural_to_sky(x)

    def image_visibilities(self, vis_arr, sphere, scale=True):
        """Create a gridless image from visibilities

        v = T s, so just use a solver to find s,

        Args:

            vis_arr (np.array): An array of visibilities
            nside (int):        The healpix nside parameter.
        """

        if vis_arr.shape[0] != self.n_v:
            raise ValueError(
                "Visibility array {} wrong shape {}".format(
                    vis_arr.shape, self.n_v)
            )

        logger.info(
            "Imaging Direct gamma={}, vis={}".format(
                self.gamma.shape, vis_arr.shape)
        )
        t0 = time.time()

        sky, residuals, rank, s = np.linalg.lstsq(
            np.array(self.gamma), np.array(vis_arr), rcond=None
        )

        t1 = time.time()
        logger.info("Elapsed {}s".format(time.time() - t0))

        sphere.set_visible_pixels(sky.flatten(), scale)
        return sky

    def image_natural(self, vis_arr, sphere, scale=True):
        """Create a gridless image from visibilities in the natural basis

        v = A_r x_r, so just use a solver to find x_r,

        then sky = to_sky(x_r)

        Since in the natural basis, A_r is diagonal.
        Args:

            vis_arr (np.array): An array of visibilities
            nside (int):        The healpix nside parameter.
        """

        logger.info("Imaging Natural nside={}".format(sphere.nside))
        t0 = time.time()

        s = self.s[0: self.rank]
        D = np.diag(s)  # / (s**2 + 0.25)) # np.diag(1.0/self.s[0:self.rank])

        logger.info("vis_arr = {}".format(vis_arr.shape))
        logger.info("A_r = {}".format(self.A_r.shape))

        #x_r = D @ self.U_1.T @ vis_arr 
        v_n = self.U_1.T @ vis_arr
        
        x_r = np.linalg.solve(self.A_r, v_n)
        # x_n = np.zeros(self.n_n())
        logger.info("x_r = {}".format(x_r.shape))

        # x = np.block([x_r.flatten(), x_n])
        # logging.info("x = {}".format(x.shape))

        # sky = self.natural_to_sky(x)
        sky = self.V_1 @ x_r
        logger.info("sky = {}".format(sky.shape))
        sphere.set_visible_pixels(sky.flatten(), scale)

        logger.info("Elapsed {}s".format(time.time() - t0))

        return sky

    def image_tikhonov(self, vis_arr, sphere, alpha, scale=True):
        """Do a Tikhonov regularization solution
        using the power of the SVD!
        """
        D = np.array(self.sigma).T
        np.fill_diagonal(D, self.s / (self.s ** 2 + alpha ** 2), wrap=False)
        logger.info("D = {}".format(D.shape))
        logger.info("vis_arr = {}".format(vis_arr.shape))

        sky = self.V @ D @ self.U.conj().T @ vis_arr
        sphere.set_visible_pixels(sky, scale)
        return sky

    def sequential_inference(self, prior, vis_arr, sigma_precision):
        """
            Perform the Bayesian Update of the prior sky. Return the posterior.

            prior is a MultivariateGaussian object

            v = U D V^H s
            U^H v = D V^H s


                y = U^H v
                A = D
                x = V^H s = (V_1h V_2h) s

        v = Gamma s = U Sigma V^H

        Gamma = [U_1 U_2] [Sigma_1 0] [ V_1^H ]
                          [ 0      0] [ V_2^H ]

        Let A = U Sigma = [ U_1 Sigma_1     0 ] = [ A_r 0 ]
                          [       0         0 ]   [  0  0 ]

        The v = A x = [ A_r  0 ] [x_r] = A_r x_r = U_1 Sigma_1 V_1^H s
                      [  0   0 ] [x_n]

        where x_r = V_1^H s
              x_n = V_2^H s

        So the steps are

        1) Solve [U_1^H] v =  [Sigma_1 0] [ V_1^H ] s
                 [U_2^H]      [ 0      0] [ V_2^H ]

                 [U_1^H v] =  [Sigma_1 0] [ V_1^H s ]
                 [U_2^H v]    [ 0      0] [ V_2^H s ]

                 [U_1^H v] =  [Sigma_1 0] [ x_r ]
                 [U_2^H v]    [ 0      0] [ x_n ]


                 [U_1^H v] =  [Sigma_1 x_r]
                 [U_2^H v]    [ 0         ]


                 U_1^H v =  Sigma_1 x_r

                 x_r = inv(Sigma_1) y

        """
        logger.info("Bayesian Inference of sky (n_s = {})".format(prior.D))
        t0 = time.time()

        # s = self.s[0:self.rank]
        # A = self.U_1.T @ s # np.diag(s / (s**2 + 0.25)) # np.diag(1.0/self.s[0:self.rank])

        logger.info("y_m = {}".format(vis_arr.shape))

        # Pull the block from the natural_prior that is the range_space prior
        prior_r = prior.block(0, self.rank)
        prior_n = prior.block(self.rank, self.n_s)

        posterior_r = prior_r.bayes_update(sigma_precision, vis_arr, self.A_r)
        posterior_n = prior_n

        posterior = MultivariateGaussian.outer(posterior_r, posterior_n)
        posterior = posterior.linear_transform(self.V)

        logger.info("Elapsed {}s".format(time.time() - t0))
        return posterior

    def get_prior(self):

        # What range should the image have.
        p05, p50, p95, p100 = self.grid.vis_stats()
        var = p95 * p95
        logger.info("Sky Prior variance={}".format(var))
        prior = MultivariateGaussian(
            np.zeros(self.n_s) + p50, sigma=var * np.identity(self.n_s)
        )
        # natural_prior = prior.linear_transform(self.Vh)

        return prior

    def plot_uv(self, name):
        uv = []

        plt.figure(num=None, figsize=(5, 4), dpi=300,
                   facecolor="w", edgecolor="k")
        for u, v, w in zip(self.grid.u_arr, self.grid.v_arr, self.grid.w_arr):
            plt.plot(u, v, ".", color="black")

        plt.grid(True)
        plt.title("{} U-V Coverage".format(name))
        plt.xlabel("u (m)")
        plt.ylabel("v (m)")
        plt.tight_layout()
        plt.savefig("{}_UV.pdf".format(name))
        plt.close()
