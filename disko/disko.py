#!/usr/bin/env python

# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3
#
# The DiSkO algorithm for imaging without gridding.
#
# Tim Molteno 2017-2019 tim@elec.ac.nz
#
import logging
import time

import dask.array as da
import healpy as hp
import numpy as np
import pylops
import scipy
import scipy.sparse.linalg as spalg
from astropy import constants as const
from sklearn import linear_model

from .healpix_sphere import HealpixFoV
from .multivariate_gaussian import MultivariateGaussian
from .resolution import Resolution

logger = logging.getLogger(__name__)


def get_all_uvw(ant_pos):
    """
    Little helper function to get the UVW positions from the antennas positions.
    The test (i != j) can be changed to (i > j) to avoid the duplicated conjugate
    measurements.
    ant pos is an array of (N_ant, 3)
    """
    # logger.info(f"get_all_uvw({ant_pos})")
    if ant_pos.shape[1] != 3:
        raise RuntimeError(
            "Ant pos (shape={}) must be an array of (N_ant, 3)".format(ant_pos.shape)
        )
    baselines = []
    num_ant = len(ant_pos)
    ant_p = np.array(ant_pos)
    for i in range(num_ant):
        for j in range(num_ant):
            if i != j:
                baselines.append([i, j])

    bl_pos = ant_p[np.array(baselines).astype(int)]
    uu_a, vv_a, ww_a = (bl_pos[:, 0] - bl_pos[:, 1]).T
    return baselines, uu_a, vv_a, ww_a


def to_column(x):
    return x.reshape([-1, 1])


def vis_to_real(vis_arr):
    return np.concatenate((np.real(vis_arr), np.imag(vis_arr)))


REAL_DATATYPE = np.float64
COMPLEX_DATATYPE = np.complex128


def omega(freq):
    r"""
    Little routine to convert a frequency into omega
    """
    wavelength = const.c.value / freq
    return 2 * np.pi / wavelength


def jomega(freq):
    r"""
    Little routine to convert a frequency into j*omega
    """
    return 1.0j * omega(freq)


def get_harmonic(p2j, l, m, n_minus_1, u, v, w, pixel_areas):  # noqa: E741 (l is a direction cosine)
    harmonic = np.exp(p2j * (u * l + v * m + w * n_minus_1)) * pixel_areas
    return harmonic


class DiSkOOperator(pylops.LinearOperator):
    """
    Linear operator for the telescope with a discrete sky
    """

    def __init__(self, u_arr, v_arr, w_arr, data, frequencies, sphere):
        super().__init__()
        self.N = sphere.npix  # Number of pixels
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr
        self.dtype = REAL_DATATYPE
        self.iteration_count = 0

        try:
            self.n_v, self.n_freq, self.npol = data.shape
        except Exception:
            raise RuntimeError("Data must be of the shape [n_v*2, n_freq, n_pol]")

        if self.n_v != len(self.u_arr) * 2:
            raise RuntimeError(
                "Vis data must be split into [real, imag] {} {}".format(
                    self.n_v, self.u_arr.shape
                )
            )

        self.M = self.n_v * self.n_freq

        self.frequencies = np.array(frequencies)
        self.sphere = sphere

        if self.sphere.l.shape[0] != self.N:
            raise RuntimeError(
                f"self.sphere.l.shape != self.N, {self.sphere.l.shape} != {self.N}"
            )

        self.shape = (self.M, self.N)
        self.explicit = False  # Can't be directly inverted
        logger.info("Creating DiSkOOperator data={}".format(self.shape))

    def __call__(self, x):
        # A callback to use during optimization routintes, should be used to write some temporary results
        if (self.iteration_count > 0) and (self.iteration_count % 10 == 0):
            logger.info(f"callback {self.sphere} {x.shape}")
            self.sphere.callback(x, self.iteration_count)
        self.iteration_count = self.iteration_count + 1

    def A(self, i, j, p2j):
        n_vis = len(self.u_arr)
        u, v, w = (
            self.u_arr[i % n_vis],
            self.v_arr[i % n_vis],
            self.w_arr[i % n_vis],
        )  # the row index (one u,v,w element per vis)
        l, m, n = (
            self.sphere.l[j],
            self.sphere.m[j],
            self.sphere.n[j],
        )  # The column index (one l,m,n element per pixel)

        z = get_harmonic(p2j, l, m, n - 1, u, v, w, self.sphere.pixel_areas[j])
        # z = np.exp(-p2j * (u * l + v * m + w * (n - 1))) * self.sphere.pixel_areas[j]
        if i < n_vis:
            return np.real(z)
        else:
            return np.imag(z)

    def Ah(self, i, j, p2j):
        return np.conj(self.A(j, i, p2j))

    def _compute_harmonics_block(self, p2j, u_blk, v_blk, w_blk):
        """Compute a block of the complex harmonics matrix.

        Args:
            p2j: complex frequency factor (1j * omega)
            u_blk, v_blk, w_blk: 1D arrays of UVW coordinates for this
                block of visibilities.

        Returns:
            H_block: complex array of shape (len(u_blk), self.N)
        """
        phase = p2j * (
            np.outer(u_blk, self.sphere.l)
            + np.outer(v_blk, self.sphere.m)
            + np.outer(w_blk, self.sphere.n_minus_1)
        )
        return np.exp(phase) * self.sphere.pixel_areas[np.newaxis, :]

    def _block_size(self):
        """Number of visibility rows per block, targeting ~200 MB per block."""
        return max(256, int(200 * 1024 * 1024 / (16 * max(1, self.N))))

    def _matvec(self, x):
        """
        Multiply by the sky x, producing the set of measurements y
        Returns returns A * x.

        ( v_real    = (T_real   x
          v_imag )     T_imag)

        Matrix-free: computes H @ x in blocks for memory efficiency.
        """
        n_u = self.u_arr.shape[0]
        y_re = np.zeros(n_u)
        y_im = np.zeros(n_u)
        block_size = self._block_size()

        for f in self.frequencies:
            p2j = jomega(f)
            for i_start in range(0, n_u, block_size):
                i_end = min(i_start + block_size, n_u)
                H_block = self._compute_harmonics_block(
                    p2j,
                    self.u_arr[i_start:i_end],
                    self.v_arr[i_start:i_end],
                    self.w_arr[i_start:i_end],
                )
                y_re[i_start:i_end] = np.real(H_block) @ x
                y_im[i_start:i_end] = np.imag(H_block) @ x

        return np.concatenate((y_re, y_im))

    def _rmatvec(self, v):
        r"""
        Returns x = A^H * v, where A^H is the conjugate transpose of A.

        x = ( T_real' T_imag') (v_real
                                v_imag)

        Matrix-free: blocked computation for memory efficiency.
        """
        assert v.shape == (self.M,)
        n_u = self.u_arr.shape[0]
        v_real = v[:n_u]
        v_imag = v[n_u:]

        ret = np.zeros(self.N)
        block_size = self._block_size()

        for f in self.frequencies:
            p2j = jomega(f)
            for i_start in range(0, n_u, block_size):
                i_end = min(i_start + block_size, n_u)
                H_block = self._compute_harmonics_block(
                    p2j,
                    self.u_arr[i_start:i_end],
                    self.v_arr[i_start:i_end],
                    self.w_arr[i_start:i_end],
                )
                ret += (
                    np.real(H_block).T @ v_real[i_start:i_end]
                    + np.imag(H_block).T @ v_imag[i_start:i_end]
                )

        return ret


class DirectImagingOperator(pylops.LinearOperator):
    r"""
    This is the approximate inverse of the DiSkOOperator, and corresponds to
    imaging by the discrete fourier transform
    """

    def __init__(self, u_arr, v_arr, w_arr, data, frequencies, sphere):
        super().__init__()
        self.N = sphere.npix  # Number of pixels
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr
        self.dtype = REAL_DATATYPE

        try:
            self.n_v, self.n_freq, self.npol = data.shape
        except Exception:
            raise RuntimeError("Data must be of the shape [n_v, n_freq, n_pol]")

        self.M = self.n_v * self.n_freq

        self.frequencies = frequencies
        self.sphere = sphere

        if self.sphere.l.shape[0] != self.N:
            raise RuntimeError(
                f"self.sphere.l.shape != self.N, {self.sphere.l.shape} != {self.N}"
            )

        self.shape = (self.N, self.M)
        self.explicit = False  # Can't be directly inverted
        logger.info("Creating DirectImagingOperator data={}".format(self.shape))

    def _compute_harmonics_block(self, p2j, u_blk, v_blk, w_blk):
        """Compute a block of the complex harmonics matrix.

        Args:
            p2j: complex frequency factor (1j * omega)
            u_blk, v_blk, w_blk: 1D arrays of UVW coordinates for this
                block of visibilities.

        Returns:
            H_block: complex array of shape (len(u_blk), self.N)
        """
        phase = p2j * (
            np.outer(u_blk, self.sphere.l)
            + np.outer(v_blk, self.sphere.m)
            + np.outer(w_blk, self.sphere.n_minus_1)
        )
        return np.exp(phase) * self.sphere.pixel_areas[np.newaxis, :]

    def _block_size(self):
        """Number of visibility rows per block, targeting ~200 MB per block."""
        return max(256, int(200 * 1024 * 1024 / (16 * max(1, self.N))))

    def _matvec(self, v):
        """
        Multiply by the measurements v, producing the sky.
        Returns A * v.

        sky = real(v_complex @ H) where H[i,j] is the harmonic at
        visibility i, pixel j.  Computed in blocks for memory efficiency.
        """
        n_vis = self.n_v // 2
        vis_complex = v[:n_vis] + 1.0j * v[n_vis:]

        sky = np.zeros(self.N, dtype=self.dtype)
        block_size = self._block_size()

        for f in self.frequencies:
            p2j = jomega(f)
            for i_start in range(0, n_vis, block_size):
                i_end = min(i_start + block_size, n_vis)
                H_block = self._compute_harmonics_block(
                    p2j,
                    self.u_arr[i_start:i_end],
                    self.v_arr[i_start:i_end],
                    self.w_arr[i_start:i_end],
                )  # shape (block_size, N)
                sky += np.real(vis_complex[i_start:i_end] @ H_block)

        return sky

    def _rmatvec(self, x):
        r"""
        Map sky vector x back to visibilities.

        v_complex = conj(H) @ x, then split into real/imag parts.
        """
        assert x.shape == (self.N,)
        n_vis = self.n_v // 2

        ret_complex = np.zeros(n_vis, dtype=COMPLEX_DATATYPE)
        block_size = self._block_size()

        for f in self.frequencies:
            p2j = jomega(f)
            for i_start in range(0, n_vis, block_size):
                i_end = min(i_start + block_size, n_vis)
                H_block = self._compute_harmonics_block(
                    p2j,
                    self.u_arr[i_start:i_end],
                    self.v_arr[i_start:i_end],
                    self.w_arr[i_start:i_end],
                )  # shape (block_size, N)
                # v_complex[i] = dot(conj(H[i,:]), x)  =>  v_block = H_block.conj() @ x
                ret_complex[i_start:i_end] = H_block.conj() @ x

        return np.concatenate((np.real(ret_complex), np.imag(ret_complex)))


class DiSkO(object):
    def __init__(self, u_arr, v_arr, w_arr, frequency):
        self.harmonics = {}  # Temporary store for harmonics
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr
        self.frequency = frequency
        self.n_v = len(self.u_arr)
        self.indices = None

    @classmethod
    def from_ant_pos(cls, ant_pos, frequency):
        # Get u, v, w from the antenna positions
        baselines, u_arr, v_arr, w_arr = get_all_uvw(ant_pos)
        ret = cls(u_arr, v_arr, w_arr, frequency)
        ret.info = {}
        return ret

    def vis_stats(self):
        vabs = np.abs(self.vis_arr)

        p05, p50, p95, p100 = np.percentile(vabs, [5, 50, 95, 100])
        logger.debug(
            "Vis Range: [{:5.4g} {:5.4g} {:5.4g} {:5.4g}]".format(p05, p50, p95, p100)
        )

        logger.debug("Vis Energy: {:5.4g}".format(np.sum(vabs)))

        return p05, p50, p95, p100

    @classmethod
    def from_cal_vis(cls, cal_vis):

        c = cal_vis.get_config()
        ant_p = np.asarray(c.get_antenna_positions())

        # We need to get the vis array to be correct for the full set of u,v,w points (baselines),
        # including the -u,-v, -w points.

        baselines, u_arr, v_arr, w_arr = get_all_uvw(ant_p)

        ret = cls(u_arr, v_arr, w_arr, c.get_operating_frequency())
        ret.vis_arr = []
        for bl in baselines:
            # Handles the conjugate bit
            v = cal_vis.get_visibility(bl[0], bl[1])
            ret.vis_arr.append(v)
            # logger.info("vis={}, bl={}".format(v, bl))
        ret.vis_arr = np.array(ret.vis_arr, dtype=COMPLEX_DATATYPE)
        ret.info = {}
        return ret

    def get_beam_width(self):
        max_u = np.max(np.abs(self.u_arr))
        max_v = np.max(np.abs(self.v_arr))
        max_w = np.max(np.abs(self.w_arr))

        beam_width = Resolution.from_baseline(
            bl=np.max([max_u, max_v, max_w]), frequency=self.frequency
        )
        logger.info(f"Resolution ({max_u}, {max_v}, {max_w}) : {beam_width}")
        return beam_width

    def get_harmonics(self, in_sphere):
        """Create the harmonics for this arrangement of sphere pixels.

        Returns a 2D complex array of shape (n_vis, n_pix).
        Uses vectorized broadcasting for performance.
        """
        cache_key = f"{in_sphere.npix}_{self.u_arr.shape[0]}"
        if cache_key in self.harmonics:
            harmonic_list = self.harmonics[cache_key]
            assert harmonic_list.shape[1] == in_sphere.npix
            return harmonic_list

        p2j = jomega(self.frequency)

        # Vectorized: compute the phase term for all (u,v,w) × (l,m,n-1) at once.
        # H[i,j] = exp(p2j * (u_i*l_j + v_i*m_j + w_i*(n_j-1))) * area_j
        phase = p2j * (
            np.outer(self.u_arr, in_sphere.l)
            + np.outer(self.v_arr, in_sphere.m)
            + np.outer(self.w_arr, in_sphere.n_minus_1)
        )
        gamma = np.exp(phase) * in_sphere.pixel_areas[np.newaxis, :]
        logger.debug("Vectorized Gamma Shape: {}".format(gamma.shape))

        self.harmonics[cache_key] = gamma
        return gamma

    def image_visibilities(self, vis_arr, sphere):
        """
        Create a DiSkO image from visibilities using the direct adjoint of the
        measurement operator (corresponds to the inverse DFT)

        Args:

            vis_arr (np.array): An array of complex visibilities
            sphere (FoV):    a sphere to place
        """

        assert len(vis_arr) == len(self.u_arr)
        logger.info("Imaging Visabilities resolution={}".format(sphere.min_res()))
        t0 = time.time()

        gamma = self.get_harmonics(sphere)  # shape (n_v, n_pix), complex
        # sum_i conj(vis_i) * H[i,:]  = conj(vis) @ H  =  H.T @ conj(vis) ...
        # Actually: pixels = sum_i vis_i * h_i  =  vis @ H  =  (H.T @ vis).T
        # Each column of H is a harmonic for one visibility.
        # pixels[j] = sum_i vis_i * H[i,j] = (vis @ H)[j]
        pixels = vis_arr @ gamma  # (n_vis,) @ (n_vis, n_pix) -> (n_pix,)

        logger.info("Elapsed {}s".format(time.time() - t0))

        sphere.set_visible_pixels(np.abs(pixels))

        return pixels.reshape(-1, 1)

    def solve_vis(self, vis_arr, sphere, scale=True):

        logger.info("Solving Visabilities res={}".format(sphere.min_res()))
        t0 = time.time()

        gamma = self.make_gamma(sphere)

        sky, residuals, rank, s = np.linalg.lstsq(
            gamma, to_column(vis_to_real(vis_arr)), rcond=None
        )

        logger.info("Elapsed {}s".format(time.time() - t0))

        sphere.set_visible_pixels(sky, scale)

        return sky.reshape(-1, 1)

    def vis_to_data(self, vis_arr=None):
        """
        Create some data with the correct shape (it has to have two additional dimensions)
        """
        data = np.zeros((self.n_v * 2, 1, 1), dtype=REAL_DATATYPE)
        if vis_arr is not None:
            data[:, 0, 0] = vis_to_real(vis_arr)
        else:
            data[:, 0, 0] = vis_to_real(self.vis_arr)
            assert data.shape[0] == self.n_v * 2

        return data

    def handle_residuals(self, operator, data, sky):
        residual = data - operator @ sky
        normalized_residuals = residual / np.std(residual)

        RESIDUAL_LIMIT = 10.0  # Arbitrary limit to show bad residuals.

        # Now reshape data back into complex data (from real appended to complex)
        c_data = np.reshape(data, (2, self.n_v))
        c_data = c_data[0] + 1.0j * c_data[1]

        c_res = np.reshape(normalized_residuals, (2, self.n_v))
        c_res = c_res[0] + 1.0j * c_res[1]

        bigguns = np.where(np.abs(c_res) > RESIDUAL_LIMIT)[0]

        half_v = self.n_v // 2
        bigguns = np.where(bigguns < half_v)[0]  # Remove the conjugate data

        logger.info(f"Residual problems {bigguns}")
        if self.indices is not None:
            logger.info("Residual List")
            logger.info(
                r"    MS_INDEX,    INDEX, RES (sd),     U,        V,        W,         VIS"
            )
            for b, i in zip(bigguns.tolist(), self.indices[bigguns]):
                logger.info(
                    f"    {i:8d}, {b:8d}, {np.abs(c_res[b]):5.2f},"
                    f"   {self.u_arr[b]:8.2f}, {self.v_arr[b]:8.2f},"
                    f" {self.w_arr[b]:8.2f}, {c_data[b]:4.2f}"
                )

    def solve_matrix_free(
        self,
        data,
        sphere,
        alpha=0.0,
        scale=True,
        fista=False,
        lsqr=True,
        lsmr=False,
        niter=25,
    ):
        """
        data = [vis_arr, n_freq, n_pol]
        """
        logger.info(f"Solving Visabilities sphere={sphere} data={data.shape}")
        assert data.shape[0] == self.n_v * 2

        frequencies = [self.frequency]
        logger.info("frequencies: {}".format(frequencies))

        A = DiSkOOperator(self.u_arr, self.v_arr, self.w_arr, data, frequencies, sphere)
        Apre = DirectImagingOperator(
            self.u_arr, self.v_arr, self.w_arr, data, frequencies, sphere
        )
        d = data.flatten()

        logger.info("Data.shape {}".format(data.shape))

        # u,s,vt = spalg.svds(A, k=min(A.shape)-2)
        # logger.info("t ={}, s={}".format(time.time() - t0, s))
        if fista:
            if alpha is not None:
                if alpha <= 0:
                    alpha = 10 ** (-np.log10(self.n_v) + 2)  # Empirical fit
                eps = 1.0 / alpha
                if eps > 0.1:
                    eps = 0.01
            else:
                eps = 1e-6  # Very weak regularization when alpha unspecified

            sky, niter, cost_history = pylops.optimization.sparsity.fista(
                Op=A,
                y=d,
                x0=np.abs(Apre @ d),
                eps=eps,
                tol=1e-10,
                niter=niter,
                alpha=alpha,
                show=True,
                threshkind="soft",
                callback=A,
            )

            logger.info(f"FISTA complete: {sky.shape} niter={niter}")

        if lsqr is True:
            if alpha < 0:
                alpha = np.mean(self.rms)
            (
                sky,
                lstop,
                itn,
                r1norm,
                r2norm,
                anorm,
                acond,
                arnorm,
                xnorm,
                var,
            ) = spalg.lsqr(A, data, damp=alpha, show=True)

            residual = d - A @ sky

            residual_norm, solution_norm = (
                np.linalg.norm(residual) ** 2,
                np.linalg.norm(sky) ** 2,
            )

            # mse = mean_squared_error(reg.coef_, np.zeros_like(reg.coef_))
            # mser = mean_squared_error(vis_aux, gamma @ sky)

            logger.info(
                "Alpha: {}: Loss: {}: rnorm: {}: snorm: {}: mse: {}: mser: {}".format(
                    alpha, itn, r2norm, solution_norm, residual_norm, r2norm
                )
            )

        if lsmr is True:
            if alpha < 0:
                alpha = np.mean(self.rms)
            x0 = Apre @ d

            sky, info = pylops.optimization.leastsquares.normal_equations_inversion(
                A, Regs=None, y=d, x0=x0, epsI=alpha, show=True
            )
            # logger.info(f"Matrix free solve elapsed={time.time()-t0} x={sky.shape}, stop={lstop}, itn={itn} r1norm={r1norm}")
            # logger.info(f"A M={} N={}".format(A.M, A.N))

            # sky, lstop, itn, normr, mormar, morma, conda, normx = spalg.lsmr(A, data, damp=alpha)
            # logger.info(f"Matrix free solve elapsed={time.time()-t0} x={sky.shape}, stop={lstop}, itn={itn} normr={normr}")
        # sky = np.abs(sky)

        self.handle_residuals(A, d, sky)

        sphere.set_visible_pixels(sky, scale)
        return sky.reshape(-1, 1)

    def make_gamma(self, sphere, makecomplex=False):
        """
        Build the telescope operator matrix. This v = Gamma s
        where s is the sky, and Gamma is the matrix
        """
        logger.info("Making Gamma Matrix npix={}".format(sphere.npix))

        gamma = self.get_harmonics(sphere)  # already 2D complex (n_v, n_s)

        n_v, n_s = gamma.shape
        logger.info("Gamma Shape: {}".format(gamma.shape))

        if makecomplex:
            return gamma

        # Build an augmented matrix for separating the real and imaginary
        # parts, so that the operator matrix can be real-valued.
        # Use np.concatenate (no copy) instead of np.block (creates copy).
        ret = np.concatenate((np.real(gamma), np.imag(gamma)), axis=0)

        logger.debug("Real Gamma Shape: {}".format(ret.shape))

        return ret

    def image_lasso(self, vis_arr, sphere, alpha, l1_ratio, scale=False, use_cv=False):
        gamma = self.make_gamma(sphere)

        vis_aux = vis_to_real(vis_arr)

        # Save proj operator for Further Analysis.
        if False:
            fname = "l1_big_files.npz"
            np.savez_compressed(
                fname, gamma_re=gamma, vis_re=np.real(vis_arr), vis_im=np.imag(vis_arr)
            )
            logger.info("Operator file {} saved".format(fname))

            logger.info("gamma = {}".format(gamma.shape))
            logger.info("vis_aux = {}".format(vis_aux.shape))

        n_s = sphere.pixels.shape[0]

        if not use_cv:
            reg = linear_model.ElasticNet(
                alpha=alpha / np.sqrt(n_s),
                l1_ratio=l1_ratio,
                tol=1e-6,
                max_iter=100000,
                positive=True,
            )
            reg.fit(gamma, vis_aux)

        else:
            reg = linear_model.ElasticNetCV(
                l1_ratio=l1_ratio, cv=5, max_iter=10000, positive=True
            )
            reg.fit(gamma, vis_aux)
            logger.info(
                "Cross Validation alpha: {} l1_ratio: {}".format(
                    reg.alpha_, reg.l1_ratio
                )
            )

        sky = reg.coef_
        logger.info("sky = {}".format(sky.shape))

        residual = vis_aux - gamma @ sky

        residual_norm = np.linalg.norm(residual) ** 2
        solution_norm = np.linalg.norm(sky) ** 2
        score = reg.score(gamma, vis_aux)

        logger.info(
            "Alpha: {}: Loss: {}: rnorm: {}: snorm: {}".format(
                alpha, score, residual_norm, solution_norm
            )
        )

        sphere.set_visible_pixels(sky, scale)
        return sky.reshape(-1, 1)

    def sequential_inference(self, sphere, real_vis):
        """

        posterior = to.sequential_inference(prior=prior, real_vis=vis_to_real(disko.vis_arr), sigma_vis=sigma_vis)

        # The image is now at posterior.mu
        sphere.set_visible_pixels(sky, scale)

        """
        gamma = self.make_gamma(sphere)
        n_s = sphere.pixels.shape[0]

        logger.info("Bayesian Inference of sky (n_s = {})".format(n_s))
        t0 = time.time()

        #
        # Create a prior (Using some indication of the expected range of the image)
        #

        p05, p50, p95, p100 = self.vis_stats()
        var = p95 * p95
        logger.info("Sky Prior variance={}".format(var))
        prior = MultivariateGaussian(np.zeros(n_s) + p50, sigma=var * np.identity(n_s))

        #
        # Create a likelihood covariance
        #
        diag = np.diagflat(self.rms**2)
        sigma_vis = np.block([[diag, 0.5 * diag], [0.5 * diag, diag]])

        precision = np.linalg.inv(sigma_vis)

        logger.info("y_m = {}".format(real_vis.shape))

        posterior = prior.bayes_update(precision, real_vis, gamma)

        logger.info("Elapsed {}s".format(time.time() - t0))
        return posterior

    def image_tikhonov(self, vis_arr, sphere, alpha, scale=True, usedask=False):
        n_s = sphere.pixels.shape[0]

        logger.info(
            f"image_tikhonov({vis_arr.shape}, {sphere}, {alpha}, scale={scale}, usedask={usedask})"
        )

        if alpha is None:
            raise RuntimeError(
                "The --alpha option must be specified when using --tikhonov"
            )

        lambduh = alpha / np.sqrt(n_s)
        if usedask is False:
            gamma = self.make_gamma(sphere)
            logger.info("augmented: {}".format(gamma.shape))

            vis_aux = vis_to_real(vis_arr)
            logger.info(
                "vis mean: {} shape: {}".format(np.mean(vis_aux), vis_aux.shape)
            )

            tol = min(alpha / 1e4, 1e-10)
            logger.info("Solving tol={} ...".format(tol))

            # reg = linear_model.ElasticNet(alpha=alpha/np.sqrt(n_s),
            # tol=1e-6,
            # l1_ratio = 0.01,
            # max_iter=100000,
            # positive=True)
            if False:
                (
                    sky,
                    lstop,
                    itn,
                    r1norm,
                    r2norm,
                    anorm,
                    acond,
                    arnorm,
                    xnorm,
                    var,
                ) = scipy.sparse.linalg.lsqr(gamma, vis_aux, damp=alpha, show=True)
                logger.info(
                    "Alpha: {}: Iterations: {}: rnorm: {}: xnorm: {}".format(
                        alpha, itn, r2norm, xnorm
                    )
                )
            else:
                reg = linear_model.Ridge(
                    alpha=alpha, tol=tol, solver="lsqr", max_iter=100000
                )

                reg.fit(gamma, vis_aux)
                logger.info("    Solve Complete, iter={}".format(reg.n_iter_))

                sky = reg.coef_  # np.from_array(reg.coef_)

                residual = vis_aux - gamma @ sky

                sky, residual_norm, solution_norm = (
                    sky,
                    np.linalg.norm(residual) ** 2,
                    np.linalg.norm(sky) ** 2,
                )

                score = reg.score(gamma, vis_aux)
                logger.info(
                    "Alpha: {}: Loss: {}: rnorm: {}: snorm: {}".format(
                        alpha, score, residual_norm, solution_norm
                    )
                )

        else:
            import dask
            import dask_glm
            from dask.distributed import Client, LocalCluster
            from dask_ml.linear_model import LinearRegression

            logger.info("Starting Dask Client")

            if True:
                cluster = LocalCluster(dashboard_address=":8231", processes=False)
                client = Client(cluster)
            else:
                client = Client("tcp://localhost:8786")

            logger.info("Client = {}".format(client))

            harmonic_list = []
            p2j = 2 * np.pi * 1.0j

            dl = sphere.l
            dm = sphere.m
            dn = sphere.n

            n_arr_minus_1 = dn - 1

            du = self.u_arr
            dv = self.v_arr
            dw = self.w_arr

            for u, v, w in zip(du, dv, dw):
                harmonic = da.from_array(
                    np.exp(p2j * (u * dl + v * dm + w * n_arr_minus_1))
                    / np.sqrt(sphere.npix),
                    chunks=(n_s,),
                )
                harmonic = client.persist(harmonic)
                harmonic_list.append(harmonic)

            gamma = da.stack(harmonic_list)
            logger.info("Gamma Shape: {}".format(gamma.shape))
            # gamma = gamma.reshape((n_v, n_s))
            gamma = gamma.conj()
            gamma = client.persist(gamma)

            logger.info("Gamma Shape: {}".format(gamma.shape))

            logger.info("Building Augmented Operator...")
            proj_operator_real = da.real(gamma)
            proj_operator_imag = da.imag(gamma)
            proj_operator = da.block([[proj_operator_real], [proj_operator_imag]])

            proj_operator = client.persist(proj_operator)

            logger.info("Proj Operator shape {}".format(proj_operator.shape))
            vis_aux = da.from_array(
                np.array(
                    np.concatenate((np.real(vis_arr), np.imag(vis_arr))),
                    dtype=np.float32,
                )
            )

            # logger.info("Solving...")

            en = dask_glm.regularizers.ElasticNet(weight=0.01)
            en = dask_glm.regularizers.L2()
            # dT = da.from_array(proj_operator, chunks=(-1, 'auto'))
            # dv = da.from_array(vis_aux)

            dask.config.set({"array.chunk-size": "1024MiB"})
            A = np.rechunk(proj_operator, chunks=("auto", n_s))
            A = client.persist(A)
            y = vis_aux  # np.rechunk(vis_aux, chunks=('auto', n_s))
            y = client.persist(y)
            # sky = dask_glm.algorithms.proximal_grad(A, y, regularizer=en, lambduh=alpha, max_iter=10000)

            logger.info("Rechunking completed.. A= {}.".format(A.shape))
            reg = LinearRegression(
                penalty=en,
                C=1.0 / lambduh,
                fit_intercept=False,
                solver="lbfgs",
                max_iter=1000,
                tol=1e-8,
            )
            sky = reg.fit(A, y)
            sky = reg.coef_
            score = reg.score(proj_operator, vis_aux)
            try:
                logger.info("Loss function: {}".format(score.compute()))
            except Exception:
                logger.info("Loss function: {}".format(score))

        logger.info("Solving Complete: sky = {}".format(sky.shape))

        sphere.set_visible_pixels(sky, scale=False)
        return sky.reshape(-1, 1)

    @classmethod
    def plot(self, plt, sphere, src_list):
        rot = (0, 90, 0)
        logger.info("sphere.pixels: {}".format(sphere.pixels.shape))
        if True:
            hp.orthview(
                sphere.pixels, rot=rot, xsize=1000, cbar=True, half_sky=True, hold=True
            )
            hp.graticule(verbose=False)
            plt.tight_layout()
        else:
            hp.mollview(sphere.pixels, rot=rot, xsize=1000, cbar=True)
            hp.graticule(verbose=True)

        if src_list is not None:
            for s in src_list:
                sphere.plot_x(s.el_r, s.az_r)

    def display(self, plt, src_list, nside):
        sphere = HealpixFoV(nside)
        self.solve_vis(self.vis_arr, sphere)
        sphere.plot(plt, src_list)

    def beam(self, plt, nside):
        sphere = HealpixFoV(nside)
        self.solve_vis(np.ones_like(self.vis_arr), nside)
        sphere.plot(plt, src_list=None)
