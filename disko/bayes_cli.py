#!/usr/bin/env python

# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3
import argparse
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import scipy
from tart.imaging import calibration, elaz, visibility
from tart.operation import settings
from tart_tools import api_imaging

from .cli import disko_from_ms
from .disko import DiSkO, vis_to_real
from .ms_helper import get_array_location, good_visibility_count
from .multivariate_gaussian import MultivariateGaussian
from .parser_support import sphere_args_parser, sphere_from_args
from .telescope_operator import MAX_COND, TelescopeOperator

logger = logging.getLogger(__name__)
logger.addHandler(
    logging.NullHandler()
)  # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)


def create_prior(vis_arr, sphere, hdf_prior):
    """Based on the size of the visibilities, try and calculate
    what range the image should have.
    """
    if hdf_prior is not None:
        return MultivariateGaussian.from_hdf5(hdf_prior)

    vabs = np.abs(vis_arr)

    p05, p50, p95, p100 = np.percentile(vabs, [5, 50, 95, 100])

    var = p95 * p95
    logger.info("Estimated Sky Prior variance={}".format(var))
    prior = MultivariateGaussian(
        np.zeros(sphere.npix) + p50, sigma=var * np.identity(sphere.npix)
    )

    return prior


def load_prior_image(fname, sphere):
    """Load a full-sky HEALPix FITS map (RING ordering) and subsample it
    onto the pixels of this sphere. Used as the prior image for null-space
    completion.
    """
    prior_map = hp.read_map(fname)
    expected = hp.nside2npix(sphere.nside)
    if len(prior_map) != expected:
        raise RuntimeError(
            "Prior image {} has {} pixels, expected {} for nside={}".format(
                fname, len(prior_map), expected, sphere.nside
            )
        )
    return np.asarray(prior_map)[sphere.pixel_indices]


def do_inference(disko, sphere, prior, sigma_v=None, max_cond=MAX_COND):
    real_vis = vis_to_real(disko.vis_arr)

    to = TelescopeOperator(disko, sphere, max_cond=max_cond)

    n_v = real_vis.shape[0]

    # TODO create a proper covariance that ensures the real and imaginary components are linked.
    if sigma_v is None:
        diag = np.diag(disko.rms**2)
    else:
        diag = np.diag(np.ones(n_v // 2) * (sigma_v) ** 2)

    logger.info(f"do_inference(sigma_v={diag[0, 0]})")

    sigma_vis = np.block([[diag, 0.5 * diag], [0.5 * diag, diag]])  # .rechunk('auto')

    # now invert sigma_vis
    sigma_precision = MultivariateGaussian.sp_inv(sigma_vis)
    del sigma_vis

    # Diagonal-prior fast path. V is orthogonal, so
    #     V^H (sigma0 I) V == sigma0 I:
    # the prior covariance needs no rotation into the natural basis
    # (saving two O(n_s^3) matrix products and their n_s x n_s
    # temporaries); only the mean is rotated.
    #
    # DIAGONAL PRIORS ONLY: a dense prior -- for example the chained
    # posterior of sequential inference over multiple snapshots -- is
    # NOT invariant, and takes the general (exact) path below.
    diagonal_prior = prior.is_scaled_identity()
    logger.info("Diagonal prior fast path: {}".format(diagonal_prior))

    if diagonal_prior:
        sigma0 = prior.sigma()[0, 0]
        mu_natural = to.sky_to_natural(prior.mu)
        prior_r = MultivariateGaussian(
            mu_natural[0 : to.rank], sigma=sigma0 * np.identity(to.rank)
        )
        del mu_natural
    else:
        # Transform to the natural basis (exact for any covariance).
        n_prior = prior.linear_transform(to.Vh)
        prior_r = n_prior.block(0, to.rank)
        prior_n = n_prior.block(to.rank, to.n_s)

    A_r = to.A_r
    V = np.asarray(to.V)
    V_1 = np.asarray(to.V_1)

    del to
    posterior_r = prior_r.bayes_update(sigma_precision, real_vis, A_r)

    del A_r
    del sigma_precision
    del prior_r

    logger.info("Transforming posterior")

    if diagonal_prior:
        # Fused posterior assembly, exact for a scaled-identity prior
        # (the null block stays sigma0 * I). This avoids the n_s x n_s
        # block-diagonal intermediate and the cross terms that
        # outer() + linear_transform(V) would compute and discard:
        #     Sigma = V_1 Sigma_r V_1^T + sigma0 (I - V_1 V_1^T)
        #     mu    = prior.mu + V_1 (mu_r - V_1^T prior.mu)
        mu = prior.mu + V_1 @ (posterior_r.mu - V_1.T @ prior.mu)
        sigma = V_1 @ posterior_r.sigma() @ V_1.T + sigma0 * (
            np.identity(prior.D) - V_1 @ V_1.T
        )
        posterior = MultivariateGaussian(mu, sigma=sigma)
        del sigma
    else:
        posterior_n = prior_n

        del prior_n
        del n_prior

        posterior = MultivariateGaussian.outer(posterior_r, posterior_n)

        del posterior_n

        posterior = posterior.linear_transform(V)

    del posterior_r
    del V

    return posterior


def _visibility_covariance(disko, sigma_v):
    """The real visibility covariance, built exactly as in do_inference:
    per-baseline sigma_v^2 (or rms^2) on the diagonal with real/imag
    coupling blocks 0.5 * diag."""
    n_v = 2 * disko.n_v
    if sigma_v is None:
        diag = np.diag(disko.rms**2)
    else:
        diag = np.diag(np.ones(n_v // 2) * (sigma_v) ** 2)
    return np.block([[diag, 0.5 * diag], [0.5 * diag, diag]])


class SequentialInfoState:
    """Low-rank (information-form) sequential posterior.

    Starting from a scaled-identity prior ``Sigma0 = s0^2 I``, the
    posterior after each turn keeps the Woodbury-ready form
        Sigma_k = s0^2 I - s0^4 L_k M_k L_k^T,
        w_k     = s0^{-2} mu0 + L_k c_k   (precision-weighted mean)
    with ``L_k`` (n_s x R) the per-turn range-space bases ``V_1`` stacked
    side by side, ``M_k = (C_k^{-1} + s0^2 L_k^T L_k)^{-1}`` the R x R
    compressed inverse (``C_k = blockdiag(B_j)``, ``B_j = A_r^T Sigma_v^{-1}
    A_r``), and ``c_k`` the stacked data terms ``A_r^T Sigma_v^{-1} y``.
    This is exactly the conjugate-Gaussian posterior
        Sigma_k^{-1} = s0^{-2} I + sum_j V_{1,j} B_j V_{1,j}^T
    so the chain never forms an n_s x n_s matrix and never rotates a
    covariance between bases; sky-space quantities are recovered with
    Woodbury identities only when an accessor is called.  The mean uses
    the cancellation-free innovation form above; the covariance-family
    accessors (variance, pcf, covariance) are best conditioned when the
    likelihood and prior precisions are comparable, as with realistic
    per-baseline RMS noise.
    """

    def __init__(self, mu0, s0):
        self.mu0 = np.asarray(mu0, dtype=np.float64).flatten()
        self.s0 = float(s0)
        self.n_s = self.mu0.shape[0]
        self.L = np.zeros((self.n_s, 0))  # stacked range bases (n_s, R)
        self.Bs = []  # per-turn B_j = A_r^T Sigma_v^{-1} A_r (for precision)
        self.hs = []  # per-turn B_j^{-1} c_j (data innovation, for the mean)
        self.w = self.mu0 / (self.s0 * self.s0)  # precision-weighted mean
        self.M = np.zeros((0, 0))  # (C^{-1} + s0^2 L^T L)^{-1}
        self._s2 = self.s0 * self.s0
        self._cache = {}

    @property
    def R(self):
        """Cumulative rank of the absorbed turns."""
        return self.L.shape[1]

    @property
    def is_scaled_identity(self):
        return self.R == 0

    def _invalidate(self):
        self._cache = {}

    def add_turn(self, to, y, sigma_precision):
        """Absorb one turn: ``to`` is its rank-truncated TelescopeOperator,
        ``y`` the real visibilities, ``sigma_precision`` the visibility
        precision ``Sigma_v^{-1}``."""
        A_r = np.asarray(to.A_r)  # (n_v, r)
        V_1 = np.asarray(to.V_1)  # (n_s, r)
        PA = sigma_precision @ A_r
        B = A_r.T @ PA  # r x r SPD
        c = A_r.T @ (sigma_precision @ np.asarray(y).ravel())
        r = B.shape[0]

        # Bordered update of M = (C^{-1} + s0^2 L^T L)^{-1}:
        #   Q      = s0^2 L^T V_1            (new off-diagonal block)
        #   F22    = B^{-1} + s0^2 I_r       (new diagonal block)
        B_inv = scipy.linalg.inv(B)
        h = B_inv @ c  # data innovation in the range basis
        if self.R == 0:
            newM = scipy.linalg.inv(B_inv + self._s2 * np.identity(r))
        else:
            Q = self._s2 * (self.L.T @ V_1)
            X = self.M @ Q
            F22 = B_inv + self._s2 * np.identity(r)
            M22 = scipy.linalg.inv(F22 - Q.T @ X)
            M12 = -X @ M22
            M11 = self.M + X @ M22 @ X.T
            newM = np.block([[M11, M12], [M12.T, M22]])

        self.L = np.hstack((self.L, V_1))
        self.Bs.append(B)
        self.hs.append(h)
        self.w = self.w + V_1 @ c
        self.M = newM
        self._invalidate()
        return self

    def _Y(self):
        """M @ L^T, cached (shared by variance, pcf and covariance)."""
        if "Y" not in self._cache:
            self._cache["Y"] = self.M @ self.L.T
        return self._cache["Y"]

    def mean(self):
        """Posterior mean in sky space.

        Computed in the cancellation-free innovation form
            mu = mu0 + s0^2 L M (C^{-1} c - L^T mu0),
        which is exact and numerically stable even when the likelihood is
        vastly more informative than the prior.
        """
        if "mu" not in self._cache:
            g = np.concatenate(self.hs) - self.L.T @ self.mu0
            t = self.M @ g
            mu = self.mu0 + self._s2 * (self.L @ t)
            self._cache["mu"] = mu
        return self._cache["mu"]

    def variance(self):
        """Per-pixel posterior variance (diagonal of Sigma_k)."""
        if "var" not in self._cache:
            Y = self._Y()
            d = np.einsum("ia,ai->i", self.L, Y)  # diag(L M L^T)
            self._cache["var"] = self._s2 - self._s2**2 * d
        return self._cache["var"]

    def pcf_row(self, i):
        """Row i of the posterior covariance (the point covariance function),
        without materializing the full n_s x n_s matrix."""
        Y = self._Y()
        row = self.L[i, :] @ Y if self.R > 0 else np.zeros(self.n_s)
        out = np.zeros(self.n_s)
        out[i] = self._s2
        return out - self._s2**2 * row

    def covariance(self):
        """Full posterior covariance (materializes n_s x n_s)."""
        Y = self._Y()
        return self._s2 * np.identity(self.n_s) - self._s2**2 * (self.L @ Y)

    def precision(self):
        """Full posterior precision (materializes n_s x n_s)."""
        prec = np.identity(self.n_s) / self._s2
        for V_1, B in zip(self._blocks(), self.Bs):
            prec = prec + V_1 @ B @ V_1.T
        return prec

    def _blocks(self):
        """The stacked n_s x r_j range-basis blocks of ``self.L``."""
        start = 0
        for B in self.Bs:
            r = B.shape[0]
            yield self.L[:, start:start + r]
            start += r

    def sample(self):
        """Draw a sample from the posterior (materializes Sigma)."""
        chol = scipy.linalg.cholesky(self.covariance(), lower=True)
        z = np.random.normal(0.0, 1.0, self.n_s)
        return self.mean() + chol @ z

    def to_hdf5(self, fname, json_info="{}"):
        """Save the posterior (mu, sigma, sigma_inv) to HDF5."""
        mu = self.mean()
        sigma = self.covariance()
        sigma_inv = self.precision()
        with h5py.File(fname, "w") as h5f:
            conftype = h5py.special_dtype(vlen=bytes)
            conf_dset = h5f.create_dataset("info", (1,), dtype=conftype)
            conf_dset[0] = json_info
            h5f.create_dataset(
                "sigma", data=sigma, compression="gzip", compression_opts=9
            )
            h5f.create_dataset(
                "sigma_inv", data=sigma_inv, compression="gzip", compression_opts=9
            )
            h5f.create_dataset("mu", data=mu, compression="gzip", compression_opts=9)


class InfoPosterior:
    """Per-step view of a :class:`SequentialInfoState` posterior.

    Exposes the same output-facing interface as ``MultivariateGaussian``
    (mean, variance, covariance, precision, sample, to_hdf5, ...) but
    materializes sky-space matrices only when an accessor is called.  The
    view reflects the state at the moment it is handed out and is valid
    until the next ``add_turn``.
    """

    def __init__(self, state):
        self._state = state

    @property
    def mu(self):
        return self._state.mean()

    @property
    def D(self):
        return self._state.n_s

    def variance(self):
        return self._state.variance()

    def pcf_row(self, i):
        return self._state.pcf_row(i)

    def sigma(self):
        return self._state.covariance()

    def sigma_inv(self):
        return self._state.precision()

    def sample(self):
        return self._state.sample()

    def to_hdf5(self, fname, json_info="{}"):
        self._state.to_hdf5(fname, json_info=json_info)

    def is_scaled_identity(self):
        return self._state.is_scaled_identity


def sequential_inference(
    disko_list, sphere, prior=None, sigma_v=None, max_cond=MAX_COND, on_step=None
):
    """Chain N turns of sequential Bayesian inference.

    Each element of ``disko_list`` is a DiSkO supplying one turn's (complex)
    visibilities.  When ``prior`` is None the first update starts from the
    heuristic diagonal prior p95(|vis|)^2 I (``create_prior``); afterwards
    every posterior becomes the next turn's prior.  Each update uses the
    reduced, rank-truncated telescope operator (``A_r = U_1 Sigma_1``).

    For a scaled-identity starting prior the chain runs in the low-rank
    information form (:class:`SequentialInfoState`): the posterior is
    carried as ``s0^{-2} I + L L^T`` and never converted to (or rotated
    between) sky-space covariance matrices.  Any other prior falls back to
    the exact dense chaining via ``do_inference``.

    ``on_step(step, posterior)`` is called after each turn with a view of
    that step's posterior (valid until the next turn).  Returns the final
    posterior.
    """
    if prior is None:
        prior = create_prior(disko_list[0].vis_arr, sphere, None)

    if prior.is_scaled_identity():
        s0 = np.sqrt(prior.sigma()[0, 0])
        state = SequentialInfoState(np.asarray(prior.mu), s0)
        # The information form keeps only mu0 and s0; the prior's n_s x n_s
        # covariance (GBs for large maps) is dead weight here. Drop our
        # reference before the per-turn SVDs begin.
        del prior
        final = None
        for i, disko in enumerate(disko_list):
            to = TelescopeOperator(disko, sphere, max_cond=max_cond)
            y = vis_to_real(disko.vis_arr)
            sigma_vis = _visibility_covariance(disko, sigma_v)
            precision = MultivariateGaussian.sp_inv(sigma_vis)
            state.add_turn(to, y, precision)
            # Release this turn's operator (its SVD Vh is the largest
            # allocation of the turn) before the next turn's SVD starts.
            del to, y, precision
            view = InfoPosterior(state)
            if on_step is not None:
                on_step(i, view)
            final = view
        return final

    # Dense (arbitrary) prior: general exact path, pre-existing behaviour.
    posteriors = []
    for disko in disko_list:
        prior = do_inference(
            disko, sphere, prior, sigma_v=sigma_v, max_cond=max_cond
        )
        if on_step is not None:
            on_step(len(posteriors), prior)
        posteriors.append(prior)
    return posteriors[-1]


def _step_fname(fname, step):
    """Insert a step label into an output filename: prior.h5 -> prior_000.h5."""
    p = Path(fname)
    return str(p.with_name("{}_{:03d}{}".format(p.stem, step, p.suffix)))


def run_sequential(ARGS, sphere, n_steps):
    """Perform ``n_steps`` turns of sequential inference on the measurement set.

    Each turn draws ``nvis`` new visibilities from the MS (disjoint from all
    previous turns while the pool lasts), conjugates them for the CASA UVW
    convention, and runs one Bayesian update with the reduced telescope
    operator; each posterior becomes the next turn's prior.  The mean and
    covariance are written at every step: ``*_step<NNN>_mu`` / ``_var`` /
    ``_pcf`` images, and a full posterior HDF5 per step when ``--posterior``
    is given.

    If the requested ``nvis`` cannot be met for every turn (``n_steps *
    nvis`` exceeds the field's pool of unflagged, resolution-limited
    visibilities) the per-turn draw is reduced to ``pool // n_steps`` so
    that every step still consumes genuinely NEW visibilities; the
    reduction is logged as a warning.
    """
    json_info = get_array_location(ARGS.ms)
    lat = json_info["lat"]
    lon = json_info["lon"]
    height = json_info["height"]

    pool = good_visibility_count(
        ARGS.ms,
        sphere.min_res().degrees(),
        channel=ARGS.channel,
        field_id=ARGS.field,
    )
    nvis_turn = ARGS.nvis
    if pool < n_steps * ARGS.nvis:
        nvis_turn = max(1, pool // n_steps)
        logger.warning(
            "Field {} has only {} unflagged, resolution-limited visibilities; "
            "{} turns x --nvis {} exceeds the pool. Drawing {} visibilities "
            "per turn so every step consumes new data.".format(
                ARGS.field, pool, n_steps, ARGS.nvis, nvis_turn
            )
        )
        if nvis_turn * n_steps > pool:
            logger.warning(
                "The pool is too small for {} disjoint turns; later turns "
                "will reuse already-seen visibilities.".format(n_steps)
            )

    rng = np.random.default_rng()
    used = np.zeros(0, dtype=int)
    turns = []
    for _ in range(n_steps):
        disko = disko_from_ms(
            ARGS.ms,
            "DATA",
            nvis_turn,
            res=sphere.min_res(),
            channel=ARGS.channel,
            field_id=ARGS.field,
            rng=rng,
            exclude=used,
        )
        # CASAcore UVW is conjugated; conjugate visibilities for consistency.
        disko.vis_arr = disko.vis_arr.conjugate()
        turns.append(disko)
        if disko.indices is not None:
            used = np.unique(np.concatenate((used, np.asarray(disko.indices))))

    sphere.set_info(
        timestamp=turns[0].timestamp, lon=lon, lat=lat, height=height
    )

    def on_step(step, posterior):
        disko = turns[step]
        posterior_fname = None
        if ARGS.posterior is not None:
            posterior_fname = _step_fname(ARGS.posterior, step)
        handle_output(
            ARGS,
            disko.timestamp,
            posterior,
            sphere,
            disko,
            title_suffix="_step{:03d}".format(step),
            posterior_fname=posterior_fname,
        )

    final = sequential_inference(
        turns,
        sphere,
        # create_prior is passed inline (no other reference): the
        # information form releases its n_s x n_s covariance once the
        # chain has extracted mu0 and s0.
        prior=create_prior(turns[0].vis_arr, sphere, ARGS.prior),
        sigma_v=ARGS.sigma_v,
        max_cond=ARGS.max_cond,
        on_step=on_step,
    )
    return final


def handle_bayes(ARGS):

    sphere = sphere_from_args(ARGS)

    if ARGS.sequential:
        if not ARGS.ms:
            raise RuntimeError("--sequential requires a measurement set (--ms)")
        if ARGS.file or ARGS.hdf:
            raise RuntimeError(
                "--file/--hdf input cannot be re-sampled per turn; use --ms "
                "with --sequential"
            )
        if ARGS.sequential < 1:
            raise RuntimeError("--sequential N must be a positive integer")

    # Create a prior.
    if ARGS.file:
        logger.info("Getting Data from file: {}".format(ARGS.file))
        # Load data from a JSON file
        with open(ARGS.file, "r") as json_file:
            calib_info = json.load(json_file)

        info = calib_info["info"]
        ant_pos = calib_info["ant_pos"]
        config = settings.from_api_json(info["info"], ant_pos)

        flag_list = []  # [4, 5, 14, 22]

        _original_positions = deepcopy  # noqa: F841(config.get_antenna_positions())

        gains_json = calib_info["gains"]
        gains = np.asarray(gains_json["gain"])
        phase_offsets = np.asarray(gains_json["phase_offset"])
        config = settings.from_api_json(info["info"], ant_pos)

        _measurements = []  # noqa: F841
        for d in calib_info["data"]:
            vis_json, source_json = d
            cv, timestamp = api_imaging.vis_calibrated(
                vis_json, config, gains, phase_offsets, flag_list
            )
            src_list = elaz.from_json(source_json, 0.0)

        if ARGS.sigma_v is None:
            raise RuntimeError(
                "The --sigma-v option must be supplied when --file JSON input is used"
            )

        timestamp = cv.get_timestamp()
        disko = DiSkO.from_cal_vis(cv)
        prior = create_prior(disko.vis_arr, sphere, ARGS.prior)

        posterior = do_inference(
            disko, sphere, prior, sigma_v=ARGS.sigma_v, max_cond=ARGS.max_cond
        )
        handle_output(ARGS, timestamp, posterior, sphere, disko)

    elif ARGS.hdf:
        logger.info(f"Getting data from file {ARGS.hdf}")
        if ARGS.sigma_v is None:
            raise RuntimeError(
                "The --sigma-v option must be supplied when HDF5 input is used"
            )

        data = visibility.from_hdf5(ARGS.hdf)

        prior = create_prior(data["vis_list"][0].v, sphere, ARGS.prior)
        posterior = None

        for v in data["vis_list"]:
            if posterior is not None:
                prior = posterior
            cv = calibration.CalibratedVisibility(v)
            cv.set_config(v.config)
            cv.set_phase_offset(
                list(range(cv.get_config().get_num_antenna())),
                np.array(data["phase_offset"]),
            )
            cv.set_gain(
                list(range(cv.get_config().get_num_antenna())), np.array(data["gain"])
            )
            timestamp = cv.get_timestamp()
            disko = DiSkO.from_cal_vis(cv)

            # TODO Calibrate the vis with gains and phases?
            posterior = do_inference(
                disko, sphere, prior, sigma_v=ARGS.sigma_v, max_cond=ARGS.max_cond
            )
            handle_output(ARGS, timestamp, posterior, sphere, disko)
    else:
        logger.info("Getting Data from MS file: {}".format(ARGS.ms))

        if not os.path.exists(ARGS.ms):
            raise RuntimeError("Measurement set {} not found".format(ARGS.ms))

        min_res = sphere.min_res()
        logger.info(f"Min Res {min_res}")

        if ARGS.sequential:
            run_sequential(ARGS, sphere, ARGS.sequential)
            return

        disko = disko_from_ms(
            ARGS.ms,
            "DATA",
            ARGS.nvis,
            res=min_res,
            channel=ARGS.channel,
            field_id=ARGS.field,
        )
        # CASAcore UVW is conjugated; conjugate visibilities for consistency.
        disko.vis_arr = disko.vis_arr.conjugate()

        # Convert from reduced Julian Date to timestamp.
        timestamp = disko.timestamp

        json_info = get_array_location(ARGS.ms)
        lat = json_info["lat"]
        lon = json_info["lon"]
        height = json_info["height"]
        sphere.set_info(timestamp=timestamp, lon=lon, lat=lat, height=height)

        prior = create_prior(disko.vis_arr, sphere, ARGS.prior)

        posterior = do_inference(
            disko, sphere, prior, sigma_v=ARGS.sigma_v, max_cond=ARGS.max_cond
        )
        handle_output(ARGS, timestamp, posterior, sphere, disko)


def handle_output(
    ARGS, timestamp, posterior, sphere, disko=None, title_suffix="", posterior_fname=None
):

    if not ARGS.show_sources:
        src_list = None

    time_repr = "{:%Y_%m_%d_%H_%M_%S_%Z}".format(timestamp)

    # Now save the files.
    fname = posterior_fname if posterior_fname is not None else ARGS.posterior
    if fname is not None:
        parent = os.path.dirname(fname)
        if parent:
            os.makedirs(parent, exist_ok=True)
        posterior.to_hdf5(fname)

    def path(ending, image_title):
        os.makedirs(ARGS.dir, exist_ok=True)
        fname = "{}.{}".format(image_title, ending)
        return os.path.join(ARGS.dir, fname)

    def save_images(image_title, source_list):
        # Save as a FITS file

        if ARGS.FITS:
            sphere.to_fits(fname=path("fits", image_title), info=disko.info)

        if ARGS.SVG:
            fname = path("svg", image_title)
            sphere.to_svg(
                fname=fname,
                show_grid=True,
                src_list=source_list,
                title=image_title,
                show_cbar=True,
            )
            logger.info("Generating {}".format(fname))
        if ARGS.PNG:
            fname = path("png", image_title)
            sphere.plot(plt, source_list)
            plt.title(image_title)
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close()
            logger.info("Generating {}".format(fname))
        if ARGS.PDF:
            fname = path("pdf", image_title)
            sphere.plot(plt, source_list)
            plt.title(image_title)
            plt.savefig(fname, dpi=600)
            plt.close()
            logger.info("Generating {}".format(fname))

    if ARGS.PDF or ARGS.PNG or ARGS.SVG or ARGS.FITS:
        if ARGS.mu:
            logger.info("Computing pixels")
            tic = time.perf_counter()
            # mu_positive = np.array(da.clip(posterior.mu, 0, None))
            logger.info(f"    Took {time.perf_counter() - tic:0.4f} seconds")
            stat = sphere.set_visible_pixels(np.array(posterior.mu), scale=False)
            stat["sigma-v"] = ARGS.sigma_v
            logger.info(json.dumps(stat, sort_keys=True))
            save_images(
                "{}_{}{}_mu".format(ARGS.title, time_repr, title_suffix),
                source_list=src_list,
            )

        if ARGS.null_prior is not None:
            logger.info(
                "Completing image with null-space prior: {}".format(ARGS.null_prior)
            )
            if disko is None:
                raise RuntimeError(
                    "--null-prior requires visibility data (internal error)"
                )
            tic = time.perf_counter()
            prior_image = load_prior_image(ARGS.null_prior, sphere)
            to = TelescopeOperator(disko, sphere, max_cond=ARGS.max_cond)
            completed = to.complete_image(
                vis_to_real(disko.vis_arr), sphere, prior_image, scale=False
            )
            logger.info(f"    Took {time.perf_counter() - tic:0.4f} seconds")
            stat = sphere.set_visible_pixels(completed.flatten(), scale=False)
            stat["max-cond"] = ARGS.max_cond
            logger.info(json.dumps(stat, sort_keys=True))
            save_images(
                "{}_{}{}_complete".format(ARGS.title, time_repr, title_suffix),
                source_list=src_list
            )

        if ARGS.var:
            tic = time.perf_counter()
            logger.info("Computing variance...")
            variance = np.array(posterior.variance())
            logger.info(f"    Took {time.perf_counter() - tic:0.4f} seconds")
            sphere.set_visible_pixels(variance, scale=False)
            save_images(
                "{}_{}{}_var".format(ARGS.title, time_repr, title_suffix),
                source_list=None,
            )

        if ARGS.pcf:
            tic = time.perf_counter()
            logger.info("Computing point covariance...")

            brightest_pixel = np.argmax(posterior.mu)
            if hasattr(posterior, "pcf_row"):
                # Matrix-free: no n_s x n_s materialization needed.
                pix_cov = np.array(posterior.pcf_row(brightest_pixel))
            else:
                pix_cov = np.array(posterior.sigma()[brightest_pixel, :])
            logger.info(f"    Took {time.perf_counter() - tic:0.4f} seconds")

            sphere.set_visible_pixels(pix_cov, scale=False)
            save_images(
                "{}_{}{}_pcf".format(ARGS.title, time_repr, title_suffix),
                source_list=None,
            )

        for i in range(ARGS.nsamples):
            sphere.set_visible_pixels(posterior.sample(), scale=False)
            save_images(
                image_title="{}_{}{}_s{:0>5}".format(
                    ARGS.title, time_repr, title_suffix, i
                ),
                source_list=None,
            )


def build_parser():
    sphere_parsers = sphere_args_parser()

    parser = argparse.ArgumentParser(
        description="DiSkO: Bayesian inference of a posterior sky",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=sphere_parsers,
    )

    parser.add_argument(
        "--hdf", required=False, default=None, help="Exported Multi-visibility file"
    )
    parser.add_argument("--ms", required=False, default=None, help="visibility file")
    parser.add_argument(
        "--file",
        required=False,
        default=None,
        help="Snapshot observation saved JSON file (visiblities, positions and more).",
    )

    parser.add_argument(
        "--channel", type=int, default=0, help="Use this frequency channel."
    )
    parser.add_argument(
        "--field",
        type=int,
        default=0,
        help="Use this FIELD_ID from the measurement set.",
    )
    parser.add_argument(
        "--sequential",
        type=int,
        default=0,
        help="Perform N steps of sequential inference: each turn draws nvis NEW "
        "visibilities from the measurement set, updates the posterior with the "
        "reduced telescope operator, and writes the mean and covariance.",
    )

    parser.add_argument("--dir", required=False, default=".", help="Output directory.")
    parser.add_argument(
        "--nvis", type=int, default=1000, help="Number of visibilities to use."
    )
    parser.add_argument(
        "--arcmin",
        type=float,
        default=None,
        help="Highest allowed res of the sky in arc minutes.",
    )

    parser.add_argument(
        "--sigma-v",
        type=float,
        default=None,
        help="Diagonal components of the visibility covariance. If not supplied use measurement set values",
    )

    parser.add_argument(
        "--max-cond",
        type=float,
        default=MAX_COND,
        help="Rank truncation factor: singular values below max(s)/max_cond "
        "are treated as null space (invisible to the telescope).",
    )

    parser.add_argument(
        "--null-prior",
        type=str,
        default=None,
        help="Full-sky HEALPix FITS map (RING ordering, same nside as the "
        "imaging sphere). Its null-space component is grafted onto the "
        "data reconstruction, producing a *_complete image that is "
        "exactly consistent with the measured visibilities.",
    )

    parser.add_argument(
        "--PNG", action="store_true", help="Generate a PNG format image."
    )
    parser.add_argument(
        "--PDF", action="store_true", help="Generate a PDF format image."
    )
    parser.add_argument(
        "--SVG", action="store_true", help="Generate a SVG format image."
    )
    parser.add_argument(
        "--FITS", action="store_true", help="Generate a FITS format image."
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show known sources on images (only works on PNG & SVG).",
    )

    parser.add_argument(
        "--prior", type=str, default=None, help="Load the from an HDF5 file."
    )
    parser.add_argument(
        "--posterior",
        type=str,
        default=None,
        help="Store the posterior in HDF5 format file.",
    )

    parser.add_argument("--uv", action="store_true", help="Plot the UV coverage.")
    parser.add_argument("--mu", action="store_true", help="Save the mean image.")
    parser.add_argument(
        "--pcf", action="store_true", help="Save the point covariance function image."
    )
    parser.add_argument(
        "--var", action="store_true", help="Save the pixel variance image."
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=0,
        help="Number of samples to save from the posterior.",
    )

    parser.add_argument(
        "--title", required=False, default="disko", help="Prefix the output files."
    )

    return parser


def main():
    np.random.seed(42)
    parser = build_parser()
    source_json = None

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.INFO)

    root = logging.getLogger()

    fh = logging.FileHandler("disko.log")
    # fh.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()  # noqa: F841
    # ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    fh.setFormatter(formatter)

    # add ch to logger
    # root.addHandler(ch)
    root.addHandler(fh)

    # client = Client()

    handle_bayes(parser.parse_args())

    # client.close()
    # local_cluster.close()
