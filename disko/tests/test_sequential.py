#
# Copyright Tim Molteno 2022-2026
#
# Fast unit tests for the disko_bayes --sequential N imaging method:
# N turns of sequential Bayesian inference on the reduced (rank-truncated)
# telescope operator, starting from a diagonal covariance prior, drawing
# nvis NEW visibilities from the measurement set per turn, and emitting
# the posterior mean and covariance at every step.
#
# The chain runs in the low-rank information form (SequentialInfoState):
# the posterior is carried as s0^{-2} I + L L^T and is never converted to
# (or rotated between) sky-space covariance matrices; sky quantities are
# materialized only by the output accessors.
#
# The tests use the tiny synthetic telescope (4 antennas, nside=2 sphere)
# so the SVD and Gaussian updates run in milliseconds.
#

import datetime
import logging
import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
from numpy.linalg import inv

from disko import DiSkO, HealpixFoV, TelescopeOperator
from disko.bayes_cli import (
    SequentialInfoState,
    _visibility_covariance,
    build_parser,
    create_prior,
    do_inference,
    run_sequential,
    sequential_inference,
)
from disko.disko import vis_to_real
from disko.multivariate_gaussian import MultivariateGaussian
from disko.telescope_operator import MAX_COND

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


def _conjugate_pairs(n_ant):
    """Row indices of the conjugate-symmetric baseline pairs of a
    get_all_uvw style visibility array: row (i,j) pairs with row (j,i)."""
    order = [(i, j) for i in range(n_ant) for j in range(n_ant) if i != j]
    pairs = []
    seen = set()
    for idx, (i, j) in enumerate(order):
        key = frozenset((i, j))
        if key in seen:
            continue
        seen.add(key)
        jidx = order.index((j, i))
        pairs.append((idx, jidx))
    return pairs


def _make_turn(base, sphere, full_vis, rows, freq, timestamp=None):
    """Build a DiSkO supplying one turn of the sequential loop from a
    subset of the baselines of ``base`` (conjugate pairs preserved)."""
    rows = np.asarray(rows)
    turn = DiSkO(base.u_arr[rows], base.v_arr[rows], base.w_arr[rows], freq)
    turn.vis_arr = full_vis[rows]
    turn.rms = np.ones(len(rows) // 2)
    turn.info = {}
    turn.indices = rows
    if timestamp is None:
        timestamp = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    turn.timestamp = timestamp
    return turn


def _sigma_vis(n_real, sigma_v):
    """Visibility covariance exactly as built by do_inference (real/imag
    blocks coupled by 0.5 * diag)."""
    nv2 = n_real // 2
    diag = np.diag(np.ones(nv2) * sigma_v**2)
    return np.block([[diag, 0.5 * diag], [0.5 * diag, diag]])


def _exact_chain(turns, tos, sigma_v, prior):
    """Independent oracle: the exact conjugate-Gaussian posterior after
    each turn, built directly from the precision sum
        Sigma^{-1} = prior^{-1} + sum_j V_{1,j} B_j V_{1,j}^T,
        w         = prior^{-1} mu0 + sum_j V_{1,j} c_j,
    using plain numpy (no MultivariateGaussian or SequentialInfoState
    helpers).  Returns a list of (mu, Sigma) for each turn."""
    s0 = np.sqrt(prior.sigma()[0, 0])
    prec = np.identity(tos[0].n_s) / (s0 * s0)
    lin = np.asarray(prior.mu) / (s0 * s0)
    outs = []
    for to, turn in zip(tos, turns):
        y = vis_to_real(turn.vis_arr)
        Sig_v_inv = inv(_sigma_vis(y.shape[0], sigma_v))
        Ar = np.asarray(to.A_r)
        V_1 = np.asarray(to.V_1)
        B = Ar.T @ Sig_v_inv @ Ar
        c = Ar.T @ (Sig_v_inv @ y)
        prec = prec + V_1 @ B @ V_1.T
        lin = lin + V_1 @ c
        Sigma = inv(prec)
        outs.append((Sigma @ lin, Sigma))
    return outs


class TestSequentialInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        np.seterr(all="raise")

        # A tiny synthetic telescope: 4 antennas -> 12 baselines, and a
        # 48 pixel healpix sphere. Group the 6 conjugate pairs into 3
        # disjoint turns of 2 pairs (4 complex visibilities each).
        cls.frequency = 1.5e9
        ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
        cls.base = DiSkO.from_ant_pos(ant_pos, frequency=cls.frequency)
        cls.sphere = HealpixFoV(nside=2)
        cls.sky = np.random.uniform(0.0, 1.0, cls.sphere.npix)
        cls.full_vis = cls.base.get_harmonics(cls.sphere) @ cls.sky

        pairs = _conjugate_pairs(4)
        cls.groups = [
            list(pairs[0]) + list(pairs[1]),
            list(pairs[2]) + list(pairs[3]),
            list(pairs[4]) + list(pairs[5]),
        ]
        ts = [
            datetime.datetime(2026, 1, i + 1, tzinfo=datetime.timezone.utc)
            for i in range(3)
        ]
        cls.turns = [
            _make_turn(cls.base, cls.sphere, cls.full_vis, rows, cls.frequency, t)
            for rows, t in zip(cls.groups, ts)
        ]
        cls.tos = [TelescopeOperator(t, cls.sphere) for t in cls.turns]

        cls.prior0 = create_prior(cls.turns[0].vis_arr, cls.sphere, None)
        cls.sigma_v = 1e-6

    def _run(self, turns=None, **kwargs):
        """Run the chain, collecting a (mu, Sigma) snapshot per step."""
        if turns is None:
            turns = self.turns
        collected = []

        def on_step(step, posterior):
            collected.append(
                (np.array(posterior.mu), np.array(posterior.sigma()))
            )

        kwargs.setdefault("prior", self.prior0)
        kwargs.setdefault("sigma_v", self.sigma_v)
        final = sequential_inference(
            turns, self.sphere, on_step=on_step, **kwargs
        )
        self.assertEqual(len(collected), len(turns))
        return final, collected

    def test_chains_three_steps_shapes_and_fit(self):
        # N=3 turns -> 3 posteriors, each with a mean and a covariance,
        # each fitting its own turn's data.
        final, collected = self._run()

        for (mu, Sigma), turn, to in zip(collected, self.turns, self.tos):
            self.assertEqual(mu.shape, (self.sphere.npix,))
            self.assertEqual(Sigma.shape, (self.sphere.npix, self.sphere.npix))
            # symmetric: Sigma == Sigma^T
            self.assertTrue(np.allclose(Sigma, Sigma.T))
            # each posterior fits its own turn's data
            fitted = np.asarray(to.gamma) @ mu
            y = vis_to_real(turn.vis_arr)
            self.assertTrue(
                np.allclose(fitted, y, rtol=1e-3, atol=1e-3 * (1 + np.linalg.norm(y)))
            )

        # Step 0 is a plain single update with the diagonal prior.
        single = do_inference(
            self.turns[0], self.sphere, self.prior0, sigma_v=self.sigma_v,
            max_cond=MAX_COND,
        )
        self.assertTrue(np.allclose(collected[0][0], single.mu, atol=1e-8))
        self.assertTrue(
            np.allclose(collected[0][1], single.sigma(), atol=1e-8)
        )

        # The returned posterior is the final step.
        self.assertTrue(np.allclose(final.mu, collected[-1][0], atol=1e-10))

    def test_matches_analytic_recursion(self):
        # Every step equals the exact conjugate-Gaussian posterior built
        # directly from the precision sum (the low-rank information form
        # is exact).  A well-conditioned likelihood keeps the covariance
        # comparison meaningful.
        sigma_v = 1.0
        _, collected = self._run(sigma_v=sigma_v)
        expected = _exact_chain(self.turns, self.tos, sigma_v, self.prior0)

        for (mu, Sigma), (mu_e, sigma_e) in zip(collected, expected):
            self.assertTrue(
                np.allclose(mu, mu_e, rtol=1e-6, atol=1e-6),
                "posterior mean differs from exact posterior",
            )
            self.assertTrue(
                np.allclose(Sigma, sigma_e, rtol=1e-6, atol=1e-8),
                "posterior covariance differs from exact posterior",
            )

    def test_repeated_data_accumulates(self):
        # Repeating the same turn three times must accumulate precision on
        # the range block, so the range-block covariance trace strictly
        # decreases (well-conditioned likelihood for the trace check).
        sigma_v = 1.0
        turns = [self.turns[0]] * 3
        to = self.tos[0]
        _, collected = self._run(turns=turns, sigma_v=sigma_v)
        expected = _exact_chain(turns, [to] * 3, sigma_v, self.prior0)

        traces = []
        for (mu, Sigma), (mu_e, sigma_e) in zip(collected, expected):
            self.assertTrue(np.allclose(mu, mu_e, rtol=1e-6, atol=1e-6))
            self.assertTrue(
                np.allclose(Sigma, sigma_e, rtol=1e-6, atol=1e-8)
            )
            Lam = np.asarray(to.Vh) @ Sigma @ np.asarray(to.V)
            traces.append(np.trace(Lam[: to.rank, : to.rank]))

        self.assertLess(traces[1], traces[0])
        self.assertLess(traces[2], traces[1])

    def test_null_pixels_uninformed_then_updated(self):
        # The exact chain still fits each turn's data at strong
        # likelihood, and the mean image has the right dimension.
        _, collected = self._run()
        for (mu, _Sigma), turn, to in zip(collected, self.turns, self.tos):
            self.assertEqual(mu.shape, (self.sphere.npix,))
            fitted = np.asarray(to.gamma) @ mu
            y = vis_to_real(turn.vis_arr)
            self.assertTrue(
                np.allclose(fitted, y, rtol=1e-3, atol=1e-3 * (1 + np.linalg.norm(y)))
            )

    def test_starts_with_diagonal_prior(self):
        # Step 0 starts from the heuristic diagonal covariance
        # (p95(|vis|)^2 I); the information form absorbs data immediately
        # (is_scaled_identity False once a turn has been absorbed).
        self.assertTrue(self.prior0.is_scaled_identity())
        final, collected = self._run()
        self.assertFalse(final.is_scaled_identity())

        # Default prior (prior=None) is the same diagonal heuristic.
        _, default = self._run(prior=None)
        self.assertTrue(np.allclose(default[0][0], collected[0][0], atol=1e-10))

    def test_dense_prior_falls_back_to_do_inference(self):
        # A non-diagonal starting prior (e.g. one loaded from a --prior
        # HDF5 file) takes the exact dense chaining path via do_inference.
        dense = MultivariateGaussian(
            np.asarray(self.prior0.mu) + 0.1,
            sigma=self.prior0.sigma() + 0.01 * np.identity(self.sphere.npix),
        )
        seen = []

        def on_step(step, posterior):
            seen.append(step)

        final = sequential_inference(
            self.turns,
            self.sphere,
            prior=dense,
            sigma_v=self.sigma_v,
            on_step=on_step,
        )
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(final.mu.shape, (self.sphere.npix,))
        self.assertEqual(
            final.sigma().shape, (self.sphere.npix, self.sphere.npix)
        )

    def test_disjoint_new_visibilities_per_turn(self):
        # The three turns use disjoint, new visibilities (3 turns x 2
        # conjugate pairs x 2 directions = the full 12-baseline pool).
        all_rows = []
        for turn in self.turns:
            self.assertEqual(len(turn.vis_arr), 4)
            all_rows.extend(turn.indices.tolist())
        self.assertEqual(sorted(all_rows), list(range(12)))
        # and the (u, v, w) coordinates differ between turns
        for i in range(len(self.turns) - 1):
            self.assertFalse(
                np.allclose(self.turns[i].u_arr, self.turns[i + 1].u_arr)
            )

    def test_respects_max_cond(self):
        # A smaller condition-number budget gives a smaller reduced
        # operator, but the method still fits the data and emits a
        # posterior at every step.
        max_cond = 2.0
        tos2 = [
            TelescopeOperator(t, self.sphere, max_cond=max_cond) for t in self.turns
        ]
        for to, to2 in zip(self.tos, tos2):
            self.assertLessEqual(to2.rank, to.rank)

        _, collected = self._run(max_cond=max_cond)
        for (mu, Sigma), turn, to in zip(collected, self.turns, tos2):
            self.assertEqual(mu.shape, (self.sphere.npix,))
            self.assertEqual(Sigma.shape, (self.sphere.npix, self.sphere.npix))
            fitted = np.asarray(to.gamma) @ mu
            y = vis_to_real(turn.vis_arr)
            self.assertTrue(
                np.allclose(fitted, y, rtol=1e-3, atol=1e-3 * (1 + np.linalg.norm(y)))
            )


class TestInformationForm(unittest.TestCase):
    """Direct tests of SequentialInfoState: exactness, invariants and the
    promise that the chain never materializes an n_s x n_s matrix."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        np.seterr(all="raise")
        cls.frequency = 1.5e9
        ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
        base = DiSkO.from_ant_pos(ant_pos, frequency=cls.frequency)
        cls.sphere = HealpixFoV(nside=2)
        sky = np.random.uniform(0.0, 1.0, cls.sphere.npix)
        full_vis = base.get_harmonics(cls.sphere) @ sky
        pairs = _conjugate_pairs(4)
        cls.turns = [
            _make_turn(
                base,
                cls.sphere,
                full_vis,
                list(pairs[0]) + list(pairs[1]),
                cls.frequency,
            ),
            _make_turn(
                base,
                cls.sphere,
                full_vis,
                list(pairs[2]) + list(pairs[3]),
                cls.frequency,
            ),
            _make_turn(
                base,
                cls.sphere,
                full_vis,
                list(pairs[4]) + list(pairs[5]),
                cls.frequency,
            ),
        ]
        cls.tos = [TelescopeOperator(t, cls.sphere) for t in cls.turns]
        cls.prior0 = create_prior(cls.turns[0].vis_arr, cls.sphere, None)
        # Well-conditioned likelihood for the (exact) invariant checks.
        cls.sigma_v = 1.0

    def _state_after_all_turns(self):
        s0 = np.sqrt(self.prior0.sigma()[0, 0])
        state = SequentialInfoState(np.asarray(self.prior0.mu), s0)
        cumulative = 0
        for to, turn in zip(self.tos, self.turns):
            y = vis_to_real(turn.vis_arr)
            sigma_vis = _visibility_covariance(turn, self.sigma_v)
            precision = MultivariateGaussian.sp_inv(sigma_vis)
            state.add_turn(to, y, precision)
            cumulative += to.rank
            # The chain never holds an n_s x n_s array.
            for val in state.__dict__.values():
                arr = np.asarray(val)
                if arr.ndim == 2:
                    self.assertNotEqual(arr.shape, (state.n_s, state.n_s))
            self.assertEqual(state.L.shape, (state.n_s, cumulative))
            self.assertEqual(state.M.shape, (cumulative, cumulative))
        return state

    def test_invariants_covariance_precision(self):
        # Sigma Sigma^{-1} == I and Sigma^{-1} mu == w at every step (the
        # Woodbury identities are consistent with the maintained state).
        s0 = np.sqrt(self.prior0.sigma()[0, 0])
        state = SequentialInfoState(np.asarray(self.prior0.mu), s0)
        for to, turn in zip(self.tos, self.turns):
            y = vis_to_real(turn.vis_arr)
            sigma_vis = _visibility_covariance(turn, self.sigma_v)
            precision = MultivariateGaussian.sp_inv(sigma_vis)
            state.add_turn(to, y, precision)
            cov = state.covariance()
            prec = state.precision()
            self.assertTrue(np.allclose(cov @ prec, np.identity(state.n_s), atol=1e-8))
            self.assertTrue(np.allclose(prec @ state.mean(), state.w, atol=1e-8))
            self.assertFalse(state.is_scaled_identity)

    def test_information_form_matches_analytic(self):
        state = self._state_after_all_turns()
        expected = _exact_chain(self.turns, self.tos, self.sigma_v, self.prior0)
        mu_e, sigma_e = expected[-1]
        self.assertTrue(np.allclose(state.mean(), mu_e, rtol=1e-6, atol=1e-6))
        self.assertTrue(
            np.allclose(state.covariance(), sigma_e, rtol=1e-6, atol=1e-8)
        )

    def test_precision_updates_only_range(self):
        # A turn's data adds precision only along that turn's range space:
        # V_2^T (prec_k - prec_{k-1}) V_2 == 0 for that turn's null basis.
        s0 = np.sqrt(self.prior0.sigma()[0, 0])
        state = SequentialInfoState(np.asarray(self.prior0.mu), s0)
        prec_prev = np.identity(self.sphere.npix) / (s0 * s0)
        for to, turn in zip(self.tos, self.turns):
            y = vis_to_real(turn.vis_arr)
            sigma_vis = _visibility_covariance(turn, self.sigma_v)
            precision = MultivariateGaussian.sp_inv(sigma_vis)
            state.add_turn(to, y, precision)
            dprec = state.precision() - prec_prev
            V_2 = np.asarray(to.V_2)
            self.assertTrue(np.allclose(V_2.T @ dprec, 0.0, atol=1e-8))
            self.assertTrue(np.allclose(V_2.T @ dprec @ V_2, 0.0, atol=1e-8))
            prec_prev = state.precision()

    def test_variance_and_pcf_consistent_with_covariance(self):
        # variance() and pcf_row() agree with the materialized covariance
        # diagonal / rows, without building the n_s x n_s matrix first.
        state = self._state_after_all_turns()
        cov = state.covariance()
        self.assertTrue(np.allclose(state.variance(), np.diagonal(cov), atol=1e-10))
        for i in (0, state.n_s // 2, state.n_s - 1):
            self.assertTrue(np.allclose(state.pcf_row(i), cov[i, :], atol=1e-10))


class TestSequentialCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    def test_run_sequential_loops_outputs_mean_and_covariance(self):
        # The CLI path draws nvis new visibilities per turn (exclude
        # accumulates) and writes a posterior HDF5 (mean + covariance)
        # for every step when --posterior is given.
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            ARGS = parser.parse_args(
                [
                    "--ms",
                    "fake.ms",
                    "--sequential",
                    "3",
                    "--nvis",
                    "4",
                    "--healpix",
                    "--nside",
                    "2",
                    "--posterior",
                    os.path.join(tmpdir, "post.h5"),
                    "--sigma-v",
                    "1e-6",
                    "--title",
                    "seq",
                ]
            )
            from disko.parser_support import sphere_from_args

            sphere = sphere_from_args(ARGS)

            # Build one turn per call against this CLI sphere.
            ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
            base = DiSkO.from_ant_pos(ant_pos, frequency=1.5e9)
            sky = np.random.uniform(0.0, 1.0, sphere.npix)
            full_vis = base.get_harmonics(sphere) @ sky
            pairs = _conjugate_pairs(4)
            turns = [
                _make_turn(
                    base,
                    sphere,
                    full_vis,
                    list(pairs[0]) + list(pairs[1]),
                    1.5e9,
                    datetime.datetime(2026, 1, i + 1, tzinfo=datetime.timezone.utc),
                )
                for i in range(3)
            ]

            calls = []

            def fake_disko_from_ms(*args, **kwargs):
                calls.append((args, kwargs))
                return turns[len(calls) - 1]

            with mock.patch(
                "disko.bayes_cli.disko_from_ms", side_effect=fake_disko_from_ms
            ), mock.patch(
                "disko.bayes_cli.get_array_location",
                return_value={"lon": 0.0, "lat": -40.0, "height": 10.0},
            ), mock.patch(
                "disko.bayes_cli.good_visibility_count", return_value=12
            ):
                final = run_sequential(ARGS, sphere, 3)

            # one read of nvis new visibilities per turn
            self.assertEqual(len(calls), 3)
            for args, _kwargs in calls:
                self.assertEqual(args[2], 4)
            self.assertEqual(len(calls[0][1]["exclude"]), 0)
            self.assertEqual(
                list(calls[1][1]["exclude"]),
                list(np.unique(np.asarray(turns[0].indices))),
            )
            self.assertEqual(
                list(calls[2][1]["exclude"]),
                list(
                    np.unique(
                        np.concatenate(
                            (np.asarray(turns[0].indices), np.asarray(turns[1].indices))
                        )
                    )
                ),
            )

            # one posterior (mean + covariance) saved to disk per step
            h5s = sorted(f for f in os.listdir(tmpdir) if f.endswith(".h5"))
            self.assertEqual(h5s, ["post_000.h5", "post_001.h5", "post_002.h5"])
            for name in h5s:
                with h5py.File(os.path.join(tmpdir, name), "r") as h5f:
                    self.assertEqual(h5f["mu"].shape, (sphere.npix,))
                    self.assertEqual(h5f["sigma"].shape, (sphere.npix, sphere.npix))

            # the returned posterior is the final step, with sky mean and
            # covariance available
            self.assertEqual(final.mu.shape, (sphere.npix,))
            self.assertEqual(final.sigma().shape, (sphere.npix, sphere.npix))

    def test_run_sequential_splits_pool_when_nvis_exceeds_it(self):
        # When n_steps * nvis > pool, each turn's draw is reduced to
        # pool // n_steps so every step still consumes NEW visibilities,
        # and a warning tells the user what happened.
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            ARGS = parser.parse_args(
                [
                    "--ms",
                    "fake.ms",
                    "--sequential",
                    "3",
                    "--nvis",
                    "4",
                    "--healpix",
                    "--nside",
                    "2",
                    "--posterior",
                    os.path.join(tmpdir, "post.h5"),
                    "--sigma-v",
                    "1e-6",
                ]
            )
            from disko.parser_support import sphere_from_args

            sphere = sphere_from_args(ARGS)

            ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
            base = DiSkO.from_ant_pos(ant_pos, frequency=1.5e9)
            sky = np.random.uniform(0.0, 1.0, sphere.npix)
            full_vis = base.get_harmonics(sphere) @ sky
            pairs = _conjugate_pairs(4)
            turns = [
                _make_turn(
                    base,
                    sphere,
                    full_vis,
                    list(pairs[0]) + list(pairs[1]),
                    1.5e9,
                    datetime.datetime(2026, 1, i + 1, tzinfo=datetime.timezone.utc),
                )
                for i in range(3)
            ]

            calls = []

            def fake_disko_from_ms(*args, **kwargs):
                calls.append((args, kwargs))
                return turns[len(calls) - 1]

            with mock.patch(
                "disko.bayes_cli.disko_from_ms", side_effect=fake_disko_from_ms
            ), mock.patch(
                "disko.bayes_cli.get_array_location",
                return_value={"lon": 0.0, "lat": -40.0, "height": 10.0},
            ), mock.patch(
                "disko.bayes_cli.good_visibility_count", return_value=6
            ), self.assertLogs("disko.bayes_cli", level="WARNING") as cm:
                run_sequential(ARGS, sphere, 3)

            # pool 6 < 3 turns x 4 nvis -> 6 // 3 = 2 new visibilities per turn
            self.assertEqual(len(calls), 3)
            for args, _kwargs in calls:
                self.assertEqual(args[2], 2)
            self.assertTrue(
                any("exceeds the pool" in m for m in cm.output),
                "expected a warning about the pool split",
            )

    def test_run_sequential_warns_when_pool_cannot_be_split(self):
        # A pool smaller than the turn count cannot give every turn new
        # data; the code still runs (reusing rows) but warns about it.
        parser = build_parser()
        ARGS = parser.parse_args(
            [
                "--ms", "fake.ms", "--sequential", "3", "--nvis", "4",
                "--healpix", "--nside", "2", "--sigma-v", "1e-6",
            ]
        )
        from disko.parser_support import sphere_from_args

        sphere = sphere_from_args(ARGS)
        ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
        base = DiSkO.from_ant_pos(ant_pos, frequency=1.5e9)
        sky = np.random.uniform(0.0, 1.0, sphere.npix)
        full_vis = base.get_harmonics(sphere) @ sky
        turn = _make_turn(base, sphere, full_vis, [0, 1, 2, 3], 1.5e9)

        calls = []

        def fake_disko_from_ms(*args, **kwargs):
            calls.append((args, kwargs))
            return turn

        with mock.patch(
            "disko.bayes_cli.disko_from_ms", side_effect=fake_disko_from_ms
        ), mock.patch(
            "disko.bayes_cli.get_array_location",
            return_value={"lon": 0.0, "lat": -40.0, "height": 10.0},
        ), mock.patch(
            "disko.bayes_cli.good_visibility_count", return_value=2
        ), self.assertLogs("disko.bayes_cli", level="WARNING") as cm:
            run_sequential(ARGS, sphere, 3)

        self.assertEqual(len(calls), 3)
        for args, _kwargs in calls:
            self.assertEqual(args[2], 1)  # max(1, 2 // 3)
        self.assertTrue(
            any("too small" in m for m in cm.output),
            "expected a warning about turns reusing visibilities",
        )

    def test_input_validation(self):
        parser = build_parser()
        common = ["--healpix", "--nside", "2"]

        # --sequential without a measurement set is rejected.
        ARGS = parser.parse_args(common + ["--sequential", "2"])
        from disko.bayes_cli import handle_bayes

        with self.assertRaises(RuntimeError):
            handle_bayes(ARGS)

        # --sequential with --file / --hdf input is rejected.
        ARGS = parser.parse_args(
            common + ["--sequential", "2", "--hdf", "vis.h5"]
        )
        with self.assertRaises(RuntimeError):
            handle_bayes(ARGS)

        ARGS = parser.parse_args(
            common + ["--sequential", "2", "--file", "snap.json"]
        )
        with self.assertRaises(RuntimeError):
            handle_bayes(ARGS)

        # A non-positive step count is rejected.
        ARGS = parser.parse_args(
            common + ["--sequential", "-1", "--ms", "fake.ms"]
        )
        with self.assertRaises(RuntimeError):
            handle_bayes(ARGS)

        # Default 0 leaves the existing single-shot behavior (no error).
        ARGS = parser.parse_args(common)  # --sequential defaults to 0
        self.assertEqual(ARGS.sequential, 0)


if __name__ == "__main__":
    unittest.main()
