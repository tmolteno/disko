#
# Copyright Tim Molteno 2022-2026
#
# Fast unit tests for the disko_bayes --sequential N imaging method:
# N turns of sequential Bayesian inference on the reduced (rank-truncated)
# telescope operator, starting from a diagonal covariance prior, drawing
# nvis NEW visibilities from the measurement set per turn, and emitting
# the posterior mean and covariance at every step.
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
    build_parser,
    create_prior,
    do_inference,
    run_sequential,
    sequential_inference,
)
from disko.disko import vis_to_real
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


def _analytic_chain(turns, tos, sigma_v, prior):
    """Independent oracle: chain the Gaussian updates in the natural basis
    with plain numpy (no MultivariateGaussian helpers). Returns a list of
    (mu, Sigma) for each turn."""
    mu = np.asarray(prior.mu).copy()
    Sigma = np.asarray(prior.sigma()).copy()
    outs = []
    for turn, to in zip(turns, tos):
        y = vis_to_real(turn.vis_arr)
        Sig_v_inv = inv(_sigma_vis(y.shape[0], sigma_v))
        Vh = np.asarray(to.Vh)
        V = np.asarray(to.V)
        r = to.rank
        Lam = Vh @ Sigma @ V  # prior in the natural basis of this turn
        Lam_r = Lam[:r, :r]
        Lam_r_inv = inv(Lam_r)
        x_nat = Vh @ mu
        x_r = x_nat[:r]
        x_n = x_nat[r:]
        Ar = np.asarray(to.A_r)
        Lam_r_inv_new = Lam_r_inv + Ar.T @ Sig_v_inv @ Ar
        Lam_r_new = inv(Lam_r_inv_new)
        mu_r_new = Lam_r_new @ (Lam_r_inv @ x_r + Ar.T @ Sig_v_inv @ y)
        Lam_new = np.zeros_like(Lam)
        Lam_new[:r, :r] = Lam_r_new
        Lam_new[r:, r:] = Lam[r:, r:]  # null block untouched
        mu = V @ np.concatenate((mu_r_new, x_n))
        Sigma = V @ Lam_new @ Vh
        outs.append((mu, Sigma))
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
        if turns is None:
            turns = self.turns
        return sequential_inference(
            turns,
            self.sphere,
            prior=kwargs.pop("prior", self.prior0),
            sigma_v=kwargs.pop("sigma_v", self.sigma_v),
            **kwargs,
        )

    def test_chains_three_steps_shapes_and_fit(self):
        # N=3 turns -> 3 posteriors, each with a mean and a covariance,
        # each fitting its own turn's data.
        posteriors = self._run()

        self.assertEqual(len(posteriors), 3)
        for posterior, turn, to in zip(posteriors, self.turns, self.tos):
            self.assertEqual(posterior.mu.shape, (self.sphere.npix,))
            self.assertEqual(
                posterior.sigma().shape, (self.sphere.npix, self.sphere.npix)
            )
            # symmetric: Sigma == Sigma^T
            self.assertTrue(np.allclose(posterior.sigma(), posterior.sigma().T))
            # each posterior fits its own turn's data
            fitted = np.asarray(to.gamma) @ posterior.mu
            y = vis_to_real(turn.vis_arr)
            self.assertTrue(
                np.allclose(fitted, y, rtol=1e-3, atol=1e-3 * (1 + np.linalg.norm(y)))
            )

        # Step 0 is a plain single update with the diagonal prior.
        single = do_inference(
            self.turns[0], self.sphere, self.prior0, sigma_v=self.sigma_v,
            max_cond=MAX_COND,
        )
        self.assertTrue(np.allclose(posteriors[0].mu, single.mu, atol=1e-10))
        self.assertTrue(
            np.allclose(posteriors[0].sigma(), single.sigma(), atol=1e-10)
        )

    def test_matches_analytic_recursion(self):
        # Every step (including the diagonal-prior fast path at step 0 and
        # the dense general path afterwards) equals the analytic Gaussian
        # recursion on the reduced operator A_r.
        posteriors = self._run()
        expected = _analytic_chain(self.turns, self.tos, self.sigma_v, self.prior0)

        for posterior, (mu_e, sigma_e) in zip(posteriors, expected):
            self.assertTrue(
                np.allclose(posterior.mu, mu_e, rtol=1e-6, atol=1e-8),
                "posterior mean differs from analytic recursion",
            )
            self.assertTrue(
                np.allclose(posterior.sigma(), sigma_e, rtol=1e-6, atol=1e-12),
                "posterior covariance differs from analytic recursion",
            )

    def test_repeated_data_accumulates(self):
        # Repeating the same turn three times must accumulate precision on
        # the range block: Lambda_r^{-1} += k A_r^T Sigma_v^{-1} A_r, so the
        # range-block covariance trace strictly decreases.
        turns = [self.turns[0]] * 3
        to = self.tos[0]
        posteriors = self._run(turns=turns)
        expected = _analytic_chain(turns, [to] * 3, self.sigma_v, self.prior0)

        traces = []
        for posterior, (mu_e, sigma_e) in zip(posteriors, expected):
            self.assertTrue(np.allclose(posterior.mu, mu_e, rtol=1e-6, atol=1e-8))
            self.assertTrue(
                np.allclose(posterior.sigma(), sigma_e, rtol=1e-6, atol=1e-12)
            )
            Lam = np.asarray(to.Vh) @ posterior.sigma() @ np.asarray(to.V)
            traces.append(np.trace(Lam[: to.rank, : to.rank]))

        self.assertLess(traces[1], traces[0])
        self.assertLess(traces[2], traces[1])

    def test_null_space_untouched_per_step(self):
        # The data must never move the null directions of that step's
        # reduced operator: delta-mu and delta-Sigma have zero null-space
        # component in the natural basis of each turn.
        posteriors = self._run()
        previous_mu = np.asarray(self.prior0.mu)
        previous_sigma = np.asarray(self.prior0.sigma())
        for posterior, to in zip(posteriors, self.tos):
            V_2 = np.asarray(to.V_2)
            dmu = posterior.mu - previous_mu
            dsigma = posterior.sigma() - previous_sigma
            self.assertTrue(
                np.allclose(V_2.T @ dmu, 0.0, atol=1e-10),
                "data leaked into the null space of the mean",
            )
            self.assertTrue(
                np.allclose(V_2.T @ dsigma @ V_2, 0.0, atol=1e-10),
                "data leaked into the null space of the covariance",
            )
            previous_mu = np.asarray(posterior.mu)
            previous_sigma = np.asarray(posterior.sigma())

    def test_starts_with_diagonal_prior(self):
        # Step 0 starts from the heuristic diagonal covariance
        # (p95(|vis|)^2 I) and uses the fast path; chained posteriors are
        # dense and take the general (exact) path.
        self.assertTrue(self.prior0.is_scaled_identity())
        posteriors = self._run()
        self.assertFalse(posteriors[0].is_scaled_identity())
        self.assertFalse(posteriors[1].is_scaled_identity())

        # Default prior (prior=None) is the same diagonal heuristic.
        default = self._run(prior=None)
        self.assertTrue(np.allclose(default[0].mu, posteriors[0].mu, atol=1e-10))

    def test_disjoint_new_visibilities_per_turn(self):
        # The three turns use disjoint, new visibilities (3 turns x 2
        # conjugate pairs x 2 directions = the full 12-baseline pool).
        all_rows = []
        for i, turn in enumerate(self.turns):
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
        # operator, but the method still fits the data and keeps the
        # null space of the reduced operator untouched.
        max_cond = 2.0
        tos2 = [TelescopeOperator(t, self.sphere, max_cond=max_cond) for t in self.turns]
        for to, to2 in zip(self.tos, tos2):
            self.assertLessEqual(to2.rank, to.rank)

        posteriors = sequential_inference(
            self.turns, self.sphere, prior=self.prior0,
            sigma_v=self.sigma_v, max_cond=max_cond,
        )
        previous_mu = np.asarray(self.prior0.mu)
        previous_sigma = np.asarray(self.prior0.sigma())
        for posterior, turn, to in zip(posteriors, self.turns, tos2):
            self.assertEqual(posterior.mu.shape, (self.sphere.npix,))
            fitted = np.asarray(to.gamma) @ posterior.mu
            y = vis_to_real(turn.vis_arr)
            self.assertTrue(
                np.allclose(fitted, y, rtol=1e-3, atol=1e-3 * (1 + np.linalg.norm(y)))
            )
            V_2 = np.asarray(to.V_2)
            self.assertTrue(np.allclose(V_2.T @ (posterior.mu - previous_mu), 0.0, atol=1e-10))
            self.assertTrue(
                np.allclose(
                    V_2.T @ (posterior.sigma() - previous_sigma) @ V_2, 0.0, atol=1e-10
                )
            )
            previous_mu = np.asarray(posterior.mu)
            previous_sigma = np.asarray(posterior.sigma())


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
                calls.append(kwargs)
                return turns[len(calls) - 1]

            with mock.patch(
                "disko.bayes_cli.disko_from_ms", side_effect=fake_disko_from_ms
            ), mock.patch(
                "disko.bayes_cli.get_array_location",
                return_value={"lon": 0.0, "lat": -40.0, "height": 10.0},
            ):
                posteriors = run_sequential(ARGS, sphere, 3)

            # one read of nvis new visibilities per turn
            self.assertEqual(len(calls), 3)
            for call in calls:
                self.assertEqual(call["exclude"] is not None, True)
            self.assertEqual(len(calls[0]["exclude"]), 0)
            self.assertEqual(
                list(calls[1]["exclude"]), list(np.unique(np.asarray(turns[0].indices)))
            )
            self.assertEqual(
                list(calls[2]["exclude"]),
                list(
                    np.unique(
                        np.concatenate(
                            (np.asarray(turns[0].indices), np.asarray(turns[1].indices))
                        )
                    )
                ),
            )

            # one posterior (mean + covariance) per step, saved to disk
            self.assertEqual(len(posteriors), 3)
            h5s = sorted(f for f in os.listdir(tmpdir) if f.endswith(".h5"))
            self.assertEqual(h5s, ["post_000.h5", "post_001.h5", "post_002.h5"])
            with h5py.File(os.path.join(tmpdir, "post_000.h5"), "r") as h5f:
                self.assertEqual(h5f["mu"].shape, (sphere.npix,))
                self.assertEqual(h5f["sigma"].shape, (sphere.npix, sphere.npix))

            # in-memory means and covariances at every step
            for posterior in posteriors:
                self.assertEqual(posterior.mu.shape, (sphere.npix,))
                self.assertEqual(
                    posterior.sigma().shape, (sphere.npix, sphere.npix)
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
