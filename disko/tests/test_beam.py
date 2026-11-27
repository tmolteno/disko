#
# Copyright Tim Molteno 2017-2026 tim@elec.ac.nz
# License GPLv3
#
"""
The primary beam: loading it, setting it on a field of view, and the
operator and the solvers with it.

--beam takes the same values as spotless's, and loads them the same way. The
mathematical tests use the tiny synthetic telescope of
test_telescope_operator.py (4 antennas, 24 real visibility rows, a 48-pixel
sphere), so the SVD and the Gaussian updates take milliseconds.
"""

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import pylops
import scipy.linalg
from astropy import constants as const
from scipy.spatial import Delaunay
from tart_beam import Beam, base_tart_beam, elaz_to_vec

from disko import AdaptiveMeshFoV, DiSkO, DiSkOOperator, HealpixFoV, HealpixSubFoV, Resolution
from disko import TelescopeOperator
from disko import bayes_cli
from disko import cli as disko_cli
from disko.bayes_cli import build_parser as bayes_parser
from disko.bayes_cli import create_prior, do_inference
from disko.beam import BUILTIN, add_beam_argument, beam_from_args, load_beam
from disko.cli import build_parser as disko_parser
from disko.fov.factory import from_hdf
from disko.image.disko import column_weights, column_weights_key
from disko.image.telescope_operator import condition_number
from disko.parser_support import sphere_args_parser

MS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "test.ms")

FREQUENCY = 1.5e9


def tiny_telescope():
    """The 4-antenna telescope of test_telescope_operator.py."""
    np.random.seed(42)
    return DiSkO.from_ant_pos(
        np.random.uniform(-2.0, 2.0, (4, 3)), frequency=FREQUENCY
    )


def as_close_as_float32(test, got, want):
    # DiSkOOperator keeps its harmonic blocks as float32, so its products
    # agree with the explicit float64 matrix to about 1e-6 of its largest
    # value, not element by element where cancellation leaves something
    # near zero.
    test.assertEqual(got.shape, want.shape)
    test.assertLessEqual(np.max(np.abs(got - want)), 1e-5 * np.max(np.abs(want)))


def gain_at(beam, el_deg, az_deg=0.0):
    """A pattern in one direction. elaz_to_vec() makes a bare vector of
    scalars, and Beam.evaluate() wants directions by rows."""
    return float(beam.evaluate(elaz_to_vec(el_deg, az_deg)[np.newaxis])[0])


class TestLoadBeam(unittest.TestCase):
    """--beam takes the same values as spotless's, and loads them the same way."""

    def test_the_built_in_names_give_the_tart_beam(self):
        for name in BUILTIN:
            with self.subTest(name=name):
                beam = load_beam(name)
                self.assertEqual(beam.name, "TART_base")
                self.assertEqual(beam.q, 1)
                # The pattern spotless loads: base_tart_beam(degree=8, q=1).
                np.testing.assert_array_equal(
                    beam.coeffs,
                    base_tart_beam(degree=8, q=1, name="TART_base").coeffs,
                )

    def test_the_tart_beam_tapers_to_the_horizon(self):
        # A least-squares fit to a pattern that is 1 above 10 degrees elevation
        # and falls to 0 at the horizon, so only roughly that.
        beam = load_beam("tart")

        def gain(el_deg):
            return gain_at(beam, el_deg)

        for el_deg in (90.0, 60.0, 20.0):
            self.assertAlmostEqual(gain(el_deg), 1.0, delta=0.05)
        self.assertAlmostEqual(gain(5.0), 0.5, delta=0.1)
        self.assertEqual(gain(-5.0), 0.0)

    def test_a_saved_beam_is_loaded_as_it_was_saved(self):
        saved = base_tart_beam(degree=4, q=1, name="saved")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beam.json")
            saved.to_json(path)
            beam = load_beam(path)

        self.assertEqual(beam.name, "saved")
        np.testing.assert_array_equal(beam.coeffs, saved.coeffs)
        self.assertEqual(beam.q, saved.q)

    def test_sample_records_are_fitted(self):
        # Elevation, azimuth and gain in degrees, as tart_beam reads them.
        records = [
            {"el": el, "az": az, "gain": 1.0 if el >= 30.0 else 0.0}
            for el in (90.0, 60.0, 30.0, 0.0)
            for az in (0.0, 90.0, 180.0, 270.0)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "samples.json")
            with open(path, "w") as f:
                json.dump(records, f)
            beam = load_beam(path)

        self.assertEqual(beam.name, "user_beam")
        high = gain_at(beam, 60.0)
        self.assertGreater(high, 0.5)
        self.assertGreater(high, gain_at(beam, 2.0))
        self.assertEqual(gain_at(beam, -5.0), 0.0)

    def test_anything_else_is_named_in_the_error(self):
        with self.assertRaisesRegex(ValueError, "no_such_beam.json"):
            load_beam("no_such_beam.json")

        for payload in ("[1, 2, 3]", "not json", '{"coeffs": "nope"}'):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "not_a_beam.json")
                with open(path, "w") as f:
                    f.write(payload)
                with self.assertRaisesRegex(ValueError, "not_a_beam.json"):
                    load_beam(path)


class TestSetBeam(unittest.TestCase):
    def setUp(self):
        self.sphere = HealpixFoV(8)

    def test_a_sphere_has_no_beam_until_one_is_set(self):
        self.assertIsNone(self.sphere.beam)
        # The harmonics are weighted by the pixel areas themselves.
        self.assertIs(column_weights(self.sphere), self.sphere.pixel_areas)

    def test_a_pattern_is_evaluated_at_the_pixel_directions(self):
        beam = load_beam("tart")
        beam_vector = self.sphere.set_beam(beam)
        expected = beam.evaluate(
            elaz_to_vec(np.degrees(self.sphere.el_r), np.degrees(self.sphere.az_r))
        )
        np.testing.assert_allclose(beam_vector, expected, atol=1e-12)
        self.assertIs(self.sphere.beam, beam_vector)

    def test_a_pattern_goes_through_tart_beam_s_own_elaz_to_vec(self):
        # tart_beam and disko take azimuth from different axes, (x, y, z) =
        # (m, l, n). Going through elaz_to_vec() is what makes disko,
        # spotless and tartball read a beam file alike. It only shows in a
        # pattern that is not symmetric about the zenith: this one is
        # 1 - 0.49 x, with x = cos(el) cos(az) in tart_beam's frame.
        beam = Beam([(0, 0), (1, 1)], [np.sqrt(4 * np.pi), -1.0], q=0)
        beam_vector = self.sphere.set_beam(beam)
        s = self.sphere

        np.testing.assert_allclose(
            beam_vector,
            beam.evaluate(np.stack([s.m, s.l, s.n], axis=-1)),
            atol=1e-10,
        )
        # On disko's (l, m, n) as they stand it would be a right angle out.
        wrong = beam.evaluate(np.stack([s.l, s.m, s.n], axis=-1))
        self.assertGreater(np.max(np.abs(beam_vector - wrong)), 0.5)

    def test_no_gain_behind_the_horizon_and_none_below_zero(self):
        beam_vector = self.sphere.set_beam(load_beam("tart"))
        behind = self.sphere.el_r < 0
        self.assertGreater(np.count_nonzero(behind), 0)
        self.assertTrue(np.all(beam_vector[behind] == 0.0))
        self.assertTrue(np.all(beam_vector >= 0.0))
        self.assertGreater(beam_vector.max(), 0.9)

    def test_an_array_of_gains(self):
        gains = np.full(self.sphere.npix, 0.5)
        gains[3] = -0.25
        beam_vector = self.sphere.set_beam(gains)
        # Clipped at zero, not floored: nothing divides by the gain.
        self.assertEqual(beam_vector[3], 0.0)
        self.assertTrue(np.all(beam_vector[[0, 1, 2, 4]] == 0.5))

        with self.assertRaises(ValueError):
            self.sphere.set_beam(gains[:-1])

    def test_copy_keeps_the_beam(self):
        beam_vector = self.sphere.set_beam(np.full(self.sphere.npix, 0.25))
        copied = self.sphere.copy()
        np.testing.assert_array_equal(copied.beam, beam_vector)
        self.assertIsNot(copied.beam, beam_vector)


class TestBeamOnAMesh(unittest.TestCase):
    def test_the_beam_is_set_again_when_the_mesh_changes(self):
        # AdaptiveMeshFoV.refine() replaces the pixels (fast_mesh() does the
        # same without gmsh). The beam must then be sampled again, as the
        # disko command line does, and an operator built meanwhile is refused
        # rather than weighting the wrong pixels.
        fov = AdaptiveMeshFoV(
            Resolution.from_arcmin(10), Resolution.from_deg(1), Resolution.from_deg(20),
            0.0, 0.0, recompute=False,
        )
        rng = np.random.default_rng(0)

        def rebuild(n):
            points = rng.normal(0, 0.05, (n, 2))
            fov.fast_mesh(points, Delaunay(points).simplices)

        rebuild(12)
        first = fov.set_beam(load_beam("tart"))
        self.assertEqual(first.shape[0], fov.npix)

        rebuild(30)
        self.assertNotEqual(fov.npix, first.shape[0])
        with self.assertRaises(RuntimeError):
            column_weights(fov)

        second = fov.set_beam(load_beam("tart"))
        self.assertEqual(second.shape[0], fov.npix)
        self.assertTrue(np.all(second >= 0.0))


class TestBeamFiles(unittest.TestCase):
    def test_the_beam_is_written_only_when_one_is_set_and_comes_back(self):
        sphere = HealpixFoV(4)
        with tempfile.TemporaryDirectory() as tmp:
            plain = os.path.join(tmp, "plain.h5")
            sphere.to_hdf(plain)
            with h5py.File(plain, "r") as h5f:
                self.assertNotIn("beam", h5f)
            self.assertIsNone(from_hdf(plain).beam)

            beam_vector = sphere.set_beam(load_beam("tart"))
            path = os.path.join(tmp, "beam.h5")
            sphere.to_hdf(path)
            with h5py.File(path, "r") as h5f:
                self.assertIn("beam", h5f)
            back = from_hdf(path)

        self.assertEqual(back.npix, sphere.npix)
        np.testing.assert_array_equal(back.beam, beam_vector)

    def test_a_sub_field_of_view_round_trips_too(self):
        sphere = HealpixSubFoV(nside=8, theta=0.0, phi=0.0, radius_rad=np.radians(60))
        beam_vector = sphere.set_beam(load_beam("tart"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub.h5")
            sphere.to_hdf(path)
            back = from_hdf(path)

        self.assertEqual(type(back).__name__, "HealpixSubFoV")
        np.testing.assert_array_equal(back.beam, beam_vector)


class TestOperatorWithBeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.disko = tiny_telescope()
        cls.plain = HealpixFoV(2)
        cls.beamed = cls.plain.copy()
        cls.beam = cls.beamed.set_beam(load_beam("tart"))
        cls.off = cls.beam == 0
        # One DiSkO for both spheres: its harmonics cache must tell them apart.
        cls.gamma = cls.disko.make_gamma(cls.plain)
        cls.gamma_b = cls.disko.make_gamma(cls.beamed)

    def test_harmonics_are_the_beam_free_ones_weighted_by_the_beam(self):
        # A_j -> b_j A_j is Gamma diag(b).
        np.testing.assert_allclose(
            self.gamma_b, self.gamma * self.beam[np.newaxis, :], rtol=1e-12, atol=1e-15
        )

    def test_the_harmonics_cache_knows_the_column_weights(self):
        other = self.plain.copy()
        other.set_beam(np.full(other.npix, 0.5))
        np.testing.assert_allclose(
            self.disko.make_gamma(other), 0.5 * self.gamma, rtol=1e-12, atol=1e-15
        )
        # As many pixels and other pixel areas, the same trap.
        resized = self.plain.copy()
        resized.pixel_areas = self.plain.pixel_areas * 0.25
        np.testing.assert_allclose(
            self.disko.make_gamma(resized), 0.25 * self.gamma, rtol=1e-12, atol=1e-15
        )
        # The first sphere is still served from the cache, and still right.
        self.assertIs(
            self.disko.get_harmonics(self.beamed), self.disko.get_harmonics(self.beamed)
        )
        np.testing.assert_array_equal(
            self.disko.make_gamma(self.beamed), self.gamma_b
        )
        keys = {column_weights_key(s) for s in (self.plain, self.beamed, other, resized)}
        self.assertEqual(len(keys), 4)

    def test_every_row_has_the_same_norm(self):
        n_h = self.gamma.shape[0] // 2
        h = self.gamma_b[:n_h] + 1j * self.gamma_b[n_h:]
        norms = np.sum(np.abs(h) ** 2, axis=1)
        expected = float(np.dot(self.beam * self.plain.pixel_areas, self.beam * self.plain.pixel_areas))
        np.testing.assert_allclose(norms, expected)

    def test_the_matrix_free_operator_is_the_explicit_matrix(self):
        tel = self.disko
        data = tel.vis_to_data(np.zeros(tel.n_v, dtype=complex))
        Op = DiSkOOperator(tel.u_arr, tel.v_arr, tel.w_arr, data, [FREQUENCY], self.beamed)
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, Op.N)
        v = rng.normal(0, 1, Op.M)

        for use_cache in (True, False):
            Op._use_cache = use_cache
            Op._block_cache = {} if use_cache else None
            as_close_as_float32(self, Op @ x, self.gamma_b @ x)
            adjoint = Op.rmatvec(v)
            self.assertEqual(adjoint.dtype, np.float64)
            as_close_as_float32(self, adjoint, self.gamma_b.T @ v)

        pylops.utils.dottest(Op, Op.M, Op.N, rtol=1e-6, complexflag=0, raiseerror=True, verb=False)

    def test_the_adjoint_is_the_beam_times_the_beam_free_adjoint(self):
        tel = self.disko
        data = tel.vis_to_data(np.zeros(tel.n_v, dtype=complex))
        Op = DiSkOOperator(tel.u_arr, tel.v_arr, tel.w_arr, data, [FREQUENCY], self.beamed)
        v = np.random.default_rng(1).normal(0, 1, Op.M)
        as_close_as_float32(self, Op.rmatvec(v), self.beam * (self.gamma.T @ v))

    def test_a_single_element_carries_the_beam(self):
        tel = self.disko
        data = tel.vis_to_data(np.zeros(tel.n_v, dtype=complex))
        Op = DiSkOOperator(tel.u_arr, tel.v_arr, tel.w_arr, data, [FREQUENCY], self.beamed)
        p2j = 2 * np.pi * 1.0j / (const.c.value / FREQUENCY)
        for i in (0, 5, tel.n_v, tel.n_v + 11):
            for j in (0, 17, 47):
                self.assertAlmostEqual(Op.A(i, j, p2j), self.gamma_b[i, j])

    def test_a_bare_sphere_object_still_builds_an_operator(self):
        # test_pylops_operator.py builds one on a sphere that is only npix and l.
        tel = self.disko
        data = tel.vis_to_data(np.zeros(tel.n_v, dtype=complex))
        bare = SimpleNamespace(npix=48, l=np.zeros(48))
        Op = DiSkOOperator(tel.u_arr, tel.v_arr, tel.w_arr, data, [FREQUENCY], bare)
        self.assertEqual(Op.shape, (tel.n_v * 2, 48))

    def test_a_beam_of_ones_leaves_the_operator_bit_for_bit(self):
        ones = self.plain.copy()
        ones.set_beam(np.ones(ones.npix))
        self.assertTrue(np.array_equal(self.disko.make_gamma(ones), self.gamma))


class TestBeamScaledOperator(unittest.TestCase):
    """Proposition 1 of doc/disko_operator_imaging.tex, on the test telescope."""

    @classmethod
    def setUpClass(cls):
        cls.disko = tiny_telescope()
        cls.plain = HealpixFoV(2)
        cls.beamed = cls.plain.copy()
        cls.beam = cls.beamed.set_beam(load_beam("tart"))
        cls.off = cls.beam == 0
        cls.gamma = np.asarray(cls.disko.make_gamma(cls.plain))
        cls.gamma_b = np.asarray(cls.disko.make_gamma(cls.beamed))

    def test_a_zero_gain_makes_an_exact_null_pixel(self):
        self.assertGreater(np.count_nonzero(self.off), 0)
        self.assertFalse(np.any(self.gamma_b[:, self.off]))
        # The range basis has no support there: no minimum-norm solution, and
        # no posterior mean or variance, is moved at such a pixel.
        V_1 = np.asarray(TelescopeOperator(self.disko, self.beamed).V_1)
        self.assertLess(np.max(np.abs(V_1[self.off, :])), 1e-10)

    def test_the_null_space_and_range_of_the_beam_scaled_operator(self):
        # N(Gamma) on the beam's support, lifted with zeros outside it: that
        # is N(Gamma) meeting R(B).
        on = ~self.off
        null_on = scipy.linalg.null_space(self.gamma[:, on])
        left_null = scipy.linalg.null_space(self.gamma.T)
        rank = np.linalg.matrix_rank(self.gamma)
        on_rank = np.linalg.matrix_rank(self.gamma[:, on])
        # rank(Gamma B) = rank(Gamma) on the beam's support, and its null
        # space is the off-beam pixels plus the null space of that restricted
        # operator: 28 + 8 here.
        n_on = int(np.count_nonzero(on))
        self.assertEqual(null_on.shape, (n_on, n_on - on_rank))
        self.assertEqual(
            TelescopeOperator(self.disko, self.beamed).n_n(),
            self.plain.npix - n_on + (n_on - on_rank),
        )
        self.assertEqual(left_null.shape, (self.gamma.shape[0], self.gamma.shape[0] - rank))
        rng = np.random.default_rng(1)
        scale = np.linalg.norm(self.gamma)

        for _ in range(4):
            # Such a direction, divided by the beam on its support, 1/b: a
            # null direction of Gamma B.
            y = np.zeros(self.plain.npix)
            y[on] = null_on @ rng.normal(0, 1, null_on.shape[1])
            rescaled = np.zeros_like(y)
            rescaled[on] = y[on] / self.beam[on]
            self.assertLess(np.max(np.abs(self.gamma @ y)), 1e-12 * scale)
            self.assertLess(np.max(np.abs(self.gamma_b @ rescaled)), 1e-12 * scale)
            # And the beam produces no visibility direction that Gamma cannot.
            s = rng.normal(0, 1, self.plain.npix)
            self.assertLess(
                np.max(np.abs(left_null.T @ (self.gamma_b @ s))), 1e-12 * scale
            )

    def test_a_strictly_positive_beam_keeps_the_rank(self):
        positive = self.plain.copy()
        positive.set_beam(np.linspace(0.2, 1.0, positive.npix))
        rank = TelescopeOperator(self.disko, self.plain).rank
        self.assertEqual(TelescopeOperator(self.disko, positive).rank, rank)
        # A beam with zeros can only lower it.
        self.assertLessEqual(TelescopeOperator(self.disko, self.beamed).rank, rank)

    def test_the_beam_lowers_no_singular_value(self):
        s = np.linalg.svd(self.gamma, compute_uv=False)
        s_b = np.linalg.svd(self.gamma_b, compute_uv=False)
        self.assertTrue(np.all(s_b <= s * self.beam.max() * (1 + 1e-12)))
        self.assertLess(s_b[0], s[0])

    def test_a_zero_singular_value_is_logged_as_an_infinite_condition_number(self):
        self.assertEqual(condition_number(np.array([2.0, 1.0, 0.0])), np.inf)
        self.assertEqual(condition_number(np.array([2.0, 1.0])), 2.0)
        # A sphere with zero columns goes through the SVD without a floating
        # point error, even where NumPy is told to raise on them.
        with np.errstate(all="raise"):
            TelescopeOperator(self.disko, self.beamed)

    def test_the_svd_cache_is_named_after_the_beam(self):
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                first = TelescopeOperator(self.disko, self.beamed, use_cache=True)
                TelescopeOperator(self.disko, self.plain, use_cache=True)
                names = sorted(n for n in os.listdir(tmp) if n.endswith(".npz"))
                again = TelescopeOperator(self.disko, self.beamed, use_cache=True)
        finally:
            os.chdir(cwd)

        self.assertIn("svd_48_24_10000.0.npz", names)
        beam_caches = [n for n in names if n.startswith("svd_48_24_10000.0_beam-")]
        self.assertEqual(len(beam_caches), 1, names)
        self.assertEqual(again.rank, first.rank)


class TestSolversWithBeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.disko = tiny_telescope()
        cls.sphere = HealpixFoV(2)
        cls.beam = cls.sphere.set_beam(load_beam("tart"))
        cls.off = cls.beam == 0
        cls.to = TelescopeOperator(cls.disko, cls.sphere)
        gamma = np.asarray(cls.to.gamma)
        rng = np.random.default_rng(2)
        cls.sky = rng.uniform(0.0, 1.0, cls.sphere.npix) * ~cls.off
        cls.vis = gamma @ cls.sky
        # The complex visibilities of the 12 baselines from the 24 real rows.
        cls.disko.vis_arr = cls.vis[: cls.disko.n_v] + 1j * cls.vis[cls.disko.n_v:]
        cls.disko.rms = np.full(cls.disko.n_v, 1e-3)

    def test_minimum_norm_images_vanish_where_the_beam_is_zero(self):
        vis = self.vis
        images = {
            "least squares": self.to.image_visibilities(vis, self.sphere, scale=False).ravel(),
            "natural basis": self.to.image_natural(vis, self.sphere, scale=False).ravel(),
            "Tikhonov": self.to.image_tikhonov(vis, self.sphere, 0.01, scale=False).ravel(),
        }
        for name, image in images.items():
            with self.subTest(solver=name):
                self.assertTrue(np.all(image[self.off] == 0.0))
                self.assertGreater(np.count_nonzero(image[~self.off]), 0)

    def test_the_l1_solve_leaves_them_at_zero_too(self):
        image = self.disko.image_lasso(
            self.disko.vis_arr, self.sphere, 0.001, 1.0, scale=False
        ).ravel()
        self.assertTrue(np.all(image[self.off] == 0.0))
        self.assertGreater(np.count_nonzero(image[~self.off]), 0)

    def test_the_posterior_keeps_its_prior_where_the_beam_is_zero(self):
        prior = create_prior(self.disko.vis_arr, self.sphere, None)
        posterior = do_inference(self.disko, self.sphere, prior, sigma_v=1e-6)
        np.testing.assert_array_equal(posterior.mu[self.off], prior.mu[self.off])
        np.testing.assert_array_equal(
            posterior.variance()[self.off], prior.variance()[self.off]
        )
        # While the beam is informed by the data.
        on = ~self.off
        self.assertTrue(
            np.any(posterior.variance()[on] < prior.variance()[on])
        )


class TestBeamOption(unittest.TestCase):
    """--beam on the command lines, with the values spotless's takes."""

    BUILDERS = {"disko": disko_parser, "disko_bayes": bayes_parser}

    @staticmethod
    def run_disko_main(argv):
        """cli.main() on a command line, with its logging setup neutralised."""
        logger = logging.getLogger("disko")
        level = logger.level
        try:
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "logging.basicConfig"
            ), contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(
                io.StringIO()
            ):
                try:
                    disko_cli.main()
                except SystemExit as stop:
                    return stop.code, err.getvalue()
                return None, err.getvalue()
        finally:
            logger.setLevel(level)

    def test_the_option_is_on_both_command_lines(self):
        for name, build in self.BUILDERS.items():
            with self.subTest(cli=name):
                self.assertIsNone(build().parse_args(["--ms", "x.ms"]).beam)
                self.assertEqual(build().parse_args(["--ms", "x.ms", "--beam"]).beam, "tart")
                self.assertEqual(
                    build().parse_args(["--ms", "x.ms", "--beam", "tart"]).beam, "tart"
                )
                self.assertEqual(
                    build().parse_args(["--ms", "x.ms", "--beam", "b.json"]).beam, "b.json"
                )

    def test_spotless_can_still_declare_its_own_beam(self):
        # spotless builds its parser on sphere_args_parser() and declares its
        # own --beam, so the shared parser must not already have one:
        # argparse rejects a duplicate option string.
        parser = argparse.ArgumentParser(parents=sphere_args_parser())
        parser.add_argument("--beam", nargs="?", const="tart", default=None)
        self.assertEqual(parser.parse_args(["--beam"]).beam, "tart")

    def test_the_shared_declaration(self):
        # What the command lines call: the same three forms.
        parser = argparse.ArgumentParser()
        add_beam_argument(parser)
        self.assertIsNone(parser.parse_args([]).beam)
        self.assertEqual(parser.parse_args(["--beam"]).beam, "tart")
        self.assertEqual(parser.parse_args(["--beam", "b.json"]).beam, "b.json")

    def test_the_beam_is_loaded_and_set_on_the_sphere(self):
        args = disko_parser().parse_args(["--ms", "x.ms", "--beam"])
        sphere = HealpixFoV(4)
        beam = beam_from_args(args, sphere)
        self.assertEqual(beam.name, "TART_base")
        behind = sphere.el_r < 0
        self.assertTrue(np.all(sphere.beam[behind] == 0.0))
        self.assertFalse(np.any(sphere.beam[sphere.el_r > np.radians(12)] < 0.9))

    def test_without_the_option_there_is_no_beam(self):
        args = disko_parser().parse_args(["--ms", "x.ms"])
        sphere = HealpixFoV(4)
        self.assertIsNone(beam_from_args(args, sphere))
        self.assertIsNone(sphere.beam)

    def test_a_beam_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beam.json")
            base_tart_beam(degree=4, q=1, name="from_file").to_json(path)
            args = disko_parser().parse_args(["--ms", "x.ms", "--beam", path])
            beam = beam_from_args(args, HealpixFoV(4))
        self.assertEqual(beam.name, "from_file")

    def test_a_beam_that_cannot_be_loaded_is_a_usage_error(self):
        args = disko_parser().parse_args(["--ms", "x.ms", "--beam", "no_such_beam.json"])
        sphere = HealpixFoV(4)
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(
            SystemExit
        ) as stop:
            beam_from_args(args, sphere, on_error=disko_parser().error)

        self.assertEqual(stop.exception.code, 2)
        # The usage line names the program it was run as, so only the message.
        self.assertIn(": error: --beam: no_such_beam.json", err.getvalue())
        self.assertIsNone(sphere.beam)
        # With no parser to report to, the same mistake raises.
        with self.assertRaisesRegex(ValueError, "no_such_beam.json"):
            beam_from_args(args, sphere)

    def test_it_is_reported_before_the_data_is_opened(self):
        # The measurement set does not exist either, but the beam is what is
        # complained about.
        argv = ["prog", "--ms", "no_such.ms", "--beam", "no_such_beam.json"]

        code, err = self.run_disko_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--beam: no_such_beam.json", err)
        self.assertNotIn("no_such.ms", err)

        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(
            SystemExit
        ) as stop:
            bayes_cli.handle_bayes(bayes_parser().parse_args(argv[1:]))
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("--beam: no_such_beam.json", err.getvalue())

    def test_the_command_line_images_through_the_beam(self):
        if not os.path.exists(MS_PATH):
            self.skipTest(f"{MS_PATH} is not here")
        with tempfile.TemporaryDirectory() as tmp:
            fov_path = os.path.join(tmp, "fov.h5")
            argv = [
                "disko", "--ms", MS_PATH, "--nside", "8", "--nvis", "100",
                "--beam", "tart", "--HDF", fov_path, "--dir", tmp, "--PNG",
            ]
            # What the image writers are asked for, without drawing: healpy's
            # plotting warns, and the suite's summary is kept clean.
            with mock.patch("disko.cli.save_images") as save:
                code, err = self.run_disko_main(argv)
            self.assertIsNone(code, err)
            back = from_hdf(fov_path)
            # save_images(fov, out_dir, image_title, ...): the beam image
            # follows the sky image, titled the same with a _beam suffix.
            titles = [call.args[2] for call in save.call_args_list]
            beam_fov = save.call_args_list[-1].args[0]
            sky_fov = save.call_args_list[0].args[0]

        # The field of view records the beam, and the image is exactly zero
        # where the beam is.
        self.assertIsNotNone(back.beam)
        np.testing.assert_array_equal(sky_fov.beam, back.beam)
        np.testing.assert_array_equal(back.pixels, sky_fov.pixels)
        self.assertTrue(np.all(back.pixels[back.beam == 0] == 0.0))
        self.assertGreater(np.count_nonzero(back.beam == 0), 0)
        self.assertTrue(np.all(back.beam[back.el_r > np.radians(12)] > 0.9))
        # And the pattern over the field of view is written as a beam image.
        self.assertEqual(len(titles), 2, titles)
        self.assertEqual(titles[1], titles[0] + "_beam")
        np.testing.assert_array_equal(beam_fov.pixels, back.beam)
        np.testing.assert_array_equal(beam_fov.beam, back.beam)
        for call in save.call_args_list:
            self.assertTrue(call.kwargs["png"])


if __name__ == "__main__":
    unittest.main()
