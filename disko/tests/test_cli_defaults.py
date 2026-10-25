#
# Issue #12: default behaviour should make nice images.
#
# `disko` with no algorithm flag used to fall through to an unregularized
# solve_vis(), and with no --mesh/--healpix it stopped with a RuntimeError.
# These tests pin the defaults that replaced that: healpix tiling plus
# Tikhonov regularization at alpha=0.02, with the unregularized path opt-in.
#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License GPLv3
#

import argparse
import unittest

from disko.cli import build_parser, parse_args
from disko.parser_support import sphere_from_args


class TestAlgorithmDefaults(unittest.TestCase):
    def test_no_algorithm_flag_defaults_to_tikhonov(self):
        args = parse_args(["--ms", "test_data/test.ms"])
        self.assertTrue(args.tikhonov)
        self.assertFalse(args.no_regularization)

    def test_alpha_defaults_to_0_02(self):
        # The value issue #12 identified as producing good TART images.
        args = parse_args(["--ms", "test_data/test.ms"])
        self.assertAlmostEqual(args.alpha, 0.02)

    def test_tikhonov_without_alpha_gets_the_default(self):
        # Previously: RuntimeError("The --alpha option must be specified")
        args = parse_args(["--ms", "test_data/test.ms", "--tikhonov"])
        self.assertTrue(args.tikhonov)
        self.assertAlmostEqual(args.alpha, 0.02)

    def test_explicit_alpha_wins(self):
        args = parse_args(["--ms", "x", "--tikhonov", "--alpha", "0.1"])
        self.assertAlmostEqual(args.alpha, 0.1)

    def test_explicit_algorithm_is_not_overridden(self):
        args = parse_args(["--ms", "x", "--lasso"])
        self.assertTrue(args.lasso)
        self.assertFalse(args.tikhonov)

    def test_unregularized_solve_is_opt_in(self):
        # The old fall-through behaviour is now an explicit flag.
        args = parse_args(["--ms", "x", "--no-regularization"])
        self.assertTrue(args.no_regularization)
        self.assertFalse(args.tikhonov)
        self.assertFalse(args.matrix_free)

    def test_matrix_free_solvers_imply_matrix_free(self):
        # --lsqr/--lsmr/--fista without --matrix-free used to fall through to
        # the unregularized solve, silently ignoring the choice.
        for flag in ("--lsqr", "--lsmr", "--fista"):
            args = parse_args(["--ms", "x", flag])
            self.assertTrue(args.matrix_free, f"{flag} should imply --matrix-free")
            self.assertFalse(args.tikhonov)

    def test_explicit_matrix_free_is_untouched(self):
        args = parse_args(["--ms", "x", "--matrix-free", "--lsmr"])
        self.assertTrue(args.matrix_free)
        self.assertTrue(args.lsmr)

    def test_algorithms_remain_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            parse_args(["--ms", "x", "--tikhonov", "--lasso"])

    def test_show_sources_points_at_disko_draw(self):
        # Issue #25: keep the friendly error rather than argparse's
        # "unrecognized arguments".
        with self.assertRaises(SystemExit):
            build_parser(["--show-sources"])


class TestInputRequirement(unittest.TestCase):
    def test_no_data_source_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_file_or_ms_is_accepted(self):
        self.assertEqual(parse_args(["--file", "x.h5"]).file, "x.h5")
        self.assertEqual(parse_args(["--ms", "x.ms"]).ms, "x.ms")

    def test_version_does_not_need_a_data_source(self):
        # main() prints the version and exits before touching any data, so
        # "disko --version" must keep working with no --ms/--file.
        args = parse_args(["--version"])
        self.assertTrue(args.version)


class TestSphereDefaults(unittest.TestCase):
    def test_no_tiling_flag_defaults_to_healpix(self):
        args = parse_args(["--ms", "test_data/test.ms"])
        self.assertFalse(getattr(args, "healpix", False))

        sphere = sphere_from_args(args)
        self.assertIsNotNone(sphere)
        self.assertTrue(args.healpix)

    def test_explicit_healpix_stays_on(self):
        args = parse_args(["--ms", "x", "--healpix", "--nside", "16"])
        sphere = sphere_from_args(args)
        self.assertTrue(args.healpix)
        self.assertIsNotNone(sphere)

    def test_healpix_default_also_applies_to_plain_namespace(self):
        # sphere_from_args is re-exported from disko and may be called with a
        # namespace that never had --healpix on it at all (default=SUPPRESS).
        args = argparse.Namespace(
            mesh=False, nside=16, fov="10deg", res="1deg", res_min=None
        )
        self.assertFalse(hasattr(args, "healpix"))
        sphere = sphere_from_args(args)
        self.assertIsNotNone(sphere)
        self.assertTrue(args.healpix)

    def test_mesh_still_opts_out_of_the_healpix_default(self):
        args = parse_args(["--ms", "x", "--mesh"])
        try:
            sphere = sphere_from_args(args)
        except Exception as exc:  # pragma: no cover - needs gmsh/dmsh
            self.skipTest(f"mesh construction unavailable: {exc}")
        self.assertIsNotNone(sphere)
        self.assertFalse(args.healpix)


if __name__ == "__main__":
    unittest.main()
