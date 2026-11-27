#
# Copyright Tim Molteno 2017-2026 tim@elec.ac.nz
# License GPLv3
#
"""
disko_svd's --psf, and --beam, which it replaced.

--beam was a flag that tried to draw the synthesised beam, and crashed. --psf
images it: the response to a unit point source at the zenith, the phase
centre, which is unit visibility on every baseline.

Nothing is drawn or written here: the sphere's plot() and the CLI's
handle_image() are replaced by recorders, which also keeps healpy's plotting
warnings out of the suite.
"""

import contextlib
import io
import logging
import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from disko import DiSkO, HealpixFoV, SquareFoV, TelescopeOperator, svd_cli
from disko.svd_cli import L1_RATIO, main

SNAPSHOT = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "test_data.json")


def run_svd(argv):
    """svd_cli.main() on a command line, without drawing or writing.

    Returns (exit code or None, stderr, the image titles asked for, what was
    asked to be plotted as (sphere, pixels at the time) in order). The root
    logger is put back as it was: main() sets it to DEBUG with a handler.
    """
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    titles, plotted = [], []

    def plot(sphere, plt, src_list):
        plotted.append((sphere, np.array(sphere.pixels)))

    try:
        with mock.patch.object(sys, "argv", ["disko_svd"] + argv), mock.patch.object(
            HealpixFoV, "plot", plot
        ), mock.patch.object(
            svd_cli,
            "handle_image",
            side_effect=lambda args, title, time_repr, src_list=None: titles.append(title),
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            try:
                main()
            except SystemExit as stop:
                return stop.code, err.getvalue(), titles, plotted
    finally:
        root.setLevel(level)
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
    return None, err.getvalue(), titles, plotted


@unittest.skipUnless(os.path.exists(SNAPSHOT), "test_data/test_data.json is not here")
class TestSynthesisedBeam(unittest.TestCase):
    def test_a_unit_source_at_the_phase_centre_gives_unit_visibility(self):
        # What --psf images, and why it can be given as unit visibilities: a
        # point source of flux F at (l, m, n) = (0, 0, 1) gives
        # F * exp(j w (n - 1)) on every baseline. SquareFoV has a pixel
        # centre exactly there; HEALPix pixels do not.
        np.random.seed(42)
        tel = DiSkO.from_ant_pos(np.random.uniform(-2.0, 2.0, (4, 3)), frequency=1.5e9)
        fov = SquareFoV(res_arcmin=60, theta=0.0, phi=0.0, width_rad=np.radians(10))
        # SquareFoV's centre pixel; it has no index_of().
        centre = (fov.width_pix // 2) * fov.width_pix + fov.width_pix // 2
        np.testing.assert_allclose(
            (fov.l[centre], fov.m[centre], fov.n[centre]), (0.0, 0.0, 1.0), atol=1e-15
        )

        flux = 2.5
        sky = np.zeros(fov.npix)
        sky[centre] = flux / fov.pixel_areas[centre]
        vis = tel.get_harmonics(fov) @ sky
        np.testing.assert_allclose(np.abs(vis), flux, rtol=1e-12)
        np.testing.assert_allclose(np.angle(vis), 0.0, atol=1e-12)

    def test_psf_is_the_response_to_a_unit_source_at_the_zenith(self):
        # What the three images are made of: unit visibility, stacked real
        # over imaginary for the SVD solvers and complex for the dirty image.
        asked = {}
        real = {
            "natural": TelescopeOperator.image_visibilities,
            "tikhonov": TelescopeOperator.image_tikhonov,
            "gridless": DiSkO.image_visibilities,
        }

        def record(name):
            def call(self, vis, sphere, *args, **kwargs):
                asked.setdefault(name, []).append(np.array(vis))
                return real[name](self, vis, sphere, *args, **kwargs)

            return call

        with mock.patch.object(
            TelescopeOperator, "image_visibilities", record("natural")
        ), mock.patch.object(
            TelescopeOperator, "image_tikhonov", record("tikhonov")
        ), mock.patch.object(DiSkO, "image_visibilities", record("gridless")):
            code, err, titles, plotted = run_svd(["--file", SNAPSHOT, "--nside", "2", "--psf"])

        self.assertIsNone(code, err)
        # The names the paper figure Makefiles build on.
        self.assertEqual(titles, ["psf_natural", "psf_tikhonov_0.10", "psf_gridless"])
        self.assertEqual(len(plotted), 3)
        n_vis = asked["gridless"][0].shape[0]
        # The dirty image takes the visibilities as they are, complex and one
        # per baseline: unit amplitude, no phase.
        unit = asked["gridless"][0]
        self.assertEqual(unit.dtype.kind, "c")
        self.assertEqual(unit.shape, (n_vis,))
        self.assertTrue(np.all(unit == 1))
        # The SVD solvers take them stacked real over imaginary, flat.
        stacked = np.concatenate([np.ones(n_vis), np.zeros(n_vis)])
        for name in ("natural", "tikhonov"):
            with self.subTest(image=name):
                self.assertEqual(asked[name][0].shape, (2 * n_vis,))
                np.testing.assert_array_equal(asked[name][0], stacked)

    def test_the_images_do_not_crash(self):
        code, err, titles, plotted = run_svd(["--file", SNAPSHOT, "--nside", "2", "--psf"])
        self.assertIsNone(code, err)
        for _, pixels in plotted:
            self.assertEqual(pixels.shape, (HealpixFoV(2).npix,))
            self.assertTrue(np.all(np.isfinite(pixels)))
            self.assertGreater(np.max(np.abs(pixels)), 0.0)


class TestBeamFlag(unittest.TestCase):
    def test_the_old_beam_flag_is_an_error_that_says_where_it_went(self):
        for flag in ("--beam", "--beam=tart"):
            with self.subTest(flag=flag):
                # The snapshot does not exist: the beam is complained about
                # before any file is opened.
                code, err, titles, plotted = run_svd(["--file", "no_such.json", flag])
                self.assertEqual(code, 2)
                self.assertIn("error: " + svd_cli.BEAM_MOVED, err)
                self.assertIn("--psf", err)
                self.assertEqual((titles, plotted), ([], []))

    def test_beam_with_a_value_is_an_error_too(self):
        # --beam tart, the form disko and disko_bayes take.
        code, err, _, _ = run_svd(["--file", "no_such.json", "--beam", "tart"])
        self.assertEqual(code, 2)
        self.assertIn(svd_cli.BEAM_MOVED, err)

    def test_the_option_is_gone_from_the_help(self):
        with mock.patch.object(sys, "argv", ["disko_svd", "--help"]), contextlib.redirect_stdout(
            io.StringIO()
        ) as out, self.assertRaises(SystemExit) as stop:
            main()
        self.assertEqual(stop.exception.code, 0)
        self.assertNotIn("--beam", out.getvalue())
        self.assertIn("--psf", out.getvalue())
        self.assertIn("synthesised beam", out.getvalue())

    def test_the_broken_method_behind_it_is_gone(self):
        # DiSkO.beam() handed an integer to solve_vis() as the sphere. It had
        # no other caller than the flag.
        self.assertFalse(hasattr(DiSkO, "beam"))


@unittest.skipUnless(os.path.exists(SNAPSHOT), "test_data/test_data.json is not here")
class TestImageSweep(unittest.TestCase):
    def test_the_sweep_runs_to_its_end(self):
        # --psf with an image format continues into the sweep over the data,
        # which called image_lasso() without the l1 ratio it gained later.
        # The sweep's first lasso fit is at alpha 0, which sklearn warns
        # about: its own doing, not this test's.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            code, err, titles, _ = run_svd(
                ["--file", SNAPSHOT, "--nside", "2", "--psf", "--PNG"]
            )
        self.assertIsNone(code, err)
        self.assertEqual(titles[:3], ["psf_natural", "psf_tikhonov_0.10", "psf_gridless"])
        self.assertEqual(titles[3], "natural")
        self.assertEqual(titles[-1], "gridless")
        self.assertEqual(len([t for t in titles if t.startswith("tikhonov_")]), 11)
        self.assertEqual(len([t for t in titles if t.startswith("lasso_")]), 11)
        self.assertEqual(len(titles), 3 + 1 + 11 + 11 + 1)
        self.assertEqual(L1_RATIO, 0.02)


if __name__ == "__main__":
    unittest.main()
