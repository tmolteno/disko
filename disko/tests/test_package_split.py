# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
# License: GPLv3

"""
Phase 3 of #10 (the package split) regression tests.

The implementation now lives in :mod:`disko.fov` (the sky model) and
:mod:`disko.image` (the imaging algorithms). Every pre-split module path
must keep working as a compatibility shim that re-exports the very same
objects, and the top-level :mod:`disko` API must not change.
"""

import importlib
import unittest
import warnings

import disko
from disko import fov, image

# Pre-split module path -> the canonical module it now re-exports.
SHIMS = {
    "disko.sphere": "disko.fov.fov",
    "disko.healpix_sphere": "disko.fov.healpix",
    "disko.sphere_mesh": "disko.fov.mesh",
    "disko.disko": "disko.image.disko",
    "disko.telescope_operator": "disko.image.telescope_operator",
    "disko.projection_lsqr": "disko.image.projection_lsqr",
}

# The public API of `disko` as declared in disko/__init__.py before the
# split. Adding to it would be fine; losing any of these is not.
PUBLIC_API = (
    "AdaptiveMeshFoV",
    "DiSkO",
    "DiSkOOperator",
    "HealpixFoV",
    "HealpixSubFoV",
    "MultivariateGaussian",
    "Resolution",
    "SquareFoV",
    "TelescopeOperator",
    "area",
    "dask_svd",
    "disko_from_ms",
    "get_all_uvw",
    "jomega",
    "mask_to_sky",
    "normal_svd",
    "plot_spectrum",
    "plot_uv",
    "plsqr",
    "sphere_args_parser",
    "sphere_from_args",
    "vis_to_real",
)

# The names each shim must keep answering with.
SHIM_NAMES = {
    "disko.sphere": (
        "FoV", "SquareFoV", "GeoLocation", "LonLat", "HpAngle", "ElAz",
        "PlotCoords", "elaz2lmn", "hp2elaz", "elaz2hp", "lonlat",
        "image_stats", "factors", "PI_OVER_2",
    ),
    "disko.healpix_sphere": (
        "HealpixFoV", "HealpixSubFoV", "create_fov", "cmap", "my_query_disk",
    ),
    "disko.sphere_mesh": (
        "AdaptiveMeshFoV", "area", "centroid", "get_mesh", "get_lmn",
    ),
    "disko.disko": (
        "DiSkO", "DiSkOOperator", "get_all_uvw", "jomega", "omega",
        "vis_to_real", "to_column", "get_harmonic",
    ),
    "disko.telescope_operator": (
        "TelescopeOperator", "dask_svd", "normal_svd", "tf_svd",
        "plot_spectrum", "plot_uv", "to_column", "SVD_TOL", "MAX_COND",
        "USE_DASK",
    ),
    "disko.projection_lsqr": ("plsqr",),
}


class TestCompatibilityShims(unittest.TestCase):
    def test_shim_modules_import(self):
        for old in SHIMS:
            with self.subTest(shim=old):
                self.assertIsNotNone(importlib.import_module(old))

    def test_shims_reexport_the_same_objects(self):
        """The shims are aliases, not copies: `is`, not `==`."""
        for old, new in SHIMS.items():
            old_mod = importlib.import_module(old)
            new_mod = importlib.import_module(new)
            with self.subTest(shim=old):
                for name in SHIM_NAMES[old]:
                    self.assertTrue(
                        hasattr(old_mod, name),
                        f"{old} lost {name}",
                    )
                    self.assertIs(
                        getattr(old_mod, name),
                        getattr(new_mod, name),
                        f"{old}.{name} is not {new}.{name}",
                    )

    def test_shims_add_nothing_and_hide_nothing(self):
        for old, new in SHIMS.items():
            old_mod = importlib.import_module(old)
            new_mod = importlib.import_module(new)
            old_public = {n for n in vars(old_mod) if not n.startswith("_")}
            new_public = {n for n in vars(new_mod) if not n.startswith("_")}
            with self.subTest(shim=old):
                self.assertEqual(
                    old_public, new_public,
                    f"{old} and {new} do not export the same names",
                )

    def test_shims_are_silent(self):
        """Decision recorded in CHANGES.md: shims emit no warning.

        A warning on module import would show up in every consumer's
        pytest run (pyproject.toml sets no filterwarnings) and would be
        raised by any consumer running with -W error, which is exactly
        the downstream breakage this phase exists to prevent.
        """
        for old in SHIMS:
            with self.subTest(shim=old):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    importlib.reload(importlib.import_module(old))
                self.assertEqual(
                    [], [str(w.message) for w in caught],
                    f"{old} warned on import",
                )

    def test_top_level_api_is_intact(self):
        for name in PUBLIC_API:
            with self.subTest(name=name):
                self.assertTrue(hasattr(disko, name), f"disko lost {name}")

    def test_top_level_api_points_at_the_canonical_modules(self):
        self.assertIs(disko.DiSkO, image.DiSkO)
        self.assertIs(disko.TelescopeOperator, image.TelescopeOperator)
        self.assertIs(disko.plsqr, image.plsqr)
        self.assertIs(disko.SquareFoV, fov.SquareFoV)
        self.assertIs(disko.HealpixSubFoV, fov.HealpixSubFoV)
        self.assertIs(disko.AdaptiveMeshFoV, fov.AdaptiveMeshFoV)

    def test_old_and_new_top_level_names_are_identical(self):
        from disko.disko import DiSkO as OldDiSkO
        from disko.healpix_sphere import HealpixSubFoV as OldSub
        from disko.sphere import SquareFoV as OldSquare
        from disko.telescope_operator import TelescopeOperator as OldTO

        self.assertIs(OldDiSkO, disko.DiSkO)
        self.assertIs(OldSub, disko.HealpixSubFoV)
        self.assertIs(OldSquare, disko.SquareFoV)
        self.assertIs(OldTO, disko.TelescopeOperator)


class TestPackageLayout(unittest.TestCase):
    def test_canonical_modules_are_where_the_plan_says(self):
        self.assertEqual("disko.fov.fov", fov.FoV.__module__)
        self.assertEqual("disko.fov.healpix", fov.HealpixFoV.__module__)
        self.assertEqual("disko.fov.mesh", fov.AdaptiveMeshFoV.__module__)
        self.assertEqual("disko.image.disko", disko.DiSkO.__module__)
        self.assertEqual(
            "disko.image.telescope_operator", disko.TelescopeOperator.__module__
        )

    def test_loggers_keep_their_pre_move_names(self):
        """The CLIs format log records with %(name)s.

        Renaming the loggers when the modules moved would change the
        text of every INFO line a CLI writes, so the moved modules pin
        their pre-split logger names.
        """
        from disko.fov import fov as fov_mod
        from disko.fov import healpix as healpix_mod
        from disko.image import disko as disko_mod

        self.assertEqual("disko.sphere", fov_mod.logger.name)
        self.assertEqual("disko.healpix_sphere", healpix_mod.logger.name)
        self.assertEqual("disko.disko", disko_mod.logger.name)

    def test_fov_and_image_packages_expose_their_subjects(self):
        self.assertIs(fov.FoV, importlib.import_module("disko.fov.fov").FoV)
        self.assertIs(image.DiSkO, importlib.import_module("disko.image.disko").DiSkO)
        self.assertTrue(callable(fov.from_hdf))


if __name__ == "__main__":
    unittest.main()
