#
# Copyright Tim Molteno 2022-2026
#
# Fast unit tests for the pylops DiSkOOperator. These use a tiny
# synthetic telescope (a few antennas and a coarse healpix sphere).
#

import logging
import tracemalloc
import unittest
from types import SimpleNamespace

import astropy.constants as const
import numpy as np
import pylops

import disko
from disko import DiSkO, HealpixFoV, HealpixSubFoV

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


def dottest(Op, nr, nc, tol):

    pylops.utils.dottest(
        Op, nr, nc, rtol=1e-06, complexflag=0, raiseerror=True, verb=True
    )

    u = np.random.randn(nc)  # random sky
    v = np.random.randn(nr)  # random vis

    logger.info("u = {}".format(u))
    logger.info("v = {}".format(v))

    y = Op.matvec(u)  # Op * u
    x = Op.rmatvec(v)  # Op'* v

    logger.info("x = {}".format(x))
    logger.info("y = {}".format(y))

    yy = np.dot(y, v)  # (Op  * u)' * v
    xx = np.dot(u, x)  # u' * (Op' * v)

    err = np.abs((yy - xx) / ((yy + xx + 1e-15) / 2))
    if err < tol:
        logger.info(
            "Dot test passed, v^T(Opu)={} - u^T(Op^Tv)={}, err={}".format(yy, xx, err)
        )
        return True
    else:
        raise ValueError(
            "Dot test failed, v^T(Opu)={} - u^T(Op^Tv)={}, err={}".format(yy, xx, err)
        )


class TestPylopsOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

        # A tiny synthetic telescope: 4 antennas -> 12 baselines ->
        # 24 real visibility rows. The sphere has 48 pixels.
        cls.ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
        cls.disko = DiSkO.from_ant_pos(cls.ant_pos, frequency=1.5e9)
        cls.sphere = HealpixFoV(nside=2)

        sky = np.random.uniform(0.0, 1.0, cls.sphere.npix)
        cls.disko.vis_arr = cls.disko.get_harmonics(cls.sphere) @ sky

        cls.gamma = cls.disko.make_gamma(cls.sphere)

    def test_pylops_dot(self):
        r"""
        Test using the build-in pylops tester for new operators
        """
        data = self.disko.vis_to_data()
        frequencies = [self.disko.frequency]

        Op = disko.DiSkOOperator(
            self.disko.u_arr,
            self.disko.v_arr,
            self.disko.w_arr,
            data,
            frequencies,
            self.sphere,
        )
        # Test that we have the same effect as matrix vector multiply

        sky = np.random.normal(0, 1, self.sphere.npix)

        vis1 = self.gamma @ sky

        vis2 = Op @ sky  # Op.matvec(sky)
        logger.info(f"vis1: {vis1[0:10]}")
        logger.info(f"vis2: {vis2[0:10]}")

        self.assertEqual(vis1.shape, vis2.shape)
        self.assertTrue(np.allclose(vis1, vis2))

        dottest(Op, self.disko.n_v * 2, self.sphere.npix, tol=1e-04)

    def test_pylops_tiny(self):
        r"""
        Test such a small gamma that we can inspect every element and
        check that the matrix is what we expect it to be.
        """
        tiny_subsphere = HealpixSubFoV(
            res_arcmin=3600, theta=np.radians(0.0), phi=0.0, radius_rad=np.radians(80)
        )
        self.assertEqual(tiny_subsphere.npix, 4)

        frequencies = [1.5e9]
        wavelength = const.c.value / frequencies[0]

        n_vis = 3
        u = np.random.uniform(0, 1, n_vis)
        v = np.random.uniform(0, 1, n_vis)
        w = np.random.uniform(0, 1, n_vis)
        tiny_disko = DiSkO(u, v, w, frequencies[0])

        tiny_gamma = tiny_disko.make_gamma(tiny_subsphere)
        logger.info("Gamma={}".format(tiny_gamma))

        data = tiny_disko.vis_to_data(
            np.random.normal(0, 1, tiny_disko.n_v)
            + 1.0j * np.random.normal(0, 1, tiny_disko.n_v)
        )
        p2j = 2 * np.pi * 1.0j / wavelength

        Op = disko.DiSkOOperator(
            tiny_disko.u_arr,
            tiny_disko.v_arr,
            tiny_disko.w_arr,
            data,
            frequencies,
            tiny_subsphere,
        )

        logger.info("Op Matrix")
        for i in range(Op.M):
            col = [Op.A(i, j, p2j) for j in range(Op.N)]
            logger.info(col)

        logger.info("Op Matrix Ajoint")
        for i in range(Op.N):
            col = [Op.Ah(i, j, p2j) for j in range(Op.M)]
            logger.info(col)

        for i in range(Op.M):
            for j in range(Op.N):
                self.assertAlmostEqual(Op.A(i, j, p2j), tiny_gamma[i, j])

        for i in range(Op.N):
            for j in range(Op.M):
                self.assertAlmostEqual(Op.Ah(i, j, p2j), tiny_gamma[j, i])

        dottest(Op, Op.M, Op.N, 1e-6)

        sky = np.random.normal(0, 1, tiny_subsphere.npix)
        logger.info("sky={}".format(sky))
        vis1 = tiny_gamma @ sky
        vis2 = Op.matvec(sky)
        logger.info(vis1)
        logger.info(vis2)

        self.assertEqual(vis1.shape, vis2.shape)
        self.assertTrue(np.allclose(vis1, vis2))

    def _small_operator(self, n_ant=8):
        """An operator on a small synthetic telescope, beside the explicit
        matrix ([Re; Im]) it stands for."""
        ant_pos = np.random.uniform(-2.0, 2.0, (n_ant, 3))
        tel = DiSkO.from_ant_pos(ant_pos, frequency=1.5e9)
        data = tel.vis_to_data(np.zeros(tel.n_v, dtype=np.complex128))
        sphere = HealpixSubFoV(
            nside=8, theta=0.0, phi=0.0, radius_rad=np.radians(40)
        )
        Op = disko.DiSkOOperator(
            tel.u_arr, tel.v_arr, tel.w_arr, data, [tel.frequency], sphere
        )
        return Op, tel.make_gamma(sphere)

    def _assert_as_close_as_float32(self, got, want):
        # The harmonics are stored as float32, so the products agree with the
        # float64 matrix to about 1e-6 of their largest value, not element by
        # element where cancellation leaves something near zero.
        self.assertEqual(got.shape, want.shape)
        self.assertLessEqual(np.max(np.abs(got - want)), 1e-5 * np.max(np.abs(want)))

    def test_products_match_the_explicit_matrix(self):
        r"""
        A @ x and A.T @ v equal the explicit matrix products, in one block or
        many, with the block cache on or off. The adjoint stays float64.
        """
        Op, gamma = self._small_operator()
        x = np.random.normal(0, 1, Op.N)
        v = np.random.normal(0, 1, Op.M)
        self.assertGreater(Op.M, 10)  # so the smaller block sizes below split it

        for block_size in (Op._block_size(), 5, 1):
            Op._block_size = lambda rows=block_size: rows
            for use_cache in (True, False):
                Op._use_cache = use_cache
                Op._block_cache = {} if use_cache else None

                self._assert_as_close_as_float32(Op @ x, gamma @ x)
                adjoint = Op.rmatvec(v)
                self.assertEqual(adjoint.dtype, np.float64)
                self._assert_as_close_as_float32(adjoint, gamma.T @ v)

    def test_rmatvec_copies_no_blocks(self):
        r"""
        The adjoint multiplies the float32 blocks in float32. Paired with a
        float64 vector, NumPy converts each block to float64 first: a copy of
        the whole block per product per call, which is where most of the
        adjoint's time went.
        """
        rng = np.random.default_rng(3)
        n_rows = 170
        tel = DiSkO(
            rng.uniform(-2.0, 2.0, n_rows),
            rng.uniform(-2.0, 2.0, n_rows),
            rng.uniform(-2.0, 2.0, n_rows),
            1.5e9,
        )
        data = tel.vis_to_data(np.zeros(n_rows, dtype=np.complex128))
        sphere = HealpixFoV(nside=16)  # 3072 pixels
        Op = disko.DiSkOOperator(
            tel.u_arr, tel.v_arr, tel.w_arr, data, [tel.frequency], sphere
        )
        v = rng.normal(0, 1, Op.M)
        Op.rmatvec(v)  # build and cache the block

        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            Op.rmatvec(v)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not was_tracing:
                tracemalloc.stop()

        rows = min(Op._block_size(), n_rows)
        float64_block = Op.N * rows * 8
        self.assertLess(peak, float64_block / 8)

    def test_block_size_meets_its_memory_target(self):
        r"""
        One block's complex128 harmonics stay within the ~200 MB target at
        any pixel count. A floor of 256 rows used to override the target above
        about 51,200 pixels: at nside 256 (N = 358,912) that made a block
        1.47 GB per array.
        """
        tel = DiSkO.from_ant_pos(np.random.uniform(-2.0, 2.0, (4, 3)), frequency=1.5e9)
        data = tel.vis_to_data(np.zeros(tel.n_v, dtype=np.complex128))
        for npix, rows in ((48, 273066), (5056, 2592), (89856, 145), (358912, 36)):
            # Only npix and the length of l are read at construction, so the
            # big skies need no real sphere.
            sphere = SimpleNamespace(npix=npix, l=np.zeros(npix))
            Op = disko.DiSkOOperator(
                tel.u_arr, tel.v_arr, tel.w_arr, data, [tel.frequency], sphere
            )
            self.assertEqual(Op._block_size(), rows)
            self.assertLessEqual(Op._block_size() * npix * 16, 200 * 1024 * 1024)
