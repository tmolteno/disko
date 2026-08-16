#
# Copyright Tim Molteno 2022-2026 tim@elec.ac.nz
#
# Fast unit tests for the DiSkO operator and solvers. These use a tiny
# synthetic telescope (a few antennas and a coarse healpix sphere) so
# that every test runs in milliseconds. The mathematical properties
# under test are size independent.
#

import logging
import unittest

import numpy as np

import disko
from disko import DiSkO, HealpixFoV, HealpixSubFoV
from disko.disko import get_all_uvw

logger = logging.getLogger(__name__)
# Add a null handler so logs can go somewhere
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


def dottest(Op, nr, nc, tol):
    u = np.random.randn(nc)  # random sky
    v = np.random.randn(nr)  # random vis

    y = Op.matvec(u)  # Op * u
    x = Op.rmatvec(v)  # Op'* v

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


class TestDiSkO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

        # A tiny synthetic telescope: 4 antennas -> 12 baselines ->
        # 24 real visibility rows. The sphere has 48 pixels, the
        # subsphere roughly a hemisphere.
        cls.ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
        cls.disko = DiSkO.from_ant_pos(cls.ant_pos, frequency=1.5e9)
        cls.sphere = HealpixFoV(nside=2)
        cls.subsphere = HealpixSubFoV(
            nside=2, theta=0.0, phi=0.0, radius_rad=np.radians(89)
        )

        # A random sky and consistent visibilities generated from it.
        cls.sky = np.random.uniform(0.0, 1.0, cls.sphere.npix)
        cls.disko.vis_arr = cls.disko.get_harmonics(cls.sphere) @ cls.sky

        cls.gamma = cls.disko.make_gamma(cls.sphere)
        cls.subgamma = cls.disko.make_gamma(cls.subsphere)

        # Consistent data on the subsphere, for the matrix-free solvers.
        cls.subsky = np.random.uniform(0.0, 1.0, cls.subsphere.npix)
        sub_harmonics = cls.disko.get_harmonics(cls.subsphere)
        cls.subvis = sub_harmonics @ cls.subsky
        cls.subdata = cls.disko.vis_to_data(cls.subvis)

    def get_point_sky(self):
        sky = np.zeros_like(self.sphere.pixels, dtype=np.float64)
        sky[-1] = 1.0
        sky = sky.reshape([-1, 1])
        return sky

    def test_harmonics_normalized(self):
        """
        Check the harmonics are normalized: each harmonic element is
        exp(j*phase) * pixel_area, so sum(|h|^2) == sum(pixel_area^2).
        """
        harmonics = self.disko.get_harmonics(self.sphere)
        a = self.sphere.pixel_areas
        expected = a @ a
        norms = np.sum(np.abs(harmonics) ** 2, axis=1)
        self.assertTrue(np.allclose(norms, expected))

    @unittest.skip("When there is only a real sky, there are no harmonics.")
    def test_vis_from_harmonic_sky(self):
        pass

    def test_vis(self):
        """
        Check that the effect of multiplication by gamma is the same as
        inner product with harmonics: Gamma = [Re H; Im H], so
        Gamma @ sky == [Re(H @ sky); Im(H @ sky)].
        """
        harmonics = self.disko.get_harmonics(self.sphere)
        sky = self.sky.reshape([-1, 1])
        vis = np.array([np.sum(h * sky.flatten()) for h in harmonics])
        vis2 = np.array(self.gamma @ sky)

        self.assertEqual(harmonics[0].shape[0], self.sphere.npix)
        self.assertTrue(
            np.allclose(
                np.concatenate((vis.real, vis.imag)), vis2.flatten(), atol=1e-10
            )
        )

    def test_from_pos(self):
        """
        Check that the DiSkO built from antenna positions has the uvw
        of every ordered antenna pair.
        """
        dut = DiSkO.from_ant_pos(self.ant_pos, frequency=1.5e9)

        baselines, uu, vv, ww = get_all_uvw(self.ant_pos)
        self.assertEqual(len(baselines), dut.n_v)
        self.assertTrue(np.allclose(dut.u_arr, uu))
        self.assertTrue(np.allclose(dut.v_arr, vv))
        self.assertTrue(np.allclose(dut.w_arr, ww))

        # The same antenna positions always give the same harmonics.
        dut2 = DiSkO.from_ant_pos(self.ant_pos, frequency=1.5e9)
        self.assertTrue(
            np.allclose(
                dut.get_harmonics(self.sphere), dut2.get_harmonics(self.sphere)
            )
        )

    def test_solve_vis(self):
        sky1 = self.disko.solve_vis(self.disko.vis_arr, self.sphere, scale=True)
        sky2 = self.disko.solve_vis(self.disko.vis_arr, self.subsphere, scale=True)
        self.assertEqual(sky1.shape[0], self.sphere.npix)
        self.assertEqual(sky2.shape[0], self.subsphere.npix)

        # The recovered sky must reproduce the visibilities it solved
        # against (the data is consistent, from a real sky).
        vis1 = self.gamma @ sky1
        self.assertTrue(
            np.allclose(vis1.flatten(), self.disko.vis_to_data().flatten(), atol=1e-6)
        )

    def test_lsqr_matrix_free(self):
        """
        Solve consistent fake data with a frequency axis and an npol axis.
        """
        sky = self.disko.solve_matrix_free(
            self.subdata,
            self.subsphere,
            alpha=0.0,
            scale=False,
            fista=False,
            lsqr=True,
            lsmr=False,
        )
        self.assertEqual(sky.shape[0], self.subsphere.npix)

        # Check that sky is a solution
        vis = self.subgamma @ sky
        self.assertEqual(vis[:, 0].shape, self.subdata[:, 0, 0].shape)
        for a, b in zip(vis[:, 0], self.subdata[:, 0, 0]):
            self.assertAlmostEqual(a, b, 3)

    def test_lsmr_matrix_free(self):
        """
        Solve consistent fake data with a frequency axis and an npol axis.
        """
        sky = self.disko.solve_matrix_free(
            self.subdata,
            self.subsphere,
            alpha=0.0,
            scale=False,
            fista=False,
            lsqr=False,
            lsmr=True,
        )
        self.assertEqual(sky.shape[0], self.subsphere.npix)

        # Check that sky is a solution
        vis = self.subgamma @ sky
        self.assertEqual(vis[:, 0].shape, self.subdata[:, 0, 0].shape)
        for a, b in zip(vis[:, 0], self.subdata[:, 0, 0]):
            self.assertAlmostEqual(a, b, 4)

    def test_fista_matrix_free(self):
        """
        Solve consistent fake data with a frequency axis and an npol axis.
        """
        sky = self.disko.solve_matrix_free(
            self.subdata,
            self.subsphere,
            niter=400,
            alpha=None,
            scale=False,
            fista=True,
            lsqr=False,
            lsmr=False,
        )
        self.assertEqual(sky.shape[0], self.subsphere.npix)

        # Check that sky is a solution.  FISTA is a first-order method and
        # converges more slowly than LSQR/LSMR on ill-conditioned problems,
        # so we use a looser per-element tolerance.
        vis = self.subgamma @ sky
        self.assertEqual(vis[:, 0].shape, self.subdata[:, 0, 0].shape)
        for a, b in zip(vis[:, 0], self.subdata[:, 0, 0]):
            self.assertAlmostEqual(a, b, 1)

    def test_dot_matrix_free(self):
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

        self.assertEqual(vis1.shape, vis2.shape)
        self.assertTrue(np.allclose(vis1, vis2))

        dottest(Op, self.disko.n_v * 2, self.sphere.npix, tol=1e-04)

    def test_tiny_gamma(self):
        """
        Test such a small gamma that we can inspect every element and
        check that the matrix is what we expect it to be.
        """
        tiny_subsphere = HealpixSubFoV(
            res_arcmin=3600, theta=np.radians(0.0), phi=0.0, radius_rad=np.radians(80)
        )
        self.assertEqual(tiny_subsphere.npix, 4)

        frequencies = [1.5e9]

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
        p2j = disko.jomega(frequencies[0])

        Op = disko.DiSkOOperator(
            tiny_disko.u_arr,
            tiny_disko.v_arr,
            tiny_disko.w_arr,
            data,
            frequencies,
            tiny_subsphere,
        )

        for i in range(Op.M):
            for j in range(Op.N):
                logger.info(f"[{i},{j}] {Op.A(i, j, p2j)} {tiny_gamma[i, j]}")
                self.assertAlmostEqual(Op.A(i, j, p2j), tiny_gamma[i, j])

        for i in range(Op.N):
            for j in range(Op.M):
                self.assertAlmostEqual(Op.Ah(i, j, p2j), tiny_gamma[j, i])

        dottest(Op, Op.M, Op.N, 1e-6)

        sky = np.random.normal(0, 1, tiny_subsphere.npix)
        vis1 = tiny_gamma @ sky
        vis2 = Op.matvec(sky)

        self.assertEqual(vis1.shape, vis2.shape)
        self.assertTrue(np.allclose(vis1, vis2))

    def test_gamma_size(self):
        dut = DiSkO.from_ant_pos(self.ant_pos, frequency=1.5e9)
        gamma = dut.make_gamma(self.sphere)
        gamma_sub = dut.make_gamma(self.subsphere)
        self.assertEqual(gamma.shape[1], self.sphere.npix)
        self.assertEqual(gamma_sub.shape[1], self.subsphere.npix)

    @unittest.skip("Should Fail as the sky can not be complex.")
    def test_ml_sky(self):
        pass

    @unittest.skip("Should Fail as the point sky is not entirely in the range space")
    def test_imaging(self):
        pass

    @unittest.skip("Should Fail as Direct DiSkO sucks")
    def test_solve_vs_direct(self):
        pass
