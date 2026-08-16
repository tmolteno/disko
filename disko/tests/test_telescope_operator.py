#
# Copyright Tim Molteno 2022-2026
#
# Fast unit tests for the TelescopeOperator and the Bayesian inference
# machinery. These use a tiny synthetic telescope (a few antennas and a
# coarse healpix sphere) so the SVD and the Gaussian updates run in
# milliseconds. The mathematical properties under test are size
# independent; coverage against real TART data lives in test_disko.py
# and test_pylops_operator.py.
#

import logging
import unittest

import numpy as np

from disko import DiSkO, HealpixFoV, TelescopeOperator
from disko.bayes_cli import create_prior, do_inference
from disko.disko import vis_to_real

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


class TestTelescopeOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        np.seterr(all="raise")

        # A tiny synthetic telescope: 4 antennas -> 12 baselines ->
        # 24 real visibility rows. The sphere has 48 pixels, so
        # n_s > n_v and Gamma has an exact null space.
        cls.frequency = 1.5e9
        ant_pos = np.random.uniform(-2.0, 2.0, (4, 3))
        cls.disko = DiSkO.from_ant_pos(ant_pos, frequency=cls.frequency)
        cls.sphere = HealpixFoV(nside=2)

        cls.to = TelescopeOperator(cls.disko, cls.sphere)
        logger.info(
            "n_v = {}, n_s = {}, rank = {}".format(cls.to.n_v, cls.to.n_s, cls.to.rank)
        )

        # A sky restricted to the range space of the telescope, so the
        # data is exactly representable by the truncated operator.
        sky_r = np.random.uniform(0.0, 1.0, cls.to.n_r())
        cls.sky = np.array(cls.to.V_1 @ sky_r).reshape(-1, 1)

        # Complex visibilities generated from that sky.
        cls.disko.vis_arr = cls.disko.get_harmonics(cls.sphere) @ cls.sky.flatten()
        cls.vis = np.array(cls.to.gamma @ cls.sky)
        cls.real_vis = vis_to_real(cls.disko.vis_arr)

    def test_svd(self):
        test = np.array(self.to.U @ self.to.sigma @ self.to.Vh)
        self.assertTrue(np.allclose(test, np.array(self.to.gamma)))
        self.assertTrue(
            np.allclose(np.identity(self.to.n_s), np.array(self.to.V @ self.to.Vh))
        )

    def test_gamma_consistency(self):
        # The real-augmented Gamma is [Re H; Im H] of the complex
        # harmonics, so it maps the sky to vis_to_real of the data.
        self.assertTrue(np.allclose(self.vis.flatten(), self.real_vis))

    def test_harmonics(self):
        # Each harmonic element is exp(j*phase) * pixel_area, so the
        # squared norm of every baseline row is sum(pixel_area^2).
        n_h = self.to.n_v // 2
        gamma = np.array(self.to.gamma)
        h = gamma[0:n_h, :] + 1.0j * gamma[n_h:, :]
        norms = np.sum(np.abs(h) ** 2, axis=1)
        expected = float(np.dot(self.sphere.pixel_areas, self.sphere.pixel_areas))
        self.assertTrue(np.allclose(norms, expected))

    def test_null_harmonics(self):
        # The null-space basis V_2 is orthonormal and is annihilated
        # by Gamma.
        V_2 = np.array(self.to.V_2)
        self.assertTrue(np.allclose(V_2.T @ V_2, np.identity(self.to.n_n())))
        self.assertTrue(
            np.allclose(
                np.array(self.to.gamma) @ V_2,
                np.zeros((self.to.n_v, self.to.n_n())),
            )
        )

    def test_null_to_sky(self):
        # A null-space vector maps back to a sky vector that is still
        # annihilated by Gamma (x = [0; x_n], so sky = V_2 @ x_n).
        x_n = np.random.normal(0.0, 1.0, self.to.n_n())
        sky = np.array(self.to.null_to_sky(x_n))

        self.assertTrue(np.allclose(sky, np.array(self.to.V_2) @ x_n))
        self.assertTrue(
            np.allclose(np.array(self.to.gamma) @ sky, np.zeros(self.to.n_v))
        )

    def test_range_harmonics(self):
        # The range-space basis V_1 is orthonormal.
        V_1 = np.array(self.to.V_1)
        self.assertTrue(np.allclose(V_1.T @ V_1, np.identity(self.to.n_r())))

    def test_sky_conversion(self):
        s = np.random.rand(self.to.n_s)
        x = np.array(self.to.sky_to_natural(s))
        s2 = np.array(self.to.natural_to_sky(x))
        self.assertTrue(np.allclose(s, s2))

    def test_vis(self):
        # v = Gamma s = A x = A_r x_r for a range-space sky.
        x = np.array(self.to.sky_to_natural(self.sky))
        A = np.array(self.to.U @ self.to.sigma)
        vis2 = A @ x

        x_r = x[0 : self.to.rank]
        vis3 = np.array(self.to.A_r) @ x_r

        self.assertTrue(np.allclose(self.vis, vis2))
        self.assertTrue(np.allclose(vis2, vis3))

    def test_vis_in_range(self):
        # Projecting the null space out of a range-space sky leaves the
        # visibilities unchanged.
        sky_r = np.array(self.to.P_r() @ self.sky)
        vis2 = np.array(self.to.gamma @ sky_r)
        self.assertTrue(np.allclose(self.vis, vis2))

    def test_imaging(self):
        imaged_sky = self.to.image_visibilities(self.vis, self.sphere, scale=False)
        vis3 = np.array(self.to.gamma @ imaged_sky)
        self.assertTrue(np.allclose(self.vis, vis3))

    def test_A(self):
        Ar = np.array(self.to.U_1 @ self.to.sigma[0 : self.to.rank, 0 : self.to.rank])
        self.assertEqual(Ar.shape, np.array(self.to.A_r).shape)

    def test_imaging_vs_natural(self):
        imaged_sky = self.to.image_natural(self.vis, self.sphere, scale=False)
        vis3 = np.array(self.to.gamma @ imaged_sky)
        self.assertTrue(np.allclose(self.vis, vis3))

    def test_bayes(self):
        prior = self.to.get_prior()  # in the image (sky) space.
        prior_r = prior.linear_transform(self.to.Vh)

        # A very strong likelihood precision, so the data dominates.
        sigma_precision = 1e10 * np.identity(self.to.n_v)

        posterior = self.to.sequential_inference(
            prior_r, self.real_vis, sigma_precision
        )

        self.assertEqual(posterior.mu.shape, (self.to.n_s,))
        self.assertEqual(posterior.sigma().shape, (self.to.n_s, self.to.n_s))

        # The posterior mean fits the data.
        fitted_vis = np.array(self.to.gamma) @ posterior.mu
        self.assertTrue(np.allclose(fitted_vis, self.real_vis, rtol=1e-4, atol=1e-4))

        # The range space is inferred from the data, while the null
        # space keeps the prior: the telescope carries no information
        # about it.
        P_n = np.array(self.to.V_2 @ self.to.V_2.T)
        expected = np.array(self.to.P_r() @ self.sky).flatten() + P_n @ prior.mu
        self.assertTrue(np.allclose(posterior.mu, expected, rtol=1e-3, atol=1e-3))

    def test_do_inference(self):
        # The inference path used by the disko_bayes CLI.
        prior = create_prior(self.disko.vis_arr, self.sphere, None)
        posterior = do_inference(self.disko, self.sphere, prior, sigma_v=1e-6)

        self.assertEqual(posterior.mu.shape, (self.to.n_s,))

        # With a tiny sigma_v the likelihood dominates: the range space
        # is recovered from the data, the null space keeps the prior.
        P_n = np.array(self.to.V_2 @ self.to.V_2.T)
        expected = np.array(self.to.P_r() @ self.sky).flatten() + P_n @ prior.mu
        self.assertTrue(np.allclose(posterior.mu, expected, rtol=1e-3, atol=1e-3))

    def test_create_prior(self):
        # The heuristic prior has mean = median |vis| and covariance
        # p95(|vis|)^2 * I (so the prior std is p95).
        vis = np.arange(1, 101).astype(np.complex128)
        prior = create_prior(vis, self.sphere, None)

        p50, p95 = np.percentile(np.abs(vis), [50, 95])
        self.assertTrue(np.allclose(prior.mu, p50 * np.ones(self.sphere.npix)))
        self.assertTrue(
            np.allclose(prior.sigma(), p95**2 * np.identity(self.sphere.npix))
        )
