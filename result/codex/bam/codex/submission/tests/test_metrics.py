"""Evaluation metrics of Sections 5.1-5.3 (KL divergences, relative errors)."""

import os
import sys
import unittest

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.divergence import kl_gaussian, kl_gaussian_samples  # noqa: E402
from bam.metrics import reconstruction_mse, relative_mean_error, relative_sd_error  # noqa: E402
from bam.targets import GaussianTarget, SinhArcsinhTarget  # noqa: E402


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.rng = np.random.default_rng(0)

    def test_kl_gaussian_matches_monte_carlo(self):
        D = 3
        p = GaussianTarget(self.rng.normal(size=D), np.eye(D) * 1.7)
        mu_q = self.rng.normal(size=D)
        Sigma_q = np.eye(D) * 0.6
        # forward direction KL(p ; q), reverse direction KL(q ; p)
        exact_f = kl_gaussian(p.mu, p.Sigma, mu_q, Sigma_q)
        exact_r = kl_gaussian(mu_q, Sigma_q, p.mu, p.Sigma)
        mc_f = kl_gaussian_samples(p, mu_q, Sigma_q, 400000, self.key, "forward")
        mc_r = kl_gaussian_samples(p, mu_q, Sigma_q, 400000, self.key, "reverse")
        self.assertLess(abs(mc_f - exact_f) / max(exact_f, 1e-8), 0.05)
        self.assertLess(abs(mc_r - exact_r) / max(exact_r, 1e-8), 0.05)

    def test_kl_of_identical_distributions_is_zero(self):
        p = GaussianTarget(self.rng.normal(size=4), np.eye(4))
        self.assertAlmostEqual(kl_gaussian(p.mu, p.Sigma, p.mu, p.Sigma), 0.0, places=12)

    def test_non_gaussian_kl_estimators_are_consistent(self):
        """For a sinh-arcsinh target with small skew the KL is close to the Gaussian one."""
        t = SinhArcsinhTarget(np.zeros(2), np.eye(2) * 0.8, skew=0.2, tail=1.0)
        mu_q = np.array([0.1, -0.1])
        Sigma_q = np.eye(2)
        forward = kl_gaussian_samples(t, mu_q, Sigma_q, 400000, self.key, "forward")
        reverse = kl_gaussian_samples(t, mu_q, Sigma_q, 400000, self.key, "reverse")
        self.assertGreater(forward, 0.0)
        self.assertGreater(reverse, 0.0)
        self.assertLess(forward, 1.0)
        self.assertLess(reverse, 1.0)

    def test_relative_errors(self):
        ref_mean = np.array([1.0, -2.0])
        ref_sd = np.array([0.5, 2.0])
        self.assertAlmostEqual(relative_mean_error(ref_mean, ref_mean, ref_sd), 0.0)
        self.assertAlmostEqual(relative_mean_error(ref_mean + ref_sd, ref_mean, ref_sd), np.sqrt(2.0))
        self.assertAlmostEqual(relative_sd_error(ref_sd, ref_sd), 0.0)
        self.assertAlmostEqual(relative_sd_error(2 * ref_sd, ref_sd), np.sqrt(2.0))

    def test_reconstruction_mse(self):
        x = np.ones(10)
        self.assertAlmostEqual(reconstruction_mse(x, x), 0.0)
        self.assertAlmostEqual(reconstruction_mse(x, x + 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
