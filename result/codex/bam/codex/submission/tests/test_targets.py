"""Checks on the target distributions used in the experiments.

* sinh-arcsinh targets: the density is normalized, the Gaussian is recovered for
  ``s = 0, tau = 1``, and the score agrees with finite differences;
* posteriordb targets: the transcription of the Stan models is validated against
  the HMC reference samples through the identity ``E_p[grad log p(z)] = 0``,
  which holds for the true posterior (in both the constrained and the
  unconstrained parameterization);
* the deep generative posterior: the score agrees with finite differences.
"""

import os
import sys
import unittest

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.deep_generative import DeepGenerativePosterior  # noqa: E402
from bam.posterior_models import (  # noqa: E402
    load_posteriordb_target,
    reference_draws_in_target_space,
)
from bam.targets import GaussianTarget, SinhArcsinhTarget  # noqa: E402
from bam.vae import VAEConfig, init_params  # noqa: E402


class TestSinhArcsinh(unittest.TestCase):
    def test_density_is_normalized(self):
        for skew, tail in ((0.0, 1.0), (0.9, 1.3), (1.8, 1.0)):
            t = SinhArcsinhTarget(np.array([0.3]), np.array([[0.7]]), skew=skew, tail=tail)
            z = np.linspace(-200, 200, 400001)
            integral = np.trapz(np.exp(t.log_density(z[:, None])), z)
            self.assertAlmostEqual(integral, 1.0, places=3)

    def test_gaussian_limit(self):
        t = SinhArcsinhTarget(np.array([0.2, -0.4]), np.eye(2), skew=0.0, tail=1.0)
        g = GaussianTarget(np.array([0.2, -0.4]), np.eye(2))
        Z = np.random.default_rng(0).normal(size=(5, 2))
        self.assertLess(np.abs(t.log_density(Z) - g.log_density(Z)).max(), 1e-10)
        self.assertLess(np.abs(t.score(Z) - g.score(Z)).max(), 1e-10)

    def test_score_matches_finite_differences(self):
        key = jax.random.PRNGKey(0)
        t = SinhArcsinhTarget(np.array([0.1, -0.2]), np.array([[1.0, 0.3], [0.3, 0.8]]),
                              skew=1.2, tail=0.8)
        Z = np.asarray(jax.random.normal(key, (3, 2)))
        S = t.score(Z)
        h = 1e-6
        for i in range(2):
            e = np.zeros(2)
            e[i] = h
            fd = (t.log_density(Z + e) - t.log_density(Z - e)) / (2 * h)
            self.assertLess(np.abs(fd - S[:, i]).max(), 1e-5)

    def test_samples_match_density(self):
        """The empirical distribution of the samples matches the density."""
        t = SinhArcsinhTarget(np.array([0.0]), np.array([[1.0]]), skew=1.0, tail=1.0)
        S = t.sample(jax.random.PRNGKey(0), 200000)[:, 0]
        grid = np.linspace(-1.0, 4.0, 51)
        z = np.linspace(-200, 200, 400001)
        # compare the empirical CDF on the grid with the CDF implied by the density
        pdf = np.exp(t.log_density(z[:, None]))
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(z))])
        cdf = cdf / cdf[-1]
        model_cdf = np.interp(grid, z, cdf)
        emp_cdf = np.asarray([(S <= g).mean() for g in grid])
        self.assertLess(np.abs(model_cdf - emp_cdf).max(), 0.01)


class TestPosteriorModels(unittest.TestCase):
    """``E_p[score] = 0`` under the HMC reference samples (a model transcription test)."""

    NAMES = ("arK", "gp_pois_regr", "eight_schools_centered")

    def test_reference_draws_are_stationary(self):
        for name in self.NAMES:
            for param in ("unconstrained", "constrained"):
                target = load_posteriordb_target(name, parameterization=param)
                draws = reference_draws_in_target_space(name, parameterization=param)
                g = target.score(draws)
                se = g.std(axis=0) / np.sqrt(draws.shape[0])
                # the Monte-Carlo average of the score should be zero up to its
                # own sampling error
                self.assertLess(np.abs(g.mean(axis=0)).max(), 10.0 * se.max() + 1e-6,
                                msg=f"{name}/{param}: E[score] far from zero -> model mismatch")

    def test_reference_draws_have_high_density(self):
        rng = np.random.default_rng(0)
        for name in self.NAMES:
            target = load_posteriordb_target(name)
            draws = reference_draws_in_target_space(name)
            lp_draws = target.log_density(draws[:500]).mean()
            perturbed = draws.mean(0) + rng.normal(size=(500, target.dim)) * draws.std(0)
            lp_pert = target.log_density(perturbed).mean()
            self.assertGreater(lp_draws, lp_pert)

    def test_variational_summaries_are_in_constrained_space(self):
        """The lognormal moments used for the relative-error metrics are exact."""
        for name in self.NAMES:
            target = load_posteriordb_target(name)
            mu = np.zeros(target.dim)
            Sigma = np.diag(np.linspace(0.01, 0.05, target.dim))
            mean, sd = target.variational_summaries(mu, Sigma)
            for d in target.positive_dims:
                self.assertAlmostEqual(mean[d], np.exp(0.5 * Sigma[d, d]), places=8)
                self.assertAlmostEqual(sd[d], np.sqrt((np.exp(Sigma[d, d]) - 1) * np.exp(Sigma[d, d])),
                                       places=8)


class TestDeepGenerativePosterior(unittest.TestCase):
    def test_score_matches_finite_differences(self):
        config = VAEConfig(latent_dim=6, c_hid=4)
        params = init_params(jax.random.PRNGKey(0), config)
        x_obs = np.random.default_rng(0).uniform(-1, 1, size=(32, 32, 3))
        target = DeepGenerativePosterior(params["dec"], config, x_obs)
        key = jax.random.PRNGKey(1)
        Z = np.asarray(jax.random.normal(key, (2, config.latent_dim))) * 0.1
        S = target.score(Z)
        h = 1e-5
        for i in range(config.latent_dim):
            e = np.zeros(config.latent_dim)
            e[i] = h
            fd = (target.log_density(Z + e) - target.log_density(Z - e)) / (2 * h)
            self.assertLess(np.abs(fd - S[:, i]).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
