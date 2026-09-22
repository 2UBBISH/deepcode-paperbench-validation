"""Properties of the score-based divergence (Section 2 and Appendix A).

These are the claims the paper makes about the divergence itself:

* Definition A.2, and its Gaussian form ``E_q[||grad log(q/p)||^2_{Cov(q)}]``;
* Proposition A.7 (closed form for two Gaussians), and consistency of the
  Monte-Carlo estimator with it, together with non-negativity / equality;
* Theorem A.4 (affine invariance);
* Theorem A.5 (annealing: ``D(q; p) = D (beta - 1)^2`` for ``p ∝ q^beta``);
* Theorem A.6 (exponential tilting: ``D(q; p) = theta^T Psi theta``);
* Corollary A.8 (``D(q;p) / 2 = KL(q;p) = KL(p;q)`` for equal covariances).
"""

import os
import sys
import unittest

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.divergence import (  # noqa: E402
    annealing_divergence,
    exponential_tilting_divergence,
    gaussian_score_divergence,
    kl_gaussian,
    score_based_divergence,
)
from bam.linalg import symmetrize  # noqa: E402
from bam.targets import GaussianTarget  # noqa: E402


def random_spd(rng, D, scale=1.0):
    A = rng.normal(size=(D, D)) * scale
    return symmetrize(A @ A.T) + 0.5 * np.eye(D)


class TestScoreBasedDivergence(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.D = 4
        self.mu_p = self.rng.normal(size=self.D)
        self.Sigma_p = random_spd(self.rng, self.D)
        self.mu_q = self.rng.normal(size=self.D)
        self.Sigma_q = random_spd(self.rng, self.D, scale=0.5)
        self.key = jax.random.PRNGKey(0)

    def test_nonnegativity_and_equality(self):
        self.assertAlmostEqual(gaussian_score_divergence(self.mu_p, self.Sigma_p,
                                                         self.mu_p, self.Sigma_p), 0.0, places=10)
        self.assertGreater(gaussian_score_divergence(self.mu_p, self.Sigma_p,
                                                     self.mu_q, self.Sigma_q), 0.0)

    def test_monte_carlo_matches_closed_form(self):
        target = GaussianTarget(self.mu_p, self.Sigma_p)
        mc = score_based_divergence(target, self.mu_q, self.Sigma_q, 400000, self.key)
        exact = gaussian_score_divergence(self.mu_p, self.Sigma_p, self.mu_q, self.Sigma_q)
        self.assertLess(abs(mc - exact) / max(exact, 1e-8), 0.05)

    def test_affine_invariance(self):
        """Theorem A.4: the divergence is invariant under affine reparameterizations."""
        A = self.rng.normal(size=(self.D, self.D))
        while abs(np.linalg.det(A)) < 1e-3:
            A = self.rng.normal(size=(self.D, self.D))
        b = self.rng.normal(size=self.D)
        base = gaussian_score_divergence(self.mu_p, self.Sigma_p, self.mu_q, self.Sigma_q)
        transformed = gaussian_score_divergence(
            A @ self.mu_p + b, symmetrize(A @ self.Sigma_p @ A.T),
            A @ self.mu_q + b, symmetrize(A @ self.Sigma_q @ A.T))
        self.assertAlmostEqual(base, transformed, places=8)

    def test_annealing(self):
        """Theorem A.5: for ``p ∝ q^beta`` the divergence equals ``D (beta - 1)^2``."""
        target = GaussianTarget(self.mu_q, self.Sigma_p)  # placeholder, replaced below
        for beta in (0.25, 0.5, 2.0, 4.0):
            # p = q^beta normalized is N(mu_q, Sigma_q / beta)
            p = GaussianTarget(self.mu_q, self.Sigma_q / beta)
            mc = score_based_divergence(p, self.mu_q, self.Sigma_q, 200000, self.key)
            self.assertAlmostEqual(mc, annealing_divergence(beta, self.D), delta=0.08 * self.D)
        del target

    def test_exponential_tilting(self):
        """Theorem A.6: for ``p ∝ q exp(theta^T z)`` the divergence is ``theta^T Psi theta``."""
        theta = self.rng.normal(size=self.D)
        # p = N(mu_q + Sigma_q theta, Sigma_q)
        p = GaussianTarget(self.mu_q + self.Sigma_q @ theta, self.Sigma_q)
        exact = exponential_tilting_divergence(theta, self.Sigma_q)
        mc = score_based_divergence(p, self.mu_q, self.Sigma_q, 200000, self.key)
        self.assertLess(abs(mc - exact) / max(exact, 1e-8), 0.05)
        self.assertLess(abs(gaussian_score_divergence(p.mu, p.Sigma, self.mu_q, self.Sigma_q) - exact),
                        1e-8 * max(exact, 1.0))

    def test_relation_to_kl_with_equal_covariance(self):
        """Corollary A.8: half the divergence equals the KL divergence."""
        p = GaussianTarget(self.mu_p, self.Sigma_p)
        q_mu = self.mu_p + self.rng.normal(size=self.D)
        D = gaussian_score_divergence(p.mu, p.Sigma, q_mu, p.Sigma)
        self.assertAlmostEqual(0.5 * D, kl_gaussian(q_mu, p.Sigma, p.mu, p.Sigma), places=8)
        self.assertAlmostEqual(kl_gaussian(q_mu, p.Sigma, p.mu, p.Sigma),
                               kl_gaussian(p.mu, p.Sigma, q_mu, p.Sigma), places=10)


if __name__ == "__main__":
    unittest.main()
