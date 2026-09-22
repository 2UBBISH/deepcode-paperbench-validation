"""Theorem 3.1 and the infinite-batch analysis of Appendix D."""

import os
import sys
import unittest

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.bam import BaM, BaMConfig, lambda_constant  # noqa: E402
from bam.targets import random_gaussian_target  # noqa: E402
from bam.theory import infinite_batch_step, run_infinite_batch, theoretical_bounds  # noqa: E402


class TestTheorem31(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.D = 6
        self.target = random_gaussian_target(self.D, seed=0)
        self.mu0 = self.rng.normal(size=self.D)
        A = self.rng.normal(size=(self.D, self.D)) / np.sqrt(self.D)
        self.Sigma0 = A @ A.T + 0.05 * np.eye(self.D)

    def test_bounds_hold_for_all_regularization_levels(self):
        for lam in (0.05, 1.0, 10.0, 1000.0):
            res = run_infinite_batch(self.mu0, self.Sigma0, self.target.mu, self.target.Sigma, lam, 150)
            bnd = theoretical_bounds(self.mu0, self.Sigma0, self.target.mu, self.target.Sigma, lam, 150)
            self.assertLessEqual(res["eps_norm"].max(), bnd["eps_bound"].max() + 1e-10)
            self.assertTrue(np.all(res["Delta_norm"] <= bnd["Delta_bound"] + 1e-8),
                            msg=f"Delta bound violated for lambda={lam}")
            # convergence is guaranteed for any lambda > 0, but it is slower for
            # small lambda (delta = lambda * beta / (1 + lambda))
            if lam >= 1.0:
                self.assertLess(res["eps_norm"][-1], 1e-8)
                self.assertLess(res["Delta_norm"][-1], 1e-6)
            else:
                self.assertLess(res["eps_norm"][-1], res["eps_norm"][0])
                self.assertLess(res["Delta_norm"][-1], res["Delta_norm"][0])

    def test_per_iteration_inequalities(self):
        """eqs. (19)-(20): ||eps_{t+1}|| <= (1-delta)||eps_t|| and
        ||Delta_{t+1}|| <= (1-delta)||Delta_t|| + ||eps_t||^2."""
        lam = 2.0
        res = run_infinite_batch(self.mu0, self.Sigma0, self.target.mu, self.target.Sigma, lam, 50)
        bnd = theoretical_bounds(self.mu0, self.Sigma0, self.target.mu, self.target.Sigma, lam, 50)
        delta = bnd["delta"]
        self.assertTrue(np.all(res["eps_norm"][1:] <= (1 - delta) * res["eps_norm"][:-1] + 1e-12))
        self.assertTrue(np.all(res["Delta_norm"][1:] <= (1 - delta) * res["Delta_norm"][:-1]
                               + res["eps_norm"][:-1] ** 2 + 1e-8))

    def test_matches_finite_batch_run(self):
        """The finite-batch run approaches the infinite-batch recursion as B grows."""
        T, lam = 3, 1.0
        target = random_gaussian_target(self.D, seed=1)
        inf = run_infinite_batch(self.mu0, self.Sigma0, target.mu, target.Sigma, lam, T)
        L = np.linalg.cholesky(target.Sigma)
        rels = {}
        for B in (256, 8192):
            bam = BaM(target, BaMConfig(batch_size=B, lam=lambda_constant(lam)))
            out = bam.run(jax.random.PRNGKey(0), T, self.mu0, self.Sigma0)
            eps_finite = np.asarray([np.linalg.norm(np.linalg.solve(L, out["mu"][i] - target.mu))
                                     for i in range(T + 1)])
            rels[B] = float((np.abs(eps_finite - inf["eps_norm"])
                             / np.maximum(inf["eps_norm"], 1e-12)).max())
        self.assertLess(rels[8192], rels[256])      # closer to the limit for larger B
        self.assertLess(rels[8192], 0.15)

    def test_one_step_convergence(self):
        """Corollary D.5: in the limit ``B -> inf`` followed by ``lambda -> inf``,
        the algorithm converges in one step."""
        errs = []
        for lam in (1e2, 1e4, 1e6):
            mu1, S1 = infinite_batch_step(self.mu0, self.Sigma0, self.target.mu, self.target.Sigma, lam)
            L = np.linalg.cholesky(self.target.Sigma)
            errs.append(np.linalg.norm(np.linalg.solve(L, mu1 - self.target.mu))
                        + np.linalg.norm(np.linalg.solve(L, (S1 - self.target.Sigma)) @ np.linalg.inv(L)))
        self.assertLess(errs[-1], errs[0])
        self.assertLess(errs[-1], 1e-3)

    def test_decay_rate_matches_theory(self):
        """The measured decay of ``||eps_t||`` is close to the predicted ``(1-delta)``."""
        lam = 1.0
        res = run_infinite_batch(self.mu0, self.Sigma0, self.target.mu, self.target.Sigma, lam, 40)
        bnd = theoretical_bounds(self.mu0, self.Sigma0, self.target.mu, self.target.Sigma, lam, 40)
        ratio = res["eps_norm"][1:] / np.maximum(res["eps_norm"][:-1], 1e-300)
        self.assertLess(ratio.max(), 1.0)                      # strictly decreasing
        self.assertLessEqual(ratio.max(), 1.0 - bnd["delta"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
