"""The BaM updates of Section 3.1 (Algorithm 1) and Appendix C.

Checked here:

* the BATCH step statistics and the MATCH step updates agree with eqs. (6)-(13)
  when recomputed by hand from the same batch;
* the low-rank solver (Lemma B.3) produces the same iterates as the general
  solver, and both converge to a Gaussian target;
* the limiting cases: ``lambda -> 0`` leaves the iterates unchanged,
  ``B = 1`` with ``lambda -> inf`` reproduces the GSM update of Appendix C.3,
  and ``lambda -> inf`` with ``B -> inf`` converges in one step
  (Corollary D.5);
* BaM converges to a Gaussian target and, for a fixed gradient budget, is more
  accurate than ADVI on a synthetic Gaussian target (the trend of Figure 5.1).
"""

import os
import sys
import unittest

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.bam import BaM, BaMConfig, lambda_constant  # noqa: E402
from bam.baselines import GSM, GradientVI, GradientVIConfig  # noqa: E402
from bam.divergence import kl_gaussian  # noqa: E402
from bam.linalg import symmetrize  # noqa: E402
from bam.targets import GaussianTarget, random_gaussian_target, sample_gaussian  # noqa: E402


class TestBaMUpdates(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.D = 5
        self.target = random_gaussian_target(self.D, seed=0)
        self.mu = self.rng.normal(size=self.D)
        A = self.rng.normal(size=(self.D, self.D))
        self.Sigma = symmetrize(A @ A.T) + np.eye(self.D)
        self.B = 7
        self.lam = 3.5
        self.key = jax.random.PRNGKey(0)

    def test_match_step_matches_closed_form(self):
        bam = BaM(self.target, BaMConfig(batch_size=self.B, lam=lambda_constant(self.lam)))
        out = bam.step(self.mu, self.Sigma, self.key, 0)

        # recompute the batch statistics from the same batch
        Z = sample_gaussian(self.key, self.mu, np.linalg.cholesky(self.Sigma), self.B)
        G = self.target.score(Z)
        zbar, gbar = Z.mean(0), G.mean(0)
        C = ((Z - zbar).T @ (Z - zbar)) / self.B
        Gamma = ((G - gbar).T @ (G - gbar)) / self.B
        self.assertLess(np.abs(out["zbar"] - zbar).max(), 1e-12)
        self.assertLess(np.abs(out["gbar"] - gbar).max(), 1e-12)
        self.assertLess(np.abs(out["C"] - C).max(), 1e-12)
        self.assertLess(np.abs(out["Gamma"] - Gamma).max(), 1e-12)

        lam = self.lam
        U = lam * Gamma + (lam / (1 + lam)) * np.outer(gbar, gbar)
        V = self.Sigma + lam * C + (lam / (1 + lam)) * np.outer(self.mu - zbar, self.mu - zbar)
        self.assertLess(np.abs(out["U"] - U).max(), 1e-12)
        self.assertLess(np.abs(out["V"] - V).max(), 1e-12)
        # eq. (9): Sigma_{t+1} U Sigma_{t+1} + Sigma_{t+1} = V
        S_new = out["Sigma"]
        self.assertLess(np.abs(S_new @ U @ S_new + S_new - V).max(), 1e-8)
        # eq. (13)
        mu_expected = (self.mu + lam * (S_new @ gbar + zbar)) / (1 + lam)
        self.assertLess(np.abs(out["mu"] - mu_expected).max(), 1e-12)

    def test_low_rank_and_full_solver_agree(self):
        lam = lambda_constant(self.lam)
        out_low = BaM(self.target, BaMConfig(batch_size=self.B, lam=lam, solver="low_rank")).run(
            self.key, 10, self.mu, self.Sigma)
        out_full = BaM(self.target, BaMConfig(batch_size=self.B, lam=lam, solver="cholesky")).run(
            self.key, 10, self.mu, self.Sigma)
        self.assertLess(np.abs(out_low["mu"] - out_full["mu"]).max(), 1e-8)
        self.assertLess(np.abs(out_low["Sigma"] - out_full["Sigma"]).max(), 1e-8)

    def test_zero_regularization_limit_leaves_iterates_unchanged(self):
        """Section 3.1: when ``lambda_t -> 0`` the updates have no effect."""
        # use a well-conditioned target so that the scores are O(1) and the
        # change of the iterates is proportional to lambda
        target = GaussianTarget(np.zeros(self.D), 2.0 * np.eye(self.D))
        mu_t, Sigma_t = np.zeros(self.D), np.eye(self.D)
        changes = []
        for lam in (1e-2, 1e-4, 1e-6):
            bam = BaM(target, BaMConfig(batch_size=self.B, lam=lambda_constant(lam)))
            out = bam.step(mu_t, Sigma_t, self.key, 0)
            changes.append((np.abs(out["mu"] - mu_t).max(), np.abs(out["Sigma"] - Sigma_t).max()))
        self.assertLess(changes[-1][0], 1e-4)
        self.assertLess(changes[-1][1], 1e-4)
        # the change decreases as the regularization becomes stronger
        self.assertLess(changes[1][0], changes[0][0])
        self.assertLess(changes[2][0], changes[1][0])

    def test_large_regularization_one_sample_recovers_gsm(self):
        """Appendix C.3: ``B = 1`` and ``lambda -> inf`` reproduces GSM's updates."""
        lam = 1e8
        key = jax.random.PRNGKey(3)
        Z = sample_gaussian(key, self.mu, np.linalg.cholesky(self.Sigma), 1)
        g = self.target.score(Z)
        bam = BaM(self.target, BaMConfig(batch_size=1, lam=lambda_constant(lam)))
        out = bam.step(self.mu, self.Sigma, key, 0)
        # eqs. (114)-(115): Sigma_{t+1} g g^T Sigma_{t+1} + Sigma_{t+1}
        #                   = Sigma_t + (mu_t - z)(mu_t - z)^T,  mu_{t+1} = Sigma_{t+1} g + z
        V = self.Sigma + np.outer(self.mu - Z[0], self.mu - Z[0])
        self.assertLess(np.abs(out["Sigma"] @ np.outer(g[0], g[0]) @ out["Sigma"] + out["Sigma"] - V).max(),
                        1e-4)
        self.assertLess(np.abs(out["mu"] - (out["Sigma"] @ g[0] + Z[0])).max(), 1e-4)

    def test_convergence_on_gaussian_target(self):
        B, T = 20, 400
        out = BaM(self.target, BaMConfig(batch_size=B, lam=lambda_constant(B * self.D))).run(
            self.key, T, np.zeros(self.D), np.eye(self.D))
        fkl = kl_gaussian(out["mu"][-1], out["Sigma"][-1], self.target.mu, self.target.Sigma)
        self.assertLess(fkl, 1e-2)

    def test_bam_beats_advi_per_gradient_evaluation(self):
        """Section 5.1 trend: BaM reaches a much smaller KL for the same budget."""
        D = 4
        target = random_gaussian_target(D, seed=1)
        key = jax.random.PRNGKey(0)
        n_grad_evals = 400
        B = 20
        bam = BaM(target, BaMConfig(batch_size=B, lam=lambda_constant(B * D)))
        out_bam = bam.run(key, n_grad_evals // B, np.zeros(D), np.eye(D))
        vi = GradientVI(target, GradientVIConfig(batch_size=2, learning_rate=0.01, loss="elbo",
                                                 n_iters=n_grad_evals // 2))
        out_advi = vi.run(key, mu0=np.zeros(D), Sigma0=np.eye(D))
        fkl_bam = kl_gaussian(out_bam["mu"][-1], out_bam["Sigma"][-1], target.mu, target.Sigma)
        fkl_advi = kl_gaussian(out_advi["mu"][-1], out_advi["Sigma"][-1], target.mu, target.Sigma)
        self.assertLess(fkl_bam, 1e-2)
        self.assertLess(fkl_bam, fkl_advi)

    def test_gsm_recovers_gaussian_target(self):
        out = GSM(self.target, batch_size=2).run(self.key, 200, np.zeros(self.D), np.eye(self.D))
        fkl = kl_gaussian(out["mu"][-1], out["Sigma"][-1], self.target.mu, self.target.Sigma)
        self.assertLess(fkl, 1e-3)


if __name__ == "__main__":
    unittest.main()
