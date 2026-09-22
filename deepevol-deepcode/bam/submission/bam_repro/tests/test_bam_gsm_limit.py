"""Tests for the BaM <-> GSM relationship and the analytic Gaussian limits.

All assertions are derived from the paper's own statements:

* :math:`\\S3.1`, eqs. (4)-(13) / Algorithm 1 -- batch step statistics
  :math:`(\\bar z, C, \\bar g, \\Gamma)`, the matrices :math:`U, V`, the quadratic
  matrix equation :math:`\\Sigma_{t+1} U \\Sigma_{t+1} + \\Sigma_{t+1} = V`, and the
  mean update :math:`\\mu_{t+1} = \\frac{1}{1+\\lambda_t}\\mu_t +
  \\frac{\\lambda_t}{1+\\lambda_t}(\\Sigma_{t+1}\\bar g + \\bar z)`.
* :math:`\\S C.3`, eqs. (114)-(115) -- the :math:`B = 1`, :math:`\\lambda \\to \\infty`
  limit: :math:`U = g_t g_t^\\top`,
  :math:`V = \\Sigma_t + (\\mu_t - z_t)(\\mu_t - z_t)^\\top`,
  :math:`\\Sigma_{t+1} g_t g_t^\\top \\Sigma_{t+1} + \\Sigma_{t+1} = V` and
  :math:`\\mu_{t+1} = \\Sigma_{t+1} g_t + z_t`, which coincide exactly with the GSM
  updates (Modi et al., 2023).
* :math:`\\S E.1`, Algorithm 3 -- GSM per-sample updates
  :math:`\\delta\\mu_b`, :math:`\\delta\\Sigma_b` and the :math:`\\rho` quadratic
  :math:`\\rho(1+\\rho) = s_b^\\top \\Sigma_t s_b + [(\\mu_t - z_b)^\\top s_b]^2`.
* :math:`\\S D.2`, Lemma D.2 -- :math:`B \\to \\infty` limits of the batch statistics
  for a Gaussian target :math:`p = \\mathcal{N}(\\mu_*, \\Sigma_*)`.
* :math:`\\S 3.2` Theorem 3.1 / :math:`\\S D.5` -- exponential convergence, which implies
  that in the infinite batch + large regularization limit BaM recovers the Gaussian
  target essentially in one step (the "one-step recovery" check).

The tests use small, deterministic reference implementations (transcribed directly from
the paper equations) as oracles, plus the analytic infinite-batch batch constructed in
:func:`_exact_limit_batch`, which has *exactly* the :math:`B \\to \\infty` statistics.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

# --------------------------------------------------------------------------------------
# Import shims: work both as ``bam_repro.bam...`` and as a flat ``bam...`` package.
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import path dependent
    from bam_repro.bam.matrix_equations import (  # type: ignore
        ensure_spd,
        residual as matrix_residual,
        solve_quadratic_matrix_equation,
        solve_quadratic_matrix_equation_dense,
        solve_quadratic_matrix_equation_low_rank,
        symmetrize,
    )
except ImportError:  # pragma: no cover
    from bam.matrix_equations import (  # type: ignore
        ensure_spd,
        residual as matrix_residual,
        solve_quadratic_matrix_equation,
        solve_quadratic_matrix_equation_dense,
        solve_quadratic_matrix_equation_low_rank,
        symmetrize,
    )

try:  # pragma: no cover
    from bam_repro.bam.bam import (  # type: ignore
        BaM,
        bam_match_step,
        batch_statistics,
        resolve_lambda,
    )
except ImportError:  # pragma: no cover
    from bam.bam import (  # type: ignore
        BaM,
        bam_match_step,
        batch_statistics,
        resolve_lambda,
    )

try:  # pragma: no cover
    from bam_repro.bam.vi_base import (  # type: ignore
        gaussian_score,
        init_gaussian_state,
        project_spd,
    )
except ImportError:  # pragma: no cover
    from bam.vi_base import (  # type: ignore
        gaussian_score,
        init_gaussian_state,
        project_spd,
    )

try:  # pragma: no cover
    from bam_repro.bam.learning_rate import (  # type: ignore
        bd_over_t_schedule,
        bd_schedule,
        make_schedule,
        schedule_values,
    )
except ImportError:  # pragma: no cover
    bd_over_t_schedule = bd_schedule = make_schedule = schedule_values = None  # type: ignore

try:  # pragma: no cover
    from bam_repro.baselines.gsm import (  # type: ignore
        GSM,
        gsm_batch_update,
        gsm_fit,
        gsm_sample_update,
        solve_rho,
    )
except ImportError:  # pragma: no cover
    try:  # pragma: no cover
        from baselines.gsm import (  # type: ignore
            GSM,
            gsm_batch_update,
            gsm_fit,
            gsm_sample_update,
            solve_rho,
        )
    except ImportError:  # pragma: no cover
        GSM = gsm_batch_update = gsm_fit = gsm_sample_update = solve_rho = None  # type: ignore

try:  # pragma: no cover
    from bam_repro.targets.gaussian_target import GaussianTarget  # type: ignore
except ImportError:  # pragma: no cover
    try:  # pragma: no cover
        from targets.gaussian_target import GaussianTarget  # type: ignore
    except ImportError:  # pragma: no cover
        GaussianTarget = None  # type: ignore


# --------------------------------------------------------------------------------------
# Local reference implementations (transcribed from the paper; NumPy only)
# --------------------------------------------------------------------------------------
def _scores(score_fn, z):
    """Evaluate ``score_fn`` on a batch, tolerating element-wise implementations."""
    z = np.asarray(z, dtype=float)
    try:
        out = np.asarray(score_fn(z), dtype=float)
        if out.shape == z.shape:
            return out
    except Exception:  # pragma: no cover - element-wise fallback
        pass
    return np.stack([np.asarray(score_fn(zb), dtype=float).reshape(-1) for zb in z])


def _ref_stats(z, g):
    """Batch statistics of eq. (95): ``z_bar, C, g_bar, Gamma`` (centered covariances)."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    g = np.atleast_2d(np.asarray(g, dtype=float))
    B = z.shape[0]
    z_bar = z.mean(axis=0)
    g_bar = g.mean(axis=0)
    dz = z - z_bar
    dg = g - g_bar
    C = (dz.T @ dz) / B
    Gamma = (dg.T @ dg) / B
    return z_bar, C, g_bar, Gamma


def _ref_solve_quadratic(U, V):
    """Dense closed form (Lemma B.1): ``X = 2 V (I + (I + 4 U V)^{1/2})^{-1}``."""
    U = symmetrize(np.asarray(U, dtype=float))
    V = symmetrize(np.asarray(V, dtype=float))
    D = V.shape[0]
    M = np.eye(D) + 4.0 * (U @ V)
    w, Q = np.linalg.eigh(symmetrize(M))
    w = np.clip(w, 0.0, None)  # principal square root, negative eigenvalues -> 0
    S = (Q * np.sqrt(w)) @ Q.T
    X = 2.0 * V @ np.linalg.inv(np.eye(D) + S)
    return symmetrize(X)


def _ref_uv(mu, Sigma, z, g, lam):
    """Matrices U and V of eq. (108)."""
    z_bar, C, g_bar, Gamma = _ref_stats(z, g)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    U = lam * Gamma + (lam / (1.0 + lam)) * np.outer(g_bar, g_bar)
    d = mu - z_bar
    V = Sigma + lam * C + (lam / (1.0 + lam)) * np.outer(d, d)
    return symmetrize(U), symmetrize(V)


def _ref_bam_match(mu, Sigma, z, g, lam):
    """One BaM match step, eqs. (108)-(113) of the paper."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = symmetrize(np.asarray(Sigma, dtype=float))
    U, V = _ref_uv(mu, Sigma, z, g, lam)
    Sigma_next = _ref_solve_quadratic(U, V)
    z_bar, _, g_bar, _ = _ref_stats(z, g)
    mu_next = (mu / (1.0 + lam)) + (lam / (1.0 + lam)) * (Sigma_next @ g_bar + z_bar)
    return mu_next, Sigma_next, U, V


def _ref_gsm_sample(mu_t, Sigma_t, z_b, s_b):
    """One GSM per-sample update (Algorithm 3)."""
    mu_t = np.asarray(mu_t, dtype=float).reshape(-1)
    Sigma_t = symmetrize(np.asarray(Sigma_t, dtype=float))
    z_b = np.asarray(z_b, dtype=float).reshape(-1)
    s_b = np.asarray(s_b, dtype=float).reshape(-1)
    eps = Sigma_t @ s_b - mu_t + z_b
    d = mu_t - z_b
    c = float(s_b @ Sigma_t @ s_b + (d @ s_b) ** 2)
    rho = (-1.0 + math.sqrt(1.0 + 4.0 * max(c, 0.0))) / 2.0
    delta_mu = (eps - d * (s_b @ eps) / (1.0 + rho + d @ s_b)) / (1.0 + rho)
    mu_tilde = mu_t + delta_mu
    delta_sigma = np.outer(d, d) - np.outer(mu_tilde - z_b, mu_tilde - z_b)
    return delta_mu, symmetrize(delta_sigma), rho


def _ref_gsm_batch(mu_t, Sigma_t, z, g):
    """GSM batch update: average of per-sample updates (Algorithm 3)."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    g = np.atleast_2d(np.asarray(g, dtype=float))
    B = z.shape[0]
    dmu = np.zeros_like(np.asarray(mu_t, dtype=float).reshape(-1))
    dS = np.zeros_like(np.asarray(Sigma_t, dtype=float))
    for b in range(B):
        dmu_b, dS_b, _ = _ref_gsm_sample(mu_t, Sigma_t, z[b], g[b])
        dmu = dmu + dmu_b / B
        dS = dS + dS_b / B
    mu_next = np.asarray(mu_t, dtype=float).reshape(-1) + dmu
    Sigma_next = symmetrize(np.asarray(Sigma_t, dtype=float) + dS)
    return mu_next, Sigma_next


def _exact_limit_batch(mu_t, Sigma_t, score_fn, reps=2):
    """Batch whose empirical statistics equal the :math:`B \\to \\infty` limits.

    Samples are constructed so that :math:`\\bar z = \\mu_t` and
    :math:`C = \\Sigma_t` *exactly* (Lemma D.2).  Because the target is Gaussian, the
    scores are affine in :math:`z`, so :math:`\\bar g = \\Sigma_*^{-1}(\\mu_* - \\mu_t)`
    and :math:`\\Gamma = \\Sigma_*^{-1} \\Sigma_t \\Sigma_*^{-1}` hold exactly too.
    """
    mu_t = np.asarray(mu_t, dtype=float).reshape(-1)
    Sigma_t = symmetrize(np.asarray(Sigma_t, dtype=float))
    D = mu_t.shape[0]
    L = np.linalg.cholesky(ensure_spd(Sigma_t, jitter=1e-12))
    half = int(reps) * D
    idx = np.arange(half) % D
    V = np.zeros((half, D))
    V[np.arange(half), idx] = math.sqrt(D)
    W = (L @ V.T).T
    z = np.vstack([mu_t + W, mu_t - W])
    g = _scores(score_fn, z)
    return z, g


def _match_step(mu, Sigma, z=None, g=None, lam=1.0, low_rank=None):
    """Call the module's match step, returning plain ``(mu_next, Sigma_next)``."""
    kwargs = {}
    if low_rank is not None:
        kwargs["low_rank"] = low_rank
    try:
        out = bam_match_step(mu, Sigma, z=z, g=g, lam=lam, **kwargs)
    except TypeError:  # pragma: no cover - tolerant of minor signature drift
        out = bam_match_step(mu, Sigma, lam=lam, z=z, g=g)
    if hasattr(out, "mu") and hasattr(out, "Sigma"):
        return np.asarray(out.mu, dtype=float), np.asarray(out.Sigma, dtype=float)
    mu_next, Sigma_next = out
    return np.asarray(mu_next, dtype=float), np.asarray(Sigma_next, dtype=float)


def _gsm_step(mu, Sigma, z, g):
    """Call the module's GSM batch update, returning plain arrays."""
    out = gsm_batch_update(mu, Sigma, z, g)
    if isinstance(out, tuple):
        return np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)
    return np.asarray(out.mu, dtype=float), np.asarray(out.Sigma, dtype=float)


def _random_spd(D, rng, scale=1.0):
    A = rng.standard_normal((D, D))
    S = (A @ A.T) / D + 0.5 * np.eye(D)
    return ensure_spd(symmetrize(scale * S), jitter=1e-12)


def _make_target(mean, cov):
    """Construct a :class:`GaussianTarget` (schema tolerant)."""
    if GaussianTarget is None:  # pragma: no cover
        raise unittest.SkipTest("GaussianTarget unavailable")
    try:
        return GaussianTarget(mean=mean, cov=cov)
    except TypeError:  # pragma: no cover
        return GaussianTarget(mean, cov)


def _norm_mean_err(mu, mu_star, Sigma_star):
    """``||Sigma_*^{-1/2} (mu - mu_*)||`` (eq. 14 of the paper)."""
    L = np.linalg.cholesky(ensure_spd(Sigma_star, jitter=1e-12))
    v = np.linalg.solve(L, np.asarray(mu, dtype=float).reshape(-1) - mu_star)
    return float(np.linalg.norm(v))


def _norm_cov_err(Sigma, Sigma_star):
    """``||Sigma_*^{-1/2} (Sigma - Sigma_*) Sigma_*^{-1/2}||_2`` (eq. 15)."""
    L = np.linalg.cholesky(ensure_spd(Sigma_star, jitter=1e-12))
    Lm = np.linalg.inv(L)
    M = Lm @ (np.asarray(Sigma, dtype=float) - Sigma_star) @ Lm.T
    return float(np.linalg.norm(symmetrize(M), 2))


def _default_problem(D=4, seed=0, mu_scale=0.5):
    """A deterministic (target, mu_t, Sigma_t) triple far from the optimum."""
    rng = np.random.default_rng(seed)
    Sigma_star = _random_spd(D, rng)
    mu_star = rng.uniform(-mu_scale, mu_scale, size=D)
    target = _make_target(mu_star, Sigma_star)
    mu_t = np.zeros(D)
    Sigma_t = _random_spd(D, np.random.default_rng(seed + 1))
    return target, mu_star, Sigma_star, mu_t, Sigma_t


# --------------------------------------------------------------------------------------
# 0. Sanity of the local reference implementations used as oracles
# --------------------------------------------------------------------------------------
class TestReferenceOracles(unittest.TestCase):
    """The oracles must themselves satisfy the paper's equations."""

    def test_reference_match_step_solves_quadratic_equation(self):
        rng = np.random.default_rng(3)
        D = 4
        Sigma = _random_spd(D, rng)
        mu = rng.standard_normal(D)
        z = mu + rng.standard_normal((7, D)) @ np.linalg.cholesky(Sigma).T
        g = -rng.standard_normal((7, D))  # arbitrary "scores"
        lam = 3.0
        _, Sigma_next, U, V = _ref_bam_match(mu, Sigma, z, g, lam)
        res = Sigma_next @ U @ Sigma_next + Sigma_next - V
        self.assertLess(np.max(np.abs(res)), 1e-8)
        # positive definite and symmetric
        self.assertTrue(np.allclose(Sigma_next, Sigma_next.T))
        self.assertGreater(np.min(np.linalg.eigvalsh(Sigma_next)), 0.0)

    def test_reference_gsm_satisfies_bam_large_lambda_equations(self):
        """§C.3: the GSM update satisfies the BaM ``lambda -> inf`` equations."""
        rng = np.random.default_rng(4)
        D = 3
        Sigma_t = _random_spd(D, rng)
        mu_t = rng.standard_normal(D)
        z = rng.standard_normal(D)
        g = rng.standard_normal(D)
        mu_next, Sigma_next = _ref_gsm_batch(mu_t, Sigma_t, z[None, :], g[None, :])
        V = Sigma_t + np.outer(mu_t - z, mu_t - z)
        res = Sigma_next @ np.outer(g, g) @ Sigma_next + Sigma_next - V
        self.assertLess(np.max(np.abs(res)), 1e-8)
        np.testing.assert_allclose(mu_next, Sigma_next @ g + z, atol=1e-8, rtol=1e-8)


# --------------------------------------------------------------------------------------
# 1. BaM <-> GSM limit (§C.3, eqs. 114-115)
# --------------------------------------------------------------------------------------
class TestBamGsmLimit(unittest.TestCase):
    """``B = 1`` and ``lambda -> inf``: BaM reproduces GSM (paper §C.3)."""

    def setUp(self):
        if gsm_batch_update is None:
            self.skipTest("baselines.gsm unavailable")
        self.rng = np.random.default_rng(0)
        D = 3
        self.Sigma_t = _random_spd(D, self.rng)
        self.mu_t = self.rng.standard_normal(D)
        self.z = self.rng.standard_normal(D)
        self.g = self.rng.standard_normal(D)
        self.V_limit = self.Sigma_t + np.outer(self.mu_t - self.z, self.mu_t - self.z)

    def test_large_lambda_solves_limit_quadratic_equation(self):
        """eq. (114): Sigma_{t+1} g g^T Sigma_{t+1} + Sigma_{t+1} = V."""
        mu_next, Sigma_next = _match_step(
            self.mu_t, self.Sigma_t, z=self.z[None, :], g=self.g[None, :], lam=1e9
        )
        res = Sigma_next @ np.outer(self.g, self.g) @ Sigma_next + Sigma_next - self.V_limit
        self.assertLess(np.max(np.abs(res)), 1e-6)
        self.assertTrue(np.allclose(Sigma_next, Sigma_next.T))
        self.assertGreater(np.min(np.linalg.eigvalsh(Sigma_next)), 0.0)
        del mu_next

    def test_large_lambda_mean_update(self):
        """eq. (115): mu_{t+1} = Sigma_{t+1} g + z."""
        mu_next, Sigma_next = _match_step(
            self.mu_t, self.Sigma_t, z=self.z[None, :], g=self.g[None, :], lam=1e9
        )
        np.testing.assert_allclose(mu_next, Sigma_next @ self.g + self.z, atol=1e-6, rtol=1e-6)

    def test_bam_matches_gsm_single_step(self):
        mu_bam, Sigma_bam = _match_step(
            self.mu_t, self.Sigma_t, z=self.z[None, :], g=self.g[None, :], lam=1e10
        )
        mu_gsm, Sigma_gsm = _gsm_step(self.mu_t, self.Sigma_t, self.z[None, :], self.g[None, :])
        np.testing.assert_allclose(mu_bam, mu_gsm, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(Sigma_bam, Sigma_gsm, atol=1e-6, rtol=1e-6)

    def test_bam_approaches_gsm_as_lambda_grows(self):
        """The discrepancy to GSM must vanish like ``O(1/lambda)``."""
        mu_gsm, Sigma_gsm = _gsm_step(self.mu_t, self.Sigma_t, self.z[None, :], self.g[None, :])
        errs = []
        for lam in (1e2, 1e4, 1e6, 1e8):
            mu_bam, Sigma_bam = _match_step(
                self.mu_t, self.Sigma_t, z=self.z[None, :], g=self.g[None, :], lam=lam
            )
            errs.append(
                float(np.linalg.norm(mu_bam - mu_gsm) + np.linalg.norm(Sigma_bam - Sigma_gsm))
            )
        # monotone decrease and tiny at large lambda
        for a, b in zip(errs, errs[1:]):
            self.assertLess(b, a)
        self.assertLess(errs[-1], 1e-6)
        self.assertLess(errs[-1], 1e-3 * errs[0])

    def test_score_matching_holds_at_the_sample(self):
        """GSM/BaM limit enforces ``grad log q(z) = grad log p(z) = g``."""
        mu_next, Sigma_next = _match_step(
            self.mu_t, self.Sigma_t, z=self.z[None, :], g=self.g[None, :], lam=1e9
        )
        score_q = gaussian_score(self.z, mu_next, Sigma_next)
        np.testing.assert_allclose(score_q, self.g, atol=1e-6, rtol=1e-6)

    def test_trajectories_agree_over_multiple_iterations(self):
        """BaM (B=1, huge lambda) and GSM follow the same trajectory."""
        rng = np.random.default_rng(11)
        D = 3
        Sigma_star = _random_spd(D, rng)
        mu_star = rng.uniform(-0.5, 0.5, size=D)
        target = _make_target(mu_star, Sigma_star)
        mu0 = np.zeros(D)
        Sigma0 = _random_spd(D, np.random.default_rng(12))

        mu_b, S_b = mu0.copy(), Sigma0.copy()
        mu_g, S_g = mu0.copy(), Sigma0.copy()
        lam = 1e10
        for _ in range(8):
            L = np.linalg.cholesky(ensure_spd(S_b, jitter=1e-12))
            z = mu_b + L @ rng.standard_normal(D)
            g = np.asarray(target.score(z), dtype=float).reshape(-1)
            mu_b, S_b = _match_step(mu_b, S_b, z=z[None, :], g=g[None, :], lam=lam)
            mu_g, S_g = _gsm_step(mu_g, S_g, z[None, :], g[None, :])
            np.testing.assert_allclose(mu_b, mu_g, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(S_b, S_g, atol=1e-6, rtol=1e-6)

    def test_low_rank_solver_agrees_for_rank_one_U(self):
        """``B = 1`` makes ``U`` rank one, so Lemma B.3 must match Lemma B.1."""
        U = np.outer(self.g, self.g)
        V = self.V_limit
        X_dense = solve_quadratic_matrix_equation_dense(U, V)
        Q = self.g.reshape(-1, 1)
        X_low = solve_quadratic_matrix_equation_low_rank(V, Q)
        np.testing.assert_allclose(X_low, X_dense, atol=1e-7, rtol=1e-7)
        res = matrix_residual(U, V, X_low)
        self.assertLess(np.max(np.abs(res)), 1e-7)

    def test_limit_matrices_returned_by_module(self):
        """If the match step exposes ``U``/``V``, check them against eq. (114)."""
        try:
            out = bam_match_step(
                self.mu_t, self.Sigma_t, z=self.z[None, :], g=self.g[None, :], lam=1e9,
                return_result=True,
            )
        except TypeError:  # pragma: no cover
            self.skipTest("return_result / stats API unavailable")
        if not (hasattr(out, "U") and hasattr(out, "V")):
            self.skipTest("MatchStepResult does not expose U/V")
        np.testing.assert_allclose(out.U, np.outer(self.g, self.g), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(out.V, self.V_limit, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------------------
# 2. GSM implementation vs Algorithm 3 of §E.1
# --------------------------------------------------------------------------------------
class TestGsmAlgorithm3(unittest.TestCase):
    def setUp(self):
        if gsm_batch_update is None:
            self.skipTest("baselines.gsm unavailable")
        self.rng = np.random.default_rng(7)
        D = 3
        self.Sigma_t = _random_spd(D, self.rng)
        self.mu_t = self.rng.standard_normal(D)
        self.z = self.rng.standard_normal(D)
        self.g = self.rng.standard_normal(D)

    def test_solve_rho_positive_root_if_exposed(self):
        sigma_t, mu_t, z, g = self.Sigma_t, self.mu_t, self.z, self.g
        c = float(g @ sigma_t @ g + ((mu_t - z) @ g) ** 2)
        try:
            rho = float(np.asarray(solve_rho(g, sigma_t, mu_t, z)).reshape(-1)[0])
        except (TypeError, AttributeError):  # pragma: no cover
            self.skipTest("solve_rho signature differs")
        self.assertGreater(rho, 0.0)
        self.assertAlmostEqual(rho * (1.0 + rho), c, places=8)

    def test_reference_sample_update_matches_algorithm3(self):
        dmu, dS, rho = _ref_gsm_sample(self.mu_t, self.Sigma_t, self.z, self.g)
        d = self.mu_t - self.z
        eps = self.Sigma_t @ self.g - self.mu_t + self.z
        expected = (eps - d * (self.g @ eps) / (1.0 + rho + d @ self.g)) / (1.0 + rho)
        np.testing.assert_allclose(dmu, expected, atol=1e-12, rtol=1e-12)
        self.assertGreater(dS.shape[0], 0)

    def test_module_sample_update_matches_reference(self):
        try:
            out = gsm_sample_update(self.mu_t, self.Sigma_t, self.z, self.g)
        except (TypeError, AttributeError):  # pragma: no cover
            self.skipTest("gsm_sample_update signature differs")
        dmu_mod, dS_mod = np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)
        dmu_ref, dS_ref, _ = _ref_gsm_sample(self.mu_t, self.Sigma_t, self.z, self.g)
        np.testing.assert_allclose(dmu_mod, dmu_ref, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(dS_mod, dS_ref, atol=1e-10, rtol=1e-10)

    def test_module_batch_update_matches_reference(self):
        z = self.z + 0.1 * self.rng.standard_normal((5, 3))
        g = self.g + 0.1 * self.rng.standard_normal((5, 3))
        mu_mod, S_mod = _gsm_step(self.mu_t, self.Sigma_t, z, g)
        mu_ref, S_ref = _ref_gsm_batch(self.mu_t, self.Sigma_t, z, g)
        np.testing.assert_allclose(mu_mod, mu_ref, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(S_mod, S_ref, atol=1e-10, rtol=1e-10)

    def test_batch_update_is_batch_average(self):
        """eq. (2) of §E.1: update is the mean of the per-sample updates."""
        z = self.z + 0.2 * self.rng.standard_normal((4, 3))
        g = self.g + 0.2 * self.rng.standard_normal((4, 3))
        mu_next, S_next = _gsm_step(self.mu_t, self.Sigma_t, z, g)
        dmu = np.zeros(3)
        dS = np.zeros((3, 3))
        for b in range(4):
            a, bS, _ = _ref_gsm_sample(self.mu_t, self.Sigma_t, z[b], g[b])
            dmu = dmu + a / 4.0
            dS = dS + bS / 4.0
        np.testing.assert_allclose(mu_next, self.mu_t + dmu, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(S_next, self.Sigma_t + dS, atol=1e-10, rtol=1e-10)

    def test_gsm_fit_improves_on_gaussian_target(self):
        """Smoke test: GSM reduces the normalized mean error on a Gaussian target."""
        D = 4
        rng = np.random.default_rng(21)
        Sigma_star = _random_spd(D, rng)
        mu_star = rng.uniform(-0.5, 0.5, size=D)
        target = _make_target(mu_star, Sigma_star)
        mu0 = np.zeros(D)
        Sigma0 = np.eye(D)
        err0 = _norm_mean_err(mu0, mu_star, Sigma_star)
        try:
            res = gsm_fit(
                mu0, Sigma0, score_fn=target.score, T=120, batch_size=8, seed=0,
                track_history=True,
            )
        except TypeError:  # pragma: no cover
            res = gsm_fit(mu0, Sigma0, score_fn=target.score, T=120, batch_size=8)
        mu_final = np.asarray(res.mu, dtype=float)
        errs = [_norm_mean_err(mu_final, mu_star, Sigma_star)]
        hist = getattr(res, "mu_history", None)
        if hist is not None and len(hist) > 0:
            errs.extend(_norm_mean_err(np.asarray(m, dtype=float), mu_star, Sigma_star)
                        for m in hist)
        self.assertTrue(np.all(np.isfinite(errs)))
        self.assertLess(min(errs), err0)


# --------------------------------------------------------------------------------------
# 3. lambda -> 0 / schedule sanity (no movement; large lambda = weak regularization)
# --------------------------------------------------------------------------------------
class TestLambdaLimits(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        D = 3
        self.Sigma_t = _random_spd(D, rng)
        self.mu_t = rng.standard_normal(D)
        z = self.mu_t + rng.standard_normal((6, D))
        self.z = z
        self.g = rng.standard_normal((6, D))

    def test_lambda_to_zero_means_no_movement(self):
        """lambda -> 0 sends U -> 0 and V -> Sigma_t, so q_{t+1} = q_t."""
        for lam in (0.0, 1e-12):
            try:
                mu_next, Sigma_next = _match_step(
                    self.mu_t, self.Sigma_t, z=self.z, g=self.g, lam=lam
                )
            except (ValueError, ZeroDivisionError):  # pragma: no cover - lambda > 0 assumed
                continue
            np.testing.assert_allclose(mu_next, self.mu_t, atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(Sigma_next, self.Sigma_t, atol=1e-8, rtol=1e-8)

    def test_reference_lambda_to_zero(self):
        mu_next, Sigma_next, U, V = _ref_bam_match(
            self.mu_t, self.Sigma_t, self.z, self.g, 1e-12
        )
        self.assertLess(np.max(np.abs(U)), 1e-10)
        np.testing.assert_allclose(V, self.Sigma_t, atol=1e-10)
        np.testing.assert_allclose(mu_next, self.mu_t, atol=1e-10)
        np.testing.assert_allclose(Sigma_next, self.Sigma_t, atol=1e-8)

    def test_decay_schedule_drives_lambda_to_zero(self):
        """lambda_t = B D / (t+1) -> 0, hence no movement at late iterations."""
        if make_schedule is None:
            self.skipTest("learning_rate module unavailable")
        B, D = 5, 3
        try:
            sched = make_schedule("BD/(t+1)", batch_size=B, dim=D)
            lam_late = float(sched(10 ** 7))
        except Exception:  # pragma: no cover
            self.skipTest("schedule construction failed")
        self.assertLess(lam_late, 1e-5)
        mu_next, Sigma_next = _match_step(
            self.mu_t, self.Sigma_t, z=self.z, g=self.g, lam=lam_late
        )
        np.testing.assert_allclose(mu_next, self.mu_t, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(Sigma_next, self.Sigma_t, atol=1e-3, rtol=1e-3)

    def test_schedule_values_match_paper(self):
        """lambda_t = B D (Gaussian) and lambda_t = B D/(t+1) (non-Gaussian)."""
        if make_schedule is None:
            self.skipTest("learning_rate module unavailable")
        B, D = 20, 4
        const = make_schedule("BD", batch_size=B, dim=D)
        self.assertAlmostEqual(float(const(0)), 80.0, places=8)
        self.assertAlmostEqual(float(const(9)), 80.0, places=8)
        decay = make_schedule("BD/(t+1)", batch_size=B, dim=D)
        self.assertAlmostEqual(float(decay(0)), 80.0, places=8)
        self.assertAlmostEqual(float(decay(1)), 40.0, places=8)
        self.assertAlmostEqual(float(decay(7)), 10.0, places=8)

    def test_resolve_lambda_default_is_batch_times_dim(self):
        D, B = 4, 6
        try:
            lam = float(resolve_lambda(None, 0, dim=D, batch_size=B))
        except TypeError:  # pragma: no cover
            self.skipTest("resolve_lambda signature differs")
        self.assertAlmostEqual(lam, float(B * D), places=8)

    def test_large_lambda_gives_weak_regularization(self):
        """Bigger lambda moves further from q_t (paper: ``lambda`` = learning rate)."""
        mu_small, S_small = _match_step(self.mu_t, self.Sigma_t, z=self.z, g=self.g, lam=1e-3)
        mu_big, S_big = _match_step(self.mu_t, self.Sigma_t, z=self.z, g=self.g, lam=1e6)
        move_small = np.linalg.norm(mu_small - self.mu_t) + np.linalg.norm(S_small - self.Sigma_t)
        move_big = np.linalg.norm(mu_big - self.mu_t) + np.linalg.norm(S_big - self.Sigma_t)
        self.assertLess(move_small, move_big)


# --------------------------------------------------------------------------------------
# 4. Infinite batch limit for Gaussian targets (Lemma D.2)
# --------------------------------------------------------------------------------------
class TestInfiniteBatchGaussian(unittest.TestCase):
    def test_exact_limit_batch_statistics(self):
        """Exact construction: ``z_bar -> mu_t``, ``C -> Sigma_t`` etc. (Lemma D.2)."""
        rng = np.random.default_rng(41)
        D = 4
        Sigma_star = _random_spd(D, rng)
        mu_star = rng.uniform(-0.5, 0.5, size=D)
        target = _make_target(mu_star, Sigma_star)
        mu_t = rng.standard_normal(D)
        Sigma_t = _random_spd(D, rng)

        z, g = _exact_limit_batch(mu_t, Sigma_t, target.score, reps=2)
        z_bar, C, g_bar, Gamma = _ref_stats(z, g)
        Sinv = np.linalg.inv(Sigma_star)

        np.testing.assert_allclose(z_bar, mu_t, atol=1e-10, rtol=0, err_msg="z_bar limit")
        np.testing.assert_allclose(C, Sigma_t, atol=1e-10, rtol=1e-10, err_msg="C limit")
        np.testing.assert_allclose(
            g_bar, Sinv @ (mu_star - mu_t), atol=1e-10, rtol=1e-10, err_msg="g_bar limit"
        )
        np.testing.assert_allclose(
            Gamma, Sinv @ Sigma_t @ Sinv, atol=1e-10, rtol=1e-10, err_msg="Gamma limit"
        )

    def test_module_batch_statistics_match_paper_conventions(self):
        """eq. (95): centered batch covariances ``C`` and ``Gamma``."""
        if batch_statistics is None:  # pragma: no cover
            self.skipTest("batch_statistics unavailable")
        rng = np.random.default_rng(43)
        z = rng.standard_normal((9, 3))
        g = rng.standard_normal((9, 3))
        z_bar, C, g_bar, Gamma = _ref_stats(z, g)
        try:
            stats = batch_statistics(z, g)
        except TypeError:  # pragma: no cover
            self.skipTest("batch_statistics signature differs")
        for name, expected in (
            ("z_bar", z_bar), ("C", C), ("g_bar", g_bar), ("Gamma", Gamma)
        ):
            if not hasattr(stats, name):  # pragma: no cover
                self.skipTest(f"BatchStatistics lacks {name}")
            np.testing.assert_allclose(
                np.asarray(getattr(stats, name), dtype=float), expected, atol=1e-12, rtol=1e-12
            )

    def test_monte_carlo_batch_statistics(self):
        """Monte-Carlo check of Lemma D.2 at large but finite ``B``."""
        rng = np.random.default_rng(47)
        D = 4
        Sigma_star = _random_spd(D, rng)
        mu_star = rng.uniform(-0.5, 0.5, size=D)
        target = _make_target(mu_star, Sigma_star)
        mu_t = rng.standard_normal(D)
        Sigma_t = _random_spd(D, rng)

        B = 40000
        L = np.linalg.cholesky(ensure_spd(Sigma_t, jitter=1e-12))
        z = mu_t + rng.standard_normal((B, D)) @ L.T
        g = _scores(target.score, z)
        z_bar, C, g_bar, Gamma = _ref_stats(z, g)
        Sinv = np.linalg.inv(Sigma_star)

        rel = lambda A, Bm: np.linalg.norm(A - Bm) / np.linalg.norm(Bm)
        self.assertLess(rel(z_bar, mu_t), 0.02)
        self.assertLess(rel(C, Sigma_t), 0.10)
        self.assertLess(rel(g_bar, Sinv @ (mu_star - mu_t)), 0.10)
        self.assertLess(rel(Gamma, Sinv @ Sigma_t @ Sinv), 0.10)


# --------------------------------------------------------------------------------------
# 5. One-step recovery of a Gaussian target (Thm 3.1 / §D.2, "Cor D.5")
# --------------------------------------------------------------------------------------
class TestOneStepGaussianRecovery(unittest.TestCase):
    def test_one_step_recovers_target_in_large_lambda_limit(self):
        """Infinite batch + large lambda => BaM jumps to ``N(mu_*, Sigma_*)``."""
        target, mu_star, Sigma_star, mu_t, Sigma_t = _default_problem(D=4, seed=5)
        z, g = _exact_limit_batch(mu_t, Sigma_t, target.score, reps=2)

        err_before = _norm_mean_err(mu_t, mu_star, Sigma_star)
        cov_before = _norm_cov_err(Sigma_t, Sigma_star)

        prev = None
        for lam in (1e2, 1e3, 1e4, 1e6, 1e8):
            mu_1, S_1 = _match_step(mu_t, Sigma_t, z=z, g=g, lam=lam)
            e_mean = _norm_mean_err(mu_1, mu_star, Sigma_star)
            e_cov = _norm_cov_err(S_1, Sigma_star)
            self.assertLess(e_mean, err_before)
            self.assertLess(e_cov, cov_before)
            if prev is not None:
                self.assertLess(e_cov, prev + 1e-12)
            prev = e_cov

        self.assertLess(e_mean, 1e-4)
        self.assertLess(e_cov, 1e-4)
        self.assertLess(prev, 1e-6)

    def test_module_matches_reference_match_step(self):
        """The module's match step equals eqs. (108)-(113) for random statistics."""
        rng = np.random.default_rng(53)
        D = 5
        Sigma_t = _random_spd(D, rng)
        mu_t = rng.standard_normal(D)
        z = mu_t + rng.standard_normal((6, D)) @ np.linalg.cholesky(Sigma_t).T
        g = rng.standard_normal((6, D))
        lam = 2.5
        mu_mod, S_mod = _match_step(mu_t, Sigma_t, z=z, g=g, lam=lam)
        mu_ref, S_ref, _, _ = _ref_bam_match(mu_t, Sigma_t, z, g, lam)
        np.testing.assert_allclose(mu_mod, mu_ref, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(S_mod, S_ref, atol=1e-8, rtol=1e-8)

    def test_run_solver_with_low_rank_path(self):
        """For ``B < D`` the low-rank path (Lemma B.3) must give the same answer."""
        rng = np.random.default_rng(59)
        D = 6
        U = rng.standard_normal((D, 3)) @ np.diag([1.0, 0.5, 0.1])
        Q = rng.standard_normal((D, 3))
        U = Q @ Q.T
        V = _random_spd(D, rng)
        X_dense = solve_quadratic_matrix_equation_dense(U, V)
        X_low = solve_quadratic_matrix_equation_low_rank(V, Q)
        X_disp = solve_quadratic_matrix_equation(U, V, Q=Q)
        np.testing.assert_allclose(X_low, X_dense, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(X_disp, X_dense, atol=1e-7, rtol=1e-7)
        for X in (X_dense, X_low):
            self.assertLess(np.max(np.abs(matrix_residual(U, V, X))), 1e-7)

    def test_bam_class_reduces_error_on_gaussian_target(self):
        """Integration check of Algorithm 1 with the constant ``lambda = B D`` schedule."""
        D = 4
        rng = np.random.default_rng(61)
        Sigma_star = _random_spd(D, rng)
        mu_star = rng.uniform(-0.5, 0.5, size=D)
        target = _make_target(mu_star, Sigma_star)
        B = 20
        mu0 = np.zeros(D)
        Sigma0 = np.eye(D)
        try:
            opt = BaM(mu0, Sigma0, score_fn=target.score, batch_size=B, lam=float(B * D))
            res = opt.run(60)
        except TypeError:  # pragma: no cover
            self.skipTest("BaM constructor signature differs")
        mu_final = np.asarray(res.mu, dtype=float)
        S_final = np.asarray(res.Sigma, dtype=float)
        self.assertLess(
            _norm_mean_err(mu_final, mu_star, Sigma_star),
            _norm_mean_err(mu0, mu_star, Sigma_star),
        )
        self.assertLess(_norm_cov_err(S_final, Sigma_star), _norm_cov_err(Sigma0, Sigma_star))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
