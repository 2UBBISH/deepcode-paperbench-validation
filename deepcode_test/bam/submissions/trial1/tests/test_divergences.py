"""Unit tests for score-based divergences and Gaussian specializations.

These tests cover the divergence-related portions of the reproduction plan:
    - Gaussian-vs-Gaussian closed form for the score-based divergence
    - Agreement between empirical (sample) estimates and closed forms
    - Batch-statistics expansion matching the sample estimate
    - KL and reverse KL Gaussian formulas
    - Fisher divergence closed form and empirical estimator
    - Generic score-based divergence definition
    - Gaussian tilt mean
    - Affine invariance and non-negativity of the score divergence
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from bam.batch_stats import compute_batch_stats
from bam.divergences import (
    fisher_divergence_estimate,
    fisher_divergence_gaussian_closed_form,
    kl_gaussian,
    reverse_kl_gaussian,
    score_divergence_batch_form,
    score_divergence_gaussian,
    score_divergence_gaussian_closed_form,
    score_divergence_general,
    symmetrize,
    tilt_gaussian_mean,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sym(a: jnp.ndarray) -> jnp.ndarray:
    """Symmetrize a square matrix."""
    return (a + a.T) / 2.0


def _random_spd(key: jax.Array, n: int, jitter: float = 1e-6) -> jnp.ndarray:
    """Create a random symmetric positive definite matrix."""
    a = jax.random.normal(key, (n, n))
    return _sym(a @ a.T / n + jitter * jnp.eye(n))


def _gaussian_score(mu: jnp.ndarray, sigma: jnp.ndarray):
    """Closed-form vectorized Gaussian score function s(z) = grad_z log p(z)."""
    sigma_inv = jnp.linalg.inv(sigma)

    def score(z: jnp.ndarray) -> jnp.ndarray:
        return -(z - mu) @ sigma_inv

    return score


def _sample_mvn(key: jax.Array, mu: jnp.ndarray, sigma: jnp.ndarray, n: int) -> jnp.ndarray:
    """Draw n samples from a multivariate normal using Cholesky sampling."""
    chol = jnp.linalg.cholesky(sigma + 1e-10 * jnp.eye(mu.shape[0]))
    eps = jax.random.normal(key, (n, mu.shape[0]))
    return mu + eps @ chol.T


def _score_divergence_closed_form(mu_q, sigma_q, mu_p, sigma_p) -> jnp.ndarray:
    """Independent transcription of the Gaussian-vs-Gaussian score divergence.

    For q = N(nu, Psi) and p = N(mu, Sigma):
        D(q;p) = tr[(I - Psi Sigma^{-1})^2]
                 + (nu - mu)^T Sigma^{-1} Psi Sigma^{-1} (nu - mu).
    """
    sigma_p_inv = jnp.linalg.inv(sigma_p)
    mat = jnp.eye(mu_q.shape[0]) - sigma_q @ sigma_p_inv
    trace_term = jnp.trace(mat @ mat)
    diff = mu_q - mu_p
    mean_term = diff @ sigma_p_inv @ sigma_q @ sigma_p_inv @ diff
    return trace_term + mean_term


def _kl_gaussian_formula(mu0, sigma0, mu1, sigma1) -> jnp.ndarray:
    """KL(N(mu0, sigma0) || N(mu1, sigma1))."""
    d = mu0.shape[0]
    sigma1_inv = jnp.linalg.inv(sigma1)
    diff = mu1 - mu0
    return 0.5 * (
        jnp.trace(sigma1_inv @ sigma0)
        + diff @ sigma1_inv @ diff
        - d
        + jnp.log(jnp.linalg.det(sigma1))
        - jnp.log(jnp.linalg.det(sigma0))
    )


def _fisher_gaussian_closed_form(mu_q, sigma_q, mu_p, sigma_p) -> jnp.ndarray:
    """Independent Fisher-divergence formula for Gaussian q and p.

    Let q = N(nu, Psi), p = N(mu, Sigma), and write
        grad log q(z) - grad log p(z) = A z + b,
    with
        A = Psi^{-1} - Sigma^{-1},
        b = Sigma^{-1} mu - Psi^{-1} nu.
    Then Fisher = E_q[||A z + b||^2]
                = tr(A Psi A^T) + (A nu + b)^T (A nu + b).
    """
    psi_inv = jnp.linalg.inv(sigma_q)
    sigma_inv = jnp.linalg.inv(sigma_p)
    a = psi_inv - sigma_inv
    b = sigma_inv @ mu_p - psi_inv @ mu_q
    trace_term = jnp.trace(a @ sigma_q @ a.T)
    center = a @ mu_q + b
    mean_term = center @ center
    return trace_term + mean_term


# ---------------------------------------------------------------------------
# Score-based divergence
# ---------------------------------------------------------------------------
def test_score_divergence_gaussian_closed_form_matches_formula():
    key = jax.random.PRNGKey(0)
    d = 4
    k1, k2 = jax.random.split(key)
    mu_q = jax.random.normal(k1, (d,))
    sigma_q = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu_p = jax.random.normal(k1, (d,))
    sigma_p = _random_spd(k2, d)

    expected = _score_divergence_closed_form(mu_q, sigma_q, mu_p, sigma_p)
    got = score_divergence_gaussian_closed_form(mu_q, sigma_q, mu_p, sigma_p)

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_score_divergence_gaussian_matches_closed_form_large_batch():
    key = jax.random.PRNGKey(1)
    d = 3
    k1, k2 = jax.random.split(key)
    mu_q = jax.random.normal(k1, (d,))
    sigma_q = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu_p = jax.random.normal(k1, (d,))
    sigma_p = _random_spd(k2, d)

    k_sample, _ = jax.random.split(k2)
    n = 60_000
    z = _sample_mvn(k_sample, mu_q, sigma_q, n)
    score_p = _gaussian_score(mu_p, sigma_p)

    empirical = score_divergence_gaussian(mu_q, sigma_q, score_p, z)
    closed = score_divergence_gaussian_closed_form(mu_q, sigma_q, mu_p, sigma_p)

    # Monte-Carlo error is small but not zero at this sample size.
    np.testing.assert_allclose(empirical, closed, rtol=0.08, atol=0.05)


def test_score_divergence_batch_form_matches_sample_form():
    key = jax.random.PRNGKey(2)
    d = 4
    k1, k2 = jax.random.split(key)
    mu_q = jax.random.normal(k1, (d,))
    sigma_q = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu_p = jax.random.normal(k1, (d,))
    sigma_p = _random_spd(k2, d)

    n = 128
    k_sample, _ = jax.random.split(k2)
    z = _sample_mvn(k_sample, mu_q, sigma_q, n)
    score_p = _gaussian_score(mu_p, sigma_p)
    g = score_p(z)

    stats = compute_batch_stats(z, g)

    from_batch = score_divergence_batch_form(
        mu_q,
        sigma_q,
        stats.zbar,
        stats.gbar,
        stats.C,
        stats.Gamma,
        z=z,
        g=g,
    )
    from_samples = score_divergence_gaussian(mu_q, sigma_q, score_p, z)

    np.testing.assert_allclose(from_batch, from_samples, rtol=1e-9, atol=1e-9)


def test_score_divergence_general_matches_gaussian_specialization():
    key = jax.random.PRNGKey(3)
    d = 4
    k1, k2 = jax.random.split(key)
    mu_q = jax.random.normal(k1, (d,))
    sigma_q = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu_p = jax.random.normal(k1, (d,))
    sigma_p = _random_spd(k2, d)

    n = 256
    k_sample, _ = jax.random.split(k2)
    z = _sample_mvn(k_sample, mu_q, sigma_q, n)

    sigma_q_inv = jnp.linalg.inv(sigma_q)
    scores_q = -(z - mu_q) @ sigma_q_inv
    scores_p = _gaussian_score(mu_p, sigma_p)(z)
    gamma_q = sigma_q_inv  # score covariance for a Gaussian

    general = score_divergence_general(scores_q, scores_p, gamma_q)
    specialized = score_divergence_gaussian(mu_q, sigma_q, _gaussian_score(mu_p, sigma_p), z)

    np.testing.assert_allclose(general, specialized, rtol=1e-9, atol=1e-9)


def test_score_divergence_is_zero_for_matching_gaussians_and_positive_otherwise():
    key = jax.random.PRNGKey(4)
    d = 4
    k1, k2 = jax.random.split(key)
    mu = jax.random.normal(k1, (d,))
    sigma = _random_spd(k2, d)

    zero = score_divergence_gaussian_closed_form(mu, sigma, mu, sigma)
    np.testing.assert_allclose(zero, 0.0, rtol=1e-10, atol=1e-10)

    k1, k2 = jax.random.split(k1)
    mu_other = mu + 0.3 * jax.random.normal(k1, (d,))
    positive = score_divergence_gaussian_closed_form(mu_other, sigma, mu, sigma)
    assert float(positive) > 0.0


def test_score_divergence_is_affine_invariant():
    key = jax.random.PRNGKey(5)
    d = 4
    k1, k2 = jax.random.split(key)
    mu_q = jax.random.normal(k1, (d,))
    sigma_q = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu_p = jax.random.normal(k1, (d,))
    sigma_p = _random_spd(k2, d)

    n = 256
    k_sample, k_affine = jax.random.split(k2)
    z = _sample_mvn(k_sample, mu_q, sigma_q, n)
    score_p = _gaussian_score(mu_p, sigma_p)

    original = score_divergence_gaussian(mu_q, sigma_q, score_p, z)

    # Non-singular affine map z -> A z + b.
    a = jnp.eye(d) + 0.2 * jax.random.normal(k_affine, (d, d))
    b = jax.random.normal(k_affine, (d,))

    z_transformed = z @ a.T + b
    mu_q_transformed = a @ mu_q + b
    sigma_q_transformed = a @ sigma_q @ a.T
    mu_p_transformed = a @ mu_p + b
    sigma_p_transformed = a @ sigma_p @ a.T

    inv_a = jnp.linalg.inv(a)

    def score_p_transformed(zt: jnp.ndarray) -> jnp.ndarray:
        z_original = (zt - b) @ inv_a.T
        return score_p(z_original) @ inv_a.T

    transformed = score_divergence_gaussian(
        mu_q_transformed,
        sigma_q_transformed,
        score_p_transformed,
        z_transformed,
    )

    np.testing.assert_allclose(transformed, original, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# KL divergences
# ---------------------------------------------------------------------------
def test_kl_gaussian_matches_formula():
    key = jax.random.PRNGKey(6)
    d = 4
    k1, k2 = jax.random.split(key)
    mu0 = jax.random.normal(k1, (d,))
    sigma0 = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu1 = jax.random.normal(k1, (d,))
    sigma1 = _random_spd(k2, d)

    expected = _kl_gaussian_formula(mu0, sigma0, mu1, sigma1)
    got = kl_gaussian(mu0, sigma0, mu1, sigma1)

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_reverse_kl_gaussian_matches_forward_kl_swapped():
    key = jax.random.PRNGKey(7)
    d = 4
    k1, k2 = jax.random.split(key)
    mu_q = jax.random.normal(k1, (d,))
    sigma_q = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu_p = jax.random.normal(k1, (d,))
    sigma_p = _random_spd(k2, d)

    got = reverse_kl_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    expected = kl_gaussian(mu_p, sigma_p, mu_q, sigma_q)

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Fisher divergence
# ---------------------------------------------------------------------------
def test_fisher_divergence_gaussian_closed_form_matches_formula():
    key = jax.random.PRNGKey(8)
    d = 4
    k1, k2 = jax.random.split(key)
    mu_q = jax.random.normal(k1, (d,))
    sigma_q = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu_p = jax.random.normal(k1, (d,))
    sigma_p = _random_spd(k2, d)

    expected = _fisher_gaussian_closed_form(mu_q, sigma_q, mu_p, sigma_p)
    got = fisher_divergence_gaussian_closed_form(mu_q, sigma_q, mu_p, sigma_p)

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_fisher_divergence_estimate_matches_closed_form_large_batch():
    key = jax.random.PRNGKey(9)
    d = 3
    k1, k2 = jax.random.split(key)
    mu_q = jax.random.normal(k1, (d,))
    sigma_q = _random_spd(k2, d)
    k1, k2 = jax.random.split(k1)
    mu_p = jax.random.normal(k1, (d,))
    sigma_p = _random_spd(k2, d)

    n = 60_000
    k_sample, _ = jax.random.split(k2)
    z = _sample_mvn(k_sample, mu_q, sigma_q, n)
    score_p = _gaussian_score(mu_p, sigma_p)

    empirical = fisher_divergence_estimate(mu_q, sigma_q, score_p, z)
    closed = fisher_divergence_gaussian_closed_form(mu_q, sigma_q, mu_p, sigma_p)

    np.testing.assert_allclose(empirical, closed, rtol=0.08, atol=0.05)


# ---------------------------------------------------------------------------
# Gaussian tilt
# ---------------------------------------------------------------------------
def test_tilt_gaussian_mean_matches_closed_form():
    key = jax.random.PRNGKey(10)
    d = 4
    k1, k2 = jax.random.split(key)
    mu = jax.random.normal(k1, (d,))
    sigma = _random_spd(k2, d)
    s = jax.random.normal(k1, (d,))

    got = tilt_gaussian_mean(mu, sigma, s)
    expected = mu + sigma @ s

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_symmetrize_produces_symmetric_matrix():
    key = jax.random.PRNGKey(11)
    a = jax.random.normal(key, (5, 5))
    out = symmetrize(a)
    np.testing.assert_allclose(out, out.T, rtol=1e-12, atol=1e-12)
