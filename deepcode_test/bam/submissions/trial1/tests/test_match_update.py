"""Tests for the Batch-and-Match mean/covariance MATCH updates.

This module verifies the closed-form BaM covariance update (quadratic matrix
equation), the closed-form mean update, the combined ``match_update`` step, the
BaM proximal objective, and the GSM limiting case implemented in
``src/bam/match_update.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Make the local ``src/bam`` package importable when tests are run directly.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Higher precision makes the matrix-equation and closed-form assertions tight.
jax.config.update("jax_enable_x64", True)

from bam.batch_stats import BatchStats, compute_batch_stats  # noqa: E402
from bam.divergences import kl_gaussian, score_divergence_batch_form  # noqa: E402
from bam.match_update import (  # noqa: E402
    bam_objective,
    match_covariance,
    match_gsm_update,
    match_mean,
    match_update,
    match_uv,
)
from bam.quadratic_solver import validate_solution  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symmetrize(a: jnp.ndarray) -> jnp.ndarray:
    """Symmetrize a square matrix."""
    return (a + a.T) / 2.0


def _random_psd(key: jax.Array, n: int, jitter: float = 1e-6) -> jnp.ndarray:
    """Return a random symmetric positive definite matrix of size n x n."""
    a = jax.random.normal(key, (n, n))
    return _symmetrize(a @ a.T) + jitter * jnp.eye(n)


def _make_stats(key: jax.Array, d: int, b: int = 64) -> tuple[jnp.ndarray, jnp.ndarray, BatchStats]:
    """Sample a batch and return ``(z, g, stats)`` with valid batch statistics."""
    key_z, key_g = jax.random.split(key)
    z = jax.random.normal(key_z, (b, d))
    g = jax.random.normal(key_g, (b, d))
    return z, g, compute_batch_stats(z, g)


# ---------------------------------------------------------------------------
# match_uv
# ---------------------------------------------------------------------------

def test_match_uv_builds_expected_matrices() -> None:
    """``match_uv`` should construct U and V exactly as in the paper."""
    d = 4
    key = jax.random.PRNGKey(0)
    k_mu, k_sigma, k_vec, k_c, k_gam = jax.random.split(key, 5)

    mu_t = jax.random.normal(k_mu, (d,))
    Sigma_t = _random_psd(k_sigma, d)
    zbar = jax.random.normal(k_vec, (d,))
    gbar = jax.random.normal(k_vec, (d,))
    C = _random_psd(k_c, d)
    Gamma = _random_psd(k_gam, d)
    lam = 1.7

    U, V = match_uv(mu_t, Sigma_t, zbar, gbar, C, Gamma, lam)

    U_expected = lam * Gamma + (lam / (1.0 + lam)) * jnp.outer(gbar, gbar)
    V_expected = (
        Sigma_t
        + lam * C
        + (lam / (1.0 + lam)) * jnp.outer(mu_t - zbar, mu_t - zbar)
    )

    np.testing.assert_allclose(U, U_expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(V, V_expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(U, U.T, atol=1e-10)
    np.testing.assert_allclose(V, V.T, atol=1e-10)


def test_match_uv_rejects_negative_lambda() -> None:
    """``match_uv`` should reject a negative learning rate."""
    d = 3
    key = jax.random.PRNGKey(1)
    mu_t = jax.random.normal(key, (d,))
    Sigma_t = _random_psd(jax.random.PRNGKey(2), d)
    zbar = jax.random.normal(jax.random.PRNGKey(3), (d,))
    gbar = jax.random.normal(jax.random.PRNGKey(4), (d,))
    C = _random_psd(jax.random.PRNGKey(5), d)
    Gamma = _random_psd(jax.random.PRNGKey(6), d)

    with pytest.raises((ValueError, AssertionError)):
        match_uv(mu_t, Sigma_t, zbar, gbar, C, Gamma, -1.0)


# ---------------------------------------------------------------------------
# match_covariance
# ---------------------------------------------------------------------------

def test_match_covariance_solves_quadratic_equation() -> None:
    """The covariance update must solve ``Sigma U Sigma + Sigma = V``."""
    d = 5
    key = jax.random.PRNGKey(7)
    k_mu, k_sigma, k_c, k_gam, k_vec = jax.random.split(key, 5)

    mu_t = jax.random.normal(k_mu, (d,))
    Sigma_t = _random_psd(k_sigma, d)
    zbar = jax.random.normal(k_vec, (d,))
    gbar = jax.random.normal(k_vec, (d,))
    C = _random_psd(k_c, d)
    Gamma = _random_psd(k_gam, d)
    lam = 0.8

    Sigma_new = match_covariance(Sigma_t, C, Gamma, zbar, gbar, mu_t, lam)
    U, V = match_uv(mu_t, Sigma_t, zbar, gbar, C, Gamma, lam)

    residual = Sigma_new @ U @ Sigma_new + Sigma_new - V
    np.testing.assert_allclose(residual, jnp.zeros((d, d)), atol=1e-9, rtol=1e-8)

    diagnostics = validate_solution(Sigma_new, U, V)
    np.testing.assert_allclose(diagnostics["residual"], 0.0, atol=1e-9, rtol=1e-8)


def test_match_covariance_is_symmetric_psd() -> None:
    """The covariance update should remain symmetric and positive semidefinite."""
    d = 6
    key = jax.random.PRNGKey(8)
    k_mu, k_sigma, k_c, k_gam, k_vec = jax.random.split(key, 5)

    mu_t = jax.random.normal(k_mu, (d,))
    Sigma_t = _random_psd(k_sigma, d)
    zbar = jax.random.normal(k_vec, (d,))
    gbar = jax.random.normal(k_vec, (d,))
    C = _random_psd(k_c, d)
    Gamma = _random_psd(k_gam, d)

    Sigma_new = match_covariance(Sigma_t, C, Gamma, zbar, gbar, mu_t, 1.0)

    np.testing.assert_allclose(Sigma_new, Sigma_new.T, atol=1e-10)
    eigvals = np.asarray(jnp.linalg.eigvalsh(Sigma_new))
    assert np.min(eigvals) > -1e-8


# ---------------------------------------------------------------------------
# match_mean
# ---------------------------------------------------------------------------

def test_match_mean_matches_closed_form() -> None:
    """The mean update should match the paper's closed-form expression."""
    d = 4
    key = jax.random.PRNGKey(9)
    k_mu, k_sigma, k_vec = jax.random.split(key, 3)

    mu_t = jax.random.normal(k_mu, (d,))
    zbar = jax.random.normal(k_vec, (d,))
    gbar = jax.random.normal(k_vec, (d,))
    Sigma_new = _random_psd(k_sigma, d)
    lam = 1.2

    expected = (lam / (1.0 + lam)) * (zbar + Sigma_new @ gbar) + (
        1.0 / (1.0 + lam)
    ) * mu_t
    actual = match_mean(mu_t, zbar, gbar, Sigma_new, lam)

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_match_mean_rejects_shape_mismatch() -> None:
    """The mean update should validate vector dimensions."""
    d = 4
    key = jax.random.PRNGKey(10)
    mu_t = jax.random.normal(key, (d,))
    zbar = jax.random.normal(key, (d + 1,))
    gbar = jax.random.normal(key, (d,))
    Sigma_new = _random_psd(jax.random.PRNGKey(11), d)

    with pytest.raises((ValueError, AssertionError)):
        match_mean(mu_t, zbar, gbar, Sigma_new, 1.0)


# ---------------------------------------------------------------------------
# match_update
# ---------------------------------------------------------------------------

def test_match_update_matches_component_updates() -> None:
    """``match_update`` should equal covariance-then-mean component updates."""
    d = 5
    key = jax.random.PRNGKey(12)
    k_mu, k_sigma, k_batch = jax.random.split(key, 3)

    mu_t = jax.random.normal(k_mu, (d,))
    Sigma_t = _random_psd(k_sigma, d)
    z, g, stats = _make_stats(k_batch, d, b=128)
    lam = 0.6

    mu_new, Sigma_new = match_update(mu_t, Sigma_t, stats, lam)

    Sigma_expected = match_covariance(
        Sigma_t, stats.C, stats.Gamma, stats.zbar, stats.gbar, mu_t, lam
    )
    mu_expected = match_mean(mu_t, stats.zbar, stats.gbar, Sigma_new, lam)

    np.testing.assert_allclose(Sigma_new, Sigma_expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(mu_new, mu_expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(Sigma_new, Sigma_new.T, atol=1e-10)


# ---------------------------------------------------------------------------
# bam_objective
# ---------------------------------------------------------------------------

def test_bam_objective_matches_manual_sum() -> None:
    """``bam_objective`` equals score divergence plus scaled KL proximal term."""
    d = 4
    key = jax.random.PRNGKey(13)
    k_mu, k_sigma, k_mu_t, k_sigma_t, k_batch = jax.random.split(key, 5)

    mu = jax.random.normal(k_mu, (d,))
    Sigma = _random_psd(k_sigma, d)
    mu_t = jax.random.normal(k_mu_t, (d,))
    Sigma_t = _random_psd(k_sigma_t, d)
    z, g, stats = _make_stats(k_batch, d, b=64)
    lam = 0.9

    zbar, gbar, C, Gamma = stats.zbar, stats.gbar, stats.C, stats.Gamma

    expected = score_divergence_batch_form(mu, Sigma, zbar, gbar, C, Gamma) + (
        2.0 / lam
    ) * kl_gaussian(mu_t, Sigma_t, mu, Sigma)
    actual = bam_objective(mu, Sigma, mu_t, Sigma_t, zbar, gbar, C, Gamma, lam)

    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)

    expected_const = score_divergence_batch_form(
        mu, Sigma, zbar, gbar, C, Gamma, z=z, g=g
    ) + (2.0 / lam) * kl_gaussian(mu_t, Sigma_t, mu, Sigma)
    actual_const = bam_objective(
        mu, Sigma, mu_t, Sigma_t, zbar, gbar, C, Gamma, lam, include_const=True, z=z, g=g
    )

    np.testing.assert_allclose(actual_const, expected_const, atol=1e-9, rtol=1e-8)


# ---------------------------------------------------------------------------
# GSM limiting case
# ---------------------------------------------------------------------------

def test_match_gsm_update_matches_limiting_case() -> None:
    """The GSM update is the B=1, lambda -> infinity BaM limit."""
    d = 3
    key = jax.random.PRNGKey(14)
    k_mu, k_sigma, k_z, k_g = jax.random.split(key, 4)

    mu_t = jax.random.normal(k_mu, (d,))
    Sigma_t = _random_psd(k_sigma, d)
    z = jax.random.normal(k_z, (d,))
    g = jax.random.normal(k_g, (d,))

    mu_new, Sigma_new = match_gsm_update(mu_t, Sigma_t, z, g)

    U = jnp.outer(g, g)
    V = Sigma_t + jnp.outer(mu_t - z, mu_t - z)

    residual = Sigma_new @ U @ Sigma_new + Sigma_new - V
    np.testing.assert_allclose(residual, jnp.zeros((d, d)), atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(mu_new, Sigma_new @ g + z, atol=1e-10, rtol=1e-10)
