"""Tests for the baseline variational-inference algorithms.

The module exercises ADAM, ADVI, the score/Fisher gradient baselines, and the
closed-form GSM update against a known Gaussian target. The Gaussian target is
particularly convenient because all gradients and per-sample updates are
available in closed form.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Ensure ``src`` is importable when running from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bam.baselines import (
    AdamState,
    GradientStepResult,
    GsmStepResult,
    adam_init,
    adam_step,
    advi_step,
    fisher_step,
    gsm_step,
    gsm_update,
    run_advi,
    run_gsm,
    score_step,
)


def _gaussian_score(mu_p, sigma_p_inv):
    """Return a vectorized score function for N(mu_p, Sigma_p)."""

    def score(z):
        # z: (B, D) -> scores: (B, D)
        return -(z - mu_p) @ sigma_p_inv

    return score


def _symmetrize(a):
    return (a + a.T) / 2.0


def _final_mu(result):
    if isinstance(result, dict):
        return result["mu"]
    return result


def _final_sigma(result):
    if isinstance(result, dict):
        if "Sigma" in result:
            return result["Sigma"]
        if "L" in result:
            return result["L"] @ result["L"].T
    if hasattr(result, "Sigma"):
        return result.Sigma
    return result


def test_adam_init_and_step_reduces_quadratic():
    """ADAM should decrease a simple scalar quadratic objective."""
    x = jnp.array(3.0)
    state = adam_init(x)
    assert isinstance(state, AdamState)
    assert state.t == 0

    values = []
    for _ in range(200):
        grad = 2.0 * x  # gradient of x^2
        x, state = adam_step(x, grad, state, learning_rate=0.05)
        values.append(float(x))

    # After a few hundred ADAM iterations on a quadratic, |x| should shrink.
    assert abs(values[-1]) < abs(values[0])


def test_advi_step_moves_mean_toward_gaussian_target():
    """One ADVI step should move q's mean toward a Gaussian target's mean."""
    rng = np.random.default_rng(0)
    D = 3
    mu_p = jnp.array([2.0, -1.5, 1.0])
    sigma_p = jnp.array([[1.5, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.2]])
    sigma_p_inv = jnp.linalg.inv(sigma_p)

    key = jax.random.PRNGKey(0)
    mu = jnp.zeros(D)
    L = jnp.eye(D)

    target_score = _gaussian_score(mu_p, sigma_p_inv)
    state = adam_init((mu, L))

    result = advi_step(
        key,
        mu,
        L,
        target_score,
        batch_size=20000,
        adam_state=state,
        learning_rate=0.5,
    )

    assert isinstance(result, GradientStepResult)
    assert result.mu.shape == (D,)
    assert result.Sigma.shape == (D, D)
    assert result.L.shape == (D, D)

    sigma_new = _symmetrize(result.Sigma)
    eigvals = jnp.linalg.eigvalsh(sigma_new)
    assert jnp.all(eigvals > 0)

    d0 = float(jnp.linalg.norm(mu - mu_p))
    d1 = float(jnp.linalg.norm(result.mu - mu_p))
    assert d1 < d0


def test_score_step_moves_mean_toward_gaussian_target():
    """The empirical score-divergence baseline should also improve the mean."""
    D = 3
    mu_p = jnp.array([2.0, -1.5, 1.0])
    sigma_p = jnp.array([[1.5, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.2]])
    sigma_p_inv = jnp.linalg.inv(sigma_p)

    key = jax.random.PRNGKey(1)
    mu = jnp.zeros(D)
    L = jnp.eye(D)

    result = score_step(
        key,
        mu,
        L,
        _gaussian_score(mu_p, sigma_p_inv),
        batch_size=20000,
        adam_state=adam_init((mu, L)),
        learning_rate=0.5,
    )

    assert isinstance(result, GradientStepResult)
    assert result.mu.shape == (D,)
    assert result.Sigma.shape == (D, D)

    d0 = float(jnp.linalg.norm(mu - mu_p))
    d1 = float(jnp.linalg.norm(result.mu - mu_p))
    assert d1 < d0


def test_fisher_step_runs_and_returns_valid_gaussian():
    """Fisher-divergence minimization should run and preserve a valid q."""
    D = 2
    mu_p = jnp.array([0.5, -0.5])
    sigma_p = jnp.eye(D)
    sigma_p_inv = jnp.linalg.inv(sigma_p)

    key = jax.random.PRNGKey(2)
    mu = jnp.zeros(D)
    L = jnp.eye(D)

    result = fisher_step(
        key,
        mu,
        L,
        _gaussian_score(mu_p, sigma_p_inv),
        batch_size=2000,
        adam_state=adam_init((mu, L)),
        learning_rate=0.1,
    )

    assert isinstance(result, GradientStepResult)
    assert result.mu.shape == (D,)
    assert result.Sigma.shape == (D, D)
    assert jnp.all(jnp.linalg.eigvalsh(_symmetrize(result.Sigma)) > 0)


def _algorithm3_single_sample(mu, sigma, z, s):
    """Direct transcription of Algorithm 3 for one sample."""
    eps = sigma @ s - mu + z
    a = (mu - z) @ s
    c = s @ sigma @ s + a * a
    rho = (-1.0 + jnp.sqrt(1.0 + 4.0 * c)) / 2.0
    denom = 1.0 + rho + a
    delta_mu = (1.0 / (1.0 + rho)) * (
        eps - (mu - z) * (s @ eps) / denom
    )
    tilde_mu = mu + delta_mu
    delta_sigma = jnp.outer(mu - z, mu - z) - jnp.outer(tilde_mu - z, tilde_mu - z)
    return tilde_mu, sigma + delta_sigma


def test_gsm_update_matches_algorithm3_single_sample():
    """The vectorized GSM update should match the scalar Algorithm 3 formula."""
    D = 3
    mu = jnp.array([0.3, -0.2, 0.4])
    sigma = jnp.array([[1.2, 0.1, 0.0], [0.1, 0.9, 0.2], [0.0, 0.2, 1.1]])
    z = jnp.array([0.5, -0.6, 0.1])
    s = jnp.array([0.8, -0.4, 0.3])

    mu_new, sigma_new = gsm_update(mu, sigma, z[None, :], s[None, :])

    mu_ref, sigma_ref = _algorithm3_single_sample(mu, sigma, z, s)

    assert mu_new.shape == (D,)
    assert sigma_new.shape == (D, D)
    assert jnp.allclose(mu_new, mu_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(sigma_new, sigma_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(sigma_new, sigma_new.T, atol=1e-6)


def test_gsm_step_moves_mean_toward_gaussian_target():
    """GSM should move the variational mean toward the Gaussian target mean."""
    D = 3
    mu_p = jnp.array([1.5, -1.0, 0.7])
    sigma_p = jnp.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.1], [0.0, 0.1, 1.0]])
    sigma_p_inv = jnp.linalg.inv(sigma_p)

    key = jax.random.PRNGKey(3)
    mu = jnp.zeros(D)
    sigma = jnp.eye(D)

    result = gsm_step(key, mu, sigma, _gaussian_score(mu_p, sigma_p_inv), batch_size=500)

    assert isinstance(result, GsmStepResult)
    assert result.mu.shape == (D,)
    assert result.Sigma.shape == (D, D)
    assert jnp.allclose(result.Sigma, result.Sigma.T, atol=1e-6)

    d0 = float(jnp.linalg.norm(mu - mu_p))
    d1 = float(jnp.linalg.norm(result.mu - mu_p))
    assert d1 < d0


def test_run_gsm_and_run_advi_return_valid_gaussians():
    """High-level run loops should return well-formed Gaussian parameters."""
    D = 2
    mu_p = jnp.array([1.0, -0.5])
    sigma_p = jnp.array([[1.0, 0.0], [0.0, 0.8]])
    sigma_p_inv = jnp.linalg.inv(sigma_p)

    key = jax.random.PRNGKey(4)
    gsm_res = run_gsm(
        key,
        jnp.zeros(D),
        jnp.eye(D),
        _gaussian_score(mu_p, sigma_p_inv),
        T=3,
        batch_size=50,
    )
    gsm_mu = _final_mu(gsm_res)
    gsm_sigma = _final_sigma(gsm_res)
    assert gsm_mu.shape == (D,)
    assert gsm_sigma.shape == (D, D)
    assert jnp.all(jnp.linalg.eigvalsh(_symmetrize(gsm_sigma)) > 0)

    key2 = jax.random.PRNGKey(5)
    advi_res = run_advi(
        key2,
        jnp.zeros(D),
        jnp.eye(D),
        _gaussian_score(mu_p, sigma_p_inv),
        T=3,
        batch_size=50,
        learning_rate=0.05,
    )
    advi_mu = _final_mu(advi_res)
    advi_sigma = _final_sigma(advi_res)
    assert advi_mu.shape == (D,)
    assert advi_sigma.shape == (D, D)
    assert jnp.all(jnp.linalg.eigvalsh(_symmetrize(advi_sigma)) > 0)
