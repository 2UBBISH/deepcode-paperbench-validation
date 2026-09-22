"""Tests for Batch-and-Match convergence on Gaussian targets.

These tests validate the qualitative behavior of Theorem 1 from the paper:

* With a very large batch size and a very large learning rate ``lambda``, one
  Batch-and-Match step nearly recovers the exact Gaussian target parameters.
* Iterating BaM with a constant learning rate contracts both the normalized
  mean error and the normalized covariance error.
* In the single-sample, large-``lambda`` limit, the BaM update recovers the
  Gaussian Score Matching (GSM) update.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.bam.algorithm import bam_step, constant_learning_rate


def _sym_sqrt_inv(a: jnp.ndarray) -> jnp.ndarray:
    """Symmetric inverse square root of a positive semidefinite matrix."""
    evals, evecs = jnp.linalg.eigh(a)
    evals = jnp.clip(evals, 1e-12)
    return (evecs * (1.0 / jnp.sqrt(evals))) @ evecs.T


def _gaussian_score(mu: jnp.ndarray, sigma_inv: jnp.ndarray):
    """Return a vectorized score function for N(mu, sigma)."""

    def score(z: jnp.ndarray) -> jnp.ndarray:
        return -jnp.einsum("ij,bj->bi", sigma_inv, z - mu)

    return score


def _normalized_mean_error(mu: jnp.ndarray, mu_star: jnp.ndarray, inv_sqrt: jnp.ndarray) -> jnp.ndarray:
    return inv_sqrt @ (mu - mu_star)


def _normalized_cov_error(sigma: jnp.ndarray, inv_sqrt: jnp.ndarray) -> jnp.ndarray:
    return inv_sqrt @ sigma @ inv_sqrt - jnp.eye(sigma.shape[0])


def test_one_step_near_convergence_large_lambda():
    """One BaM step with a huge batch and lambda should nearly match the target."""
    key = jax.random.PRNGKey(0)
    d = 4
    key_a, key_mu, key_init, key_run = jax.random.split(key, 4)

    a = jax.random.normal(key_a, (d, d))
    sigma_star = a @ a.T + jnp.eye(d)
    mu_star = jax.random.normal(key_mu, (d,))
    sigma_inv = jnp.linalg.inv(sigma_star)

    mu0 = jax.random.uniform(key_init, (d,), minval=0.0, maxval=0.1)
    sigma0 = jnp.eye(d)

    target_score = _gaussian_score(mu_star, sigma_inv)

    b = 20000
    lam = 1e5
    result = bam_step(key_run, mu0, sigma0, target_score=target_score, b=b, lam=lam)

    inv_sqrt = _sym_sqrt_inv(sigma_star)
    eps = _normalized_mean_error(result.mu, mu_star, inv_sqrt)
    delta = _normalized_cov_error(result.sigma, inv_sqrt)

    assert jnp.linalg.norm(eps) < 0.05
    assert jnp.linalg.norm(delta) < 0.10


def test_bam_iterates_contract_gaussian_errors():
    """Iterating BaM on a Gaussian target contracts normalized mean/cov errors."""
    key = jax.random.PRNGKey(1)
    d = 4
    key_a, key_mu, key_init, key_run = jax.random.split(key, 4)

    a = jax.random.normal(key_a, (d, d))
    sigma_star = a @ a.T + jnp.eye(d)
    mu_star = jax.random.normal(key_mu, (d,))
    sigma_inv = jnp.linalg.inv(sigma_star)

    mu0 = jax.random.uniform(key_init, (d,), minval=0.0, maxval=0.1)
    sigma0 = jnp.eye(d)

    target_score = _gaussian_score(mu_star, sigma_inv)

    b = 5000
    lr = constant_learning_rate(b, d)
    lam = lr(0)

    mu = mu0
    sigma = sigma0
    key = key_run
    for _ in range(6):
        key, subkey = jax.random.split(key)
        result = bam_step(subkey, mu, sigma, target_score=target_score, b=b, lam=lam)
        mu = result.mu
        sigma = result.sigma

    inv_sqrt = _sym_sqrt_inv(sigma_star)
    eps0 = jnp.linalg.norm(_normalized_mean_error(mu0, mu_star, inv_sqrt))
    eps_t = jnp.linalg.norm(_normalized_mean_error(mu, mu_star, inv_sqrt))
    delta0 = jnp.linalg.norm(_normalized_cov_error(sigma0, inv_sqrt))
    delta_t = jnp.linalg.norm(_normalized_cov_error(sigma, inv_sqrt))

    assert eps_t < eps0
    assert delta_t < delta0


def test_bam_recovers_gsm_in_single_sample_large_lambda_limit():
    """BaM with B=1 and lambda -> infinity should match the GSM closed-form update."""
    from src.bam.match_update import match_gsm_update

    key = jax.random.PRNGKey(2)
    d = 3
    key_a, key_mu, key_init, key_run = jax.random.split(key, 4)

    a = jax.random.normal(key_a, (d, d))
    sigma_star = a @ a.T + jnp.eye(d)
    mu_star = jax.random.normal(key_mu, (d,))
    sigma_inv = jnp.linalg.inv(sigma_star)

    mu0 = jax.random.normal(key_init, (d,))
    sigma0 = jax.random.normal(key_a, (d, d))
    sigma0 = sigma0 @ sigma0.T + 0.5 * jnp.eye(d)

    target_score = _gaussian_score(mu_star, sigma_inv)

    lam = 1e6
    result = bam_step(key_run, mu0, sigma0, target_score=target_score, b=1, lam=lam)

    mu_gsm, sigma_gsm = match_gsm_update(mu0, sigma0, result.z[0], result.g[0])

    np.testing.assert_allclose(np.asarray(result.mu), np.asarray(mu_gsm), atol=2e-3)
    np.testing.assert_allclose(np.asarray(result.sigma), np.asarray(sigma_gsm), atol=2e-3)


if __name__ == "__main__":
    test_one_step_near_convergence_large_lambda()
    test_bam_iterates_contract_gaussian_errors()
    test_bam_recovers_gsm_in_single_sample_large_lambda_limit()
    print("All Gaussian convergence tests passed.")
