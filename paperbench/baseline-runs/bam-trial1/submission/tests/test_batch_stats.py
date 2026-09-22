"""Unit tests for the batch sample/score statistics used by BaM.

These tests verify that :func:`compute_batch_stats` produces correctly
shaped, symmetric statistics and recovers the known moments of a Gaussian
distribution in the large-batch Monte Carlo limit.
"""

import os
import sys

import jax
import jax.numpy as jnp
import pytest

# Ensure the project root is importable (src layout, namespace package).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bam.batch_stats import BatchStats, batch_stats_to_dict, compute_batch_stats


def test_shapes_and_types():
    """Batch statistics should have the expected shapes and container type."""
    B, D = 5, 3
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (B, D))
    g = -z

    stats = compute_batch_stats(z, g)

    assert isinstance(stats, BatchStats)
    assert stats.zbar.shape == (D,)
    assert stats.gbar.shape == (D,)
    assert stats.C.shape == (D, D)
    assert stats.Gamma.shape == (D, D)


def test_gaussian_moments_in_expectation():
    """For a Gaussian target, batch statistics converge to the exact moments.

    If z ~ N(mu, Sigma) and g = grad_z log N(z; mu, Sigma)
    = -Sigma^{-1}(z - mu), then:
        zbar -> mu,       C -> Sigma,
        gbar -> 0,        Gamma -> Sigma^{-1}.
    """
    D = 4
    mu = jnp.array([1.0, -2.0, 0.5, 3.0])
    A = jnp.array(
        [
            [1.0, 0.3, 0.0, 0.1],
            [0.3, 1.5, -0.2, 0.0],
            [0.0, -0.2, 0.8, 0.4],
            [0.1, 0.0, 0.4, 2.0],
        ]
    )
    Sigma = A @ A.T
    Sigma_inv = jnp.linalg.inv(Sigma)

    key = jax.random.PRNGKey(1)
    B = 200_000
    eps = jax.random.normal(key, (B, D))
    L = jnp.linalg.cholesky(Sigma)
    z = mu[None, :] + eps @ L.T
    g = -((z - mu[None, :]) @ Sigma_inv)

    stats = compute_batch_stats(z, g)

    assert jnp.allclose(stats.zbar, mu, atol=0.05)
    assert jnp.allclose(stats.C, Sigma, atol=0.1)
    assert jnp.allclose(stats.gbar, jnp.zeros(D), atol=0.05)
    assert jnp.allclose(stats.Gamma, Sigma_inv, atol=0.2)


def test_covariances_are_symmetric_psd():
    """Sample covariance matrices should be symmetric positive semidefinite."""
    key = jax.random.PRNGKey(2)
    z = jax.random.normal(key, (100, 6))
    g = jax.random.normal(jax.random.fold_in(key, 1), (100, 6))

    stats = compute_batch_stats(z, g)

    for name, matrix in (("C", stats.C), ("Gamma", stats.Gamma)):
        assert jnp.allclose(matrix, matrix.T, atol=1e-6), f"{name} is not symmetric"
        assert jnp.min(jnp.linalg.eigvalsh(matrix)) >= -1e-6, f"{name} is not PSD"


def test_batch_stats_to_dict():
    """Conversion to dict should preserve all four statistics."""
    key = jax.random.PRNGKey(3)
    z = jax.random.normal(key, (12, 5))
    g = jax.random.normal(jax.random.fold_in(key, 4), (12, 5))

    stats = compute_batch_stats(z, g)
    as_dict = batch_stats_to_dict(stats)

    assert set(as_dict.keys()) == {"zbar", "gbar", "C", "Gamma"}
    assert jnp.allclose(as_dict["zbar"], stats.zbar)
    assert jnp.allclose(as_dict["gbar"], stats.gbar)
    assert jnp.allclose(as_dict["C"], stats.C)
    assert jnp.allclose(as_dict["Gamma"], stats.Gamma)


def test_input_validation():
    """Non-2D inputs and shape mismatches should be rejected."""
    # 1D inputs are not valid batches.
    z = jnp.ones(5)
    g = jnp.ones(5)
    with pytest.raises((ValueError, AssertionError)):
        compute_batch_stats(z, g)

    # Mismatched batch dimensions between z and g.
    z = jnp.ones((4, 3))
    g = jnp.ones((4, 2))
    with pytest.raises((ValueError, AssertionError)):
        compute_batch_stats(z, g)
