"""Tests for the quadratic matrix equation solver used by Batch-and-Match.

The solver targets
    X U X + X = V
where U and V are symmetric positive-semidefinite matrices.

These tests verify:
  * the full-rank closed-form solution satisfies the equation,
  * the low-rank solution agrees with the full-rank solution when U = Q Q^T,
  * factor_psd reconstructs a valid low-rank factor,
  * the dispatcher chooses a valid solution,
  * returned X is symmetric and positive-semidefinite,
  * validate_solution reports small residuals.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Make the local ``bam`` package importable when running tests directly.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from bam.quadratic_solver import (  # noqa: E402
    factor_psd,
    solve_quadratic,
    solve_quadratic_full,
    solve_quadratic_low_rank,
    validate_solution,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


def _random_psd(key: jax.Array, n: int, jitter: float = 1e-6) -> jnp.ndarray:
    """Return a random symmetric positive-definite n x n matrix."""
    a = jax.random.normal(key, (n, n))
    m = a @ a.T
    return 0.5 * (m + m.T) + jitter * jnp.eye(n)


def _random_sym(key: jax.Array, n: int) -> jnp.ndarray:
    """Return a random symmetric n x n matrix."""
    a = jax.random.normal(key, (n, n))
    return 0.5 * (a + a.T)


def _residual(X: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Residual of the quadratic matrix equation X U X + X = V."""
    return X @ U @ X + X - V


def test_full_solver_satisfies_equation() -> None:
    """The full-rank solver should satisfy X U X + X = V."""
    key = jax.random.PRNGKey(0)
    n = 6
    key_u, key_v = jax.random.split(key)
    U = _random_psd(key_u, n)
    V = _random_psd(key_v, n)

    X = solve_quadratic_full(U, V)

    np.testing.assert_allclose(_residual(X, U, V), jnp.zeros((n, n)), atol=1e-5)


def test_full_solver_produces_symmetric_psd_solution() -> None:
    """The full-rank solution should be symmetric and positive-semidefinite."""
    key = jax.random.PRNGKey(1)
    n = 5
    key_u, key_v = jax.random.split(key)
    U = _random_psd(key_u, n)
    V = _random_psd(key_v, n)

    X = solve_quadratic_full(U, V)

    np.testing.assert_allclose(X, X.T, atol=1e-10)
    eigvals = np.asarray(jnp.linalg.eigvalsh(X))
    assert eigvals.min() >= -1e-8


def test_low_rank_solver_matches_full_solver() -> None:
    """When U = Q Q^T, the low-rank and full-rank solvers should agree."""
    key = jax.random.PRNGKey(2)
    D, K = 8, 3
    key_q, key_v = jax.random.split(key)
    Q = jax.random.normal(key_q, (D, K))
    U = Q @ Q.T
    V = _random_psd(key_v, D)

    X_low = solve_quadratic_low_rank(Q, V)
    X_full = solve_quadratic_full(U, V)

    np.testing.assert_allclose(X_low, X_full, atol=1e-5)
    np.testing.assert_allclose(_residual(X_low, U, V), jnp.zeros((D, D)), atol=1e-5)


def test_low_rank_solution_is_symmetric_psd() -> None:
    """The low-rank solution should be symmetric and positive-semidefinite."""
    key = jax.random.PRNGKey(3)
    D, K = 7, 2
    key_q, key_v = jax.random.split(key)
    Q = jax.random.normal(key_q, (D, K))
    V = _random_psd(key_v, D)

    X = solve_quadratic_low_rank(Q, V)

    np.testing.assert_allclose(X, X.T, atol=1e-10)
    eigvals = np.asarray(jnp.linalg.eigvalsh(X))
    assert eigvals.min() >= -1e-8


def test_factor_psd_reconstructs_low_rank_matrix() -> None:
    """factor_psd should recover a factor Q with Q Q^T approximately U."""
    key = jax.random.PRNGKey(4)
    D, K = 9, 3
    key_q = jax.random.PRNGKey(10)
    Q = jax.random.normal(key_q, (D, K))
    U = Q @ Q.T
    # Symmetrize and add a tiny jitter to remove exact numerical null directions.
    U = 0.5 * (U + U.T) + 1e-10 * jnp.eye(D)

    Q_hat = factor_psd(U, tol=1e-7)

    assert Q_hat.ndim == 2
    assert Q_hat.shape[0] == D
    assert Q_hat.shape[1] <= D
    np.testing.assert_allclose(Q_hat @ Q_hat.T, U, atol=1e-5)


def test_solve_quadratic_dispatch_uses_valid_solution() -> None:
    """The dispatcher should solve the equation regardless of solver choice."""
    key = jax.random.PRNGKey(5)
    D, K = 10, 2
    key_q, key_v = jax.random.split(key)
    Q = jax.random.normal(key_q, (D, K))
    U = Q @ Q.T
    V = _random_psd(key_v, D)

    X = solve_quadratic(U, V, rank_tol=1e-8)

    np.testing.assert_allclose(_residual(X, U, V), jnp.zeros((D, D)), atol=1e-5)
    np.testing.assert_allclose(X, X.T, atol=1e-10)


def test_validate_solution_reports_small_residual() -> None:
    """validate_solution should report a small residual and symmetry error."""
    key = jax.random.PRNGKey(6)
    n = 6
    key_u, key_v = jax.random.split(key)
    U = _random_psd(key_u, n)
    V = _random_psd(key_v, n)

    X = solve_quadratic_full(U, V)
    info = validate_solution(X, U, V)

    assert float(jnp.abs(info["residual"])) < 1e-5
    assert float(info["symmetry_error"]) < 1e-9
    assert float(info["min_eigval"]) >= -1e-8
    assert float(info["frobenius_norm"]) >= 0.0


def test_full_solver_handles_zero_U() -> None:
    """If U = 0, the equation reduces to X = V."""
    key = jax.random.PRNGKey(7)
    n = 4
    key_v = jax.random.PRNGKey(11)
    U = jnp.zeros((n, n))
    V = _random_psd(key_v, n)

    X = solve_quadratic_full(U, V)

    np.testing.assert_allclose(X, V, atol=1e-5)


def test_solvers_reject_non_square_matrices() -> None:
    """Input validation should reject matrices with incompatible shapes."""
    key = jax.random.PRNGKey(8)
    U = jnp.ones((3, 4))
    V = jnp.ones((4, 3))
    with pytest.raises((ValueError, AssertionError)):
        solve_quadratic_full(U, V)
