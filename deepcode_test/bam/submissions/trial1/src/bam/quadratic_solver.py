"""Quadratic matrix equation solver for the Batch-and-Match (BaM) update.

The BaM covariance update requires solving, for symmetric positive
semidefinite matrices :math:`U, V \\in \\mathbb{R}^{D \\times D}`,

.. math::
    X U X + X = V.

Section 3 / Appendix C of "Batch and Match: Score-Based Black-Box
Variational Inference" gives two closed-form solutions:

* a full-rank :math:`O(D^3)` solution using a matrix square root,
* a low-rank :math:`O(K D^2 + K^3)` solution when :math:`U = Q Q^\\top`
  with :math:`Q \\in \\mathbb{R}^{D \\times K}` and :math:`K \\ll D`.

All computations are implemented in JAX so that the solver can be used
inside differentiable pipelines when desired.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "solve_quadratic_full",
    "solve_quadratic_low_rank",
    "factor_psd",
    "solve_quadratic",
    "validate_solution",
]


def _symmetrize(a: jnp.ndarray) -> jnp.ndarray:
    """Return the symmetric part of a square matrix."""
    return 0.5 * (a + a.T)


def _validate_psd_pair(U: jnp.ndarray, V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Symmetrize U and V and enforce small positive diagonal jitter.

    The theoretical update assumes symmetric positive semidefinite inputs.
    Numerical round-off can introduce tiny negative eigenvalues; clamping
    with negligible diagonal jitter keeps the matrix square roots real and
    stable without meaningfully changing the solution.
    """
    U = _symmetrize(U)
    V = _symmetrize(V)
    d = U.shape[-1]
    # Use a tiny multiple of the trace to keep conditioning reasonable while
    # guaranteeing positive definiteness of the shifted operators.
    jitter = 1e-12 * (jnp.abs(jnp.trace(U)) / d + 1.0)
    U = U + jitter * jnp.eye(d)
    V = V + jitter * jnp.eye(d)
    return U, V


def solve_quadratic_full(U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Solve ``X U X + X = V`` with the full-rank :math:`O(D^3)` formula.

    Parameters
    ----------
    U : (D, D) array
        Symmetric positive semidefinite matrix.
    V : (D, D) array
        Symmetric positive semidefinite matrix.

    Returns
    -------
    X : (D, D) array
        Symmetric positive semidefinite solution
        ``X = 2 V [I + (I + 4 U V)^{1/2}]^{-1}``.
    """
    U, V = _validate_psd_pair(U, V)
    d = U.shape[-1]
    I = jnp.eye(d)
    # For symmetric positive definite M = I + 4 U V, we use a symmetric
    # eigendecomposition square root, which is more numerically robust than
    # a generic matrix square root for the near-symmetric products arising
    # from Monte Carlo estimates.
    M = I + 4.0 * U @ V
    M = _symmetrize(M)
    evals, evecs = jnp.linalg.eigh(M)
    evals = jnp.clip(evals, 0.0, None)
    sqrt_M = (evecs * jnp.sqrt(evals)[None, :]) @ evecs.T
    sqrt_M = _symmetrize(sqrt_M)
    inner = I + sqrt_M
    X = 2.0 * V @ jnp.linalg.inv(inner)
    return _symmetrize(X)


def factor_psd(U: jnp.ndarray, tol: float = 1e-10) -> jnp.ndarray:
    """Factor a symmetric PSD matrix as ``U = Q Q^T``.

    Only eigenvectors corresponding to eigenvalues above ``tol * lambda_max``
    are retained, producing a low-rank factor :math:`Q \\in \\mathbb{R}^{D
    \\times K}`.

    Parameters
    ----------
    U : (D, D) array
        Symmetric positive semidefinite matrix.
    tol : float
        Relative eigenvalue threshold.

    Returns
    -------
    Q : (D, K) array
        Low-rank factor with :math:`U \\approx Q Q^\\top`.
    """
    U = _symmetrize(U)
    evals, evecs = jnp.linalg.eigh(U)
    evals = jnp.clip(evals, 0.0, None)
    threshold = tol * jnp.maximum(evals[-1], 0.0)
    mask = evals >= threshold
    idx = jnp.argsort(mask, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Keep the largest eigenvalues first.
    kept = mask.sum()
    kept = jnp.asarray(kept, dtype=jnp.int32)
    K = jnp.maximum(kept, 1)
    Q = evecs[:, :K] * jnp.sqrt(jnp.maximum(evals[:K], 0.0))[None, :]
    return Q


def solve_quadratic_low_rank(Q: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Solve ``X U X + X = V`` when ``U = Q Q^T`` is low-rank.

    Parameters
    ----------
    Q : (D, K) array
        Low-rank factor of ``U``.
    V : (D, D) array
        Symmetric positive semidefinite matrix.

    Returns
    -------
    X : (D, D) array
        Symmetric positive semidefinite solution

        .. math::
            X = V - V Q [0.5 I + (Q^T V Q + 0.25 I)^{1/2}]^{-2} Q^T V.
    """
    V = _symmetrize(V)
    d = V.shape[-1]
    K = Q.shape[-1]
    I_k = jnp.eye(K)
    inner = Q.T @ V @ Q + 0.25 * I_k
    inner = _symmetrize(inner)
    evals, evecs = jnp.linalg.eigh(inner)
    evals = jnp.clip(evals, 0.0, None)
    sqrt_inner = (evecs * jnp.sqrt(evals)[None, :]) @ evecs.T
    sqrt_inner = _symmetrize(sqrt_inner)
    denom = 0.5 * I_k + sqrt_inner
    inv_denom = jnp.linalg.inv(denom)
    inv_denom_sq = inv_denom @ inv_denom
    VQ = V @ Q
    X = V - VQ @ inv_denom_sq @ Q.T @ V
    # Small diagonal jitter keeps the result well conditioned in practice.
    X = X + 1e-12 * jnp.eye(d)
    return _symmetrize(X)


def solve_quadratic(U: jnp.ndarray, V: jnp.ndarray, rank_tol: float = 1e-10) -> jnp.ndarray:
    """Solve ``X U X + X = V`` choosing the best available closed form.

    If the numerical rank of ``U`` is at most half its dimension, the
    low-rank solver is used; otherwise the full-rank solver is used.

    Parameters
    ----------
    U : (D, D) array
        Symmetric positive semidefinite matrix.
    V : (D, D) array
        Symmetric positive semidefinite matrix.
    rank_tol : float
        Relative eigenvalue threshold used for rank estimation.

    Returns
    -------
    X : (D, D) array
        Symmetric positive semidefinite solution.
    """
    U, V = _validate_psd_pair(U, V)
    d = U.shape[-1]
    Q = factor_psd(U, tol=rank_tol)
    rank = Q.shape[-1]
    # JAX jitting needs a static branch; the boolean rank check is constant
    # for a fixed input shape, so a Python conditional is safe.
    if rank <= d // 2:
        return solve_quadratic_low_rank(Q, V)
    return solve_quadratic_full(U, V)


def validate_solution(X: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Check residual, symmetry, and positive semidefiniteness of ``X``.

    Parameters
    ----------
    X, U, V : (D, D) arrays
        Candidate solution and the defining matrices.

    Returns
    -------
    dict
        Contains ``residual`` (Frobenius norm of ``X U X + X - V``),
        ``symmetry_error``, ``min_eigval``, and ``fro_norm``.
    """
    U, V = _validate_psd_pair(U, V)
    X = _symmetrize(X)
    residual = jnp.linalg.norm(X @ U @ X + X - V)
    symmetry_error = jnp.linalg.norm(X - X.T)
    min_eigval = jnp.min(jnp.linalg.eigvalsh(_symmetrize(X)))
    fro_norm = jnp.linalg.norm(X)
    return {
        "residual": residual,
        "symmetry_error": symmetry_error,
        "min_eigval": min_eigval,
        "fro_norm": fro_norm,
    }
