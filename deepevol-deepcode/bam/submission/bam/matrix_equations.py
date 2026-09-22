"""Quadratic matrix equation solvers for the Batch-and-Match (BaM) algorithm.

The core numerical primitive of BaM is the *quadratic matrix equation*

    X U X + X = V                                             (eq. 54, §B)

where ``U ⪰ 0`` and ``V ≻ 0``.  Lemma B.1 of the paper gives the closed-form
solution

    X = 2 V [ I + (I + 4 U V)^{1/2} ]^{-1}                    (eq. 55, Lemma B.1)

and Lemma B.2 shows that this ``X`` is symmetric and positive definite.  Lemma B.3
gives an O(K D^2 + K^3) low-rank formula for the case ``U = Q Q^T`` with
``Q ∈ R^{D×K}``:

    X = V - V^T Q [ (1/2) I + (Q^T V Q + (1/4) I)^{1/2} ]^{-2} Q^T V
                                                              (eq. 60, Lemma B.3)

All matrix square roots are *principal* square roots, computed here via a symmetric
eigendecomposition with negative eigenvalues clipped to zero.

The functions accept either NumPy or JAX arrays; the array namespace is inferred
from the inputs so that the same code can be used inside ``jax.jit``.
"""

from __future__ import annotations

import numpy as np

try:  # JAX is the reference implementation language of the paper
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover - JAX is expected to be installed
    jnp = None
    _HAS_JAX = False


__all__ = [
    "array_namespace",
    "symmetrize",
    "matrix_sqrt",
    "matrix_sqrt_inv",
    "solve_symmetric",
    "inverse_spd",
    "ensure_spd",
    "solve_quadratic_matrix_equation",
    "solve_quadratic_matrix_equation_dense",
    "solve_quadratic_matrix_equation_low_rank",
    "residual",
]


# --------------------------------------------------------------------------------------
# namespace / helper utilities
# --------------------------------------------------------------------------------------
def array_namespace(*arrays):
    """Return the array module that should be used to process ``arrays``.

    ``jax.numpy`` is used whenever any input is a JAX array (so the routine is
    jit/vmap friendly); otherwise NumPy is used.
    """
    if _HAS_JAX:
        for a in arrays:
            if a is None:
                continue
            mod = type(a).__module__
            if mod.startswith("jax") or mod.startswith("jaxlib"):
                return jnp
    return np


def symmetrize(A):
    """Return the symmetric part ``(A + A^T) / 2``."""
    xp = array_namespace(A)
    return 0.5 * (A + xp.swapaxes(A, -1, -2))


def _eye_like(A, n=None, xp=None):
    xp = array_namespace(A) if xp is None else xp
    n = A.shape[-1] if n is None else n
    dtype = getattr(A, "dtype", None)
    return xp.eye(n, dtype=dtype)


def matrix_sqrt(A, clip_negative=True, floor=0.0):
    """Principal square root of a symmetric PSD matrix.

    Parameters
    ----------
    A : (D, D) array_like
        Symmetric positive semi-definite matrix (assumed, but not checked).
    clip_negative : bool
        If True, negative eigenvalues (numerical noise) are clipped to ``floor``.
    floor : float
        Value used for the clipping.

    Returns
    -------
    (D, D) array
        The principal square root ``A^{1/2}`` (symmetric PSD).
    """
    xp = array_namespace(A)
    A = symmetrize(A)
    w, V = xp.linalg.eigh(A)
    if clip_negative:
        w = xp.maximum(w, floor)
    return (V * xp.sqrt(w)) @ xp.swapaxes(V, -1, -2)


def matrix_sqrt_inv(A, clip_negative=True, floor=1e-12):
    """Inverse principal square root ``A^{-1/2}`` of a symmetric PD matrix."""
    xp = array_namespace(A)
    A = symmetrize(A)
    w, V = xp.linalg.eigh(A)
    if clip_negative:
        w = xp.maximum(w, floor)
    return (V * (1.0 / xp.sqrt(w))) @ xp.swapaxes(V, -1, -2)


def solve_symmetric(A, B):
    """Solve ``A X = B`` for symmetric positive definite ``A`` (uses Cholesky)."""
    xp = array_namespace(A, B)
    A = symmetrize(A)
    L = xp.linalg.cholesky(A)
    Y = xp.linalg.solve(L, B)
    return xp.linalg.solve(xp.swapaxes(L, -1, -2), Y)


def inverse_spd(A, jitter=0.0):
    """Inverse of a symmetric positive definite matrix via Cholesky."""
    xp = array_namespace(A)
    A = symmetrize(A)
    if jitter:
        n = A.shape[-1]
        A = A + jitter * _eye_like(A, n, xp)
    L = xp.linalg.cholesky(A)
    Linv = xp.linalg.inv(L)
    return symmetrize(xp.swapaxes(Linv, -1, -2) @ Linv)


def ensure_spd(A, jitter=1e-10, min_eig=1e-8):
    """Symmetrize ``A`` and, if needed, shift it into the positive definite cone.

    A Cholesky factorisation is attempted; on failure (or if the smallest eigenvalue
    is below ``min_eig``) a jitter is added to the diagonal.  This mirrors the
    "sym/PD re-projection after every covariance update" contract of the paper's
    implementation.
    """
    xp = array_namespace(A)
    A = symmetrize(A)
    n = A.shape[-1]
    w = xp.linalg.eigvalsh(A)
    bad = xp.min(w) <= min_eig
    shift = xp.where(bad, min_eig - xp.min(w) + jitter, 0.0)
    return A + shift * _eye_like(A, n, xp)


# --------------------------------------------------------------------------------------
# dense solver:  X U X + X = V      (Lemma B.1)
# --------------------------------------------------------------------------------------
def solve_quadratic_matrix_equation_dense(U, V, jitter=0.0):
    """Solve ``X U X + X = V`` with the dense closed form of Lemma B.1.

    .. math::
        X = 2 V \\left[ I + (I + 4 U V)^{1/2} \\right]^{-1}

    Parameters
    ----------
    U : (D, D) array_like
        Positive semi-definite.
    V : (D, D) array_like
        Positive definite.
    jitter : float
        Optional diagonal jitter added to ``V`` for numerical robustness.

    Returns
    -------
    X : (D, D) array
        Symmetric positive definite solution.
    """
    xp = array_namespace(U, V)
    U = symmetrize(U)
    V = symmetrize(V)
    D = V.shape[-1]
    I = _eye_like(V, D, xp)
    if jitter:
        V = V + jitter * I

    # (I + 4 U V)^{1/2}  -- note UV is not symmetric in general, but I + 4UV has
    # strictly positive eigenvalues (Lemma B.1) so we use its principal sqrt.
    M = I + 4.0 * (U @ V)
    M_sqrt = matrix_sqrt(M)

    A = I + M_sqrt                      # symmetric positive definite
    # X = 2 V A^{-1} = 2 (A^{-1} V^T)^T  with A symmetric
    X = 2.0 * solve_symmetric(A, xp.swapaxes(V, -1, -2))
    X = xp.swapaxes(X, -1, -2)
    return symmetrize(X)


# --------------------------------------------------------------------------------------
# low-rank solver:  U = Q Q^T      (Lemma B.3)
# --------------------------------------------------------------------------------------
def solve_quadratic_matrix_equation_low_rank(V, Q, jitter=0.0):
    """Solve ``X Q Q^T X + X = V`` using the low-rank closed form of Lemma B.3.

    .. math::
        X = V - V^T Q \\left[ \\tfrac{1}{2} I + \\big(Q^T V Q + \\tfrac14 I\\big)^{1/2}
            \\right]^{-2} Q^T V

    Cost is ``O(K D^2 + K^3)`` instead of ``O(D^3)`` for the dense solver, where
    ``K = Q.shape[1]``.

    Parameters
    ----------
    V : (D, D) array_like
        Positive definite.
    Q : (D, K) array_like
        Low-rank factor, ``U = Q Q^T ⪰ 0``.
    jitter : float
        Optional diagonal jitter added to ``V``.

    Returns
    -------
    X : (D, D) array
        Symmetric positive definite solution.
    """
    xp = array_namespace(V, Q)
    V = symmetrize(V)
    D = V.shape[-1]
    K = Q.shape[-1]
    I_D = _eye_like(V, D, xp)
    if jitter:
        V = V + jitter * I_D

    I_K = _eye_like(V, K, xp)

    # Q^T V Q + (1/4) I  is symmetric positive definite (K x K)
    inner = xp.swapaxes(Q, -1, -2) @ V @ Q + 0.25 * I_K
    inner_sqrt = matrix_sqrt(inner)
    M = 0.5 * I_K + inner_sqrt                 # symmetric PD (K x K)

    # M^{-2} = (M^{-1})^T M^{-1} with M symmetric  ->  compute M^{-1} once
    M_inv = inverse_spd(M)
    M_inv2 = symmetrize(M_inv @ M_inv)

    # V^T Q M^{-2} Q^T V
    VtQ = xp.swapaxes(V, -1, -2) @ Q           # (D, K)
    X = V - VtQ @ M_inv2 @ xp.swapaxes(VtQ, -1, -2)
    return symmetrize(X)


# --------------------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------------------
def solve_quadratic_matrix_equation(U, V, Q=None, jitter=0.0, low_rank=None):
    """Solve ``X U X + X = V``.

    If ``Q`` is given (with ``U = Q Q^T``) the fast low-rank formulation of Lemma B.3
    is used; otherwise the dense closed form of Lemma B.1 is used.

    Parameters
    ----------
    U : (D, D) array_like or None
        Positive semi-definite matrix.  May be ``None`` when ``Q`` is provided.
    V : (D, D) array_like
        Positive definite matrix.
    Q : (D, K) array_like, optional
        Low-rank factor such that ``U = Q Q^T``.
    jitter : float
        Diagonal jitter for robustness.
    low_rank : bool, optional
        Force one of the two code paths.  By default the low-rank path is used when
        ``Q`` is provided and ``K < D``.

    Returns
    -------
    X : (D, D) array
    """
    xp = array_namespace(V)
    D = V.shape[-1]
    if low_rank is None:
        low_rank = Q is not None and Q.shape[-1] < D
    if low_rank:
        if Q is None:
            # factor U without importing anything heavy
            U = symmetrize(U)
            w, Vv = xp.linalg.eigh(U)
            w = xp.maximum(w, 0.0)
            keep = w > 1e-12
            Q = Vv * xp.sqrt(w) * keep
        return solve_quadratic_matrix_equation_low_rank(V, Q, jitter=jitter)
    if U is None:
        raise ValueError("U must be provided when using the dense solver.")
    return solve_quadratic_matrix_equation_dense(U, V, jitter=jitter)


def residual(U, V, X):
    """Return ``X U X + X - V`` (useful for unit tests / validation)."""
    xp = array_namespace(U, V, X)
    return symmetrize(X @ U @ X + X) - symmetrize(V)
