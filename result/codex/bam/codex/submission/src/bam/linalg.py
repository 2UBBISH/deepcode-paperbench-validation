"""Linear algebra helpers used by batch-and-match (BaM).

The central object here is the quadratic matrix equation

    X U X + X = V,          U >= 0,  V > 0           (eq. (9) of the paper)

whose symmetric positive-definite solution is (Lemma B.1)

    X = 2 V [ I + (I + 4 U V)^{1/2} ]^{-1}.           (eq. (12) of the paper)

Two solvers are provided:

* :func:`solve_quadratic_matrix_eq` -- the general ``O(D^3)`` solver.
* :func:`solve_quadratic_matrix_eq_low_rank` -- the ``O(D^2 B + B^3)`` solver
  of Lemma B.3, which applies when ``U = Q Q^T`` with ``Q`` of shape
  ``(D, K)`` and ``K << D`` (this is the regime ``B < D`` for BaM, where
  ``U = lambda * Gamma + lambda/(1+lambda) gbar gbar^T`` has rank at most ``B``).

Both are written with plain NumPy so that they can be used inside as well as
outside of JAX-transformed code.
"""

from __future__ import annotations

import numpy as np


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part ``(A + A^T) / 2`` of a matrix."""
    A = np.asarray(A, dtype=np.float64)
    return 0.5 * (A + A.T)


def psd_eigh(A: np.ndarray, clip: float = 0.0):
    """Eigendecomposition of a symmetric PSD matrix with clipped eigenvalues.

    Returns ``(eigenvalues, eigenvectors)`` with ``A ~= Q diag(w) Q^T`` and
    ``w >= clip``.  Negative eigenvalues that appear only because of round-off
    are set to ``clip`` (usually zero).
    """
    A = symmetrize(A)
    w, Q = np.linalg.eigh(A)
    if clip is not None:
        w = np.maximum(w, clip)
    return w, Q


def psd_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root ``A^{1/2}`` of a PSD matrix."""
    w, Q = psd_eigh(A)
    return (Q * np.sqrt(w)) @ Q.T


def psd_pinv(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Moore-Penrose pseudo-inverse of a symmetric PSD matrix."""
    A = symmetrize(A)
    w, Q = np.linalg.eigh(A)
    wmax = max(float(w.max()), 0.0) if w.size else 0.0
    cut = tol * max(wmax, 1.0)
    w_inv = np.where(w > cut, 1.0 / np.where(w > cut, w, 1.0), 0.0)
    return (Q * w_inv) @ Q.T


def solve_quadratic_matrix_eq(U: np.ndarray, V: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Solve ``X U X + X = V`` for ``U >= 0`` and ``V > 0`` (Lemma B.1).

    Uses the symmetric form of the closed-form solution,

        X = 2 V^{1/2} [ I + (I + 4 V^{1/2} U V^{1/2})^{1/2} ]^{-1} V^{1/2},

    which is numerically more stable than forming the (nonsymmetric) product
    ``U V``.  The returned matrix is symmetric and positive definite.
    """
    U = symmetrize(U)
    V = symmetrize(V)
    if jitter:
        V = V + jitter * np.eye(V.shape[0])
    w_v, Q_v = psd_eigh(V, clip=0.0)
    V_sqrt = (Q_v * np.sqrt(w_v)) @ Q_v.T
    S = symmetrize(V_sqrt @ U @ V_sqrt)
    w_s, Q_s = psd_eigh(S, clip=0.0)
    # [ I + (I + 4 S)^{1/2} ]^{-1} in the eigenbasis of S
    inv_scale = 1.0 / (1.0 + np.sqrt(1.0 + 4.0 * w_s))
    X = 2.0 * (V_sqrt @ Q_s * inv_scale) @ Q_s.T @ V_sqrt
    return symmetrize(X)


def solve_quadratic_matrix_eq_low_rank(Q: np.ndarray, V: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Low-rank solver of Lemma B.3 for ``U = Q Q^T``.

        X = V - V Q [ 1/2 I + (Q^T V Q + 1/4 I)^{1/2} ]^{-2} Q^T V

    with cost ``O(D^2 K + K^3)`` for ``Q`` of shape ``(D, K)``.
    """
    Q = np.asarray(Q, dtype=np.float64)
    V = symmetrize(V)
    if jitter:
        V = V + jitter * np.eye(V.shape[0])
    VQ = V @ Q
    M = symmetrize(Q.T @ VQ)
    w_m, Q_m = psd_eigh(M, clip=0.0)
    w_inv = 1.0 / (0.5 + np.sqrt(w_m + 0.25))
    # (1/2 I + (M + 1/4 I)^{1/2})^{-2}
    scale = w_inv**2
    W = (Q_m * scale) @ Q_m.T
    X = V - VQ @ W @ VQ.T
    return symmetrize(X)


def low_rank_factor_of_U(Gamma: np.ndarray, gbar: np.ndarray, lam: float, batch: np.ndarray | None = None) -> np.ndarray:
    """Return ``Q`` with ``Q Q^T = U = lam * Gamma + lam/(1+lam) gbar gbar^T``.

    ``Gamma = (1/B) sum_b (g_b - gbar)(g_b - gbar)^T`` so it has rank at most
    ``B - 1``.  If the centered scores ``batch`` (shape ``(B, D)``) are supplied,
    the factor is formed directly from them, i.e.

        Q = [ sqrt(lam/B) (g_1 - gbar), ..., sqrt(lam/B) (g_B - gbar),
              sqrt(lam/(1+lam)) gbar ]  ,

    which is exact and costs no eigendecomposition.  Otherwise a factor of
    ``Gamma`` is obtained from its eigendecomposition.
    """
    lam = float(lam)
    blocks = []
    if batch is not None and lam > 0.0:
        centered = np.asarray(batch, dtype=np.float64) - np.asarray(gbar, dtype=np.float64)
        blocks.append(np.sqrt(lam / centered.shape[0]) * centered.T)
    elif Gamma is not None and lam > 0.0:
        w, Q = psd_eigh(Gamma, clip=0.0)
        keep = w > 1e-14 * max(float(w.max()), 1.0)
        blocks.append(np.sqrt(lam) * (Q[:, keep] * np.sqrt(w[keep])))
    # gbar gbar^T is exactly rank one (including gbar = 0, which contributes nothing)
    gbar = np.asarray(gbar, dtype=np.float64)
    if lam > 0.0 and np.any(gbar != 0.0):
        blocks.append(np.sqrt(lam / (1.0 + lam)) * gbar[:, None])
    if not blocks:
        return np.zeros((Gamma.shape[0] if Gamma is not None else gbar.shape[0], 0))
    return np.concatenate(blocks, axis=1)


def ensure_positive_definite(A: np.ndarray, abs_floor: float = 1e-12, rel_floor: float = 1e-12) -> np.ndarray:
    """Project a symmetric matrix onto the positive-definite cone.

    The BaM update is positive definite in exact arithmetic (Lemma B.2), but
    when the matrices involved have eigenvalues spanning many orders of
    magnitude (which happens for the ill-conditioned GP posterior of
    Section 5.2) round-off can produce slightly negative eigenvalues of order
    ``eps * lambda_max``.  Such eigenvalues are floored at
    ``max(abs_floor, rel_floor * lambda_max)``; everything else is untouched.
    """
    A = symmetrize(A)
    w, Q = np.linalg.eigh(A)
    floor = max(float(abs_floor), float(rel_floor) * max(float(w.max()), 0.0))
    if float(w.min()) >= floor:
        return A
    w = np.maximum(w, floor)
    return symmetrize((Q * w) @ Q.T)
