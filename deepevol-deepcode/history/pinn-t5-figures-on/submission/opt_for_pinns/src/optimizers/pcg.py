"""NyströmPCG (Algorithm 6) — preconditioned conjugate gradient.

Solves the (regularized) linear system

    (A + mu I) x = b

where ``A`` is a symmetric positive semi-definite operator accessed only through
matrix-vector products (e.g. the PINN Hessian ``H_L(w_k)`` via Hessian-vector
products), using the randomized Nyström approximation ``A ~= U Lambda U^T`` as a
preconditioner.

The preconditioner is (see paper, Algorithm 6 / Section 7.2)

    P^{-1} = (lambda_s + mu) U (Lambda + mu I)^{-1} U^T + (I - U U^T)

where ``lambda_s`` is the smallest retained Nyström eigenvalue (``Lambda[-1]``).
Applying ``P^{-1}`` to a vector ``r`` is done in the low-rank subspace plus the
orthogonal complement, which is cheap since ``U`` has only ``s`` columns.

The routine supports warm-starting from a previous direction ``x0`` (used by
NNCG, which warm-starts PCG with the previous Newton direction ``d_{k-1}``).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


__all__ = ["nystrom_pcg", "apply_preconditioner", "PCGResult"]


class PCGResult:
    """Container for the outcome of a NyströmPCG solve.

    Attributes:
        x: The (approximate) solution vector, shape ``(p,)``.
        iters: Number of PCG iterations performed.
        residual_norm: Final residual norm ``||b - (A + mu I) x||``.
        converged: Whether the tolerance was reached within ``max_iters``.
    """

    __slots__ = ("x", "iters", "residual_norm", "converged")

    def __init__(self, x: torch.Tensor, iters: int, residual_norm: float, converged: bool):
        self.x = x
        self.iters = iters
        self.residual_norm = residual_norm
        self.converged = converged

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PCGResult(iters={self.iters}, residual_norm={self.residual_norm:.3e}, "
            f"converged={self.converged})"
        )


def apply_preconditioner(
    r: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """Apply the Nyström preconditioner ``P^{-1}`` to a vector ``r``.

    ``P^{-1} = (lambda_s + mu) U (Lambda + mu I)^{-1} U^T + (I - U U^T)``

    Args:
        r: Residual vector, shape ``(p,)``.
        U: Orthonormal low-rank basis, shape ``(p, s)``.
        lam: Nyström eigenvalues (descending, non-negative), shape ``(s,)``.
        mu: Damping / regularization parameter.

    Returns:
        ``P^{-1} r``, shape ``(p,)``.
    """
    # Projection coefficients in the low-rank subspace.
    Ut_r = U.transpose(0, 1) @ r  # (s,)
    # (Lambda + mu I)^{-1} U^T r
    inv = Ut_r / (lam + mu)
    # Low-rank part: (lambda_s + mu) U (Lambda + mu I)^{-1} U^T r
    lam_s = lam[-1] if lam.numel() > 0 else torch.zeros((), dtype=r.dtype, device=r.device)
    low_rank = (lam_s + mu) * (U @ inv)
    # Orthogonal complement part: (I - U U^T) r
    complement = r - U @ Ut_r
    return low_rank + complement


def nystrom_pcg(
    A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    mu: float = 1e-2,
    tol: float = 1e-16,
    max_iters: int = 1000,
    x0: Optional[torch.Tensor] = None,
) -> PCGResult:
    """Preconditioned conjugate gradient on ``(A + mu I) x = b`` (Algorithm 6).

    Args:
        A: Matvec callable implementing the symmetric operator ``A`` (e.g. a
            Hessian-vector product). Must accept and return a flat vector of the
            same shape as ``b``.
        b: Right-hand side, shape ``(p,)`` (typically the gradient).
        U: Nyström basis, shape ``(p, s)``.
        lam: Nyström eigenvalues, shape ``(s,)``.
        mu: Damping added to ``A`` (and used inside the preconditioner).
        tol: Relative residual tolerance for convergence.
        max_iters: Maximum number of PCG iterations (``M`` in the paper).
        x0: Optional warm-start vector (e.g. previous Newton direction).

    Returns:
        A :class:`PCGResult` with the solution and diagnostics.
    """
    p = b.shape[0]
    dtype = b.dtype
    device = b.device

    # Ensure the low-rank factors match the working dtype/device.
    U = U.to(dtype=dtype, device=device)
    lam = lam.to(dtype=dtype, device=device)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return A(v) + mu * v

    # Warm start (or zero).
    if x0 is None:
        x = torch.zeros(p, dtype=dtype, device=device)
    else:
        x = x0.to(dtype=dtype, device=device).clone()

    r = b - matvec(x)
    b_norm = torch.linalg.norm(b)
    if b_norm == 0:
        return PCGResult(x, 0, float(torch.linalg.norm(r)), True)

    # Convergence threshold: relative to ||b||.
    threshold = tol * b_norm

    z = apply_preconditioner(r, U, lam, mu)
    p_vec = z.clone()
    rz_old = torch.dot(r, z)

    iters = 0
    converged = False
    for iters in range(1, max_iters + 1):
        Ap = matvec(p_vec)
        pAp = torch.dot(p_vec, Ap)
        if not torch.isfinite(pAp) or pAp <= 0:
            # Breakdown (non-positive curvature or numerical issue): stop.
            break
        alpha = rz_old / pAp
        x = x + alpha * p_vec
        r = r - alpha * Ap

        r_norm = torch.linalg.norm(r)
        if r_norm <= threshold:
            converged = True
            break

        z = apply_preconditioner(r, U, lam, mu)
        rz_new = torch.dot(r, z)
        if not torch.isfinite(rz_new) or rz_old == 0:
            break
        beta = rz_new / rz_old
        p_vec = z + beta * p_vec
        rz_old = rz_new

    residual_norm = float(torch.linalg.norm(b - matvec(x)))
    return PCGResult(x, iters, residual_norm, converged)
