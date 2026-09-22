"""Randomized Nyström approximation (Algorithm 5 of the paper).

Given a symmetric positive semi-definite operator ``M`` (accessed only through
matrix-vector products) this module computes a low-rank approximation

    M ≈ V Λ Vᵀ

where ``V`` has orthonormal columns and ``Λ`` is a diagonal matrix of
non-negative eigenvalues.  The implementation follows Algorithm 5
(``RandomizedNystromApproximation``) of

    "Challenges in Training PINNs: A Loss Landscape Perspective"

The routine is written so that it can operate on either a dense matrix or a
callable ``matvec`` (used for Hessian-vector products of the PINN loss).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def _as_matvec(M) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable ``v -> M v`` for a matrix or an existing matvec."""
    if callable(M):
        return M

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return M @ v

    return matvec


def randomized_nystrom_approximation(
    M,
    s: int,
    p: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    shift: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomized Nyström approximation of a symmetric PSD operator.

    Parameters
    ----------
    M:
        Either a dense ``(p, p)`` tensor/matrix or a callable implementing the
        matrix-vector product ``v -> M v``.
    s:
        Target rank of the approximation (number of random test vectors).
    p:
        Dimension of the operator.  Required when ``M`` is a callable.
    dtype, device:
        Working precision / device for the random sketch.
    generator:
        Optional ``torch.Generator`` for reproducible sketching.
    shift:
        Optional non-negative shift added to the eigenvalues before clipping
        (used by the Cholesky fallback path).

    Returns
    -------
    (V, Lambda):
        ``V`` is ``(p, s)`` with orthonormal columns and ``Lambda`` is a
        ``(s,)`` tensor of non-negative eigenvalues (descending).
    """
    matvec = _as_matvec(M)

    if p is None:
        if callable(M):
            raise ValueError("`p` must be provided when `M` is a callable.")
        p = int(M.shape[0])

    if device is None:
        if not callable(M):
            device = M.device
        else:
            device = torch.device("cpu")

    # --- Step 1: random sketch S in R^{p x s} -----------------------------
    S = torch.randn(p, s, dtype=dtype, device=device, generator=generator)

    # --- Step 2: orthonormal basis Q of the range of S --------------------
    Q, _ = torch.linalg.qr(S, mode="reduced")

    # --- Step 3: Y = M Q (via matvec) -------------------------------------
    Y = matvec(Q)
    Y = Y.to(dtype=dtype, device=device)

    # --- Step 4: regularization nu = sqrt(p) * eps * ||Y||_2 --------------
    eps = torch.finfo(dtype).eps
    Y_norm = torch.linalg.norm(Y, ord=2)
    nu = float(torch.sqrt(torch.tensor(float(p), dtype=dtype, device=device)) * eps * Y_norm)

    # --- Step 5: Y_nu = Y + nu Q ------------------------------------------
    Y_nu = Y + nu * Q

    # --- Step 6: Cholesky of Qᵀ Y_nu (with eig fallback) ------------------
    C = Q.t() @ Y_nu
    C = 0.5 * (C + C.t())  # symmetrize to guard against round-off

    try:
        L = torch.linalg.cholesky(C)
        B = torch.linalg.solve_triangular(L, Y.t(), upper=False).t()
    except Exception:
        # Fallback: eigendecomposition with a small shift.
        lam, Vc = torch.linalg.eigh(C)
        lam = torch.clamp(lam, min=0.0) + shift
        inv_sqrt = torch.diag(1.0 / torch.sqrt(lam))
        B = Y @ Vc @ inv_sqrt

    # --- Step 7: SVD of B --------------------------------------------------
    U, Sigma, _ = torch.linalg.svd(B, full_matrices=False)

    # --- Step 8: eigenvalues Lambda = max(0, Sigma^2 - (nu + |shift|)) ----
    lam = torch.clamp(Sigma.pow(2) - (nu + abs(shift)), min=0.0)

    # Sort descending (SVD already returns descending Sigma).
    order = torch.argsort(lam, descending=True)
    lam = lam[order]
    U = U[:, order]

    return U, lam


class NystromApproximation:
    """Convenience wrapper storing the low-rank factors ``(U, Lambda)``."""

    def __init__(self, U: torch.Tensor, lam: torch.Tensor):
        self.U = U
        self.lam = lam

    @property
    def rank(self) -> int:
        return int(self.U.shape[1])

    def to(self, dtype: torch.dtype, device: torch.device) -> "NystromApproximation":
        return NystromApproximation(self.U.to(dtype=dtype, device=device),
                                    self.lam.to(dtype=dtype, device=device))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"NystromApproximation(rank={self.rank}, lam_max={float(self.lam.max()):.3e})"


__all__ = ["randomized_nystrom_approximation", "NystromApproximation"]
