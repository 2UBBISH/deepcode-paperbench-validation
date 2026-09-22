"""Randomized Nyström approximation of a symmetric (possibly indefinite) matrix.

Implements Algorithm 5 from the paper "Challenges in Training PINNs: A Loss
Landscape Perspective".

Given a symmetric matrix ``M`` (accessed only through matrix-vector products,
e.g. Hessian-vector products) and a target rank ``s``, this routine returns an
orthonormal matrix ``U`` (p x s) and a diagonal matrix ``Lambda_hat`` (s x s)
such that ``M ~= U Lambda_hat U^T``.

The algorithm is robust to indefinite ``M``: if the Cholesky factorization of
``Q^T Y_nu`` fails, we fall back to an eigendecomposition and shift by the most
negative eigenvalue (the "fail-safe" branch described in the paper).

Reference: Algorithm 5 (RandomizedNystromApproximation).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def _matvec(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply a matrix (or matvec callable) to a vector/matrix ``v``."""
    if callable(M):
        return M(v)
    return M @ v


def randomized_nystrom_approximation(
    M: Callable[[torch.Tensor], torch.Tensor],
    s: int,
    p: Optional[int] = None,
    eps: float = 1e-16,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomized Nyström approximation (Algorithm 5).

    Parameters
    ----------
    M : callable
        Symmetric linear operator; ``M(V)`` returns ``M @ V`` for a matrix ``V``
        of shape ``(p, k)``. Typically a Hessian-vector-product closure.
    s : int
        Target rank (number of columns of the sketch).
    p : int, optional
        Dimension of the operator. If ``None`` it is inferred from the output of
        a probe matvec.
    eps : float
        Small constant used to compute the shift ``nu``.
    device, dtype : optional
        Device / dtype for the random sketch.
    generator : torch.Generator, optional
        For reproducible random sketches.

    Returns
    -------
    U : torch.Tensor
        Orthonormal matrix of shape ``(p, s)``.
    Lambda_hat : torch.Tensor
        Diagonal matrix of shape ``(s, s)`` with non-negative entries.
    """
    # Infer dimension if not provided.
    if p is None:
        probe = torch.zeros(1, device=device, dtype=dtype)
        # We cannot infer p from a scalar probe; require p explicitly in that case.
        raise ValueError("`p` (operator dimension) must be provided.")

    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64

    # --- Step 1: random sketch S (p x s), orthonormalize -> Q ---
    S = torch.randn(p, s, device=device, dtype=dtype, generator=generator)
    Q, _ = torch.linalg.qr(S, mode="reduced")  # (p, s)

    # --- Step 2: Y = M Q ---
    Y = _matvec(M, Q)  # (p, s)

    # --- Step 3: shift nu = sqrt(p) * eps * ||Y||_2 ---
    norm_Y = torch.linalg.norm(Y, ord=2)
    nu = (p ** 0.5) * eps * norm_Y
    Y_nu = Y + nu * Q  # (p, s)

    # --- Step 4: attempt Cholesky of Q^T Y_nu ---
    QTY = Q.transpose(0, 1) @ Y_nu  # (s, s)
    # Symmetrize to guard against tiny numerical asymmetry.
    QTY = 0.5 * (QTY + QTY.transpose(0, 1))

    lam = 0.0
    try:
        C = torch.linalg.cholesky(QTY)  # lower triangular
        # B = Y C^{-1}
        B = torch.linalg.solve_triangular(C, Y.transpose(0, 1), upper=False).transpose(0, 1)
    except Exception:
        # Fail-safe branch: eigendecomposition of Q^T Y_nu.
        Gamma, W = torch.linalg.eigh(QTY)  # ascending eigenvalues
        lam = float(Gamma.min().item())
        # R = W (Gamma + |lam| I)^{-1/2} W^T
        shifted = Gamma + abs(lam)
        shifted = torch.clamp(shifted, min=eps)
        inv_sqrt = torch.diag(1.0 / torch.sqrt(shifted))
        R = W @ inv_sqrt @ W.transpose(0, 1)
        B = Y @ R  # (p, s)

    # --- Step 5: SVD of B ---
    V_hat, Sigma, _ = torch.linalg.svd(B, full_matrices=False)  # V_hat (p,s), Sigma (s,)

    # --- Step 6: Lambda_hat = max{0, Sigma^2 - (nu + |lam|) I} ---
    diag = Sigma ** 2 - (nu + abs(lam))
    diag = torch.clamp(diag, min=0.0)
    Lambda_hat = torch.diag(diag)

    return V_hat, Lambda_hat


class RandomizedNystromApproximation:
    """Class wrapper around :func:`randomized_nystrom_approximation`.

    Stores the operator and configuration so that repeated calls (e.g. every
    ``F`` iterations of NysNewton-CG) are convenient.
    """

    def __init__(
        self,
        s: int = 60,
        eps: float = 1e-16,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.s = s
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        M: Callable[[torch.Tensor], torch.Tensor],
        p: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return randomized_nystrom_approximation(
            M,
            s=self.s,
            p=p,
            eps=self.eps,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

    def approximate(
        self,
        M: Callable[[torch.Tensor], torch.Tensor],
        p: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self(M, p, generator=generator)


__all__ = [
    "randomized_nystrom_approximation",
    "RandomizedNystromApproximation",
]
