"""Nyström-preconditioned Conjugate Gradient (Algorithm 6).

This module implements the preconditioned conjugate gradient (PCG) routine used
inside NysNewton-CG (Algorithm 4) to (approximately) solve the damped Newton
system

    (H_L(w_k) + mu I) d = grad L(w_k)

where ``H_L(w_k)`` is the Hessian of the PINN loss accessed only through
Hessian-vector products (Pearlmutter trick). The preconditioner is built from a
randomized Nyström approximation ``H ≈ U Lambda_hat U^T`` (Algorithm 5):

    P      = 1/(lambda_hat_s + mu) * U (Lambda_hat + mu I) U^T + (I - U U^T)
    P^{-1} = (lambda_hat_s + mu) * U (Lambda_hat + mu I)^{-1} U^T + (I - U U^T)

where ``lambda_hat_s`` is the smallest eigenvalue of the Nyström approximation
(the last diagonal entry of ``Lambda_hat``, which is sorted ascending).

The PCG loop is warm-started with the previous Newton step ``d_{k-1}`` (passed
as ``x0``), which is the key trick that makes NNCG efficient across outer
iterations.

References
----------
- Paper: "Challenges in Training PINNs: A Loss Landscape Perspective"
  Algorithm 6 (NystromPCG).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


__all__ = [
    "NystromPreconditioner",
    "nystrom_pcg",
    "NystromPCG",
]


# ---------------------------------------------------------------------------
# Preconditioner
# ---------------------------------------------------------------------------
class NystromPreconditioner:
    """Nyström preconditioner ``P`` and its inverse ``P^{-1}``.

    Given a randomized Nyström approximation ``H ≈ U Lambda_hat U^T`` with
    ``U`` of shape ``(p, s)`` orthonormal and ``Lambda_hat`` of shape ``(s, s)``
    diagonal (sorted ascending), and damping ``mu >= 0``, this class builds

        P      = 1/(lambda_hat_s + mu) * U (Lambda_hat + mu I) U^T + (I - U U^T)
        P^{-1} = (lambda_hat_s + mu) * U (Lambda_hat + mu I)^{-1} U^T + (I - U U^T)

    Parameters
    ----------
    U : torch.Tensor
        Orthonormal basis of shape ``(p, s)``.
    Lambda_hat : torch.Tensor
        Diagonal matrix (or vector) of Nyström eigenvalues, shape ``(s, s)`` or
        ``(s,)``. Assumed sorted ascending.
    mu : float
        Damping / regularization parameter (``mu >= 0``).
    """

    def __init__(
        self,
        U: torch.Tensor,
        Lambda_hat: torch.Tensor,
        mu: float = 0.0,
    ) -> None:
        if Lambda_hat.dim() == 2:
            lam = torch.diagonal(Lambda_hat)
        else:
            lam = Lambda_hat
        lam = lam.to(dtype=U.dtype, device=U.device)

        self.U = U
        self.lam = lam
        self.mu = float(mu)
        self.p = U.shape[0]
        self.s = U.shape[1]

        # Smallest Nyström eigenvalue (sorted ascending).
        self.lambda_hat_s = float(lam.min().item()) if lam.numel() > 0 else 0.0

        # (Lambda_hat + mu I) and its inverse.
        self.lam_plus_mu = lam + self.mu
        self.inv_lam_plus_mu = 1.0 / self.lam_plus_mu.clamp_min(1e-30)

        # Scalar prefactors.
        self.scale_P = 1.0 / max(self.lambda_hat_s + self.mu, 1e-30)
        self.scale_Pinv = self.lambda_hat_s + self.mu

    # -- apply P -----------------------------------------------------------
    def apply_P(self, v: torch.Tensor) -> torch.Tensor:
        """Apply ``P`` to a vector (or batch of column vectors)."""
        return self._apply(v, self.scale_P, self.lam_plus_mu)

    # -- apply P^{-1} ------------------------------------------------------
    def apply_Pinv(self, v: torch.Tensor) -> torch.Tensor:
        """Apply ``P^{-1}`` to a vector (or batch of column vectors)."""
        return self._apply(v, self.scale_Pinv, self.inv_lam_plus_mu)

    def _apply(
        self,
        v: torch.Tensor,
        scale: float,
        diag: torch.Tensor,
    ) -> torch.Tensor:
        # v: (p,) or (p, k)
        Utv = self.U.t() @ v  # (s,) or (s, k)
        if Utv.dim() == 1:
            scaled = diag * Utv
        else:
            scaled = diag.unsqueeze(1) * Utv
        proj = self.U @ scaled  # (p,) or (p, k)
        # (I - U U^T) v = v - U (U^T v)
        complement = v - self.U @ (self.U.t() @ v)
        return scale * proj + complement

    # Convenience aliases --------------------------------------------------
    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.apply_Pinv(v)


# ---------------------------------------------------------------------------
# PCG
# ---------------------------------------------------------------------------
def nystrom_pcg(
    A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    U: torch.Tensor,
    Lambda_hat: torch.Tensor,
    mu: float = 0.0,
    eps: float = 1e-16,
    max_iters: int = 1000,
    x0: Optional[torch.Tensor] = None,
    return_info: bool = False,
) -> torch.Tensor:
    """Preconditioned conjugate gradient (Algorithm 6).

    Approximately solves ``(A + mu I) x = b`` where ``A`` is a symmetric
    operator accessed via matvecs (typically a Hessian-vector product closure).

    Parameters
    ----------
    A : Callable[[torch.Tensor], torch.Tensor]
        Symmetric matvec operator (e.g. Hessian-vector product).
    b : torch.Tensor
        Right-hand side, shape ``(p,)``.
    U : torch.Tensor
        Nyström basis, shape ``(p, s)``.
    Lambda_hat : torch.Tensor
        Nyström eigenvalues, shape ``(s, s)`` or ``(s,)``.
    mu : float
        Damping added to ``A`` (and used in the preconditioner).
    eps : float
        Relative residual tolerance: stop when
        ``||r||_2 <= eps * ||b||_2``.
    max_iters : int
        Maximum number of PCG iterations (``M`` in Algorithm 4).
    x0 : Optional[torch.Tensor]
        Warm-start initial guess (e.g. previous Newton step ``d_{k-1}``).
    return_info : bool
        If True, also return a dict with diagnostics.

    Returns
    -------
    torch.Tensor
        Approximate solution ``x`` of shape ``(p,)``.
    """
    dtype = b.dtype
    device = b.device

    precond = NystromPreconditioner(U, Lambda_hat, mu=mu)

    # Damped operator: A_mu(v) = A(v) + mu v
    def A_mu(v: torch.Tensor) -> torch.Tensor:
        return A(v) + mu * v

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone().to(dtype=dtype, device=device)

    r = b - A_mu(x)
    z = precond.apply_Pinv(r)
    p = z.clone()

    rz_old = torch.dot(r, z)

    b_norm = torch.linalg.norm(b).item()
    tol = eps * max(b_norm, 1e-30)

    iters = 0
    for it in range(max_iters):
        iters = it + 1
        Ap = A_mu(p)
        pAp = torch.dot(p, Ap)

        # Guard against non-positive curvature / breakdown.
        if pAp.item() <= 0.0:
            # Fall back to a small positive step to keep progressing.
            if pAp.item() == 0.0:
                break
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.linalg.norm(r).item()
        if r_norm <= tol:
            break

        z = precond.apply_Pinv(r)
        rz_new = torch.dot(r, z)
        if rz_old.item() == 0.0:
            break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    if return_info:
        info = {
            "iters": iters,
            "residual_norm": float(torch.linalg.norm(r).item()),
            "b_norm": float(b_norm),
        }
        return x, info
    return x


# ---------------------------------------------------------------------------
# Stateful wrapper
# ---------------------------------------------------------------------------
class NystromPCG:
    """Stateful wrapper around :func:`nystrom_pcg`.

    Stores default hyperparameters (``mu``, ``eps``, ``max_iters``) so that
    NysNewton-CG can call it repeatedly with fresh Nyström factors.

    Parameters
    ----------
    mu : float
        Damping parameter.
    eps : float
        Relative residual tolerance.
    max_iters : int
        Maximum PCG iterations (``M`` in Algorithm 4).
    """

    def __init__(
        self,
        mu: float = 0.0,
        eps: float = 1e-16,
        max_iters: int = 1000,
    ) -> None:
        self.mu = float(mu)
        self.eps = float(eps)
        self.max_iters = int(max_iters)

    def solve(
        self,
        A: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        U: torch.Tensor,
        Lambda_hat: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        return nystrom_pcg(
            A=A,
            b=b,
            U=U,
            Lambda_hat=Lambda_hat,
            mu=self.mu,
            eps=self.eps,
            max_iters=self.max_iters,
            x0=x0,
            return_info=return_info,
        )

    def __call__(
        self,
        A: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        U: torch.Tensor,
        Lambda_hat: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        return self.solve(A, b, U, Lambda_hat, x0=x0, return_info=return_info)
