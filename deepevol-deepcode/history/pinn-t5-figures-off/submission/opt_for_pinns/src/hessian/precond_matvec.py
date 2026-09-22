"""Preconditioned Hessian matvec (Algorithm 3 of the paper).

This module implements the matrix-vector product with the *preconditioned*
Hessian

    H_tilde_k^T H_L(w_k) H_tilde_k

where ``H_tilde_k`` is the implicit L-BFGS preconditioner built from the stored
curvature pairs ``{(s_i, y_i, rho_i)}`` (see ``lbfgs_unroll.py`` / Algorithm 2).

The operator ``H_tilde_k^T H_L H_tilde_k`` is symmetric positive semi-definite
(whenever ``H_L`` is PSD), which makes it suitable for Stochastic Lanczos
Quadrature (SLQ) based spectral density estimation (Figures 3 and 7 of the
paper).

Algorithm 3 (Preconditioned Hessian matvec)
-------------------------------------------
Given a vector ``v`` of length ``p + m`` (``p`` = number of parameters,
``m`` = number of stored curvature pairs), split it as

    v = [v_1 ; v_2],   v_1 in R^p,  v_2 in R^m.

Then

    H_tilde_k^T H_L H_tilde_k v
        = [ H_tilde_k^T H_L (H_tilde_k v_1) + H_tilde_k^T H_L (V_tilde v_2) ; V_tilde^T H_L (H_tilde_k v_1) ]

where ``V_tilde`` is the ``(p, m)`` matrix whose columns are the L-BFGS
correction vectors ``[rho_i s_i, y_i]`` (see ``build_preconditioner_columns``).

Because ``H_tilde_k`` is only available implicitly (via the two-loop
recursion), we apply it through ``lbfgs_two_loop``.  The correction columns
``V_tilde`` are applied explicitly.

The resulting operator is symmetric, so it can be fed directly into the SLQ
routine in ``spectral_density.py``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from .lbfgs_unroll import (
    LBFGSHistory,
    build_preconditioner_columns,
    lbfgs_two_loop,
)

__all__ = [
    "PreconditionedHessianOperator",
    "preconditioned_hessian_matvec",
    "build_preconditioned_matvec",
]


def _apply_htilde(v: torch.Tensor, history: LBFGSHistory, gamma: Optional[float] = None) -> torch.Tensor:
    """Apply the implicit L-BFGS preconditioner ``H_tilde_k`` to ``v``."""
    return lbfgs_two_loop(v, history, gamma=gamma)


def _apply_htilde_T(v: torch.Tensor, history: LBFGSHistory, gamma: Optional[float] = None) -> torch.Tensor:
    """Apply ``H_tilde_k^T`` to ``v``.

    ``H_tilde_k`` is symmetric in exact arithmetic (it is a product of
    symmetric rank-one updates), so ``H_tilde_k^T == H_tilde_k``.  We keep a
    separate entry point for clarity and to allow future non-symmetric
    variants.
    """
    return lbfgs_two_loop(v, history, gamma=gamma)


def preconditioned_hessian_matvec(
    v: torch.Tensor,
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    history: LBFGSHistory,
    p: int,
    m: Optional[int] = None,
    gamma: Optional[float] = None,
) -> torch.Tensor:
    """Compute ``H_tilde_k^T H_L H_tilde_k v`` (Algorithm 3).

    Parameters
    ----------
    v : torch.Tensor
        Input vector of length ``p + m`` (stacked ``[v_1 ; v_2]``).
    hvp_fn : Callable[[Tensor], Tensor]
        Matrix-free Hessian-vector product ``w -> H_L(w) w`` operating on
        vectors of length ``p``.
    history : LBFGSHistory
        Stored L-BFGS curvature pairs.
    p : int
        Number of model parameters.
    m : int, optional
        Number of curvature pairs.  Defaults to ``len(history)``.
    gamma : float, optional
        L-BFGS scaling factor (``gamma_k = s^T y / y^T y``).  If ``None`` the
        two-loop recursion uses its own default.

    Returns
    -------
    torch.Tensor
        Result vector of length ``p + m``.
    """
    if m is None:
        m = len(history)

    v = v.reshape(-1)
    if v.numel() != p + m:
        raise ValueError(
            f"Expected input of length p + m = {p + m}, got {v.numel()}."
        )

    v1 = v[:p]
    v2 = v[p:]

    # Explicit correction columns V_tilde of shape (p, m).
    V = build_preconditioner_columns(history, dtype=v.dtype, device=v.device)
    if V.shape[1] != m:
        # history may have fewer pairs than requested; truncate v2 accordingly.
        m = V.shape[1]
        v2 = v2[:m]

    # H_tilde_k v_1  (implicit, via two-loop recursion)
    h_v1 = _apply_htilde(v1, history, gamma=gamma)

    # V_tilde v_2  (explicit)
    if m > 0:
        Vv2 = V @ v2
    else:
        Vv2 = torch.zeros_like(v1)

    # Combined vector fed through the Hessian.
    combined = h_v1 + Vv2

    # H_L (H_tilde_k v_1 + V_tilde v_2)
    H_combined = hvp_fn(combined)

    # Top block: H_tilde_k^T H_L H_tilde_k v_1 + H_tilde_k^T H_L V_tilde v_2
    top = _apply_htilde_T(H_combined, history, gamma=gamma)

    # Bottom block: V_tilde^T H_L (H_tilde_k v_1 + V_tilde v_2)
    if m > 0:
        bottom = V.t() @ H_combined
    else:
        bottom = torch.zeros(m, dtype=v.dtype, device=v.device)

    return torch.cat([top, bottom], dim=0)


class PreconditionedHessianOperator:
    """Callable matrix-free operator for ``H_tilde_k^T H_L H_tilde_k``.

    The operator acts on vectors of length ``p + m`` and is symmetric, so it
    can be passed directly to the SLQ spectral-density estimator.

    Parameters
    ----------
    hvp_fn : Callable[[Tensor], Tensor]
        Hessian-vector product closure ``w -> H_L(w) w`` on length-``p``
        vectors.
    history : LBFGSHistory
        Stored L-BFGS curvature pairs.
    p : int
        Number of model parameters.
    gamma : float, optional
        L-BFGS scaling factor.
    """

    def __init__(
        self,
        hvp_fn: Callable[[torch.Tensor], torch.Tensor],
        history: LBFGSHistory,
        p: int,
        gamma: Optional[float] = None,
    ) -> None:
        self.hvp_fn = hvp_fn
        self.history = history
        self.p = int(p)
        self.m = len(history)
        self.gamma = gamma

    @property
    def shape(self) -> Tuple[int, int]:
        n = self.p + self.m
        return (n, n)

    @property
    def numel(self) -> int:
        return self.p + self.m

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return preconditioned_hessian_matvec(
            v,
            self.hvp_fn,
            self.history,
            self.p,
            m=self.m,
            gamma=self.gamma,
        )

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.matvec(v)


def build_preconditioned_matvec(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    history: LBFGSHistory,
    p: int,
    gamma: Optional[float] = None,
) -> PreconditionedHessianOperator:
    """Convenience factory returning a :class:`PreconditionedHessianOperator`."""
    return PreconditionedHessianOperator(hvp_fn, history, p, gamma=gamma)
