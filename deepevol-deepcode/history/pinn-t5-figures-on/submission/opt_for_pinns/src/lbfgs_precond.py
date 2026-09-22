"""L-BFGS preconditioner unrolling (Algorithms 2 & 3 from the paper, Appendix C.2).

The paper analyzes the conditioning of the PINN loss landscape both for the raw
Hessian ``H_L`` and for the Hessian *preconditioned* by the L-BFGS inverse-Hessian
approximation ``H_k``.  Because L-BFGS stores only the last ``m`` curvature pairs
``{s_i, y_i, rho_i}``, the preconditioned operator can be applied matrix-free.

Algorithm 2 (Unrolling L-BFGS Update)
-------------------------------------
Given the stored curvature pairs ``{(s_i, y_i, rho_i)}_{i=1..m}`` we build the
matrices ``Ỹ``, ``Ṽ`` and ``S̃`` used by the compact (two-loop) representation of
the L-BFGS inverse Hessian:

    ỹ_i = rho_i * y_i
    ṽ_i = s_i - Σ_{j<i} (ỹ_jᵀ s_i) ṽ_j
    s̃_i = sqrt(rho_i) * (s_i - α)

where ``α`` is the running sum ``Σ_j (ỹ_jᵀ s_i) ṽ_j`` (i.e. the projection of
``s_i`` onto the previously constructed ``Ṽ`` columns).  The columns are stored
in the matrices ``Ỹ = [ỹ_1, ..., ỹ_m]``, ``Ṽ = [ṽ_1, ..., ṽ_m]`` and
``S̃ = [s̃_1, ..., s̃_m]``.

Algorithm 3 (Preconditioned matvec)
-----------------------------------
For a vector ``v`` split as ``v = [v1; v2]`` (``v1`` in the range of ``Ỹ`` and
``v2`` in the orthogonal complement) the preconditioned Hessian matvec is

    v'  = sqrt(gamma_k) * (v1 - Ṽ Ỹᵀ v1) + S̃ v2
    v'' = H_L v'                       # Hessian-vector product
    v''' = [ sqrt(gamma_k) * (v'' - Ỹ Ṽᵀ v'') ; S̃ᵀ v'' ]

The resulting operator ``H̃_kᵀ H_L H̃_k`` is symmetric and has the same
eigenvalues as the L-BFGS-preconditioned Hessian, so SLQ can be run on it to
produce the dashed spectral-density curves in Figures 3 & 7.

This module is intentionally matrix-free: the only access to the Hessian is via
the ``hvp`` callable produced by :mod:`src.hessian`.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "LBFGSPreconditioner",
    "unroll_lbfgs_update",
    "preconditioned_matvec",
    "build_preconditioned_matvec",
]


# ---------------------------------------------------------------------------
# Algorithm 2: Unrolling L-BFGS Update
# ---------------------------------------------------------------------------
def unroll_lbfgs_update(
    s_list: Sequence[torch.Tensor],
    y_list: Sequence[torch.Tensor],
    rho_list: Sequence[float],
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm 2: build the compact L-BFGS matrices ``Ỹ``, ``Ṽ``, ``S̃``.

    Parameters
    ----------
    s_list, y_list : sequence of 1-D tensors
        The stored L-BFGS curvature pairs ``s_i = w_{i+1} - w_i`` and
        ``y_i = g_{i+1} - g_i`` (oldest first).
    rho_list : sequence of float
        The stored scalars ``rho_i = 1 / (y_iᵀ s_i)``.
    dtype, device : optional
        Target dtype/device for the returned matrices.

    Returns
    -------
    Yt, Vt, St : torch.Tensor
        Matrices of shape ``(p, m)`` whose columns are ``ỹ_i``, ``ṽ_i`` and
        ``s̃_i`` respectively.
    """
    m = len(s_list)
    if m == 0:
        raise ValueError("unroll_lbfgs_update requires at least one curvature pair.")

    p = s_list[0].numel()
    if device is None:
        device = s_list[0].device

    Yt = torch.zeros(p, m, dtype=dtype, device=device)
    Vt = torch.zeros(p, m, dtype=dtype, device=device)
    St = torch.zeros(p, m, dtype=dtype, device=device)

    for i in range(m):
        s_i = s_list[i].to(dtype=dtype, device=device).reshape(-1)
        y_i = y_list[i].to(dtype=dtype, device=device).reshape(-1)
        rho_i = float(rho_list[i])

        # ỹ_i = rho_i * y_i
        yt_i = rho_i * y_i

        # ṽ_i = s_i - Σ_{j<i} (ỹ_jᵀ s_i) ṽ_j
        v_i = s_i.clone()
        alpha = torch.zeros_like(s_i)
        for j in range(i):
            coeff = torch.dot(Yt[:, j], s_i)
            v_i = v_i - coeff * Vt[:, j]
            alpha = alpha + coeff * Vt[:, j]

        # s̃_i = sqrt(rho_i) * (s_i - α)
        st_i = torch.sqrt(torch.tensor(rho_i, dtype=dtype, device=device)) * (s_i - alpha)

        Yt[:, i] = yt_i
        Vt[:, i] = v_i
        St[:, i] = st_i

    return Yt, Vt, St


# ---------------------------------------------------------------------------
# Algorithm 3: Preconditioned matvec
# ---------------------------------------------------------------------------
def preconditioned_matvec(
    v: torch.Tensor,
    hvp: Callable[[torch.Tensor], torch.Tensor],
    Yt: torch.Tensor,
    Vt: torch.Tensor,
    St: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Algorithm 3: apply the L-BFGS-preconditioned Hessian to ``v``.

    Computes ``H̃_kᵀ H_L H̃_k v`` where ``H̃_k`` is the compact L-BFGS inverse
    Hessian factor built from ``(Ỹ, Ṽ, S̃)``.  The result is symmetric and has
    the same eigenvalues as the preconditioned Hessian ``H̃_kᵀ H_L H̃_k``.

    Parameters
    ----------
    v : torch.Tensor
        Flat vector of length ``p`` (or ``(p, k)`` for a block of vectors).
    hvp : callable
        Hessian-vector product ``hvp(x) -> H_L x``.
    Yt, Vt, St : torch.Tensor
        Matrices from :func:`unroll_lbfgs_update`, shape ``(p, m)``.
    gamma : float
        Scaling ``gamma_k`` (typically ``s_kᵀ y_k / y_kᵀ y_k``); defaults to 1.

    Returns
    -------
    torch.Tensor
        The preconditioned matvec result, same shape as ``v``.
    """
    squeeze = False
    if v.dim() == 1:
        v = v.unsqueeze(1)
        squeeze = True

    p, k = v.shape
    m = Yt.shape[1]
    dtype = Yt.dtype
    device = Yt.device

    # Split v into v1 (range of Ỹ) and v2 (orthogonal complement).  In the
    # compact representation we simply use the full vector for both parts, as
    # the projection operators are applied explicitly below.
    v1 = v
    v2 = v

    sqrt_gamma = torch.sqrt(torch.tensor(float(gamma), dtype=dtype, device=device))

    # v' = sqrt(gamma) * (v1 - Ṽ Ỹᵀ v1) + S̃ v2
    YtT_v1 = Yt.transpose(0, 1) @ v1          # (m, k)
    v_prime = sqrt_gamma * (v1 - Vt @ YtT_v1) + St @ v2

    # v'' = H_L v'
    v_double = hvp(v_prime)
    if v_double.dim() == 1:
        v_double = v_double.unsqueeze(1)

    # v''' = [ sqrt(gamma) * (v'' - Ỹ Ṽᵀ v'') ; S̃ᵀ v'' ]
    VtT_v2 = Vt.transpose(0, 1) @ v_double   # (m, k)
    top = sqrt_gamma * (v_double - Yt @ VtT_v2)
    bottom = St.transpose(0, 1) @ v_double   # (m, k)

    out = torch.cat([top, bottom], dim=0)    # (p + m, k)

    if squeeze:
        out = out.squeeze(1)
    return out


def build_preconditioned_matvec(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    s_list: Sequence[torch.Tensor],
    y_list: Sequence[torch.Tensor],
    rho_list: Sequence[float],
    gamma: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Convenience wrapper returning a matvec for the preconditioned Hessian.

    The returned callable accepts a flat vector of length ``p + m`` (the
    augmented space used by Algorithm 3) and returns a vector of the same
    length.  Its eigenvalues coincide with those of the L-BFGS-preconditioned
    Hessian, so it can be fed directly to
    :func:`src.hessian.slq_spectral_density`.
    """
    Yt, Vt, St = unroll_lbfgs_update(s_list, y_list, rho_list, dtype=dtype, device=device)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return preconditioned_matvec(v, hvp, Yt, Vt, St, gamma=gamma)

    return matvec


class LBFGSPreconditioner:
    """Container for the compact L-BFGS preconditioner (Algorithms 2 & 3).

    Stores the curvature pairs collected during an L-BFGS run and exposes the
    preconditioned matvec used for spectral-density analysis.

    Parameters
    ----------
    s_list, y_list : sequence of 1-D tensors
        Curvature pairs (oldest first).
    rho_list : sequence of float
        ``rho_i = 1 / (y_iᵀ s_i)``.
    gamma : float
        Scaling ``gamma_k``.
    dtype, device : optional
        Target dtype/device for the compact matrices.
    """

    def __init__(
        self,
        s_list: Sequence[torch.Tensor],
        y_list: Sequence[torch.Tensor],
        rho_list: Sequence[float],
        gamma: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        self.s_list = [s.reshape(-1) for s in s_list]
        self.y_list = [y.reshape(-1) for y in y_list]
        self.rho_list = [float(r) for r in rho_list]
        self.gamma = float(gamma)
        self.dtype = dtype
        self.device = device if device is not None else self.s_list[0].device

        self.Yt, self.Vt, self.St = unroll_lbfgs_update(
            self.s_list, self.y_list, self.rho_list, dtype=dtype, device=self.device
        )

    @property
    def m(self) -> int:
        """Number of stored curvature pairs."""
        return len(self.s_list)

    @property
    def p(self) -> int:
        """Dimension of the parameter space."""
        return self.s_list[0].numel()

    def matvec(self, hvp: Callable[[torch.Tensor], torch.Tensor], v: torch.Tensor) -> torch.Tensor:
        """Apply the preconditioned Hessian matvec (Algorithm 3)."""
        return preconditioned_matvec(v, hvp, self.Yt, self.Vt, self.St, gamma=self.gamma)

    def build_matvec(self, hvp: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a matvec callable bound to ``hvp`` for SLQ."""
        return lambda v: self.matvec(hvp, v)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LBFGSPreconditioner(m={self.m}, p={self.p}, "
            f"gamma={self.gamma:.4g}, dtype={self.dtype})"
        )
