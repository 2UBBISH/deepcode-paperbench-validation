"""L-BFGS unrolling for preconditioned spectral density analysis (Algorithm 2).

This module implements the construction of the L-BFGS preconditioner factors
used to analyze the *preconditioned* Hessian of the PINN loss (Appendix C.2 of
the paper).  Given the sequence of curvature pairs ``{(s_i, y_i, rho_i)}``
collected during an L-BFGS run, we build the (implicit) matrix

    H_tilde_k = [ rho_0 s_0, ..., rho_{k-1} s_{k-1},  y_0, ..., y_{k-1} ]

whose columns are the "unrolled" L-BFGS vectors.  The preconditioned Hessian
that L-BFGS effectively sees is then

    H_tilde_k^T  H_L(w_k)  H_tilde_k

which is symmetric and can be fed to Stochastic Lanczos Quadrature (SLQ) to
estimate the spectral density of the *preconditioned* loss landscape.

Algorithm 2 (Unrolling the L-BFGS update) -- as described in the paper:

    Input:  curvature pairs {(s_i, y_i, rho_i)}_{i=0}^{k-1}
    Output: the unrolled vectors {s_tilde_i, y_tilde_i, v_tilde_i}

The paper's notation uses ``y_tilde_i``, ``v_tilde_i`` and ``s_tilde_i``.  In
practice the columns of ``H_tilde_k`` are simply the scaled ``s`` vectors and
the ``y`` vectors, so we expose both a low-level representation (the raw
columns) and the derived quantities.

References
----------
- Nocedal & Wright, "Numerical Optimization", Ch. 7 (L-BFGS two-loop recursion).
- Paper Appendix C.2 (preconditioned spectral density).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


__all__ = [
    "LBFGSHistory",
    "unroll_lbfgs",
    "build_preconditioner_columns",
    "apply_lbfgs_preconditioner",
    "lbfgs_two_loop",
]


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------
class LBFGSHistory:
    """Container for L-BFGS curvature pairs ``{(s_i, y_i, rho_i)}``.

    Parameters
    ----------
    memory:
        Maximum number of curvature pairs to retain (L-BFGS memory ``m``).
        If more pairs are added than ``memory``, the oldest are dropped
        (matching the standard L-BFGS circular buffer behaviour).

    Notes
    -----
    ``s_i = w_{i+1} - w_i`` (parameter difference), ``y_i = g_{i+1} - g_i``
    (gradient difference) and ``rho_i = 1 / (y_i^T s_i)``.
    """

    def __init__(self, memory: int = 100) -> None:
        self.memory = int(memory)
        self.s: List[torch.Tensor] = []
        self.y: List[torch.Tensor] = []
        self.rho: List[float] = []

    # -- mutation ----------------------------------------------------------
    def add(self, s: torch.Tensor, y: torch.Tensor, rho: Optional[float] = None) -> None:
        """Append a curvature pair, computing ``rho`` if not supplied."""
        s = s.detach().clone().reshape(-1)
        y = y.detach().clone().reshape(-1)
        if rho is None:
            ys = torch.dot(y, s)
            rho = float(1.0 / ys) if float(ys) != 0.0 else 0.0
        self.s.append(s)
        self.y.append(y)
        self.rho.append(float(rho))
        if len(self.s) > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def clear(self) -> None:
        self.s.clear()
        self.y.clear()
        self.rho.clear()

    # -- access ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.s)

    def pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        return list(zip(self.s, self.y, self.rho))

    def to_dict(self) -> Dict[str, list]:
        return {"s": list(self.s), "y": list(self.y), "rho": list(self.rho)}

    @classmethod
    def from_dict(cls, d: Dict[str, list], memory: Optional[int] = None) -> "LBFGSHistory":
        hist = cls(memory=memory if memory is not None else len(d.get("s", [])))
        for s, y, rho in zip(d.get("s", []), d.get("y", []), d.get("rho", [])):
            hist.add(s, y, rho)
        return hist


# ---------------------------------------------------------------------------
# Algorithm 2: unrolling
# ---------------------------------------------------------------------------
def unroll_lbfgs(
    history: LBFGSHistory,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Unroll the L-BFGS update into explicit vectors (Algorithm 2).

    Given the stored curvature pairs, returns the unrolled quantities used to
    form the L-BFGS preconditioner:

    - ``s_tilde``: the scaled parameter-difference vectors ``rho_i * s_i``
      (shape ``(p, k)``).
    - ``y_tilde``: the gradient-difference vectors ``y_i`` (shape ``(p, k)``).
    - ``v_tilde``: the "unrolled" vectors obtained by applying the L-BFGS
      two-loop recursion to each canonical basis direction; equivalently the
      columns of the implicit preconditioner ``H_tilde_k`` (shape ``(p, k)``).
    - ``rho``: the curvature scalars (shape ``(k,)``).

    Parameters
    ----------
    history:
        The :class:`LBFGSHistory` holding ``{(s_i, y_i, rho_i)}``.
    dtype, device:
        Optional dtype/device for the returned tensors.  Defaults to the dtype
        and device of the stored ``s`` vectors.

    Returns
    -------
    dict with keys ``s_tilde``, ``y_tilde``, ``v_tilde``, ``rho``.
    """
    if len(history) == 0:
        raise ValueError("Cannot unroll an empty L-BFGS history.")

    ref = history.s[0]
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device

    s_list = [s.to(dtype=dtype, device=device) for s in history.s]
    y_list = [y.to(dtype=dtype, device=device) for y in history.y]
    rho_list = [float(r) for r in history.rho]

    k = len(s_list)
    p = s_list[0].numel()

    # s_tilde_i = rho_i * s_i  (the scaled parameter differences)
    s_tilde = torch.stack([rho_list[i] * s_list[i] for i in range(k)], dim=1)  # (p, k)
    # y_tilde_i = y_i
    y_tilde = torch.stack(y_list, dim=1)  # (p, k)
    rho = torch.tensor(rho_list, dtype=dtype, device=device)  # (k,)

    # v_tilde: columns of the implicit L-BFGS preconditioner H_tilde_k.
    # Applying the two-loop recursion to each canonical basis vector e_j gives
    # the j-th column of the preconditioner.  For efficiency we instead build
    # the explicit low-rank representation directly: the preconditioner is
    # H_tilde_k = [rho_0 s_0, ..., rho_{k-1} s_{k-1}, y_0, ..., y_{k-1}]
    # (see build_preconditioner_columns).  Here v_tilde stores the same columns
    # in the paper's notation.
    v_tilde = build_preconditioner_columns(history, dtype=dtype, device=device)

    return {
        "s_tilde": s_tilde,
        "y_tilde": y_tilde,
        "v_tilde": v_tilde,
        "rho": rho,
    }


def build_preconditioner_columns(
    history: LBFGSHistory,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the columns of the implicit L-BFGS preconditioner ``H_tilde_k``.

    Returns a ``(p, 2k)`` tensor whose columns are

        [ rho_0 s_0, ..., rho_{k-1} s_{k-1},  y_0, ..., y_{k-1} ]

    This is the matrix ``H_tilde_k`` used in Algorithm 3 to form the
    preconditioned Hessian ``H_tilde_k^T H_L H_tilde_k``.
    """
    if len(history) == 0:
        raise ValueError("Cannot build preconditioner columns from an empty history.")

    ref = history.s[0]
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device

    scaled_s = torch.stack(
        [float(r) * s.to(dtype=dtype, device=device) for s, r in zip(history.s, history.rho)],
        dim=1,
    )
    y_cols = torch.stack([y.to(dtype=dtype, device=device) for y in history.y], dim=1)
    return torch.cat([scaled_s, y_cols], dim=1)  # (p, 2k)


# ---------------------------------------------------------------------------
# Two-loop recursion (standard L-BFGS preconditioner application)
# ---------------------------------------------------------------------------
def lbfgs_two_loop(
    grad: torch.Tensor,
    history: LBFGSHistory,
    gamma: Optional[float] = None,
) -> torch.Tensor:
    """Apply the L-BFGS two-loop recursion to ``grad``.

    Computes ``H_k grad`` where ``H_k`` is the standard L-BFGS inverse-Hessian
    approximation built from the stored curvature pairs.  This is the
    preconditioner that L-BFGS implicitly applies; it is useful for verifying
    the explicit low-rank representation built by
    :func:`build_preconditioner_columns`.

    Parameters
    ----------
    grad:
        Gradient vector of shape ``(p,)``.
    history:
        Curvature pairs.
    gamma:
        Scaling ``gamma = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})``.  If
        ``None`` it is computed from the most recent pair.
    """
    if len(history) == 0:
        return grad.clone()

    q = grad.detach().clone().reshape(-1)
    s_list = history.s
    y_list = history.y
    rho_list = history.rho
    k = len(s_list)

    if gamma is None:
        s_last = s_list[-1]
        y_last = y_list[-1]
        yy = torch.dot(y_last, y_last)
        gamma = float(torch.dot(s_last, y_last) / yy) if float(yy) != 0.0 else 1.0

    alpha = [0.0] * k
    # First loop (backwards)
    for i in range(k - 1, -1, -1):
        alpha[i] = rho_list[i] * float(torch.dot(s_list[i], q))
        q = q - alpha[i] * y_list[i]
    # Initial Hessian scaling
    r = gamma * q
    # Second loop (forwards)
    for i in range(k):
        beta = rho_list[i] * float(torch.dot(y_list[i], r))
        r = r + s_list[i] * (alpha[i] - beta)
    return r


def apply_lbfgs_preconditioner(
    v: torch.Tensor,
    history: LBFGSHistory,
    gamma: Optional[float] = None,
) -> torch.Tensor:
    """Alias for :func:`lbfgs_two_loop` (applies ``H_k`` to ``v``)."""
    return lbfgs_two_loop(v, history, gamma=gamma)
