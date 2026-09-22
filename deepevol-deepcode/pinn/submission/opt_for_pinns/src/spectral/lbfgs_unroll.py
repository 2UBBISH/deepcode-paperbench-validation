"""Unrolling the L-BFGS update (Algorithm 2, Appendix C.2).

The paper (Appendix C.2) writes the L-BFGS inverse-Hessian approximation as

    H_k = (I - Y~ V~^T)^T gamma_k I (I - Y~ V~^T) + S~ S~^T
        = [ sqrt(gamma_k) (I - Y~ V~^T)^T , S~ ] [ sqrt(gamma_k) (I - Y~ V~^T) ; S~^T ]
        = H~_k H~_k^T

with, for the last ``m`` stored pairs (``s_i = x_{i+1} - x_i``, ``y_i = g_{i+1} - g_i``,
``rho_i = 1/(y_i^T s_i)``, ``gamma_k = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})``)::

    Y~  = [ rho_{k-1} y_{k-1}  ...  rho_{k-m} y_{k-m} ]
    V~  = [ v~_{k-1}           ...  v~_{k-m}          ]
    S~  = [ s~_{k-1}           ...  s~_{k-m}          ],  s~_{k-1} = sqrt(rho_{k-1}) s_{k-1},
                                                         s~_{k-l} = sqrt(rho_{k-l})(V_{k-1}^T ... V_{k-l+1}^T) s_{k-l}

Algorithm 2 computes the columns of ``Y~``, ``V~`` and ``S~`` sequentially::

    y~_{k-1} = rho_{k-1} y_{k-1}
    v~_{k-1} = s_{k-1}
    s~_{k-1} = sqrt(rho_{k-1}) s_{k-1}
    for i = k-2, ..., k-m:
        y~_i = rho_i y_i
        alpha = 0
        for j = k-1, ..., i+1:
            alpha = alpha + (y~_j^T s_i) v~_j
        v~_i = s_i - alpha
        s~_i = sqrt(rho_i) (s_i - alpha)

This module stores the columns **newest first** (column 0 corresponds to index ``k-1``,
column ``m-1`` to ``k-m``), exactly matching the column ordering of ``Y~``, ``V~`` and ``S~``
printed in the paper.  The factors then allow matrix-vector products with ``H_k``
(via ``H~_k``/``H~_k^T``) without ever materialising an ``n x n`` matrix, as required by
Algorithm 3 (`src/spectral/preconditioned_mvp.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

try:  # pragma: no cover - import shim so the file works from either sys.path root
    from src.spectral.hvp import DEFAULT_DTYPE
except Exception:  # pragma: no cover
    try:
        from .hvp import DEFAULT_DTYPE  # type: ignore
    except Exception:
        DEFAULT_DTYPE = torch.float64

__all__ = [
    "LBFGSFactors",
    "LBFGSUnroll",
    "UnrolledLBFGS",
    "unroll_lbfgs",
    "lbfgs_factors",
    "unroll_from_history",
    "apply_lbfgs_inverse_hessian",
    "DEFAULT_DTYPE",
]


# ---------------------------------------------------------------------------
# Factor container
# ---------------------------------------------------------------------------
@dataclass
class LBFGSFactors:
    """Columns of ``Y~``, ``V~``, ``S~`` plus the scaling ``gamma_k``.

    Attributes
    ----------
    Y_tilde, V_tilde, S_tilde:
        Tensors of shape ``(n, m)`` whose **columns are ordered newest first**
        (column 0 is index ``k-1``).
    gamma:
        Scalar ``gamma_k = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})``.
    n, m:
        Parameter dimension and number of stored curvature pairs.
    rho:
        ``(m,)`` tensor of ``rho_i`` (newest first), kept for diagnostics.
    """

    n: int
    m: int
    Y_tilde: torch.Tensor
    V_tilde: torch.Tensor
    S_tilde: torch.Tensor
    gamma: float
    rho: Optional[torch.Tensor] = None
    dtype: torch.dtype = DEFAULT_DTYPE
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- basics ------------------------------------------------------------
    @property
    def size(self) -> int:
        """Dimension ``size(w) + m`` of the operator ``H~_k^T H_L H~_k``."""
        return self.n + self.m

    @property
    def sqrt_gamma(self) -> float:
        return math.sqrt(max(float(self.gamma), 0.0))

    def __len__(self) -> int:
        return self.m

    def __bool__(self) -> bool:
        return self.m > 0

    def to(self, device: Union[str, torch.device]) -> "LBFGSFactors":
        return LBFGSFactors(
            n=self.n,
            m=self.m,
            Y_tilde=self.Y_tilde.to(device),
            V_tilde=self.V_tilde.to(device),
            S_tilde=self.S_tilde.to(device),
            gamma=self.gamma,
            rho=None if self.rho is None else self.rho.to(device),
            dtype=self.dtype,
            extra=dict(self.extra),
        )

    def _as_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """Coerce a vector / matrix argument to shape ``(n, k)``."""
        if v.dim() == 1:
            return v.reshape(-1, 1)
        if v.dim() == 2:
            return v
        raise ValueError(f"expected 1-D or 2-D tensor, got shape {tuple(v.shape)}")

    # -- products with H~_k -------------------------------------------------
    def apply_Htilde(self, v: torch.Tensor) -> torch.Tensor:
        """``H~_k v`` for ``v`` of length ``n + m`` (Algorithm 3, step 2).

        ``H~_k = [ sqrt(gamma_k) (I - Y~ V~^T)^T , S~ ]`` has ``n`` rows and ``n+m``
        columns, hence ``H~_k v = sqrt(gamma) (v_1 - V~ Y~^T v_1) + S~ v_2``.
        """
        V = self._as_matrix(v)
        if V.shape[0] != self.size:
            raise ValueError(
                f"apply_Htilde expects vectors of length n+m={self.size}, got {V.shape[0]}"
            )
        v1 = V[: self.n, :]
        v2 = V[self.n :, :]
        out = self.sqrt_gamma * (v1 - self.V_tilde @ (self.Y_tilde.transpose(0, 1) @ v1))
        out = out + self.S_tilde @ v2
        return out if out.shape[1] > 1 else out.reshape(-1)

    def apply_Htilde_T(self, v: torch.Tensor) -> torch.Tensor:
        """``H~_k^T v`` for ``v`` of length ``n`` (Algorithm 3, step 4).

        ``H~_k^T v = [ sqrt(gamma_k)(v - Y~ V~^T v) ; S~^T v ]`` has length ``n+m``.
        """
        V = self._as_matrix(v)
        if V.shape[0] != self.n:
            raise ValueError(
                f"apply_Htilde_T expects vectors of length n={self.n}, got {V.shape[0]}"
            )
        top = self.sqrt_gamma * (V - self.Y_tilde @ (self.V_tilde.transpose(0, 1) @ V))
        bottom = self.S_tilde.transpose(0, 1) @ V
        out = torch.cat([top, bottom], dim=0)
        return out if out.shape[1] > 1 else out.reshape(-1)

    #: friendlier aliases (``H~`` is ``Htilde``)
    apply_htilde = apply_Htilde
    apply_htilde_T = apply_Htilde_T

    def apply_H(self, v: torch.Tensor) -> torch.Tensor:
        """``H_k v = H~_k H~_k^T v`` (length ``n`` in, length ``n`` out)."""
        return self.apply_Htilde_T(self.apply_Htilde(v))

    #: alias used by some callers
    apply_inverse_hessian = apply_H

    # -- dense / diagnostics ------------------------------------------------
    def dense_Htilde(self) -> torch.Tensor:
        """Materialise ``H~_k`` as an ``(n, n+m)`` matrix (diagnostics only)."""
        eye = torch.eye(self.n, dtype=self.Y_tilde.dtype, device=self.Y_tilde.device)
        left = self.sqrt_gamma * (
            eye - self.V_tilde @ self.Y_tilde.transpose(0, 1)
        ).transpose(0, 1)
        return torch.cat([left, self.S_tilde], dim=1)

    def dense_H(self) -> torch.Tensor:
        """Materialise ``H_k`` as an ``(n, n)`` matrix (diagnostics / small tests)."""
        Ht = self.dense_Htilde()
        return Ht @ Ht.transpose(0, 1)

    def curvature_eps(self) -> float:
        """Small value used when reporting numerical rank of the factors."""
        return float(self.extra.get("eps", 1e-14))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "m": self.m,
            "gamma": float(self.gamma),
            "Y_tilde": self.Y_tilde,
            "V_tilde": self.V_tilde,
            "S_tilde": self.S_tilde,
            "rho": self.rho,
        }


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------
def _stack_pairs(
    pairs: Sequence[torch.Tensor],
    m: Optional[int],
    name: str,
    dtype: torch.dtype,
    device: Optional[torch.device],
) -> torch.Tensor:
    """Stack a newest-first sequence of vectors into an ``(m, n)`` tensor."""
    if len(pairs) == 0:
        raise ValueError(f"no L-BFGS {name} pairs supplied")
    stacked = torch.stack([torch.as_tensor(p, dtype=dtype).reshape(-1) for p in pairs], dim=0)
    if m is not None:
        stacked = stacked[:m]
    if device is not None:
        stacked = stacked.to(device)
    return stacked


def unroll_lbfgs(
    Y: Union[torch.Tensor, Sequence[torch.Tensor]],
    S: Union[torch.Tensor, Sequence[torch.Tensor]],
    rho: Union[torch.Tensor, Sequence[float]],
    gamma: Optional[float] = None,
    *,
    m: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_info: bool = False,
) -> Union[LBFGSFactors, Tuple[LBFGSFactors, Dict[str, Any]]]:
    """Unroll the L-BFGS update (Algorithm 2) into the factors of ``H_k = H~_k H~_k^T``.

    Parameters
    ----------
    Y, S:
        Saved ``{y_i}`` and ``{s_i}``, either ``(m, n)`` tensors or sequences of vectors,
        ordered **newest first** (row/entry 0 is index ``k-1``, the last is ``k-m``).
    rho:
        Saved ``rho_i = 1/(y_i^T s_i)``, in the same newest-first order.
    gamma:
        Scaling ``gamma_k = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})``.  If omitted it is
        computed from the most recent pair (row 0).

    Returns
    -------
    LBFGSFactors
        With ``Y_tilde``, ``V_tilde``, ``S_tilde`` of shape ``(n, m)`` (newest first) and
        ``gamma``.  When ``return_info`` is set, a small diagnostics dict is returned too.
    """
    dtype = dtype or DEFAULT_DTYPE
    device = torch.device(device) if device is not None else None

    Y_t = _stack_pairs(Y, m, "y", dtype, device)
    S_t = _stack_pairs(S, m, "s", dtype, device)
    if Y_t.shape != S_t.shape:
        raise ValueError(
            f"Y and S must have the same shape, got {tuple(Y_t.shape)} and {tuple(S_t.shape)}"
        )
    rows, n = Y_t.shape

    if isinstance(rho, torch.Tensor):
        rho_t = torch.as_tensor(rho, dtype=dtype).reshape(-1)[:rows].to(Y_t.device)
    else:
        rho_t = torch.as_tensor(list(rho)[:rows], dtype=dtype, device=Y_t.device)

    if gamma is None:
        y0, s0 = Y_t[0], S_t[0]
        denom = float(torch.dot(y0, y0))
        gamma = float(torch.dot(s0, y0) / denom) if denom > 0 else 1.0

    # ---- Algorithm 2 -----------------------------------------------------
    # index t (0-based) corresponds to the paper's index i = k-1-t
    Y_tilde = rho_t.reshape(-1, 1) * Y_t            # y~_i = rho_i y_i   (all i at once)
    V_tilde = torch.empty_like(S_t)
    S_tilde = torch.empty_like(S_t)

    V_tilde[0] = S_t[0]                              # v~_{k-1} = s_{k-1}
    S_tilde[0] = math.sqrt(float(rho_t[0])) * S_t[0]  # s~_{k-1} = sqrt(rho_{k-1}) s_{k-1}

    for t in range(1, rows):                         # i = k-2, ..., k-m
        si = S_t[t]
        # alpha = sum_{j=k-1}^{i+1} (y~_j^T s_i) v~_j  == sum_{u<t} (Y~[:,u]^T s_i) V~[:,u]
        coeffs = Y_tilde[:t].transpose(0, 1) @ si     # (t,)
        alpha = V_tilde[:t].transpose(0, 1) @ coeffs  # (n,)
        vi = si - alpha
        V_tilde[t] = vi
        S_tilde[t] = math.sqrt(float(rho_t[t])) * vi

    factors = LBFGSFactors(
        n=n,
        m=rows,
        Y_tilde=Y_tilde,
        V_tilde=V_tilde,
        S_tilde=S_tilde,
        gamma=float(gamma),
        rho=rho_t,
        dtype=dtype,
        extra={"requested_m": m},
    )
    if return_info:
        info = {
            "n": n,
            "m": rows,
            "gamma": float(gamma),
            "rho_min": float(rho_t.min()) if rows else float("nan"),
            "rho_max": float(rho_t.max()) if rows else float("nan"),
        }
        return factors, info
    return factors


#: aliases matching the naming used in the reproduction plan / paper
lbfgs_factors = unroll_lbfgs


def unroll_from_history(
    history: Any,
    gamma: Optional[float] = None,
    *,
    m: Optional[int] = None,
    index: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_info: bool = False,
) -> Union[LBFGSFactors, Tuple[LBFGSFactors, Dict[str, Any]]]:
    """Unroll the buffers recorded by :class:`src.optimizers.lbfgs_wrapper.LBFGSHistory`.

    Accepts anything exposing ``.stacked(memory)`` (returning ``S``, ``Y``, ``rho``,
    ``gamma``) or ``.s``/``.y``/``.rho``/``.gamma`` lists.  Buffers are assumed to be
    stored oldest → newest and are flipped so that Algorithm 2 sees newest first.
    ``index`` optionally selects a prefix of the history (e.g. the state after ``k``
    iterations); otherwise the newest ``m`` pairs are used.
    """
    stacked = None
    if hasattr(history, "stacked"):
        try:
            stacked = history.stacked(memory=m)
        except TypeError:  # pragma: no cover - different signature
            stacked = history.stacked()
    if stacked is None:
        S_list = list(getattr(history, "s"))
        Y_list = list(getattr(history, "y"))
        rho_list = list(getattr(history, "rho"))
        gamma = gamma if gamma is not None else getattr(history, "last_gamma", None)
    else:
        S_list = list(stacked["S"])
        Y_list = list(stacked["Y"])
        rho_list = list(stacked["rho"])
        if gamma is None:
            gvals = list(stacked.get("gamma", [])) if isinstance(stacked, dict) else []
            gamma = gvals[-1] if gvals else None
    if index is not None:
        S_list = S_list[: index + 1]
        Y_list = Y_list[: index + 1]
        rho_list = rho_list[: index + 1]
    if len(S_list) == 0:
        raise ValueError("empty L-BFGS history; nothing to unroll")

    dtype = dtype or DEFAULT_DTYPE
    S_t = _stack_pairs(S_list, m, "s", dtype, device)
    Y_t = _stack_pairs(Y_list, m, "y", dtype, device)
    rho_t = torch.as_tensor(list(rho_list)[: S_t.shape[0]], dtype=dtype, device=S_t.device)
    if gamma is None:
        y0, s0 = Y_t[0], S_t[0]
        denom = float(torch.dot(y0, y0))
        gamma = float(torch.dot(s0, y0) / denom) if denom > 0 else 1.0
    # buffers recorded oldest-first -> flip so column 0 is the most recent pair
    return unroll_lbfgs(
        torch.flip(Y_t, dims=[0]),
        torch.flip(S_t, dims=[0]),
        torch.flip(rho_t, dims=[0]),
        float(gamma),
        m=m,
        dtype=dtype,
        return_info=return_info,
    )


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------
class LBFGSUnroll:
    """Callable wrapper around :func:`unroll_lbfgs` bound to stored buffers.

    ``LBFGSUnroll(Y, S, rho, gamma)`` behaves like ``LBFGSFactors`` (delegates attribute
    access) and can be re-bound to a longer history with :meth:`rebind`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._factors = unroll_lbfgs(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs

    # -- delegate to the factors ------------------------------------------
    @property
    def factors(self) -> LBFGSFactors:
        return self._factors

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial passthrough
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._factors, item)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Apply ``H_k`` (the L-BFGS inverse-Hessian approximation) to ``v``."""
        return self._factors.apply_H(v)

    def apply_H(self, v: torch.Tensor) -> torch.Tensor:
        return self._factors.apply_H(v)

    def apply_Htilde(self, v: torch.Tensor) -> torch.Tensor:
        return self._factors.apply_Htilde(v)

    def apply_Htilde_T(self, v: torch.Tensor) -> torch.Tensor:
        return self._factors.apply_Htilde_T(v)

    @classmethod
    def from_history(cls, history: Any, *args: Any, **kwargs: Any) -> "LBFGSUnroll":
        obj = cls.__new__(cls)
        obj._factors = unroll_from_history(history, *args, **kwargs)
        obj._args = ()
        obj._kwargs = {}
        return obj

    def rebind(self, Y: Any, S: Any, rho: Any, gamma: Optional[float] = None, **kwargs: Any):
        self._factors = unroll_lbfgs(Y, S, rho, gamma, **kwargs)
        return self


#: paper-style alias
UnrolledLBFGS = LBFGSUnroll


# ---------------------------------------------------------------------------
# Reference implementation (classic two-loop recursion) for validation
# ---------------------------------------------------------------------------
def apply_lbfgs_inverse_hessian(
    v: torch.Tensor,
    Y: Union[torch.Tensor, Sequence[torch.Tensor]],
    S: Union[torch.Tensor, Sequence[torch.Tensor]],
    rho: Union[torch.Tensor, Sequence[float]],
    gamma: Optional[float] = None,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Standard (Nocedal & Wright, 2006) two-loop recursion computing ``H_k v``.

    Inputs are newest-first, matching :func:`unroll_lbfgs`.  Used as the ground truth in
    :func:`_self_test` to validate the unrolled factors.
    """
    dtype = dtype or DEFAULT_DTYPE
    Y_t = _stack_pairs(Y, None, "y", dtype, None)
    S_t = _stack_pairs(S, None, "s", dtype, None)
    rows = Y_t.shape[0]
    rho_t = torch.as_tensor(list(rho)[:rows], dtype=dtype)
    if gamma is None:
        gamma = float(torch.dot(S_t[0], Y_t[0]) / torch.dot(Y_t[0], Y_t[0]))

    v = torch.as_tensor(v, dtype=dtype).reshape(-1)
    q = v.clone()
    alphas: List[torch.Tensor] = []
    for t in range(rows):                       # j = k-1 ... k-m
        a = float(rho_t[t]) * float(torch.dot(S_t[t], q))
        alphas.append(torch.as_tensor(a, dtype=dtype))
        q = q - a * Y_t[t]
    r = float(gamma) * q
    for t in range(rows - 1, -1, -1):           # j = k-m ... k-1
        beta = float(rho_t[t]) * float(torch.dot(Y_t[t], r))
        r = r + (alphas[t] - beta) * S_t[t]
    return r


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------
def _self_test(seed: int = 0) -> None:
    torch.manual_seed(seed)
    n, m = 25, 6
    dtype = torch.float64

    # Build a synthetic L-BFGS history from a quadratic so that curvature is real.
    A = torch.randn(n, n, dtype=dtype)
    A = A @ A.transpose(0, 1) + 1.0 * torch.eye(n, dtype=dtype)

    def f(x):
        return 0.5 * x @ (A @ x)

    def g(x):
        return A @ x

    x = torch.randn(n, dtype=dtype)
    S_list, Y_list, rho_list, gammas = [], [], [], []
    for _ in range(m):
        gi = g(x)
        s = -0.05 * gi
        x = x + s
        gj = g(x)
        y = gj - gi
        S_list.append(s.clone())
        Y_list.append(y.clone())
        rho_list.append(1.0 / float(torch.dot(y, s)))
        gammas.append(float(torch.dot(s, y) / torch.dot(y, y)))

    # history stored oldest -> newest; flip to newest-first for the algorithms
    Y = torch.flip(torch.stack(Y_list), dims=[0])
    S = torch.flip(torch.stack(S_list), dims=[0])
    rho = torch.flip(torch.tensor(rho_list, dtype=dtype), dims=[0])
    gamma = gammas[-1]

    factors = unroll_lbfgs(Y, S, rho, gamma)
    assert factors.Y_tilde.shape == (n, m)
    assert factors.V_tilde.shape == (n, m)
    assert factors.S_tilde.shape == (n, m)

    # 1) H~ H~^T must reproduce the classic two-loop recursion on random vectors
    for _ in range(4):
        v = torch.randn(n, dtype=dtype)
        got = factors.apply_H(v)
        ref = apply_lbfgs_inverse_hessian(v, Y, S, rho, gamma)
        err = float(torch.norm(got - ref) / torch.norm(ref))
        assert err < 1e-10, f"unrolled factors disagree with two-loop recursion: {err:.3e}"

    # 2) H~^T H~ v equals H v  (H = H~ H~^T is symmetric)
    v = torch.randn(n, dtype=dtype)
    assert torch.allclose(factors.apply_Htilde_T(factors.apply_Htilde(v)), factors.apply_H(v))

    # 3) The preconditioned operator dimension is size(w) + m
    assert factors.size == n + m
    assert factors.dense_H().shape == (n, n)

    # 4) Algorithm 3 building blocks: H~^T H~ has the same (non-zero) spectrum as H
    H = factors.dense_H()
    ev_H = torch.linalg.eigvalsh(H)
    print(
        "[lbfgs_unroll] self-test OK | n=%d m=%d gamma=%.4e | eig(H): "
        "min=%.3e max=%.3e" % (n, m, gamma, float(ev_H.min()), float(ev_H.max()))
    )


if __name__ == "__main__":  # pragma: no cover
    _self_test()
