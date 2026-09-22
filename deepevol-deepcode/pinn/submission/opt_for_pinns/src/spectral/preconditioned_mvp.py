"""Preconditioned matrix-vector products with the L-BFGS-preconditioned Hessian.

This module implements Algorithm 3 of Appendix C.2 of

    "Challenges in Training PINNs: A Loss Landscape Perspective" (ICML 2024).

Background (Appendix C.2)
-------------------------
The L-BFGS inverse-Hessian approximation ``H_k`` factors as

    H_k = (I - Y~ V~^T)^T gamma_k I (I - Y~ V~^T) + S~ S~^T
        = [ sqrt(gamma_k) (I - Y~ V~^T)^T   S~ ] [ sqrt(gamma_k) (I - Y~ V~^T) ; S~^T ]
        = H~_k H~_k^T,

with ``H~_k`` of shape ``(n, n + m)`` (``n = size(w)``, ``m`` = L-BFGS memory).  Since the
non-zero eigenvalues of ``H~_k^T H_L(w) H~_k`` equal those of the preconditioned Hessian
``H_k H_L(w) = H~_k H~_k^T H_L(w)`` (Theorem 1.3.22 of Horn & Johnson, 2012), the *symmetric*
``(n + m) x (n + m)`` operator ``H~_k^T H_L(w) H~_k`` is what the stochastic Lanczos quadrature
(SLQ) spectral-density pipeline of ``spectral_density.py`` analyses.

Algorithm 3 (Performing matrix-vector product)
---------------------------------------------
::

    input: matrices Y~, V~, S~ (from unrolling), vector v, gamma_k
    split v (length size(w) + m) into v1 (size(w)) and v2 (m)
    v'  = sqrt(gamma_k) (v1 - V~ Y~^T v1) + S~ v2          # = H~_k v
    v'' = H_L(w) v'                                        # Hessian-vector product
    stack sqrt(gamma_k) (v'' - Y~ V~^T v'') and S~^T v''   # = H~_k^T v''
    output v'''                                            # = H~_k^T H_L(w) H~_k v

Note that ``(I - Y~ V~^T)^T = I - V~ Y~^T``, which is the form printed in Algorithm 3 for the
first (forward) application, while the transpose application uses ``(I - Y~ V~^T)`` directly.

Public API
----------
- :class:`PreconditionedHessian` / :class:`PreconditionedMVP`: symmetric operator
  ``H~_k^T H_L(w) H~_k`` on flat vectors of length ``n + m``.
- :func:`preconditioned_mvp` / :func:`preconditioned_matvec` / :func:`preconditioned_hvp`:
  functional form of Algorithm 3 (single vector or a batch of columns).
- :func:`h_tilde_matvec`: forward application ``H~_k v`` (``n + m -> n``).
- :func:`h_tilde_transpose_matvec`: transpose application ``H~_k^T u`` (``n -> n + m``).
- :func:`dense_preconditioned_hessian`: materialises the dense ``(n + m) x (n + m)`` matrix
  ``H~_k^T A H~_k`` (small dimensional tests / validation only).
- :func:`make_spectral_operator`, :func:`operator_from_history`: convenience constructors.
- :func:`preconditioned_hessian_eigenvalues`: dense reference eigenvalues (tests only).

All routines operate on ``torch.float64`` by default (second-order numerics) and are agnostic to
whether the underlying Hessian is supplied as a callable (HVP oracle), a dense
:class:`torch.Tensor`, or an object exposing ``.matvec`` (e.g. ``hvp.HessianOperator``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

try:  # pragma: no cover - import shim so the module loads from either sys.path root
    from .hvp import DEFAULT_DTYPE
except ImportError:  # pragma: no cover
    try:
        from src.spectral.hvp import DEFAULT_DTYPE  # type: ignore
    except ImportError:
        DEFAULT_DTYPE = torch.float64

try:  # pragma: no cover
    from .lbfgs_unroll import (
        LBFGSFactors,
        unroll_lbfgs,
        unroll_from_history,
    )
except ImportError:  # pragma: no cover
    try:
        from src.spectral.lbfgs_unroll import (  # type: ignore
            LBFGSFactors,
            unroll_lbfgs,
            unroll_from_history,
        )
    except ImportError:  # pragma: no cover
        LBFGSFactors = None  # type: ignore
        unroll_lbfgs = None  # type: ignore
        unroll_from_history = None  # type: ignore


__all__ = [
    "PreconditionedHessian",
    "PreconditionedMVP",
    "preconditioned_mvp",
    "preconditioned_matvec",
    "preconditioned_hvp",
    "h_tilde_matvec",
    "h_tilde_transpose_matvec",
    "apply_htilde",
    "apply_htilde_t",
    "dense_htilde",
    "dense_preconditioned_hessian",
    "make_spectral_operator",
    "operator_from_history",
    "preconditioned_hessian_eigenvalues",
    "FactorTensors",
    "extract_factor_tensors",
    "DEFAULT_DTYPE",
]


MatVec = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------------------
# Factor extraction / coercion helpers
# --------------------------------------------------------------------------------------
@dataclass
class FactorTensors:
    """Minimal container for the unrolled L-BFGS factors.

    Attributes
    ----------
    Y_tilde, V_tilde, S_tilde:
        Tensors of shape ``(n, m)`` with columns ordered **newest first** (column 0 corresponds to
        index ``k-1``), exactly as produced by ``lbfgs_unroll.unroll_lbfgs``.
    gamma:
        Scalar ``gamma_k = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})``.
    """

    Y_tilde: torch.Tensor
    V_tilde: torch.Tensor
    S_tilde: torch.Tensor
    gamma: float = 1.0
    source: Any = None

    @property
    def n(self) -> int:
        return int(self.V_tilde.shape[0])

    @property
    def m(self) -> int:
        return int(self.V_tilde.shape[1])

    @property
    def size(self) -> int:
        return self.n + self.m

    @property
    def sqrt_gamma(self) -> float:
        return math.sqrt(max(float(self.gamma), 0.0))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "m": self.m,
            "gamma": float(self.gamma),
            "dtype": str(self.V_tilde.dtype),
        }


def _stack_pairs(pairs: Any, m: Optional[int], dtype: torch.dtype) -> torch.Tensor:
    """Coerce a sequence of 1-D tensors into a ``(n, len(pairs))`` matrix (newest first)."""
    if torch.is_tensor(pairs):
        if pairs.dim() == 1:
            pairs = pairs.unsqueeze(1)
        mat = pairs.to(dtype=dtype)
        return mat if m is None else mat[:, :m]
    rows = [torch.as_tensor(p, dtype=dtype).reshape(-1) for p in pairs]
    if not rows:
        raise ValueError("empty list of L-BFGS vectors")
    mat = torch.stack(rows, dim=1)
    return mat if m is None else mat[:, :m]


def extract_factor_tensors(factors: Any, *, dtype: Optional[torch.dtype] = None) -> FactorTensors:
    """Normalise the many possible representations of the unrolled factors.

    Accepted inputs
    ---------------
    * :class:`FactorTensors` (returned unchanged / re-cast).
    * ``LBFGSFactors`` (from ``lbfgs_unroll.py``), i.e. anything exposing ``Y_tilde``,
      ``V_tilde``, ``S_tilde`` and ``gamma``.
    * ``dict`` with keys ``Y_tilde``/``V_tilde``/``S_tilde`` (or ``Y``/``V``/``S``) plus
      optional ``gamma``.
    * ``LBFGSHistory``-like object (has ``.Y``/``.S``/``.rho`` or ``.stacked``) -> unrolled via
      ``lbfgs_unroll.unroll_from_history``.
    * ``(Y, S, rho)`` or ``(Y, S, rho, gamma)`` tuple -> unrolled via ``lbfgs_unroll.unroll_lbfgs``.
    """
    dt = dtype or DEFAULT_DTYPE

    if isinstance(factors, FactorTensors):
        return factors

    if isinstance(factors, dict):
        keys = {k.lower(): v for k, v in factors.items()}
        Y = keys.get("y_tilde", keys.get("ytilde", keys.get("y")))
        V = keys.get("v_tilde", keys.get("vtilde", keys.get("v")))
        S = keys.get("s_tilde", keys.get("stilde", keys.get("s")))
        gamma = keys.get("gamma", None)
        if Y is None or V is None or S is None:
            # Perhaps raw (Y, S, rho) curvature pairs were passed as a dict.
            if "rho" in keys and Y is not None and S is not None:
                return _make_from_pairs(Y, S, keys["rho"], gamma, dtype=dt)
            raise ValueError(
                "factor dict must provide Y_tilde/V_tilde/S_tilde (or Y/V/S)"
            )
        m = None if Y is None else min(Y.shape[-1] if torch.is_tensor(Y) else len(Y),
                                       V.shape[-1] if torch.is_tensor(V) else len(V))
        return FactorTensors(
            Y_tilde=_stack_pairs(Y, m, dt),
            V_tilde=_stack_pairs(V, m, dt),
            S_tilde=_stack_pairs(S, m, dt),
            gamma=float(gamma) if gamma is not None else 1.0,
            source=factors,
        )

    # LBFGSHistory-like object with curvature buffers (no unrolled factors yet).
    looks_like_history = (
        hasattr(factors, "Y")
        and hasattr(factors, "S")
        and (hasattr(factors, "rho") or hasattr(factors, "stacked"))
        and not all(
            hasattr(factors, a) for a in ("Y_tilde", "V_tilde", "S_tilde")
        )
    )
    if looks_like_history:
        if unroll_from_history is None:
            raise RuntimeError("lbfgs_unroll.unroll_from_history is unavailable")
        unrolled = unroll_from_history(factors, dtype=dt)
        if isinstance(unrolled, tuple):  # (factors, info)
            unrolled = unrolled[0]
        return extract_factor_tensors(unrolled, dtype=dt)

    # Unrolled factor object (LBFGSFactors or duck-typed equivalent).
    if all(hasattr(factors, a) for a in ("Y_tilde", "V_tilde", "S_tilde")):
        m = int(min(factors.Y_tilde.shape[-1], factors.V_tilde.shape[-1], factors.S_tilde.shape[-1]))
        gamma = float(getattr(factors, "gamma", 1.0))
        return FactorTensors(
            Y_tilde=factors.Y_tilde.to(dtype=dt),
            V_tilde=factors.V_tilde.to(dtype=dt),
            S_tilde=factors.S_tilde.to(dtype=dt),
            gamma=gamma,
            source=factors,
        )

    # Raw curvature pairs.
    if isinstance(factors, (tuple, list)):
        vals = list(factors)
        if len(vals) in (3, 4):
            Y, S, rho = vals[0], vals[1], vals[2]
            gamma = vals[3] if len(vals) == 4 else None
            return _make_from_pairs(Y, S, rho, gamma, dtype=dt)

    raise TypeError(
        "unsupported `factors` object: expected LBFGSFactors, dict, history or "
        f"(Y, S, rho[, gamma]) tuple, got {type(factors)!r}"
    )


def _make_from_pairs(
    Y: Any, S: Any, rho: Any, gamma: Any, *, dtype: torch.dtype
) -> FactorTensors:
    """Unroll raw curvature pairs gathered from a history object."""
    if hasattr(Y, "stacked") or hasattr(Y, "Y_tilde"):
        return extract_factor_tensors(Y, dtype=dtype)

    Ym = _stack_pairs(Y, None, dtype)
    Sm = _stack_pairs(S, None, dtype)
    if torch.is_tensor(rho):
        rm = rho.to(dtype=dtype).reshape(-1)
    else:
        rm = torch.as_tensor(list(rho), dtype=dtype)
    m = min(Ym.shape[1], Sm.shape[1], rm.numel())
    Ym, Sm, rm = Ym[:, :m], Sm[:, :m], rm[:m]
    if gamma is None:
        ys = (Ym[:, 0] * Sm[:, 0]).sum()
        yy = (Ym[:, 0] * Ym[:, 0]).sum()
        gamma = float(ys / yy) if float(yy) > 0.0 else 1.0
    if unroll_lbfgs is None:
        raise RuntimeError("lbfgs_unroll.unroll_lbfgs is unavailable")
    unrolled = unroll_lbfgs(Ym, Sm, rm, float(gamma), m=m, dtype=dtype)
    if isinstance(unrolled, tuple):
        unrolled = unrolled[0]
    return extract_factor_tensors(unrolled, dtype=dtype)


def _as_matvec(op: Any) -> MatVec:
    """Normalise an operator (callable / dense tensor / operator object) into a matvec callable."""
    if op is None:
        raise ValueError("a Hessian operator (`hvp`) must be provided")
    if callable(op):
        return op
    if torch.is_tensor(op):
        return lambda v: op @ v
    if hasattr(op, "matvec"):
        return op.matvec  # type: ignore[no-any-return]
    if hasattr(op, "hvp"):
        return op.hvp  # type: ignore[no-any-return]
    raise TypeError(f"cannot interpret {type(op)!r} as a Hessian matvec oracle")


def _as_2d(v: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return ``(v_2d, was_1d)`` for a vector or a matrix of columns."""
    if v.dim() == 1:
        return v.unsqueeze(1), True
    if v.dim() == 2:
        return v, False
    raise ValueError(f"expected a 1-D or 2-D vector, got shape {tuple(v.shape)}")


# --------------------------------------------------------------------------------------
# H~_k applications (Algorithm 3 building blocks)
# --------------------------------------------------------------------------------------
def h_tilde_matvec(factors: Any, v: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Apply ``H~_k`` (shape ``(n, n + m)``) to ``v`` of length ``n + m`` -> length ``n``.

    This is the first line of Algorithm 3::

        v' = sqrt(gamma_k) (v1 - V~ Y~^T v1) + S~ v2

    using the identity ``(I - Y~ V~^T)^T = I - V~ Y~^T`` (c.f. Eq. (4) of Appendix C.2).
    Accepts a single vector ``(n+m,)`` or a block of columns ``(n+m, k)``.
    """
    ft = extract_factor_tensors(factors, dtype=dtype)
    V, Y, S = ft.V_tilde, ft.Y_tilde, ft.S_tilde
    v2d, was_1d = _as_2d(torch.as_tensor(v, dtype=V.dtype))
    if v2d.shape[0] != ft.size:
        raise ValueError(
            f"expected a vector of length n + m = {ft.size}, got {v2d.shape[0]}"
        )
    v1, v2 = v2d[: ft.n], v2d[ft.n:]
    out = ft.sqrt_gamma * (v1 - V @ (Y.transpose(0, 1) @ v1)) + S @ v2
    return out.squeeze(1) if was_1d else out


def h_tilde_transpose_matvec(
    factors: Any, u: torch.Tensor, *, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Apply ``H~_k^T`` (shape ``(n + m, n)``) to ``u`` of length ``n`` -> length ``n + m``.

    This is the last line of Algorithm 3::

        v''' = [ sqrt(gamma_k) (v'' - Y~ V~^T v'') ; S~^T v'' ]

    Accepts a single vector ``(n,)`` or a block of columns ``(n, k)``.
    """
    ft = extract_factor_tensors(factors, dtype=dtype)
    V, Y, S = ft.V_tilde, ft.Y_tilde, ft.S_tilde
    u2d, was_1d = _as_2d(torch.as_tensor(u, dtype=V.dtype))
    if u2d.shape[0] != ft.n:
        raise ValueError(f"expected a vector of length n = {ft.n}, got {u2d.shape[0]}")
    top = ft.sqrt_gamma * (u2d - Y @ (V.transpose(0, 1) @ u2d))
    bottom = S.transpose(0, 1) @ u2d
    out = torch.cat([top, bottom], dim=0)
    return out.squeeze(1) if was_1d else out


# Paper-style lowercase aliases.
apply_htilde = h_tilde_matvec
apply_htilde_t = h_tilde_transpose_matvec


def dense_htilde(factors: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Materialise ``H~_k = [ sqrt(gamma_k)(I - Y~ V~^T)^T , S~ ]`` of shape ``(n, n + m)``."""
    ft = extract_factor_tensors(factors, dtype=dtype)
    V, Y, S = ft.V_tilde, ft.Y_tilde, ft.S_tilde
    eye = torch.eye(ft.n, dtype=V.dtype, device=V.device)
    left = ft.sqrt_gamma * (eye - Y @ V.transpose(0, 1)).transpose(0, 1)
    return torch.cat([left, S], dim=1)


# --------------------------------------------------------------------------------------
# Algorithm 3: preconditioned matrix-vector product
# --------------------------------------------------------------------------------------
def preconditioned_mvp(
    factors: Any,
    v: torch.Tensor,
    hvp: Any,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Algorithm 3: apply ``H~_k^T H_L(w) H~_k`` to ``v`` of length ``n + m``.

    Parameters
    ----------
    factors:
        Unrolled L-BFGS factors (``LBFGSFactors``, dict, history, or ``(Y, S, rho)`` tuple) --
        see :func:`extract_factor_tensors`.
    v:
        Vector of length ``n + m`` (or a matrix with that many rows).
    hvp:
        Hessian-vector-product oracle ``u -> H_L(w) u`` on flat vectors of length ``n``; may be a
        callable, a dense matrix, or an object exposing ``.matvec``.

    Returns
    -------
    ``torch.Tensor`` with the same leading dimension as ``v``.
    """
    ft = extract_factor_tensors(factors, dtype=dtype)
    matvec = _as_matvec(hvp)

    # Step 1: v' = H~_k v  (n + m -> n)
    v_prime = h_tilde_matvec(ft, v)

    # Step 2: v'' = H_L(w) v'  (n -> n); loop over columns when a block is passed.
    if v_prime.dim() == 2:
        columns = [matvec(v_prime[:, j]) for j in range(v_prime.shape[1])]
        v_dprime = torch.stack([torch.as_tensor(c).reshape(-1) for c in columns], dim=1)
    else:
        v_dprime = torch.as_tensor(matvec(v_prime)).reshape(-1)

    # Step 3: v''' = H~_k^T v''  (n -> n + m)
    return h_tilde_transpose_matvec(ft, v_dprime)


# Aliases used by the experiment runners / spectral pipeline.
preconditioned_matvec = preconditioned_mvp
preconditioned_hvp = preconditioned_mvp


# --------------------------------------------------------------------------------------
# Operator wrapper (consumed by stochastic Lanczos quadrature)
# --------------------------------------------------------------------------------------
class PreconditionedHessian:
    """Symmetric matrix-free operator ``H~_k^T H_L(w) H~_k`` (Algorithm 3).

    The operator acts on flat vectors of length ``size = n + m``.  It exposes the small
    ``LinearOperator``-compatible surface expected by the SLQ routines in
    ``spectral_density.py``: ``matvec``, ``matmat``, ``shape``, ``dtype``, ``device`` and
    ``__call__``.

    Parameters
    ----------
    factors:
        Unrolled L-BFGS factors (see :func:`extract_factor_tensors`).
    hvp:
        Hessian-vector-product oracle for ``H_L(w)`` (callable, dense tensor, or object with
        ``.matvec``).
    dtype:
        Computation dtype (defaults to ``float64``).
    """

    def __init__(self, factors: Any, hvp: Any, *, dtype: Optional[torch.dtype] = None):
        self.factors_raw = factors
        self.factors = extract_factor_tensors(factors, dtype=dtype)
        self._hvp = _as_matvec(hvp)
        self.dtype = self.factors.V_tilde.dtype
        self.device = self.factors.V_tilde.device
        self.n_matvecs = 0  # number of Hessian-vector products performed (diagnostics)

    # -- basic operator surface ---------------------------------------------------------
    @property
    def n(self) -> int:
        return self.factors.n

    @property
    def m(self) -> int:
        return self.factors.m

    @property
    def size(self) -> int:
        return self.factors.size

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.size, self.size)

    @property
    def gamma(self) -> float:
        return float(self.factors.gamma)

    @property
    def sqrt_gamma(self) -> float:
        return self.factors.sqrt_gamma

    def to(self, device: Union[str, torch.device]) -> "PreconditionedHessian":
        self.factors = FactorTensors(
            Y_tilde=self.factors.Y_tilde.to(device),
            V_tilde=self.factors.V_tilde.to(device),
            S_tilde=self.factors.S_tilde.to(device),
            gamma=self.factors.gamma,
            source=self.factors.source,
        )
        self.device = self.factors.V_tilde.device
        return self

    # -- Algorithm 3 --------------------------------------------------------------------
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Apply ``H~_k^T H_L(w) H~_k`` to ``v`` (length ``n + m`` or a block of columns)."""
        self.n_matvecs += 1
        return preconditioned_mvp(self.factors, v, self._hvp)

    __call__ = matvec

    def matmat(self, V: torch.Tensor) -> torch.Tensor:
        """Apply the operator to each column of ``V`` (``(n+m, k)`` -> ``(n+m, k)``)."""
        self.n_matvecs += int(V.shape[1]) if V.dim() == 2 else 1
        return preconditioned_mvp(self.factors, V, self._hvp)

    # -- adjoint: the operator is symmetric so T is the identity operator ----------------
    def T(self) -> "PreconditionedHessian":
        return self

    @property
    def T_op(self) -> "PreconditionedHessian":
        return self

    def as_operator(self) -> "PreconditionedHessian":
        return self

    def operator(self) -> MatVec:
        return self.matvec

    def quadratic_form(self, v: torch.Tensor) -> float:
        """Return ``v^T (H~_k^T H_L H~_k) v``."""
        return float(torch.dot(v.reshape(-1), self.matvec(v).reshape(-1)))

    # -- dense reference -----------------------------------------------------------------
    def dense(self) -> torch.Tensor:
        """Materialise the dense ``(n + m) x (n + m)`` matrix (small ``n`` only)."""
        return dense_preconditioned_hessian(self.factors, self._hvp)

    to_dense = dense

    def via_htilde(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(H~_k, H_k = H~_k H~_k^T)`` (dense, small ``n`` only)."""
        Ht = dense_htilde(self.factors)
        return Ht, Ht @ Ht.transpose(0, 1)

    # -- constructors ---------------------------------------------------------------------
    @classmethod
    def from_pairs(
        cls,
        Y: Any,
        S: Any,
        rho: Any,
        hvp: Any,
        gamma: Optional[float] = None,
        *,
        m: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "PreconditionedHessian":
        """Build the operator from raw L-BFGS curvature pairs."""
        dtype = dtype or DEFAULT_DTYPE
        Ym = _stack_pairs(Y, m, dtype)
        Sm = _stack_pairs(S, m, dtype)
        rm = (
            rho.to(dtype=dtype).reshape(-1)
            if torch.is_tensor(rho)
            else torch.as_tensor(list(rho), dtype=dtype)
        )
        k = min(Ym.shape[1], Sm.shape[1], rm.numel())
        Ym, Sm, rm = Ym[:, :k], Sm[:, :k], rm[:k]
        if gamma is None:
            ys = float((Ym[:, 0] * Sm[:, 0]).sum())
            yy = float((Ym[:, 0] * Ym[:, 0]).sum())
            gamma = ys / yy if yy > 0 else 1.0
        if unroll_lbfgs is None:
            raise RuntimeError("lbfgs_unroll.unroll_lbfgs is unavailable")
        factors = unroll_lbfgs(Ym, Sm, rm, gamma, m=k, dtype=dtype)
        if isinstance(factors, tuple):
            factors = factors[0]
        return cls(factors, hvp, dtype=dtype)

    @classmethod
    def from_history(
        cls,
        history: Any,
        hvp: Any,
        *,
        m: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "PreconditionedHessian":
        """Build the operator from a recorded ``LBFGSHistory`` object."""
        if unroll_from_history is None:
            raise RuntimeError("lbfgs_unroll.unroll_from_history is unavailable")
        dtype = dtype or DEFAULT_DTYPE
        factors = unroll_from_history(history, m=m, dtype=dtype)
        if isinstance(factors, tuple):
            factors = factors[0]
        return cls(factors, hvp, dtype=dtype)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "m": self.m,
            "size": self.size,
            "gamma": self.gamma,
            "dtype": str(self.dtype),
            "n_matvecs": self.n_matvecs,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PreconditionedHessian(n={self.n}, m={self.m}, gamma={self.gamma:.6g}, "
            f"dtype={self.dtype})"
        )


# Paper-facing alias.
PreconditionedMVP = PreconditionedHessian


def make_spectral_operator(
    factors: Any, hvp: Any, *, dtype: Optional[torch.dtype] = None
) -> PreconditionedHessian:
    """Return the symmetric ``H~_k^T H_L(w) H~_k`` operator analysed by SLQ."""
    return PreconditionedHessian(factors, hvp, dtype=dtype)


def operator_from_history(
    history: Any, hvp: Any, *, m: Optional[int] = None, dtype: Optional[torch.dtype] = None
) -> PreconditionedHessian:
    """Convenience wrapper around :meth:`PreconditionedHessian.from_history`."""
    return PreconditionedHessian.from_history(history, hvp, m=m, dtype=dtype)


# --------------------------------------------------------------------------------------
# Dense reference (tests / small problems)
# --------------------------------------------------------------------------------------
def dense_preconditioned_hessian(
    factors: Any, hvp: Any, *, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Materialise ``H~_k^T A H~_k`` where ``A`` is the (small) Hessian given by ``hvp``.

    The matrix is formed by applying the exact Algorithm 3 chain to the identity columns, so it
    is an independent reference for :func:`preconditioned_mvp`.
    """
    ft = extract_factor_tensors(factors, dtype=dtype)
    matvec = _as_matvec(hvp)
    n = ft.n
    eye_n = torch.eye(n, dtype=ft.V_tilde.dtype, device=ft.V_tilde.device)
    cols = [matvec(eye_n[:, j]) for j in range(n)]
    A = torch.stack([torch.as_tensor(c).reshape(-1) for c in cols], dim=1)
    Ht = dense_htilde(ft)
    return Ht.transpose(0, 1) @ A @ Ht


def preconditioned_hessian_eigenvalues(
    factors: Any, hvp: Any, *, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Eigenvalues of the dense ``H~_k^T A H~_k`` (small problems only)."""
    return torch.linalg.eigvalsh(dense_preconditioned_hessian(factors, hvp, dtype=dtype))


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------
def _self_test(seed: int = 0) -> Dict[str, float]:  # pragma: no cover - manual sanity check
    """Validate Algorithm 3 against dense algebra and the eigenvalue equivalence theorem."""
    if unroll_lbfgs is None:
        raise RuntimeError("lbfgs_unroll module unavailable")

    torch.manual_seed(seed)
    n, m = 8, 3
    S = torch.randn(n, m, dtype=torch.float64)
    Y = S + 0.5 * torch.randn(n, m, dtype=torch.float64)  # keep y^T s mostly positive
    rho = 1.0 / (Y * S).sum(dim=0)
    gamma = float((S[:, 0] * Y[:, 0]).sum() / (Y[:, 0] * Y[:, 0]).sum())
    factors = unroll_lbfgs(Y, S, rho, gamma, m=m, dtype=torch.float64)

    M = torch.randn(n, n, dtype=torch.float64)
    A = M @ M.transpose(0, 1) / n + 0.1 * torch.eye(n, dtype=torch.float64)
    hvp = lambda v: A @ v  # noqa: E731

    op = PreconditionedHessian(factors, hvp)
    dense = op.dense()
    assert dense.shape == (n + m, n + m)

    # (1) symmetry
    sym_err = float((dense - dense.transpose(0, 1)).abs().max())

    # (2) matvec consistency with the dense matrix
    g = torch.Generator().manual_seed(seed + 1)
    v = torch.randn(n + m, dtype=torch.float64, generator=g)
    mvp_err = float((op.matvec(v) - dense @ v).abs().max())

    # (3) H~_k v then H~_k^T (.) reproduce the dense factors
    Ht = dense_htilde(factors)
    ht_err = float((h_tilde_matvec(factors, v) - Ht @ v).abs().max())
    u = torch.randn(n, dtype=torch.float64, generator=g)
    ht_t_err = float((h_tilde_transpose_matvec(factors, u) - Ht.transpose(0, 1) @ u).abs().max())

    # (4) H_k = H~_k H~_k^T must equal the classical L-BFGS inverse-Hessian operator
    Hk = Ht @ Ht.transpose(0, 1)
    classical_err = 0.0
    try:
        from .lbfgs_unroll import apply_lbfgs_inverse_hessian  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from src.spectral.lbfgs_unroll import apply_lbfgs_inverse_hessian  # type: ignore
        except Exception:
            apply_lbfgs_inverse_hessian = None
    if apply_lbfgs_inverse_hessian is not None:
        ref = apply_lbfgs_inverse_hessian(u, Y, S, rho, gamma)
        classical_err = float((Hk @ u - ref).abs().max())

    # (5) Theorem 1.3.22 (Horn & Johnson): non-zero eigenvalues of H~^T A H~ equal those of
    #     H~ H~^T A = H_k A.
    lam_dense = torch.linalg.eigvalsh(dense)
    lam_dense = lam_dense[lam_dense.abs() > 1e-8 * max(1.0, float(lam_dense.abs().max()))]
    lam_prod = torch.linalg.eigvals(Hk @ A)
    order = torch.argsort(lam_prod.real, descending=True)
    lam_prod = lam_prod.real[order]
    k = min(lam_dense.numel(), lam_prod.numel())
    lam_dense_sorted = torch.sort(lam_dense, descending=True).values[:k]
    eig_err = float((lam_dense_sorted - lam_prod[:k]).abs().max()) if k else 0.0

    info = {
        "symmetry_error": sym_err,
        "matvec_error": mvp_err,
        "htilde_error": ht_err,
        "htilde_transpose_error": ht_t_err,
        "classical_hk_error": classical_err,
        "eigenvalue_equivalence_error": eig_err,
    }
    for name, err in info.items():
        if err > 1e-8:
            raise AssertionError(f"self-test failure: {name} = {err:.3e}")
    return info


if __name__ == "__main__":  # pragma: no cover
    print(_self_test())
