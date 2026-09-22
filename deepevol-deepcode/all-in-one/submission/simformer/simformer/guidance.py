"""Diffusion guidance for the Simformer (Sec. 3.4, Sec. A1.3, Sec. A3.3).

The Simformer can condition on *exact* observations by simply clamping the
observed variables while running the reverse SDE (Sec. 3.3).  Conditions that
are not point observations -- e.g. **intervals** (``x <= u``), box constraints or
arbitrary set constraints ``c(x) <= 0`` -- are handled by *guidance*: the score
estimate is augmented with the gradient of the log-sigmoid of the constraint
evaluated on a **denoised estimate** (Bansal et al. 2023; Lugmayr et al. 2022).
The general formulation is (Sec. 3.4, Eq. 2; Sec. A1.3, Eq. 4)::

    s_phi(x_t, t | c) ~= s_phi(x_t, t)
                        + grad_{x_t} sum_{i=1}^{K} log sigmoid(-s(t) c_i(x_t))

with the scaling function ``s(t) -> inf`` as ``t -> 0``; the paper uses
``s(t) = 1 / sigma(t)^2`` (Sec. A3.3), i.e. inversely proportional to the
variance of the approximate marginal SDE scores.  For an interval upper bound
``u`` the constraint function is ``c(x) = x - u`` (Sec. 3.4).

Algorithm 1 ("General Guidance", Sec. A3.3) is implemented exactly, including
the **self-recurrence** loop of ``r`` steps and the forward-SDE resampling of
the future point between successive inner steps::

    dt = (T_max - T_min) / T
    x_{T} ~ N(mu_T, sigma_T I)
    for i = 1 .. T:                      # t_i = T_max - i * dt
        for j = 1 .. r:
            eps ~ N(0, I)
            s = s_phi(x_{t_{i+1}}, t_i)                      # marginal score
            x_~0 = (x_{t_{i+1}} + sigma(t_{i+1})^2 s) / mu(t_{i+1})   # denoise
            s~ = s + grad_x log sigmoid(s(t) c(x_~0))        # constraint score
            x_{t_i} = x_{t_{i+1}} - (f(x_{t_{i+1}}, t_i) - g(t_i)^2 s~) dt
                                  - g(t_i) sqrt(dt) eps
            if r > 0:                                        # resample future pt
                eps ~ N(0, I)
                x_{t_{i+1}} = x_{t_i} + f(x_{t_{i+1}}, t_i) dt
                            + g(t_i) sqrt(dt) eps
    return x_{T_min}

Remarks / defaults where the paper is silent (documented in the code below):

* ``j = 1 .. r`` degenerates for ``r = 0``; ``r = 0`` is interpreted as *no
  self-recurrence*, i.e. a single inner evaluation per reverse step.  The paper
  reports ``r = 5`` as the setting that recovers model-based conditioning
  accuracy at 5x cost (Sec. A3.3).
* Algorithm 1 writes ``log sigmoid(s(t) c(x_~0))`` without the minus sign while
  Sec. 3.4 (Eq. 2) and Sec. A1.3 (Eq. 4) include it.  The minus sign is the one
  that actually enforces ``c <= 0`` (it pushes ``c`` downwards), so it is the
  default here; ``sign`` is configurable for ablation.
* The scaling function is evaluated at the time of the current (noisier) point.
* ``dt = t_cur - t_next > 0`` with the reverse iteration ``t_max -> t_min``.
* The forward resampling step at the *last* inner iteration is skipped because
  its result is immediately discarded by the next outer iteration (pure compute
  saving; no change to the returned trajectory).

Exact/point conditions are *model-based* (clamped, ``M_C``), set constraints are
guided -- combining both is exactly the protocol of Fig. A16b.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .diffusion import (
    DEFAULT_STEPS,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    MIN_RECOMMENDED_STEPS,
    SDE,
    get_sde,
    time_grid,
)

try:  # torch is required by the rest of the package; keep the import soft here.
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore
    _HAS_TORCH = False


__all__ = [
    # constants
    "DEFAULT_GUIDANCE_STEPS",
    "DEFAULT_SELF_RECURRENCE",
    "RECOMMENDED_SELF_RECURRENCE",
    "DEFAULT_SIGN",
    "DEFAULT_SCALE",
    "DEFAULT_FD_EPS",
    # scaling functions
    "inverse_variance_scaling",
    "variance_scaling",
    "std_scaling",
    "constant_scaling",
    "get_scaling_function",
    # constraints
    "Constraint",
    "IntervalConstraint",
    "EqualityConstraint",
    "CallableConstraint",
    "CombinedConstraint",
    "combine_constraints",
    "interval_constraint",
    "upper_bound_constraint",
    "lower_bound_constraint",
    "box_constraint",
    "equality_constraint",
    "constraint_violation",
    "constraint_satisfaction",
    # score machinery
    "log_sigmoid",
    "denoise_from_score",
    "guidance_scale_factor",
    "constraint_score",
    "guided_score",
    "guidance_step",
    "general_guidance",
    "sample_with_guidance",
    "sample_interval_conditional",
    # higher level
    "GuidanceConfig",
    "GuidanceResult",
    "GuidedSampler",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_GUIDANCE_STEPS: int = DEFAULT_STEPS  # 500 reverse-SDE steps (Sec. A2.1)
DEFAULT_SELF_RECURRENCE: int = 0  # r = 0 -> no self-recurrence
RECOMMENDED_SELF_RECURRENCE: int = 5  # r = 5 in Fig. A15/A16 (5x compute)
DEFAULT_SIGN: float = -1.0  # log sigmoid(-s(t) c(x)) as in Sec. 3.4 / A1.3
DEFAULT_SCALE: str = "inverse_variance"  # s(t) = 1 / sigma(t)^2 (Sec. A3.3)
DEFAULT_FD_EPS: float = 1e-3

ArrayLike = Union[np.ndarray, Any]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _lib_of(x: Any):
    """Return ``torch`` for torch tensors and ``numpy`` otherwise."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return torch
    return np


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors / scalars / lists to a detached NumPy array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _as_float(x: Any) -> float:
    """Scalar float from a python/numpy/torch 0-d (or size-1) value."""
    arr = np.asarray(_to_numpy(x)).reshape(-1)
    return float(arr[0])


def _as_timestep(sde: SDE, t: Any) -> float:
    """Clamp a time value to the SDE's integration interval."""
    t_val = _as_float(t)
    return float(min(max(t_val, sde.t_min), sde.t_max))


def _marginal_std(sde: SDE, t: Any) -> float:
    return float(np.asarray(_to_numpy(sde.marginal_std(t))).reshape(-1)[0])


def _marginal_mean(sde: SDE, t: Any) -> float:
    return float(np.asarray(_to_numpy(sde.marginal_mean(t))).reshape(-1)[0])


def _sigmoid(x: Any):
    lib = _lib_of(x)
    if lib is np:
        return 1.0 / (1.0 + np.exp(-np.asarray(x)))
    return torch.sigmoid(x)


def log_sigmoid(x: Any):
    """Numerically stable ``log sigmoid(x)`` (numpy or torch)."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return torch.nn.functional.logsigmoid(x)
    arr = np.asarray(x)
    return -np.logaddexp(0.0, -arr)


def _sample_prior(sde: SDE, shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Draw from the terminal (prior) distribution of the SDE."""
    out = None
    for kwargs in ({"rng": rng}, {}):
        try:
            out = sde.sample_prior(shape, **kwargs)
            break
        except TypeError:
            continue
    if out is None:  # pragma: no cover - last-resort fallback
        mu = _marginal_mean(sde, sde.t_max)
        sigma = _marginal_std(sde, sde.t_max)
        out = mu + sigma * rng.standard_normal(shape)
    out = _to_numpy(out)
    if out.shape != tuple(shape):
        out = np.broadcast_to(out, shape)
    return np.asarray(out, dtype=np.float64)


def _call_score_fn(score_fn: Callable, x: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a score function with a scalar or per-sample time argument."""
    try:
        out = score_fn(x, t)
    except TypeError:
        out = score_fn(x, np.full((x.shape[0],), float(t)))
    return np.asarray(_to_numpy(out), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Scaling functions s(t)  (Sec. A3.3)
# --------------------------------------------------------------------------- #


def inverse_variance_scaling(sde: SDE, t: Any) -> float:
    """``s(t) = 1 / sigma(t)^2`` -- the paper's default scaling (Sec. A3.3)."""
    sigma = _marginal_std(sde, t)
    return 1.0 / max(sigma * sigma, 1e-300)


def variance_scaling(sde: SDE, t: Any) -> float:
    """``s(t) = sigma(t)^2`` (ablation only)."""
    sigma = _marginal_std(sde, t)
    return float(sigma * sigma)


def std_scaling(sde: SDE, t: Any) -> float:
    """``s(t) = sigma(t)`` (ablation only)."""
    return float(_marginal_std(sde, t))


def constant_scaling(value: float = 1.0) -> Callable[[SDE, Any], float]:
    """``s(t) = value`` (ablation only)."""

    def fn(sde: SDE, t: Any) -> float:  # noqa: ARG001 - signature parity
        return float(value)

    return fn


def get_scaling_function(
    scale: Union[str, float, Callable, None] = DEFAULT_SCALE,
    sde: Optional[SDE] = None,
    *,
    clip: Optional[float] = None,
) -> Callable[[Any], float]:
    """Build ``s(t)`` as a function of ``t`` only.

    Parameters
    ----------
    scale:
        * ``None`` / ``"inverse_variance"`` / ``"inv_var"`` / ``"1/sigma^2"`` ->
          ``1 / sigma(t)^2`` (paper default).
        * ``"variance"`` / ``"sigma^2"`` -> ``sigma(t)^2``.
        * ``"std"`` / ``"sigma"`` -> ``sigma(t)``.
        * ``"constant"`` / ``"one"`` -> ``1``.
        * a float -> constant scaling with that value.
        * a callable ``f(t) -> float`` -> used directly (``sde`` is bound if the
          callable accepts two arguments).
    clip:
        Optional upper bound on the scaling value (numerical safeguard; the
        paper's ``1/sigma(t)^2`` diverges as ``t -> 0``).
    """
    if sde is None:
        sde = get_sde("vesde")

    if callable(scale):
        try:
            _raw = scale(_as_timestep(sde, sde.t_max))  # probe: single-arg?
            base = lambda t: float(np.asarray(_to_numpy(scale(t))).reshape(-1)[0])
        except TypeError:
            base = lambda t: float(np.asarray(_to_numpy(scale(sde, t))).reshape(-1)[0])
        del _raw
    elif scale is None:
        base = lambda t: inverse_variance_scaling(sde, t)
    elif isinstance(scale, (int, float, np.floating, np.integer)):
        base = lambda t, _v=float(scale): float(_v)
    else:
        key = str(scale).strip().lower()
        if key in ("inverse_variance", "inv_var", "inverse-variance", "1/sigma^2", "invvar"):
            base = lambda t: inverse_variance_scaling(sde, t)
        elif key in ("variance", "sigma^2", "sigma2"):
            base = lambda t: variance_scaling(sde, t)
        elif key in ("std", "sigma"):
            base = lambda t: std_scaling(sde, t)
        elif key in ("constant", "one", "none"):
            base = lambda t: 1.0
        else:
            raise ValueError(f"Unknown scaling function specification: {scale!r}")

    if clip is None:
        return base

    def clipped(t: Any) -> float:
        return float(min(base(t), float(clip)))

    return clipped


# --------------------------------------------------------------------------- #
# Constraints  c(x) <= 0   (Sec. 3.4 / A1.3)
# --------------------------------------------------------------------------- #


class Constraint:
    """Base class for constraint functions ``c(x) <= 0``.

    ``violation(x)`` returns an array of shape ``(..., K)`` holding ``K``
    constraint values per sample; the values are satisfied when ``<= 0``.
    Subclasses may implement ``jacobian_entries`` for an efficient analytic
    gradient (list of ``(dim_index, d c_k / d x_j)`` per constraint) and/or
    ``grad`` for a dense Jacobian of shape ``(..., K, d)``.
    """

    name: str = "constraint"

    def __init__(self, name: Optional[str] = None) -> None:
        if name is not None:
            self.name = name

    # -- required ---------------------------------------------------------
    @property
    def n_constraints(self) -> int:
        raise NotImplementedError

    def violation(self, x: ArrayLike) -> ArrayLike:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- optional ---------------------------------------------------------
    def jacobian_entries(self, x: ArrayLike, k: int):
        """Return ``(dim_index, d c_k / d x_dim per sample)`` for constraint ``k``.

        Default: ``NotImplementedError`` (finite differences are then used).
        """
        raise NotImplementedError

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.violation(x)

    def satisfied(self, x: ArrayLike, tol: float = 0.0) -> np.ndarray:
        c = _to_numpy(self.violation(x))
        return c <= tol

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r})"


class IntervalConstraint(Constraint):
    """Interval (box) constraints on selected dimensions.

    For each selected dimension ``i``: ``lower_i <= x_i <= upper_i``, expressed
    in the paper's inequality form ``c(x) <= 0`` as ``x_i - upper_i <= 0`` and
    ``lower_i - x_i <= 0`` (cf. Sec. 3.4: ``c(x) = x - u`` for an upper bound).

    Parameters
    ----------
    indices:
        Dimensions the constraint acts on.  ``None`` means "all dimensions"
        (only allowed with scalar bounds; ``n_constraints`` is then dynamic).
    lower, upper:
        Bound values; scalars (broadcast) or sequences matching ``indices``.
        ``None`` disables that side.
    """

    def __init__(
        self,
        indices: Optional[Sequence[int]] = None,
        lower: Union[None, float, Sequence[float]] = None,
        upper: Union[None, float, Sequence[float]] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "interval")
        if lower is None and upper is None:
            raise ValueError("IntervalConstraint requires at least one of lower/upper.")
        self.indices = None if indices is None else np.asarray(list(indices), dtype=np.int64)
        self.lower = self._normalize_bounds(lower)
        self.upper = self._normalize_bounds(upper)
        self.dynamic_dim = self.indices is None
        self._k = None if self.dynamic_dim else self._k_for(self.indices.size)
        self._bound_types: Optional[np.ndarray] = None
        self._bound_values: Optional[np.ndarray] = None
        self._bound_indices: Optional[np.ndarray] = None

    # -- construction helpers --------------------------------------------
    @staticmethod
    def _normalize_bounds(bound):
        if bound is None:
            return None
        arr = np.asarray(bound, dtype=np.float64)
        return arr.reshape(()) if arr.ndim == 0 else arr.reshape(-1)

    def _k_for(self, n_dims: int) -> int:
        k = 0
        if self.upper is not None:
            k += self._size(self.upper, n_dims)
        if self.lower is not None:
            k += self._size(self.lower, n_dims)
        return k

    @staticmethod
    def _size(bound: np.ndarray, n_dims: int) -> int:
        return n_dims if bound.ndim == 0 else max(int(bound.shape[0]), n_dims)

    def _resolve(self, n_dims: int):
        """Build flat ``(indices, coefficients, values)`` for the K constraints."""
        idx = np.arange(n_dims, dtype=np.int64) if self.indices is None else self.indices
        if idx.max(initial=-1) >= n_dims:
            raise ValueError(
                f"IntervalConstraint indices {idx.tolist()} exceed dimension {n_dims}."
            )
        indices, coefs, values = [], [], []
        if self.upper is not None:
            up = np.broadcast_to(self.upper, idx.shape) if self.upper.ndim == 0 else self.upper
            up = np.resize(up, idx.shape) if up.shape[0] != idx.shape[0] else up
            for j, v in zip(idx, up):
                indices.append(int(j))
                coefs.append(1.0)  # d(x - u)/dx = +1
                values.append(float(v))
        if self.lower is not None:
            lo = np.broadcast_to(self.lower, idx.shape) if self.lower.ndim == 0 else self.lower
            lo = np.resize(lo, idx.shape) if lo.shape[0] != idx.shape[0] else lo
            for j, v in zip(idx, lo):
                indices.append(int(j))
                coefs.append(-1.0)  # d(l - x)/dx = -1
                values.append(float(v))
        return (
            np.asarray(indices, dtype=np.int64),
            np.asarray(coefs, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )

    def _ensure_resolved(self, n_dims: int) -> None:
        if self._bound_indices is None or self._bound_indices.size != self._k_for(n_dims):
            (idx, coefs, values) = self._resolve(n_dims)
            self._bound_indices, self._bound_coefs, self._bound_values = idx, coefs, values
            if self._k is None:
                self._k = int(idx.size)

    # -- Constraint API ---------------------------------------------------
    @property
    def n_constraints(self) -> int:
        if self._k is None:  # dynamic (indices=None): resolved on first call
            raise ValueError("n_constraints is dynamic; call violation(x) first.")
        return int(self._k)

    def violation(self, x: ArrayLike) -> ArrayLike:
        lib = _lib_of(x)
        n_dims = int(x.shape[-1])
        self._ensure_resolved(n_dims)
        idx, coefs, values = self._bound_indices, self._bound_coefs, self._bound_values
        if lib is np:
            vals = np.asarray(x)[..., idx]
            return vals * coefs - values * coefs
        c = torch.as_tensor(coefs, dtype=x.dtype, device=x.device)
        v = torch.as_tensor(values, dtype=x.dtype, device=x.device)
        return x[..., torch.as_tensor(idx, device=x.device)] * c - v * c

    def jacobian_entries(self, x: ArrayLike, k: int) -> Tuple[int, np.ndarray]:
        n_dims = int(x.shape[-1])
        self._ensure_resolved(n_dims)
        return int(self._bound_indices[k]), np.full(
            (_n_samples_of(x),), float(self._bound_coefs[k])
        )

    def grad(self, x: ArrayLike) -> np.ndarray:
        """Dense Jacobian ``(..., K, d)`` (analytic, mostly zeros)."""
        n_dims = int(x.shape[-1])
        self._ensure_resolved(n_dims)
        batch = _n_samples_of(x)
        out = np.zeros((batch, self._bound_indices.size, n_dims), dtype=np.float64)
        for k, (j, coef) in enumerate(zip(self._bound_indices, self._bound_coefs)):
            out[:, k, j] = coef
        return out


class EqualityConstraint(Constraint):
    """Equality constraint ``x_i = target`` written as ``|x_i - target|^2 - tol^2 <= 0``.

    The squared-distance form keeps the constraint a smooth function ``c <= 0``
    compatible with the general formulation of Sec. 3.4 / A1.3.
    """

    def __init__(
        self,
        indices: Sequence[int],
        target: Sequence[float],
        tol: float = 1e-6,
        *,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "equality")
        self.indices = np.asarray(list(indices), dtype=np.int64).reshape(-1)
        self.target = np.asarray(target, dtype=np.float64).reshape(-1)
        if self.target.size == 1 and self.indices.size > 1:
            self.target = np.full(self.indices.size, float(self.target[0]))
        if self.target.size != self.indices.size:
            raise ValueError("target must be scalar or match indices length.")
        self.tol = float(tol)
        self._k = 1

    @property
    def n_constraints(self) -> int:
        return 1

    def violation(self, x: ArrayLike) -> ArrayLike:
        lib = _lib_of(x)
        if lib is np:
            diff = np.asarray(x)[..., self.indices] - self.target
            return (diff**2).sum(axis=-1, keepdims=True) - self.tol**2
        tgt = torch.as_tensor(self.target, dtype=x.dtype, device=x.device)
        diff = x[..., torch.as_tensor(self.indices, device=x.device)] - tgt
        return (diff**2).sum(dim=-1, keepdim=True) - float(self.tol) ** 2

    def grad(self, x: ArrayLike) -> np.ndarray:
        arr = np.asarray(_to_numpy(x))
        diff = arr[..., self.indices] - self.target
        out = np.zeros(arr.shape[:-1] + (1, arr.shape[-1]), dtype=np.float64)
        out[..., 0, self.indices] = 2.0 * diff
        return out


class CallableConstraint(Constraint):
    """Wrap an arbitrary callable ``x -> c(x) <= 0`` (shape ``(..., K)``)."""

    def __init__(
        self,
        fn: Callable[[ArrayLike], ArrayLike],
        n_constraints: int = 1,
        *,
        grad_fn: Optional[Callable[[ArrayLike], np.ndarray]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "callable")
        self.fn = fn
        self._k = int(n_constraints)
        self.grad_fn = grad_fn

    @property
    def n_constraints(self) -> int:
        return self._k

    def violation(self, x: ArrayLike) -> ArrayLike:
        out = self.fn(x)
        lib = _lib_of(x)
        if lib is not np and not (_HAS_TORCH and isinstance(out, torch.Tensor)):
            out = torch.as_tensor(np.asarray(out), dtype=x.dtype, device=x.device)
        if out.ndim == x.ndim - 1:
            out = out[..., None]
        return out

    def grad(self, x: ArrayLike) -> np.ndarray:
        if self.grad_fn is None:
            raise NotImplementedError
        return np.asarray(self.grad_fn(x), dtype=np.float64)


class CombinedConstraint(Constraint):
    """Concatenation of several constraints (Sec. A1.3, Eq. 4 with K terms)."""

    def __init__(self, *constraints: Constraint, name: Optional[str] = None) -> None:
        constraints = tuple(_as_constraint(c) for c in constraints)
        if not constraints:
            raise ValueError("CombinedConstraint requires at least one constraint.")
        super().__init__(name=name or "combined")
        self.constraints: Tuple[Constraint, ...] = constraints

    @property
    def n_constraints(self) -> int:
        return int(sum(c.n_constraints for c in self.constraints))

    def violation(self, x: ArrayLike) -> ArrayLike:
        lib = _lib_of(x)
        parts = [c.violation(x) for c in self.constraints]
        if lib is np:
            return np.concatenate([np.asarray(p) for p in parts], axis=-1)
        return torch.cat([p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in parts], dim=-1)

    def grad(self, x: ArrayLike) -> np.ndarray:
        parts = []
        for c in self.constraints:
            try:
                parts.append(c.grad(x))
            except NotImplementedError:
                parts.append(_dense_fd_grad(c, x))
        return np.concatenate(parts, axis=-2)


def _as_constraint(obj: Any) -> Constraint:
    if isinstance(obj, Constraint):
        return obj
    if callable(obj):
        return CallableConstraint(obj)
    raise TypeError(f"Cannot interpret {obj!r} as a Constraint.")


def _n_samples_of(x: ArrayLike) -> int:
    shape = tuple(x.shape)
    return int(shape[0]) if len(shape) > 1 else 1


def combine_constraints(*constraints: Constraint) -> CombinedConstraint:
    """Union of constraints; the K log-sigmoids are summed (Sec. A1.3, Eq. 4)."""
    flat: List[Constraint] = []
    for c in constraints:
        if isinstance(c, CombinedConstraint):
            flat.extend(c.constraints)
        else:
            flat.append(_as_constraint(c))
    return CombinedConstraint(*flat)


def interval_constraint(
    indices: Optional[Sequence[int]] = None,
    lower=None,
    upper=None,
    *,
    name: Optional[str] = None,
) -> IntervalConstraint:
    """Build an interval constraint ``lower <= x[indices] <= upper``."""
    return IntervalConstraint(indices, lower, upper, name=name)


def upper_bound_constraint(indices: Sequence[int], upper, *, name=None) -> IntervalConstraint:
    """Upper-bound interval constraint ``x[indices] <= upper`` (Sec. 3.4: c = x - u)."""
    return IntervalConstraint(indices, None, upper, name=name or "upper_bound")


def lower_bound_constraint(indices: Sequence[int], lower, *, name=None) -> IntervalConstraint:
    """Lower-bound interval constraint ``x[indices] >= lower``."""
    return IntervalConstraint(indices, lower, None, name=name or "lower_bound")


def box_constraint(
    indices: Sequence[int],
    lower,
    upper,
    *,
    name: Optional[str] = None,
) -> IntervalConstraint:
    """Two-sided interval constraint."""
    return IntervalConstraint(indices, lower, upper, name=name or "box")


def equality_constraint(
    indices: Sequence[int], target, tol: float = 1e-6, *, name=None
) -> EqualityConstraint:
    """Equality constraint ``x[indices] = target`` (squared tolerance form)."""
    return EqualityConstraint(indices, target, tol=tol, name=name)


def constraint_violation(constraint: Any, x: ArrayLike) -> np.ndarray:
    """Evaluate ``c(x)`` for a constraint or bare callable, as NumPy."""
    return np.asarray(_to_numpy(_as_constraint(constraint).violation(x)), dtype=np.float64)


def constraint_satisfaction(
    samples: ArrayLike, constraint: Any, tol: float = 0.0
) -> Dict[str, Any]:
    """Fraction of samples satisfying each constraint (and all constraints)."""
    c = constraint_violation(constraint, samples)
    ok = c <= tol
    return {
        "per_constraint": ok.mean(axis=0),
        "overall": float(ok.all(axis=1).mean()),
        "mean_violation": c.mean(axis=0),
        "max_violation": c.max(axis=0),
    }


# --------------------------------------------------------------------------- #
# Constraint score  grad log sigmoid(-s(t) c(x))   (Sec. 3.4 Eq. 2, Alg. 1)
# --------------------------------------------------------------------------- #


def denoise_from_score(sde: SDE, x_t: ArrayLike, t: Any, score: ArrayLike) -> ArrayLike:
    """One-step Tweedie/Jacobi denoised estimate used in Algorithm 1.

    ``x_~0 = (x_t + sigma(t)^2 * s) / mu(t)``
    """
    if hasattr(sde, "twedie_denoise"):
        try:
            return sde.twedie_denoise(x_t, t, score)
        except Exception:  # pragma: no cover - fall through to explicit formula
            pass
    mu = _marginal_mean(sde, t)
    sigma = _marginal_std(sde, t)
    x_arr = np.asarray(_to_numpy(x_t), dtype=np.float64)
    s_arr = np.asarray(_to_numpy(score), dtype=np.float64)
    return (x_arr + (sigma**2) * s_arr) / max(mu, 1e-300)


def _dense_fd_grad(constraint: Constraint, x: np.ndarray, eps: float = DEFAULT_FD_EPS) -> np.ndarray:
    """Finite-difference Jacobian ``(B, K, d)`` for a constraint without analytic grad."""
    x = np.asarray(x, dtype=np.float64)
    b, d = x.shape[0], x.shape[-1]
    c0 = constraint_violation(constraint, x)
    k = c0.shape[-1]
    out = np.zeros((b, k, d), dtype=np.float64)
    for j in range(d):
        xp = x.copy()
        xp[:, j] += eps
        xm = x.copy()
        xm[:, j] -= eps
        cp = constraint_violation(constraint, xp)
        cm = constraint_violation(constraint, xm)
        out[:, :, j] = (cp - cm) / (2.0 * eps)
    return out


def _torch_constraint_score(
    constraint: Constraint,
    x_denoised: np.ndarray,
    scale: float,
    sign: float,
    weight,
) -> Optional[np.ndarray]:
    """Constraint score via torch autograd on the denoised estimate."""
    if not _HAS_TORCH:
        return None
    try:
        xd = torch.as_tensor(np.asarray(x_denoised))
        xd = xd.detach().clone().requires_grad_(True)
        c = constraint.violation(xd)
        if c.ndim == xd.ndim - 1:
            c = c.unsqueeze(-1)
        z = float(sign) * float(scale) * c
        logp = log_sigmoid(z)
        if weight is not None:
            w = torch.as_tensor(np.asarray(weight, dtype=np.float64), dtype=logp.dtype)
            logp = logp * w
        obj = logp.sum()
        (grad,) = torch.autograd.grad(obj, xd, allow_unused=False)
    except Exception:
        return None
    return np.asarray(grad.detach().cpu().numpy(), dtype=np.float64)


def constraint_score(
    constraint: Any,
    x_denoised: ArrayLike,
    t: Any,
    sde: SDE,
    *,
    scale_fn: Optional[Callable[[Any], float]] = None,
    scale: Union[str, float, Callable, None] = None,
    sign: float = DEFAULT_SIGN,
    weight: Union[float, Sequence[float]] = 1.0,
    clip_scale: Optional[float] = None,
    differentiate: str = "denoised",
    fd_eps: float = DEFAULT_FD_EPS,
) -> np.ndarray:
    """``grad_{x} log sigmoid(sign * s(t) * c(x_denoised))`` (Sec. 3.4, Eq. 2).

    Parameters
    ----------
    constraint:
        :class:`Constraint` (or a callable returning ``c(x) <= 0``).
    x_denoised:
        Denoised estimate ``x_~0`` (shape ``(B, d)``).
    t:
        Current diffusion time (used only to evaluate the scaling function).
    scale_fn / scale:
        ``scale_fn`` wins if given; otherwise built from ``scale`` via
        :func:`get_scaling_function`.
    sign:
        ``-1`` for the paper's ``log sigmoid(-s(t)c)`` (default).
    weight:
        Scalar or per-constraint weight multiplying each log-sigmoid term.
    clip_scale:
        Upper bound on ``s(t)``.
    differentiate:
        ``"denoised"``: differentiate w.r.t. ``x_~0`` (Algorithm 1 notation
        ``grad_x`` with ``x = x_~0``).  ``"sample"``: additionally propagate the
        chain rule through the denoising map, i.e. divide by ``mu(t)``.
    """
    cons = _as_constraint(constraint)
    if scale_fn is None:
        scale_fn = get_scaling_function(scale, sde, clip=clip_scale)
    s_t = float(scale_fn(t))
    if clip_scale is not None:
        s_t = float(min(s_t, float(clip_scale)))

    x_arr = np.asarray(_to_numpy(x_denoised), dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[None, :]
    w = None if np.isscalar(weight) and float(weight) == 1.0 else weight

    grad = _torch_constraint_score(cons, x_arr, s_t, sign, w)

    if grad is None:
        # Analytic / finite-difference fallback in NumPy.
        c = constraint_violation(cons, x_arr)
        z = float(sign) * s_t * c  # (B, K)
        coeff = (1.0 - np.asarray(_sigmoid(z)))  # d/dz log sigmoid(z)
        if w is not None:
            w_arr = np.broadcast_to(np.asarray(w, dtype=np.float64).reshape(-1), coeff.shape[1:])
            coeff = coeff * w_arr[None, :]
        coeff = coeff * float(sign) * s_t  # dlog sigma/d c_k * dc_k/dx
        grad = np.zeros_like(x_arr)
        # Try the sparse analytic Jacobian first, else fall back to finite diffs.
        entries_ok = True
        for k in range(c.shape[-1]):
            try:
                j, coef_k = cons.jacobian_entries(x_arr, k)
            except NotImplementedError:
                entries_ok = False
                break
            grad[:, int(j)] += coeff[:, k] * np.asarray(coef_k, dtype=np.float64).reshape(-1)
        if not entries_ok:
            jac = _dense_fd_grad(cons, x_arr, eps=fd_eps)  # (B, K, d)
            grad = np.einsum("bk,bkd->bd", coeff, jac)

    if differentiate in ("sample", "x_t", "xt"):
        mu = _marginal_mean(sde, t)
        grad = grad / max(mu, 1e-300)
    elif differentiate not in ("denoised", "x_denoised", "hat"):
        raise ValueError(f"Unknown differentiate mode: {differentiate!r}")

    if w is None and np.isscalar(weight) and float(weight) != 1.0:
        grad = grad * float(weight)
    return grad


def guidance_scale_factor(sde: SDE, t: Any, scale: Union[str, float, Callable, None] = None,
                          clip_scale: Optional[float] = None) -> float:
    """Convenience accessor for the scalar guidance scaling ``s(t)``."""
    return float(get_scaling_function(scale, sde, clip=clip_scale)(t))


def guided_score(
    score: ArrayLike,
    constraint: Any,
    x_t: ArrayLike,
    t: Any,
    sde: SDE,
    *,
    guidance_scale: float = 1.0,
    scale: Union[str, float, Callable, None] = None,
    scale_fn: Optional[Callable[[Any], float]] = None,
    sign: float = DEFAULT_SIGN,
    weight: Union[float, Sequence[float]] = 1.0,
    clip_scale: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    differentiate: str = "denoised",
    active_mask: Optional[np.ndarray] = None,
    fd_eps: float = DEFAULT_FD_EPS,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Attach the constraint score to a marginal score estimate.

    Returns ``(s_tilde, info)`` with ``s_tilde = s + guidance_scale * grad`` where
    ``grad = grad log sigmoid(sign * s(t) * c(x_~0))`` and ``x_~0`` is the denoised
    estimate of ``x_t`` under the (unguided) score ``s``.
    """
    s_arr = np.asarray(_to_numpy(score), dtype=np.float64)
    x_arr = np.asarray(_to_numpy(x_t), dtype=np.float64)
    x_denoised = denoise_from_score(sde, x_arr, t, s_arr)

    grad = constraint_score(
        constraint,
        x_denoised,
        t,
        sde,
        scale_fn=scale_fn,
        scale=scale,
        sign=sign,
        weight=weight,
        clip_scale=clip_scale,
        differentiate=differentiate,
        fd_eps=fd_eps,
    )
    if active_mask is not None:
        grad = grad * np.asarray(active_mask, dtype=np.float64).reshape(1, -1)

    raw_norm = float(np.linalg.norm(grad))
    if clip_grad_norm is not None and raw_norm > clip_grad_norm > 0:
        grad = grad * (float(clip_grad_norm) / raw_norm)

    s_tilde = s_arr + float(guidance_scale) * grad
    info = {
        "grad_norm": raw_norm,
        "score_norm": float(np.linalg.norm(s_arr)),
        "constraint_score_norm": float(np.linalg.norm(grad)),
        "denoised_norm": float(np.linalg.norm(x_denoised)),
    }
    return s_tilde, info


# --------------------------------------------------------------------------- #
# Algorithm 1: reverse step with guidance
# --------------------------------------------------------------------------- #


def guidance_step(
    sde: SDE,
    x: np.ndarray,
    t_cur: float,
    t_next: float,
    score: ArrayLike,
    constraint: Any,
    *,
    noise: Optional[np.ndarray] = None,
    scale: Union[str, float, Callable, None] = None,
    scale_fn: Optional[Callable[[Any], float]] = None,
    sign: float = DEFAULT_SIGN,
    weight: Union[float, Sequence[float]] = 1.0,
    guidance_scale: float = 1.0,
    clip_scale: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    differentiate: str = "denoised",
    active_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    fd_eps: float = DEFAULT_FD_EPS,
    return_info: bool = False,
):
    """One guided reverse-SDE step ``x_{t_cur} -> x_{t_next}`` (Algorithm 1).

    ``x_next = x - (f(x, t_next) - g(t_next)^2 * s~) * dt - g(t_next) sqrt(dt) eps``
    with ``dt = t_cur - t_next > 0`` and ``s~`` the guided score.
    """
    x = np.asarray(x, dtype=np.float64)
    dt = float(t_cur) - float(t_next)
    if dt <= 0:
        raise ValueError(f"guidance_step expects t_cur > t_next, got {t_cur} -> {t_next}.")
    if noise is None:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.standard_normal(x.shape)

    s_tilde, info = guided_score(
        score,
        constraint,
        x,
        t_cur,
        sde,
        guidance_scale=guidance_scale,
        scale=scale,
        scale_fn=scale_fn,
        sign=sign,
        weight=weight,
        clip_scale=clip_scale,
        clip_grad_norm=clip_grad_norm,
        differentiate=differentiate,
        active_mask=active_mask,
        fd_eps=fd_eps,
    )

    f = np.asarray(_to_numpy(sde.drift(x, t_next)), dtype=np.float64)
    g = float(np.asarray(_to_numpy(sde.diffusion(t_next))).reshape(-1)[0])
    drift_term = f - (g**2) * s_tilde
    x_next = x - drift_term * dt - g * math.sqrt(dt) * np.asarray(noise, dtype=np.float64)
    info["dt"] = dt
    info["g"] = g
    info["t_cur"] = float(t_cur)
    info["t_next"] = float(t_next)
    return (x_next, info) if return_info else x_next


# --------------------------------------------------------------------------- #
# Guidance configuration / result containers
# --------------------------------------------------------------------------- #


@dataclass
class GuidanceConfig:
    """Numerical settings for guided reverse-SDE sampling (Algorithm 1)."""

    n_steps: int = DEFAULT_GUIDANCE_STEPS
    t_min: float = DEFAULT_T_MIN
    t_max: float = DEFAULT_T_MAX
    self_recurrence: int = DEFAULT_SELF_RECURRENCE  # r in Algorithm 1
    scale: Union[str, float, Callable, None] = DEFAULT_SCALE  # s(t)
    scale_fn: Optional[Callable[[Any], float]] = None
    sign: float = DEFAULT_SIGN
    clip_scale: Optional[float] = None
    guidance_scale: float = 1.0
    weight: Union[float, Sequence[float]] = 1.0
    clip_grad_norm: Optional[float] = None
    differentiate: str = "denoised"
    n_samples: int = 1000
    batch_size: Optional[int] = None
    clamp_every_step: bool = True  # clamp exact (model-based) conditions
    return_trajectory: bool = False
    return_times: bool = False
    seed: Optional[int] = None
    device: Optional[str] = None
    dtype: Any = None
    fd_eps: float = DEFAULT_FD_EPS
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        if callable(d.get("scale")):
            d["scale"] = getattr(self.scale, "__name__", "callable")
        if callable(d.get("scale_fn")):
            d["scale_fn"] = getattr(self.scale_fn, "__name__", "callable")
        if d.get("dtype") is not None:
            d["dtype"] = str(d["dtype"])
        return d

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "GuidanceConfig":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in dict(cfg).items() if k in known}
        extra = {k: v for k, v in dict(cfg).items() if k not in known}
        out = cls(**kwargs)
        if extra:
            out.extra.update(extra)
        return out

    def times(self, sde: Optional[SDE] = None) -> np.ndarray:
        return np.linspace(float(self.t_max), float(self.t_min), int(self.n_steps) + 1)


@dataclass
class GuidanceResult:
    """Output of a guided sampling run (Algorithm 1)."""

    samples: np.ndarray
    times: Optional[np.ndarray] = None
    trajectory: Optional[np.ndarray] = None
    n_steps: int = DEFAULT_GUIDANCE_STEPS
    self_recurrence: int = DEFAULT_SELF_RECURRENCE
    condition_mask: Optional[np.ndarray] = None
    condition_values: Optional[np.ndarray] = None
    constraint_satisfaction: Optional[Dict[str, Any]] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    mode: str = "guidance"

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def as_array(self) -> np.ndarray:
        return self.samples


# --------------------------------------------------------------------------- #
# Algorithm 1 in full
# --------------------------------------------------------------------------- #


def _initial_state(
    sde: SDE,
    n_samples: int,
    dim: int,
    rng: np.random.Generator,
    x_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    if x_init is not None:
        x = np.asarray(_to_numpy(x_init), dtype=np.float64)
        if x.ndim == 1:
            x = np.broadcast_to(x, (n_samples, dim)).copy()
        return x
    return _sample_prior(sde, (n_samples, dim), rng)


def _clamp_conditions(
    x: np.ndarray,
    condition_mask: Optional[np.ndarray],
    condition_values: Optional[np.ndarray],
) -> np.ndarray:
    """Set exact/observed coordinates to their clean values (Sec. 3.3).

    ``x^{M_C}_t = (1 - M_C) * x_t + M_C * x_0``.
    """
    if condition_mask is None or condition_values is None:
        return x
    m = np.asarray(_to_numpy(condition_mask), dtype=np.float64).reshape(1, -1)
    v = np.asarray(_to_numpy(condition_values), dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if v.shape[0] == 1 and x.shape[0] > 1:
        v = np.broadcast_to(v, (x.shape[0], v.shape[-1]))
    if m.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"condition_mask width {m.shape[-1]} does not match sample dim {x.shape[-1]}."
        )
    return (1.0 - m) * x + m * v


def _reverse_sde_chunk(
    score_fn: Callable,
    sde: SDE,
    constraint: Any,
    *,
    n_samples: int,
    dim: int,
    n_steps: int,
    t_min: float,
    t_max: float,
    self_recurrence: int,
    scale: Union[str, float, Callable, None],
    scale_fn: Optional[Callable[[Any], float]],
    sign: float,
    weight: Union[float, Sequence[float]],
    guidance_scale: float,
    clip_scale: Optional[float],
    clip_grad_norm: Optional[float],
    differentiate: str,
    condition_mask: Optional[np.ndarray],
    condition_values: Optional[np.ndarray],
    active_mask: Optional[np.ndarray],
    clamp_every_step: bool,
    rng: np.random.Generator,
    x_init: Optional[np.ndarray],
    fd_eps: float,
    return_trajectory: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Run Algorithm 1 for one chunk of samples (NumPy integration)."""
    if scale_fn is None:
        scale_fn = get_scaling_function(scale, sde, clip=clip_scale)

    times = np.linspace(float(t_max), float(t_min), int(n_steps) + 1)
    x = _initial_state(sde, n_samples, dim, rng, x_init)
    x = _clamp_conditions(x, condition_mask, condition_values)

    n_inner = max(1, int(self_recurrence))  # r = 0 -> single inner evaluation

    trajectory: List[np.ndarray] = []
    grad_norms: List[float] = []
    if return_trajectory:
        trajectory.append(x.copy())

    for i in range(int(n_steps)):
        t_cur = float(times[i])  # paper's t_{i+1}: the point currently held
        t_next = float(times[i + 1])  # paper's t_i: next (less noisy) time
        x_carry = x.copy()  # x_{t_{i+1}} used by the forward resampling step
        x_next = x
        info: Dict[str, float] = {}
        for j in range(n_inner):
            eps = rng.standard_normal(x.shape)
            score = _call_score_fn(score_fn, x, t_cur)
            x_next, info = guidance_step(
                sde,
                x,
                t_cur,
                t_next,
                score,
                constraint,
                noise=eps,
                scale_fn=scale_fn,
                sign=sign,
                weight=weight,
                guidance_scale=guidance_scale,
                clip_scale=clip_scale,
                clip_grad_norm=clip_grad_norm,
                differentiate=differentiate,
                active_mask=active_mask,
                rng=rng,
                fd_eps=fd_eps,
                return_info=True,
            )
            if clamp_every_step:
                x_next = _clamp_conditions(x_next, condition_mask, condition_values)
            grad_norms.append(float(info.get("grad_norm", float("nan"))))
            if self_recurrence > 0 and j < n_inner - 1:
                # Resample the future point using the forward SDE (Algorithm 1).
                eps2 = rng.standard_normal(x.shape)
                f_fwd = np.asarray(_to_numpy(sde.drift(x_carry, t_next)), dtype=np.float64)
                g_fwd = float(np.asarray(_to_numpy(sde.diffusion(t_next))).reshape(-1)[0])
                dt = t_cur - t_next
                x = x_next + f_fwd * dt + g_fwd * math.sqrt(dt) * eps2
                x = _clamp_conditions(x, condition_mask, condition_values)
            else:
                x = x_next
        x = x_next
        if not clamp_every_step:
            x = _clamp_conditions(x, condition_mask, condition_values)
        if return_trajectory:
            trajectory.append(x.copy())

    x = _clamp_conditions(x, condition_mask, condition_values)
    stats = {
        "grad_norm_mean": float(np.nanmean(grad_norms)) if grad_norms else 0.0,
        "grad_norm_max": float(np.nanmax(grad_norms)) if grad_norms else 0.0,
        "times": times,
        "scale_at_t_min": float(scale_fn(float(t_min))),
    }
    traj = np.stack(trajectory, axis=1) if (return_trajectory and trajectory) else None
    return x, traj, times, stats


def general_guidance(
    score_fn: Callable,
    sde: SDE,
    constraint: Any,
    *,
    n_samples: int = 1000,
    n_steps: int = DEFAULT_GUIDANCE_STEPS,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    self_recurrence: int = DEFAULT_SELF_RECURRENCE,
    scale: Union[str, float, Callable, None] = DEFAULT_SCALE,
    scale_fn: Optional[Callable[[Any], float]] = None,
    sign: float = DEFAULT_SIGN,
    weight: Union[float, Sequence[float]] = 1.0,
    guidance_scale: float = 1.0,
    clip_scale: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    differentiate: str = "denoised",
    condition_mask: Optional[np.ndarray] = None,
    condition_values: Optional[np.ndarray] = None,
    active_mask: Optional[np.ndarray] = None,
    clamp_every_step: bool = True,
    chunk_size: Optional[int] = None,
    seed: Optional[int] = None,
    x_init: Optional[np.ndarray] = None,
    fd_eps: float = DEFAULT_FD_EPS,
    return_trajectory: bool = False,
    return_result: bool = True,
    constraint_for_eval: Optional[Any] = None,
):
    """Algorithm 1 -- general guidance for arbitrary constraint sets.

    Parameters mirror Algorithm 1's ``Require`` list: ``n_steps`` (``T``),
    ``t_min``/``t_max`` (``T_min``/``T_max``), ``self_recurrence`` (``r``),
    ``scale``/``scale_fn`` (``s(t)``) and ``constraint`` (``c(x)``).  Point
    conditions are passed through ``condition_mask``/``condition_values`` and are
    clamped exactly (model-based conditioning).

    Returns a :class:`GuidanceResult` (or just the samples if
    ``return_result=False``).
    """
    if sde is None:
        sde = get_sde("vesde")
    t_min = float(sde.t_min if t_min is None else t_min)
    t_max = float(sde.t_max if t_max is None else t_max)
    if t_max <= t_min:
        raise ValueError(f"t_max ({t_max}) must exceed t_min ({t_min}).")
    if n_steps < MIN_RECOMMENDED_STEPS:
        # >= 50 steps are reported to be sufficient (Sec. A3.1); warn instead of fail.
        pass

    cons = _as_constraint(constraint)
    if np.isscalar(constraint) or (isinstance(constraint, (list, tuple)) and not constraint):
        raise ValueError("general_guidance requires a constraint.")

    dim = None
    if condition_mask is not None:
        dim = int(np.asarray(_to_numpy(condition_mask)).reshape(-1).shape[0])
    if x_init is not None:
        xi = np.asarray(_to_numpy(x_init))
        dim = int(xi.shape[-1]) if xi.ndim > 1 else int(xi.shape[0])
    if dim is None:
        raise ValueError(
            "Cannot infer the joint dimension: pass x_init or condition_mask "
            "(or use GuidedSampler, which knows the model dimensionality)."
        )

    rng = np.random.default_rng(seed)
    chunk = int(chunk_size) if chunk_size else int(n_samples)
    chunk = max(1, min(chunk, int(n_samples)))

    samples: List[np.ndarray] = []
    trajs: List[np.ndarray] = []
    times = None
    stats: Dict[str, Any] = {}
    remaining = int(n_samples)
    while remaining > 0:
        k = min(chunk, remaining)
        xs, traj, times, stats = _reverse_sde_chunk(
            score_fn,
            sde,
            cons,
            n_samples=k,
            dim=dim,
            n_steps=int(n_steps),
            t_min=t_min,
            t_max=t_max,
            self_recurrence=int(self_recurrence),
            scale=scale,
            scale_fn=scale_fn,
            sign=sign,
            weight=weight,
            guidance_scale=guidance_scale,
            clip_scale=clip_scale,
            clip_grad_norm=clip_grad_norm,
            differentiate=differentiate,
            condition_mask=condition_mask,
            condition_values=condition_values,
            active_mask=active_mask,
            clamp_every_step=clamp_every_step,
            rng=rng,
            x_init=x_init,
            fd_eps=fd_eps,
            return_trajectory=return_trajectory,
        )
        samples.append(xs)
        if traj is not None:
            trajs.append(traj)
        remaining -= k

    x_out = np.concatenate(samples, axis=0) if len(samples) > 1 else samples[0]
    traj_out = np.concatenate(trajs, axis=0) if trajs else None

    sat = None
    try:
        sat = constraint_satisfaction(x_out, cons)
    except Exception:  # pragma: no cover - constraint may be non-vectorisable here
        sat = None

    if not return_result:
        return x_out

    return GuidanceResult(
        samples=x_out,
        times=times,
        trajectory=traj_out,
        n_steps=int(n_steps),
        self_recurrence=int(self_recurrence),
        condition_mask=None if condition_mask is None else np.asarray(_to_numpy(condition_mask)),
        condition_values=None if condition_values is None else np.asarray(_to_numpy(condition_values)),
        constraint_satisfaction=sat,
        stats=stats,
        mode="general_guidance",
    )


# Friendly alias with the paper's terminology.
sample_with_guidance = general_guidance


def sample_interval_conditional(
    score_fn: Callable,
    sde: SDE,
    *,
    dim: int,
    upper: Union[None, float, Sequence[float]] = None,
    lower: Union[None, float, Sequence[float]] = None,
    indices: Optional[Sequence[int]] = None,
    n_samples: int = 1000,
    n_steps: int = DEFAULT_GUIDANCE_STEPS,
    self_recurrence: int = DEFAULT_SELF_RECURRENCE,
    scale: Union[str, float, Callable, None] = DEFAULT_SCALE,
    sign: float = DEFAULT_SIGN,
    guidance_scale: float = 1.0,
    clip_scale: Optional[float] = None,
    condition_mask: Optional[np.ndarray] = None,
    condition_values: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    x_init: Optional[np.ndarray] = None,
    return_result: bool = True,
    **kwargs,
):
    """Sample with an interval constraint ``lower <= x[indices] <= upper``.

    This is the Sec. 4.4 (Hodgkin-Huxley energy interval) use case: a point
    condition is handled model-based (``condition_mask``/``condition_values``)
    while the interval is enforced through guidance.
    """
    cons = interval_constraint(indices, lower, upper)
    if x_init is None:
        x_zero = np.zeros((1, int(dim)), dtype=np.float64)
    else:
        x_zero = np.asarray(_to_numpy(x_init), dtype=np.float64)
    return general_guidance(
        score_fn,
        sde,
        cons,
        n_samples=n_samples,
        n_steps=n_steps,
        t_min=sde.t_min,
        t_max=sde.t_max,
        self_recurrence=self_recurrence,
        scale=scale,
        sign=sign,
        guidance_scale=guidance_scale,
        clip_scale=clip_scale,
        condition_mask=condition_mask,
        condition_values=condition_values,
        seed=seed,
        x_init=None if x_init is None else x_zero,
        return_result=return_result,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# High-level guided sampler
# --------------------------------------------------------------------------- #


class GuidedSampler:
    """Guided reverse-SDE sampling for a trained Simformer.

    Combines the two conditioning mechanisms of the paper:

    * exact point conditions -> clamped (model-based), via ``condition_mask`` and
      ``condition_values`` (Sec. 3.3),
    * interval / arbitrary set constraints -> diffusion guidance with the
      general formulation of Sec. 3.4 and Algorithm 1 (Sec. A3.3).

    Example
    -------
    >>> sampler = GuidedSampler.from_model(model, sde=sde, tokenizer=tok)
    >>> res = sampler.sample(upper_bound_constraint([5], 0.0), n_samples=100)
    >>> res.constraint_satisfaction["overall"]  # fraction satisfying c <= 0
    """

    def __init__(
        self,
        model: Any = None,
        sde: Optional[SDE] = None,
        tokenizer: Any = None,
        *,
        score_fn: Optional[Callable] = None,
        config: Optional[GuidanceConfig] = None,
        condition_mask: Optional[np.ndarray] = None,
        condition_values: Optional[np.ndarray] = None,
        attention_mask: Optional[Any] = None,
        attention_mask_fn: Optional[Callable] = None,
        function_values: Optional[Any] = None,
        dim: Optional[int] = None,
        n_parameters: Optional[int] = None,
        n_data: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Any = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sde = sde if sde is not None else get_sde("vesde")
        self.config = config if config is not None else GuidanceConfig()
        self.attention_mask = attention_mask
        self.attention_mask_fn = attention_mask_fn
        self.function_values = function_values
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size

        self.n_parameters = n_parameters
        self.n_data = n_data
        if tokenizer is not None:
            self.n_parameters = getattr(tokenizer, "n_parameter_variables", n_parameters)
            self.n_data = getattr(tokenizer, "n_data_variables", n_data)

        self.dim = int(dim) if dim is not None else self._infer_dim()
        self._score_fn = score_fn
        self.condition_mask = None if condition_mask is None else np.asarray(condition_mask)
        self.condition_values = None if condition_values is None else np.asarray(condition_values)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_model(cls, model: Any, **kwargs) -> "GuidedSampler":
        return cls(model, **kwargs)

    def _infer_dim(self) -> Optional[int]:
        if self.condition_mask is not None:
            return int(np.asarray(self.condition_mask).reshape(-1).shape[0])
        for attr in ("n_variables", "input_dim"):
            if self.tokenizer is not None and hasattr(self.tokenizer, attr):
                val = getattr(self.tokenizer, attr)
                if callable(val):
                    val = val()
                try:
                    return int(val)
                except Exception:  # pragma: no cover
                    continue
        if self.n_parameters is not None and self.n_data is not None:
            return int(self.n_parameters) + int(self.n_data)
        return None

    # -- score function ---------------------------------------------------
    def build_score_fn(
        self,
        condition_mask: Optional[np.ndarray] = None,
        condition_values: Optional[np.ndarray] = None,
    ) -> Callable:
        """NumPy score function ``(x, t) -> grad log p_t`` (Sec. 3.3)."""
        if self._score_fn is not None and condition_mask is None and condition_values is None:
            return self._score_fn
        if self.model is None:
            raise ValueError("GuidedSampler needs either `model` or `score_fn`.")
        from .sampling import make_score_fn  # local import to avoid a cycle

        cm = self.condition_mask if condition_mask is None else condition_mask
        cv = self.condition_values if condition_values is None else condition_values
        return make_score_fn(
            self.model,
            self.sde,
            condition_mask=cm,
            condition_values=cv,
            attention_mask=self.attention_mask,
            attention_mask_fn=self.attention_mask_fn,
            function_values=self.function_values,
            dim=self.dim,
            device=self.device,
            dtype=self.dtype,
            batch_chunk=self.chunk_size,
        )

    # -- sampling ---------------------------------------------------------
    def sample(
        self,
        constraint: Any,
        *,
        n_samples: Optional[int] = None,
        condition_mask: Optional[np.ndarray] = None,
        condition_values: Optional[np.ndarray] = None,
        n_steps: Optional[int] = None,
        self_recurrence: Optional[int] = None,
        scale: Union[str, float, Callable, None] = None,
        sign: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        clip_scale: Optional[float] = None,
        clip_grad_norm: Optional[float] = None,
        differentiate: Optional[str] = None,
        active_mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        x_init: Optional[np.ndarray] = None,
        return_result: bool = True,
        **kwargs,
    ):
        """Run Algorithm 1 with `constraint` (matched to the model's dims)."""
        cfg = self.config
        cm = self.condition_mask if condition_mask is None else np.asarray(condition_mask)
        cv = self.condition_values if condition_values is None else np.asarray(condition_values)
        score_fn = self.build_score_fn(cm, cv)
        dim = self.dim
        if dim is None and cm is not None:
            dim = int(np.asarray(cm).reshape(-1).shape[0])
        if dim is None:
            raise ValueError("GuidedSampler.dim is unknown; pass `dim=` at construction.")
        return general_guidance(
            score_fn,
            self.sde,
            constraint,
            n_samples=int(n_samples if n_samples is not None else cfg.n_samples),
            n_steps=int(n_steps if n_steps is not None else cfg.n_steps),
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            self_recurrence=int(
                self_recurrence if self_recurrence is not None else cfg.self_recurrence
            ),
            scale=cfg.scale if scale is None else scale,
            scale_fn=cfg.scale_fn,
            sign=cfg.sign if sign is None else sign,
            weight=cfg.weight,
            guidance_scale=(
                guidance_scale if guidance_scale is not None else cfg.guidance_scale
            ),
            clip_scale=cfg.clip_scale if clip_scale is None else clip_scale,
            clip_grad_norm=(
                clip_grad_norm if clip_grad_norm is not None else cfg.clip_grad_norm
            ),
            differentiate=cfg.differentiate if differentiate is None else differentiate,
            condition_mask=cm,
            condition_values=cv,
            active_mask=active_mask,
            clamp_every_step=cfg.clamp_every_step,
            chunk_size=self.chunk_size,
            seed=cfg.seed if seed is None else seed,
            x_init=x_init,
            fd_eps=cfg.fd_eps,
            return_trajectory=cfg.return_trajectory,
            return_result=return_result,
            **kwargs,
        )

    # -- convenient entry points -----------------------------------------
    def sample_intervals(
        self,
        *,
        upper: Union[None, float, Sequence[float]] = None,
        lower: Union[None, float, Sequence[float]] = None,
        indices: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """Sample subject to ``lower <= x[indices] <= upper``."""
        cons = interval_constraint(indices, lower, upper)
        return self.sample(cons, **kwargs)

    def posterior_intervals(
        self,
        x_obs: np.ndarray,
        *,
        upper: Union[None, float, Sequence[float]] = None,
        lower: Union[None, float, Sequence[float]] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = None,
        **kwargs,
    ):
        """Posterior ``p(theta | x_obs)`` further constrained by an interval on theta.

        The observations are exact (model-based); the interval on the parameters is
        enforced through guidance -- the protocol of Sec. 4.4 / Fig. A16b.
        """
        from .condition_masks import posterior_condition_mask

        n_par = int(self.n_parameters if self.n_parameters is not None else 0)
        n_dat = int(self.n_data if self.n_data is not None else 0)
        if n_par is None or n_dat is None or (n_par + n_dat) == 0:
            raise ValueError("posterior_intervals requires n_parameters and n_data.")
        mask = posterior_condition_mask(n_par, n_dat)
        values = np.zeros((self.dim,), dtype=np.float64)
        x_obs = np.asarray(_to_numpy(x_obs), dtype=np.float64).reshape(-1)
        if x_obs.size != n_dat:
            raise ValueError(f"x_obs has {x_obs.size} entries, expected {n_dat}.")
        values[n_par : n_par + n_dat] = x_obs
        # Guidance may only act on the latent parameters.
        active = np.zeros((self.dim,), dtype=np.float64)
        active[:n_par] = 1.0
        if indices is None:
            indices = list(range(n_par))
        kwargs.setdefault("active_mask", active)
        return self.sample_intervals(
            upper=upper, lower=lower, indices=indices, n_samples=n_samples,
            condition_mask=mask, condition_values=values, **kwargs,
        )

    def likelihood_intervals(
        self,
        theta_obs: np.ndarray,
        *,
        upper=None,
        lower=None,
        indices: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = None,
        **kwargs,
    ):
        """Likelihood ``p(x | theta_obs)`` with an interval constraint on x."""
        from .condition_masks import likelihood_condition_mask

        n_par = int(self.n_parameters)
        n_dat = int(self.n_data)
        mask = likelihood_condition_mask(n_par, n_dat)
        values = np.zeros((self.dim,), dtype=np.float64)
        theta_obs = np.asarray(_to_numpy(theta_obs), dtype=np.float64).reshape(-1)
        if theta_obs.size != n_par:
            raise ValueError(f"theta_obs has {theta_obs.size} entries, expected {n_par}.")
        values[:n_par] = theta_obs
        active = np.zeros((self.dim,), dtype=np.float64)
        active[n_par : n_par + n_dat] = 1.0
        if indices is None:
            indices = list(range(n_par, n_par + n_dat))
        kwargs.setdefault("active_mask", active)
        return self.sample_intervals(
            upper=upper, lower=lower, indices=indices, n_samples=n_samples,
            condition_mask=mask, condition_values=values, **kwargs,
        )
