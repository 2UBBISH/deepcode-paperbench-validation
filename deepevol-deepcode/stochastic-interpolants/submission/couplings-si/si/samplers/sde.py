"""Stochastic (SDE) samplers for stochastic interpolants with data-dependent couplings.

This module implements the two stochastic generative models given in Corollary 3.1 of
*Stochastic Interpolants with Data-Dependent Couplings*:

Forward SDE (Eq. 11)::

    dX_t^F = b_t(X_t^F) dt - eps_t * gamma_t^{-1} g_t(X_t^F) dt + sqrt(2 eps_t) dW_t
    X_{t=1}^F ~ rho_1(x_1)   if   X_{t=0}^F ~ rho_0(x_0)

Backward SDE (Eq. 13)::

    dX_t^R = b_t(X_t^R) dt + eps_t * gamma_t^{-1} g_t(X_t^R) dt + sqrt(2 eps_t) dW_t
    X_{t=0}^R ~ rho_0(x_0)   if   X_{t=1}^R ~ rho_1(x_1)

using the score identity of Theorem 3.1, ``nabla log rho_t(x) = -gamma_t^{-1} g_t(x)``,
valid for every ``t`` with ``gamma_t != 0`` (Eq. 6).  The forward SDE is integrated with
*increasing* ``t`` from 0 to 1 (it turns ``rho_0`` into ``rho_1``); the backward SDE is
integrated with *decreasing* ``t`` from 1 to 0 (it turns ``rho_1`` into ``rho_0``) -- this
is exactly the statement of Corollary 3.1.  Both are discretized with Euler--Maruyama
(optionally Heun/Milstein), and the task conditioning ``xi`` (missingness mask for
in-painting, upsampled low-resolution image for super-resolution) is re-supplied to the
networks at every step.

The paper's ImageNet experiments use the deterministic probability flow ODE (Section 3.1:
"for simplicity we will focus on the deterministic probability flow ODE"); this module is
the stochastic counterpart, useful whenever a distribution of samples per conditioning
input is desired (Section 3.4: "the SDE produces a collection of samples whose spread can
be controlled by the diffusion coefficient eps_t").
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:  # pragma: no cover - import guard; module also works standalone
    from ..interpolants.coefficients import Coefficients, get_coefficients

    _COEFFICIENTS_AVAILABLE = True
except Exception:  # pragma: no cover
    Coefficients = None  # type: ignore[assignment]
    get_coefficients = None  # type: ignore[assignment]
    _COEFFICIENTS_AVAILABLE = False

try:  # pragma: no cover
    from ..interpolants.interpolant import Interpolant

    _INTERPOLANT_AVAILABLE = True
except Exception:  # pragma: no cover
    Interpolant = None  # type: ignore[assignment]
    _INTERPOLANT_AVAILABLE = False


__all__ = [
    "sde_drift",
    "make_sde_drift",
    "forward_sde_sample",
    "reverse_sde_sample",
    "sde_sample",
    "solve_sde",
    "euler_maruyama",
    "SDESampler",
    "sde_from_coupling",
    "resolve_epsilon",
    "epsilon_schedule",
    "score_from_g",
    "DEFAULT_EPSILON",
    "DEFAULT_GAMMA_EPS",
    "SDE_METHODS",
]

#: Default diffusion coefficient.  Corollary 3.1 only requires ``eps_t >= 0``.
DEFAULT_EPSILON: float = 0.1

#: Accepted discretization names for the SDE integrator.
SDE_METHODS: Tuple[str, ...] = (
    "euler_maruyama",
    "euler-maruyama",
    "em",
    "euler",
    "heun",
    "milstein",
)

#: Guard used when inverting ``gamma_t``; ``gamma_t^{-1} g_t`` is only meaningful where
#: ``gamma_t != 0`` (Theorem 3.1).
DEFAULT_GAMMA_EPS: float = 1e-8


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _as_tensor_time(t: Union[float, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Return a time tensor broadcastable against ``x`` (batch-shaped if ``t`` is batched)."""
    if torch.is_tensor(t):
        t = t.to(device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            return t
        return t.reshape(t.shape[0], *([1] * (x.ndim - 1)))
    return torch.as_tensor(float(t), device=x.device, dtype=x.dtype)


def resolve_interpolant(
    interpolant: Optional[Any] = None, coefficients: Optional[Any] = None
) -> Optional[Any]:
    """Normalize ``interpolant``/``coefficients`` to an ``Interpolant`` instance.

    Accepts an :class:`Interpolant`, a coefficient container, a preset name, or ``None``
    (defaults to the ``"linear"`` preset).
    """
    for candidate in (interpolant, coefficients):
        if candidate is not None and hasattr(candidate, "alpha_beta_gamma"):
            return candidate
    if not _INTERPOLANT_AVAILABLE:
        return None
    if isinstance(interpolant, str):
        return Interpolant(get_coefficients(interpolant))  # type: ignore[misc]
    if isinstance(coefficients, str):
        return Interpolant(get_coefficients(coefficients))  # type: ignore[misc]
    if coefficients is not None and _COEFFICIENTS_AVAILABLE and isinstance(coefficients, Coefficients):
        return Interpolant(coefficients)  # type: ignore[misc]
    if interpolant is not None:
        return Interpolant(interpolant)  # type: ignore[misc]
    if coefficients is not None:
        return Interpolant(coefficients)  # type: ignore[misc]
    return Interpolant(get_coefficients("linear"))  # type: ignore[misc]


def epsilon_schedule(
    kind: str = "constant", eps_max: float = DEFAULT_EPSILON, eps_min: float = 0.0
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``t -> eps_t`` for one of several simple diffusion schedules.

    ``"constant"``, ``"zero"`` (deterministic/ODE limit), ``"linear"``
    (``eps_max * (1 - t)``), ``"linear_increasing"`` (``eps_max * t``), ``"cosine"``
    (``eps_min + (eps_max - eps_min) * cos(pi t / 2)``).
    """
    kind = str(kind).lower()
    if kind in ("constant", "const", "fixed"):
        return lambda t: torch.full_like(t, float(eps_max))
    if kind in ("zero", "none", "deterministic"):
        return lambda t: torch.zeros_like(t)
    if kind in ("linear", "linear_decreasing", "decreasing"):
        return lambda t: float(eps_max) * (1.0 - t)
    if kind in ("linear_increasing", "increasing"):
        return lambda t: float(eps_max) * t
    if kind in ("cosine", "cos"):
        return lambda t: float(eps_min) + (float(eps_max) - float(eps_min)) * torch.cos(
            math.pi * t / 2.0
        )
    raise ValueError(
        f"Unknown epsilon schedule {kind!r}; expected one of 'constant', 'zero', "
        "'linear', 'linear_increasing', 'cosine'"
    )


def resolve_epsilon(
    epsilon: Union[float, Callable[[torch.Tensor], torch.Tensor], str, None],
    t: torch.Tensor,
    eps_max: Optional[float] = None,
) -> torch.Tensor:
    """Evaluate the diffusion coefficient ``eps_t`` on the time tensor ``t``."""
    if epsilon is None:
        epsilon = DEFAULT_EPSILON
    if isinstance(epsilon, str):
        scale = eps_max if eps_max is not None else DEFAULT_EPSILON
        return epsilon_schedule(epsilon, eps_max=scale)(t)
    if callable(epsilon):
        value = epsilon(t)
        if not torch.is_tensor(value):
            value = torch.full_like(t, float(value))
        return value
    return torch.full_like(t, float(epsilon))


def _safe_inverse_gamma(
    gamma: torch.Tensor, gamma_eps: float = DEFAULT_GAMMA_EPS
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(1/gamma, safe_mask)``; entries with ``|gamma| < eps`` are set to zero."""
    safe = gamma.abs() >= float(gamma_eps)
    denom = torch.where(safe, gamma, torch.ones_like(gamma))
    return torch.where(safe, 1.0 / denom, torch.zeros_like(gamma)), safe


def _broadcast_t(t_tensor: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Reshape a 1-D time tensor ``(B,)`` to a scalar when the batch is homogeneous."""
    if t_tensor.ndim == 1 and t_tensor.numel() > 1 and t_tensor.numel() != n_samples:
        return t_tensor.reshape(-1)
    return t_tensor


def score_from_g(
    g: torch.Tensor,
    t: Union[float, torch.Tensor],
    interpolant: Optional[Any] = None,
    coefficients: Optional[Any] = None,
    gamma_eps: float = DEFAULT_GAMMA_EPS,
) -> torch.Tensor:
    """Convert the learned ``g_t`` into a score via ``nabla log rho_t = -gamma_t^{-1} g_t``.

    Entries where ``|gamma_t| < gamma_eps`` (the interpolation endpoints of the ``linear``
    preset, or structurally-zero ``gamma_t`` presets such as ``"gamma0"``) carry no score
    information -- Theorem 3.1 requires ``gamma_t != 0`` -- and are set to zero.
    """
    interp = resolve_interpolant(interpolant, coefficients)
    if interp is None or not hasattr(interp, "alpha_beta_gamma"):
        raise ValueError(
            "score_from_g requires an Interpolant/Coefficients object (or a preset name) "
            "to evaluate gamma_t"
        )
    t_tensor = t if torch.is_tensor(t) else torch.as_tensor(float(t), device=g.device, dtype=g.dtype)
    if t_tensor.ndim == 0:
        t_tensor = t_tensor.reshape(1)
    _, _, gamma = interp.alpha_beta_gamma(t_tensor)
    if gamma.ndim == 1 and gamma.numel() > 1:
        gamma = gamma.reshape(gamma.shape[0], *([1] * (g.ndim - 1)))
    gamma = gamma.to(device=g.device, dtype=g.dtype)
    inv_gamma, _ = _safe_inverse_gamma(gamma, gamma_eps=gamma_eps)
    return -inv_gamma * g


# ---------------------------------------------------------------------------------------
# network invocation (tolerant to conditioning keyword aliases)
# ---------------------------------------------------------------------------------------
_XI_ALIASES = ("xi", "cond", "conditioning", "context", "mask", "c")
_Y_ALIASES = ("y", "label", "labels", "class_labels", "cls")


def _unwrap(out: Any) -> torch.Tensor:
    """Extract the tensor prediction from a dict/tuple/list model output."""
    if isinstance(out, dict):
        for key in ("velocity", "b_hat", "score", "g_hat", "v", "sample", "x", "out", "output"):
            if key in out:
                return out[key]
        raise KeyError(f"could not find a prediction entry in model output keys {list(out)}")
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _call_net(
    net: nn.Module, x: torch.Tensor, t: torch.Tensor, xi: Any = None, y: Any = None
) -> torch.Tensor:
    """Call ``net(x, t)`` forwarding ``xi``/``y`` under their accepted keyword names."""
    try:
        sig = inspect.signature(net.forward)  # type: ignore[attr-defined]
        names = {p.name for p in sig.parameters.values()}
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        names, accepts_kwargs = set(), True

    kwargs: Dict[str, Any] = {}
    if xi is not None:
        for key in _XI_ALIASES:
            if key in names:
                kwargs[key] = xi
                break
        else:
            if accepts_kwargs:
                kwargs["xi"] = xi
    if y is not None:
        for key in _Y_ALIASES:
            if key in names:
                kwargs[key] = y
                break
        else:
            if accepts_kwargs:
                kwargs["y"] = y
    return _unwrap(net(x, t, **kwargs))


# ---------------------------------------------------------------------------------------
# drift construction
# ---------------------------------------------------------------------------------------
def sde_drift(
    x: torch.Tensor,
    t: Union[float, torch.Tensor],
    velocity_model: Optional[nn.Module] = None,
    score_model: Optional[nn.Module] = None,
    xi: Any = None,
    y: Any = None,
    epsilon: Union[float, Callable, str, None] = DEFAULT_EPSILON,
    direction: str = "forward",
    interpolant: Optional[Any] = None,
    coefficients: Optional[Any] = None,
    mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    velocity_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    score_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    score_kind: str = "g",
    gamma_eps: float = DEFAULT_GAMMA_EPS,
    eps_max: Optional[float] = None,
) -> torch.Tensor:
    """Evaluate the SDE drift at ``(x, t)``.

    ``direction="forward"`` -> ``b_t(x) - eps_t * gamma_t^{-1} g_t(x)``  (Eq. 11)

    ``direction="reverse"`` -> ``b_t(x) + eps_t * gamma_t^{-1} g_t(x)``  (Eq. 13)

    ``score_kind`` selects whether the second network outputs ``"g"`` (``E[z | I_t]``, the
    paper's parameterization) or a ``"score"`` directly.
    """
    t_tensor = _as_tensor_time(t, x)
    direction = str(direction).lower()
    if direction in ("forward", "f", "fwd"):
        sign = -1.0
    elif direction in ("reverse", "r", "backward", "bwd"):
        sign = 1.0
    else:
        raise ValueError(f"unknown SDE direction {direction!r}; expected 'forward' or 'reverse'")

    if velocity_fn is not None:
        velocity = velocity_fn(x, t_tensor)
    elif velocity_model is not None:
        velocity = _call_net(velocity_model, x, t_tensor, xi=xi, y=y)
    else:
        raise ValueError("sde_drift requires a velocity model or a velocity_fn")

    if mask_fn is not None and xi is not None:
        velocity = mask_fn(velocity, xi)

    eps_t = resolve_epsilon(epsilon, _as_tensor_time(t, x), eps_max=eps_max)
    eps_t = eps_t.to(device=x.device, dtype=x.dtype)
    if eps_t.ndim == 1 and eps_t.numel() > 1:
        eps_t = eps_t.reshape(eps_t.shape[0], *([1] * (x.ndim - 1)))

    if (score_model is None and score_fn is None) or bool(torch.all(eps_t == 0)):
        return velocity

    if score_fn is not None:
        g = score_fn(x, t_tensor)
    else:
        g = _call_net(score_model, x, t_tensor, xi=xi, y=y)  # type: ignore[arg-type]

    if str(score_kind).lower() in ("score", "grad_log", "nabla_log"):
        score = g
    else:
        interp = resolve_interpolant(interpolant, coefficients)
        if interp is None or not hasattr(interp, "alpha_beta_gamma"):
            raise ValueError(
                "an interpolant/coefficients object is required to convert g_t into a score"
            )
        _, _, gamma = interp.alpha_beta_gamma(
            t_tensor.reshape(1) if t_tensor.ndim == 0 else t_tensor
        )
        if gamma.ndim == 1 and gamma.numel() > 1:
            gamma = gamma.reshape(gamma.shape[0], *([1] * (g.ndim - 1)))
        gamma = gamma.to(device=g.device, dtype=g.dtype)
        inv_gamma, _ = _safe_inverse_gamma(gamma, gamma_eps=gamma_eps)
        score = -inv_gamma * g

    return velocity + sign * eps_t * score


def make_sde_drift(
    velocity_model: Optional[nn.Module] = None,
    score_model: Optional[nn.Module] = None,
    xi: Any = None,
    y: Any = None,
    epsilon: Union[float, Callable, str, None] = DEFAULT_EPSILON,
    direction: str = "forward",
    interpolant: Optional[Any] = None,
    coefficients: Optional[Any] = None,
    mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    velocity_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    score_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    score_kind: str = "g",
    gamma_eps: float = DEFAULT_GAMMA_EPS,
    eps_max: Optional[float] = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Close over the conditioning and return ``drift(x, t)`` for :func:`solve_sde`."""

    def drift(x: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        return sde_drift(
            x,
            t,
            velocity_model=velocity_model,
            score_model=score_model,
            xi=xi,
            y=y,
            epsilon=epsilon,
            direction=direction,
            interpolant=interpolant,
            coefficients=coefficients,
            mask_fn=mask_fn,
            velocity_fn=velocity_fn,
            score_fn=score_fn,
            score_kind=score_kind,
            gamma_eps=gamma_eps,
            eps_max=eps_max,
        )

    return drift


# ---------------------------------------------------------------------------------------
# integrators
# ---------------------------------------------------------------------------------------
def _noise(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Unit-variance Gaussian noise matching ``x`` (the ``dW`` increment is scaled outside)."""
    if generator is None:
        return torch.randn_like(x)
    try:
        return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    except TypeError:  # pragma: no cover - very old torch
        return torch.randn_like(x)


def euler_maruyama(
    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    steps: int = 100,
    t0: float = 0.0,
    t1: float = 1.0,
    epsilon: Union[float, Callable, str, None] = DEFAULT_EPSILON,
    method: str = "euler_maruyama",
    generator: Optional[torch.Generator] = None,
    return_trajectory: bool = False,
    project: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    time_grid: Optional[Sequence[float]] = None,
    eps_max: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Euler--Maruyama integration of ``dX = drift(X,t) dt + sqrt(2 eps_t) dW``.

    ``t0``/``t1`` may be given in either order; the returned sample corresponds to ``t1``.
    ``project`` is an optional per-step map (e.g. re-imposing observed pixels for
    in-painting) applied to the state before the next step.
    """
    dtype, device = x0.dtype, x0.device
    method = str(method).lower()
    if method not in SDE_METHODS:
        raise ValueError(f"unknown SDE method {method!r}; expected one of {sorted(SDE_METHODS)}")

    if time_grid is not None:
        ts = torch.as_tensor(list(time_grid), device=device, dtype=dtype)
        if ts.ndim != 1 or ts.numel() < 2:
            raise ValueError("time_grid must contain at least two times")
        steps = ts.numel() - 1
    else:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        ts = torch.linspace(float(t0), float(t1), int(steps) + 1, device=device, dtype=dtype)

    x = x0
    trajectory: List[torch.Tensor] = [x0.detach().clone()] if return_trajectory else []

    for n in range(steps):
        t_n, t_next = ts[n], ts[n + 1]
        dt = float((t_next - t_n).item())
        eps_t = resolve_epsilon(epsilon, t_n.reshape(1).expand(x.shape[0]), eps_max=eps_max)
        eps_t = eps_t.to(device=device, dtype=dtype)
        if eps_t.ndim == 1 and eps_t.numel() > 1:
            eps_t = eps_t.reshape(eps_t.shape[0], *([1] * (x.ndim - 1)))
        noise_scale = torch.sqrt(2.0 * eps_t * abs(dt))

        if method == "heun" and bool(torch.all(eps_t == 0)):
            # deterministic limit of Heun: trapezoidal rule
            d0 = drift(x, t_n)
            d1 = drift(x + dt * d0, t_next)
            x = x + 0.5 * dt * (d0 + d1)
        elif method == "heun":
            d0 = drift(x, t_n)
            x_pred = x + dt * d0 + noise_scale * _noise(x, generator)
            d1 = drift(x_pred, t_next)
            x = x + 0.5 * dt * (d0 + d1) + noise_scale * _noise(x, generator)
        else:  # euler_maruyama / euler / milstein (drift-only Milstein update)
            d0 = drift(x, t_n)
            x = x + dt * d0 + noise_scale * _noise(x, generator)

        if project is not None:
            x = project(x, t_next)
        if return_trajectory:
            trajectory.append(x.detach().clone())

    traj = torch.stack(trajectory, dim=0) if return_trajectory else None
    return x, traj


def solve_sde(
    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    steps: int = 100,
    t0: float = 0.0,
    t1: float = 1.0,
    epsilon: Union[float, Callable, str, None] = DEFAULT_EPSILON,
    method: str = "euler_maruyama",
    generator: Optional[torch.Generator] = None,
    return_trajectory: bool = False,
    project: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    eps_max: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Generic SDE integrator returning ``X_{t1}`` (and optionally the trajectory)."""
    x, traj = euler_maruyama(
        drift,
        x0,
        steps=steps,
        t0=t0,
        t1=t1,
        epsilon=epsilon,
        method=method,
        generator=generator,
        return_trajectory=return_trajectory,
        project=project,
        eps_max=eps_max,
    )
    if return_trajectory:
        return x, traj  # type: ignore[return-value]
    return x


# ---------------------------------------------------------------------------------------
# task-level sampling entry points
# ---------------------------------------------------------------------------------------
def _maybe_project(
    project: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    xi: Optional[torch.Tensor],
    x1: Optional[torch.Tensor],
    observed_value: float = 1.0,
) -> Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Build a projection re-imposing the observed (``xi == 1``) pixels of ``x1``."""
    if project is not None:
        return project
    if xi is None or x1 is None:
        return None
    mask = xi
    while mask.ndim < x1.ndim:
        mask = mask.unsqueeze(0)
    observed = (mask == float(observed_value)).to(dtype=x1.dtype)

    def _project(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        return observed * x1 + (1.0 - observed) * x

    return _project


def forward_sde_sample(
    velocity_model: nn.Module,
    x0: torch.Tensor,
    score_model: Optional[nn.Module] = None,
    xi: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    steps: int = 100,
    epsilon: Union[float, Callable, str, None] = DEFAULT_EPSILON,
    method: str = "euler_maruyama",
    interpolant: Optional[Any] = None,
    coefficients: Optional[Any] = None,
    mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    score_kind: str = "g",
    gamma_eps: float = DEFAULT_GAMMA_EPS,
    generator: Optional[torch.Generator] = None,
    return_trajectory: bool = False,
    project: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    x1: Optional[torch.Tensor] = None,
    observed_value: float = 1.0,
    eps_max: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Forward SDE (Eq. 11), ``t: 0 -> 1``: ``X_0 ~ rho_0`` becomes ``X_1 ~ rho_1``."""
    drift = make_sde_drift(
        velocity_model=velocity_model,
        score_model=score_model,
        xi=xi,
        y=y,
        epsilon=epsilon,
        direction="forward",
        interpolant=interpolant,
        coefficients=coefficients,
        mask_fn=mask_fn,
        score_kind=score_kind,
        gamma_eps=gamma_eps,
        eps_max=eps_max,
    )
    project = _maybe_project(project, xi, x1, observed_value=observed_value)
    return solve_sde(
        drift,
        x0,
        steps=steps,
        t0=0.0,
        t1=1.0,
        epsilon=epsilon,
        method=method,
        generator=generator,
        return_trajectory=return_trajectory,
        project=project,
        eps_max=eps_max,
    )


def reverse_sde_sample(
    velocity_model: nn.Module,
    x1: torch.Tensor,
    score_model: Optional[nn.Module] = None,
    xi: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    steps: int = 100,
    epsilon: Union[float, Callable, str, None] = DEFAULT_EPSILON,
    method: str = "euler_maruyama",
    interpolant: Optional[Any] = None,
    coefficients: Optional[Any] = None,
    mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    score_kind: str = "g",
    gamma_eps: float = DEFAULT_GAMMA_EPS,
    generator: Optional[torch.Generator] = None,
    return_trajectory: bool = False,
    project: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    eps_max: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Backward SDE (Eq. 13), ``t: 1 -> 0``: ``X_1 ~ rho_1`` becomes ``X_0 ~ rho_0``.

    The drift is ``b + eps_t gamma_t^{-1} g`` and time is integrated *backwards*, which is
    precisely the statement of Corollary 3.1 for the backward SDE.
    """
    drift = make_sde_drift(
        velocity_model=velocity_model,
        score_model=score_model,
        xi=xi,
        y=y,
        epsilon=epsilon,
        direction="reverse",
        interpolant=interpolant,
        coefficients=coefficients,
        mask_fn=mask_fn,
        score_kind=score_kind,
        gamma_eps=gamma_eps,
        eps_max=eps_max,
    )
    return solve_sde(
        drift,
        x1,
        steps=steps,
        t0=1.0,
        t1=0.0,
        epsilon=epsilon,
        method=method,
        generator=generator,
        return_trajectory=return_trajectory,
        project=project,
        eps_max=eps_max,
    )


def sde_sample(
    velocity_model: nn.Module,
    x: torch.Tensor,
    score_model: Optional[nn.Module] = None,
    direction: str = "forward",
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Dispatch to :func:`forward_sde_sample` or :func:`reverse_sde_sample`."""
    if str(direction).lower() in ("forward", "f", "fwd"):
        return forward_sde_sample(velocity_model, x, score_model=score_model, **kwargs)
    if str(direction).lower() in ("reverse", "backward", "r", "bwd"):
        return reverse_sde_sample(velocity_model, x, score_model=score_model, **kwargs)
    raise ValueError(f"unknown SDE direction {direction!r}; expected 'forward' or 'reverse'")


def sde_from_coupling(
    coupling: Any,
    x1: torch.Tensor,
    velocity_model: nn.Module,
    score_model: Optional[nn.Module] = None,
    direction: str = "forward",
    steps: int = 100,
    epsilon: Union[float, Callable, str, None] = DEFAULT_EPSILON,
    zeta: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_trajectory: bool = False,
    return_base: bool = False,
    method: str = "euler_maruyama",
    eps_max: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Build the coupled base sample ``X_0 = m(x_1) + sigma zeta`` and run the SDE."""
    x0, xi = coupling.build_x0(x1, zeta=zeta, generator=generator, return_xi=True)
    coefficients = kwargs.pop("coefficients", None)
    if coefficients is None:
        coefficients = getattr(coupling, "coefficients_name", None)
    mask_fn = kwargs.pop("mask_fn", None)
    if mask_fn is None and hasattr(coupling, "mask_velocity"):
        mask_fn = coupling.mask_velocity
    out = sde_sample(
        velocity_model,
        x0,
        score_model=score_model,
        direction=direction,
        xi=xi,
        y=y if y is not None else kwargs.pop("y", None),
        steps=steps,
        epsilon=epsilon,
        method=method,
        coefficients=coefficients,
        mask_fn=mask_fn,
        generator=generator,
        return_trajectory=return_trajectory,
        project=kwargs.pop("project", None),
        eps_max=eps_max,
        **kwargs,
    )
    if return_base:
        return out, x0, xi
    return out


class SDESampler(nn.Module):
    """Config-friendly SDE sampler module (Eq. 11 forward / Eq. 13 backward).

    Parameters
    ----------
    method:
        Discretization name, one of :data:`SDE_METHODS`.
    steps:
        Number of integration steps ``N``.
    epsilon:
        Diffusion coefficient ``eps_t``: a float, a callable ``t -> eps_t``, or a schedule
        name (``"constant"``/``"linear"``/``"cosine"``/``"zero"``).
    direction:
        ``"forward"`` (``t: 0 -> 1``, Eq. 11) or ``"reverse"`` (``t: 1 -> 0``, Eq. 13).
    """

    def __init__(
        self,
        method: str = "euler_maruyama",
        steps: int = 100,
        epsilon: Union[float, Callable, str, None] = DEFAULT_EPSILON,
        direction: str = "forward",
        interpolant: Optional[Any] = None,
        coefficients: Optional[Any] = None,
        mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        score_kind: str = "g",
        gamma_eps: float = DEFAULT_GAMMA_EPS,
        return_trajectory: bool = False,
        project_observed: bool = False,
        observed_value: float = 1.0,
        generator: Optional[torch.Generator] = None,
        eps_max: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.method = method
        self.steps = int(steps)
        self.epsilon = epsilon
        self.direction = direction
        self.interpolant = interpolant
        self.coefficients = coefficients
        self.mask_fn = mask_fn
        self.score_kind = score_kind
        self.gamma_eps = float(gamma_eps)
        self.return_trajectory = bool(return_trajectory)
        self.project_observed = bool(project_observed)
        self.observed_value = float(observed_value)
        self.generator = generator
        self.eps_max = eps_max

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        score_model: Optional[nn.Module] = None,
        xi: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
        direction: Optional[str] = None,
        steps: Optional[int] = None,
        method: Optional[str] = None,
        epsilon: Optional[Union[float, Callable, str]] = None,
        return_trajectory: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> Any:
        project = kwargs.pop("project", None)
        if self.project_observed and project is None and xi is not None and x1 is not None:
            project = _maybe_project(None, xi, x1, observed_value=self.observed_value)
        return sde_sample(
            model,
            x,
            score_model=score_model,
            direction=direction if direction is not None else self.direction,
            xi=xi,
            y=y,
            steps=steps if steps is not None else self.steps,
            epsilon=epsilon if epsilon is not None else self.epsilon,
            method=method if method is not None else self.method,
            interpolant=self.interpolant,
            coefficients=self.coefficients,
            mask_fn=self.mask_fn,
            score_kind=self.score_kind,
            gamma_eps=self.gamma_eps,
            generator=generator if generator is not None else self.generator,
            return_trajectory=(
                self.return_trajectory if return_trajectory is None else bool(return_trajectory)
            ),
            project=project,
            eps_max=self.eps_max,
            **kwargs,
        )

    @classmethod
    def from_coupling(cls, coupling: Any, x1: torch.Tensor, model: nn.Module, **kwargs: Any) -> Any:
        """Class-level convenience wrapper around :func:`sde_from_coupling`."""
        return sde_from_coupling(coupling, x1, model, **kwargs)

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"method={self.method}, steps={self.steps}, epsilon={self.epsilon}, "
            f"direction={self.direction}, score_kind={self.score_kind}, "
            f"project_observed={self.project_observed}"
        )


# ---------------------------------------------------------------------------------------
# analytic toy models + self test
# ---------------------------------------------------------------------------------------
class _AnalyticVelocity(nn.Module):
    """Exact ``b_t`` for ``x_0 ~ N(0, I)``, ``x_1 ~ N(0, s1^2 I)``, ``"linear"`` preset."""

    def __init__(self, s1: float = 2.0) -> None:
        super().__init__()
        self.s1 = float(s1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D102
        t = t.reshape(-1, *([1] * (x.ndim - 1))) if t.ndim == 1 else t
        alpha, beta = 1.0 - t, t
        gamma = torch.sqrt(torch.clamp(2.0 * t * (1.0 - t), min=1e-24))
        alpha_dot = -torch.ones_like(t)
        beta_dot = torch.ones_like(t)
        gamma_dot = (1.0 - 2.0 * t) / gamma
        var = alpha**2 + beta**2 * self.s1**2 + gamma**2
        num = beta_dot * beta * self.s1**2 - alpha * alpha_dot + gamma_dot * gamma
        return num / var * x


class _AnalyticG(nn.Module):
    """Exact ``g_t(x) = E[z | I_t = x] = gamma_t / s_t^2 * x`` for the same toy problem."""

    def __init__(self, s1: float = 2.0) -> None:
        super().__init__()
        self.s1 = float(s1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D102
        t = t.reshape(-1, *([1] * (x.ndim - 1))) if t.ndim == 1 else t
        alpha, beta = 1.0 - t, t
        gamma = torch.sqrt(torch.clamp(2.0 * t * (1.0 - t), min=1e-24))
        var = alpha**2 + beta**2 * self.s1**2 + gamma**2
        return gamma / var * x


def _self_test() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)
    d, s1, n = 2, 2.0, 20000

    vel = _AnalyticVelocity(s1)
    gnet = _AnalyticG(s1)

    # forward SDE: rho_0 = N(0, I) -> rho_1 = N(0, s1^2 I)
    x0 = torch.randn(n, d)
    out = forward_sde_sample(vel, x0, score_model=gnet, steps=200, epsilon=0.5, coefficients="linear")
    assert abs(float(out.std(dim=0).mean()) - s1) < 0.15, float(out.std(dim=0).mean())
    assert float(out.mean(dim=0).abs().max()) < 0.15

    # backward SDE: rho_1 = N(0, s1^2 I) -> rho_0 = N(0, I)
    x1 = s1 * torch.randn(n, d)
    back = reverse_sde_sample(vel, x1, score_model=gnet, steps=200, epsilon=0.5, coefficients="linear")
    assert abs(float(back.std(dim=0).mean()) - 1.0) < 0.15, float(back.std(dim=0).mean())

    # eps_t = 0 -> deterministic probability-flow ODE
    ode = forward_sde_sample(vel, x0, score_model=gnet, steps=200, epsilon=0.0, coefficients="linear")
    assert ode.shape == x0.shape

    # trajectory + drift plumbing
    _, traj = forward_sde_sample(
        vel, x0[:8], score_model=gnet, steps=10, epsilon=0.2, coefficients="linear",
        return_trajectory=True,
    )
    assert traj is not None and traj.shape == (11, 8, d)

    drift = make_sde_drift(velocity_model=vel, score_model=gnet, epsilon=0.0, coefficients="linear")
    got = drift(x0[:4], torch.full((4,), 0.5))
    ref = vel(x0[:4], torch.full((4,), 0.5))
    assert torch.allclose(got, ref, atol=1e-6)

    # score convention: score = -gamma_t^{-1} g
    t = torch.full((4,), 0.4)
    g = gnet(x0[:4], t)
    sc = score_from_g(g, t, coefficients="linear")
    _, _, gamma = Interpolant(get_coefficients("linear")).alpha_beta_gamma(t)
    assert torch.allclose(sc, -g / gamma.reshape(-1, 1), atol=1e-5)
    assert torch.allclose(score_from_g(g, t, coefficients="gamma0"), torch.zeros_like(g))

    # module wrapper
    sampler = SDESampler(steps=5, epsilon=0.1, coefficients="linear")
    y_out = sampler(vel, x0[:5], score_model=gnet)
    assert y_out.shape == x0[:5].shape

    # coupled-base plumbing with the in-painting coupling (gamma_t = 0 -> ODE limit)
    from ..couplings.inpainting import InpaintingCoupling

    coupling = InpaintingCoupling(sigma=1.0)
    x1_img = torch.randn(2, 3, 32, 32)
    base, xi = coupling.build_x0(x1_img)
    masked = coupling.mask_velocity(torch.ones_like(base), xi)
    assert torch.allclose(masked * (1 - xi.expand_as(masked)), torch.zeros_like(masked))

    print("si.samplers.sde self-test passed")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
