"""Reverse-time sampling and density evaluation for NPSE / SNPSE / NLSE.

This module implements the sampling machinery of Section 2.2 of the paper:

* the reverse-time SDE (paper eq. 2, listed as eq. (3) in the plan)::

      d theta_bar_t = [ -f(theta_bar_t, T-t) + g^2(T-t) grad log p_{T-t}(theta_bar_t | x) ] dt
                      + g(T-t) dw_t

* the probability-flow ODE (paper eq. 3, "eq. 4" in the plan)::

      d theta_t / dt = f(theta_t, t) - 1/2 g^2(t) grad log p_t(theta_t | x)

* the instantaneous change-of-variables formula (paper eq. 4, "eq. 5" in the plan)::

      d log p_t(theta_t | x) / dt = -Tr[ grad_theta ( f(theta_t, t)
                                                  - 1/2 g^2(t) grad log p_t(theta_t | x) ) ]

In practice ``grad log p_t(theta_t | x)`` is replaced by the trained score network
``s_psi(theta_t, x, t)`` (step (iii) of Section 2.2).

Following Appendix E.3.3 the ODE is solved with an off-the-shelf RK45 solver; a
self-contained Dormand-Prince 5(4) implementation is provided so that the module
does not depend on ``torchdiffeq`` (which is used when available/selected).

Conventions
-----------
* Time runs over ``t in [0, T]`` with ``T = 1`` (``sdes.T_FINAL``).
* Integrating *backwards* in forward-time ``t`` is performed with the reversed
  variable ``tau = T - t`` (``tau`` goes from 0 to ``T``), which is how the
  reverse process / reverse ODE is stated in the paper.
* The density is evaluated by integrating the *augmented* state
  ``(theta_tau, log p_{tau})``; along the reverse pass this directly returns both
  the posterior samples and their log densities (this is what HPR_eps needs).
* ``theta`` and ``x`` can optionally be standardised (Appendix E.3.3); pass
  ``theta_shift``/``theta_scale`` and ``x_shift``/``x_scale`` and the module
  takes care of the change of variables for the log density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Robust import of the SDE definitions (the module is importable both as
# ``snpse.sampler`` and as a top-level ``sampler``).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import bookkeeping
    from .sdes import SDE, T_FINAL
except Exception:  # pragma: no cover
    try:
        from snpse.sdes import SDE, T_FINAL  # type: ignore
    except Exception:
        try:
            from sdes import SDE, T_FINAL  # type: ignore
        except Exception:
            SDE = object  # type: ignore
            T_FINAL = 1.0


__all__ = [
    "SamplerConfig",
    "ProbabilityFlowSampler",
    "ProbabilityFlowODESolver",
    "sample_reference",
    "prior_samples",
    "probability_flow_drift",
    "reverse_sde_drift",
    "divergence",
    "integrate",
    "dopri5",
    "sample_posterior",
    "posterior_samples",
    "log_prob",
    "posterior_log_prob",
    "estimate_log_prob",
    "reverse_sde_sample",
    "reference_log_prob",
]


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class SamplerConfig:
    """Numerical settings for the reverse-time samplers.

    Defaults follow Appendix E.3.3 ("we use an off-the-shelf solver (RK45)") and
    the reproduction plan (RK45 with ``atol = rtol = 1e-5``).
    """

    method: str = "rk45"                       # rk45 | euler | heun | rk4
    atol: float = 1e-5
    rtol: float = 1e-5
    n_steps: int = 1000                        # fixed-step methods only
    first_step: Optional[float] = None
    max_steps: int = 100000
    t_max: float = float(T_FINAL)              # forward-time upper limit (T)
    t_min: float = 0.0                         # forward-time lower limit
    trace_estimator: str = "auto"              # auto | exact | hutchinson
    hutchinson_probes: int = 1
    chunk_size: int = 512                      # exact-divergence chunking
    seed: Optional[int] = None
    solver: Optional[str] = None               # alias for ``method``

    def __post_init__(self) -> None:
        if self.solver is not None:
            self.method = self.solver

    @property
    def t_span(self) -> float:
        return float(self.t_max - self.t_min)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "atol": self.atol,
            "rtol": self.rtol,
            "n_steps": self.n_steps,
            "first_step": self.first_step,
            "max_steps": self.max_steps,
            "t_max": self.t_max,
            "t_min": self.t_min,
            "trace_estimator": self.trace_estimator,
            "hutchinson_probes": self.hutchinson_probes,
        }


def _as_config(config: Optional[SamplerConfig]) -> SamplerConfig:
    if config is None:
        return SamplerConfig()
    if isinstance(config, SamplerConfig):
        return config
    if isinstance(config, dict):
        return SamplerConfig(**config)
    raise TypeError(f"Unsupported sampler config type: {type(config)}")


# ===========================================================================
# Small helpers
# ===========================================================================
def _apply_score(score_fn: Callable, theta: torch.Tensor, x, t) -> torch.Tensor:
    """Evaluate a score network that may return a bare tensor or a tuple."""
    out = score_fn(theta, x, t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _col(value, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reshape a scalar / (n,) time-dependent coefficient to broadcast with ``ref``."""
    if not torch.is_tensor(value):
        return value
    if value.dim() == 0:
        return value
    if ref is not None and ref.dim() >= 1 and value.dim() == 1:
        if ref.dim() >= 2 and value.shape[0] == ref.shape[0] and value.shape[0] != ref.shape[-1]:
            return value.unsqueeze(-1)
    return value


def _as_time(t: float, ref: torch.Tensor):
    """Convert a python float time into a 0-dim tensor matching ``ref``."""
    if torch.is_tensor(t):
        return t.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(float(t), device=ref.device, dtype=ref.dtype)


def _infer_theta_dim(score_fn) -> Optional[int]:
    for attr in ("theta_dim", "dim", "d"):
        value = getattr(score_fn, attr, None)
        if isinstance(value, int):
            return value
    energy = getattr(score_fn, "energy_net", None)
    if energy is not None:
        value = getattr(energy, "theta_dim", None)
        if isinstance(value, int):
            return value
    return None


def _flatten_theta(theta: torch.Tensor) -> torch.Tensor:
    if theta.dim() == 1:
        return theta.unsqueeze(0)
    return theta


def _standardise(value: torch.Tensor, shift, scale) -> torch.Tensor:
    if shift is not None:
        value = value - shift
    if scale is not None:
        value = value / scale
    return value


def _unstandardise(value: torch.Tensor, shift, scale) -> torch.Tensor:
    if scale is not None:
        value = value * scale
    if shift is not None:
        value = value + shift
    return value


def _log_abs_det_scale(scale, dim: int, ref: torch.Tensor) -> Optional[torch.Tensor]:
    """``sum_i log scale_i`` for an element-wise (diagonal) rescaling of theta."""
    if scale is None:
        return None
    scale = torch.as_tensor(scale, device=ref.device, dtype=ref.dtype)
    if scale.dim() == 0:
        return torch.log(scale.abs()) * dim
    return torch.log(scale.abs()).sum()


# ===========================================================================
# Drift coefficients (paper eq. 2 and eq. 3, with s_psi in place of the score)
# ===========================================================================
def probability_flow_drift(sde: "SDE", theta: torch.Tensor, score: torch.Tensor, t) -> torch.Tensor:
    """``f(theta, t) - 1/2 g^2(t) score``  (paper eq. 3)."""
    drift = sde.drift(theta, t)
    g2 = sde.diffusion_sq(t)
    return drift - 0.5 * _col(g2, theta) * score


def reverse_sde_drift(sde: "SDE", theta: torch.Tensor, score: torch.Tensor, t) -> torch.Tensor:
    """``-f(theta, t) + g^2(t) score``  (bracket of paper eq. 2)."""
    drift = sde.drift(theta, t)
    g2 = sde.diffusion_sq(t)
    return -drift + _col(g2, theta) * score


def _pf_drift_from_score_fn(sde, score_fn, theta, x, t) -> torch.Tensor:
    score = _apply_score(score_fn, theta, x, t)
    return probability_flow_drift(sde, theta, score, t)


# ===========================================================================
# Divergence (trace of the Jacobian) of the ODE drift
# ===========================================================================
def divergence(
    vector_fn: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    method: str = "auto",
    n_probes: int = 1,
    generator: Optional[torch.Generator] = None,
    chunk_size: int = 512,
    exact_threshold: int = 32,
) -> torch.Tensor:
    """Estimate ``Tr[grad_theta vector_fn(theta)]`` elementwise over the batch.

    ``vector_fn`` maps ``(n, d) -> (n, d)``.

    * ``"exact"``: exact trace via ``d`` backward passes (autograd).
    * ``"hutchinson"``: unbiased Hutchinson / Skilling estimate using Rademacher
      probes, ``(eps^T J eps)`` — one backward pass per probe.
    * ``"auto"``: exact when ``d <= exact_threshold``, Hutchinson otherwise.
    """
    theta = _flatten_theta(theta)
    d = theta.shape[-1]
    if method == "auto":
        method = "exact" if d <= exact_threshold else "hutchinson"

    # ---- exact ---------------------------------------------------------
    if method == "exact":
        outs = []
        for start in range(0, theta.shape[0], max(1, chunk_size)):
            block = theta[start : start + max(1, chunk_size)]
            th = block.detach().clone().requires_grad_(True)
            out = vector_fn(th)
            if isinstance(out, (tuple, list)):
                out = out[0]
            div = torch.zeros(th.shape[0], device=th.device, dtype=th.dtype)
            for i in range(d):
                grad = torch.autograd.grad(
                    out[:, i].sum(),
                    th,
                    retain_graph=(i < d - 1),
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad is not None:
                    div = div + grad[:, i]
            outs.append(div.detach())
        return torch.cat(outs, dim=0)

    # ---- Hutchinson / Skilling ----------------------------------------
    if method not in ("hutchinson", "skilling", "hutchinson_trace"):
        raise ValueError(f"Unknown divergence estimator: {method!r}")

    total = None
    for _ in range(max(1, n_probes)):
        th = theta.detach().clone().requires_grad_(True)
        eps = torch.randint(
            0, 2, th.shape, device=th.device, dtype=th.dtype, generator=generator
        ) * 2.0 - 1.0
        out = vector_fn(th)
        if isinstance(out, (tuple, list)):
            out = out[0]
        grad, = torch.autograd.grad((out * eps).sum(), th, create_graph=False, allow_unused=True)
        est = (grad * eps).sum(dim=-1) if grad is not None else torch.zeros_like(theta[:, 0])
        total = est.detach() if total is None else total + est.detach()
    return total / float(max(1, n_probes))


# ===========================================================================
# ODE integrators
# ===========================================================================
# Dormand-Prince 5(4) tableau
_DP_C = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]
_DP_A = [
    [1.0 / 5.0],
    [3.0 / 40.0, 9.0 / 40.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
]
_DP_B5 = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0]
_DP_B4 = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
]


def _dopri5_step(f: Callable[[float, torch.Tensor], torch.Tensor], t, y, h):
    """One Dormand-Prince 5(4) step; returns ``(y_5th_order, y_error)``."""
    ks = []
    for i in range(6):
        yi = y
        if i > 0:
            for j, a in enumerate(_DP_A[i - 1]):
                if a != 0.0:
                    yi = yi + h * a * ks[j]
        ks.append(f(t + h * _DP_C[i], yi))
    y_new = y
    for b, k in zip(_DP_B5, ks):
        if b != 0.0:
            y_new = y_new + h * b * k
    k7 = f(t + h, y_new)
    ks.append(k7)
    y_err = None
    for b5, b4, k in zip(_DP_B5, _DP_B4, ks):
        c = (b5 - b4) * h
        if c != 0.0:
            y_err = c * k if y_err is None else y_err + c * k
    if y_err is None:
        y_err = torch.zeros_like(y_new)
    return y_new, y_err


def dopri5(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    t1: float,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    first_step: Optional[float] = None,
    max_steps: int = 100000,
) -> torch.Tensor:
    """Adaptive Dormand-Prince RK45 integration of ``dy/dt = f(t, y)``.

    Works for both forward (``t1 > t0``) and backward (``t1 < t0``) integration.
    The step size is shared across the batch (max-norm error control).
    """
    span = float(t1) - float(t0)
    if span == 0.0:
        return y0
    direction = 1.0 if span > 0 else -1.0
    total = abs(span)

    y = y0
    t = float(t0)
    h = float(first_step) if first_step else total / 100.0
    h = direction * min(abs(h), total)
    if h == 0.0:
        h = direction * total / 100.0

    n_steps = 0
    while abs(t1 - t) > 1e-12 * max(1.0, total) and n_steps < max_steps:
        if direction * (t + h - t1) > 0.0:
            h = t1 - t
        y_new, y_err = _dopri5_step(f, t, y, h)
        scale = atol + rtol * torch.maximum(y.abs(), y_new.abs())
        err = torch.sqrt(torch.mean((y_err / scale) ** 2)).item()

        if err <= 1.0:
            t = t + h
            y = y_new
            n_steps += 1
            if abs(h) <= 1e-14 * max(1.0, total):
                break

        if err == 0.0:
            factor = 5.0
        else:
            factor = 0.9 * err ** (-0.2)
        factor = min(5.0, max(0.2, factor))
        h = h * factor
        if abs(h) > total:
            h = direction * total
        if abs(h) < 1e-14 * max(1.0, total):
            break

    return y


def _fixed_step_integrate(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    t1: float,
    n_steps: int,
    method: str,
) -> torch.Tensor:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive for fixed-step integrators")
    h = (float(t1) - float(t0)) / float(n_steps)
    t = float(t0)
    y = y0
    for _ in range(n_steps):
        if method == "euler":
            y = y + h * f(t, y)
        elif method in ("heun", "rk2"):
            k1 = f(t, y)
            k2 = f(t + h, y + h * k1)
            y = y + 0.5 * h * (k1 + k2)
        elif method == "rk4":
            k1 = f(t, y)
            k2 = f(t + h / 2.0, y + h / 2.0 * k1)
            k3 = f(t + h / 2.0, y + h / 2.0 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:  # pragma: no cover
            raise ValueError(f"Unknown fixed-step method: {method!r}")
        t = t + h
    return y


def integrate(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    t1: float,
    config: Optional[SamplerConfig] = None,
) -> torch.Tensor:
    """Integrate ``dy/dt = f(t, y)`` from ``t0`` to ``t1`` using ``config``."""
    config = _as_config(config)
    method = str(config.method).lower()
    if method in ("rk45", "dopri5", "adaptive", "tsit5"):
        return dopri5(
            f,
            y0,
            t0,
            t1,
            atol=config.atol,
            rtol=config.rtol,
            first_step=config.first_step,
            max_steps=config.max_steps,
        )
    if method in ("euler", "heun", "rk2", "rk4"):
        return _fixed_step_integrate(f, y0, t0, t1, config.n_steps, method)
    raise ValueError(
        f"Unknown ODE solver {config.method!r}; use rk45/euler/heun/rk4."
    )


# ===========================================================================
# Reference (prior-at-T) distribution
# ===========================================================================
def reference_std(sde: "SDE", t: Optional[float] = None, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Marginal standard deviation of the forward process at time ``t`` (default ``T``)."""
    t = float(T_FINAL) if t is None else float(t)
    if ref is None:
        ref = torch.zeros(1)
    std = sde.marginal_std(_as_time(t, ref))
    std = torch.as_tensor(std, device=ref.device, dtype=ref.dtype)
    return std


def sample_reference(
    sde: "SDE",
    num_samples: int,
    dim: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw ``theta_T ~ pi``, i.e. the (approximately standard) reference sample."""
    if device is None:
        device = torch.device("cpu")
    std = reference_std(sde, T_FINAL, ref=torch.zeros(1, device=device, dtype=dtype))
    noise = torch.randn(
        num_samples, dim, generator=generator, device=device, dtype=dtype
    )
    std = torch.as_tensor(std, device=device, dtype=dtype)
    return noise * std


#: alias
prior_samples = sample_reference


def reference_log_prob(sde: "SDE", theta: torch.Tensor, t: Optional[float] = None) -> torch.Tensor:
    """``log pi(theta)`` with ``pi = N(0, sigma_T^2 I)`` (approx. the prior at time T)."""
    theta = _flatten_theta(theta)
    d = theta.shape[-1]
    std = reference_std(sde, T_FINAL if t is None else t, ref=theta)
    std = torch.as_tensor(std, device=theta.device, dtype=theta.dtype)
    return -0.5 * (
        (theta ** 2).sum(dim=-1) / std ** 2
        + d * math.log(2.0 * math.pi)
        + 2.0 * d * torch.log(std)
    )


# ===========================================================================
# Reverse-time probability-flow ODE: sampling + change of variables
# ===========================================================================
def _reverse_field(
    sde: "SDE",
    score_fn: Callable,
    x: torch.Tensor,
    theta_dim: int,
    t_max: float,
    with_log_prob: bool,
    config: SamplerConfig,
    generator: Optional[torch.Generator],
):
    """RHS of the reverse-time (tau = t_max - t) augmented probability-flow ODE.

    State layout: ``[theta (d) , log p (1)]`` (the second block is omitted when
    ``with_log_prob=False``).  With ``y(tau) = theta_{t_max - tau}``:

        dy/dtau = -[ f(y, t) - 1/2 g^2(t) s_psi(y, x, t) ]      (paper eq. 3)
        dlogp/dtau = +Tr[ grad_y ( f - 1/2 g^2 s_psi ) ]        (paper eq. 4)
    """

    def rhs(tau, y):
        t = _as_time(float(t_max) - float(tau), y)
        theta = y[..., :theta_dim]
        drift = _pf_drift_from_score_fn(sde, score_fn, theta, x, t)
        dtheta = (-drift).detach()
        if not with_log_prob:
            return dtheta

        def vector_fn(th):
            return _pf_drift_from_score_fn(sde, score_fn, th, x, t)

        div = divergence(
            vector_fn,
            theta,
            method=config.trace_estimator,
            n_probes=config.hutchinson_probes,
            generator=generator,
            chunk_size=config.chunk_size,
        )
        return torch.cat([dtheta, div.reshape(-1, 1).to(dtheta)], dim=-1)

    return rhs


def sample_posterior(
    sde: "SDE",
    score_fn: Callable,
    x_obs: torch.Tensor,
    num_samples: int,
    config: Optional[SamplerConfig] = None,
    with_log_prob: bool = False,
    theta_dim: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    theta_shift=None,
    theta_scale=None,
    x_shift=None,
    x_scale=None,
    theta_init: Optional[torch.Tensor] = None,
):
    """Sample the approximate posterior by integrating the probability-flow ODE.

    Parameters
    ----------
    x_obs : observation, shape ``(p,)`` or ``(1, p)``.
    num_samples : number of posterior samples.
    with_log_prob : if ``True``, also return the log density ``log p_psi(theta | x)``
        evaluated with the instantaneous change-of-variables formula (eq. 4); this
        is the quantity used by ``HPR_eps`` and by the truncated proposal.
    theta_shift/theta_scale/x_shift/x_scale : optional standardisation constants
        (Appendix E.3.3).  Samples are returned in the *original* theta space and
        the log density accounts for the diagonal Jacobian correction.

    Returns
    -------
    ``theta`` of shape ``(num_samples, d)`` or ``(theta, log_prob)``.
    """
    config = _as_config(config)
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = getattr(score_fn, "device", None) or torch.device("cpu")
    device = torch.device(device)

    d = theta_dim if theta_dim is not None else _infer_theta_dim(score_fn)
    if d is None:
        raise ValueError("Could not infer theta_dim; pass it explicitly.")

    x_obs = torch.as_tensor(x_obs, device=device, dtype=dtype)
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)
    x_net = _standardise(x_obs, x_shift, x_scale)

    if generator is None and config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(config.seed))

    if theta_init is None:
        theta_t = sample_reference(sde, num_samples, d, generator=generator, device=device, dtype=dtype)
    else:
        theta_t = torch.as_tensor(theta_init, device=device, dtype=dtype)
        if theta_t.dim() == 1:
            theta_t = theta_t.unsqueeze(0)

    t_max = float(config.t_max)
    tau_end = float(config.t_max) - float(config.t_min)

    if with_log_prob:
        logp0 = reference_log_prob(sde, theta_t, t=t_max).reshape(-1, 1).to(dtype)
        state = torch.cat([theta_t, logp0], dim=-1)
    else:
        state = theta_t

    rhs = _reverse_field(sde, score_fn, x_net, d, t_max, with_log_prob, config, generator)
    with torch.no_grad():
        out = integrate(rhs, state, 0.0, tau_end, config)

    theta_net = out[..., :d].detach()
    theta = _unstandardise(theta_net, theta_shift, theta_scale)

    if not with_log_prob:
        return theta

    logp = out[..., d].detach()
    correction = _log_abs_det_scale(theta_scale, d, theta)
    if correction is not None:
        logp = logp - correction
    return theta, logp


#: alias
posterior_samples = sample_posterior


def estimate_log_prob(
    sde: "SDE",
    score_fn: Callable,
    x_obs: torch.Tensor,
    theta: torch.Tensor,
    config: Optional[SamplerConfig] = None,
    theta_dim: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    theta_shift=None,
    theta_scale=None,
    x_shift=None,
    x_scale=None,
) -> torch.Tensor:
    """Log density ``log p_psi(theta | x_obs)`` for *given* ``theta`` (eq. 4).

    The probability-flow ODE is integrated *forward* in time ``t`` from
    ``t_min`` to ``t_max``, accumulating ``d log p_t / dt = -Tr[grad(drift)]``::

        log p_0(theta_0 | x) = log pi(theta_T) + int_0^T Tr[ grad drift ] dt
    """
    config = _as_config(config)
    if dtype is None:
        dtype = theta.dtype if torch.is_tensor(theta) else torch.float32
    if device is None:
        device = theta.device if torch.is_tensor(theta) else torch.device("cpu")
    device = torch.device(device)

    d = theta_dim if theta_dim is not None else _infer_theta_dim(score_fn)
    if d is None:
        raise ValueError("Could not infer theta_dim; pass it explicitly.")

    theta = _flatten_theta(torch.as_tensor(theta, device=device, dtype=dtype))
    x_obs = torch.as_tensor(x_obs, device=device, dtype=dtype)
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)
    x_net = _standardise(x_obs, x_shift, x_scale)
    theta_net = _standardise(theta, theta_shift, theta_scale)

    t_start = float(config.t_min)
    t_end = float(config.t_max)

    def rhs(t, y):
        th = y[..., :d]
        t_tensor = _as_time(t, y)
        drift = _pf_drift_from_score_fn(sde, score_fn, th, x_net, t_tensor)

        def vector_fn(z):
            return _pf_drift_from_score_fn(sde, score_fn, z, x_net, t_tensor)

        div = divergence(
            vector_fn,
            th,
            method=config.trace_estimator,
            n_probes=config.hutchinson_probes,
            generator=generator,
            chunk_size=config.chunk_size,
        )
        return torch.cat([drift.detach(), div.reshape(-1, 1).to(drift)], dim=-1)

    state = torch.cat([theta_net, torch.zeros(theta_net.shape[0], 1, device=device, dtype=dtype)], dim=-1)
    with torch.no_grad():
        out = integrate(rhs, state, t_start, t_end, config)

    logp_net = out[..., d].detach() + reference_log_prob(sde, out[..., :d].detach(), t=t_end)
    correction = _log_abs_det_scale(theta_scale, d, theta)
    if correction is not None:
        logp_net = logp_net - correction
    return logp_net


#: alias
log_prob = estimate_log_prob
posterior_log_prob = estimate_log_prob


# ===========================================================================
# Reverse-time SDE sampling (Euler-Maruyama, paper eq. 2)
# ===========================================================================
def reverse_sde_sample(
    sde: "SDE",
    score_fn: Callable,
    x_obs: torch.Tensor,
    num_samples: int,
    config: Optional[SamplerConfig] = None,
    theta_dim: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    theta_shift=None,
    theta_scale=None,
    x_shift=None,
    x_scale=None,
    n_steps: Optional[int] = None,
) -> torch.Tensor:
    """Sample by simulating the reverse-time SDE (paper eq. 2) with Euler-Maruyama.

    ``d theta_bar_tau = [ -f(theta_bar, T-tau) + g^2(T-tau) s_psi(...) ] dtau
                         + g(T-tau) dw_tau``
    """
    config = _as_config(config)
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = getattr(score_fn, "device", None) or torch.device("cpu")
    device = torch.device(device)

    d = theta_dim if theta_dim is not None else _infer_theta_dim(score_fn)
    if d is None:
        raise ValueError("Could not infer theta_dim; pass it explicitly.")

    x_obs = torch.as_tensor(x_obs, device=device, dtype=dtype)
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)
    x_net = _standardise(x_obs, x_shift, x_scale)

    if generator is None and config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(config.seed))

    y = sample_reference(sde, num_samples, d, generator=generator, device=device, dtype=dtype)
    n_steps = int(n_steps if n_steps is not None else config.n_steps)
    t_max = float(config.t_max)
    dt = (t_max - float(config.t_min)) / float(n_steps)

    with torch.no_grad():
        for k in range(n_steps):
            t_fwd = t_max - k * dt
            t_tensor = _as_time(t_fwd, y)
            score = _apply_score(score_fn, y, x_net, t_tensor)
            bracket = reverse_sde_drift(sde, y, score, t_tensor)
            g2 = sde.diffusion_sq(t_tensor)
            g = torch.sqrt(torch.clamp(torch.as_tensor(g2, device=device, dtype=dtype), min=0.0))
            noise = torch.randn(y.shape, generator=generator, device=device, dtype=dtype)
            y = y + bracket * dt + _col(g, y) * math.sqrt(dt) * noise

    return _unstandardise(y, theta_shift, theta_scale)


# ===========================================================================
# Convenience wrapper
# ===========================================================================
class ProbabilityFlowSampler:
    """Thin convenience wrapper bundling an SDE, a score network and a config."""

    def __init__(
        self,
        sde: "SDE",
        score_fn: Callable,
        config: Optional[SamplerConfig] = None,
        theta_dim: Optional[int] = None,
        theta_shift=None,
        theta_scale=None,
        x_shift=None,
        x_scale=None,
    ) -> None:
        self.sde = sde
        self.score_fn = score_fn
        self.config = _as_config(config)
        self.theta_dim = theta_dim if theta_dim is not None else _infer_theta_dim(score_fn)
        self.theta_shift = theta_shift
        self.theta_scale = theta_scale
        self.x_shift = x_shift
        self.x_scale = x_scale

    # -- sampling -------------------------------------------------------
    def sample(self, x_obs, num_samples: int, with_log_prob: bool = False, **kwargs):
        return sample_posterior(
            self.sde,
            self.score_fn,
            x_obs,
            num_samples,
            config=self.config,
            with_log_prob=with_log_prob,
            theta_dim=self.theta_dim,
            theta_shift=self.theta_shift,
            theta_scale=self.theta_scale,
            x_shift=self.x_shift,
            x_scale=self.x_scale,
            **kwargs,
        )

    # -- density of given theta -----------------------------------------
    def log_prob(self, x_obs, theta, **kwargs):
        return estimate_log_prob(
            self.sde,
            self.score_fn,
            x_obs,
            theta,
            config=self.config,
            theta_dim=self.theta_dim,
            theta_shift=self.theta_shift,
            theta_scale=self.theta_scale,
            x_shift=self.x_shift,
            x_scale=self.x_scale,
            **kwargs,
        )

    # -- reverse-time SDE ----------------------------------------------
    def sample_reverse_sde(self, x_obs, num_samples: int, n_steps: Optional[int] = None, **kwargs):
        return reverse_sde_sample(
            self.sde,
            self.score_fn,
            x_obs,
            num_samples,
            config=self.config,
            theta_dim=self.theta_dim,
            n_steps=n_steps,
            theta_shift=self.theta_shift,
            theta_scale=self.theta_scale,
            x_shift=self.x_shift,
            x_scale=self.x_scale,
            **kwargs,
        )


#: alias kept for backwards compatibility with the plan's naming
ProbabilityFlowODESolver = ProbabilityFlowSampler


# ===========================================================================
# Self-test: analytic Gaussian posterior recovered by the ODE + change of vars
# ===========================================================================
def _selftest(dim: int = 2, sde_name: str = "ve", n: int = 4000) -> None:  # pragma: no cover
    """Validate the sampler / change-of-variables against an analytic Gaussian.

    We take ``p_0 = N(mu0, Sigma0)`` and the *exact* time-dependent score of the
    forward perturbation, ``p_t = N(alpha(t) mu0, alpha(t)^2 Sigma0 + sigma_t^2 I)``.
    The ODE then transports the reference Gaussian onto ``p_0`` exactly, and the
    change-of-variables formula must return the analytic ``log N(theta; mu0, Sigma0)``.
    """
    try:
        from .sdes import get_sde
    except Exception:
        try:
            from snpse.sdes import get_sde  # type: ignore
        except Exception:
            from sdes import get_sde  # type: ignore

    torch.manual_seed(0)
    sde = get_sde(sde_name, dim=dim, data=torch.zeros(1, dim))
    sde = get_sde(sde_name, dim=dim) if sde is None else sde

    mu0 = torch.linspace(-0.5, 0.5, dim)
    A = torch.randn(dim, dim) * 0.4 + torch.eye(dim)
    Sigma0 = A @ A.T + 0.5 * torch.eye(dim)
    Sigma0_inv = torch.linalg.inv(Sigma0)

    def alpha(t):
        m = sde.marginal_mean(torch.ones(1, dim), _as_time(t, mu0))
        return torch.as_tensor(m, dtype=torch.float32).reshape(-1)[0]

    def analytic_score(theta, x, t):
        a = alpha(t)
        s = torch.as_tensor(sde.marginal_std(_as_time(t, theta)), dtype=theta.dtype)
        cov = (a ** 2) * Sigma0 + (s ** 2) * torch.eye(dim)
        cov_inv = torch.linalg.inv(cov)
        return -(theta - a * mu0) @ cov_inv.T

    x_obs = torch.zeros(1, 1)
    cfg = SamplerConfig(atol=1e-6, rtol=1e-6, trace_estimator="exact")
    theta, logp = sample_posterior(
        sde, analytic_score, x_obs, n, config=cfg, with_log_prob=True, theta_dim=dim
    )

    emp_mean = theta.mean(0)
    emp_cov = torch.cov(theta.T)
    print(f"[{sde_name}] empirical mean : {emp_mean.numpy()}")
    print(f"[{sde_name}] target    mean : {mu0.numpy()}")
    print(f"[{sde_name}] max |mean err| : {float((emp_mean - mu0).abs().max()):.4f}")
    print(f"[{sde_name}] max |cov  err| : {float((emp_cov - Sigma0).abs().max()):.4f}")

    theta_given = theta[:64]
    est = estimate_log_prob(sde, analytic_score, x_obs, theta_given, config=cfg, theta_dim=dim)
    d = theta_given - mu0
    truth = -0.5 * (torch.einsum("ij,jk,ik->i", d, Sigma0_inv, d))
    truth = truth - 0.5 * (dim * math.log(2 * math.pi) + torch.logdet(Sigma0))
    err_logp = float((est - truth).abs().max())
    print(f"[{sde_name}] max |log p err| : {err_logp:.4f}")
    assert float((emp_mean - mu0).abs().max()) < 0.1
    assert err_logp < 0.2
    print("selftest OK")


if __name__ == "__main__":  # pragma: no cover
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "ve"
    _selftest(sde_name=name)
