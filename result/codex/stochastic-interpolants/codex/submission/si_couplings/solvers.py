"""ODE / SDE integrators for the generative models of Corollary 3.1.

Corollary 3.1 gives four ways of generating samples with a coupled
interpolant:

* the probability-flow ODE          X_dot_t = b_t(X_t, xi)                    (eq. 8)
* the forward SDE
      dX_t = b_t(X_t, xi) dt - eps_t gamma_t^{-1} g_t(X_t, xi) dt + sqrt(2 eps_t) dW_t   (eq. 11)
* the backward SDE
      dX_t = b_t(X_t, xi) dt + eps_t gamma_t^{-1} g_t(X_t, xi) dt + sqrt(2 eps_t) dW_t   (eq. 13)

The paper integrates the ODE with the Dopri solver from ``torchdiffeq``
(Appendix B); we provide a self-contained Dormand-Prince (dopri5) adaptive
solver with the same interface so that the repository has no hard dependency
on ``torchdiffeq``, and it falls back on it if the library is installed and
requested.  Algorithm 2 of the paper (forward Euler with N steps) is
implemented as ``method="euler"``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor], Tensor]


# ---------------------------------------------------------------------------
# Probability-flow ODE
# ---------------------------------------------------------------------------
def euler_integrate(
    velocity_fn: VelocityFn,
    x: Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    steps: int = 250,
    **kwargs,
) -> Tensor:
    """Forward Euler integration (Algorithm 2), ``steps`` uniform steps."""
    dt = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        x = x + dt * velocity_fn(x, torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype))
        t = t + dt
    return x


def heun_integrate(
    velocity_fn: VelocityFn,
    x: Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    steps: int = 100,
    **kwargs,
) -> Tensor:
    """Second-order Heun (improved Euler) integrator."""
    dt = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        tb = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        k1 = velocity_fn(x, tb)
        k2 = velocity_fn(x + dt * k1, tb + dt)
        x = x + 0.5 * dt * (k1 + k2)
        t = t + dt
    return x


# Dormand-Prince 5(4) coefficients (the same as the default of torchdiffeq)
_DOPRI_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
_DOPRI_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_DOPRI_B = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
_DOPRI_BHAT = [
    5179 / 57600,
    0.0,
    7571 / 16695,
    393 / 640,
    -92097 / 339200,
    187 / 2100,
    1 / 40,
]


def dopri5_integrate(
    velocity_fn: VelocityFn,
    x: Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    first_step: Optional[float] = None,
    max_steps: int = 4096,
    **kwargs,
) -> Tensor:
    """Adaptive Dormand-Prince (dopri5) integrator, torchdiffeq-compatible.

    Implements the standard FSAL RK45 pair with the error control used by
    ``torchdiffeq.odeint`` (the solver the paper used for sampling).
    """
    direction = 1.0 if t1 >= t0 else -1.0
    t = t0
    h = direction * (first_step if first_step is not None else (t1 - t0) / 100.0)
    h = torch.tensor(h, dtype=x.dtype)

    def f(x_: Tensor, t_: Tensor) -> Tensor:
        tb = torch.full((x_.shape[0],), float(t_), device=x.device, dtype=x.dtype)
        return velocity_fn(x_, tb)

    k = [f(x, t)]
    n_steps = 0
    while direction * (t1 - t) > 1e-12 and n_steps < max_steps:
        h = torch.clamp(h, max=abs(t1 - t0))
        if direction * (t + h - t1) > 0:
            h = torch.tensor(t1 - t, dtype=x.dtype)

        ks = [k[0]]
        for i in range(1, 7):
            xi = x + h * sum(a * ki for a, ki in zip(_DOPRI_A[i], ks))
            ti = t + float(_DOPRI_C[i]) * float(h)
            ks.append(f(xi, torch.tensor(ti)))

        x5 = x + h * sum(b * ki for b, ki in zip(_DOPRI_B, ks))
        x4 = x + h * sum(b * ki for b, ki in zip(_DOPRI_BHAT, ks))
        # error norm: mixed absolute/relative, scaled by the state size
        scale = atol + rtol * torch.maximum(x.abs(), x5.abs())
        err = torch.sqrt(torch.mean(((x5 - x4) / scale) ** 2))
        err = float(err)

        accept = err <= 1.0 or abs(float(h)) <= 1e-8
        if accept:
            t = t + float(h)
            x = x5
            k = [ks[6]]  # FSAL
            n_steps += 1
        if err == 0:
            factor = 5.0
        else:
            factor = min(5.0, max(0.2, 0.9 * err ** (-1 / 5)))
        h = h * factor
        if abs(float(h)) < 1e-8:
            h = torch.tensor(direction * 1e-8, dtype=x.dtype)
    return x


def odeint(
    velocity_fn: VelocityFn,
    x: Tensor,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    method: str = "dopri5",
    **kwargs,
) -> Tensor:
    """Dispatch to the requested solver.

    ``method`` is one of ``dopri5`` (default, matching the paper),
    ``torchdiffeq`` (use the reference implementation if installed),
    ``euler`` (Algorithm 2) or ``heun``.
    """
    method = method.lower()
    if method == "dopri5":
        return dopri5_integrate(velocity_fn, x, t0=t0, t1=t1, **kwargs)
    if method == "euler":
        return euler_integrate(velocity_fn, x, t0=t0, t1=t1, **kwargs)
    if method == "heun":
        return heun_integrate(velocity_fn, x, t0=t0, t1=t1, **kwargs)
    if method == "torchdiffeq":
        from torchdiffeq import odeint as td_odeint

        def rhs(t, x_):
            tb = torch.full((x_.shape[0],), float(t), device=x.device, dtype=x.dtype)
            return velocity_fn(x_, tb)

        ts = torch.tensor([t0, t1], device=x.device, dtype=x.dtype)
        return td_odeint(rhs, x, ts, method="dopri5", atol=1e-5, rtol=1e-5)[-1]
    raise ValueError(f"unknown ODE method {method!r}")


# ---------------------------------------------------------------------------
# SDEs of Corollary 3.1
# ---------------------------------------------------------------------------
class ConstantEpsilon(torch.nn.Module):
    """eps_t = const (valid: the corollary holds for any eps_t >= 0)."""

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = float(eps)

    def __call__(self, t: Tensor) -> Tensor:
        return torch.full_like(t, self.eps)


class LinearEpsilon(torch.nn.Module):
    """eps_t = eps * t (vanishes at t = 0, often more stable in practice)."""

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = float(eps)

    def __call__(self, t: Tensor) -> Tensor:
        return self.eps * t


def forward_sde_sample(
    velocity_fn: VelocityFn,
    score_fn: Callable[[Tensor, Tensor], Tensor],
    gamma_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    eps_fn: Callable[[Tensor], Tensor] = ConstantEpsilon(1.0),
    steps: int = 250,
    t0: float = 0.0,
    t1: float = 1.0,
    log_score: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Euler-Maruyama integration of the forward SDE (eq. 11).

    dX_t = [b_t(X_t) - eps_t gamma_t^{-1} g_t(X_t)] dt + sqrt(2 eps_t) dW_t

    ``score_fn`` returns g_t(x) = E[z | I_t = x]; because
    gamma_t^{-1} g_t = -grad log rho_t the ratio is finite even where
    gamma_t = 0.  If ``score_fn`` instead returns grad log rho_t(x) directly,
    set ``log_score=True`` and the drift becomes b_t + eps_t grad log rho_t.
    """
    dt = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        tb = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        eps = eps_fn(tb)
        if log_score:
            drift = velocity_fn(x, tb) + eps.view(-1, *([1] * (x.dim() - 1))) * score_fn(x, tb)
        else:
            gamma = gamma_fn(tb).clamp(min=1e-6)
            drift = velocity_fn(x, tb) - (eps / gamma).view(-1, *([1] * (x.dim() - 1))) * score_fn(x, tb)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        noise_scale = torch.sqrt(2 * eps * dt).view(-1, *([1] * (x.dim() - 1)))
        x = x + drift * dt + noise_scale * noise
        t = t + dt
    return x


def backward_sde_sample(
    velocity_fn: VelocityFn,
    score_fn: Callable[[Tensor, Tensor], Tensor],
    gamma_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    eps_fn: Callable[[Tensor], Tensor] = ConstantEpsilon(1.0),
    steps: int = 250,
    t1: float = 1.0,
    t0: float = 0.0,
    log_score: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Euler-Maruyama integration of the backward SDE (eq. 13), t: 1 -> 0.

    dX_t = [b_t(X_t) + eps_t gamma_t^{-1} g_t(X_t)] dt + sqrt(2 eps_t) dW_t
    integrated backwards in time, which maps rho_1 to rho_0.  With
    ``log_score=True`` the drift term becomes -eps_t grad log rho_t.
    """
    dt = (t1 - t0) / steps
    t = t1
    for _ in range(steps):
        tb = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        eps = eps_fn(tb)
        if log_score:
            drift = velocity_fn(x, tb) - eps.view(-1, *([1] * (x.dim() - 1))) * score_fn(x, tb)
        else:
            gamma = gamma_fn(tb).clamp(min=1e-6)
            drift = velocity_fn(x, tb) + (eps / gamma).view(-1, *([1] * (x.dim() - 1))) * score_fn(x, tb)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        noise_scale = torch.sqrt(2 * eps * dt).view(-1, *([1] * (x.dim() - 1)))
        x = x - drift * dt + noise_scale * noise
        t = t - dt
    return x


__all__ = [
    "odeint",
    "euler_integrate",
    "heun_integrate",
    "dopri5_integrate",
    "ConstantEpsilon",
    "LinearEpsilon",
    "forward_sde_sample",
    "backward_sde_sample",
]
