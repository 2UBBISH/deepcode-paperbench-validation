"""Self-contained adaptive ODE solver (Dormand-Prince RK45).

The paper uses an off-the-shelf RK45 solver (Appendix E.3.3) to integrate the
probability flow ODE (Eq. 4) and the augmented change-of-variables ODE (Eq. 5).
``torchdiffeq`` is an optional dependency, and we do not want to require it, so
this module implements the classical Dormand-Prince 4(5) embedded Runge-Kutta
pair in pure PyTorch.  It supports integration in the reverse direction
(``t1 < t0``), which is how the reverse-time / probability-flow dynamics are
integrated (from the reference noise level ``t = T`` down to ``t = 0``).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch

# Dormand-Prince 5(4) coefficients -------------------------------------------
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)

_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)

_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_B4 = (
    5179 / 57600,
    0.0,
    7571 / 16695,
    393 / 640,
    -92097 / 339200,
    187 / 2100,
    1 / 40,
)


def _as_state(y: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(y):
        raise TypeError("odeint expects a single torch.Tensor state")
    return y


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2))


def _dopri5_step(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Dormand-Prince step. Returns (y5, error_estimate)."""
    ks: List[torch.Tensor] = []
    for stage in range(7):
        if stage == 0:
            ti, yi = t, y
        elif stage < 6:
            acc = torch.zeros_like(y)
            for j, a in enumerate(_A[stage]):
                acc = acc + a * ks[j]
            ti, yi = t + _C[stage] * h, y + h * acc
        else:
            # k7 is evaluated at the 5th-order solution (FSAL property)
            acc = torch.zeros_like(y)
            for j, a in enumerate(_A[6]):
                if a != 0.0:
                    acc = acc + a * ks[j]
            ti, yi = t + h, y + h * acc
        ks.append(func(ti, yi))

    y5 = y
    for b, k in zip(_B5, ks):
        if b != 0.0:
            y5 = y5 + h * b * k

    y4 = y
    for b, k in zip(_B4, ks):
        if b != 0.0:
            y4 = y4 + h * b * k

    return y5, y5 - y4


def odeint(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    t1: float,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    first_step: Optional[float] = None,
    max_num_steps: int = 100_000,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Integrate ``dy/dt = func(t, y)`` from ``t0`` to ``t1``.

    Args:
        func: vector field, called as ``func(t, y)`` where ``t`` is a scalar
            tensor (dtype ``y0.dtype``) and ``y`` has the shape of ``y0``.
        y0: initial state.
        t0: initial time.
        t1: final time.  May be smaller than ``t0`` (reverse time).
        rtol, atol: relative / absolute tolerances used for step-size control.
        first_step: optional initial step size (signed, magnitude used).
        max_num_steps: safety cap on the number of accepted steps.
        return_trajectory: if ``True``, also return the states at all accepted
            times (used for debugging / visualisation).

    Returns:
        The state at ``t1`` (and optionally the trajectory).
    """
    y = _as_state(y0)
    dtype, device = y.dtype, y.device
    t0_t = torch.as_tensor(float(t0), dtype=dtype, device=device)

    direction = 1.0 if float(t1) > float(t0) else -1.0
    span = abs(float(t1) - float(t0))
    if span == 0.0:
        return (y, [t0_t], [y.clone()]) if return_trajectory else y

    if first_step is None:
        h_abs = span / 100.0
    else:
        h_abs = abs(float(first_step))

    t = t0_t
    traj_t: List[torch.Tensor] = [t.clone()]
    traj_y: List[torch.Tensor] = [y.clone()]

    target = float(t1)
    n_steps = 0
    while (direction > 0 and float(t) < target) or (direction < 0 and float(t) > target):
        n_steps += 1
        if n_steps > max_num_steps:
            raise RuntimeError("odeint: exceeded max_num_steps")

        # do not step past the endpoint
        h_signed = direction * h_abs
        if direction * (float(t) + h_signed - target) > 0:
            h_signed = target - float(t)
        h = torch.as_tensor(h_signed, dtype=dtype, device=device)

        y_new, err = _dopri5_step(func, t, y, h)
        scale = atol + rtol * torch.maximum(y.abs(), y_new.abs())
        err_norm = float(_norm(err / scale))

        if err_norm <= 1.0 or h_abs <= 1e-12 * max(1.0, span):
            # accept
            t = t + h
            y = y_new
            traj_t.append(t.clone())
            traj_y.append(y.clone())
            if err_norm == 0.0:
                factor = 5.0
            else:
                factor = 0.9 * err_norm ** (-1.0 / 5.0)
            h_abs = h_abs * min(5.0, max(0.2, factor))
        else:
            # reject and retry with a smaller step
            factor = max(0.1, 0.9 * err_norm ** (-1.0 / 5.0))
            h_abs = h_abs * factor

    if return_trajectory:
        return y, traj_t, traj_y
    return y


def odeint_fixed(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    t1: float,
    num_steps: int = 100,
) -> torch.Tensor:
    """Classical fixed-step RK4 integrator (used for cheap smoke tests)."""
    y = _as_state(y0)
    dtype, device = y.dtype, y.device
    h = (float(t1) - float(t0)) / num_steps
    t = torch.as_tensor(float(t0), dtype=dtype, device=device)
    h_t = torch.as_tensor(h, dtype=dtype, device=device)
    for _ in range(num_steps):
        k1 = func(t, y)
        k2 = func(t + 0.5 * h_t, y + 0.5 * h_t * k1)
        k3 = func(t + 0.5 * h_t, y + 0.5 * h_t * k2)
        k4 = func(t + h_t, y + h_t * k3)
        y = y + (h_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h_t
    return y
