"""Simulation-free regression objectives of Theorem 3.1 / Theorem A.1.

The paper shows that the velocity and the score of the interpolant are the
*unique* minimisers of

    L_b(b_hat) = int_0^1 E[ |b_hat_t(I_t, xi)|^2 - 2 I_dot_t . b_hat_t(I_t, xi) ] dt
                                                                      (eq. 7 / 29)
    L_g(g_hat) = int_0^1 E[ |g_hat_t(I_t, xi)|^2 - 2 z . g_hat_t(I_t, xi) ] dt

with the expectation taken over (x_0, x_1) ~ rho(x_0, x_1 | xi), xi ~ eta(xi)
and z ~ N(0, Id).  Both objectives are estimated with a single Monte-Carlo
draw of t ~ U([0, 1]) per example, exactly as in Algorithm 1:

    L_hat_b(b_hat) = 1/n_b sum_i [ |b_hat_{t_i}(I_{t_i})|^2
                                   - 2 I_dot_{t_i} . b_hat_{t_i}(I_{t_i}) ].

Because E|b_hat - I_dot|^2 = E|b_hat|^2 - 2 E[I_dot . b_hat] + E|I_dot|^2 and
the last term does not depend on b_hat, minimising ``L_hat_b`` is equivalent
(up to the additive constant E|I_dot|^2) to regressing the model onto the
interpolant velocity.  We therefore expose both the exact objective of the
paper and the equivalent "MSE form", since the latter has better numerical
behaviour.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .couplings import CoupledBatch
from .interpolants import InterpolantSchedule


def sample_t(batch_size: int, device=None, dtype=torch.float32) -> Tensor:
    """t ~ U([0, 1]) (Algorithm 1)."""
    return torch.rand(batch_size, device=device, dtype=dtype)


def flatten_sum(x: Tensor) -> Tensor:
    """Sum over everything but the batch dimension (used for the inner product)."""
    return x.flatten(1).sum(dim=1)


def velocity_objective(
    b_hat: Tensor,
    i_dot: Tensor,
    *,
    mse_form: bool = False,
) -> Tensor:
    """Per-example value of |b_hat|^2 - 2 I_dot . b_hat (or |b_hat - I_dot|^2)."""
    if mse_form:
        return flatten_sum((b_hat - i_dot) ** 2)
    return flatten_sum(b_hat**2) - 2.0 * flatten_sum(b_hat * i_dot)


def score_objective(g_hat: Tensor, z: Tensor, *, mse_form: bool = False) -> Tensor:
    """Per-example value of |g_hat|^2 - 2 z . g_hat (or |g_hat - z|^2)."""
    if mse_form:
        return flatten_sum((g_hat - z) ** 2)
    return flatten_sum(g_hat**2) - 2.0 * flatten_sum(g_hat * z)


def velocity_loss(
    velocity_model,
    schedule: InterpolantSchedule,
    batch: CoupledBatch,
    *,
    t: Optional[Tensor] = None,
    mse_form: bool = False,
    n_samples: int = 1,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, dict]:
    """Monte-Carlo estimate of L_b (eq. 22 / Algorithm 1).

    Parameters
    ----------
    velocity_model : callable (x_t, t, cond, labels) -> predicted velocity
    schedule : the interpolant schedule (alpha_t, beta_t, gamma_t)
    batch : a :class:`~si_couplings.couplings.CoupledBatch`
    t : optional pre-drawn times (default: t ~ U([0, 1]))
    mse_form : use the equivalent MSE objective (recommended; see module docstring)
    n_samples : number of Monte-Carlo draws of (t, z) per example

    Returns
    -------
    (loss, info) where ``info`` contains diagnostics such as the squared norm
    of the interpolant velocity, E|I_dot_t|^2, whose integral over t is the
    upper bound on the transport cost of Proposition 3.1.
    """
    x0, x1 = batch.x0, batch.x1
    total = None
    info: dict = {}
    for _ in range(n_samples):
        tt = sample_t(x0.shape[0], x0.device, x0.dtype) if t is None else t
        if schedule.coefficients(tt).gamma.abs().max() > 0:
            z = torch.randn_like(x0)
        else:
            z = None
        i_t = schedule.interpolate(tt, x0, x1, z)
        i_dot = schedule.interpolate_velocity(tt, x0, x1, z)
        b_hat = velocity_model(i_t, tt, batch.cond, batch.labels, batch.mask)
        per_example = velocity_objective(b_hat, i_dot, mse_form=mse_form)
        total = per_example if total is None else total + per_example
        with torch.no_grad():
            info["idot_sq"] = flatten_sum(i_dot**2).mean().detach()
    return (total / n_samples).mean(), info


def score_loss(
    score_model,
    schedule: InterpolantSchedule,
    batch: CoupledBatch,
    *,
    t: Optional[Tensor] = None,
    mse_form: bool = False,
    n_samples: int = 1,
) -> tuple[Tensor, dict]:
    """Monte-Carlo estimate of L_g (eq. 7) for a network g_hat(z | I_t, xi)."""
    x0, x1 = batch.x0, batch.x1
    total = None
    info: dict = {}
    for _ in range(n_samples):
        tt = sample_t(x0.shape[0], x0.device, x0.dtype) if t is None else t
        z = torch.randn_like(x0)
        i_t = schedule.interpolate(tt, x0, x1, z)
        g_hat = score_model(i_t, tt, batch.cond, batch.labels)
        per_example = score_objective(g_hat, z, mse_form=mse_form)
        total = per_example if total is None else total + per_example
    return (total / n_samples).mean(), info


def transport_cost_upper_bound(
    schedule: InterpolantSchedule,
    batch: CoupledBatch,
    *,
    n_times: int = 101,
) -> Tensor:
    """Monte-Carlo estimate of int_0^1 E|I_dot_t|^2 dt (Proposition 3.1).

    This quantity upper-bounds the squared transport cost
    E_{x_0 ~ rho_0}[|X_{t=1}(x_0) - x_0|^2] of the probability-flow ODE, and
    is the object the paper compares between couplings in eq. (21).
    """
    x0, x1 = batch.x0, batch.x1
    ts = torch.linspace(0.0, 1.0, n_times, device=x0.device, dtype=x0.dtype)
    vals = []
    for t in ts:
        tb = t.expand(x0.shape[0])
        z = torch.randn_like(x0) if schedule.coefficients(tb).gamma.abs().max() > 0 else None
        i_dot = schedule.interpolate_velocity(tb, x0, x1, z)
        vals.append(flatten_sum(i_dot**2).mean())
    vals = torch.stack(vals)
    return torch.trapz(vals, ts)


__all__ = [
    "sample_t",
    "flatten_sum",
    "velocity_objective",
    "score_objective",
    "velocity_loss",
    "score_loss",
    "transport_cost_upper_bound",
]
