"""Probability-flow ODE sampling for posterior score models.

This module solves the probability-flow ODE

    dtheta = [f(theta, t) - 0.5 * g(t)^2 * s(theta, x, t)] dt

backwards in time from ``t=T`` to ``t=0``. The solution at ``t=0`` is an
approximate draw from the posterior ``p(theta | x)``.

We provide both a torchdiffeq-based integrator (RK45) and a lightweight
Euler fallback for environments where ``torchdiffeq`` is not installed.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .sde import SDE

__all__ = [
    "ProbabilityFlowSampler",
    "sample_probability_flow",
    "euler_probability_flow_sample",
]


def _integrate_ode(
    ode_func: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t_span: tuple[float, float],
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method: str = "rk45",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Integrate ``dy/dt = ode_func(t, y)`` from ``t_span[0]`` to ``t_span[1]``.

    Uses ``torchdiffeq.odeint`` when available, otherwise falls back to a dense
    Euler integrator (only suitable for validation on small problems).
    """
    try:
        from torchdiffeq import odeint  # type: ignore

        t_tensor = torch.tensor(t_span, device=device, dtype=dtype)
        solution = odeint(
            lambda t, y: ode_func(t, y),
            y0,
            t_tensor,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        # odeint returns [len(t_span), batch, dim]; return last time point.
        return solution[-1]
    except Exception as exc:  # pragma: no cover - environment fallback
        if method not in ("rk45", "dopri5", "euler"):
            raise
        if method in ("rk45", "dopri5"):
            # Deliberately fall through to Euler; RK45 without torchdiffeq is
            # not implemented here.
            pass
        return _euler_integrate(ode_func, y0, t_span, device=device, dtype=dtype)


def _euler_integrate(
    ode_func: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t_span: tuple[float, float],
    steps: int = 200,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Fixed-step Euler integrator from ``t_span[0]`` down to ``t_span[1]``."""
    t_start, t_end = t_span
    y = y0
    dt = (t_end - t_start) / float(steps)
    for i in range(steps):
        t = t_start + dt * i
        y = y + dt * ode_func(t, y)
    return y


class ProbabilityFlowSampler:
    """Sampler that solves the probability-flow ODE backwards in time.

    Parameters
    ----------
    sde:
        Forward SDE defining the perturbation and prior.
    score_fn:
        Callable ``(theta, x, t) -> score``. Must be vectorized over the batch
        dimension of ``theta`` and ``x`` and return a tensor with the same shape
        as ``theta``.
    rtol, atol:
        Relative/absolute tolerances passed to the ODE integrator.
    method:
        ODE integration method (``"rk45"`` is the paper default).
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "rk45",
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def _velocity(
        self, t: torch.Tensor, theta: torch.Tensor, x_obs: torch.Tensor
    ) -> torch.Tensor:
        """Probability-flow ODE velocity ``f - 0.5 g^2 s``."""
        # Broadcast t to a tensor shaped like the batch dimension.
        if torch.is_tensor(t):
            t_value = t[0] if t.numel() == 1 else t
        else:
            t_value = float(t)
        t_batch = torch.full(
            (theta.shape[0],), float(t_value), device=theta.device, dtype=theta.dtype
        )
        score = self.score_fn(theta, x_obs, t_batch)
        return self.sde.ode_velocity(theta, t_batch, score)

    def sample(
        self,
        n_samples: int,
        x_obs: torch.Tensor,
        theta_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Draw ``n_samples`` approximate posterior samples for ``x_obs``.

        ``x_obs`` may be a single observation (shape ``(x_dim,)``) or a batch
        (shape ``(batch, x_dim)``). In the batched case, each observation uses
        the same number of initial theta draws and the returned tensor has shape
        ``(batch, n_samples, theta_dim)``.
        """
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
            single = True
        else:
            single = False

        batch_size = x_obs.shape[0]
        if theta_dim is None:
            theta_dim = x_obs.shape[1]

        if device is None:
            device = x_obs.device
        if dtype is None:
            dtype = x_obs.dtype

        theta_T = self.sde.prior_sample(n_samples * batch_size, theta_dim, device).to(
            dtype
        )

        # odeint integrates a single batch of states, so flatten batch * n.
        x_expanded = x_obs.repeat_interleave(n_samples, dim=0)

        y0 = theta_T
        solution = _integrate_ode(
            lambda t, y: self._velocity(t, y, x_expanded),
            y0,
            (float(self.sde.T), 0.0),
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            device=device,
            dtype=dtype,
        )

        if not single:
            solution = solution.reshape(batch_size, n_samples, theta_dim)
        return solution

    def score(
        self, theta: torch.Tensor, x_obs: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Convenience wrapper around the internal score function."""
        return self.score_fn(theta, x_obs, t)


def sample_probability_flow(
    sde: SDE,
    score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    n_samples: int,
    x_obs: torch.Tensor,
    theta_dim: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method: str = "rk45",
) -> torch.Tensor:
    """Functional convenience wrapper for probability-flow sampling.

    Returns
    -------
    torch.Tensor
        Approximate posterior samples at ``t=0`` with shape
        ``(n_samples, theta_dim)`` for a single observation.
    """
    sampler = ProbabilityFlowSampler(
        sde=sde, score_fn=score_fn, rtol=rtol, atol=atol, method=method
    )
    return sampler.sample(
        n_samples=n_samples,
        x_obs=x_obs,
        theta_dim=theta_dim,
        device=device,
        dtype=dtype,
    )


def euler_probability_flow_sample(
    sde: SDE,
    score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    n_samples: int,
    x_obs: torch.Tensor,
    theta_dim: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    steps: int = 200,
) -> torch.Tensor:
    """Explicit Euler probability-flow sampling (no torchdiffeq dependency).

    Useful for unit tests and CPU validation. Accuracy is lower than RK45, so
    the default paper pipeline uses :func:`sample_probability_flow`.
    """
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)
        single = True
    else:
        single = False

    batch_size = x_obs.shape[0]
    if theta_dim is None:
        theta_dim = x_obs.shape[1]

    if device is None:
        device = x_obs.device
    if dtype is None:
        dtype = x_obs.dtype

    theta = sde.prior_sample(n_samples * batch_size, theta_dim, device).to(dtype)
    x_expanded = x_obs.repeat_interleave(n_samples, dim=0)

    dt = -float(sde.T) / float(steps)
    for i in range(steps):
        t_value = float(sde.T) + dt * i
        t = torch.full(
            (theta.shape[0],), t_value, device=device, dtype=dtype
        )
        score = score_fn(theta, x_expanded, t)
        velocity = sde.ode_velocity(theta, t, score)
        theta = theta + dt * velocity

    if not single:
        theta = theta.reshape(batch_size, n_samples, theta_dim)
    return theta
