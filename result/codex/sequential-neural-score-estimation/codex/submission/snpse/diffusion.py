"""Conditional score-based diffusion model for SBI (Sections 2.2 and E.3).

This module turns a trained score network
``s_psi(theta_t, x, t) ~ grad_theta log p_t(theta_t | x)`` into

* samples from the approximate posterior, by integrating the probability flow
  ODE (Eq. 4) backwards in time (Appendix E.3.3, "Sampling"), and
* log densities of those samples, using the instantaneous change-of-variables
  formula (Eq. 5) -- required by TSNPSE to define the truncated proposal.

Note on standardization (Appendix E.3.3, "Standardization"): the paper feeds
standardized parameters/observations to the score network.  We implement this by
performing the whole diffusion in the standardized parameter space ``z`` and
mapping back at the end.  This is an affine reparameterisation of the dynamics,
so it leaves all posterior summaries (and hence C2ST) unchanged, while making
the prior and reference distribution well conditioned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .odeint import odeint, odeint_fixed
from .sde import SDE


def _expand(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.expand(batch_size)
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise ValueError("cannot broadcast to batch size")


def exact_divergence(fn, y: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """Exact trace of the Jacobian of ``fn`` evaluated at ``y`` (Eq. 5).

    ``fn`` maps ``(B, d) -> (B, d)``.  This costs ``d`` backward passes, which
    is cheap for the low-dimensional parameter spaces considered here.  For
    higher dimensions, ``hutchinson_divergence`` gives an unbiased estimator.
    """
    y = y.requires_grad_(True)
    with torch.enable_grad():
        f = fn(y)
        div = torch.zeros(y.shape[:-1], device=y.device, dtype=y.dtype)
        for i in range(y.shape[-1]):
            (grad,) = torch.autograd.grad(
                f[..., i].sum(),
                y,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=False,
            )
            div = div + grad[..., i]
    return div


def hutchinson_divergence(
    fn, y: torch.Tensor, num_samples: int = 1, create_graph: bool = False
) -> torch.Tensor:
    """Skilling-Hutchinson trace estimator (see the Discussion section)."""
    y = y.requires_grad_(True)
    with torch.enable_grad():
        f = fn(y)
        div_total = torch.zeros(y.shape[:-1], device=y.device, dtype=y.dtype)
        for _ in range(num_samples):
            eps = torch.randn_like(y)
            (vjp,) = torch.autograd.grad(
                (f * eps).sum(),
                y,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=False,
            )
            div_total = div_total + (vjp * eps).sum(-1)
        return div_total / num_samples


@dataclass
class DiffusionConfig:
    solver: str = "rk45"  # rk45 (adaptive dopri5) | rk4 | euler
    rtol: float = 1e-5
    atol: float = 1e-5
    num_steps: int = 100  # for fixed-step solvers
    exact_trace: bool = True
    hutchinson_samples: int = 1


class DiffusionPosterior:
    """Approximate posterior given by a trained score network and an SDE."""

    def __init__(
        self,
        score_network: torch.nn.Module,
        sde: SDE,
        dim_parameters: int,
        config: Optional[DiffusionConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.net = score_network
        self.sde = sde
        self.dim_parameters = dim_parameters
        self.config = config or DiffusionConfig()
        self.device = device

    # -- velocity fields ----------------------------------------------------
    def score(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the score network ``s_psi(theta_t, x, t)``."""
        t = t.reshape(-1)
        if t.shape[0] == 1 and theta_t.shape[0] > 1:
            t = t.expand(theta_t.shape[0])
        return self.net(theta_t, x, t)

    def velocity(self, theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Probability flow ODE velocity (Eq. 4):
        ``f(theta, t) - 1/2 g(t)^2 grad_theta log p_t(theta | x)``.
        """
        f, g = self.sde.drift_and_diffusion(theta, t)
        s = self.score(theta, x, t)
        return f - 0.5 * (g ** 2) * s

    def reverse_velocity(self, theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse-time SDE drift (Eq. 3), parameterised in forward time."""
        t_rev = self.sde.T - t
        f_rev, g_rev = self.sde.drift_and_diffusion(theta, t_rev)
        s = self.score(theta, x, t_rev)
        return -f_rev + (g_rev ** 2) * s

    # -- integration --------------------------------------------------------
    def _integrate(self, func, y0: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        cfg = self.config
        if cfg.solver == "rk45":
            return odeint(func, y0, t0, t1, rtol=cfg.rtol, atol=cfg.atol)
        if cfg.solver == "rk4":
            return odeint_fixed(func, y0, t0, t1, num_steps=cfg.num_steps)
        if cfg.solver in ("euler", "fixed"):
            return _euler(func, y0, t0, t1, cfg.num_steps)
        raise ValueError(f"unknown solver {cfg.solver}")

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        x_obs: torch.Tensor,
        num_samples: int,
        use_reverse_sde: bool = False,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples from the approximate posterior (in ``z`` space)."""
        if seed is not None:
            torch.manual_seed(seed)
        sde = self.sde
        dtype = torch.get_default_dtype()
        z = sde.sample_reference((num_samples, self.dim_parameters), device=self.device, dtype=dtype)
        x = _expand(x_obs.reshape(1, -1), num_samples).to(z.dtype).to(z.device)

        if not use_reverse_sde:
            return self._integrate(lambda t, y: self.velocity(y, x, t.reshape(1)), z, sde.T, 0.0)

        steps = num_steps or self.config.num_steps
        dt = sde.T / steps
        for i in range(steps):
            tau = i * dt
            t_fwd = torch.full((num_samples,), float(sde.T - tau))
            drift = self.reverse_velocity(z, x, t_fwd)
            _, g = sde.drift_and_diffusion(z, sde.T - t_fwd)
            z = z + drift * dt + g * math.sqrt(dt) * torch.randn_like(z)
        return z

    # -- log density --------------------------------------------------------
    def _augmented_velocity(self, x: torch.Tensor):
        d = self.dim_parameters
        cfg = self.config

        def func(t, y):
            theta = y[:, :d]
            t_vec = t.reshape(1).expand(theta.shape[0]).to(theta.dtype)

            def vel(a):
                return self.velocity(a, x, t_vec)

            dzdt = vel(theta)
            if cfg.exact_trace:
                div = exact_divergence(vel, theta, create_graph=False)
            else:
                div = hutchinson_divergence(
                    vel, theta, num_samples=cfg.hutchinson_samples, create_graph=False
                )
            return torch.cat([dzdt, (-div).reshape(-1, 1)], dim=-1)

        return func

    def log_prob(self, x_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Log density of the approximate posterior at ``z`` (Eq. 5).

        Integrates ``d z_t / dt = v(z_t, t)`` from ``t = 0`` to ``t = T``
        together with ``d a / dt = -Tr[grad_z v(z_t, t)]``; then
        ``log p_0(z) = log p_T(z_T) - a(T)``.
        """
        num = z.shape[0]
        x = _expand(x_obs.reshape(1, -1), num).to(z.dtype).to(z.device)
        y0 = torch.cat(
            [z, torch.zeros(num, 1, device=z.device, dtype=z.dtype)], dim=-1
        ).detach()
        yT = self._integrate(self._augmented_velocity(x), y0, 0.0, self.sde.T)
        zT = yT[:, : self.dim_parameters]
        accT = yT[:, self.dim_parameters :].reshape(-1)
        return self.sde.log_reference(zT) - accT

    @torch.no_grad()
    def sample_and_log_prob(
        self, x_obs: torch.Tensor, num_samples: int, seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the approximate posterior and evaluate its log density.

        Both quantities come from a *single* integration of the augmented
        probability flow ODE, starting at the reference distribution.  This is
        exactly the procedure used in Appendix E.3.3 to estimate
        ``HPR_epsilon`` of the approximate posterior.
        """
        if seed is not None:
            torch.manual_seed(seed)
        sde = self.sde
        zT = sde.sample_reference((num_samples, self.dim_parameters), device=self.device)
        x = _expand(x_obs.reshape(1, -1), num_samples).to(zT.dtype).to(zT.device)
        log_ref = sde.log_reference(zT)
        y0 = torch.cat(
            [zT, torch.zeros(num_samples, 1, device=zT.device, dtype=zT.dtype)], dim=-1
        ).detach()
        y = self._integrate(self._augmented_velocity(x), y0, sde.T, 0.0)
        z0 = y[:, : self.dim_parameters]
        a0 = y[:, self.dim_parameters :].reshape(-1)
        return z0, log_ref + a0


def _euler(func, y0: torch.Tensor, t0: float, t1: float, num_steps: int) -> torch.Tensor:
    y = y0
    h = (float(t1) - float(t0)) / num_steps
    t = torch.as_tensor(float(t0), dtype=y0.dtype, device=y0.device)
    h_t = torch.as_tensor(h, dtype=y0.dtype, device=y0.device)
    for _ in range(num_steps):
        y = y + h_t * func(t, y)
        t = t + h_t
    return y
