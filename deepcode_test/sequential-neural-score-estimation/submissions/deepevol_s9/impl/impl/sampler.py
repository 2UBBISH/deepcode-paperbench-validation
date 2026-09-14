'''Probability flow ODE posterior sampling and log-density evaluation.

Sampling uses the time-reversed probability flow ODE with the trained score
network substituted in place of the posterior score.  Log-density evaluation
augments the same ODE with the instantaneous change-of-variables trace.
'''

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import Tensor


NODE_RTOL = 1e-5
NODE_ATOL = 1e-6


def _device_of(score_model):
    return next(score_model.parameters()).device


def _reference_log_prob(theta_T: Tensor, sde) -> float:
    '''Log probability of the reference distribution at ``theta_T``.'''
    dim = theta_T.numel()
    flat = theta_T.reshape(-1)
    sigma = float(
        sde.std(torch.tensor([sde.t_max], dtype=torch.float32, device=flat.device))
        .detach()
        .cpu()
        .item()
    )
    const = -0.5 * dim * math.log(2.0 * math.pi) - dim * math.log(sigma)
    return const - 0.5 * torch.sum(flat * flat).detach().cpu().item() / (sigma * sigma)


def _trace_divergence(v: Tensor, theta: Tensor) -> Tensor:
    '''Compute the trace of the Jacobian of ``v`` with respect to ``theta``.'''
    dim = theta.size(1)
    trace = torch.zeros_like(theta[:, 0])
    for j in range(dim):
        grad_outputs = torch.zeros_like(v)
        grad_outputs[:, j] = 1.0
        grad_j = torch.autograd.grad(
            v,
            theta,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
        )[0]
        trace = trace + grad_j[:, j]
    return trace


def _log_density_single(theta, score_model, sde, observation) -> Tensor:
    device = _device_of(score_model)
    theta = theta.detach().to(device).reshape(-1)
    dim = theta.numel()

    obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
    if obs.ndim == 0:
        obs = obs.reshape(1, 1)
    elif obs.ndim == 1:
        obs = obs.unsqueeze(0)
    else:
        obs = obs.reshape(1, -1)

    t_span = (float(sde.t_min), float(sde.t_max))
    y0 = np.concatenate(
        [theta.detach().cpu().numpy().reshape(-1).astype(np.float64), [0.0]]
    )

    def rhs(t, state):
        theta_t = torch.tensor(
            state[:dim], dtype=torch.float32, device=device, requires_grad=True
        ).reshape(1, dim)
        tt = torch.tensor([t], dtype=torch.float32, device=device)
        with torch.enable_grad():
            score = score_model(theta_t, obs, tt)
            drift = sde.f(theta_t, tt) - 0.5 * (sde.g(tt) ** 2) * score
            divergence = _trace_divergence(drift, theta_t)
            dlogp = -divergence.detach().cpu().item()
        drift_flat = drift.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return np.concatenate([drift_flat, np.array([dlogp])])

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method='RK45',
        rtol=NODE_RTOL,
        atol=NODE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"probability flow ODE failed: {sol.message}")

    theta_T = torch.tensor(sol.y[:dim, -1], dtype=torch.float32, device=device)
    final_logp = float(sol.y[-1, -1])
    reference_logp = _reference_log_prob(theta_T, sde)
    return torch.tensor(reference_logp - final_logp, dtype=torch.float32, device=device)


def instantaneous_change_of_variables_log_density(
    theta: Tensor,
    score_model,
    sde,
    observation,
) -> Tensor:
    '''Compute approximate posterior log densities for posterior samples.

    The function integrates the probability flow ODE forward from ``t_min`` to
    ``t_max`` while accumulating the negative trace of the ODE vector field.
    The final log density is the reference log probability at ``t_max`` minus
    the accumulated trace, which approximates the posterior log density at the
    supplied ``theta``.
    '''
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32)
    single = theta_tensor.ndim == 1
    if single:
        theta_tensor = theta_tensor.unsqueeze(0)
    if theta_tensor.ndim != 2:
        raise ValueError('theta must be a 1-D or 2-D tensor')

    logs = [
        _log_density_single(theta_tensor[i], score_model, sde, observation)
        for i in range(theta_tensor.size(0))
    ]
    result = torch.stack(logs)
    if single:
        result = result[0]
    return result


class ProbabilityFlowODESampler:
    '''Sample approximate posterior draws through the probability flow ODE.

    Parameters
    ----------
    score_model:
        Conditional score network ``s_psi(theta_t, x, t)``.
    sde:
        Forward noising SDE with ``f``, ``g``, ``t_min``, and ``t_max``.
    observation:
        Observed summary statistics.
    solver:
        SciPy ODE solver name. The paper uses ``RK45``.
    '''

    def __init__(self, score_model, sde, observation, solver='RK45'):
        self.score_model = score_model
        self.sde = sde
        self.observation = torch.as_tensor(observation, dtype=torch.float32)
        if self.observation.ndim == 1:
            self.observation = self.observation.unsqueeze(0)
        elif self.observation.ndim > 2:
            raise ValueError('observation must be a 1-D or 2-D tensor')
        self.solver = solver
        self.theta_dim = int(score_model.theta_dim)
        self.score_model.eval()

    def _obs_batch(self, batch_size):
        return self.observation.repeat(batch_size, 1)

    def _draw_reference_samples(self, n_samples):
        device = _device_of(self.score_model)
        sigma_ref = float(
            self.sde.std(torch.tensor([self.sde.t_max], device=device))
            .detach()
            .cpu()
            .item()
        )
        return torch.randn(n_samples, self.theta_dim, device=device) * sigma_ref

    def _ode_rhs_np(self, t, theta_np):
        device = _device_of(self.score_model)
        theta = torch.tensor(
            theta_np, dtype=torch.float32, device=device
        ).reshape(1, self.theta_dim)
        tt = torch.tensor([t], dtype=torch.float32, device=device)
        obs = self._obs_batch(1).to(device)
        with torch.no_grad():
            score = self.score_model(theta, obs, tt)
            drift = self.sde.f(theta, tt) - 0.5 * (self.sde.g(tt) ** 2) * score
        return drift.detach().cpu().numpy().astype(np.float64).reshape(-1)

    def sample(self, n_samples: int) -> Tensor:
        '''Return ``n_samples`` approximate posterior draws.'''
        n_samples = int(n_samples)
        if n_samples <= 0:
            return torch.empty((0, self.theta_dim), device=_device_of(self.score_model))

        initial = self._draw_reference_samples(n_samples)
        samples = []
        for i in range(n_samples):
            y0 = initial[i].detach().cpu().numpy().astype(np.float64)
            sol = solve_ivp(
                self._ode_rhs_np,
                [float(self.sde.t_max), float(self.sde.t_min)],
                y0,
                method=self.solver,
                rtol=NODE_RTOL,
                atol=NODE_ATOL,
            )
            if not sol.success:
                raise RuntimeError(f"probability flow ODE failed: {sol.message}")
            samples.append(sol.y[:, -1])

        return torch.tensor(
            np.stack(samples), dtype=torch.float32, device=_device_of(self.score_model)
        )
