"""Instantaneous change-of-variables density evaluation for probability-flow ODEs.

The probability-flow ODE

    dtheta = [f(theta, t) - 0.5 * g(t)^2 * s_psi(theta, x, t)] dt =: v(theta, t) dt

defines a normalizing flow between the prior distribution p_T(theta_T) (the
marginal of the forward SDE at time T) and the model posterior p_0(theta_0 | x).
The instantaneous change-of-variables formula (Chen et al., 2018) gives

    log p_0(theta_0 | x) = log p_T(theta_T) + integral_0^T div(v)(theta_t, t) dt,

where theta_t is obtained by integrating the ODE *forward* from t=0 to t=T
starting at theta_0.  The divergence is estimated with the Hutchinson trace
estimator

    div(v)(theta) = E_{eps ~ N(0, I)} [ eps^T (J_v(theta) eps) ],
    eps^T (J_v eps) = eps^T grad_theta (v(theta) . eps),

which is computed with reverse-mode automatic differentiation.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .sde import SDE

__all__ = [
    "hutchinson_divergence",
    "ProbabilityFlowDensity",
    "log_probability_flow",
]

# Optional lazy import for torchdiffeq (required for adaptive ODE integration).
try:  # pragma: no cover - import guard
    from torchdiffeq import odeint as _odeint
except ImportError:  # pragma: no cover
    _odeint = None


def _as_float(t) -> float:
    """Return *t* as a Python float, supporting tensors and arrays."""
    if torch.is_tensor(t):
        return float(t.detach().cpu().item())
    try:
        return float(t)
    except (TypeError, ValueError):
        return float(torch.as_tensor(t).item())


def _expand_to_batch(value: torch.Tensor, batch: int) -> torch.Tensor:
    """Broadcast an observation vector/row to a batch of *batch* rows."""
    if value.dim() == 1:
        return value.unsqueeze(0).expand(batch, -1)
    if value.shape[0] == 1 and batch > 1:
        return value.expand(batch, -1)
    return value


def hutchinson_divergence(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Estimate div_theta(v(theta, t)) with the Hutchinson trace estimator.

    Args:
        velocity_fn: Callable ``(theta, t) -> v`` where ``theta`` has shape
            ``(batch, d)`` and ``v`` has the same shape.
        theta: Point(s) at which to evaluate the divergence, shape ``(batch, d)``.
        t: Time tensor broadcastable against the batch.
        noise: Optional Gaussian noise ``(batch, d)``.  If not given, fresh
            ``N(0, I)`` noise is sampled.

    Returns:
        Per-sample divergence estimates with shape ``(batch,)``.
    """
    theta = theta.detach().clone().requires_grad_(True)
    v = velocity_fn(theta, t)
    if noise is None:
        noise = torch.randn_like(theta)
    v_dot_eps = (v * noise).sum(dim=-1)  # (batch,)
    # grad_theta sum_i v_i * eps_i = J_v^T eps, hence eps^T J_v eps below.
    (grad,) = torch.autograd.grad(v_dot_eps.sum(), theta, create_graph=False)
    return (grad * noise).sum(dim=-1)


class ProbabilityFlowDensity:
    """Evaluate log p_0(theta_0 | x) for a probability-flow posterior model.

    The score function must have the signature used throughout the codebase,

        ``score_fn(theta, x, t) -> scores``,

    with ``theta`` of shape ``(batch, theta_dim)``, ``x`` of shape
    ``(batch, x_dim)``, and ``t`` of shape ``(batch,)``.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        x_obs: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "rk45",
        hutchinson_samples: int = 1,
    ) -> None:
        self.sde = sde
        self.score_fn = score_fn
        self.x_obs = x_obs
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.hutchinson_samples = int(hutchinson_samples)
        self._noise: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Velocity and dynamics
    # ------------------------------------------------------------------
    def velocity(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Probability-flow ODE velocity ``v(theta, t)``."""
        batch = theta.shape[0]
        x = _expand_to_batch(self.x_obs, batch)
        t_val = _as_float(t)
        t_batch = torch.full(
            (batch,), t_val, device=theta.device, dtype=theta.dtype
        )
        score = self.score_fn(theta, x, t_batch)
        return self.sde.ode_velocity(theta, t_batch, score)

    def _augmented_dynamics(
        self, t: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Augmented dynamics ``[v(theta, t); div(v)(theta, t)]``.

        The state has shape ``(batch, theta_dim + 1)`` where the final column is
        the accumulated log-density increment.  Integrating forward in time adds
        ``+ div(v)`` to this column, matching the instantaneous
        change-of-variables formula.
        """
        theta = state[:, :-1].clone().requires_grad_(True)
        v = self.velocity(theta, t)

        if self._noise is None:
            noise = torch.randn_like(theta)
        else:
            noise = self._noise

        v_dot_eps = (v * noise).sum(dim=-1)
        (grad,) = torch.autograd.grad(v_dot_eps.sum(), theta, create_graph=False)
        div = (grad * noise).sum(dim=-1, keepdim=True)  # (batch, 1)
        return torch.cat([v, div], dim=-1)

    # ------------------------------------------------------------------
    # Density evaluation
    # ------------------------------------------------------------------
    def log_prob(self, theta0: torch.Tensor, x_obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute ``log p_0(theta_0 | x_obs)`` for posterior samples ``theta0``.

        Args:
            theta0: Approximate posterior samples at time zero, shape
                ``(batch, theta_dim)`` (or ``(theta_dim,)`` for a single sample).
            x_obs: Optional observation override; defaults to the observation
                stored at construction.

        Returns:
            Log-density estimates, shape ``(batch,)`` (or scalar for a single
            sample).
        """
        if _odeint is None:
            raise ImportError(
                "torchdiffeq is required for density evaluation. "
                "Install it with `pip install torchdiffeq`."
            )

        if x_obs is not None:
            self.x_obs = x_obs

        single = theta0.dim() == 1
        if single:
            theta0 = theta0.unsqueeze(0)

        batch, dim = theta0.shape
        device = theta0.device
        dtype = theta0.dtype
        T = float(self.sde.T)

        # Average several independent Hutchinson trajectories when requested.
        log_probs = []
        for _ in range(self.hutchinson_samples):
            self._noise = torch.randn_like(theta0)
            state0 = torch.cat(
                [theta0, torch.zeros(batch, 1, device=device, dtype=dtype)], dim=-1
            )
            t_span = torch.tensor([0.0, T], device=device, dtype=dtype)
            traj = _odeint(
                self._augmented_dynamics,
                state0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
            final_state = traj[-1]
            theta_T = final_state[:, :-1]
            logp_delta = final_state[:, -1]
            log_p0 = logp_delta + self.sde.prior_logp(theta_T)
            log_probs.append(log_p0)

        self._noise = None

        if self.hutchinson_samples == 1:
            log_p0 = log_probs[0]
        else:
            log_p0 = torch.stack(log_probs, dim=0).logsumexp(dim=0) - float(
                torch.tensor(self.hutchinson_samples).log()
            )

        if single:
            return log_p0.squeeze(0)
        return log_p0


def log_probability_flow(
    sde: SDE,
    score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    theta0: torch.Tensor,
    x_obs: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method: str = "rk45",
    hutchinson_samples: int = 1,
) -> torch.Tensor:
    """Convenience wrapper around :class:`ProbabilityFlowDensity`."""
    estimator = ProbabilityFlowDensity(
        sde=sde,
        score_fn=score_fn,
        x_obs=x_obs,
        rtol=rtol,
        atol=atol,
        method=method,
        hutchinson_samples=hutchinson_samples,
    )
    return estimator.log_prob(theta0)
