"""Gaussian Linear benchmark.

Task specification (from the paper):
    theta in R^10, p(theta) = N(theta | 0, 0.1 * I),
    p(x | theta) = N(x | theta, 0.1 * I), x in R^10.

Because both prior and likelihood are conjugate Gaussians with equal variance,
the posterior is available in closed form:

    p(theta | x) = N(theta | x / 2, 0.05 * I).

This benchmark is used in the unit-validation stage: the probability-flow ODE
sampler should recover these posterior moments on this analytically tractable
task.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import Benchmark, to_torch


class GaussianLinear(Benchmark):
    """Gaussian linear benchmark with an analytic Gaussian posterior."""

    name: str = "gaussian_linear"

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        dim: int = 10,
        prior_var: float = 0.1,
        likelihood_var: float = 0.1,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.theta_dim = int(dim)
        self.x_dim = int(dim)
        self.prior_var = float(prior_var)
        self.likelihood_var = float(likelihood_var)

        # Precomputed posterior precision and variance (scalar, isotropic).
        self.posterior_precision = 1.0 / self.prior_var + 1.0 / self.likelihood_var
        self.posterior_var = 1.0 / self.posterior_precision

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------
    def prior_sample(self, n: int) -> torch.Tensor:
        """Sample theta ~ N(0, prior_var * I)."""
        std = math.sqrt(self.prior_var)
        return std * torch.randn(
            (int(n), self.theta_dim), device=self.device, dtype=self.dtype
        )

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log probability of the Gaussian prior, summed over dimensions."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        var = self.prior_var
        const = -0.5 * self.theta_dim * math.log(2.0 * math.pi * var)
        return const - 0.5 * torch.sum(theta * theta, dim=-1) / var

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------
    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Sample x | theta ~ N(theta, likelihood_var * I)."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        std = math.sqrt(self.likelihood_var)
        noise = std * torch.randn_like(theta)
        return theta + noise

    # ------------------------------------------------------------------
    # Analytic posterior / references
    # ------------------------------------------------------------------
    def posterior_mean(self, x_obs: torch.Tensor) -> torch.Tensor:
        """Closed-form posterior mean for Gaussian linear."""
        x_obs = to_torch(x_obs, device=self.device, dtype=self.dtype)
        return (self.posterior_var / self.likelihood_var) * x_obs

    def reference_posterior_samples(
        self, x_obs: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Draw exact posterior samples N(x / 2, 0.05 * I).

        Accepts a single observation of shape ``(theta_dim,)`` (returns
        ``(n, theta_dim)``) or a batch ``(batch, theta_dim)`` (returns
        ``(batch, n, theta_dim)``).
        """
        x_obs = to_torch(x_obs, device=self.device, dtype=self.dtype)
        single = x_obs.dim() == 1
        if single:
            x_obs = x_obs.unsqueeze(0)

        batch = x_obs.shape[0]
        mean = self.posterior_mean(x_obs)  # (batch, theta_dim)
        std = math.sqrt(self.posterior_var)
        samples = mean.unsqueeze(1) + std * torch.randn(
            (batch, int(n), self.theta_dim), device=self.device, dtype=self.dtype
        )

        if single:
            return samples.squeeze(0)
        return samples

    def sample_observation(self, n: int = 1) -> torch.Tensor:
        """Sample observations from the prior predictive distribution."""
        theta = self.prior_sample(int(n))
        return self.simulator(theta)
