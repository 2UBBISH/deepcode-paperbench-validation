"""Two Moons benchmark simulator.

Specification
-------------
Prior:       theta in R^2 with p(theta) = U(-1, 1)^2.
Simulator:   for each parameter vector theta = (theta_1, theta_2):
                 alpha ~ U(-pi/2, pi/2)
                 r     ~ N(0.1, 0.01^2)
                 x_1 = r * cos(alpha) + 0.25 - |theta_1 + theta_2| / sqrt(2)
                 x_2 = r * sin(alpha) + (-theta_1 + theta_2) / sqrt(2)

The reference posterior is not available in closed form. If ``sbibm`` is
installed, reference posterior samples are loaded from the sbibm package
(otherwise ``reference_posterior_samples`` returns ``None``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from npse.benchmarks.base import Benchmark, to_torch

__all__ = ["TwoMoons"]


class TwoMoons(Benchmark):
    """Two Moons simulation-based inference benchmark."""

    name = "two_moons"
    theta_dim = 2
    x_dim = 2

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        low: float = -1.0,
        high: float = 1.0,
        radius: float = 0.1,
        radius_std: float = 0.01,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.low = low
        self.high = high
        self.radius = radius
        self.radius_std = radius_std

    def prior_sample(self, n: int) -> torch.Tensor:
        """Sample theta ~ U(low, high)^2."""
        return (
            (self.high - self.low)
            * torch.rand(n, self.theta_dim, device=self.device, dtype=self.dtype)
            + self.low
        )

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log probability of the uniform prior on [low, high]^2."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        inside = torch.all(
            (theta >= self.low) & (theta <= self.high), dim=-1
        )
        log_density = math.log(1.0 / (self.high - self.low) ** self.theta_dim)
        logp = torch.full(
            theta.shape[:-1],
            log_density,
            device=theta.device,
            dtype=theta.dtype,
        )
        return torch.where(
            inside, logp, torch.full_like(logp, -math.inf)
        )

    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Sample x | theta for a batch of parameters.

        Parameters
        ----------
        theta : torch.Tensor
            Parameter vectors of shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Simulated observations of shape ``(batch, 2)``.
        """
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        batch = theta.shape[0]
        a = theta[..., 0]
        b = theta[..., 1]

        alpha = (
            torch.rand(batch, device=theta.device, dtype=theta.dtype) * math.pi
            - math.pi / 2.0
        )
        r = (
            torch.randn(batch, device=theta.device, dtype=theta.dtype)
            * self.radius_std
            + self.radius
        )

        x1 = r * torch.cos(alpha) + 0.25 - torch.abs(a + b) / math.sqrt(2.0)
        x2 = r * torch.sin(alpha) + (-a + b) / math.sqrt(2.0)
        return torch.stack([x1, x2], dim=-1)

    def sample_observation(self, n: int = 1) -> torch.Tensor:
        """Sample observations from the prior predictive distribution."""
        return self.simulator(self.prior_sample(n))

    def reference_posterior_samples(
        self, x_obs: torch.Tensor, n: int
    ) -> Optional[torch.Tensor]:
        """Return reference posterior samples for an observation.

        Uses the sbibm reference posterior when the ``sbibm`` package is
        available. Returns ``None`` otherwise (implicit reference).
        """
        try:
            import sbibm
        except Exception:
            return None

        try:
            x_obs = to_torch(x_obs, device=self.device, dtype=self.dtype)
            observation = x_obs.reshape(1, -1).to(torch.float32)

            task = sbibm.get_task("two_moons")
            posterior = task.get_reference_posterior()
            samples = posterior.sample(
                num_samples=n, observation=observation
            )
            if isinstance(samples, torch.Tensor):
                return samples.to(device=self.device, dtype=self.dtype)
            return to_torch(
                samples, device=self.device, dtype=self.dtype
            )
        except Exception:
            # Observation may not be one of sbibm's stored reference
            # observations; in that case no reference is available.
            return None
