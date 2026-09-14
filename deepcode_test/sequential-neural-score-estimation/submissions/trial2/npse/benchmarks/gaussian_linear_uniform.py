"""Gaussian Linear Uniform benchmark.

Specification from the paper:

    theta in R^10,  p(theta) = U(-1, 1)^10
    p(x | theta) = N(theta, 0.1 * I),   x in R^10

This is a benchmark where the prior is uniform and the likelihood is
isotropic Gaussian with identity mean. The exact posterior is a truncated
multivariate Gaussian restricted to the prior hypercube; we therefore rely
on sbibm reference posterior samples when available.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from npse.benchmarks.base import Benchmark, to_torch

__all__ = ["GaussianLinearUniform"]


class GaussianLinearUniform(Benchmark):
    """Gaussian linear simulator with a uniform prior."""

    name = "gaussian_linear_uniform"

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        dim: int = 10,
        low: float = -1.0,
        high: float = 1.0,
        likelihood_var: float = 0.1,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.dim = int(dim)
        self.low = float(low)
        self.high = float(high)
        self.likelihood_var = float(likelihood_var)
        self.likelihood_std = math.sqrt(self.likelihood_var)

        self.theta_dim = self.dim
        self.x_dim = self.dim

    def prior_sample(self, n: int) -> torch.Tensor:
        """Sample theta ~ U(low, high)^dim."""
        shape = (int(n), self.dim)
        return torch.empty(shape, device=self.device, dtype=self.dtype).uniform_(
            self.low, self.high
        )

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log density of U(low, high)^dim (constant inside the hypercube)."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        inside = (
            (theta >= self.low) & (theta <= self.high)
        ).all(dim=-1)
        log_norm = -float(self.dim) * math.log(self.high - self.low)
        return torch.where(
            inside,
            torch.full_like(inside, log_norm, dtype=self.dtype),
            torch.full_like(inside, -math.inf, dtype=self.dtype),
        )

    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Sample x ~ N(theta, likelihood_var * I)."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        noise = torch.randn_like(theta) * self.likelihood_std
        return theta + noise

    def sample_observation(self, n: int = 1) -> torch.Tensor:
        """Sample from the prior predictive distribution."""
        theta = self.prior_sample(n)
        return self.simulator(theta)

    def reference_posterior_samples(
        self, x_obs: torch.Tensor, n: int
    ) -> Optional[torch.Tensor]:
        """Return sbibm reference posterior samples if available.

        The exact posterior is a truncated Gaussian and cannot be sampled in
        closed form, so we delegate to the sbibm reference implementation.
        """
        try:
            from sbibm.tasks import get_task
        except Exception:
            return None

        x_obs = to_torch(x_obs, device=self.device, dtype=self.dtype)
        if x_obs.dim() == 1:
            observations = [x_obs]
            single = True
        else:
            observations = [x_obs[i] for i in range(x_obs.shape[0])]
            single = False

        samples_list = []
        try:
            task = get_task("gaussian_linear_uniform")
        except Exception:
            return None

        for obs in observations:
            try:
                ref = task.get_reference_posterior_samples(
                    num_observation=int(n), observation=obs.cpu().numpy()
                )
            except Exception:
                return None
            if ref is None:
                return None
            samples_list.append(to_torch(ref, device=self.device, dtype=self.dtype))

        if single:
            return samples_list[0]
        return torch.stack(samples_list, dim=0)
