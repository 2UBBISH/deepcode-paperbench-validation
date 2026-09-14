"""Gaussian Mixture benchmark.

Two-dimensional simulation-based inference benchmark with a uniform prior
and a two-component Gaussian likelihood::

    p(theta) = U(-10, 10)^2
    p(x | theta) = 0.5 * N(x | theta, I) + 0.5 * N(x | theta, 0.01 * I)

The posterior is a two-component truncated-Gaussian mixture on
[-10, 10]^2 and can be sampled exactly.  This makes the benchmark useful
for unit validation of probability-flow sampling and density evaluation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from npse.benchmarks.base import Benchmark, to_torch

__all__ = ["GaussianMixture"]


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF evaluated element-wise."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _truncated_gaussian_mass(
    x: torch.Tensor, sigma: float, low: float, high: float
) -> torch.Tensor:
    """Mass of N(mu=x, sigma^2 I) inside the hypercube [low, high]^d.

    Args:
        x: Observation / truncated-Gaussian mean, shape ``(..., d)``.
        sigma: Isotropic standard deviation.
        low: Lower truncation bound.
        high: Upper truncation bound.

    Returns:
        Tensor of shape ``x.shape[:-1]`` containing, for each row, the
        product over dimensions of ``Phi((high - x)/sigma) - Phi((low - x)/sigma)``.
    """
    mass = _normal_cdf((high - x) / sigma) - _normal_cdf((low - x) / sigma)
    return mass.prod(dim=-1)


def _sample_truncated_normal(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    low: float,
    high: float,
) -> torch.Tensor:
    """Sample from N(mu, sigma^2 I) truncated independently to [low, high]^d.

    Sampling uses the inverse-CDF transform for each dimension.

    Args:
        mu: Mean, shape ``(..., d)``.
        sigma: Per-sample isotropic standard deviation, broadcastable against
            ``mu`` (e.g. shape ``(..., 1)``).
        low: Lower truncation bound.
        high: Upper truncation bound.

    Returns:
        Samples with the same shape as ``mu``.
    """
    a = _normal_cdf((low - mu) / sigma)
    b = _normal_cdf((high - mu) / sigma)
    u = torch.rand_like(a)
    z = a + u * (b - a)
    z = z.clamp(1e-7, 1.0 - 1e-7)
    return mu + sigma * math.sqrt(2.0) * torch.erfinv(2.0 * z - 1.0)


class GaussianMixture(Benchmark):
    """Gaussian Mixture SBI benchmark (two-dimensional)."""

    name = "gaussian_mixture"

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        low: float = -10.0,
        high: float = 10.0,
        wide_var: float = 1.0,
        narrow_var: float = 0.01,
        mixture_weight: float = 0.5,
    ):
        super().__init__(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.theta_dim = 2
        self.x_dim = 2
        self.low = low
        self.high = high
        self.wide_var = wide_var
        self.narrow_var = narrow_var
        self.wide_std = math.sqrt(wide_var)
        self.narrow_std = math.sqrt(narrow_var)
        self.mixture_weight = mixture_weight

    # ------------------------------------------------------------------
    # Prior and simulator
    # ------------------------------------------------------------------
    def prior_sample(self, n: int) -> torch.Tensor:
        """Sample ``theta ~ U(low, high)^2``."""
        return self.low + (self.high - self.low) * torch.rand(
            n, self.theta_dim, device=self.device, dtype=self.dtype
        )

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log density of the uniform prior on ``[low, high]^2``."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        inside = ((theta >= self.low) & (theta <= self.high)).all(dim=-1)
        log_density = -theta.shape[-1] * math.log(self.high - self.low)
        logp = torch.full(
            theta.shape[:-1],
            -float("inf"),
            device=theta.device,
            dtype=theta.dtype,
        )
        return torch.where(
            inside,
            torch.full_like(logp, log_density),
            logp,
        )

    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Sample ``x`` from the two-component Gaussian mixture likelihood."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        batch = theta.shape[0]

        choose_wide = (
            torch.rand(batch, 1, device=self.device, dtype=self.dtype)
            < self.mixture_weight
        )
        wide_std = torch.full(
            (batch, 1), self.wide_std, device=self.device, dtype=self.dtype
        )
        narrow_std = torch.full(
            (batch, 1), self.narrow_std, device=self.device, dtype=self.dtype
        )
        std = torch.where(choose_wide, wide_std, narrow_std)

        noise = torch.randn(
            batch, self.x_dim, device=self.device, dtype=self.dtype
        )
        return theta + std * noise

    # ------------------------------------------------------------------
    # Exact posterior (truncated Gaussian mixture)
    # ------------------------------------------------------------------
    def reference_posterior_samples(
        self, x_obs: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Exact posterior samples for the two-component mixture posterior.

        For a single observation ``x_obs`` of shape ``(x_dim,)`` the result has
        shape ``(n, theta_dim)``.  For a batch ``(batch, x_dim)`` the result
        has shape ``(batch, n, theta_dim)``.
        """
        x_obs = to_torch(x_obs, device=self.device, dtype=self.dtype)
        single = x_obs.dim() == 1
        x = x_obs.unsqueeze(0) if single else x_obs
        batch = x.shape[0]

        # Unnormalized mixture component weights, accounting for prior
        # truncation of each Gaussian component to the hypercube.
        wide_mass = _truncated_gaussian_mass(
            x, self.wide_std, self.low, self.high
        )
        narrow_mass = _truncated_gaussian_mass(
            x, self.narrow_std, self.low, self.high
        )
        w_wide = self.mixture_weight * wide_mass
        w_narrow = (1.0 - self.mixture_weight) * narrow_mass
        total = (w_wide + w_narrow).clamp_min(1e-30)
        p_wide = (w_wide / total).unsqueeze(1).expand(batch, n)

        choose_wide = (
            torch.rand(batch, n, device=self.device, dtype=self.dtype) < p_wide
        )
        sigma = torch.where(
            choose_wide.unsqueeze(-1),
            torch.full((1,), self.wide_std, device=self.device, dtype=self.dtype),
            torch.full((1,), self.narrow_std, device=self.device, dtype=self.dtype),
        )
        mu = x.unsqueeze(1).expand(batch, n, self.x_dim)
        samples = _sample_truncated_normal(mu, sigma, self.low, self.high)

        if single:
            return samples.squeeze(0)
        return samples

    def sample_observation(self, n: int = 1) -> torch.Tensor:
        """Sample an observation from the prior predictive distribution."""
        theta = self.prior_sample(n)
        return self.simulator(theta)
