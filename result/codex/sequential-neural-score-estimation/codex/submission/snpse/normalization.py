"""Standardization of parameters and observations (Appendix E.3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Standardizer:
    """Per-dimension centering and scaling."""

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-8) -> "Standardizer":
        x = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        std = torch.where(std < eps, torch.ones_like(std), std)
        return cls(mean=mean, std=std, eps=eps)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def log_abs_det_jacobian(self) -> torch.Tensor:
        """log |det dz/dtheta| = -sum log std."""
        return -torch.log(self.std).sum()

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Standardizer":
        return cls(mean=d["mean"], std=d["std"])


class StandardizedDistribution:
    """A torch distribution pushed forward by an affine standardization.

    Used to represent the prior ``p(theta)`` in the standardized parameter
    space ``z = (theta - mu) / sigma`` in which the diffusion process runs.
    Densities are returned *per parameter sample* (i.e. already summed over
    dimensions) and include the change-of-variables term, so that
    ``log_prob_z(z) == log p(theta(z)) + log|det dtheta/dz|``.
    """

    def __init__(self, distribution, standardizer: Standardizer) -> None:
        self.distribution = distribution
        self.standardizer = standardizer

    def sample(self, num_samples: int, device: Optional[str] = None) -> torch.Tensor:
        theta = self.distribution.sample((num_samples,))
        if device is not None:
            theta = theta.to(device)
        return self.standardizer.transform(theta)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        theta = self.standardizer.inverse(z)
        lp = self.distribution.log_prob(theta)
        if lp.dim() > 1:
            lp = lp.sum(-1)
        return lp + self.standardizer.log_abs_det_jacobian().to(lp.device)

    def rsample(self, num_samples: int, device: Optional[str] = None) -> torch.Tensor:
        theta = self.distribution.rsample((num_samples,))
        if device is not None:
            theta = theta.to(device)
        return self.standardizer.transform(theta)


class BoxUniformStandardized(StandardizedDistribution):
    """Uniform prior on a box, standardized (used when the task only exposes
    the prior bounds)."""

    def __init__(self, low: torch.Tensor, high: torch.Tensor, standardizer: Standardizer) -> None:
        import torch.distributions as D

        super().__init__(D.Uniform(low, high), standardizer)
        self.low = low
        self.high = high
