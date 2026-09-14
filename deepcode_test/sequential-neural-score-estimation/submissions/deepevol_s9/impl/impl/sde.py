"""Variance-exploding and variance-preserving forward noising SDEs.

This module exposes the transition kernels used to diffuse original parameter
samples and compute denoising score-matching targets. It intentionally keeps
the time convention on :math:`(0, 1]`, with a small positive ``t_min`` used by
the ODE sampler so the forward process never reaches the singular zero-time
point during numerical integration.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from impl.config import SDEConfig


class VESDE:
    """Variance-exploding SDE from equation (134) of the paper.

    The process is drift-free with
    ``dtheta_t = sigma_min * (sigma_min / sigma_max)^t * sqrt(2 * log(sigma_max / sigma_min)) dw_t``.
    Its transition density is ``N(theta_t | theta_0, sigma(t)^2 I)`` with
    ``sigma(t) = sigma_min * (sigma_max / sigma_min)^t``.
    """

    def __init__(self, sigma_min: float = 0.05, sigma_max: float = 1.0):
        if not (0.0 < sigma_min < sigma_max):
            raise ValueError("VESDE requires 0 < sigma_min < sigma_max")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.t_min = 1e-5
        self.t_max = 1.0
        self._log_ratio = math.log(self.sigma_max / self.sigma_min)

    def _time(self, t) -> Tensor:
        """Normalise time to a tensor, converting 1D batch time to 2D batch."""
        t = torch.as_tensor(t, dtype=torch.float32)
        if t.ndim == 1:
            return t.unsqueeze(-1)
        return t

    def std(self, t) -> Tensor:
        """Return the transition standard deviation ``sigma(t)``."""
        t = self._time(t)
        return self.sigma_min * torch.exp(self._log_ratio * t)

    def log_std(self, t) -> Tensor:
        """Return ``log sigma(t)``."""
        return torch.log(self.std(t))

    def mean(self, theta_0: Tensor, t) -> Tensor:
        """Return the transition mean; VE has identity drift so mean is theta_0."""
        return theta_0

    def score_target(self, theta_t: Tensor, theta_0: Tensor, t) -> Tensor:
        """Return the conditional transition score target."""
        t = self._time(t)
        var = self.std(t) ** 2
        return -(theta_t - self.mean(theta_0, t)) / var

    def f(self, theta_t: Tensor, t) -> Tensor:
        """Drift coefficient of the forward SDE (zero for VE)."""
        return torch.zeros_like(theta_t)

    def g(self, t) -> Tensor:
        """Diffusion coefficient of the forward SDE."""
        t = self._time(t)
        sigma = self.sigma_min * torch.exp(self._log_ratio * t)
        return sigma * math.sqrt(2.0 * self._log_ratio)

    def drift(self, theta_t: Tensor, t) -> Tensor:
        return self.f(theta_t, t)

    def diffusion(self, t) -> Tensor:
        return self.g(t)

    def sample_transition(self, theta_0: Tensor, t, noise: Tensor | None = None) -> Tensor:
        """Draw ``theta_t`` from ``p_{t|0}(theta_t | theta_0)``."""
        t = self._time(t)
        if noise is None:
            noise = torch.randn_like(theta_0)
        return self.mean(theta_0, t) + self.std(t) * noise

    @classmethod
    def from_config(cls, config: SDEConfig) -> "VESDE":
        if config.kind != "ve":
            raise ValueError(f"expected kind='ve', got {config.kind!r}")
        return cls(config.sigma_min, config.sigma_max)


class VPSDE:
    """Variance-preserving SDE from equation (136) of the paper.

    With ``beta_t = beta_min + t * (beta_max - beta_min)``, the process is
    ``dtheta_t = -1/2 beta_t theta_t dt + sqrt(beta_t) dw_t``. Its transition
    density is ``N(theta_t | m(t) theta_0, (1 - m(t)^2) I)``, where
    ``m(t) = exp(-1/2 integral_0^t beta_s ds)``.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 11.0):
        if not (0.0 < beta_min <= beta_max):
            raise ValueError("VPSDE requires 0 < beta_min <= beta_max")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.t_min = 1e-5
        self.t_max = 1.0

    def _time(self, t) -> Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        if t.ndim == 1:
            return t.unsqueeze(-1)
        return t

    def _beta(self, t: Tensor) -> Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def _beta_integral(self, t: Tensor) -> Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t * t

    def std(self, t) -> Tensor:
        t = self._time(t)
        return torch.sqrt(1.0 - torch.exp(-self._beta_integral(t)))

    def log_std(self, t) -> Tensor:
        return torch.log(self.std(t))

    def mean(self, theta_0: Tensor, t) -> Tensor:
        t = self._time(t)
        coefficient = torch.exp(-0.5 * self._beta_integral(t))
        return theta_0 * coefficient

    def score_target(self, theta_t: Tensor, theta_0: Tensor, t) -> Tensor:
        t = self._time(t)
        mean = self.mean(theta_0, t)
        var = self.std(t) ** 2
        return -(theta_t - mean) / var

    def f(self, theta_t: Tensor, t) -> Tensor:
        """Drift coefficient of the forward SDE."""
        t = self._time(t)
        beta = self._beta(t)
        return -0.5 * beta * theta_t

    def g(self, t) -> Tensor:
        """Diffusion coefficient of the forward SDE."""
        return torch.sqrt(self._beta(self._time(t)))

    def drift(self, theta_t: Tensor, t) -> Tensor:
        return self.f(theta_t, t)

    def diffusion(self, t) -> Tensor:
        return self.g(t)

    def sample_transition(self, theta_0: Tensor, t, noise: Tensor | None = None) -> Tensor:
        """Draw ``theta_t`` from ``p_{t|0}(theta_t | theta_0)``."""
        t = self._time(t)
        if noise is None:
            noise = torch.randn_like(theta_0)
        return self.mean(theta_0, t) + self.std(t) * noise

    @classmethod
    def from_config(cls, config: SDEConfig) -> "VPSDE":
        if config.kind != "vp":
            raise ValueError(f"expected kind='vp', got {config.kind!r}")
        return cls(config.beta_min, config.beta_max)


def build_sde(config: SDEConfig):
    """Return the SDE object described by ``config``."""
    if config.kind == "ve":
        return VESDE.from_config(config)
    if config.kind == "vp":
        return VPSDE.from_config(config)
    raise ValueError(f"unknown SDE kind {config.kind!r}")


def choose_sigma_max(data: Tensor) -> Tensor:
    """Approximate Technique 1 from Song & Ermon (2020).

    This helper computes the maximum Euclidean distance between any two rows in
    a small reference batch. It returns a scalar so the VE schedule is globally
    scaled. For large datasets, callers may choose to cap the batch before
    calling this function.
    """
    if data.numel() == 0:
        return torch.tensor(1.0, dtype=data.dtype, device=data.device)
    sample = data[:1024] if data.ndim >= 2 else data
    pairwise = torch.cdist(sample, sample)
    return pairwise.max()
