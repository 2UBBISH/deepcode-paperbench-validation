"""Diffusion noise schedules and the forward (noising) process.

Implements Section 3 (Preliminary) of the paper:

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
    x_t          = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps

with ``alpha_t = 1 - beta_t`` and ``alpha_bar_t = prod_{i<=t} (1 - beta_i)``.

The paper uses the DDPM (Ho et al., 2020) linear schedule
(``beta_start=1e-4``, ``beta_end=2e-2``, ``T=1000``) for the DDPM backbone and
the cosine schedule for the LDM backbone; both are supported here.

This module also provides ``sigma_hat_t``, the time-dependent scaling of the
classifier-gradient term appearing in the similarity-guided loss
(Eq. 5 of the paper):

    sigma_hat_t = (1 - alpha_bar_{t-1}) * sqrt(alpha_t / (1 - alpha_bar_t))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch


def _linear_betas(num_timesteps: int, beta_start: float, beta_end: float) -> np.ndarray:
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)


def _scaled_linear_betas(num_timesteps: int, beta_start: float, beta_end: float) -> np.ndarray:
    """The latent-diffusion schedule: linear in ``sqrt(beta)``."""
    return np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=np.float64) ** 2


def _cosine_betas(num_timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine schedule of Nichol & Dhariwal (2021), as used by the LDM."""
    steps = num_timesteps + 1
    x = np.linspace(0, num_timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0, 0.999)


@dataclass
class DiffusionSchedule:
    """Discrete Gaussian diffusion schedule plus forward-process helpers."""

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "linear"  # {"linear", "scaled_linear", "cosine"}
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schedule == "linear":
            betas = _linear_betas(self.num_timesteps, self.beta_start, self.beta_end)
        elif self.schedule == "scaled_linear":
            betas = _scaled_linear_betas(self.num_timesteps, self.beta_start, self.beta_end)
        elif self.schedule == "cosine":
            betas = _cosine_betas(self.num_timesteps)
        else:
            raise ValueError(f"unknown noise schedule: {self.schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)

        # posterior q(x_{t-1} | x_t, x_0) -- Ho et al. (2020), Eq. (6)-(7)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Eq. 5 of the paper (scaling of the classifier gradient)
        sigma_hat = (1.0 - alphas_cumprod_prev) * np.sqrt(
            alphas / np.maximum(1.0 - alphas_cumprod, 1e-12)
        )
        # sigma_t of Section 3 with eta = 1 (DDPM): sqrt(posterior variance)
        sigma = np.sqrt(posterior_variance)

        def tensor(values: np.ndarray) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32)

        self.betas = tensor(betas)
        self.alphas = tensor(alphas)
        self.alphas_cumprod = tensor(alphas_cumprod)
        self.alphas_cumprod_prev = tensor(alphas_cumprod_prev)
        self.alphas_cumprod_next = tensor(alphas_cumprod_next)
        self.sqrt_alphas_cumprod = tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = tensor(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = tensor(np.log(np.maximum(1.0 - alphas_cumprod, 1e-12)))
        self.sqrt_recip_alphas_cumprod = tensor(np.sqrt(1.0 / np.maximum(alphas_cumprod, 1e-12)))
        self.sqrt_recipm1_alphas_cumprod = tensor(
            np.sqrt(1.0 / np.maximum(alphas_cumprod, 1e-12) - 1.0)
        )
        self.posterior_variance = tensor(posterior_variance)
        self.posterior_log_variance_clipped = tensor(
            np.log(np.maximum(posterior_variance, 1e-20))
        )
        self.posterior_mean_coef1 = tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.sigma_hat = tensor(sigma_hat)
        self.sigma = tensor(sigma)
        self._cache = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, broadcast: torch.Tensor):
        """Gather ``values[t]`` for every ``t`` in ``timesteps`` and broadcast."""
        timesteps = timesteps.to(device=values.device)
        out = values.gather(0, timesteps.long().reshape(-1))
        while out.ndim < broadcast.ndim:
            out = out.unsqueeze(-1)
        return out

    def to(self, device: Union[str, torch.device]) -> "DiffusionSchedule":
        device = torch.device(device)
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        self._cache = {}
        return self

    @property
    def device(self) -> torch.device:
        return self.sqrt_alphas_cumprod.device

    def sigma_hat_t(self, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """``sigma_hat_t`` of Eq. (5), broadcast to the shape of ``like``."""
        return self._extract(self.sigma_hat, timesteps, like)

    def sigma_t(self, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sigma, timesteps, like)

    # ------------------------------------------------------------------ #
    # forward process
    # ------------------------------------------------------------------ #
    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample ``x_t`` from ``q(x_t | x_0)`` (Section 3, Eq. 1)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """The "predicted x_0" of the reverse process (Section 3)."""
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t)
        return sqrt_recip * x_t - sqrt_recipm1 * eps

    def eps_from_x0(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, x_start: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, timesteps, x_t)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t)
        return (x_t - sqrt_alpha_bar * x_start) / sqrt_one_minus

    def q_posterior_mean(
        self, x_start: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        coef1 = self._extract(self.posterior_mean_coef1, timesteps, x_t)
        coef2 = self._extract(self.posterior_mean_coef2, timesteps, x_t)
        return coef1 * x_start + coef2 * x_t


def respaced_timesteps(num_timesteps: int, respacing: Optional[str], device=None) -> torch.Tensor:
    """Sub-sequence of timesteps used for DDIM-style fast sampling.

    ``respacing`` follows the guided-diffusion convention: ``""`` keeps all
    steps, ``"ddim50"`` selects 50 uniformly spaced steps, ``"ddim250"`` etc.
    """
    if not respacing:
        steps = list(range(num_timesteps))
    elif respacing.startswith("ddim"):
        count = int(respacing[len("ddim") :])
        steps = np.linspace(0, num_timesteps - 1, count).round().astype(int).tolist()
    else:
        raise ValueError(f"unsupported respacing string: {respacing}")
    return torch.tensor(steps, dtype=torch.long, device=device)
