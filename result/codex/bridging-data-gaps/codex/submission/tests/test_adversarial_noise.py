"""Checks of the adversarial noise selection (Section 4.2, Eqs. 6-7)."""

from __future__ import annotations

import torch
import torch.nn as nn

from dpms_ant.adversarial_noise import (
    AdversarialNoiseConfig,
    adversarial_noise_objective,
    normalize_noise,
    select_adversarial_noise,
)
from dpms_ant.schedules import DiffusionSchedule


class IdentityDenoiser(nn.Module):
    """``eps_theta(x_t, t) = 0``."""

    def forward(self, x, timesteps):
        return torch.zeros_like(x)


class PartiallyCorrectDenoiser(nn.Module):
    """A model whose residual is linear in the injected noise."""

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, x, timesteps):
        return self.weight * x


def test_normalize_noise_has_zero_mean_unit_std():
    noise = torch.randn(16, 3, 8, 8) * 3 + 1.5
    normalized = normalize_noise(noise)
    assert normalized.mean().abs().item() < 1e-5
    per_sample_std = normalized.reshape(16, -1).std(dim=1, unbiased=False)
    assert torch.allclose(per_sample_std, torch.ones(16), atol=1e-4)


def test_global_normalization():
    noise = torch.randn(4, 2) * 5 - 2
    normalized = normalize_noise(noise, mode="global")
    assert normalized.mean().abs().item() < 1e-5
    assert abs(normalized.std(unbiased=False).item() - 1.0) < 1e-4


def test_ascent_increases_the_objective():
    """Without ``Norm(.)`` the plain gradient ascent maximises the objective."""
    torch.manual_seed(0)
    schedule = DiffusionSchedule(num_timesteps=100)
    model = PartiallyCorrectDenoiser()
    x_start = torch.randn(8, 3, 4, 4)
    timesteps = torch.full((8,), 50, dtype=torch.long)
    initial = torch.randn_like(x_start)
    config = AdversarialNoiseConfig(num_steps=10, step_size=0.005, normalize=False)
    worse_case = select_adversarial_noise(
        model, schedule, x_start, timesteps, config=config, initial_noise=initial
    )
    before = adversarial_noise_objective(model, schedule, x_start, timesteps, initial)
    after = adversarial_noise_objective(model, schedule, x_start, timesteps, worse_case)
    assert after > before


def test_normalised_ascent_keeps_the_gaussian_shape():
    """With ``Norm(.)`` the "worse-case" noise stays a standard Gaussian."""
    torch.manual_seed(0)
    schedule = DiffusionSchedule(num_timesteps=100)
    model = PartiallyCorrectDenoiser()
    x_start = torch.randn(8, 3, 4, 4)
    timesteps = torch.full((8,), 50, dtype=torch.long)
    initial = torch.randn_like(x_start)
    worse_case = select_adversarial_noise(
        model,
        schedule,
        x_start,
        timesteps,
        config=AdversarialNoiseConfig(num_steps=10, step_size=0.02, normalize=True),
        initial_noise=initial,
    )
    assert abs(worse_case.mean().item()) < 0.1
    assert abs(worse_case.std().item() - 1.0) < 0.1
    # the perturbation actually moved
    assert not torch.allclose(worse_case, initial)


def test_zero_steps_returns_the_initial_noise():
    schedule = DiffusionSchedule(num_timesteps=100)
    model = IdentityDenoiser()
    x_start = torch.randn(4, 3, 4, 4)
    timesteps = torch.full((4,), 20, dtype=torch.long)
    initial = torch.randn_like(x_start)
    out = select_adversarial_noise(
        model,
        schedule,
        x_start,
        timesteps,
        config=AdversarialNoiseConfig(num_steps=0),
        initial_noise=initial,
    )
    assert torch.allclose(out, initial)


def test_history_is_returned():
    schedule = DiffusionSchedule(num_timesteps=100)
    model = PartiallyCorrectDenoiser()
    x_start = torch.randn(2, 3, 4, 4)
    timesteps = torch.full((2,), 30, dtype=torch.long)
    noise, history = select_adversarial_noise(
        model,
        schedule,
        x_start,
        timesteps,
        config=AdversarialNoiseConfig(num_steps=4),
        return_history=True,
    )
    assert len(history) == 5  # J + 1 iterates
