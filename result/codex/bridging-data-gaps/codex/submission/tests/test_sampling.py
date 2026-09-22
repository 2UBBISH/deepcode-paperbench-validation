"""Checks of the reverse process."""

from __future__ import annotations

import torch
import torch.nn as nn

from dpms_ant.sampling import ddpm_step, sample_loop
from dpms_ant.schedules import DiffusionSchedule, respaced_timesteps


class MeanDenoiser(nn.Module):
    """Predicts the noise that would take ``x_t`` back to the origin."""

    def __init__(self, schedule: DiffusionSchedule):
        super().__init__()
        self.schedule = schedule

    def forward(self, x, timesteps):
        return self.schedule.eps_from_x0(x, timesteps, torch.zeros_like(x))


def test_sampling_shapes():
    schedule = DiffusionSchedule(num_timesteps=50)
    model = MeanDenoiser(schedule)
    samples = sample_loop(schedule, (4, 3, 8, 8), model, device="cpu")
    assert samples.shape == (4, 3, 8, 8)


def test_ddim_sampling_with_respacing():
    schedule = DiffusionSchedule(num_timesteps=100)
    model = MeanDenoiser(schedule)
    steps = torch.flip(respaced_timesteps(100, "ddim10"), dims=[0])
    samples = sample_loop(
        schedule, (2, 3, 8, 8), model, steps=steps, eta=0.0, device="cpu"
    )
    assert samples.shape == (2, 3, 8, 8)


def test_ddpm_step_clipping_flag():
    schedule = DiffusionSchedule(num_timesteps=100)
    x_t = torch.full((1, 3, 4, 4), 10.0)
    eps = torch.zeros_like(x_t)
    t = torch.tensor([50])
    clipped = ddpm_step(schedule, x_t, t, eps, eta=0.0, clip_denoised=True)
    unclipped = ddpm_step(schedule, x_t, t, eps, eta=0.0, clip_denoised=False)
    assert clipped.abs().max() <= unclipped.abs().max()

