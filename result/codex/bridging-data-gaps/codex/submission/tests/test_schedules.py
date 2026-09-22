"""Checks of the diffusion schedule and of the forward process (Section 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dpms_ant.schedules import DiffusionSchedule, respaced_timesteps


def test_alpha_bars_are_consistent():
    schedule = DiffusionSchedule(num_timesteps=1000)
    alphas = schedule.alphas.numpy()
    alpha_bar = np.cumprod(alphas)
    assert np.allclose(alpha_bar, schedule.alphas_cumprod.numpy(), atol=1e-6)
    assert np.allclose(schedule.alphas_cumprod_prev.numpy()[1:], alpha_bar[:-1], atol=1e-6)
    assert schedule.alphas_cumprod_prev[0] == 1.0


def test_sigma_hat_matches_the_paper_formula():
    schedule = DiffusionSchedule(num_timesteps=1000)
    alpha_bar = schedule.alphas_cumprod
    expected = (1 - schedule.alphas_cumprod_prev) * torch.sqrt(
        schedule.alphas / (1 - alpha_bar)
    )
    assert torch.allclose(schedule.sigma_hat, expected, atol=1e-4)
    assert schedule.sigma_hat[0].item() == pytest.approx(0.0, abs=1e-7)
    assert schedule.sigma_hat[-1] > schedule.sigma_hat[0]


def test_q_sample_and_predict_x0_round_trip():
    schedule = DiffusionSchedule(num_timesteps=100)
    x0 = torch.randn(4, 3, 8, 8)
    t = torch.tensor([1, 17, 55, 99])
    noise = torch.randn_like(x0)
    x_t = schedule.q_sample(x0, t, noise)
    recovered = schedule.predict_x0_from_eps(x_t, t, noise)
    assert torch.allclose(recovered, x0, atol=1e-4)


def test_eps_from_x0_inverts_q_sample():
    schedule = DiffusionSchedule(num_timesteps=100)
    x0 = torch.randn(2, 3, 4, 4)
    t = torch.tensor([3, 70])
    noise = torch.randn_like(x0)
    x_t = schedule.q_sample(x0, t, noise)
    assert torch.allclose(schedule.eps_from_x0(x_t, t, x0), noise, atol=1e-4)


def test_cosine_and_scaled_linear_schedules():
    cosine = DiffusionSchedule(num_timesteps=1000, schedule="cosine")
    assert 0.0 < cosine.alphas_cumprod[-1] < 0.1
    scaled = DiffusionSchedule(
        num_timesteps=1000, schedule="scaled_linear", beta_start=0.0015, beta_end=0.0195
    )
    assert scaled.betas[0] == pytest.approx(0.0015, rel=1e-6)
    assert scaled.betas[-1] == pytest.approx(0.0195, rel=1e-6)


def test_respacing():
    steps = respaced_timesteps(1000, "ddim50")
    assert len(steps) == 50
    assert steps[0] == 0 and steps[-1] == 999
    assert len(respaced_timesteps(1000, "")) == 1000
