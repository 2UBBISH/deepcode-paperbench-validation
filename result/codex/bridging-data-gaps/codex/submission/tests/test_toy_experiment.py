"""Fast checks of the Section 5.1 toy utilities."""

from __future__ import annotations

import torch

from dpms_ant.schedules import DiffusionSchedule
from dpms_ant.toy_experiment import (
    ToyConfig,
    ToyDenoiser,
    energy_distance,
    mean_vector,
    projection_direction,
    sample,
    target_mean,
    train_source_model,
)


def test_toy_means_and_projection():
    config = ToyConfig()
    assert torch.allclose(target_mean(config), torch.tensor([-1.0, -1.0]))
    assert torch.allclose(mean_vector(1.0, 3), torch.ones(3))
    assert abs(float(projection_direction(config).norm()) - 1.0) < 1e-6


def test_toy_denoiser_shapes():
    model = ToyDenoiser(hidden=16, depth=1)
    x = torch.randn(5, 2)
    t = torch.randint(0, 100, (5,))
    assert model(x, t).shape == (5, 2)


def test_source_model_learns_a_gaussian_mean():
    """A short run should at least move the generated mean away from 0."""
    torch.manual_seed(0)
    schedule = DiffusionSchedule(num_timesteps=200)
    config = ToyConfig(num_timesteps=200, source_iterations=200, eval_every=10)
    model = train_source_model(schedule, config)
    samples = sample(model, schedule, 128, seed=0)
    assert samples.reshape(-1, 2).mean(dim=0).mean().item() > 0.2


def test_energy_distance_properties():
    a = torch.randn(64, 2)
    assert abs(energy_distance(a, a.clone())) < 1e-5
    assert energy_distance(a, a + 5.0) > energy_distance(a, a + 0.5)
