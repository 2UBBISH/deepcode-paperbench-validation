"""Checks of the similarity-guided loss (Section 4.1, Eq. 5)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dpms_ant.guidance import classifier_guidance, similarity_guided_loss, vanilla_ddpm_loss
from dpms_ant.schedules import DiffusionSchedule


class LinearClassifier(nn.Module):
    """``p(y | x_t)`` linear in ``x_t`` (so its gradient is analytic)."""

    def __init__(self, dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(dim, 2, bias=False)

    def forward(self, x, timesteps=None):
        return self.linear(x)


def test_guidance_equals_scaled_classifier_gradient():
    torch.manual_seed(0)
    schedule = DiffusionSchedule(num_timesteps=100)
    classifier = LinearClassifier(2)
    x_t = torch.randn(8, 2)
    t = torch.full((8,), 40, dtype=torch.long)
    gamma = 5.0

    guidance = classifier_guidance(classifier, schedule, x_t, t, target_class=1, gamma=gamma)

    # sigma_hat^2 * gamma * grad log p(y=T | x_t); for a linear classifier the
    # gradient of the log-softmax is W[T] - sum_j softmax_j * W[j]
    sigma_hat = schedule.sigma_hat_t(t, x_t)
    logits = classifier.linear(x_t)
    probabilities = logits.softmax(dim=-1)
    weighted = torch.einsum("nk,kd->nd", probabilities, classifier.linear.weight)
    expected = gamma * sigma_hat ** 2 * (classifier.linear.weight[1].reshape(1, 2) - weighted)
    assert torch.allclose(guidance, expected, atol=1e-5)


def test_guidance_points_towards_the_target_class():
    torch.manual_seed(0)
    schedule = DiffusionSchedule(num_timesteps=100)
    classifier = LinearClassifier(2)
    with torch.no_grad():
        classifier.linear.weight[1] = torch.tensor([1.0, 0.0])
        classifier.linear.weight[0] = torch.tensor([-1.0, 0.0])
    x_t = torch.zeros(1, 2)
    t = torch.full((1,), 30, dtype=torch.long)
    guidance = classifier_guidance(classifier, schedule, x_t, t, target_class=1, gamma=1.0)
    assert guidance[0, 0] > 0


def test_similarity_guided_loss_matches_manual_formula():
    torch.manual_seed(0)
    eps_pred = torch.randn(4, 2)
    eps_target = torch.randn(4, 2)
    guidance = torch.randn(4, 2)
    loss = similarity_guided_loss(eps_pred, eps_target, guidance)
    residual = eps_target - eps_pred - guidance
    assert torch.allclose(loss, residual.pow(2).mean())


def test_vanilla_loss_is_plain_mse():
    eps_pred = torch.randn(4, 2)
    eps_target = torch.randn(4, 2)
    assert torch.allclose(
        vanilla_ddpm_loss(eps_pred, eps_target), (eps_target - eps_pred).pow(2).mean()
    )


def test_guidance_scale_flags():
    torch.manual_seed(0)
    schedule = DiffusionSchedule(num_timesteps=100)
    classifier = LinearClassifier(2)
    x_t = torch.randn(4, 2)
    t = torch.full((4,), 50, dtype=torch.long)
    raw = classifier_guidance(classifier, schedule, x_t, t, gamma=1.0)
    scaled = classifier_guidance(classifier, schedule, x_t, t, gamma=3.0)
    assert torch.allclose(scaled, 3.0 * raw, atol=1e-6)
    # `target_rms` rescales the correction to the requested magnitude
    rescaled = classifier_guidance(classifier, schedule, x_t, t, target_rms=0.25)
    assert rescaled.pow(2).mean().sqrt().item() == pytest.approx(0.25, rel=1e-4)
