"""Numerical checks of the ranking based NCE objective (Eqs. 2-3)."""

from __future__ import annotations

import math

import torch

from bbox_adapter.adapter.losses import (
    l2_energy_regularizer,
    ranking_nce_loss,
    ranking_nce_softmax_loss,
)


def test_pairwise_gradient_matches_eq3():
    positive = torch.tensor([0.5, -0.2, 1.3], dtype=torch.float64, requires_grad=True)
    negative = torch.tensor([-0.4, 0.7, 0.1], dtype=torch.float64, requires_grad=True)
    alpha = 0.01
    loss = ranking_nce_loss(positive, negative, alpha=alpha, reduction="sum")
    loss.backward()
    # d/dg+ = -1 + 2 alpha g+   and   d/dg- = 1 + 2 alpha g-
    assert torch.allclose(positive.grad, -1 + 2 * alpha * positive.detach())
    assert torch.allclose(negative.grad, 1 + 2 * alpha * negative.detach())


def test_listwise_gradient_is_posterior_minus_indicator():
    """Appendix B: d l / d g(x_m) = p_theta(x_m) - 1[m == 0]."""

    energies = torch.tensor([[1.0, 0.5, -0.3, 2.0]], dtype=torch.float64, requires_grad=True)
    loss = ranking_nce_softmax_loss(energies, alpha=0.0)
    loss.backward()
    posterior = torch.softmax(energies.detach(), dim=-1)
    indicator = torch.zeros_like(posterior)
    indicator[0, 0] = 1.0
    assert torch.allclose(energies.grad, posterior - indicator, atol=1e-6)


def test_listwise_loss_equals_negative_log_posterior():
    energies = torch.tensor([[0.3, -0.7, 1.1]], dtype=torch.float64)
    expected = -math.log(math.exp(0.3) / sum(math.exp(v) for v in [0.3, -0.7, 1.1]))
    value = ranking_nce_softmax_loss(energies, alpha=0.0).item()
    assert abs(value - expected) < 1e-9


def test_l2_regularizer_penalises_large_energies():
    small = l2_energy_regularizer(torch.zeros(4), alpha=1.0)
    large = l2_energy_regularizer(torch.full((4,), 3.0), alpha=1.0)
    assert small.item() == 0.0
    assert large.item() == 9.0


def test_alpha_penalises_the_positive_energy_too():
    positive = torch.tensor([2.0])
    negative = torch.tensor([0.0])
    with_reg = ranking_nce_loss(positive, negative, alpha=1.0).item()
    without_reg = ranking_nce_loss(positive, negative, alpha=0.0).item()
    assert with_reg > without_reg  # the regulariser keeps the energy bounded
