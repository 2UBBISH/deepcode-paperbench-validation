"""Tests for the GAE / n-step return computation of Sec. 4.1."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.algorithms.storage import compute_gae, compute_nstep_returns


def test_gae_no_dones_matches_geometric_sum():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros(3, 1)
    dones = torch.zeros(3, 1)
    last_value = torch.zeros(1)
    gamma, lam = 0.99, 0.95
    advantages = compute_gae(rewards, values, dones, last_value, gamma, lam)
    # A_t = sum_{k>=t} (gamma*lam)^{k-t} r_k
    a0 = 1.0 + gamma * lam * 2.0 + (gamma * lam) ** 2 * 3.0
    a1 = 2.0 + gamma * lam * 3.0
    a2 = 3.0
    assert torch.allclose(advantages[:, 0], torch.tensor([a0, a1, a2]), atol=1e-6)


def test_gae_stops_at_done():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros(3, 1)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    advantages = compute_gae(rewards, values, dones, torch.zeros(1), 0.99, 0.95)
    # episode ends at t=1, so A_1 = 2 and A_0 = 1 + gamma*lam*(1-done_0)*A_1
    a1 = 2.0
    a0 = 1.0 + 0.99 * 0.95 * a1
    assert torch.allclose(advantages[:, 0], torch.tensor([a0, a1, 3.0]), atol=1e-6)


def test_nstep_return_matches_manual():
    gamma = 0.9
    rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
    dones = torch.zeros(4, 1)
    last_value = torch.tensor([0.0])
    targets = compute_nstep_returns(rewards, values, dones, last_value, gamma, nstep=3)
    # V_3(s_0) = 1 + gamma + gamma^2 + gamma^3 * V_old(s_3)
    expected0 = 1 + gamma + gamma ** 2 + gamma ** 3 * 0.5
    expected1 = 1 + gamma + gamma ** 2 + gamma ** 3 * 0.0
    assert torch.allclose(targets[0, 0], torch.tensor(expected0), atol=1e-6)
    assert torch.allclose(targets[1, 0], torch.tensor(expected1), atol=1e-6)


def test_nstep_return_respects_termination():
    rewards = torch.tensor([[1.0], [5.0], [1.0]])
    values = torch.zeros(3, 1)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    targets = compute_nstep_returns(rewards, values, dones, torch.zeros(1), 0.99, nstep=3)
    # the terminal transition at t=1 contributes its reward but no bootstrap
    assert torch.allclose(targets[0, 0], torch.tensor(1.0 + 0.99 * 5.0), atol=1e-6)
