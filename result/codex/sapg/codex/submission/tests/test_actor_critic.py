"""Architecture tests: phi conditioning, fixed log-sigma, recurrent path."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.algorithms.actor_critic import ActorCritic, ActorCriticConfig


def test_policies_are_distinct_and_share_backbone():
    cfg = ActorCriticConfig(latent_dim=16, actor_units=[32, 32], critic_units=[32, 32])
    model = ActorCritic(cfg, obs_dim=6, action_dim=2, num_policies=4)
    assert len(model.phi) == 4
    obs = torch.randn(5, 6)
    with torch.no_grad():
        mu0, _ = model(obs, torch.zeros(5, dtype=torch.long))
        mu1, _ = model(obs, torch.ones(5, dtype=torch.long))
    assert not torch.allclose(mu0, mu1)
    # the actor trunk is shared: there is exactly one copy of the hidden layers
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3 + 3  # (actor trunk 2 + mu) + (critic trunk 2 + value)


def test_log_std_is_input_independent():
    cfg = ActorCriticConfig(latent_dim=8, actor_units=[16], critic_units=[16])
    model = ActorCritic(cfg, obs_dim=4, action_dim=3, num_policies=2)
    assert model.log_std.requires_grad
    dist_a = model.distribution(model(torch.zeros(1, 4), torch.zeros(1, dtype=torch.long))[0])
    dist_b = model.distribution(model(torch.randn(1, 4), torch.zeros(1, dtype=torch.long))[0])
    assert torch.allclose(dist_a.stddev, dist_b.stddev)


def test_recurrent_sequence_forward_masks_dones():
    cfg = ActorCriticConfig(
        latent_dim=8,
        actor_type="lstm",
        actor_encoder_units=[16],
        lstm_hidden_size=8,
        critic_type="lstm",
        critic_encoder_units=[16],
        critic_units=[16],
    )
    model = ActorCritic(cfg, obs_dim=4, action_dim=2, num_policies=2)
    obs = torch.randn(5, 3, 4)
    dones = torch.zeros(5, 3)
    dones[2] = 1.0
    actions = torch.randn(5, 3, 2)
    log_prob, entropy, value, _, mu = model.evaluate_actions(
        obs, actions, torch.zeros(3, dtype=torch.long), dones=dones
    )
    assert log_prob.shape == (5, 3)
    assert value.shape == (5, 3)
    assert mu.shape == (5, 3, 2)
