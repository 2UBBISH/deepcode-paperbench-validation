"""Tests for the off-policy (importance-sampled) update of Sec. 4.1."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.algorithms.actor_critic import ActorCritic, ActorCriticConfig
from sapg.algorithms.losses import compute_off_policy_loss, compute_on_policy_loss


def make_model(obs_dim=4, action_dim=2, latent=8, num_policies=3, seed=0):
    torch.manual_seed(seed)
    cfg = ActorCriticConfig(
        latent_dim=latent,
        actor_type="mlp",
        actor_units=[16, 16],
        critic_type="mlp",
        critic_units=[16, 16],
    )
    return ActorCritic(cfg, obs_dim=obs_dim, action_dim=action_dim, num_policies=num_policies)


def make_batch(model, batch_size=64, obs_dim=4, action_dim=2, policy_id=0, seed=0):
    torch.manual_seed(seed)
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)
    policy_ids = torch.full((batch_size,), policy_id, dtype=torch.long)
    with torch.no_grad():
        logprob, _, value, _, _ = model.evaluate_actions(obs, actions, policy_ids)
    advantages = torch.randn(batch_size)
    value_targets = value + 0.1 * torch.randn(batch_size)
    return {
        "obs": obs,
        "actions": actions,
        "old_logprobs": logprob,
        "values": value,
        "policy_ids": policy_ids,
        "advantages": advantages,
        "value_targets": value_targets,
    }


def test_offpolicy_reduces_to_onpolicy_for_own_data():
    """With j == i, mu = pi_{i,old}/pi_j = 1 and the off-policy loss is PPO."""
    model = make_model()
    batch = make_batch(model)
    on_policy = compute_on_policy_loss(
        model,
        obs=batch["obs"],
        actions=batch["actions"],
        old_logprobs=batch["old_logprobs"],
        advantages=batch["advantages"],
        value_targets=batch["value_targets"],
        policy_ids=batch["policy_ids"],
        clip_epsilon=0.1,
    )
    off_policy = compute_off_policy_loss(
        model,
        obs=batch["obs"],
        actions=batch["actions"],
        behavior_logprobs=batch["old_logprobs"],  # data was produced by pi_{i,old}
        mu_weights=torch.ones_like(batch["old_logprobs"]),
        advantages=batch["advantages"],
        value_targets=batch["value_targets"],
        policy_ids=batch["policy_ids"],
        clip_epsilon=0.1,
    )
    assert torch.allclose(on_policy["loss"], off_policy["loss"], atol=1e-6)


def test_behavior_policy_receives_no_gradient():
    """The data-collecting follower pi_j is a constant: only pi_i is differentiated."""
    model = make_model(num_policies=3)
    batch = make_batch(model, policy_id=0)  # data is used to update policy i = 0
    behavior = batch["old_logprobs"] - 0.5  # pretend it came from follower j = 2
    mu = torch.exp(batch["old_logprobs"] - behavior)
    model.zero_grad(set_to_none=True)
    loss = compute_off_policy_loss(
        model,
        obs=batch["obs"],
        actions=batch["actions"],
        behavior_logprobs=behavior,
        mu_weights=mu,
        advantages=batch["advantages"],
        value_targets=batch["value_targets"],
        policy_ids=batch["policy_ids"],
        clip_epsilon=0.1,
    )["loss"]
    loss.backward()
    assert model.phi.grad[0].abs().sum() > 0, "pi_i must be updated"
    assert model.phi.grad[2].abs().sum() == 0, "pi_j is a constant"


def test_strictly_clipped_branch_has_no_policy_gradient():
    """Once r_{pi_i} is clipped, the surrogate is flat in pi_i's parameters."""
    model = make_model(num_policies=2)
    batch = make_batch(model, batch_size=32)
    # ratio = exp(+2) = 7.39 while mu = 0.2 -> upper bound = 0.22 < ratio, so
    # the clipped branch is selected and its gradient w.r.t. pi_i vanishes.
    behavior = batch["old_logprobs"] - 2.0
    mu = torch.full_like(batch["old_logprobs"], 0.2)
    model.zero_grad(set_to_none=True)
    loss = compute_off_policy_loss(
        model,
        obs=batch["obs"],
        actions=batch["actions"],
        behavior_logprobs=behavior,
        mu_weights=mu,
        advantages=torch.ones_like(batch["advantages"]),
        value_targets=batch["value_targets"],
        policy_ids=batch["policy_ids"],
        clip_epsilon=0.1,
        critic_coef=0.0,  # isolate the policy term
    )["loss"]
    loss.backward()
    policy_params = [p for p in model.actor.parameters()]
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in policy_params)
    assert torch.allclose(loss, -torch.tensor(0.2 * 1.1), atol=1e-5)


def test_phi_gradients_are_local_to_their_policy():
    """Sec. 4.4: phi_j only receives gradients from policy j's objective."""
    model = make_model(num_policies=3)
    batch = make_batch(model, policy_id=1)
    model.zero_grad(set_to_none=True)
    loss = compute_on_policy_loss(
        model,
        obs=batch["obs"],
        actions=batch["actions"],
        old_logprobs=batch["old_logprobs"],
        advantages=batch["advantages"],
        value_targets=batch["value_targets"],
        policy_ids=batch["policy_ids"],
        clip_epsilon=0.1,
    )["loss"]
    loss.backward()
    assert model.phi.grad[1].abs().sum() > 0
    assert model.phi.grad[0].abs().sum() == 0
    assert model.phi.grad[2].abs().sum() == 0


def test_positive_advantage_is_clipped_at_the_mu_upper_bound():
    """A very stale sample is clipped at ``mu(1+eps)`` instead of blowing up."""
    model = make_model()
    batch = make_batch(model)
    loss = compute_off_policy_loss(
        model,
        obs=batch["obs"],
        actions=batch["actions"],
        behavior_logprobs=batch["old_logprobs"] - 10.0,  # ratio = e^10 in favour of pi_i
        mu_weights=torch.full_like(batch["old_logprobs"], 50.0),
        advantages=torch.ones_like(batch["advantages"]),
        value_targets=batch["value_targets"],
        policy_ids=batch["policy_ids"],
        clip_epsilon=0.1,
        critic_coef=0.0,
    )
    assert torch.isfinite(loss["loss"])
    assert torch.allclose(loss["policy_loss"], -torch.tensor(55.0), atol=1e-4)


def test_negative_advantage_is_not_clipped_above():
    """PPO's pessimistic min: when A < 0 the unclipped branch is selected.

    This is the standard clipped-surrogate behaviour: the update is only
    prevented from *increasing* the probability of bad actions, it is still
    allowed to push their probability down.
    """
    model = make_model()
    batch = make_batch(model)
    loss = compute_off_policy_loss(
        model,
        obs=batch["obs"],
        actions=batch["actions"],
        behavior_logprobs=batch["old_logprobs"],
        mu_weights=torch.ones_like(batch["old_logprobs"]),
        advantages=-torch.ones_like(batch["advantages"]),
        value_targets=batch["value_targets"],
        policy_ids=batch["policy_ids"],
        clip_epsilon=0.1,
        critic_coef=0.0,
    )
    # ratio == 1 for fresh data: surrogate = -1 -> loss = +1
    assert torch.allclose(loss["policy_loss"], torch.tensor(1.0), atol=1e-4)


def test_clipping_bounds_follow_mu():
    """The clipping range is mu(1-eps) .. mu(1+eps) -- it *scales* with mu.

    ``mu = pi_{i,old}/pi_j`` and ``r = pi_i/pi_j`` coincide at the start of an
    update, so a stale (large-mu) transition is only clipped once the policy has
    moved within the update.  Here the policy is deliberately pushed towards the
    data so that ``r > mu(1+eps)`` and the surrogate must saturate at
    ``mu(1+eps) = 5.445`` rather than at ``1+eps = 1.1``.
    """
    model = make_model()
    batch = make_batch(model, batch_size=128)
    c = 1.6
    behavior = batch["old_logprobs"] - c          # mu = e^c = 4.95
    mu_value = float(torch.exp(torch.tensor(c)))
    mu = torch.full_like(batch["old_logprobs"], mu_value)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    for _ in range(40):  # push the policy towards the batch actions
        log_prob, _, _, _, _ = model.evaluate_actions(
            batch["obs"], batch["actions"], batch["policy_ids"]
        )
        loss = -log_prob.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        log_prob, _, _, _, _ = model.evaluate_actions(
            batch["obs"], batch["actions"], batch["policy_ids"]
        )
    ratio = float(torch.exp(log_prob - behavior).mean())
    assert ratio > mu_value * 1.1, "the test needs r > mu(1+eps)"

    loss = compute_off_policy_loss(
        model,
        obs=batch["obs"],
        actions=batch["actions"],
        behavior_logprobs=behavior,
        mu_weights=mu,
        advantages=torch.ones_like(batch["advantages"]),
        value_targets=batch["value_targets"],
        policy_ids=batch["policy_ids"],
        clip_epsilon=0.1,
        critic_coef=0.0,
    )
    with torch.no_grad():
        ratio = torch.exp(log_prob - behavior)
        expected = -torch.minimum(ratio, torch.clamp(ratio, mu * 0.9, mu * 1.1)).mean()
        isotropic = -torch.minimum(ratio, torch.clamp(ratio, 0.9, 1.1)).mean()
    assert torch.allclose(loss["policy_loss"], expected, atol=1e-4)
    assert not torch.allclose(loss["policy_loss"], isotropic, atol=1e-2)
