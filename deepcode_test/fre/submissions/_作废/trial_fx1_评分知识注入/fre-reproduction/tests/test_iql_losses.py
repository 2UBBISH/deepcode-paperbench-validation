"""Unit tests for FRE-conditioned IQL losses.

These tests are intentionally CPU-only and use small networks/batches so they
can run without MuJoCo, D4RL, or ExORL.  They verify the three IQL loss
behaviours called out in the reproduction plan:

* the Q loss decreases on a deterministic batch,
* the expectile loss penalizes underestimation more than overestimation when
  ``expectile > 0.5``,
* the AWR policy loss improves the likelihood of high-advantage actions.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

# Make repository root importable without installation (mirrors other tests).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fre.iql import (  # noqa: E402
    IQLNetworks,
    SquashedGaussianPolicy,
    expectile_loss,
)


def _as_loss_dict(result):
    """Normalize the output of ``compute_losses``/``compute_iql_losses``.

    Different implementations may return a dictionary or a tuple.  This helper
    accepts the common conventions without making tests brittle.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, (tuple, list)):
        keys = ("q_loss", "v_loss", "policy_loss", "total_loss")
        return dict(zip(keys, result))
    raise TypeError(f"Unsupported loss return type: {type(result)!r}")


def _float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _make_small_iql(
    state_dim=5,
    action_dim=3,
    latent_dim=8,
    hidden_dims=(32, 32),
    gamma=0.99,
    expectile=0.9,
    awr_temperature=3.0,
    target_tau=0.005,
    advantage_clip=(-5.0, 2.0),
):
    return IQLNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        gamma=gamma,
        expectile=expectile,
        awr_temperature=awr_temperature,
        target_tau=target_tau,
        advantage_clip=advantage_clip,
    )


def _deterministic_batch(state_dim=5, action_dim=3, latent_dim=8, batch_size=64):
    torch.manual_seed(7)
    states = torch.randn(batch_size, state_dim)
    actions = torch.rand(batch_size, action_dim) * 2.0 - 1.0
    next_states = torch.randn(batch_size, state_dim)
    rewards = torch.randn(batch_size)
    z = torch.randn(batch_size, latent_dim)
    dones = torch.zeros(batch_size)
    return states, actions, rewards, next_states, z, dones


def test_q_loss_decreases_on_deterministic_batch():
    """Train Q networks on a fixed batch and confirm the Q loss decreases."""
    torch.manual_seed(0)
    iql = _make_small_iql()
    batch = _deterministic_batch()
    states, actions, rewards, next_states, z, dones = batch

    q_params = list(iql.q1.parameters()) + list(iql.q2.parameters())
    optimizer = torch.optim.Adam(q_params, lr=3e-3)

    initial = None
    for step in range(15):
        optimizer.zero_grad()
        losses = _as_loss_dict(
            iql.compute_losses(states, actions, rewards, next_states, z, dones)
        )
        q_loss = losses["q_loss"]
        if step == 0:
            initial = _float(q_loss)
        q_loss.backward()
        optimizer.step()

    final = _float(
        _as_loss_dict(
            iql.compute_losses(states, actions, rewards, next_states, z, dones)
        )["q_loss"]
    )

    assert math.isfinite(initial)
    assert math.isfinite(final)
    assert final < initial - 1e-4


def test_v_expectile_loss_penalizes_underestimation_more():
    """For expectile > 0.5, positive residuals (target > V) cost more."""
    tau = 0.9

    # u > 0 means the target exceeds the current value, i.e. underestimation.
    underestimation = torch.tensor([1.0])
    overestimation = torch.tensor([-1.0])

    loss_under = _float(expectile_loss(underestimation, expectile=tau))
    loss_over = _float(expectile_loss(overestimation, expectile=tau))

    assert loss_under > loss_over

    # At tau = 0.5 the loss should be symmetric.
    sym = 0.5
    loss_under_sym = _float(expectile_loss(underestimation, expectile=sym))
    loss_over_sym = _float(expectile_loss(overestimation, expectile=sym))
    assert abs(loss_under_sym - loss_over_sym) < 1e-6


def test_policy_loss_increases_likelihood_of_high_advantage_actions():
    """AWR policy updates should decrease the weighted NLL for fixed actions.

    We keep the Q and V networks fixed, update only the policy, and verify that
    the policy loss decreases on the same batch.  Since the loss is
    ``-E[exp(advantage / temperature) * log pi(a|s,z)]``, a decrease implies a
    higher (advantage-weighted) action likelihood.
    """
    torch.manual_seed(3)
    iql = _make_small_iql()
    batch = _deterministic_batch()
    states, actions, rewards, next_states, z, dones = batch

    policy_params = list(iql.policy.parameters())
    optimizer = torch.optim.Adam(policy_params, lr=3e-3)

    initial = None
    for step in range(10):
        optimizer.zero_grad()
        losses = _as_loss_dict(
            iql.compute_losses(states, actions, rewards, next_states, z, dones)
        )
        policy_loss = losses["policy_loss"]
        if step == 0:
            initial = _float(policy_loss)
        policy_loss.backward()
        optimizer.step()

    final = _float(
        _as_loss_dict(
            iql.compute_losses(states, actions, rewards, next_states, z, dones)
        )["policy_loss"]
    )

    assert math.isfinite(initial)
    assert math.isfinite(final)
    # The policy is being optimized, so its loss should strictly improve on a
    # deterministic batch.  A modest tolerance guards against numerical noise.
    assert final < initial - 1e-4


def test_policy_outputs_are_bounded_after_tanh():
    """Sanity-check that sampled and deterministic actions stay in [-1, 1]."""
    torch.manual_seed(1)
    policy = SquashedGaussianPolicy(
        state_dim=5,
        latent_dim=8,
        action_dim=3,
        hidden_dims=(32, 32),
    )
    states = torch.randn(16, 5)
    z = torch.randn(16, 8)

    deterministic = policy.get_action(states, z, deterministic=True)
    assert isinstance(deterministic, torch.Tensor) or hasattr(deterministic, "__array__")
    det_np = deterministic.cpu().numpy() if torch.is_tensor(deterministic) else deterministic
    assert det_np.min() >= -1.0 - 1e-5
    assert det_np.max() <= 1.0 + 1e-5

    sampled_output = policy.sample(states, z)
    if isinstance(sampled_output, (tuple, list)):
        sampled_action = sampled_output[0]
    else:
        sampled_action = sampled_output
    sampled_np = sampled_action.detach().cpu().numpy()
    assert sampled_np.min() >= -1.0 - 1e-5
    assert sampled_np.max() <= 1.0 + 1e-5
