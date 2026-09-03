"""Unit tests for the zero-shot offline RL baselines.

These tests use small synthetic batches so they can run quickly on CPU without
MuJoCo, D4RL, or ExORL being installed.  They verify the main baseline classes
are importable, their training steps produce finite losses, and their policies
return valid, bounded actions.
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest
import torch

# Make repository root importable (e.g. when running pytest from a different CWD).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.fb import FB
from baselines.gc_bc import GCBC
from baselines.gc_iql import GCIQL
from baselines.opal import OPAL
from baselines.sf import SF


def _synthetic_batch(
    state_dim: int = 11,
    action_dim: int = 3,
    batch_size: int = 32,
    seed: int = 0,
):
    """Build a deterministic synthetic transition batch."""
    torch.manual_seed(seed)
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim).clamp(-1.0, 1.0)
    next_states = torch.randn(batch_size, state_dim)
    rewards = torch.randn(batch_size, 1)
    dones = torch.zeros(batch_size, 1)
    return states, actions, next_states, rewards, dones


def _assert_finite_metrics(metrics):
    """All returned metric values must be real and finite."""
    assert metrics, "training step returned no metrics"
    assert isinstance(metrics, dict), "training step should return a dict of metrics"
    for key, value in metrics.items():
        assert np.isfinite(float(value)), f"metric {key} is not finite: {value}"


def _try_train_step(baseline, states, actions, next_states, dones, rewards):
    """Call a baseline `train_step` method without knowing its exact signature.

    Baseline files intentionally expose slightly different training-step
    signatures.  This helper tries the most common positional/packed conventions
    and returns the first one that does not raise a signature `TypeError`.
    """
    method = baseline.train_step
    params = inspect.signature(method).parameters

    # If a single packed argument is expected, pass a dict containing all fields.
    packed = {
        "states": states,
        "actions": actions,
        "next_states": next_states,
        "dones": dones,
        "rewards": rewards,
        "terminals": dones,
    }

    attempts = [
        lambda: method(states, actions, next_states, dones),
        lambda: method(states, actions, rewards, next_states, dones),
        lambda: method(states, actions, next_states, rewards, dones),
        lambda: method(packed),
    ]

    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:  # signature mismatch
            last_error = exc
            continue
    raise last_error if last_error is not None else RuntimeError("no training convention worked")


def test_fb_train_step_no_nan():
    """FB representation + actor-critic updates stay finite on small batches."""
    state_dim, action_dim, batch_size = 11, 3, 32
    states, actions, next_states, rewards, dones = _synthetic_batch(
        state_dim, action_dim, batch_size
    )

    fb = FB(
        state_dim=state_dim,
        action_dim=action_dim,
        repr_dim=32,
        hidden_dims=(64, 64),
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=batch_size,
        device="cpu",
    )

    for _ in range(3):
        metrics = _try_train_step(fb, states, actions, next_states, dones, rewards)
        _assert_finite_metrics(metrics)


def test_sf_train_step_no_nan():
    """SF ICM/feature/TD updates stay finite on small batches."""
    state_dim, action_dim, batch_size = 11, 3, 32
    states, actions, next_states, rewards, dones = _synthetic_batch(
        state_dim, action_dim, batch_size
    )

    sf = SF(
        state_dim=state_dim,
        action_dim=action_dim,
        feature_dim=32,
        hidden_dims=(64, 64),
        icm_lr=1e-3,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=batch_size,
        device="cpu",
    )

    for _ in range(3):
        metrics = _try_train_step(sf, states, actions, next_states, dones, rewards)
        _assert_finite_metrics(metrics)


def test_gc_iql_returns_valid_actions_and_trains():
    """GC-IQL trains and its goal-conditioned policy returns bounded actions."""
    state_dim, action_dim, batch_size = 11, 3, 32
    states, actions, next_states, _, dones = _synthetic_batch(
        state_dim, action_dim, batch_size
    )

    gc_iql = GCIQL(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=state_dim,
        hidden_dims=(64, 64),
        lr=1e-3,
        gamma=0.99,
        expectile=0.9,
        awr_temperature=3.0,
        target_tau=0.005,
        batch_size=batch_size,
        device="cpu",
    )

    goal = torch.zeros(state_dim)
    goal[:2] = 1.0

    metrics = gc_iql.train_step(states, actions, next_states, dones, goals=goal.expand(batch_size, -1))
    _assert_finite_metrics(metrics)

    policy_fn = gc_iql.get_task_policy(goal)
    obs = np.random.RandomState(0).randn(state_dim).astype(np.float32)
    action = policy_fn(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (action_dim,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0 + 1e-6)


def test_gc_bc_returns_valid_actions_and_improves():
    """GC-BC trains on a fixed batch and returns bounded goal-conditioned actions."""
    state_dim, action_dim, batch_size = 32, 3, 32
    states, actions, next_states, _, dones = _synthetic_batch(
        state_dim, action_dim, batch_size
    )

    gc_bc = GCBC(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=state_dim,
        hidden_dims=(64, 64),
        lr=3e-3,
        batch_size=batch_size,
        device="cpu",
    )

    # A fixed batch repeated several times should reduce the BC negative
    # log-likelihood reliably; if it merely stays finite, that is acceptable
    # for a smoke test, but we assert that it decreases.
    losses = []
    for _ in range(50):
        metrics = gc_bc.train_step(states, actions, next_states, dones)
        _assert_finite_metrics(metrics)
        losses.append(float(metrics["policy_loss"]))

    assert losses[-1] < losses[0], "BC policy loss did not decrease on a fixed batch"

    goal = torch.zeros(state_dim)
    goal[0] = 1.0
    policy_fn = gc_bc.get_task_policy(goal)
    obs = np.random.RandomState(0).randn(state_dim).astype(np.float32)
    action = policy_fn(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (action_dim,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0 + 1e-6)


def test_opal_reconstruction_loss_decreases():
    """OPAL trajectory autoencoder reconstruction loss decreases on a fixed batch."""
    state_dim, action_dim = 8, 3
    horizon = 8
    batch_size = 32

    states, actions, next_states, _, dones = _synthetic_batch(
        state_dim, action_dim, batch_size
    )

    opal = OPAL(
        state_dim=state_dim,
        action_dim=action_dim,
        skill_dim=4,
        hidden_dims=(64, 64),
        encoder_hidden=64,
        decoder_hidden=64,
        lr=3e-3,
        beta=1.0,
        batch_size=batch_size,
        horizon=horizon,
        device="cpu",
    )

    losses = []
    for _ in range(60):
        metrics = opal.train_step(states, actions, next_states, dones, horizon=horizon)
        _assert_finite_metrics(metrics)
        # Prefer a named total/reconstruction loss if available, otherwise use
        # the first key containing "loss".
        loss_key = None
        for key in metrics:
            if key in ("total_loss", "loss", "recon_loss", "reconstruction_loss"):
                loss_key = key
                break
        if loss_key is None:
            for key in metrics:
                if "loss" in key.lower():
                    loss_key = key
                    break
        assert loss_key is not None, "OPAL did not return any loss metric"
        losses.append(float(metrics[loss_key]))

    assert losses[-1] < losses[0], "OPAL reconstruction loss did not decrease on a fixed batch"


def test_baseline_policies_are_deterministic_given_context():
    """Goal/skill-conditioned policies return the same action for the same input."""
    state_dim, action_dim = 10, 3
    goal = torch.zeros(state_dim)
    goal[0] = 1.0
    obs = np.random.RandomState(42).randn(state_dim).astype(np.float32)

    gc_iql = GCIQL(state_dim=state_dim, action_dim=action_dim, hidden_dims=(32, 32), device="cpu")
    gc_bc = GCBC(state_dim=state_dim, action_dim=action_dim, hidden_dims=(32, 32), device="cpu")

    for policy_fn in (gc_iql.get_task_policy(goal), gc_bc.get_task_policy(goal)):
        a1 = policy_fn(obs)
        a2 = policy_fn(obs)
        np.testing.assert_allclose(a1, a2, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
