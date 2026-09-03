"""Unit tests for the random reward-function prior in ``fre.reward_prior``.

These tests intentionally avoid MuJoCo/D4RL and only require ``numpy``,
``torch``, and the reward-prior module.  They exercise the three reward
families described in the paper:

1. Singleton goal-reaching rewards (``SingletonGoalReward``).
2. Sparse random-linear rewards (``LinearReward``).
3. Random two-layer MLP rewards (``MLPReward``).

The public classes are used directly where deterministic behavior is needed,
while ``RewardPrior.sample_reward_fn`` is used to verify the uniform-mixture
sampling interface.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# Make the repository root importable when running ``pytest`` from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fre.reward_prior import (  # noqa: E402
    LinearReward,
    MLPReward,
    RewardPrior,
    SingletonGoalReward,
)


@pytest.fixture
def fixed_states() -> torch.Tensor:
    """A small deterministic batch of 2-D states for reward testing."""
    return torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )


def test_singleton_goal_reward_returns_minus_one_or_zero(fixed_states: torch.Tensor) -> None:
    """Singleton rewards must be ``0`` within epsilon and ``-1`` otherwise."""
    goal = torch.tensor([0.0, 0.0], dtype=torch.float32)
    reward_fn = SingletonGoalReward(goal=goal, epsilon=1.0)

    rewards = reward_fn(fixed_states)
    assert isinstance(rewards, torch.Tensor)
    assert rewards.shape == (fixed_states.shape[0],)
    assert rewards.dtype == torch.float32

    expected = torch.tensor([0.0, 0.0, -1.0, -1.0, -1.0], dtype=torch.float32)
    assert torch.allclose(rewards, expected, atol=1e-6), rewards


def test_linear_reward_matches_masked_dot_product() -> None:
    """Linear rewards equal ``<w, s>`` on unmasked coordinates (pre-clip).

    The implementation clips the final output to ``[-1, 1]``, so the chosen
    example keeps the raw dot product inside that interval.
    """
    weights = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    reward_fn = LinearReward(weights=weights, mask=mask)

    state = torch.tensor([[1.0, 10.0, 2.0]], dtype=torch.float32)
    rewards = reward_fn(state)
    assert rewards.shape == (1,)

    expected = 0.1 * 1.0 + 0.3 * 2.0  # 0.2 * 10 is masked out
    assert torch.allclose(rewards, torch.tensor([expected]), atol=1e-6)


def test_mlp_reward_is_bounded_after_clipping(fixed_states: torch.Tensor) -> None:
    """Random MLP rewards must be clipped to ``[-1, 1]``."""
    torch.manual_seed(0)
    # A 2-layer MLP exactly as described by the paper's reward prior.
    net = nn.Sequential(
        nn.Linear(fixed_states.shape[-1], 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    reward_fn = MLPReward(net=net)

    rewards = reward_fn(fixed_states)
    assert rewards.shape == (fixed_states.shape[0],)
    assert torch.all(rewards >= -1.0 - 1e-6)
    assert torch.all(rewards <= 1.0 + 1e-6)


def test_reward_prior_samples_uniform_mixture_of_three_families() -> None:
    """Over many samples, all three reward families must appear."""
    state_dim = 4
    prior = RewardPrior(state_dim=state_dim, device="cpu", seed=123)
    prior.seed(123)

    seen_kinds = set()
    for _ in range(300):
        reward_fn = prior.sample_reward_fn()
        assert reward_fn.kind in {"goal", "linear", "mlp"}
        seen_kinds.add(reward_fn.kind)

    assert seen_kinds == {"goal", "linear", "mlp"}


def test_reward_prior_scalar_and_batch_outputs() -> None:
    """Sampled reward functions should handle both single states and batches."""
    state_dim = 3
    state_pool = torch.randn(100, state_dim)
    prior = RewardPrior(state_dim=state_dim, state_pool=state_pool, device="cpu", seed=7)
    prior.seed(7)

    for _ in range(20):
        reward_fn = prior.sample_reward_fn()

        single = torch.randn(state_dim)
        single_reward = reward_fn(single)
        assert single_reward.ndim == 0 or single_reward.numel() == 1

        batch = torch.randn(8, state_dim)
        batch_rewards = reward_fn(batch)
        assert batch_rewards.shape == (8,)
        assert torch.all(batch_rewards >= -1.0 - 1e-6)
        assert torch.all(batch_rewards <= 1.0 + 1e-6)


def test_reward_prior_uses_state_pool_for_goals() -> None:
    """When a state pool is provided, singleton goals are drawn from it."""
    state_dim = 2
    state_pool = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    prior = RewardPrior(state_dim=state_dim, state_pool=state_pool, device="cpu", seed=0)
    prior.seed(0)

    goals_seen = []
    for _ in range(50):
        reward_fn = prior.sample_reward_fn()
        if reward_fn.kind == "goal":
            # The sampled goal is stored on the reward function.
            if hasattr(reward_fn, "goal"):
                goals_seen.append(reward_fn.goal.cpu().numpy().tolist())

    assert len(goals_seen) > 0
    for goal in goals_seen:
        assert any(np.allclose(goal, pool_row.numpy()) for pool_row in state_pool)
