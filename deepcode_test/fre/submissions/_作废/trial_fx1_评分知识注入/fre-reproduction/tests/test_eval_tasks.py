"""Unit tests for downstream task definitions and zero-shot evaluation helpers.

These tests intentionally avoid MuJoCo/D4RL/ExORL dependencies. They exercise
the reward factories and 32-example state/reward samplers for AntMaze, Kitchen,
and ExORL tasks, plus the eval_zero_shot dispatch helpers that can run on CPU.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from envs.antmaze_wrapper import (  # noqa: E402
    ANTMAZE_TASKS,
    make_antmaze_task_reward,
    sample_task_reward_states as sample_antmaze_states,
)
from envs.exorl_wrapper import (  # noqa: E402
    EXORL_TASKS,
    make_exorl_task_reward,
    sample_task_reward_states as sample_exorl_states,
)
from envs.kitchen_wrapper import (  # noqa: E402
    KITCHEN_TASKS,
    make_kitchen_task_reward,
    sample_task_reward_states as sample_kitchen_states,
)

import eval_zero_shot  # noqa: E402


def _make_pool(n_rows: int, state_dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=(n_rows, state_dim)).astype(np.float32)


def _assert_exact_sample_count(states, rewards, expected_count, state_dim):
    assert isinstance(states, np.ndarray)
    assert isinstance(rewards, np.ndarray)
    assert states.shape[0] == expected_count
    assert rewards.shape[0] == expected_count
    assert states.shape[1] == state_dim
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(rewards))


# ---------------------------------------------------------------------------
# Task reward definitions
# ---------------------------------------------------------------------------

def test_antmaze_task_reward_definitions_exist():
    # All six paper-defined AntMaze evaluation tasks should be available.
    assert len(ANTMAZE_TASKS) == 6
    for task_name in ANTMAZE_TASKS:
        reward_fn = make_antmaze_task_reward(task_name)
        assert reward_fn is not None


def test_antmaze_goal_reward_is_sparse():
    states = _make_pool(64, 29)
    reward_fn = make_antmaze_task_reward("ant-goal-reaching")
    rewards = reward_fn(states)
    assert rewards.shape == (64,)
    unique = np.unique(rewards)
    assert set(unique).issubset({-1.0, 0.0})


def test_kitchen_task_rewards_are_sparse():
    states = _make_pool(64, 60)
    for task_name in KITCHEN_TASKS:
        reward_fn = make_kitchen_task_reward(task_name)
        rewards = reward_fn(states)
        assert rewards.shape == (64,)
        assert np.all((rewards == 0.0) | (rewards == -1.0))


def test_exorl_goal_rewards_are_sparse():
    walker_states = _make_pool(64, 24)
    cheetah_states = _make_pool(64, 18)

    for task_name, pool in [("walker-goals", walker_states),
                            ("cheetah-goals", cheetah_states)]:
        reward_fn = make_exorl_task_reward(task_name)
        rewards = reward_fn(pool)
        assert rewards.shape == (pool.shape[0],)
        assert set(np.unique(rewards)).issubset({-1.0, 0.0})


# ---------------------------------------------------------------------------
# 32-example state/reward sampling
# ---------------------------------------------------------------------------

def test_antmaze_sample_task_reward_states_exact_count():
    pool = _make_pool(128, 29)
    states, rewards = sample_antmaze_states(
        "ant-goal-reaching", pool, num_examples=32, seed=7
    )
    _assert_exact_sample_count(states, rewards, 32, 29)


def test_kitchen_sample_task_reward_states_exact_count():
    pool = _make_pool(128, 60)
    states, rewards = sample_kitchen_states(
        "microwave", pool, num_examples=32, seed=7
    )
    _assert_exact_sample_count(states, rewards, 32, 60)


def test_exorl_sample_task_reward_states_exact_count():
    walker_pool = _make_pool(128, 24)
    cheetah_pool = _make_pool(128, 18)

    states, rewards = sample_exorl_states(
        "walker-goals", walker_pool, num_examples=32, seed=7
    )
    _assert_exact_sample_count(states, rewards, 32, 24)

    states, rewards = sample_exorl_states(
        "cheetah-goals", cheetah_pool, num_examples=32, seed=7
    )
    _assert_exact_sample_count(states, rewards, 32, 18)


# ---------------------------------------------------------------------------
# eval_zero_shot helper integration
# ---------------------------------------------------------------------------

def test_resolve_tasks_returns_known_domain_tasks():
    assert list(eval_zero_shot.resolve_tasks("antmaze")) == list(ANTMAZE_TASKS)
    assert list(eval_zero_shot.resolve_tasks("kitchen")) == list(KITCHEN_TASKS)


def test_sample_task_pairs_antmaze():
    pool = _make_pool(128, 29)
    states, rewards = eval_zero_shot.sample_task_pairs(
        "antmaze", "ant-goal-reaching", pool, num_examples=32, seed=11
    )
    _assert_exact_sample_count(states, rewards, 32, 29)
    assert set(np.unique(rewards)).issubset({-1.0, 0.0})


def test_sample_task_pairs_kitchen():
    pool = _make_pool(128, 60)
    states, rewards = eval_zero_shot.sample_task_pairs(
        "kitchen", "microwave", pool, num_examples=32, seed=11
    )
    _assert_exact_sample_count(states, rewards, 32, 60)
    assert np.all((rewards == 0.0) | (rewards == -1.0))


def test_build_policy_fn_conditions_on_latent():
    class DummyAgent:
        def get_action(self, *args, **kwargs):
            # Return a fixed action regardless of how the closure calls us.
            return np.array([0.25, -0.5, 0.75], dtype=np.float32)

    agent = DummyAgent()
    z = np.zeros(8, dtype=np.float32)
    policy_fn = eval_zero_shot.build_policy_fn(agent, z)

    action = policy_fn(np.zeros(11, dtype=np.float32))
    assert np.allclose(action, np.array([0.25, -0.5, 0.75], dtype=np.float32))
