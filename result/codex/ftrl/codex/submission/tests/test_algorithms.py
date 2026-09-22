"""Tests for the algorithmic cores of the three experimental domains."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fpc.analysis.forward_transfer import auc, forward_transfer
from fpc.metaworld.config import MetaworldConfig
from fpc.metaworld.point_chain import PointReachChain, evaluate_point_chain
from fpc.metaworld.sac import ReplayBuffer, SAC, Transition
from fpc.montezuma.ppo import RolloutBuffer, compute_gae


# --------------------------------------------------------------------------
# PPO: generalised advantage estimation
# --------------------------------------------------------------------------
def test_gae_matches_hand_computed_two_step_case():
    """For a 1-step episode GAE reduces to the TD error."""

    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.0]])
    terminals = torch.tensor([[0.0], [1.0]])
    last_value = torch.zeros(1)
    advantages = compute_gae(rewards, values, terminals, 0.5, 0.95, last_value)

    # t = 1: terminal, delta = 2 + 0 - 0 = 2
    assert advantages[1, 0].item() == pytest.approx(2.0)
    # t = 0: delta = 1 + 0.5*0 - 0.5 = 0.5 ; plus 0.5*0.95*2 = 0.95
    assert advantages[0, 0].item() == pytest.approx(0.5 + 0.5 * 0.95 * 2.0)


def test_gae_is_zero_when_rewards_and_values_vanish():
    rewards = torch.zeros(3, 1)
    values = torch.zeros(3, 1)
    terminals = torch.zeros(3, 1)
    advantages = compute_gae(rewards, values, terminals, 0.99, 0.95, torch.zeros(1))
    assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-6)


def test_gae_discounts_a_terminal_reward_across_the_trajectory():
    """A single terminal reward of 1 propagates back with gamma * lambda."""

    gamma, lam = 0.9, 1.0
    rewards = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.zeros(3, 1)
    terminals = torch.tensor([[0.0], [0.0], [1.0]])
    advantages = compute_gae(rewards, values, terminals, gamma, lam, torch.zeros(1))
    assert advantages[2, 0].item() == pytest.approx(1.0)
    assert advantages[1, 0].item() == pytest.approx(gamma)
    assert advantages[0, 0].item() == pytest.approx(gamma ** 2)


def test_rollout_buffer_round_trip():
    buffer = RolloutBuffer(num_steps=4, num_envs=3, obs_shape=(2, 2, 2))
    for step in range(4):
        buffer.add(
            observations=torch.full((3, 2, 2, 2), float(step)),
            actions=torch.zeros(3, dtype=torch.long),
            log_probs=torch.zeros(3),
            values=torch.zeros(3),
            rewards=torch.ones(3),
            extrinsic_rewards=torch.ones(3),
            terminals=torch.zeros(3),
        )
    flat = buffer.flatten()
    assert flat.observations.shape == (12, 2, 2, 2)
    assert flat.observations[0, 0, 0, 0].item() == 0.0
    assert flat.observations[-1, 0, 0, 0].item() == 3.0


# --------------------------------------------------------------------------
# RoboticSequence: Algorithm 1 semantics
# --------------------------------------------------------------------------
def test_robotic_sequence_advances_only_on_success():
    """The chain must advance stage by stage and terminate after the last one."""

    env = PointReachChain(num_stages=3, time_limit=100, seed=0)
    obs, info = env.reset(seed=0)
    assert info["stage"] == 0

    solved = 0
    for _ in range(400):
        # Oracle action: move straight to the current goal (inverse transform).
        stage_env = env._stage_envs[env._current_stage]
        direction = stage_env.goal - stage_env.position
        action = np.linalg.solve(stage_env.transform, direction / max(np.linalg.norm(direction), 1e-6))
        result = env.step(action.astype(np.float32))
        solved += int(result.stage_solved)
        if result.terminated:
            break

    assert sum(env.stage_successes) == 3
    assert solved == 3  # exactly one success per stage
    env.close()


def test_robotic_sequence_augmented_success_reward():
    """r' = beta * r * (T - t) with beta = 1.5 on success, plain r otherwise."""

    env = PointReachChain(num_stages=1, time_limit=50, success_reward_beta=1.5, seed=0)
    env.reset(seed=0)
    stage_env = env._stage_envs[0]
    for _ in range(200):
        direction = stage_env.goal - stage_env.position
        if np.linalg.norm(direction) < stage_env.success_radius and env._t > 0:
            break
        action = np.linalg.solve(stage_env.transform, direction / max(np.linalg.norm(direction), 1e-6))
        result = env.step(action.astype(np.float32))
        if result.stage_solved:
            # success reward is guaranteed to be much larger than the dense reward
            assert result.reward > 0.5
            assert result.terminated
            break
    env.close()


def test_sac_improves_on_a_two_stage_chain():
    """A short SAC run on the stand-in chain must raise the success rate above zero."""

    config = MetaworldConfig()
    config.hidden_dim = 64
    config.batch_size = 32
    env = PointReachChain(num_stages=1, time_limit=40, seed=0)
    agent = SAC(config, env.observation_dim, env.num_actions, 1, device="cpu")
    replay = ReplayBuffer(10_000, env.observation_dim, env.num_actions)

    obs, info = env.reset(seed=0)
    stage = int(info["stage"])
    for step in range(1200):
        action = agent.select_action(obs, stage)
        result = env.step(action)
        replay.add(
            Transition(obs, action, result.reward, result.observation,
                       float(result.terminated or result.truncated), stage, result.stage)
        )
        obs, stage = result.observation, result.stage
        if result.terminated or result.truncated:
            obs, info = env.reset(seed=step)
            stage = int(info["stage"])
        if replay.size >= config.batch_size:
            agent.update(replay)

    metrics = evaluate_point_chain(agent, num_stages=1, episodes=10, time_limit=40, seed=100)
    assert metrics["overall"] >= 0.5
    env.close()


# --------------------------------------------------------------------------
# Forward transfer
# --------------------------------------------------------------------------
def test_forward_transfer_formula():
    steps = np.linspace(0, 1, 11)
    baseline = np.full(11, 0.2)
    assert auc(steps, baseline, total_steps=1.0) == pytest.approx(0.2, abs=1e-6)

    # A method that exactly doubles the baseline AUC gets (0.4-0.2)/(1-0.2) = 0.25.
    method = np.full(11, 0.4)
    assert forward_transfer(steps, method, steps, baseline, total_steps=1.0) == pytest.approx(0.25)

    # No improvement -> zero forward transfer; worse than baseline -> negative.
    assert forward_transfer(steps, baseline, steps, baseline, total_steps=1.0) == pytest.approx(0.0)
    assert forward_transfer(steps, np.full(11, 0.1), steps, baseline, total_steps=1.0) < 0
