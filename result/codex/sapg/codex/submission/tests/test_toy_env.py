"""Tests for the CPU toy suite used in smoke runs."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.envs.toy.multimodal_collect import MultiModalCollectEnv


def test_shapes_and_spaces():
    env = MultiModalCollectEnv(num_envs=8, seed=0)
    obs = env.reset()
    assert obs.shape == (8, env.obs_dim)
    step = env.step(torch.zeros(8, env.action_dim))
    assert step.obs.shape == (8, env.obs_dim)
    assert step.rewards.shape == (8,)
    assert step.dones.shape == (8,)


def test_reward_when_reaching_a_landmark():
    env = MultiModalCollectEnv(num_envs=1, seed=0, capture_radius=0.5, step_size=1.0)
    env.reset()
    target = torch.as_tensor(env.landmarks[0], dtype=torch.float32)
    start = env._x[0]
    action = torch.clamp(torch.as_tensor(target, dtype=torch.float32) - torch.as_tensor(start), -1, 1)
    step = env.step(action.unsqueeze(0))
    assert float(step.rewards[0]) > 0.0


def test_episode_ends_when_all_landmarks_captured():
    env = MultiModalCollectEnv(num_envs=1, seed=0, num_landmarks=2, capture_radius=10.0, max_steps=10)
    env.reset()
    step = env.step(torch.zeros(1, env.action_dim))
    assert float(step.dones[0]) == 1.0
    assert float(step.infos["successes"][0]) == 2.0


def test_episode_stats_reports_successes():
    env = MultiModalCollectEnv(num_envs=4, seed=0, num_landmarks=2, capture_radius=10.0, max_steps=10)
    env.reset()
    env.step(torch.zeros(4, env.action_dim))
    stats = env.episode_stats()
    assert "successes" in stats and stats["successes"] == 2.0
