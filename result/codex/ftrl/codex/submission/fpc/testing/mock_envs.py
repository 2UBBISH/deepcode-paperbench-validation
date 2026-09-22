"""Minimal mock environments mirroring the real interfaces used by the trainers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..nethack.config import NetHackConfig


class MockAtariVecEnv:
    """Vectorised mock of the Montezuma's Revenge wrapper for the PPO+RND loop."""

    def __init__(self, num_envs: int = 4, height: int = 84, width: int = 84, num_actions: int = 18) -> None:
        self.num_envs = num_envs
        self.height = height
        self.width = width
        self.num_actions = num_actions
        self.observation_shape = (4, height, width)
        self._step = 0
        self._rng = np.random.default_rng(0)

    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        self._step = 0
        obs = self._rng.random((self.num_envs, 4, self.height, self.width), dtype=np.float32)
        return obs, [{} for _ in range(self.num_envs)]

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._step += 1
        obs = self._rng.random((self.num_envs, 4, self.height, self.width), dtype=np.float32)
        rewards = self._rng.random(self.num_envs).astype(np.float32)
        dones = (self._step % 20 == 0) * np.ones(self.num_envs, dtype=np.float32)
        return obs, rewards, dones


class MockNetHackEnv:
    """Mock NLE environment returning the four observation components."""

    def __init__(self, config: NetHackConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def _obs(self) -> Dict[str, np.ndarray]:
        cfg = self.config
        h, w = cfg.obs_screen_shape
        return {
            "glyphs": self._rng.integers(0, cfg.num_chars, size=(h, w)).astype(np.int64),
            "colors": self._rng.integers(0, cfg.num_colors, size=(h, w)).astype(np.int64),
            "blstats": self._rng.normal(size=(cfg.blstats_length,)).astype(np.float32),
            "message": self._rng.integers(0, cfg.num_chars, size=(cfg.message_length,)).astype(np.int64),
        }

    def reset(self) -> Dict[str, np.ndarray]:
        self._t = 0
        return self._obs()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        self._t += 1
        done = self._t >= 8
        return self._obs(), float(self._rng.random()), done, {}

    def close(self) -> None:
        pass


class MockRoboticSequence:
    """Mock RoboticSequence with a handful of stages and a toy solve rule."""

    class _Result:
        def __init__(self, observation, reward, terminated, truncated, info, stage, stage_solved):
            self.observation = observation
            self.reward = reward
            self.terminated = terminated
            self.truncated = truncated
            self.info = info
            self.stage = stage
            self.stage_solved = stage_solved

    def __init__(self, obs_dim: int = 10, action_dim: int = 4, num_stages: int = 3, time_limit: int = 20) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_stages = num_stages
        self.time_limit = time_limit
        self.observation_dim = obs_dim
        self.num_actions = action_dim
        self.stages = [f"stage-{i}" for i in range(num_stages)]
        self.stage_successes = [False] * num_stages
        self._rng = np.random.default_rng(0)
        self._stage = 0
        self._t = 0
        self._stage_t = 0

    def reset(self, seed: int | None = None):
        self._stage = 0
        self._t = 0
        self._stage_t = 0
        self.stage_successes = [False] * self.num_stages
        return self._rng.normal(size=self.obs_dim).astype(np.float32), {"stage": 0}

    def step(self, action):
        self._t += 1
        self._stage_t += 1
        solved = self._stage_t >= 5  # a stage is solved after 5 steps
        reward = 1.0
        terminated = truncated = False
        if solved:
            self.stage_successes[self._stage] = True
            if self._stage + 1 < self.num_stages:
                self._stage += 1
                self._stage_t = 0
            else:
                terminated = True
        if self._t >= self.time_limit:
            truncated = True
        obs = self._rng.normal(size=self.obs_dim).astype(np.float32)
        return self._Result(obs, reward, terminated, truncated, {}, self._stage, solved)

    def close(self) -> None:
        pass
