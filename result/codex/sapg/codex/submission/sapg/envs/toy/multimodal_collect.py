"""A CPU-only vectorised control task that mirrors the paper's setting.

Why this environment exists
---------------------------
The paper's five benchmark tasks require IsaacGym and tens of thousands of
parallel environments, which cannot be run here (no GPU, and experiments are
re-run later outside this environment).  This suite is a *small-scale surrogate*
that reproduces the structural properties SAPG is designed for, so that the
algorithm, baselines and analyses can be smoke-tested end-to-end on CPU:

* a single task with several **disjoint reward modes** (K landmarks that must
  be captured in turn) so that a unimodal Gaussian policy that samples near its
  mean keeps re-visiting the mode it already found -- exactly the "many
  environments execute the same action, so more parallel data does not help"
  argument of Sec. 4 / Figure 2;
* rewards are sparse and binary, so a policy that never explores a mode gets no
  signal from it;
* the visited-state distribution depends on behaviour, so the Sec. 6.4
  diversity metrics (PCA / MLP reconstruction error) are meaningful here.

Observation = [position (2), visited mask (K), time fraction (1)].
Action      = 2-D velocity command, x_{t+1} = x_t + step_size * clip(a, -1, 1).
Reward      = +1 the first time the agent enters the capture radius of a
              landmark; episodes end after ``max_steps`` or when all landmarks
              have been captured.  ``successes`` = landmarks captured.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..base import EpisodeTracker, StepResult, VecEnv


class MultiModalCollectEnv(VecEnv):
    def __init__(
        self,
        num_envs: int = 1024,
        seed: int = 0,
        num_landmarks: int = 6,
        max_steps: int = 64,
        capture_radius: float = 0.12,
        step_size: float = 0.15,
        device: str = "cpu",
        boundary: float = 1.2,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_landmarks = int(num_landmarks)
        self.max_steps = int(max_steps)
        self.capture_radius = float(capture_radius)
        self.step_size = float(step_size)
        self.boundary = float(boundary)
        self.device = torch.device(device)
        self.obs_dim = 2 + self.num_landmarks + 1
        self.action_dim = 2

        rng = np.random.RandomState(seed)
        # landmarks spread out on a ring so that the modes are well separated
        angles = np.linspace(0.0, 2.0 * np.pi, self.num_landmarks, endpoint=False)
        radius = 0.9
        self.landmarks = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
        self.landmark_weights = np.ones(self.num_landmarks)
        self.landmark_weights[: self.num_landmarks // 2] = 1.0 + 0.25 * rng.rand(self.num_landmarks // 2)

        self._obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self._x = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._visited = np.zeros((self.num_envs, self.num_landmarks), dtype=np.float32)
        self._t = np.zeros(self.num_envs, dtype=np.int64)
        self.tracker = EpisodeTracker(self.num_envs)
        self._rng = rng
        self.episode_count = 0

    # ------------------------------------------------------------------ #
    def _compose_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self._x,
                self._visited,
                (self._t / float(self.max_steps))[:, None].astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

    def reset(self) -> torch.Tensor:
        self._x = self._rng.uniform(-0.3, 0.3, size=(self.num_envs, 2)).astype(np.float32)
        self._visited = np.zeros((self.num_envs, self.num_landmarks), dtype=np.float32)
        self._t = np.zeros(self.num_envs, dtype=np.int64)
        self._obs = self._compose_obs()
        self.tracker = EpisodeTracker(self.num_envs)
        return torch.as_tensor(self._obs, device=self.device)

    def step(self, actions: torch.Tensor) -> StepResult:
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        actions_np = np.clip(actions_np, -1.0, 1.0)
        self._x = np.clip(self._x + self.step_size * actions_np, -self.boundary, self.boundary)

        # capture landmarks that are (a) unvisited and (b) within the radius
        dists = np.linalg.norm(self._x[:, None, :] - self.landmarks[None, :, :], axis=-1)
        inside = dists <= self.capture_radius
        new_captures = inside & (self._visited < 0.5)
        rewards = (new_captures * self.landmark_weights[None, :]).sum(axis=-1).astype(np.float32)
        successes = new_captures.sum(axis=-1).astype(np.float32)
        self._visited = np.clip(self._visited + new_captures.astype(np.float32), 0.0, 1.0)

        self._t += 1
        all_captured = self._visited.sum(axis=-1) >= self.num_landmarks
        timeouts = (self._t >= self.max_steps) & ~all_captured
        dones = all_captured

        obs = self._compose_obs()
        # captured the whole set: start a new episode immediately
        if dones.any():
            idx = np.nonzero(dones)[0]
            self._x[idx] = self._rng.uniform(-0.3, 0.3, size=(idx.size, 2)).astype(np.float32)
            self._visited[idx] = 0.0
            self._t[idx] = 0
            self._obs = self._compose_obs()
            obs = self._obs
        if timeouts.any():
            idx = np.nonzero(timeouts)[0]
            self._x[idx] = self._rng.uniform(-0.3, 0.3, size=(idx.size, 2)).astype(np.float32)
            self._visited[idx] = 0.0
            self._t[idx] = 0
            self._obs = self._compose_obs()
            obs = self._obs

        done_flags = dones | timeouts
        self.tracker.step(rewards, successes, done_flags.astype(bool))
        return StepResult(
            obs=torch.as_tensor(obs, device=self.device),
            rewards=torch.as_tensor(rewards, device=self.device),
            dones=torch.as_tensor(dones.astype(np.float32), device=self.device),
            timeouts=torch.as_tensor(timeouts.astype(np.float32), device=self.device),
            infos={"successes": torch.as_tensor(successes, device=self.device)},
        )

    def scalar_rewards(self, rewards: torch.Tensor) -> np.ndarray:
        return rewards.detach().cpu().numpy()

    def episode_stats(self) -> Dict[str, float]:
        stats = self.tracker.pop_stats()
        stats["mean_landmarks"] = stats.get("successes", 0.0)
        return stats

    # ------------------------------------------------------------------ #
    def state_batch(self, num_samples: int = 4096) -> np.ndarray:
        """Sample observations currently in the buffer (for diversity metrics)."""
        idx = self._rng.randint(0, self.num_envs, size=num_samples)
        return self._obs[idx].copy()
