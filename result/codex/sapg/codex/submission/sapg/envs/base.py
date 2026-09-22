"""Vectorised-environment interface used by every trainer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


@dataclass
class StepResult:
    obs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor          # episode terminated
    timeouts: torch.Tensor       # episode truncated (horizon reached)
    infos: Dict[str, torch.Tensor] = field(default_factory=dict)


class EpisodeTracker:
    """Accumulates per-env episode returns / success counts.

    ``successes`` mirrors the paper's metric for the AllegroKuka tasks ("the
    number of successes in a single episode", Sec. 5.1); ``episode_reward``
    mirrors the metric used for the two in-hand reorientation tasks.
    """

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.successes = np.zeros(num_envs, dtype=np.float64)
        self.lengths = np.zeros(num_envs, dtype=np.int64)
        self.finished_returns: List[float] = []
        self.finished_successes: List[float] = []
        self.finished_lengths: List[float] = []

    def step(self, rewards: np.ndarray, successes: np.ndarray, dones: np.ndarray) -> None:
        self.returns += rewards
        self.successes += successes
        self.lengths += 1
        for idx in np.nonzero(dones)[0]:
            self.finished_returns.append(float(self.returns[idx]))
            self.finished_successes.append(float(self.successes[idx]))
            self.finished_lengths.append(float(self.lengths[idx]))
            self.returns[idx] = 0.0
            self.successes[idx] = 0.0
            self.lengths[idx] = 0

    def pop_stats(self) -> Dict[str, float]:
        """Return the mean over episodes finished since the last call."""
        stats = {}
        if self.finished_returns:
            stats.update(self.summary())
        self.finished_returns.clear()
        self.finished_successes.clear()
        self.finished_lengths.clear()
        return stats

    def summary(self) -> Dict[str, float]:
        """Means over every episode finished so far (no clearing)."""
        if not self.finished_returns:
            return {
                "episode_reward": 0.0,
                "successes": 0.0,
                "episode_length": 0.0,
                "num_episodes": 0.0,
            }
        return {
            "episode_reward": float(np.mean(self.finished_returns)),
            "successes": float(np.mean(self.finished_successes)),
            "episode_length": float(np.mean(self.finished_lengths)),
            "num_episodes": float(len(self.finished_returns)),
        }


class VecEnv(ABC):
    """Minimal interface shared by the IsaacGym tasks and the toy suite."""

    num_envs: int
    obs_dim: int
    action_dim: int
    device: torch.device

    @abstractmethod
    def reset(self) -> torch.Tensor:
        ...

    @abstractmethod
    def step(self, actions: torch.Tensor) -> StepResult:
        ...

    def episode_stats(self) -> Dict[str, float]:
        """Mean of episode metrics over episodes finished since the last call."""
        return {}

    def close(self) -> None:  # pragma: no cover - trivial
        pass
