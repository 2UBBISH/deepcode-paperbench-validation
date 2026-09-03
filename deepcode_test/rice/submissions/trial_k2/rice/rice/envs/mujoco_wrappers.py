"""MuJoCo environment wrappers for RICE.

This module provides:
  - Observation normalization wrappers (used for Walker2d-v3 and HalfCheetah-v3).
  - Sparse-reward wrappers for Hopper-v3, Walker2d-v3, and HalfCheetah-v3 as
    defined by Mazoure et al. (2019).
  - Convenience factories to build dense/sparse MuJoCo environments.
"""

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


class RunningObsNormalizer(gym.Wrapper):
    """Online observation normalizer using running mean and standard deviation.

    This matches the observation-normalization scheme commonly used with
    Stable-Baselines3 VecNormalize, but implemented as a single-env wrapper
    for compatibility with the RICE training loop.
    """

    def __init__(
        self,
        env: gym.Env,
        eps: float = 1e-8,
        clip: float = 10.0,
    ):
        super().__init__(env)
        self.eps = eps
        self.clip = clip

        obs_shape = self.observation_space.shape
        self.mean = np.zeros(obs_shape, dtype=np.float64)
        self.var = np.ones(obs_shape, dtype=np.float64)
        self.count = eps

    def _update(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float64)
        batch_mean = obs
        batch_var = np.zeros_like(obs)
        batch_count = 1.0

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        std = np.sqrt(self.var) + self.eps
        normalized = (obs - self.mean) / std
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._update(obs)
        return self._normalize(obs), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update(obs)
        return self._normalize(obs), reward, terminated, truncated, info

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize an observation without updating running statistics."""
        return self._normalize(obs)

    def denormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert a normalized observation back to raw scale."""
        obs = np.asarray(obs, dtype=np.float64)
        std = np.sqrt(self.var) + self.eps
        return (obs * std + self.mean).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.mean = np.asarray(state_dict["mean"], dtype=np.float64)
        self.var = np.asarray(state_dict["var"], dtype=np.float64)
        self.count = float(state_dict["count"])


class SparseRewardWrapper(gym.Wrapper):
    """Base sparse-reward wrapper for MuJoCo locomotion tasks.

    Replaces the environment reward with a binary signal: +1 if the agent has
    moved forward past a threshold, otherwise 0. The threshold and the x-position
    accessor are task-specific and implemented by subclasses.
    """

    def __init__(self, env: gym.Env, threshold: float):
        super().__init__(env)
        self.threshold = threshold

    def _x_position(self, info: Dict[str, Any]) -> float:
        """Return the current forward (x) position."""
        raise NotImplementedError

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = self._x_position(info)
        sparse_reward = 1.0 if x > self.threshold else 0.0
        info["dense_reward"] = reward
        info["x_position"] = x
        return obs, sparse_reward, terminated, truncated, info


class SparseHopperWrapper(SparseRewardWrapper):
    """Sparse Hopper-v3: reward only if x > 0.6."""

    def __init__(self, env: gym.Env, threshold: float = 0.6):
        super().__init__(env, threshold)

    def _x_position(self, info: Dict[str, Any]) -> float:
        # Hopper-v3 info dict contains "x_position" in recent Gymnasium versions.
        if "x_position" in info:
            return float(info["x_position"])
        # Fallback: reconstruct from qpos if available.
        if hasattr(self.env.unwrapped, "data"):
            return float(self.env.unwrapped.data.qpos[0])
        return 0.0


class SparseWalker2dWrapper(SparseRewardWrapper):
    """Sparse Walker2d-v3: reward only if x > 0.6."""

    def __init__(self, env: gym.Env, threshold: float = 0.6):
        super().__init__(env, threshold)

    def _x_position(self, info: Dict[str, Any]) -> float:
        if "x_position" in info:
            return float(info["x_position"])
        if hasattr(self.env.unwrapped, "data"):
            return float(self.env.unwrapped.data.qpos[0])
        return 0.0


class SparseHalfCheetahWrapper(SparseRewardWrapper):
    """Sparse HalfCheetah-v3: reward only if x > 5.0."""

    def __init__(self, env: gym.Env, threshold: float = 5.0):
        super().__init__(env, threshold)

    def _x_position(self, info: Dict[str, Any]) -> float:
        if "x_position" in info:
            return float(info["x_position"])
        if hasattr(self.env.unwrapped, "data"):
            return float(self.env.unwrapped.data.qpos[0])
        return 0.0


def make_mujoco_env(
    env_id: str,
    normalize_obs: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> gym.Env:
    """Create a dense-reward MuJoCo environment.

    Args:
        env_id: Gymnasium environment id, e.g. ``Hopper-v3``.
        normalize_obs: If True, wrap with :class:`RunningObsNormalizer`.
        seed: Optional seed passed to ``env.reset``.
        **kwargs: Additional arguments forwarded to ``gym.make``.

    Returns:
        A Gymnasium environment.
    """
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    if normalize_obs:
        env = RunningObsNormalizer(env)
    return env


def make_sparse_mujoco_env(
    env_id: str,
    normalize_obs: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> gym.Env:
    """Create a sparse-reward MuJoCo environment.

    Supported ids and their default thresholds follow Mazoure et al. (2019):
      - Hopper-v3 / Walker2d-v3: x > 0.6
      - HalfCheetah-v3: x > 5.0

    Args:
        env_id: Base Gymnasium environment id.
        normalize_obs: Whether to add observation normalization.
        seed: Optional seed.
        **kwargs: Additional arguments forwarded to ``gym.make``.

    Returns:
        A sparse-reward Gymnasium environment.
    """
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)

    sparse_wrappers = {
        "Hopper-v3": SparseHopperWrapper,
        "Hopper-v4": SparseHopperWrapper,
        "Walker2d-v3": SparseWalker2dWrapper,
        "Walker2d-v4": SparseWalker2dWrapper,
        "HalfCheetah-v3": SparseHalfCheetahWrapper,
        "HalfCheetah-v4": SparseHalfCheetahWrapper,
    }

    base_id = env_id.split("/")[-1]
    if base_id not in sparse_wrappers:
        raise ValueError(
            f"No sparse wrapper defined for {env_id}. "
            f"Supported: {list(sparse_wrappers.keys())}"
        )

    env = sparse_wrappers[base_id](env)
    if normalize_obs:
        env = RunningObsNormalizer(env)
    return env


def is_mujoco_env_id(env_id: str) -> bool:
    """Return True if ``env_id`` is a known MuJoCo locomotion task."""
    known = {
        "Hopper-v3",
        "Hopper-v4",
        "Walker2d-v3",
        "Walker2d-v4",
        "HalfCheetah-v3",
        "HalfCheetah-v4",
        "Reacher-v2",
        "Reacher-v4",
        "Ant-v3",
        "Ant-v4",
        "Humanoid-v3",
        "Humanoid-v4",
    }
    base_id = env_id.split("/")[-1]
    return base_id in known


def should_normalize_obs(env_id: str) -> bool:
    """Return True if the paper recommends observation normalization for ``env_id``."""
    normalize_ids = {"Walker2d-v3", "Walker2d-v4", "HalfCheetah-v3", "HalfCheetah-v4"}
    base_id = env_id.split("/")[-1]
    return base_id in normalize_ids
