"""MuJoCo environment wrappers for RICE.

Provides dense/sparse reward variants and observation normalization for the
MuJoCo continuous-control tasks used in the paper:
    * Hopper-v3
    * Walker2d-v3
    * Reacher-v2
    * HalfCheetah-v3

Sparse thresholds follow the paper appendix:
    * Hopper / Walker2d: reward only when x-position > 0.6
    * HalfCheetah: reward only when x-position > 5.0
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType


class DenseRewardWrapper(gym.Wrapper):
    """Identity wrapper that keeps the original dense MuJoCo reward.

    Useful for explicit tagging of dense-reward experiments in the RICE pipeline.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._sparse = False

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)


class SparseRewardWrapper(gym.Wrapper):
    """Base sparse-reward wrapper.

    Replaces the environment reward with a binary success signal when the
    agent's x-position exceeds ``threshold``; otherwise the reward is zero.
    The original termination / truncation signals are preserved.
    """

    def __init__(self, env: gym.Env, threshold: float = 0.6):
        super().__init__(env)
        self.threshold = threshold
        self._sparse = True

    def _x_position(self, obs: ObsType, info: Dict[str, Any]) -> float:
        """Extract x-position from observation or info dict."""
        # MuJoCo locomotion environments expose x-position in info["x_position"]
        # for Gymnasium >= 0.28; fall back to qpos[0] if available.
        if "x_position" in info:
            return float(info["x_position"])
        if hasattr(self.unwrapped, "data"):
            return float(self.unwrapped.data.qpos[0])
        if hasattr(self.unwrapped, "sim"):
            return float(self.unwrapped.sim.data.qpos[0])
        # Last resort: assume first observation coordinate is forward position.
        return float(np.asarray(obs).reshape(-1)[0])

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = self._x_position(obs, info)
        sparse_reward = 1.0 if x > self.threshold else 0.0
        info["original_reward"] = reward
        info["x_position"] = x
        return obs, sparse_reward, terminated, truncated, info


class SparseHopperWrapper(SparseRewardWrapper):
    """Sparse Hopper: reward only when x-position > 0.6."""

    def __init__(self, env: gym.Env):
        super().__init__(env, threshold=0.6)


class SparseWalker2dWrapper(SparseRewardWrapper):
    """Sparse Walker2d: reward only when x-position > 0.6."""

    def __init__(self, env: gym.Env):
        super().__init__(env, threshold=0.6)


class SparseHalfCheetahWrapper(SparseRewardWrapper):
    """Sparse HalfCheetah: reward only when x-position > 5.0."""

    def __init__(self, env: gym.Env):
        super().__init__(env, threshold=5.0)


class ReacherWrapper(gym.Wrapper):
    """Lightweight wrapper for Reacher-v2 that exposes dense reward unchanged.

    Reacher is already dense and does not require sparse variants in the paper.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._sparse = False


class NormalizeObservationWrapper(gym.Wrapper):
    """Running mean/std observation normalizer for single Gymnasium envs.

    Mirrors the observation normalization used by Stable-Baselines3
    ``VecNormalize`` but operates on a single environment.  Statistics are
    updated online during ``reset`` and ``step``.

    Parameters
    ----------
    env : gym.Env
        Environment to wrap.
    clip : float
        Clipping range for normalized observations.
    epsilon : float
        Small constant for numerical stability.
    """

    def __init__(self, env: gym.Env, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__(env)
        self.clip = clip
        self.epsilon = epsilon

        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise ValueError("NormalizeObservationWrapper only supports Box observation spaces")

        self.obs_rms_mean = np.zeros(obs_space.shape, dtype=np.float64)
        self.obs_rms_var = np.ones(obs_space.shape, dtype=np.float64)
        self.obs_count = epsilon

    def _update_stats(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float64)
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_count = obs.shape[0] if obs.ndim > 1 else 1

        delta = batch_mean - self.obs_rms_mean
        total_count = self.obs_count + batch_count

        new_mean = self.obs_rms_mean + delta * batch_count / total_count
        m_a = self.obs_rms_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.obs_count * batch_count / total_count
        new_var = m2 / total_count

        self.obs_rms_mean = new_mean
        self.obs_rms_var = new_var
        self.obs_count = total_count

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        std = np.sqrt(self.obs_rms_var + self.epsilon)
        normalized = (obs - self.obs_rms_mean) / std
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._update_stats(obs)
        return self._normalize(obs), info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_stats(obs)
        return self._normalize(obs), reward, terminated, truncated, info

    def get_stats(self) -> Dict[str, np.ndarray]:
        return {"mean": self.obs_rms_mean.copy(), "var": self.obs_rms_var.copy()}

    def set_stats(self, mean: np.ndarray, var: np.ndarray) -> None:
        self.obs_rms_mean = np.asarray(mean, dtype=np.float64)
        self.obs_rms_var = np.asarray(var, dtype=np.float64)


class TerminationWrapper(gym.Wrapper):
    """Wrapper that optionally disables early termination for MuJoCo tasks.

    Some RICE experiments keep the original termination behavior; others may
    want to let the agent continue until ``max_episode_steps`` truncation.
    This wrapper makes the choice explicit and stores the original termination
    flag in ``info``.
    """

    def __init__(self, env: gym.Env, terminate_on_unhealthy: bool = True):
        super().__init__(env)
        self.terminate_on_unhealthy = terminate_on_unhealthy

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_terminated"] = terminated
        if not self.terminate_on_unhealthy:
            terminated = False
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

SPARSE_THRESHOLDS = {
    "Hopper-v3": 0.6,
    "Walker2d-v3": 0.6,
    "HalfCheetah-v3": 5.0,
}

NORMALIZED_ENVS = {"Walker2d-v3", "HalfCheetah-v3"}


def make_mujoco_env(
    env_id: str,
    sparse: bool = False,
    normalize_obs: Optional[bool] = None,
    terminate_on_unhealthy: bool = True,
    **kwargs: Any,
) -> gym.Env:
    """Create a MuJoCo environment with RICE wrappers applied.

    Parameters
    ----------
    env_id : str
        Gymnasium environment id, e.g. ``Hopper-v3``.
    sparse : bool
        If True, replace the reward with a sparse binary signal.
    normalize_obs : bool or None
        If True, add running observation normalization.  If None, use the
        paper's default set (Walker2d and HalfCheetah).
    terminate_on_unhealthy : bool
        Whether to keep the default early-termination behavior.
    **kwargs
        Forwarded to ``gymnasium.make``.

    Returns
    -------
    gym.Env
        Wrapped environment ready for training / evaluation.
    """
    env = gym.make(env_id, **kwargs)

    if sparse:
        if env_id.startswith("Hopper"):
            env = SparseHopperWrapper(env)
        elif env_id.startswith("Walker2d"):
            env = SparseWalker2dWrapper(env)
        elif env_id.startswith("HalfCheetah"):
            env = SparseHalfCheetahWrapper(env)
        elif env_id.startswith("Reacher"):
            raise ValueError("Reacher does not have a sparse variant in RICE")
        else:
            threshold = SPARSE_THRESHOLDS.get(env_id, 0.6)
            env = SparseRewardWrapper(env, threshold=threshold)
    else:
        env = DenseRewardWrapper(env)

    if normalize_obs is None:
        normalize_obs = any(env_id.startswith(name) for name in NORMALIZED_ENVS)

    if normalize_obs:
        env = NormalizeObservationWrapper(env)

    env = TerminationWrapper(env, terminate_on_unhealthy=terminate_on_unhealthy)
    return env


def is_sparse_env(env: gym.Env) -> bool:
    """Return True if ``env`` (or any of its wrappers) is a sparse variant."""
    current: gym.Env = env
    while hasattr(current, "env"):
        if isinstance(current, SparseRewardWrapper):
            return True
        current = current.env  # type: ignore[assignment]
    return isinstance(current, SparseRewardWrapper)


def get_x_position(env: gym.Env) -> float:
    """Best-effort extraction of the agent's x-position from a MuJoCo env."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "data"):
        return float(unwrapped.data.qpos[0])
    if hasattr(unwrapped, "sim"):
        return float(unwrapped.sim.data.qpos[0])
    return 0.0
