"""Sparse reward variants of the MuJoCo locomotion tasks.

The paper uses the sparse reward versions of the MuJoCo games introduced by
Mazoure et al. (2019), where "the reward informs the x position of the agent
only if x > threshold" (Section C.2 of the paper):

    Hopper       x > 0.6
    Walker2d     x > 0.6
    HalfCheetah  x > 5
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym


SPARSE_X_THRESHOLD = {
    "Hopper": 0.6,
    "Walker2d": 0.6,
    "HalfCheetah": 5.0,
}


class SparseLocomotionWrapper(gym.Wrapper):
    """Replace the dense locomotion reward by the sparse x-position reward."""

    def __init__(self, env: gym.Env, x_threshold: float):
        super().__init__(env)
        self.x_threshold = float(x_threshold)

    def _x_position(self) -> float:
        env = self.env.unwrapped
        try:
            return float(env.data.qpos[0])
        except AttributeError:  # pragma: no cover - defensive
            return float(env.state[0])

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        x = self._x_position()
        reward = x if x > self.x_threshold else 0.0
        info = dict(info)
        info["dense_reward"] = info.get("reward", None)
        info["x_position"] = x
        return obs, float(reward), terminated, truncated, info


def make_sparse(env: gym.Env, base_name: str) -> gym.Env:
    """Wrap ``env`` so that it returns the sparse reward of ``base_name``."""
    if base_name not in SPARSE_X_THRESHOLD:
        raise KeyError("no sparse reward definition for {}".format(base_name))
    return SparseLocomotionWrapper(env, SPARSE_X_THRESHOLD[base_name])


def is_healthy(env: gym.Env) -> Optional[bool]:
    """Best effort healthy flag for the locomotion tasks."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "is_healthy"):
        try:
            return bool(unwrapped.is_healthy)
        except TypeError:
            return bool(unwrapped.is_healthy())
    return None
