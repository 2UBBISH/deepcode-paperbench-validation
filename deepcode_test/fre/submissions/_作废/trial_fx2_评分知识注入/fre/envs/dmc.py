"""DeepMind Control Suite (DMC) environment wrapper for ExORL domains.

This module provides a small, dependency-tolerant wrapper around
``dm_control`` environments so ExORL Walker/Cheetah tasks can be evaluated with
the same ``reset()``/``step()`` interface used by the FRE evaluation pipeline.

The wrapper intentionally defers importing ``dm_control`` (and ``gym``) until an
environment is actually instantiated, allowing the rest of the repository to be
imported on machines without MuJoCo/DMC installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DMCEnv", "make_dmc_env", "parse_dmc_env_name"]


def _load_dm_control_suite():
    """Import ``dm_control.suite`` lazily and raise a clear error if missing."""
    try:
        from dm_control import suite  # type: ignore
        return suite
    except Exception as exc:  # pragma: no cover - depends on local install
        raise ImportError(
            "dm_control is required for ExORL DMC environments. "
            "Install it with `pip install dm_control` or use the official ExORL "
            "evaluation protocol."
        ) from exc


def _maybe_gym_spaces():
    """Return gym.spaces if available, otherwise return simple placeholder classes."""
    try:
        import gym  # type: ignore
        return gym.spaces
    except Exception:
        return None


def parse_dmc_env_name(env_name: str) -> Tuple[str, str]:
    """Parse ``walker_walk`` or ``walker/walk`` into ``(domain_name, task_name)``.

    Args:
        env_name: Either an underscore-separated name (e.g. ``walker_walk``) or a
            slash-separated name (e.g. ``walker/walk``).

    Returns:
        A ``(domain_name, task_name)`` tuple.
    """
    if "/" in env_name:
        parts = env_name.split("/", 1)
        return parts[0], parts[1]
    if "_" in env_name:
        parts = env_name.split("_", 1)
        return parts[0], parts[1]
    return env_name, "walk"


class DMCEnv:
    """Gym-like wrapper around a single DeepMind Control Suite environment.

    Observations are flattened from the ordered ``dm_control`` observation dict
    into a single ``float32`` vector. This matches the state representation used
    by ExORL datasets after their row-wise observation flattening.
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str = "walk",
        seed: Optional[int] = None,
        max_episode_steps: int = 1000,
        task_kwargs: Optional[Dict[str, Any]] = None,
        from_pixels: bool = False,
        **kwargs: Any,
    ):
        self.domain_name = domain_name
        self.task_name = task_name
        self.max_episode_steps = int(max_episode_steps)
        self.from_pixels = from_pixels
        self._seed_value = seed
        self._task_kwargs = dict(task_kwargs or {})
        self._elapsed_steps = 0

        suite = _load_dm_control_suite()
        self._env = suite.load(
            domain_name,
            task_name,
            task_kwargs=self._task_kwargs,
        )

        self._obs_spec = self._env.observation_spec()
        self._action_spec = self._env.action_spec()

        # Flatten observation dim by summing per-component sizes.
        self.observation_dim = 0
        for value in self._obs_spec.values():
            arr = np.asarray(value)
            self.observation_dim += int(np.prod(arr.shape, dtype=int))

        action_arr = np.asarray(self._action_spec)
        self.action_dim = int(np.prod(action_arr.shape, dtype=int))

        spaces = _maybe_gym_spaces()
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_dim,),
                dtype=np.float32,
            )
            # DMC action specs are usually bounded, but fall back to [-1, 1].
            if hasattr(self._action_spec, "minimum") and hasattr(self._action_spec, "maximum"):
                low = np.asarray(self._action_spec.minimum, dtype=np.float32).reshape(-1)
                high = np.asarray(self._action_spec.maximum, dtype=np.float32).reshape(-1)
            else:
                low = -np.ones(self.action_dim, dtype=np.float32)
                high = np.ones(self.action_dim, dtype=np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = None
            self.action_space = None

        # Legacy four-tuple step API, matching D4RL environments and the
        # existing AntMaze/Kitchen wrappers.
        self._new_step_api = False

    @property
    def np_random(self) -> np.random.Generator:
        return np.random.default_rng(self._seed_value)

    def seed(self, seed: Optional[int] = None) -> None:
        self._seed_value = int(seed) if seed is not None else int(os.environ.get("FRE_SEED", 0))
        np.random.seed(self._seed_value)
        if hasattr(self._env, "seed"):
            try:
                self._env.seed(self._seed_value)
            except Exception:
                pass

    def _flatten_obs(self, observation: Any) -> np.ndarray:
        """Flatten a DMC observation (dict or array) into one vector."""
        if isinstance(observation, dict):
            pieces = []
            for key in self._obs_spec.keys():
                value = observation.get(key)
                if value is None:
                    value = np.zeros_like(np.asarray(self._obs_spec[key]))
                pieces.append(np.asarray(value, dtype=np.float32).reshape(-1))
            if not pieces:
                pieces.append(np.asarray(observation, dtype=np.float32).reshape(-1))
            return np.concatenate(pieces).astype(np.float32)
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)
        self._elapsed_steps = 0
        time_step = self._env.reset()
        self._last_time_step = time_step
        return self._flatten_obs(time_step.observation)

    def _action_to_numpy(self, action: Any) -> np.ndarray:
        import torch  # local import to keep torch optional for this module

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)

        if hasattr(self._action_spec, "minimum") and hasattr(self._action_spec, "maximum"):
            low = np.asarray(self._action_spec.minimum, dtype=np.float32).reshape(-1)
            high = np.asarray(self._action_spec.maximum, dtype=np.float32).reshape(-1)
            action = np.clip(action, low, high)
        return action

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step and return ``(obs, reward, done, info)``.

        ``info`` carries ``timeout`` and ``discount`` fields for downstream
        bootstrap masking.
        """
        action = self._action_to_numpy(action)
        self._elapsed_steps += 1

        time_step = self._env.step(action)
        self._last_time_step = time_step

        obs = self._flatten_obs(time_step.observation)
        reward = float(time_step.reward if time_step.reward is not None else 0.0)
        timeout = self._elapsed_steps >= self.max_episode_steps
        terminal = bool(time_step.last())
        info = {
            "discount": float(time_step.discount if time_step.discount is not None else 1.0),
            "timeout": timeout,
            "step": self._elapsed_steps,
        }
        return obs, reward, terminal, info

    def render(self, mode: str = "rgb_array", **kwargs: Any) -> Any:
        if hasattr(self._env, "physics"):
            return self._env.physics.render(**kwargs)
        raise NotImplementedError("Rendering is not available for this DMC environment.")

    def close(self) -> None:
        if hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:
                pass

    @property
    def physics(self) -> Any:
        return getattr(self._env, "physics", None)

    def get_state(self) -> Any:
        """Return the most recent DMC observation dict (useful for visualization)."""
        time_step = getattr(self, "_last_time_step", None)
        if time_step is None:
            return None
        return time_step.observation


def make_dmc_env(
    env_name: str = "walker_walk",
    seed: Optional[int] = None,
    max_episode_steps: int = 1000,
    task_kwargs: Optional[Dict[str, Any]] = None,
    from_pixels: bool = False,
    **kwargs: Any,
) -> DMCEnv:
    """Create a DMC environment from an ExORL-style environment name.

    Args:
        env_name: ``walker_walk``, ``cheetah_run``, ``walker/walk``, etc.
        seed: Optional random seed.
        max_episode_steps: Maximum episode length before a timeout flag is set.
        task_kwargs: Optional kwargs passed to ``dm_control.suite.load``.
        from_pixels: Whether pixel observations are requested.

    Returns:
        A :class:`DMCEnv` wrapper.
    """
    domain_name, task_name = parse_dmc_env_name(env_name)
    return DMCEnv(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        task_kwargs=task_kwargs,
        from_pixels=from_pixels,
        **kwargs,
    )
