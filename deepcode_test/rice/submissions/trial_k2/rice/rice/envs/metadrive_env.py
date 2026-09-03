"""MetaDrive autonomous-driving environment wrapper for RICE.

The paper evaluates RICE on MetaDrive ``Macro-v1``.  This module exposes a
Gymnasium-compatible wrapper that:

* converts the 2-D continuous action ``a in [-1, 1]^2`` into MetaDrive's
  ``[steering, acceleration, brake]`` command,
* flattens the image/vector observation to a fixed-length vector when needed,
* aggregates a short macro-episode (``Macro-v1``) into a single RL episode,
* and falls back to a lightweight mock environment when MetaDrive is not
  installed so the rest of RICE can still be imported and unit-tested.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


try:  # pragma: no cover
    from metadrive import MetaDriveEnv as _MetaDriveEnv
    from metadrive.constants import RENDER_MODE_NONE

    _METADRIVE_AVAILABLE = True
except Exception:  # pragma: no cover
    _METADRIVE_AVAILABLE = False


DEFAULT_CONFIG: Dict[str, Any] = {
    "map": "XSOT",
    "traffic_density": 0.1,
    "num_scenarios": 1,
    "start_seed": 0,
    "accident_prob": 0.0,
    "use_render": False,
    "manual_control": False,
    "random_traffic": False,
    "decision_repeat": 5,
    "physics_world_step_size": 0.02,
}


def _flatten_obs(obs: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    """Recursively flatten a MetaDrive observation into a 1-D float vector."""
    if isinstance(obs, dict):
        parts = [_flatten_obs(v, dtype) for v in obs.values()]
        return np.concatenate(parts).astype(dtype)
    if isinstance(obs, (list, tuple)):
        return np.asarray(obs, dtype=dtype).ravel()
    arr = np.asarray(obs, dtype=dtype).ravel()
    return arr


def _convert_action(action: np.ndarray) -> np.ndarray:
    """Map ``a in [-1, 1]^2`` to MetaDrive's ``[steering, acceleration, brake]``.

    The paper states that the continuous action ``a in [-1, 1]^2`` is converted
    to steering/acceleration/brake.  We interpret the first dimension as
    steering (left/right) and the second dimension as throttle/brake: positive
    values accelerate, negative values brake.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != 2:
        raise ValueError(
            f"MetaDrive Macro-v1 expects a 2-D continuous action, got shape {action.shape}"
        )
    steering = float(np.clip(action[0], -1.0, 1.0))
    throttle_brake = float(np.clip(action[1], -1.0, 1.0))
    if throttle_brake >= 0.0:
        acceleration = throttle_brake
        brake = 0.0
    else:
        acceleration = 0.0
        brake = -throttle_brake
    return np.array([steering, acceleration, brake], dtype=np.float32)


class MetaDriveMacroEnv(gym.Env):
    """Gymnasium wrapper around MetaDrive ``Macro-v1`` for RICE.

    A *macro* episode consists of ``n_macro_steps`` consecutive MetaDrive
    environment steps.  The wrapper returns the accumulated reward and only
    terminates after the macro horizon is reached (or the vehicle crashes/out
    of road).  This mirrors the ``Macro-v1`` setting used in the paper.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        n_macro_steps: int = 100,
        flatten_obs: bool = True,
        target_obs_dim: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.config = dict(DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)
        self.n_macro_steps = max(1, int(n_macro_steps))
        self.flatten_obs = flatten_obs
        self.target_obs_dim = target_obs_dim
        self._seed = seed

        if _METADRIVE_AVAILABLE:
            self._env = _MetaDriveEnv(config=self.config)
            if seed is not None:
                self._env.reset(seed=seed)
        else:
            warnings.warn(
                "MetaDrive is not installed; using a mock MetaDrive environment. "
                "Install MetaDrive to run real autonomous-driving experiments."
            )
            obs_dim = target_obs_dim or 256
            self._env = _MockMetaDriveEnv(
                obs_dim=obs_dim, trial_length=n_macro_steps, seed=seed
            )

        # Action space: continuous 2-D vector in [-1, 1]^2.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space: infer from a reset sample.
        obs_sample = self._get_observation_sample()
        obs_dim = obs_sample.shape[0]
        obs_low = -np.inf * np.ones(obs_dim, dtype=np.float32)
        obs_high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(obs_dim,), dtype=np.float32
        )

        self._current_macro_step = 0
        self._last_obs: Optional[np.ndarray] = None

    def _get_observation_sample(self) -> np.ndarray:
        """Return a single observation vector to infer the observation shape."""
        if _METADRIVE_AVAILABLE:
            obs, _ = self._env.reset(seed=self._seed)
            obs = _flatten_obs(obs) if self.flatten_obs else np.asarray(obs, dtype=np.float32).ravel()
            if self.target_obs_dim is not None and obs.shape[0] != self.target_obs_dim:
                obs = _pad_or_truncate(obs, self.target_obs_dim)
            return obs
        return self._env.reset()[0]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._seed = seed
        if _METADRIVE_AVAILABLE and seed is not None:
            obs, info = self._env.reset(seed=seed)
        else:
            obs, info = self._env.reset()
        self._current_macro_step = 0
        self._last_obs = self._process_observation(obs)
        return self._last_obs, info

    def step(
        self, action: Union[np.ndarray, List[float], Tuple[float, ...]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        md_action = _convert_action(np.asarray(action, dtype=np.float32))

        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        for _ in range(self.n_macro_steps):
            if _METADRIVE_AVAILABLE:
                obs, reward, terminated, truncated, info = self._env.step(md_action)
            else:
                obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += float(reward)
            self._current_macro_step += 1
            if terminated or truncated:
                break

        self._last_obs = self._process_observation(obs)
        info["macro_step"] = self._current_macro_step
        info["macro_reward"] = total_reward
        return self._last_obs, total_reward, terminated, truncated, info

    def _process_observation(self, obs: Any) -> np.ndarray:
        obs = _flatten_obs(obs) if self.flatten_obs else np.asarray(obs, dtype=np.float32).ravel()
        if self.target_obs_dim is not None:
            obs = _pad_or_truncate(obs, self.target_obs_dim)
        return obs.astype(np.float32)

    def render(self) -> Optional[np.ndarray]:
        if _METADRIVE_AVAILABLE:
            return self._env.render()
        return None

    def close(self) -> None:
        if _METADRIVE_AVAILABLE and hasattr(self._env, "close"):
            self._env.close()

    def get_state(self) -> Any:
        """Best-effort state retrieval for mixed-reset refinement."""
        if _METADRIVE_AVAILABLE and hasattr(self._env, "get_state"):
            return self._env.get_state()
        return None

    def set_state(self, state: Any) -> None:
        """Best-effort state restoration for mixed-reset refinement."""
        if _METADRIVE_AVAILABLE and hasattr(self._env, "set_state"):
            self._env.set_state(state)


def _pad_or_truncate(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad with zeros or truncate ``arr`` to ``target_dim``."""
    if arr.shape[0] == target_dim:
        return arr
    if arr.shape[0] > target_dim:
        return arr[:target_dim]
    pad = np.zeros(target_dim - arr.shape[0], dtype=arr.dtype)
    return np.concatenate([arr, pad])


class _MockMetaDriveEnv:
    """Lightweight mock MetaDrive environment for import/testing without MetaDrive."""

    def __init__(self, obs_dim: int = 256, trial_length: int = 100, seed: Optional[int] = None):
        self.obs_dim = obs_dim
        self.trial_length = trial_length
        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        obs = self.rng.standard_normal(self.obs_dim).astype(np.float32)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        obs = self.rng.standard_normal(self.obs_dim).astype(np.float32)
        reward = float(self.rng.standard_normal())
        terminated = bool(self.rng.random() < 0.01)
        truncated = self._step_count >= self.trial_length
        info = {"mock": True}
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        pass


def make_metadrive_env(
    config: Optional[Dict[str, Any]] = None,
    n_macro_steps: int = 100,
    flatten_obs: bool = True,
    target_obs_dim: Optional[int] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """Factory that builds a Gymnasium-compatible MetaDrive ``Macro-v1`` env.

    Parameters
    ----------
    config:
        MetaDrive configuration dict.  See ``DEFAULT_CONFIG`` for defaults.
    n_macro_steps:
        Number of MetaDrive steps aggregated into one RL step (macro action).
    flatten_obs:
        Whether to flatten dict/list observations to a 1-D vector.
    target_obs_dim:
        If given, pad/truncate observations to this fixed dimension.
    seed:
        Random seed.

    Returns
    -------
    gym.Env
        A ``MetaDriveMacroEnv`` instance.
    """
    return MetaDriveMacroEnv(
        config=config,
        n_macro_steps=n_macro_steps,
        flatten_obs=flatten_obs,
        target_obs_dim=target_obs_dim,
        seed=seed,
    )
