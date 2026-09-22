"""MetaDrive ("Macro-v1") wrapper for the autonomous driving application.

The paper trains a PPO agent with the DI-drive implementation of MetaDrive
(Section C.2).  The qualitative analysis of the driving trajectories is out of
scope for this reproduction, but the *quantitative* refining result (the
"Auto Driving" columns of Table 1) is in scope, hence this wrapper.

Install the simulator with ``pip install metadrive-simulator``; the wrapper is
imported lazily so that the rest of the repository works without it.  Note that
recent versions of ``metadrive-simulator`` require ``numpy>=2``, which is
incompatible with ``torch 2.2`` and ``stable-baselines3 2.3``; install it in a
separate environment (the *refining* code of this repository only needs the
environment to expose ``reset``/``step``/state snapshots, so any version of the
simulator can be plugged in).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from rice.envs.adapters import StatefulEnv

MetaDriveEnv = None
METADRIVE_AVAILABLE = False
for _module in ("metadrive", "metadrive.envs"):
    try:  # pragma: no cover - optional dependency
        _mod = __import__(_module, fromlist=["MetaDriveEnv"])
        MetaDriveEnv = getattr(_mod, "MetaDriveEnv")
        METADRIVE_AVAILABLE = True
        break
    except Exception:  # pragma: no cover - optional dependency
        continue


def _flatten_observation(obs) -> np.ndarray:
    """MetaDrive returns a dict observation in recent versions; flatten it."""
    if isinstance(obs, dict):
        return np.concatenate(
            [
                np.asarray(value, dtype=np.float32).reshape(-1)
                for _, value in sorted(obs.items())
            ]
        )
    return np.asarray(obs, dtype=np.float32)


DEFAULT_CONFIG = {
    "environment_num": 100,
    "start_seed": 0,
    "traffic_density": 0.1,
    "map": "C",
    "random_traffic": False,
    "manual_control": False,
    "use_render": False,
}


class MetaDriveStatefulEnv(StatefulEnv):
    """Macro-v1 (MetaDrive) wrapper with action replay based snapshots."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: str = "metadrive"):
        if not METADRIVE_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError(
                "MetaDrive is not installed. Run `pip install metadrive-simulator` "
                "to enable the autonomous driving experiments."
            )
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        self._cfg = cfg
        self._env = MetaDriveEnv(cfg)
        self._action_log: list = []
        super().__init__(env=self._env, name=name)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        if getattr(self, "_obs_dim", None) is None:
            obs, _ = self.reset()
            self._obs_dim = int(np.asarray(obs).size)
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        out = self._env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        self._action_log = []
        return _flatten_observation(obs), {}

    def step(self, action):
        out = self._env.step(action)
        self._action_log.append(np.asarray(action, dtype=np.float32).copy())
        obs, reward, terminated, truncated, info = out
        return _flatten_observation(obs), reward, terminated, truncated, info

    def get_state(self) -> Dict[str, Any]:
        return {"actions": [a.copy() for a in self._action_log]}

    def set_state(self, state: Dict[str, Any]) -> None:
        self._env.reset()
        self._action_log = []
        for action in state["actions"]:
            self.step(action)
