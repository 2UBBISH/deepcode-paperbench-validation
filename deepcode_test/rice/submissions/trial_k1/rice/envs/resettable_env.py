"""Resettable environment wrapper for RICE mixed-initial-state refining.

This wrapper implements the mixed initial-state distribution used in RICE:

    μ(s) = p · d_ρ^π(s) + (1-p) · ρ(s)

where with probability `p` an episode starts from a stored critical state
(and simulator state), and otherwise from the default initial distribution.
"""

from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class CriticalStateBuffer:
    """Storage for critical states recovered by the mask network."""

    def __init__(self, capacity: Optional[int] = None) -> None:
        self.capacity = capacity
        self._states: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._states)

    def add(self, state_dict: Dict[str, Any]) -> None:
        """Add a critical state dictionary.

        A state dictionary should contain at minimum an ``obs`` key.  For
        physics-based environments it is strongly recommended to also store
        ``simulator_state`` (e.g. MuJoCo qpos/qvel) so that the episode can be
        resumed deterministically.
        """
        self._states.append(copy.deepcopy(state_dict))
        if self.capacity is not None and len(self._states) > self.capacity:
            self._states.pop(0)

    def sample(self) -> Dict[str, Any]:
        if len(self) == 0:
            raise IndexError("Sampling from an empty CriticalStateBuffer")
        return copy.deepcopy(random.choice(self._states))

    def top_k(self, k: int) -> List[Dict[str, Any]]:
        """Return the ``k`` most recently added critical states."""
        return copy.deepcopy(self._states[-k:])

    def clear(self) -> None:
        self._states.clear()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._states, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._states = pickle.load(f)


class ResettableEnv(gym.Wrapper):
    """Wrapper that resets to a stored critical state with probability ``p``.

    Parameters
    ----------
    env :
        The base gymnasium environment.  Must expose ``unwrapped`` so that
        simulator state can be restored when available.
    critical_buffer :
        Buffer of critical states.  If ``None`` an empty buffer is created.
    p :
        Probability of starting from a critical state rather than the default
        initial distribution.
    """

    def __init__(
        self,
        env: gym.Env,
        critical_buffer: Optional[CriticalStateBuffer] = None,
        p: float = 0.25,
    ) -> None:
        super().__init__(env)
        self.p = p
        self.buffer = critical_buffer if critical_buffer is not None else CriticalStateBuffer()
        self._last_reset_from_critical = False

    @property
    def last_reset_from_critical(self) -> bool:
        """Whether the most recent reset started from a critical state."""
        return self._last_reset_from_critical

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        options = options or {}
        force_critical = options.pop("force_critical", None)

        use_critical = False
        if force_critical is True:
            use_critical = len(self.buffer) > 0
        elif force_critical is False:
            use_critical = False
        else:
            use_critical = len(self.buffer) > 0 and self.np_random.random() < self.p

        if use_critical:
            self._last_reset_from_critical = True
            state_dict = self.buffer.sample()
            obs, info = self._restore_state(state_dict, seed=seed, options=options)
            info["critical_reset"] = True
            return obs, info

        self._last_reset_from_critical = False
        obs, info = self.env.reset(seed=seed, options=options)
        info["critical_reset"] = False
        return obs, info

    def _restore_state(
        self,
        state_dict: Dict[str, Any],
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Restore simulator state and return the corresponding observation."""
        options = options or {}

        # First reset the environment to obtain a fresh info dict and to make
        # sure any internal rng/state is initialised.
        obs, info = self.env.reset(seed=seed, options=options)

        simulator_state = state_dict.get("simulator_state", None)
        if simulator_state is not None:
            self._set_simulator_state(simulator_state)

        # If an explicit observation was stored, trust it; otherwise re-observe.
        stored_obs = state_dict.get("obs", None)
        if stored_obs is not None:
            obs = np.asarray(stored_obs, dtype=self.observation_space.dtype)

        info["critical_reset"] = True
        return obs, info

    def _set_simulator_state(self, simulator_state: Any) -> None:
        """Attempt to restore simulator state using common APIs."""
        env = self.env.unwrapped

        # MuJoCo gymnasium environments expose set_state(qpos, qvel).
        if isinstance(simulator_state, (tuple, list)) and len(simulator_state) == 2:
            qpos, qvel = simulator_state
            if hasattr(env, "set_state"):
                env.set_state(np.asarray(qpos), np.asarray(qvel))
                return

        # Generic get_state/set_state interface used by some external domains.
        if hasattr(env, "set_state"):
            try:
                env.set_state(simulator_state)
                return
            except TypeError:
                # Fall through to attribute assignment if the signature differs.
                pass

        # Last resort: try to assign state attributes directly.
        if isinstance(simulator_state, dict):
            for key, value in simulator_state.items():
                if hasattr(env, key):
                    setattr(env, key, copy.deepcopy(value))

    def add_critical_state(
        self,
        obs: np.ndarray,
        simulator_state: Optional[Any] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a critical state into the buffer."""
        state_dict: Dict[str, Any] = {"obs": np.asarray(obs)}
        if simulator_state is not None:
            state_dict["simulator_state"] = simulator_state
        if extras is not None:
            state_dict.update(extras)
        self.buffer.add(state_dict)

    def save_buffer(self, path: Path) -> None:
        self.buffer.save(path)

    def load_buffer(self, path: Path) -> None:
        self.buffer.load(path)


def make_resettable(
    env: gym.Env,
    p: float = 0.25,
    buffer_path: Optional[Path] = None,
    capacity: Optional[int] = None,
) -> ResettableEnv:
    """Convenience factory wrapping ``env`` with a ``ResettableEnv``."""
    buffer = CriticalStateBuffer(capacity=capacity)
    if buffer_path is not None and Path(buffer_path).exists():
        buffer.load(buffer_path)
    return ResettableEnv(env, critical_buffer=buffer, p=p)
