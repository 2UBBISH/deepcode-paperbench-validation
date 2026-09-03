"""Replay buffers for RICE.

This module provides trajectory storage and critical-state replay utilities.
It is designed to complement :class:`rice.envs.resettable_env.CriticalStateBuffer`,
adding full-episode trajectory buffers, prioritized critical-state sampling, and
serialization helpers used by mask training and refining.
"""
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Transition:
    """Single environment transition."""

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)
    mask_score: Optional[float] = None


class TrajectoryBuffer:
    """Buffer that stores full trajectories (episodes) of transitions.

    Parameters
    ----------
    capacity:
        Maximum number of trajectories to retain. ``None`` means unbounded.
    """

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = capacity
        self.trajectories: List[List[Transition]] = []
        self._current: List[Transition] = []

    def start_episode(self) -> None:
        """Begin a new trajectory."""
        self._current = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
        mask_score: Optional[float] = None,
    ) -> None:
        """Append one transition to the currently open trajectory."""
        info = info or {}
        self._current.append(
            Transition(
                obs=np.asarray(obs),
                action=np.asarray(action),
                reward=float(reward),
                next_obs=np.asarray(next_obs),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
                mask_score=float(mask_score) if mask_score is not None else None,
            )
        )

    def end_episode(self) -> Optional[List[Transition]]:
        """Close the current trajectory and store it.

        Returns the stored trajectory, or ``None`` if it was empty.
        """
        if not self._current:
            return None
        traj = self._current
        self.trajectories.append(traj)
        self._current = []
        if self.capacity is not None and len(self.trajectories) > self.capacity:
            self.trajectories.pop(0)
        return traj

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> List[Transition]:
        return self.trajectories[idx]

    def sample_trajectory(self) -> Optional[List[Transition]]:
        """Sample one trajectory uniformly at random."""
        if not self.trajectories:
            return None
        return random.choice(self.trajectories)

    def sample_transitions(self, n: int) -> List[Transition]:
        """Sample ``n`` transitions uniformly from all trajectories."""
        all_transitions = [t for traj in self.trajectories for t in traj]
        if not all_transitions:
            return []
        n = min(n, len(all_transitions))
        return random.sample(all_transitions, k=n)

    def clear(self) -> None:
        """Remove all stored trajectories."""
        self.trajectories.clear()
        self._current = []

    def save(self, path: Union[str, Path]) -> None:
        """Persist the buffer to disk via pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.trajectories, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load trajectories from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory buffer not found: {path}")
        with path.open("rb") as f:
            self.trajectories = pickle.load(f)
        self._current = []

    def returns(self) -> List[float]:
        """Return the cumulative return of each stored trajectory."""
        returns = []
        for traj in self.trajectories:
            ret = sum(t.reward for t in traj)
            returns.append(ret)
        return returns

    def top_returns(self, k: int = 10) -> List[Tuple[int, float]]:
        """Return indices and returns of the top-``k`` trajectories."""
        returns = self.returns()
        indexed = sorted(enumerate(returns), key=lambda x: x[1], reverse=True)
        return indexed[:k]


class CriticalStateReplayBuffer:
    """Replay buffer specialized for critical states used during refining.

    In addition to the FIFO storage in :class:`CriticalStateBuffer`, this buffer
    supports:

    * prioritized sampling by mask score,
    * stratified sampling (top-k vs uniform vs default),
    * and persistence.

    Parameters
    ----------
    capacity:
        Maximum number of critical states to store. ``None`` means unbounded.
    alpha:
        Prioritization exponent. ``0`` means uniform sampling; ``1`` means
        proportional to mask score.
    """

    def __init__(self, capacity: Optional[int] = None, alpha: float = 1.0):
        self.capacity = capacity
        self.alpha = float(alpha)
        self.states: List[Dict[str, Any]] = []

    def add(self, state_dict: Dict[str, Any]) -> None:
        """Add a critical-state dictionary.

        Expected keys include ``obs`` and optionally ``simulator_state`` and
        ``mask_score``.
        """
        self.states.append(state_dict)
        if self.capacity is not None and len(self.states) > self.capacity:
            self.states.pop(0)

    def add_batch(self, state_dicts: Iterable[Dict[str, Any]]) -> None:
        """Add multiple critical-state dictionaries."""
        for sd in state_dicts:
            self.add(sd)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.states[idx]

    def sample(self, n: int = 1) -> List[Dict[str, Any]]:
        """Sample ``n`` critical states.

        If ``alpha > 0`` and states contain ``mask_score``, sampling is
        proportional to ``score ** alpha``. Otherwise uniform sampling is used.
        """
        if not self.states:
            return []
        n = min(n, len(self.states))
        if self.alpha > 0 and all("mask_score" in s for s in self.states):
            scores = np.array(
                [float(s.get("mask_score", 0.0)) for s in self.states], dtype=np.float64
            )
            scores = np.clip(scores, a_min=1e-8, a_max=None)
            probs = scores**self.alpha
            probs /= probs.sum()
            idx = np.random.choice(len(self.states), size=n, replace=False, p=probs)
            return [self.states[i] for i in idx]
        return random.sample(self.states, k=n)

    def top_k(self, k: int) -> List[Dict[str, Any]]:
        """Return the ``k`` states with highest ``mask_score``."""
        scored = [(i, float(s.get("mask_score", 0.0))) for i, s in enumerate(self.states)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self.states[i] for i, _ in scored[:k]]

    def clear(self) -> None:
        """Remove all stored states."""
        self.states.clear()

    def save(self, path: Union[str, Path]) -> None:
        """Persist the buffer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"capacity": self.capacity, "alpha": self.alpha, "states": self.states}, f
            )

    def load(self, path: Union[str, Path]) -> None:
        """Load a critical-state buffer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Critical state buffer not found: {path}")
        with path.open("rb") as f:
            data = pickle.load(f)
        self.capacity = data.get("capacity")
        self.alpha = data.get("alpha", 1.0)
        self.states = data.get("states", [])

    def to_resettable_buffer(self) -> "CriticalStateBuffer":
        """Convert to the FIFO :class:`rice.envs.resettable_env.CriticalStateBuffer`."""
        from rice.envs.resettable_env import CriticalStateBuffer

        buf = CriticalStateBuffer(capacity=self.capacity)
        for s in self.states:
            buf.add(s)
        return buf

    @classmethod
    def from_resettable_buffer(
        cls, buffer: "CriticalStateBuffer", alpha: float = 1.0
    ) -> "CriticalStateReplayBuffer":
        """Build a prioritized replay buffer from a FIFO critical-state buffer."""
        new = cls(capacity=buffer.capacity, alpha=alpha)
        for s in buffer.states:
            new.add(s)
        return new


def trajectories_to_critical_states(
    trajectories: List[List[Dict[str, Any]]],
    top_k: Optional[int] = None,
    percentile: Optional[float] = None,
    include_simulator_state: bool = True,
) -> List[Dict[str, Any]]:
    """Extract critical states from trajectory dictionaries.

    Each trajectory is a list of step dictionaries. Expected keys per step:
    ``obs``, ``mask_score`` (or ``xi``), and optionally ``simulator_state``.

    Parameters
    ----------
    trajectories:
        List of trajectories, each a list of step dictionaries.
    top_k:
        If given, return the top-``top_k`` states by mask score across all
        trajectories.
    percentile:
        If given, return all states whose mask score is at least this
        percentile (0-100).
    include_simulator_state:
        Whether to retain ``simulator_state`` in the output dictionaries.

    Returns
    -------
    List of critical-state dictionaries sorted by descending mask score.
    """
    states: List[Dict[str, Any]] = []
    for traj in trajectories:
        for step in traj:
            score = step.get("mask_score", step.get("xi", 0.0))
            sd: Dict[str, Any] = {
                "obs": step["obs"],
                "mask_score": float(score),
            }
            if include_simulator_state and "simulator_state" in step:
                sd["simulator_state"] = step["simulator_state"]
            if "info" in step:
                sd["info"] = step["info"]
            states.append(sd)

    states.sort(key=lambda s: s["mask_score"], reverse=True)

    if top_k is not None:
        states = states[:top_k]
    elif percentile is not None:
        if not states:
            return []
        threshold = np.percentile([s["mask_score"] for s in states], percentile)
        states = [s for s in states if s["mask_score"] >= threshold]

    return states


def merge_critical_state_buffers(
    buffers: Iterable["CriticalStateReplayBuffer"],
    capacity: Optional[int] = None,
    alpha: float = 1.0,
) -> CriticalStateReplayBuffer:
    """Merge several critical-state replay buffers into one."""
    merged = CriticalStateReplayBuffer(capacity=capacity, alpha=alpha)
    for buf in buffers:
        merged.add_batch(buf.states)
    if capacity is not None and len(merged.states) > capacity:
        merged.states = merged.states[-capacity:]
    return merged
