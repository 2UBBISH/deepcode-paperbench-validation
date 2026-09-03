"""Critical-state buffer for RICE refinement.

The buffer collects states visited by the frozen target policy, ranks them by the
learned criticality score ``ξ(s)``, and supports uniform sampling for the mixed
initial-state distribution used during refinement.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class CriticalState:
    """Container for a single critical state."""

    def __init__(
        self,
        observation: np.ndarray,
        xi: float,
        env_state: Optional[Any] = None,
        action_history: Optional[List[Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ):
        self.observation = np.asarray(observation)
        self.xi = float(xi)
        self.env_state = env_state
        self.action_history = action_history or []
        self.info = info or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation,
            "xi": self.xi,
            "env_state": self.env_state,
            "action_history": self.action_history,
            "info": self.info,
        }


class CriticalStateBuffer:
    """Replay-like buffer that stores and samples critical states.

    Parameters
    ----------
    capacity : int, optional
        Maximum number of critical states to retain. If ``None``, the buffer is
        unbounded.
    selection_mode : {"top_p", "threshold"}
        How states are selected from raw trajectories.
    top_p : float
        Percentile (0-1) of highest-ξ states to keep when ``selection_mode`` is
        ``"top_p"``.
    threshold : float
        Minimum ξ value to keep when ``selection_mode`` is ``"threshold"``.
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        selection_mode: str = "top_p",
        top_p: float = 0.25,
        threshold: float = 0.5,
    ):
        if selection_mode not in ("top_p", "threshold"):
            raise ValueError("selection_mode must be 'top_p' or 'threshold'")
        if top_p is not None and not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        self.capacity = capacity
        self.selection_mode = selection_mode
        self.top_p = top_p
        self.threshold = threshold
        self._buffer: List[CriticalState] = []

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def add_trajectory(
        self,
        observations: List[np.ndarray],
        xi_values: List[float],
        env_states: Optional[List[Any]] = None,
        action_histories: Optional[List[List[Any]]] = None,
        infos: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Add states from one target-policy trajectory.

        Only states satisfying the selection criterion are retained. If the
        buffer would exceed ``capacity``, lowest-ξ states are discarded.

        Returns
        -------
        int
            Number of states actually added.
        """
        if len(observations) != len(xi_values):
            raise ValueError("observations and xi_values must have the same length")

        env_states = env_states or [None] * len(observations)
        action_histories = action_histories or [[] for _ in observations]
        infos = infos or [{} for _ in observations]

        candidates = [
            CriticalState(obs, xi, es, ah, info)
            for obs, xi, es, ah, info in zip(
                observations, xi_values, env_states, action_histories, infos
            )
        ]

        selected = self._select_candidates(candidates)
        self._buffer.extend(selected)
        self._enforce_capacity()
        return len(selected)

    def add_state(self, state: CriticalState) -> None:
        """Add a single pre-selected critical state."""
        self._buffer.append(state)
        self._enforce_capacity()

    def _select_candidates(self, candidates: List[CriticalState]) -> List[CriticalState]:
        if not candidates:
            return []

        if self.selection_mode == "threshold":
            return [c for c in candidates if c.xi >= self.threshold]

        # top_p mode
        if self.top_p >= 1.0:
            return candidates
        if self.top_p <= 0.0:
            return []

        scores = np.array([c.xi for c in candidates])
        cutoff = np.percentile(scores, 100 * (1.0 - self.top_p))
        return [c for c in candidates if c.xi >= cutoff]

    def _enforce_capacity(self) -> None:
        if self.capacity is None:
            return
        if len(self._buffer) <= self.capacity:
            return
        # Keep the highest-ξ states when over capacity.
        self._buffer.sort(key=lambda s: s.xi, reverse=True)
        self._buffer = self._buffer[: self.capacity]

    def sample(
        self, n: int = 1, replace: bool = True
    ) -> Union[CriticalState, List[CriticalState]]:
        """Sample ``n`` critical states uniformly.

        Returns a single ``CriticalState`` when ``n == 1``, otherwise a list.
        """
        if self.is_empty:
            raise RuntimeError("Cannot sample from an empty CriticalStateBuffer")

        indices = np.random.choice(
            len(self._buffer), size=n, replace=replace and len(self._buffer) >= n
        )
        states = [self._buffer[i] for i in indices]
        return states[0] if n == 1 else states

    def get_top_k(self, k: int = 10) -> List[CriticalState]:
        """Return the ``k`` states with highest ξ."""
        sorted_states = sorted(self._buffer, key=lambda s: s.xi, reverse=True)
        return sorted_states[:k]

    def get_all(self) -> List[CriticalState]:
        """Return all stored states (unsorted)."""
        return list(self._buffer)

    def clear(self) -> None:
        """Remove all stored states."""
        self._buffer.clear()

    def summary(self) -> Dict[str, Any]:
        """Return a small statistics dict for logging."""
        if self.is_empty:
            return {"size": 0, "mean_xi": 0.0, "max_xi": 0.0, "min_xi": 0.0}
        xis = [s.xi for s in self._buffer]
        return {
            "size": len(self._buffer),
            "mean_xi": float(np.mean(xis)),
            "max_xi": float(np.max(xis)),
            "min_xi": float(np.min(xis)),
        }

    def save(self, path: str) -> None:
        """Save the buffer to disk as a NumPy archive."""
        data = {
            "observations": np.stack([s.observation for s in self._buffer]),
            "xi_values": np.array([s.xi for s in self._buffer]),
        }
        np.savez(path, **data)

    def load(self, path: str) -> None:
        """Load observations and ξ values from a NumPy archive."""
        data = np.load(path)
        observations = data["observations"]
        xi_values = data["xi_values"]
        self._buffer = [
            CriticalState(obs, xi) for obs, xi in zip(observations, xi_values)
        ]


def build_critical_buffer_from_trajectories(
    trajectories: List[Dict[str, Any]],
    capacity: Optional[int] = None,
    selection_mode: str = "top_p",
    top_p: float = 0.25,
    threshold: float = 0.5,
) -> CriticalStateBuffer:
    """Convenience factory that builds a buffer from a list of trajectory dicts.

    Each trajectory dict should contain keys ``observations`` and ``xi_values``,
    and optionally ``env_states``, ``action_histories``, and ``infos``.
    """
    buffer = CriticalStateBuffer(
        capacity=capacity,
        selection_mode=selection_mode,
        top_p=top_p,
        threshold=threshold,
    )
    for traj in trajectories:
        buffer.add_trajectory(
            observations=traj["observations"],
            xi_values=traj["xi_values"],
            env_states=traj.get("env_states"),
            action_histories=traj.get("action_histories"),
            infos=traj.get("infos"),
        )
    return buffer
