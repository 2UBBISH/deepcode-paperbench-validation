"""Replay-based retention: episodic memory (EM).

At the beginning of fine-tuning a set of transitions from the pre-training task
is collected and inserted into the off-policy replay buffer.  That protected
region is never overwritten during training.  In the Meta-World experiments
10000 transitions are protected (10% of the 100k buffer, Appendix B.3) and in
Montezuma's Revenge a buffer of 500 trajectories gathered by the pre-trained
agent is used.

EM is only applicable to off-policy algorithms that maintain a replay buffer
(SAC), see Appendix C.3 -- on-policy APPO/PPO cannot use it trivially.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .base import RetentionConfig, RetentionMethod


class ProtectedReplayBuffer:
    """A replay buffer with a protected region reserved for old (pre-training) data.

    The first ``protected_size`` slots are reserved for the pre-training
    transitions and are never overwritten.  New transitions are stored in the
    remaining ``capacity - protected_size`` slots with a circular pointer.
    """

    def __init__(self, capacity: int, protected_size: int = 0, seed: Optional[int] = None) -> None:
        if protected_size > capacity:
            raise ValueError("protected_size cannot exceed capacity")
        self.capacity = int(capacity)
        self.protected_size = int(protected_size)
        self._rng = np.random.default_rng(seed)
        self._size = 0            # total filled slots
        self._ptr = self.protected_size  # circular pointer over the non-protected region
        self._storage: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _ensure(self, sample: Dict[str, np.ndarray], batched: bool = False) -> None:
        if self._storage:
            return
        for key, value in sample.items():
            value = np.asarray(value)
            per_sample = value.shape[1:] if batched else value.shape
            shape = (self.capacity,) + per_sample
            dtype = np.float32 if np.issubdtype(value.dtype, np.floating) else value.dtype
            self._storage[key] = np.zeros(shape, dtype=dtype)

    @property
    def num_protected(self) -> int:
        return min(self._size, self.protected_size)

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._size

    # ------------------------------------------------------------------
    def add_protected(self, transitions: Dict[str, np.ndarray]) -> None:
        """Insert the pre-training transitions into the protected region."""

        self._ensure(transitions, batched=True)
        n = len(next(iter(transitions.values())))
        n = min(n, self.protected_size)
        for key in self._storage:
            self._storage[key][:n] = np.asarray(transitions[key])[:n]
        self._size = max(self._size, n)
        self._ptr = max(self._ptr, n)

    def add(self, transition: Dict[str, np.ndarray]) -> None:
        """Insert a single new transition (never touches the protected region)."""

        self._ensure(transition)
        idx = self._ptr
        for key in self._storage:
            self._storage[key][idx] = np.asarray(transition[key])
        self._ptr = self.protected_size + (self._ptr - self.protected_size + 1) % max(
            self.capacity - self.protected_size, 1
        )
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, transitions: Dict[str, np.ndarray]) -> None:
        self._ensure(transitions, batched=True)
        n = len(next(iter(transitions.values())))
        for i in range(n):
            self.add({k: v[i] for k, v in transitions.items()})

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            key: torch.as_tensor(value[idx]).float() if value.dtype.kind == "f"
            else torch.as_tensor(value[idx])
            for key, value in self._storage.items()
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "protected_size": self.protected_size,
            "size": self._size,
            "ptr": self._ptr,
        }


class EpisodicMemory(RetentionMethod):
    """Thin :class:`RetentionMethod` wrapper around :class:`ProtectedReplayBuffer`.

    Episodic memory does not add an explicit auxiliary loss -- it is implemented
    by protecting part of the replay buffer.  ``aux_loss`` therefore returns a
    zero tensor and the actual logic lives in ``buffer``.
    """

    def __init__(self, config: Optional[RetentionConfig] = None, buffer: Optional[ProtectedReplayBuffer] = None) -> None:
        super().__init__(config)
        self.buffer = buffer

    def attach_buffer(self, buffer: ProtectedReplayBuffer) -> None:
        self.buffer = buffer

    def fill(self, transitions: Dict[str, np.ndarray]) -> None:
        if self.buffer is None:
            raise RuntimeError("EpisodicMemory has no buffer attached.")
        self.buffer.add_protected(transitions)

    def aux_loss(self, *_, **__) -> Tensor:
        return torch.zeros(())
