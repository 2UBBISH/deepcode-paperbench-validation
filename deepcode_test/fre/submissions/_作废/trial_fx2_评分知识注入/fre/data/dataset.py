"""Offline dataset abstraction for FRE.

The :class:`OfflineDataset` class is intentionally agnostic to the source of
the data (D4RL, ExORL, or custom buffers).  It stores transitions as PyTorch
tensors, infers episode boundaries, and exposes the three sampling primitives
required by the rest of the codebase:

* uniform state sampling for FRE reward/encoder contexts,
* uniform transition sampling for offline RL updates,
* full-trajectory sampling for hindsight relabeling and evaluation.

State normalization is handled here as well.  By default states are normalized
with dataset statistics; this is usually important for AntMaze and Kitchen
where state scales differ across coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from fre.config import DataConfig


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence]


def _as_numpy(x: Optional[ArrayLike], name: str) -> Optional[np.ndarray]:
    """Convert an array-like object to a float32 numpy array."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return arr


def _to_tensor(x: Optional[np.ndarray], dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return torch.as_tensor(x, dtype=dtype)


@dataclass
class TransitionBatch:
    """A minibatch of RL transitions.

    All fields are ``[batch_size, ...]`` tensors.  ``terminals`` mark true
    episode termination, while ``timeouts`` mark artificially truncated
    trajectories.  In offline RL the timeout flag is normally *not* treated as
    an absorbing state, so value bootstrapping should use ``terminals`` only.
    """

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    timeouts: Optional[torch.Tensor] = None
    goals: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.states.shape[0])


@dataclass
class Episode:
    """A full trajectory (episode) stored as tensors."""

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    timeouts: torch.Tensor
    goals: Optional[torch.Tensor] = None
    index: int = -1

    def __len__(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def return_(self) -> float:
        return float(self.rewards.sum().item())


class OfflineDataset:
    """Uniform interface over an offline transition dataset.

    Parameters
    ----------
    data:
        A dictionary containing at least ``states``/``observations``,
        ``actions``, ``next_states``/``next_observations``, and ``rewards``.
        The keys ``terminals``/``dones``, ``timeouts``, ``goals``, and
        ``episode_ids`` are optional.  Arrays are copied into torch tensors.
    cfg:
        Optional :class:`~fre.config.DataConfig`; used only for
        ``normalize_states`` and ``seed`` defaults.
    device:
        Default device used by sampling helpers when ``device=None``.
    state_mean, state_std:
        Optional precomputed normalization statistics.  If omitted and
        normalization is enabled, they are estimated from ``states``.
    seed:
        RNG seed for dataset sampling.
    """

    def __init__(
        self,
        data: Optional[Dict[str, ArrayLike]] = None,
        cfg: Optional[DataConfig] = None,
        device: str = "cpu",
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.cfg = cfg
        self.device = device
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        if data is None:
            data = kwargs

        states = _as_numpy(data.get("states", data.get("observations")), "states")
        actions = _as_numpy(data.get("actions"), "actions")
        next_states = _as_numpy(
            data.get("next_states", data.get("next_observations")), "next_states"
        )
        rewards = _as_numpy(data.get("rewards"), "rewards")
        terminals = _as_numpy(data.get("terminals", data.get("dones")), "terminals")
        timeouts = _as_numpy(data.get("timeouts"), "timeouts")
        goals = _as_numpy(data.get("goals"), "goals")
        episode_ids = _as_numpy(data.get("episode_ids"), "episode_ids")

        if states is None or actions is None or next_states is None or rewards is None:
            raise ValueError(
                "OfflineDataset requires states, actions, next_states, and rewards arrays."
            )

        # Squeeze accidental trailing singleton dimensions (common with scalar rewards).
        if rewards.ndim == 2 and rewards.shape[1] == 1:
            rewards = rewards[:, 0]
        if terminals is not None and terminals.ndim == 2 and terminals.shape[1] == 1:
            terminals = terminals[:, 0]
        if timeouts is not None and timeouts.ndim == 2 and timeouts.shape[1] == 1:
            timeouts = timeouts[:, 0]

        n_transitions = int(states.shape[0])
        if terminals is None:
            terminals = np.zeros(n_transitions, dtype=np.float32)
        if timeouts is None:
            timeouts = np.zeros(n_transitions, dtype=np.float32)

        terminals = terminals.astype(np.float32)
        timeouts = timeouts.astype(np.float32)
        rewards = rewards.astype(np.float32)

        self._raw_states = states.astype(np.float32)
        self._raw_next_states = next_states.astype(np.float32)
        self._actions = actions.astype(np.float32)
        self._rewards = rewards
        self._terminals = terminals
        self._timeouts = timeouts
        self._goals = goals.astype(np.float32) if goals is not None else None
        self._n_transitions = n_transitions

        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------
        normalize = True
        if cfg is not None:
            normalize = bool(getattr(cfg, "normalize_states", True))
        self.normalize_states_flag = normalize

        if normalize:
            if state_mean is None:
                state_mean = self._raw_states.mean(axis=0, keepdims=True)
            if state_std is None:
                state_std = self._raw_states.std(axis=0, keepdims=True) + 1e-6
            self.state_mean = np.asarray(state_mean, dtype=np.float32).reshape(1, -1)
            self.state_std = np.asarray(state_std, dtype=np.float32).reshape(1, -1)
            self._states = (self._raw_states - self.state_mean) / self.state_std
            self._next_states = (self._raw_next_states - self.state_mean) / self.state_std
        else:
            self.state_mean = np.zeros((1, self._raw_states.shape[1]), dtype=np.float32)
            self.state_std = np.ones((1, self._raw_states.shape[1]), dtype=np.float32)
            self._states = self._raw_states.copy()
            self._next_states = self._raw_next_states.copy()

        # ------------------------------------------------------------------
        # Episode boundaries
        # ------------------------------------------------------------------
        if episode_ids is not None:
            episode_ids = episode_ids.astype(np.int64)
            boundaries = list(np.where(episode_ids[1:] != episode_ids[:-1])[0] + 1)
            # Ensure final boundary exists even if the last episode is unterminated.
            if not boundaries or boundaries[-1] != n_transitions:
                boundaries.append(n_transitions)
        else:
            end_flags = (terminals > 0.5) | (timeouts > 0.5)
            boundaries = [int(i) + 1 for i in np.where(end_flags)[0]]
            if not boundaries or boundaries[-1] != n_transitions:
                boundaries.append(n_transitions)

        self._starts: List[int] = []
        self._ends: List[int] = []
        start = 0
        for end in boundaries:
            end = min(int(end), n_transitions)
            if end <= start:
                continue
            self._starts.append(start)
            self._ends.append(end)
            start = end

        if not self._starts:
            self._starts = [0]
            self._ends = [n_transitions]

        self.num_episodes = len(self._starts)
        self._episode_index = np.empty(n_transitions, dtype=np.int64)
        for ep in range(self.num_episodes):
            self._episode_index[self._starts[ep] : self._ends[ep]] = ep

        # ------------------------------------------------------------------
        # Torch storage for fast indexing
        # ------------------------------------------------------------------
        self._states_t = torch.as_tensor(self._states)
        self._next_states_t = torch.as_tensor(self._next_states)
        self._actions_t = torch.as_tensor(self._actions)
        self._rewards_t = torch.as_tensor(self._rewards)
        self._terminals_t = torch.as_tensor(self._terminals)
        self._timeouts_t = torch.as_tensor(self._timeouts)
        self._goals_t = torch.as_tensor(self._goals) if self._goals is not None else None

        self._episode_returns = self._compute_episode_returns()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return int(self._states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self._actions.shape[1])

    @property
    def size(self) -> int:
        return self._n_transitions

    @property
    def episode_returns(self) -> List[float]:
        return list(self._episode_returns)

    @property
    def raw_states(self) -> torch.Tensor:
        return torch.as_tensor(self._raw_states)

    @property
    def raw_next_states(self) -> torch.Tensor:
        return torch.as_tensor(self._raw_next_states)

    @property
    def states(self) -> torch.Tensor:
        return self._states_t

    @property
    def actions(self) -> torch.Tensor:
        return self._actions_t

    @property
    def next_states(self) -> torch.Tensor:
        return self._next_states_t

    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards_t

    @property
    def terminals(self) -> torch.Tensor:
        return self._terminals_t

    @property
    def timeouts(self) -> torch.Tensor:
        return self._timeouts_t

    @property
    def goals(self) -> Optional[torch.Tensor]:
        return self._goals_t

    def __len__(self) -> int:
        return self._n_transitions

    # ------------------------------------------------------------------
    # State normalization helpers
    # ------------------------------------------------------------------
    def normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize raw states using the dataset statistics."""
        mean = torch.as_tensor(self.state_mean, dtype=states.dtype, device=states.device)
        std = torch.as_tensor(self.state_std, dtype=states.dtype, device=states.device)
        return (states - mean) / std

    def unnormalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """Map normalized states back to raw environment coordinates."""
        mean = torch.as_tensor(self.state_mean, dtype=states.dtype, device=states.device)
        std = torch.as_tensor(self.state_std, dtype=states.dtype, device=states.device)
        return states * std + mean

    # ------------------------------------------------------------------
    # Sampling primitives
    # ------------------------------------------------------------------
    def _device(self, device: Optional[str]) -> torch.device:
        if device is None:
            device = self.device
        return torch.device(device)

    def sample_states(self, batch_size: int, device: Optional[str] = None) -> torch.Tensor:
        """Uniformly sample states for FRE encoder/decoder contexts."""
        idx = torch.randint(0, self._n_transitions, (batch_size,))
        return self._states_t[idx].to(self._device(device))

    def sample_transitions(
        self, batch_size: int, device: Optional[str] = None
    ) -> TransitionBatch:
        """Uniformly sample transitions for RL updates."""
        idx = torch.randint(0, self._n_transitions, (batch_size,))
        return self._transition_batch_from_idx(idx, device=device)

    def _transition_batch_from_idx(
        self, idx: torch.Tensor, device: Optional[str] = None
    ) -> TransitionBatch:
        dev = self._device(device)
        goals = None
        if self._goals_t is not None:
            goals = self._goals_t[idx].to(dev)
        return TransitionBatch(
            states=self._states_t[idx].to(dev),
            actions=self._actions_t[idx].to(dev),
            next_states=self._next_states_t[idx].to(dev),
            rewards=self._rewards_t[idx].to(dev),
            terminals=self._terminals_t[idx].to(dev),
            timeouts=self._timeouts_t[idx].to(dev),
            goals=goals,
            indices=idx.to(dev),
        )

    def sample_trajectory(self, device: Optional[str] = None) -> Episode:
        """Sample a uniformly random episode."""
        ep = int(self._rng.integers(0, self.num_episodes))
        return self.get_trajectory(ep, device=device)

    def get_trajectory(self, episode_idx: int, device: Optional[str] = None) -> Episode:
        """Return episode ``episode_idx`` as an :class:`Episode`."""
        if episode_idx < 0 or episode_idx >= self.num_episodes:
            raise IndexError(f"Episode index {episode_idx} out of range.")
        start, end = self._starts[episode_idx], self._ends[episode_idx]
        sl = slice(start, end)
        dev = self._device(device)
        goals = None
        if self._goals_t is not None:
            goals = self._goals_t[sl].to(dev)
        return Episode(
            states=self._states_t[sl].to(dev),
            actions=self._actions_t[sl].to(dev),
            next_states=self._next_states_t[sl].to(dev),
            rewards=self._rewards_t[sl].to(dev),
            terminals=self._terminals_t[sl].to(dev),
            timeouts=self._timeouts_t[sl].to(dev),
            goals=goals,
            index=episode_idx,
        )

    def get_raw_trajectory(self, episode_idx: int, device: Optional[str] = None) -> Episode:
        """Like :meth:`get_trajectory` but returns raw (unnormalized) states."""
        ep = self.get_trajectory(episode_idx, device=device)
        raw_states = torch.as_tensor(self._raw_states[self._starts[episode_idx] : self._ends[episode_idx]])
        raw_next = torch.as_tensor(self._raw_next_states[self._starts[episode_idx] : self._ends[episode_idx]])
        dev = self._device(device)
        ep.states = raw_states.to(dev)
        ep.next_states = raw_next.to(dev)
        return ep

    def sample_future_state_pairs(
        self,
        batch_size: int,
        device: Optional[str] = None,
        future_goal_prob: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample transitions and a future state from the same episode.

        Returns
        -------
        states, actions, goals
            ``goals`` is a future state chosen uniformly from ``[t, T_e)`` for
            each sampled transition, where ``T_e`` is the episode end.
        """
        dev = self._device(device)
        idx = self._rng.integers(0, self._n_transitions, size=batch_size)
        ep = self._episode_index[idx]
        starts = np.asarray(self._starts, dtype=np.int64)
        ends = np.asarray(self._ends, dtype=np.int64)
        future_idx = np.empty(batch_size, dtype=np.int64)
        use_self = self._rng.random(batch_size) > float(future_goal_prob)
        for i in range(batch_size):
            e = int(ep[i])
            lo = int(idx[i])
            hi = int(ends[e])
            if hi <= lo:
                hi = lo + 1
            # For a transition at index t, future state indices range from t to
            # end-1; the state at t is valid, but usually future goals are
            # sampled from (t, end).  Clamp if episode length is one.
            if use_self[i] or hi == lo:
                future_idx[i] = lo
            else:
                future_idx[i] = int(self._rng.integers(lo, hi))
        future_idx = np.minimum(future_idx, self._n_transitions - 1)
        states = self._states_t[torch.as_tensor(idx)].to(dev)
        actions = self._actions_t[torch.as_tensor(idx)].to(dev)
        goals = self._states_t[torch.as_tensor(future_idx)].to(dev)
        return states, actions, goals

    def sample_goal_transitions(
        self,
        batch_size: int,
        device: Optional[str] = None,
        future_goal_prob: float = 0.8,
    ) -> TransitionBatch:
        """Sample goal-conditioned transitions using hindsight relabeling.

        The goal is a future state from the same episode.  This is convenient
        for GC-IQL/GC-BC training.
        """
        dev = self._device(device)
        states, actions, goals = self.sample_future_state_pairs(
            batch_size, device=device, future_goal_prob=future_goal_prob
        )
        # Use a deterministic batch index for the other fields; this is slightly
        # redundant but keeps the returned TransitionBatch self-contained.
        idx = torch.randint(0, self._n_transitions, (batch_size,))
        batch = self._transition_batch_from_idx(idx, device=device)
        batch.states = states
        batch.actions = actions
        batch.goals = goals
        return batch

    def sample_hindsight_batch(
        self,
        batch_size: int,
        device: Optional[str] = None,
        future_goal_prob: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``(s, a, s', g, terminal)`` for goal-conditioned updates.

        Returns normalized states/next_states and future goals.
        """
        states, actions, goals = self.sample_future_state_pairs(
            batch_size, device=device, future_goal_prob=future_goal_prob
        )
        idx = self._rng.integers(0, self._n_transitions, size=batch_size)
        next_states = self._next_states_t[torch.as_tensor(idx)].to(self._device(device))
        terminals = self._terminals_t[torch.as_tensor(idx)].to(self._device(device))
        return states, actions, next_states, goals, terminals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_episode_returns(self) -> np.ndarray:
        returns = np.empty(self.num_episodes, dtype=np.float32)
        for ep in range(self.num_episodes):
            sl = slice(self._starts[ep], self._ends[ep])
            returns[ep] = float(self._rewards[sl].sum())
        return returns

    def to(self, device: str) -> "OfflineDataset":
        """Return a shallow copy with a different default device."""
        new = object.__new__(OfflineDataset)
        new.__dict__.update(self.__dict__)
        new.device = device
        return new


# Backwards-friendly alias used throughout the codebase.
Dataset = OfflineDataset


__all__ = [
    "ArrayLike",
    "TransitionBatch",
    "Episode",
    "OfflineDataset",
    "Dataset",
]
