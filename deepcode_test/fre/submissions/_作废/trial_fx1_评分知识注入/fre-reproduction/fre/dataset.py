"""Offline dataset loading and sampling utilities.

This module provides a lightweight offline dataset wrapper used by both the
FRE agent and the baseline algorithms.  It stores state-action transition
data and a separate state-only pool used for:

* sampling reward-prior states while training the FRE encoder
* sampling context states when encoding reward functions into latents

The wrapper supports D4RL AntMaze, D4RL Kitchen, and ExORL walker/cheetah
exploratory datasets.  States are normalized to zero mean and unit variance
using dataset statistics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor, List, Tuple]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a list, tuple, torch tensor, or ndarray to float32 numpy."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    return arr


def _is_arraylike(x: Any) -> bool:
    return isinstance(x, (np.ndarray, torch.Tensor, list, tuple))


class OfflineDataset:
    """Offline transition dataset with normalized states and state pool.

    Parameters
    ----------
    states: np.ndarray
        Array of shape ``(N, state_dim)``.
    actions: np.ndarray
        Array of shape ``(N, action_dim)``.
    next_states: Optional[np.ndarray]
        Array of shape ``(N, state_dim)``.  If omitted, next_states is copied
        from ``states`` and terminals is set to all ones.
    rewards: Optional[np.ndarray]
        Array of shape ``(N,)`` or ``(N, 1)``.
    terminals: Optional[np.ndarray]
        Array of shape ``(N,)`` or ``(N, 1)``, 1 indicates episode termination.
    state_pool: Optional[np.ndarray]
        Prebuilt pool of states for encoder/reward-prior sampling.  If omitted,
        the pool is built from the concatenation of states and next_states.
    normalize_states: bool
        Whether to normalize states using dataset statistics.
    state_mean: Optional[np.ndarray]
        Precomputed mean.  If ``normalize_states`` is True and this is None,
        it is estimated from states and next_states.
    state_std: Optional[np.ndarray]
        Precomputed std.  If ``normalize_states`` is True and this is None,
        it is estimated from states and next_states.
    """

    def __init__(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        next_states: Optional[ArrayLike] = None,
        rewards: Optional[ArrayLike] = None,
        terminals: Optional[ArrayLike] = None,
        state_pool: Optional[ArrayLike] = None,
        normalize_states: bool = True,
        state_mean: Optional[ArrayLike] = None,
        state_std: Optional[ArrayLike] = None,
    ) -> None:
        states = _to_numpy(states)
        actions = _to_numpy(actions)

        if states.ndim == 1:
            states = states.reshape(1, -1)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Mismatched dataset lengths: {states.shape[0]} vs {actions.shape[0]}"
            )

        if next_states is None:
            next_states = states.copy()
        else:
            next_states = _to_numpy(next_states)
            if next_states.ndim == 1:
                next_states = next_states.reshape(1, -1)
            if next_states.shape[0] != states.shape[0]:
                raise ValueError("next_states length must match states")

        if rewards is None:
            rewards = np.zeros(states.shape[0], dtype=np.float32)
        else:
            rewards = _to_numpy(rewards).reshape(-1)

        if terminals is None:
            terminals = np.zeros(states.shape[0], dtype=np.float32)
        else:
            terminals = _to_numpy(terminals).reshape(-1)

        if rewards.shape[0] != states.shape[0]:
            rewards = np.broadcast_to(rewards, (states.shape[0],)).astype(np.float32)
        if terminals.shape[0] != states.shape[0]:
            terminals = np.broadcast_to(terminals, (states.shape[0],)).astype(np.float32)

        # Estimate normalization statistics before mutating states.
        all_states = np.concatenate([states, next_states], axis=0)
        if normalize_states:
            if state_mean is None:
                state_mean = all_states.mean(axis=0, keepdims=False)
            else:
                state_mean = _to_numpy(state_mean)
            if state_std is None:
                state_std = all_states.std(axis=0, keepdims=False) + 1e-6
            else:
                state_std = _to_numpy(state_std) + 1e-6
            states = (states - state_mean) / state_std
            next_states = (next_states - state_mean) / state_std
            if state_pool is not None:
                state_pool = _to_numpy(state_pool)
                state_pool = (state_pool - state_mean) / state_std
        else:
            state_mean = np.zeros(states.shape[1], dtype=np.float32)
            state_std = np.ones(states.shape[1], dtype=np.float32)
            if state_pool is not None:
                state_pool = _to_numpy(state_pool)

        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.next_states = next_states.astype(np.float32)
        self.rewards = rewards.astype(np.float32)
        self.terminals = terminals.astype(np.float32)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.normalize_states = normalize_states

        if state_pool is None:
            state_pool = all_states
        self.state_pool = _to_numpy(state_pool).astype(np.float32)

        self._size = int(states.shape[0])
        self.state_dim = int(states.shape[1])
        self.action_dim = int(actions.shape[1])

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def sample_batch(self, batch_size: int, device: str = "cpu") -> Dict[str, np.ndarray]:
        """Sample a random transition batch.

        Returns a dictionary with keys ``states``, ``actions``,
        ``next_states``, ``rewards``, and ``terminals``.  Values are numpy
        arrays by default; the FRE agent converts them to tensors internally.
        """
        batch_size = min(batch_size, self._size)
        idx = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "next_states": self.next_states[idx],
            "rewards": self.rewards[idx],
            "terminals": self.terminals[idx],
        }
        if device != "cpu":
            batch = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in batch.items()}
        return batch

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, np.ndarray]:
        """Alias for :meth:`sample_batch`."""
        return self.sample_batch(batch_size, device=device)

    def sample_states(self, batch_size: int, device: str = "cpu") -> np.ndarray:
        """Sample raw states from the state-only pool."""
        batch_size = min(batch_size, self.state_pool.shape[0])
        idx = np.random.randint(0, self.state_pool.shape[0], size=batch_size)
        states = self.state_pool[idx]
        if device != "cpu":
            states = torch.as_tensor(states, device=device, dtype=torch.float32)
        return states

    def sample_state_pool(self, batch_size: int, device: str = "cpu") -> np.ndarray:
        """Alias for :meth:`sample_states`."""
        return self.sample_states(batch_size, device=device)

    def to_torch(self, device: str = "cpu") -> "TorchOfflineDataset":
        """Return a torch-tensor-backed wrapper for fast sampling."""
        return TorchOfflineDataset(
            torch.as_tensor(self.states, device=device, dtype=torch.float32),
            torch.as_tensor(self.actions, device=device, dtype=torch.float32),
            torch.as_tensor(self.next_states, device=device, dtype=torch.float32),
            torch.as_tensor(self.rewards, device=device, dtype=torch.float32),
            torch.as_tensor(self.terminals, device=device, dtype=torch.float32),
            torch.as_tensor(self.state_pool, device=device, dtype=torch.float32),
        )


class TorchOfflineDataset:
    """GPU-friendly dataset wrapper storing all arrays as torch tensors."""

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        state_pool: torch.Tensor,
    ) -> None:
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.terminals = terminals
        self.state_pool = state_pool
        self._size = int(states.shape[0])
        self.state_dim = int(states.shape[1])
        self.action_dim = int(actions.shape[1])

    def __len__(self) -> int:
        return self._size

    def sample_batch(self, batch_size: int, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        batch_size = min(batch_size, self._size)
        idx = torch.randint(0, self._size, (batch_size,), device=self.states.device)
        batch = {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "next_states": self.next_states[idx],
            "rewards": self.rewards[idx],
            "terminals": self.terminals[idx],
        }
        if device is not None and device != self.states.device:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    def sample(self, batch_size: int, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        return self.sample_batch(batch_size, device=device)

    def sample_states(self, batch_size: int, device: Optional[str] = None) -> torch.Tensor:
        batch_size = min(batch_size, self.state_pool.shape[0])
        idx = torch.randint(0, self.state_pool.shape[0], (batch_size,), device=self.state_pool.device)
        states = self.state_pool[idx]
        if device is not None and device != states.device:
            states = states.to(device)
        return states


# ---------------------------------------------------------------------------
# Domain loaders
# ---------------------------------------------------------------------------

def _load_d4rl_qlearning(env_name: str) -> Dict[str, np.ndarray]:
    """Load a D4RL dataset via ``d4rl.qlearning_dataset``.

    Falls back to loading from a local file if D4RL is unavailable.
    """
    try:
        import d4rl  # noqa: F401
        import gym

        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        env.close()
        return dataset
    except Exception as exc:  # pragma: no cover - fallback path
        raise RuntimeError(
            f"Failed to load D4RL dataset '{env_name}'. "
            "Install d4rl and MuJoCo, or provide a local dataset file."
        ) from exc


def load_d4rl_antmaze(dataset_name: str = "antmaze-large-diverse-v2") -> OfflineDataset:
    """Load an AntMaze D4RL dataset.

    The default environment ``antmaze-large-diverse-v2`` matches the paper's
    AntMaze-large-diverse benchmark.  States contain (x, y, z, theta, ...) and
    actions are 8-dimensional.
    """
    data = _load_d4rl_qlearning(dataset_name)
    return OfflineDataset(
        states=data["observations"],
        actions=data["actions"],
        next_states=data["next_observations"],
        rewards=data["rewards"],
        terminals=data.get("terminals", data.get("dones")),
        normalize_states=True,
    )


def load_d4rl_kitchen(dataset_name: str = "kitchen-complete-v0") -> OfflineDataset:
    """Load a D4RL Kitchen dataset.

    Kitchen states are D4RL's 30-dimensional state representation and the
    downstream tasks correspond to 7 standard subtasks.  Rewards in the stored
    dataset are not used by FRE; task rewards are recomputed at evaluation
    time in :mod:`envs.kitchen_wrapper`.
    """
    data = _load_d4rl_qlearning(dataset_name)
    return OfflineDataset(
        states=data["observations"],
        actions=data["actions"],
        next_states=data["next_observations"],
        rewards=data["rewards"],
        terminals=data.get("terminals", data.get("dones")),
        normalize_states=True,
    )


def _load_exorl_hdf5(path: str) -> Dict[str, np.ndarray]:
    """Load an ExORL HDF5 exploratory dataset.

    ExORL releases data as HDF5 files with keys ``observations``,
    ``actions``, ``rewards``, ``discounts`` (or ``terminals``), and
    ``next_observations``.  This function supports that common layout.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading ExORL datasets requires h5py.") from exc

    with h5py.File(path, "r") as f:
        observations = f["observations"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]
        if "next_observations" in f:
            next_observations = f["next_observations"][:]
        else:
            # Reconstruct from observations by shifting one timestep.
            next_observations = np.empty_like(observations)
            next_observations[:-1] = observations[1:]
            next_observations[-1] = observations[-1]
        if "terminals" in f:
            terminals = f["terminals"][:]
        elif "discounts" in f:
            discounts = f["discounts"][:]
            terminals = 1.0 - discounts
        else:
            terminals = np.zeros(len(observations), dtype=np.float32)

    return {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
        "terminals": terminals,
    }


def load_exorl(domain: str, dataset_path: str) -> OfflineDataset:
    """Load an ExORL exploratory dataset from an HDF5 file.

    Parameters
    ----------
    domain: str
        One of ``"walker"`` or ``"cheetah"`` (used only for metadata).
    dataset_path: str
        Path to the ExORL HDF5 dataset file.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ExORL dataset not found: {dataset_path}")

    data = _load_exorl_hdf5(dataset_path)
    return OfflineDataset(
        states=data["observations"],
        actions=data["actions"],
        next_states=data["next_observations"],
        rewards=data["rewards"],
        terminals=data["terminals"],
        normalize_states=True,
    )


def load_offline_dataset(
    domain: str,
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> OfflineDataset:
    """Convenience dispatch for loading benchmark datasets.

    Parameters
    ----------
    domain:
        ``"antmaze"``, ``"kitchen"``, ``"walker"``, or ``"cheetah"``.
    dataset_name:
        For D4RL domains, the gym dataset/env name.
    dataset_path:
        For ExORL domains, path to an HDF5 dataset file.

    Returns
    -------
    OfflineDataset
    """
    domain = domain.lower()
    if domain == "antmaze":
        return load_d4rl_antmaze(dataset_name or "antmaze-large-diverse-v2")
    if domain == "kitchen":
        return load_d4rl_kitchen(dataset_name or "kitchen-complete-v0")
    if domain in {"walker", "cheetah"}:
        if dataset_path is None:
            raise ValueError("ExORL datasets require dataset_path.")
        return load_exorl(domain, dataset_path)
    raise ValueError(f"Unknown domain: {domain}")


def build_state_pool(dataset: OfflineDataset, max_pool_size: Optional[int] = None) -> np.ndarray:
    """Return a state-only pool used for reward-prior and encoder sampling.

    The pool is the dataset's prebuilt ``state_pool``, optionally subsampled.
    """
    pool = dataset.state_pool
    if max_pool_size is not None and pool.shape[0] > max_pool_size:
        idx = np.random.choice(pool.shape[0], size=max_pool_size, replace=False)
        pool = pool[idx]
    return pool


def make_synthetic_dataset(
    state_dim: int = 17,
    action_dim: int = 8,
    size: int = 10_000,
    seed: int = 0,
) -> OfflineDataset:
    """Build a small synthetic dataset for debugging and unit tests."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(size, state_dim)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(size, action_dim)).astype(np.float32)
    next_states = states + 0.1 * rng.normal(size=(size, state_dim)).astype(np.float32)
    rewards = rng.uniform(-1.0, 1.0, size=size).astype(np.float32)
    terminals = (rng.uniform(size=size) < 0.02).astype(np.float32)
    return OfflineDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        terminals=terminals,
        normalize_states=True,
    )
