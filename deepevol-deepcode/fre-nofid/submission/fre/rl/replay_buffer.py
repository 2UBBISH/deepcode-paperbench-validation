"""Replay buffers for FRE offline / z-conditioned IQL training.

The FRE pipeline trains phase-2 IQL fully offline over a fixed dataset
(AntMaze ``antmaze-large-diverse-v2``, Kitchen, or the ExORL RND exploratory
datasets).  Because FRE re-samples a *reward function* ``eta ~ p(eta)`` from the
reward prior at every RL iteration and evaluates ``r = eta(s)`` on the sampled
batch, the buffer only needs to serve ``(s, a, s', done)`` transitions cheaply.
Stored dataset rewards are kept for diagnostics and for the supervised
baselines (GC-BC).

This module is intentionally dependency-light: it only needs ``numpy`` and
``torch``.  Offline datasets are stored as contiguous numpy arrays (no python
object references), so memory usage is one ``float32`` copy per field.

Public interface
----------------
* :class:`Batch`               - dict-like + attribute-accessible tensor batch.
* :class:`ReplayBuffer`        - numpy-backed offline replay buffer.
* :class:`OfflineReplayBuffer` - alias used by training drivers.
* :class:`TrajectoryReplayBuffer` - buffer that also exposes episode boundaries
  (needed for goal-reaching HER sampling and velocity diagnostics).
* :func:`make_batch`, :func:`concatenate_buffers`, :func:`d4rl_dict_to_arrays`
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is required for batching but keep import guarded for error clarity
    import torch
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "fre.rl.replay_buffer requires PyTorch (pip install torch>=1.13)"
    ) from _exc


__all__ = [
    "Batch",
    "ReplayBuffer",
    "OfflineReplayBuffer",
    "TrajectoryReplayBuffer",
    "make_batch",
    "concatenate_buffers",
    "d4rl_dict_to_arrays",
    "TRANSITION_KEYS",
]


TRANSITION_KEYS: Tuple[str, ...] = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
)

# Common aliases found in D4RL / ExORL / custom datasets.
_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "observations": ("observations", "obs", "observation", "o"),
    "actions": ("actions", "action", "a"),
    "rewards": ("rewards", "reward", "r"),
    "next_observations": (
        "next_observations",
        "next_obs",
        "obs_next",
        "observations_next",
        "next_observation",
    ),
    "terminals": ("terminals", "terminal", "dones", "done", "masks"),
}

DEFAULT_CAPACITY: int = 2_000_000


# --------------------------------------------------------------------------- #
# Batch container
# --------------------------------------------------------------------------- #
class Batch(Mapping):
    """A tensor batch that behaves like a dict *and* like a namespace.

    Supports both ``batch["observations"]`` and ``batch.observations`` so any
    consumer style in the training drivers keeps working.

    Example
    -------
    >>> batch = Batch({"observations": torch.zeros(4, 3)})
    >>> batch.observations.shape, batch["observations"].shape
    (torch.Size([4, 3]), torch.Size([4, 3]))
    """

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        merged: Dict[str, Any] = {}
        if data is not None:
            if isinstance(data, Batch):
                merged.update(data._data)
            elif isinstance(data, Mapping):
                merged.update(data)
            else:  # pragma: no cover - defensive
                raise TypeError(f"Batch expects a mapping, got {type(data)!r}")
        merged.update(kwargs)
        object.__setattr__(self, "_data", merged)

    # -- Mapping protocol -------------------------------------------------- #
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:  # pragma: no cover - trivial
        return key in self._data

    # -- attribute access -------------------------------------------------- #
    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        data = object.__getattribute__(self, "_data")
        if item in data:
            return data[item]
        raise AttributeError(f"Batch has no field '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value

    # -- conveniences ------------------------------------------------------ #
    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def to(
        self,
        device: Union[str, "torch.device"] = "cpu",
        dtype: Optional["torch.dtype"] = None,
        non_blocking: bool = False,
    ) -> "Batch":
        """Move every tensor field to ``device`` (optionally casting dtype)."""
        out: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, torch.Tensor):
                v = value.to(device=device, non_blocking=non_blocking)
                if dtype is not None and v.is_floating_point():
                    v = v.to(dtype=dtype)
                out[key] = v
            else:
                out[key] = value
        return Batch(out)

    def cpu(self) -> "Batch":
        return self.to("cpu")

    def detach(self) -> "Batch":
        out = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in self._data.items()
        }
        return Batch(out)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parts = []
        for key, value in self._data.items():
            if isinstance(value, torch.Tensor):
                parts.append(f"{key}={tuple(value.shape)}")
            else:
                parts.append(f"{key}={type(value).__name__}")
        return f"Batch({', '.join(parts)})"


def _to_tensor(value: Any, device: Union[str, "torch.device"]) -> Any:
    """Convert numpy arrays / scalars to torch tensors, leave tensors alone."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        if value.dtype == np.float64:
            value = value.astype(np.float32, copy=False)
        return torch.as_tensor(np.ascontiguousarray(value), device=device)
    if isinstance(value, (np.floating, float)):
        return torch.tensor(float(value), device=device, dtype=torch.float32)
    if isinstance(value, (np.integer, int, bool, np.bool_)):
        return torch.tensor(int(value), device=device)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(np.asarray(value), device=device)
    return value


def make_batch(
    arrays: Mapping[str, Any],
    device: Union[str, "torch.device"] = "cpu",
    dtype: Optional["torch.dtype"] = torch.float32,
) -> Batch:
    """Build a :class:`Batch` from a mapping of numpy arrays / tensors."""
    out = {}
    for key, value in arrays.items():
        tensor = _to_tensor(value, device)
        if (
            dtype is not None
            and isinstance(tensor, torch.Tensor)
            and tensor.is_floating_point()
        ):
            tensor = tensor.to(dtype=dtype)
        out[key] = tensor
    return Batch(out)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_keys(dataset: Mapping[str, Any]) -> Dict[str, str]:
    """Map canonical transition keys onto the keys present in ``dataset``."""
    available = set(dataset.keys())
    resolved: Dict[str, str] = {}
    for canonical, aliases in _KEY_ALIASES.items():
        for alias in aliases:
            if alias in available:
                resolved[canonical] = alias
                break
    return resolved


def d4rl_dict_to_arrays(
    dataset: Mapping[str, Any],
    require_rewards: bool = False,
    infer_terminals: bool = True,
) -> Dict[str, np.ndarray]:
    """Normalise a D4RL / ExORL style dataset dict into canonical numpy arrays.

    Returns a dict with the keys in :data:`TRANSITION_KEYS` (``rewards`` and
    ``terminals`` are synthesised with zeros when absent and
    ``infer_terminals``/``require_rewards`` allow it).
    """
    resolved = _resolve_keys(dataset)
    if "observations" not in resolved:
        raise KeyError(
            f"dataset is missing observations; found keys: {sorted(dataset.keys())}"
        )
    if "actions" not in resolved:
        raise KeyError(
            f"dataset is missing actions; found keys: {sorted(dataset.keys())}"
        )

    obs = np.asarray(dataset[resolved["observations"]], dtype=np.float32)
    actions = np.asarray(dataset[resolved["actions"]], dtype=np.float32)

    if "next_observations" in resolved:
        next_obs = np.asarray(dataset[resolved["next_observations"]], dtype=np.float32)
    else:  # synthesise by shifting, with a copy of the last observation
        next_obs = np.empty_like(obs)
        next_obs[:-1] = obs[1:]
        next_obs[-1] = obs[-1]

    if "rewards" in resolved:
        rewards = np.asarray(dataset[resolved["rewards"]], dtype=np.float32).reshape(-1)
    elif require_rewards:
        raise KeyError("dataset is missing rewards")
    else:
        rewards = np.zeros(len(obs), dtype=np.float32)

    if "terminals" in resolved:
        terminal_key = resolved["terminals"]
        raw = np.asarray(dataset[terminal_key])
        terminals = raw.reshape(-1).astype(np.float32)
        if terminal_key == "masks":
            # D4RL "masks" use 1 - done convention.
            terminals = 1.0 - terminals
        terminals = (terminals > 0.5).astype(np.float32)
    elif infer_terminals:
        # D4RL datasets without explicit terminals: mark the very last step.
        terminals = np.zeros(len(obs), dtype=np.float32)
        if len(terminals) > 0:
            terminals[-1] = 1.0
    else:
        terminals = np.zeros(len(obs), dtype=np.float32)

    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_obs,
        "terminals": terminals,
    }


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Numpy-backed replay buffer for offline (and optionally online) RL.

    Parameters
    ----------
    capacity:
        Maximum number of transitions.  Offline data is truncated to the first
        ``capacity`` transitions (with a warning) when it does not fit.
    obs_dim, action_dim:
        Optional dimensionality hints used when appending single transitions to
        an empty buffer.
    device:
        Device used when materialising sampled batches.
    seed:
        Seed for the internal :class:`numpy.random.Generator`.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        device: Union[str, "torch.device"] = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)

        # Storage is allocated lazily so an offline dataset can be assigned
        # wholesale without a redundant copy.
        self._arrays: Dict[str, np.ndarray] = {}
        self._size = 0
        self._ptr = 0

        # Optional observation normalisation (ExORL std-normalisation).
        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std: Optional[np.ndarray] = None
        self._normalisation_applied = False

        # Lazily computed episode index (for HER goal sampling).
        self._trajectory_id: Optional[np.ndarray] = None
        self._timestep: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_arrays(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        terminals: Optional[np.ndarray] = None,
        device: Union[str, "torch.device"] = "cpu",
        seed: Optional[int] = None,
        capacity: Optional[int] = None,
    ) -> "ReplayBuffer":
        buf = cls(
            capacity=int(capacity) if capacity is not None else max(len(observations), 1),
            obs_dim=np.asarray(observations).shape[-1],
            action_dim=np.asarray(actions).shape[-1],
            device=device,
            seed=seed,
        )
        buf.set_arrays(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            terminals=terminals,
        )
        return buf

    @classmethod
    def from_dataset(
        cls,
        dataset: Mapping[str, Any],
        device: Union[str, "torch.device"] = "cpu",
        seed: Optional[int] = None,
        require_rewards: bool = False,
        capacity: Optional[int] = None,
        infer_terminals: bool = True,
    ) -> "ReplayBuffer":
        arrays = d4rl_dict_to_arrays(
            dataset, require_rewards=require_rewards, infer_terminals=infer_terminals
        )
        return cls.from_arrays(device=device, seed=seed, capacity=capacity, **arrays)

    # ------------------------------------------------------------------ #
    # storage
    # ------------------------------------------------------------------ #
    def set_arrays(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        terminals: Optional[np.ndarray] = None,
        replace: bool = True,
    ) -> None:
        """Assign (offline) arrays directly, skipping per-transition copies."""
        obs = np.ascontiguousarray(np.asarray(observations, dtype=np.float32))
        acts = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
        n = len(obs)

        if next_observations is None:
            next_obs = np.empty_like(obs)
            if n > 1:
                next_obs[:-1] = obs[1:]
            if n > 0:
                next_obs[-1] = obs[-1]
        else:
            next_obs = np.ascontiguousarray(
                np.asarray(next_observations, dtype=np.float32)
            )
        rew = (
            np.zeros(n, dtype=np.float32)
            if rewards is None
            else np.ascontiguousarray(np.asarray(rewards, dtype=np.float32).reshape(-1))
        )
        term = (
            np.zeros(n, dtype=np.float32)
            if terminals is None
            else np.ascontiguousarray(np.asarray(terminals).reshape(-1)).astype(
                np.float32
            )
        )

        if not (len(obs) == len(acts) == len(next_obs) == len(rew) == len(term)):
            raise ValueError(
                "inconsistent dataset lengths: "
                f"obs={len(obs)} act={len(acts)} next={len(next_obs)} "
                f"rew={len(rew)} term={len(term)}"
            )

        if n > self.capacity:
            obs, acts, next_obs, rew, term = (
                obs[: self.capacity],
                acts[: self.capacity],
                next_obs[: self.capacity],
                rew[: self.capacity],
                term[: self.capacity],
            )
            n = self.capacity

        if replace:
            self._arrays = {}
        self._arrays["observations"] = obs
        self._arrays["actions"] = acts
        self._arrays["next_observations"] = next_obs
        self._arrays["rewards"] = rew
        self._arrays["terminals"] = term
        self.obs_dim = int(obs.shape[-1]) if obs.ndim >= 1 and n > 0 else self.obs_dim
        self.action_dim = (
            int(acts.shape[-1]) if acts.ndim >= 1 and n > 0 else self.action_dim
        )
        self._size = n
        self._ptr = 0
        self._invalidate_episodes()

    def _invalidate_episodes(self) -> None:
        self._trajectory_id = None
        self._timestep = None

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float = 0.0,
        next_observation: Optional[np.ndarray] = None,
        terminal: Union[bool, float] = False,
        **extra: Any,
    ) -> None:
        """Append a single transition (circular overwrite at capacity)."""
        if not self._arrays:
            obs_dim = int(np.asarray(observation).reshape(-1).shape[0])
            act_dim = int(np.asarray(action).reshape(-1).shape[0])
            self._arrays = {
                "observations": np.zeros((self.capacity, obs_dim), dtype=np.float32),
                "actions": np.zeros((self.capacity, act_dim), dtype=np.float32),
                "next_observations": np.zeros((self.capacity, obs_dim), dtype=np.float32),
                "rewards": np.zeros((self.capacity,), dtype=np.float32),
                "terminals": np.zeros((self.capacity,), dtype=np.float32),
            }
            self.obs_dim, self.action_dim = obs_dim, act_dim
            self._size = 0
            self._ptr = 0

        idx = self._ptr
        self._arrays["observations"][idx] = np.asarray(observation, dtype=np.float32)
        self._arrays["actions"][idx] = np.asarray(action, dtype=np.float32)
        if next_observation is None:
            next_observation = observation
        self._arrays["next_observations"][idx] = np.asarray(
            next_observation, dtype=np.float32
        )
        self._arrays["rewards"][idx] = float(reward)
        self._arrays["terminals"][idx] = float(terminal)
        for key, value in extra.items():
            self._add_extra(idx, key, value)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._invalidate_episodes()

    def add_batch(self, batch: Mapping[str, Any]) -> None:
        """Append a batch of transitions stored in a dict of arrays."""
        arrays = {k: np.asarray(v) for k, v in batch.items()}
        lengths = {len(v) for v in arrays.values() if np.ndim(v) > 0}
        if len(lengths) > 1:
            raise ValueError(f"inconsistent batch lengths: {lengths}")
        n = lengths.pop() if lengths else 1
        for i in range(n):
            row = {k: (v[i] if np.ndim(v) > 0 else v) for k, v in arrays.items()}
            self.add(**row)

    def extend(self, other: "ReplayBuffer") -> None:
        """Append every transition of another buffer."""
        self.add_batch(other.as_numpy())

    def _add_extra(self, idx: int, key: str, value: Any) -> None:
        arr = self._arrays.get(key)
        if arr is None:
            value_arr = np.asarray(value)
            shape = (self.capacity,) + value_arr.shape
            self._arrays[key] = np.zeros(shape, dtype=np.float64 if value_arr.size == 0 else value_arr.dtype)
            arr = self._arrays[key]
        self._arrays[key][idx] = value

    # ------------------------------------------------------------------ #
    # observation normalisation
    # ------------------------------------------------------------------ #
    def compute_observation_normalization(
        self, eps: float = 1e-3, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-dimension mean/std of the stored observations."""
        obs = self._arrays["observations"][: self._size]
        mean = obs.mean(axis=0).astype(np.float32)
        std = obs.std(axis=0).astype(np.float32)
        std = np.maximum(std, eps)
        self.obs_mean, self.obs_std = mean, std
        if verbose:
            print(
                f"[replay_buffer] obs normalisation: mean|{mean.shape}| "
                f"std in [{std.min():.4f}, {std.max():.4f}]"
            )
        return mean, std

    def set_observation_normalization(
        self, mean: np.ndarray, std: np.ndarray, apply: bool = False
    ) -> None:
        self.obs_mean = np.asarray(mean, dtype=np.float32)
        self.obs_std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
        if apply:
            self.apply_observation_normalization()

    def apply_observation_normalization(self) -> None:
        """In-place ``(obs - mean) / std`` on observations *and* next-states."""
        if self.obs_mean is None or self.obs_std is None:
            raise RuntimeError("call compute/set_observation_normalization first")
        if self._normalisation_applied:
            return
        for key in ("observations", "next_observations"):
            self._arrays[key] = (
                (self._arrays[key] - self.obs_mean) / self.obs_std
            ).astype(np.float32, copy=False)
        self._normalisation_applied = True

    def normalize_observations(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is None or self.obs_std is None:
            return np.asarray(obs, dtype=np.float32)
        return ((np.asarray(obs, dtype=np.float32) - self.obs_mean) / self.obs_std).astype(
            np.float32
        )

    def denormalize_observations(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is None or self.obs_std is None:
            return np.asarray(obs, dtype=np.float32)
        return (np.asarray(obs, dtype=np.float32) * self.obs_std + self.obs_mean).astype(
            np.float32
        )

    # ------------------------------------------------------------------ #
    # episode structure
    # ------------------------------------------------------------------ #
    def _build_episode_index(self) -> None:
        """Derive trajectory ids / timesteps from terminal flags (lazy)."""
        if self._trajectory_id is not None:
            return
        terminals = self._arrays["terminals"][: self._size]
        is_start = np.zeros(self._size, dtype=bool)
        if self._size:
            is_start[0] = True
            if self._size > 1:
                is_start[1:] = terminals[:-1] > 0.5
        traj_id = np.cumsum(is_start) - 1
        traj_id = np.maximum(traj_id, 0)
        # timestep within episode
        timestep = np.arange(self._size, dtype=np.int64)
        starts = np.flatnonzero(is_start)
        if len(starts):
            base = np.repeat(starts, np.diff(np.append(starts, self._size)))
            if len(base) == self._size:
                timestep = timestep - base
        self._trajectory_id = traj_id.astype(np.int64)
        self._timestep = timestep.astype(np.int64)

    @property
    def trajectory_id(self) -> np.ndarray:
        self._build_episode_index()
        return self._trajectory_id  # type: ignore[return-value]

    @property
    def timestep(self) -> np.ndarray:
        self._build_episode_index()
        return self._timestep  # type: ignore[return-value]

    @property
    def trajectory_ids(self) -> np.ndarray:
        return np.unique(self.trajectory_id)

    def trajectory_indices(self, traj_id: int) -> np.ndarray:
        return np.flatnonzero(self.trajectory_id == traj_id)

    def num_trajectories(self) -> int:
        return int(len(self.trajectory_ids))

    def episode_ends(self) -> np.ndarray:
        """Indices at which stored trajectories end (inclusive)."""
        tid = self.trajectory_id
        if self._size == 0:
            return np.zeros(0, dtype=np.int64)
        change = np.flatnonzero(np.diff(tid) != 0)
        return np.concatenate([change, [self._size - 1]])

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def sample_indices(
        self, batch_size: int, replace: bool = True
    ) -> np.ndarray:
        return self._rng.integers(0, self._size, size=int(batch_size))

    def sample(
        self,
        batch_size: int,
        device: Optional[Union[str, "torch.device"]] = None,
        keys: Optional[Sequence[str]] = None,
        mask_terminal_next_obs: bool = False,
        as_batch: bool = True,
        dtype: Optional["torch.dtype"] = torch.float32,
    ) -> Union[Batch, Dict[str, Any]]:
        """Sample a random transition batch.

        Parameters
        ----------
        mask_terminal_next_obs:
            When ``True``, terminal transitions get ``next_obs = obs`` (the
            standard offline-RL masking that prevents bootstrapping past the
            end of an episode).
        """
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        idx = self.sample_indices(batch_size)
        return self.sample_by_index(
            idx,
            device=device,
            keys=keys,
            mask_terminal_next_obs=mask_terminal_next_obs,
            as_batch=as_batch,
            dtype=dtype,
        )

    def sample_by_index(
        self,
        indices: np.ndarray,
        device: Optional[Union[str, "torch.device"]] = None,
        keys: Optional[Sequence[str]] = None,
        mask_terminal_next_obs: bool = False,
        as_batch: bool = True,
        dtype: Optional["torch.dtype"] = torch.float32,
        with_indices: bool = True,
    ) -> Union[Batch, Dict[str, Any]]:
        device = self.device if device is None else device
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        wanted = list(keys) if keys is not None else list(self._arrays.keys())

        out: Dict[str, Any] = {}
        for key in wanted:
            arr = self._arrays.get(key)
            if arr is None:
                continue
            out[key] = arr[idx]
        if mask_terminal_next_obs and "terminals" in out and "next_observations" in out:
            term = np.asarray(out["terminals"]).reshape(-1) > 0.5
            if term.any():
                nxt = np.array(out["next_observations"], copy=True)
                nxt[term] = np.asarray(out["observations"])[term]
                out["next_observations"] = nxt
        out["indices"] = idx

        batch = make_batch(out, device=device, dtype=dtype)
        return batch if as_batch else batch.as_dict()

    def sample_trajectory_batch(
        self,
        batch_size: int,
        device: Optional[Union[str, "torch.device"]] = None,
        mask_terminal_next_obs: bool = False,
    ) -> Batch:
        """Same as :meth:`sample` but always includes trajectory/timestep info."""
        return self.sample(
            batch_size,
            device=device,
            mask_terminal_next_obs=mask_terminal_next_obs,
            as_batch=True,
        )

    def sample_context(
        self,
        batch_size: int,
        context_size: int,
        device: Optional[Union[str, "torch.device"]] = None,
        keys: Sequence[str] = ("observations",),
    ) -> Batch:
        """Sample ``(B, K)`` index grids, e.g. for goal-reaching HER context."""
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        traj = self.trajectory_id
        starts = np.flatnonzero(
            np.concatenate([[True], np.diff(traj) != 0])
        )
        ends = self.episode_ends()
        traj_of_start = traj[starts]
        chosen = self._rng.integers(0, len(starts), size=int(batch_size))
        out: Dict[str, Any] = {}
        idx = np.zeros((batch_size, context_size), dtype=np.int64)
        for b, t in enumerate(chosen):
            lo, hi = int(starts[t]), int(ends[t])
            if hi <= lo:
                hi = lo + 1
            idx[b] = self._rng.integers(lo, hi, size=int(context_size))
        for key in keys:
            arr = self._arrays.get(key)
            if arr is None:
                continue
            out[key] = arr[idx]
        out["indices"] = idx
        out["_trajectory_id_of_start"] = traj_of_start[chosen]
        return make_batch(out, device=self.device if device is None else device)

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #
    def as_numpy(
        self,
        keys: Optional[Sequence[str]] = None,
        copy: bool = False,
    ) -> Dict[str, np.ndarray]:
        keys = list(keys) if keys is not None else list(self._arrays.keys())
        out = {}
        for key in keys:
            arr = self._arrays.get(key)
            if arr is None:
                continue
            sliced = arr[: self._size] if key in TRANSITION_KEYS else arr
            out[key] = sliced.copy() if copy else sliced
        return out

    def as_tensors(
        self,
        device: Optional[Union[str, "torch.device"]] = None,
        keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, "torch.Tensor"]:
        device = self.device if device is None else device
        return {
            k: torch.as_tensor(v, device=device) for k, v in self.as_numpy(keys).items()
        }

    def get(self, key: str, indices: Optional[np.ndarray] = None) -> np.ndarray:
        arr = self._arrays[key]
        if indices is None:
            return arr[: self._size] if key in TRANSITION_KEYS else arr
        return arr[np.asarray(indices)]

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: object) -> bool:
        return key in self._arrays

    @property
    def size(self) -> int:
        return self._size

    @property
    def keys(self) -> List[str]:
        return list(self._arrays.keys())

    def observations(self) -> np.ndarray:
        return self._arrays["observations"][: self._size]

    def actions(self) -> np.ndarray:
        return self._arrays["actions"][: self._size]

    def rewards(self) -> np.ndarray:
        return self._arrays["rewards"][: self._size]

    def next_observations(self) -> np.ndarray:
        return self._arrays["next_observations"][: self._size]

    def terminals(self) -> np.ndarray:
        return self._arrays["terminals"][: self._size]

    @property
    def observation_dim(self) -> Optional[int]:
        return (
            int(self._arrays["observations"].shape[-1]) if self._arrays else self.obs_dim
        )

    @property
    def action_dim_actual(self) -> Optional[int]:
        return int(self._arrays["actions"].shape[-1]) if self._arrays else self.action_dim

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        payload = {
            "capacity": self.capacity,
            "size": self._size,
            "arrays": self.as_numpy(copy=True),
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "normalisation_applied": self._normalisation_applied,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str, device: Union[str, "torch.device"] = "cpu") -> "ReplayBuffer":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        buf = cls(
            capacity=payload.get("capacity", DEFAULT_CAPACITY),
            obs_dim=payload.get("obs_dim"),
            action_dim=payload.get("action_dim"),
            device=device,
        )
        arrays = payload["arrays"]
        buf.set_arrays(
            observations=arrays["observations"],
            actions=arrays["actions"],
            next_observations=arrays.get("next_observations"),
            rewards=arrays.get("rewards"),
            terminals=arrays.get("terminals"),
        )
        buf.obs_mean = payload.get("obs_mean")
        buf.obs_std = payload.get("obs_std")
        buf._normalisation_applied = payload.get("normalisation_applied", False)
        return buf

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "observations": self._arrays["observations"],
            "actions": self._arrays["actions"],
            "rewards": self._arrays["rewards"],
            "next_observations": self._arrays["next_observations"],
            "terminals": self._arrays["terminals"],
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.set_arrays(
            observations=state["observations"],
            actions=state["actions"],
            next_observations=state.get("next_observations"),
            rewards=state.get("rewards"),
            terminals=state.get("terminals"),
        )
        self.obs_mean = state.get("obs_mean")
        self.obs_std = state.get("obs_std")

    # ------------------------------------------------------------------ #
    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(size={self._size}, capacity={self.capacity}, "
            f"obs_dim={self.observation_dim}, action_dim={self.action_dim_actual}, "
            f"device={self.device})"
        )


# Alias used by the training drivers (`OfflineReplayBuffer`).
class OfflineReplayBuffer(ReplayBuffer):
    """Replay buffer specialised for fixed offline datasets.

    Identical to :class:`ReplayBuffer` but the default constructor arguments
    reflect the offline use case (no circular writes, capacity inferred from the
    dataset).
    """

    def __init__(
        self,
        dataset: Optional[Mapping[str, Any]] = None,
        device: Union[str, "torch.device"] = "cpu",
        seed: Optional[int] = None,
        capacity: Optional[int] = None,
        normalize_observations: bool = False,
        require_rewards: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(capacity=capacity or DEFAULT_CAPACITY, device=device, seed=seed)
        self.normalize = bool(normalize_observations)
        if dataset is not None:
            self.load_dataset(
                dataset,
                capacity=capacity,
                normalize_observations=normalize_observations,
                require_rewards=require_rewards,
                verbose=verbose,
            )

    def load_dataset(
        self,
        dataset: Mapping[str, Any],
        capacity: Optional[int] = None,
        normalize_observations: Optional[bool] = None,
        require_rewards: bool = False,
        verbose: bool = False,
    ) -> None:
        arrays = d4rl_dict_to_arrays(dataset, require_rewards=require_rewards)
        if capacity is not None:
            self.capacity = int(capacity)
        else:
            self.capacity = max(len(arrays["observations"]), 1)
        self.set_arrays(**arrays)
        if normalize_observations is None:
            normalize_observations = self.normalize
        if normalize_observations:
            self.compute_observation_normalization(verbose=verbose)
            self.apply_observation_normalization()

    @classmethod
    def from_dataset(  # type: ignore[override]
        cls,
        dataset: Mapping[str, Any],
        device: Union[str, "torch.device"] = "cpu",
        seed: Optional[int] = None,
        capacity: Optional[int] = None,
        normalize_observations: bool = False,
        require_rewards: bool = False,
        verbose: bool = False,
    ) -> "OfflineReplayBuffer":
        return cls(
            dataset=dataset,
            device=device,
            seed=seed,
            capacity=capacity,
            normalize_observations=normalize_observations,
            require_rewards=require_rewards,
            verbose=verbose,
        )


class TrajectoryReplayBuffer(ReplayBuffer):
    """Replay buffer that materialises trajectory indices eagerly.

    Used when the reward prior needs episode structure (goal-reaching HER
    sampling with p(current)=0.2, p(future)=0.5, p(random)=0.3), and for the
    ExORL goal tasks that pick fixed goal states per trajectory.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def set_arrays(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        super().set_arrays(*args, **kwargs)
        self._build_episode_index()

    def goal_indices(self) -> np.ndarray:
        """One goal index per stored trajectory (its final state)."""
        return self.episode_ends()

    def sample_goals(
        self,
        indices: np.ndarray,
        strategy: str = "geometric",
        p_current: float = 0.2,
        p_future: float = 0.5,
        p_random: float = 0.3,
    ) -> np.ndarray:
        """Sample goal *indices* for the given state indices.

        Mirrors the goal-reaching prior: ``p_current`` (goal = current state),
        ``p_future`` (a state later in the same trajectory, geometric bias),
        and ``p_random`` (any state in the dataset).
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        traj = self.trajectory_id
        ends = self.episode_ends()
        ends_of_traj = ends  # ends[i] corresponds to i-th trajectory in order
        goals = np.empty_like(indices)
        u = self._rng.random(len(indices))
        for i, idx in enumerate(indices):
            r = u[i]
            if r < p_current or strategy == "current":
                goals[i] = idx
            elif r < p_current + p_future or strategy == "future":
                t = traj[idx]
                lo, hi = int(ends_of_traj[t]), int(ends_of_traj[t])
                # start of the trajectory
                start = int(ends_of_traj[t - 1] + 1) if t > 0 else 0
                if strategy == "geometric" and self._rng.random() < 0.5:
                    # geometric/backward bias towards the trajectory end
                    frac = self._rng.random() ** 0.5
                    cand = int(idx + frac * (hi - idx))
                else:
                    cand = int(self._rng.integers(min(idx, hi), hi + 1))
                goals[i] = min(max(cand, start), hi)
            else:
                goals[i] = int(self._rng.integers(0, self._size))
        return goals

    def sample_goal_observations(
        self,
        indices: np.ndarray,
        strategy: str = "geometric",
    ) -> np.ndarray:
        return self.get("observations", self.sample_goals(indices, strategy=strategy))


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def concatenate_buffers(
    buffers: Sequence[ReplayBuffer],
    device: Union[str, "torch.device"] = "cpu",
    seed: Optional[int] = None,
) -> ReplayBuffer:
    """Concatenate several buffers (e.g. ExORL walker + cheetah shards)."""
    buffers = [b for b in buffers if b is not None and len(b) > 0]
    if not buffers:
        raise ValueError("need at least one non-empty buffer to concatenate")
    key_set = {k for b in buffers for k in b._arrays}
    arrays: Dict[str, np.ndarray] = {}
    for key in key_set:
        parts = [
            (b.as_numpy([key])[key] if key in b._arrays else None) for b in buffers
        ]
        have = [p for p in parts if p is not None]
        if len(have) != len(buffers):
            # Fill missing entries with zeros of matching trailing shape.
            ref = next(p for p in parts if p is not None)
            shape = (0,) + ref.shape[1:]
            have = []
            for p in parts:
                if p is None:
                    have.append(np.zeros(shape, dtype=ref.dtype))
                else:
                    have.append(p)
        arrays[key] = np.concatenate(have, axis=0)
    total = sum(len(b) for b in buffers)
    buf = ReplayBuffer(capacity=total, device=device, seed=seed)
    buf.set_arrays(
        observations=arrays["observations"],
        actions=arrays["actions"],
        next_observations=arrays.get("next_observations"),
        rewards=arrays.get("rewards"),
        terminals=arrays.get("terminals"),
    )
    return buf


def batch_to_device(
    batch: Union[Batch, Mapping[str, Any]],
    device: Union[str, "torch.device"],
    dtype: Optional["torch.dtype"] = None,
) -> Batch:
    """Move an arbitrary (dict or :class:`Batch`) batch to ``device``."""
    if isinstance(batch, Batch):
        return batch.to(device, dtype=dtype)
    return Batch(batch).to(device, dtype=dtype)


def add_reward_to_batch(
    batch: Batch,
    reward: "torch.Tensor",
    key: str = "rewards",
) -> Batch:
    """Attach (or replace) the reward field of a batch, e.g. ``r = eta(s)``."""
    out = batch.as_dict()
    out[key] = reward
    return Batch(out)


def dataset_observation_stats(
    dataset: Mapping[str, Any], eps: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience helper returning (mean, std) of a raw dataset dict."""
    obs = np.asarray(
        dataset[_resolve_keys(dataset)["observations"]], dtype=np.float32
    )
    return obs.mean(axis=0).astype(np.float32), np.maximum(
        obs.std(axis=0).astype(np.float32), eps
    )
