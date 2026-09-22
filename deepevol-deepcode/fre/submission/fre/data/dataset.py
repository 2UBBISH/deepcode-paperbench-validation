"""Offline dataset loaders and trajectory indexing for FRE.

This module implements the data pipeline described in Appendix C of
"Zero-Shot Reinforcement Learning via Functional Reward Encodings":

* ``C.1 AntMaze``: the ``antmaze-large-diverse-v2`` dataset from D4RL.  The
  X / Y coordinates are discretized into 32 bins by the *encoder* preprocessing
  (see :mod:`fre.data.preprocessing`); the raw states are kept here.
* ``C.2 ExORL``: ``walker-*`` / ``cheetah-*`` RND datasets.  Because the true
  rewards of the Cheetah / Walker environments are functions of the underlying
  *physics*, auxiliary physics information is appended to the observations
  **for the encoder only**.  Each state dimension is normalized by its standard
  deviation inside the offline dataset, and augmented dimensions are ignored
  when computing goal distance.
* ``C.3 Kitchen``: the seven sparse D4RL Kitchen subtasks are used directly as
  evaluation tasks (their sparse rewards are the reward functions).

Everything in this file is deliberately dependency-light: ``d4rl``, ``gym``,
``h5py`` and ``dm_control`` are imported lazily so that the rest of the code
base (encoders, priors, IQL, evaluation) can be exercised without the full
simulation stack installed.  A :func:`synthetic_dataset` builder is provided so
that unit tests and smoke tests can run without any external dataset.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = [
    "OfflineDataset",
    "TransitionBatch",
    "build_trajectory_index",
    "future_indices",
    "load_dataset",
    "load_d4rl_dataset",
    "load_exorl_dataset",
    "load_antmaze_dataset",
    "load_kitchen_dataset",
    "load_npz_dataset",
    "synthetic_dataset",
    "ANTMAZE_DATASET",
    "KITCHEN_DATASET",
    "EXORL_DATASETS",
    "DATASET_REGISTRY",
]


# ---------------------------------------------------------------------------
# Constants / registry
# ---------------------------------------------------------------------------

#: D4RL dataset used for the AntMaze experiments (Appendix C.1).
ANTMAZE_DATASET = "antmaze-large-diverse-v2"

#: D4RL Kitchen dataset (Appendix C.3).
KITCHEN_DATASET = "kitchen-mixed-v0"

#: ExORL (RND) datasets used in the paper (Appendix C.2).  The paper trains on
#: the RND dataset for each domain.
EXORL_DATASETS = (
    "walker-run",
    "walker-walk",
    "cheetah-run",
    "cheetah-walk",
    "cheetah-run-backwards",
    "cheetah-walk-backwards",
)

#: Domain key -> {"kind", "env_name"} used by :func:`load_dataset`.
DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "antmaze": {"kind": "d4rl", "env_name": ANTMAZE_DATASET},
    "kitchen": {"kind": "d4rl", "env_name": KITCHEN_DATASET},
}
for _env in EXORL_DATASETS:
    DATASET_REGISTRY["exorl:" + _env] = {"kind": "exorl", "env_name": _env}


# ---------------------------------------------------------------------------
# Trajectory indexing helpers
# ---------------------------------------------------------------------------


def build_trajectory_index(
    dones: np.ndarray,
    timeouts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index transitions by trajectory id.

    Args:
        dones: boolean array of length ``N``; ``dones[i] = True`` marks that
            transition ``i`` is the last transition of its trajectory.
        timeouts: optional boolean array marking time-limit truncations, used
            only to sanity-check that ``dones`` already includes them.  (ExORL
            episodes end on a time limit, so both must be treated as terminal
            boundaries; see Appendix C.2.)

    Returns:
        ``(trajectory_ids, traj_starts, traj_ends)`` where ``trajectory_ids``
        has length ``N`` and ``traj_starts`` / ``traj_ends`` have length
        ``num_trajectories`` (``traj_ends`` is inclusive).
    """
    dones = np.asarray(dones).astype(bool).reshape(-1)
    n = dones.shape[0]
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty

    trajectory_ids = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(dones[:-1], dtype=np.int64)]
    )
    num_trajectories = int(trajectory_ids[-1]) + 1
    boundaries = np.arange(num_trajectories, dtype=np.int64)
    traj_starts = np.searchsorted(trajectory_ids, boundaries, side="left")
    traj_ends = np.searchsorted(trajectory_ids, boundaries, side="right") - 1

    if timeouts is not None:
        timeouts = np.asarray(timeouts).astype(bool).reshape(-1)
        if timeouts.shape[0] == n and timeouts.any():
            # A timeout also terminates a trajectory; warn (do not fail) if the
            # caller passed a `dones` array that ignores them.
            if not bool(np.all(dones[timeouts])):
                warnings.warn(
                    "`timeouts` are true at transitions that are not marked as "
                    "done; trajectory boundaries will follow `dones`.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return trajectory_ids, traj_starts, traj_ends


def future_indices(
    indices: np.ndarray,
    trajectory_ids: np.ndarray,
    traj_ends: np.ndarray,
    rng: np.random.Generator,
    min_offset: int = 1,
) -> np.ndarray:
    """Sample a *future* index inside the same trajectory for each index.

    Indices that have no future transition inside their trajectory are returned
    unchanged (i.e. they fall back to the "current" state).
    """
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return idx.copy()
    traj = trajectory_ids[idx]
    lo = idx + int(min_offset)
    hi = traj_ends[traj]
    out = idx.copy()
    valid = lo <= hi
    if np.any(valid):
        lo_v = lo[valid]
        hi_v = hi[valid]
        span = (hi_v - lo_v + 1).astype(np.float64)
        sampled = lo_v + np.floor(rng.random(lo_v.shape[0]) * span).astype(np.int64)
        out[valid] = np.minimum(sampled, hi_v)
    return out


# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------


@dataclass
class TransitionBatch:
    """A batch of offline transitions (all tensors have leading batch dim)."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    indices: Optional[torch.Tensor] = None
    trajectory_ids: Optional[torch.Tensor] = None

    def to(self, device: Union[str, torch.device]) -> "TransitionBatch":
        return TransitionBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_observations=self.next_observations.to(device),
            dones=self.dones.to(device),
            indices=None if self.indices is None else self.indices.to(device),
            trajectory_ids=(
                None if self.trajectory_ids is None else self.trajectory_ids.to(device)
            ),
        )

    def as_dict(self) -> Dict[str, torch.Tensor]:
        out = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones,
        }
        if self.indices is not None:
            out["indices"] = self.indices
        if self.trajectory_ids is not None:
            out["trajectory_ids"] = self.trajectory_ids
        return out

    def __len__(self) -> int:
        return int(self.observations.shape[0])


# ---------------------------------------------------------------------------
# Offline dataset
# ---------------------------------------------------------------------------


class OfflineDataset:
    """An offline dataset of transitions with trajectory bookkeeping.

    The dataset stores transitions ``(s, a, r, s', done)`` as contiguous numpy
    arrays plus a trajectory id per transition.  Trajectory ids make it possible
    to implement the hindsight (HER) goal distribution used by the FRE prior
    reward functions: ``0.2`` current state, ``0.5`` future state inside the
    trajectory, ``0.3`` uniformly random dataset state.

    Args:
        observations: ``(N, state_dim)`` float array.
        actions: ``(N, action_dim)`` float array.
        rewards: ``(N,)`` float array (optional, defaults to zeros).
        dones: ``(N,)`` bool array marking the last transition of a trajectory.
        timeouts: ``(N,)`` bool array of time-limit truncations (optional).
        trajectory_ids: precomputed trajectory ids (optional).
        state_mean / state_std: optional precomputed normalization statistics.
        name: human readable dataset name.
        augment_dim: number of *appended* physics dimensions (ExORL).  These are
            excluded from goal distances (Appendix C.2).
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        dones: Optional[np.ndarray] = None,
        timeouts: Optional[np.ndarray] = None,
        trajectory_ids: Optional[np.ndarray] = None,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        name: str = "dataset",
        augment_dim: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.float32)
        if self.observations.ndim != 2:
            raise ValueError(
                "`observations` must be 2-D (num_transitions, state_dim), got "
                f"shape {self.observations.shape}"
            )
        num_transitions = int(self.observations.shape[0])

        self.actions = np.asarray(actions, dtype=np.float32)
        if self.actions.ndim == 1:
            self.actions = self.actions[:, None]
        if self.actions.shape[0] != num_transitions:
            raise ValueError(
                "`actions` must have the same leading dimension as "
                f"`observations` ({self.actions.shape[0]} vs {num_transitions})"
            )

        if rewards is None:
            rewards = np.zeros(num_transitions, dtype=np.float32)
        self.rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if self.rewards.shape[0] != num_transitions:
            raise ValueError("`rewards` must have length num_transitions")

        if dones is None:
            # No episode information: treat the dataset as one long trajectory.
            dones = np.zeros(num_transitions, dtype=bool)
            if num_transitions:
                dones[-1] = True
        self.dones = np.asarray(dones).astype(bool).reshape(-1)
        if self.dones.shape[0] != num_transitions:
            raise ValueError("`dones` must have length num_transitions")

        self.timeouts = (
            None if timeouts is None else np.asarray(timeouts).astype(bool).reshape(-1)
        )

        if trajectory_ids is None:
            trajectory_ids, _, _ = build_trajectory_index(self.dones, self.timeouts)
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64).reshape(-1)

        self.name = name
        self.augment_dim = int(augment_dim)
        self.metadata: Dict[str, Any] = dict(metadata or {})

        self.state_mean = (
            None if state_mean is None else np.asarray(state_mean, dtype=np.float32)
        )
        self.state_std = (
            None if state_std is None else np.asarray(state_std, dtype=np.float32)
        )

        self._traj_starts: Optional[np.ndarray] = None
        self._traj_ends: Optional[np.ndarray] = None
        self._torch_cache: Dict[str, torch.Tensor] = {}

    # -- basic properties ---------------------------------------------------
    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_transitions(self) -> int:
        return int(self.observations.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_trajectories(self) -> int:
        return int(self.trajectory_ids[-1]) + 1 if len(self) else 0

    @property
    def traj_starts(self) -> np.ndarray:
        if self._traj_starts is None:
            _, self._traj_starts, self._traj_ends = build_trajectory_index(self.dones)
        return self._traj_starts

    @property
    def traj_ends(self) -> np.ndarray:
        if self._traj_ends is None:
            _, self._traj_starts, self._traj_ends = build_trajectory_index(self.dones)
        return self._traj_ends

    @property
    def goal_state_dim(self) -> int:
        """Number of state dims used for goal distance (drops augmentation)."""
        return self.state_dim - self.augment_dim

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OfflineDataset(name={self.name!r}, transitions={len(self)}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"trajectories={self.num_trajectories}, augment_dim={self.augment_dim})"
        )

    # -- transition access --------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        index = int(index)
        next_index = index + 1 if index + 1 < len(self) else index
        return {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "rewards": self.rewards[index : index + 1].astype(np.float32),
            "next_observations": self.observations[next_index],
            "dones": np.asarray(self.dones[index], dtype=np.float32),
        }

    def state_tensor(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        key = f"states::{device}::{dtype}"
        if key not in self._torch_cache:
            self._torch_cache[key] = torch.as_tensor(
                self.observations, dtype=dtype, device=device
            )
        return self._torch_cache[key]

    def reward_tensor(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        key = f"rewards::{device}::{dtype}"
        if key not in self._torch_cache:
            self._torch_cache[key] = torch.as_tensor(
                self.rewards, dtype=dtype, device=device
            )
        return self._torch_cache[key]

    def action_tensor(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        key = f"actions::{device}::{dtype}"
        if key not in self._torch_cache:
            self._torch_cache[key] = torch.as_tensor(
                self.actions, dtype=dtype, device=device
            )
        return self._torch_cache[key]

    # -- sampling -----------------------------------------------------------
    def sample_indices(
        self, batch_size: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.integers(0, len(self), size=int(batch_size), dtype=np.int64)

    def sample_states(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        return_indices: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Uniformly sample states from the dataset.

        This is the distribution ``D`` used by the FRE encoder / decoder to draw
        ``K = 32`` encoder states and ``K' = 8`` decoder states.
        """
        idx = self.sample_indices(batch_size, rng)
        states = self.observations[idx]
        if return_indices:
            return states, idx
        return states

    def sample_states_tensor(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, np.ndarray]:
        states, idx = self.sample_states(batch_size, rng, return_indices=True)
        return torch.as_tensor(states, device=device), idx

    def sample_transitions(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> TransitionBatch:
        """Uniformly sample transitions ``(s, a, r, s', done)`` for IQL."""
        idx = self.sample_indices(batch_size, rng)
        next_idx = np.minimum(idx + 1, len(self) - 1)
        # Keep (s, s') inside the same trajectory.
        same_traj = self.trajectory_ids[next_idx] == self.trajectory_ids[idx]
        next_idx = np.where(same_traj, next_idx, idx)
        return TransitionBatch(
            observations=torch.as_tensor(self.observations[idx], device=device),
            actions=torch.as_tensor(self.actions[idx], device=device),
            rewards=torch.as_tensor(self.rewards[idx][:, None], device=device),
            next_observations=torch.as_tensor(
                self.observations[next_idx], device=device
            ),
            dones=torch.as_tensor(
                self.dones[idx].astype(np.float32)[:, None], device=device
            ),
            indices=torch.as_tensor(idx, device=device),
            trajectory_ids=torch.as_tensor(self.trajectory_ids[idx], device=device),
        )

    def sample_trajectory_states(
        self,
        num_trajectories: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[np.ndarray]:
        """Return the full state sequence of ``num_trajectories`` trajectories."""
        rng = rng if rng is not None else np.random.default_rng()
        num = min(int(num_trajectories), self.num_trajectories)
        traj = rng.integers(0, self.num_trajectories, size=num)
        return [
            self.observations[self.traj_starts[t] : self.traj_ends[t] + 1] for t in traj
        ]

    def future_state_indices(
        self,
        indices: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        min_offset: int = 1,
    ) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return future_indices(
            indices, self.trajectory_ids, self.traj_ends, rng, min_offset
        )

    def future_states(
        self,
        indices: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        min_offset: int = 1,
    ) -> np.ndarray:
        return self.observations[self.future_state_indices(indices, rng, min_offset)]

    def sample_her_goals(
        self,
        indices: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        current_fraction: float = 0.2,
        future_fraction: float = 0.5,
    ) -> np.ndarray:
        """Sample goals with the HER distribution used by the FRE goal prior.

        The remaining probability mass (``1 - current - future``, i.e. ``0.3``
        in the paper, Appendix C / §4.2) is sampled uniformly from the dataset.
        """
        rng = rng if rng is not None else np.random.default_rng()
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return self.observations[idx]

        random_fraction = max(0.0, 1.0 - current_fraction - future_fraction)
        probs = np.array(
            [current_fraction, future_fraction, random_fraction], dtype=np.float64
        )
        probs = probs / probs.sum()
        choice = rng.choice(3, size=idx.shape[0], p=probs)

        goals = np.empty((idx.shape[0], self.state_dim), dtype=np.float32)
        goals[:] = self.observations[idx]
        future_mask = choice == 1
        if np.any(future_mask):
            goals[future_mask] = self.future_states(idx[future_mask], rng)
        random_mask = choice == 2
        if np.any(random_mask):
            goals[random_mask] = self.sample_states(int(random_mask.sum()), rng)
        return goals

    # -- normalization (Appendix C.2) ---------------------------------------
    def compute_normalization(
        self,
        max_samples: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-dimension mean / std over the offline dataset."""
        if max_samples is not None and max_samples < len(self):
            rng = rng if rng is not None else np.random.default_rng(0)
            idx = rng.choice(len(self), size=int(max_samples), replace=False)
            obs = self.observations[idx]
        else:
            obs = self.observations
        self.state_mean = obs.mean(axis=0).astype(np.float32)
        self.state_std = obs.std(axis=0).astype(np.float32)
        return self.state_mean, self.state_std

    def normalization(self, min_std: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        if self.state_mean is None or self.state_std is None:
            self.compute_normalization()
        std = np.maximum(self.state_std, float(min_std))
        return self.state_mean, std

    def normalized_observations(self, min_std: float = 1e-3) -> np.ndarray:
        mean, std = self.normalization(min_std)
        return (self.observations - mean) / std

    # -- sub-setting / persistence ------------------------------------------
    def subsample(
        self, num_transitions: int, rng: Optional[np.random.Generator] = None
    ) -> "OfflineDataset":
        rng = rng if rng is not None else np.random.default_rng(0)
        num = min(int(num_transitions), len(self))
        idx = np.sort(rng.choice(len(self), size=num, replace=False))
        return OfflineDataset(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
            timeouts=None if self.timeouts is None else self.timeouts[idx],
            state_mean=self.state_mean,
            state_std=self.state_std,
            name=self.name + f"-sub{num}",
            augment_dim=self.augment_dim,
            metadata=self.metadata,
        )

    def with_observations(
        self, observations: np.ndarray, name: Optional[str] = None, **metadata: Any
    ) -> "OfflineDataset":
        """Return a copy with replaced observations (e.g. physics-augmented)."""
        meta = dict(self.metadata)
        meta.update(metadata)
        return OfflineDataset(
            observations=observations,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            timeouts=self.timeouts,
            state_mean=self.state_mean,
            state_std=self.state_std,
            name=name or self.name,
            augment_dim=self.augment_dim,
            metadata=meta,
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "trajectory_ids": self.trajectory_ids,
            "augment_dim": np.int64(self.augment_dim),
            "name": np.array(self.name),
        }
        if self.timeouts is not None:
            payload["timeouts"] = self.timeouts
        if self.state_mean is not None and self.state_std is not None:
            payload["state_mean"] = self.state_mean
            payload["state_std"] = self.state_std
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "OfflineDataset":
        with np.load(path, allow_pickle=True) as data:
            name = str(data["name"]) if "name" in data else os.path.basename(path)
            return cls(
                observations=data["observations"],
                actions=data["actions"],
                rewards=data["rewards"] if "rewards" in data else None,
                dones=data["dones"] if "dones" in data else None,
                timeouts=data["timeouts"] if "timeouts" in data else None,
                trajectory_ids=(
                    data["trajectory_ids"] if "trajectory_ids" in data else None
                ),
                state_mean=data["state_mean"] if "state_mean" in data else None,
                state_std=data["state_std"] if "state_std" in data else None,
                name=name,
                augment_dim=int(data["augment_dim"]) if "augment_dim" in data else 0,
            )


# ---------------------------------------------------------------------------
# Raw-array helpers / flexible key lookup
# ---------------------------------------------------------------------------


def _lookup(container: Any, keys: Sequence[str]) -> Optional[np.ndarray]:
    """Return the first present key of ``keys`` inside a mapping-like object."""
    for key in keys:
        try:
            if key in container:
                return np.asarray(container[key])
        except TypeError:  # pragma: no cover - exotic containers
            continue
    return None


def _dataset_from_raw(
    raw: Dict[str, np.ndarray], name: str, **kwargs: Any
) -> OfflineDataset:
    observations = _lookup(raw, ["observations", "observation", "obs", "states"])
    actions = _lookup(raw, ["actions", "action"])
    if observations is None or actions is None:
        raise KeyError(
            f"Could not find observation/action arrays in {name}; available keys: "
            f"{sorted(raw.keys())}"
        )
    rewards = _lookup(raw, ["rewards", "reward"])
    dones = _lookup(raw, ["terminals", "dones", "done"])
    timeouts = _lookup(raw, ["timeouts", "timeout"])
    if dones is not None:
        dones = dones.reshape(-1)
        if dones.dtype.kind == "f":
            dones = dones > 0.5
    return OfflineDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        timeouts=timeouts,
        name=name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# D4RL loaders
# ---------------------------------------------------------------------------


def load_d4rl_dataset(
    env_name: str,
    dataset_path: Optional[str] = None,
    name: Optional[str] = None,
) -> OfflineDataset:
    """Load an AntMaze / Kitchen dataset from D4RL.

    The paper pins D4RL to a pre-June-2024 commit for reproducibility (see the
    Addendum and the README).  If ``dataset_path`` is given, the transitions are
    read from a cached ``.npz`` / ``.hdf5`` file instead of the ``d4rl`` package.
    """
    name = name or env_name
    if dataset_path is not None:
        return load_npz_dataset(dataset_path, name=name)

    try:  # pragma: no cover - requires d4rl/mujoco
        import gym  # noqa: F401  (D4RL registers its envs on import)
        import d4rl  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Loading D4RL datasets requires the `d4rl`, `gym` and `mujoco` "
            "packages (see requirements.txt). Alternatively pass an explicit "
            "`dataset_path` to load a cached .npz/.hdf5 file."
        ) from exc

    env = gym.make(env_name)  # pragma: no cover - requires mujoco
    raw: Dict[str, np.ndarray] = {}
    try:  # pragma: no cover
        dataset = env.get_dataset()
        raw = {k: np.asarray(v) for k, v in dataset.items() if k != "infos"}
    except Exception:  # pragma: no cover
        raw = {}
    if "observations" not in raw:  # pragma: no cover
        qdata = d4rl.qlearning_dataset(env)
        raw = {
            "observations": qdata["observations"],
            "actions": qdata["actions"],
            "rewards": qdata["rewards"],
            "terminals": qdata["terminals"],
        }
        if "timeouts" in qdata:
            raw["timeouts"] = qdata["timeouts"]
    return _dataset_from_raw(raw, name=name)


def load_kitchen_dataset(
    env_name: str = KITCHEN_DATASET, dataset_path: Optional[str] = None
) -> OfflineDataset:
    """Load the D4RL Kitchen dataset (Appendix C.3, ``kitchen-mixed-v0``)."""
    return load_d4rl_dataset(env_name, dataset_path=dataset_path, name=env_name)


def load_antmaze_dataset(
    env_name: str = ANTMAZE_DATASET,
    dataset_path: Optional[str] = None,
    discretize_xy: bool = False,
) -> OfflineDataset:
    """Load ``antmaze-large-diverse-v2`` (Appendix C.1).

    Args:
        discretize_xy: if True, run the FRE / GC-IQL / GC-BC / OPAL
            preprocessing of Appendix C.1 that discretizes the X and Y
            coordinates into 32 bins (implemented in
            :mod:`fre.data.preprocessing`).
    """
    dataset = load_d4rl_dataset(env_name, dataset_path=dataset_path, name=env_name)
    if discretize_xy:
        from fre.data.preprocessing import discretize_antmaze_xy  # local import

        dataset = discretize_antmaze_xy(dataset)
    return dataset


# ---------------------------------------------------------------------------
# ExORL loaders
# ---------------------------------------------------------------------------


def _find_exorl_file(directory: str, env_name: str) -> str:
    candidates = [
        os.path.join(directory, env_name + ".npz"),
        os.path.join(directory, env_name + ".hdf5"),
        os.path.join(directory, env_name + ".h5"),
        os.path.join(directory, env_name, "dataset.npz"),
        os.path.join(directory, env_name, "dataset.hdf5"),
        os.path.join(directory, env_name, "rnd.npz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # Fall back to any npz/hdf5 whose name contains the environment name.
    if os.path.isdir(directory):
        for entry in sorted(os.listdir(directory)):
            if env_name in entry and entry.endswith((".npz", ".hdf5", ".h5")):
                return os.path.join(directory, entry)
    raise FileNotFoundError(
        f"Could not locate an ExORL dataset for {env_name!r} inside {directory!r}. "
        "Download the ExORL RND datasets and point `--exorl-dir` at them."
    )


def load_exorl_dataset(
    env_name: str,
    dataset_dir: str,
    dataset_path: Optional[str] = None,
    augment_physics: bool = True,
    physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> OfflineDataset:
    """Load an ExORL (RND) dataset for ``walker-*`` / ``cheetah-*``.

    Appends the physics dimensions described in Appendix C.2 *for the encoder*:

    * Walker: ``horizontal_velocity()`` (2), ``torso_upright()`` (1),
      ``torso_height()`` (1)
    * Cheetah: ``speed()`` (1)

    The augmentation is skipped (with a warning) if the physics values cannot be
    obtained; ``physics_fn`` may be provided to compute them from observations.
    """
    path = dataset_path or _find_exorl_file(dataset_dir, env_name)
    dataset = load_npz_dataset(path, name=env_name, augment_dim=0)
    if not augment_physics:
        return dataset

    try:
        from fre.data.preprocessing import augment_exorl_physics

        return augment_exorl_physics(dataset, env_name=env_name, physics_fn=physics_fn)
    except Exception as exc:  # pragma: no cover - depends on external assets
        warnings.warn(
            f"ExORL physics augmentation unavailable ({exc}); using raw "
            "observations. Rewards for Cheetah/Walker tasks require the "
            "augmented physics dimensions (Appendix C.2).",
            RuntimeWarning,
            stacklevel=2,
        )
        return dataset


# ---------------------------------------------------------------------------
# Generic file loaders
# ---------------------------------------------------------------------------


def load_npz_dataset(
    path: str, name: Optional[str] = None, **kwargs: Any
) -> OfflineDataset:
    """Load a dataset from an ``.npz`` / ``.hdf5`` file with flexible keys."""
    name = name or os.path.splitext(os.path.basename(path))[0]
    if path.endswith(".npz") or path.endswith(".npy"):
        with np.load(path, allow_pickle=True) as data:
            raw = {k: np.asarray(data[k]) for k in data.files}
    else:
        try:  # pragma: no cover - requires h5py
            import h5py
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Reading {path} requires `h5py` (pip install h5py)."
            ) from exc
        raw = {}
        with h5py.File(path, "r") as handle:  # pragma: no cover
            def collect(prefix: str, group: Any) -> None:
                for key in group.keys():
                    item = group[key]
                    full = f"{prefix}{key}"
                    if hasattr(item, "keys"):
                        collect(full + "/", item)
                    else:
                        raw[full] = np.asarray(item)

            collect("", handle)

        def find(key_names: Sequence[str]) -> Optional[np.ndarray]:
            for key in sorted(raw):
                tail = key.split("/")[-1]
                if tail in key_names:
                    return raw[key]
            return None

        observations = find(["observations", "observation", "obs", "states"])
        actions = find(["actions", "action"])
        if observations is None or actions is None:
            raise KeyError(
                f"Could not find observation/action arrays in {path}; available "
                f"keys: {sorted(raw.keys())}"
            )
        rewards = find(["rewards", "reward"])
        dones = find(["terminals", "dones", "done"])
        timeouts = find(["timeouts", "timeout"])
        if dones is not None:
            dones = dones.reshape(-1)
            if dones.dtype.kind == "f":
                dones = dones > 0.5
        return OfflineDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            name=name,
            **kwargs,
        )
    return _dataset_from_raw(raw, name=name, **kwargs)


# ---------------------------------------------------------------------------
# Synthetic dataset (for tests / smoke runs)
# ---------------------------------------------------------------------------


def synthetic_dataset(
    num_transitions: int = 2000,
    state_dim: int = 29,
    action_dim: int = 8,
    episode_length: int = 100,
    seed: int = 0,
    augment_dim: int = 0,
    name: str = "synthetic",
) -> OfflineDataset:
    """Build a random dataset with realistic trajectory structure.

    Useful for unit-testing the pipeline without external datasets.  Rewards are
    drawn uniformly from ``[-1, 0]`` to mimic the goal-reaching prior.
    """
    rng = np.random.default_rng(seed)
    num_trajectories = max(1, num_transitions // max(1, episode_length))
    total = num_trajectories * episode_length

    # Smooth random walks so that "future state" goals are meaningful.
    steps = rng.normal(scale=0.1, size=(total, state_dim)).astype(np.float32)
    offsets = np.repeat(
        rng.normal(scale=1.0, size=(num_trajectories, state_dim)).astype(np.float32),
        episode_length,
        axis=0,
    )
    observations = np.cumsum(steps, axis=0) + offsets
    actions = rng.uniform(-1.0, 1.0, size=(total, action_dim)).astype(np.float32)
    rewards = rng.uniform(-1.0, 0.0, size=(total,)).astype(np.float32)
    dones = np.zeros(total, dtype=bool)
    dones[episode_length - 1 :: episode_length] = True
    return OfflineDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        name=name,
        augment_dim=augment_dim,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def load_dataset(domain: str = "antmaze", **kwargs: Any) -> OfflineDataset:
    """Load the offline dataset for a domain.

    Args:
        domain: one of ``"antmaze"``, ``"kitchen"`` or ``"exorl:<env>"`` (e.g.
            ``"exorl:walker-run"``), matching the per-domain configs in
            ``fre/configs/*.yaml``.
        **kwargs: passed through to the underlying loader, e.g.
            ``dataset_path=...``, ``exorl_dir=...``, ``discretize_xy=True``.

    Returns:
        The loaded :class:`OfflineDataset`.
    """
    key = domain.lower()
    exorl_dir = kwargs.pop("exorl_dir", kwargs.pop("dataset_dir", None))
    dataset_path = kwargs.pop("dataset_path", None)

    if key in ("antmaze", "antmaze-large-diverse-v2"):
        return load_antmaze_dataset(dataset_path=dataset_path, **kwargs)
    if key in ("kitchen", "kitchen-mixed-v0"):
        return load_kitchen_dataset(dataset_path=dataset_path, **kwargs)
    for prefix in ("exorl:", "exorl/"):
        if key.startswith(prefix):
            env_name = key.split(prefix, 1)[1]
            if exorl_dir is None:
                raise ValueError(
                    "Loading ExORL datasets requires `exorl_dir=<path to datasets>`."
                )
            return load_exorl_dataset(
                env_name, exorl_dir, dataset_path=dataset_path, **kwargs
            )
    raise KeyError(
        f"Unknown dataset domain {domain!r}. Known domains: "
        f"{sorted(DATASET_REGISTRY)} (+ 'exorl:<env>')."
    )
