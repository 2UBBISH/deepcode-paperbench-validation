"""D4RL Kitchen offline dataset utilities for FRE.

Paper references
----------------
Source: C.3. Kitchen (Appendix C, "Environments and Evaluation Tasks"):

    "For the second domain, we utilize the seven standard subtasks within the D4RL
    Kitchen environment. Because each task already defines a sparse reward, we directly
    use those sparse rewards as evaluation tasks."

Source: A. Hyperparameters (Table 3): the Kitchen domain uses the "1M for
ExORL/Kitchen" step budget for both encoder and policy training.

Source: 5.2 (Table 1 caption / protocol): "All methods are evaluated using a mean over
twenty evaluation episodes, and each agent is trained using five random seeds".

This module mirrors :mod:`fre.data.antmaze_dataset` and :mod:`fre.data.exorl_dataset`:
it loads the raw D4RL Kitchen transition arrays (or an exported HDF5/NPZ dump), computes
per-dimension dataset statistics used for observation normalization, exposes the seven
sparse subtask reward functions used at zero-shot evaluation time, and builds the
unlabeled :class:`~fre.data.replay.ReplayBuffer` consumed by the FRE encoder/decoder and
the IQL policy trainer.

Deviations / unspecified details (marked "Source: not specified in the paper")
-----------------------------------------------------------------------------
* Concrete dataset id: the paper only says "the D4RL Kitchen environment".  We default
  to ``kitchen-complete-v0`` and accept the other standard ids (``kitchen-partial-v0``,
  ``kitchen-mixed-v0``).  The offline transitions are unlabeled in FRE anyway (only the
  observations/actions are used), so the choice affects only the state distribution.
* Episode length: the appendix is silent for Kitchen, so we use the D4RL Kitchen
  default of 1000 steps.
* Task achievement threshold: D4RL's own Kitchen environment treats a subtask as
  achieved when the relevant ``qpos`` dimensions are within a L2 distance of 0.3 of the
  goal values; that constant and the element index/goal tables below are copied from the
  D4RL Kitchen environment definition and are overridable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Intra-package import (with a fallback so the module can also be executed or
# imported directly without installing the package).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import plumbing
    from fre.data.replay import Batch, ReplayBuffer, Trajectory, make_replay_buffer  # type: ignore
except Exception:  # pragma: no cover - direct execution fallback
    Batch = ReplayBuffer = Trajectory = make_replay_buffer = None  # type: ignore
    try:
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        _PKG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
        if _PKG_ROOT not in sys.path:
            sys.path.insert(0, _PKG_ROOT)
        from fre.data.replay import (  # type: ignore
            Batch,
            ReplayBuffer,
            Trajectory,
            make_replay_buffer,
        )
    except Exception:  # pragma: no cover - keep the module importable regardless
        pass


__all__ = [
    # constants
    "KITCHEN_TASKS",
    "KITCHEN_DATASET_NAME",
    "KITCHEN_DATASET_NAMES",
    "KITCHEN_DATASET_PATHS_ENV",
    "KITCHEN_EVAL_EPISODE_LENGTH",
    "KITCHEN_TASK_THRESHOLD",
    "KITCHEN_ELEMENT_INDICES",
    "KITCHEN_ELEMENT_GOALS",
    "KITCHEN_ELEMENT_THRESHOLDS",
    "KITCHEN_TASK_ELEMENTS",
    "KITCHEN_OBS_DIM",
    # reward functions
    "element_reached",
    "compute_task_reward",
    "compute_all_task_rewards",
    "make_task_reward_function",
    "resolve_task_name",
    # dataset
    "KitchenDataset",
    "make_kitchen_dataset",
    "load_kitchen_arrays",
    "load_kitchen_dataset",
]


# ===========================================================================
# Constants: the seven standard D4RL Kitchen subtasks
# ===========================================================================

#: The seven standard subtasks of the D4RL Kitchen domain (Source: C.3. Kitchen).
KITCHEN_TASKS: Tuple[str, ...] = (
    "microwave",
    "kettle",
    "light switch",
    "slide cabinet",
    "bottom burner",
    "top burner",
    "hinge cabinet",
)

#: Convenience aliases (normalized spellings) -> canonical task name.
_TASK_ALIASES: Dict[str, str] = {
    "microwave": "microwave",
    "kettle": "kettle",
    "light switch": "light switch",
    "light_switch": "light switch",
    "lightswitch": "light switch",
    "slide cabinet": "slide cabinet",
    "slide_cabinet": "slide cabinet",
    "slidecabinet": "slide cabinet",
    "slider": "slide cabinet",
    "bottom burner": "bottom burner",
    "bottom_burner": "bottom burner",
    "bottomburner": "bottom burner",
    "top burner": "top burner",
    "top_burner": "top burner",
    "topburner": "top burner",
    "hinge cabinet": "hinge cabinet",
    "hinge_cabinet": "hinge cabinet",
    "hingecabinet": "hinge cabinet",
}

#: Observation indices (into the 60-d Kitchen ``qpos``-based observation) that define
#: each subtask, taken from the D4RL Kitchen environment definition.
#: (Source: not specified in the paper; D4RL constants.)
KITCHEN_ELEMENT_INDICES: Dict[str, Tuple[int, ...]] = {
    "bottom burner": (11, 12),
    "top burner": (15, 16),
    "light switch": (17, 18),
    "slide cabinet": (19,),
    "hinge cabinet": (20, 21),
    "microwave": (22,),
    "kettle": (23, 24, 25, 26, 27, 28, 29),
}

#: Goal values for the indexed dimensions of each subtask.
#: (Source: not specified in the paper; D4RL constants.)
KITCHEN_ELEMENT_GOALS: Dict[str, Tuple[float, ...]] = {
    "bottom burner": (-0.88, -0.01),
    "top burner": (-0.92, -0.01),
    "light switch": (-0.69, -0.05),
    "slide cabinet": (0.37,),
    "hinge cabinet": (0.0, 1.45),
    "microwave": (-0.75,),
    "kettle": (-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06),
}

#: L2 tolerance below which a subtask counts as solved.  D4RL's Kitchen environment
#: uses 0.3 for the element-completion checks.  (Source: not specified in the paper.)
KITCHEN_TASK_THRESHOLD: float = 0.3

KITCHEN_ELEMENT_THRESHOLDS: Dict[str, float] = {
    task: KITCHEN_TASK_THRESHOLD for task in KITCHEN_TASKS
}

#: Each evaluation subtask is defined by the set of elements that must be achieved.
#: The seven standard subtasks are single-element tasks.
KITCHEN_TASK_ELEMENTS: Dict[str, Tuple[str, ...]] = {
    task: (task,) for task in KITCHEN_TASKS
}

#: Raw observation dimension of the D4RL Kitchen environment.
KITCHEN_OBS_DIM: int = 60

#: Standard D4RL Kitchen dataset ids.
KITCHEN_DATASET_NAMES: Dict[str, str] = {
    "complete": "kitchen-complete-v0",
    "partial": "kitchen-partial-v0",
    "mixed": "kitchen-mixed-v0",
}

#: Default dataset id.  (Source: not specified in the paper.)
KITCHEN_DATASET_NAME: str = "kitchen-complete-v0"

#: Environment variables pointing at an exported Kitchen dump (``.npz``/``.hdf5``).
KITCHEN_DATASET_PATHS_ENV: Tuple[str, ...] = (
    "FRE_KITCHEN_DATASET",
    "FRE_KITCHEN_DATASET_PATH",
)

#: Maximum episode length for evaluation.  (Source: not specified in the paper;
#: D4RL Kitchen episodes are 1000 steps.)
KITCHEN_EVAL_EPISODE_LENGTH: int = 1000


# ===========================================================================
# Sparse subtask reward functions
# ===========================================================================


def resolve_task_name(task: str) -> str:
    """Normalize ``task`` to one of :data:`KITCHEN_TASKS`.

    Raises ``ValueError`` for unknown task names.
    """
    key = str(task).strip().lower().replace("-", " ")
    if key in _TASK_ALIASES:
        return _TASK_ALIASES[key]
    # allow "goal-<task>" style spellings used by evaluators
    for prefix in ("goal ", "eval "):
        if key.startswith(prefix):
            candidate = _TASK_ALIASES.get(key[len(prefix) :])
            if candidate is not None:
                return candidate
    raise ValueError(f"Unknown Kitchen task {task!r}; expected one of {KITCHEN_TASKS}")


def _as_2d_float(observations: Any) -> np.ndarray:
    """Coerce ``observations`` to a 2-D float array (promoting single states)."""
    if hasattr(observations, "detach"):  # torch tensor support
        observations = observations.detach().cpu().numpy()
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D observations, got shape {obs.shape}")
    return obs


def element_reached(
    observations: Any,
    element: str,
    threshold: float = KITCHEN_TASK_THRESHOLD,
) -> np.ndarray:
    """Return a boolean array marking states where ``element`` is at its goal.

    A state counts as achieved when the Euclidean distance between the element's
    observation dimensions and the corresponding goal values is within ``threshold``.
    Observations missing the element's dimensions return an all-``False`` result
    instead of raising, so reduced observation spaces degrade gracefully.
    """
    obs = _as_2d_float(observations)
    indices = KITCHEN_ELEMENT_INDICES.get(element)
    if indices is None:
        return np.zeros(obs.shape[0], dtype=bool)
    if obs.shape[1] <= max(indices):
        return np.zeros(obs.shape[0], dtype=bool)
    goal = np.asarray(KITCHEN_ELEMENT_GOALS[element], dtype=np.float64)
    diff = obs[:, list(indices)] - goal[None, :]
    distance = np.sqrt(np.sum(diff * diff, axis=1))
    return distance <= float(threshold)


def compute_task_reward(
    observations: Any,
    task: str,
    threshold: Optional[float] = None,
    reward_success: float = 1.0,
    reward_failure: float = 0.0,
    elements: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Sparse reward of one Kitchen subtask.

    The paper states that "each task already defines a sparse reward, we directly use
    those sparse rewards as evaluation tasks" (Source: C.3. Kitchen), i.e. the reward is
    ``reward_success`` (default 1) once the subtask's element(s) are achieved and
    ``reward_failure`` (default 0) otherwise.
    """
    task_name = resolve_task_name(task)
    element_names = tuple(elements) if elements is not None else KITCHEN_TASK_ELEMENTS[task_name]
    rewards: Optional[np.ndarray] = None
    for element in element_names:
        step_threshold = (
            float(threshold)
            if threshold is not None
            else float(KITCHEN_ELEMENT_THRESHOLDS.get(element, KITCHEN_TASK_THRESHOLD))
        )
        achieved = element_reached(observations, element, step_threshold)
        rewards = achieved if rewards is None else np.logical_and(rewards, achieved)
    if rewards is None:
        obs = _as_2d_float(observations)
        return np.full(obs.shape[0], float(reward_failure))
    return np.where(rewards, float(reward_success), float(reward_failure))


def compute_all_task_rewards(
    observations: Any,
    tasks: Sequence[str] = KITCHEN_TASKS,
    threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Return a mapping ``task -> reward array`` for every requested subtask."""
    return {name: compute_task_reward(observations, name, threshold=threshold) for name in tasks}


def make_task_reward_function(
    task: str,
    threshold: Optional[float] = None,
) -> Callable[[Any], np.ndarray]:
    """Return a callable ``observations -> rewards`` for a single Kitchen subtask."""
    task_name = resolve_task_name(task)

    def reward_function(observations: Any) -> np.ndarray:
        return compute_task_reward(observations, task_name, threshold=threshold)

    reward_function.__name__ = f"kitchen_reward[{task_name}]"
    return reward_function


# ===========================================================================
# Dataset
# ===========================================================================


@dataclass
class KitchenDataset:
    """D4RL Kitchen offline dataset with the seven sparse subtask rewards.

    The dataset is *unlabeled* from FRE's perspective: only ``observations`` and
    ``actions`` (plus trajectory boundaries) are used to train the encoder/decoder and
    the IQL agent.  The sparse subtask rewards are only applied when the FRE encoder
    encodes the evaluation task and when the evaluation rollout is scored.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: Optional[np.ndarray] = None
    terminals: Optional[np.ndarray] = None
    timeouts: Optional[np.ndarray] = None
    next_observations: Optional[np.ndarray] = None
    ends: Optional[np.ndarray] = None
    dataset_name: str = KITCHEN_DATASET_NAME
    tasks: Tuple[str, ...] = KITCHEN_TASKS
    task_threshold: float = KITCHEN_TASK_THRESHOLD
    eval_episode_length: int = KITCHEN_EVAL_EPISODE_LENGTH
    normalize: bool = False
    state_mean: Optional[np.ndarray] = None
    state_std: Optional[np.ndarray] = None
    seed: int = 0
    buffer: Optional[Any] = None
    _stats: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        observations: Any,
        actions: Any,
        rewards: Any = None,
        terminals: Any = None,
        timeouts: Any = None,
        next_observations: Any = None,
        ends: Any = None,
        **kwargs: Any,
    ) -> "KitchenDataset":
        """Build a dataset from flat D4RL-style arrays."""
        obs = _as_2d_float(observations)
        acts = np.asarray(actions, dtype=np.float32)
        if acts.ndim == 1:
            acts = acts[:, None]
        if ends is None:
            ends = _compute_ends(terminals, timeouts, next_observations, obs.shape[0])
        dataset = cls(
            observations=obs.astype(np.float32),
            actions=acts.astype(np.float32),
            rewards=None if rewards is None else np.asarray(rewards, dtype=np.float32),
            terminals=None if terminals is None else np.asarray(terminals, dtype=np.float32),
            timeouts=None if timeouts is None else np.asarray(timeouts, dtype=np.float32),
            next_observations=(
                None if next_observations is None else _as_2d_float(next_observations).astype(np.float32)
            ),
            ends=np.asarray(ends, dtype=np.int64),
            **kwargs,
        )
        return dataset

    def __post_init__(self) -> None:
        self.observations = _as_2d_float(self.observations).astype(np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.actions.ndim == 1:
            self.actions = self.actions[:, None]
        if self.ends is None:
            self.ends = _compute_ends(self.terminals, self.timeouts, self.next_observations, self.observations.shape[0])
        self.ends = np.asarray(self.ends, dtype=np.int64)
        self.tasks = tuple(self.tasks) if self.tasks is not None else KITCHEN_TASKS
        if self.state_mean is not None:
            self.state_mean = np.asarray(self.state_mean, dtype=np.float64)
        if self.state_std is not None:
            self.state_std = np.asarray(self.state_std, dtype=np.float64)

    # -- basic geometry ----------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def act_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_transitions(self) -> int:
        return int(self.observations.shape[0])

    def __len__(self) -> int:
        return self.num_transitions

    @property
    def num_trajectories(self) -> int:
        return int(self.ends.size)

    @property
    def num_states(self) -> int:
        """Number of distinct states (transitions plus one terminal state each)."""
        return int(self.observations.shape[0] + self.ends.size)

    @property
    def observation_space(self) -> Any:
        """Lightweight stand-in for a gym observation space (``shape``/``dtype``)."""
        try:
            import gym  # type: ignore

            return gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
        except Exception:  # pragma: no cover - gym optional

            class _Box:  # minimal duck-typed space
                def __init__(self, shape: Tuple[int, ...]) -> None:
                    self.shape = shape
                    self.dtype = np.float32

                def __repr__(self) -> str:  # pragma: no cover
                    return f"Box(shape={self.shape})"

            return _Box((self.obs_dim,))

    # -- statistics / normalization ---------------------------------------
    def statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-dimension mean and standard deviation of the offline observations."""
        if self._stats is None:
            mean = self.observations.mean(axis=0).astype(np.float64)
            std = self.observations.std(axis=0).astype(np.float64)
            std = np.maximum(std, 1e-6)
            self._stats = (mean, std)
        return self._stats

    def state_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Alias of :meth:`statistics` (interface symmetry with the other loaders)."""
        if self.state_mean is not None and self.state_std is not None:
            return self.state_mean, self.state_std
        return self.statistics()

    def state_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate per-dimension state box (mean +- 3 std)."""
        mean, std = self.state_statistics()
        return mean - 3.0 * std, mean + 3.0 * std

    def normalized_observations(self) -> np.ndarray:
        """Observations normalized by the dataset per-dimension mean/std."""
        mean, std = self.state_statistics()
        return ((self.observations - mean[None, :]) / std[None, :]).astype(np.float32)

    # -- trajectories ------------------------------------------------------
    def trajectory_slice(self, traj_index: int) -> slice:
        """Return the ``slice`` of the flat arrays belonging to trajectory ``traj_index``."""
        if traj_index < 0:
            traj_index += self.num_trajectories
        if not 0 <= traj_index < self.num_trajectories:
            raise IndexError(f"trajectory index {traj_index} out of range")
        start = 0 if traj_index == 0 else int(self.ends[traj_index - 1]) + 1
        stop = int(self.ends[traj_index]) + 1
        return slice(start, stop)

    def trajectory_length(self, traj_index: int) -> int:
        """Number of *transitions* in trajectory ``traj_index``."""
        sl = self.trajectory_slice(traj_index)
        return int(sl.stop - sl.start)

    def trajectory_observations(self, traj_index: int) -> np.ndarray:
        """Observations (including the terminal observation) of a trajectory."""
        sl = self.trajectory_slice(traj_index)
        obs = self.observations[sl]
        if sl.stop - 1 < self.observations.shape[0] and self.next_observations is not None:
            terminal_obs = self.next_observations[sl.stop - 1]
            obs = np.concatenate([obs, terminal_obs[None, :]], axis=0)
        return obs

    def start_state(self, traj_index: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Initial observation for evaluation.

        Kitchen episodes in D4RL reset to one of a small set of initial configurations.
        The appendix does not specify the evaluation start state for Kitchen, so we
        default to the first observation of a fixed (or randomly chosen) offline
        trajectory, which guarantees an in-distribution initial pose.
        """
        if traj_index is None:
            if rng is None:
                traj_index = 0
            else:
                traj_index = int(rng.integers(self.num_trajectories))
        sl = self.trajectory_slice(int(traj_index))
        return self.observations[sl.start].copy()

    # -- reward functions --------------------------------------------------
    def task_reward(self, observations: Any, task: str) -> np.ndarray:
        """Sparse reward of ``task`` (Source: C.3. Kitchen)."""
        return compute_task_reward(observations, task, threshold=self.task_threshold)

    def all_task_rewards(self, observations: Any) -> Dict[str, np.ndarray]:
        """Sparse rewards for all seven standard subtasks."""
        return compute_all_task_rewards(observations, self.tasks, threshold=self.task_threshold)

    def task_success(self, observations: Any, task: str) -> np.ndarray:
        """Boolean success indicator for ``task``."""
        return self.task_reward(observations, task) > 0.0

    # -- replay buffer -----------------------------------------------------
    def build_buffer(
        self,
        seed: Optional[int] = None,
        exclude_final_states: bool = True,
        attach_stats: bool = True,
    ) -> Any:
        """Create (and cache) the unlabeled :class:`ReplayBuffer` for FRE training."""
        if make_replay_buffer is None:  # pragma: no cover - replay module unavailable
            raise RuntimeError("fre.data.replay is unavailable; cannot build a ReplayBuffer")
        buffer = make_replay_buffer(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            timeouts=self.timeouts,
            next_observations=self.next_observations,
            ends=self.ends,
            seed=self.seed if seed is None else int(seed),
            exclude_final_states=exclude_final_states,
        )
        if attach_stats:
            mean, std = self.state_statistics()
            try:  # optional hooks used by the reward priors
                if not hasattr(buffer, "state_box"):
                    buffer.state_box = lambda: (mean - 3.0 * std, mean + 3.0 * std)  # type: ignore[attr-defined]
                if not hasattr(buffer, "state_mean_std"):
                    buffer.state_mean_std = lambda: (mean, std)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - buffer may be immutable
                pass
        self.buffer = buffer
        return buffer

    @property
    def replay_buffer(self) -> Any:
        if self.buffer is None:
            self.build_buffer()
        return self.buffer

    def sample_states(
        self,
        num_states: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Uniformly sample ``num_states`` observations from the offline dataset."""
        return self.replay_buffer.sample_states(num_states, rng=rng)

    def sample_states_with_metadata(
        self,
        num_states: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample states together with their ``(traj_index, step_index)``."""
        return self.replay_buffer.sample_states_with_metadata(num_states, rng=rng)

    # -- io / description --------------------------------------------------
    def to_npz(self, path: str) -> str:
        """Export the raw arrays to a compressed ``.npz`` file."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "observations": self.observations,
            "actions": self.actions,
            "ends": self.ends,
        }
        if self.terminals is not None:
            payload["terminals"] = self.terminals
        if self.timeouts is not None:
            payload["timeouts"] = self.timeouts
        if self.next_observations is not None:
            payload["next_observations"] = self.next_observations
        np.savez_compressed(path, **payload)
        return path

    def describe(self) -> Dict[str, Any]:
        """Human-readable summary of the dataset (for logging)."""
        mean, std = self.state_statistics()
        return {
            "dataset_name": self.dataset_name,
            "domain": "kitchen",
            "num_transitions": self.num_transitions,
            "num_trajectories": self.num_trajectories,
            "num_states": self.num_states,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "tasks": list(self.tasks),
            "task_threshold": float(self.task_threshold),
            "eval_episode_length": int(self.eval_episode_length),
            "obs_mean_first": float(mean[0]) if mean.size else None,
            "obs_std_first": float(std[0]) if std.size else None,
        }


# ===========================================================================
# Helpers
# ===========================================================================


def _compute_ends(
    terminals: Any,
    timeouts: Any,
    next_observations: Any,
    num_transitions: int,
) -> np.ndarray:
    """Derive trajectory end indices from ``terminals``/``timeouts`` when absent."""
    if next_observations is not None:
        next_obs = _as_2d_float(next_observations)
        if next_obs.shape[0] == num_transitions:
            return np.arange(num_transitions, dtype=np.int64)
    mask = np.zeros(num_transitions, dtype=bool)
    if terminals is not None:
        mask |= np.asarray(terminals, dtype=np.float64).reshape(-1) > 0.5
    if timeouts is not None:
        mask |= np.asarray(timeouts, dtype=np.float64).reshape(-1) > 0.5
    mask[-1] = True
    return np.nonzero(mask)[0].astype(np.int64)


def _load_hdf5(path: str) -> Dict[str, np.ndarray]:
    """Load arrays from a D4RL-style HDF5 dump (requires ``h5py``)."""
    import h5py  # imported lazily

    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        grp = handle["data"] if "data" in handle else handle

        def collect(group: Any, prefix: str = "") -> None:
            for key in group.keys():
                item = group[key]
                name = f"{prefix}{key}"
                if hasattr(item, "keys"):
                    collect(item, f"{name}/")
                else:
                    arrays[name] = np.array(item)

        collect(grp)
    merged: Dict[str, np.ndarray] = {}
    for name in ("observations", "actions", "rewards", "terminals", "timeouts", "next_observations"):
        for key, value in arrays.items():
            if key.split("/")[-1] == name and name not in merged:
                merged[name] = value
    return merged


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _empty_arrays() -> Dict[str, np.ndarray]:
    obs = np.zeros((0, KITCHEN_OBS_DIM), dtype=np.float32)
    return {
        "observations": obs,
        "actions": np.zeros((0, 9), dtype=np.float32),
        "rewards": np.zeros((0,), dtype=np.float32),
        "terminals": np.zeros((0,), dtype=np.float32),
        "timeouts": np.zeros((0,), dtype=np.float32),
        "next_observations": obs.copy(),
    }


def load_kitchen_arrays(
    path: Optional[str] = None,
    dataset_name: str = KITCHEN_DATASET_NAME,
    env_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load the raw D4RL Kitchen transition arrays.

    Resolution order:

    1. ``path`` (``.npz`` or ``.hdf5``),
    2. the ``FRE_KITCHEN_DATASET`` / ``FRE_KITCHEN_DATASET_PATH`` environment variables,
    3. D4RL / gym (``gym.make(env_id).get_dataset()``), if installed.

    Returns an empty (but correctly shaped) array dict if none of these are available,
    so that the surrounding pipeline can still be exercised.
    """
    candidates: List[str] = []
    if path:
        candidates.append(path)
    for env_var in KITCHEN_DATASET_PATHS_ENV:
        value = os.environ.get(env_var)
        if value:
            candidates.append(value)

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        if candidate.endswith(".npz") or candidate.endswith(".npy"):
            arrays = _load_npz(candidate)
            return _truncate(arrays, limit)
        if candidate.endswith(".hdf5") or candidate.endswith(".h5"):
            arrays = _load_hdf5(candidate)
            if arrays:
                return _truncate(arrays, limit)

    # D4RL / gym fallback (optional dependency).
    try:  # pragma: no cover - requires d4rl + mujoco
        import gym  # type: ignore
        import d4rl  # noqa: F401  (registers the environments)

        env_id = env_id or dataset_name
        env = gym.make(env_id)
        try:
            dataset = env.get_dataset()
        finally:
            try:
                env.close()
            except Exception:
                pass
        arrays = {
            "observations": np.asarray(dataset["observations"], dtype=np.float32),
            "actions": np.asarray(dataset["actions"], dtype=np.float32),
        }
        for key in ("rewards", "terminals", "timeouts", "next_observations"):
            if key in dataset:
                arrays[key] = np.asarray(dataset[key], dtype=np.float32)
        return _truncate(arrays, limit)
    except Exception as exc:  # pragma: no cover - optional dependency
        print(
            f"[fre.data.kitchen_dataset] Could not load the D4RL Kitchen dataset "
            f"({type(exc).__name__}: {exc}); returning empty arrays. Set FRE_KITCHEN_DATASET "
            f"to an exported .npz/.hdf5 dump to use the loader offline.",
            file=sys.stderr,
        )
        return _empty_arrays()


def _truncate(arrays: Mapping[str, np.ndarray], limit: Optional[int]) -> Dict[str, np.ndarray]:
    """Optionally truncate every per-transition array to the first ``limit`` entries."""
    result: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if limit is not None and key in ("observations", "actions", "rewards", "terminals", "timeouts"):
            result[key] = value[:limit]
        else:
            result[key] = value
    return result


def make_kitchen_dataset(
    observations: Any,
    actions: Any,
    rewards: Any = None,
    terminals: Any = None,
    timeouts: Any = None,
    next_observations: Any = None,
    ends: Any = None,
    **kwargs: Any,
) -> KitchenDataset:
    """Factory mirroring :func:`fre.data.antmaze_dataset.make_antmaze_dataset`."""
    return KitchenDataset.from_arrays(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        next_observations=next_observations,
        ends=ends,
        **kwargs,
    )


def load_kitchen_dataset(
    path: Optional[str] = None,
    dataset_name: str = KITCHEN_DATASET_NAME,
    env_id: Optional[str] = None,
    tasks: Sequence[str] = KITCHEN_TASKS,
    task_threshold: float = KITCHEN_TASK_THRESHOLD,
    eval_episode_length: int = KITCHEN_EVAL_EPISODE_LENGTH,
    build_buffer: bool = True,
    exclude_final_states: bool = True,
    attach_stats: bool = True,
    limit: Optional[int] = None,
    seed: int = 0,
    **kwargs: Any,
) -> KitchenDataset:
    """End-to-end Kitchen loader used by the training / evaluation entry points.

    Returns a :class:`KitchenDataset` with the dataset statistics computed from the
    loaded transitions and (by default) an attached unlabeled ``ReplayBuffer``.
    """
    arrays = load_kitchen_arrays(
        path=path, dataset_name=dataset_name, env_id=env_id, limit=limit
    )
    dataset = KitchenDataset.from_arrays(
        observations=arrays.get("observations", np.zeros((0, KITCHEN_OBS_DIM), dtype=np.float32)),
        actions=arrays.get("actions", np.zeros((0, 9), dtype=np.float32)),
        rewards=arrays.get("rewards"),
        terminals=arrays.get("terminals"),
        timeouts=arrays.get("timeouts"),
        next_observations=arrays.get("next_observations"),
        ends=arrays.get("ends"),
        dataset_name=dataset_name,
        tasks=tuple(tasks),
        task_threshold=task_threshold,
        eval_episode_length=eval_episode_length,
        seed=seed,
        **kwargs,
    )
    if build_buffer and dataset.num_transitions > 0:
        dataset.build_buffer(
            seed=seed, exclude_final_states=exclude_final_states, attach_stats=attach_stats
        )
    return dataset
