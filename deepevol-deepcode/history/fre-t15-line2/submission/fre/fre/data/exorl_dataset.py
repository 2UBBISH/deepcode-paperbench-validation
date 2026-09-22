"""ExORL (cheetah / walker) offline dataset wrapper for FRE.

Paper references
----------------
Section C.2 (ExORL):
    "We utilize cheetah-run, cheetah-walk, cheetah-run-backwards,
    cheetah-walk-backwards and walker-run, walker-walk as evaluation tasks.
    Agents are evaluated for 1000 timesteps. For goal-reaching tasks, we select
    five consistent goal states from the offline dataset."

    "FRE assumes that reward functions must be pure functions of the environment
    state. Because the Cheetah and Walker environments utilize rewards that are a
    function of the underlying physics, we append information about the physics
    onto the offline dataset during encoder training. Specifically, we append the
    values of
        self.physics.horizontal_velocity()
        self.physics.torso_upright()
        self.physics.torso_height()
    to Walker, and
        self.physics.speed()
    to Cheetah.
    The above auxiliary information is neccessary only for the encoder network,
    in order to define the true reward functions of the ExORL tasks, which are
    based on physics states. We found that performance was not greatly affected
    whether or not the value functions and policy networks have access to the
    auxilliary information, and are instead trained on the underlying observation
    space of the environment."

    "Goals in ExORL are computed when the Euclidean distance between the current
    state and the goal state is less than 0.1. Each state dimension is normalized
    according to the standard deviation along that dimension within the offline
    dataset. Augmented information is not utilized when calculating goal distance."

Design notes
------------
* The offline dataset used for *training* is the RND exploratory dataset of each
  domain (addendum: "training uses the RND dataset for each domain").  The
  cheetah / walker evaluation tasks all share the same domain-wise training
  dataset, so the loader resolves ``dataset=EXORL_RND_DATASET`` by default.
* ``observations`` always hold the *raw* environment observation (what the Q /
  value / policy networks consume).  ``physics_features`` holds the appended
  physics quantities; ``encoder_observations`` == concat(raw, physics) is what the
  FRE encoder consumes (``ReplayBuffer.encoder_observations``).
* Goal distance uses the *raw*, per-dimension standard-deviation-normalized
  observations and never the appended physics values ("Augmented information is
  not utilized when calculating goal distance").
* Because a full MuJoCo replay of the offline trajectories is expensive, the
  physics quantities are recomputed through dm_control when it is importable
  (``physics_fn`` / ``method="env"``) and otherwise through a documented
  index-based fallback (``method="indices"``).  The fallback indices are derived
  from the layout of the dm_control observation dictionaries:
  walker = [position (qpos, 9), velocity (qvel, 9), ..., height (last dim)],
  cheetah = [position (qpos, 8), velocity (qvel, 9)].
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# imports of the sibling replay buffer (works both as a package and standalone)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import plumbing
    from fre.data.replay import Batch, ReplayBuffer, Trajectory, make_replay_buffer
except Exception:  # pragma: no cover - direct execution / partial checkout
    Batch = ReplayBuffer = Trajectory = make_replay_buffer = None  # type: ignore
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(_HERE))
    if os.path.isdir(os.path.join(_ROOT, "fre")) and _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    try:
        from fre.data.replay import (  # type: ignore
            Batch,
            ReplayBuffer,
            Trajectory,
            make_replay_buffer,
        )
    except Exception:
        pass


__all__ = [
    "EXORL_DATASET_NAMES",
    "EXORL_DOMAINS",
    "EXORL_RND_DATASET",
    "EXORL_EVAL_EPISODE_LENGTH",
    "EXORL_GOAL_DISTANCE_THRESHOLD",
    "NUM_EXORL_GOALS",
    "WALKER_PHYSICS_FEATURES",
    "CHEETAH_PHYSICS_FEATURES",
    "PHYSICS_FEATURES",
    "ExORLDataset",
    "append_physics_features",
    "compute_physics_features",
    "normalize_observations",
    "unnormalize_observations",
    "load_exorl_arrays",
    "load_exorl_dataset",
    "make_exorl_dataset",
]


# ======================================================================================
# Constants
# ======================================================================================

#: domains with ExORL RND exploratory datasets used by the paper.
EXORL_DOMAINS: Tuple[str, ...] = ("cheetah", "walker")

#: the exploratory dataset used for training (addendum: "training uses the RND dataset").
EXORL_RND_DATASET: str = "rnd"

#: Section C.2: "Agents are evaluated for 1000 timesteps."
EXORL_EVAL_EPISODE_LENGTH: int = 1000

#: Section C.2: goal when Euclidean distance < 0.1 (on std-normalized dimensions).
EXORL_GOAL_DISTANCE_THRESHOLD: float = 0.1

#: Section C.2: "we select five consistent goal states from the offline dataset."
NUM_EXORL_GOALS: int = 5

#: Appendix C.2 physics quantities appended for Walker (order as in the paper).
WALKER_PHYSICS_FEATURES: Tuple[str, ...] = (
    "horizontal_velocity",
    "torso_upright",
    "torso_height",
)

#: Appendix C.2 physics quantities appended for Cheetah.
CHEETAH_PHYSICS_FEATURES: Tuple[str, ...] = ("speed",)

PHYSICS_FEATURES: Dict[str, Tuple[str, ...]] = {
    "walker": WALKER_PHYSICS_FEATURES,
    "cheetah": CHEETAH_PHYSICS_FEATURES,
}

#: The six ExORL evaluation tasks of Section C.2 (domain / dm_control task / training dataset).
#: ``file`` is the relative path inside the ExORL dataset dump (or ``$FRE_EXORL_DATASET_DIR``).
EXORL_DATASET_NAMES: Dict[str, Dict[str, Any]] = {
    "cheetah-run": {
        "domain": "cheetah",
        "task": "run",
        "dataset": EXORL_RND_DATASET,
        "env_id": "cheetah-run",
        "file": "cheetah/rnd.hdf5",
    },
    "cheetah-walk": {
        "domain": "cheetah",
        "task": "walk",
        "dataset": EXORL_RND_DATASET,
        "env_id": "cheetah-walk",
        "file": "cheetah/rnd.hdf5",
    },
    "cheetah-run-backwards": {
        "domain": "cheetah",
        "task": "run_backwards",
        "dataset": EXORL_RND_DATASET,
        "env_id": "cheetah-run-backwards",
        "file": "cheetah/rnd.hdf5",
    },
    "cheetah-walk-backwards": {
        "domain": "cheetah",
        "task": "walk_backwards",
        "dataset": EXORL_RND_DATASET,
        "env_id": "cheetah-walk-backwards",
        "file": "cheetah/rnd.hdf5",
    },
    "walker-run": {
        "domain": "walker",
        "task": "run",
        "dataset": EXORL_RND_DATASET,
        "env_id": "walker-run",
        "file": "walker/rnd.hdf5",
    },
    "walker-walk": {
        "domain": "walker",
        "task": "walk",
        "dataset": EXORL_RND_DATASET,
        "env_id": "walker-walk",
        "file": "walker/rnd.hdf5",
    },
}

#: Domain -> the RND training dataset relative path (all eval tasks of a domain share it).
EXORL_TRAINING_DATASETS: Dict[str, str] = {
    "cheetah": "cheetah/rnd.hdf5",
    "walker": "walker/rnd.hdf5",
}

#: Documented (not used automatically) download root of the ExORL dataset dumps.
EXORL_DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/exorl/"

#: Environment variable pointing at a local ExORL dataset directory.
EXORL_DATASET_DIR_ENV = "FRE_EXORL_DATASET_DIR"

#: Number of leading dm_control observation dims that form ``[qpos, qvel]`` (fallback replay).
_EXORL_QPOS_QVEL_DIMS: Dict[str, int] = {"walker": 18, "cheetah": 17}

#: Index of the first velocity (qvel[0]) inside the raw observation (fallback heuristics).
_EXORL_ROOT_VEL_INDEX: Dict[str, int] = {"walker": 9, "cheetah": 8}

_STD_FLOOR = 1e-6


# ======================================================================================
# small helpers
# ======================================================================================


def _as_2d_float(array: Any) -> np.ndarray:
    """Coerce ``array`` into a contiguous 2-D float32 numpy array."""
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return np.ascontiguousarray(out)


def _as_1d_float(array: Any) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    return np.ascontiguousarray(out.reshape(-1))


def _resolve_domain(domain: Optional[str], task: Optional[str], dataset_name: Optional[str]) -> str:
    """Resolve the ExORL domain (``"cheetah"`` / ``"walker"``) from any of the identifiers."""
    if domain is not None:
        domain = str(domain).lower()
    if dataset_name is not None and str(dataset_name) in EXORL_DATASET_NAMES:
        domain = str(EXORL_DATASET_NAMES[str(dataset_name)]["domain"])
    if domain is None and task is not None:
        for name, spec in EXORL_DATASET_NAMES.items():
            if spec["task"] == task:
                domain = spec["domain"]
                break
    if domain is None:
        domain = "walker"
    if domain not in PHYSICS_FEATURES:
        raise ValueError(
            f"unknown ExORL domain {domain!r}; expected one of {sorted(PHYSICS_FEATURES)}"
        )
    return domain


def _resolve_dataset_name(
    domain: str,
    task: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> str:
    """Resolve the evaluation-task identifier (key of :data:`EXORL_DATASET_NAMES`)."""
    if dataset_name is not None and str(dataset_name) in EXORL_DATASET_NAMES:
        return str(dataset_name)
    if dataset_name is not None and str(dataset_name) == domain:
        dataset_name = None
    if task is not None:
        candidate = f"{domain}-{task}"
        if candidate in EXORL_DATASET_NAMES:
            return candidate
    # default: the forward-running task of the domain
    return f"{domain}-run"


def _resolve_dataset_file(
    path: Optional[str] = None,
    domain: str = "walker",
    dataset: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Optional[str]:
    """Locate an ExORL dump file: explicit path > ``$FRE_EXORL_DATASET_DIR`` > ``None``."""
    candidates: List[str] = []
    if path is not None:
        candidates.append(str(path))
    env_dir = os.environ.get(EXORL_DATASET_DIR_ENV)
    if env_dir:
        rel: List[str] = []
        if dataset_name is not None and str(dataset_name) in EXORL_DATASET_NAMES:
            rel.append(str(EXORL_DATASET_NAMES[str(dataset_name)]["file"]))
        if dataset is not None and dataset != EXORL_RND_DATASET:
            rel.append(os.path.join(domain, f"{dataset}.hdf5"))
        rel.append(EXORL_TRAINING_DATASETS.get(domain, os.path.join(domain, "rnd.hdf5")))
        for r in rel:
            candidates.append(os.path.join(env_dir, r))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates and os.path.exists(candidates[0]) else None


def _compute_ends(
    terminals: Optional[np.ndarray],
    timeouts: Optional[np.ndarray],
    next_observations: Optional[np.ndarray],
) -> np.ndarray:
    """Derive trajectory end indices from ``terminals`` / ``timeouts`` (D4RL convention)."""
    if terminals is None and timeouts is None:
        if next_observations is None:
            raise ValueError("cannot infer trajectory boundaries for the ExORL dataset")
        return np.arange(len(next_observations) - 1, len(next_observations), dtype=np.int64)
    total = len(terminals if terminals is not None else timeouts)
    ends = np.zeros(total, dtype=bool)
    if terminals is not None:
        ends |= np.asarray(terminals).astype(bool).reshape(-1)
    if timeouts is not None:
        ends |= np.asarray(timeouts).astype(bool).reshape(-1)
    return np.nonzero(ends)[0].astype(np.int64)


# ======================================================================================
# physics augmentation
# ======================================================================================


def _physics_from_indices(observations: np.ndarray, domain: str) -> np.ndarray:
    """Index-based fallback for the physics quantities of Section C.2.

    Walker (dm_control ``walker`` observation layout: position 9, velocity 9, ...,
    height last):

    * ``horizontal_velocity`` = first root velocity (``obs[9]``, i.e. ``qvel[0]``)
    * ``torso_upright``       = ``1 - 2 * (qx^2 + qy^2)`` from the torso quaternion
      stored in ``qpos[3:7]`` (dm_control's own definition)
    * ``torso_height``        = last observation dimension

    Cheetah (position 8, velocity 9):

    * ``speed`` = first root velocity (``obs[8]``, i.e. ``qvel[0]``), matching
      ``physics.speed()`` of dm_control's cheetah which returns the root x velocity.

    Note: this fallback is only used when a dm_control replay is unavailable
    (Source: not specified in the paper, which queries the simulator directly).
    """
    if domain == "walker":
        root_vel = observations[:, _EXORL_ROOT_VEL_INDEX["walker"]]
        qx = observations[:, 4]
        qy = observations[:, 5]
        upright = 1.0 - 2.0 * (qx**2 + qy**2)
        height = observations[:, -1]
        return np.stack([root_vel, upright, height], axis=-1).astype(np.float32)
    if domain == "cheetah":
        root_vel = observations[:, _EXORL_ROOT_VEL_INDEX["cheetah"]]
        return root_vel.reshape(-1, 1).astype(np.float32)
    raise ValueError(f"no physics fallback for domain {domain!r}")


def _physics_from_env(observations: np.ndarray, domain: str) -> np.ndarray:
    """Recompute the physics quantities by replaying states through dm_control.

    Sources the exact quantities used by the paper via the environment's own
    physics object: ``horizontal_velocity`` / ``torso_upright`` / ``torso_height``
    for Walker and ``speed`` for Cheetah.
    """
    from dm_control import suite  # local import: optional dependency

    env = suite.load(domain_name=domain, task_name="run")
    physics = env.physics
    qpos_dim = int(physics.data.qpos.shape[0])
    qvel_dim = int(physics.data.qvel.shape[0])
    if qpos_dim + qvel_dim > observations.shape[1]:
        raise ValueError("dm_control state dims do not fit the offline observations")

    features = np.zeros((observations.shape[0], len(PHYSICS_FEATURES[domain])), dtype=np.float32)
    for i in range(observations.shape[0]):
        state = observations[i]
        with physics.reset_context():
            physics.data.qpos[:] = state[:qpos_dim]
            physics.data.qvel[:] = state[qpos_dim : qpos_dim + qvel_dim]
        physics.forward()
        for j, name in enumerate(PHYSICS_FEATURES[domain]):
            features[i, j] = float(getattr(physics, name)())
    return features


def compute_physics_features(
    observations: Any,
    domain: str = "walker",
    physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    method: str = "auto",
) -> np.ndarray:
    """Compute the Appendix C.2 physics quantities for a batch of observations.

    Parameters
    ----------
    observations:
        Raw environment observations, shape ``(N, obs_dim)``.
    domain:
        ``"walker"`` or ``"cheetah"``.
    physics_fn:
        Optional callable mapping ``(N, obs_dim) -> (N, num_physics)``; takes
        precedence over everything else (useful for tests / precomputed dumps).
    method:
        ``"auto"`` (dm_control replay if importable, else indices), ``"env"`` or
        ``"indices"``.

    Returns
    -------
    np.ndarray of shape ``(N, len(PHYSICS_FEATURES[domain]))``.
    """
    obs = _as_2d_float(observations)
    domain = _resolve_domain(domain, None, None)

    if physics_fn is not None:
        return _as_2d_float(physics_fn(obs))

    if method in ("auto", "env"):
        try:
            return _as_2d_float(_physics_from_env(obs, domain))
        except Exception:
            if method == "env":
                raise

    if method not in ("auto", "indices"):
        raise ValueError(f"unknown physics method {method!r}")
    return _physics_from_indices(obs, domain)


def append_physics_features(
    observations: Any,
    domain: str = "walker",
    physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    method: str = "auto",
    features: Optional[Any] = None,
    in_place: bool = False,
) -> np.ndarray:
    """Append the physics quantities to the raw observations (encoder input only).

    Implements Appendix C.2: the encoder sees ``concat(observation, physics)``
    whereas the value / policy networks and the goal distances use the raw
    observation space only.

    Returns an array of shape ``(N, obs_dim + num_physics)``.
    """
    obs = _as_2d_float(observations) if not in_place else np.asarray(observations, dtype=np.float32)
    if features is None:
        features = compute_physics_features(obs, domain=domain, physics_fn=physics_fn, method=method)
    features = _as_2d_float(features)
    if features.shape[0] != obs.shape[0]:
        raise ValueError("physics features and observations disagree on the number of samples")
    return np.concatenate([obs, features], axis=-1).astype(np.float32)


# ======================================================================================
# normalization (Appendix C.2: "Each state dimension is normalized according to the
# standard deviation along that dimension within the offline dataset")
# ======================================================================================


def normalize_observations(
    observations: Any,
    mean: Optional[Any] = None,
    std: Optional[Any] = None,
    stats: Optional[Tuple[Any, Any]] = None,
    in_place: bool = False,
    std_floor: float = _STD_FLOOR,
) -> np.ndarray:
    """``(x - mean) / max(std, std_floor)`` along the last dimension."""
    obs = np.asarray(observations, dtype=np.float32)
    if stats is not None:
        mean, std = stats
    if mean is None or std is None:
        raise ValueError("normalize_observations requires mean/std (or a (mean, std) tuple)")
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_arr = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1), float(std_floor))
    out = (obs - mean_arr) / std_arr
    if in_place and isinstance(observations, np.ndarray):
        observations[...] = out
        return observations
    return out.astype(np.float32)


def unnormalize_observations(
    observations: Any,
    mean: Any,
    std: Any,
    std_floor: float = _STD_FLOOR,
) -> np.ndarray:
    """Inverse of :func:`normalize_observations`."""
    obs = np.asarray(observations, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_arr = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1), float(std_floor))
    return (obs * std_arr + mean_arr).astype(np.float32)


# ======================================================================================
# loading
# ======================================================================================

#: accepted key spellings in the ExORL hdf5 dumps (checked in order).
_OBS_KEYS = ("observations", "obs", "observation", "states")
_ACT_KEYS = ("actions", "action", "acts")
_REWARD_KEYS = ("rewards", "reward")
_TERMINAL_KEYS = ("terminals", "terminal", "dones", "done")
_TIMEOUT_KEYS = ("timeouts", "timeout", "truncated")
_NEXT_OBS_KEYS = ("next_observations", "next_obs", "next_states")


def _hdf5_get(handle: Any, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in handle:
            return np.asarray(handle[key])
    return None


def _load_hdf5(path: str) -> Dict[str, np.ndarray]:
    """Load an ExORL hdf5 dump, accepting the common key spellings / ``train`` groups."""
    import h5py  # local import: optional dependency

    with h5py.File(path, "r") as handle:
        scope: Any = handle
        # some ExORL dumps nest the transitions under a "train" group.
        for group in ("train", "data"):
            if group in handle and not any(k in handle for k in _OBS_KEYS):
                scope = handle[group]
                break
        out: Dict[str, np.ndarray] = {}
        obs = _hdf5_get(scope, _OBS_KEYS)
        if obs is None:
            raise KeyError(f"no observation dataset found in {path!r}")
        out["observations"] = obs
        for name, keys in (
            ("actions", _ACT_KEYS),
            ("rewards", _REWARD_KEYS),
            ("terminals", _TERMINAL_KEYS),
            ("timeouts", _TIMEOUT_KEYS),
            ("next_observations", _NEXT_OBS_KEYS),
        ):
            value = _hdf5_get(scope, keys)
            if value is not None:
                out[name] = value
    return out


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def load_exorl_arrays(
    path: Optional[str] = None,
    domain: str = "walker",
    dataset: Optional[str] = None,
    dataset_name: Optional[str] = None,
    max_transitions: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load raw ExORL transition arrays (``.hdf5`` / ``.npz``).

    ``path`` may also be given through the ``FRE_EXORL_DATASET_DIR`` environment
    variable, in which case ``<domain>/<dataset>.hdf5`` is used
    (Source: the paper only names the ExORL datasets and their domains).
    """
    resolved = _resolve_dataset_file(path, domain=domain, dataset=dataset, dataset_name=dataset_name)
    if resolved is None:
        raise FileNotFoundError(
            "could not locate the ExORL dataset dump; pass an explicit path or set "
            f"{EXORL_DATASET_DIR_ENV} (see {EXORL_DATASET_URL})"
        )
    if resolved.endswith(".npz"):
        arrays = _load_npz(resolved)
    else:
        arrays = _load_hdf5(resolved)

    observations = _as_2d_float(arrays["observations"])
    if max_transitions is not None:
        observations = observations[: int(max_transitions)]
    out: Dict[str, np.ndarray] = {"observations": observations}
    for name in ("actions", "rewards", "terminals", "timeouts", "next_observations"):
        if name in arrays:
            value = np.asarray(arrays[name])
            if name in ("terminals", "timeouts"):
                value = value.astype(bool).reshape(-1)
            elif value.ndim == 1 and name != "rewards":
                value = value.reshape(-1, 1)
            if max_transitions is not None:
                value = value[: int(max_transitions)]
            out[name] = value
    return out


# ======================================================================================
# dataset container
# ======================================================================================


@dataclass
class ExORLDataset:
    """ExORL offline dataset with the Appendix C.2 preprocessing.

    Fields
    ------
    observations:
        raw environment observations, ``(N, obs_dim)`` — used by Q / V / policy and
        by the goal distances.
    actions:
        ``(N, act_dim)``.
    physics_features:
        ``(N, num_physics)`` physics quantities of Appendix C.2 (None when the
        state dimension available in the dump cannot support them).
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: Optional[np.ndarray] = None
    terminals: Optional[np.ndarray] = None
    timeouts: Optional[np.ndarray] = None
    next_observations: Optional[np.ndarray] = None
    ends: Optional[np.ndarray] = None

    domain: str = "walker"
    task: str = "run"
    dataset_name: str = "walker-run"
    dataset: str = EXORL_RND_DATASET
    dataset_file: Optional[str] = None

    physics_features: Optional[np.ndarray] = None
    physics_names: Tuple[str, ...] = ()

    normalize: bool = True
    state_mean: Optional[np.ndarray] = None
    state_std: Optional[np.ndarray] = None
    encoder_state_mean: Optional[np.ndarray] = None
    encoder_state_std: Optional[np.ndarray] = None

    goal_states: Optional[np.ndarray] = None
    num_goals: int = NUM_EXORL_GOALS
    goal_distance_threshold: float = EXORL_GOAL_DISTANCE_THRESHOLD
    eval_episode_length: int = EXORL_EVAL_EPISODE_LENGTH

    seed: int = 0
    buffer: Any = None
    _start_state: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------ construction
    def __post_init__(self) -> None:
        self.observations = _as_2d_float(self.observations)
        self.actions = _as_2d_float(self.actions)
        if self.actions.shape[0] != self.observations.shape[0]:
            raise ValueError("observations and actions must have the same length")
        if self.rewards is not None:
            self.rewards = _as_1d_float(self.rewards)
        if self.terminals is not None:
            self.terminals = np.asarray(self.terminals).astype(bool).reshape(-1)
        if self.timeouts is not None:
            self.timeouts = np.asarray(self.timeouts).astype(bool).reshape(-1)
        if self.next_observations is not None:
            self.next_observations = _as_2d_float(self.next_observations)
        if self.ends is None:
            self.ends = _compute_ends(self.terminals, self.timeouts, self.next_observations)
        else:
            self.ends = np.asarray(self.ends, dtype=np.int64).reshape(-1)

        self.domain = _resolve_domain(self.domain, self.task, self.dataset_name)
        self.dataset_name = _resolve_dataset_name(self.domain, self.task, self.dataset_name)
        spec = EXORL_DATASET_NAMES[self.dataset_name]
        self.task = str(spec["task"])
        if not self.physics_names:
            self.physics_names = PHYSICS_FEATURES[self.domain]
        if self.physics_features is not None:
            self.physics_features = _as_2d_float(self.physics_features)
            if self.physics_features.shape[0] != self.observations.shape[0]:
                raise ValueError("physics features must align with the observations")

        self._compute_statistics()
        if self.normalize:
            self.normalize_()

    # ------------------------------------------------------------------ statistics
    def _compute_statistics(self) -> None:
        """Per-dimension mean / std of the offline dataset (Section C.2)."""
        if self.state_mean is None or self.state_std is None:
            self.state_mean = self.observations.mean(axis=0).astype(np.float32)
            self.state_std = np.maximum(self.observations.std(axis=0), _STD_FLOOR).astype(np.float32)
        if self.physics_features is not None:
            aug = np.concatenate([self.observations, self.physics_features], axis=-1)
            self.encoder_state_mean = aug.mean(axis=0).astype(np.float32)
            self.encoder_state_std = np.maximum(aug.std(axis=0), _STD_FLOOR).astype(np.float32)
        else:
            self.encoder_state_mean = self.state_mean
            self.encoder_state_std = self.state_std

    def normalize_(self) -> "ExORLDataset":
        """Normalize the stored arrays in place with the dataset statistics."""
        self.observations = normalize_observations(self.observations, self.state_mean, self.state_std)
        if self.next_observations is not None:
            self.next_observations = normalize_observations(
                self.next_observations, self.state_mean, self.state_std
            )
        if self.physics_features is not None:
            raw = np.concatenate([self.observations, self.physics_features], axis=-1)
            aug = normalize_observations(raw, self.encoder_state_mean, self.encoder_state_std)
            self.physics_features = aug[:, self.raw_obs_dim :].astype(np.float32)
        self.normalize = True
        return self

    def denormalized_observations(self) -> np.ndarray:
        """Raw (un-normalized) observations, e.g. for logging / dm_control replay."""
        return unnormalize_observations(self.observations, self.state_mean, self.state_std)

    # ------------------------------------------------------------------ shapes
    @property
    def raw_obs_dim(self) -> int:
        """Dimension of the observation space used by Q / V / policy."""
        return int(self.observations.shape[-1])

    @property
    def obs_dim(self) -> int:
        return self.raw_obs_dim

    @property
    def num_physics_features(self) -> int:
        return 0 if self.physics_features is None else int(self.physics_features.shape[-1])

    @property
    def encoder_obs_dim(self) -> int:
        """Dimension of the encoder input (observation + physics, Appendix C.2)."""
        return self.raw_obs_dim + self.num_physics_features

    @property
    def act_dim(self) -> int:
        return int(self.actions.shape[-1])

    @property
    def num_transitions(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_states(self) -> int:
        return self.num_transitions + int(self.num_trajectories)

    @property
    def num_trajectories(self) -> int:
        return int(self.ends.shape[0])

    def __len__(self) -> int:
        return self.num_transitions

    @property
    def observation_space(self) -> Tuple[int, ...]:
        """Encoder observation space (includes the augmented physics dims)."""
        return (self.encoder_obs_dim,)

    @property
    def agent_observation_space(self) -> Tuple[int, ...]:
        return (self.raw_obs_dim,)

    @property
    def encoder_observation_space(self) -> Tuple[int, ...]:
        return (self.encoder_obs_dim,)

    def physics_index(self, name: str) -> int:
        """Index of a physics quantity inside the *encoder* observation vector."""
        if name not in self.physics_names:
            raise KeyError(f"{name!r} is not appended for domain {self.domain!r}")
        return self.raw_obs_dim + list(self.physics_names).index(name)

    # ------------------------------------------------------------------ observations
    def agent_observations(self) -> np.ndarray:
        """The raw observation space the RL components are trained on."""
        return self.observations

    def encoder_observations(self) -> np.ndarray:
        """``concat(observation, physics)`` — the FRE encoder input (Appendix C.2)."""
        if self.physics_features is None:
            return self.observations
        return np.concatenate([self.observations, self.physics_features], axis=-1).astype(np.float32)

    def augment_observations(
        self,
        observations: Any,
        physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        method: str = "auto",
        normalized: bool = True,
    ) -> np.ndarray:
        """Append physics features to arbitrary (possibly un-normalized) observations.

        Used at evaluation time, where states come straight from the environment
        rather than from the offline dataset.
        """
        obs = _as_2d_float(observations)
        if normalized and self.state_mean is not None:
            obs_norm = normalize_observations(obs, self.state_mean, self.state_std)
        else:
            obs_norm = obs
        if physics_fn is None:
            physics_fn = self._physics_fn
        features = compute_physics_features(
            obs_norm if normalized else obs, domain=self.domain, physics_fn=physics_fn, method=method
        )
        if normalized and self.encoder_state_mean is not None:
            physics_norm = (features - self.encoder_state_mean[self.raw_obs_dim :]) / np.maximum(
                self.encoder_state_std[self.raw_obs_dim :], _STD_FLOOR
            )
            return np.concatenate([obs_norm, physics_norm], axis=-1).astype(np.float32)
        return np.concatenate([obs_norm, features], axis=-1).astype(np.float32)

    _physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = field(default=None, repr=False)

    # ------------------------------------------------------------------ goals
    def select_goal_states(
        self,
        num_goals: int = NUM_EXORL_GOALS,
        seed: Optional[int] = None,
        method: str = "uniform",
    ) -> np.ndarray:
        """Select the five *consistent* goal states from the offline dataset (Section C.2).

        The selection is deterministic given ``seed`` so that all methods are
        evaluated on the exact same goals ("kept fixed throughout the online
        evaluation").
        """
        num_goals = int(num_goals)
        if num_goals <= 0:
            raise ValueError("num_goals must be positive")
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        total = self.observations.shape[0]
        if method == "uniform":
            # uniformly spaced anchors, jittered deterministically
            anchors = np.linspace(0, total - 1, num_goals).astype(np.int64)
        else:
            anchors = rng.choice(total, size=num_goals, replace=False).astype(np.int64)
        jitter = rng.integers(0, max(1, total // (num_goals * 4) or 1), size=num_goals)
        idx = np.unique(np.clip(anchors + jitter, 0, total - 1))
        if idx.shape[0] < num_goals:
            extra = rng.choice(total, size=num_goals, replace=False)
            idx = np.unique(np.concatenate([idx, extra]))[:num_goals]
        goals = self.observations[idx[:num_goals]].astype(np.float32)
        self.goal_states = goals
        return goals

    def goal_distance(
        self,
        observations: Any,
        goal: Any,
        normalize: bool = True,
        use_physics: bool = False,
    ) -> np.ndarray:
        """Euclidean distance to ``goal`` in the (std-normalized) raw observation space.

        Section C.2: "Each state dimension is normalized according to the standard
        deviation along that dimension within the offline dataset. Augmented
        information is not utilized when calculating goal distance."
        """
        obs = _as_2d_float(observations)
        if obs.shape[-1] != self.raw_obs_dim:
            obs = obs[:, : self.raw_obs_dim]
        goal_arr = _as_1d_float(goal)[: self.raw_obs_dim]
        if use_physics:
            raise ValueError("augmented information is not utilized when calculating goal distance")
        if not normalize:
            raw_obs = unnormalize_observations(obs, self.state_mean, self.state_std)
            raw_goal = unnormalize_observations(goal_arr.reshape(1, -1), self.state_mean, self.state_std)[0]
            return np.linalg.norm(raw_obs - raw_goal, axis=-1)
        return np.linalg.norm(obs - goal_arr, axis=-1)

    def goal_reached(
        self,
        observations: Any,
        goal: Any,
        threshold: Optional[float] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Boolean 0.1-distance goal test of Section C.2."""
        thr = self.goal_distance_threshold if threshold is None else float(threshold)
        return self.goal_distance(observations, goal, normalize=normalize) < thr

    def goal_distances(self, observations: Any, normalize: bool = True) -> np.ndarray:
        """Distances to all cached goal states, shape ``(num_goals, N)``."""
        goals = self.goal_states
        if goals is None:
            goals = self.select_goal_states()
        return np.stack([self.goal_distance(observations, g, normalize=normalize) for g in goals], axis=0)

    # ------------------------------------------------------------------ trajectories
    def trajectory_slice(self, traj_index: int) -> slice:
        prev = -1 if traj_index == 0 else int(self.ends[traj_index - 1])
        return slice(prev + 1, int(self.ends[traj_index]) + 1)

    def trajectory_length(self, traj_index: int) -> int:
        sl = self.trajectory_slice(traj_index)
        return int(sl.stop - sl.start)

    def start_state(self, observations: Optional[np.ndarray] = None):
        """Evaluation start state: mean of the dataset's first states (documented default).

        Source: not specified in the paper for ExORL; the dm_control default reset
        distribution is used by the environment, and the *encoder context* is sampled
        from the offline dataset, so no fixed start state is required.
        """
        if self._start_state is None:
            starts = [self.trajectory_slice(i).start for i in range(self.num_trajectories)]
            self._start_state = self.observations[starts].mean(axis=0).astype(np.float32)
        return self._start_state

    def state_at(self, traj_index: int, step_index: int) -> np.ndarray:
        sl = self.trajectory_slice(traj_index)
        idx = int(np.clip(sl.start + int(step_index), sl.start, sl.stop))
        return self.observations[idx]

    # ------------------------------------------------------------------ buffers
    def statistics(self):
        """``(mean, std)`` of the raw observations used by Q / V / policy."""
        return self.state_mean, self.state_std

    def state_statistics(self):
        return self.state_mean, self.state_std

    def encoder_statistics(self):
        """``(mean, std)`` of the encoder input (observation + physics)."""
        return self.encoder_state_mean, self.encoder_state_std

    def state_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Min / max of the encoder-input states (used to bound linear reward ranges)."""
        low = (self.observations.min(axis=0) - 3.0 * self.state_std).astype(np.float32)
        high = (self.observations.max(axis=0) + 3.0 * self.state_std).astype(np.float32)
        if self.num_physics_features == 0:
            return low, high
        enc = self.encoder_observations()
        enc_low = low
        enc_high = high
        phys_mean = self.encoder_state_mean[self.raw_obs_dim :]
        phys_std = self.encoder_state_std[self.raw_obs_dim :]
        enc_low = np.concatenate([enc_low, (phys_mean - 3.0 * phys_std).astype(np.float32)])
        enc_high = np.concatenate([enc_high, (phys_mean + 3.0 * phys_std).astype(np.float32)])
        del enc
        return enc_low, enc_high

    def build_buffer(
        self,
        normalize: Optional[bool] = None,
        seed: Optional[int] = None,
        exclude_final_states: bool = True,
        attach_stats: bool = True,
    ):
        """Build the :class:`~fre.data.replay.ReplayBuffer` consumed by training.

        The buffer's ``observations`` are the raw (normalized) environment states,
        while ``encoder_observations`` carry the Appendix C.2 physics augmentation.
        """
        if make_replay_buffer is None:  # pragma: no cover - guarded import
            raise RuntimeError("fre.data.replay is unavailable; cannot build the replay buffer")
        if normalize is not None and bool(normalize) != bool(self.normalize):
            if not self.normalize:
                # normalize in place first, then rebuild the physics features
                raw_physics = self.physics_features
                self._compute_statistics()
                self.normalize_()
                del raw_physics
        buffer = make_replay_buffer(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            timeouts=self.timeouts,
            next_observations=self.next_observations,
            ends=self.ends,
            encoder_observations=self.encoder_observations(),
            seed=self.seed if seed is None else int(seed),
            exclude_final_states=exclude_final_states,
        )
        if attach_stats:
            try:
                buffer.state_box = lambda: self.state_box()  # type: ignore[attr-defined]
                buffer.state_mean_std = lambda: (self.state_mean, self.state_std)  # type: ignore[attr-defined]
            except Exception:
                pass
        self.buffer = buffer
        return buffer

    @property
    def replay_buffer(self):
        if self.buffer is None:
            self.build_buffer()
        return self.buffer

    def sample_states(self, num_states: int, rng=None, encoder_input: bool = True) -> np.ndarray:
        return self.replay_buffer.sample_states(num_states, rng=rng, encoder_input=encoder_input)

    def sample_states_with_metadata(self, num_states: int, rng=None):
        return self.replay_buffer.sample_states_with_metadata(num_states, rng=rng)

    # ------------------------------------------------------------------ misc
    def describe(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "task": self.task,
            "dataset_name": self.dataset_name,
            "dataset": self.dataset,
            "dataset_file": self.dataset_file,
            "num_transitions": self.num_transitions,
            "num_trajectories": self.num_trajectories,
            "obs_dim": self.raw_obs_dim,
            "encoder_obs_dim": self.encoder_obs_dim,
            "act_dim": self.act_dim,
            "physics_features": list(self.physics_names),
            "num_goals": int(0 if self.goal_states is None else self.goal_states.shape[0]),
            "goal_distance_threshold": self.goal_distance_threshold,
            "eval_episode_length": self.eval_episode_length,
            "normalized": bool(self.normalize),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ExORLDataset(domain={self.domain!r}, task={self.task!r}, "
            f"transitions={self.num_transitions}, obs_dim={self.raw_obs_dim}, "
            f"encoder_obs_dim={self.encoder_obs_dim})"
        )


# ======================================================================================
# factories
# ======================================================================================


def make_exorl_dataset(
    observations: Any,
    actions: Any,
    rewards: Optional[Any] = None,
    terminals: Optional[Any] = None,
    timeouts: Optional[Any] = None,
    next_observations: Optional[Any] = None,
    ends: Optional[Any] = None,
    domain: str = "walker",
    task: Optional[str] = None,
    dataset_name: Optional[str] = None,
    physics_features: Optional[Any] = None,
    physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    physics_method: str = "auto",
    normalize: bool = True,
    select_goals: bool = True,
    seed: int = 0,
    **kwargs: Any,
) -> ExORLDataset:
    """Factory mirroring :func:`fre.data.antmaze_dataset.make_antmaze_dataset`."""
    raw = _as_2d_float(observations)
    domain = _resolve_domain(domain, task, dataset_name)
    if physics_features is None:
        physics_features = compute_physics_features(
            raw, domain=domain, physics_fn=physics_fn, method=physics_method
        )
    dataset = ExORLDataset(
        observations=raw,
        actions=_as_2d_float(actions),
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        next_observations=next_observations,
        ends=ends,
        domain=domain,
        task=task or "run",
        dataset_name=dataset_name or _resolve_dataset_name(domain, task, dataset_name),
        physics_features=physics_features,
        normalize=normalize,
        seed=seed,
        **kwargs,
    )
    dataset._physics_fn = physics_fn  # type: ignore[attr-defined]
    if select_goals and dataset.goal_states is None:
        dataset.select_goal_states(num_goals=NUM_EXORL_GOALS, seed=seed)
    return dataset


def load_exorl_dataset(
    path: Optional[str] = None,
    domain: str = "walker",
    task: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset: str = EXORL_RND_DATASET,
    physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    physics_method: str = "auto",
    normalize: bool = True,
    select_goals: bool = True,
    num_goals: int = NUM_EXORL_GOALS,
    build_buffer: bool = True,
    exclude_final_states: bool = True,
    attach_stats: bool = True,
    max_transitions: Optional[int] = None,
    seed: int = 0,
    **kwargs: Any,
) -> ExORLDataset:
    """Full ExORL pipeline: load RND dump -> physics augmentation -> normalization.

    Parameters mirror the paper's Appendix C.2 requirements; ``dataset`` defaults
    to the RND exploratory dataset used for training.
    """
    domain = _resolve_domain(domain, task, dataset_name)
    dataset_name = _resolve_dataset_name(domain, task, dataset_name)
    arrays = load_exorl_arrays(
        path=path,
        domain=domain,
        dataset=dataset,
        dataset_name=dataset_name,
        max_transitions=max_transitions,
    )
    file_path = _resolve_dataset_file(path, domain=domain, dataset=dataset, dataset_name=dataset_name)
    dataset_obj = make_exorl_dataset(
        observations=arrays["observations"],
        actions=arrays.get("actions", np.zeros((arrays["observations"].shape[0], 6), dtype=np.float32)),
        rewards=arrays.get("rewards"),
        terminals=arrays.get("terminals"),
        timeouts=arrays.get("timeouts"),
        next_observations=arrays.get("next_observations"),
        domain=domain,
        task=task,
        dataset_name=dataset_name,
        physics_fn=physics_fn,
        physics_method=physics_method,
        normalize=normalize,
        select_goals=select_goals,
        seed=seed,
        **kwargs,
    )
    dataset_obj.dataset = dataset
    dataset_obj.dataset_file = file_path
    if num_goals != NUM_EXORL_GOALS:
        dataset_obj.select_goal_states(num_goals=num_goals, seed=seed)
    if build_buffer:
        dataset_obj.build_buffer(
            seed=seed, exclude_final_states=exclude_final_states, attach_stats=attach_stats
        )
    return dataset_obj
