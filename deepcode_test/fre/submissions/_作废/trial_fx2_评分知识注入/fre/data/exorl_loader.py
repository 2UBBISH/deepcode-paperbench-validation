"""ExORL offline dataset loading utilities.

The ExORL datasets used by FRE (Walker and Cheetah exploratory data) are
typically distributed as one HDF5 file per environment.  This module keeps the
loader dependency-light and format-tolerant: it accepts ``.hdf5``/``.h5``
files, ``.npz`` archives, or a directory containing either of those.  The
resulting transitions are converted into the internal :class:`OfflineDataset`
representation used throughout the rest of the repository.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fre.config import DataConfig
from fre.data.dataset import OfflineDataset

__all__ = [
    "load_exorl_dataset",
    "load_exorl_dataset_and_env",
    "load_walker_dataset",
    "load_cheetah_dataset",
]


# Canonical ExORL exploratory dataset names.  Forward/backward velocity and
# goal-reaching evaluations reuse the same offline exploratory data.
_EXORL_DATASET_BY_DOMAIN = {
    "walker": "walker_walk",
    "cheetah": "cheetah_run",
}

_TRANSITION_KEYS = ("states", "actions", "next_states", "rewards", "terminals", "timeouts")


def _resolve_exorl_dataset_name(cfg: DataConfig, env_name: Optional[str] = None) -> str:
    """Return the dataset file base name for an ExORL environment."""
    if env_name is None:
        env_name = getattr(cfg, "env_name", None) or getattr(cfg, "dataset_name", None)

    if env_name is not None:
        env_name = str(env_name).lower()
        # Accept both 'walker' and longer identifiers such as 'walker-goal'.
        for domain, dataset_name in _EXORL_DATASET_BY_DOMAIN.items():
            if domain in env_name:
                return dataset_name

    explicit = getattr(cfg, "exorl_dataset_name", None) or getattr(cfg, "dataset_name", None)
    if explicit is not None and str(explicit).strip():
        return str(explicit)

    raise ValueError(
        "Could not determine the ExORL dataset name. "
        "Set cfg.env_name to 'walker'/'cheetah' or cfg.exorl_dataset_name explicitly."
    )


def _resolve_exorl_path(cfg: DataConfig, env_name: Optional[str] = None) -> str:
    """Locate the ExORL dataset file/directory on disk."""
    dataset_name = _resolve_exorl_dataset_name(cfg, env_name)
    root = getattr(cfg, "exorl_data_path", None) or os.environ.get("EXORL_DATA_PATH", None)
    if root is None or not str(root).strip():
        # Fall back to D4RL path only if the caller placed ExORL data there.
        root = getattr(cfg, "d4rl_data_path", None) or os.environ.get("D4RL_DATA_PATH", None)
    if root is None or not str(root).strip():
        root = os.path.join(os.getcwd(), "exorl_data")

    root = os.path.expanduser(str(root))
    candidates = [
        os.path.join(root, dataset_name),
        os.path.join(root, f"{dataset_name}.hdf5"),
        os.path.join(root, f"{dataset_name}.h5"),
        os.path.join(root, f"{dataset_name}.npz"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # Be permissive with suffixes.
    for pattern in (f"{dataset_name}.*",):
        matches = sorted(glob.glob(os.path.join(root, pattern)))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"ExORL dataset not found. Expected one of: {candidates}. "
        "Set cfg.exorl_data_path or EXORL_DATA_PATH."
    )


# ---------------------------------------------------------------------------
# Observation flattening helpers
# ---------------------------------------------------------------------------

def _as_leaf_array(value: Any) -> np.ndarray:
    """Convert an HDF5 dataset, group, or array-like leaf into a numpy array."""
    # h5py Dataset/Group detection is duck-typed to avoid a hard h5py import.
    if hasattr(value, "__iter__") and hasattr(value, "shape") and not isinstance(value, dict):
        arr = np.asarray(value)
    elif hasattr(value, "keys") and hasattr(value, "values"):
        arr = _flatten_observation_group(value)
    elif isinstance(value, dict):
        arr = _flatten_observation_group(value)
    else:
        arr = np.asarray(value)
    if arr.dtype == object:
        # A few ExORL snapshots store observations as arrays of dicts.
        rows = [_flatten_observation_group(row) for row in arr]
        if rows:
            return np.stack(rows, axis=0)
        return arr.reshape(-1, 0)
    return arr


def _flatten_observation_group(obs_group: Any) -> np.ndarray:
    """Flatten a nested observation container into a [T, obs_dim] matrix."""
    if hasattr(obs_group, "keys"):
        keys = sorted(obs_group.keys())
    elif isinstance(obs_group, dict):
        keys = sorted(obs_group.keys())
    else:
        arr = np.asarray(obs_group)
        return arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)

    columns: List[np.ndarray] = []
    for key in keys:
        arr = _as_leaf_array(obs_group[key])
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim == 1:
            arr = arr[:, None]
        else:
            arr = arr.reshape(arr.shape[0], -1)
        columns.append(arr)

    if not columns:
        raise ValueError("Observation group contains no fields.")

    lengths = {c.shape[0] for c in columns}
    if len(lengths) > 1:
        # Pad or trim inconsistent fields; this should rarely happen.
        target = max(lengths)
        padded = []
        for c in columns:
            if c.shape[0] < target:
                pad = np.zeros((target - c.shape[0], c.shape[1]), dtype=np.float32)
                c = np.concatenate([c, pad], axis=0)
            padded.append(c[:target])
        columns = padded
    return np.concatenate(columns, axis=1)


def _observation_matrix(container: Any, key_candidates: Tuple[str, ...]) -> Optional[np.ndarray]:
    """Read the first present observation key from an HDF5 group/dict."""
    for key in key_candidates:
        if hasattr(container, "keys") and key in container:
            return _as_leaf_array(container[key])
        if isinstance(container, dict) and key in container:
            return _as_leaf_array(container[key])
    return None


def _array_matrix(container: Any, key_candidates: Tuple[str, ...]) -> Optional[np.ndarray]:
    """Read a regular 1-D/2-D transition array from an HDF5 group/dict."""
    for key in key_candidates:
        if hasattr(container, "keys") and key in container:
            arr = np.asarray(container[key])
            if arr.ndim == 0:
                arr = arr.reshape(1)
            return arr
        if isinstance(container, dict) and key in container:
            arr = np.asarray(container[key])
            if arr.ndim == 0:
                arr = arr.reshape(1)
            return arr
    return None


def _boolean_array(container: Any, key_candidates: Tuple[str, ...], length: int, default: int) -> np.ndarray:
    arr = _array_matrix(container, key_candidates)
    if arr is None:
        return np.full(length, default, dtype=np.bool_)
    arr = arr.reshape(-1)
    if arr.shape[0] != length:
        arr = np.resize(arr, length).astype(np.bool_)
    return arr.astype(np.bool_)


def _derive_next_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.shape[0] < 2:
        return states.copy()
    return np.concatenate([states[1:], states[-1:]], axis=0)


def _episode_to_transitions(group: Any) -> Optional[Dict[str, np.ndarray]]:
    """Convert one HDF5 episode group into a transition dictionary."""
    states = _observation_matrix(group, ("observations", "observation", "states", "state"))
    if states is None:
        return None
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(-1, 1)

    length = states.shape[0]
    actions = _array_matrix(group, ("actions", "action"))
    if actions is None:
        # No actions => not a usable RL trajectory.
        return None
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    actions = actions[:length]

    next_states = _observation_matrix(group, ("next_observations", "next_observation", "next_states", "next_state"))
    if next_states is None:
        next_states = _derive_next_states(states)
    else:
        next_states = np.asarray(next_states, dtype=np.float32)
        if next_states.ndim == 1:
            next_states = next_states.reshape(-1, 1)
        next_states = next_states[:length]

    rewards = _array_matrix(group, ("rewards", "reward"))
    if rewards is None:
        rewards = np.zeros(length, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)[:length]

    terminals = _boolean_array(
        group,
        ("terminals", "terminal", "is_terminal", "dones", "done"),
        length,
        default=0,
    )
    # HDF5 boolean datasets sometimes arrive as numeric arrays; enforce bool.
    terminals = terminals.astype(np.bool_)

    # ExORL marks episode boundaries with is_last/is_first.  The last
    # transition is a timeout when it is the final step but not terminal.
    is_last = _boolean_array(group, ("is_last", "last"), length, default=0).astype(np.bool_)
    timeouts = _boolean_array(group, ("timeouts", "timeout"), length, default=0).astype(np.bool_)
    if not timeouts.any() and is_last.any():
        timeouts = (is_last & ~terminals).astype(np.bool_)
    if terminals.any():
        timeouts = timeouts & ~terminals

    return {
        "states": states,
        "actions": actions,
        "next_states": next_states,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }


def _concatenate_transitions(
    transitions: List[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    if not transitions:
        raise ValueError("No transitions found in ExORL dataset.")
    out: Dict[str, np.ndarray] = {}
    for key in _TRANSITION_KEYS:
        out[key] = np.concatenate([t[key] for t in transitions], axis=0)
    return out


def _load_transitions_from_hdf5(path: str) -> Dict[str, np.ndarray]:
    """Load transitions from a D4RL-style or episode-grouped HDF5 file."""
    import h5py  # lazy import so the rest of the package works without ExORL data

    with h5py.File(path, "r") as f:
        root_keys = list(f.keys())

        # Case 1: flat transition arrays at the root.
        flat = _episode_to_transitions(f)
        if flat is not None and _array_matrix(f, ("actions", "action")) is not None:
            return flat

        # Case 2: one group per episode (the canonical ExORL format).
        transitions: List[Dict[str, np.ndarray]] = []
        for key in sorted(root_keys):
            node = f[key]
            if hasattr(node, "keys"):
                ep = _episode_to_transitions(node)
                if ep is not None:
                    transitions.append(ep)

        if transitions:
            return _concatenate_transitions(transitions)

        # Case 3: root itself was not flat but a single episode group-like object.
        if flat is not None:
            return flat

    raise ValueError(f"Could not parse ExORL HDF5 dataset: {path}")


def _load_transitions_from_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        states = _observation_matrix(data, ("observations", "observation", "states", "state"))
        if states is None:
            states = data[keys[0]] if keys else np.zeros((0, 1), dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(-1, 1)

        actions = _array_matrix(data, ("actions", "action"))
        if actions is None:
            raise ValueError(f"ExORL npz dataset has no actions: {path}")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        next_states = _observation_matrix(data, ("next_observations", "next_observation", "next_states", "next_state"))
        if next_states is None:
            next_states = _derive_next_states(states)
        next_states = np.asarray(next_states, dtype=np.float32)
        if next_states.ndim == 1:
            next_states = next_states.reshape(-1, 1)

        rewards = _array_matrix(data, ("rewards", "reward"))
        rewards = np.zeros(states.shape[0], dtype=np.float32) if rewards is None else np.asarray(rewards, dtype=np.float32).reshape(-1)

        length = states.shape[0]
        terminals = _boolean_array(data, ("terminals", "terminal", "is_terminal", "dones", "done"), length, 0)
        timeouts = _boolean_array(data, ("timeouts", "timeout"), length, 0)

        return {
            "states": states,
            "actions": actions[:length],
            "next_states": next_states[:length],
            "rewards": rewards[:length],
            "terminals": terminals,
            "timeouts": timeouts,
        }


def load_exorl_transitions(path: str) -> Dict[str, np.ndarray]:
    """Load raw ExORL transitions from a file or directory."""
    if os.path.isdir(path):
        transitions: List[Dict[str, np.ndarray]] = []
        files = sorted(
            glob.glob(os.path.join(path, "*.hdf5"))
            + glob.glob(os.path.join(path, "*.h5"))
            + glob.glob(os.path.join(path, "*.npz"))
        )
        if not files:
            raise FileNotFoundError(f"No ExORL data files found in directory: {path}")
        for f in files:
            transitions.append(load_exorl_transitions(f))
        return _concatenate_transitions(transitions)

    if path.endswith(".npz"):
        return _load_transitions_from_npz(path)
    if path.endswith((".hdf5", ".h5")):
        return _load_transitions_from_hdf5(path)
    raise ValueError(f"Unsupported ExORL data file format: {path}")


def load_exorl_dataset(
    cfg: DataConfig,
    env_name: Optional[str] = None,
    device: str = "cpu",
    data_path: Optional[str] = None,
) -> OfflineDataset:
    """Load an ExORL offline dataset into :class:`OfflineDataset`.

    Parameters
    ----------
    cfg:
        Dataset configuration. ``exorl_data_path`` and ``env_name`` /
        ``exorl_dataset_name`` are used to locate the data.
    env_name:
        Optional override, e.g. ``"walker"`` or ``"cheetah"``.
    device:
        Default device for sampled tensors.
    data_path:
        Optional explicit file/directory path; bypasses config resolution.
    """
    path = data_path if data_path is not None else _resolve_exorl_path(cfg, env_name)
    transitions = load_exorl_transitions(path)
    return OfflineDataset(transitions, cfg=cfg, device=device)


def load_exorl_dataset_and_env(
    cfg: DataConfig,
    env_name: Optional[str] = None,
    device: str = "cpu",
    data_path: Optional[str] = None,
) -> Tuple[OfflineDataset, Any]:
    """Load the dataset and, if possible, the corresponding live DMC environment."""
    dataset = load_exorl_dataset(cfg, env_name=env_name, device=device, data_path=data_path)
    domain = _resolve_exorl_dataset_name(cfg, env_name)
    env = None
    try:
        from fre.envs.dmc import make_dmc_env

        env = make_dmc_env(domain, seed=getattr(cfg, "seed", 0))
    except Exception:
        env = None
    return dataset, env


def load_walker_dataset(cfg: DataConfig, device: str = "cpu") -> OfflineDataset:
    """Load the canonical Walker exploratory dataset."""
    return load_exorl_dataset(cfg, env_name="walker", device=device)


def load_cheetah_dataset(cfg: DataConfig, device: str = "cpu") -> OfflineDataset:
    """Load the canonical Cheetah exploratory dataset."""
    return load_exorl_dataset(cfg, env_name="cheetah", device=device)
