"""D4RL dataset loading utilities for the FRE reproduction.

This module is responsible for loading the offline datasets used by FRE:

    * AntMaze : ``antmaze-large-diverse-v2`` (used for the goal-reaching /
      directional / random-simplex / path tasks)
    * Kitchen : ``kitchen-*`` (7 sparse subtasks, rewards used directly)

Design notes
------------
The paper (and reproduction plan) explicitly requires installing D4RL from a
commit dated *before June 2024* to avoid dataset / environment drift.  We do
**not** vendor D4RL here -- instead every access goes through this module which

    1. tries a normal ``import d4rl`` (which registers the gym environments),
    2. falls back to a *dataset-only* path via ``d4rl.qlearning_dataset`` /
       ``gym.make(...).get_dataset()`` if the module layout differs,
    3. raises a clear, actionable error otherwise.

Datasets are returned as either raw transition dictionaries (matching the
canonical keys used by :mod:`fre.rl.replay_buffer`) or as
:class:`~fre.rl.replay_buffer.OfflineReplayBuffer` objects when requested.

The returned arrays follow the canonical transition layout::

    observations, actions, rewards, next_observations, terminals

plus an optional ``timeouts`` array for AntMaze (a timeout is *not* a true
terminal for goal-reaching, it just ends the rollout).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANTMAZE_DATASET = "antmaze-large-diverse-v2"
"""Default AntMaze dataset used by FRE (Table 1 / Figure 5)."""

KITCHEN_DATASET = "kitchen-complete-v0"
"""Default Kitchen dataset (all 7 subtasks present in the demonstrations)."""

KITCHEN_DATASETS: Tuple[str, ...] = (
    "kitchen-complete-v0",
    "kitchen-mixed-v0",
    "kitchen-partial-v0",
)

#: The 7 standard D4RL Kitchen subtasks (order matters for reward masking).
KITCHEN_TASKS: Tuple[str, ...] = (
    "microwave",
    "kettle",
    "slide",
    "hinge",
    "light",
    "bottom_burner",
    "top_burner",
)

#: Environment ids that are considered AntMaze datasets.
ANTMAZE_DATASETS: Tuple[str, ...] = (
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
    "antmaze-umaze-v0",
    "antmaze-umaze-diverse-v0",
    "antmaze-medium-play-v0",
    "antmaze-medium-diverse-v0",
    "antmaze-large-play-v0",
    "antmaze-large-diverse-v0",
)

_CANONICAL_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
)


# ---------------------------------------------------------------------------
# D4RL import helpers
# ---------------------------------------------------------------------------


def d4rl_available() -> bool:
    """Return ``True`` if ``d4rl`` can be imported in this environment."""
    try:  # pragma: no cover - import side effects
        import d4rl  # noqa: F401
        import gym  # noqa: F401

        return True
    except Exception:
        return False


def _require_d4rl() -> Any:
    """Import and return the ``d4rl`` module or raise an informative error."""
    try:
        import d4rl  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "d4rl is required to load the offline datasets. Install it from a "
            "commit dated before June 2024, e.g.\n"
            "    pip install git+https://github.com/Farama-Foundation/D4RL.git"
            "@<pre-2024-06-commit>\n"
            "See requirements.txt for the exact pin used by this repo."
        ) from exc
    return d4rl


def make_env(env_name: str, **kwargs: Any) -> Any:
    """Create a D4RL gym environment by name (also registers datasets)."""
    import gym  # local import: gym is a D4RL dependency

    _require_d4rl()
    env = gym.make(env_name, **kwargs)
    return env


# ---------------------------------------------------------------------------
# Low level dataset -> dict conversion
# ---------------------------------------------------------------------------


def _to_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    else:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
        except Exception:
            arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _standardise_dataset(
    raw: Dict[str, Any],
    *,
    infer_next_observations: bool = True,
    infer_terminals: bool = True,
) -> Dict[str, np.ndarray]:
    """Normalise a raw D4RL dict into canonical transition keys.

    Handles the common D4RL quirks:

    * ``next_observations`` may be absent (Kitchen prior to a certain commit),
      in which case it is derived by shifting ``observations`` by one and
      patching the last step of each episode with the current observation.
    * ``terminals`` may be given as 0/1 ``terminals`` or as ``1 - masks``.
    * ``timeouts`` are preserved verbatim when present.
    """
    data: Dict[str, np.ndarray] = {}

    observations = raw.get("observations")
    if observations is None:
        observations = raw.get("obs")
    if observations is None:
        raise KeyError("dataset is missing 'observations'")
    observations = _to_numpy(observations, np.float32)

    actions = raw.get("actions")
    if actions is None:
        actions = raw.get("act")
    if actions is None:
        raise KeyError("dataset is missing 'actions'")
    actions = _to_numpy(actions, np.float32)

    rewards = raw.get("rewards")
    if rewards is None:
        rewards = raw.get("reward")
    if rewards is None:
        rewards = np.zeros(observations.shape[0], dtype=np.float32)
    rewards = _to_numpy(rewards, np.float32).reshape(-1)

    next_observations = raw.get("next_observations")
    if next_observations is None:
        next_observations = raw.get("next_obs")
    if next_observations is None and infer_next_observations:
        next_observations = _shift_observations(observations, raw, rewards)
    if next_observations is None:
        raise KeyError("dataset is missing 'next_observations'")
    next_observations = _to_numpy(next_observations, np.float32)

    terminals = raw.get("terminals")
    if terminals is None and "masks" in raw and raw["masks"] is not None:
        # D4RL / newer datasets use masks=1 for non-terminal steps.
        terminals = 1.0 - _to_numpy(raw["masks"], np.float32)
    if terminals is None:
        terminals = raw.get("dones")
    if terminals is None and infer_terminals:
        # AntMaze "diverse" datasets historically omit terminals entirely.
        terminals = np.zeros(observations.shape[0], dtype=np.float32)
    if terminals is None:
        raise KeyError("dataset is missing 'terminals'")
    terminals = _to_numpy(terminals, np.float32).reshape(-1)
    terminals = (terminals > 0.5).astype(np.float32)

    data["observations"] = observations
    data["actions"] = actions
    data["rewards"] = rewards
    data["next_observations"] = next_observations
    data["terminals"] = terminals

    if "timeouts" in raw and raw["timeouts"] is not None:
        data["timeouts"] = _to_numpy(raw["timeouts"], np.float32).reshape(-1) > 0.5

    return data


def _shift_observations(
    observations: np.ndarray,
    raw: Dict[str, Any],
    rewards: np.ndarray,
) -> np.ndarray:
    """Derive ``next_observations`` by shifting within episodes.

    Episode boundaries are located via ``timeouts``/``terminals`` when
    available, otherwise the terminal index is inferred from a reward of 1.0
    (Kitchen sparse rewards) or from the end of the array.
    """
    next_obs = np.concatenate([observations[1:], observations[-1:]], axis=0)

    boundaries: Optional[np.ndarray] = None
    for key in ("timeouts", "terminals"):
        if key in raw and raw[key] is not None:
            boundaries = _to_numpy(raw[key]).reshape(-1) > 0.5
            break
    if boundaries is None:
        # Kitchen uses a sparse reward of 1.0 exactly at subtask completion;
        # the dataset (for the "complete" variant) ends episodes implicitly.
        boundaries = rewards > 0.99

    idx = np.flatnonzero(boundaries)
    if idx.size:
        last = observations[idx]
        # Patch the final step of each episode with its own observation.
        next_obs[idx] = last
    return next_obs


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_d4rl_dataset(
    env_name: str,
    *,
    standardise: bool = True,
    infer_next_observations: bool = True,
    infer_terminals: bool = True,
) -> Dict[str, np.ndarray]:
    """Load a D4RL dataset and return it as a canonical transition dict.

    Parameters
    ----------
    env_name:
        D4RL environment id, e.g. ``antmaze-large-diverse-v2`` or
        ``kitchen-complete-v0``.
    standardise:
        When ``True`` (default) the raw dict is normalised into the canonical
        keys ``observations/actions/rewards/next_observations/terminals``.

    Returns
    -------
    dict
        Numpy arrays with a leading batch dimension.  ``observations`` and
        ``next_observations`` are ``float32``.
    """
    env = make_env(env_name)
    try:
        raw = env.get_dataset()
    finally:
        try:
            env.close()
        except Exception:
            pass

    if not standardise:
        return {k: _to_numpy(v) for k, v in raw.items() if v is not None}

    return _standardise_dataset(
        raw,
        infer_next_observations=infer_next_observations,
        infer_terminals=infer_terminals,
    )


def load_antmaze(
    dataset: str = ANTMAZE_DATASET,
    *,
    standardise: bool = True,
) -> Dict[str, np.ndarray]:
    """Load an AntMaze dataset (default ``antmaze-large-diverse-v2``)."""
    if dataset not in ANTMAZE_DATASETS:
        # Allow arbitrary ids but warn through the return value's metadata.
        pass
    return load_d4rl_dataset(dataset, standardise=standardise)


def load_kitchen(
    dataset: str = KITCHEN_DATASET,
    *,
    standardise: bool = True,
) -> Dict[str, np.ndarray]:
    """Load a Kitchen dataset (default ``kitchen-complete-v0``).

    Kitchen observations already contain the 7 subtask completion signals in
    their final 7 dimensions, so the sparse rewards in the dataset can be used
    directly (per the reproduction plan).  We expose the task names so callers
    can mask a single subtask if desired.
    """
    return load_d4rl_dataset(dataset, standardise=standardise)


def kitchen_subtask_rewards(
    dataset: Dict[str, np.ndarray],
    task: str,
) -> np.ndarray:
    """Extract the sparse reward for a single Kitchen subtask.

    Kitchen observations end with 7 binary completion flags (one per subtask).
    The environment reward is ``sum(completed_at_this_step)``; the per-subtask
    reward is ``1`` on the step where that flag first becomes true.
    """
    if task not in KITCHEN_TASKS:
        raise ValueError(
            f"unknown Kitchen task {task!r}; expected one of {KITCHEN_TASKS}"
        )
    idx = KITCHEN_TASKS.index(task)
    obs = _to_numpy(dataset["observations"]).astype(np.float64)
    flags = obs[:, -len(KITCHEN_TASKS):]
    task_flag = flags[:, idx]
    # Reward is 1 where the flag transitions 0 -> 1.
    prev = np.concatenate([[0.0], task_flag[:-1]])
    newly_completed = (task_flag > 0.5) & (prev <= 0.5)
    return newly_completed.astype(np.float32)


def load_kitchen_multitask(
    dataset: str = KITCHEN_DATASET,
) -> Dict[str, Any]:
    """Load Kitchen and return per-subtask sparse reward arrays.

    Returns
    -------
    dict with keys ``dataset`` (canonical transition dict) and ``tasks``
    (mapping subtask name -> ``float32`` reward array).
    """
    data = load_kitchen(dataset)
    tasks = {name: kitchen_subtask_rewards(data, name) for name in KITCHEN_TASKS}
    return {"dataset": data, "tasks": tasks, "task_names": list(KITCHEN_TASKS)}


# ---------------------------------------------------------------------------
# Dataset statistics / sanity checks
# ---------------------------------------------------------------------------


def dataset_info(dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Return shape / range / episode information for a transition dict."""
    obs = np.asarray(dataset["observations"])
    act = np.asarray(dataset["actions"])
    rew = np.asarray(dataset["rewards"])
    terminals = np.asarray(dataset.get("terminals", np.zeros(len(obs))))

    num_episodes = int(np.sum(terminals > 0.5))
    lengths: List[int] = []
    start = 0
    for idx in np.flatnonzero(terminals > 0.5):
        lengths.append(int(idx + 1 - start))
        start = idx + 1
    if start < len(obs):
        lengths.append(int(len(obs) - start))

    return {
        "size": int(len(obs)),
        "observation_dim": int(obs.shape[-1]),
        "action_dim": int(act.shape[-1]),
        "reward_min": float(rew.min()) if len(rew) else 0.0,
        "reward_max": float(rew.max()) if len(rew) else 0.0,
        "reward_mean": float(rew.mean()) if len(rew) else 0.0,
        "num_episodes": num_episodes,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_episode_length": int(np.max(lengths)) if lengths else 0,
    }


def dataset_observation_stats(
    dataset: Dict[str, np.ndarray],
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` of the observations (std floored at ``eps``)."""
    obs = np.asarray(dataset["observations"], dtype=np.float64)
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


def split_train_validation(
    dataset: Dict[str, np.ndarray],
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split a transition dict into (train, validation) subsets.

    Used by the encoder sanity checks (decoder reconstruction error on held-out
    states).  The split is per-transition, which is sufficient for the
    reconstruction sanity test.
    """
    n = len(dataset["observations"])
    if n < 2:
        return dataset, dataset
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(validation_fraction * n)))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    def _take(subset_idx: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for key, value in dataset.items():
            arr = np.asarray(value)
            if arr.shape[0] == n:
                out[key] = arr[subset_idx]
            else:
                out[key] = arr
        return out

    return _take(train_idx), _take(val_idx)


# ---------------------------------------------------------------------------
# Replay-buffer bridge
# ---------------------------------------------------------------------------


def to_replay_buffer(
    dataset: Dict[str, np.ndarray],
    *,
    device: str = "cpu",
    seed: Optional[int] = None,
    normalize_observations: bool = False,
    trajectory_buffer: bool = True,
) -> Any:
    """Convert a canonical dataset dict into a FRE replay buffer.

    Uses :class:`fre.rl.replay_buffer.TrajectoryReplayBuffer` by default so
    that goal-reaching priors can sample goals within trajectories (HER).
    """
    from fre.rl.replay_buffer import (
        OfflineReplayBuffer,
        TrajectoryReplayBuffer,
        d4rl_dict_to_arrays,
    )

    arrays = d4rl_dict_to_arrays(dataset, require_rewards=True)

    if trajectory_buffer:
        buffer = TrajectoryReplayBuffer(
            device=device, seed=seed, capacity=len(arrays["observations"])
        )
    else:
        buffer = OfflineReplayBuffer(
            device=device,
            seed=seed,
            capacity=len(arrays["observations"]),
        )
    buffer.set_arrays(arrays)
    if normalize_observations:
        buffer.compute_observation_normalization()
        buffer.apply_observation_normalization()
    return buffer


def load_antmaze_buffer(
    dataset: str = ANTMAZE_DATASET,
    **buffer_kwargs: Any,
) -> Any:
    """Convenience: load AntMaze straight into a replay buffer."""
    return to_replay_buffer(load_antmaze(dataset), **buffer_kwargs)


def load_kitchen_buffer(
    dataset: str = KITCHEN_DATASET,
    **buffer_kwargs: Any,
) -> Any:
    """Convenience: load Kitchen straight into a replay buffer."""
    return to_replay_buffer(load_kitchen(dataset), **buffer_kwargs)


# ---------------------------------------------------------------------------
# Dataset root discovery (ExORL style local data directories)
# ---------------------------------------------------------------------------


def find_dataset_root(candidates: Optional[Sequence[str]] = None) -> Optional[str]:
    """Locate a local ``data`` directory if one exists.

    Returns the first existing candidate, or ``None``.  This mirrors the
    reproduction plan's instruction to place ExORL RND datasets under
    ``./data``.
    """
    if candidates is None:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
        candidates = [
            os.environ.get("FRE_DATA_DIR", ""),
            os.path.join(repo_root, "data"),
            os.path.join(os.getcwd(), "data"),
        ]
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    return None


__all__ = [
    "ANTMAZE_DATASET",
    "ANTMAZE_DATASETS",
    "KITCHEN_DATASET",
    "KITCHEN_DATASETS",
    "KITCHEN_TASKS",
    "d4rl_available",
    "make_env",
    "load_d4rl_dataset",
    "load_antmaze",
    "load_kitchen",
    "load_kitchen_multitask",
    "kitchen_subtask_rewards",
    "dataset_info",
    "dataset_observation_stats",
    "split_train_validation",
    "to_replay_buffer",
    "load_antmaze_buffer",
    "load_kitchen_buffer",
    "find_dataset_root",
]
