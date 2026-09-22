"""ExORL (Exploratory Off-policy RL) dataset loading for the FRE reproduction.

The FRE paper pretrains its encoder + policy on the *RND* exploratory datasets from
`ExORL <https://github.com/denisyarats/exorl>`_ for the ``walker`` and ``cheetah``
control domains.  Those datasets are stored as ``*.npz`` archives of off-policy
transitions collected by an RND exploration agent:

.. code-block:: python

    with np.load(path) as f:
        observations   # (N, obs_dim) float32
        actions        # (N, act_dim) float32
        rewards        # (N,)        float32   (environment reward, mostly unused by FRE)
        terminals      # (N,)        float32
        timeouts       # (N,)        float32
        infos          # (N,)        object array of dicts; infos[i]["physics"] -> nth physics vector

FRE only needs ``(observations, actions, terminals)``: the phase-2 reward is *sampled*
from the reward prior ``eta(s)`` rather than read from the dataset.  Three details from
the paper are handled here:

1. **Physics augmentation (encoder only).**  Extra physical quantities are appended to the
   observations that feed the *encoder* (``walker``: horizontal velocity, torso upright,
   torso height; ``cheetah``: speed).  The decoder / policy continue to consume the raw
   observations, so the loader exposes both ``observations`` (raw) and
   ``encoder_observations`` (raw + physics).  Goal / velocity evaluation rewards are
   computed in the physics-augmented space.
2. **Per-dimension std-normalisation.**  ``encoder_observations`` are divided by their
   per-dimension standard deviation (this is what stabilises the ExORL latent space).
3. **Task definitions.**  Goal-reaching tasks use a Euclidean distance threshold of
   ``0.1`` in the physics space, velocity tasks use the fixed threshold grids from the
   paper (``cheetah``: 10, 1; ``walker``: 0.1, 1, 4, 8), and five fixed goal states per
   domain are deterministically selected from the dataset.

The module deliberately does **not** import ``dm_control`` / ``exorl``: only the local
``.npz`` files (or ``torch`` / ``pickle`` archives) are required.  Place them under
``./data`` (or point ``FRE_EXORL_DIR`` / ``root=`` at them).
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Maximum episode length used for ExORL evaluation in the paper.
MAX_EPISODE_STEPS = 1000

#: Default dataset flavour (the RND exploratory datasets used by FRE).
DEFAULT_DATASET_KIND = "rnd"

#: Euclidean goal distance threshold (paper: "goal euclid dist < 0.1").
GOAL_DISTANCE_THRESHOLD = 0.1

#: Number of fixed goal states held out per domain.
NUM_GOAL_STATES = 5

#: Physics quantities appended to the encoder observations, in order.
WALKER_PHYSICS = ("horizontal_velocity", "torso_upright", "torso_height")
CHEETAH_PHYSICS = ("speed",)

#: Fallback indices into the raw ExORL ``physics`` vector when names are unavailable.
#: dm_control ``walker`` reports [horizontal_velocity, vertical_velocity, torso_upright,
#: torso_height] (and possibly more); ``cheetah`` reports [velocity_x, velocity_y] with
#: speed being the signed / unsigned first component.
WALKER_PHYSICS_FALLBACK_INDEX = {
    "horizontal_velocity": 0,
    "vertical_velocity": 1,
    "torso_upright": 2,
    "torso_height": 3,
}
CHEETAH_PHYSICS_FALLBACK_INDEX = {
    "speed": 0,
    "velocity": 0,
}

#: Velocity-task threshold grids from the paper (Table 1 / App. C).
VELOCITY_THRESHOLDS = {
    "cheetah": (10.0, 1.0),
    "walker": (0.1, 1.0, 4.0, 8.0),
}

#: Raw observation dimensions of the dm_control tasks used by ExORL.
RAW_OBS_DIM = {
    "walker": 24,
    "cheetah": 17,
}

EXORL_DOMAINS = ("walker", "cheetah")

#: Canonical transition keys produced by this loader.
TRANSITION_KEYS = ("observations", "actions", "rewards", "next_observations", "terminals")

#: cwd-relative / repo-relative locations searched for the ExORL archives.
_DEFAULT_SUBDIRS = (
    os.path.join("data", "exorl"),
    os.path.join("data", "exorl", "rnd"),
    os.path.join("data", "exorl", "rnd", "dm_control"),
    os.path.join("data", "rnd"),
    os.path.join("data"),
    os.path.join("exorl"),
    os.path.join("exorl", "rnd"),
    ".",
)


# --------------------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------------------


def _default_roots() -> List[str]:
    """Return candidate directories in which ExORL archives may live."""

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    roots: List[str] = []
    env_dir = os.environ.get("FRE_EXORL_DIR") or os.environ.get("EXORL_DATA_DIR")
    if env_dir:
        roots.append(env_dir)
    for base in (os.getcwd(), repo):
        for sub in _DEFAULT_SUBDIRS:
            roots.append(os.path.join(base, sub))
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def find_exorl_root(root: Optional[str] = None) -> Optional[str]:
    """Return the first existing directory that contains ExORL archives."""

    if root is not None:
        return root if os.path.isdir(root) else None
    for candidate in _default_roots():
        if not os.path.isdir(candidate):
            continue
        if find_exorl_files(candidate):
            return candidate
    return None


def find_exorl_files(
    root: Optional[str] = None,
    domain: Optional[str] = None,
    dataset_kind: str = DEFAULT_DATASET_KIND,
) -> List[str]:
    """Locate ``.npz`` (or ``.pkl``) ExORL archives.

    Parameters
    ----------
    root:
        Directory to search.  When ``None`` the default candidate roots are searched.
    domain:
        Restrict to one of ``{"walker", "cheetah"}``; ``None`` matches any domain.
    dataset_kind:
        Sub-directory / filename hint, e.g. ``"rnd"``.
    """

    if domain is not None and domain not in EXORL_DOMAINS:
        raise ValueError(f"unknown ExORL domain {domain!r}; expected one of {EXORL_DOMAINS}")

    roots = [root] if root is not None else _default_roots()
    matches: List[str] = []
    for base in roots:
        if not base or not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            # Do not descend into huge unrelated trees (e.g. site-packages).
            depth = dirpath[len(base):].count(os.sep)
            if depth > 4:
                _dirnames[:] = []
                continue
            for fname in filenames:
                if not fname.endswith((".npz", ".pkl", ".pickle")):
                    continue
                full = os.path.join(dirpath, fname)
                low = full.lower()
                if dataset_kind and dataset_kind not in low and "rnd" not in low:
                    # Allow archives that simply live under a matching root.
                    if not any(k in low for k in EXORL_DOMAINS):
                        continue
                if domain is not None and domain not in low:
                    continue
                if domain is None and not any(k in low for k in EXORL_DOMAINS):
                    continue
                matches.append(full)
    # Deterministic ordering.
    return sorted(set(matches))


def exorl_available(root: Optional[str] = None, domain: Optional[str] = None) -> bool:
    """Return ``True`` if at least one ExORL archive can be found."""

    try:
        return len(find_exorl_files(root=root, domain=domain)) > 0
    except Exception:  # pragma: no cover - defensive
        return False


# --------------------------------------------------------------------------------------
# Low-level archive reading
# --------------------------------------------------------------------------------------


def _to_numpy(value: Any) -> Any:
    """Convert tensors / lists to numpy arrays without importing torch eagerly."""

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception:
            return value
    if hasattr(value, "detach"):  # torch tensor
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return value


def _load_archive(path: str) -> Dict[str, Any]:
    """Load an ``.npz`` or pickled dataset into a plain dict."""

    if path.endswith(".npz") or path.endswith(".npy"):
        with np.load(path, allow_pickle=True) as handle:
            data = {k: handle[k] for k in handle.files}
    else:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"pickled ExORL dataset at {path} is not a dict")
    return {k: _to_numpy(v) for k, v in data.items()}


def _merge_archives(archives: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate multiple dataset shards along the transition axis."""

    if len(archives) == 1:
        return archives[0]
    merged: Dict[str, Any] = {}
    common = set(archives[0].keys())
    for arch in archives[1:]:
        common &= set(arch.keys())
    for key in common:
        arrays = [a[key] for a in archives]
        try:
            merged[key] = np.concatenate(arrays, axis=0)
        except Exception:
            # ``infos`` may be object arrays of dicts -> vstack-free fallback.
            merged[key] = np.array([item for arr in arrays for item in arr], dtype=object)
    # Keep keys that only some shards have (best effort).
    for arch in archives:
        for key, value in arch.items():
            if key not in merged:
                merged[key] = value
    return merged


# --------------------------------------------------------------------------------------
# Physics handling
# --------------------------------------------------------------------------------------


def physics_names(domain: str) -> Tuple[str, ...]:
    """Ordered physics quantities appended to the encoder for ``domain``."""

    if domain == "walker":
        return WALKER_PHYSICS
    if domain == "cheetah":
        return CHEETAH_PHYSICS
    raise ValueError(f"unknown ExORL domain {domain!r}")


def physics_dim(domain: str) -> int:
    """Number of physics dimensions appended to the encoder observations."""

    return len(physics_names(domain))


def _physics_from_infos(infos: Any) -> Optional[np.ndarray]:
    """Extract a ``(N, P)`` physics matrix from the ExORL ``infos`` object array."""

    if infos is None:
        return None
    infos = _to_numpy(infos)
    if not isinstance(infos, np.ndarray) or infos.size == 0:
        return None
    if infos.dtype != object and infos.ndim == 2 and np.issubdtype(infos.dtype, np.number):
        return np.asarray(infos, dtype=np.float32)

    rows: List[np.ndarray] = []
    for item in infos.reshape(-1):
        if item is None:
            return None
        phys = None
        if isinstance(item, dict):
            for key in ("physics", "physics_vec", "procprio_physics"):
                if key in item:
                    phys = item[key]
                    break
        if phys is None:
            return None
        rows.append(np.atleast_1d(np.asarray(phys, dtype=np.float32)))
    if not rows:
        return None
    try:
        return np.stack(rows, axis=0).astype(np.float32)
    except Exception:
        return None


def _select_physics_columns(
    domain: str,
    physics: np.ndarray,
    names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Sub-select the paper's physics quantities from a raw physics matrix."""

    physics = np.atleast_2d(np.asarray(physics, dtype=np.float32))
    if physics.shape[0] == 0:
        return physics
    wanted = list(physics_names(domain))
    fallback = WALKER_PHYSICS_FALLBACK_INDEX if domain == "walker" else CHEETAH_PHYSICS_FALLBACK_INDEX

    if names is not None and len(names) == physics.shape[1]:
        index_map = {str(n).lower(): i for i, n in enumerate(names)}
    else:
        index_map = {}

    cols: List[np.ndarray] = []
    for quantity in wanted:
        idx: Optional[int] = None
        for candidate, i in index_map.items():
            if candidate == quantity or quantity in candidate or candidate in quantity:
                idx = i
                break
        if idx is None:
            idx = fallback.get(quantity)
        if idx is None or idx >= physics.shape[1]:
            # Graceful degradation: append zeros rather than crashing.
            cols.append(np.zeros(physics.shape[0], dtype=np.float32))
        else:
            cols.append(physics[:, idx])
    out = np.stack(cols, axis=-1).astype(np.float32)

    if domain == "cheetah" and out.shape[1] == 1:
        # The paper uses *speed* (magnitude), recorded as a non-negative quantity.
        out = np.abs(out)
    return out


def compute_physics(
    domain: str,
    observations: np.ndarray,
    physics: Optional[np.ndarray] = None,
    physics_names_: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return the ``(N, P)`` physics matrix used by FRE's encoder and eval rewards.

    If a raw ``physics`` array (or ``infos``) is available it is sub-selected to the
    paper's quantities; otherwise a best-effort approximation is derived from the raw
    observations using the known dm_control layouts:

    * ``walker`` — horizontal velocity from the torso x-velocity components,
      ``torso_upright`` from the torso z-axis orientation, ``torso_height`` from the
      torso z position.
    * ``cheetah`` — ``speed`` as the magnitude of the torso linear velocity.
    """

    observations = np.asarray(observations, dtype=np.float32)
    if physics is not None:
        physics = _to_numpy(physics)
        if isinstance(physics, np.ndarray) and physics.dtype == object:
            physics = _physics_from_infos(physics)
        if physics is not None:
            physics = np.asarray(physics, dtype=np.float32)
            if physics.ndim == 1:
                physics = physics[:, None]
            if physics.shape[0] == observations.shape[0]:
                return _select_physics_columns(domain, physics, physics_names_)

    # ---- fallback: derive from the raw observation layout -------------------------------
    n, dim = observations.shape
    if domain == "walker" and dim >= 17:
        # dm_control walker layout:
        #   [0:3] torso position (x, y, z); [3:6] torso orientation (rotmat z-axis);
        #   [6:9] torso linear velocity; ...
        horizontal_velocity = observations[:, 6].astype(np.float32)
        torso_upright = observations[:, 5].astype(np.float32)
        torso_height = observations[:, 2].astype(np.float32)
        return np.stack([horizontal_velocity, torso_upright, torso_height], axis=-1)
    if domain == "cheetah" and dim >= 9:
        # dm_control cheetah layout: [0:3] root position, [3:9] root velocity (xyz, ...)
        speed = np.linalg.norm(observations[:, 3:6], axis=-1).astype(np.float32)
        return speed[:, None]
    # Unknown layout -> zeros (keeps the pipeline running).
    return np.zeros((n, physics_dim(domain)), dtype=np.float32)


def append_physics(
    observations: np.ndarray,
    physics: np.ndarray,
    scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concatenate physics onto raw observations (optionally pre-scaled)."""

    observations = np.asarray(observations, dtype=np.float32)
    physics = np.asarray(physics, dtype=np.float32)
    if physics.ndim == 1:
        physics = physics[:, None]
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float32).reshape(1, -1)
        physics = physics * scale
    return np.concatenate([observations, physics], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


def observation_stats(observations: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dimension ``(mean, std)`` with a floor on the std."""

    observations = np.asarray(observations, dtype=np.float32)
    mean = observations.mean(axis=0).astype(np.float32)
    std = observations.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std


def normalize_observations(
    observations: np.ndarray,
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Std-normalise observations (paper: "std-normalize dims")."""

    observations = np.asarray(observations, dtype=np.float32)
    if stats is None:
        stats = observation_stats(observations, eps=eps)
    mean, std = stats
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    normalized = ((observations - mean) / np.maximum(std, eps)).astype(np.float32)
    return normalized, (mean, std)


# --------------------------------------------------------------------------------------
# Main dataset loader
# --------------------------------------------------------------------------------------


def _standardise_dataset(raw: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Normalise an ExORL archive into FRE's canonical transition dictionary."""

    out: Dict[str, Any] = {}

    obs = raw.get("observations", raw.get("obs", raw.get("o")))
    if obs is None:
        raise KeyError("ExORL archive is missing 'observations'")
    obs = np.asarray(obs, dtype=np.float32)
    n = obs.shape[0]

    actions = raw.get("actions", raw.get("action", raw.get("a")))
    actions = np.zeros((n, 0), dtype=np.float32) if actions is None else np.asarray(actions, dtype=np.float32)

    rewards = raw.get("rewards", raw.get("reward", raw.get("r")))
    rewards = np.zeros(n, dtype=np.float32) if rewards is None else np.asarray(rewards, dtype=np.float32).reshape(-1)

    terminals = raw.get("terminals", raw.get("terminal", raw.get("dones", raw.get("done"))))
    if terminals is None and "masks" in raw:
        terminals = 1.0 - np.asarray(raw["masks"], dtype=np.float32).reshape(-1)
    if terminals is None and "timeouts" in raw:
        # ExORL sets ``timeouts`` at truncation; those are NOT true terminals for FRE.
        terminals = np.zeros(n, dtype=np.float32)
    if terminals is None:
        terminals = np.zeros(n, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)

    timeouts = raw.get("timeouts")
    if timeouts is not None:
        timeouts = np.asarray(timeouts, dtype=np.float32).reshape(-1)

    next_obs = raw.get("next_observations", raw.get("next_obs"))
    if next_obs is None:
        next_obs = np.empty_like(obs)
        next_obs[:-1] = obs[1:]
        next_obs[-1] = obs[-1]
    next_obs = np.asarray(next_obs, dtype=np.float32)

    physics = raw.get("physics")
    phys_names = raw.get("physics_names")
    if physics is None and "infos" in raw:
        physics = _physics_from_infos(raw["infos"])
    physics = compute_physics(domain, obs, physics, phys_names)

    out.update(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminals=terminals,
        physics=physics,
        domain=domain,
    )
    if timeouts is not None:
        out["timeouts"] = timeouts
    return out


def load_exorl_dataset(
    domain: str = "walker",
    root: Optional[str] = None,
    *,
    dataset_kind: str = DEFAULT_DATASET_KIND,
    append_physics_to_obs: bool = True,
    normalise: bool = True,
    max_transitions: Optional[int] = None,
    verbose: bool = False,
    paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load an ExORL RND dataset for ``domain`` (``"walker"`` or ``"cheetah"``).

    Returns a canonical dictionary with:

    ``observations``
        Raw dm_control observations (``obs_dim`` dims) — used by the decoder/policy.
    ``encoder_observations``
        ``observations`` with physics appended and std-normalised — used by the *encoder*
        (paper: "append physics ... to ENCODER ONLY; std-normalize dims").
    ``physics``
        The ``(N, P)`` physics matrix (walker: h-velocity, torso upright, torso height;
        cheetah: speed) — used by the goal / velocity evaluation rewards.
    ``actions``/``rewards``/``terminals``/``timeouts``/``next_observations``
        Transition arrays; ``rewards`` is kept for completeness but FRE re-samples ``eta``.
    ``obs_mean``/``obs_std``/``physics_mean``/``physics_std``
        Normalisation statistics (physics statistics are returned so rewards can be
        computed in the normalised space).
    """

    if domain not in EXORL_DOMAINS:
        raise ValueError(f"unknown ExORL domain {domain!r}; expected one of {EXORL_DOMAINS}")

    if paths is None:
        paths = find_exorl_files(root=root, domain=domain, dataset_kind=dataset_kind)
    if not paths:
        raise FileNotFoundError(
            f"No ExORL dataset found for domain {domain!r}. Place the ExORL RND archive "
            f"(e.g. '{dataset_kind}/{domain}/{domain}.npz') under ./data/exorl, or set the "
            "FRE_EXORL_DIR environment variable. See the module docstring for the expected "
            "npz keys (observations, actions, rewards, terminals, timeouts, infos)."
        )

    if verbose:
        print(f"[exorl] loading {len(paths)} archive(s) for {domain}: {paths[0]} ...")

    archives = [_load_archive(p) for p in sorted(paths)]
    dataset = _standardise_dataset(_merge_archives(archives), domain)

    if max_transitions is not None and max_transitions > 0:
        for key, value in list(dataset.items()):
            if isinstance(value, np.ndarray) and value.shape[:1] == (dataset["observations"].shape[0],):
                dataset[key] = value[:max_transitions]
        dataset["observations"] = dataset["observations"][:max_transitions]

    raw_obs = dataset["observations"]
    physics = dataset["physics"]

    # ---- encoder observations: raw + physics, then std-normalised ----------------------
    if append_physics_to_obs:
        enc_obs = append_physics(raw_obs, physics)
    else:
        enc_obs = raw_obs.copy()

    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if normalise:
        enc_obs, stats = normalize_observations(enc_obs)

    dataset["encoder_observations"] = enc_obs.astype(np.float32)
    dataset["observation_stats"] = stats
    dataset["obs_mean"] = None if stats is None else stats[0].astype(np.float32)
    dataset["obs_std"] = None if stats is None else stats[1].astype(np.float32)

    phys_mean, phys_std = observation_stats(physics)
    dataset["physics_mean"] = phys_mean
    dataset["physics_std"] = phys_std
    dataset["raw_observations"] = raw_obs  # alias, explicit
    dataset["obs_dim"] = int(raw_obs.shape[1])
    dataset["encoder_obs_dim"] = int(dataset["encoder_observations"].shape[1])
    dataset["physics_dim"] = int(physics.shape[1])
    dataset["action_dim"] = int(dataset["actions"].shape[1])
    dataset["num_transitions"] = int(raw_obs.shape[0])
    dataset["dataset_paths"] = list(paths)

    if verbose:
        print(
            f"[exorl] {domain}: N={dataset['num_transitions']}, obs_dim={dataset['obs_dim']}, "
            f"encoder_obs_dim={dataset['encoder_obs_dim']}, physics_dim={dataset['physics_dim']}"
        )
    return dataset


def load_exorl_walker(root: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Convenience wrapper: :func:`load_exorl_dataset` for ``walker``."""

    return load_exorl_dataset("walker", root=root, **kwargs)


def load_exorl_cheetah(root: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Convenience wrapper: :func:`load_exorl_dataset` for ``cheetah``."""

    return load_exorl_dataset("cheetah", root=root, **kwargs)


def load_exorl(domain: str = "walker", **kwargs: Any) -> Dict[str, Any]:
    """Generic ExORL loader entry point."""

    return load_exorl_dataset(domain, **kwargs)


def load_exorl_multitask(
    domains: Sequence[str] = EXORL_DOMAINS,
    root: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Load one dataset per domain, returning ``{domain: dataset}``."""

    return {d: load_exorl_dataset(d, root=root, **kwargs) for d in domains}


# --------------------------------------------------------------------------------------
# Evaluation tasks (goals + velocity thresholds)
# --------------------------------------------------------------------------------------


def normalize_physics(dataset: Dict[str, Any], physics: np.ndarray) -> np.ndarray:
    """Std-normalise a physics matrix with the dataset's stored statistics."""

    mean = np.asarray(dataset.get("physics_mean", 0.0), dtype=np.float32).reshape(-1)
    std = np.asarray(dataset.get("physics_std", 1.0), dtype=np.float32).reshape(-1)
    return ((np.asarray(physics, dtype=np.float32) - mean) / np.maximum(std, 1e-3)).astype(np.float32)


def select_goal_states(
    dataset: Dict[str, Any],
    num_goals: int = NUM_GOAL_STATES,
    seed: int = 0,
    mode: str = "quantile",
) -> np.ndarray:
    """Deterministically choose ``num_goals`` goal states (physics space).

    The paper fixes five goal states per ExORL domain.  Without the exact published
    values we select states spread across the dataset (uniform quantiles of the
    transition index), which is deterministic, reproducible, and covers the state
    distribution — exactly the property needed for a fair zero-shot benchmark.
    ``mode="random"`` draws a seeded sample instead.
    """

    physics = np.asarray(dataset["physics"], dtype=np.float32)
    n = physics.shape[0]
    if n == 0:
        return physics.reshape(0, physics_dim(dataset.get("domain", "walker")))

    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(num_goals, n), replace=False)
    else:
        quantiles = (np.arange(num_goals) + 0.5) / max(num_goals, 1)
        idx = np.clip((quantiles * n).astype(np.int64), 0, n - 1)
    return physics[np.sort(idx)].astype(np.float32)


def goal_reward(
    physics: np.ndarray,
    goal: np.ndarray,
    threshold: float = GOAL_DISTANCE_THRESHOLD,
) -> np.ndarray:
    """Sparse goal-reaching reward: ``0`` within ``threshold`` (Euclidean), else ``-1``."""

    physics = np.atleast_2d(np.asarray(physics, dtype=np.float32))
    goal = np.asarray(goal, dtype=np.float32).reshape(1, -1)
    dist = np.linalg.norm(physics - goal, axis=-1)
    return np.where(dist < threshold, 0.0, -1.0).astype(np.float32)


def goal_done(physics: np.ndarray, goal: np.ndarray, threshold: float = GOAL_DISTANCE_THRESHOLD) -> np.ndarray:
    """Boolean done-mask for goal-reaching (True once the goal is reached)."""

    physics = np.atleast_2d(np.asarray(physics, dtype=np.float32))
    goal = np.asarray(goal, dtype=np.float32).reshape(1, -1)
    return (np.linalg.norm(physics - goal, axis=-1) < threshold).astype(np.float32)


def velocity_reward(domain: str, physics: np.ndarray, threshold: float) -> np.ndarray:
    """Sparse velocity reward: ``0`` while velocity ``>= threshold``, else ``-1``.

    ``threshold > 0`` uses the paper's forward-velocity tasks; negative thresholds give
    the reversed direction (used by the ``cheetah -1`` style tasks).
    """

    physics = np.atleast_2d(np.asarray(physics, dtype=np.float32))
    if domain == "cheetah":
        velocity = physics[:, 0]
    elif domain == "walker":
        velocity = physics[:, 0]
    else:
        raise ValueError(f"unknown ExORL domain {domain!r}")
    if threshold >= 0:
        return np.where(velocity >= threshold, 0.0, -1.0).astype(np.float32)
    return np.where(velocity <= threshold, 0.0, -1.0).astype(np.float32)


def velocity_done(domain: str, physics: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean done-mask for velocity tasks."""

    rewards = velocity_reward(domain, physics, threshold)
    return (rewards == 0.0).astype(np.float32)


def make_goal_reward_fn(
    dataset: Dict[str, Any],
    goal: np.ndarray,
    threshold: float = GOAL_DISTANCE_THRESHOLD,
    normalised: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a reward function mapping *raw* observations to goal-reaching rewards.

    When ``normalised`` is True the observations are the encoder observations (i.e. raw
    physics is embedded in the first ``obs_dim``/``encoder_obs_dim - physics_dim`` dims);
    otherwise a fully physics-augmented observation is expected.
    """

    goal = np.asarray(goal, dtype=np.float32)
    if normalised:
        goal = (goal - np.asarray(dataset["physics_mean"], dtype=np.float32)) / np.maximum(
            np.asarray(dataset["physics_std"], dtype=np.float32), 1e-3
        )
    phys_dim = int(dataset["physics_dim"])

    def reward_fn(observations: np.ndarray) -> np.ndarray:
        observations = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        physics = observations[:, -phys_dim:]
        return goal_reward(physics, goal, threshold=threshold)

    return reward_fn


def make_velocity_reward_fn(
    domain: str,
    dataset: Dict[str, Any],
    threshold: float,
    normalised: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a reward function mapping observations to velocity-task rewards."""

    phys_dim = int(dataset["physics_dim"])
    mean = np.asarray(dataset.get("physics_mean", 0.0), dtype=np.float32)
    std = np.asarray(dataset.get("physics_std", 1.0), dtype=np.float32)

    def reward_fn(observations: np.ndarray) -> np.ndarray:
        observations = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        physics = observations[:, -phys_dim:]
        if normalised:
            physics = physics * np.maximum(std, 1e-3) + mean
        return velocity_reward(domain, physics, threshold)

    return reward_fn


def evaluation_tasks(
    dataset: Dict[str, Any],
    domain: Optional[str] = None,
    goals: Optional[np.ndarray] = None,
    goal_threshold: float = GOAL_DISTANCE_THRESHOLD,
    include_velocity: bool = True,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Assemble the ExORL zero-shot task suite for one domain (Table 1 columns).

    Returns ``{task_name: {"kind", "reward", "done", "goal"/"threshold", "eval_obs"}}``:

    * ``{domain}-goal-{i}`` — the five fixed goal states (Euclidean dist < 0.1).
    * ``{domain}-velocity-{t}`` — the domain's velocity threshold grid
      (cheetah: 10, 1; walker: 0.1, 1, 4, 8).
    """

    domain = domain or dataset.get("domain", "walker")
    physics = np.asarray(dataset["physics"], dtype=np.float32)
    norm_phys = normalize_physics(dataset, physics)

    if goals is None:
        goals = select_goal_states(dataset, NUM_GOAL_STATES, seed=seed)
    goals = np.asarray(goals, dtype=np.float32)

    tasks: Dict[str, Dict[str, Any]] = {}
    for i, goal in enumerate(goals):
        norm_goal = normalize_physics(dataset, goal)
        tasks[f"{domain}-goal-{i}"] = {
            "kind": "goal",
            "domain": domain,
            "goal": goal,
            "normalized_goal": norm_goal,
            "threshold": goal_threshold,
            "reward": goal_reward(norm_phys, norm_goal, goal_threshold),
            "done": goal_done(norm_phys, norm_goal, goal_threshold),
            "eval_obs": np.concatenate(
                [dataset["raw_observations"], normalize_physics(dataset, physics)], axis=-1
            ).astype(np.float32),
        }

    if include_velocity:
        for threshold in VELOCITY_THRESHOLDS.get(domain, ()):
            tasks[f"{domain}-velocity-{threshold:g}"] = {
                "kind": "velocity",
                "domain": domain,
                "threshold": float(threshold),
                "reward": velocity_reward(domain, norm_phys, float(threshold)),
                "done": velocity_done(domain, norm_phys, float(threshold)),
                "eval_obs": np.concatenate(
                    [dataset["raw_observations"], norm_phys], axis=-1
                ).astype(np.float32),
            }
    return tasks


# --------------------------------------------------------------------------------------
# Replay-buffer bridge & serialisation
# --------------------------------------------------------------------------------------


def to_replay_buffer(
    dataset: Dict[str, Any],
    *,
    device: str = "cpu",
    seed: Optional[int] = None,
    normalize_observations_: Optional[bool] = None,
    trajectory_buffer: bool = True,
) -> Any:
    """Convert an ExORL dataset into a FRE replay buffer.

    The buffer stores the **encoder observations** (physics-augmented, std-normalised)
    because FRE's policy is conditioned on latents encoded from that space; the raw
    observations remain available on the dataset dict for reward computation.
    """

    from fre.rl.replay_buffer import (
        OfflineReplayBuffer,
        TrajectoryReplayBuffer,
        d4rl_dict_to_arrays,
    )

    payload = {
        "observations": dataset["encoder_observations"],
        "actions": dataset["actions"],
        "rewards": dataset["rewards"],
        "next_observations": np.concatenate(
            [dataset["encoder_observations"][1:], dataset["encoder_observations"][-1:]], axis=0
        ),
        "terminals": dataset["terminals"],
    }
    arrays = d4rl_dict_to_arrays(payload, require_rewards=False, infer_terminals=True)

    if trajectory_buffer:
        buffer = TrajectoryReplayBuffer(device=device, seed=seed)
    else:
        buffer = OfflineReplayBuffer(device=device, seed=seed)
    buffer.set_arrays(**arrays)

    mean = dataset.get("obs_mean")
    std = dataset.get("obs_std")
    if normalize_observations_ is True and mean is not None:
        buffer.set_observation_normalization(mean, std, apply=True)
    return buffer


def save_exorl_dataset(dataset: Dict[str, Any], path: str) -> str:
    """Serialise a loaded ExORL dataset (numpy-friendly keys only) to ``.npz``/pickle."""

    serialisable = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            serialisable[key] = value
        elif isinstance(value, (int, float, str, bool)) or value is None:
            serialisable[key] = np.array(value, dtype=object)
        elif isinstance(value, tuple):
            serialisable[key] = np.array(value, dtype=object)
    if path.endswith(".npz"):
        np.savez_compressed(path, **serialisable)
    else:
        with open(path, "wb") as fh:
            pickle.dump(serialisable, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def dataset_info(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Summary statistics mirroring :func:`fre.data.d4rl_loader.dataset_info`."""

    obs = dataset["observations"]
    actions = dataset["actions"]
    terminals = dataset["terminals"]
    info: Dict[str, Any] = {
        "domain": dataset.get("domain"),
        "num_transitions": int(obs.shape[0]),
        "obs_dim": int(obs.shape[1]),
        "encoder_obs_dim": int(dataset["encoder_observations"].shape[1]),
        "physics_dim": int(dataset["physics_dim"]),
        "action_dim": int(actions.shape[1]),
        "num_episodes": int(np.sum(terminals > 0.5)) + 1,
        "reward_mean": float(np.mean(dataset["rewards"])) if dataset["rewards"].size else 0.0,
        "paths": dataset.get("dataset_paths", []),
    }
    ends = np.flatnonzero(terminals > 0.5)
    if ends.size:
        lengths = np.diff(np.concatenate([[-1], ends]))
        info["mean_episode_length"] = float(np.mean(lengths))
        info["max_episode_length"] = int(np.max(lengths))
    else:
        info["mean_episode_length"] = float(obs.shape[0])
        info["max_episode_length"] = int(obs.shape[0])
    return info


__all__ = [
    "MAX_EPISODE_STEPS",
    "DEFAULT_DATASET_KIND",
    "GOAL_DISTANCE_THRESHOLD",
    "NUM_GOAL_STATES",
    "WALKER_PHYSICS",
    "CHEETAH_PHYSICS",
    "VELOCITY_THRESHOLDS",
    "RAW_OBS_DIM",
    "EXORL_DOMAINS",
    "find_exorl_files",
    "find_exorl_root",
    "exorl_available",
    "physics_names",
    "physics_dim",
    "compute_physics",
    "append_physics",
    "observation_stats",
    "normalize_observations",
    "load_exorl_dataset",
    "load_exorl",
    "load_exorl_walker",
    "load_exorl_cheetah",
    "load_exorl_multitask",
    "normalize_physics",
    "select_goal_states",
    "goal_reward",
    "goal_done",
    "velocity_reward",
    "velocity_done",
    "make_goal_reward_fn",
    "make_velocity_reward_fn",
    "evaluation_tasks",
    "to_replay_buffer",
    "save_exorl_dataset",
    "dataset_info",
]
