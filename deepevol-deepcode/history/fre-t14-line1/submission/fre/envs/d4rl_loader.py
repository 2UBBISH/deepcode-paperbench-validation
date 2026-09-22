"""Offline dataset loading for the FRE reproduction.

This module is the single entry point used by :class:`fre.rl.replay_buffer.ReplayBuffer`
to materialise offline data for the three evaluation domains of the paper:

* **AntMaze** -- the most challenging D4RL offline AntMaze dataset,
  ``antmaze-large-diverse-v2`` (Appendix C.1: "We utilize the
  antmaze-large-diverse-v2 dataset from D4RL").  Online evaluation runs for
  2000 timesteps with the ant placed at the centre of the maze.
* **ExORL** -- the ExORL datasets for the ``walker`` and ``cheetah`` domains
  (Section 5: "We consider the walker and cheetah domains, in accordance with
  Touati et al. (2022)").  The RND dataset is used per the reproduction plan.
* **Kitchen** -- the D4RL Kitchen environment (Appendix C.3: "we utilize the
  seven standard subtasks within the D4RL Kitchen environment").

Appendix C.2 additionally specifies a *physics augmentation* used only for the
FRE encoder on ExORL: the Cheetah/Walker reward functions are functions of the
underlying physics rather than of the observation, so the values of

    Walker : self.physics.horizontal_velocity(), torso_upright, torso_height
    Cheetah: self.physics.speed()

are appended to the offline dataset during encoder training.  "Augmented
information is not utilized when calculating goal distance."

Loading order for every domain (each step optional, first success wins):

1. an explicit local file (``.npz`` / ``.hdf5`` / ``.h5``) or directory,
2. the official ``d4rl`` / ``gym`` API (requires ``d4rl`` + ``mujoco``),
3. a deterministic *synthetic* fallback so that the FRE pipeline can be smoke
   tested on machines without MuJoCo installed.  The fallback is clearly
   flagged in the returned buffer name and in ``dataset.extra`` so that it can
   never be mistaken for a real result.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DATASET_DIR",
    "D4RL_ENV_IDS",
    "EXORL_DATASETS",
    "PHYSICS_FIELDS",
    "load_offline_dataset",
    "load_antmaze",
    "load_exorl",
    "load_kitchen",
    "load_local_dataset_file",
    "resolve_dataset_path",
    "compute_physics",
    "walker_physics",
    "cheetah_physics",
    "append_physics",
    "physics_dim",
    "state_dim_for",
    "action_dim_for",
    "dataset_std",
    "normalized_goal_distance",
    "make_synthetic_dataset",
    "available_local_datasets",
]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Root directory searched for locally downloaded datasets.  Override with the
#: ``FRE_DATASET_DIR`` environment variable.
DATASET_DIR = os.environ.get("FRE_DATASET_DIR", "datasets")

#: D4RL environment ids per domain.  ``v2`` datasets are required for AntMaze.
D4RL_ENV_IDS: Dict[str, str] = {
    "antmaze": "antmaze-large-diverse-v2",
    "kitchen": "kitchen-complete-v0",
}

#: ExORL dataset names, one per ExORL domain.  The plan pins the RND dataset.
EXORL_DATASETS: Dict[str, str] = {
    "exorl_walker": "walker",
    "exorl_cheetah": "cheetah",
}

#: Physics fields appended to the encoder input per ExORL domain (Appendix C.2).
PHYSICS_FIELDS: Dict[str, Tuple[str, ...]] = {
    "exorl_walker": ("horizontal_velocity", "torso_upright", "torso_height"),
    "exorl_cheetah": ("speed",),
}

#: Observation / action dimensionalities of the underlying environments.  Used
#: only for synthetic fallbacks and shape validation -- real datasets provide
#: their own shapes.
_OBS_DIM: Dict[str, int] = {
    "antmaze": 29,
    "exorl_walker": 24,
    "exorl_cheetah": 17,
    "kitchen": 59,
}
_ACTION_DIM: Dict[str, int] = {
    "antmaze": 8,
    "exorl_walker": 6,
    "exorl_cheetah": 6,
    "kitchen": 9,
}

#: Aliases accepted for domain names.
_DOMAIN_ALIASES: Dict[str, str] = {
    "ant": "antmaze",
    "antmaze": "antmaze",
    "antmaze-large-diverse-v2": "antmaze",
    "walker": "exorl_walker",
    "exorl-walker": "exorl_walker",
    "exorl_walker": "exorl_walker",
    "walker2d": "exorl_walker",
    "cheetah": "exorl_cheetah",
    "exorl-cheetah": "exorl_cheetah",
    "exorl_cheetah": "exorl_cheetah",
    "halfcheetah": "exorl_cheetah",
    "kitchen": "kitchen",
}


def _canonical_domain(domain: Optional[str]) -> Optional[str]:
    """Map a (possibly aliased) domain string onto a canonical domain name."""
    if domain is None:
        return None
    key = str(domain).strip().lower().replace(" ", "_")
    return _DOMAIN_ALIASES.get(key, key)


# --------------------------------------------------------------------------------------
# Local file loading
# --------------------------------------------------------------------------------------


def available_local_datasets(dataset_dir: Optional[str] = None) -> List[str]:
    """Return the dataset files found under ``dataset_dir`` (recursively)."""
    root = dataset_dir or DATASET_DIR
    found: List[str] = []
    if not os.path.isdir(root):
        return found
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".npz", ".hdf5", ".h5", ".npy")):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def resolve_dataset_path(
    domain: str,
    variant: str = "rnd",
    dataset_dir: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> Optional[str]:
    """Locate an offline dataset file for ``domain`` on disk.

    Search order:

    1. ``dataset_path`` verbatim (file or directory),
    2. ``{dataset_dir}/{domain}_{variant}.{ext}`` and friends,
    3. ``{dataset_dir}/{domain}/{variant}.{ext}`` and friends (ExORL layout),
    4. any file under ``dataset_dir`` whose name contains both the environment
       name and the variant (case-insensitive).

    Returns ``None`` when nothing is found.
    """
    exts = (".npz", ".hdf5", ".h5", ".npy")

    def _try(path: str) -> Optional[str]:
        if path is None:
            return None
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            hits = available_local_datasets(path)
            if len(hits) == 1:
                return hits[0]
            # prefer a file matching the variant name
            for hit in hits:
                if variant.lower() in os.path.basename(hit).lower():
                    return hit
            return hits[0] if hits else None
        return None

    if dataset_path is not None:
        hit = _try(dataset_path)
        if hit is not None:
            return hit

    root = dataset_dir or DATASET_DIR
    env_name = EXORL_DATASETS.get(domain, domain)
    stems = [
        f"{domain}_{variant}",
        f"{domain}-{variant}",
        f"{env_name}_{variant}",
        f"{env_name}-{variant}",
        f"{variant}_{env_name}",
        f"{variant}-{env_name}",
        f"{domain}_{variant}_dataset",
        f"{env_name}_{variant}_dataset",
    ]
    subdirs = [root, os.path.join(root, domain), os.path.join(root, env_name)]
    for sub in subdirs:
        for stem in stems:
            for ext in exts:
                hit = _try(os.path.join(sub, stem + ext))
                if hit is not None:
                    return hit

    # last resort: fuzzy match on file names
    for hit in available_local_datasets(root):
        base = os.path.basename(hit).lower()
        if env_name.lower() in base and variant.lower() in base:
            return hit
    return None


def load_local_dataset_file(path: str) -> Dict[str, np.ndarray]:
    """Load a ``.npz`` / ``.hdf5`` / ``.npy`` dataset into a dict of arrays.

    Keys are normalised to the canonical names used by
    :class:`fre.rl.replay_buffer.ReplayBuffer`.  ExORL hdf5 files store their
    transitions in a flat layout or under a group (``exp``/``data``); both are
    handled.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset file not found: {path}")
    lower = path.lower()
    if lower.endswith(".npz"):
        with np.load(path, allow_pickle=True) as archive:
            raw = {k: np.asarray(archive[k]) for k in archive.files}
    elif lower.endswith(".npy"):
        raw = {"observations": np.load(path, allow_pickle=True)}
    elif lower.endswith((".hdf5", ".h5")):
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "h5py is required to read ExORL hdf5 datasets; install it with "
                "`pip install h5py` or download the .npz variant of the dataset."
            ) from exc
        raw = {}
        with h5py.File(path, "r") as handle:

            def _collect(group, prefix=""):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        _collect(item, prefix=f"{prefix}{key}/")
                    else:
                        raw[f"{prefix}{key}"] = np.asarray(item)

            _collect(handle)
    else:
        raise ValueError(f"unsupported dataset format: {path}")

    return _canonicalise_arrays(raw)


_ARRAY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "observations": ("observations", "obs", "observation", "states", "state"),
    "actions": ("actions", "action", "acs"),
    "rewards": ("rewards", "reward", "rew"),
    "next_observations": ("next_observations", "next_obs", "nextob", "next_state", "next_states"),
    "terminals": ("terminals", "dones", "done", "terminal", "is_terminal"),
    "timeouts": ("timeouts", "timeout", "is_timeout"),
    "physics": ("physics", "physics_states", "augmented", "aux", "qpos_qvel"),
    "qpos": ("qpos", "positions"),
    "qvel": ("qvel", "velocities"),
    "trajectory_ids": ("trajectory_ids", "episode_ids", "traj_ids", "episode"),
}


def _canonicalise_arrays(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Rename raw dataset keys to canonical names (case/prefix insensitive)."""
    out: Dict[str, np.ndarray] = {}
    lowered = {str(k).lower().strip(): v for k, v in raw.items()}
    for canonical, aliases in _ARRAY_ALIASES.items():
        for alias in aliases:
            match = None
            # exact-ish match first
            for key, value in lowered.items():
                tail = key.split("/")[-1]
                if tail == alias:
                    match = value
                    break
            if match is None:
                for key, value in lowered.items():
                    tail = key.split("/")[-1]
                    if alias in tail:
                        match = value
                        break
            if match is not None:
                out[canonical] = np.asarray(match)
                break
    # keep unknown arrays for inspection
    for key, value in raw.items():
        if key not in out:
            out.setdefault(key, value)
    return out


# --------------------------------------------------------------------------------------
# Physics augmentation (Appendix C.2)
# --------------------------------------------------------------------------------------


def walker_physics(
    qpos: np.ndarray,
    qvel: np.ndarray,
    horizontal_velocity: str = "x",
) -> np.ndarray:
    """Compute the Walker physics features used for encoder augmentation.

    Implements the quantities named in Appendix C.2::

        self.physics.horizontal_velocity()
        self.physics.torso_upright()
        self.physics.torso_height()

    The Walker is planar: ``qpos = [rootx, rootz, torso_angle, joint_1..6]`` and
    ``qvel = [rootx_dot, rootz_dot, torso_angle_dot, joint_1..6_dot]``.  Hence

    * ``horizontal_velocity`` = forward (x) root velocity, ``qvel[0]``;
    * ``torso_height``        = root height, ``qpos[1]``;
    * ``torso_upright``       = ``xmat[torso, 'zz']``, which for a rotation of
      the torso about the y-axis equals ``cos(torso_angle)``; the torso angle is
      read from ``qpos[2]`` (dm_control's ``walker`` hinge ordering).

    dm_control exposes ``horizontal_velocity()`` as a 2-vector; FRE only needs
    it to express the Walker *run/walk* reward, which is the forward velocity,
    so the x-component is used by default.  Set ``horizontal_velocity="xy"`` to
    keep both components.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qvel.ndim == 1:
        qvel = qvel[None, :]

    if horizontal_velocity == "xy" and qvel.shape[1] >= 2:
        hv = qvel[:, :2]
    else:
        hv = qvel[:, :1]

    torso_angle = qpos[:, 2] if qpos.shape[1] > 2 else np.zeros(qpos.shape[0])
    upright = np.cos(torso_angle)[:, None]
    height = qpos[:, 1:2]
    return np.concatenate([hv, upright, height], axis=1).astype(np.float32)


def cheetah_physics(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Compute the Cheetah physics feature used for encoder augmentation.

    Appendix C.2 appends ``self.physics.speed()``, which in dm_control's
    HalfCheetah is the horizontal velocity of the root joint
    (``qvel['rootx']``).  In the standard cheetah qpos/qvel layout the root x
    position/velocity occupy index 0.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qvel.ndim == 1:
        qvel = qvel[None, :]
    return qvel[:, :1].astype(np.float32)


def physics_dim(domain: str) -> int:
    """Number of physics features appended for ``domain`` (0 when none)."""
    domain = _canonical_domain(domain) or domain
    if domain == "exorl_walker":
        return 3
    if domain == "exorl_cheetah":
        return 1
    return 0


def compute_physics(
    domain: str,
    observations: np.ndarray,
    physics: Optional[np.ndarray] = None,
    qpos: Optional[np.ndarray] = None,
    qvel: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build the physics-augmentation array for ``domain``.

    ``physics`` is returned verbatim when supplied (many ExORL mirrors store
    the augmented quantities directly).  Otherwise the features are computed
    from ``qpos``/``qvel`` when available.  For AntMaze and Kitchen (whose
    rewards are pure functions of the observation) ``None`` is returned, as no
    augmentation is used.
    """
    domain = _canonical_domain(domain) or domain
    if domain not in PHYSICS_FIELDS:
        return None

    if physics is not None:
        physics = np.asarray(physics, dtype=np.float32)
        if physics.ndim == 1:
            physics = physics[:, None]
        return physics

    if qpos is None or qvel is None:
        # No raw simulator state available: fall back to a best-effort recovery
        # from the observation vector.  The DM Control observation ordering puts
        # the root position first and the root velocity after the position
        # block, so: Walker obs = [rootx, rootz, torso_angle, joints...,
        # rootx_dot, rootz_dot, torso_angle_dot, joint_v...]; Cheetah follows the
        # same convention with a shorter block.
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs[None, :]
        if domain == "exorl_cheetah":
            half = obs.shape[1] // 2
            qpos = obs[:, :half]
            qvel = obs[:, half : 2 * half]
            return cheetah_physics(qpos, qvel)
        # walker: positions = 9, velocities = 9 in the DM Control layout
        half = obs.shape[1] // 2
        qpos = obs[:, :half]
        qvel = obs[:, half : 2 * half]
        return walker_physics(qpos, qvel)

    if domain == "exorl_cheetah":
        return cheetah_physics(qpos, qvel)
    return walker_physics(qpos, qvel)


def append_physics(observations: np.ndarray, physics: np.ndarray) -> np.ndarray:
    """Return ``observations`` with the physics features concatenated (encoder input)."""
    observations = np.asarray(observations, dtype=np.float32)
    physics = np.asarray(physics, dtype=np.float32)
    if physics.ndim == 1:
        physics = physics[:, None]
    return np.concatenate([observations, physics], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------------------
# D4RL / gym loading
# --------------------------------------------------------------------------------------


def _load_d4rl_dataset(env_id: str) -> Dict[str, np.ndarray]:
    """Load a D4RL dataset through the ``d4rl``/``gym`` API.

    Tries :func:`d4rl.qlearning_dataset` first (the canonical transition layout)
    and falls back to ``env.get_dataset()``.
    """
    try:
        import gym  # type: ignore
        import d4rl  # noqa: F401  (import registers the environments)
    except ImportError as exc:  # pragma: no cover - heavy optional dependency
        raise ImportError(
            "d4rl/gym are required to load D4RL datasets.  Install with "
            "`pip install gym==0.23.1 d4rl` (pin d4rl to a commit before June 2024) "
            "or point `dataset_path` at a local .npz/.hdf5 copy of the dataset."
        ) from exc

    env = gym.make(env_id)
    data: Dict[str, np.ndarray]
    try:
        import d4rl as _d4rl  # type: ignore

        data = dict(_d4rl.qlearning_dataset(env))
    except Exception:  # pragma: no cover - depends on d4rl version
        data = dict(env.get_dataset())

    # 'infos/*' style keys are dropped; failures=False rows are kept verbatim.
    return {k: np.asarray(v) for k, v in data.items() if not isinstance(v, dict)}


def load_antmaze(
    env_id: str = "antmaze-large-diverse-v2",
    dataset_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    variant: str = "large-diverse-v2",
    **kwargs: Any,
):
    """Load the ``antmaze-large-diverse-v2`` offline dataset (Appendix C.1)."""
    from fre.rl.replay_buffer import ReplayBuffer

    path = dataset_path or resolve_dataset_path("antmaze", variant, dataset_dir)
    if path is not None:
        data = load_local_dataset_file(path)
        name = os.path.basename(path)
    else:
        data = _load_d4rl_dataset(env_id)
        name = env_id
    return _to_buffer(data, name=name, domain="antmaze", **kwargs)


def load_exorl(
    domain: str = "exorl_walker",
    variant: str = "rnd",
    dataset_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    use_physics: bool = True,
    **kwargs: Any,
):
    """Load an ExORL dataset (walker/cheetah, RND) with physics augmentation.

    The physics features described in Appendix C.2 are computed (or read from
    the file when present) and stored on the returned buffer, so the encoder can
    consume ``observations + physics`` while the policy/value networks use the
    plain observation space.
    """
    domain = _canonical_domain(domain) or domain
    if domain not in EXORL_DATASETS:
        raise ValueError(f"unknown ExORL domain: {domain!r}")

    path = dataset_path or resolve_dataset_path(domain, variant, dataset_dir)
    if path is not None:
        data = load_local_dataset_file(path)
        name = f"{os.path.basename(path)}"
    else:
        raise FileNotFoundError(
            "No ExORL dataset found.  Download the RND dataset for "
            f"{EXORL_DATASETS[domain]} (see https://github.com/denisyarats/exorl) and "
            f"place it under {dataset_dir or DATASET_DIR}/, or pass `dataset_path=...`."
        )

    physics = None
    if use_physics:
        physics = compute_physics(
            domain,
            data.get("observations", np.zeros((0, 0))),
            physics=data.get("physics"),
            qpos=data.get("qpos"),
            qvel=data.get("qvel"),
        )
    return _to_buffer(
        data, name=name, domain=domain, physics=physics, use_physics=use_physics, **kwargs
    )


def load_kitchen(
    env_id: str = "kitchen-complete-v0",
    dataset_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    variant: str = "complete",
    **kwargs: Any,
):
    """Load the D4RL Kitchen dataset (Appendix C.3)."""
    path = dataset_path or resolve_dataset_path("kitchen", variant, dataset_dir)
    if path is not None:
        data = load_local_dataset_file(path)
        name = os.path.basename(path)
    else:
        data = _load_d4rl_dataset(env_id)
        name = env_id
    return _to_buffer(data, name=name, domain="kitchen", **kwargs)


# --------------------------------------------------------------------------------------
# Top level entry point
# --------------------------------------------------------------------------------------


def _to_buffer(
    data: Dict[str, np.ndarray],
    name: str,
    domain: str,
    physics: Optional[np.ndarray] = None,
    use_physics: bool = False,
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
    normalize_actions: bool = False,
    **kwargs: Any,
):
    """Wrap a dict of arrays into a :class:`ReplayBuffer`."""
    from fre.rl.replay_buffer import ReplayBuffer

    observations = data.get("observations")
    if observations is None:
        raise KeyError(f"dataset {name!r} does not contain observations")
    observations = np.asarray(observations, dtype=np.float32)

    actions = data.get("actions")
    actions = None if actions is None else np.asarray(actions, dtype=np.float32)
    rewards = data.get("rewards")
    rewards = None if rewards is None else np.asarray(rewards, dtype=np.float32).reshape(-1)
    next_observations = data.get("next_observations")
    if next_observations is None and actions is not None:
        next_observations = np.concatenate([observations[1:], observations[-1:]], axis=0)
    next_observations = (
        None if next_observations is None else np.asarray(next_observations, dtype=np.float32)
    )
    terminals = data.get("terminals")
    terminals = None if terminals is None else np.asarray(terminals).astype(bool).reshape(-1)
    timeouts = data.get("timeouts")
    timeouts = None if timeouts is None else np.asarray(timeouts).astype(bool).reshape(-1)
    trajectory_ids = data.get("trajectory_ids")
    trajectory_ids = (
        None if trajectory_ids is None else np.asarray(trajectory_ids).reshape(-1)
    )

    if physics is not None and use_physics is False:
        physics = None

    buffer = ReplayBuffer(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
        physics=physics,
        trajectory_ids=trajectory_ids,
        reward_scale=reward_scale,
        reward_shift=reward_shift,
        name=name,
        normalize_actions=normalize_actions,
        **kwargs,
    )
    # provenance metadata (never used for results, only for bookkeeping)
    try:
        setattr(buffer, "domain", domain)
        setattr(buffer, "source", name)
        setattr(buffer, "use_physics", bool(use_physics and physics is not None))
    except Exception:  # pragma: no cover - ReplayBuffer may use __slots__
        pass
    return buffer


def load_offline_dataset(
    config: Any = None,
    domain: Optional[str] = None,
    env_id: Optional[str] = None,
    dataset_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    variant: Optional[str] = None,
    use_physics: Optional[bool] = None,
    allow_synthetic: bool = False,
    synthetic_transitions: int = 20_000,
    **kwargs: Any,
):
    """Load the offline dataset for a domain, honouring a :class:`Config`.

    ``config`` may be a ``fre.config.default.Config``-like object (attributes
    ``domain``, ``dataset_id``/``env_id``, ``dataset_path``, ``dataset_dir``,
    ``exorl_variant``, ``use_physics_augmentation``) or a plain string naming
    the domain.  Any keyword argument overrides the corresponding config value.

    With ``allow_synthetic=True`` a deterministic synthetic dataset is returned
    when no real data is available, so that the training/evaluation pipeline can
    be exercised without MuJoCo or D4RL installed.
    """
    # --- resolve the domain -----------------------------------------------------
    cfg = config
    if isinstance(config, str) and domain is None:
        domain = config
        cfg = None

    if cfg is not None and domain is None:
        domain = getattr(cfg, "domain", None)
    domain = _canonical_domain(domain) or "antmaze"

    if variant is None and cfg is not None:
        variant = getattr(cfg, "exorl_variant", None) or getattr(cfg, "dataset_variant", None)
    if variant is None:
        variant = "rnd" if domain.startswith("exorl") else _default_variant(domain)

    if dataset_path is None and cfg is not None:
        dataset_path = getattr(cfg, "dataset_path", None)
    if dataset_dir is None and cfg is not None:
        dataset_dir = getattr(cfg, "dataset_dir", None) or DATASET_DIR
    dataset_dir = dataset_dir or DATASET_DIR

    if env_id is None and cfg is not None:
        env_id = getattr(cfg, "env_id", None) or getattr(cfg, "dataset_id", None)
    if env_id is None:
        env_id = D4RL_ENV_IDS.get(domain, "")

    if use_physics is None and cfg is not None:
        use_physics = bool(getattr(cfg, "use_physics_augmentation", False))
    if use_physics is None:
        use_physics = domain.startswith("exorl")

    # forward only constructor kwargs the ReplayBuffer understands
    buffer_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "reward_scale",
            "reward_shift",
            "normalize_actions",
            "clip_actions",
            "seed",
            "max_episode_steps",
        }
    }

    try:
        if domain == "antmaze":
            return load_antmaze(
                env_id=env_id or "antmaze-large-diverse-v2",
                dataset_path=dataset_path,
                dataset_dir=dataset_dir,
                variant=variant,
                **buffer_kwargs,
            )
        if domain.startswith("exorl"):
            return load_exorl(
                domain=domain,
                variant=variant,
                dataset_path=dataset_path,
                dataset_dir=dataset_dir,
                use_physics=use_physics,
                **buffer_kwargs,
            )
        if domain == "kitchen":
            return load_kitchen(
                env_id=env_id or "kitchen-complete-v0",
                dataset_path=dataset_path,
                dataset_dir=dataset_dir,
                variant=variant,
                **buffer_kwargs,
            )
    except (FileNotFoundError, ImportError) as exc:
        if not allow_synthetic:
            raise
        import warnings

        warnings.warn(
            f"[d4rl_loader] falling back to a SYNTHETIC dataset for {domain!r}: {exc}",
            stacklevel=2,
        )
        return make_synthetic_dataset(
            domain,
            num_transitions=synthetic_transitions,
            use_physics=use_physics,
            **buffer_kwargs,
        )

    raise ValueError(f"unknown domain: {domain!r}")


def _default_variant(domain: str) -> str:
    if domain == "antmaze":
        return "large-diverse-v2"
    if domain == "kitchen":
        return "complete"
    return "rnd"


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def state_dim_for(domain: str, use_physics: bool = False) -> int:
    """Observation dimensionality (optionally including physics augmentation)."""
    domain = _canonical_domain(domain) or domain
    base = _OBS_DIM.get(domain, 0)
    if use_physics:
        base += physics_dim(domain)
    return base


def action_dim_for(domain: str) -> int:
    """Action dimensionality of ``domain``."""
    domain = _canonical_domain(domain) or domain
    return _ACTION_DIM.get(domain, 0)


def dataset_std(states: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-dimension standard deviation used to normalise ExORL goal distances.

    Appendix C.2: "Each state dimension is normalized according to the standard
    deviation along that dimension within the offline dataset.  Augmented
    information is not utilized when calculating goal distance."
    """
    states = np.asarray(states, dtype=np.float64)
    std = states.std(axis=0)
    return np.maximum(std, eps).astype(np.float32)


def normalized_goal_distance(
    states: np.ndarray,
    goals: np.ndarray,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Euclidean distance in std-normalised state space (Appendix C.2)."""
    states = np.asarray(states, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    if std is None:
        std = dataset_std(states, eps=eps)
    std = np.asarray(std, dtype=np.float64)
    scale = 1.0 / np.maximum(std, eps)
    scale = scale[: states.shape[-1]]
    if goals.ndim == 1:
        diff = (states - goals[None, :]) * scale[None, :]
        return np.linalg.norm(diff, axis=-1)
    diff = states[None, :, :] - goals[:, None, :]
    diff = diff * scale[None, None, :]
    return np.linalg.norm(diff, axis=-1)


# --------------------------------------------------------------------------------------
# Synthetic fallback (never used for reported results)
# --------------------------------------------------------------------------------------


def make_synthetic_dataset(
    domain: str,
    num_transitions: int = 20_000,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    use_physics: bool = False,
    seed: int = 0,
    episode_length: int = 200,
    **buffer_kwargs: Any,
):
    """Deterministic random-walk dataset, for smoke testing without MuJoCo.

    The returned buffer is tagged with ``synthetic=True`` so downstream code can
    warn that any metric computed on it is meaningless.
    """
    domain = _canonical_domain(domain) or domain
    rng = np.random.default_rng(seed)
    obs_dim = state_dim if state_dim is not None else _OBS_DIM.get(domain, 29)
    act_dim = action_dim if action_dim is not None else _ACTION_DIM.get(domain, 8)

    num_episodes = max(1, int(np.ceil(num_transitions / episode_length)))
    obs = np.zeros((num_episodes, episode_length, obs_dim), dtype=np.float32)
    act = np.zeros((num_episodes, episode_length, act_dim), dtype=np.float32)
    rew = np.zeros((num_episodes, episode_length), dtype=np.float32)
    term = np.zeros((num_episodes, episode_length), dtype=bool)
    state = rng.normal(size=obs_dim).astype(np.float32)
    for ep in range(num_episodes):
        for t in range(episode_length):
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
            act[ep, t] = a
            obs[ep, t] = state
            rew[ep, t] = float(rng.normal())
            state = (state + 0.1 * rng.normal(size=obs_dim)).astype(np.float32)
    term[:, -1] = True
    flat_obs = obs.reshape(-1, obs_dim)
    flat_act = act.reshape(-1, act_dim)
    flat_rew = rew.reshape(-1)
    flat_term = term.reshape(-1)

    physics = None
    if use_physics and domain in PHYSICS_FIELDS:
        physics = compute_physics(domain, flat_obs)

    buffer = _to_buffer(
        {
            "observations": flat_obs,
            "actions": flat_act,
            "rewards": flat_rew,
            "next_observations": np.concatenate([flat_obs[1:], flat_obs[-1:]], axis=0),
            "terminals": flat_term,
        },
        name=f"synthetic-{domain}",
        domain=domain,
        physics=physics,
        use_physics=use_physics,
        **buffer_kwargs,
    )
    try:
        setattr(buffer, "synthetic", True)
    except Exception:  # pragma: no cover
        pass
    return buffer
