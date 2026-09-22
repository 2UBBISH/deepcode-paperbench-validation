"""Data preprocessing utilities for FRE.

Implements the domain-specific preprocessing described in the paper:

* AntMaze (Appendix C.1): the ``antmaze-large-diverse-v2`` dataset from D4RL.
  "The FRE, GC-IQL, GC-BC, and OPAL agents all utilize a discretized
  preprocessing procedure, where the X and Y coordinates are discretized into
  32 bins."

* ExORL (Appendix C.2): "FRE assumes that reward functions must be pure
  functions of the environment state.  Because the Cheetah and Walker
  environments utilize rewards that are a function of the underlying physics, we
  append information about the physics onto the offline dataset during encoder
  training.  Specifically, we append the values of
  ``self.physics.horizontal_velocity()``, ``self.physics.torso_upright()``,
  ``self.physics.torso_height()`` to Walker, and ``self.physics.speed()`` to
  Cheetah."  Also: "Each state dimension is normalized according to the standard
  deviation along that dimension within the offline dataset.  Augmented
  information is not utilized when calculating goal distance."  And goals are
  reached "when the Euclidean distance between the current state and the goal
  state is less than 0.1".

* Kitchen (Appendix C.3): the seven standard sparse subtasks of D4RL Kitchen
  are used directly (no extra preprocessing beyond using the environment's own
  sparse rewards).

The module is intentionally dependency-light: ``dm_control`` is imported lazily
and only when physics augmentation is actually requested, so that the rest of
the code base (AntMaze / Kitchen / synthetic data) works without it.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    # constants
    "NUM_XY_BINS",
    "ANTMAZE_XY_INDICES",
    "EXORL_PHYSICS_FIELDS",
    "EXORL_AUGMENT_DIM",
    "EXORL_DM_CONTROL_DOMAINS",
    # AntMaze
    "discretize_antmaze_xy",
    "antmaze_xy_bins",
    "count_xy_bins",
    # ExORL physics augmentation
    "augment_exorl_physics",
    "exorl_physics_augmentation",
    "exorl_domain",
    "exorl_augment_dim",
    "exorl_goal_state_dims",
    # normalization
    "compute_state_normalization",
    "normalize_states",
    "denormalize_states",
    "scale_by_std",
    # goals
    "euclidean_goal_distance",
    "goal_reached_mask",
    # high-level helpers
    "preprocess_antmaze",
    "preprocess_exorl",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

NUM_XY_BINS: int = 32
"""AntMaze X/Y coordinates are discretized into this many bins (Appendix C.1)."""

ANTMAZE_XY_INDICES: Tuple[int, int] = (0, 1)
"""Indices of the X/Y coordinates in the AntMaze observation vector."""

EXORL_PHYSICS_FIELDS: Dict[str, Tuple[str, ...]] = {
    "walker": ("horizontal_velocity_x", "horizontal_velocity_y", "torso_upright", "torso_height"),
    "cheetah": ("speed",),
}
"""Physics quantities appended to the ExORL observations for encoder training (C.2)."""

EXORL_AUGMENT_DIM: Dict[str, int] = {
    "walker": 4,  # horizontal_velocity (2) + torso_upright (1) + torso_height (1)
    "cheetah": 1,  # speed (1)
}

EXORL_DM_CONTROL_DOMAINS: Dict[str, Tuple[str, str]] = {
    "walker": ("walker", "walk"),
    "cheetah": ("cheetah", "run"),
}
"""Mapping from ExORL domain to the DeepMind Control Suite (domain, task) pair."""


# --------------------------------------------------------------------------------------
# AntMaze X/Y discretization (Appendix C.1)
# --------------------------------------------------------------------------------------


def antmaze_xy_bins(
    observations: np.ndarray,
    xy_indices: Sequence[int] = ANTMAZE_XY_INDICES,
    num_bins: int = NUM_XY_BINS,
    xy_min: Optional[Sequence[float]] = None,
    xy_max: Optional[Sequence[float]] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute integer bins for the AntMaze X/Y coordinates.

    The coordinates are rescaled to ``[0, 1]`` using the (provided or dataset)
    min/max and floored into ``num_bins`` equal-width bins, yielding integers in
    ``[0, num_bins - 1]``.

    Args:
        observations: ``(N, state_dim)`` (or ``(..., state_dim)``) array.
        xy_indices: Indices of the X and Y coordinates.
        num_bins: Number of bins (32 in the paper).
        xy_min: Optional per-coordinate minimum (defaults to the dataset min).
        xy_max: Optional per-coordinate maximum (defaults to the dataset max).
        eps: Denominator guard.

    Returns:
        ``int64`` array of shape ``(..., 2)`` with bin indices in ``[0, num_bins-1]``.
    """
    obs = np.asarray(observations, dtype=np.float64)
    xy = obs[..., list(xy_indices)]
    lo = np.asarray(xy_min, dtype=np.float64) if xy_min is not None else xy.min(axis=tuple(range(xy.ndim - 1)))
    hi = np.asarray(xy_max, dtype=np.float64) if xy_max is not None else xy.max(axis=tuple(range(xy.ndim - 1)))
    span = np.maximum(hi - lo, eps)
    scaled = (xy - lo) / span
    bins = np.floor(np.clip(scaled, 0.0, 1.0) * num_bins).astype(np.int64)
    # r == max maps to num_bins -> clamp to the last bin.
    bins = np.clip(bins, 0, num_bins - 1)
    return bins


def discretize_antmaze_xy(
    observations: np.ndarray,
    xy_indices: Sequence[int] = ANTMAZE_XY_INDICES,
    num_bins: int = NUM_XY_BINS,
    xy_min: Optional[Sequence[float]] = None,
    xy_max: Optional[Sequence[float]] = None,
    in_place: bool = False,
    dtype: np.dtype = np.float32,
    **kwargs: Any,
) -> np.ndarray:
    """Replace the AntMaze X/Y coordinates with their discretized bin values.

    Args:
        observations: ``(N, state_dim)`` observation array.
        xy_indices: Indices of the X and Y coordinates (default ``(0, 1)``).
        num_bins: Number of bins (32 in the paper, Appendix C.1).
        xy_min / xy_max: Optional per-coordinate normalization range.
        in_place: If ``True`` modify (a float copy of) the input array in place.
        dtype: Output dtype.
        **kwargs: Ignored extra keyword arguments (kept for forward compatibility
            with different call sites).

    Returns:
        A copy of ``observations`` whose X/Y entries are the integer bin values
        (stored as floats).
    """
    obs = np.asarray(observations, dtype=dtype)
    if in_place and obs is observations:
        out = obs
    else:
        out = np.array(obs, dtype=dtype, copy=True)
    if out.ndim == 1:
        bins = antmaze_xy_bins(
            out[None, :], xy_indices=xy_indices, num_bins=num_bins, xy_min=xy_min, xy_max=xy_max
        )[0]
        out[list(xy_indices)] = bins.astype(dtype)
        return out
    bins = antmaze_xy_bins(out, xy_indices=xy_indices, num_bins=num_bins, xy_min=xy_min, xy_max=xy_max)
    out[..., list(xy_indices)] = bins.astype(dtype)
    return out


def count_xy_bins(
    observations: np.ndarray,
    xy_indices: Sequence[int] = ANTMAZE_XY_INDICES,
    num_bins: int = NUM_XY_BINS,
) -> np.ndarray:
    """Return the ``(num_bins, num_bins)`` histogram of discretized XY positions."""
    bins = antmaze_xy_bins(observations, xy_indices=xy_indices, num_bins=num_bins)
    flat = bins.reshape(-1, 2)
    hist = np.zeros((num_bins, num_bins), dtype=np.int64)
    np.add.at(hist, (flat[:, 0], flat[:, 1]), 1)
    return hist


# --------------------------------------------------------------------------------------
# ExORL physics augmentation (Appendix C.2)
# --------------------------------------------------------------------------------------


def exorl_domain(env_name: str) -> str:
    """Extract the ExORL domain (``walker`` / ``cheetah``) from an env name.

    Accepts names such as ``walker-run``, ``cheetah-walk-backwards`` or a plain
    domain name.
    """
    name = str(env_name or "").lower()
    for domain in EXORL_PHYSICS_FIELDS:
        if domain in name:
            return domain
    return name.split("-")[0]


def exorl_augment_dim(env_name: str) -> int:
    """Number of physics dimensions appended for the given ExORL environment."""
    return int(EXORL_AUGMENT_DIM.get(exorl_domain(env_name), 0))


def exorl_goal_state_dims(env_name: str, state_dim: Optional[int] = None, augment_dim: Optional[int] = None) -> slice:
    """Slice of state dimensions used for goal distance (augmented dims excluded).

    Appendix C.2: "Augmented information is not utilized when calculating goal
    distance."
    """
    aug = exorl_augment_dim(env_name) if augment_dim is None else int(augment_dim)
    if state_dim is None:
        return slice(0, -aug if aug > 0 else None)
    return slice(0, int(state_dim) - aug)


def _physics_values_dm_control(
    domain: str,
    observations: np.ndarray,
    seed: int = 0,
    progress: bool = False,
) -> np.ndarray:
    """Compute physics augmentation by replaying states in DeepMind Control Suite.

    Requires ``dm_control`` (and a working MuJoCo installation).  Raises on
    failure; :func:`exorl_physics_augmentation` handles the fallback.
    """
    from dm_control import suite  # type: ignore  # lazy import

    dm_domain, dm_task = EXORL_DM_CONTROL_DOMAINS[domain]
    env = suite.load(domain_name=dm_domain, task_name=dm_task, task_kwargs={"random": seed})
    physics = env.physics

    obs = np.asarray(observations, dtype=np.float64)
    n = obs.shape[0]
    dim = EXORL_AUGMENT_DIM[domain]
    out = np.zeros((n, dim), dtype=np.float32)

    try:
        for i in range(n):
            state = obs[i]
            try:
                physics.set_state(state)
            except Exception:  # pragma: no cover - depends on state layout
                physics.set_state(state[: physics.get_state().shape[0]])
            physics.after_reset()
            physics.forward()
            if domain == "walker":
                vel = np.asarray(physics.horizontal_velocity(), dtype=np.float64).reshape(-1)
                out[i, 0] = vel[0]
                out[i, 1] = vel[1] if vel.size > 1 else 0.0
                out[i, 2] = float(physics.torso_upright())
                out[i, 3] = float(physics.torso_height())
            elif domain == "cheetah":
                out[i, 0] = float(physics.speed())
            else:  # pragma: no cover - unknown domain
                raise ValueError(f"Unknown ExORL domain: {domain}")
            if progress and (i + 1) % 100000 == 0:
                print(f"[preprocessing] physics augmentation {i + 1}/{n}")
    finally:
        try:
            env.close()
        except Exception:  # pragma: no cover
            pass
    return out


def exorl_physics_augmentation(
    env_name: str = "walker",
    observations: Optional[np.ndarray] = None,
    physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    dataset_dir: Optional[str] = None,
    seed: int = 0,
    allow_fallback: bool = True,
    progress: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Physics augmentation values for ExORL states (Appendix C.2).

    Args:
        env_name: ExORL environment name (e.g. ``"walker-run"``).
        observations: ``(N, state_dim)`` raw environment states.
        physics_fn: Optional user-provided callable ``states -> (N, aug_dim)``
            (used as an escape hatch when ``dm_control`` is unavailable).
        dataset_dir: Unused; accepted for API symmetry with the dataset loaders.
        seed: Random seed used when constructing the DMC environment.
        allow_fallback: If ``True`` and the physics replay fails, warn and return
            zeros of the correct shape instead of raising.
        progress: Print progress while replaying states.
        **kwargs: Ignored extra keyword arguments.

    Returns:
        ``(N, augment_dim)`` float32 array of physics values.
    """
    if observations is None:
        raise ValueError("`observations` is required for physics augmentation.")
    obs = np.asarray(observations, dtype=np.float64)
    domain = exorl_domain(env_name)
    dim = exorl_augment_dim(env_name)

    if physics_fn is not None:
        values = np.asarray(physics_fn(obs), dtype=np.float32)
        values = values.reshape(obs.shape[0], -1)
        if values.shape[1] != dim and dim > 0:
            warnings.warn(
                f"physics_fn returned {values.shape[1]} dims but {dim} were expected for '{env_name}'.",
                RuntimeWarning,
            )
        return values

    if dim == 0 or domain not in EXORL_PHYSICS_FIELDS:
        return np.zeros((obs.shape[0], 0), dtype=np.float32)

    try:
        return _physics_values_dm_control(domain, obs, seed=seed, progress=progress)
    except Exception as exc:  # pragma: no cover - environment dependent
        if not allow_fallback:
            raise
        warnings.warn(
            f"Could not compute ExORL physics augmentation with dm_control ({exc!r}); "
            f"falling back to zeros. Provide `physics_fn` for paper-faithful augmentation.",
            RuntimeWarning,
        )
        return np.zeros((obs.shape[0], dim), dtype=np.float32)


def augment_exorl_physics(
    env_name: Union[str, np.ndarray] = "walker",
    observations: Optional[np.ndarray] = None,
    physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    dataset_dir: Optional[str] = None,
    return_info: bool = False,
    allow_fallback: bool = True,
    progress: bool = False,
    **kwargs: Any,
):
    """Append ExORL physics quantities to the observations (Appendix C.2).

    This is the function used by :mod:`fre.data.dataset` when loading ExORL RND
    datasets.  It accepts both ``(env_name, observations)`` and
    ``(observations, env_name)`` argument orders for robustness.

    Returns:
        ``discretized_observations`` (``(N, state_dim + augment_dim)`` float32),
        or ``(observations, info)`` when ``return_info=True``, where ``info`` is a
        dict with ``{"augment_dim", "augment_indices", "domain", "fields"}``.
    """
    # Tolerate swapped positional arguments.
    if isinstance(env_name, np.ndarray) or (observations is not None and not isinstance(observations, (np.ndarray, list, tuple))):
        env_name, observations = observations, env_name  # type: ignore[assignment]
    if observations is None:
        raise ValueError("`observations` is required for physics augmentation.")

    env_name = str(env_name)
    obs = np.asarray(observations, dtype=np.float32)
    domain = exorl_domain(env_name)
    values = exorl_physics_augmentation(
        env_name=env_name,
        observations=obs,
        physics_fn=physics_fn,
        dataset_dir=dataset_dir,
        allow_fallback=allow_fallback,
        progress=progress,
        **kwargs,
    )
    if values.shape[1] == 0:
        augmented = obs
    else:
        augmented = np.concatenate([obs, values.astype(np.float32)], axis=-1)

    if not return_info:
        return augmented
    info = {
        "augment_dim": int(values.shape[1]),
        "augment_indices": tuple(range(obs.shape[-1], augmented.shape[-1])),
        "domain": domain,
        "fields": EXORL_PHYSICS_FIELDS.get(domain, ()),
    }
    return augmented, info


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------


def compute_state_normalization(
    observations: np.ndarray,
    eps: float = 1e-6,
    clip_std: Optional[float] = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and standard deviation of an observation array.

    Appendix C.2: "Each state dimension is normalized according to the standard
    deviation along that dimension within the offline dataset."
    """
    obs = np.asarray(observations, dtype=np.float64)
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    std = np.maximum(std, eps)
    if clip_std is not None:
        std = np.maximum(std, float(clip_std) / 100.0)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_states(
    observations: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-6,
    clip: Optional[float] = None,
) -> np.ndarray:
    """Normalize observations as ``(x - mean) / std`` (per dimension)."""
    obs = np.asarray(observations, dtype=np.float32)
    if mean is None or std is None:
        m, s = compute_state_normalization(obs, eps=eps)
        mean = m if mean is None else np.asarray(mean, dtype=np.float32)
        std = s if std is None else np.asarray(std, dtype=np.float32)
    out = (obs - np.asarray(mean, dtype=np.float32)) / np.maximum(np.asarray(std, dtype=np.float32), eps)
    if clip is not None:
        out = np.clip(out, -float(clip), float(clip))
    return out.astype(np.float32)


def denormalize_states(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Invert :func:`normalize_states`."""
    obs = np.asarray(observations, dtype=np.float32)
    return (obs * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)).astype(np.float32)


def scale_by_std(
    states: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Divide states by the per-dimension standard deviation (no mean shift).

    Used when computing (normalized) Euclidean goal distances for ExORL.
    """
    return (np.asarray(states, dtype=np.float32) / np.maximum(np.asarray(std, dtype=np.float32), eps)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Goal distances
# --------------------------------------------------------------------------------------


def euclidean_goal_distance(
    states: np.ndarray,
    goals: np.ndarray,
    state_std: Optional[np.ndarray] = None,
    dims: Optional[Union[slice, Sequence[int], np.ndarray]] = None,
) -> np.ndarray:
    """Euclidean distance between states and goals.

    Args:
        states: ``(N, state_dim)`` current states.
        goals: ``(N, state_dim)`` goal states (broadcastable).
        state_std: Optional per-dimension std used to normalize the distance.
        dims: Optional subset of dimensions (e.g. excluding the augmented
            physics dims) used for the distance.

    Returns:
        ``(N,)`` float32 array of distances.
    """
    s = np.asarray(states, dtype=np.float32)
    g = np.asarray(goals, dtype=np.float32)
    if dims is not None:
        idx = dims if not isinstance(dims, slice) else np.arange(s.shape[-1])[dims]
        s = s[..., idx]
        g = g[..., idx]
        if state_std is not None:
            state_std = np.asarray(state_std, dtype=np.float32)[idx]
    diff = s - g
    if state_std is not None:
        diff = diff / np.maximum(np.asarray(state_std, dtype=np.float32), 1e-6)
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def goal_reached_mask(
    states: np.ndarray,
    goals: np.ndarray,
    threshold: float = 0.1,
    state_std: Optional[np.ndarray] = None,
    dims: Optional[Union[slice, Sequence[int], np.ndarray]] = None,
) -> np.ndarray:
    """Boolean mask of goal achievement (Appendix C.2: distance < 0.1)."""
    return euclidean_goal_distance(states, goals, state_std=state_std, dims=dims) < float(threshold)


# --------------------------------------------------------------------------------------
# High-level helpers
# --------------------------------------------------------------------------------------


def preprocess_antmaze(
    observations: np.ndarray,
    discretize_xy: bool = True,
    num_bins: int = NUM_XY_BINS,
    xy_indices: Sequence[int] = ANTMAZE_XY_INDICES,
    **kwargs: Any,
) -> np.ndarray:
    """Apply the AntMaze preprocessing used by FRE/GC-IQL/GC-BC/OPAL (C.1)."""
    obs = np.asarray(observations, dtype=np.float32)
    if not discretize_xy:
        return obs
    return discretize_antmaze_xy(obs, xy_indices=xy_indices, num_bins=num_bins)


def preprocess_exorl(
    observations: np.ndarray,
    env_name: str = "walker",
    augment_physics: bool = True,
    physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    state_std: Optional[np.ndarray] = None,
    normalize: bool = True,
    allow_fallback: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Apply the ExORL preprocessing used for the encoder (C.2).

    Physics quantities are appended, then each state dimension is normalized by
    its dataset standard deviation.
    """
    obs = np.asarray(observations, dtype=np.float32)
    if augment_physics:
        obs = augment_exorl_physics(
            env_name,
            obs,
            physics_fn=physics_fn,
            allow_fallback=allow_fallback,
        )
    if normalize:
        if state_std is None:
            state_std = compute_state_normalization(obs)[1]
        obs = scale_by_std(obs, np.asarray(state_std, dtype=np.float32))
    return obs.astype(np.float32)
