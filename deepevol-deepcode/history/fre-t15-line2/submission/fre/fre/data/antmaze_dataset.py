"""AntMaze offline dataset utilities for FRE (Appendix C.1).

Paper references
----------------
* "We utilize the ``antmaze-large-diverse-v2`` dataset from D4RL (Fu et al., 2020).
  Online evaluation is performed with a length of 2000 timesteps. The ant robot is
  placed in the center of the maze to allow for more diverse behavior, in comparison
  to the original start position in the bottom-left." (Appendix C.1)
* "For the goal-reaching tasks, we utilize a reward function that considers the goal
  reached if an agent reaches within a distance of 2 with the target position. The
  FRE, GC-IQL, GC-BC, and OPAL agents all utilize a discretized preprocessing
  procedure, where the X and Y coordinates are discretized into 32 bins." (Appendix C.1)

This module provides

* :func:`discretize_xy` -- the 32-bin X/Y discretization shared by FRE, GC-IQL, GC-BC
  and OPAL (Appendix C.1).
* :class:`AntMazeDataset` -- a thin wrapper around the raw D4RL arrays that keeps
  trajectory boundaries, provides dataset statistics, the maze-center start state,
  and a ready-to-use :class:`~fre.data.replay.ReplayBuffer`.
* :func:`load_antmaze_dataset` -- loader from ``d4rl`` / an HDF5 file / an ``.npz``
  dump / user-supplied arrays.

Notes on details the paper leaves unspecified (marked inline as "paper silent")
------------------------------------------------------------------------------
* ``NUM_XY_BINS = 32`` comes from the addendum/C.1; the *grid bounds* used to compute
  the bins are not stated.  The appendix reports the hand-crafted goal locations on an
  "(X,Y) grid with origin at the bottom left" (e.g. ``goal-top`` at ``(35, 24)``), i.e.
  in the same units as the raw MuJoCo ant position.  We therefore default to the fixed
  bounds ``[0, 32]`` per axis so that integer coordinates (e.g. ``(28, 0)``) map to
  themselves, and offer ``bounds="dataset"`` to use the empirical min/max instead.
* The exact center coordinate of the maze is not given; ``CENTER_POSITION`` is the
  geometric center of the ``[0, 32]`` grid and the actual start observation is chosen
  as the *closest state present in the offline dataset*, which guarantees a feasible
  (non-penetrating) ant configuration.
* ``discretize_mode``: the paper only states that coordinates are "discretized into 32
  bins".  We default to replacing X/Y with the integer bin index (``"index"``, matching
  the grid coordinates the goals are expressed in) and also support ``"center"``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------------------
# Replay buffer import (with a fallback path so the module also works stand-alone)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import plumbing
    from fre.data.replay import Batch, ReplayBuffer, make_replay_buffer
except ImportError:  # pragma: no cover
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(_here, "..", ".."))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from fre.data.replay import Batch, ReplayBuffer, make_replay_buffer  # type: ignore
    except ImportError:  # pragma: no cover - allow degraded standalone use
        ReplayBuffer = None  # type: ignore
        Batch = None  # type: ignore
        make_replay_buffer = None  # type: ignore


__all__ = [
    "ANTMAZE_DATASET_NAME",
    "ANTMAZE_DATASET_ENV_ID",
    "ANTMAZE_DATASET_ENV_IDS",
    "NUM_XY_BINS",
    "POSITION_DIMS",
    "DEFAULT_XY_BOUNDS",
    "CENTER_POSITION",
    "GOAL_DISTANCE_THRESHOLD",
    "EVAL_EPISODE_LENGTH",
    "AntMazeDataset",
    "discretize_xy",
    "continuous_xy_from_bins",
    "make_antmaze_dataset",
    "load_antmaze_dataset",
    "load_antmaze_arrays",
]


# --------------------------------------------------------------------------------------
# Constants (paper: Appendix C.1 and the addendum's AntMaze task list)
# --------------------------------------------------------------------------------------
#: D4RL dataset identifier used by the paper.
ANTMAZE_DATASET_NAME = "antmaze-large-diverse-v2"

#: ``gym.make`` id corresponding to the dataset (d4rl keeps the ``-v2`` suffix).
ANTMAZE_DATASET_ENV_ID = "antmaze-large-diverse-v2"

#: Aliases that resolve to the same antmaze-large-diverse-v2 arrays.
ANTMAZE_DATASET_ENV_IDS = {
    "antmaze-large-diverse-v2": "antmaze-large-diverse-v2",
    "antmaze-large-diverse": "antmaze-large-diverse-v2",
    "ant-large-diverse-v2": "antmaze-large-diverse-v2",
    "large-diverse-v2": "antmaze-large-diverse-v2",
    ANTMAZE_DATASET_NAME: "antmaze-large-diverse-v2",
}

#: Number of bins for the X and Y coordinates (Appendix C.1).
NUM_XY_BINS = 32

#: X and Y positions live in the first two entries of the antmaze observation.
POSITION_DIMS: Tuple[int, int] = (0, 1)

#: Fixed grid bounds used for the 32-bin discretization ("paper silent").
#: The appendix expresses goal locations on an (X, Y) grid with the origin at the
#: bottom-left (e.g. (28, 0), (35, 24)), i.e. in raw position units inside [0, ~36].
DEFAULT_XY_BOUNDS: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 32.0), (0.0, 32.0))

#: Geometric center of the [0, 32] grid -- the ant is placed here at evaluation time
#: ("The ant robot is placed in the center of the maze", Appendix C.1).
CENTER_POSITION: Tuple[float, float] = (16.0, 16.0)

#: Goal-reaching success radius (Appendix C.1: "within a distance of 2").
GOAL_DISTANCE_THRESHOLD = 2.0

#: Online evaluation length, in timesteps (Appendix C.1).
EVAL_EPISODE_LENGTH = 2000


# --------------------------------------------------------------------------------------
# 32-bin discretization
# --------------------------------------------------------------------------------------
def _resolve_xy_bounds(
    bounds: Union[str, Sequence[Sequence[float]], np.ndarray, None],
    xy: Optional[np.ndarray] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Resolve the ``bounds`` argument of :func:`discretize_xy`.

    ``bounds`` may be ``None``/``"default"`` (fixed :data:`DEFAULT_XY_BOUNDS`),
    ``"dataset"`` (empirical min/max of ``xy``) or an explicit sequence
    ``((xmin, xmax), (ymin, ymax))`` / ``(D, 2)`` array (only the position dims are used).
    """
    if bounds is None or (isinstance(bounds, str) and bounds.lower() in ("default", "fixed", "grid")):
        return DEFAULT_XY_BOUNDS
    if isinstance(bounds, str):
        if bounds.lower() in ("dataset", "data", "empirical") and xy is not None:
            arr = np.asarray(xy, dtype=np.float64)
            lo = arr.min(axis=0)
            hi = arr.max(axis=0)
            span = np.maximum(hi - lo, 1e-6)
            hi = lo + span  # guard against a degenerate dimension
            return ((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])))
        raise ValueError(f"Unknown bounds specification: {bounds!r}")
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape == (2, 2):
        return ((float(arr[0, 0]), float(arr[0, 1])), (float(arr[1, 0]), float(arr[1, 1])))
    if arr.ndim == 2 and arr.shape[-1] == 2:
        # (D, 2) per-dimension (min, max); take the position dimensions in order.
        return ((float(arr[0, 0]), float(arr[0, 1])), (float(arr[1, 0]), float(arr[1, 1])))
    raise ValueError(f"bounds must be ((xmin, xmax), (ymin, ymax)); got shape {arr.shape}")


def discrete_xy_values(
    xy: np.ndarray,
    num_bins: int = NUM_XY_BINS,
    bounds: Union[str, Sequence[Sequence[float]], None] = None,
) -> np.ndarray:
    """Continuous XY positions -> integer bin indices in ``[0, num_bins - 1]``."""
    arr = np.asarray(xy, dtype=np.float64)
    single = arr.ndim == 1
    arr = np.atleast_2d(arr)
    if arr.shape[-1] != 2:
        raise ValueError(f"Expected XY positions with last dim 2, got shape {arr.shape}")
    (xmin, xmax), (ymin, ymax) = _resolve_xy_bounds(bounds, arr)
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    bx = np.floor((arr[:, 0] - xmin) / span_x * num_bins)
    by = np.floor((arr[:, 1] - ymin) / span_y * num_bins)
    bins = np.stack([bx, by], axis=-1)
    bins = np.clip(bins, 0, num_bins - 1)
    if single:
        return bins[0].astype(np.int64)
    return bins.astype(np.int64)


def continuous_xy_from_bins(
    bins: np.ndarray,
    num_bins: int = NUM_XY_BINS,
    bounds: Union[str, Sequence[Sequence[float]], None] = None,
) -> np.ndarray:
    """Inverse of :func:`discretize_xy` for ``mode="center"`` (bin centre coordinates)."""
    b = np.asarray(bins, dtype=np.float64)
    single = b.ndim == 1
    b = np.atleast_2d(b)
    (xmin, xmax), (ymin, ymax) = _resolve_xy_bounds(bounds, None)
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    x = xmin + (b[:, 0] + 0.5) * span_x / num_bins
    y = ymin + (b[:, 1] + 0.5) * span_y / num_bins
    out = np.stack([x, y], axis=-1)
    return out[0] if single else out


def discretize_xy(
    observations: np.ndarray,
    num_bins: int = NUM_XY_BINS,
    position_dims: Sequence[int] = POSITION_DIMS,
    bounds: Union[str, Sequence[Sequence[float]], None] = None,
    mode: str = "index",
    in_place: bool = False,
) -> np.ndarray:
    """Discretize the X and Y coordinates of antmaze observations into 32 bins.

    Matches Appendix C.1: "the X and Y coordinates are discretized into 32 bins" for
    the FRE, GC-IQL, GC-BC and OPAL agents.

    Parameters
    ----------
    observations:
        Array of shape ``(..., obs_dim)`` (or ``(obs_dim,)`` for a single state).
    num_bins:
        Number of bins per axis (32 in the paper).
    position_dims:
        Indices of the X and Y entries inside the observation (``(0, 1)`` for
        ``antmaze-large-diverse-v2``).
    bounds:
        ``None``/``"default"`` -> :data:`DEFAULT_XY_BOUNDS` (raw grid units, so the
        appendix's integer goal coordinates are their own bin index); ``"dataset"`` ->
        empirical min/max; or an explicit ``((xmin, xmax), (ymin, ymax))``.
    mode:
        ``"index"`` (default) stores the integer bin index; ``"center"`` stores the
        bin centre expressed in the original coordinate units ("paper silent").
    in_place:
        Write into ``observations`` when it is a float array (avoids a copy).
    """
    arr = np.asarray(observations)
    single = arr.ndim == 1
    work = np.atleast_2d(arr).astype(np.float64, copy=not in_place or arr.dtype != np.float64)
    if work.shape[-1] <= max(position_dims):
        raise ValueError(
            f"observations of shape {np.asarray(observations).shape} do not contain "
            f"position dims {tuple(position_dims)}"
        )
    xy = work[:, list(position_dims)]
    bins = discrete_xy_values(xy, num_bins=num_bins, bounds=bounds)
    if mode in ("index", "bin", "int"):
        values = bins.astype(np.float64)
    elif mode in ("center", "centre", "continuous"):
        values = continuous_xy_from_bins(bins, num_bins=num_bins, bounds=bounds)
    else:
        raise ValueError(f"Unknown discretize mode {mode!r}; use 'index' or 'center'")
    work[:, position_dims[0]] = values[:, 0]
    work[:, position_dims[1]] = values[:, 1]
    if single:
        return work[0]
    if in_place and isinstance(observations, np.ndarray) and observations.dtype == np.float64:
        observations[...] = work
        return observations
    return work


# --------------------------------------------------------------------------------------
# Trajectory-boundary helpers
# --------------------------------------------------------------------------------------
def _compute_ends(terminals: Optional[np.ndarray], timeouts: Optional[np.ndarray], n: int) -> np.ndarray:
    """End index (exclusive) of every trajectory from terminal/timeout flags."""
    done = np.zeros(n, dtype=bool)
    if terminals is not None:
        done |= np.asarray(terminals, dtype=bool).reshape(-1)[:n]
    if timeouts is not None:
        done |= np.asarray(timeouts, dtype=bool).reshape(-1)[:n]
    if not done.any():
        done[-1] = True
    ends = (np.flatnonzero(done) + 1).tolist()
    if ends[-1] != n:
        ends.append(n)
    return np.asarray(ends, dtype=np.int64)


def _as_2d_float(array: Any, name: str) -> np.ndarray:
    if array is None:
        raise ValueError(f"'{name}' must not be None")
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


# --------------------------------------------------------------------------------------
# Dataset container
# --------------------------------------------------------------------------------------
@dataclass
class AntMazeDataset:
    """``antmaze-large-diverse-v2`` wrapper with 32-bin XY discretization.

    The class keeps the raw arrays, the discretized copy used as observations for the
    FRE/GC-IQL/GC-BC/OPAL agents, dataset statistics, and an (optional) lazily built
    :class:`~fre.data.replay.ReplayBuffer` for uniform state / transition sampling.

    Construction normally happens through :func:`load_antmaze_dataset` or
    :meth:`from_arrays`.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: Optional[np.ndarray] = None
    terminals: Optional[np.ndarray] = None
    timeouts: Optional[np.ndarray] = None
    next_observations: Optional[np.ndarray] = None
    ends: Optional[np.ndarray] = None
    dataset_name: str = ANTMAZE_DATASET_NAME
    num_bins: int = NUM_XY_BINS
    position_dims: Tuple[int, int] = POSITION_DIMS
    bounds: Union[str, Sequence[Sequence[float]], None] = None
    discretize_mode: str = "index"
    discretize: bool = True
    center_position: Tuple[float, float] = CENTER_POSITION
    goal_distance_threshold: float = GOAL_DISTANCE_THRESHOLD
    eval_episode_length: int = EVAL_EPISODE_LENGTH
    seed: int = 0
    buffer: Optional[Any] = None
    _discretized: Optional[np.ndarray] = field(default=None, repr=False)
    _stats: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    _center_state: Optional[np.ndarray] = field(default=None, repr=False)

    # ---------------------------------------------------------------- construction
    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        n = len(self.observations)
        if self.rewards is None:
            self.rewards = np.zeros(n, dtype=np.float32)
        else:
            self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        if self.terminals is None:
            self.terminals = np.zeros(n, dtype=np.float32)
        else:
            self.terminals = np.asarray(self.terminals, dtype=np.float32).reshape(-1)
        if self.timeouts is not None:
            self.timeouts = np.asarray(self.timeouts, dtype=np.float32).reshape(-1)
        if self.next_observations is not None:
            self.next_observations = np.asarray(self.next_observations, dtype=np.float32)
        if self.ends is None:
            self.ends = _compute_ends(self.terminals, self.timeouts, len(self.terminals))
        else:
            self.ends = np.asarray(self.ends, dtype=np.int64)
        if self.position_dims is POSITION_DIMS or self.position_dims is None:
            self.position_dims = POSITION_DIMS
        if self.discretize:
            self._discretized = self._compute_discretized(self.observations)

    @classmethod
    def from_arrays(
        cls,
        observations: Any,
        actions: Any,
        rewards: Optional[Any] = None,
        terminals: Optional[Any] = None,
        timeouts: Optional[Any] = None,
        next_observations: Optional[Any] = None,
        ends: Optional[Any] = None,
        **kwargs: Any,
    ) -> "AntMazeDataset":
        """Build a dataset from D4RL-style flat arrays (``ends`` optional)."""
        kwargs.pop("buffer", None)
        return cls(
            observations=_as_2d_float(observations, "observations"),
            actions=_as_2d_float(actions, "actions"),
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
            next_observations=next_observations,
            ends=ends,
            **kwargs,
        )

    # ---------------------------------------------------------------- discretization
    def _compute_discretized(self, observations: np.ndarray) -> np.ndarray:
        return discretize_xy(
            observations,
            num_bins=self.num_bins,
            position_dims=self.position_dims,
            bounds=self.bounds,
            mode=self.discretize_mode,
        ).astype(np.float32)

    @property
    def observation_space(self) -> np.ndarray:
        """The discretized observations used by the FRE/GC agents (Appendix C.1)."""
        if self._discretized is None:
            return self.observations
        return self._discretized

    def agent_observations(self) -> np.ndarray:
        """Observations handed to the learning agents (discretized when enabled)."""
        return self.observation_space

    def xy(self, observations: Optional[np.ndarray] = None) -> np.ndarray:
        """Continuous-ish X/Y positions currently stored in the observations."""
        obs = self.observation_space if observations is None else np.asarray(observations)
        obs = np.atleast_2d(obs)
        return obs[:, list(self.position_dims)]

    def discretize_observations(self, observations: Optional[np.ndarray] = None, in_place: bool = False) -> np.ndarray:
        """Apply :func:`discretize_xy` to arbitrary observations."""
        obs = self.observation_space if observations is None else observations
        return discretize_xy(
            obs,
            num_bins=self.num_bins,
            position_dims=self.position_dims,
            bounds=self.bounds,
            mode=self.discretize_mode,
            in_place=in_place,
        )

    def position_bins(self, position: Sequence[float]) -> np.ndarray:
        """Map a hand-crafted grid coordinate (e.g. ``(28, 0)``) to its bin index."""
        return discrete_xy_values(np.asarray(position, dtype=np.float64).reshape(1, 2), self.num_bins, self.bounds)[0]

    # ---------------------------------------------------------------- statistics
    def statistics(self, use_discretized: bool = True) -> Dict[str, np.ndarray]:
        """Dataset mean/std/min/max of the agent observations (cached)."""
        if self._stats is None:
            obs = self.observation_space if use_discretized else self.observations
            obs = np.asarray(obs, dtype=np.float64)
            std = obs.std(axis=0)
            std = np.maximum(std, 1e-6)
            self._stats = {
                "mean": obs.mean(axis=0).astype(np.float32),
                "std": std.astype(np.float32),
                "min": obs.min(axis=0).astype(np.float32),
                "max": obs.max(axis=0).astype(np.float32),
            }
        return self._stats

    def state_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        stats = self.statistics()
        return stats["mean"], stats["std"]

    def state_box(self) -> Tuple[np.ndarray, np.ndarray]:
        stats = self.statistics()
        return stats["min"], stats["max"]

    def xy_bounds_actual(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Empirical ``((xmin, xmax), (ymin, ymax))`` of the stored coordinates."""
        xy = np.asarray(self.xy(), dtype=np.float64)
        return ((float(xy[:, 0].min()), float(xy[:, 0].max())), (float(xy[:, 1].min()), float(xy[:, 1].max())))

    @property
    def obs_dim(self) -> int:
        return int(self.observation_space.shape[-1])

    @property
    def raw_obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def act_dim(self) -> int:
        return int(self.actions.shape[-1])

    @property
    def num_transitions(self) -> int:
        return int(len(self.observations))

    @property
    def num_trajectories(self) -> int:
        return int(len(self.ends))

    def __len__(self) -> int:
        return self.num_transitions

    def trajectory_slice(self, traj_index: int) -> Tuple[int, int]:
        """``(start, end)`` transition indices of trajectory ``traj_index``."""
        start = 0 if traj_index == 0 else int(self.ends[traj_index - 1])
        return start, int(self.ends[traj_index])

    # ---------------------------------------------------------------- center start
    def _center_xy_discrete(self) -> np.ndarray:
        """Centre of the maze expressed in the stored (possibly discretized) units."""
        bins = self.position_bins(self.center_position)
        if self.discretize and self.discretize_mode.startswith("center"):
            xy = continuous_xy_from_bins(bins, self.num_bins, self.bounds)
        else:
            xy = bins.astype(np.float64)
        return xy

    def center_start_state(self, observations: Optional[np.ndarray] = None) -> np.ndarray:
        """State used to place the ant in the centre of the maze (Appendix C.1).

        The paper says the ant starts at the centre "to allow for more diverse
        behavior".  The centre coordinate is not specified, so we search the offline
        dataset for the observation whose X/Y is closest to :data:`CENTER_POSITION`;
        this guarantees a physically feasible (non-penetrating) start pose and keeps
        the velocity dimensions in-distribution.
        """
        if self._center_state is not None and observations is None:
            return self._center_state.copy()
        obs = self.observation_space if observations is None else np.asarray(observations, dtype=np.float32)
        obs = np.atleast_2d(obs).astype(np.float32)
        target = self._center_xy_discrete()
        xy = np.asarray(self.xy(obs), dtype=np.float64)
        dist = np.linalg.norm(xy - target[None, :], axis=-1)
        # Restrict to transitions that are the *first* step of a trajectory when
        # possible, so the returned configuration is a genuine reset state.
        candidate = int(np.argmin(dist))
        state = obs[candidate].copy()
        if observations is None:
            self._center_state = state
        return state

    def with_position(self, state: np.ndarray, position: Sequence[float] = None) -> np.ndarray:
        """Return ``state`` with its X/Y entries set to the (discretized) ``position``."""
        out = np.array(state, dtype=np.float32, copy=True)
        if position is None:
            target = self._center_xy_discrete()
        else:
            pos = np.asarray(position, dtype=np.float64).reshape(2)
            if self.discretize and self.discretize_mode.startswith("center"):
                pos = continuous_xy_from_bins(self.position_bins(pos), self.num_bins, self.bounds)
            elif self.discretize:
                pos = self.position_bins(pos).astype(np.float64)
            target = pos
        out[self.position_dims[0]] = target[0]
        out[self.position_dims[1]] = target[1]
        return out

    def start_state(self, observations: Optional[np.ndarray] = None) -> np.ndarray:
        """Alias of :meth:`center_start_state` (maze-centre start, Appendix C.1)."""
        return self.center_start_state(observations)

    # ---------------------------------------------------------------- goals / rewards
    def goal_distance(self, observations: np.ndarray, goal: Sequence[float]) -> np.ndarray:
        """Euclidean X/Y distance between ``observations`` and a goal location.

        Distances are computed on the *discretized* coordinates, matching the
        preprocessing used by FRE/GC-IQL/GC-BC/OPAL (Appendix C.1).
        """
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        if not isinstance(goal, (list, tuple, np.ndarray)):
            raise TypeError("goal must be a 2D position")
        goal_arr = np.asarray(goal, dtype=np.float64).reshape(2)
        if self.discretize:
            # Interpret the goal as a grid coordinate / position in the same units as
            # the (possibly discretized) observations.
            goal_arr = np.asarray(self.position_bins(goal_arr), dtype=np.float64)
        xy = obs[:, list(self.position_dims)]
        return np.linalg.norm(xy - goal_arr[None, :], axis=-1)

    def goal_reached(
        self,
        observations: np.ndarray,
        goal: Sequence[float],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Boolean mask: ``True`` where the agent is within ``threshold`` of ``goal``."""
        thr = self.goal_distance_threshold if threshold is None else float(threshold)
        return self.goal_distance(observations, goal) <= thr

    # ---------------------------------------------------------------- replay buffer
    def build_buffer(
        self,
        discretize: Optional[bool] = None,
        seed: Optional[int] = None,
        exclude_final_states: bool = True,
        attach_stats: bool = True,
    ) -> Any:
        """Create (and cache) a :class:`~fre.data.replay.ReplayBuffer` for this dataset.

        ``discretize`` overrides :attr:`discretize`; ``exclude_final_states`` defaults to
        ``True`` so that sampled context/decoder states are genuine environment states.
        """
        if make_replay_buffer is None:  # pragma: no cover
            raise ImportError("fre.data.replay is unavailable; cannot build a ReplayBuffer")
        use_disc = self.discretize if discretize is None else bool(discretize)
        obs = self._compute_discretized(self.observations) if use_disc else self.observations
        next_obs = None
        if self.next_observations is not None:
            next_obs = self._compute_discretized(self.next_observations) if use_disc else self.next_observations
        else:
            # Reconstruct `next_observations` from the flat transition ordering when the
            # dataset does not carry them (D4RL antmaze files do, but be defensive).
            next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
        buffer = make_replay_buffer(
            observations=obs,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            timeouts=self.timeouts,
            next_observations=next_obs,
            ends=self.ends,
            seed=self.seed if seed is None else int(seed),
            exclude_final_states=exclude_final_states,
        )
        if attach_stats and buffer is not None:
            # Expose dataset statistics / bounds to consumers (reward priors use
            # `state_box()` when available to compute exact reward ranges; the paper is
            # silent about this, but the linear prior needs a bounded state space).
            mean, std = self.state_statistics()
            lo, hi = self.state_box()
            try:
                if not hasattr(buffer, "state_box"):
                    setattr(buffer, "state_box", lambda: (lo.copy(), hi.copy()))
                if not hasattr(buffer, "state_mean_std"):
                    setattr(buffer, "state_mean_std", lambda: (mean.copy(), std.copy()))
            except Exception:  # pragma: no cover - defensive
                pass
        self.buffer = buffer
        return buffer

    @property
    def replay_buffer(self) -> Any:
        if self.buffer is None:
            self.build_buffer()
        return self.buffer

    def sample_states(self, num_states: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Uniform states from the offline dataset (delegates to the replay buffer)."""
        rng = np.random.default_rng(self.seed) if rng is None else rng
        return self.replay_buffer.sample_states(num_states, rng=rng)

    def sample_states_with_metadata(self, num_states: int, rng: Optional[np.random.Generator] = None):
        rng = np.random.default_rng(self.seed) if rng is None else rng
        return self.replay_buffer.sample_states_with_metadata(num_states, rng=rng)

    # ---------------------------------------------------------------- misc
    def to_npz(self, path: str) -> str:
        payload = {
            "observations": self.observations,
            "actions": self.actions,
            "terminals": self.terminals,
            "ends": self.ends,
        }
        if self.rewards is not None:
            payload["rewards"] = self.rewards
        if self.next_observations is not None:
            payload["next_observations"] = self.next_observations
        np.savez_compressed(path, **payload)
        return path

    def describe(self) -> Dict[str, Any]:
        (xmin, xmax), (ymin, ymax) = self.xy_bounds_actual()
        return {
            "dataset_name": self.dataset_name,
            "num_transitions": self.num_transitions,
            "num_trajectories": self.num_trajectories,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "discretize": bool(self.discretize),
            "num_xy_bins": int(self.num_bins),
            "position_dims": tuple(int(d) for d in self.position_dims),
            "discretize_mode": self.discretize_mode,
            "bounds": self.bounds if isinstance(self.bounds, str) or self.bounds is None else list(map(list, self.bounds)),
            "xy_range": [[xmin, xmax], [ymin, ymax]],
            "center_position": list(self.center_position),
            "goal_distance_threshold": float(self.goal_distance_threshold),
            "eval_episode_length": int(self.eval_episode_length),
        }


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _load_hdf5(path: str) -> Dict[str, np.ndarray]:
    """Load a D4RL-style antmaze HDF5 dump (keys: observations/actions/terminals/...)."""
    import h5py  # local import: h5py is only needed for file loading

    data: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in ("observations", "actions", "rewards", "terminals", "timeouts", "next_observations"):
            if key in f:
                data[key] = np.asarray(f[key][()])
        # Some releases store per-trajectory groups instead of flat datasets.
        if "observations" not in data and any(k.startswith("traj") for k in f.keys()):
            obs, act, term = [], [], []
            for tname in sorted(f.keys()):
                grp = f[tname]
                obs.append(np.asarray(grp["observations"][()]))
                act.append(np.asarray(grp["actions"][()]))
                if "terminals" in grp:
                    term.append(np.asarray(grp["terminals"][()]))
                else:
                    t = np.zeros(len(grp["actions"]), dtype=np.float32)
                    t[-1] = 1.0
                    term.append(t)
            data["observations"] = np.concatenate(obs, axis=0)
            data["actions"] = np.concatenate(act, axis=0)
            data["terminals"] = np.concatenate(term, axis=0)
    if "observations" not in data:
        raise ValueError(f"Could not find 'observations' inside {path}")
    return data


def load_antmaze_arrays(
    path: Optional[str] = None,
    dataset_name: str = ANTMAZE_DATASET_NAME,
    env_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load the raw ``antmaze-large-diverse-v2`` transition arrays.

    Resolution order:

    1. ``path`` (``.hdf5``/``.h5``/``.npz``) when given;
    2. ``$FRE_ANTMAZE_DATASET`` when set;
    3. ``d4rl`` via ``gym.make(env).get_dataset()`` (requires D4RL pinned *before*
       June 2024, as noted in the paper's addendum).
    """
    path = path or os.environ.get("FRE_ANTMAZE_DATASET")
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"AntMaze dataset file not found: {path}")
        if path.endswith((".hdf5", ".h5", ".hdf")):
            data = _load_hdf5(path)
        else:
            with np.load(path, allow_pickle=True) as npz:
                data = {k: np.asarray(npz[k]) for k in npz.files}
    else:
        env_id = env_id or ANTMAZE_DATASET_ENV_IDS.get(dataset_name, ANTMAZE_DATASET_ENV_ID)
        try:  # pragma: no cover - requires d4rl/gym/mujoco
            import gym  # noqa: F401
            import d4rl  # noqa: F401  (registers the antmaze environments)

            env = gym.make(env_id)
            data = env.get_dataset()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not load the antmaze-large-diverse-v2 dataset from D4RL "
                "(paper: 'We utilize the antmaze-large-diverse-v2 dataset from D4RL'). "
                "Install D4RL from a commit predating June 2024, or pass an explicit "
                "HDF5/NPZ path / set $FRE_ANTMAZE_DATASET."
            ) from exc

    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    rewards = None if "rewards" not in data else np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
    terminals = None if "terminals" not in data else np.asarray(data["terminals"], dtype=np.float32).reshape(-1)
    timeouts = None if "timeouts" not in data else np.asarray(data["timeouts"], dtype=np.float32).reshape(-1)
    next_observations = None
    if "next_observations" in data:
        next_observations = np.asarray(data["next_observations"], dtype=np.float32)
    if "ends" in data:
        ends = np.asarray(data["ends"], dtype=np.int64)
    else:
        ends = _compute_ends(terminals, timeouts, len(terminals if terminals is not None else observations))
    if limit is not None:
        limit = int(min(limit, ends[-1]))
        # keep whole trajectories only
        keep = int(np.searchsorted(ends, limit, side="right"))
        keep = max(keep, 1)
        ends = ends[:keep]
        n = int(ends[-1])
        observations = observations[:n]
        actions = actions[:n]
        rewards = None if rewards is None else rewards[:n]
        terminals = None if terminals is None else terminals[:n]
        timeouts = None if timeouts is None else timeouts[:n]
        next_observations = None if next_observations is None else next_observations[:n]
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "next_observations": next_observations,
        "ends": ends,
    }


def make_antmaze_dataset(
    observations: Any,
    actions: Any,
    rewards: Optional[Any] = None,
    terminals: Optional[Any] = None,
    timeouts: Optional[Any] = None,
    next_observations: Optional[Any] = None,
    ends: Optional[Any] = None,
    **kwargs: Any,
) -> AntMazeDataset:
    """Factory mirroring the other ``make_*`` helpers in the codebase."""
    return AntMazeDataset.from_arrays(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        next_observations=next_observations,
        ends=ends,
        **kwargs,
    )


def load_antmaze_dataset(
    path: Optional[str] = None,
    dataset_name: str = ANTMAZE_DATASET_NAME,
    env_id: Optional[str] = None,
    num_bins: int = NUM_XY_BINS,
    discretize: bool = True,
    discretize_mode: str = "index",
    bounds: Union[str, Sequence[Sequence[float]], None] = None,
    build_buffer: bool = True,
    exclude_final_states: bool = True,
    attach_stats: bool = True,
    limit: Optional[int] = None,
    seed: int = 0,
    **kwargs: Any,
) -> AntMazeDataset:
    """Load ``antmaze-large-diverse-v2`` and apply the paper's preprocessing.

    Parameters
    ----------
    path:
        Optional HDF5/NPZ file.  When omitted, ``$FRE_ANTMAZE_DATASET`` or D4RL is used.
    num_bins:
        X/Y discretization resolution (32 in Appendix C.1).
    discretize:
        Replace the X/Y coordinates with their 32-bin values for the agent observations.
    discretize_mode:
        ``"index"`` (integer bin index, default) or ``"center"`` (bin centre).
    bounds:
        ``None``/``"default"`` -> fixed ``[0, 32]`` grid; ``"dataset"`` -> empirical
        min/max; or explicit ``((xmin, xmax), (ymin, ymax))``.  "Paper silent" -- the
        default matches the appendix's integer goal coordinates.
    build_buffer:
        Immediately construct the :class:`~fre.data.replay.ReplayBuffer`.
    """
    arrays = load_antmaze_arrays(path=path, dataset_name=dataset_name, env_id=env_id, limit=limit)
    dataset = AntMazeDataset.from_arrays(
        observations=arrays["observations"],
        actions=arrays["actions"],
        rewards=arrays["rewards"],
        terminals=arrays["terminals"],
        timeouts=arrays["timeouts"],
        next_observations=arrays["next_observations"],
        ends=arrays["ends"],
        dataset_name=dataset_name,
        num_bins=num_bins,
        discretize=discretize,
        discretize_mode=discretize_mode,
        bounds=bounds,
        seed=seed,
        **kwargs,
    )
    if build_buffer:
        dataset.build_buffer(
            discretize=discretize,
            seed=seed,
            exclude_final_states=exclude_final_states,
            attach_stats=attach_stats,
        )
    return dataset
