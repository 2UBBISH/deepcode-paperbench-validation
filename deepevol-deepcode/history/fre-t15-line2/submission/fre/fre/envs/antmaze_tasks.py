"""AntMaze zero-shot evaluation tasks for FRE.

Implements every AntMaze evaluation reward function described in the paper
(Appendix C.1 and the addendum's *Ant Maze evaluation tasks*) so that the
zero-shot harness can encode ~32 ``(state, eta(state))`` pairs of a novel task
and roll out the frozen FRE policy with no further training.

Paper specification used here
-----------------------------
Source: Addendum -- Ant Maze evaluation tasks / C.1. AntMaze

* Episodes have a **maximum** length of 2000 steps per trajectory.
* ``ant-goal-reaching``: average of 5 hand-crafted fixed goal-reaching reward
  functions.  "The reward is set to -1 for every timestep that the goal is not
  achieved."  Goal locations on an (X,Y) grid with the origin at the bottom left:

  - ``goal-bottom`` at ``(28, 0)``
  - ``goal-left``   at ``(0, 15)``
  - ``goal-top``    at ``(35, 24)``
  - ``goal-center`` at ``(12, 24)``
  - ``goal-right``  at ``(33, 16)``

  "we utilize a reward function that considers the goal reached if an agent
  reaches within a distance of 2 with the target position." (C.1)

* ``ant-directional``: each task specifies a target velocity in the (X,Y) plane
  and "the reward function checks the agent's actual velocity and grants higher
  reward the closer it is to the target velocity, using a simple dot product".
  The four directions listed in the addendum are ``vel_left (-1,0)``,
  ``vel_up (0,1)``, ``vel_down (0,-1)`` and ``vel_right (1,0)``.  (The addendum
  prose says "5 directional tasks" but enumerates four; we implement the four
  that are enumerated, which is what the reported average requires.)

* ``ant-random-simplex``: five seeded tasks (seeds 1-5).  Each is "a random 2D
  noise 'height map' plus velocity preferences in the (X,Y) grid of the AntMaze
  generated via opensimplex"; the agent "gets baseline negative reward (-1) at
  each step, a bonus if it stands in higher 'height' regions, and an additional
  bonus for moving in the local 'preferred' velocity direction indicated by the
  noise field."

* ``ant-path-center`` / ``ant-path-loop`` / ``ant-path-edges``: "reward
  functions that reward the agent for moving along hand-crafted corridors placed
  in the center of the grid, for moving in a hand-crafted loop around the grid,
  and for moving along the edges of the grid".

* Observations: "The FRE, GC-IQL, GC-BC, and OPAL agents all utilize a
  discretized preprocessing procedure, where the X and Y coordinates are
  discretized into 32 bins."  Every task here therefore accepts *either*
  discretized bin-index observations (converted back to continuous units with
  bin centres, using ``bounds``) or raw continuous observations, selected with
  the ``discretized`` flag.

Details the paper does not specify (documented defaults)
-------------------------------------------------------
* The continuous (X,Y) extent used to invert the 32-bin discretization is not
  given; the addendum's goal ``(35, 24)`` implies a grid larger than 32 units,
  so we default to ``ANTMAZE_XY_BOUNDS = ((0, 36), (0, 36))`` (the antmaze-large
  maze spans 9 cells of the D4RL ``maze_size_scaling = 4``).  Pass explicit
  ``bounds`` to match a different loader convention.
* The exact height/velocity bonus weights of the simplex task, the velocity
  scale of the directional task and the corridor geometries/bandwidths of the
  path tasks are not specified; sensible defaults are exposed as keyword
  arguments and constants.
* Episode return normalisation bounds (0-100 per Table 1) are exposed per task
  through ``metadata["score_bounds"]``; the exact normalisation is applied by
  :mod:`fre.eval.zero_shot_eval`.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# imports of the shared env-task containers (lazy in fre.envs, so no cycle)
# --------------------------------------------------------------------------
try:  # package-relative import
    from . import EvalTask, TaskSuite, build_suite
except Exception:  # pragma: no cover - direct execution / partial install
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from fre.envs import EvalTask, TaskSuite, build_suite  # type: ignore

try:  # opensimplex is only required for the random-simplex tasks
    import opensimplex  # type: ignore

    HAS_OPENSIMPLEX = True
except Exception:  # pragma: no cover
    opensimplex = None  # type: ignore
    HAS_OPENSIMPLEX = False


__all__ = [
    "AntMazeTask",
    "AntMazeGoalTask",
    "AntMazeDirectionTask",
    "AntMazeSimplexTask",
    "AntMazePathTask",
    "make_antmaze_goal_tasks",
    "make_antmaze_directional_tasks",
    "make_antmaze_random_simplex_tasks",
    "make_antmaze_path_tasks",
    "make_antmaze_path_center_task",
    "make_antmaze_path_loop_task",
    "make_antmaze_path_edges_task",
    "make_antmaze_all_tasks",
    "make_antmaze_task_suite",
    "antmaze_suite_for_group",
    "antmaze_task_groups",
    "antmaze_goal_locations",
    "antmaze_directions",
    "antmaze_velocity",
    "antmaze_positions",
    "polyline_distance",
    "simplex_noise_field",
    "ANTMAZE_DOMAIN",
    "ANTMAZE_EVAL_EPISODE_LENGTH",
    "ANTMAZE_GOAL_THRESHOLD",
    "ANTMAZE_GOAL_LOCATIONS",
    "ANTMAZE_DIRECTIONS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "ANTMAZE_POSITION_DIMS",
    "ANTMAZE_VELOCITY_DIMS",
    "ANTMAZE_XY_BOUNDS",
    "ANTMAZE_NUM_XY_BINS",
    "ANTMAZE_TASK_GROUPS",
    "ANTMAZE_GOAL_TASK_NAMES",
    "ANTMAZE_DIRECTIONAL_TASK_NAMES",
    "ANTMAZE_SIMPLEX_TASK_NAMES",
    "ANTMAZE_PATH_TASK_NAMES",
    "ANTMAZE_CENTER_PATH",
    "ANTMAZE_LOOP_PATH",
    "ANTMAZE_EDGE_PATH",
    "ANTMAZE_PATH_WIDTH",
    "ANTMAZE_HEIGHT_BONUS",
    "ANTMAZE_VELOCITY_BONUS",
    "ANTMAZE_VELOCITY_SCALE",
]

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
ANTMAZE_DOMAIN = "antmaze"

#: Maximum length of an evaluation trajectory (Addendum: "maximum length of
#: 2000 steps per trajectory"; C.1: "a length of 2000 timesteps").
ANTMAZE_EVAL_EPISODE_LENGTH = 2000

#: C.1: goal considered reached within a distance of 2 of the target position.
ANTMAZE_GOAL_THRESHOLD = 2.0

#: 5 hand-crafted goal-reaching locations, (X, Y) with origin bottom-left.
ANTMAZE_GOAL_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "goal-bottom": (28.0, 0.0),
    "goal-left": (0.0, 15.0),
    "goal-top": (35.0, 24.0),
    "goal-center": (12.0, 24.0),
    "goal-right": (33.0, 16.0),
}

#: Target (unit) velocities of the four enumerated directional tasks.
ANTMAZE_DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "vel_left": (-1.0, 0.0),
    "vel_up": (0.0, 1.0),
    "vel_down": (0.0, -1.0),
    "vel_right": (1.0, 0.0),
}

#: Five fixed opensimplex seeds for ``ant-random-simplex``.
ANTMAZE_SIMPLEX_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: Dimensions of the (X, Y) position in the antmaze observation.
ANTMAZE_POSITION_DIMS: Tuple[int, int] = (0, 1)

#: Dimensions of the (X, Y) linear velocity in the 29-d antmaze observation
#: (qpos = dims 0..14, qvel = dims 15..28).  Consistent with the velocity dims
#: used by the FRE-hint directional prior in :mod:`fre.reward_priors.mixture`.
ANTMAZE_VELOCITY_DIMS: Tuple[int, int] = (15, 16)

#: Continuous extent assumed when inverting the 32-bin X/Y discretization.
#: (Paper does not specify the bounds; see module docstring.)
ANTMAZE_XY_BOUNDS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (0.0, 36.0),
    (0.0, 36.0),
)

#: C.1: "the X and Y coordinates are discretized into 32 bins".
ANTMAZE_NUM_XY_BINS = 32

ANTMAZE_GOAL_TASK_NAMES: Tuple[str, ...] = tuple(ANTMAZE_GOAL_LOCATIONS.keys())
ANTMAZE_DIRECTIONAL_TASK_NAMES: Tuple[str, ...] = tuple(ANTMAZE_DIRECTIONS.keys())
ANTMAZE_SIMPLEX_TASK_NAMES: Tuple[str, ...] = tuple(
    f"simplex-{seed}" for seed in ANTMAZE_SIMPLEX_SEEDS
)
ANTMAZE_PATH_TASK_NAMES: Tuple[str, ...] = ("path-center", "path-loop", "path-edges")

#: Task groups usable with :func:`make_antmaze_task_suite`.
ANTMAZE_TASK_GROUPS: Tuple[str, ...] = (
    "goal-reaching",
    "directional",
    "random-simplex",
    "path-center",
    "path-loop",
    "path-edges",
    "path-all",
    "all",
)

# --------------------------------------------------------------------------
# hand-crafted corridor geometries for the path tasks (paper: "hand-crafted")
# --------------------------------------------------------------------------
# Centre corridors: a horizontal corridor through the middle of the grid plus a
# vertical one crossing it.
ANTMAZE_CENTER_PATH: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
    ((4.0, 18.0), (32.0, 18.0)),
    ((18.0, 4.0), (18.0, 32.0)),
)

# Loop around the grid (closed square ring inset from the border).
ANTMAZE_LOOP_PATH: Tuple[Tuple[float, float], ...] = (
    (12.0, 12.0),
    (24.0, 12.0),
    (24.0, 24.0),
    (12.0, 24.0),
    (12.0, 12.0),
)

# Corridors along the edges of the grid (closed outer ring).
ANTMAZE_EDGE_PATH: Tuple[Tuple[float, float], ...] = (
    (4.0, 4.0),
    (32.0, 4.0),
    (32.0, 32.0),
    (4.0, 32.0),
    (4.0, 4.0),
)

#: Half-width of the corridor reward band (paper: not specified).
ANTMAZE_PATH_WIDTH = 4.0

#: Simplex task bonus weights (paper: "a bonus ... and an additional bonus").
ANTMAZE_HEIGHT_BONUS = 0.5
ANTMAZE_VELOCITY_BONUS = 0.5

#: Scale applied to the raw velocity dot product in the directional tasks.
ANTMAZE_VELOCITY_SCALE = 1.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_2d(observations: Any) -> np.ndarray:
    """Coerce observations to a ``(N, obs_dim)`` float array."""
    arr = np.asarray(observations, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _normalize_bounds(
    bounds: Optional[Sequence[Sequence[float]]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if bounds is None:
        return ANTMAZE_XY_BOUNDS
    b = tuple((float(lo), float(hi)) for lo, hi in bounds)
    if len(b) != 2:
        raise ValueError(f"bounds must provide (x, y) ranges, got {bounds!r}")
    return b  # type: ignore[return-value]


def antmaze_positions(
    observations: Any,
    position_dims: Sequence[int] = ANTMAZE_POSITION_DIMS,
    num_bins: int = ANTMAZE_NUM_XY_BINS,
    bounds: Optional[Sequence[Sequence[float]]] = None,
    discretized: bool = False,
) -> np.ndarray:
    """Return continuous ``(N, 2)`` (X, Y) positions from observations.

    If ``discretized`` is True the two position dimensions are interpreted as
    32-bin indices and converted back to continuous units using the *centres* of
    the bins implied by ``bounds`` (paper: the X/Y coordinates are discretized
    into 32 bins before being consumed by the agents).
    """
    arr = _as_2d(observations)
    if arr.shape[1] <= max(position_dims):
        raise ValueError(
            f"observations have {arr.shape[1]} dims, cannot read position dims "
            f"{tuple(position_dims)}"
        )
    xy = arr[:, list(position_dims)].astype(np.float64)
    if not discretized:
        return xy
    lo_hi = _normalize_bounds(bounds)
    out = np.empty_like(xy)
    for axis in range(2):
        lo, hi = lo_hi[axis]
        width = (hi - lo) / float(num_bins)
        out[:, axis] = lo + (np.clip(xy[:, axis], 0.0, num_bins - 1.0) + 0.5) * width
    return out


def antmaze_velocity(
    observations: Any,
    velocity_dims: Sequence[int] = ANTMAZE_VELOCITY_DIMS,
    scale: float = 1.0,
) -> np.ndarray:
    """Return the ``(N, 2)`` (X, Y) linear velocity of the ant.

    The addendum's directional tasks use "the agent's actual velocity"; the
    discretization applied to the observations only modifies the X/Y position
    dimensions, so the velocity dims are used as-is.
    """
    arr = _as_2d(observations)
    if arr.shape[1] <= max(velocity_dims):
        raise ValueError(
            f"observations have {arr.shape[1]} dims, cannot read velocity dims "
            f"{tuple(velocity_dims)}"
        )
    return arr[:, list(velocity_dims)].astype(np.float64) * float(scale)


def _segments_from_path(path: Sequence[Sequence[float]]):
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError(f"path must be a sequence of (x, y) points, got {path!r}")
    return np.stack([pts[:-1], pts[1:]], axis=1)


def polyline_distance(points: Any, path: Sequence[Sequence[float]]) -> np.ndarray:
    """Minimum Euclidean distance from each point to a polyline ``path``."""
    pts = _as_2d(points)[:, :2]
    segs = _segments_from_path(path)
    best = np.full(len(pts), np.inf, dtype=np.float64)
    for (a, b) in segs:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            d = np.linalg.norm(pts - proj, axis=1)
        best = np.minimum(best, d)
    return best


def polyline_tangent(points: Any, path: Sequence[Sequence[float]]) -> np.ndarray:
    """Unit tangent of the nearest polyline segment for each point."""
    pts = _as_2d(points)[:, :2]
    segs = _segments_from_path(path)
    best_dist = np.full(len(pts), np.inf, dtype=np.float64)
    best_tan = np.zeros((len(pts), 2), dtype=np.float64)
    for (a, b) in segs:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            d = np.linalg.norm(pts - proj, axis=1)
        closer = d < best_dist
        if np.any(closer):
            norm = max(float(np.linalg.norm(ab)), 1e-12)
            best_tan[closer] = ab / norm
            best_dist[closer] = d[closer]
    return best_tan


# --------------------------------------------------------------------------
# opensimplex height/preference field
# --------------------------------------------------------------------------
class _FallbackNoise:
    """Deterministic smooth pseudo-noise used when opensimplex is unavailable.

    Only a stand-in so that the evaluation pipeline is runnable without the
    optional dependency; the paper's tasks use opensimplex noise.
    """

    def __init__(self, seed: int, frequency: float = 1.0):
        self.seed = int(seed)
        self.frequency = float(frequency)
        rng = np.random.default_rng(self.seed)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=(4, 2))
        self.freqs = np.array([[1.0, 0.7], [0.5, 1.3], [1.7, -0.9], [-1.1, 1.5]])
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])

    def noise2(self, x: float, y: float) -> float:
        x = float(x) * self.frequency
        y = float(y) * self.frequency
        total = 0.0
        for (fx, fy), (px, py), w in zip(self.freqs, self.phases, self.weights):
            total += w * np.sin(fx * x + px) * np.cos(fy * y + py)
        return float(np.clip(total, -1.0, 1.0))


def simplex_noise_field(seed: int, frequency: float = 0.15) -> Callable[[Any, Any], np.ndarray]:
    """Return a vectorized 2-D noise sampler for ``seed``.

    Uses :mod:`opensimplex` when installed (matching the paper's
    ``ant-random-simplex`` construction) and a documented deterministic
    fallback otherwise.
    """
    if HAS_OPENSIMPLEX:  # pragma: no cover - depends on optional dependency
        try:
            field = opensimplex.OpenSimplex(seed=int(seed))
        except Exception:
            opensimplex.noise_seed(int(seed))
            field = opensimplex
        noise2array = getattr(opensimplex, "noise2array", None)

        def _noise(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
            xs = np.asarray(xs, dtype=np.float64) * frequency
            ys = np.asarray(ys, dtype=np.float64) * frequency
            if noise2array is not None:
                return np.asarray(noise2array(xs, ys), dtype=np.float64)
            flat = np.array(
                [float(field.noise2(float(a), float(b))) for a, b in zip(xs.ravel(), ys.ravel())],
                dtype=np.float64,
            )
            return flat.reshape(xs.shape)

        return _noise

    fallback = _FallbackNoise(seed, frequency=frequency)
    return lambda xs, ys: np.array(
        [fallback.noise2(a, b) for a, b in zip(np.ravel(xs), np.ravel(ys))],
        dtype=np.float64,
    ).reshape(np.shape(xs))


# --------------------------------------------------------------------------
# task classes
# --------------------------------------------------------------------------
class AntMazeTask(EvalTask):
    """Base class for AntMaze evaluation tasks.

    Subclasses :class:`~fre.envs.EvalTask` (so ``reward``, ``__call__``,
    ``success`` and ``describe`` are inherited) but installs attributes directly
    instead of going through the dataclass constructor.
    """

    def __init__(
        self,
        name: str,
        reward_fn: Callable[[Any], np.ndarray],
        domain: str = ANTMAZE_DOMAIN,
        task_group: str = ANTMAZE_DOMAIN,
        is_goal_task: bool = False,
        goal: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        self.name = name
        self.reward_fn = reward_fn
        self.domain = domain
        self.task_group = task_group
        self.is_goal_task = bool(is_goal_task)
        self.goal = None if goal is None else np.asarray(goal, dtype=np.float64)
        self.threshold = threshold
        self.eval_episode_length = int(eval_episode_length)
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.metadata: Dict[str, Any] = dict(metadata or {})
        for key, value in extra.items():
            setattr(self, key, value)

    # -- convenience ------------------------------------------------------
    def rewards(self, observations: Any) -> np.ndarray:
        """Alias of :meth:`reward`."""
        return np.asarray(self.reward(observations), dtype=np.float64).reshape(-1)

    def copy_with_name(self, name: str) -> "AntMazeTask":
        self.name = name
        return self

    def scores_bounds(self) -> Tuple[float, float]:
        """Per-episode return bounds used for the 0-100 normalisation."""
        bounds = self.metadata.get("score_bounds")
        if bounds is not None:
            return float(bounds[0]), float(bounds[1])
        lo = self.reward_min if self.reward_min is not None else 0.0
        hi = self.reward_max if self.reward_max is not None else 0.0
        return lo * self.eval_episode_length, hi * self.eval_episode_length

    def describe(self) -> Dict[str, Any]:  # pragma: no cover - thin wrapper
        info: Dict[str, Any] = {
            "name": self.name,
            "domain": self.domain,
            "task_group": self.task_group,
            "is_goal_task": self.is_goal_task,
            "eval_episode_length": self.eval_episode_length,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
        }
        if self.goal is not None:
            info["goal"] = self.goal.tolist()
        if self.threshold is not None:
            info["threshold"] = float(self.threshold)
        if self.metadata:
            info["metadata"] = dict(self.metadata)
        return info


class AntMazeGoalTask(AntMazeTask):
    """Goal-reaching reward: ``-1`` until the goal is reached, ``0`` afterwards.

    Source (Addendum): "The reward is set to -1 for every timestep that the goal
    is not achieved."  Source (C.1): "considers the goal reached if an agent
    reaches within a distance of 2 with the target position."
    """

    def __init__(
        self,
        name: str,
        goal: Sequence[float],
        threshold: float = ANTMAZE_GOAL_THRESHOLD,
        position_dims: Sequence[int] = ANTMAZE_POSITION_DIMS,
        num_bins: int = ANTMAZE_NUM_XY_BINS,
        bounds: Optional[Sequence[Sequence[float]]] = None,
        discretized: bool = True,
        reward_unreached: float = -1.0,
        reward_reached: float = 0.0,
        terminate_on_success: bool = True,
        eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
        task_group: str = "goal-reaching",
        **extra: Any,
    ) -> None:
        self._goal_xy = np.asarray(goal, dtype=np.float64).reshape(2)
        self._position_dims = tuple(position_dims)
        self._num_bins = int(num_bins)
        self._bounds = _normalize_bounds(bounds)
        self._discretized = bool(discretized)
        self._reward_unreached = float(reward_unreached)
        self._reward_reached = float(reward_reached)
        self.terminate_on_success = bool(terminate_on_success)
        metadata = dict(extra.pop("metadata", {}) or {})
        metadata.update(
            {
                "goal_xy": self._goal_xy.tolist(),
                "discretized": self._discretized,
                "bounds": [list(b) for b in self._bounds],
                # || return in [-L, 0] : 100 * (return + L) / L
                "score_bounds": (
                    float(reward_unreached) * int(eval_episode_length),
                    float(reward_reached) * int(eval_episode_length),
                ),
            }
        )
        super().__init__(
            name=name,
            reward_fn=self.reward,
            task_group=task_group,
            is_goal_task=True,
            goal=self._goal_xy,
            threshold=float(threshold),
            eval_episode_length=eval_episode_length,
            reward_min=float(reward_unreached),
            reward_max=float(reward_reached),
            metadata=metadata,
            **extra,
        )

    # -- geometry ---------------------------------------------------------
    def positions(self, observations: Any) -> np.ndarray:
        return antmaze_positions(
            observations,
            position_dims=self._position_dims,
            num_bins=self._num_bins,
            bounds=self._bounds,
            discretized=self._discretized,
        )

    def distances(self, observations: Any) -> np.ndarray:
        xy = self.positions(observations)
        return np.linalg.norm(xy - self._goal_xy[None, :], axis=1)

    # -- reward -----------------------------------------------------------
    def reward(self, observations: Any) -> np.ndarray:
        reached = self.distances(observations) <= float(self.threshold)
        return np.where(reached, self._reward_reached, self._reward_unreached)

    def done(self, observations: Any) -> np.ndarray:
        if not self.terminate_on_success:
            return np.zeros(self.distances(observations).shape[0], dtype=bool)
        return self.distances(observations) <= float(self.threshold)


class AntMazeDirectionTask(AntMazeTask):
    """Directional reward: dot product of the ant's velocity with a direction.

    Source (Addendum): "The reward function checks the agent's actual velocity
    and grants higher reward the closer it is to the target velocity, using a
    simple dot product."
    """

    def __init__(
        self,
        name: str,
        direction: Sequence[float],
        velocity_dims: Sequence[int] = ANTMAZE_VELOCITY_DIMS,
        velocity_scale: float = ANTMAZE_VELOCITY_SCALE,
        reward_scale: float = 1.0,
        clip: Optional[float] = None,
        eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
        task_group: str = "directional",
        **extra: Any,
    ) -> None:
        direction = np.asarray(direction, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(direction))
        self.direction = direction / norm if norm > 0 else direction
        self._velocity_dims = tuple(velocity_dims)
        self.velocity_scale = float(velocity_scale)
        self.reward_scale = float(reward_scale)
        self._clip = None if clip is None else float(clip)
        per_step_max = abs(self.velocity_scale) * abs(self.reward_scale)
        metadata = dict(extra.pop("metadata", {}) or {})
        metadata.update(
            {
                "direction": self.direction.tolist(),
                "velocity_dims": list(self._velocity_dims),
                "score_bounds": (
                    -per_step_max * int(eval_episode_length),
                    per_step_max * int(eval_episode_length),
                ),
            }
        )
        super().__init__(
            name=name,
            reward_fn=self.reward,
            task_group=task_group,
            is_goal_task=False,
            goal=self.direction,
            threshold=None,
            eval_episode_length=eval_episode_length,
            reward_min=-per_step_max,
            reward_max=per_step_max,
            metadata=metadata,
            **extra,
        )

    def velocity(self, observations: Any) -> np.ndarray:
        return antmaze_velocity(observations, self._velocity_dims, scale=self.velocity_scale)

    def reward(self, observations: Any) -> np.ndarray:
        vel = self.velocity(observations)
        out = self.reward_scale * (vel @ self.direction[None, :].T).reshape(-1)
        if self._clip is not None:
            out = np.clip(out, -self._clip, self._clip)
        return out


class AntMazeSimplexTask(AntMazeTask):
    """Opensimplex "height map" + preferred-velocity reward (5 seeded tasks).

    Source (Addendum): "The agent gets baseline negative reward (-1) at each
    step, a bonus if it stands in higher 'height' regions, and an additional
    bonus for moving in the local 'preferred' velocity direction indicated by
    the noise field."
    """

    def __init__(
        self,
        name: str,
        seed: int,
        frequency: float = 0.15,
        height_bonus: float = ANTMAZE_HEIGHT_BONUS,
        velocity_bonus: float = ANTMAZE_VELOCITY_BONUS,
        baseline: float = -1.0,
        position_dims: Sequence[int] = ANTMAZE_POSITION_DIMS,
        velocity_dims: Sequence[int] = ANTMAZE_VELOCITY_DIMS,
        velocity_scale: float = 1.0,
        num_bins: int = ANTMAZE_NUM_XY_BINS,
        bounds: Optional[Sequence[Sequence[float]]] = None,
        discretized: bool = True,
        eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
        task_group: str = "random-simplex",
        **extra: Any,
    ) -> None:
        self.seed = int(seed)
        self.frequency = float(frequency)
        self.height_bonus = float(height_bonus)
        self.velocity_bonus = float(velocity_bonus)
        self.baseline = float(baseline)
        self._position_dims = tuple(position_dims)
        self._velocity_dims = tuple(velocity_dims)
        self._velocity_scale = float(velocity_scale)
        self._num_bins = int(num_bins)
        self._bounds = _normalize_bounds(bounds)
        self._discretized = bool(discretized)
        self._noise = simplex_noise_field(self.seed, frequency=self.frequency)
        metadata = dict(extra.pop("metadata", {}) or {})
        metadata.update(
            {
                "seed": self.seed,
                "height_bonus": self.height_bonus,
                "velocity_bonus": self.velocity_bonus,
                "baseline": self.baseline,
                "has_opensimplex": bool(HAS_OPENSIMPLEX),
                # baseline -1 per step is the floor; bonuses raise it to <= 0
                "score_bounds": (
                    float(baseline) * int(eval_episode_length),
                    (float(baseline) + abs(self.height_bonus) + abs(self.velocity_bonus))
                    * int(eval_episode_length),
                ),
            }
        )
        super().__init__(
            name=name,
            reward_fn=self.reward,
            task_group=task_group,
            is_goal_task=False,
            goal=None,
            threshold=None,
            eval_episode_length=eval_episode_length,
            reward_min=float(baseline),
            reward_max=float(baseline) + abs(self.height_bonus) + abs(self.velocity_bonus),
            metadata=metadata,
            **extra,
        )

    # -- field ------------------------------------------------------------
    def positions(self, observations: Any) -> np.ndarray:
        return antmaze_positions(
            observations,
            position_dims=self._position_dims,
            num_bins=self._num_bins,
            bounds=self._bounds,
            discretized=self._discretized,
        )

    def heights(self, observations: Any) -> np.ndarray:
        xy = self.positions(observations)
        return np.asarray(self._noise(xy[:, 0], xy[:, 1]), dtype=np.float64).reshape(-1)

    def preferred_velocity(self, observations: Any, eps: float = 1e-3) -> np.ndarray:
        """Gradient of the noise field, i.e. the local preferred direction."""
        xy = self.positions(observations)
        hx = (self._noise(xy[:, 0] + eps, xy[:, 1]) - self._noise(xy[:, 0] - eps, xy[:, 1])) / (
            2.0 * eps
        )
        hy = (self._noise(xy[:, 0], xy[:, 1] + eps) - self._noise(xy[:, 0], xy[:, 1] - eps)) / (
            2.0 * eps
        )
        pref = np.stack([hx, hy], axis=1)
        norm = np.linalg.norm(pref, axis=1, keepdims=True)
        return pref / np.maximum(norm, 1e-12)

    # -- reward -----------------------------------------------------------
    def reward(self, observations: Any) -> np.ndarray:
        height = np.clip(self.heights(observations), -1.0, 1.0)
        height_term = self.height_bonus * np.clip(height, 0.0, 1.0)
        vel = antmaze_velocity(observations, self._velocity_dims, scale=self._velocity_scale)
        pref = self.preferred_velocity(observations)
        alignment = np.clip(np.sum(vel * pref, axis=1), -1.0, 1.0)
        velocity_term = self.velocity_bonus * alignment
        return self.baseline + height_term + velocity_term


class AntMazePathTask(AntMazeTask):
    """Corridor-following reward for the three ``ant-path-*`` tasks.

    Source (Addendum): "The ``ant-path-center``, ``ant-path-loop`` and
    ``ant-path-edges`` are simply reward functions that reward the agent for
    moving along hand-crafted corridors placed in the center of the grid, for
    moving in a hand-crafted loop around the grid, and for moving along the
    edges of the grid, respectively."

    The reward is dense (the paper does not give a per-step value): the agent
    receives ``proximity`` in ``[0, 1]`` based on its distance to the corridor
    centreline, weighted by how well its velocity aligns with the corridor
    direction.  ``proximity = clip(1 - dist / width, 0, 1)`` and the returned
    reward is ``reward_scale * proximity * (1 - alignment_weight +
    alignment_weight * (0.5 + 0.5 * |cos|))`` so that standing on the corridor
    without moving still yields a positive reward and moving along it yields the
    maximum.
    """

    def __init__(
        self,
        name: str,
        path: Sequence[Sequence[float]],
        width: float = ANTMAZE_PATH_WIDTH,
        reward_scale: float = 1.0,
        alignment_weight: float = 0.5,
        position_dims: Sequence[int] = ANTMAZE_POSITION_DIMS,
        velocity_dims: Sequence[int] = ANTMAZE_VELOCITY_DIMS,
        num_bins: int = ANTMAZE_NUM_XY_BINS,
        bounds: Optional[Sequence[Sequence[float]]] = None,
        discretized: bool = True,
        task_group: str = "path",
        eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
        **extra: Any,
    ) -> None:
        self.path = tuple(tuple(float(v) for v in p) for p in path)
        self.width = float(width)
        self.reward_scale = float(reward_scale)
        self.alignment_weight = float(np.clip(alignment_weight, 0.0, 1.0))
        self._position_dims = tuple(position_dims)
        self._velocity_dims = tuple(velocity_dims)
        self._num_bins = int(num_bins)
        self._bounds = _normalize_bounds(bounds)
        self._discretized = bool(discretized)
        self._segments = _segments_from_path(self.path)
        metadata = dict(extra.pop("metadata", {}) or {})
        metadata.update(
            {
                "path": [list(p) for p in self.path],
                "path_width": self.width,
                "alignment_weight": self.alignment_weight,
                "score_bounds": (
                    0.0,
                    abs(self.reward_scale) * int(eval_episode_length),
                ),
            }
        )
        super().__init__(
            name=name,
            reward_fn=self.reward,
            task_group=task_group,
            is_goal_task=False,
            goal=None,
            threshold=None,
            eval_episode_length=eval_episode_length,
            reward_min=0.0,
            reward_max=abs(self.reward_scale),
            metadata=metadata,
            **extra,
        )

    def positions(self, observations: Any) -> np.ndarray:
        return antmaze_positions(
            observations,
            position_dims=self._position_dims,
            num_bins=self._num_bins,
            bounds=self._bounds,
            discretized=self._discretized,
        )

    def distances(self, observations: Any) -> np.ndarray:
        return polyline_distance(self.positions(observations), self.path)

    def reward(self, observations: Any) -> np.ndarray:
        xy = self.positions(observations)
        dist = polyline_distance(xy, self.path)
        proximity = np.clip(1.0 - dist / max(self.width, 1e-6), 0.0, 1.0)
        if self.alignment_weight > 0.0:
            tangent = polyline_tangent(xy, self.path)
            vel = antmaze_velocity(observations, self._velocity_dims)
            speed = np.linalg.norm(vel, axis=1)
            cos = np.zeros_like(speed)
            moving = speed > 1e-8
            cos[moving] = (
                np.sum(vel[moving] * tangent[moving], axis=1) / speed[moving]
            )
            # either travel direction along the corridor counts
            alignment = 0.5 + 0.5 * np.abs(np.clip(cos, -1.0, 1.0))
        else:
            alignment = np.ones_like(proximity)
        weight = (1.0 - self.alignment_weight) + self.alignment_weight * alignment
        return self.reward_scale * proximity * weight


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------
def _finish(tasks: Iterable[AntMazeTask]) -> List[AntMazeTask]:
    return [t for t in tasks]


def make_antmaze_goal_tasks(
    threshold: float = ANTMAZE_GOAL_THRESHOLD,
    locations: Optional[Dict[str, Sequence[float]]] = None,
    discretized: bool = True,
    num_bins: int = ANTMAZE_NUM_XY_BINS,
    bounds: Optional[Sequence[Sequence[float]]] = None,
    eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
    **kwargs: Any,
) -> List[AntMazeGoalTask]:
    """The 5 hand-crafted goal-reaching tasks (``ant-goal-reaching``)."""
    locations = dict(locations or ANTMAZE_GOAL_LOCATIONS)
    return [
        AntMazeGoalTask(
            name=name,
            goal=goal,
            threshold=threshold,
            discretized=discretized,
            num_bins=num_bins,
            bounds=bounds,
            eval_episode_length=eval_episode_length,
            **kwargs,
        )
        for name, goal in locations.items()
    ]


def make_antmaze_directional_tasks(
    directions: Optional[Dict[str, Sequence[float]]] = None,
    velocity_scale: float = ANTMAZE_VELOCITY_SCALE,
    eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
    **kwargs: Any,
) -> List[AntMazeDirectionTask]:
    """The 4 directional tasks (``ant-directional``)."""
    directions = dict(directions or ANTMAZE_DIRECTIONS)
    return [
        AntMazeDirectionTask(
            name=name,
            direction=direction,
            velocity_scale=velocity_scale,
            eval_episode_length=eval_episode_length,
            **kwargs,
        )
        for name, direction in directions.items()
    ]


def make_antmaze_random_simplex_tasks(
    seeds: Sequence[int] = ANTMAZE_SIMPLEX_SEEDS,
    frequency: float = 0.15,
    height_bonus: float = ANTMAZE_HEIGHT_BONUS,
    velocity_bonus: float = ANTMAZE_VELOCITY_BONUS,
    discretized: bool = True,
    num_bins: int = ANTMAZE_NUM_XY_BINS,
    bounds: Optional[Sequence[Sequence[float]]] = None,
    eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
    **kwargs: Any,
) -> List[AntMazeSimplexTask]:
    """The 5 seeded opensimplex tasks (``ant-random-simplex``)."""
    return [
        AntMazeSimplexTask(
            name=f"simplex-{int(seed)}",
            seed=int(seed),
            frequency=frequency,
            height_bonus=height_bonus,
            velocity_bonus=velocity_bonus,
            discretized=discretized,
            num_bins=num_bins,
            bounds=bounds,
            eval_episode_length=eval_episode_length,
            **kwargs,
        )
        for seed in seeds
    ]


def make_antmaze_path_center_task(**kwargs: Any) -> AntMazePathTask:
    kwargs.setdefault("task_group", "path-center")
    return AntMazePathTask(
        name="path-center", path=ANTMAZE_CENTER_PATH, **kwargs
    )


def make_antmaze_path_loop_task(**kwargs: Any) -> AntMazePathTask:
    kwargs.setdefault("task_group", "path-loop")
    return AntMazePathTask(name="path-loop", path=ANTMAZE_LOOP_PATH, **kwargs)


def make_antmaze_path_edges_task(**kwargs: Any) -> AntMazePathTask:
    kwargs.setdefault("task_group", "path-edges")
    return AntMazePathTask(name="path-edges", path=ANTMAZE_EDGE_PATH, **kwargs)


def make_antmaze_path_tasks(**kwargs: Any) -> List[AntMazePathTask]:
    """The 3 hand-crafted corridor tasks (``ant-path-center/loop/edges``)."""
    return [
        make_antmaze_path_center_task(**kwargs),
        make_antmaze_path_loop_task(**kwargs),
        make_antmaze_path_edges_task(**kwargs),
    ]


def make_antmaze_all_tasks(**kwargs: Any) -> List[AntMazeTask]:
    """Every AntMaze evaluation task (18 tasks: 5 + 4 + 5 + 3 + 1)."""
    tasks: List[AntMazeTask] = []
    tasks.extend(make_antmaze_goal_tasks(**kwargs))
    tasks.extend(make_antmaze_directional_tasks(**kwargs))
    tasks.extend(make_antmaze_random_simplex_tasks(**kwargs))
    tasks.extend(make_antmaze_path_tasks(**kwargs))
    return tasks


# --------------------------------------------------------------------------
# task suites
# --------------------------------------------------------------------------
_GROUP_ALIASES: Dict[str, str] = {
    "goals": "goal-reaching",
    "goal": "goal-reaching",
    "ant-goal-reaching": "goal-reaching",
    "antmaze-goal-reaching": "goal-reaching",
    "ant-directional": "directional",
    "antmaze-directional": "directional",
    "velocities": "directional",
    "simplex": "random-simplex",
    "ant-random-simplex": "random-simplex",
    "antmaze-random-simplex": "random-simplex",
    "ant-path-center": "path-center",
    "ant-path-loop": "path-loop",
    "ant-path-edges": "path-edges",
    "paths": "path-all",
    "path": "path-all",
    "ant-path-all": "path-all",
    "antmaze": "all",
    "antmaze-all": "all",
    "ant-all": "all",
}


def antmaze_task_groups() -> Tuple[str, ...]:
    """Canonical AntMaze task-group names."""
    return ANTMAZE_TASK_GROUPS


def antmaze_goal_locations() -> Dict[str, Tuple[float, float]]:
    return dict(ANTMAZE_GOAL_LOCATIONS)


def antmaze_directions() -> Dict[str, Tuple[float, float]]:
    return dict(ANTMAZE_DIRECTIONS)


def _resolve_group(group: str) -> str:
    key = str(group).strip().lower()
    key = _GROUP_ALIASES.get(key, key)
    if key not in ANTMAZE_TASK_GROUPS:
        raise ValueError(
            f"unknown AntMaze task group {group!r}; expected one of "
            f"{ANTMAZE_TASK_GROUPS} (aliases: {sorted(_GROUP_ALIASES)})"
        )
    return key


def make_antmaze_task_suite(
    group: str = "all",
    eval_episode_length: int = ANTMAZE_EVAL_EPISODE_LENGTH,
    **kwargs: Any,
) -> TaskSuite:
    """Build a :class:`~fre.envs.TaskSuite` for an AntMaze task group.

    Groups: ``goal-reaching``, ``directional``, ``random-simplex``,
    ``path-center``, ``path-loop``, ``path-edges``, ``path-all``, ``all``
    (domain-prefixed aliases such as ``ant-goal-reaching`` are accepted).
    """
    key = _resolve_group(group)
    kwargs.setdefault("eval_episode_length", eval_episode_length)

    if key == "goal-reaching":
        tasks = make_antmaze_goal_tasks(**kwargs)
        suite_name = "ant-goal-reaching"
    elif key == "directional":
        tasks = make_antmaze_directional_tasks(**kwargs)
        suite_name = "ant-directional"
    elif key == "random-simplex":
        tasks = make_antmaze_random_simplex_tasks(**kwargs)
        suite_name = "ant-random-simplex"
    elif key == "path-center":
        tasks = [make_antmaze_path_center_task(**kwargs)]
        suite_name = "ant-path-center"
    elif key == "path-loop":
        tasks = [make_antmaze_path_loop_task(**kwargs)]
        suite_name = "ant-path-loop"
    elif key == "path-edges":
        tasks = [make_antmaze_path_edges_task(**kwargs)]
        suite_name = "ant-path-edges"
    elif key == "path-all":
        tasks = make_antmaze_path_tasks(**kwargs)
        suite_name = "ant-path-all"
    else:  # "all"
        tasks = make_antmaze_all_tasks(**kwargs)
        suite_name = "antmaze-all"

    aggregate = "mean" if len(tasks) > 1 else None
    metadata = {
        "group": key,
        "num_tasks": len(tasks),
        "eval_episode_length": int(eval_episode_length),
        "normalization": "0-100 via task score_bounds",
    }
    try:
        return build_suite(
            name=suite_name,
            tasks=tasks,
            eval_episode_length=int(eval_episode_length),
            aggregate=aggregate,
            domain=ANTMAZE_DOMAIN,
            metadata=metadata,
        )
    except TypeError:  # pragma: no cover - defensive against signature drift
        return TaskSuite(
            name=suite_name,
            tasks=tasks,
            eval_episode_length=int(eval_episode_length),
            aggregate=aggregate,
            domain=ANTMAZE_DOMAIN,
            metadata=metadata,
        )


def antmaze_suite_for_group(group: str = "all", **kwargs: Any) -> TaskSuite:
    """Alias of :func:`make_antmaze_task_suite` (used by the eval harness)."""
    return make_antmaze_task_suite(group=group, **kwargs)


# --------------------------------------------------------------------------
# smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    obs_dim = 29
    obs = np.zeros((3, obs_dim))
    obs[:, 0] = [28.0, 0.0, 35.0]
    obs[:, 1] = [0.0, 15.0, 24.0]
    obs[:, 15] = [0.5, -0.2, 1.0]
    obs[:, 16] = [0.1, 0.4, -0.3]

    for task in make_antmaze_all_tasks():
        r = task.rewards(obs)
        assert r.shape == (3,), (task.name, r.shape)
        sb = task.scores_bounds()
        print(f"{task.name:>16s} reward={np.round(r, 3)} bounds={np.round(sb, 1)}")

    suite = make_antmaze_task_suite("all")
    print(f"\nsuite {suite.name}: {len(suite)} tasks -> {suite.names}")
    print("opensimplex available:", HAS_OPENSIMPLEX)
