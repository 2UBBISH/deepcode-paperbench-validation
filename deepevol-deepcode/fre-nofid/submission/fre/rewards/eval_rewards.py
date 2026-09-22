"""Ground-truth zero-shot evaluation reward functions for FRE (Section 4.4 / Sec 7 of the plan).

This module turns every FRE *evaluation task* into a :class:`~fre.rewards.base.RewardFunction`
(the same Markovian interface ``eta: S -> [-1, 1]`` used by the unsupervised reward prior), so the
evaluation harness can treat prior-sampled and ground-truth reward functions identically.

Task suites (matching the reproduction plan):

* **AntMaze** (``antmaze-large-diverse-v2``, 29-d obs, 2000-step episodes)
    - 5 goal-reaching tasks           -> goals ``[(28,0),(0,15),(35,24),(12,24),(33,16)]`` (XY bins, dist <= 2)
    - 4 directional velocity tasks    -> dot product of XY velocity with ``{(-1,0),(0,1),(0,-1),(1,0)}``
    - 5 random-simplex tasks          -> ``opensimplex`` noise fields, seeds 1..5
    - 3 corridor tasks                -> ``path-center`` / ``path-loop`` / ``path-edges``
* **ExORL** (RND ``walker`` / ``cheetah`` datasets, 1000-step episodes)
    - 5 goal tasks per domain         -> Euclidean distance in *physics* space < 0.1
    - velocity tasks                  -> cheetah thresholds ``(10, 1)``, walker ``(0.1, 1, 4, 8)``
* **Kitchen** (``kitchen-complete-v0``)
    - 7 standard D4RL subtasks, sparse completion rewards read from the last 7 observation dims.

Normalisation
-------------
Every task carries a :class:`TaskScoring`, which converts raw episode quantities into the FRE
normalised score in ``[0, 100]`` (success rate, mean reward mapped from ``[-1, 1]``, or a
min/max-normalised return).  Aggregation helpers compute the per-family and per-domain means that
feed Table 1 / Figure 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# ----------------------------------------------------------------------------------------------
# base class import (package, absolute, or standalone)
# ----------------------------------------------------------------------------------------------
try:  # pragma: no cover - import shim
    from .base import RewardFunction, get_observations, to_numpy
except Exception:  # pragma: no cover
    try:
        from fre.rewards.base import RewardFunction, get_observations, to_numpy  # type: ignore
    except Exception:
        try:
            from base import RewardFunction, get_observations, to_numpy  # type: ignore
        except Exception:  # last-resort stub so the module stays importable

            class RewardFunction:  # type: ignore
                def __init__(self, state_dim=None, name=None, clip=1.0, **kwargs):
                    self.state_dim = state_dim
                    self.name = name
                    self.clip = clip

            def get_observations(source):  # type: ignore
                return np.asarray(source, dtype=np.float32)

            def to_numpy(x, dtype=np.float32):  # type: ignore
                return np.asarray(x, dtype=dtype)


# ==============================================================================================
# Constants
# ==============================================================================================
EVAL_REWARD_FAMILY = "eval_rewards"

# ---- AntMaze ---------------------------------------------------------------------------------
ANTMAZE_DATASET = "antmaze-large-diverse-v2"
ANTMAZE_OBS_DIM = 29
ANTMAZE_ACTION_DIM = 8
ANTMAZE_XY_DIMS = (0, 1)
ANTMAZE_VEL_XY_DIMS = (15, 16)  # qvel[0:2] inside the 29-d AntMaze observation
ANTMAZE_MAX_STEPS = 2000
ANTMAZE_XY_BINS = 32
ANTMAZE_XY_EXTENT = 36.0
ANTMAZE_GOAL_THRESHOLD = 2.0  # in *bins*, 32x32 grid -> matches the plan's "dist <= 2"

#: 5 goal-reaching goals (XY *bin* coordinates), from the reproduction plan.
ANTMAZE_GOAL_TASKS: Tuple[Tuple[int, int], ...] = (
    (28, 0),
    (0, 15),
    (35, 24),
    (12, 24),
    (33, 16),
)
#: 4 directional (dot-product velocity) tasks.
ANTMAZE_DIRECTIONAL_TASKS: Tuple[Tuple[float, float], ...] = (
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
)
#: 5 opensimplex seeds for the random-simplex reward family.
ANTMAZE_SIMPLEX_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
#: The 3 corridor tasks.
ANTMAZE_PATH_TASKS: Tuple[str, ...] = ("path-center", "path-loop", "path-edges")

#: Approximate corridor routes in metres (documented approximations of the paper's corridors).
ANTMAZE_PATH_ROUTES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "path-center": ((18.0, 3.0), (18.0, 33.0)),
    "path-loop": ((18.0, 3.0), (33.0, 18.0), (18.0, 33.0), (3.0, 18.0), (18.0, 3.0)),
    "path-edges": ((2.0, 2.0), (34.0, 2.0), (34.0, 34.0), (2.0, 34.0), (2.0, 2.0)),
}

# ---- ExORL -----------------------------------------------------------------------------------
EXORL_DOMAINS: Tuple[str, ...] = ("walker", "cheetah")
EXORL_MAX_STEPS = 1000
EXORL_GOAL_THRESHOLD = 0.1
EXORL_NUM_GOAL_STATES = 5
#: Paper velocity thresholds ("cheetah 10/1, walker 0.1/1/4/8").
EXORL_VELOCITY_THRESHOLDS: Dict[str, Tuple[float, ...]] = {
    "cheetah": (10.0, 1.0),
    "walker": (0.1, 1.0, 4.0, 8.0),
}
#: Physics layout when appended *after* the raw observation (index within the encoder observation).
EXORL_PHYSICS_DIMS: Dict[str, Tuple[int, ...]] = {
    # horizontal_velocity (dim 0), torso_upright (1), torso_height (2)
    "walker": (0, 1, 2),
    # speed (dim 0)
    "cheetah": (0,),
}
EXORL_RAW_OBS_DIM: Dict[str, int] = {"walker": 24, "cheetah": 17}

# ---- Kitchen ---------------------------------------------------------------------------------
KITCHEN_DATASET = "kitchen-complete-v0"
KITCHEN_OBS_DIM = 60
KITCHEN_NUM_SUBTASKS = 7
KITCHEN_MAX_STEPS = 1000
KITCHEN_TASKS: Tuple[str, ...] = (
    "microwave",
    "kettle",
    "slide",
    "hinge",
    "light",
    "bottom_burner",
    "top_burner",
)
KITCHEN_FLAG_OFFSET = -7  # completion flags are the last 7 observation dims
KITCHEN_FLAG_THRESHOLD = 0.5

# ---- Scoring modes ---------------------------------------------------------------------------
SCORE_SUCCESS_RATE = "success_rate"
SCORE_MEAN_REWARD = "mean_reward"
SCORE_NORMALIZED_RETURN = "normalized_return"
SCORE_MODES = (SCORE_SUCCESS_RATE, SCORE_MEAN_REWARD, SCORE_NORMALIZED_RETURN)

_EPS = 1e-8

__all__ = [
    # constants
    "EVAL_REWARD_FAMILY",
    "ANTMAZE_GOAL_TASKS",
    "ANTMAZE_DIRECTIONAL_TASKS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "ANTMAZE_PATH_TASKS",
    "ANTMAZE_PATH_ROUTES",
    "ANTMAZE_XY_DIMS",
    "ANTMAZE_VEL_XY_DIMS",
    "EXORL_VELOCITY_THRESHOLDS",
    "EXORL_PHYSICS_DIMS",
    "KITCHEN_TASKS",
    "SCORE_SUCCESS_RATE",
    "SCORE_MEAN_REWARD",
    "SCORE_NORMALIZED_RETURN",
    # rewards
    "AntMazeGoalReward",
    "AntMazeDirectionalReward",
    "AntMazeSimplexReward",
    "AntMazePathReward",
    "ExORLGoalReward",
    "ExORLVelocityReward",
    "KitchenSubtaskReward",
    # task specs
    "TaskScoring",
    "EvalTask",
    "antmaze_eval_tasks",
    "exorl_eval_tasks",
    "kitchen_eval_tasks",
    "all_eval_tasks",
    "task_groups",
    "tasks_for_group",
    # scoring helpers
    "normalized_return",
    "success_rate_score",
    "mean_reward_score",
    "score_episodes",
    "aggregate_scores",
    # geometry helpers
    "antmaze_xy",
    "antmaze_velocity",
    "discretize_xy",
    "bin_to_xy",
    "bin_distance",
    "polylines",
]


# ==============================================================================================
# Geometry helpers
# ==============================================================================================
def _as_states(states: Any) -> np.ndarray:
    """Return a ``(..., D)`` float64 numpy view of ``states`` (torch tensors accepted)."""
    arr = to_numpy(states, dtype=np.float64) if not isinstance(states, np.ndarray) else states.astype(np.float64)
    return arr


def antmaze_xy(states: Any, xy_dims: Sequence[int] = ANTMAZE_XY_DIMS) -> np.ndarray:
    """Extract ``(..., 2)`` XY positions from AntMaze observations."""
    arr = _as_states(states)
    return np.ascontiguousarray(arr[..., list(xy_dims)])


def antmaze_velocity(states: Any, vel_dims: Sequence[int] = ANTMAZE_VEL_XY_DIMS) -> np.ndarray:
    """Extract ``(..., 2)`` XY joint velocities from AntMaze observations."""
    arr = _as_states(states)
    return np.ascontiguousarray(arr[..., list(vel_dims)])


def discretize_xy(xy: Any, extent: float = ANTMAZE_XY_EXTENT, num_bins: int = ANTMAZE_XY_BINS) -> np.ndarray:
    """Discretise continuous XY into ``num_bins`` bins per axis (matching the AntMaze wrapper)."""
    xy = np.asarray(xy, dtype=np.float64)
    bins = np.floor(xy / float(extent) * float(num_bins))
    bins = np.clip(bins, 0, num_bins - 1)
    return bins.astype(np.int64)


def bin_to_xy(bins: Any, extent: float = ANTMAZE_XY_EXTENT, num_bins: int = ANTMAZE_XY_BINS) -> np.ndarray:
    """Centres of the given XY bins."""
    bins = np.asarray(bins, dtype=np.float64)
    return (bins + 0.5) * (float(extent) / float(num_bins))


def bin_distance(a: Any, b: Any) -> np.ndarray:
    """Euclidean distance between integer XY bins."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b, axis=-1)


def polylines(route: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a polyline route into ``(starts, ends)`` segment arrays of shape ``(M, 2)``."""
    pts = np.asarray(route, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("route must be a sequence of >= 2 XY points")
    return pts[:-1], pts[1:]


def _distance_to_route(xy: np.ndarray, route: Sequence[Sequence[float]]) -> np.ndarray:
    """Minimum Euclidean distance from ``xy`` ``(..., 2)`` to a polyline route."""
    starts, ends = polylines(route)
    flat = xy.reshape(-1, 2)
    seg = ends - starts  # (M,2)
    seg_len2 = np.sum(seg * seg, axis=-1)  # (M,)
    # (N,1,2) - (1,M,2)
    diff = flat[:, None, :] - starts[None, :, :]
    t = np.sum(diff * seg[None, :, :], axis=-1) / np.maximum(seg_len2[None, :], _EPS)
    t = np.clip(t, 0.0, 1.0)
    proj = starts[None, :, :] + t[..., None] * seg[None, :, :]
    dist = np.linalg.norm(flat[:, None, :] - proj, axis=-1)  # (N, M)
    return np.min(dist, axis=-1).reshape(xy.shape[:-1])


# ==============================================================================================
# Simple noise field (opensimplex with deterministic Fourier fallback)
# ==============================================================================================
def _opensimplex_field(xy: np.ndarray, seed: int, scale: float = 1.0) -> np.ndarray:
    """2-D ``opensimplex`` noise sampled at ``xy`` (falls back to a deterministic Fourier field)."""
    try:  # pragma: no cover - depends on optional dependency
        from opensimplex import OpenSimplex  # type: ignore

        gen = OpenSimplex(seed=int(seed))
        flat = xy.reshape(-1, 2) / max(scale, _EPS)
        vals = np.array([gen.noise2d(float(x), float(y)) for x, y in flat], dtype=np.float64)
        return vals.reshape(xy.shape[:-1])
    except Exception:
        # Deterministic Fourier fallback (used when opensimplex is unavailable).
        rng = np.random.default_rng(1234 + int(seed))
        freqs = rng.uniform(0.5, 2.5, size=(4, 2))
        phases = rng.uniform(0.0, 2.0 * math.pi, size=(4,))
        weights = rng.uniform(0.4, 1.0, size=(4,))
        weights = weights / weights.sum()
        flat = xy.reshape(-1, 2) / max(scale, _EPS)
        vals = np.zeros(flat.shape[0], dtype=np.float64)
        for w, f, p in zip(weights, freqs, phases):
            vals += w * np.sin(flat @ f + p)
        return np.clip(vals, -1.0, 1.0).reshape(xy.shape[:-1])


# ==============================================================================================
# AntMaze evaluation rewards
# ==============================================================================================
class AntMazeGoalReward(RewardFunction):
    """Sparse goal-reaching reward on the 32x32 discretised AntMaze grid.

    ``-1`` until the agent's XY *bin* is within ``threshold`` bins of the goal, ``0`` at the goal
    (the FRE prior's goal-reaching convention).
    """

    family = "antmaze_goal"

    def __init__(
        self,
        goal: Sequence[float] = (28, 0),
        threshold: float = ANTMAZE_GOAL_THRESHOLD,
        num_bins: int = ANTMAZE_XY_BINS,
        extent: float = ANTMAZE_XY_EXTENT,
        discrete: bool = True,
        xy_dims: Sequence[int] = ANTMAZE_XY_DIMS,
        reward_success: float = 0.0,
        reward_failure: float = -1.0,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(state_dim=state_dim, name=name or f"antmaze_goal_{tuple(goal)}", clip=clip, **kwargs)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.threshold = float(threshold)
        self.num_bins = int(num_bins)
        self.extent = float(extent)
        self.discrete = bool(discrete)
        self.xy_dims = tuple(int(d) for d in xy_dims)
        self.reward_success = float(reward_success)
        self.reward_failure = float(reward_failure)

    def distance(self, states: Any) -> np.ndarray:
        xy = antmaze_xy(states, self.xy_dims)
        if self.discrete:
            return bin_distance(discretize_xy(xy, self.extent, self.num_bins), self.goal)
        return np.linalg.norm(xy - self.goal, axis=-1)

    def is_success(self, states: Any) -> np.ndarray:
        return self.distance(states) <= self.threshold

    def _compute(self, states: Any) -> np.ndarray:
        success = self.is_success(states)
        return np.where(success, self.reward_success, self.reward_failure)

    def done_numpy(self, states: Any) -> np.ndarray:
        return np.asarray(self.is_success(states), dtype=bool)

    def describe(self) -> Dict[str, Any]:
        info = super().describe() if hasattr(super(), "describe") else {}
        info.update(
            {
                "goal": self.goal.tolist(),
                "threshold": self.threshold,
                "discrete": self.discrete,
                "reward_success": self.reward_success,
                "reward_failure": self.reward_failure,
            }
        )
        return info


class AntMazeDirectionalReward(RewardFunction):
    """Directional task: reward is the dot product of XY velocity with a fixed direction.

    The raw dot product is scaled and clipped into ``[-1, 1]`` so it matches the reward prior's
    range; success (for scoring) is declared when the dot product exceeds ``done_threshold``.
    """

    family = "antmaze_directional"

    def __init__(
        self,
        direction: Sequence[float] = (1.0, 0.0),
        vel_dims: Sequence[int] = ANTMAZE_VEL_XY_DIMS,
        scale: float = 1.0,
        done_threshold: float = 1.0,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            state_dim=state_dim, name=name or f"antmaze_directional_{tuple(direction)}", clip=clip, **kwargs
        )
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        self.direction = direction / norm if norm > _EPS else direction
        self.vel_dims = tuple(int(d) for d in vel_dims)
        self.scale = float(scale)
        self.done_threshold = float(done_threshold)

    def dot(self, states: Any) -> np.ndarray:
        vel = antmaze_velocity(states, self.vel_dims)
        return vel @ self.direction

    def _compute(self, states: Any) -> np.ndarray:
        return np.clip(self.dot(states) * self.scale, -1.0, 1.0)

    def done_numpy(self, states: Any) -> np.ndarray:
        return np.asarray(self.dot(states) >= self.done_threshold, dtype=bool)

    def describe(self) -> Dict[str, Any]:
        info = super().describe() if hasattr(super(), "describe") else {}
        info.update({"direction": self.direction.tolist(), "scale": self.scale})
        return info


class AntMazeSimplexReward(RewardFunction):
    """Random-simplex task: reward is a seeded ``opensimplex`` noise field of the XY position."""

    family = "antmaze_simplex"

    def __init__(
        self,
        seed: int = 1,
        extent: float = ANTMAZE_XY_EXTENT,
        scale: Optional[float] = None,
        xy_dims: Sequence[int] = ANTMAZE_XY_DIMS,
        done_threshold: Optional[float] = None,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(state_dim=state_dim, name=name or f"antmaze_simplex_{int(seed)}", clip=clip, **kwargs)
        self.seed = int(seed)
        self.extent = float(extent)
        self.scale = float(scale) if scale is not None else float(extent) * 0.5
        self.xy_dims = tuple(int(d) for d in xy_dims)
        self.done_threshold = done_threshold

    def raw_field(self, states: Any) -> np.ndarray:
        xy = antmaze_xy(states, self.xy_dims)
        return _opensimplex_field(xy, self.seed, scale=self.scale)

    def _compute(self, states: Any) -> np.ndarray:
        return np.clip(self.raw_field(states), -1.0, 1.0)

    def done_numpy(self, states: Any) -> np.ndarray:
        if self.done_threshold is None:
            return np.zeros(np.asarray(self.raw_field(states)).shape, dtype=bool)
        return np.asarray(self.raw_field(states) >= self.done_threshold, dtype=bool)

    def describe(self) -> Dict[str, Any]:
        info = super().describe() if hasattr(super(), "describe") else {}
        info.update({"seed": self.seed, "scale": self.scale})
        return info


class AntMazePathReward(RewardFunction):
    """Corridor task: ``0`` when within ``tolerance`` of the route, ``-1`` otherwise."""

    family = "antmaze_path"

    def __init__(
        self,
        task: str = "path-center",
        route: Optional[Sequence[Sequence[float]]] = None,
        tolerance: float = 1.5,
        xy_dims: Sequence[int] = ANTMAZE_XY_DIMS,
        dense: bool = False,
        reward_success: float = 0.0,
        reward_failure: float = -1.0,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        key = str(task)
        if key not in ANTMAZE_PATH_ROUTES and route is None:
            raise ValueError(f"unknown path task {task!r}; expected one of {sorted(ANTMAZE_PATH_ROUTES)}")
        super().__init__(state_dim=state_dim, name=name or f"antmaze_{key}", clip=clip, **kwargs)
        self.task = key
        self.route = tuple(tuple(map(float, p)) for p in (route or ANTMAZE_PATH_ROUTES[key]))
        self.tolerance = float(tolerance)
        self.xy_dims = tuple(int(d) for d in xy_dims)
        self.dense = bool(dense)
        self.reward_success = float(reward_success)
        self.reward_failure = float(reward_failure)

    def distance(self, states: Any) -> np.ndarray:
        return _distance_to_route(antmaze_xy(states, self.xy_dims), self.route)

    def is_success(self, states: Any) -> np.ndarray:
        return self.distance(states) <= self.tolerance

    def _compute(self, states: Any) -> np.ndarray:
        dist = self.distance(states)
        if self.dense:
            return np.clip(1.0 - dist / max(self.tolerance, _EPS), -1.0, 1.0)
        return np.where(dist <= self.tolerance, self.reward_success, self.reward_failure)

    def done_numpy(self, states: Any) -> np.ndarray:
        return np.asarray(self.is_success(states), dtype=bool)

    def describe(self) -> Dict[str, Any]:
        info = super().describe() if hasattr(super(), "describe") else {}
        info.update({"task": self.task, "tolerance": self.tolerance, "dense": self.dense})
        return info


# ==============================================================================================
# ExORL evaluation rewards
# ==============================================================================================
class ExORLGoalReward(RewardFunction):
    """Sparse goal-reaching reward in ExORL *physics* space (Euclidean distance < 0.1)."""

    family = "exorl_goal"

    def __init__(
        self,
        goal: Sequence[float],
        domain: str = "walker",
        threshold: float = EXORL_GOAL_THRESHOLD,
        physics_dims: Optional[Sequence[int]] = None,
        reward_success: float = 0.0,
        reward_failure: float = -1.0,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        domain = str(domain).lower()
        super().__init__(state_dim=state_dim, name=name or f"exorl_{domain}_goal", clip=clip, **kwargs)
        self.domain = domain
        self.goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        self.threshold = float(threshold)
        dims = physics_dims if physics_dims is not None else EXORL_PHYSICS_DIMS.get(domain)
        self.physics_dims = tuple(int(d) for d in dims) if dims is not None else None
        self.reward_success = float(reward_success)
        self.reward_failure = float(reward_failure)

    def physics(self, states: Any) -> np.ndarray:
        """Extract the physics features used for the goal distance.

        If ``physics_dims`` is ``None`` the **first** ``len(goal)`` dimensions of the state are
        assumed to be the (already appended) physics block.
        """
        arr = _as_states(states)
        if self.physics_dims is None:
            return np.ascontiguousarray(arr[..., : self.goal.shape[0]])
        # When physics is appended *after* the raw observation, the caller may either pass the
        # physics-only slice directly (already the right width) or the full encoder observation.
        if arr.shape[-1] == self.goal.shape[0] or max(self.physics_dims) >= arr.shape[-1]:
            return np.ascontiguousarray(arr[..., : self.goal.shape[0]])
        return np.ascontiguousarray(arr[..., list(self.physics_dims)])

    def distance(self, states: Any) -> np.ndarray:
        return np.linalg.norm(self.physics(states) - self.goal, axis=-1)

    def is_success(self, states: Any) -> np.ndarray:
        return self.distance(states) < self.threshold

    def _compute(self, states: Any) -> np.ndarray:
        return np.where(self.is_success(states), self.reward_success, self.reward_failure)

    def done_numpy(self, states: Any) -> np.ndarray:
        return np.asarray(self.is_success(states), dtype=bool)

    def describe(self) -> Dict[str, Any]:
        info = super().describe() if hasattr(super(), "describe") else {}
        info.update({"domain": self.domain, "goal": self.goal.tolist(), "threshold": self.threshold})
        return info


class ExORLVelocityReward(RewardFunction):
    """Velocity task: ``reward_value`` once the selected physics feature reaches ``threshold``.

    Following the ExORL loader convention the reward is sparse (``1.0`` above the threshold,
    ``0.0`` otherwise); the per-episode normalised score is then the fraction of successful steps.
    """

    family = "exorl_velocity"

    def __init__(
        self,
        domain: str = "walker",
        threshold: float = 1.0,
        feature_index: int = 0,
        physics_dims: Optional[Sequence[int]] = None,
        reward_value: float = 1.0,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        domain = str(domain).lower()
        super().__init__(
            state_dim=state_dim, name=name or f"exorl_{domain}_velocity_{threshold}", clip=clip, **kwargs
        )
        self.domain = domain
        dims = physics_dims if physics_dims is not None else EXORL_PHYSICS_DIMS.get(domain)
        self.physics_dims = tuple(int(d) for d in dims) if dims is not None else None
        self.feature_index = int(feature_index)
        self.threshold = float(threshold)
        self.reward_value = float(reward_value)

    def feature(self, states: Any) -> np.ndarray:
        arr = _as_states(states)
        dims = self.physics_dims
        if dims is None or max(dims) >= arr.shape[-1] or arr.shape[-1] == len(dims or ()):
            # physics-only slice
            return arr[..., self.feature_index]
        return arr[..., dims[self.feature_index]]

    def is_success(self, states: Any) -> np.ndarray:
        return self.feature(states) >= self.threshold

    def _compute(self, states: Any) -> np.ndarray:
        return np.where(self.is_success(states), self.reward_value, 0.0)

    def done_numpy(self, states: Any) -> np.ndarray:
        return np.asarray(self.is_success(states), dtype=bool)

    def describe(self) -> Dict[str, Any]:
        info = super().describe() if hasattr(super(), "describe") else {}
        info.update(
            {
                "domain": self.domain,
                "threshold": self.threshold,
                "feature_index": self.feature_index,
            }
        )
        return info


# ==============================================================================================
# Kitchen evaluation rewards
# ==============================================================================================
def _kitchen_flags(obs: Any) -> np.ndarray:
    arr = _as_states(obs)
    return arr[..., KITCHEN_FLAG_OFFSET:]


def _kitchen_flag_index(task: str) -> int:
    key = str(task).lower()
    if key in KITCHEN_TASKS:
        return KITCHEN_TASKS.index(key)
    # tolerate aliases such as "kitchen-microwave" / "bottom-burner"
    normalised = key.replace("kitchen-", "").replace("kitchen_", "").replace("_", "").replace("-", "")
    for i, cand in enumerate(KITCHEN_TASKS):
        if cand.replace("_", "") == normalised:
            return i
    raise ValueError(f"unknown Kitchen subtask {task!r}; expected one of {KITCHEN_TASKS}")


class KitchenSubtaskReward(RewardFunction):
    """Sparse Markovian reward for a D4RL Kitchen subtask.

    The seven completion flags live in the last seven observation dimensions, so the subtask reward
    is a genuine function of state: ``reward_success`` once the flag is set, ``reward_failure``
    before that.  ``sparse_transition=True`` alternatively reproduces D4RL's transition reward
    (``+1`` on the 0->1 flip of the completion flag).
    """

    family = "kitchen_subtask"

    def __init__(
        self,
        task: str = "microwave",
        reward_success: float = 1.0,
        reward_failure: float = 0.0,
        flag_threshold: float = KITCHEN_FLAG_THRESHOLD,
        sparse_transition: bool = False,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(state_dim=state_dim, name=name or f"kitchen_{task}", clip=clip, **kwargs)
        self.task = str(task)
        self.flag_index = _kitchen_flag_index(task)
        self.reward_success = float(reward_success)
        self.reward_failure = float(reward_failure)
        self.flag_threshold = float(flag_threshold)
        self.sparse_transition = bool(sparse_transition)

    def flag(self, obs: Any) -> np.ndarray:
        return _kitchen_flags(obs)[..., self.flag_index]

    def is_success(self, obs: Any) -> np.ndarray:
        return self.flag(obs) > self.flag_threshold

    def _compute(self, states: Any) -> np.ndarray:
        return np.where(self.is_success(states), self.reward_success, self.reward_failure)

    def done_numpy(self, states: Any) -> np.ndarray:
        return np.asarray(self.is_success(states), dtype=bool)

    def transition_reward(self, prev_states: Any, states: Any) -> np.ndarray:
        """D4RL-style sparse transition reward: ``1`` on a 0->1 completion flip, else ``0``."""
        prev = np.asarray(self.is_success(prev_states))
        cur = np.asarray(self.is_success(states))
        return np.where(cur & ~prev, 1.0, 0.0)

    def describe(self) -> Dict[str, Any]:
        info = super().describe() if hasattr(super(), "describe") else {}
        info.update({"task": self.task, "flag_index": self.flag_index})
        return info


# ==============================================================================================
# Task descriptors & scoring
# ==============================================================================================
@dataclass
class TaskScoring:
    """Converts per-episode raw quantities into the FRE normalised score in ``[0, 100]``."""

    mode: str = SCORE_SUCCESS_RATE
    min_return: Optional[float] = None
    max_return: Optional[float] = None
    reward_min: float = -1.0
    reward_max: float = 1.0
    clip: bool = True

    def __post_init__(self) -> None:
        if self.mode not in SCORE_MODES:
            raise ValueError(f"unknown score mode {self.mode!r}; expected one of {SCORE_MODES}")

    def score(self, episode_return: float, episode_length: int, success: bool) -> float:
        if self.mode == SCORE_SUCCESS_RATE:
            return 100.0 * float(bool(success))
        if self.mode == SCORE_MEAN_REWARD:
            length = max(int(episode_length), 1)
            mean_r = float(episode_return) / length
            span = max(self.reward_max - self.reward_min, _EPS)
            score = 100.0 * (mean_r - self.reward_min) / span
        else:  # SCORE_NORMALIZED_RETURN
            lo = self.min_return if self.min_return is not None else -1.0
            hi = self.max_return if self.max_return is not None else 1.0
            span = max(hi - lo, _EPS)
            score = 100.0 * (float(episode_return) - lo) / span
        if self.clip:
            score = min(max(score, 0.0), 100.0)
        return float(score)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "clip": self.clip,
        }


@dataclass
class EvalTask:
    """A single FRE zero-shot evaluation task."""

    name: str
    domain: str
    family: str
    reward_fn: RewardFunction
    max_episode_steps: int = ANTMAZE_MAX_STEPS
    scoring: TaskScoring = field(default_factory=TaskScoring)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score_mode(self) -> str:
        return self.scoring.mode

    def reward(self, states: Any) -> np.ndarray:
        """Evaluate the ground-truth reward field ``eta(s)``."""
        return np.asarray(self.reward_fn(states))

    def done(self, states: Any) -> np.ndarray:
        return np.asarray(self.reward_fn.done(states), dtype=bool) if hasattr(self.reward_fn, "done") else np.zeros(
            np.asarray(self.reward(states)).shape, dtype=bool
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "family": self.family,
            "max_episode_steps": self.max_episode_steps,
            "scoring": self.scoring.as_dict(),
            "metadata": dict(self.metadata),
        }


# ---- AntMaze suite ---------------------------------------------------------------------------
def antmaze_eval_tasks() -> Dict[str, EvalTask]:
    """The complete FRE AntMaze zero-shot task suite (17 tasks)."""
    tasks: Dict[str, EvalTask] = {}

    # 5 goal-reaching tasks
    for i, goal in enumerate(ANTMAZE_GOAL_TASKS):
        name = f"antmaze-goal-{i + 1}"
        tasks[name] = EvalTask(
            name=name,
            domain="antmaze",
            family="goal-reaching",
            reward_fn=AntMazeGoalReward(goal=goal, state_dim=ANTMAZE_OBS_DIM, name=name),
            max_episode_steps=ANTMAZE_MAX_STEPS,
            scoring=TaskScoring(mode=SCORE_SUCCESS_RATE),
            metadata={"goal": tuple(goal), "goal_bins": tuple(goal)},
        )

    # 4 directional tasks
    for i, direction in enumerate(ANTMAZE_DIRECTIONAL_TASKS):
        name = f"antmaze-directional-{i + 1}"
        tasks[name] = EvalTask(
            name=name,
            domain="antmaze",
            family="directional",
            reward_fn=AntMazeDirectionalReward(direction=direction, state_dim=ANTMAZE_OBS_DIM, name=name),
            max_episode_steps=ANTMAZE_MAX_STEPS,
            scoring=TaskScoring(mode=SCORE_SUCCESS_RATE),
            metadata={"direction": tuple(direction)},
        )

    # 5 random-simplex tasks
    for seed in ANTMAZE_SIMPLEX_SEEDS:
        name = f"antmaze-simplex-{seed}"
        tasks[name] = EvalTask(
            name=name,
            domain="antmaze",
            family="random-simplex",
            reward_fn=AntMazeSimplexReward(seed=seed, state_dim=ANTMAZE_OBS_DIM, name=name),
            max_episode_steps=ANTMAZE_MAX_STEPS,
            scoring=TaskScoring(mode=SCORE_MEAN_REWARD, reward_min=-1.0, reward_max=1.0),
            metadata={"seed": int(seed)},
        )

    # 3 corridor tasks
    for key in ANTMAZE_PATH_TASKS:
        name = f"antmaze-{key}"
        tasks[name] = EvalTask(
            name=name,
            domain="antmaze",
            family="path",
            reward_fn=AntMazePathReward(task=key, state_dim=ANTMAZE_OBS_DIM, name=name),
            max_episode_steps=ANTMAZE_MAX_STEPS,
            scoring=TaskScoring(mode=SCORE_SUCCESS_RATE),
            metadata={"task": key},
        )
    return tasks


# ---- ExORL suite -----------------------------------------------------------------------------
def _exorl_source(domain: str, dataset: Any = None, root: Optional[str] = None, seed: int = 0) -> Tuple[Any, np.ndarray]:
    """Return ``(dataset, goal_states)`` for an ExORL domain (lazy import of the loader)."""
    if dataset is None:
        try:  # pragma: no cover - optional dependency on local ExORL data
            from ..data.exorl_loader import load_exorl_dataset, select_goal_states  # type: ignore

            dataset = load_exorl_dataset(domain=domain, root=root, verbose=False)
        except Exception:
            try:
                from fre.data.exorl_loader import load_exorl_dataset, select_goal_states  # type: ignore

                dataset = load_exorl_dataset(domain=domain, root=root, verbose=False)
            except Exception:
                return None, np.zeros((0, 0), dtype=np.float64)
    else:
        try:
            from ..data.exorl_loader import select_goal_states  # type: ignore
        except Exception:
            try:
                from fre.data.exorl_loader import select_goal_states  # type: ignore
            except Exception:
                select_goal_states = None  # type: ignore
    goals = None
    if select_goal_states is not None:
        try:
            goals = np.asarray(select_goal_states(dataset, num_goals=EXORL_NUM_GOAL_STATES, seed=seed), dtype=np.float64)
        except Exception:
            goals = None
    if goals is None:
        physics = np.asarray(dataset.get("physics", dataset.get("encoder_observations")), dtype=np.float64)
        idx = np.linspace(0, max(len(physics) - 1, 0), EXORL_NUM_GOAL_STATES).astype(np.int64)
        goals = physics[idx]
    return dataset, goals


def exorl_eval_tasks(
    domain: str = "walker",
    dataset: Any = None,
    root: Optional[str] = None,
    goals: Optional[Sequence[Sequence[float]]] = None,
    seed: int = 0,
    include_velocity: bool = True,
) -> Dict[str, EvalTask]:
    """The FRE ExORL zero-shot task suite for one domain (5 goals + velocity tasks)."""
    domain = str(domain).lower()
    if goals is None:
        _, goals = _exorl_source(domain, dataset=dataset, root=root, seed=seed)
    goals = np.asarray(goals, dtype=np.float64).reshape(-1, len(EXORL_PHYSICS_DIMS[domain])) if len(goals) else goals

    tasks: Dict[str, EvalTask] = {}
    for i in range(min(len(goals), EXORL_NUM_GOAL_STATES)):
        name = f"exorl-{domain}-goal-{i + 1}"
        tasks[name] = EvalTask(
            name=name,
            domain=f"exorl-{domain}",
            family="goal-reaching",
            reward_fn=ExORLGoalReward(goal=goals[i], domain=domain, name=name),
            max_episode_steps=EXORL_MAX_STEPS,
            scoring=TaskScoring(mode=SCORE_SUCCESS_RATE),
            metadata={"goal_index": i},
        )
    if include_velocity:
        for t in EXORL_VELOCITY_THRESHOLDS[domain]:
            name = f"exorl-{domain}-velocity-{t}"
            tasks[name] = EvalTask(
                name=name,
                domain=f"exorl-{domain}",
                family="velocity",
                reward_fn=ExORLVelocityReward(domain=domain, threshold=t, feature_index=0, name=name),
                max_episode_steps=EXORL_MAX_STEPS,
                scoring=TaskScoring(mode=SCORE_SUCCESS_RATE),
                metadata={"threshold": float(t), "feature_index": 0},
            )
    return tasks


# ---- Kitchen suite ---------------------------------------------------------------------------
def kitchen_eval_tasks(dataset: Any = None, reward_style: str = "sparse") -> Dict[str, EvalTask]:
    """The 7 standard D4RL Kitchen subtasks as FRE zero-shot evaluation tasks."""
    tasks: Dict[str, EvalTask] = {}
    for task in KITCHEN_TASKS:
        name = f"kitchen-{task}"
        if reward_style == "goal":
            reward_fn = KitchenSubtaskReward(
                task=task, reward_success=0.0, reward_failure=-1.0, state_dim=KITCHEN_OBS_DIM, name=name
            )
            scoring = TaskScoring(mode=SCORE_SUCCESS_RATE)
        elif reward_style == "transition":
            reward_fn = KitchenSubtaskReward(task=task, sparse_transition=True, state_dim=KITCHEN_OBS_DIM, name=name)
            scoring = TaskScoring(mode=SCORE_SUCCESS_RATE)
        else:  # sparse Markovian
            reward_fn = KitchenSubtaskReward(
                task=task, reward_success=1.0, reward_failure=0.0, state_dim=KITCHEN_OBS_DIM, name=name
            )
            scoring = TaskScoring(mode=SCORE_SUCCESS_RATE)
        tasks[name] = EvalTask(
            name=name,
            domain="kitchen",
            family="kitchen",
            reward_fn=reward_fn,
            max_episode_steps=KITCHEN_MAX_STEPS,
            scoring=scoring,
            metadata={"task": task, "reward_style": reward_style},
        )
    return tasks


# ---- combined --------------------------------------------------------------------------------
def all_eval_tasks(
    exorl_datasets: Optional[Dict[str, Any]] = None,
    exorl_root: Optional[str] = None,
    include_exorl: bool = True,
    include_kitchen: bool = True,
    include_antmaze: bool = True,
) -> Dict[str, EvalTask]:
    """Every FRE zero-shot evaluation task (AntMaze + ExORL walker/cheetah + Kitchen)."""
    tasks: Dict[str, EvalTask] = {}
    if include_antmaze:
        tasks.update(antmaze_eval_tasks())
    if include_exorl:
        for domain in EXORL_DOMAINS:
            ds = None if exorl_datasets is None else exorl_datasets.get(domain)
            try:
                tasks.update(exorl_eval_tasks(domain=domain, dataset=ds, root=exorl_root))
            except Exception:
                # still provide the velocity tasks (no dataset needed) when data is unavailable
                for t in EXORL_VELOCITY_THRESHOLDS[domain]:
                    name = f"exorl-{domain}-velocity-{t}"
                    tasks[name] = EvalTask(
                        name=name,
                        domain=f"exorl-{domain}",
                        family="velocity",
                        reward_fn=ExORLVelocityReward(domain=domain, threshold=t, name=name),
                        max_episode_steps=EXORL_MAX_STEPS,
                        scoring=TaskScoring(mode=SCORE_SUCCESS_RATE),
                    )
    if include_kitchen:
        tasks.update(kitchen_eval_tasks())
    return tasks


#: The task groups used for the Table 1 sub-task reporting.
TASK_GROUPS: Dict[str, Tuple[str, ...]] = {
    "ant-goal-reaching": tuple(f"antmaze-goal-{i + 1}" for i in range(len(ANTMAZE_GOAL_TASKS))),
    "ant-directional": tuple(f"antmaze-directional-{i + 1}" for i in range(len(ANTMAZE_DIRECTIONAL_TASKS))),
    "ant-random-simplex": tuple(f"antmaze-simplex-{s}" for s in ANTMAZE_SIMPLEX_SEEDS),
    "ant-path": tuple(f"antmaze-{k}" for k in ANTMAZE_PATH_TASKS),
    "ant-path-center": ("antmaze-path-center",),
    "ant-path-loop": ("antmaze-path-loop",),
    "ant-path-edges": ("antmaze-path-edges",),
    "exorl-walker-goals": tuple(f"exorl-walker-goal-{i + 1}" for i in range(EXORL_NUM_GOAL_STATES)),
    "exorl-cheetah-goals": tuple(f"exorl-cheetah-goal-{i + 1}" for i in range(EXORL_NUM_GOAL_STATES)),
    "exorl-walker-velocity": tuple(
        f"exorl-walker-velocity-{t}" for t in EXORL_VELOCITY_THRESHOLDS["walker"]
    ),
    "exorl-cheetah-velocity": tuple(
        f"exorl-cheetah-velocity-{t}" for t in EXORL_VELOCITY_THRESHOLDS["cheetah"]
    ),
    "kitchen": tuple(f"kitchen-{t}" for t in KITCHEN_TASKS),
    "antmaze-all": tuple(
        [f"antmaze-goal-{i + 1}" for i in range(len(ANTMAZE_GOAL_TASKS))]
        + [f"antmaze-directional-{i + 1}" for i in range(len(ANTMAZE_DIRECTIONAL_TASKS))]
        + [f"antmaze-simplex-{s}" for s in ANTMAZE_SIMPLEX_SEEDS]
        + [f"antmaze-{k}" for k in ANTMAZE_PATH_TASKS]
    ),
    "exorl-all": tuple(
        [f"exorl-{d}-goal-{i + 1}" for d in EXORL_DOMAINS for i in range(EXORL_NUM_GOAL_STATES)]
        + [f"exorl-{d}-velocity-{t}" for d in EXORL_DOMAINS for t in EXORL_VELOCITY_THRESHOLDS[d]]
    ),
}
TASK_GROUPS["all"] = TASK_GROUPS["antmaze-all"] + TASK_GROUPS["exorl-all"] + TASK_GROUPS["kitchen"]


def task_groups() -> Dict[str, Tuple[str, ...]]:
    """Return the task-group -> task-name mapping (Table 1 / Figure 5 grouping)."""
    return dict(TASK_GROUPS)


def tasks_for_group(group: str, tasks: Optional[Dict[str, EvalTask]] = None) -> Dict[str, EvalTask]:
    """Select the subset of ``tasks`` belonging to the named group."""
    if tasks is None:
        tasks = all_eval_tasks()
    if group not in TASK_GROUPS:
        raise KeyError(f"unknown task group {group!r}; expected one of {sorted(TASK_GROUPS)}")
    return {name: tasks[name] for name in TASK_GROUPS[group] if name in tasks}


# ==============================================================================================
# Scoring helpers
# ==============================================================================================
def normalized_return(returns: Any, min_return: float, max_return: float, clip: bool = True) -> np.ndarray:
    """Map raw returns into ``[0, 100]`` using ``(r - min) / (max - min) * 100``."""
    returns = np.asarray(returns, dtype=np.float64)
    span = max(float(max_return) - float(min_return), _EPS)
    score = 100.0 * (returns - float(min_return)) / span
    return np.clip(score, 0.0, 100.0) if clip else score


def success_rate_score(successes: Any) -> float:
    """Normalised score = ``100 * mean(success)``."""
    successes = np.asarray(successes, dtype=np.float64).reshape(-1)
    if successes.size == 0:
        return 0.0
    return float(100.0 * successes.mean())


def mean_reward_score(returns: Any, lengths: Any, reward_min: float = -1.0, reward_max: float = 1.0) -> float:
    """Normalised score from the mean per-step reward (linear map ``[reward_min, reward_max]``)."""
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    if returns.size == 0:
        return 0.0
    if lengths.size == returns.size:
        per_step = returns / np.maximum(lengths, 1.0)
    else:
        per_step = returns
    span = max(float(reward_max) - float(reward_min), _EPS)
    return float(np.clip(100.0 * (per_step.mean() - reward_min) / span, 0.0, 100.0))


def score_episodes(
    task: EvalTask,
    episode_returns: Sequence[float],
    episode_lengths: Optional[Sequence[int]] = None,
    successes: Optional[Sequence[bool]] = None,
) -> float:
    """Score a set of evaluation episodes for one task, using its :class:`TaskScoring`."""
    returns = np.asarray(episode_returns, dtype=np.float64).reshape(-1)
    if episode_lengths is None:
        lengths = np.full_like(returns, task.max_episode_steps)
    else:
        lengths = np.asarray(episode_lengths, dtype=np.float64).reshape(-1)

    if task.scoring.mode == SCORE_SUCCESS_RATE:
        if successes is None:
            # fall back to "return == best possible return" heuristic
            successes = returns >= (return_ceiling(task) - 1e-6)
        return success_rate_score(successes)
    if task.scoring.mode == SCORE_MEAN_REWARD:
        return mean_reward_score(
            returns, lengths, task.scoring.reward_min, task.scoring.reward_max
        )
    lo = task.scoring.min_return
    hi = task.scoring.max_return
    if lo is None:
        lo = -float(task.max_episode_steps)
    if hi is None:
        hi = float(task.max_episode_steps)
    return float(normalized_return(returns.mean(), lo, hi, task.scoring.clip))


def return_ceiling(task: EvalTask) -> float:
    """Best achievable episode return for a success-rate task (0 for goal-reaching style rewards)."""
    return 0.0


def aggregate_scores(
    per_task_scores: Dict[str, float],
    groups: Optional[Dict[str, Sequence[str]]] = None,
    compute_std: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-task scores into the Table 1 groups (``mean`` and ``std``)."""
    groups = groups if groups is not None else TASK_GROUPS
    out: Dict[str, Dict[str, float]] = {}
    for group, names in groups.items():
        vals = [per_task_scores[n] for n in names if n in per_task_scores]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[group] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if (compute_std and arr.size > 1) else 0.0,
            "num_tasks": int(arr.size),
        }
    # global summary rows used by Table 1
    for key, members in (
        ("antmaze-all", "antmaze-all"),
        ("exorl-all", "exorl-all"),
        ("kitchen", "kitchen"),
        ("all", "all"),
    ):
        if key not in out and members in groups:
            vals = [per_task_scores[n] for n in groups[members] if n in per_task_scores]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                out[key] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if (compute_std and arr.size > 1) else 0.0,
                    "num_tasks": int(arr.size),
                }
    return out


# ==============================================================================================
# Self-test
# ==============================================================================================
def _self_test() -> Dict[str, Any]:  # pragma: no cover - manual sanity check
    rng = np.random.default_rng(0)
    results: Dict[str, Any] = {}

    # AntMaze: observation inside the maze
    obs = np.zeros((8, ANTMAZE_OBS_DIM), dtype=np.float64)
    obs[:, 0:2] = rng.uniform(0, 36, size=(8, 2))
    obs[:, 15:17] = rng.uniform(-1, 1, size=(8, 2))

    goal_fn = AntMazeGoalReward(goal=(28, 0))
    r = np.asarray(goal_fn(obs))
    assert r.shape == (8,) and set(np.unique(r)).issubset({-1.0, 0.0}), r
    results["antmaze_goal_range"] = (float(r.min()), float(r.max()))

    dir_fn = AntMazeDirectionalReward(direction=(1.0, 0.0))
    d = np.asarray(dir_fn(obs))
    assert d.shape == (8,) and d.min() >= -1.0 - 1e-6 and d.max() <= 1.0 + 1e-6
    results["antmaze_directional_range"] = (float(d.min()), float(d.max()))

    simp_fn = AntMazeSimplexReward(seed=1)
    s = np.asarray(simp_fn(obs))
    assert s.shape == (8,) and s.min() >= -1.0 - 1e-6 and s.max() <= 1.0 + 1e-6
    results["antmaze_simplex_range"] = (float(s.min()), float(s.max()))

    path_fn = AntMazePathReward(task="path-center")
    p = np.asarray(path_fn(obs))
    assert p.shape == (8,) and set(np.unique(p)).issubset({-1.0, 0.0})
    results["antmaze_path_range"] = (float(p.min()), float(p.max()))

    # ExORL: physics-only states
    phys = rng.uniform(-1, 1, size=(6, 3))
    exg = ExORLGoalReward(goal=np.zeros(3), domain="walker")
    rg = np.asarray(exg(phys))
    assert rg.shape == (6,)
    exv = ExORLVelocityReward(domain="walker", threshold=1.0)
    rv = np.asarray(exv(phys))
    assert rv.shape == (6,) and set(np.unique(rv)).issubset({0.0, 1.0})
    results["exorl_ranges"] = (float(rg.min()), float(rg.max()), float(rv.max()))

    # Kitchen
    kobs = np.zeros((4, KITCHEN_OBS_DIM), dtype=np.float64)
    kobs[1, -7] = 1.0
    kobs[3, -1] = 1.0
    kfn = KitchenSubtaskReward(task="microwave")
    kr = np.asarray(kfn(kobs))
    assert kr.shape == (4,)
    assert kr[0] == 0.0 and kr[1] == 1.0 and kr[3] == 0.0
    results["kitchen_reward"] = kr.tolist()

    # Suites + scoring
    tasks = all_eval_tasks(include_exorl=False)
    results["num_tasks"] = len(tasks)
    groups = task_groups()
    results["num_groups"] = len(groups)

    scores = {name: float(rng.uniform(0, 100)) for name in tasks}
    agg = aggregate_scores(scores, groups=groups)
    results["aggregate_keys"] = sorted(agg.keys())
    results["antmaze_all"] = agg.get("antmaze-all", {}).get("mean")

    # scoring modes
    assert abs(success_rate_score([True, False, True, True]) - 75.0) < 1e-6
    assert abs(normalized_return(0.0, -10.0, 10.0) - 50.0) < 1e-6
    assert abs(mean_reward_score([0.0, 0.0], [10, 10]) - 50.0) < 1e-6
    return results


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(_self_test(), indent=2, default=str))
