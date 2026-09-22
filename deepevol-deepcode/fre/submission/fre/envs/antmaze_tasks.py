"""AntMaze evaluation tasks for FRE (Section 5.2 / Appendix C.1).

This module implements the *online evaluation* task suite used for the
``antmaze-large-diverse-v2`` domain:

``ant-goal-reaching``
    Average of 5 hand-crafted goal-reaching reward functions.  The reward is
    ``-1`` for every timestep the goal is not achieved (``0`` on achievement);
    a goal counts as reached within a (grid) distance of ``2``.  The five goal
    locations, on an ``(X, Y)`` grid with the origin at the bottom left, are

    =============== =============
    ``goal-bottom`` ``(28, 0)``
    ``goal-left``   ``(0, 15)``
    ``goal-top``    ``(35, 24)``
    ``goal-center`` ``(12, 24)``
    ``goal-right``  ``(33, 16)``
    =============== =============

``ant-directional``
    Average over the directional velocity tasks: the reward is the dot product
    of the agent's ``(X, Y)`` velocity with a target unit direction
    (``(-1, 0)``, ``(0, 1)``, ``(0, -1)``, ``(1, 0)``).

``ant-random-simplex``
    Five seeded tasks (seeds 1..5) whose reward is a random 2D opensimplex
    "height" field plus an opensimplex-generated preferred velocity field:
    the agent receives a baseline reward of ``-1`` at each step, a bonus for
    standing in higher "height" regions and an additional bonus for moving in
    the local preferred velocity direction.

``ant-path-center`` / ``ant-path-loop`` / ``ant-path-edges``
    Reward functions that reward the agent for moving along hand-crafted
    corridors placed respectively in the center of the grid, in a loop around
    the grid, and along the edges of the grid.

Additional FRE-specific details (Appendix C.1):

* the ant robot is placed in the *center of the maze* at reset (rather than
  the original bottom-left start position),
* online evaluation is capped at **2000 steps** per trajectory,
* the X/Y coordinates are discretized into **32 bins** for the encoding stream
  of FRE / GC-IQL / GC-BC / OPAL (see :func:`antmaze_encoder_states`).

Task rewards are *state functions* ``eta(s)`` so that they can be evaluated on
arbitrary offline dataset states (which is what the FRE encoder consumes).

.. note::
    The paper's prose calls ``ant-directional`` "5 directional tasks" but then
    explicitly lists only the four directions above; we implement the four
    listed directions (the plan/Table 1 also use four) and expose the number of
    directional tasks through :data:`DEFAULT_DIRECTIONS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from fre.priors.goal_functions import (
    DEFAULT_GOAL_THRESHOLD,
    DirectionalRewardFunction,
    GoalRewardFunction,
    RewardFunction,
    goal_distances,
    goal_reached_mask,
)

try:  # optional; only used for the reward-function signature tests
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency of the project
    torch = None  # type: ignore

__all__ = [
    # constants
    "ANTMAZE_ENV_NAME",
    "ANTMAZE_MAX_EPISODE_STEPS",
    "ANTMAZE_XY_INDICES",
    "ANTMAZE_XY_VELOCITY_INDICES",
    "ANTMAZE_GOAL_DISTANCE",
    "ANTMAZE_GRID_EXTENT",
    "ANTMAZE_GRID_CENTER",
    "NUM_XY_BINS",
    "ANTMAZE_GOAL_TASKS",
    "DEFAULT_DIRECTIONS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "ANTMAZE_PATH_TASKS",
    "ANTMAZE_TASK_SETS",
    # low level helpers
    "ant_xy",
    "ant_velocity",
    "xy_grid_extent",
    "maze_center_xy",
    "path_distance",
    "path_progress",
    # reward functions
    "AntMazeGoalReward",
    "AntMazeDirectionalReward",
    "AntMazeSimplexReward",
    "AntMazePathReward",
    # task specs
    "TaskSpec",
    "make_goal_task",
    "make_directional_task",
    "make_simplex_task",
    "make_path_task",
    "build_task_set",
    "build_tasks",
    "list_task_sets",
    "get_task",
    "task_names",
    # env utilities
    "make_antmaze_env",
    "get_sim_state",
    "set_sim_state",
    "reset_to_center",
    "make_antmaze_task_env",
    "antmaze_encoder_states",
    "encoding_samples_for_task",
    "encoding_samples_from_env",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ANTMAZE_ENV_NAME = "antmaze-large-diverse-v2"
ANTMAZE_MAX_EPISODE_STEPS = 2000
NUM_XY_BINS = 32

#: Indices of the ``(x, y)`` position inside the 29-d AntMaze observation.
ANTMAZE_XY_INDICES: Tuple[int, int] = (0, 1)
#: Indices of the ``(x, y)`` linear velocity inside the 29-d AntMaze observation
#: (qvel follows the 15-d qpos, so ``qvel[0]`` is at index 15).
ANTMAZE_XY_VELOCITY_INDICES: Tuple[int, int] = (15, 16)

#: A goal is reached within this Euclidean distance (Appendix C.1).
ANTMAZE_GOAL_DISTANCE = 2.0

#: Extent of the maze grid in environment coordinates.  ``antmaze-large`` is a
#: 9x9 cell maze with a cell size of 4, i.e. coordinates in ``[0, 36]``.
ANTMAZE_GRID_EXTENT: Tuple[float, float] = (36.0, 36.0)
ANTMAZE_GRID_CENTER: Tuple[float, float] = (18.0, 18.0)

#: ``(name, goal_xy)`` pairs of the 5 hand-crafted goal-reaching tasks.
ANTMAZE_GOAL_TASKS: Tuple[Tuple[str, Tuple[float, float]], ...] = (
    ("goal-bottom", (28.0, 0.0)),
    ("goal-left", (0.0, 15.0)),
    ("goal-top", (35.0, 24.0)),
    ("goal-center", (12.0, 24.0)),
    ("goal-right", (33.0, 16.0)),
)

#: ``(name, unit_direction)`` pairs of the directional velocity tasks.
DEFAULT_DIRECTIONS: Tuple[Tuple[str, Tuple[float, float]], ...] = (
    ("vel_left", (-1.0, 0.0)),
    ("vel_up", (0.0, 1.0)),
    ("vel_down", (0.0, -1.0)),
    ("vel_right", (1.0, 0.0)),
)

ANTMAZE_SIMPLEX_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)

ANTMAZE_PATH_TASKS: Tuple[str, ...] = ("ant-path-center", "ant-path-loop", "ant-path-edges")

#: Correlation length of the opensimplex fields (in "noise units" across the
#: whole maze extent); larger => smoother fields.
DEFAULT_NOISE_FREQUENCY = 3.0

#: Corridor half-width used by the path tasks (in environment coordinates).
DEFAULT_CORRIDOR_WIDTH = 3.0


# --------------------------------------------------------------------------- #
# Low-level state helpers
# --------------------------------------------------------------------------- #


def _as_float_array(states: Any) -> Tuple[np.ndarray, Any]:
    """Convert ``states`` to a float64 numpy array.

    Returns ``(array, original)`` where ``original`` is ``None`` for non-torch
    inputs and the original tensor otherwise (so the output can be cast back).
    """
    if torch is not None and isinstance(states, torch.Tensor):
        return states.detach().cpu().numpy().astype(np.float64), states
    return np.asarray(states, dtype=np.float64), None


def _restore(values: np.ndarray, original: Any) -> Any:
    """Cast a numpy result back to a torch tensor when the input was a tensor."""
    if original is not None:
        return torch.as_tensor(
            values,
            dtype=original.dtype if original.dtype.is_floating_point else torch.float32,
            device=original.device,
        )
    return values


def ant_xy(states: Any, xy_indices: Tuple[int, int] = ANTMAZE_XY_INDICES) -> Any:
    """Return the ``(x, y)`` position from AntMaze observations.

    Shape ``states.shape[:-1] + (2,)``; torch in -> torch out.
    """
    array, original = _as_float_array(states)
    xy = np.asarray(array[..., list(xy_indices)], dtype=np.float64)
    return _restore(xy, original)


def ant_velocity(
    states: Any, velocity_indices: Tuple[int, int] = ANTMAZE_XY_VELOCITY_INDICES
) -> Any:
    """Return the ``(x, y)`` linear velocity from AntMaze observations."""
    array, original = _as_float_array(states)
    vel = np.asarray(array[..., list(velocity_indices)], dtype=np.float64)
    return _restore(vel, original)


def xy_grid_extent(
    env: Any = None, default: Tuple[float, float] = ANTMAZE_GRID_EXTENT
) -> Tuple[float, float]:
    """Best-effort ``(width, height)`` of the maze in environment coordinates.

    Falls back to :data:`ANTMAZE_GRID_EXTENT` when the extent cannot be read
    from the environment.
    """
    if env is None:
        return tuple(default)  # type: ignore[return-value]
    unwrapped = getattr(env, "unwrapped", env)
    for attr in ("maze_extent", "xy_extent", "arena_extent"):
        value = getattr(unwrapped, attr, None)
        if value is not None:
            try:
                extent = tuple(float(v) for v in np.asarray(value).reshape(-1)[:2])
            except Exception:
                continue
            if len(extent) == 2 and extent[0] > 0 and extent[1] > 0:
                return extent  # type: ignore[return-value]
    maze = getattr(unwrapped, "_maze", None) or getattr(unwrapped, "maze", None)
    if maze is not None:
        for w_attr, h_attr in (("maze_width", "maze_height"), ("width", "height")):
            width = getattr(maze, w_attr, None)
            height = getattr(maze, h_attr, None)
            scaling = getattr(maze, "maze_size_scaling", None)
            if scaling is None:
                scaling = getattr(unwrapped, "maze_size_scaling", 4.0)
            if width and height and scaling:
                return (float(width) * float(scaling), float(height) * float(scaling))
    return tuple(default)  # type: ignore[return-value]


def maze_center_xy(
    env: Any = None, default: Tuple[float, float] = ANTMAZE_GRID_CENTER
) -> np.ndarray:
    """Center of the maze grid (Appendix C.1: the ant starts at the center)."""
    extent = xy_grid_extent(env)
    if env is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(extent, dtype=np.float64) / 2.0


# --------------------------------------------------------------------------- #
# opensimplex height / velocity fields
# --------------------------------------------------------------------------- #


class _FallbackNoise:
    """Deterministic smooth noise in ``[-1, 1]`` used when opensimplex is absent.

    Implemented as a small sum of sinusoids with seed-dependent frequencies,
    which reproduces the qualitative behaviour (smooth random "height" field)
    of opensimplex.
    """

    def __init__(self, seed: int = 0, num_components: int = 5) -> None:
        rng = np.random.RandomState(int(seed) % (2**31 - 1))
        self.freqs = rng.uniform(0.5, 2.5, size=(num_components, 2))
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=num_components)
        self.weights = rng.uniform(0.5, 1.0, size=num_components)
        self.weights = self.weights / self.weights.sum()

    def noise2(self, x: float, y: float) -> float:  # noqa: D102 - mirror opensimplex API
        value = float(
            np.sum(
                self.weights
                * np.sin(self.freqs[:, 0] * float(x) + self.phases)
                * np.cos(self.freqs[:, 1] * float(y) + self.phases)
            )
        )
        return float(np.clip(value, -1.0, 1.0))


def _make_noise(seed: int) -> Any:
    """Create an opensimplex (or fallback) 2-d noise source for ``seed``."""
    try:  # pragma: no cover - exercised depending on the environment
        from opensimplex import OpenSimplex  # type: ignore

        try:
            return OpenSimplex(seed=int(seed))
        except TypeError:  # older/newer signatures
            return OpenSimplex(seed=int(seed), randomize=False)
    except Exception:
        return _FallbackNoise(seed=int(seed))


def _noise2(noise: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised ``noise2`` (falls back to element-wise evaluation)."""
    flat_x = np.asarray(x, dtype=np.float64).reshape(-1)
    flat_y = np.asarray(y, dtype=np.float64).reshape(-1)
    try:
        out = noise.noise2(flat_x, flat_y)
        out = np.asarray(out, dtype=np.float64)
        if out.shape == flat_x.shape:
            return np.clip(np.nan_to_num(out), -1.0, 1.0)
    except Exception:
        pass
    out = np.empty_like(flat_x)
    for i in range(flat_x.size):
        out[i] = float(noise.noise2(float(flat_x[i]), float(flat_y[i])))
    return np.clip(np.nan_to_num(out), -1.0, 1.0)


def _normalise_xy(
    xy: np.ndarray, extent: Tuple[float, float], frequency: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Map environment ``(x, y)`` to noise coordinates in ``[0, frequency]``."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    scale = np.asarray(extent, dtype=np.float64).reshape(2)
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    return xy[:, 0] / scale[0] * frequency, xy[:, 1] / scale[1] * frequency


# --------------------------------------------------------------------------- #
# Path helpers (corridor tasks)
# --------------------------------------------------------------------------- #


def _polyline_segments(
    waypoints: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(starts, ends, cumulative_arclength)`` of a polyline."""
    points = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 2:
        raise ValueError("a path needs at least two waypoints")
    starts = points[:-1]
    ends = points[1:]
    lengths = np.linalg.norm(ends - starts, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    return starts, ends, cum


def path_distance(xy: Any, waypoints: Optional[np.ndarray] = None, path: Optional[np.ndarray] = None) -> np.ndarray:
    """Minimum Euclidean distance from each ``(x, y)`` to a polyline path.

    Parameters
    ----------
    xy: array of shape ``(..., 2)``.
    waypoints / path: sequence of ``(2,)`` waypoints describing the polyline.

    Returns
    -------
    np.ndarray of shape ``xy.shape[:-1]``.
    """
    points = waypoints if waypoints is not None else path
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    flat = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    starts, ends, _ = _polyline_segments(points)
    seg = ends - starts
    seg_len_sq = np.maximum(np.sum(seg * seg, axis=-1), 1e-12)
    # (N, S, 2)
    rel = flat[:, None, :] - starts[None, :, :]
    t = np.clip(np.sum(rel * seg[None, :, :], axis=-1) / seg_len_sq[None, :], 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
    dists = np.linalg.norm(flat[:, None, :] - closest, axis=-1)
    out = dists.min(axis=-1)
    return out.reshape(np.asarray(xy).shape[:-1])


def path_progress(
    xy: Any,
    waypoints: Optional[np.ndarray] = None,
    path: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalised arclength position ``s / total_length`` in ``[0, 1]``.

    The projection uses the closest segment of the polyline.
    """
    points = waypoints if waypoints is not None else path
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    flat = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    starts, ends, cum = _polyline_segments(points)
    seg = ends - starts
    seg_len = np.linalg.norm(seg, axis=-1)
    seg_len_sq = np.maximum(np.sum(seg * seg, axis=-1), 1e-12)
    rel = flat[:, None, :] - starts[None, :, :]
    t = np.clip(np.sum(rel * seg[None, :, :], axis=-1) / seg_len_sq[None, :], 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
    dists = np.linalg.norm(flat[:, None, :] - closest, axis=-1)
    best = np.argmin(dists, axis=-1)
    rows = np.arange(flat.shape[0])
    s = cum[best] + t[rows, best] * seg_len[best]
    total = float(cum[-1]) if cum[-1] > 1e-12 else 1.0
    return (s / total).reshape(np.asarray(xy).shape[:-1])


# --------------------------------------------------------------------------- #
# Reward functions
# --------------------------------------------------------------------------- #


class AntMazeGoalReward(GoalRewardFunction):
    """Singleton goal-reaching reward on the AntMaze ``(x, y)`` plane.

    ``-1`` while the goal is unachieved, ``0`` once the distance is below
    :data:`ANTMAZE_GOAL_DISTANCE` (i.e. ``2``).
    """

    def __init__(
        self,
        goal: Sequence[float],
        threshold: float = ANTMAZE_GOAL_DISTANCE,
        distance_dims: Sequence[int] = ANTMAZE_XY_INDICES,
        state_dim: Optional[int] = None,
        reward_unachieved: float = -1.0,
        reward_achieved: float = 0.0,
        state_std: Optional[Any] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            goal=goal,
            threshold=float(threshold),
            state_dim=state_dim,
            distance_dims=tuple(distance_dims),
            state_std=state_std,
            reward_unachieved=reward_unachieved,
            reward_achieved=reward_achieved,
            name=name,
            **kwargs,
        )

    @property
    def family(self) -> str:
        return "antmaze-goal"


class AntMazeDirectionalReward(DirectionalRewardFunction):
    """Directional velocity reward: ``dot(v_xy, direction) + bias``."""

    def __init__(
        self,
        direction: Sequence[float],
        velocity_indices: Sequence[int] = ANTMAZE_XY_VELOCITY_INDICES,
        state_dim: Optional[int] = None,
        scale: float = 1.0,
        bias: float = 0.0,
        normalise_velocity: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.bias = float(bias)
        self.normalise_velocity = bool(normalise_velocity)
        super().__init__(
            direction=direction,
            velocity_indices=tuple(velocity_indices),
            state_dim=state_dim,
            scale=float(scale),
            name=name,
            **kwargs,
        )

    def reward(self, states: Any) -> Any:  # type: ignore[override]
        if not self.normalise_velocity:
            out = super().reward(states)
            return out + self.bias if self.bias else out
        array, original = _as_float_array(states)
        vel = np.asarray(array[..., list(self.velocity_indices)], dtype=np.float64)
        norm = np.linalg.norm(vel, axis=-1, keepdims=True)
        vel = vel / np.maximum(norm, 1e-8)
        direction = np.asarray(self.direction, dtype=np.float64).reshape(-1)[:2]
        out = float(self.scale) * (vel * direction).sum(axis=-1) + self.bias
        return _restore(out, original)

    @property
    def family(self) -> str:
        return "antmaze-directional"


class AntMazeSimplexReward(RewardFunction):
    """Seeded opensimplex "height" + preferred-velocity reward.

    ``eta(s) = baseline + height_weight * height(x, y)
              + velocity_weight * alignment(v, preferred(x, y))``

    where ``height`` is an opensimplex field rescaled to ``[0, 1]``, and
    ``preferred`` is a unit direction read from a second opensimplex field
    (its angle) -- both fixed by the task seed (1..5 in the paper).
    """

    def __init__(
        self,
        seed: int = 1,
        extent: Sequence[float] = ANTMAZE_GRID_EXTENT,
        noise_frequency: float = DEFAULT_NOISE_FREQUENCY,
        baseline: float = -1.0,
        height_weight: float = 1.0,
        velocity_weight: float = 1.0,
        velocity_indices: Sequence[int] = ANTMAZE_XY_VELOCITY_INDICES,
        normalise_velocity: bool = True,
        alignment_mode: str = "raw",  # "raw" | "positive"
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        noise: Any = None,
        direction_noise: Any = None,
        **kwargs: Any,
    ) -> None:
        self.seed = int(seed)
        self.extent = tuple(float(v) for v in np.asarray(extent).reshape(-1)[:2])
        self.noise_frequency = float(noise_frequency)
        self.baseline = float(baseline)
        self.height_weight = float(height_weight)
        self.velocity_weight = float(velocity_weight)
        self.velocity_indices = tuple(velocity_indices)
        self.normalise_velocity = bool(normalise_velocity)
        if alignment_mode not in ("raw", "positive"):
            raise ValueError("alignment_mode must be 'raw' or 'positive'")
        self.alignment_mode = alignment_mode
        self.state_dim = state_dim
        self._noise = noise if noise is not None else _make_noise(self.seed)
        self._direction_noise = (
            direction_noise if direction_noise is not None else _make_noise(self.seed + 1000)
        )
        self.name = name or f"simplex-{self.seed}"

    # -- fields ---------------------------------------------------------- #
    def height(self, xy: Any) -> np.ndarray:
        """Noise "height" in ``[0, 1]`` for each ``(x, y)``."""
        array = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        nx, ny = _normalise_xy(array, self.extent, self.noise_frequency)  # type: ignore[arg-type]
        raw = _noise2(self._noise, nx, ny)
        return np.clip(0.5 * (raw + 1.0), 0.0, 1.0)

    def preferred_direction(self, xy: Any) -> np.ndarray:
        """Unit preferred velocity direction for each ``(x, y)``."""
        array = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        nx, ny = _normalise_xy(array, self.extent, self.noise_frequency)  # type: ignore[arg-type]
        angle = np.pi * _noise2(self._direction_noise, nx, ny)
        direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        return direction

    # -- reward ---------------------------------------------------------- #
    def reward(self, states: Any) -> Any:  # type: ignore[override]
        array, original = _as_float_array(states)
        flat = array.reshape(-1, array.shape[-1])
        xy = flat[:, list(ANTMAZE_XY_INDICES)]
        vel = flat[:, list(self.velocity_indices)]

        height = self.height(xy)
        direction = self.preferred_direction(xy)
        if self.normalise_velocity:
            norm = np.linalg.norm(vel, axis=-1, keepdims=True)
            vel = vel / np.maximum(norm, 1e-8)
        align = np.sum(vel * direction, axis=-1)
        if self.alignment_mode == "positive":
            align = np.maximum(align, 0.0)
        align = np.clip(align, -1.0, 1.0)

        out = (
            self.baseline
            + self.height_weight * height
            + self.velocity_weight * align
        ).astype(np.float64)
        out = out.reshape(array.shape[:-1])
        return _restore(out, original)

    @property
    def family(self) -> str:
        return "antmaze-simplex"

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"seed={self.seed}, baseline={self.baseline}, "
            f"height_weight={self.height_weight}, velocity_weight={self.velocity_weight}"
        )


class AntMazePathReward(RewardFunction):
    """Corridor reward for the hand-crafted path tasks.

    The agent receives ``reward_inside`` while it stays within
    ``corridor_width`` of the (poly)line and ``reward_outside`` otherwise.  An
    optional stateless progress shaping term rewards being further along the
    path.

    Paths are specified in *fractional* grid coordinates (``[0, 1]``), which are
    scaled by the maze extent so that the tasks are independent of the exact
    coordinate range of the environment.
    """

    def __init__(
        self,
        waypoints: Optional[Sequence[Sequence[float]]] = None,
        extent: Sequence[float] = ANTMAZE_GRID_EXTENT,
        corridor_width: float = DEFAULT_CORRIDOR_WIDTH,
        reward_inside: float = 0.0,
        reward_outside: float = -1.0,
        progress_weight: float = 0.0,
        success_at_end: bool = True,
        normalised_waypoints: bool = True,
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        path: Optional[Sequence[Sequence[float]]] = None,
        **kwargs: Any,
    ) -> None:
        points = waypoints if waypoints is not None else path
        if points is None:
            raise ValueError("AntMazePathReward requires waypoints")
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if normalised_waypoints:
            extent_arr = np.asarray(extent, dtype=np.float64).reshape(2)
            points = points * extent_arr[None, :]
        self.waypoints = points
        self.extent = tuple(float(v) for v in np.asarray(extent).reshape(-1)[:2])
        self.corridor_width = float(corridor_width)
        self.reward_inside = float(reward_inside)
        self.reward_outside = float(reward_outside)
        self.progress_weight = float(progress_weight)
        self.success_at_end = bool(success_at_end)
        self.state_dim = state_dim
        self.name = name or "ant-path"

    # -- helpers --------------------------------------------------------- #
    def distance(self, states: Any) -> Any:
        """Distance from each state's ``(x, y)`` to the path."""
        xy = ant_xy(states)
        array, original = _as_float_array(xy)
        out = path_distance(array, self.waypoints)
        return _restore(out, original)

    def progress(self, states: Any) -> Any:
        """Normalised arclength position along the path in ``[0, 1]``."""
        xy = ant_xy(states)
        array, original = _as_float_array(xy)
        out = path_progress(array, self.waypoints)
        return _restore(out, original)

    def success(self, states: Any) -> Any:
        """Boolean mask: the agent reached the end of the path."""
        dist_to_end = goal_distances(
            np.asarray(_as_float_array(states)[0], dtype=np.float64),
            np.asarray(self.waypoints, dtype=np.float64)[-1:],
            dims=ANTMAZE_XY_INDICES,
        )
        return dist_to_end <= max(self.corridor_width, ANTMAZE_GOAL_DISTANCE)

    # -- reward ---------------------------------------------------------- #
    def reward(self, states: Any) -> Any:  # type: ignore[override]
        array, original = _as_float_array(states)
        flat = array.reshape(-1, array.shape[-1])
        xy = flat[:, list(ANTMAZE_XY_INDICES)]
        dist = path_distance(xy, self.waypoints)
        inside = dist <= self.corridor_width
        out = np.where(inside, self.reward_inside, self.reward_outside).astype(np.float64)
        if self.progress_weight:
            out = out + self.progress_weight * path_progress(xy, self.waypoints)
        out = out.reshape(array.shape[:-1])
        return _restore(out, original)

    @property
    def family(self) -> str:
        return "antmaze-path"

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"waypoints={len(self.waypoints)}, corridor_width={self.corridor_width}, "
            f"progress_weight={self.progress_weight}"
        )


# --------------------------------------------------------------------------- #
# Path definitions (fractional grid coordinates, origin bottom-left)
# --------------------------------------------------------------------------- #

#: Corridors through the center of the grid (a cross of two corridors).
PATH_CENTER_WAYPOINTS: Tuple[Tuple[float, float], ...] = (
    (0.30, 0.50),
    (0.70, 0.50),
)
PATH_CENTER_EXTRA = ((0.50, 0.30), (0.50, 0.70))

#: A loop around the grid.
PATH_LOOP_WAYPOINTS: Tuple[Tuple[float, float], ...] = (
    (0.20, 0.20),
    (0.80, 0.20),
    (0.80, 0.80),
    (0.20, 0.80),
    (0.20, 0.20),
)

#: Corridors along the edges of the grid.
PATH_EDGES_WAYPOINTS: Tuple[Tuple[float, float], ...] = (
    (0.05, 0.05),
    (0.95, 0.05),
    (0.95, 0.95),
    (0.05, 0.95),
    (0.05, 0.05),
)


# --------------------------------------------------------------------------- #
# Task specifications
# --------------------------------------------------------------------------- #


@dataclass
class TaskSpec:
    """Description of one AntMaze evaluation task.

    ``reward_fn`` is a state reward function ``eta(s)`` (callable or object with
    a ``.reward(states)`` method).  ``success_fn`` returns a boolean mask
    indicating task success (used for the goal tasks and reported separately).
    """

    name: str
    task_set: str
    reward_fn: Any
    goal: Optional[np.ndarray] = None
    success_fn: Optional[Callable[[Any], Any]] = None
    done_fn: Optional[Callable[[Any], Any]] = None
    max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS
    coordinate_scale: Tuple[float, float] = ANTMAZE_GRID_EXTENT
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ----------------------------------------------------- #
    @property
    def family(self) -> str:
        """Task family (e.g. ``"goal"``, ``"directional"``, ``"simplex"``)."""
        return self.task_set

    def reward(self, states: Any) -> np.ndarray:
        """Evaluate the task reward on (batched) states."""
        fn = self.reward_fn
        if hasattr(fn, "reward") and not callable(fn):
            return np.asarray(fn.reward(states))
        if hasattr(fn, "reward"):
            return np.asarray(fn.reward(states))
        return np.asarray(fn(states))

    def is_success(self, states: Any) -> np.ndarray:
        """Boolean (batched) success mask; ``False`` when undefined."""
        if self.success_fn is None:
            return np.zeros(np.asarray(states).shape[:-1], dtype=bool)
        return np.asarray(self.success_fn(states)).astype(bool)

    def is_done(self, states: Any) -> np.ndarray:
        """Boolean (batched) done mask; ``False`` for non-terminating tasks."""
        if self.done_fn is not None:
            return np.asarray(self.done_fn(states)).astype(bool)
        if self.success_fn is not None:
            return self.is_success(states)
        array = np.asarray(states)
        return np.zeros(array.shape[:-1], dtype=bool)

    def success_fn_for_wrapper(self) -> Optional[Callable[[Any], Any]]:
        """Success predicate usable by :mod:`fre.envs.reward_wrappers`."""
        if self.success_fn is not None:
            return lambda state: bool(np.asarray(self.success_fn(state)).reshape(-1)[-1])
        if self.goal is not None:
            goal = np.asarray(self.goal, dtype=np.float64)
            dims = ANTMAZE_XY_INDICES
            threshold = ANTMAZE_GOAL_DISTANCE
            return lambda state: bool(
                goal_distances(
                    np.asarray(state, dtype=np.float64), goal, dims=dims
                )
                <= threshold
            )
        return None

    def encoder_reward(self, states: Any) -> np.ndarray:
        """Rewards used to label encoder/decoder states (per-reward-function
        min/max normalisation happens inside :class:`RewardEmbedding`)."""
        return np.asarray(self.reward(states), dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_set": self.task_set,
            "goal": None if self.goal is None else np.asarray(self.goal).tolist(),
            "max_episode_steps": int(self.max_episode_steps),
            "metadata": dict(self.metadata),
        }


def make_goal_task(
    name: str, goal: Sequence[float], threshold: float = ANTMAZE_GOAL_DISTANCE, **kwargs: Any
) -> TaskSpec:
    """Create one hand-crafted goal-reaching task (``-1`` until achieved)."""
    goal_arr = np.asarray(goal, dtype=np.float64).reshape(2)
    reward_fn = AntMazeGoalReward(goal=goal_arr, threshold=threshold, name=name)
    success_fn = lambda states, g=goal_arr, t=threshold: goal_reached_mask(  # noqa: E731
        np.asarray(states, dtype=np.float64), g, threshold=t, dims=ANTMAZE_XY_INDICES
    )
    return TaskSpec(
        name=name,
        task_set="goal",
        reward_fn=reward_fn,
        goal=goal_arr,
        success_fn=success_fn,
        done_fn=success_fn,
        metadata={"goal_xy": goal_arr.tolist(), "threshold": float(threshold)},
        **kwargs,
    )


def make_directional_task(
    name: str, direction: Sequence[float], **kwargs: Any
) -> TaskSpec:
    """Create one directional velocity task (dot product with unit direction)."""
    direction_arr = np.asarray(direction, dtype=np.float64).reshape(2)
    reward_fn = AntMazeDirectionalReward(direction=direction_arr, name=name)
    return TaskSpec(
        name=name,
        task_set="directional",
        reward_fn=reward_fn,
        metadata={"direction": direction_arr.tolist()},
        **kwargs,
    )


def make_simplex_task(seed: int, extent: Sequence[float] = ANTMAZE_GRID_EXTENT, **kwargs: Any) -> TaskSpec:
    """Create one seeded opensimplex height + velocity task."""
    reward_fn = AntMazeSimplexReward(seed=int(seed), extent=extent, name=f"simplex-{int(seed)}")
    return TaskSpec(
        name=f"simplex-{int(seed)}",
        task_set="simplex",
        reward_fn=reward_fn,
        metadata={"seed": int(seed)},
        **kwargs,
    )


def make_path_task(
    name: str,
    waypoints: Optional[Sequence[Sequence[float]]] = None,
    extent: Sequence[float] = ANTMAZE_GRID_EXTENT,
    corridor_width: float = DEFAULT_CORRIDOR_WIDTH,
    progress_weight: float = 0.0,
    extra_waypoints: Optional[Sequence[Sequence[float]]] = None,
    **kwargs: Any,
) -> TaskSpec:
    """Create one corridor/path task (``ant-path-center|loop|edges``)."""
    if waypoints is None:
        if name.endswith("center"):
            waypoints = PATH_CENTER_WAYPOINTS
        elif name.endswith("loop"):
            waypoints = PATH_LOOP_WAYPOINTS
        elif name.endswith("edges"):
            waypoints = PATH_EDGES_WAYPOINTS
        else:
            raise ValueError(f"unknown path task: {name}")
    reward_fn = AntMazePathReward(
        waypoints=waypoints,
        extent=extent,
        corridor_width=corridor_width,
        progress_weight=progress_weight,
        name=name,
    )
    success_fn = lambda states, fn=reward_fn: fn.success(states)  # noqa: E731
    metadata: Dict[str, Any] = {
        "waypoints": np.asarray(reward_fn.waypoints).tolist(),
        "corridor_width": float(corridor_width),
    }
    if extra_waypoints is not None:
        metadata["extra_waypoints"] = np.asarray(extra_waypoints).tolist()
    return TaskSpec(
        name=name,
        task_set="path",
        reward_fn=reward_fn,
        success_fn=success_fn,
        metadata=metadata,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Task registries
# --------------------------------------------------------------------------- #

#: Individual task names inserted into :data:`ANTMAZE_PREFIX` below.
_GOAL_TASK_NAMES = tuple(f"ant-{n}" for n, _ in ANTMAZE_GOAL_TASKS)
_DIRECTIONAL_TASK_NAMES = tuple(f"ant-{n}" for n, _ in DEFAULT_DIRECTIONS)
_SIMPLEX_TASK_NAMES = tuple(f"ant-simplex-{s}" for s in ANTMAZE_SIMPLEX_SEEDS)

#: Aggregate evaluation task sets reported in Table 1.
ANTMAZE_TASK_SETS: Dict[str, Tuple[str, ...]] = {
    "goal-reaching": _GOAL_TASK_NAMES,
    "directional": _DIRECTIONAL_TASK_NAMES,
    "random-simplex": _SIMPLEX_TASK_NAMES,
    "path": ANTMAZE_PATH_TASKS,
    "all": _GOAL_TASK_NAMES
    + _DIRECTIONAL_TASK_NAMES
    + _SIMPLEX_TASK_NAMES
    + ANTMAZE_PATH_TASKS,
    # aliases matching the names used in the paper's text / plan
    "ant-goal-reaching": _GOAL_TASK_NAMES,
    "ant-directional": _DIRECTIONAL_TASK_NAMES,
    "ant-random-simplex": _SIMPLEX_TASK_NAMES,
    "ant-path": ANTMAZE_PATH_TASKS,
}

#: Human readable aliases for individual tasks.
_TASK_ALIASES: Dict[str, str] = {
    "goal-bottom": "ant-goal-bottom",
    "goal-left": "ant-goal-left",
    "goal-top": "ant-goal-top",
    "goal-center": "ant-goal-center",
    "goal-right": "ant-goal-right",
    "vel_left": "ant-vel_left",
    "vel_up": "ant-vel_up",
    "vel_down": "ant-vel_down",
    "vel_right": "ant-vel_right",
    "path-center": "ant-path-center",
    "path-loop": "ant-path-loop",
    "path-edges": "ant-path-edges",
    "center": "ant-path-center",
    "loop": "ant-path-loop",
    "edges": "ant-path-edges",
}


def _all_task_names() -> Tuple[str, ...]:
    return tuple(ANTMAZE_TASK_SETS["all"])


def list_task_sets() -> Tuple[str, ...]:
    """Names of the available aggregate task sets."""
    return tuple(ANTMAZE_TASK_SETS.keys())


def task_names(task_set: Optional[str] = None) -> Tuple[str, ...]:
    """Names of the individual tasks in ``task_set`` (all tasks if ``None``)."""
    if task_set is None:
        return _all_task_names()
    key = task_set.lower()
    if key in ANTMAZE_TASK_SETS:
        return tuple(ANTMAZE_TASK_SETS[key])
    if key in {t.lower() for t in _all_task_names()}:
        return (key,)
    raise KeyError(f"unknown task set: {task_set!r} (available: {list(ANTMAZE_TASK_SETS)})")


def get_task(name: str, **kwargs: Any) -> TaskSpec:
    """Instantiate a single AntMaze task by name."""
    key = _TASK_ALIASES.get(name, name)
    key_lower = key.lower()

    for task_name, goal in ANTMAZE_GOAL_TASKS:
        if key_lower == f"ant-{task_name}".lower():
            return make_goal_task(f"ant-{task_name}", goal, **kwargs)
    for dir_name, direction in DEFAULT_DIRECTIONS:
        if key_lower == f"ant-{dir_name}".lower():
            return make_directional_task(f"ant-{dir_name}", direction, **kwargs)
    if key_lower.startswith("ant-simplex-"):
        seed = int(key_lower.rsplit("-", 1)[-1])
        return make_simplex_task(seed, **kwargs)
    if key_lower in {p.lower() for p in ANTMAZE_PATH_TASKS}:
        return make_path_task(f"ant-path-{key_lower.rsplit('-', 1)[-1]}", **kwargs)
    raise KeyError(f"unknown AntMaze task: {name!r}")


def build_task_set(
    task_set: str = "all",
    extent: Sequence[float] = ANTMAZE_GRID_EXTENT,
    corridor_width: float = DEFAULT_CORRIDOR_WIDTH,
    path_progress_weight: float = 0.0,
    directions: Optional[Sequence[Tuple[str, Sequence[float]]]] = None,
    goal_distance: float = ANTMAZE_GOAL_DISTANCE,
    simplex_seeds: Sequence[int] = ANTMAZE_SIMPLEX_SEEDS,
    **kwargs: Any,
) -> List[TaskSpec]:
    """Build the list of :class:`TaskSpec` for an aggregate task set.

    ``task_set`` may be any key of :data:`ANTMAZE_TASK_SETS` (e.g.
    ``"goal-reaching"``, ``"directional"``, ``"random-simplex"``, ``"path"``,
    ``"all"``) or the name of a single task (see :func:`get_task`).
    """
    key = task_set.lower()
    if key not in {k.lower() for k in ANTMAZE_TASK_SETS}:
        return [get_task(task_set, **kwargs)]

    canonical = next(k for k in ANTMAZE_TASK_SETS if k.lower() == key)
    tasks: List[TaskSpec] = []
    goal_set = canonical in ("goal-reaching", "ant-goal-reaching", "all")
    dir_set = canonical in ("directional", "ant-directional", "all")
    simplex_set = canonical in ("random-simplex", "ant-random-simplex", "all")
    path_set = canonical in ("path", "ant-path", "all")

    if goal_set:
        for name, goal in ANTMAZE_GOAL_TASKS:
            tasks.append(
                make_goal_task(
                    f"ant-{name}",
                    goal,
                    threshold=goal_distance,
                    coordinate_scale=tuple(extent),
                    **kwargs,
                )
            )
    if dir_set:
        for name, direction in (directions or DEFAULT_DIRECTIONS):
            tasks.append(
                make_directional_task(
                    f"ant-{name}", direction, coordinate_scale=tuple(extent), **kwargs
                )
            )
    if simplex_set:
        for seed in simplex_seeds:
            tasks.append(
                make_simplex_task(seed, extent=extent, coordinate_scale=tuple(extent), **kwargs)
            )
    if path_set:
        tasks.append(
            make_path_task(
                "ant-path-center",
                extent=extent,
                corridor_width=corridor_width,
                progress_weight=path_progress_weight,
                coordinate_scale=tuple(extent),
                **kwargs,
            )
        )
        tasks.append(
            make_path_task(
                "ant-path-loop",
                extent=extent,
                corridor_width=corridor_width,
                progress_weight=path_progress_weight,
                coordinate_scale=tuple(extent),
                **kwargs,
            )
        )
        tasks.append(
            make_path_task(
                "ant-path-edges",
                extent=extent,
                corridor_width=corridor_width,
                progress_weight=path_progress_weight,
                coordinate_scale=tuple(extent),
                **kwargs,
            )
        )
    return tasks


def build_tasks(task_set: str = "all", **kwargs: Any) -> List[TaskSpec]:
    """Alias of :func:`build_task_set`."""
    return build_task_set(task_set=task_set, **kwargs)


# --------------------------------------------------------------------------- #
# Environment utilities
# --------------------------------------------------------------------------- #


def make_antmaze_env(
    env_name: str = ANTMAZE_ENV_NAME,
    seed: Optional[int] = None,
    center_start: bool = True,
    max_episode_steps: Optional[int] = ANTMAZE_MAX_EPISODE_STEPS,
    **kwargs: Any,
) -> Any:
    """Create the D4RL AntMaze environment.

    Requires ``gym`` (or ``gymnasium``) and ``d4rl``; ``import d4rl`` triggers
    the environment registration.  ``max_episode_steps`` is enforced through
    :class:`fre.envs.reward_wrappers.TimeLimitWrapper` so that the evaluation
    horizon of 2000 steps is respected even when the reward is custom.
    """
    import gym  # type: ignore  # local import: heavy dependency

    try:  # pragma: no cover - registers the D4RL environments
        import d4rl  # noqa: F401
    except Exception:
        try:
            import d4rl.gym_mujoco  # noqa: F401
        except Exception:  # pragma: no cover
            pass

    env = gym.make(env_name, **kwargs)
    if seed is not None:
        try:
            env.seed(int(seed))
        except Exception:
            try:
                env.reset(seed=int(seed))
            except Exception:
                pass
        try:
            env.action_space.seed(int(seed))
        except Exception:
            pass
    env.center_start = bool(center_start)  # type: ignore[attr-defined]
    if max_episode_steps is not None:
        from fre.envs.reward_wrappers import TimeLimitWrapper

        env = TimeLimitWrapper(env, max_episode_steps=int(max_episode_steps))
    return env


def get_sim_state(env: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Best-effort ``(qpos, qvel)`` extraction from a MuJoCo env."""
    unwrapped = getattr(env, "unwrapped", env)
    # mujoco_py / gym-mujoco
    sim = getattr(unwrapped, "sim", None)
    data = getattr(sim, "data", None) if sim is not None else None
    if data is None:
        data = getattr(unwrapped, "data", None)
    if data is not None:
        qpos = getattr(data, "qpos", None)
        qvel = getattr(data, "qvel", None)
        if qpos is not None and qvel is not None:
            return np.array(qpos, dtype=np.float64), np.array(qvel, dtype=np.float64)
    # dm_control style
    physics = getattr(unwrapped, "physics", None)
    if physics is not None:
        try:
            qpos = np.array(physics.data.qpos, dtype=np.float64)
            qvel = np.array(physics.data.qvel, dtype=np.float64)
            return qpos, qvel
        except Exception:
            pass
    # gymnasium mujoco
    try:
        state = unwrapped.unwrapped.data.qpos, unwrapped.unwrapped.data.qvel
        return np.array(state[0], dtype=np.float64), np.array(state[1], dtype=np.float64)
    except Exception:
        return None


def set_sim_state(env: Any, qpos: np.ndarray, qvel: np.ndarray) -> bool:
    """Best-effort ``(qpos, qvel)`` assignment on a MuJoCo env."""
    unwrapped = getattr(env, "unwrapped", env)
    for attempt in (
        lambda: unwrapped.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)),
        lambda: unwrapped.sim.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)),
    ):
        try:
            attempt()
            return True
        except Exception:
            continue
    # direct mujoco[_py] assignment
    sim = getattr(unwrapped, "sim", None)
    data = getattr(sim, "data", None) if sim is not None else None
    if data is None:
        data = getattr(unwrapped, "data", None)
    if data is not None and hasattr(data, "qpos") and hasattr(data, "qvel"):
        try:
            data.qpos[:] = np.asarray(qpos, dtype=np.float64)
            data.qvel[:] = np.asarray(qvel, dtype=np.float64)
            if sim is not None and hasattr(sim, "forward"):
                sim.forward()
            return True
        except Exception:
            pass
    return False


def reset_to_center(
    env: Any,
    xy_center: Optional[Sequence[float]] = None,
    **reset_kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Reset the env with the ant placed in the center of the maze.

    Matches Appendix C.1: "*The ant robot is placed in the center of the maze to
    allow for more diverse behavior, in comparison to the original start
    position in the bottom-left.*"
    """
    from fre.envs.reward_wrappers import call_env_reset

    obs, info = call_env_reset(env, **reset_kwargs)
    center = (
        np.asarray(xy_center, dtype=np.float64).reshape(2)
        if xy_center is not None
        else maze_center_xy(env)
    )
    state = get_sim_state(env)
    if state is not None:
        qpos, qvel = state
        if qpos.shape[0] >= 2:
            qpos = np.array(qpos, dtype=np.float64)
            qpos[:2] = center
            if qvel.shape[0] >= 2:
                qvel = np.zeros_like(qvel)
            if set_sim_state(env, qpos, qvel):
                obs = _current_observation(env)
                info = dict(info) if isinstance(info, dict) else {}
                info["center_start_xy"] = center.tolist()
    if isinstance(info, dict):
        info.setdefault("maze_center_xy", center.tolist())
    return obs, info


def _current_observation(env: Any) -> Any:
    """Read the current observation after a manual state assignment."""
    for getter in (
        lambda: env.unwrapped._get_obs(),
        lambda: env.unwrapped._get_observation(),
        lambda: env.unwrapped.get_obs(),
    ):
        try:
            return getter()
        except Exception:
            continue
    return None


def make_antmaze_task_env(
    task: Union[TaskSpec, str],
    env_name: str = ANTMAZE_ENV_NAME,
    seed: Optional[int] = None,
    center_start: bool = True,
    max_episode_steps: Optional[int] = ANTMAZE_MAX_EPISODE_STEPS,
    env: Any = None,
    count_success_as_done: bool = True,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Wrap an AntMaze env with the reward function of ``task``.

    Returns a :class:`fre.envs.reward_wrappers.RewardWrapper` whose
    ``reward_fn`` is the task's state reward function.  The ant is placed in the
    center of the maze at reset (Appendix C.1).
    """
    from fre.envs.reward_wrappers import wrap_env

    spec = get_task(task) if isinstance(task, str) else task
    if env is None:
        env = make_antmaze_env(env_name=env_name, seed=seed, center_start=center_start)
    success_fn = spec.success_fn_for_wrapper()
    wrapped = wrap_env(
        env,
        reward_fn=spec.reward_fn,
        success_fn=success_fn,
        max_episode_steps=max_episode_steps,
        domain="antmaze",
        count_success_as_done=count_success_as_done,
        name=name or spec.name,
        **kwargs,
    )
    if center_start:
        original_reset = wrapped.reset

        def _reset(**reset_kwargs: Any):  # pragma: no cover - thin wrapper
            obs, info = reset_to_center(wrapped.env, **reset_kwargs)
            # propagate through the wrapper bookkeeping
            try:
                wrapped._episode_steps = 0
            except Exception:
                pass
            return obs, info

        try:
            wrapped.reset = _reset  # type: ignore[assignment]
        except Exception:
            pass
    return wrapped


# --------------------------------------------------------------------------- #
# Encoder-side preprocessing / zero-shot encoding samples
# --------------------------------------------------------------------------- #


def antmaze_encoder_states(
    states: Any, num_bins: int = NUM_XY_BINS, discretize: bool = True
) -> np.ndarray:
    """Prepare AntMaze states for the FRE encoder (32-bin XY discretization)."""
    array = np.asarray(states, dtype=np.float32)
    if not discretize:
        return array
    from fre.data.preprocessing import discretize_antmaze_xy

    return discretize_antmaze_xy(array, num_bins=num_bins)


def encoding_samples_for_task(
    task: Union[TaskSpec, str],
    dataset_states: Any,
    num_samples: int = 32,
    discretize_xy: bool = True,
    num_bins: int = NUM_XY_BINS,
    rng: Optional[Any] = None,
    replace_last_with_goal: bool = False,
    device: Any = None,
) -> Dict[str, Any]:
    """Sample ``(s, eta(s))`` pairs of ``task`` for zero-shot encoding.

    ``K = 32`` states are drawn uniformly from ``D`` (the offline dataset) and
    labeled with the task reward function, exactly as done for the unsupervised
    priors.  Returns a dict with ``states``, ``rewards`` (original, ``(K, 1)``),
    ``normalized_rewards`` (per-reward-function min/max rescaled to ``[0, 1]``)
    and ``done`` (goal achievement mask, ``False`` for non-goal tasks).
    """
    spec = get_task(task) if isinstance(task, str) else task
    states_np = np.asarray(dataset_states, dtype=np.float32)
    if states_np.ndim != 2:
        raise ValueError("dataset_states must be a 2-D array of shape (N, state_dim)")
    rng = rng if rng is not None else np.random.default_rng(0)
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    idx = rng.choice(states_np.shape[0], size=int(num_samples), replace=False)
    sample_states = states_np[idx]
    if replace_last_with_goal and spec.goal is not None:
        sample_states = np.array(sample_states, copy=True)
        goal_state = np.array(sample_states[-1], copy=True)
        goal_state[:2] = np.asarray(spec.goal, dtype=np.float32).reshape(2)
        sample_states[-1] = goal_state

    rewards = np.asarray(spec.reward(sample_states), dtype=np.float32).reshape(-1, 1)
    done = np.asarray(spec.is_success(sample_states), dtype=bool).reshape(-1, 1)
    encoder_states = antmaze_encoder_states(
        sample_states, num_bins=num_bins, discretize=discretize_xy
    )
    # per-reward-function min/max rescaling (as in RewardEmbedding)
    r_min = float(rewards.min())
    r_max = float(rewards.max())
    if r_max - r_min > 1e-6:
        normalized = (rewards - r_min) / (r_max - r_min)
    else:
        normalized = np.zeros_like(rewards)

    result: Dict[str, Any] = {
        "states": sample_states,
        "encoder_states": encoder_states,
        "rewards": rewards,
        "normalized_rewards": normalized.astype(np.float32),
        "done": done,
        "indices": idx,
        "task": spec.name,
        "task_set": spec.task_set,
    }
    if torch is not None:
        dev = device if device is not None else torch.device("cpu")
        result["states_tensor"] = torch.as_tensor(sample_states, dtype=torch.float32, device=dev)
        result["encoder_states_tensor"] = torch.as_tensor(
            encoder_states, dtype=torch.float32, device=dev
        )
        result["rewards_tensor"] = torch.as_tensor(rewards, dtype=torch.float32, device=dev)
        result["mask_tensor"] = torch.ones(
            (int(num_samples), 1), dtype=torch.bool, device=dev
        )
    return result


def encoding_samples_from_env(
    task: Union[TaskSpec, str],
    env: Any,
    num_samples: int = 32,
    discretize_xy: bool = True,
    num_bins: int = NUM_XY_BINS,
    policy: Optional[Callable[[Any], Any]] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Collect ``(s, eta(s))`` samples by rolling out ``env`` with a random policy."""
    spec = get_task(task) if isinstance(task, str) else task
    rng = np.random.default_rng(0 if seed is None else int(seed))
    from fre.envs.reward_wrappers import call_env_reset, call_env_step

    states: List[np.ndarray] = []
    for _ in range(int(num_samples)):
        try:
            obs, _info = call_env_reset(env)
        except Exception:
            obs = env.reset()
        if policy is not None:
            action = policy(obs)
        else:
            try:
                action = env.action_space.sample()
            except Exception:
                action = rng.uniform(-1.0, 1.0, size=8).astype(np.float32)
        states.append(np.asarray(obs, dtype=np.float32))
        try:
            call_env_step(env, action)
        except Exception:
            pass
    states_np = np.stack(states, axis=0)
    return encoding_samples_for_task(
        spec,
        states_np,
        num_samples=int(num_samples),
        discretize_xy=discretize_xy,
        num_bins=num_bins,
        rng=rng,
        **kwargs,
    )
