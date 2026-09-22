"""AntMaze zero-shot evaluation task suite for FRE (paper Section 5, Appendix C.1
and the addendum section "Ant Maze evaluation tasks").

This module defines the *online* evaluation tasks used to score a frozen,
z-conditioned policy on top of the D4RL ``antmaze-large-diverse-v2`` dataset.

Task families (exactly as described in the paper / addendum):

* ``ant-goal-reaching``: 5 hand-crafted fixed reward functions that reward the
  agent for reaching a goal location.  ``reward = -1`` for every timestep the
  goal is not achieved, ``0`` otherwise.  The goal is considered reached when
  the agent is within a distance of ``2`` of the target position
  (Appendix C.1).  Goal locations on the (X, Y) grid with origin at the bottom
  left: bottom (28, 0), left (0, 15), top (35, 24), center (12, 24),
  right (33, 16).  The reported score is the average over the 5 tasks.
* ``ant-directional``: 4 tasks, each specifying a target velocity in the (X, Y)
  plane -- left (-1, 0), up (0, 1), down (0, -1), right (1, 0).  The reward is
  the dot product between the agent's actual (X, Y) velocity and the target
  velocity.  The reported score is the average over the 4 (addendum says 5,
  but only 4 directions are listed -- we follow the explicit list).
* ``ant-random-simplex``: 5 seeded tasks defined by a random 2D noise "height
  map" plus local velocity preferences generated via ``opensimplex``.  The
  agent receives a baseline reward of ``-1`` per step, a bonus for standing in
  higher "height" regions, and an additional bonus for moving in the local
  "preferred" velocity direction indicated by the noise field.  Scores are
  averaged over the 5 fixed seeds (1..5).
* ``ant-path-{loop,edges,center}``: reward the agent for moving along a
  hand-crafted loop around the grid, along the edges of the grid, and along a
  central corridor respectively.

Agents (FRE, GC-IQL, GC-BC, OPAL) receive *discretized* X and Y coordinates
(32 bins) as part of the observation (Appendix C.1).  The helper
:func:`discretize_antmaze_observations` performs this preprocessing and must be
applied consistently to the offline dataset, the encoder (state, reward) pairs
and the online observations.

Online evaluation uses a *maximum* length of 2000 steps per trajectory and the
ant robot is placed at the centre of the maze (Appendix C.1), in contrast to
the original bottom-left start position.  Returns are normalised between 0 and
100 (Section 5); see :meth:`AntMazeTask.normalize_return`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Constants taken verbatim from the paper / addendum
# --------------------------------------------------------------------------------------

#: Fixed hand-crafted goal locations (X, Y), addendum "ant-goal-reaching".
ANTMAZE_GOAL_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "goal-bottom": (28.0, 0.0),
    "goal-left": (0.0, 15.0),
    "goal-top": (35.0, 24.0),
    "goal-center": (12.0, 24.0),
    "goal-right": (33.0, 16.0),
}

#: Target velocities in the (X, Y) plane, addendum "ant-directional".
ANTMAZE_DIRECTION_VECTORS: Dict[str, Tuple[float, float]] = {
    "vel_left": (-1.0, 0.0),
    "vel_up": (0.0, 1.0),
    "vel_down": (0.0, -1.0),
    "vel_right": (1.0, 0.0),
}

#: Fixed opensimplex seeds used by the random-simplex tasks.
ANTMAZE_SIMPLEX_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: Hand-designed path tasks.
ANTMAZE_PATH_KINDS: Tuple[str, ...] = ("loop", "edges", "center")

#: Goal-reaching success radius (Appendix C.1: "within a distance of 2").
ANTMAZE_GOAL_THRESHOLD: float = 2.0

#: X / Y coordinates are discretized into this many bins (Appendix C.1).
ANTMAZE_DISCRETIZE_BINS: int = 32

#: Range used for the 32-bin discretization.  Matches
#: ``fre.config.envs.antmaze_discretize_xy`` (low=0, high=40).
ANTMAZE_DISCRETIZE_LOW: float = 0.0
ANTMAZE_DISCRETIZE_HIGH: float = 40.0

#: Online evaluation length (Appendix C.1, "maximum length of 2000 steps").
ANTMAZE_MAX_EPISODE_STEPS: int = 2000

#: Number of (state, reward) pairs handed to the encoder at zero-shot test time.
ANTMAZE_ENCODER_SAMPLES: int = 32

#: Default maze coordinate extents used to lay out hand-designed paths.  Derived
#: from the goal locations above -- the paper does not state them explicitly, so
#: these are the smallest box containing all fixed goals (plus a margin).
ANTMAZE_MAP_MIN: Tuple[float, float] = (0.0, 0.0)
ANTMAZE_MAP_MAX: Tuple[float, float] = (40.0, 28.0)

# --------------------------------------------------------------------------------------
# Observation helpers
# --------------------------------------------------------------------------------------


def extract_antmaze_xy(obs: np.ndarray) -> np.ndarray:
    """Return the (X, Y) position of the ant from a D4RL AntMaze observation.

    The AntMaze observation is ``qpos (15) + qvel (14)``; the free-joint position
    is the first two entries and the corresponding velocities the first two
    entries of ``qvel``.
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    if obs.shape[0] < 2:
        raise ValueError(f"AntMaze observation too short to contain XY: {obs.shape}")
    return obs[:2].copy()


def extract_antmaze_velocity(obs: np.ndarray) -> np.ndarray:
    """Return the (dX, dY) velocity of the ant from a D4RL AntMaze observation."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    if obs.shape[0] < 17:
        raise ValueError(
            "AntMaze observation too short to contain XY velocities "
            f"(expected qpos+qvel layout, got {obs.shape[0]})"
        )
    return obs[15:17].copy()


def antmaze_discretize_xy(
    xy: np.ndarray,
    bins: int = ANTMAZE_DISCRETIZE_BINS,
    low: float = ANTMAZE_DISCRETIZE_LOW,
    high: float = ANTMAZE_DISCRETIZE_HIGH,
    to_bin_center: bool = True,
) -> np.ndarray:
    """Discretize (X, Y) coordinates into ``bins`` bins in ``[low, high]``.

    The FRE / GC-IQL / GC-BC / OPAL preprocessing snaps each coordinate to a bin
    of size ``(high - low) / bins`` (Appendix C.1).  When ``to_bin_center`` the
    returned value is the bin centre (a value the agent can also observe for
    states that were never in the dataset), otherwise the integer bin index.
    """
    xy = np.asarray(xy, dtype=np.float64)
    width = (high - low) / float(bins)
    idx = np.floor((xy - low) / width)
    idx = np.clip(idx, 0, bins - 1)
    if to_bin_center:
        return low + (idx + 0.5) * width
    return idx.astype(np.int64)


def discretize_antmaze_observations(
    observations: np.ndarray,
    bins: int = ANTMAZE_DISCRETIZE_BINS,
    low: float = ANTMAZE_DISCRETIZE_LOW,
    high: float = ANTMAZE_DISCRETIZE_HIGH,
    in_place: bool = False,
) -> np.ndarray:
    """Apply the 32-bin X/Y discretization used by all compared agents.

    Parameters
    ----------
    observations:
        Array of shape ``(..., obs_dim)`` with the D4RL AntMaze layout
        (``qpos`` followed by ``qvel``); at least 17 dimensions are required.
    """
    obs = np.asarray(observations, dtype=np.float64)
    if obs.shape[-1] < 17:
        raise ValueError(
            "Expected observations with at least 17 dims (qpos 15 + qvel 2), got "
            f"{obs.shape}"
        )
    out = obs if in_place else obs.copy()
    out[..., :2] = antmaze_discretize_xy(out[..., :2], bins=bins, low=low, high=high)
    return out


# --------------------------------------------------------------------------------------
# Task definitions
# --------------------------------------------------------------------------------------


class AntMazeTask:
    """Base class for a single AntMaze evaluation reward function.

    Subclasses implement :meth:`reward` (computed from the *next* observation and
    the action) and :meth:`success`.  ``min_return`` / ``max_return`` describe the
    theoretical return range used to normalise scores between 0 and 100 (the paper
    does not state the normalisation constants explicitly, so each task declares
    the tightest range implied by its own reward definition).
    """

    name: str = "ant-task"
    max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS
    min_return: float = 0.0
    max_return: float = 0.0

    # -- reward interface ---------------------------------------------------------
    def reward(self, obs, next_obs, action=None, info: Optional[dict] = None) -> float:
        raise NotImplementedError

    def success(self, obs, info: Optional[dict] = None) -> bool:
        return False

    def done(self, obs, info: Optional[dict] = None) -> bool:
        """Whether the episode should terminate early (never for AntMaze)."""
        return False

    def reset(self, env=None) -> dict:
        """Per-episode reset hook (returns an info dict merged into the step info)."""
        return {}

    def reward_from_state(self, state: np.ndarray) -> float:
        """Reward of a single *offline* state, used to build encoder pairs.

        Defaults to evaluating :meth:`reward` with ``next_obs = state`` and no
        action, which is exact for the state-only reward functions.
        """
        return float(self.reward(state, state, None))

    # -- normalisation ------------------------------------------------------------
    def normalize_return(self, total_return: float) -> float:
        """Linearly map a raw episode return onto the paper's [0, 100] scale."""
        lo, hi = float(self.min_return), float(self.max_return)
        if hi <= lo:
            return 0.0
        return 100.0 * (float(total_return) - lo) / (hi - lo)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


class GoalReachingTask(AntMazeTask):
    """``ant-goal-reaching`` reward: ``-1`` until the goal is reached, then ``0``.

    The goal is reached when the (discretized) agent position is within
    ``threshold`` of the target position (Appendix C.1 uses a distance of 2).
    """

    def __init__(
        self,
        goal_xy: Sequence[float],
        label: str = "goal",
        threshold: float = ANTMAZE_GOAL_THRESHOLD,
        discretize: bool = True,
        bins: int = ANTMAZE_DISCRETIZE_BINS,
        max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
        reward_on_success: float = 0.0,
        reward_off_success: float = -1.0,
    ) -> None:
        self.goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        self.label = label
        self.threshold = float(threshold)
        self.discretize = bool(discretize)
        self.bins = int(bins)
        self.max_episode_steps = int(max_episode_steps)
        self.reward_on_success = float(reward_on_success)
        self.reward_off_success = float(reward_off_success)
        self.name = f"ant-goal-reaching-{label}"
        # Reward is 0 on success and -1 otherwise for at most max_episode_steps.
        self.min_return = self.reward_off_success * self.max_episode_steps
        self.max_return = self.reward_on_success * self.max_episode_steps
        if self.reward_off_success == self.reward_on_success:  # degenerate guard
            self.max_return = self.reward_off_success * self.max_episode_steps

    # -- helpers ------------------------------------------------------------------
    def _position(self, obs) -> np.ndarray:
        xy = extract_antmaze_xy(obs)
        if self.discretize:
            xy = antmaze_discretize_xy(xy, bins=self.bins)
        return xy

    def goal_position(self) -> np.ndarray:
        goal = self.goal_xy
        if self.discretize:
            goal = antmaze_discretize_xy(goal, bins=self.bins)
        return np.asarray(goal, dtype=np.float64)

    def distance(self, obs) -> float:
        return float(np.linalg.norm(self._position(obs) - self.goal_position()))

    # -- reward interface ---------------------------------------------------------
    def reward(self, obs, next_obs, action=None, info: Optional[dict] = None) -> float:
        if self.distance(next_obs) <= self.threshold:
            return self.reward_on_success
        return self.reward_off_success

    def success(self, obs, info: Optional[dict] = None) -> bool:
        return self.distance(obs) <= self.threshold

    def reward_from_state(self, state: np.ndarray) -> float:
        if self.distance(state) <= self.threshold:
            return self.reward_on_success
        return self.reward_off_success

    def encoder_pairs(
        self, dataset=None, num_samples: int = ANTMAZE_ENCODER_SAMPLES, rng=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(state, reward) pairs used to encode ``z`` for this task.

        The paper guarantees at least one encoder sample is exactly at the goal
        state so that the encoding is informative; we prepend the goal state.
        """
        states = _sample_task_states(self, dataset, num_samples, rng)
        states[0] = self._dataset_state_for(self.goal_xy, dataset, states)
        rewards = np.asarray([self.reward_from_state(s) for s in states], dtype=np.float32)
        return states, rewards

    def _dataset_state_for(self, xy, dataset, fallback_states) -> np.ndarray:
        """Find the closest dataset state to ``xy`` (so the goal token looks like data)."""
        raw = _dataset_raw_states(dataset)
        if raw is None or raw.shape[0] == 0:
            state = np.array(fallback_states[0], dtype=np.float64, copy=True)
            state[:2] = self.goal_position()
            return state
        positions = raw[:, :2]
        dists = np.linalg.norm(positions - np.asarray(xy, dtype=np.float64).reshape(1, 2), axis=1)
        return np.array(raw[int(np.argmin(dists))], dtype=np.float64, copy=True)


class DirectionalTask(AntMazeTask):
    """``ant-directional`` reward: dot product of the agent velocity and a target.

    ``reward = <v_xy, target_dir>`` (addendum: "grant higher reward the closer it
    is to the target velocity, using a simple dot product").  The optional
    ``reward_clip`` bounds per-step rewards to ``[0, reward_clip]``; the paper is
    silent about clipping, so the default is ``None`` (raw dot product) and the
    normalisation range is derived from it.
    """

    def __init__(
        self,
        direction: Sequence[float],
        label: str = "dir",
        max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
        reward_clip: Optional[float] = None,
        max_velocity: float = 2.0,
    ) -> None:
        self.direction = np.asarray(direction, dtype=np.float64).reshape(2)
        self.label = label
        self.max_episode_steps = int(max_episode_steps)
        self.reward_clip = None if reward_clip is None else float(reward_clip)
        self.max_velocity = float(max_velocity)
        self.name = f"ant-directional-{label}"
        hi = self.max_velocity * float(np.linalg.norm(self.direction))
        if self.reward_clip is not None:
            hi = min(hi, self.reward_clip)
        self.max_return = hi * self.max_episode_steps
        self.min_return = -self.max_return

    def reward(self, obs, next_obs, action=None, info: Optional[dict] = None) -> float:
        vel = extract_antmaze_velocity(next_obs)
        r = float(np.dot(vel, self.direction))
        if self.reward_clip is not None:
            r = float(np.clip(r, -self.reward_clip, self.reward_clip))
        return r

    def reward_from_state(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape[0] < 17:
            return 0.0
        r = float(np.dot(state[15:17], self.direction))
        if self.reward_clip is not None:
            r = float(np.clip(r, -self.reward_clip, self.reward_clip))
        return r


class RandomSimplexTask(AntMazeTask):
    """``ant-random-simplex``: procedural noise height map + velocity preference.

    Reward per step (addendum): ``-1`` baseline, plus a bonus for standing in
    higher "height" regions of an ``opensimplex`` noise field, plus a bonus for
    moving in the local "preferred" velocity direction indicated by the noise
    field.  Both bonuses are expressed in ``[0, 1]`` so a single step reward lies
    in ``[-1, 2]``; the return is normalised against ``[-T, 2T]``.
    """

    def __init__(
        self,
        seed: int = 1,
        max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
        resolution: float = 4.0,
        velocity_bonus_scale: float = 1.0,
        height_bonus_scale: float = 1.0,
        baseline: float = -1.0,
        max_velocity: float = 2.0,
        noise_scale: float = 1.0,
    ) -> None:
        self.seed = int(seed)
        self.name = f"ant-random-simplex-{self.seed}"
        self.max_episode_steps = int(max_episode_steps)
        self.resolution = float(resolution)
        self.velocity_bonus_scale = float(velocity_bonus_scale)
        self.height_bonus_scale = float(height_bonus_scale)
        self.baseline = float(baseline)
        self.max_velocity = float(max_velocity)
        self.noise_scale = float(noise_scale)
        self.min_return = self.baseline * self.max_episode_steps
        self.max_return = (
            self.baseline + self.height_bonus_scale + self.velocity_bonus_scale
        ) * self.max_episode_steps
        self._field = None  # lazily built (numpy only, no opensimplex dependency)
        self._preferred = None

    # -- noise field --------------------------------------------------------------
    def _grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xs = np.arange(ANTMAZE_MAP_MIN[0], ANTMAZE_MAP_MAX[0] + 1e-9, self.resolution)
        ys = np.arange(ANTMAZE_MAP_MIN[1], ANTMAZE_MAP_MAX[1] + 1e-9, self.resolution)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return xs, ys, gx, gy

    def build_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(xs, ys, height, pref_vx, pref_vy)`` for this task's seed.

        Uses :mod:`opensimplex` when available (as in the paper); otherwise falls
        back to a smooth Gaussian-random-field interpolation built from
        ``numpy.random.default_rng(seed)`` so the task remains runnable offline.
        """
        if self._field is not None:
            return self._field
        xs, ys, gx, gy = self._grid()
        height = _noise2(self.seed, gx * self.noise_scale, gy * self.noise_scale)
        # Normalise the noise to [0, 1] so the height bonus matches the paper's
        # qualitative description ("bonus if it stands in higher regions").
        height = (height - height.min()) / max(height.max() - height.min(), 1e-8)
        # Preferred velocity = direction of steepest ascent of the height field.
        dhdx, dhdy = np.gradient(height, xs, ys, edge_order=1)
        norm = np.sqrt(dhdx ** 2 + dhdy ** 2)
        norm[norm < 1e-8] = 1.0
        pref_vx, pref_vy = dhdx / norm, dhdy / norm
        self._field = (xs, ys, height, pref_vx, pref_vy)
        return self._field

    def _lookup(self, xy: np.ndarray) -> Tuple[float, np.ndarray]:
        xs, ys, height, pref_vx, pref_vy = self.build_field()
        ix = int(np.clip(np.searchsorted(xs, xy[0]) - 1, 0, len(xs) - 1))
        iy = int(np.clip(np.searchsorted(ys, xy[1]) - 1, 0, len(ys) - 1))
        return float(height[ix, iy]), np.array([pref_vx[ix, iy], pref_vy[ix, iy]])

    # -- reward interface ---------------------------------------------------------
    def reward(self, obs, next_obs, action=None, info: Optional[dict] = None) -> float:
        xy = extract_antmaze_xy(next_obs)
        vel = extract_antmaze_velocity(next_obs)
        h, preferred = self._lookup(xy)
        height_bonus = self.height_bonus_scale * h
        alignment = float(np.dot(vel, preferred)) / max(self.max_velocity, 1e-8)
        velocity_bonus = self.velocity_bonus_scale * float(np.clip(alignment, 0.0, 1.0))
        return self.baseline + height_bonus + velocity_bonus

    def reward_from_state(self, state: np.ndarray) -> float:
        """State-only reward (used for the encoder tokens).

        Velocity is unavailable for a single offline state, so only the baseline
        and the height bonus are used here; this is a documented approximation
        when building the 32 (state, reward) pairs for the encoder.
        """
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        h, _ = self._lookup(state[:2])
        return self.baseline + self.height_bonus_scale * h


class PathTask(AntMazeTask):
    """``ant-path-{loop,edges,center}``: reward for staying on a hand-crafted path.

    Each task defines a polyline inside the maze; the per-step reward is ``1``
    when the ant is within ``tolerance`` of the polyline and ``0`` otherwise (the
    paper only states that the agent is "rewarded for moving along" the corridor
    / loop / edges, so this is the simplest faithful realisation and is
    documented as a paper-unspecified default).
    """

    def __init__(
        self,
        kind: str = "center",
        tolerance: float = 3.0,
        max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
        waypoints: Optional[Sequence[Sequence[float]]] = None,
        closed: Optional[bool] = None,
    ) -> None:
        if kind not in ANTMAZE_PATH_KINDS and waypoints is None:
            raise ValueError(f"Unknown path kind {kind!r}; expected one of {ANTMAZE_PATH_KINDS}")
        self.kind = kind
        self.tolerance = float(tolerance)
        self.max_episode_steps = int(max_episode_steps)
        if waypoints is None:
            waypoints, default_closed = default_path_waypoints(kind)
        else:
            default_closed = False
        self.waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
        self.closed = bool(default_closed if closed is None else closed)
        self.name = f"ant-path-{kind}"
        # reward in {0, 1} each step.
        self.min_return = 0.0
        self.max_return = 1.0 * self.max_episode_steps

    # -- geometry -----------------------------------------------------------------
    def distance_to_path(self, xy: Sequence[float]) -> float:
        return float(point_to_polyline_distance(np.asarray(xy, dtype=np.float64), self.waypoints, self.closed))

    def reward(self, obs, next_obs, action=None, info: Optional[dict] = None) -> float:
        xy = extract_antmaze_xy(next_obs)
        return 1.0 if self.distance_to_path(xy) <= self.tolerance else 0.0

    def reward_from_state(self, state: np.ndarray) -> float:
        xy = np.asarray(state, dtype=np.float64).reshape(-1)[:2]
        return 1.0 if self.distance_to_path(xy) <= self.tolerance else 0.0

    def success(self, obs, info: Optional[dict] = None) -> bool:
        return self.distance_to_path(extract_antmaze_xy(obs)) <= self.tolerance


def default_path_waypoints(kind: str) -> Tuple[List[List[float]], bool]:
    """Hand-crafted waypoints for the three path tasks (paper-unspecified layout).

    Returns ``(waypoints, closed)``.  The layouts are derived from the maze
    extents (:data:`ANTMAZE_MAP_MIN` / :data:`ANTMAZE_MAP_MAX`) and the fixed
    evaluation goals so that ``center`` runs along the middle of the maze,
    ``edges`` hugs the boundary corridors and ``loop`` traces a rectangle in
    between.
    """
    (x0, y0), (x1, y1) = ANTMAZE_MAP_MIN, ANTMAZE_MAP_MAX
    if kind == "center":
        ym = 0.5 * (y0 + y1)
        waypoints = [[x0 + 4.0, ym], [x1 - 4.0, ym]]
        return waypoints, False
    if kind == "edges":
        inset = 2.0
        waypoints = [
            [x0 + inset, y0 + inset],
            [x1 - inset, y0 + inset],
            [x1 - inset, y1 - inset],
            [x0 + inset, y1 - inset],
        ]
        return waypoints, True
    if kind == "loop":
        inset = 6.0
        waypoints = [
            [x0 + inset, y0 + inset],
            [x1 - inset, y0 + inset],
            [x1 - inset, y1 - inset],
            [x0 + inset, y1 - inset],
        ]
        return waypoints, True
    raise ValueError(f"Unknown path kind {kind!r}")


def point_to_polyline_distance(point: np.ndarray, waypoints: np.ndarray, closed: bool = False) -> float:
    """Minimum Euclidean distance from ``point`` to a polyline."""
    point = np.asarray(point, dtype=np.float64).reshape(2)
    pts = np.asarray(waypoints, dtype=np.float64)
    if pts.shape[0] == 1:
        return float(np.linalg.norm(point - pts[0]))
    best = math.inf
    n_seg = pts.shape[0] if closed else pts.shape[0] - 1
    for i in range(n_seg):
        a = pts[i]
        b = pts[(i + 1) % pts.shape[0]]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            d = float(np.linalg.norm(point - a))
        else:
            t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
            d = float(np.linalg.norm(point - (a + t * ab)))
        best = min(best, d)
    return float(best)


# --------------------------------------------------------------------------------------
# Noise helper (opensimplex with a numpy fallback)
# --------------------------------------------------------------------------------------


def _noise2(seed: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2D simplex noise; falls back to a smooth Gaussian field without opensimplex."""
    try:  # pragma: no cover - depends on optional dependency
        from opensimplex import OpenSimplex  # type: ignore

        gen = OpenSimplex(seed=int(seed))
        flat = np.asarray(
            [gen.noise2(float(a), float(b)) for a, b in zip(np.ravel(x), np.ravel(y))],
            dtype=np.float64,
        )
        return flat.reshape(np.shape(x))
    except Exception:
        # Deterministic smooth field: low-resolution Gaussian noise upsampled with
        # bilinear interpolation.  Only used when opensimplex is unavailable.
        return _smooth_gaussian_field(seed, np.shape(x))


def _smooth_gaussian_field(seed: int, shape: Tuple[int, ...], resolution: int = 6) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    coarse = rng.standard_normal((resolution, resolution))
    n0, n1 = shape
    xi = np.linspace(0, resolution - 1, n0)
    yi = np.linspace(0, resolution - 1, n1)
    x0 = np.clip(np.floor(xi).astype(int), 0, resolution - 2)
    y0 = np.clip(np.floor(yi).astype(int), 0, resolution - 2)
    tx = (xi - x0)[:, None]
    ty = (yi - y0)[None, :]
    c00 = coarse[x0][:, y0]
    c01 = coarse[x0][:, y0 + 1]
    c10 = coarse[x0 + 1][:, y0]
    c11 = coarse[x0 + 1][:, y0 + 1]
    top = c00 * (1 - tx) + c10 * tx
    bottom = c01 * (1 - tx) + c11 * tx
    return top * (1 - ty) + bottom * ty


# --------------------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------------------


def _dataset_raw_states(dataset) -> Optional[np.ndarray]:
    """Best-effort access to a dataset's raw (un-augmented) states."""
    if dataset is None:
        return None
    for attr in ("states", "observations"):
        arr = getattr(dataset, attr, None)
        if arr is not None:
            return np.asarray(arr, dtype=np.float64)
    return None


def _sample_task_states(
    task: AntMazeTask,
    dataset,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``num_samples`` states from the dataset (random if unavailable)."""
    num_samples = int(num_samples)
    raw = _dataset_raw_states(dataset)
    if raw is not None and raw.shape[0] > 0:
        if hasattr(dataset, "sample_states"):
            try:
                return np.array(
                    np.asarray(dataset.sample_states(num_samples), dtype=np.float64), copy=True
                )
            except Exception:
                pass
        gen = rng if rng is not None else np.random.default_rng(0)
        idx = gen.integers(0, raw.shape[0], size=num_samples)
        return np.array(raw[idx], dtype=np.float64, copy=True)
    gen = rng if rng is not None else np.random.default_rng(0)
    state_dim = 29
    states = np.zeros((num_samples, state_dim), dtype=np.float64)
    states[:, :2] = gen.uniform(
        [ANTMAZE_MAP_MIN[0], ANTMAZE_MAP_MIN[1]],
        [ANTMAZE_MAP_MAX[0], ANTMAZE_MAP_MAX[1]],
        size=(num_samples, 2),
    )
    return states


# --------------------------------------------------------------------------------------
# Task suite construction
# --------------------------------------------------------------------------------------


def make_goal_reaching_tasks(
    threshold: float = ANTMAZE_GOAL_THRESHOLD,
    discretize: bool = True,
    bins: int = ANTMAZE_DISCRETIZE_BINS,
    max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
) -> List[GoalReachingTask]:
    """The 5 fixed goal-reaching tasks (addendum "ant-goal-reaching")."""
    return [
        GoalReachingTask(
            goal, label=name.replace("goal-", ""), threshold=threshold,
            discretize=discretize, bins=bins, max_episode_steps=max_episode_steps,
        )
        for name, goal in ANTMAZE_GOAL_LOCATIONS.items()
    ]


def make_directional_tasks(
    max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
    reward_clip: Optional[float] = None,
    max_velocity: float = 2.0,
) -> List[DirectionalTask]:
    """The 4 directional tasks (addendum "ant-directional")."""
    return [
        DirectionalTask(
            vec, label=name, max_episode_steps=max_episode_steps,
            reward_clip=reward_clip, max_velocity=max_velocity,
        )
        for name, vec in ANTMAZE_DIRECTION_VECTORS.items()
    ]


def make_random_simplex_tasks(
    seeds: Sequence[int] = ANTMAZE_SIMPLEX_SEEDS,
    max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
) -> List[RandomSimplexTask]:
    """The 5 seeded random-simplex tasks (addendum "ant-random-simplex")."""
    return [RandomSimplexTask(seed=s, max_episode_steps=max_episode_steps) for s in seeds]


def make_path_tasks(
    kinds: Sequence[str] = ANTMAZE_PATH_KINDS,
    tolerance: float = 3.0,
    max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
) -> List[PathTask]:
    """The three hand-designed path tasks (loop / edges / center)."""
    return [
        PathTask(kind=k, tolerance=tolerance, max_episode_steps=max_episode_steps) for k in kinds
    ]


#: Suite key -> factory producing the list of tasks averaged into ``ant-<key>``.
TASK_FACTORIES: Dict[str, Callable[[], List[AntMazeTask]]] = {
    "goal-reaching": make_goal_reaching_tasks,
    "directional": make_directional_tasks,
    "random-simplex": make_random_simplex_tasks,
    "path-loop": lambda: make_path_tasks(["loop"]),
    "path-edges": lambda: make_path_tasks(["edges"]),
    "path-center": lambda: make_path_tasks(["center"]),
}

#: Order of the reported Table 1 rows (antmaze-all averages these suites).
SUITE_ORDER: Tuple[str, ...] = (
    "goal-reaching",
    "directional",
    "random-simplex",
    "path-loop",
    "path-edges",
    "path-center",
)


def get_task_suite(name: str, **kwargs) -> List[AntMazeTask]:
    """Instantiate one evaluation suite by name (``"goal-reaching"`` etc.)."""
    if name not in TASK_FACTORIES:
        raise KeyError(f"Unknown AntMaze suite {name!r}; expected one of {SUITE_ORDER}")
    return TASK_FACTORIES[name](**kwargs) if kwargs else TASK_FACTORIES[name]()


def make_antmaze_task_suite(**kwargs) -> Dict[str, List[AntMazeTask]]:
    """All AntMaze evaluation suites, keyed by suite name."""
    return {key: TASK_FACTORIES[key](**kwargs) for key in SUITE_ORDER}


# --------------------------------------------------------------------------------------
# Environment wrapper
# --------------------------------------------------------------------------------------


class AntMazeEvalWrapper:
    """Wrap an AntMaze gym environment for online zero-shot evaluation.

    Responsibilities:

    * optionally apply the 32-bin X/Y discretization shared by FRE / GC-IQL /
      GC-BC / OPAL (Appendix C.1),
    * compute the task reward from the *next* observation,
    * terminate on success for goal-reaching tasks (the paper clamps episodes at
      2000 steps; terminating early on success does not change the return because
      the success reward is 0 and no further reward can be gained, but it does
      save rollout time),
    * expose ``info["success"]`` for auxiliary reporting.
    """

    def __init__(
        self,
        env,
        task: AntMazeTask,
        discretize_xy: bool = True,
        terminate_on_success: bool = True,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        self.env = env
        self.task = task
        self.discretize_xy = bool(discretize_xy)
        self.terminate_on_success = bool(terminate_on_success)
        self.max_episode_steps = int(max_episode_steps or task.max_episode_steps)
        self._steps = 0
        self._episode_return = 0.0

    # -- gym-ish API --------------------------------------------------------------
    @property
    def observation_space(self):  # pragma: no cover - passthrough
        return getattr(self.env, "observation_space", None)

    @property
    def action_space(self):  # pragma: no cover - passthrough
        return getattr(self.env, "action_space", None)

    def seed(self, seed=None):  # pragma: no cover - legacy gym API
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def _process_obs(self, obs):
        if self.discretize_xy and obs is not None:
            return discretize_antmaze_observations(np.asarray(obs, dtype=np.float64))
        return obs

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):  # gymnasium-style (obs, info)
            obs, info = out
            info = dict(info or {})
        else:
            obs, info = out, {}
        self._steps = 0
        self._episode_return = 0.0
        info.update(self.task.reset(self.env))
        info["success"] = bool(self.task.success(obs))
        return self._process_obs(obs), info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            next_obs, _reward, terminated, truncated, info = out
            info = dict(info or {})
        else:
            next_obs, _reward, terminated, info = out
            truncated = False
            info = dict(info or {})
        self._steps += 1
        reward = float(self.task.reward(None, next_obs, action, info))
        success = bool(self.task.success(next_obs, info))
        self._episode_return += reward
        info.update(
            success=success,
            task_name=self.task.name,
            episode_return=self._episode_return,
            task_reward=reward,
        )
        done = bool(terminated) or bool(truncated)
        if self.terminate_on_success and success and self.task.done(next_obs, info):
            done = True
        if self._steps >= self.max_episode_steps:
            done = True
            truncated = True
        return self._process_obs(next_obs), reward, done, info

    def close(self):  # pragma: no cover - passthrough
        if hasattr(self.env, "close"):
            self.env.close()


# --------------------------------------------------------------------------------------
# Rollout / evaluation
# --------------------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Outcome of a single online rollout."""

    task_name: str
    total_return: float
    normalized_return: float
    length: int
    success: bool
    success_steps: Optional[int] = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def rollout_episode(
    env,
    act_fn: Callable[[np.ndarray], np.ndarray],
    max_episode_steps: int = ANTMAZE_MAX_EPISODE_STEPS,
    seed: Optional[int] = None,
    task: Optional[AntMazeTask] = None,
) -> EpisodeResult:
    """Run a single online rollout and return the (normalized) return."""
    task = task if task is not None else getattr(env, "task", None)
    if task is None:
        raise ValueError("rollout_episode requires a task (argument or env.task)")

    reset_out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out
    total = 0.0
    success = False
    success_steps: Optional[int] = None
    steps = 0
    for steps in range(1, int(max_episode_steps) + 1):
        action = act_fn(obs)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = step_out
        total += float(reward)
        if info.get("success", False):
            if not success:
                success_steps = steps
            success = True
        if done:
            break
    return EpisodeResult(
        task_name=task.name,
        total_return=float(total),
        normalized_return=float(task.normalize_return(total)),
        length=int(steps),
        success=bool(success),
        success_steps=success_steps,
    )


def evaluate_task(
    task: AntMazeTask,
    act_fn: Callable[[np.ndarray], np.ndarray],
    env=None,
    env_id: str = "antmaze-large-diverse-v2",
    num_episodes: int = 20,
    max_episode_steps: Optional[int] = None,
    seed: int = 0,
    discretize_xy: bool = True,
    terminate_on_success: bool = True,
) -> Dict[str, float]:
    """Evaluate ``act_fn`` on one AntMaze task over ``num_episodes`` rollouts."""
    owns_env = env is None
    base_env = make_antmaze_env(env_id, seed=seed) if owns_env else env
    wrapper = AntMazeEvalWrapper(
        base_env,
        task,
        discretize_xy=discretize_xy,
        terminate_on_success=terminate_on_success,
        max_episode_steps=max_episode_steps,
    )
    results: List[EpisodeResult] = []
    for ep in range(int(num_episodes)):
        results.append(
            rollout_episode(
                wrapper,
                act_fn,
                max_episode_steps=max_episode_steps or task.max_episode_steps,
                seed=seed + ep,
                task=task,
            )
        )
    if owns_env:
        try:
            base_env.close()
        except Exception:  # pragma: no cover
            pass
    normalized = np.asarray([r.normalized_return for r in results], dtype=np.float64)
    raw = np.asarray([r.total_return for r in results], dtype=np.float64)
    success = np.asarray([1.0 if r.success else 0.0 for r in results], dtype=np.float64)
    return {
        "task": task.name,
        "num_episodes": int(num_episodes),
        "score": float(normalized.mean()),
        "score_std": float(normalized.std()),
        "normalized_return": float(normalized.mean()),
        "raw_return": float(raw.mean()),
        "raw_return_std": float(raw.std()),
        "success_rate": float(success.mean()),
        "episode_length": float(np.mean([r.length for r in results])),
    }


def evaluate_suite(
    suite: Sequence[AntMazeTask],
    act_fn: Callable[[np.ndarray], np.ndarray],
    env=None,
    env_id: str = "antmaze-large-diverse-v2",
    num_episodes: int = 20,
    max_episode_steps: Optional[int] = None,
    seed: int = 0,
    discretize_xy: bool = True,
) -> Dict[str, object]:
    """Evaluate one task family; ``score`` is the mean over its tasks."""
    per_task = [
        evaluate_task(
            task,
            act_fn,
            env=env,
            env_id=env_id,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps or task.max_episode_steps,
            seed=seed,
            discretize_xy=discretize_xy,
        )
        for task in suite
    ]
    scores = np.asarray([t["score"] for t in per_task], dtype=np.float64)
    return {
        "suite": suite[0].name.split("-", 2)[-1] if suite else "unknown",
        "score": float(scores.mean()) if scores.size else 0.0,
        "score_std": float(scores.std()) if scores.size else 0.0,
        "per_task": per_task,
    }


def evaluate_antmaze_suite(
    act_fn_factory: Callable[[AntMazeTask], Callable[[np.ndarray], np.ndarray]],
    suites: Optional[Dict[str, Sequence[AntMazeTask]]] = None,
    env_id: str = "antmaze-large-diverse-v2",
    num_episodes: int = 20,
    seed: int = 0,
    discretize_xy: bool = True,
    envs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Evaluate every AntMaze suite and aggregate into ``antmaze-all``.

    ``act_fn_factory`` receives the task (so the caller can encode the 32
    ``(state, reward)`` pairs for that specific task into ``z``) and returns a
    callable mapping observations to actions.
    """
    suites = suites if suites is not None else make_antmaze_task_suite()
    envs = envs or {}
    per_suite: Dict[str, Dict[str, object]] = {}
    for key in SUITE_ORDER:
        suite = suites[key]
        results = [
            evaluate_task(
                task,
                act_fn_factory(task),
                env=envs.get(key),
                env_id=env_id,
                num_episodes=num_episodes,
                seed=seed,
                discretize_xy=discretize_xy,
            )
            for task in suite
        ]
        scores = np.asarray([r["score"] for r in results], dtype=np.float64)
        per_suite[f"ant-{key}"] = {
            "score": float(scores.mean()) if scores.size else 0.0,
            "score_std": float(scores.std()) if scores.size else 0.0,
            "per_task": results,
        }
    all_scores = np.asarray([v["score"] for v in per_suite.values()], dtype=np.float64)
    per_suite["antmaze-all"] = {
        "score": float(all_scores.mean()),
        "score_std": float(all_scores.std()),
    }
    return per_suite


# --------------------------------------------------------------------------------------
# Environment construction
# --------------------------------------------------------------------------------------


def make_antmaze_env(
    env_id: str = "antmaze-large-diverse-v2",
    seed: Optional[int] = None,
    start_at_center: bool = True,
    **kwargs,
):
    """Create a D4RL AntMaze gym environment.

    D4RL requires ``import d4rl`` before ``gym.make``; the import is lazy so this
    module (and consequently the config layer) stays import-light.  AntMaze uses
    ``mujoco-py``/``gym``'s legacy API, so the raw env is returned unchanged; the
    task-specific behaviour is added by :class:`AntMazeEvalWrapper`.
    """
    try:  # pragma: no cover - requires d4rl/mujoco
        import gym  # type: ignore
        import d4rl  # noqa: F401  (registers the D4RL environments)
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "AntMaze evaluation requires `gym` and `d4rl` (import d4rl before gym.make). "
            f"Original error: {exc}"
        ) from exc
    env = gym.make(env_id)
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)
    if start_at_center:
        _enable_center_start(env)
    return env


def _enable_center_start(env) -> None:
    """Place the ant at the maze centre rather than the D4RL bottom-left start.

    Appendix C.1: "The ant robot is placed in the center of the maze to allow for
    more diverse behavior, in comparison to the original start position in the
    bottom-left."  D4RL exposes the init qpos as a ``mujoco_py`` global
    (``env.unwrapped.init_qpos``); we shift the free-joint XY to the maze centre,
    which is a documented re-implementation of that statement.
    """
    unwrapped = getattr(env, "unwrapped", env)
    init_qpos = getattr(unwrapped, "init_qpos", None)
    if init_qpos is None:
        return
    try:
        center = np.array(
            [0.5 * (ANTMAZE_MAP_MIN[0] + ANTMAZE_MAP_MAX[0]),
             0.5 * (ANTMAZE_MAP_MIN[1] + ANTMAZE_MAP_MAX[1])],
            dtype=np.float64,
        )
        unwrapped.init_qpos[:2] = center
    except Exception:  # pragma: no cover - best effort only
        pass


# --------------------------------------------------------------------------------------
# Convenience: policy adapter
# --------------------------------------------------------------------------------------


def make_iql_policy_fn(agent, z, deterministic: bool = True, device: str = "cpu"):
    """Wrap a z-conditioned IQL agent into an ``act_fn(obs) -> action`` callable.

    ``z`` is the 128-dim task latent produced by the frozen FRE encoder from this
    task's 32 ``(state, reward)`` pairs.  The returned callable performs the
    observation -> tensor -> action -> numpy round trip needed by the rollout
    loop.
    """
    import torch  # local import: keep this module import-light

    z_tensor = None
    if z is not None:
        z_tensor = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device).reshape(1, -1)
        if z_tensor.dim() == 1:
            z_tensor = z_tensor.unsqueeze(0)

    def act_fn(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.asarray(obs, dtype=np.float32), device=device
            ).reshape(1, -1)
            action = agent.select_action(
                obs_t, z_tensor, deterministic=deterministic
            )
            action = np.asarray(action.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        return action

    return act_fn


__all__ = [
    "ANTMAZE_GOAL_LOCATIONS",
    "ANTMAZE_DIRECTION_VECTORS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "ANTMAZE_PATH_KINDS",
    "ANTMAZE_GOAL_THRESHOLD",
    "ANTMAZE_DISCRETIZE_BINS",
    "ANTMAZE_MAX_EPISODE_STEPS",
    "ANTMAZE_ENCODER_SAMPLES",
    "ANTMAZE_MAP_MIN",
    "ANTMAZE_MAP_MAX",
    "extract_antmaze_xy",
    "extract_antmaze_velocity",
    "antmaze_discretize_xy",
    "discretize_antmaze_observations",
    "AntMazeTask",
    "GoalReachingTask",
    "DirectionalTask",
    "RandomSimplexTask",
    "PathTask",
    "default_path_waypoints",
    "point_to_polyline_distance",
    "make_goal_reaching_tasks",
    "make_directional_tasks",
    "make_random_simplex_tasks",
    "make_path_tasks",
    "get_task_suite",
    "make_antmaze_task_suite",
    "TASK_FACTORIES",
    "SUITE_ORDER",
    "AntMazeEvalWrapper",
    "EpisodeResult",
    "rollout_episode",
    "evaluate_task",
    "evaluate_suite",
    "evaluate_antmaze_suite",
    "make_antmaze_env",
    "make_iql_policy_fn",
]
