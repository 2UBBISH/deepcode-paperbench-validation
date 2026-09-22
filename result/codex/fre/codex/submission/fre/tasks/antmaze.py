"""AntMaze evaluation tasks (Section 5, Table 1, and the addendum).

Environment (Appendix C.1):
    * ``antmaze-large-diverse-v2`` from D4RL, online evaluation with episodes
      of at most 2000 timesteps;
    * the ant is placed in the *centre* of the maze rather than the original
      bottom-left start position, to encourage more diverse behaviour;
    * FRE / GC-IQL / GC-BC / OPAL use a discretised preprocessing where the
      (x, y) coordinates are discretised into 32 bins.

Observation layout of ``antmaze-*-v2`` (29 dimensions):
    ``obs[0:2]``   torso (x, y) position in maze coordinates,
    ``obs[2]``     torso z,
    ``obs[3:15]``  12 ant joint angles,
    ``obs[15:17]`` torso (x, y) linear velocity,
    ``obs[17:29]`` remaining joint velocities.

The maze used by the ``large`` variants is ``HARDEST_MAZE_TEST`` with
``maze_size_scaling = 4.0``; the D4RL code shifts the maze so the reset tile is
at the origin, which puts the free space in ``x in [0, 36]``, ``y in [0, 24]``
(origin at the bottom left, exactly the coordinate frame the addendum uses for
the goal locations).

Task families (Section 5):
    * ``goal-reaching``    -- 5 hand-crafted goal locations;
    * ``directional``      -- 4 unit target velocities;
    * ``random-simplex``   -- 5 seeded procedural noise reward functions;
    * ``path-*``           -- 3 hand-crafted corridor rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from fre.tasks.base import EvalTask, GoalReachingEvalTask, TaskSuite
from fre.utils.simplex import make_height_and_velocity_fields


# --------------------------------------------------------------------------------------
# Observation constants / helpers
# --------------------------------------------------------------------------------------
ANTMAZE_OBS_DIM = 29
XY_SLICE = slice(0, 2)
VELOCITY_SLICE = slice(15, 17)

MAZE_SIZE_SCALING = 4.0
# Extent of the free space of HARDEST_MAZE_TEST (12 x 9 cells shifted so the
# reset tile at (row=1, col=1) sits at x = 0, y = 0).
MAZE_XY_MIN = np.array([0.0, 0.0], dtype=np.float32)
MAZE_XY_MAX = np.array([36.0, 24.0], dtype=np.float32)
MAZE_XY_CENTER = np.array([20.0, 8.0], dtype=np.float32)  # nearest free cell to the maze centre

# The 5 hand-crafted goal locations from the addendum, on an (X, Y) grid with
# the origin at the bottom left.
GOAL_LOCATIONS: Tuple[Tuple[str, Tuple[float, float]], ...] = (
    ("goal-bottom", (28.0, 0.0)),
    ("goal-left", (0.0, 15.0)),
    ("goal-top", (35.0, 24.0)),
    ("goal-center", (12.0, 24.0)),
    ("goal-right", (33.0, 16.0)),
)

# The 4 directional tasks from the addendum (the text says "5 directional
# tasks" but then lists and averages four unit directions).
DIRECTIONAL_TASKS: Tuple[Tuple[str, Tuple[float, float]], ...] = (
    ("vel_left", (-1.0, 0.0)),
    ("vel_up", (0.0, 1.0)),
    ("vel_down", (0.0, -1.0)),
    ("vel_right", (1.0, 0.0)),
)

SIMPLEX_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)


def xy_from_obs(obs: np.ndarray) -> np.ndarray:
    """Extract the torso (x, y) position from an AntMaze observation."""
    return np.asarray(obs, dtype=np.float64)[..., XY_SLICE]


def velocity_from_obs(obs: np.ndarray) -> np.ndarray:
    """Extract the torso (vx, vy) world-frame linear velocity."""
    return np.asarray(obs, dtype=np.float64)[..., VELOCITY_SLICE]


class AntMazeStatePreprocessor:
    """Discretises the (x, y) coordinates into ``num_bins`` bins.

    Appendix C.1: "The FRE, GC-IQL, GC-BC, and OPAL agents all utilize a
    discretized preprocessing procedure, where the X and Y coordinates are
    discretized into 32 bins."  This preprocessor is applied to the states
    *fed to the model* (encoder, decoder, Q/V/policy); reward functions are
    evaluated on the raw state, since FRE assumes rewards are pure functions of
    the environment state.
    """

    def __init__(
        self,
        num_bins: int = 32,
        xy_min: Sequence[float] = MAZE_XY_MIN,
        xy_max: Sequence[float] = MAZE_XY_MAX,
    ) -> None:
        self.num_bins = int(num_bins)
        self.xy_min = np.asarray(xy_min, dtype=np.float64)
        self.xy_max = np.asarray(xy_max, dtype=np.float64)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32).copy()
        xy = states[..., XY_SLICE]
        scaled = (xy - self.xy_min) / (self.xy_max - self.xy_min)
        bins = np.floor(np.clip(scaled, 0.0, 1.0) * self.num_bins)
        states[..., XY_SLICE] = np.clip(bins, 0, self.num_bins - 1)
        return states

    @property
    def state_dim(self) -> int:
        return ANTMAZE_OBS_DIM


# --------------------------------------------------------------------------------------
# Task definitions
# --------------------------------------------------------------------------------------
class DirectionalTask(EvalTask):
    """Reward the agent for moving in a target (x, y) direction."""

    group = "directional"

    def __init__(
        self,
        name: str,
        direction: Sequence[float],
        max_episode_steps: int = 2000,
        velocity_reference: float = 5.0,
    ) -> None:
        self.name = name
        direction = np.asarray(direction, dtype=np.float64)
        self.direction = direction / (np.linalg.norm(direction) + 1e-8)
        self.max_episode_steps = int(max_episode_steps)
        self.velocity_reference = float(velocity_reference)
        # Rewards are the dot product between the agent velocity and the target
        # direction, scaled by a reference speed so that they lie in [-1, 1].
        self.reward_range = (-1.0, 1.0)

    def reward(self, obs, action=None, next_obs=None) -> np.ndarray:
        vel = velocity_from_obs(obs)
        raw = vel @ self.direction / self.velocity_reference
        return np.clip(raw, -1.0, 1.0).astype(np.float32)


class RandomSimplexTask(EvalTask):
    """Procedurally generated reward from a seeded 2-D noise field.

    From the addendum: baseline reward ``-1`` each step, a bonus for standing in
    higher "height" regions, and an additional bonus for moving in the local
    preferred velocity direction indicated by the noise field.
    """

    group = "random-simplex"
    # The reward is clipped to [-1, 1] so that it lives in the same range as the
    # random linear / MLP prior families the encoder was trained on.
    reward_range = (-1.0, 1.0)

    def __init__(
        self,
        seed: int,
        max_episode_steps: int = 2000,
        frequency: float = 0.08,
        velocity_reference: float = 5.0,
        use_opensimplex: bool = True,
    ) -> None:
        self.name = f"random-simplex-{seed}"
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.velocity_reference = float(velocity_reference)
        self.field = make_height_and_velocity_fields(
            self.seed, MAZE_XY_MIN, MAZE_XY_MAX, frequency=frequency, use_opensimplex=use_opensimplex
        )

    def reward(self, obs, action=None, next_obs=None) -> np.ndarray:
        xy = xy_from_obs(obs)
        height, vx_pref, vy_pref = self.field(xy)
        vel = velocity_from_obs(obs)
        vel_bonus = (vel[..., 0] * vx_pref + vel[..., 1] * vy_pref) / self.velocity_reference
        reward = -1.0 + height + np.clip(vel_bonus, -1.0, 1.0)
        reward = np.clip(reward, self.reward_range[0], self.reward_range[1])
        return reward.astype(np.float32)


class CorridorTask(EvalTask):
    """Reward for moving along a hand-crafted corridor (a polyline in (x, y)).

    The paper defines three corridor tasks (centre, loop, edges) but does not
    publish the exact waypoint lists, so the corridors below are hand-crafted
    from the ``HARDEST_MAZE_TEST`` free-space layout (documented in
    :data:`CORRIDOR_WAYPOINTS`).  The reward is ``-1`` at every step plus a
    bonus of ``+2`` while the agent is within ``radius`` of a corridor.
    """

    group = "path"
    reward_range = (-1.0, 1.0)

    def __init__(
        self,
        name: str,
        polylines: Sequence[Sequence[Tuple[float, float]]],
        radius: float = 2.0,
        max_episode_steps: int = 2000,
    ) -> None:
        self.name = name
        self.polylines = [np.asarray(p, dtype=np.float64) for p in polylines]
        self.radius = float(radius)
        self.max_episode_steps = int(max_episode_steps)

    def _distance(self, xy: np.ndarray) -> np.ndarray:
        best = np.full(xy.shape[:-1], np.inf)
        for poly in self.polylines:
            for a, b in zip(poly[:-1], poly[1:]):
                ab = b - a
                denom = float(ab @ ab) + 1e-8
                t = np.clip(((xy - a) @ ab) / denom, 0.0, 1.0)
                proj = a + t[..., None] * ab
                dist = np.linalg.norm(xy - proj, axis=-1)
                best = np.minimum(best, dist)
        return best

    def reward(self, obs, action=None, next_obs=None) -> np.ndarray:
        xy = xy_from_obs(obs)
        on_corridor = self._distance(xy) < self.radius
        return np.where(on_corridor, 1.0, -1.0).astype(np.float32)


# Hand-crafted corridors, expressed as polylines in maze (x, y) coordinates.
# Column x = 20 (map column 6) is a fully free vertical corridor through the
# middle of HARDEST_MAZE_TEST, which is used as the "centre" path.
CORRIDOR_WAYPOINTS = {
    "path-center": [[(20.0, 0.0), (20.0, 24.0)]],
    "path-loop": [
        [
            (0.0, 0.0), (12.0, 0.0), (12.0, 12.0), (20.0, 12.0), (20.0, 24.0),
            (36.0, 24.0), (36.0, 16.0), (28.0, 16.0), (28.0, 4.0), (20.0, 4.0),
            (20.0, 0.0), (0.0, 0.0),
        ]
    ],
    "path-edges": [
        [(0.0, 0.0), (0.0, 16.0)],
        [(36.0, 0.0), (36.0, 16.0)],
        [(0.0, 24.0), (36.0, 24.0)],
        [(0.0, 0.0), (12.0, 0.0)],
        [(20.0, 0.0), (36.0, 0.0)],
    ],
}


def make_antmaze_goal_tasks(
    max_episode_steps: int = 2000, threshold: float = 2.0
) -> List[EvalTask]:
    """The 5 hand-crafted goal-reaching tasks."""
    return [
        GoalReachingEvalTask(
            name=name,
            goal=np.asarray(loc, dtype=np.float32),
            position_fn=xy_from_obs,
            threshold=threshold,
            max_episode_steps=max_episode_steps,
        )
        for name, loc in GOAL_LOCATIONS
    ]


def make_antmaze_directional_tasks(max_episode_steps: int = 2000) -> List[EvalTask]:
    """The 4 directional movement tasks."""
    return [
        DirectionalTask(name=name, direction=vec, max_episode_steps=max_episode_steps)
        for name, vec in DIRECTIONAL_TASKS
    ]


def make_antmaze_simplex_tasks(
    max_episode_steps: int = 2000, use_opensimplex: bool = True
) -> List[EvalTask]:
    """The 5 seeded ``ant-random-simplex`` tasks."""
    return [
        RandomSimplexTask(seed=s, max_episode_steps=max_episode_steps, use_opensimplex=use_opensimplex)
        for s in SIMPLEX_SEEDS
    ]


def make_antmaze_path_tasks(max_episode_steps: int = 2000) -> List[EvalTask]:
    """The 3 hand-crafted corridor tasks."""
    return [
        CorridorTask(name=name, polylines=polys, max_episode_steps=max_episode_steps)
        for name, polys in CORRIDOR_WAYPOINTS.items()
    ]


def make_antmaze_suites(
    max_episode_steps: int = 2000, use_opensimplex: bool = True
) -> List[TaskSuite]:
    """All AntMaze task groups reported in Table 1 (plus a ``path-all`` suite).

    ``path-all`` is the average over the three corridor tasks and appears in
    Table 4 / Figure 5.
    """
    goals = make_antmaze_goal_tasks(max_episode_steps)
    directional = make_antmaze_directional_tasks(max_episode_steps)
    simplex = make_antmaze_simplex_tasks(max_episode_steps, use_opensimplex)
    paths = make_antmaze_path_tasks(max_episode_steps)
    return [
        TaskSuite("ant-goal-reaching", goals, description="Average over 5 goal-reaching tasks"),
        TaskSuite("ant-directional", directional, description="Average over 4 directional tasks"),
        TaskSuite("ant-random-simplex", simplex, description="Average over 5 seeded noise tasks"),
        TaskSuite("ant-path-all", paths, description="Average over the 3 corridor tasks"),
        TaskSuite("ant-path-loop", [paths[1]], description="Loop corridor"),
        TaskSuite("ant-path-edges", [paths[2]], description="Edge corridors"),
        TaskSuite("ant-path-center", [paths[0]], description="Central corridor"),
    ]


# --------------------------------------------------------------------------------------
# Environment construction
# --------------------------------------------------------------------------------------
def center_reset_xy() -> np.ndarray:
    """Reset location used for AntMaze evaluation.

    Appendix C.1: "The ant robot is placed in the center of the maze to allow
    for more diverse behavior, in comparison to the original start position in
    the bottom-left."  We use the free maze cell nearest to the geometric
    centre of the free space.
    """
    return MAZE_XY_CENTER.copy()


def make_antmaze_env(env_name: str = "antmaze-large-diverse-v2"):
    """Create the D4RL AntMaze environment used for evaluation."""
    try:
        import d4rl  # noqa: F401
        import gym
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "gym and d4rl are required to run AntMaze evaluation; see README."
        ) from exc
    env = gym.make(env_name)
    return env


def reset_ant_at_center(env, seed: Optional[int] = None):
    """Reset the AntMaze environment with the ant at the centre of the maze."""
    obs = env.reset()
    try:
        base = env.unwrapped
        loco = getattr(base, "wrapped_env", None)
        if loco is not None and hasattr(loco, "set_xy"):
            loco.set_xy(center_reset_xy())
            obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else obs
    except Exception:  # pragma: no cover - depends on the installed d4rl version
        pass
    return obs
