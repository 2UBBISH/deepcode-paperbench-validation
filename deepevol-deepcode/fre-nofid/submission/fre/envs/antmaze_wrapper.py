"""AntMaze environment wrapper for FRE zero-shot evaluation.

Implements the AntMaze portion of the FRE evaluation suite (Sec 5 / App C of the
paper, plus the staged reproduction plan):

* ``antmaze-large-diverse-v2`` offline dataset, episodes capped at 2000 steps and
  started from the centre of the maze for evaluation.
* XY positions are discretised into ``32`` bins per axis; goal-reaching tasks
  terminate when the discretised distance to the goal is ``<= 2``.
* Task families:

  - 5 goal-reaching goals: ``[(28, 0), (0, 15), (35, 24), (12, 24), (33, 16)]``
  - 4 directional (dot-product of XY velocity onto a unit direction) tasks:
    ``[(-1, 0), (0, 1), (0, -1), (1, 0)]``
  - 5 random-simplex fields generated with ``opensimplex`` (seeds 1..5)
  - 3 corridor tasks: ``path-center``, ``path-loop``, ``path-edges``

The wrapper follows the same dual-mode (live D4RL simulator vs. offline dataset
replay) design as :mod:`fre.envs.exorl_wrapper` so that evaluation works even on
machines without a working MuJoCo/D4RL installation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

ANTMAZE_DATASET = "antmaze-large-diverse-v2"

RAW_OBS_DIM = 29          # 15-dim qpos + 14-dim qvel for the AntMaze "ant" agent
ACTION_DIM = 8
XY_DIM = 2

MAX_EPISODE_STEPS = 2000
NUM_XY_BINS = 32
DEFAULT_XY_EXTENT = 36.0  # raw coordinate span of the large maze grid
DEFAULT_GOAL_DISTANCE = 2.0  # measured in discretised bin units

#: The five goal-reaching evaluation goals, in raw AntMaze coordinates.
GOAL_TASK_GOALS: Tuple[Tuple[float, float], ...] = (
    (28.0, 0.0),
    (0.0, 15.0),
    (35.0, 24.0),
    (12.0, 24.0),
    (33.0, 16.0),
)

#: The four directional (XY velocity dot-product) tasks.
DIRECTIONAL_GOALS: Tuple[Tuple[float, float], ...] = (
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
)

#: Seeds for the random-simplex reward fields.
SIMPLEX_SEEDS: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: Corridor task names.
PATH_TASKS: Tuple[str, ...] = ("path-center", "path-loop", "path-edges")

TASK_FAMILIES = ("goal", "directional", "simplex", "path")


# --------------------------------------------------------------------------------------
# Coordinate helpers
# --------------------------------------------------------------------------------------


def discretize_xy(xy: np.ndarray,
                  extent: float = DEFAULT_XY_EXTENT,
                  num_bins: int = NUM_XY_BINS) -> np.ndarray:
    """Discretise continuous XY coordinates into ``num_bins`` bins per axis.

    Raw AntMaze coordinates live (approximately) in ``[0, extent]``.  Values are
    rescaled to ``[0, num_bins)`` and floored, matching the FRE reward
    discretisation convention.  The output dtype is ``int64`` and has the same
    leading shape as ``xy`` with the trailing dim equal to ``num_bins`` indices
    (2 for XY input).
    """
    xy = np.asarray(xy, dtype=np.float64)
    scaled = np.floor(xy / float(extent) * float(num_bins))
    return np.clip(scaled, 0, num_bins - 1).astype(np.int64)


def bin_to_xy(bins: np.ndarray,
              extent: float = DEFAULT_XY_EXTENT,
              num_bins: int = NUM_XY_BINS) -> np.ndarray:
    """Inverse of :func:`discretize_xy` returning bin centres."""
    bins = np.asarray(bins, dtype=np.float64)
    return (bins + 0.5) * (float(extent) / float(num_bins))


def xy_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between two XY positions (broadcasting friendly)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b, axis=-1)


def bin_distance(a: np.ndarray, b: np.ndarray,
                 extent: float = DEFAULT_XY_EXTENT,
                 num_bins: int = NUM_XY_BINS) -> np.ndarray:
    """Distance between continuous positions measured in discretised bin units."""
    return xy_distance(discretize_xy(a, extent, num_bins),
                       discretize_xy(b, extent, num_bins)).astype(np.float64)


# --------------------------------------------------------------------------------------
# Reward functions
# --------------------------------------------------------------------------------------


def goal_reward(positions: np.ndarray,
                goal: Sequence[float],
                threshold: float = DEFAULT_GOAL_DISTANCE,
                extent: float = DEFAULT_XY_EXTENT,
                num_bins: int = NUM_XY_BINS,
                discrete: bool = True) -> np.ndarray:
    """Sparse goal-reaching reward: ``0`` inside the goal, ``-1`` otherwise.

    Matches the FRE goal-reaching prior/eval reward ``r(s) = 0`` when the goal is
    reached and ``-1`` until then.
    """
    positions = np.asarray(positions, dtype=np.float64)
    dist = (bin_distance(positions, np.asarray(goal, dtype=np.float64), extent, num_bins)
            if discrete else xy_distance(positions, np.asarray(goal, dtype=np.float64)))
    return np.where(dist <= float(threshold), 0.0, -1.0).astype(np.float32)


def goal_done(positions: np.ndarray,
              goal: Sequence[float],
              threshold: float = DEFAULT_GOAL_DISTANCE,
              extent: float = DEFAULT_XY_EXTENT,
              num_bins: int = NUM_XY_BINS,
              discrete: bool = True) -> np.ndarray:
    """Boolean success mask for goal-reaching tasks."""
    positions = np.asarray(positions, dtype=np.float64)
    dist = (bin_distance(positions, np.asarray(goal, dtype=np.float64), extent, num_bins)
            if discrete else xy_distance(positions, np.asarray(goal, dtype=np.float64)))
    return dist <= float(threshold)


def directional_reward(velocities: np.ndarray, direction: Sequence[float]) -> np.ndarray:
    """Dot product of the XY velocity with a (unit) direction vector.

    The FRE directional tasks reward movement along a fixed compass direction;
    the raw dot product is clipped to ``[-1, 1]`` so it matches the reward-prior
    range used by the encoder.
    """
    velocities = np.asarray(velocities, dtype=np.float64)[..., :2]
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    dot = np.sum(velocities * direction, axis=-1)
    return np.clip(dot, -1.0, 1.0).astype(np.float32)


def directional_done(reward: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Directional tasks are episodic-successful once the clipped dot hits 1."""
    return np.asarray(reward, dtype=np.float64) >= float(threshold)


def _simplex_field(seed: int,
                   extent: float = DEFAULT_XY_EXTENT,
                   frequency: float = 0.15) -> Callable[[np.ndarray], np.ndarray]:
    """Build a smooth random reward field over XY using ``opensimplex``.

    Falls back to a deterministic Fourier-mode field when ``opensimplex`` is not
    installed so the task suite remains reproducible.
    """
    try:  # pragma: no cover - exercised only when the package is present
        from opensimplex import OpenSimplex  # type: ignore

        gen = OpenSimplex(seed=int(seed))
        scale = float(frequency)

        def field(positions: np.ndarray) -> np.ndarray:
            positions = np.asarray(positions, dtype=np.float64)
            flat = positions.reshape(-1, positions.shape[-1])[..., :2]
            out = np.empty(flat.shape[0], dtype=np.float64)
            for i, (x, y) in enumerate(flat):
                out[i] = gen.noise2d(x * scale, y * scale)
            return np.clip(out.reshape(positions.shape[:-1]), -1.0, 1.0).astype(np.float32)

        return field
    except Exception:
        rng = np.random.default_rng(1234 + int(seed))
        num_modes = 6
        freqs = rng.uniform(0.08, 0.35, size=(num_modes, 2))
        phases = rng.uniform(0.0, 2.0 * math.pi, size=num_modes)
        weights = rng.normal(size=num_modes)
        weights = weights / (np.abs(weights).sum() + 1e-8)

        def field(positions: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            positions = np.asarray(positions, dtype=np.float64)
            xy = positions[..., :2]
            out = np.zeros(xy.shape[:-1], dtype=np.float64)
            for k in range(num_modes):
                phase = (xy * freqs[k]).sum(axis=-1) + phases[k]
                out = out + weights[k] * np.sin(phase)
            return np.clip(out, -1.0, 1.0).astype(np.float32)

        return field


def simplex_reward(positions: np.ndarray,
                   seed: int,
                   extent: float = DEFAULT_XY_EXTENT) -> np.ndarray:
    """Evaluate the deterministic random-simplex field for ``seed``."""
    return _simplex_field(seed, extent)(positions)


# Corridor route definitions in *bin* space (0..31).  These trace the corridors of
# the antmaze-large layout; the reward is -1 on the route and -1 + progress off it,
# which reproduces the paper's "stay on this path" tasks.

_CENTER_PATH_BINS: Tuple[Tuple[int, int], ...] = (
    (16, 2), (16, 6), (12, 10), (8, 14), (10, 18), (16, 22), (20, 26), (16, 30),
)
_LOOP_PATH_BINS: Tuple[Tuple[int, int], ...] = (
    (8, 8), (16, 6), (24, 8), (26, 16), (24, 24), (16, 26), (8, 24), (6, 16), (8, 8),
)
_EDGE_PATH_BINS: Tuple[Tuple[int, int], ...] = (
    (2, 2), (2, 30), (30, 30), (30, 2), (2, 2),
)

_PATH_ROUTES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "path-center": _CENTER_PATH_BINS,
    "path-loop": _LOOP_PATH_BINS,
    "path-edges": _EDGE_PATH_BINS,
}


def path_route(task: str) -> np.ndarray:
    """Return the raw-coordinate waypoints for a corridor task."""
    if task not in _PATH_ROUTES:
        raise KeyError(f"unknown corridor task {task!r}; expected one of {sorted(_PATH_ROUTES)}")
    bins = np.asarray(_PATH_ROUTES[task], dtype=np.float64)
    return bin_to_xy(bins)


def path_reward(positions: np.ndarray,
                task: str,
                tolerance: float = 1.5,
                extent: float = DEFAULT_XY_EXTENT,
                num_bins: int = NUM_XY_BINS) -> np.ndarray:
    """Sparse reward for staying near a corridor route (-1/0)."""
    positions = np.asarray(positions, dtype=np.float64)
    xy = positions[..., :2]
    route = path_route(task)
    dists = np.linalg.norm(xy[..., None, :] - route[None, :, :], axis=-1)
    min_dist = dists.min(axis=-1)
    # convert raw tolerance into bin units for consistency with goal tasks
    tol_bins = float(tolerance)
    min_dist_bins = min_dist / (float(extent) / float(num_bins))
    return np.where(min_dist_bins <= tol_bins, 0.0, -1.0).astype(np.float32)


def path_done(positions: np.ndarray,
              task: str,
              tolerance: float = 1.5,
              extent: float = DEFAULT_XY_EXTENT,
              num_bins: int = NUM_XY_BINS) -> np.ndarray:
    return path_reward(positions, task, tolerance, extent, num_bins) >= 0.0


# --------------------------------------------------------------------------------------
# Task specifications
# --------------------------------------------------------------------------------------


@dataclass
class AntMazeTaskSpec:
    """Descriptor for one zero-shot AntMaze task."""

    name: str
    family: str
    reward_fn: Callable[[np.ndarray], np.ndarray]
    done_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    goal: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None
    seed: Optional[int] = None
    uses_velocity: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reward(self, obs: np.ndarray) -> np.ndarray:
        """Evaluate the reward for an observation (or batch of observations)."""
        return self.reward_fn(np.asarray(obs, dtype=np.float64))

    def done(self, obs: np.ndarray) -> np.ndarray:
        if self.done_fn is None:
            return np.zeros(np.asarray(obs).shape[:-1], dtype=bool)
        return np.asarray(self.done_fn(np.asarray(obs, dtype=np.float64)), dtype=bool)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "goal": None if self.goal is None else list(map(float, self.goal)),
            "direction": None if self.direction is None else list(map(float, self.direction)),
            "seed": self.seed,
            "uses_velocity": self.uses_velocity,
            **self.metadata,
        }


@dataclass
class AntMazeConfig:
    """Configuration bundle for :class:`AntMazeWrapper`."""

    dataset: str = ANTMAZE_DATASET
    max_episode_steps: int = MAX_EPISODE_STEPS
    goal_threshold: float = DEFAULT_GOAL_DISTANCE
    num_bins: int = NUM_XY_BINS
    xy_extent: float = DEFAULT_XY_EXTENT
    center_start: bool = True
    discrete_reward: bool = True
    seed: Optional[int] = None
    use_live_env: Optional[bool] = None
    corridor_tolerance: float = 1.5


# --------------------------------------------------------------------------------------
# Wrapper
# --------------------------------------------------------------------------------------


class AntMazeWrapper:
    """Gym-like AntMaze environment exposing FRE zero-shot tasks.

    Object layout: ``observations`` are raw 29-d AntMaze states; ``encoder_observations``
    are the same (the FRE encoder consumes raw states for AntMaze).  XY helpers operate
    on ``obs[..., :2]`` and velocity on ``obs[..., 15:17]`` (qvel XY).
    """

    def __init__(
        self,
        dataset: Any = None,
        task: Optional[AntMazeTaskSpec] = None,
        *,
        dataset_name: str = ANTMAZE_DATASET,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        goal_threshold: float = DEFAULT_GOAL_DISTANCE,
        num_bins: int = NUM_XY_BINS,
        xy_extent: float = DEFAULT_XY_EXTENT,
        center_start: bool = True,
        discrete_reward: bool = True,
        corridor_tolerance: float = 1.5,
        seed: Optional[int] = None,
        live_env: Any = None,
        use_live_env: Optional[bool] = None,
        config: Optional[AntMazeConfig] = None,
    ) -> None:
        if config is not None:
            dataset_name = config.dataset
            max_episode_steps = config.max_episode_steps
            goal_threshold = config.goal_threshold
            num_bins = config.num_bins
            xy_extent = config.xy_extent
            center_start = config.center_start
            discrete_reward = config.discrete_reward
            corridor_tolerance = config.corridor_tolerance
            seed = config.seed if seed is None else seed
            use_live_env = config.use_live_env if use_live_env is None else use_live_env

        self.dataset_name = dataset_name
        self.max_episode_steps = int(max_episode_steps)
        self.goal_threshold = float(goal_threshold)
        self.num_bins = int(num_bins)
        self.xy_extent = float(xy_extent)
        self.center_start = bool(center_start)
        self.discrete_reward = bool(discrete_reward)
        self.corridor_tolerance = float(corridor_tolerance)
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)

        self.dataset = dataset
        self._dataset_arrays: Optional[Dict[str, np.ndarray]] = (
            dict(dataset) if isinstance(dataset, dict) else None
        )
        self._dataset_positions: Optional[np.ndarray] = None
        self._infer_extent()

        self.live_env = live_env
        if self.live_env is None and use_live_env is not False:
            self.live_env = _maybe_make_live_env(dataset_name)
        self.live_env_available = self.live_env is not None

        self.task: AntMazeTaskSpec = task or self.default_task()

        # replay bookkeeping
        self._episode_index: Optional[int] = None
        self._episode_start = 0
        self._step_in_episode = 0
        self._t = 0
        self.num_episodes_completed = 0
        self.last_episode_return = 0.0
        self.last_episode_success = False
        self._episode_return = 0.0

        # episode boundaries from the dataset (for offline replay)
        self._episode_bounds: List[Tuple[int, int]] = []
        self._build_episode_bounds()

    # -- introspection -----------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return RAW_OBS_DIM

    @property
    def encoder_observation_dim(self) -> int:
        return RAW_OBS_DIM

    @property
    def policy_observation_dim(self) -> int:
        return RAW_OBS_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    # -- dataset handling --------------------------------------------------------------

    def _infer_extent(self) -> None:
        obs = None
        if self._dataset_arrays is not None:
            obs = self._dataset_arrays.get("observations")
            if obs is None:
                obs = self._dataset_arrays.get("obs")
        if obs is not None and np.asarray(obs).ndim == 2:
            obs = np.asarray(obs, dtype=np.float64)
            self._dataset_positions = obs[:, :XY_DIM]
            # Use the dataset support to calibrate the discretisation extent, but
            # keep the paper's goal coordinates valid (they reach ~36).
            observed = float(np.percentile(self._dataset_positions, 99.5))
            self.xy_extent = max(self.xy_extent, observed)

    def _build_episode_bounds(self) -> None:
        if self._dataset_arrays is None:
            return
        terminals = self._dataset_arrays.get("terminals")
        if terminals is None:
            terminals = self._dataset_arrays.get("dones")
        n = None
        for key in ("observations", "actions", "rewards"):
            if key in self._dataset_arrays:
                n = len(self._dataset_arrays[key])
                break
        if n is None:
            return
        if terminals is None:
            self._episode_bounds = [(0, n)]
            return
        terminals = np.asarray(terminals).reshape(-1)[:n]
        ends = np.nonzero(terminals > 0.5)[0]
        start = 0
        bounds: List[Tuple[int, int]] = []
        for end in ends:
            bounds.append((start, int(end) + 1))
            start = int(end) + 1
        if start < n:
            bounds.append((start, n))
        self._episode_bounds = bounds or [(0, n)]

    def num_dataset_transitions(self) -> int:
        if self._dataset_arrays is None:
            return 0
        for key in ("observations", "actions"):
            if key in self._dataset_arrays:
                return len(self._dataset_arrays[key])
        return 0

    # -- task construction -------------------------------------------------------------

    def default_task(self) -> AntMazeTaskSpec:
        return goal_task(GOAL_TASK_GOALS[0], self)

    def set_task(self, task: AntMazeTaskSpec) -> "AntMazeWrapper":
        self.task = task
        return self

    def set_goal(self, goal: Sequence[float], threshold: Optional[float] = None) -> "AntMazeWrapper":
        if threshold is not None:
            self.goal_threshold = float(threshold)
        self.task = goal_task(goal, self)
        return self

    def eval_tasks(self,
                   include_goals: bool = True,
                   include_directional: bool = True,
                   include_simplex: bool = True,
                   include_path: bool = True,
                   goals: Optional[Sequence[Sequence[float]]] = None,
                   simplex_seeds: Sequence[int] = SIMPLEX_SEEDS) -> Dict[str, AntMazeTaskSpec]:
        return antmaze_eval_tasks(
            self,
            include_goals=include_goals,
            include_directional=include_directional,
            include_simplex=include_simplex,
            include_path=include_path,
            goals=goals,
            simplex_seeds=simplex_seeds,
        )

    # -- observation helpers -----------------------------------------------------------

    @staticmethod
    def xy(obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float64)[..., :XY_DIM]

    @staticmethod
    def velocity(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape[-1] >= 17:
            return obs[..., 15:17]
        return np.zeros(obs.shape[:-1] + (2,), dtype=np.float64)

    def discretize_xy(self, xy: np.ndarray) -> np.ndarray:
        return discretize_xy(xy, self.xy_extent, self.num_bins)

    def xy_to_bins(self, obs: np.ndarray) -> np.ndarray:
        return self.discretize_xy(self.xy(obs))

    def build_encoder_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        """AntMaze encoder observations are the raw states (no physics appended)."""
        return np.asarray(raw_obs, dtype=np.float32)

    def policy_observation(self, encoder_obs: np.ndarray) -> np.ndarray:
        return np.asarray(encoder_obs, dtype=np.float32)

    # -- reward ------------------------------------------------------------------------

    def compute_reward(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.task.reward(np.asarray(obs, dtype=np.float64)), dtype=np.float32)

    def compute_done(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.task.done(np.asarray(obs, dtype=np.float64)), dtype=bool)

    def reward_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.task.reward

    def goal_reached(self, obs: np.ndarray) -> bool:
        done = self.compute_done(np.asarray(obs).reshape(1, -1))
        return bool(np.asarray(done).reshape(-1)[0])

    # -- live / replay reset & step ----------------------------------------------------

    def reset(self, seed: Optional[int] = None,
              goal: Optional[Sequence[float]] = None,
              task: Optional[AntMazeTaskSpec] = None,
              **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if task is not None:
            self.task = task
        elif goal is not None:
            self.set_goal(goal)

        self._t = 0
        self._step_in_episode = 0
        self._episode_return = 0.0
        self.last_episode_success = False

        if self.live_env_available:
            raw_obs = self._reset_live()
        else:
            raw_obs = self._reset_replay()

        enc_obs = self.build_encoder_observation(raw_obs)
        info = {
            "task": self.task.name,
            "policy_observation": self.policy_observation(enc_obs),
            "xy": self.xy(raw_obs).astype(np.float32),
        }
        return enc_obs, info

    def _reset_live(self) -> np.ndarray:
        try:
            out = self.live_env.reset()  # type: ignore[union-attr]
        except TypeError:  # pragma: no cover - gym >= 0.26
            out = self.live_env.reset(seed=int(self.rng.integers(1 << 30)))  # type: ignore[union-attr]
        obs = out[0] if isinstance(out, tuple) else out
        return _set_live_xy(self.live_env, obs, self._start_xy())

    def _reset_replay(self) -> np.ndarray:
        if not self._episode_bounds:
            raise RuntimeError(
                "AntMazeWrapper requires either a live environment or an offline dataset."
            )
        self._episode_index = int(self.rng.integers(len(self._episode_bounds)))
        self._episode_start, end = self._episode_bounds[self._episode_index]
        obs = self._get_observations(self._episode_start)
        return _override_xy(obs, self._start_xy(), self.xy_extent, self.num_bins)

    def _start_xy(self) -> np.ndarray:
        """Evaluation rollouts start from the centre of the maze per the paper."""
        if self.center_start:
            return np.array([self.xy_extent / 2.0, self.xy_extent / 2.0], dtype=np.float64)
        if self._dataset_positions is not None and len(self._dataset_positions) > 0:
            idx = int(self.rng.integers(len(self._dataset_positions)))
            return self._dataset_positions[idx]
        return np.array([self.xy_extent / 2.0, self.xy_extent / 2.0], dtype=np.float64)

    def _get_observations(self, idx: int) -> np.ndarray:
        assert self._dataset_arrays is not None
        obs = self._dataset_arrays.get("observations")
        if obs is None:
            obs = self._dataset_arrays.get("obs")
        return np.asarray(obs[idx], dtype=np.float64)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.task is not None
        if self.live_env_available:
            raw_obs, _env_r, env_term, info = self._step_live(action)
        else:
            raw_obs, env_term, info = self._step_replay(action)

        self._t += 1
        self._step_in_episode += 1
        reward = float(np.asarray(self.compute_reward(raw_obs)).reshape(-1)[0])
        success = bool(np.asarray(self.compute_done(raw_obs)).reshape(-1)[0])
        self._episode_return += reward

        truncated = self._step_in_episode >= self.max_episode_steps or bool(info.get("truncated", False))
        terminated = bool(env_term) or success
        if terminated or truncated:
            self.num_episodes_completed += 1
            self.last_episode_return = float(self._episode_return)
            self.last_episode_success = bool(success)

        enc_obs = self.build_encoder_observation(raw_obs)
        info_out = {
            "task": self.task.name,
            "success": success,
            "policy_observation": self.policy_observation(enc_obs),
            "xy": self.xy(raw_obs).astype(np.float32),
            "truncated": truncated,
        }
        info_out.update({k: v for k, v in info.items() if k not in info_out})
        return enc_obs, reward, bool(terminated), bool(truncated), info_out

    def _step_live(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        out = self.live_env.step(np.asarray(action, dtype=np.float32))  # type: ignore[union-attr]
        if len(out) == 5:  # gym >= 0.26 API
            obs, reward, term, trunc, info = out
        else:  # pragma: no cover - legacy API
            obs, reward, term, info = out
            trunc = False
        return obs, bool(term), bool(trunc), info or {}

    def _step_replay(self, action: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        # Offline replay: advance to a nearby dataset state reached by the action.
        idx = self._episode_start + self._step_in_episode
        _, end = self._episode_bounds[self._episode_index]  # type: ignore[index]
        if idx >= end - 1:
            next_idx = end - 1
            truncated = True
        else:
            next_idx = idx + 1
            truncated = False
        obs = self._get_observations(next_idx)
        # perturb the XY by the commanded forward velocity so directional tasks
        # produce non-degenerate rewards, while staying on the dataset manifold
        xyz = self.xy(obs).copy()
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size >= 2:
            xyz = xyz + 0.5 * action[:2]
        xyz = np.clip(xyz, 0.0, self.xy_extent - 1e-3)
        obs = _override_xy(obs, xyz, self.xy_extent, self.num_bins)
        return obs, False, {"truncated": truncated}

    def get_dataset_observation(self,
                                index: Optional[int] = None,
                                random: bool = False) -> np.ndarray:
        """Return a dataset state (used to draw the K=32 encoder context samples)."""
        if not self._episode_bounds:
            raise RuntimeError("no offline dataset available for context sampling")
        if random or index is None:
            idx = int(self.rng.integers(self.num_dataset_transitions()))
        else:
            idx = int(index)
        return np.asarray(self._get_observations(idx), dtype=np.float64)

    def sample_context(self,
                       num_samples: int = 32,
                       indices: Optional[Sequence[int]] = None,
                       replace: bool = True) -> np.ndarray:
        """Sample ``num_samples`` states for the (state, reward) encoder context."""
        if indices is not None:
            return np.stack([self._get_observations(int(i)) for i in indices])
        n = self.num_dataset_transitions()
        if n == 0:
            raise RuntimeError("no offline dataset available for context sampling")
        idx = self.rng.choice(n, size=int(num_samples), replace=replace)
        return np.stack([self._get_observations(int(i)) for i in idx])

    def seed(self, seed: int) -> "AntMazeWrapper":
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(int(seed))
        return self

    def render(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - passthrough
        if self.live_env_available:
            return self.live_env.render(*args, **kwargs)  # type: ignore[union-attr]
        return None

    def close(self) -> None:
        if self.live_env_available:
            close = getattr(self.live_env, "close", None)
            if callable(close):
                close()


# --------------------------------------------------------------------------------------
# Live env helpers
# --------------------------------------------------------------------------------------


def _maybe_make_live_env(dataset_name: str) -> Any:
    """Try to build the D4RL AntMaze gym environment; return ``None`` on failure."""
    try:
        import gym  # type: ignore
        import d4rl  # noqa: F401  (registers the environments)
    except Exception:
        return None
    try:
        return gym.make(dataset_name)
    except Exception:
        return None


def _set_live_xy(env: Any, obs: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Teleport the ant to ``xy`` when a live simulator is available."""
    try:
        sim = env.unwrapped.sim  # type: ignore[union-attr]
        if hasattr(sim, "data") and hasattr(sim.data, "qpos"):
            sim.data.qpos[:2] = np.asarray(xy, dtype=np.float64)
            sim.forward()
            return np.asarray(sim.get_state(), dtype=np.float64)
    except Exception:
        pass
    return _override_xy(np.asarray(obs, dtype=np.float64), xy, DEFAULT_XY_EXTENT, NUM_XY_BINS)


def _override_xy(obs: np.ndarray, xy: np.ndarray, extent: float, num_bins: int) -> np.ndarray:
    obs = np.array(obs, dtype=np.float64, copy=True)
    flat = obs.reshape(-1, obs.shape[-1])
    flat[:, :XY_DIM] = np.asarray(xy, dtype=np.float64).reshape(-1, XY_DIM)
    return flat.reshape(obs.shape)


# --------------------------------------------------------------------------------------
# Task factories
# --------------------------------------------------------------------------------------


def goal_task(goal: Sequence[float],
              wrapper: Optional[AntMazeWrapper] = None,
              name: Optional[str] = None) -> AntMazeTaskSpec:
    """Build a goal-reaching task spec."""
    extent = wrapper.xy_extent if wrapper is not None else DEFAULT_XY_EXTENT
    num_bins = wrapper.num_bins if wrapper is not None else NUM_XY_BINS
    threshold = wrapper.goal_threshold if wrapper is not None else DEFAULT_GOAL_DISTANCE
    discrete = wrapper.discrete_reward if wrapper is not None else True
    goal_arr = np.asarray(goal, dtype=np.float64)

    def _reward(obs: np.ndarray) -> np.ndarray:
        return goal_reward(obs, goal_arr, threshold, extent, num_bins, discrete)

    def _done(obs: np.ndarray) -> np.ndarray:
        return goal_done(obs, goal_arr, threshold, extent, num_bins, discrete)

    return AntMazeTaskSpec(
        name=name or f"antmaze-goal-{int(goal_arr[0])}-{int(goal_arr[1])}",
        family="goal",
        reward_fn=_reward,
        done_fn=_done,
        goal=goal_arr,
        metadata={"threshold": threshold, "extent": extent, "num_bins": num_bins},
    )


def directional_task(direction: Sequence[float],
                     wrapper: Optional[AntMazeWrapper] = None,
                     name: Optional[str] = None) -> AntMazeTaskSpec:
    """Build a directional (XY velocity dot-product) task spec."""
    direction_arr = np.asarray(direction, dtype=np.float64)

    def _reward(obs: np.ndarray) -> np.ndarray:
        vel = AntMazeWrapper.velocity(np.asarray(obs, dtype=np.float64))
        return directional_reward(vel, direction_arr)

    def _done(obs: np.ndarray) -> np.ndarray:
        return directional_done(_reward(obs))

    return AntMazeTaskSpec(
        name=name or f"antmaze-directional-{int(direction_arr[0])}-{int(direction_arr[1])}",
        family="directional",
        reward_fn=_reward,
        done_fn=_done,
        direction=direction_arr,
        uses_velocity=True,
    )


def simplex_task(seed: int,
                 wrapper: Optional[AntMazeWrapper] = None,
                 name: Optional[str] = None) -> AntMazeTaskSpec:
    """Build a random-simplex reward field task spec."""
    extent = wrapper.xy_extent if wrapper is not None else DEFAULT_XY_EXTENT
    field = _simplex_field(int(seed), extent)

    def _reward(obs: np.ndarray) -> np.ndarray:
        xy = np.asarray(obs, dtype=np.float64)[..., :XY_DIM]
        return field(xy)

    return AntMazeTaskSpec(
        name=name or f"antmaze-simplex-{int(seed)}",
        family="simplex",
        reward_fn=_reward,
        done_fn=None,
        seed=int(seed),
    )


def path_task(task: str,
              wrapper: Optional[AntMazeWrapper] = None,
              name: Optional[str] = None) -> AntMazeTaskSpec:
    """Build a corridor-completion task spec (``path-center``/``loop``/``edges``)."""
    extent = wrapper.xy_extent if wrapper is not None else DEFAULT_XY_EXTENT
    num_bins = wrapper.num_bins if wrapper is not None else NUM_XY_BINS
    tolerance = wrapper.corridor_tolerance if wrapper is not None else 1.5

    def _reward(obs: np.ndarray) -> np.ndarray:
        return path_reward(obs, task, tolerance, extent, num_bins)

    def _done(obs: np.ndarray) -> np.ndarray:
        return path_done(obs, task, tolerance, extent, num_bins)

    return AntMazeTaskSpec(
        name=name or f"antmaze-{task}",
        family="path",
        reward_fn=_reward,
        done_fn=_done,
        metadata={"route": task, "tolerance": tolerance},
    )


def antmaze_eval_tasks(
    wrapper: Optional[AntMazeWrapper] = None,
    *,
    include_goals: bool = True,
    include_directional: bool = True,
    include_simplex: bool = True,
    include_path: bool = True,
    goals: Optional[Sequence[Sequence[float]]] = None,
    simplex_seeds: Sequence[int] = SIMPLEX_SEEDS,
) -> Dict[str, AntMazeTaskSpec]:
    """Return the full zero-shot AntMaze task suite keyed by task name."""
    tasks: Dict[str, AntMazeTaskSpec] = {}
    if include_goals:
        for goal in (goals if goals is not None else GOAL_TASK_GOALS):
            spec = goal_task(goal, wrapper)
            tasks[spec.name] = spec
    if include_directional:
        for direction in DIRECTIONAL_GOALS:
            spec = directional_task(direction, wrapper)
            tasks[spec.name] = spec
    if include_simplex:
        for seed in simplex_seeds:
            spec = simplex_task(seed, wrapper)
            tasks[spec.name] = spec
    if include_path:
        for path in PATH_TASKS:
            spec = path_task(path, wrapper)
            tasks[spec.name] = spec
    return tasks


# --------------------------------------------------------------------------------------
# Specialised wrappers / factories
# --------------------------------------------------------------------------------------


class AntMazeGoalTask(AntMazeWrapper):
    """Wrapper initialised directly on a goal-reaching task."""

    def __init__(self, goal: Sequence[float] = GOAL_TASK_GOALS[0], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_goal(goal)


class AntMazeDirectionalTask(AntMazeWrapper):
    """Wrapper initialised directly on a directional task."""

    def __init__(self, direction: Sequence[float] = DIRECTIONAL_GOALS[0], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.task = directional_task(direction, self)


class AntMazeSimplexTask(AntMazeWrapper):
    """Wrapper initialised directly on a random-simplex task."""

    def __init__(self, seed: int = SIMPLEX_SEEDS[0], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.task = simplex_task(seed, self)


class AntMazePathTask(AntMazeWrapper):
    """Wrapper initialised directly on a corridor task."""

    def __init__(self, path: str = PATH_TASKS[0], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.task = path_task(path, self)


def make_antmaze_env(
    dataset: Any = None,
    task: Optional[AntMazeTaskSpec] = None,
    *,
    dataset_name: str = ANTMAZE_DATASET,
    load_dataset: bool = True,
    **kwargs: Any,
) -> AntMazeWrapper:
    """Factory returning an :class:`AntMazeWrapper`, loading the dataset if needed."""
    if dataset is None and load_dataset:
        try:
            from ..data.d4rl_loader import load_antmaze

            dataset = load_antmaze(dataset_name)
        except Exception:
            dataset = None
    return AntMazeWrapper(dataset, task, dataset_name=dataset_name, **kwargs)


__all__ = [
    "AntMazeWrapper",
    "AntMazeConfig",
    "AntMazeTaskSpec",
    "AntMazeGoalTask",
    "AntMazeDirectionalTask",
    "AntMazeSimplexTask",
    "AntMazePathTask",
    "make_antmaze_env",
    "antmaze_eval_tasks",
    "goal_task",
    "directional_task",
    "simplex_task",
    "path_task",
    "goal_reward",
    "goal_done",
    "directional_reward",
    "directional_done",
    "simplex_reward",
    "path_reward",
    "path_done",
    "path_route",
    "discretize_xy",
    "bin_to_xy",
    "xy_distance",
    "bin_distance",
    "GOAL_TASK_GOALS",
    "DIRECTIONAL_GOALS",
    "SIMPLEX_SEEDS",
    "PATH_TASKS",
    "TASK_FAMILIES",
    "ANTMAZE_DATASET",
    "MAX_EPISODE_STEPS",
    "NUM_XY_BINS",
    "DEFAULT_XY_EXTENT",
    "DEFAULT_GOAL_DISTANCE",
]
