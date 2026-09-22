"""ExORL (cheetah / walker) online evaluation tasks for FRE.

This module implements the evaluation task suite described in Appendix C.2 of the
paper ("Environment Details / ExORL") and in the addendum section "ExORL
evaluation tasks":

* Online evaluation is performed with a *maximum* length of 1000 steps per
  trajectory.
* ``exorl-cheetah-velocity`` -- average of 4 custom tasks (``cheetah-run``,
  ``cheetah-run-backwards``, ``cheetah-walk``, ``cheetah-walk-backwards``).
  The agent receives a reward of 1 if its (forward or backward) horizontal
  velocity is at least the threshold (10 for run, 1 for walk) and the reward
  linearly decays to 0 for values below the threshold.  If the velocity points
  in the opposite direction of the target velocity the reward is 0.
* ``exorl-cheetah-goals`` -- average of 5 goal reaching tasks.  Five random
  states are selected from the offline dataset and kept fixed.  The reward is
  ``-1`` at each step unless the Euclidean distance to the goal is below
  ``0.1``, in which case the reward is ``0``.
* ``exorl-walker-velocity`` -- average of 4 custom tasks with thresholds
  ``0.1, 1, 4, 8`` (referred to as ``walker-run`` / ``walker-walk`` in the
  paper).
* ``exorl-walker-goals`` -- average of 5 goal reaching tasks (same definition
  as the cheetah ones).

Per Appendix C.2 the physics quantities used to define the true reward
functions (``horizontal_velocity``, ``torso_upright``, ``torso_height`` for
walker, ``speed`` for cheetah) are appended to the observations and this
auxiliary information *is also used during evaluation*.  Goal distances use the
underlying (non-augmented) observation coordinates with each dimension
normalized by its offline-dataset standard deviation.

The module is intentionally dependency-light: numpy is the only hard
requirement.  ``dm_control`` / ``gym`` are imported lazily inside the
environment-creation helpers so that reward-function definitions, task
registries and encoding helpers can be used (and unit tested) without an
installed simulator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - torch is optional for this module
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

from fre.priors.goal_functions import (
    DEFAULT_GOAL_THRESHOLD,
    GoalRewardFunction,
    RewardFunction,
    goal_distances,
    goal_reached_mask,
)

__all__ = [
    # constants
    "EXORL_DOMAINS",
    "EXORL_MAX_EPISODE_STEPS",
    "EXORL_GOAL_DISTANCE",
    "EXORL_AUGMENT_DIM",
    "EXORL_BASE_STATE_DIM",
    "EXORL_VELOCITY_OFFSET",
    "EXORL_ENV_TASKS",
    "WALKER_VELOCITY_THRESHOLDS",
    "CHEETAH_RUN_THRESHOLD",
    "CHEETAH_WALK_THRESHOLD",
    "EXORL_NUM_GOALS",
    "EXORL_VELOCITY_TASKS",
    "EXORL_TASK_SETS",
    # reward classes
    "ExORLVelocityReward",
    "ExORLGoalReward",
    "TaskSpec",
    # velocity helpers
    "exorl_domain",
    "exorl_velocity_index",
    "exorl_velocity",
    "exorl_physics_features",
    "exorl_score_velocity",
    # goal helpers
    "select_goal_states",
    "exorl_goal_dims",
    # builders
    "make_velocity_task",
    "make_goal_task",
    "list_task_sets",
    "task_names",
    "get_task",
    "build_task_set",
    "build_tasks",
    # envs
    "make_exorl_env",
    "reset_exorl_env",
    "make_exorl_task_env",
    "exorl_encoder_states",
    "encoding_samples_for_task",
    "encoding_samples_from_env",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXORL_DOMAINS: Tuple[str, str] = ("walker", "cheetah")

#: Online evaluation maximum length (addendum: "maximum length of 1000 steps").
EXORL_MAX_EPISODE_STEPS = 1000

#: Goal threshold used for ExORL goal-reaching tasks (Appendix C.2).
EXORL_GOAL_DISTANCE = 0.1

#: Number of appended physics dimensions per domain (Appendix C.2).
EXORL_AUGMENT_DIM: Dict[str, int] = {"walker": 4, "cheetah": 1}

#: Assumed dimensionality of the *un-augmented* observation (used only to
#: detect whether a state array already carries the appended physics fields).
#: These values follow the standard DeepMind Control state observations that
#: ExORL RND datasets ship with; pass ``base_state_dim`` explicitly to override.
EXORL_BASE_STATE_DIM: Dict[str, int] = {"walker": 24, "cheetah": 17}

#: Offset (from the end of the state vector) of the velocity entry inside the
#: appended physics block.  Walker appends
#: ``(horizontal_velocity_x, horizontal_velocity_y, torso_upright,
#: torso_height)`` so the x-velocity is the *first* appended dimension; cheetah
#: appends a single ``speed`` dimension.
EXORL_VELOCITY_OFFSET: Dict[str, int] = {"walker": -4, "cheetah": -1}

#: Mapping from evaluation task names to the dm_control (domain, task) pair.
EXORL_ENV_TASKS: Dict[str, Tuple[str, str]] = {
    "walker-run": ("walker", "run"),
    "walker-walk": ("walker", "walk"),
    "cheetah-run": ("cheetah", "run"),
    "cheetah-walk": ("cheetah", "walk"),
    "cheetah-run-backwards": ("cheetah", "run"),
    "cheetah-walk-backwards": ("cheetah", "walk"),
}

#: Walker velocity thresholds (addendum: ".1, 1, 4 and 8 respectively").
WALKER_VELOCITY_THRESHOLDS: Tuple[float, ...] = (0.1, 1.0, 4.0, 8.0)

#: Cheetah run/walk thresholds.
CHEETAH_RUN_THRESHOLD = 10.0
CHEETAH_WALK_THRESHOLD = 1.0

#: Number of fixed goal states per ExORL domain.
EXORL_NUM_GOALS = 5

#: Velocity task definitions: (task name, domain, threshold, direction).
EXORL_VELOCITY_TASKS: Tuple[Tuple[str, str, float, float], ...] = (
    ("walker-velocity-0.1", "walker", 0.1, 1.0),
    ("walker-velocity-1", "walker", 1.0, 1.0),
    ("walker-velocity-4", "walker", 4.0, 1.0),
    ("walker-velocity-8", "walker", 8.0, 1.0),
    ("cheetah-run", "cheetah", CHEETAH_RUN_THRESHOLD, 1.0),
    ("cheetah-run-backwards", "cheetah", CHEETAH_RUN_THRESHOLD, -1.0),
    ("cheetah-walk", "cheetah", CHEETAH_WALK_THRESHOLD, 1.0),
    ("cheetah-walk-backwards", "cheetah", CHEETAH_WALK_THRESHOLD, -1.0),
)

#: Aggregate task sets, mirroring the names used in Table 1 of the paper.
EXORL_TASK_SETS: Dict[str, str] = {
    "walker-velocity": "walker",
    "walker-goals": "walker",
    "cheetah-velocity": "cheetah",
    "cheetah-goals": "cheetah",
    # paper / table aliases
    "exorl-walker-speed": "walker",
    "exorl-walker-velocity": "walker",
    "exorl-walker-goals": "walker",
    "exorl-cheetah-velocity": "cheetah",
    "exorl-cheetah-goals": "cheetah",
    "exorl-walker": "walker",
    "exorl-cheetah": "cheetah",
    "all": "both",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_float_array(values: Any) -> np.ndarray:
    """Convert numpy / torch inputs to a float32 numpy array."""
    if _HAS_TORCH and isinstance(values, torch.Tensor):  # pragma: no cover
        return values.detach().cpu().numpy().astype(np.float32)
    return np.asarray(values, dtype=np.float32)


def _restore(original: Any, array: np.ndarray) -> Any:
    """Return ``array`` as the same container type as ``original``."""
    if _HAS_TORCH and isinstance(original, torch.Tensor):  # pragma: no cover
        return torch.as_tensor(array, device=original.device, dtype=original.dtype)
    return array


def exorl_domain(env_name: str) -> str:
    """Extract the ``walker`` / ``cheetah`` domain from a task or env name."""
    if env_name is None:
        raise ValueError("env_name must not be None")
    name = str(env_name).lower()
    for domain in EXORL_DOMAINS:
        if domain in name:
            return domain
    raise ValueError(f"Could not infer ExORL domain from {env_name!r}")


def exorl_velocity_index(
    env_name: str,
    state_dim: Optional[int] = None,
    augment_dim: Optional[int] = None,
) -> int:
    """Index of the velocity entry inside an (augmented) state vector.

    The ExORL encoder states are the base observations with the physics
    features appended (Appendix C.2).  Walker appends
    ``(horizontal_velocity_x, horizontal_velocity_y, torso_upright,
    torso_height)`` and cheetah appends a single ``speed`` dimension, so the
    velocity lives at a fixed offset from the end of the vector.  When
    ``state_dim`` is provided the index is returned as a non-negative position.
    """
    domain = exorl_domain(env_name)
    offset = EXORL_VELOCITY_OFFSET[domain]
    if state_dim is None:
        return offset
    state_dim = int(state_dim)
    if augment_dim is None:
        augment_dim = EXORL_AUGMENT_DIM[domain]
    base_dim = state_dim - int(augment_dim)
    if domain == "walker":
        idx = base_dim  # first appended feature == horizontal velocity x
    else:
        idx = state_dim - 1  # appended feature == speed
    if idx < 0:
        # States are not augmented (raw observations): fall back to the
        # dimension holding the x-velocity of the root body when known.
        idx = min(abs(offset), state_dim - 1)
    return int(idx)


# ---------------------------------------------------------------------------
# Physics / velocity extraction (online evaluation)
# ---------------------------------------------------------------------------
def _first_scalar(value: Any) -> Optional[float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if array.size == 0:
        return None
    out = float(array[0])
    if not np.isfinite(out):
        return None
    return out


def exorl_velocity(env_or_physics: Any, domain: Optional[str] = None) -> float:
    """Signed forward velocity used by the ExORL velocity tasks.

    * walker: ``physics.horizontal_velocity()[0]`` (x-component)
    * cheetah: the signed horizontal velocity of the root body
      (``physics.velocity()[0]`` when available, otherwise
      ``physics.speed()``)

    Reference dm_control behaviour: the cheetah root joint is a slide along the
    x axis, hence the first entry of ``physics.velocity()`` is the signed
    forward speed.  ``physics.speed()`` is used as a fallback.
    """
    if domain is None:
        domain = getattr(env_or_physics, "domain", None)
    physics = getattr(env_or_physics, "physics", env_or_physics)
    if domain is None:
        name = str(getattr(env_or_physics, "env_name", "") or "")
        try:
            domain = exorl_domain(name) if name else None
        except ValueError:
            domain = None
    if domain is None:
        domain = "walker"

    if domain == "walker":
        if hasattr(physics, "horizontal_velocity"):
            try:
                value = _first_scalar(physics.horizontal_velocity())
                if value is not None:
                    return value
            except Exception:
                pass
        return 0.0

    # cheetah: prefer a signed value
    if hasattr(physics, "velocity"):
        try:
            value = _first_scalar(physics.velocity())
            if value is not None:
                return value
        except Exception:
            pass
    if hasattr(physics, "speed"):
        try:
            value = _first_scalar(physics.speed())
            if value is not None:
                return value
        except Exception:
            pass
    return 0.0


def exorl_physics_features(env_or_physics: Any, domain: Optional[str] = None) -> np.ndarray:
    """Physics features appended to the encoder observations (Appendix C.2)."""
    if domain is None:
        domain = getattr(env_or_physics, "domain", None)
    physics = getattr(env_or_physics, "physics", env_or_physics)
    if domain is None:
        try:
            domain = exorl_domain(getattr(env_or_physics, "env_name", ""))
        except (ValueError, AttributeError):
            domain = "walker"

    values: List[float] = []
    if domain == "walker":
        for attr in ("horizontal_velocity", "torso_upright", "torso_height"):
            if not hasattr(physics, attr):
                continue
            try:
                out = np.asarray(getattr(physics, attr)(), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            values.extend(float(v) for v in out)
    else:
        for attr in ("speed",):
            if not hasattr(physics, attr):
                continue
            try:
                out = np.asarray(getattr(physics, attr)(), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            values.extend(float(v) for v in out)

    expected = EXORL_AUGMENT_DIM[domain]
    if len(values) < expected:  # pad so the layout is stable
        values.extend([0.0] * (expected - len(values)))
    return np.asarray(values[:expected], dtype=np.float32)


def exorl_score_velocity(
    velocity: Union[float, np.ndarray, Any],
    threshold: float,
    direction: float = 1.0,
    linear_decay: bool = True,
) -> Union[float, np.ndarray, Any]:
    """Paper's velocity reward.

    ``reward = clip(direction * velocity / threshold, 0, 1)`` so that the reward
    is 1 once the aligned velocity reaches the threshold and decays linearly to
    0 as the aligned velocity goes to 0.  Velocities pointing in the opposite
    direction of the target receive a reward of 0.  When ``linear_decay`` is
    ``False`` the task degrades to a step function.
    """
    threshold = float(threshold) if threshold else 1e-8
    if _HAS_TORCH and isinstance(velocity, torch.Tensor):  # pragma: no cover
        aligned = float(direction) * velocity
        if not linear_decay:
            return (aligned >= threshold).to(velocity.dtype)
        return torch.clamp(aligned / threshold, min=0.0, max=1.0)
    arr = np.asarray(velocity, dtype=np.float64)
    aligned = float(direction) * arr
    if not linear_decay:
        return (aligned >= threshold).astype(np.float32)
    return np.clip(aligned / threshold, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
class ExORLVelocityReward(RewardFunction):
    """ExORL velocity task: ``eta(s) = clip(dir * v(s) / threshold, 0, 1)``.

    The velocity is read from the velocity entry of the state vector
    (``velocity_index``); when ``velocity_index is None`` the entry is inferred
    from the state dimensionality, assuming the physics block is appended
    (Appendix C.2).  ``velocity_fn`` can be supplied to read the velocity from an
    environment's physics object during online evaluation.
    """

    family = "exorl-velocity"

    def __init__(
        self,
        threshold: float,
        direction: float = 1.0,
        velocity_index: Optional[int] = None,
        state_dim: Optional[int] = None,
        domain: Optional[str] = None,
        augment_dim: Optional[int] = None,
        scale: float = 1.0,
        linear_decay: bool = True,
        velocity_fn: Optional[Callable[[Any], float]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.threshold = float(threshold)
        self.direction = float(np.sign(direction) or 1.0)
        self.domain = exorl_domain(domain) if isinstance(domain, str) and not domain.lower() in EXORL_DOMAINS else domain
        self.scale = float(scale)
        self.linear_decay = bool(linear_decay)
        self.velocity_fn = velocity_fn
        self.name = name or f"exorl-velocity-{self.threshold:g}"
        self._state_dim = state_dim
        self._velocity_index = velocity_index

        if velocity_index is None and domain is not None:
            try:
                velocity_index = exorl_velocity_index(domain, state_dim=state_dim, augment_dim=augment_dim)
            except ValueError:
                velocity_index = None
        self.velocity_index = -1 if velocity_index is None else int(velocity_index)
        self.transform = (lambda x: x) if abs(self.scale - 1.0) < 1e-12 else (lambda x: self.scale * x)

    # -- velocity extraction -------------------------------------------------
    def _velocity(self, states: Any) -> Any:
        arr = _as_float_array(states)
        index = self.velocity_index
        if index < 0:
            index = arr.shape[-1] + index
        index = int(np.clip(index, 0, arr.shape[-1] - 1))
        velocity = arr[..., index]
        if _HAS_TORCH and isinstance(states, torch.Tensor):  # pragma: no cover
            return torch.as_tensor(velocity, device=states.device, dtype=states.dtype)
        return velocity

    # -- reward --------------------------------------------------------------
    def reward(self, states: Any) -> Any:
        velocity = self._velocity(states)
        out = exorl_score_velocity(
            velocity,
            threshold=self.threshold,
            direction=self.direction,
            linear_decay=self.linear_decay,
        )
        out = self.transform(out)
        return _restore(states, _as_float_array(out)) if not isinstance(out, float) else out

    # -- misc ----------------------------------------------------------------
    @property
    def state_dim(self) -> Optional[int]:
        return self._state_dim

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold:g}, direction={self.direction:g}, "
            f"velocity_index={self.velocity_index}, linear_decay={self.linear_decay}"
        )


class ExORLGoalReward(GoalRewardFunction):
    """Singleton goal-reaching reward used by the ExORL goal tasks.

    Reward ``-1`` at every step unless the (per-dimension normalized, augmented
    dimensions excluded) Euclidean distance to the goal is below the threshold,
    in which case the reward is ``0`` (Appendix C.2, addendum).
    """

    family = "exorl-goal"

    def __init__(
        self,
        goal: Any,
        threshold: float = EXORL_GOAL_DISTANCE,
        distance_dims: Optional[Iterable[int]] = None,
        state_std: Optional[np.ndarray] = None,
        state_dim: Optional[int] = None,
        reward_unachieved: float = -1.0,
        reward_achieved: float = 0.0,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            goal=goal,
            threshold=threshold,
            state_dim=state_dim,
            distance_dims=None if distance_dims is None else tuple(int(d) for d in distance_dims),
            state_std=state_std,
            reward_unachieved=reward_unachieved,
            reward_achieved=reward_achieved,
            name=name,
        )


# ---------------------------------------------------------------------------
# Task specification (mirrors fre.envs.antmaze_tasks.TaskSpec)
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    """Description of a single ExORL evaluation task."""

    name: str
    task_set: str = "exorl"
    reward_fn: Optional[Callable[[Any], Any]] = None
    goal: Optional[np.ndarray] = None
    success_fn: Optional[Callable[[Any], bool]] = None
    done_fn: Optional[Callable[[Any], bool]] = None
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS
    threshold: float = EXORL_GOAL_DISTANCE
    domain: Optional[str] = None
    state_std: Optional[np.ndarray] = None
    goal_dims: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- properties ---------------------------------------------------------
    @property
    def family(self) -> str:
        if self.reward_fn is None:
            return "exorl"
        return getattr(self.reward_fn, "family", type(self.reward_fn).__name__)

    @property
    def is_goal_task(self) -> bool:
        return self.goal is not None or self.family == "exorl-goal"

    # -- evaluation helpers -------------------------------------------------
    def reward(self, states: Any) -> Any:
        if self.reward_fn is None:
            return np.zeros(_as_float_array(states).shape[:-1], dtype=np.float32)
        fn = getattr(self.reward_fn, "reward", None)
        if callable(fn):
            return fn(states)
        return self.reward_fn(states)

    def encoder_reward(self, states: Any) -> Any:
        """Reward used to build ``(s, eta(s))`` pairs for zero-shot encoding."""
        return self.reward(states)

    def is_success(self, states: Any) -> Any:
        if self.success_fn is not None:
            result = self.success_fn(states)
            if isinstance(result, (bool, np.bool_)):
                return bool(result)
            arr = np.asarray(result)
            return arr if arr.ndim > 0 else bool(np.all(arr))
        if self.goal is None:
            return False
        return goal_reached_mask(
            states,
            self.goal,
            threshold=self.threshold,
            state_std=self.state_std,
            distance_dims=self.goal_dims,
        )

    def is_done(self, states: Any) -> Any:
        if self.done_fn is not None:
            result = self.done_fn(states)
            if isinstance(result, (bool, np.bool_)):
                return bool(result)
            return np.asarray(result)
        return self.is_success(states)

    def success_fn_for_wrapper(self) -> Optional[Callable[[Any], bool]]:
        if self.success_fn is not None:
            return self.success_fn
        if self.goal is None:
            return None

        def _success(state: Any) -> bool:
            mask = self.is_success(np.asarray(state, dtype=np.float32)[None, :])
            return bool(np.asarray(mask).reshape(-1)[0])

        return _success

    def copy(self, **overrides: Any) -> "TaskSpec":
        new = copy.deepcopy(self)
        for key, value in overrides.items():
            setattr(new, key, value)
        return new

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_set": self.task_set,
            "family": self.family,
            "goal": None if self.goal is None else np.asarray(self.goal).tolist(),
            "threshold": float(self.threshold),
            "domain": self.domain,
            "max_episode_steps": int(self.max_episode_steps),
            "goal_dims": None if self.goal_dims is None else list(self.goal_dims),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Goal selection / goal dimensions
# ---------------------------------------------------------------------------
def exorl_goal_dims(
    state_dim: int,
    domain: Optional[str] = None,
    augment_dim: Optional[int] = None,
) -> Tuple[int, ...]:
    """Dimensions used for goal distances.

    Appendix C.2: "Augmented information is not utilized when calculating goal
    distance", i.e. the physics dimensions appended for the encoder are dropped.
    """
    if domain is not None:
        domain = exorl_domain(domain)
        if augment_dim is None:
            augment_dim = EXORL_AUGMENT_DIM[domain]
    if augment_dim is None:
        augment_dim = 0
    base_dim = int(state_dim) - int(augment_dim)
    if base_dim <= 0:
        raise ValueError(
            f"state_dim={state_dim} is not larger than augment_dim={augment_dim}"
        )
    return tuple(range(base_dim))


def select_goal_states(
    states: Any,
    num_goals: int = EXORL_NUM_GOALS,
    seed: int = 0,
    state_std: Optional[np.ndarray] = None,
    goal_dims: Optional[Sequence[int]] = None,
    domain: Optional[str] = None,
    augment_dim: Optional[int] = None,
    min_distance: float = 0.2,
    num_candidates: int = 64,
) -> np.ndarray:
    """Pick ``num_goals`` fixed goal states from the offline dataset.

    The addendum specifies ``5 random states ... selected from the offline
    dataset and used as goal states, and kept fixed throughout the online
    evaluation``.  To avoid degenerate goals that are already reachable at
    reset, candidates are filtered so that pairwise distances (in normalized
    goal coordinates) exceed ``min_distance`` when possible.
    """
    array = _as_float_array(states)
    if array.ndim != 2:
        raise ValueError(f"states must be a 2D array, got shape {array.shape}")
    num_goals = int(num_goals)
    if num_goals <= 0:
        raise ValueError("num_goals must be positive")

    rng = np.random.default_rng(seed)
    if goal_dims is None:
        if domain is not None:
            goal_dims = exorl_goal_dims(array.shape[-1], domain, augment_dim)
        else:
            goal_dims = tuple(range(array.shape[-1]))
    goal_dims = tuple(int(d) for d in goal_dims)

    std = None if state_std is None else np.asarray(state_std, dtype=np.float64)[list(goal_dims)]
    coords = np.asarray(array[:, list(goal_dims)], dtype=np.float64)
    if std is not None:
        coords = coords / np.where(np.abs(std) < 1e-8, 1.0, std)

    pool = rng.choice(coords.shape[0], size=min(int(num_candidates), coords.shape[0]), replace=False)
    chosen: List[int] = []
    for candidate in pool:
        if len(chosen) >= num_goals:
            break
        if not chosen:
            chosen.append(int(candidate))
            continue
        dists = np.linalg.norm(coords[chosen] - coords[candidate], axis=-1)
        if np.all(dists >= float(min_distance)):
            chosen.append(int(candidate))
    # Top up if the dataset is too small / too clustered.
    if len(chosen) < num_goals:
        remaining = [i for i in range(coords.shape[0]) if i not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(remaining[: num_goals - len(chosen)])
    return np.asarray(array[chosen], dtype=np.float32)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------
def make_velocity_task(
    name: str,
    threshold: float,
    direction: float = 1.0,
    domain: Optional[str] = None,
    state_dim: Optional[int] = None,
    augment_dim: Optional[int] = None,
    velocity_index: Optional[int] = None,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    task_set: str = "exorl-velocity",
    linear_decay: bool = True,
    **kwargs: Any,
) -> TaskSpec:
    """Create an ExORL velocity task (linear decay from the threshold to 0)."""
    if domain is None:
        try:
            domain = exorl_domain(name)
        except ValueError:
            domain = None
    reward_fn = ExORLVelocityReward(
        threshold=threshold,
        direction=direction,
        velocity_index=velocity_index,
        state_dim=state_dim,
        domain=domain,
        augment_dim=augment_dim,
        linear_decay=linear_decay,
        name=name,
        **kwargs,
    )
    metadata = {
        "threshold": float(threshold),
        "direction": float(direction),
        "sort": "velocity",
    }
    return TaskSpec(
        name=name,
        task_set=task_set,
        reward_fn=reward_fn,
        goal=None,
        max_episode_steps=int(max_episode_steps),
        threshold=float(threshold),
        domain=domain,
        metadata=metadata,
    )


def make_goal_task(
    name: str,
    goal: Any,
    threshold: float = EXORL_GOAL_DISTANCE,
    domain: Optional[str] = None,
    state_dim: Optional[int] = None,
    state_std: Optional[np.ndarray] = None,
    goal_dims: Optional[Sequence[int]] = None,
    augment_dim: Optional[int] = None,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    task_set: str = "exorl-goals",
    **kwargs: Any,
) -> TaskSpec:
    """Create an ExORL goal-reaching task (``-1`` until within ``threshold``)."""
    goal_array = _as_float_array(goal).reshape(-1)
    if goal_dims is None and domain is not None:
        try:
            goal_dims = exorl_goal_dims(goal_array.shape[-1], domain, augment_dim)
        except ValueError:
            goal_dims = None
    reward_fn = ExORLGoalReward(
        goal=goal_array,
        threshold=threshold,
        distance_dims=goal_dims,
        state_std=state_std,
        state_dim=state_dim if state_dim is not None else int(goal_array.shape[-1]),
        name=name,
        **kwargs,
    )
    return TaskSpec(
        name=name,
        task_set=task_set,
        reward_fn=reward_fn,
        goal=goal_array,
        max_episode_steps=int(max_episode_steps),
        threshold=float(threshold),
        domain=domain,
        state_std=None if state_std is None else np.asarray(state_std, dtype=np.float32),
        goal_dims=None if goal_dims is None else tuple(int(d) for d in goal_dims),
        metadata={"sort": "goal"},
    )


def _velocity_task_specs(
    domain: str,
    state_dim: Optional[int] = None,
    augment_dim: Optional[int] = None,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    for name, task_domain, threshold, direction in EXORL_VELOCITY_TASKS:
        if task_domain != domain:
            continue
        tasks.append(
            make_velocity_task(
                name=name,
                threshold=threshold,
                direction=direction,
                domain=task_domain,
                state_dim=state_dim,
                augment_dim=augment_dim,
                max_episode_steps=max_episode_steps,
                task_set=f"exorl-{domain}-velocity",
            )
        )
    return tasks


def _goal_task_specs(
    domain: str,
    goal_states: Any,
    state_std: Optional[np.ndarray] = None,
    state_dim: Optional[int] = None,
    augment_dim: Optional[int] = None,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    goal_distance: float = EXORL_GOAL_DISTANCE,
) -> List[TaskSpec]:
    goals = _as_float_array(goal_states)
    if goals.ndim == 1:
        goals = goals[None, :]
    tasks: List[TaskSpec] = []
    for index in range(goals.shape[0]):
        tasks.append(
            make_goal_task(
                name=f"exorl-{domain}-goal-{index}",
                goal=goals[index],
                threshold=goal_distance,
                domain=domain,
                state_dim=state_dim,
                state_std=state_std,
                augment_dim=augment_dim,
                max_episode_steps=max_episode_steps,
                task_set=f"exorl-{domain}-goals",
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Task registries
# ---------------------------------------------------------------------------
def list_task_sets() -> Tuple[str, ...]:
    """Names of the aggregate ExORL task sets."""
    return (
        "exorl-walker-velocity",
        "exorl-walker-goals",
        "exorl-cheetah-velocity",
        "exorl-cheetah-goals",
        "exorl-walker",
        "exorl-cheetah",
        "all",
    )


def _resolve_task_set(
    task_set: str,
    domain: Optional[str],
    env_name: Optional[str],
) -> Tuple[str, str]:
    """Return ``(kind, domain)`` for an aggregate task-set name."""
    name = str(task_set)
    candidates = {
        "walker-velocity": "walker",
        "exorl-walker-velocity": "walker",
        "exorl-walker-speed": "walker",
        "walker-goals": "walker",
        "exorl-walker-goals": "walker",
        "exorl-walker": "walker",
        "cheetah-velocity": "cheetah",
        "exorl-cheetah-velocity": "cheetah",
        "cheetah-goals": "cheetah",
        "exorl-cheetah-goals": "cheetah",
        "exorl-cheetah": "cheetah",
    }
    if name in candidates:
        target = candidates[name]
        kind = "all"
        if "velocity" in name or "speed" in name:
            kind = "velocity"
        elif "goal" in name:
            kind = "goals"
        return kind, target
    if name == "all":
        return "all", "both"
    # Fall back to a single task name (e.g. "cheetah-run").
    if name in EXORL_ENV_TASKS:
        return "single", exorl_domain(name)
    if domain is not None:
        return name, exorl_domain(domain)
    if env_name is not None:
        return name, exorl_domain(env_name)
    raise ValueError(f"Unknown ExORL task set {task_set!r}")


def task_names(task_set: Optional[str] = None, domain: Optional[str] = None) -> Tuple[str, ...]:
    """Names of the individual tasks in an aggregate task set."""
    if task_set is None:
        task_set = "all" if domain is None else f"exorl-{exorl_domain(domain)}"
    kind, target = _resolve_task_set(task_set, domain, None)
    names: List[str] = []
    domains = EXORL_DOMAINS if target == "both" else (target,)
    for dom in domains:
        if kind in ("velocity", "all"):
            names.extend(n for n, d, _, _ in EXORL_VELOCITY_TASKS if d == dom)
        if kind in ("goals", "all"):
            names.extend(f"exorl-{dom}-goal-{i}" for i in range(EXORL_NUM_GOALS))
        if kind == "single":
            names.extend(n for n, d, _, _ in EXORL_VELOCITY_TASKS if d == dom)
    if kind == "single":
        return (task_set,)
    return tuple(names)


def get_task(
    name: str,
    goal: Any = None,
    state_std: Optional[np.ndarray] = None,
    state_dim: Optional[int] = None,
    augment_dim: Optional[int] = None,
    goal_states: Any = None,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    **kwargs: Any,
) -> TaskSpec:
    """Instantiate a single ExORL task by name.

    Velocity tasks (``walker-velocity-*``, ``cheetah-run``,
    ``cheetah-run-backwards``, ``cheetah-walk``, ``cheetah-walk-backwards``) are
    fully specified by the paper.  Goal tasks (``exorl-<domain>-goal-<i>``)
    require the fixed goal states, supplied either through ``goal`` or through
    ``goal_states``.
    """
    name = str(name)
    for task_name, domain, threshold, direction in EXORL_VELOCITY_TASKS:
        if name == task_name:
            return make_velocity_task(
                name=name,
                threshold=threshold,
                direction=direction,
                domain=domain,
                state_dim=state_dim,
                augment_dim=augment_dim,
                max_episode_steps=max_episode_steps,
                **kwargs,
            )

    if name in EXORL_ENV_TASKS:
        domain = exorl_domain(name)
        # e.g. "walker-run" / "walker-walk" are the paper's velocity tasks with
        # the thresholds 4 / 1 respectively.
        threshold = 4.0 if domain == "walker" and name.endswith("run") else 1.0
        direction = -1.0 if name.endswith("backwards") else 1.0
        return make_velocity_task(
            name=name,
            threshold=threshold,
            direction=direction,
            domain=domain,
            state_dim=state_dim,
            augment_dim=augment_dim,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

    if name.startswith("exorl-") and "goal" in name:
        domain = exorl_domain(name)
        index = None
        parts = name.split("-")
        if parts and parts[-1].isdigit():
            index = int(parts[-1])
        if goal is None and goal_states is not None:
            goals = _as_float_array(goal_states)
            if goals.ndim == 1:
                goals = goals[None, :]
            index = 0 if index is None else index
            goal = goals[min(index, goals.shape[0] - 1)]
        if goal is None:
            raise ValueError(
                f"Goal task {name!r} requires goal states from the offline dataset; "
                "pass `goal=` or `goal_states=`."
            )
        return make_goal_task(
            name=name,
            goal=goal,
            domain=domain,
            state_dim=state_dim,
            state_std=state_std,
            augment_dim=augment_dim,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

    raise ValueError(f"Unknown ExORL task {name!r}")


def build_task_set(
    task_set: str = "all",
    dataset_states: Any = None,
    env_name: Optional[str] = None,
    domain: Optional[str] = None,
    state_std: Optional[np.ndarray] = None,
    state_dim: Optional[int] = None,
    augment_dim: Optional[int] = None,
    goals: Any = None,
    num_goals: int = EXORL_NUM_GOALS,
    goal_seed: int = 0,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    goal_distance: float = EXORL_GOAL_DISTANCE,
    **kwargs: Any,
) -> List[TaskSpec]:
    """Build the list of ExORL tasks for an aggregate task set.

    ``task_set`` accepts ``"exorl-walker-velocity"``, ``"exorl-walker-goals"``,
    ``"exorl-cheetah-velocity"``, ``"exorl-cheetah-goals"``,
    ``"exorl-walker"``, ``"exorl-cheetah"``, ``"all"`` as well as the shorter
    aliases ``"walker-velocity"`` etc.  Goal tasks need five fixed goal states,
    taken from ``goals`` or sampled from ``dataset_states`` (addendum).
    """
    if env_name is not None and domain is None:
        domain = exorl_domain(env_name)
    kind, target = _resolve_task_set(task_set, domain, env_name)
    domains = EXORL_DOMAINS if target == "both" else (target,)

    if state_dim is None and dataset_states is not None:
        state_dim = int(_as_float_array(dataset_states).shape[-1])

    tasks: List[TaskSpec] = []
    for dom in domains:
        if kind in ("velocity", "all", "single"):
            if kind == "single":
                tasks.append(
                    get_task(
                        task_set,
                        state_std=state_std,
                        state_dim=state_dim,
                        augment_dim=augment_dim,
                        max_episode_steps=max_episode_steps,
                    )
                )
            else:
                tasks.extend(
                    _velocity_task_specs(
                        dom,
                        state_dim=state_dim,
                        augment_dim=augment_dim,
                        max_episode_steps=max_episode_steps,
                    )
                )
        if kind in ("goals", "all"):
            domain_goals = None
            if goals is not None:
                if isinstance(goals, dict):
                    domain_goals = goals.get(dom)
                else:
                    domain_goals = goals
            if domain_goals is None:
                if dataset_states is None:
                    raise ValueError(
                        "Goal-reaching ExORL tasks require the fixed goal states: pass "
                        "`dataset_states=` (offline dataset) or `goals=`."
                    )
                domain_goals = select_goal_states(
                    dataset_states,
                    num_goals=num_goals,
                    seed=goal_seed,
                    state_std=state_std,
                    domain=dom,
                    augment_dim=augment_dim,
                )
            tasks.extend(
                _goal_task_specs(
                    dom,
                    domain_goals,
                    state_std=state_std,
                    state_dim=state_dim,
                    augment_dim=augment_dim,
                    max_episode_steps=max_episode_steps,
                    goal_distance=goal_distance,
                )
            )
    return tasks


def build_tasks(task_set: str = "all", **kwargs: Any) -> List[TaskSpec]:
    """Alias for :func:`build_task_set`."""
    return build_task_set(task_set=task_set, **kwargs)


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------
class DMControlEnv:
    """Minimal gym-like adapter around a ``dm_control.suite`` environment.

    Exposes ``reset() -> (obs, info)``, ``step(action) -> (obs, reward, done,
    info)``, ``physics``, ``observation_space``/``action_space`` so that the
    wrappers in :mod:`fre.envs.reward_wrappers` can be reused.  The observation
    is the flattened DeepMind Control state (matching the ExORL RND datasets).
    """

    def __init__(
        self,
        domain: str,
        task: str,
        seed: Optional[int] = None,
        max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
        observation_mode: str = "physics_state",
        env: Any = None,
    ) -> None:
        self.domain = domain
        self.dmc_task = task
        self.env_name = f"{domain}-{task}"
        self.max_episode_steps = int(max_episode_steps)
        self.observation_mode = observation_mode
        self._steps = 0
        self._env = env if env is not None else self._load(domain, task, seed)
        self._action_low, self._action_high = self._action_bounds()

    # -- construction --------------------------------------------------------
    @staticmethod
    def _load(domain: str, task: str, seed: Optional[int]) -> Any:  # pragma: no cover
        from dm_control import suite  # local import: heavy / optional dependency

        if seed is not None:
            try:
                import random

                random.seed(seed)
                np.random.seed(seed)
            except Exception:
                pass
        return suite.load(domain_name=domain, task_name=task)

    def _action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            spec = self._env.action_spec()
            return np.asarray(spec.minimum, dtype=np.float32), np.asarray(spec.maximum, dtype=np.float32)
        except Exception:
            return np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

    # -- spaces --------------------------------------------------------------
    @property
    def physics(self) -> Any:
        return getattr(self._env, "physics", None)

    @property
    def unwrapped(self) -> Any:
        return self

    @property
    def action_space(self) -> Any:
        class _Space:
            def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
                self.low = low
                self.high = high
                self.shape = low.shape
                self.dtype = np.float32

            def sample(self) -> np.ndarray:
                return np.asarray(self.low + (self.high - self.low) * np.random.rand(*self.shape), dtype=np.float32)

        return _Space(self._action_low, self._action_high)

    @property
    def observation_space(self) -> Any:
        size = int(self._observation().shape[-1])

        class _Space:
            def __init__(self, dim: int) -> None:
                self.shape = (dim,)
                self.dtype = np.float32
                self.low = np.full(dim, -np.inf, dtype=np.float32)
                self.high = np.full(dim, np.inf, dtype=np.float32)

        return _Space(size)

    # -- observations --------------------------------------------------------
    def _observation(self) -> np.ndarray:
        physics = self.physics
        if physics is not None and self.observation_mode in ("physics_state", "auto"):
            try:
                return np.asarray(physics.state(), dtype=np.float32).reshape(-1)
            except Exception:
                pass
        try:
            obs = self._env.observation()
        except Exception:
            return np.zeros(1, dtype=np.float32)
        if isinstance(obs, dict):
            parts = []
            for value in obs.values():
                parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
            return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def base_observation(self) -> np.ndarray:
        return self._observation()

    def physics_features(self) -> np.ndarray:
        return exorl_physics_features(self, domain=self.domain)

    def encoder_observation(self) -> np.ndarray:
        """Observation with the appended physics features (Appendix C.2)."""
        obs = self._observation()
        return np.concatenate([obs, self.physics_features()]).astype(np.float32)

    # -- gym-like API --------------------------------------------------------
    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        time_step = self._env.reset()
        self._steps = 0
        info: Dict[str, Any] = {"time_step": time_step}
        if hasattr(time_step, "observation"):
            info["dm_control_observation"] = time_step.observation
        return self._observation(), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(self._action_low.shape)
        # dm_control's `suite.load` environments expect actions in [-1, 1].
        time_step = self._env.step(action)
        self._steps += 1
        obs = self._observation()
        reward = float(getattr(time_step, "reward", 0.0) or 0.0)
        done = bool(getattr(time_step, "last", False))
        truncated = self._steps >= self.max_episode_steps
        info = {
            "time_step": time_step,
            "truncated": truncated and not done,
            "physics": self.physics,
        }
        return obs, reward, done, info

    def close(self) -> None:
        try:
            self._env.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    def __repr__(self) -> str:  # pragma: no cover
        return f"DMControlEnv({self.env_name}, obs_dim={self.observation_space.shape[0]})"


def make_exorl_env(
    env_name: str = "walker-run",
    seed: Optional[int] = None,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    observation_mode: str = "physics_state",
    env: Any = None,
    **kwargs: Any,
) -> Any:
    """Create the online ExORL environment for a task name.

    ``env_name`` may be an evaluation task name (``cheetah-run-backwards``,
    ``walker-velocity-4`` ...) or a ``domain-task`` pair (``walker-walk``).
    """
    name = str(env_name)
    if name in EXORL_ENV_TASKS:
        domain, task = EXORL_ENV_TASKS[name]
    else:
        domain = exorl_domain(name)
        if "-" in name:
            task = name.split("-", 1)[1]
        else:
            task = "run"
        if name.endswith("backwards"):
            task = task.replace("-backwards", "")
        task = task.split("-")[0] if task not in ("run", "walk") else task
    if env is not None:
        return DMControlEnv(
            domain,
            task,
            seed=seed,
            max_episode_steps=max_episode_steps,
            observation_mode=observation_mode,
            env=env,
        )
    return DMControlEnv(
        domain,
        task,
        seed=seed,
        max_episode_steps=max_episode_steps,
        observation_mode=observation_mode,
    )


def reset_exorl_env(env: Any, seed: Optional[int] = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
    """Reset an ExORL environment, normalizing gym / dm_control signatures."""
    from fre.envs.reward_wrappers import call_env_reset

    if seed is not None:
        try:
            env.seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass
    return call_env_reset(env, **kwargs)


def make_exorl_task_env(
    task: TaskSpec,
    env_name: Optional[str] = None,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    env: Any = None,
    count_success_as_done: bool = True,
    physics_augment: bool = True,
    state_fn: Optional[Callable[[Any], np.ndarray]] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Wrap an ExORL environment with a task reward function.

    The wrapper computes ``eta(s)`` from the *augmented* observation (the physics
    information is also used during evaluation, per the addendum), while the
    policy/encoder consume the base observation separately.
    """
    from fre.envs.reward_wrappers import wrap_env

    env_name = env_name or task.metadata.get("env_name") or f"{task.domain}-run"
    base_env = env if env is not None else make_exorl_env(
        env_name,
        seed=seed,
        max_episode_steps=max_episode_steps or task.max_episode_steps,
    )

    if state_fn is None:
        if physics_augment and hasattr(base_env, "encoder_observation"):
            state_fn = lambda obs=None, _env=base_env: _env.encoder_observation()  # noqa: E731
        else:

            def state_fn(obs=None):  # type: ignore[misc]
                return np.asarray(obs, dtype=np.float32)

    wrapped = base_env if physics_augment else base_env
    return wrap_env(
        wrapped,
        reward_fn=task.reward_fn,
        state_fn=state_fn,
        success_fn=task.success_fn_for_wrapper(),
        done_fn=None,
        max_episode_steps=max_episode_steps or task.max_episode_steps,
        domain="exorl",
        count_success_as_done=count_success_as_done,
        name=name or task.name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Zero-shot encoding helpers
# ---------------------------------------------------------------------------
def exorl_encoder_states(
    states: Any,
    env_name: Optional[str] = None,
    domain: Optional[str] = None,
    base_state_dim: Optional[int] = None,
    augment_dim: Optional[int] = None,
    augmented: Optional[bool] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Prepare ExORL states for the FRE encoder.

    Training-time encoder states are base observations with the physics features
    appended (Appendix C.2).  If the given states already contain the appended
    dimensions they are returned unchanged; otherwise the physics features are
    reconstructed through :func:`fre.data.preprocessing.augment_exorl_physics`
    (which replays the dataset through ``dm_control``) and, failing that, are
    padded with zeros so the encoder input dimensionality stays consistent.
    """
    array = _as_float_array(states)
    if domain is None:
        domain = exorl_domain(env_name) if env_name is not None else None
    if augment_dim is None:
        augment_dim = EXORL_AUGMENT_DIM.get(domain or "", 0)
    if base_state_dim is None:
        base_state_dim = EXORL_BASE_STATE_DIM.get(domain or "")

    if augmented is None and base_state_dim is not None:
        augmented = array.shape[-1] >= int(base_state_dim) + int(augment_dim)
    if augmented:
        return array.astype(np.float32)

    if domain is not None and augment_dim:
        try:
            from fre.data.preprocessing import augment_exorl_physics

            augmented_states = augment_exorl_physics(
                env_name=env_name or domain, observations=array, return_info=False, **kwargs
            )
            augmented_states = _as_float_array(augmented_states)
            if augmented_states.shape[-1] == array.shape[-1] + int(augment_dim):
                return augmented_states.astype(np.float32)
        except Exception:
            pass
        pad = np.zeros(array.shape[:-1] + (int(augment_dim),), dtype=np.float32)
        return np.concatenate([array, pad], axis=-1).astype(np.float32)
    return array.astype(np.float32)


def _normalize_rewards(rewards: np.ndarray) -> np.ndarray:
    """Per-reward-function min/max rescaling to ``[0, 1]`` (encoder input)."""
    r = np.asarray(rewards, dtype=np.float32)
    if r.size == 0:
        return r
    lo = np.min(r)
    hi = np.max(r)
    if abs(hi - lo) < 1e-8:
        return np.zeros_like(r)
    return (r - lo) / (hi - lo)


def encoding_samples_for_task(
    task: TaskSpec,
    dataset_states: Any,
    num_samples: int = 32,
    rng: Optional[np.random.Generator] = None,
    replace_last_with_goal: bool = False,
    normalize_rewards: bool = True,
    state_std: Optional[np.ndarray] = None,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Sample ``(s, eta(s))`` pairs used to encode a task into ``z``.

    States are drawn uniformly from the offline dataset, rescaled by the
    (augmented) physics layout expected by the encoder, and labelled by the
    test-task reward function.  Returns a dict with ``states``/``rewards``
    (numpy, or torch when ``device`` is given) plus bookkeeping entries.
    """
    states = exorl_encoder_states(
        dataset_states,
        domain=task.domain,
        state_std=state_std,
    )
    num_samples = int(num_samples)
    if states.shape[0] < num_samples:
        raise ValueError(
            f"dataset has {states.shape[0]} states, cannot sample {num_samples} encoding pairs"
        )
    if rng is None:
        rng = np.random.default_rng(0)
    indices = rng.choice(states.shape[0], size=num_samples, replace=False)
    encoder_states = states[indices]

    if task.is_goal_task and replace_last_with_goal and task.goal is not None:
        goal = np.asarray(task.goal, dtype=np.float32).reshape(-1)
        if goal.shape[-1] == encoder_states.shape[-1]:
            encoder_states = encoder_states.copy()
            encoder_states[-1] = goal
    elif task.is_goal_task and task.goal is not None:
        goal = np.asarray(task.goal, dtype=np.float32).reshape(-1)
        if goal.shape[-1] == encoder_states.shape[-1]:
            idx = int(rng.integers(0, encoder_states.shape[0]))
            encoder_states[idx] = goal

    rewards = np.asarray(task.reward(encoder_states), dtype=np.float32).reshape(-1)
    if normalize_rewards:
        rewards = _normalize_rewards(rewards)

    out: Dict[str, Any] = {
        "states": encoder_states.astype(np.float32),
        "rewards": rewards.astype(np.float32),
        "indices": np.asarray(indices),
        "task": task.name,
        "discretize_xy": False,
    }
    if device is not None and _HAS_TORCH:  # pragma: no cover
        out["states"] = torch.as_tensor(out["states"], device=device)
        out["rewards"] = torch.as_tensor(out["rewards"], device=device)
    return out


def encoding_samples_from_env(
    task: TaskSpec,
    env: Any,
    num_samples: int = 32,
    policy: Optional[Callable[[Any], Any]] = None,
    seed: Optional[int] = None,
    normalize_rewards: bool = True,
    max_steps: Optional[int] = None,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Collect ``(s, eta(s))`` pairs by rolling out ``env`` under ``policy``.

    Useful when no offline dataset is available; a random policy is used by
    default and episodes are stepped until ``num_samples`` pairs are gathered
    (bounded by ``max_steps``).
    """
    rng = np.random.default_rng(0 if seed is None else seed)
    obs, _ = reset_exorl_env(env, seed=seed)
    budget = int(max_steps if max_steps is not None else max(10 * num_samples, num_samples))

    collected_states: List[np.ndarray] = []
    while len(collected_states) < num_samples and budget > 0:
        if hasattr(env, "encoder_observation"):
            state = np.asarray(env.encoder_observation(), dtype=np.float32).reshape(-1)
        else:
            state = np.asarray(obs, dtype=np.float32).reshape(-1)
        collected_states.append(state)

        if policy is None:
            action = rng.uniform(-1.0, 1.0, size=getattr(env.action_space, "shape", (1,))).astype(np.float32)
        else:
            action = policy(obs)
        obs, _, done, _ = env.step(action)
        budget -= 1
        if done:
            obs, _ = reset_exorl_env(env, seed=int(rng.integers(0, 2 ** 31 - 1)))

    states = np.asarray(collected_states[:num_samples], dtype=np.float32)
    rewards = np.asarray(task.reward(states), dtype=np.float32).reshape(-1)
    if normalize_rewards:
        rewards = _normalize_rewards(rewards)
    out: Dict[str, Any] = {
        "states": states,
        "rewards": rewards,
        "task": task.name,
        "discretize_xy": False,
    }
    if device is not None and _HAS_TORCH:  # pragma: no cover
        out["states"] = torch.as_tensor(out["states"], device=device)
        out["rewards"] = torch.as_tensor(out["rewards"], device=device)
    return out
