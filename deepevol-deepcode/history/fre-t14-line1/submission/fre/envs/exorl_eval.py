"""ExORL zero-shot online evaluation suite for FRE (paper Section 5, Appendix C.2).

This module implements *everything* needed to reproduce the four ExORL rows of
Table 1 of the FRE paper (``Zero-Shot Reinforcement Learning via Functional
Reward Encodings``):

==========================  =====================================================
Table 1 row                 contents
==========================  =====================================================
``exorl-walker-goals``      average of 5 goal-reaching tasks (fixed dataset goals)
``exorl-walker-velocity``   average of 4 velocity tasks (thresholds 0.1/1/4/8)
``exorl-cheetah-goals``     average of 5 goal-reaching tasks (fixed dataset goals)
``exorl-cheetah-velocity``  average of 4 tasks (run 10, walk 1, and backwards)
``exorl-all``               average of the four rows above
==========================  =====================================================

Paper specification (verbatim, "ExORL evaluation tasks" / Appendix C.2):

* online evaluation uses a *maximum* length of **1000 steps per trajectory**;
* velocity rewards: "The reward is 1 if the velocity is at least the threshold
  value and linearly decays to 0 for values below the threshold value. If the
  agent's horizontal velocity is in the opposite direction of the target
  velocity, the reward is 0."  Walker thresholds are ``0.1, 1, 4, 8``; Cheetah
  run uses ``10`` and walk uses ``1`` (with forward and backward variants);
* goal rewards: "The agent is assigned a reward of -1 at each step unless it is
  within a threshold distance of 0.1 of the goal state, in which case it is
  assigned a reward of 0", where "The distance is the euclidean distance between
  the agent's current state and the goal state." Goals are 5 states selected
  from the offline dataset and kept fixed throughout the evaluation;
* "Goals in ExORL are computed when the Euclidean distance between the current
  state and the goal state is less than 0.1. Each state dimension is normalized
  according to the standard deviation along that dimension within the offline
  dataset. Augmented information is not utilized when calculating goal
  distance.";
* physics augmentation (Appendix C.2): ``horizontal_velocity``,
  ``torso_upright``, ``torso_height`` are appended to Walker states and
  ``speed`` to Cheetah states -- this is "necessary only for the encoder
  network"; "performance was not greatly affected whether or not the value
  functions and policy networks have access to the auxiliary information, and
  are instead trained on the underlying observation space of the environment."

Return normalisation.  The paper reports "results ... normalized between 0 and
100".  We use a fixed reference horizon (= ``max_episode_steps``, i.e. 1000) so
that the normalisation does not depend on early termination:

* velocity tasks: per-step reward in ``[0, 1]`` -> score = ``100 * mean(reward)``
  (equivalently ``100 * (return - 0) / (1000 - 0)``);
* goal tasks: per-step reward ``-1`` off goal, ``0`` at goal -> score =
  ``100 * (return + 1000) / 1000`` = 100 * (fraction of steps at the goal).

Both conventions are documented on the corresponding task classes
(:attr:`ExoRLTask.min_return` / :attr:`ExoRLTask.max_return`).

Dependencies are layered exactly like :mod:`fre.envs.d4rl_loader`: a real
``dm_control``/``exorl`` simulator is used when available, and a deterministic
:class:`SyntheticExoRLEnv` fallback (which exposes the same
``env.physics`` interface) keeps the whole evaluation pipeline runnable and
unit-testable offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    # constants
    "EXORL_MAX_EPISODE_STEPS",
    "EXORL_ENCODER_SAMPLES",
    "EXORL_GOAL_THRESHOLD",
    "EXORL_NUM_GOALS",
    "EXORL_WALKER_VELOCITY_THRESHOLDS",
    "EXORL_CHEETAH_RUN_THRESHOLD",
    "EXORL_CHEETAH_WALK_THRESHOLD",
    "EXORL_CHEETAH_VELOCITY_TASKS",
    "EXORL_WALKER_PHYSICS",
    "EXORL_CHEETAH_PHYSICS",
    "EXORL_PHYSICS_DIM",
    "EXORL_DOMAINS",
    "EXORL_SUITE_ROWS",
    "EXORL_TABLE1_REFERENCE",
    "EXORL_TABLE1_FRE_PER_TASK",
    # physics helpers
    "physics_features",
    "walker_physics_from_observation",
    "cheetah_physics_from_observation",
    "augment_observations",
    "encode_states_for_domain",
    # dataset helpers
    "as_state_array",
    "as_physics_array",
    "select_goal_states",
    "dataset_state_std",
    "normalized_goal_distance",
    # tasks
    "ExoRLTask",
    "VelocityTask",
    "GoalReachingTask",
    "make_velocity_tasks",
    "make_goal_tasks",
    "get_task_suite",
    "make_exorl_task_suite",
    # envs / rollouts
    "SyntheticExoRLEnv",
    "ExoRLEvalWrapper",
    "make_exorl_env",
    "ExoRLEpisodeResult",
    "rollout_episode",
    "evaluate_task",
    "evaluate_suite",
    "evaluate_exorl_suite",
    "make_exorl_policy_fn",
    "encode_task_latent",
]

# ---------------------------------------------------------------------------
# Constants (paper: "ExORL evaluation tasks", Appendix C.2)
# ---------------------------------------------------------------------------

EXORL_DOMAINS: Tuple[str, ...] = ("exorl_walker", "exorl_cheetah")

#: "online evaluation is performed with a maximum length of 1000 steps per trajectory".
EXORL_MAX_EPISODE_STEPS: int = 1000

#: FRE evaluates zero-shot using K=32 (state, reward) samples.
EXORL_ENCODER_SAMPLES: int = 32

#: "... within a threshold distance of 0.1 of the goal state".
EXORL_GOAL_THRESHOLD: float = 0.1

#: "5 random states are selected from the offline dataset and used as goal states".
EXORL_NUM_GOALS: int = 5

#: "The 4 tasks use values of 0.1, 1, 4, and 8 respectively."
EXORL_WALKER_VELOCITY_THRESHOLDS: Tuple[float, ...] = (0.1, 1.0, 4.0, 8.0)

#: "the agent is assigned a reward if the agent's horizontal forward velocity is at least 10".
EXORL_CHEETAH_RUN_THRESHOLD: float = 10.0
#: "cheetah-walk: Same as cheetah-run, but the agent is rewarded for a velocity of at least 1."
EXORL_CHEETAH_WALK_THRESHOLD: float = 1.0

#: (name, threshold, direction) with direction +1 = forward, -1 = backwards.
EXORL_CHEETAH_VELOCITY_TASKS: Tuple[Tuple[str, float, int], ...] = (
    ("cheetah-run", EXORL_CHEETAH_RUN_THRESHOLD, +1),
    ("cheetah-walk", EXORL_CHEETAH_WALK_THRESHOLD, +1),
    ("cheetah-run-backwards", EXORL_CHEETAH_RUN_THRESHOLD, -1),
    ("cheetah-walk-backwards", EXORL_CHEETAH_WALK_THRESHOLD, -1),
)

#: Walker velocity variants: the paper bundles "walker-run"/"walker-walk" into 4 thresholds.
EXORL_WALKER_VELOCITY_TASKS: Tuple[Tuple[str, float, int], ...] = tuple(
    (f"walker-velocity-{i}", float(thr), +1)
    for i, thr in enumerate(EXORL_WALKER_VELOCITY_THRESHOLDS)
)

#: Appendix C.2 physics augmentation (encoder input only).
EXORL_WALKER_PHYSICS: Tuple[str, ...] = ("horizontal_velocity", "torso_upright", "torso_height")
EXORL_CHEETAH_PHYSICS: Tuple[str, ...] = ("speed",)
EXORL_PHYSICS_DIM: Dict[str, int] = {"exorl_walker": 3, "exorl_cheetah": 1}

#: Table-1 suite (row) names in evaluation order.
EXORL_SUITE_ROWS: Tuple[str, ...] = (
    "exorl-walker-goals",
    "exorl-walker-velocity",
    "exorl-cheetah-goals",
    "exorl-cheetah-velocity",
)

#: Reference numbers from Table 1 (mean +/- std over 5 seeds) for validation printing.
EXORL_TABLE1_REFERENCE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "exorl-walker-goals": {"FRE": (94.0, 2.0), "FB": (58.0, 30.0), "SF": (100.0, 0.0),
                           "GC-IQL": (92.0, 4.0), "GC-BC": (52.0, 18.0), "OPAL-10": (88.0, 8.0)},
    "exorl-cheetah-goals": {"FRE": (58.0, 8.0), "FB": (1.0, 2.0), "SF": (0.0, 0.0),
                            "GC-IQL": (100.0, 0.0), "GC-BC": (14.0, 6.0), "OPAL-10": (0.0, 0.0)},
    "exorl-walker-velocity": {"FRE": (34.0, 13.0), "FB": (64.0, 1.0), "SF": (38.0, 4.0),
                              "OPAL-10": (8.0, 0.0)},
    "exorl-cheetah-velocity": {"FRE": (20.0, 2.0), "FB": (51.0, 3.0), "SF": (25.0, 3.0),
                               "OPAL-10": (17.0, 8.0)},
    "exorl-all": {"FRE": (51.5, 6.3), "FB": (43.4, 9.1), "SF": (40.9, 1.9), "OPAL-10": (28.2, 4.0)},
}

#: FRE per-task reference (Table 1) for quick console comparison.
EXORL_TABLE1_FRE_PER_TASK: Dict[str, Tuple[float, float]] = {
    "exorl-walker-goals": (94.0, 2.0),
    "exorl-cheetah-goals": (58.0, 8.0),
    "exorl-walker-velocity": (34.0, 13.0),
    "exorl-cheetah-velocity": (20.0, 2.0),
    "exorl-all": (51.5, 6.3),
}

# Observation dimensionalities of the DeepMind Control Suite tasks used by ExORL
# (walker: 14 orientation + 1 height + 9 velocity = 24; cheetah: 17).
WALKER_OBS_DIM: int = 24
CHEETAH_OBS_DIM: int = 17

# Nominal torso height of the walker, used only by the observation-based physics
# fallback (dm_control reports the true value through ``physics.torso_height()``).
WALKER_NOMINAL_HEIGHT: float = 1.2


# ---------------------------------------------------------------------------
# Physics features (Appendix C.2)
# ---------------------------------------------------------------------------
def _scalar_from_method(obj: Any, names: Sequence[str]) -> Optional[float]:
    """Call the first available zero-argument method in ``names`` and return a float."""
    if obj is None:
        return None
    for name in names:
        fn = getattr(obj, name, None)
        if not callable(fn):
            continue
        try:
            value = np.asarray(fn(), dtype=np.float64).ravel()
        except Exception:  # pragma: no cover - exotic physics objects
            continue
        if value.size:
            return float(value[0])
    return None


def _walker_qpos_qvel_from_obs(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Best-effort (qpos, qvel) reconstruction from a dm_control Walker observation.

    dm_control Walker observations are ``concat(orientations(14), height(1),
    velocity(9))``.  The velocities are expressed in the torso frame, which is a
    good proxy for the true physics quantities when the (MuJoCo) model itself is
    unavailable.
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    qvel = np.zeros(9, dtype=np.float64)
    qvel[: min(9, obs.size)] = obs[-min(9, obs.size):]
    qpos = np.zeros(7, dtype=np.float64)
    qpos[2] = float(obs[-10]) if obs.size >= 10 else WALKER_NOMINAL_HEIGHT
    return qpos, qvel


def walker_physics_from_observation(obs: np.ndarray) -> np.ndarray:
    """``[horizontal_velocity, torso_upright, torso_height]`` from a Walker observation.

    Only used when the environment does not expose a MuJoCo ``physics`` object
    (see :func:`physics_features`).  ``torso_upright`` is approximated by the
    torso height relative to the nominal standing height, which is monotone in
    the uprightness of the torso.
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    height = float(obs[-10]) if obs.size >= 10 else WALKER_NOMINAL_HEIGHT
    # Root translational velocity in the torso frame (obs[-9], obs[-8]).
    vx = float(obs[-9]) if obs.size >= 9 else 0.0
    torso_upright = float(np.clip(height / WALKER_NOMINAL_HEIGHT, -1.0, 1.0))
    return np.array([vx, torso_upright, height], dtype=np.float64)


def cheetah_physics_from_observation(obs: np.ndarray) -> np.ndarray:
    """``[speed]`` from a Cheetah observation.

    dm_control Cheetah observations are ``concat(position(9), velocity(8))``;
    the horizontal speed is the root x-velocity, which is the first velocity
    component (index 9 of 17).
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    if obs.size >= 10:
        speed = float(obs[-8])
    elif obs.size >= 1:
        speed = float(obs[0])
    else:  # pragma: no cover
        speed = 0.0
    return np.array([speed], dtype=np.float64)


def physics_features(
    domain: str,
    obs: np.ndarray,
    env: Any = None,
    physics: Optional[Any] = None,
    qpos: Optional[Any] = None,
    qvel: Optional[Any] = None,
) -> np.ndarray:
    """Return the Appendix C.2 physics features ``[f_1, ..., f_d]`` for ``domain``.

    Resolution order (most accurate first):

    1. an explicitly provided ``physics`` array/tensor;
    2. the live MuJoCo model available as ``env.physics`` (dm_control), i.e. the
       exact ``physics.horizontal_velocity()`` / ``torso_upright()`` /
       ``torso_height()`` (Walker) and ``physics.speed()`` (Cheetah) calls from
       Appendix C.2 -- with a ``qpos``/``qvel`` route through
       :mod:`fre.envs.d4rl_loader` in between when those are supplied;
    3. an observation-based approximation so that the pipeline stays runnable
       without MuJoCo.
    """
    domain = _canonical_domain(domain)

    if physics is not None:
        arr = np.asarray(physics, dtype=np.float64).ravel()
        if arr.size:
            return arr

    # 2a. explicit qpos/qvel -> kinematic model (shared with the dataset loader).
    if qpos is not None and qvel is not None:
        try:
            from fre.envs.d4rl_loader import cheetah_physics as _ch, walker_physics as _wa

            if domain == "exorl_walker":
                return np.asarray(_wa(np.asarray(qpos), np.asarray(qvel)), dtype=np.float64).ravel()
            return np.asarray(_ch(np.asarray(qpos), np.asarray(qvel)), dtype=np.float64).ravel()
        except Exception:  # pragma: no cover - loader unavailable / bad shapes
            pass

    # 2b. live dm_control physics object.
    ph = getattr(env, "physics", None) if env is not None else None
    if ph is not None:
        if domain == "exorl_walker":
            hv = _scalar_from_method(ph, ("horizontal_velocity", "torsov"))
            up = _scalar_from_method(ph, ("torso_upright",))
            hg = _scalar_from_method(ph, ("torso_height", "height"))
            if hv is not None and up is not None and hg is not None:
                return np.array([hv, up, hg], dtype=np.float64)
        else:
            sp = _scalar_from_method(ph, ("speed", "horizontal_velocity"))
            if sp is not None:
                return np.array([sp], dtype=np.float64)
            # fall back to the root x-velocity of the MuJoCo model.
            try:
                return np.array([float(np.asarray(ph.data.qvel).ravel()[0])], dtype=np.float64)
            except Exception:  # pragma: no cover
                pass

    # 3. observation-based approximation.
    if domain == "exorl_walker":
        return walker_physics_from_observation(obs)
    return cheetah_physics_from_observation(obs)


def augment_observations(
    domain: str,
    observations: np.ndarray,
    physics: Optional[Union[np.ndarray, Any]] = None,
) -> np.ndarray:
    """Concatenate the ExORL physics features onto observations (encoder input).

    Appendix C.2: the auxiliary information "is necessary only for the encoder
    network".  Shapes: ``observations`` ``(N, obs_dim)`` -> ``(N, obs_dim + d)``.
    """
    obs = np.asarray(observations, dtype=np.float64)
    single = obs.ndim == 1
    if single:
        obs = obs[None]
    if physics is None:
        phys = np.stack([physics_features(domain, o) for o in obs], axis=0)
    else:
        phys = np.asarray(physics, dtype=np.float64)
        if phys.ndim == 1:
            phys = phys[None]
        if phys.shape[0] == 1 and obs.shape[0] > 1:
            phys = np.repeat(phys, obs.shape[0], axis=0)
    out = np.concatenate([obs, phys], axis=-1)
    return out[0] if single else out


#: Backwards-compatible alias (the plan refers to "physics augmentation for the encoder").
encode_states_for_domain = augment_observations


def _canonical_domain(domain: str) -> str:
    """Map domain aliases (``walker``, ``cheetah``) onto ``exorl_walker``/``exorl_cheetah``."""
    if domain is None:
        return "exorl_walker"
    d = str(domain).lower()
    if d in ("exorl_walker", "walker", "exorl-walker", "walk"):
        return "exorl_walker"
    if d in ("exorl_cheetah", "cheetah", "exorl-cheetah"):
        return "exorl_cheetah"
    return d


def obs_dim_for(domain: str) -> int:
    """Observation dimensionality of the underlying (non-augmented) ExORL task."""
    return WALKER_OBS_DIM if _canonical_domain(domain) == "exorl_walker" else CHEETAH_OBS_DIM


def action_dim_for(domain: str) -> int:
    """Action dimensionality of the walker (6) and cheetah (6) tasks."""
    return 6


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def as_state_array(dataset: Any) -> np.ndarray:
    """Extract a ``(N, obs_dim)`` float array of states from a dataset-like object."""
    if dataset is None:
        raise ValueError("An offline dataset is required to select ExORL goal states.")
    if isinstance(dataset, np.ndarray):
        return np.asarray(dataset, dtype=np.float64)
    for attr in ("states", "observations", "obs"):
        value = getattr(dataset, attr, None)
        if value is not None:
            arr = value
            if hasattr(arr, "detach"):  # torch tensor
                arr = arr.detach().cpu().numpy()
            return np.asarray(arr, dtype=np.float64)
    if isinstance(dataset, dict):
        for key in ("states", "observations", "obs"):
            if key in dataset:
                return np.asarray(dataset[key], dtype=np.float64)
    raise TypeError(f"Cannot extract states from dataset of type {type(dataset)!r}")


def as_physics_array(dataset: Any, domain: str) -> Optional[np.ndarray]:
    """Extract (or recompute) the physics features stored alongside a dataset."""
    phys = getattr(dataset, "physics", None)
    if phys is None and isinstance(dataset, dict):
        phys = dataset.get("physics")
    if phys is not None:
        if hasattr(phys, "detach"):
            phys = phys.detach().cpu().numpy()
        arr = np.asarray(phys, dtype=np.float64)
        if arr.ndim == 2 and arr.size:
            return arr
    return None


def dataset_state_std(dataset: Any, eps: float = 1e-6) -> np.ndarray:
    """Per-dimension standard deviation used to normalise ExORL goal distances.

    "Each state dimension is normalized according to the standard deviation along
    that dimension within the offline dataset."  Prefers a precomputed
    ``state_std``/``obs_std`` attribute (see :mod:`fre.rl.replay_buffer`) and
    otherwise computes it from the stored states.
    """
    for attr in ("state_std", "obs_std"):
        value = getattr(dataset, attr, None)
        if value is not None:
            std = np.asarray(value, dtype=np.float64).ravel()
            if std.size:
                return np.maximum(std, eps)
    states = as_state_array(dataset)
    if states.ndim == 1:
        states = states[None]
    return np.maximum(states.std(axis=0), eps)


def normalized_goal_distance(
    states: np.ndarray,
    goals: np.ndarray,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Std-normalised Euclidean distance ("Augmented information is not utilized").

    Supports both a single goal (returns ``(N,)``) and several goals (returns
    ``(N, num_goals)``).
    """
    states = np.asarray(states, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    if states.ndim == 1:
        states = states[None]
    single_goal = goals.ndim == 1
    if single_goal:
        goals = goals[None]
    dims = min(states.shape[-1], goals.shape[-1])
    states, goals = states[..., :dims], goals[..., :dims]
    if std is None:
        std = np.maximum(states.std(axis=0), eps)
    std = np.maximum(np.asarray(std, dtype=np.float64).ravel()[:dims], eps)
    diff = (states[:, None, :] - goals[None, :, :]) / std
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist[:, 0] if single_goal else dist


def select_goal_states(
    dataset: Any,
    num_goals: int = EXORL_NUM_GOALS,
    seed: int = 0,
    indices: Optional[Sequence[int]] = None,
    dims: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select the fixed goal states used for goal-reaching (paper: "5 random states").

    The selection is *deterministic* given ``(dataset, num_goals, seed)`` so that
    goals are "kept fixed throughout the online evaluation" and identical across
    agents being compared.

    Returns ``(goals (num_goals, dims), indices (num_goals,))``.
    """
    states = as_state_array(dataset)
    if states.ndim == 1:
        states = states[None]
    n = states.shape[0]
    if n == 0:
        raise ValueError("Offline dataset contains no states; cannot select goals.")
    num_goals = int(max(1, min(num_goals, n)))
    if indices is None:
        rng = np.random.RandomState(seed)
        pick = rng.choice(n, size=num_goals, replace=False)
    else:
        pick = np.asarray(list(indices), dtype=np.int64)[:num_goals]
    goals = states[pick]
    if dims is not None:
        goals = goals[:, : int(dims)]
    return np.ascontiguousarray(goals, dtype=np.float64), pick.astype(np.int64)


# ---------------------------------------------------------------------------
# Task base class
# ---------------------------------------------------------------------------
class ExoRLTask:
    """Base class for ExORL zero-shot evaluation tasks.

    Reward convention and return normalisation follow the paper (see module
    docstring).  Subclasses implement :meth:`reward_from_state` (the pure
    state -> reward map, which is also what FRE's encoder consumes) and may
    override :meth:`min_return` / :meth:`max_return`.
    """

    kind: str = "base"

    def __init__(
        self,
        name: str,
        domain: str,
        max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    ) -> None:
        self.name = name
        self.domain = _canonical_domain(domain)
        self.max_episode_steps = int(max_episode_steps)
        self._episode_steps = 0
        self._total_return = 0.0
        self._success_steps = 0
        self._success = False

    # -- reward interface ---------------------------------------------------
    def reward_from_state(self, state: np.ndarray) -> float:
        """Pure state -> reward map; also used to label encoder samples."""
        raise NotImplementedError

    def reward(
        self,
        obs: np.ndarray,
        next_obs: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Per-step reward.  ``info['physics']`` may carry the ExORL physics features."""
        return float(self.reward_from_state(obs))

    def success(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> bool:
        """Task-specific success indicator (used for diagnostics only)."""
        return False

    # -- bookkeeping / normalisation ---------------------------------------
    @property
    def min_return(self) -> float:
        """Return of the worst possible ``max_episode_steps``-long trajectory."""
        return -float(self.max_episode_steps)

    @property
    def max_return(self) -> float:
        """Return of the best possible ``max_episode_steps``-long trajectory."""
        return 0.0

    @property
    def reference_steps(self) -> int:
        return self.max_episode_steps

    def normalize_return(self, total_return: float) -> float:
        """Map a raw episode return to the paper's 0-100 scale."""
        span = self.max_return - self.min_return
        if span <= 0:
            return 0.0
        return float(np.clip(100.0 * (total_return - self.min_return) / span, 0.0, 100.0))

    def reset(self) -> None:
        self._episode_steps = 0
        self._total_return = 0.0
        self._success_steps = 0
        self._success = False

    def observe(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> float:
        """Accumulate one step of statistics; returns the per-step reward."""
        r = self.reward(obs, None, None, info)
        self._episode_steps += 1
        self._total_return += r
        if self.success(obs, info):
            self._success_steps += 1
            self._success = True
        return r

    # -- encoder tokens -----------------------------------------------------
    def encoder_pairs(
        self,
        dataset: Any,
        num_samples: int = EXORL_ENCODER_SAMPLES,
        rng: Optional[np.random.RandomState] = None,
        use_physics: bool = True,
        dataset_physics: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample ``(states, rewards)`` pairs labelling this task for the encoder.

        Mirrors the AntMaze implementation: ``num_samples`` states drawn from the
        offline dataset (FRE uses K=32 zero-shot), augmented with ExORL physics
        features (Appendix C.2), each labelled by the task's reward function.
        Goal-reaching tasks additionally guarantee that the goal state itself is
        part of the set (otherwise a -1-only context is uninformative).
        """
        return sample_task_encoder_pairs(
            self,
            dataset,
            num_samples=num_samples,
            rng=rng,
            use_physics=use_physics,
            dataset_physics=dataset_physics,
        )

    # -- misc ---------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "kind": self.kind,
            "max_episode_steps": self.max_episode_steps,
            "min_return": self.min_return,
            "max_return": self.max_return,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, domain={self.domain!r})"


class VelocityTask(ExoRLTask):
    """Forward/backward locomotion velocity task ("ExORL evaluation tasks").

    Reward: "The reward is 1 if the velocity is at least the threshold value and
    linearly decays to 0 for values below the threshold value. If the agent's
    horizontal velocity is in the opposite direction of the target velocity, the
    reward is 0."  The horizontal velocity is the first physics feature appended
    by Appendix C.2 (``horizontal_velocity`` for Walker, ``speed`` for Cheetah).
    """

    kind = "velocity"

    def __init__(
        self,
        domain: str,
        threshold: float,
        direction: int = +1,
        name: Optional[str] = None,
        max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
        physics_index: int = 0,
    ) -> None:
        domain = _canonical_domain(domain)
        label = name or f"{domain.replace('exorl_', '')}-velocity-{threshold:g}"
        super().__init__(label, domain, max_episode_steps=max_episode_steps)
        if threshold <= 0:
            raise ValueError("Velocity threshold must be positive.")
        self.threshold = float(threshold)
        self.direction = +1 if float(direction) >= 0 else -1
        self.physics_index = int(physics_index)

    # min_return = 0 (per-step reward in [0, 1]); max_return = horizon
    @property
    def min_return(self) -> float:
        return 0.0

    @property
    def max_return(self) -> float:
        return float(self.max_episode_steps)

    def reward_from_state(self, state: np.ndarray) -> float:
        velocity = self._horizontal_velocity(state)
        return self.reward_from_velocity(velocity)

    def reward_from_velocity(self, velocity: float) -> float:
        """Paper-exact velocity reward: linear decay above the threshold, 0 if reversed."""
        v = self.direction * float(velocity)  # signed velocity along the target direction
        if v <= 0.0:  # "opposite direction of the target velocity" -> reward 0
            return 0.0
        return float(min(v / self.threshold, 1.0))

    def reward(
        self,
        obs: np.ndarray,
        next_obs: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        physics = None if info is None else info.get("physics")
        if physics is not None:
            arr = np.asarray(physics, dtype=np.float64).ravel()
            if arr.size > self.physics_index:
                return self.reward_from_velocity(arr[self.physics_index])
        return self.reward_from_state(obs)

    def success(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> bool:
        physics = None if info is None else info.get("physics")
        if physics is not None:
            arr = np.asarray(physics, dtype=np.float64).ravel()
            if arr.size > self.physics_index:
                return self.direction * float(arr[self.physics_index]) >= self.threshold
        return self.direction * self._horizontal_velocity(obs) >= self.threshold

    def _horizontal_velocity(self, state: np.ndarray) -> float:
        arr = np.asarray(state, dtype=np.float64).ravel()
        # Augmented encoder states already carry the physics features in their tail.
        return float(arr[self.physics_index])

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({"threshold": self.threshold, "direction": self.direction})
        return d


class GoalReachingTask(ExoRLTask):
    """Goal-reaching task with the paper's -1/0 reward and std-normalised distance.

    "The agent is assigned a reward of -1 at each step unless it is within a
    threshold distance of 0.1 of the goal state, in which case it is assigned a
    reward of 0."  Distances are Euclidean on the (non-augmented) observation
    space with each dimension normalised by the offline dataset's per-dim std.
    """

    kind = "goal"

    def __init__(
        self,
        domain: str,
        goal_state: np.ndarray,
        state_std: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        index: int = 0,
        threshold: float = EXORL_GOAL_THRESHOLD,
        max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
        reward_on_success: float = 0.0,
        reward_off_success: float = -1.0,
    ) -> None:
        domain = _canonical_domain(domain)
        label = name or f"{domain.replace('exorl_', '')}-goal-{index}"
        super().__init__(label, domain, max_episode_steps=max_episode_steps)
        self.goal = np.asarray(goal_state, dtype=np.float64).ravel().copy()
        self.goal_index = int(index)
        self.threshold = float(threshold)
        self.reward_on_success = float(reward_on_success)
        self.reward_off_success = float(reward_off_success)
        self._state_std = None if state_std is None else np.asarray(state_std, dtype=np.float64).ravel()

    # goal tasks: best case is 0 (always at the goal), worst case is -horizon.
    @property
    def min_return(self) -> float:
        return float(self.reward_off_success) * float(self.max_episode_steps)

    @property
    def max_return(self) -> float:
        return float(self.reward_on_success) * float(self.max_episode_steps)

    def distance(self, state: np.ndarray, state_std: Optional[np.ndarray] = None) -> float:
        """Std-normalised Euclidean distance to the goal (physics excluded)."""
        std = state_std if state_std is not None else self._state_std
        return float(normalized_goal_distance(state, self.goal, std=std))

    def reward_from_state(self, state: np.ndarray) -> float:
        return self.reward_from_distance(self.distance(state))

    def reward_from_distance(self, distance: float) -> float:
        return self.reward_on_success if float(distance) < self.threshold else self.reward_off_success

    def reward(
        self,
        obs: np.ndarray,
        next_obs: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        return self.reward_from_state(obs)

    def success(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> bool:
        return self.distance(obs) < self.threshold

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "threshold": self.threshold,
            "goal_index": self.goal_index,
            "goal": self.goal.tolist(),
        })
        return d


# ---------------------------------------------------------------------------
# Task factories
# ---------------------------------------------------------------------------
def make_velocity_tasks(
    domain: str = "exorl_walker",
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
) -> List[VelocityTask]:
    """Velocity tasks for ``domain``.

    Walker: four forward tasks with thresholds ``0.1, 1, 4, 8``.
    Cheetah: ``run`` (10), ``walk`` (1), ``run-backwards`` (10), ``walk-backwards`` (1).
    """
    domain = _canonical_domain(domain)
    if domain == "exorl_walker":
        specs = EXORL_WALKER_VELOCITY_TASKS
    else:
        specs = EXORL_CHEETAH_VELOCITY_TASKS
    prefix = "walker" if domain == "exorl_walker" else "cheetah"
    tasks: List[VelocityTask] = []
    for name, threshold, direction in specs:
        label = name if domain != "exorl_walker" else f"{prefix}-{name}"
        if domain != "exorl_walker":
            label = f"{prefix}-{name.replace('cheetah-', '')}"
        tasks.append(
            VelocityTask(
                domain,
                threshold,
                direction=direction,
                name=label,
                max_episode_steps=max_episode_steps,
            )
        )
    return tasks


def make_goal_tasks(
    domain: str = "exorl_walker",
    dataset: Any = None,
    num_goals: int = EXORL_NUM_GOALS,
    seed: int = 0,
    goals: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    goal_indices: Optional[Sequence[int]] = None,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
) -> List[GoalReachingTask]:
    """Five fixed goal-reaching tasks ("5 random states ... kept fixed throughout")."""
    domain = _canonical_domain(domain)
    prefix = "walker" if domain == "exorl_walker" else "cheetah"
    if goals is None:
        if dataset is None:
            raise ValueError("Either `goals` or `dataset` must be provided for goal tasks.")
        goals, goal_indices = select_goal_states(
            dataset, num_goals=num_goals, seed=seed, indices=goal_indices
        )
    goals = np.atleast_2d(np.asarray(goals, dtype=np.float64))
    if state_std is None and dataset is not None:
        state_std = dataset_state_std(dataset)
    if state_std is not None:
        state_std = np.asarray(state_std, dtype=np.float64).ravel()[: goals.shape[1]]
    return [
        GoalReachingTask(
            domain,
            goal,
            state_std=state_std,
            name=f"{prefix}-goal-{i}",
            index=i,
            max_episode_steps=max_episode_steps,
        )
        for i, goal in enumerate(goals)
    ]


def get_task_suite(
    name: str,
    dataset: Any = None,
    seed: int = 0,
    goals: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    **kwargs: Any,
) -> List[ExoRLTask]:
    """Return the task list for one ExORL suite key.

    Keys: ``"exorl-walker-goals"``, ``"exorl-walker-velocity"``,
    ``"exorl-cheetah-goals"``, ``"exorl-cheetah-velocity"``.
    """
    key = str(name).lower().replace("_", "-")
    if key in ("walker", "exorl-walker"):
        return make_velocity_tasks("exorl_walker", max_episode_steps) + make_goal_tasks(
            "exorl_walker", dataset, seed=seed, goals=goals, state_std=state_std,
            max_episode_steps=max_episode_steps, **kwargs
        )
    if key in ("cheetah", "exorl-cheetah"):
        return make_velocity_tasks("exorl_cheetah", max_episode_steps) + make_goal_tasks(
            "exorl_cheetah", dataset, seed=seed, goals=goals, state_std=state_std,
            max_episode_steps=max_episode_steps, **kwargs
        )
    if key.endswith("-goals"):
        domain = "exorl_walker" if "walker" in key else "exorl_cheetah"
        return make_goal_tasks(domain, dataset, seed=seed, goals=goals, state_std=state_std,
                               max_episode_steps=max_episode_steps, **kwargs)
    if key.endswith("-velocity"):
        domain = "exorl_walker" if "walker" in key else "exorl_cheetah"
        return make_velocity_tasks(domain, max_episode_steps)
    raise KeyError(f"Unknown ExORL suite {name!r}")


def make_exorl_task_suite(
    dataset: Any = None,
    domains: Sequence[str] = EXORL_DOMAINS,
    seed: int = 0,
    num_goals: int = EXORL_NUM_GOALS,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    goals_by_domain: Optional[Dict[str, np.ndarray]] = None,
    state_std_by_domain: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, List[ExoRLTask]]:
    """Build the four ExORL rows of Table 1 as ``{row_name: [tasks]}``."""
    suite: Dict[str, List[ExoRLTask]] = {}
    goals_by_domain = dict(goals_by_domain or {})
    state_std_by_domain = dict(state_std_by_domain or {})
    for domain in domains:
        d = _canonical_domain(domain)
        prefix = "walker" if d == "exorl_walker" else "cheetah"
        goals = goals_by_domain.get(domain, goals_by_domain.get(d))
        std = state_std_by_domain.get(domain, state_std_by_domain.get(d))
        suite[f"exorl-{prefix}-goals"] = make_goal_tasks(
            d, dataset, num_goals=num_goals, seed=seed, goals=goals, state_std=std,
            max_episode_steps=max_episode_steps,
        )
        suite[f"exorl-{prefix}-velocity"] = make_velocity_tasks(d, max_episode_steps)
    return suite


# ---------------------------------------------------------------------------
# Encoder (state, reward) pairs
# ---------------------------------------------------------------------------
def sample_task_encoder_pairs(
    task: ExoRLTask,
    dataset: Any,
    num_samples: int = EXORL_ENCODER_SAMPLES,
    rng: Optional[np.random.RandomState] = None,
    use_physics: bool = True,
    dataset_physics: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label ``num_samples`` dataset states with ``task``'s reward function.

    Returns ``(states (K, obs_dim [+ physics]), rewards (K,))``.  For
    goal-reaching tasks the goal state occupies the first slot so that the
    encoder context always contains at least one goal-reaching (reward 0) token
    -- the same safeguard the prior sampler uses in ``fre/fre/prior.py``.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    states = as_state_array(dataset)
    if states.ndim == 1:
        states = states[None]
    n = states.shape[0]
    k = int(max(1, num_samples))

    if isinstance(task, GoalReachingTask):
        extra = 1
        idx = rng.randint(0, n, size=max(0, k - extra))
        sel = np.concatenate([[0], idx]) if k > 1 else np.array([0])
        sampled = states[sel]
        sampled[0] = task.goal[: sampled.shape[1]] if task.goal.size >= sampled.shape[1] else sampled[0]
        goal_row = task.goal
        if goal_row.size != sampled.shape[1]:
            goal_row = goal_row[: sampled.shape[1]]
        sampled[0] = goal_row
    else:
        sel = rng.randint(0, n, size=k)
        sampled = states[sel]

    physics = dataset_physics
    if physics is None and use_physics:
        physics = as_physics_array(dataset, task.domain)
    if use_physics:
        if physics is not None:
            physics = np.asarray(physics, dtype=np.float64)[sel] if physics.shape[0] == n else physics
            enc_states = augment_observations(task.domain, sampled, physics=physics)
        else:
            enc_states = augment_observations(task.domain, sampled)
    else:
        enc_states = np.asarray(sampled, dtype=np.float64)

    # Rewards are computed from the *raw* (non-augmented) state for goal tasks
    # ("Augmented information is not utilized when calculating goal distance")
    # and from the physics features for velocity tasks.
    rewards = np.zeros(enc_states.shape[0], dtype=np.float64)
    for i, raw in enumerate(sampled):
        if isinstance(task, VelocityTask):
            phys_i = physics_features(task.domain, raw) if physics is None else np.asarray(physics)[i]
            rewards[i] = task.reward_from_velocity(np.asarray(phys_i).ravel()[task.physics_index])
        else:
            rewards[i] = task.reward_from_state(raw)
    return enc_states.astype(np.float64), rewards


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------
class _SyntheticPhysics:
    """Minimal stand-in for dm_control's ``env.physics`` (fallback env only)."""

    def __init__(self, domain: str) -> None:
        self.domain = _canonical_domain(domain)
        self._velocity = 0.0
        self._upright = 1.0
        self._height = WALKER_NOMINAL_HEIGHT

    def horizontal_velocity(self) -> np.ndarray:
        return np.array([self._velocity, 0.0], dtype=np.float64)

    def torso_upright(self) -> float:
        return float(self._upright)

    def torso_height(self) -> float:
        return float(self._height)

    def speed(self) -> float:
        return float(self._velocity)


class SyntheticExoRLEnv:
    """Deterministic stand-in for the ExORL Walker/Cheetah simulators.

    Used only when neither the ExORL datasets nor ``dm_control`` are installed,
    so that :func:`evaluate_exorl_suite` (and its scripts) remain runnable and
    testable end-to-end.  It exposes the same ``physics`` interface as
    dm_control (``horizontal_velocity``/``torso_upright``/``torso_height`` for
    Walker, ``speed`` for Cheetah), so the reward functions and physics
    augmentation code paths are exercised identically.
    """

    def __init__(self, domain: str = "exorl_walker", seed: Optional[int] = None) -> None:
        self.domain = _canonical_domain(domain)
        self.obs_dim = obs_dim_for(self.domain)
        self.action_dim = action_dim_for(self.domain)
        self.physics = _SyntheticPhysics(self.domain)
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        self._vel = 0.0
        self._height = WALKER_NOMINAL_HEIGHT
        self._upright = 1.0
        self._x = 0.0
        self._t = 0
        self.action_space = _BoxSpace(low=-1.0, high=1.0, shape=(self.action_dim,))
        self.observation_space = _BoxSpace(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
        self.synthetic = True

    # -- gym-like API -------------------------------------------------------
    def reset(self, **kwargs: Any) -> np.ndarray:
        self._vel = 0.0
        self._height = WALKER_NOMINAL_HEIGHT
        self._upright = 1.0
        self._x = 0.0
        self._t = 0
        return self._obs()

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        drive = float(a[0]) if a.size else 0.0
        if self.domain == "exorl_walker":
            self._vel += 0.1 * drive - 0.02 * self._vel
            self._height += 0.01 * (np.cos(self._t * 0.05) - 0.1) - 0.02 * (self._height - WALKER_NOMINAL_HEIGHT)
            self._upright = float(np.clip(self._height / WALKER_NOMINAL_HEIGHT, -1.0, 1.0))
        else:
            self._vel += 0.1 * drive - 0.01 * self._vel
        self._x += self._vel / 20.0
        self._t += 1
        self.physics._velocity = self._vel
        self.physics._upright = self._upright
        self.physics._height = self._height
        obs = self._obs()
        info = {"physics": physics_features(self.domain, obs, env=self), "x_position": self._x}
        return obs, 0.0, False, info

    def close(self) -> None:  # pragma: no cover - parity with gym API
        return None

    def seed(self, seed: Optional[int] = None) -> None:  # pragma: no cover
        self._rng = np.random.RandomState(seed if seed is not None else 0)

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float64)
        if self.obs_dim >= 10:
            obs[-10] = self._height
        obs[-8] = self._vel
        if self.obs_dim >= 1:
            obs[0] = self._x
        return obs


class _BoxSpace:
    """Tiny ``gym.spaces.Box`` stand-in (avoids a hard gym dependency)."""

    def __init__(self, low: float, high: float, shape: Tuple[int, ...]) -> None:
        self.low = np.full(shape, low, dtype=np.float64)
        self.high = np.full(shape, high, dtype=np.float64)
        self.shape = shape

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float64)

    def contains(self, x: np.ndarray) -> bool:  # pragma: no cover
        x = np.asarray(x)
        return bool(np.all(x >= self.low) and np.all(x <= self.high))


def make_exorl_env(
    domain: str = "exorl_walker",
    seed: Optional[int] = None,
    env_id: Optional[str] = None,
    allow_synthetic: bool = True,
    **kwargs: Any,
) -> Any:
    """Create the online evaluation environment for an ExORL domain.

    Resolution order: ``dm_control`` suite (via ``gym``-style wrapper) ->
    :class:`SyntheticExoRLEnv` (only when ``allow_synthetic=True``).  The real
    ExORL data-collection simulators are DeepMind Control Suite tasks, so the
    dm_control route matches the paper's evaluation environments exactly.
    """
    domain = _canonical_domain(domain)
    task_name = "walker" if domain == "exorl_walker" else "cheetah"

    try:  # pragma: no cover - depends on the optional dm_control install
        from dm_control import suite  # type: ignore

        dm_env = suite.load(task_name, "walk", task_kwargs={"random": seed} if seed is not None else None)
        return _DMControlWrapper(dm_env)
    except Exception:
        pass

    if not allow_synthetic:
        raise RuntimeError(
            "dm_control/ExORL simulators unavailable and allow_synthetic=False; "
            "install `dm_control` (or the `exorl` package) to run real ExORL evaluation."
        )
    return SyntheticExoRLEnv(domain, seed=seed, **kwargs)


class _DMControlWrapper:
    """``gym``-style wrapper around a ``dm_control`` environment (thin shim)."""

    def __init__(self, dm_env: Any) -> None:  # pragma: no cover - requires dm_control
        self._env = dm_env
        self.physics = getattr(dm_env, "physics", None)
        self.action_space = _dmc_to_box(dm_env.action_spec())
        self.observation_space = _dmc_to_box(dm_env.observation_spec())
        self.domain = "exorl_walker" if "walker" in type(dm_env).__module__ else "exorl_cheetah"

    def reset(self, **kwargs: Any):  # pragma: no cover
        ts = self._env.reset()
        return np.asarray(ts.observation, dtype=np.float64)

    def step(self, action):  # pragma: no cover
        ts = self._env.step(np.asarray(action, dtype=np.float64))
        obs = np.asarray(ts.observation, dtype=np.float64)
        term = bool(getattr(ts, "last", lambda: False)())
        reward = float(getattr(ts, "reward", 0.0) or 0.0)
        return obs, reward, term, {"physics": None}

    def close(self):  # pragma: no cover
        return None

    def seed(self, seed=None):  # pragma: no cover
        return None


def _dmc_to_box(spec) -> _BoxSpace:  # pragma: no cover - requires dm_control
    """Convert a dm_control spec into the local :class:`_BoxSpace`."""
    if hasattr(spec, "shape"):  # Array spec
        return _BoxSpace(float(getattr(spec, "minimum", -np.inf)), float(getattr(spec, "maximum", np.inf)), tuple(spec.shape))
    names, lows, highs, total = [], [], [], 0
    for key, sub in spec.items():
        shape = tuple(getattr(sub, "shape", ()))
        names.append(key)
        lows.append(np.full(shape, float(getattr(sub, "minimum", -np.inf))))
        highs.append(np.full(shape, float(getattr(sub, "maximum", np.inf))))
        total += int(np.prod(shape)) if shape else 1
    return _BoxSpace(float(np.min(lows)) if lows else -np.inf,
                     float(np.max(highs)) if highs else np.inf, (total,))


class ExoRLEvalWrapper:
    """Gym wrapper turning an ExORL simulator into a task-reward rollout env.

    * applies the task reward (:class:`VelocityTask` / :class:`GoalReachingTask`);
    * extracts the Appendix C.2 physics features (exact via ``env.physics`` when
      available) and passes them through ``info['physics']``;
    * exposes ``last_encoder_obs`` -- the *augmented* observation (physics
      concatenated) that the frozen FRE encoder consumes, while ``step``
      returns the plain observation that the policy/value networks are trained
      on ("are instead trained on the underlying observation space").
    """

    def __init__(
        self,
        env: Any,
        task: ExoRLTask,
        max_episode_steps: Optional[int] = None,
        terminate_on_success: bool = False,
        use_physics: bool = True,
    ) -> None:
        self.env = env
        self.task = task
        self.domain = task.domain
        self.max_episode_steps = int(max_episode_steps or task.max_episode_steps)
        self.terminate_on_success = bool(terminate_on_success)
        self.use_physics = bool(use_physics)
        self._steps = 0
        self.last_encoder_obs: Optional[np.ndarray] = None

    # -- gym-like API -------------------------------------------------------
    def reset(self, **kwargs: Any):
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        obs = np.asarray(obs, dtype=np.float64)
        self._steps = 0
        self.task.reset()
        self.last_encoder_obs = self._augment(obs)
        return obs

    def step(self, action: np.ndarray):
        out = self.env.step(np.asarray(action, dtype=np.float64))
        if len(out) == 5:  # gymnasium API
            obs, _r, term, trunc, info = out
            done = bool(term or trunc)
        else:
            obs, _r, done, info = out
        obs = np.asarray(obs, dtype=np.float64)
        info = dict(info or {})
        physics = None
        if self.use_physics:
            physics = info.get("physics", None)
            if physics is None or np.asarray(physics).size == 0:
                physics = physics_features(self.domain, obs, env=self.env)
        info["physics"] = physics
        info["task_name"] = self.task.name
        info["domain"] = self.domain

        reward = self.task.reward(obs, obs, action, info)
        self._steps += 1
        self.last_encoder_obs = self._augment(obs, physics) if self.use_physics else obs

        timeout = self._steps >= self.max_episode_steps
        if self.terminate_on_success and self.task.success(obs, info):
            done = True
        done = bool(done or timeout)
        info["timeout"] = bool(timeout)
        return obs, float(reward), done, info

    def close(self) -> None:  # pragma: no cover
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def seed(self, seed: Optional[int] = None) -> None:  # pragma: no cover
        seed_fn = getattr(self.env, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)

    # -- helpers ------------------------------------------------------------
    def _augment(self, obs: np.ndarray, physics: Optional[np.ndarray] = None) -> np.ndarray:
        """Physics-augmented observation = FRE encoder state (Appendix C.2)."""
        if physics is None:
            physics = physics_features(self.domain, obs, env=self.env)
        return augment_observations(self.domain, obs, physics=physics)

    @property
    def action_space(self):
        return getattr(self.env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self.env, "observation_space", None)


# ---------------------------------------------------------------------------
# Rollouts / evaluation
# ---------------------------------------------------------------------------
@dataclass
class ExoRLEpisodeResult:
    """Statistics of a single ExORL evaluation episode (0-100 normalised score)."""

    task_name: str
    total_return: float
    normalized_return: float
    length: int
    success: bool
    success_steps: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "total_return": self.total_return,
            "normalized_return": self.normalized_return,
            "length": self.length,
            "success": self.success,
            "success_steps": self.success_steps,
        }


def _to_numpy_action(action: Any) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float64).ravel()


def rollout_episode(
    env: ExoRLEvalWrapper,
    act_fn: Callable[[np.ndarray], Any],
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> ExoRLEpisodeResult:
    """Roll out ``act_fn`` in a wrapped ExORL eval env for one episode.

    ``act_fn`` receives the *plain* observation (policy networks are trained on
    the underlying observation space, Appendix C.2) and returns an action.
    """
    if seed is not None:
        seed_fn = getattr(env, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)
        else:  # pragma: no cover
            np.random.seed(seed)
    obs = env.reset()
    steps = int(max_episode_steps or env.max_episode_steps)
    total = 0.0
    success = False
    success_steps = 0
    length = 0
    for _ in range(steps):
        action = _to_numpy_action(act_fn(obs))
        obs, reward, done, info = env.step(action)
        total += float(reward)
        length += 1
        if env.task.success(obs, info):
            success = True
            success_steps += 1
        if done:
            break
    return ExoRLEpisodeResult(
        task_name=env.task.name,
        total_return=float(total),
        normalized_return=float(env.task.normalize_return(total)),
        length=length,
        success=bool(success),
        success_steps=int(success_steps),
    )


def evaluate_task(
    task: ExoRLTask,
    act_fn: Callable[[np.ndarray], Any],
    env: Any = None,
    domain: Optional[str] = None,
    num_episodes: int = 20,
    max_episode_steps: Optional[int] = None,
    seed: int = 0,
    terminate_on_success: bool = False,
    use_physics: bool = True,
) -> Dict[str, float]:
    """Evaluate one ExORL task: ``num_episodes`` rollouts, returns in [0, 100].

    Follows the paper's protocol (5 seeds x 20 episodes, mean +/- std); the
    per-seed aggregation is handled by :func:`evaluate_suite` /
    :func:`evaluate_exorl_suite`.
    """
    domain = task.domain if domain is None else _canonical_domain(domain)
    close_env = False
    if env is None:
        env = make_exorl_env(domain, seed=seed)
        close_env = True
    wrapped = env if isinstance(env, ExoRLEvalWrapper) else ExoRLEvalWrapper(
        env, task, max_episode_steps=max_episode_steps, terminate_on_success=terminate_on_success,
        use_physics=use_physics,
    )
    scores: List[float] = []
    returns: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []
    try:
        for ep in range(int(num_episodes)):
            res = rollout_episode(wrapped, act_fn, max_episode_steps=max_episode_steps, seed=seed + ep)
            scores.append(res.normalized_return)
            returns.append(res.total_return)
            successes.append(1.0 if res.success else 0.0)
            lengths.append(res.length)
    finally:
        if close_env:
            wrapped.close()
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "task": task.name,
        "domain": domain,
        "score": float(arr.mean()) if arr.size else 0.0,
        "score_std": float(arr.std()) if arr.size else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "total_return": float(np.mean(returns)) if returns else 0.0,
        "episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "num_episodes": int(num_episodes),
        "seed": int(seed),
    }


def evaluate_suite(
    suite: Sequence[ExoRLTask],
    act_fn_factory: Callable[[ExoRLTask], Callable[[np.ndarray], Any]],
    envs: Optional[Dict[str, Any]] = None,
    domain: str = "exorl_walker",
    num_episodes: int = 20,
    max_episode_steps: Optional[int] = None,
    seed: int = 0,
    terminate_on_success: bool = False,
    use_physics: bool = True,
) -> Dict[str, Any]:
    """Evaluate a list of tasks and aggregate them into one Table 1 row.

    ``act_fn_factory(task)`` returns the rollout callable for that task (usually
    a z-conditioned policy with ``z`` encoded from the task's 32 (state, reward)
    samples -- see :func:`encode_task_latent`).
    """
    domain = _canonical_domain(domain)
    env = None if envs is None else envs.get(domain)
    results: Dict[str, Dict[str, float]] = {}
    scores: List[float] = []
    for task in suite:
        res = evaluate_task(
            task,
            act_fn_factory(task),
            env=env,
            domain=task.domain,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
            terminate_on_success=terminate_on_success,
            use_physics=use_physics,
        )
        results[task.name] = res
        scores.append(res["score"])
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "tasks": results,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "num_tasks": int(arr.size),
        "domain": domain,
    }


def evaluate_exorl_suite(
    act_fn_factory: Callable[[str, ExoRLTask], Callable[[np.ndarray], Any]],
    dataset: Any = None,
    domains: Sequence[str] = EXORL_DOMAINS,
    num_episodes: int = 20,
    max_episode_steps: int = EXORL_MAX_EPISODE_STEPS,
    seed: int = 0,
    goals_by_domain: Optional[Dict[str, np.ndarray]] = None,
    state_std_by_domain: Optional[Dict[str, np.ndarray]] = None,
    envs: Optional[Dict[str, Any]] = None,
    terminate_on_success: bool = False,
    use_physics: bool = True,
    dataset_goals: bool = True,
) -> Dict[str, Any]:
    """Evaluate the full ExORL benchmark (the four Table 1 rows + ``exorl-all``).

    ``act_fn_factory(domain, task)`` must return the rollout callable for a task.
    When ``dataset_goals`` is False the five goal states default to zero vectors
    (useful for smoke tests without an offline dataset).
    """
    goals_by_domain = dict(goals_by_domain or {})
    out: Dict[str, Any] = {}
    row_means: List[float] = []

    for domain in domains:
        d = _canonical_domain(domain)
        prefix = "walker" if d == "exorl_walker" else "cheetah"
        if d not in goals_by_domain:
            if dataset is None:
                goals_by_domain[d] = np.zeros((EXORL_NUM_GOALS, obs_dim_for(d)), dtype=np.float64)
                state_std = state_std_by_domain.get(d) if state_std_by_domain else np.ones(obs_dim_for(d))
            else:
                goals, _ = select_goal_states(dataset, num_goals=EXORL_NUM_GOALS, seed=seed)
                goals_by_domain[d] = goals if dataset_goals else np.zeros_like(goals)
                state_std = state_std_by_domain.get(d) if state_std_by_domain else dataset_state_std(dataset)[: goals.shape[1]]
        else:
            goals = np.atleast_2d(goals_by_domain[d])
            state_std = state_std_by_domain.get(d) if state_std_by_domain else None
        if state_std_by_domain and d in state_std_by_domain:
            state_std = state_std_by_domain[d]
        if state_std is None and dataset is not None:
            state_std = dataset_state_std(dataset)
        if state_std is None:
            state_std = np.ones(obs_dim_for(d), dtype=np.float64)

        for kind, tasks in (
            ("goals", make_goal_tasks(d, dataset, goals=goals_by_domain[d], state_std=state_std,
                                      max_episode_steps=max_episode_steps)),
            ("velocity", make_velocity_tasks(d, max_episode_steps)),
        ):
            row = f"exorl-{prefix}-{kind}"
            res = evaluate_suite(
                tasks,
                lambda task: act_fn_factory(d, task),
                envs=envs,
                domain=d,
                num_episodes=num_episodes,
                max_episode_steps=max_episode_steps,
                seed=seed,
                terminate_on_success=terminate_on_success,
                use_physics=use_physics,
            )
            out[row] = res
            row_means.append(res["mean"])

    all_mean = float(np.mean(row_means)) if row_means else 0.0
    all_std = float(np.std(row_means)) if row_means else 0.0
    out["exorl-all"] = {"mean": all_mean, "std": all_std, "num_tasks": len(row_means)}
    out["reference"] = EXORL_TABLE1_REFERENCE
    return out


# ---------------------------------------------------------------------------
# Policy / latent helpers
# ---------------------------------------------------------------------------
def make_exorl_policy_fn(
    agent: Any,
    z: Any,
    deterministic: bool = True,
    clip: Optional[Sequence[float]] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a z-conditioned IQL agent into an ``act_fn(obs) -> action`` callable.

    The policy consumes the plain observation (Appendix C.2: policy/value
    networks "are trained on the underlying observation space"), with ``z`` the
    latent encoded from the task's 32 reward-annotated states.
    """
    import torch  # local import: keep module import-light

    z_arr = z
    if not hasattr(z_arr, "detach"):
        z_arr = torch.as_tensor(np.asarray(z_arr, dtype=np.float32))

    def act(obs: np.ndarray) -> np.ndarray:
        return agent.select_action(obs, z_arr, deterministic=deterministic, clip=clip)

    return act


def encode_task_latent(
    encoder: Any,
    task: ExoRLTask,
    dataset: Any,
    num_samples: int = EXORL_ENCODER_SAMPLES,
    seed: int = 0,
    use_physics: bool = True,
    dataset_physics: Optional[np.ndarray] = None,
    deterministic: bool = True,
    device: str = "cpu",
) -> Any:
    """Encode a task's 32 (state, reward) pairs into the 128-dim latent ``z``.

    Uses the frozen FRE encoder; by default returns the posterior mean
    (``deterministic=True``) which is what zero-shot evaluation uses.
    """
    import torch

    states, rewards = sample_task_encoder_pairs(
        task, dataset, num_samples=num_samples, rng=np.random.RandomState(seed),
        use_physics=use_physics, dataset_physics=dataset_physics,
    )
    s = torch.as_tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
    r = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            dist = encoder(s, r)
            return dist.mean.squeeze(0)
        return encoder.encode(s, r, sample=True).squeeze(0)


# ---------------------------------------------------------------------------
# Smoke test (runnable offline, no dm_control / ExORL install required)
# ---------------------------------------------------------------------------
def _smoke_test(seed: int = 0, num_episodes: int = 2) -> Dict[str, Any]:  # pragma: no cover
    """End-to-end smoke test with the synthetic env and a random policy."""
    rng = np.random.RandomState(seed)
    dataset_states = rng.normal(size=(256, WALKER_OBS_DIM))
    dataset = {
        "states": dataset_states,
        "physics": np.stack([physics_features("exorl_walker", s) for s in dataset_states]),
    }
    stats = dataset_state_std(dataset)

    vel_tasks = make_velocity_tasks("exorl_walker")
    goal_tasks = make_goal_tasks("exorl_walker", dataset, state_std=stats)
    print(f"[exorl] velocity tasks: {[t.name for t in vel_tasks]}")
    print(f"[exorl] goal tasks:     {[t.name for t in goal_tasks]}")
    print(f"[exorl] cheetah tasks:  {[t.name for t in make_velocity_tasks('exorl_cheetah')]}")

    for task in (vel_tasks[0], goal_tasks[0]):
        s, r = sample_task_encoder_pairs(task, dataset)
        assert s.shape[0] == EXORL_ENCODER_SAMPLES, s.shape
        print(f"[exorl] {task.name}: encoder states {s.shape}, rewards range "
              f"[{r.min():.2f}, {r.max():.2f}]")

    env = SyntheticExoRLEnv("exorl_walker", seed=seed)
    for task in (vel_tasks[0], goal_tasks[0]):
        res = evaluate_task(task, lambda o: rng.uniform(-1, 1, size=6), env=env,
                            num_episodes=num_episodes, max_episode_steps=50, seed=seed)
        print(f"[exorl] random-policy {task.name}: score={res['score']:.2f} (should be ~0)")

    # Validation of the paper's reward conventions.
    vt = VelocityTask("exorl_cheetah", threshold=10.0, name="cheetah-run")
    assert vt.reward_from_velocity(-1.0) == 0.0            # opposite direction -> 0
    assert abs(vt.reward_from_velocity(5.0) - 0.5) < 1e-9  # linear decay
    assert vt.reward_from_velocity(20.0) == 1.0            # saturates at 1
    gt = GoalReachingTask("exorl_walker", np.zeros(WALKER_OBS_DIM), state_std=np.ones(WALKER_OBS_DIM))
    assert gt.reward_from_distance(0.05) == 0.0
    assert gt.reward_from_distance(0.2) == -1.0
    assert abs(gt.normalize_return(-500.0) - 50.0) < 1e-9
    print("[exorl] reward-convention assertions passed.")
    return {"ok": True}


if __name__ == "__main__":  # pragma: no cover
    _smoke_test()
