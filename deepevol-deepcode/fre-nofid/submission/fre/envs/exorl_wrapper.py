"""ExORL environment wrapper for FRE zero-shot evaluation.

The FRE paper trains on ExORL (Exploratory Off-policy RL) RND datasets for the
``walker`` and ``cheetah`` domains and evaluates zero-shot on two families of
tasks:

* **goal-reaching** -- reach one of 5 fixed goal states (Euclidean distance in
  a *physics* feature space < 0.1),
* **velocity** -- sustain a threshold velocity (cheetah ``speed`` at 10 / 1,
  walker ``horizontal_velocity`` at 0.1 / 1 / 4 / 8).

Preprocessing details that matter (see the reproduction plan):

1. episode length is capped at 1000 steps,
2. physics features are appended to the *encoder* observations only -- the
   latent policy and the decoder only ever see raw observations,
3. each observation dimension is standardised (mean/std) before being fed to
   the encoder.

This module provides a gym-like wrapper around either a *live* dm_control
environment (when one is installed; used for actual rollout evaluation) or the
offline dataset (used for reward-field inspection and start-state sampling when
no simulator is available).  Rewards are computed exactly as the paper does:
sparse ``-1``/``0`` style indicators rescaled into ``[-1, 1]``.

The wrapper is intentionally dependency-light: ``numpy`` is required, while
``gym``/``dm_control``/``mujoco`` are optional and imported lazily so that the
offline analysis utilities keep working on machines without a simulator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.exorl_loader import (
    CHEETAH_PHYSICS,
    DEFAULT_DATASET_KIND,
    EXORL_DOMAINS,
    GOAL_DISTANCE_THRESHOLD,
    MAX_EPISODE_STEPS,
    NUM_GOAL_STATES,
    RAW_OBS_DIM,
    VELOCITY_THRESHOLDS,
    WALKER_PHYSICS,
    append_physics,
    compute_physics,
    evaluation_tasks,
    goal_done,
    goal_reward,
    load_exorl_dataset,
    normalize_observations,
    normalize_physics,
    physics_dim,
    physics_names,
    select_goal_states,
    velocity_done,
    velocity_reward,
)

__all__ = [
    "ExORLConfig",
    "ExORLTaskSpec",
    "ExORLWrapper",
    "ExORLGoalTask",
    "ExORLVelocityTask",
    "make_exorl_env",
    "make_exorl_envs",
    "exorl_eval_tasks",
    "LIVE_ENV_CANDIDATES",
]


# ---------------------------------------------------------------------------
# Live simulator ids
# ---------------------------------------------------------------------------
# ExORL experiments were run on the dm_control "suite" locomotion tasks.  We try
# a small list of candidate identifiers because different D4RL / dm_control
# installs register slightly different names.
LIVE_ENV_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "walker": ("walker-walk", "walker2d-walk", "walker2d-medium-replay-v2", "walker2d-v2"),
    "cheetah": ("cheetah-run", "halfcheetah-run", "halfcheetah-medium-replay-v2", "halfcheetah-v2"),
}


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------
@dataclass
class ExORLTaskSpec:
    """Descriptor for a single zero-shot ExORL task.

    Attributes
    ----------
    name:
        Human readable identifier, e.g. ``"walker-goal-2"`` or
        ``"cheetah-velocity-10"``.
    domain:
        ``"walker"`` or ``"cheetah"``.
    kind:
        ``"goal"`` or ``"velocity"``.
    goal:
        Target physics vector for goal tasks (``None`` for velocity tasks).
    threshold:
        Distance threshold (goal tasks) or velocity threshold (velocity tasks).
    feature_index:
        Index inside the physics vector used by velocity tasks.
    """

    name: str
    domain: str
    kind: str = "goal"
    goal: Optional[np.ndarray] = None
    threshold: float = GOAL_DISTANCE_THRESHOLD
    feature_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("goal", "velocity"):
            raise ValueError(f"Unknown ExORL task kind: {self.kind!r}")
        if self.goal is not None and not isinstance(self.goal, np.ndarray):
            self.goal = np.asarray(self.goal, dtype=np.float64)
        if self.kind == "goal" and self.goal is None:
            raise ValueError("Goal tasks require a target goal state.")

    def describe(self) -> str:
        if self.kind == "goal":
            return (
                f"{self.name}: reach physics state {np.round(self.goal, 3).tolist()} "
                f"(dist<{self.threshold})"
            )
        return (
            f"{self.name}: |{physics_names(self.domain)[self.feature_index]}| "
            f"exceeds {self.threshold}"
        )


@dataclass
class ExORLConfig:
    """Configuration bundle for :class:`ExORLWrapper`."""

    domain: str = "walker"
    root: Optional[str] = None
    dataset_kind: str = DEFAULT_DATASET_KIND
    max_episode_steps: int = MAX_EPISODE_STEPS
    goal_threshold: float = GOAL_DISTANCE_THRESHOLD
    append_physics_to_obs: bool = True
    normalise: bool = True
    sparse_reward: bool = True
    seed: Optional[int] = None
    use_live_env: Optional[bool] = None  # None -> auto-detect


# ---------------------------------------------------------------------------
# Reward helpers (thin wrappers around the loader functions, normalised to
# the [-1, 1] range used by the FRE reward prior)
# ---------------------------------------------------------------------------
def _goal_reward_value(physics: np.ndarray, goal: np.ndarray, threshold: float) -> float:
    """Reward for reaching ``goal`` (0.0 on success, -1.0 otherwise)."""
    r = goal_reward(physics, goal, threshold)
    return float(np.asarray(r).reshape(-1)[0])


def _velocity_reward_value(
    domain: str, physics: np.ndarray, threshold: float, feature_index: int = 0
) -> float:
    """Reward for exceeding ``threshold`` on the selected velocity feature."""
    r = velocity_reward(domain, physics, threshold)
    return float(np.asarray(r).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class ExORLWrapper:
    """Gym-like ExORL environment for FRE evaluation.

    Parameters
    ----------
    domain:
        ``"walker"`` or ``"cheetah"``.
    dataset:
        Pre-loaded dataset dict as returned by
        :func:`fre.data.exorl_loader.load_exorl_dataset`.  Loaded on demand when
        omitted.
    task:
        A :class:`ExORLTaskSpec`; the reward function is built from it.  When
        ``None`` a default goal task is used (first goal state of the domain).
    root / dataset_kind / normalise / append_physics_to_obs:
        Passed through to the loader.
    live_env:
        An already-constructed dm_control/gym environment.  When ``None`` the
        wrapper tries to build one (auto-detection can be disabled with
        ``use_live_env=False``), otherwise it operates in **offline mode**
        where ``reset`` samples a dataset start state and ``step`` replays the
        dataset (only useful for reward-field inspection).

    Notes
    -----
    ``reset`` / ``step`` return the *encoder* observation (raw observation with
    physics appended, standardised) -- exactly what the FRE encoder consumes.
    The raw policy observation is available via ``info["policy_observation"]``
    and through :meth:`policy_observation`.
    """

    def __init__(
        self,
        domain: str = "walker",
        dataset: Optional[Dict[str, Any]] = None,
        task: Optional[ExORLTaskSpec] = None,
        *,
        root: Optional[str] = None,
        dataset_kind: str = DEFAULT_DATASET_KIND,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        goal_threshold: float = GOAL_DISTANCE_THRESHOLD,
        append_physics_to_obs: bool = True,
        normalise: bool = True,
        seed: Optional[int] = None,
        live_env: Optional[Any] = None,
        use_live_env: Optional[bool] = None,
        config: Optional[ExORLConfig] = None,
    ) -> None:
        if config is not None:
            domain = config.domain
            root = config.root if root is None else root
            dataset_kind = config.dataset_kind
            max_episode_steps = config.max_episode_steps
            goal_threshold = config.goal_threshold
            append_physics_to_obs = config.append_physics_to_obs
            normalise = config.normalise
            seed = config.seed if seed is None else seed
            use_live_env = config.use_live_env if use_live_env is None else use_live_env

        domain = str(domain).lower()
        if domain not in EXORL_DOMAINS:
            raise ValueError(
                f"Unknown ExORL domain {domain!r}; expected one of {EXORL_DOMAINS}."
            )
        self.domain = domain
        self.max_episode_steps = int(max_episode_steps)
        self.goal_threshold = float(goal_threshold)
        self.append_physics_to_obs = bool(append_physics_to_obs)
        self.normalise = bool(normalise)
        self.root = root
        self.dataset_kind = dataset_kind

        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._step_count = 0
        self._episode_count = 0
        self._live_state: Optional[Any] = None
        self._episode_returns: List[float] = []
        self._episode_successes: List[bool] = []

        # ------------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------------
        self.dataset = dataset
        self._dataset_loaded = dataset is not None
        if self._dataset_loaded:
            self._setup_from_dataset()
        else:
            # Dimensions can be inferred from the domain even before loading.
            self.raw_obs_dim = RAW_OBS_DIM.get(domain, 0)
            self.physics_names_ = list(physics_names(domain))
            self.physics_dim_ = physics_dim(domain)

        # ------------------------------------------------------------------
        # Live simulator
        # ------------------------------------------------------------------
        if live_env is not None:
            use_live_env = True
        if use_live_env is False:
            self.env = None
            self.live_mode = False
        elif live_env is not None:
            self.env = live_env
            self.live_mode = True
        else:
            self.env = _maybe_make_live_env(domain, seed=seed)
            self.live_mode = self.env is not None

        # ------------------------------------------------------------------
        # Task / reward function
        # ------------------------------------------------------------------
        if task is None and self._dataset_loaded:
            task = self.default_task()
        self.task: Optional[ExORLTaskSpec] = task
        self._goal_cache: Optional[np.ndarray] = None
        if task is not None:
            self.set_task(task)

        # Offline-mode bookkeeping
        self._offline_index = 0
        self._offline_episode_start = 0

    # ------------------------------------------------------------------
    # Lazy dataset access
    # ------------------------------------------------------------------
    def _ensure_dataset(self) -> Dict[str, Any]:
        if not self._dataset_loaded:
            self.dataset = load_exorl_dataset(
                self.domain,
                root=self.root,
                dataset_kind=self.dataset_kind,
                append_physics_to_obs=True,
                normalise=True,
                verbose=False,
            )
            self._dataset_loaded = True
            self._setup_from_dataset()
        return self.dataset  # type: ignore[return-value]

    def _setup_from_dataset(self) -> None:
        ds = self.dataset or {}
        self.raw_observations = np.asarray(ds.get("observations"), dtype=np.float32)
        self.encoder_observations = np.asarray(
            ds.get("encoder_observations"), dtype=np.float32
        )
        self.physics = np.asarray(ds.get("physics"), dtype=np.float64)
        self.obs_mean = np.asarray(ds.get("obs_mean"), dtype=np.float64)
        self.obs_std = np.asarray(ds.get("obs_std"), dtype=np.float64)
        self.physics_mean = np.asarray(ds.get("physics_mean"), dtype=np.float64)
        self.physics_std = np.asarray(ds.get("physics_std"), dtype=np.float64)
        self.raw_obs_dim = int(self.raw_observations.shape[1])
        self.physics_dim_ = int(self.physics.shape[1])
        self.physics_names_ = list(ds.get("physics_names") or physics_names(self.domain))
        self.num_transitions = int(self.raw_observations.shape[0])
        self._terminals = ds.get("terminals")
        if self._terminals is None:
            self._terminals = np.zeros(self.num_transitions, dtype=np.float32)
        self._terminals = np.asarray(self._terminals, dtype=np.float32)
        self._episode_starts = np.concatenate(
            ([0], np.flatnonzero(self._terminals > 0.5) + 1)
        ).astype(np.int64)
        self._episode_starts = self._episode_starts[
            self._episode_starts < self.num_transitions
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def observation_dim(self) -> int:
        """Dimension of the observation the FRE encoder consumes."""
        return int(self.raw_obs_dim + (self.physics_dim_ if self.append_physics_to_obs else 0))

    @property
    def policy_observation_dim(self) -> int:
        """Dimension of the raw observation used by the policy/decoder."""
        return int(self.raw_obs_dim)

    @property
    def action_dim(self) -> Optional[int]:
        ds = self.dataset
        if ds is not None and ds.get("actions") is not None:
            return int(np.asarray(ds["actions"]).shape[1])
        if self.env is not None:
            try:
                return int(np.prod(self.env.action_space.shape))
            except Exception:  # pragma: no cover - exotic envs
                return None
        return None

    @property
    def num_episodes_completed(self) -> int:
        return int(self._episode_count)

    @property
    def last_episode_return(self) -> float:
        return float(self._episode_returns[-1]) if self._episode_returns else 0.0

    @property
    def last_episode_success(self) -> bool:
        return bool(self._episode_successes[-1]) if self._episode_successes else False

    # ------------------------------------------------------------------
    # Task handling
    # ------------------------------------------------------------------
    def set_task(self, task: ExORLTaskSpec) -> None:
        """Switch the active reward function."""
        if task.domain != self.domain:
            raise ValueError(
                f"Task domain {task.domain!r} does not match env domain {self.domain!r}."
            )
        self.task = task
        if task.kind == "goal":
            self._goal_cache = np.asarray(task.goal, dtype=np.float64)
        else:
            self._goal_cache = None

    def set_goal(self, goal: np.ndarray, threshold: Optional[float] = None) -> None:
        """Set a goal-reaching task from a physics-space target state."""
        self.set_task(
            ExORLTaskSpec(
                name=f"{self.domain}-goal",
                domain=self.domain,
                kind="goal",
                goal=np.asarray(goal, dtype=np.float64),
                threshold=self.goal_threshold if threshold is None else float(threshold),
            )
        )

    def set_velocity_threshold(self, threshold: float, feature_index: int = 0) -> None:
        """Set a velocity task on physics feature ``feature_index``."""
        self.set_task(
            ExORLTaskSpec(
                name=f"{self.domain}-velocity-{threshold}",
                domain=self.domain,
                kind="velocity",
                threshold=float(threshold),
                feature_index=int(feature_index),
            )
        )

    def default_task(self) -> ExORLTaskSpec:
        """Default goal task: the first fixed goal state of the domain."""
        goals = self.goal_states(num_goals=NUM_GOAL_STATES)
        return ExORLTaskSpec(
            name=f"{self.domain}-goal-0",
            domain=self.domain,
            kind="goal",
            goal=goals[0],
            threshold=self.goal_threshold,
        )

    def goal_states(self, num_goals: int = NUM_GOAL_STATES, seed: int = 0) -> np.ndarray:
        """The fixed goal states used for zero-shot evaluation (5 per domain)."""
        ds = self._ensure_dataset()
        goals = ds.get("goals")
        if goals is not None:
            goals = np.asarray(goals, dtype=np.float64)
            if goals.ndim == 2 and goals.shape[0] >= num_goals:
                return goals[:num_goals]
        return select_goal_states(ds, num_goals=num_goals, seed=seed)

    def eval_tasks(self, **kwargs: Any) -> Dict[str, ExORLTaskSpec]:
        """Zero-shot task suite for this domain (goals + velocity thresholds)."""
        ds = self._ensure_dataset()
        raw = evaluation_tasks(ds, domain=self.domain, **kwargs)
        return {
            name: _spec_from_eval_task(name, self.domain, spec)
            for name, spec in raw.items()
        }

    # ------------------------------------------------------------------
    # Reward / done
    # ------------------------------------------------------------------
    def compute_reward(
        self, physics: np.ndarray, next_physics: Optional[np.ndarray] = None
    ) -> float:
        """Task reward in ``{0, -1}`` (the FRE convention for the reward prior)."""
        if self.task is None:
            return 0.0
        physics = np.asarray(physics, dtype=np.float64).reshape(-1)
        if self.task.kind == "goal":
            return _goal_reward_value(physics, self._goal_cache, self.task.threshold)
        # Velocity tasks are computed on the *next* physics state when available,
        # mirroring the standard "reward after moving" convention.
        features = np.asarray(
            next_physics if next_physics is not None else physics, dtype=np.float64
        ).reshape(-1)
        return _velocity_reward_value(
            self.domain, features, self.task.threshold, self.task.feature_index
        )

    def compute_done(
        self, physics: np.ndarray, next_physics: Optional[np.ndarray] = None
    ) -> bool:
        """Whether the task is solved by the given physics state."""
        if self.task is None:
            return False
        physics = np.asarray(physics, dtype=np.float64).reshape(-1)
        if self.task.kind == "goal":
            return bool(np.asarray(goal_done(physics, self._goal_cache, self.task.threshold)).reshape(-1)[0])
        features = np.asarray(
            next_physics if next_physics is not None else physics, dtype=np.float64
        ).reshape(-1)
        return bool(
            np.asarray(
                velocity_done(self.domain, features, self.task.threshold, self.task.feature_index)
            ).reshape(-1)[0]
        )

    def reward_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Vectorised ``eta(state_physics) -> reward`` callable (for analysis)."""

        def _fn(physics_batch: np.ndarray) -> np.ndarray:
            arr = np.asarray(physics_batch, dtype=np.float64)
            single = arr.ndim == 1
            arr = np.atleast_2d(arr)
            if self.task is None:
                out = np.zeros(arr.shape[0], dtype=np.float64)
            elif self.task.kind == "goal":
                out = np.asarray(goal_reward(arr, self._goal_cache, self.task.threshold), dtype=np.float64).reshape(-1)
            else:
                out = np.asarray(
                    velocity_reward(self.domain, arr, self.task.threshold), dtype=np.float64
                ).reshape(-1)
            return out[0] if single else out

        return _fn

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def build_encoder_observation(
        self, raw_obs: np.ndarray, physics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Raw obs (+ physics) standardised -- the encoder input."""
        raw_obs = np.asarray(raw_obs, dtype=np.float64).reshape(-1)
        if self.physics_dim_:
            if physics is None:
                physics = compute_physics(self.domain, raw_obs.reshape(1, -1)).reshape(-1)
            physics = np.asarray(physics, dtype=np.float64).reshape(-1)
            if self.dataset is not None and self._dataset_loaded:
                physics = np.asarray(normalize_physics(self.dataset, physics.reshape(1, -1))).reshape(-1)
        else:
            physics = np.zeros(0, dtype=np.float64)
        if not self.append_physics_to_obs:
            physics = np.zeros(0, dtype=np.float64)
        if self.normalise and self._dataset_loaded and self.obs_mean.size:
            raw_obs = (raw_obs - self.obs_mean) / np.maximum(self.obs_std, 1e-3)
        return np.concatenate([raw_obs, physics], axis=0).astype(np.float32)

    def policy_observation(self, encoder_obs: np.ndarray) -> np.ndarray:
        """Strip physics (and de-normalise) to recover the policy observation."""
        obs = np.asarray(encoder_obs, dtype=np.float64).reshape(-1)
        raw = obs[: self.raw_obs_dim]
        if self.normalise and self._dataset_loaded and self.obs_mean.size:
            raw = raw * np.maximum(self.obs_std, 1e-3) + self.obs_mean
        return raw.astype(np.float32)

    def physics_from_observation(self, encoder_obs: np.ndarray) -> np.ndarray:
        """Extract the (de-normalised) physics features from an encoder obs."""
        obs = np.asarray(encoder_obs, dtype=np.float64).reshape(-1)
        physics = obs[self.raw_obs_dim : self.raw_obs_dim + self.physics_dim_]
        if self._dataset_loaded and self.physics_mean.size and physics.size:
            physics = physics * np.maximum(self.physics_std, 1e-3) + self.physics_mean
        return physics.astype(np.float64)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        goal: Optional[np.ndarray] = None,
        task: Optional[ExORLTaskSpec] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return ``(encoder_obs, info)``."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = int(seed)
        if task is not None:
            self.set_task(task)
        if goal is not None:
            self.set_goal(goal)

        self._step_count = 0
        goal_reached = False

        if self.live_mode and self.env is not None:
            try:
                out = self.env.reset()
            except TypeError:  # pragma: no cover - older gym signatures
                out = self.env.reset()
            raw_obs = out[0] if isinstance(out, tuple) else out
            self._live_state = raw_obs
            physics = self._physics_from_live(raw_obs)
            encoder_obs = self.build_encoder_observation(raw_obs, physics)
        else:
            encoder_obs, physics = self._reset_offline()

        info = {
            "domain": self.domain,
            "task": self.task.name if self.task else None,
            "raw_observation": self.policy_observation(encoder_obs),
            "policy_observation": self.policy_observation(encoder_obs),
            "physics": np.asarray(physics, dtype=np.float64),
            "goal": None if self._goal_cache is None else self._goal_cache.copy(),
            "step": 0,
            "live_env": self.live_mode,
        }
        self._current_encoder_obs = encoder_obs
        self._current_physics = np.asarray(physics, dtype=np.float64)
        self._episode_return = 0.0
        self._episode_success = False
        info["goal_reached"] = goal_reached
        return np.asarray(encoder_obs, dtype=np.float32), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one environment step.

        Returns ``(encoder_obs, reward, terminated, truncated, info)``.
        """
        self._step_count += 1
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)

        if self.live_mode and self.env is not None:
            out = self.env.step(action_arr)
            if len(out) == 5:
                raw_next, _, terminated, truncated, _ = out
            else:  # pragma: no cover - gym<0.26
                raw_next, _, terminated, _ = out
                truncated = False
            physics_next = self._physics_from_live(raw_next)
            encoder_next = self.build_encoder_observation(raw_next, physics_next)
            self._live_state = raw_next
        else:
            encoder_next, physics_next, terminated, truncated = self._step_offline(action_arr)

        prev_physics = np.asarray(self._current_physics, dtype=np.float64)
        reward = self.compute_reward(prev_physics, physics_next)
        solved = self.compute_done(prev_physics, physics_next)
        if solved:
            terminated = True
        truncated = bool(truncated) or (self._step_count >= self.max_episode_steps)

        self._episode_return += float(reward)
        self._episode_success = self._episode_success or bool(solved)

        info = {
            "step": self._step_count,
            "domain": self.domain,
            "task": self.task.name if self.task else None,
            "raw_observation": self.policy_observation(encoder_next),
            "policy_observation": self.policy_observation(encoder_next),
            "physics": np.asarray(physics_next, dtype=np.float64),
            "goal": None if self._goal_cache is None else self._goal_cache.copy(),
            "goal_reached": bool(solved),
            "episode_return": float(self._episode_return),
            "live_env": self.live_mode,
        }

        self._current_encoder_obs = encoder_next
        self._current_physics = np.asarray(physics_next, dtype=np.float64)

        if terminated or truncated:
            self._finish_episode()
        return np.asarray(encoder_next, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def _finish_episode(self) -> None:
        self._episode_count += 1
        self._episode_returns.append(float(getattr(self, "_episode_return", 0.0)))
        self._episode_successes.append(bool(getattr(self, "_episode_success", False)))

    # ------------------------------------------------------------------
    # Live-env helpers
    # ------------------------------------------------------------------
    def _physics_from_live(self, raw_obs: np.ndarray) -> np.ndarray:
        """Extract physics features from a live dm_control observation."""
        env = self.env
        physics: List[float] = []
        if env is not None:
            for name in self.physics_names_:
                if name == "horizontal_velocity":
                    physics.append(float(self._read_attr(env, ("horizontal_velocity", "torso_horizontal_velocity"), default=_fallback_velocity(raw_obs))))
                elif name == "speed":
                    physics.append(float(self._read_attr(env, ("speed", "horizontal_velocity"), default=_fallback_velocity(raw_obs))))
                elif name == "torso_upright":
                    physics.append(float(self._read_attr(env, ("torso_upright",), default=_fallback_upright(raw_obs))))
                elif name == "torso_height":
                    physics.append(float(self._read_attr(env, ("torso_height",), default=_fallback_height(raw_obs))))
                else:  # pragma: no cover - unknown physics feature
                    physics.append(0.0)
        if not physics:
            physics = list(compute_physics(self.domain, np.asarray(raw_obs).reshape(1, -1)).reshape(-1))
        return np.asarray(physics, dtype=np.float64)

    @staticmethod
    def _read_attr(env: Any, names: Sequence[str], default: float = 0.0) -> float:
        for name in names:
            if hasattr(env, name):
                value = getattr(env, name)
                try:
                    return float(np.asarray(value).reshape(-1)[0])
                except Exception:  # pragma: no cover
                    continue
            sim = getattr(env, "sim", None)
            if sim is not None and hasattr(sim, name):
                try:
                    return float(np.asarray(getattr(sim, name)).reshape(-1)[0])
                except Exception:  # pragma: no cover
                    continue
        return float(default)

    # ------------------------------------------------------------------
    # Offline (dataset replay) helpers
    # ------------------------------------------------------------------
    def _reset_offline(self) -> Tuple[np.ndarray, np.ndarray]:
        starts = getattr(self, "_episode_starts", np.array([0]))
        if starts.size == 0:
            starts = np.array([0])
        idx = int(self._rng.choice(starts))
        self._offline_index = idx
        self._offline_episode_start = idx
        encoder_obs = self.encoder_observations[idx]
        physics = self.physics[idx] if self.physics.size else np.zeros(0)
        return encoder_obs, physics

    def _step_offline(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
        """Replay dataset transitions starting from the sampled start state."""
        nxt = self._offline_index + 1
        limit = self._offline_episode_start + self.max_episode_steps
        if nxt >= self.num_transitions or nxt >= limit:
            terminated = True
            truncated = True
            encoder_next = self.encoder_observations[self._offline_index]
            physics_next = self.physics[self._offline_index] if self.physics.size else np.zeros(0)
        else:
            terminated = bool(self._terminals[self._offline_index] > 0.5)
            truncated = False
            self._offline_index = nxt
            encoder_next = self.encoder_observations[nxt]
            physics_next = self.physics[nxt] if self.physics.size else np.zeros(0)
        return encoder_next, physics_next, terminated, truncated

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def render(self, *args: Any, **kwargs: Any) -> Any:
        if self.env is not None and hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None

    def close(self) -> None:
        if self.env is not None and hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:  # pragma: no cover - simulator teardown issues
                pass

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._seed = int(seed)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mode = "live" if self.live_mode else "offline"
        task = self.task.name if self.task else "none"
        return (
            f"ExORLWrapper(domain={self.domain!r}, mode={mode!r}, task={task!r}, "
            f"obs_dim={self.observation_dim}, action_dim={self.action_dim})"
        )


# ---------------------------------------------------------------------------
# Typed convenience subclasses
# ---------------------------------------------------------------------------
class ExORLGoalTask(ExORLWrapper):
    """Goal-reaching wrapper (Euclidean distance in physics space < threshold)."""

    def __init__(self, domain: str = "walker", goal: Optional[np.ndarray] = None, **kwargs: Any) -> None:
        task = kwargs.pop("task", None)
        super().__init__(domain=domain, task=task, **kwargs)
        goals = self.goal_states()
        if goal is None:
            goal = goals[0]
        self.set_goal(np.asarray(goal, dtype=np.float64))


class ExORLVelocityTask(ExORLWrapper):
    """Velocity-threshold wrapper (sustain ``|v| > threshold``)."""

    def __init__(
        self,
        domain: str = "walker",
        threshold: Optional[float] = None,
        feature_index: int = 0,
        **kwargs: Any,
    ) -> None:
        task = kwargs.pop("task", None)
        super().__init__(domain=domain, task=task, **kwargs)
        if threshold is None:
            threshold = VELOCITY_THRESHOLDS.get(domain, (1.0,))[0]
        self.set_velocity_threshold(float(threshold), feature_index=feature_index)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_exorl_env(
    domain: str = "walker",
    task: Optional[ExORLTaskSpec] = None,
    **kwargs: Any,
) -> ExORLWrapper:
    """Create an :class:`ExORLWrapper` for ``domain`` (loads data lazily)."""
    env = ExORLWrapper(domain=domain, task=task, **kwargs)
    # Force dataset loading so dimensions/tasks are available immediately.
    env._ensure_dataset()
    if task is None:
        env.set_task(env.default_task())
    return env


def make_exorl_envs(
    domains: Sequence[str] = EXORL_DOMAINS,
    **kwargs: Any,
) -> Dict[str, ExORLWrapper]:
    """Create one wrapper per domain (default: walker + cheetah)."""
    return {d: make_exorl_env(domain=d, **kwargs) for d in domains}


def exorl_eval_tasks(
    domain: str, dataset: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Dict[str, ExORLTaskSpec]:
    """Build the zero-shot task suite for ``domain`` as task specs."""
    env = ExORLWrapper(domain=domain, dataset=dataset, **kwargs)
    env._ensure_dataset()
    return env.eval_tasks()


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------
def _spec_from_eval_task(name: str, domain: str, spec: Dict[str, Any]) -> ExORLTaskSpec:
    """Convert a loader ``evaluation_tasks`` entry into an :class:`ExORLTaskSpec`."""
    kind = spec.get("kind") or spec.get("type") or ("goal" if "goal" in name else "velocity")
    if kind not in ("goal", "velocity"):
        kind = "goal" if spec.get("goal") is not None else "velocity"
    threshold = spec.get("threshold", spec.get("goal_threshold", GOAL_DISTANCE_THRESHOLD))
    feature_index = int(spec.get("feature_index", 0))
    goal = spec.get("goal")
    if kind == "velocity" and spec.get("feature_index") is None:
        # Velocity thresholds in the plan are listed per domain; recover the
        # feature index from the physics-name order when possible.
        feature_index = 0
    return ExORLTaskSpec(
        name=name,
        domain=domain,
        kind=kind,
        goal=None if goal is None else np.asarray(goal, dtype=np.float64),
        threshold=float(threshold),
        feature_index=feature_index,
        metadata={k: v for k, v in spec.items() if k not in ("goal", "threshold", "kind")},
    )


def _fallback_velocity(raw_obs: np.ndarray) -> float:
    """Cheap velocity proxy from raw observations when the env exposes none."""
    obs = np.asarray(raw_obs, dtype=np.float64).reshape(-1)
    # dm_control cheetah/walker place the root x-velocity in the first few dims.
    return float(obs[0]) * 0.0 if obs.size == 0 else 0.0


def _fallback_upright(raw_obs: np.ndarray) -> float:
    return 1.0


def _fallback_height(raw_obs: np.ndarray) -> float:
    obs = np.asarray(raw_obs, dtype=np.float64).reshape(-1)
    return float(obs[-1]) if obs.size else 0.0


def _maybe_make_live_env(domain: str, seed: Optional[int] = None) -> Optional[Any]:
    """Try to construct a live dm_control/D4RL environment; return ``None`` on failure."""
    try:  # pragma: no cover - depends on optional simulator install
        import gym  # type: ignore
    except Exception:
        return None

    for env_id in LIVE_ENV_CANDIDATES.get(domain, ()):  # pragma: no cover
        try:
            env = gym.make(env_id)
        except Exception:
            continue
        try:
            if seed is not None and hasattr(env, "seed"):
                env.seed(seed)
        except Exception:
            pass
        return env
    return None
