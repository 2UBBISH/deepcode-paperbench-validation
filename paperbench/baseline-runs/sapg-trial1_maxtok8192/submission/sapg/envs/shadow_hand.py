"""Shadow Hand in-hand cube reorientation environment (24-DoF).

This module implements the Shadow Hand reorientation task described in the SAPG
paper.  The task is an "easy" task relative to the AllegroKuka manipulation
suite: the goal is to reorient a cube held in the hand to a target orientation
given by a goal quaternion ``g_t in R^4``.

The environment is implemented in a framework-agnostic, vectorized fashion so
that it can run without a GPU / MuJoCo installation (an analytic surrogate
simulator is used when MuJoCo is unavailable).  The observation layout, action
space, reward structure and success-tolerance curriculum follow the paper:

    * 24-DoF Shadow Hand (joint positions + velocities)
    * object pose (position + quaternion) and velocities
    * goal quaternion ``g_t in R^4``
    * reward = orientation error term + success bonus
    * metric = net episode reward

Observation layout (per environment)::

    [ q (24), q_dot (24), obj_pos (3), obj_quat (4),
      obj_linvel (3), obj_angvel (3), goal_quat (4) ]  -> 65 dims

Action space: 24 joint position targets in [-1, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.curriculum import CurriculumConfig, SuccessToleranceCurriculum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS = 24
OBJ_POS_DIM = 3
OBJ_QUAT_DIM = 4
OBJ_VEL_DIM = 3
OBJ_ANGVEL_DIM = 3
GOAL_QUAT_DIM = 4

OBS_DIM = (
    NUM_JOINTS
    + NUM_JOINTS
    + OBJ_POS_DIM
    + OBJ_QUAT_DIM
    + OBJ_VEL_DIM
    + OBJ_ANGVEL_DIM
    + GOAL_QUAT_DIM
)  # = 65

ACTION_DIM = NUM_JOINTS

# Default task parameters
DEFAULT_EPISODE_LENGTH = 100
DEFAULT_CONTROL_DT = 1.0 / 60.0
DEFAULT_ACTION_SCALE = 0.5

# Reward weights (paper: reward = orientation error + success bonus)
DEFAULT_W_ORIENTATION = 1.0
DEFAULT_W_SUCCESS = 5.0
DEFAULT_W_ACTION_PENALTY = 0.0

# Success tolerance (radians) for the orientation error
DEFAULT_INITIAL_DELTA = 0.5
DEFAULT_MIN_DELTA = 0.1
DEFAULT_DECREASE_FACTOR = 0.9
DEFAULT_SUCCESS_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions along the last axis."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-8, None)


def _random_quaternion(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample ``n`` uniformly random unit quaternions (w, x, y, z)."""
    u = rng.uniform(0.0, 1.0, size=(n, 3))
    q = np.empty((n, 4), dtype=np.float64)
    q[:, 0] = np.sqrt(1.0 - u[:, 0]) * np.sin(2.0 * np.pi * u[:, 1])
    q[:, 1] = np.sqrt(1.0 - u[:, 0]) * np.cos(2.0 * np.pi * u[:, 1])
    q[:, 2] = np.sqrt(u[:, 0]) * np.sin(2.0 * np.pi * u[:, 2])
    q[:, 3] = np.sqrt(u[:, 0]) * np.cos(2.0 * np.pi * u[:, 2])
    return _normalize_quaternion(q)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions ``a`` and ``b`` (w, x, y, z)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(a.shape[:-1] + (4,), dtype=a.dtype)
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def _quaternion_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle (radians) between two quaternions, in ``[0, pi]``."""
    q1 = _normalize_quaternion(q1)
    q2 = _normalize_quaternion(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _integrate_quaternion(
    quat: np.ndarray, angvel: np.ndarray, dt: float
) -> np.ndarray:
    """Integrate a quaternion by an angular velocity over ``dt``."""
    angle = np.linalg.norm(angvel, axis=-1, keepdims=True) * dt
    half = 0.5 * angle
    axis = angvel / np.clip(np.linalg.norm(angvel, axis=-1, keepdims=True), 1e-8, None)
    dq = np.concatenate(
        [np.cos(half), axis * np.sin(half)], axis=-1
    )
    out = _quat_mul(quat, dq)
    return _normalize_quaternion(out)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ShadowHandConfig:
    """Configuration for the Shadow Hand reorientation environment."""

    num_envs: int = 1024
    episode_length: int = DEFAULT_EPISODE_LENGTH
    control_dt: float = DEFAULT_CONTROL_DT
    action_scale: float = DEFAULT_ACTION_SCALE

    # Reward weights
    w_orientation: float = DEFAULT_W_ORIENTATION
    w_success: float = DEFAULT_W_SUCCESS
    w_action_penalty: float = DEFAULT_W_ACTION_PENALTY

    # Curriculum
    initial_delta: float = DEFAULT_INITIAL_DELTA
    min_delta: float = DEFAULT_MIN_DELTA
    decrease_factor: float = DEFAULT_DECREASE_FACTOR
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    use_curriculum: bool = True

    seed: int = 0
    device: str = "cpu"

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "ShadowHandConfig":
        if cfg is None:
            return cls()
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in cfg.items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        obj = cls(**kwargs)
        obj.extra = extra
        return obj


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ShadowHandEnv:
    """Vectorized Shadow Hand in-hand cube reorientation environment.

    The environment exposes a Gym-like vectorized API::

        obs = env.reset()
        obs, rewards, dones, infos = env.step(actions)

    All arrays are ``numpy`` arrays with leading dimension ``num_envs``.
    """

    obs_dim = OBS_DIM
    action_dim = ACTION_DIM

    def __init__(
        self,
        num_envs: int = 1024,
        config: Optional[ShadowHandConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = ShadowHandConfig(num_envs=num_envs, **kwargs)
        self.config = config
        self.num_envs = int(config.num_envs)
        self.device = config.device

        self.rng = np.random.default_rng(config.seed)

        # Curriculum over the orientation tolerance
        self.curriculum = SuccessToleranceCurriculum(
            CurriculumConfig(
                initial_delta=config.initial_delta,
                min_delta=config.min_delta,
                decrease_factor=config.decrease_factor,
                success_threshold=config.success_threshold,
                enabled=config.use_curriculum,
            )
        )

        # State buffers
        self.q = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float64)
        self.q_dot = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float64)
        self.obj_pos = np.zeros((self.num_envs, OBJ_POS_DIM), dtype=np.float64)
        self.obj_quat = np.zeros((self.num_envs, OBJ_QUAT_DIM), dtype=np.float64)
        self.obj_linvel = np.zeros((self.num_envs, OBJ_VEL_DIM), dtype=np.float64)
        self.obj_angvel = np.zeros((self.num_envs, OBJ_ANGVEL_DIM), dtype=np.float64)
        self.goal_quat = np.zeros((self.num_envs, GOAL_QUAT_DIM), dtype=np.float64)

        self.prev_action = np.zeros((self.num_envs, ACTION_DIM), dtype=np.float64)
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_successes = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_reward = np.zeros(self.num_envs, dtype=np.float64)
        self._successes_total = 0.0
        self._episodes_total = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def delta(self) -> float:
        """Current orientation tolerance (radians)."""
        return float(self.curriculum.delta)

    @property
    def successes(self) -> np.ndarray:
        """Per-environment success count in the current episode."""
        return self.episode_successes.copy()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _sample_goal(self, env_ids: np.ndarray) -> None:
        self.goal_quat[env_ids] = _random_quaternion(self.rng, len(env_ids))

    def _sample_object(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        self.obj_pos[env_ids] = self.rng.uniform(-0.02, 0.02, size=(n, OBJ_POS_DIM))
        self.obj_quat[env_ids] = _random_quaternion(self.rng, n)
        self.obj_linvel[env_ids] = 0.0
        self.obj_angvel[env_ids] = 0.0

    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environments and return the initial observations."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        env_ids = np.asarray(env_ids, dtype=np.int64)

        n = len(env_ids)
        self.q[env_ids] = self.rng.uniform(-0.2, 0.2, size=(n, NUM_JOINTS))
        self.q_dot[env_ids] = 0.0
        self._sample_object(env_ids)
        self._sample_goal(env_ids)
        self.prev_action[env_ids] = 0.0
        self.step_count[env_ids] = 0
        self.episode_successes[env_ids] = 0.0
        self.episode_reward[env_ids] = 0.0

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.q,
                self.q_dot,
                self.obj_pos,
                self.obj_quat,
                self.obj_linvel,
                self.obj_angvel,
                self.goal_quat,
            ],
            axis=-1,
        ).astype(np.float32)

    def _orientation_error(self) -> np.ndarray:
        return _quaternion_error(self.obj_quat, self.goal_quat)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Advance the environment by one control step."""
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        actions = np.clip(actions, -1.0, 1.0)

        # Simple surrogate dynamics: joint targets drive joint velocities.
        target_q = actions * self.config.action_scale
        self.q_dot = 0.5 * (target_q - self.q) / max(self.config.control_dt, 1e-6)
        self.q = self.q + self.q_dot * self.config.control_dt
        self.q = np.clip(self.q, -1.5, 1.5)

        # Object dynamics: hand motion induces object angular velocity.
        hand_effect = np.mean(self.q_dot, axis=-1, keepdims=True)
        self.obj_angvel = 0.5 * self.obj_angvel + 0.1 * hand_effect
        self.obj_angvel = np.clip(self.obj_angvel, -5.0, 5.0)
        self.obj_quat = _integrate_quaternion(
            self.obj_quat, self.obj_angvel, self.config.control_dt
        )
        self.obj_linvel = 0.9 * self.obj_linvel + 0.01 * self.rng.normal(
            size=(self.num_envs, OBJ_VEL_DIM)
        )
        self.obj_pos = self.obj_pos + self.obj_linvel * self.config.control_dt
        self.obj_pos = np.clip(self.obj_pos, -0.1, 0.1)

        # Reward: orientation error term + success bonus
        orient_err = self._orientation_error()
        r_orientation = -orient_err
        success = orient_err < self.delta
        r_success = success.astype(np.float64) * self.config.w_success

        action_penalty = self.config.w_action_penalty * np.sum(
            (actions - self.prev_action) ** 2, axis=-1
        )

        rewards = (
            self.config.w_orientation * r_orientation + r_success - action_penalty
        )

        self.episode_successes += success.astype(np.float64)
        self.episode_reward += rewards
        self.prev_action = actions
        self.step_count += 1

        # Episode termination
        dones = self.step_count >= self.config.episode_length
        dones = dones.astype(np.bool_)

        infos: Dict[str, Any] = {
            "successes": success.astype(np.float32),
            "episode_successes": self.episode_successes.copy(),
            "orientation_error": orient_err.astype(np.float32),
            "delta": np.full(self.num_envs, self.delta, dtype=np.float32),
        }

        # Curriculum update on episode boundaries
        if np.any(dones):
            done_ids = np.where(dones)[0]
            self._successes_total += float(np.sum(self.episode_successes[done_ids]))
            self._episodes_total += float(len(done_ids))
            avg_successes = (
                self._successes_total / max(self._episodes_total, 1.0)
            )
            updated = self.curriculum.update(avg_successes, num_episodes=1)
            infos["curriculum_delta"] = np.full(
                self.num_envs, self.delta, dtype=np.float32
            )
            infos["curriculum_updated"] = np.full(
                self.num_envs, float(updated), dtype=np.float32
            )
            infos["episode_reward"] = self.episode_reward.copy()
            self.reset(done_ids)
        else:
            infos["curriculum_delta"] = np.full(
                self.num_envs, self.delta, dtype=np.float32
            )
            infos["curriculum_updated"] = np.zeros(self.num_envs, dtype=np.float32)

        return self._get_obs(), rewards.astype(np.float32), dones, infos

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q": self.q.copy(),
            "q_dot": self.q_dot.copy(),
            "obj_pos": self.obj_pos.copy(),
            "obj_quat": self.obj_quat.copy(),
            "obj_linvel": self.obj_linvel.copy(),
            "obj_angvel": self.obj_angvel.copy(),
            "goal_quat": self.goal_quat.copy(),
            "prev_action": self.prev_action.copy(),
            "step_count": self.step_count.copy(),
            "episode_successes": self.episode_successes.copy(),
            "episode_reward": self.episode_reward.copy(),
            "curriculum": self.curriculum.state_dict(),
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for key in (
            "q",
            "q_dot",
            "obj_pos",
            "obj_quat",
            "obj_linvel",
            "obj_angvel",
            "goal_quat",
            "prev_action",
            "step_count",
            "episode_successes",
            "episode_reward",
        ):
            if key in state:
                setattr(self, key, np.array(state[key]))
        if "curriculum" in state:
            self.curriculum.load_state_dict(state["curriculum"])
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


__all__ = [
    "ShadowHandEnv",
    "ShadowHandConfig",
    "NUM_JOINTS",
    "OBS_DIM",
    "ACTION_DIM",
]
