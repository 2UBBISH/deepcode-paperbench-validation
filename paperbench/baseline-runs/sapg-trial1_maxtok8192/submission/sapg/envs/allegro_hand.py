"""Allegro Hand 16-DoF in-hand cube reorientation environment.

This is one of the "easy" tasks in the SAPG paper (Section 5). The Allegro Hand
has 16 degrees of freedom (4 fingers x 4 joints) and the task is to reorient a
cube to a target orientation.

The implementation is framework-agnostic (NumPy) with an analytic surrogate
simulator so that it can run without MuJoCo/IsaacGym. The observation layout,
action space, reward structure and curriculum integration mirror the paper's
description and the Shadow Hand implementation.

Observation layout (dim = 57):
    [ q (16), q_dot (16), obj_pos (3), obj_quat (4),
      obj_linvel (3), obj_angvel (3), goal_quat (4) ]

Reward:
    r = w_orientation * (-orientation_error) + w_success * success
        - w_action_penalty * ||a||^2

Success:
    orientation error < delta (tolerance curriculum, 0.5 -> 0.1 rad, x0.9 when
    average successes per episode > 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..utils.curriculum import CurriculumConfig, SuccessToleranceCurriculum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_JOINTS = 16
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
)  # = 57
ACTION_DIM = NUM_JOINTS

DEFAULT_EPISODE_LENGTH = 100
DEFAULT_CONTROL_DT = 1.0 / 60.0
DEFAULT_ACTION_SCALE = 0.5
DEFAULT_W_ORIENTATION = 1.0
DEFAULT_W_SUCCESS = 5.0
DEFAULT_W_ACTION_PENALTY = 0.0
DEFAULT_INITIAL_DELTA = 0.5
DEFAULT_MIN_DELTA = 0.1
DEFAULT_DECREASE_FACTOR = 0.9
DEFAULT_SUCCESS_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-8, None)


def _random_quaternion(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n uniform random unit quaternions in (w, x, y, z) order."""
    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)
    q = np.stack(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ],
        axis=-1,
    )
    # Convert (x, y, z, w) -> (w, x, y, z)
    q = q[..., [3, 0, 1, 2]]
    return _normalize_quaternion(q)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quaternion_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle (radians) between two quaternions, in [0, pi]."""
    q1 = _normalize_quaternion(q1)
    q2 = _normalize_quaternion(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _integrate_quaternion(quat: np.ndarray, angvel: np.ndarray, dt: float) -> np.ndarray:
    """Integrate a quaternion by an angular velocity over dt."""
    theta = np.linalg.norm(angvel, axis=-1, keepdims=True) * dt
    half = 0.5 * theta
    axis = angvel / np.clip(np.linalg.norm(angvel, axis=-1, keepdims=True), 1e-8, None)
    dq = np.concatenate(
        [np.cos(half), axis * np.sin(half)],
        axis=-1,
    )
    out = _quat_mul(quat, dq)
    return _normalize_quaternion(out)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AllegroHandConfig:
    """Configuration for the Allegro Hand reorientation environment."""

    num_envs: int = 1024
    episode_length: int = DEFAULT_EPISODE_LENGTH
    control_dt: float = DEFAULT_CONTROL_DT
    action_scale: float = DEFAULT_ACTION_SCALE
    w_orientation: float = DEFAULT_W_ORIENTATION
    w_success: float = DEFAULT_W_SUCCESS
    w_action_penalty: float = DEFAULT_W_ACTION_PENALTY
    initial_delta: float = DEFAULT_INITIAL_DELTA
    min_delta: float = DEFAULT_MIN_DELTA
    decrease_factor: float = DEFAULT_DECREASE_FACTOR
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    use_curriculum: bool = True
    seed: int = 0
    device: str = "cpu"
    extra: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "AllegroHandConfig":
        if cfg is None:
            return cls()
        cfg = dict(cfg)
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        extra = {k: v for k, v in cfg.items() if k not in known}
        obj = cls(**kwargs)
        obj.extra = extra
        return obj


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class AllegroHandEnv:
    """Vectorized Allegro Hand 16-DoF in-hand cube reorientation environment."""

    obs_dim = OBS_DIM
    action_dim = ACTION_DIM

    def __init__(
        self,
        num_envs: int = 1024,
        config: Optional[AllegroHandConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = AllegroHandConfig(num_envs=num_envs, **kwargs)
        self.config = config
        self.num_envs = int(config.num_envs)
        self.device = config.device

        self.rng = np.random.default_rng(config.seed)

        # Joint state
        self.q = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float64)
        self.q_dot = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float64)
        self.q_target = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float64)

        # Object state
        self.obj_pos = np.zeros((self.num_envs, OBJ_POS_DIM), dtype=np.float64)
        self.obj_quat = np.zeros((self.num_envs, OBJ_QUAT_DIM), dtype=np.float64)
        self.obj_linvel = np.zeros((self.num_envs, OBJ_VEL_DIM), dtype=np.float64)
        self.obj_angvel = np.zeros((self.num_envs, OBJ_ANGVEL_DIM), dtype=np.float64)

        # Goal orientation
        self.goal_quat = np.zeros((self.num_envs, GOAL_QUAT_DIM), dtype=np.float64)

        # Episode bookkeeping
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_successes = np.zeros(self.num_envs, dtype=np.int64)
        self._successes = np.zeros(self.num_envs, dtype=np.int64)

        # Curriculum
        self.curriculum: Optional[SuccessToleranceCurriculum] = None
        if config.use_curriculum:
            self.curriculum = SuccessToleranceCurriculum(
                CurriculumConfig(
                    initial_delta=config.initial_delta,
                    min_delta=config.min_delta,
                    decrease_factor=config.decrease_factor,
                    success_threshold=config.success_threshold,
                )
            )

        self.reset()

    # -- properties --------------------------------------------------------
    @property
    def delta(self) -> float:
        if self.curriculum is not None:
            return float(self.curriculum.delta)
        return float(self.config.initial_delta)

    @property
    def successes(self) -> np.ndarray:
        return self._successes

    # -- helpers -----------------------------------------------------------
    def _sample_goal(self, env_ids: np.ndarray) -> None:
        self.goal_quat[env_ids] = _random_quaternion(self.rng, len(env_ids))

    def _sample_object(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        self.obj_pos[env_ids] = self.rng.uniform(-0.02, 0.02, size=(n, OBJ_POS_DIM))
        self.obj_quat[env_ids] = _random_quaternion(self.rng, n)
        self.obj_linvel[env_ids] = 0.0
        self.obj_angvel[env_ids] = 0.0

    def _orientation_error(self) -> np.ndarray:
        return _quaternion_error(self.obj_quat, self.goal_quat)

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

    # -- gym API -----------------------------------------------------------
    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        env_ids = np.asarray(env_ids, dtype=np.int64)
        n = len(env_ids)

        self.q[env_ids] = self.rng.uniform(-0.2, 0.2, size=(n, NUM_JOINTS))
        self.q_dot[env_ids] = 0.0
        self.q_target[env_ids] = self.q[env_ids]
        self._sample_object(env_ids)
        self._sample_goal(env_ids)

        self.episode_lengths[env_ids] = 0
        self.episode_successes[env_ids] = 0
        self._successes[env_ids] = 0

        return self._get_obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        actions = np.clip(actions, -1.0, 1.0)

        # Joint target tracking (simple first-order dynamics)
        self.q_target = actions * self.config.action_scale
        self.q_dot = (self.q_target - self.q) / max(self.config.control_dt, 1e-6)
        self.q = self.q + self.q_dot * self.config.control_dt

        # Object dynamics: hand motion induces object angular velocity
        hand_angvel = np.mean(self.q_dot, axis=-1, keepdims=True) * 0.1
        self.obj_angvel = 0.9 * self.obj_angvel + 0.1 * np.concatenate(
            [hand_angvel, hand_angvel, hand_angvel], axis=-1
        )
        self.obj_quat = _integrate_quaternion(
            self.obj_quat, self.obj_angvel, self.config.control_dt
        )
        self.obj_pos = self.obj_pos + self.obj_linvel * self.config.control_dt

        # Reward
        orient_err = self._orientation_error()
        success = (orient_err < self.delta).astype(np.float64)
        action_penalty = np.sum(actions ** 2, axis=-1)

        rewards = (
            self.config.w_orientation * (-orient_err)
            + self.config.w_success * success
            - self.config.w_action_penalty * action_penalty
        )

        self._successes = success.astype(np.int64)
        self.episode_successes += self._successes
        self.episode_lengths += 1

        # On success, resample goal and object (paper: reset target/object to a
        # random location after each success).
        success_ids = np.where(success > 0.5)[0]
        if len(success_ids) > 0:
            self._sample_goal(success_ids)
            self._sample_object(success_ids)

        # Episode termination
        dones = (self.episode_lengths >= self.config.episode_length).astype(np.float64)
        done_ids = np.where(dones > 0.5)[0]

        infos: Dict[str, np.ndarray] = {
            "successes": self._successes.astype(np.float32),
            "episode_successes": self.episode_successes.astype(np.float32),
            "delta": np.full(self.num_envs, self.delta, dtype=np.float32),
            "orientation_error": orient_err.astype(np.float32),
        }

        if len(done_ids) > 0:
            avg_successes = float(np.mean(self.episode_successes[done_ids]))
            if self.curriculum is not None:
                updated = self.curriculum.update(avg_successes, num_episodes=len(done_ids))
                infos["curriculum_updated"] = np.array([float(updated)])
                infos["curriculum_delta"] = np.full(
                    self.num_envs, self.delta, dtype=np.float32
                )
            # Reset finished envs
            self.reset(done_ids)

        return self._get_obs(), rewards.astype(np.float32), dones.astype(np.float32), infos

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> Dict:
        state = {
            "q": self.q,
            "q_dot": self.q_dot,
            "obj_pos": self.obj_pos,
            "obj_quat": self.obj_quat,
            "obj_linvel": self.obj_linvel,
            "obj_angvel": self.obj_angvel,
            "goal_quat": self.goal_quat,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "rng": self.rng.bit_generator.state,
        }
        if self.curriculum is not None:
            state["curriculum"] = self.curriculum.state_dict()
        return state

    def load_state_dict(self, state: Dict) -> None:
        for key in (
            "q",
            "q_dot",
            "obj_pos",
            "obj_quat",
            "obj_linvel",
            "obj_angvel",
            "goal_quat",
            "episode_lengths",
            "episode_successes",
        ):
            if key in state:
                setattr(self, key, np.asarray(state[key]))
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]
        if self.curriculum is not None and "curriculum" in state:
            self.curriculum.load_state_dict(state["curriculum"])

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


__all__ = [
    "AllegroHandEnv",
    "AllegroHandConfig",
    "OBS_DIM",
    "ACTION_DIM",
    "NUM_JOINTS",
]
