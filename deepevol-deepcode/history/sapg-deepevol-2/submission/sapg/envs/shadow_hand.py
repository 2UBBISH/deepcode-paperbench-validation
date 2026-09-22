"""ShadowHand in-hand cube reorientation environment (24-DoF).

This module implements a vectorized, CPU/GPU-friendly reference environment for
the ShadowHand (24-DoF) in-hand cube reorientation task described in the SAPG
paper (Table 3).  It provides a lightweight kinematic proxy for IsaacGym so the
full SAPG/PPO stack can be exercised end-to-end without an NVIDIA GPU.

Observation layout (matching the paper's convention used across envs)::

    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]

where

    q       : joint positions            (24,)
    q_dot   : joint velocities           (24,)
    x_t     : object pose (pos + quat)   (7,)
    v_t     : object linear velocity     (3,)
    omega_t : object angular velocity    (3,)
    g_t     : goal orientation quaternion(4,)
    z_t     : phase / time feature       (1,)

Reward::

    r = w1 * r_reach + r_orientation + r_success

Success requires the orientation error to stay below the curriculum tolerance
for ``success_steps`` consecutive steps.  The reported metric for this task is
the net episode reward.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .curriculum import SuccessToleranceCurriculum, build_curriculum

__all__ = ["ShadowHandEnv"]


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention)
# ---------------------------------------------------------------------------
def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions of shape (..., 4)."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-8, None)


def _quat_random(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample ``n`` uniformly random unit quaternions, shape (n, 4)."""
    u = rng.random((n, 3))
    q = np.empty((n, 4), dtype=np.float64)
    q[:, 0] = np.sqrt(1.0 - u[:, 0]) * np.sin(2.0 * np.pi * u[:, 1])
    q[:, 1] = np.sqrt(1.0 - u[:, 0]) * np.cos(2.0 * np.pi * u[:, 1])
    q[:, 2] = np.sqrt(u[:, 0]) * np.sin(2.0 * np.pi * u[:, 2])
    q[:, 3] = np.sqrt(u[:, 0]) * np.cos(2.0 * np.pi * u[:, 2])
    return _quat_normalize(q)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions with broadcasting, shape (..., 4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(a.shape[:-1] + (4,), dtype=np.float64)
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate, shape (..., 4)."""
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular distance (radians) between two batches of quaternions."""
    q1 = _quat_normalize(q1)
    q2 = _quat_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _cfg_getter(cfg):
    """Return a ``get(key, default)`` callable for dict or attribute configs."""
    if cfg is None:
        return lambda key, default=None: default
    if isinstance(cfg, dict):
        return lambda key, default=None: cfg.get(key, default)

    def _get(key, default=None):
        return getattr(cfg, key, default)

    return _get


class ShadowHandEnv:
    """Vectorized ShadowHand (24-DoF) in-hand cube reorientation environment."""

    num_joints = 24
    hand_dof = 24
    obj_pose_dim = 7
    obj_vel_dim = 6
    goal_dim = 4
    z_dim = 1

    def __init__(
        self,
        num_envs: int = 1,
        task: str = "reorientation",
        cfg: Any = None,
        device: Any = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.num_envs = int(num_envs)
        self.task = task
        self.cfg = cfg
        self.device = device
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        get = _cfg_getter(cfg)

        # Dimensions -------------------------------------------------------
        self.obs_dim = (
            self.num_joints          # q
            + self.num_joints        # q_dot
            + self.obj_pose_dim      # x_t
            + 3                      # v_t
            + 3                      # omega_t
            + self.goal_dim          # g_t
            + self.z_dim             # z_t
        )
        self.action_dim = self.num_joints

        # Episode / success parameters ------------------------------------
        self.episode_length = int(get("episode_length", 300))
        self.success_steps = int(get("success_steps", 30))
        self.control_freq = float(get("control_freq", 20))

        # Reward weights ---------------------------------------------------
        self.w1 = float(get("reward_reach_weight", 1.0))
        self.r_orientation_w = float(get("reward_orientation_weight", 1.0))
        self.r_success_w = float(get("reward_success_weight", 10.0))

        # Curriculum -------------------------------------------------------
        self.curriculum: Optional[SuccessToleranceCurriculum] = build_curriculum(cfg)
        if self.curriculum is None:
            self.curriculum = SuccessToleranceCurriculum(
                initial_tolerance=float(get("curriculum_initial_tolerance", 0.5)),
                min_tolerance=float(get("curriculum_min_tolerance", 0.1)),
                decay=float(get("curriculum_decay", 0.9)),
                success_threshold=float(get("curriculum_success_threshold", 3.0)),
            )

        # State buffers ----------------------------------------------------
        self.q = np.zeros((self.num_envs, self.num_joints), dtype=np.float64)
        self.q_dot = np.zeros((self.num_envs, self.num_joints), dtype=np.float64)
        self.obj_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_quat = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.obj_vel = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_omega = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.goal_quat = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.phase = np.zeros((self.num_envs, 1), dtype=np.float64)

        self.episode_step = np.zeros(self.num_envs, dtype=np.int64)
        self.success_counter = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_successes = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_reward = np.zeros(self.num_envs, dtype=np.float64)

        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset all (or selected) environments and return observations."""
        if indices is None:
            idx = np.arange(self.num_envs)
        else:
            idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        n = idx.shape[0]
        if n == 0:
            return self._get_obs()

        # Joint state: small random perturbation around zero.
        self.q[idx] = self.rng.normal(0.0, 0.1, size=(n, self.num_joints))
        self.q_dot[idx] = 0.0

        # Object starts near the palm with a random orientation.
        self.obj_pos[idx] = self.rng.normal(0.0, 0.01, size=(n, 3))
        self.obj_quat[idx] = _quat_random(self.rng, n)
        self.obj_vel[idx] = 0.0
        self.obj_omega[idx] = 0.0

        # Goal orientation.
        self.goal_quat[idx] = _quat_random(self.rng, n)

        self.phase[idx] = 0.0
        self.episode_step[idx] = 0
        self.success_counter[idx] = 0
        self.episode_successes[idx] = 0.0
        self.episode_reward[idx] = 0.0

        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Advance the environment by one control step."""
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_envs, -1)
        actions = np.clip(actions, -1.0, 1.0)

        # Simplified kinematic dynamics: actions drive joint velocities and
        # the object orientation is nudged toward the goal.
        self.q_dot = 0.5 * actions
        self.q = self.q + self.q_dot / max(self.control_freq, 1.0)

        # Object orientation error drives a proportional correction.
        err = _quat_angular_distance(self.obj_quat, self.goal_quat)  # (N,)
        # Convert the error into a small rotation toward the goal.
        step_scale = 0.05
        # Blend object quaternion toward goal quaternion.
        blend = np.clip(step_scale * (1.0 + err[:, None]), 0.0, 0.5)
        self.obj_quat = _quat_normalize((1.0 - blend) * self.obj_quat + blend * self.goal_quat)
        self.obj_omega = self.obj_omega * 0.9
        self.obj_vel = self.obj_vel * 0.9

        self.phase = np.clip(self.phase + 1.0 / max(self.episode_length, 1), 0.0, 1.0)
        self.episode_step += 1

        reward = self._compute_reward()
        self.episode_reward += reward

        success = self._check_success()
        self.success_counter = np.where(success, self.success_counter + 1, 0)
        newly_successful = self.success_counter >= self.success_steps
        self.episode_successes += newly_successful.astype(np.float64)

        done = self.episode_step >= self.episode_length

        info: Dict[str, Any] = {
            "successes": newly_successful.astype(np.float64),
            "episode_successes": self.episode_successes.copy(),
            "episode_reward": self.episode_reward.copy(),
            "tolerance": self.curriculum.get_tolerance(),
            "task": self.task,
        }

        if np.any(done):
            done_idx = np.where(done)[0]
            mean_successes = float(np.mean(self.episode_successes[done_idx]))
            self.curriculum.update(mean_successes)
            self.reset(done_idx)

        return self._get_obs(), reward, done.astype(np.float64), info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_reward(self) -> np.ndarray:
        """Reward = w1 * r_reach + r_orientation + r_success."""
        err = _quat_angular_distance(self.obj_quat, self.goal_quat)  # (N,)
        r_reach = -np.sum(self.q ** 2, axis=-1) * 0.01
        r_orientation = -err
        r_success = self._check_success().astype(np.float64) * self.r_success_w
        reward = self.w1 * r_reach + self.r_orientation_w * r_orientation + r_success
        return reward

    def _check_success(self) -> np.ndarray:
        """Boolean success mask based on orientation tolerance."""
        err = _quat_angular_distance(self.obj_quat, self.goal_quat)
        return err < self.curriculum.get_tolerance()

    def _get_obs(self) -> np.ndarray:
        """Assemble the observation vector."""
        return np.concatenate(
            [
                self.q,
                self.q_dot,
                np.concatenate([self.obj_pos, self.obj_quat], axis=-1),
                self.obj_vel,
                self.obj_omega,
                self.goal_quat,
                self.phase,
            ],
            axis=-1,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection / serialization
    # ------------------------------------------------------------------
    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def get_state_dict(self) -> Dict[str, Any]:
        return {"curriculum": self.curriculum.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state and "curriculum" in state:
            self.curriculum.load_state_dict(state["curriculum"])

    def __len__(self) -> int:
        return self.num_envs
