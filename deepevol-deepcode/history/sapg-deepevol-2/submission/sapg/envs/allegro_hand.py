"""AllegroHand in-hand reorientation environment (16-DoF).

This module implements a vectorized, CPU/GPU-friendly reference environment for
the AllegroHand in-hand cube reorientation task described in the SAPG paper
(Section 4 / Table 4).  The real paper uses IsaacGym for massively parallel
simulation; here we provide a lightweight kinematic proxy so that the full
SAPG/PPO algorithm stack can be exercised end-to-end without an NVIDIA GPU.

Task
----
A 16-DoF Allegro hand must reorient a cube held in the palm to match a target
orientation ``g_t`` (a unit quaternion in R^4).  The observation follows the
same layout used by the AllegroKuka env:

    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]

where ``q, q_dot`` are the 16 joint positions/velocities, ``x_t`` is the object
pose (position R^3 + quaternion R^4 = R^7), ``v_t, omega_t`` are the object
linear/angular velocities (R^6), ``g_t`` is the goal quaternion (R^4) and
``z_t`` is a scalar phase/curriculum signal.

Reward
------
    r = w1 * r_reach + r_orientation + r_success

where ``r_orientation`` is a shaped term based on the angular distance between
the current and goal orientation, and ``r_success`` is a sparse bonus granted
when the orientation error stays below the curriculum tolerance for
``success_steps`` consecutive steps.

Metric
------
The paper reports *net episode reward* for the easier in-hand tasks (as opposed
to successes-per-episode for the harder AllegroKuka tasks).  We expose both via
``info``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .curriculum import SuccessToleranceCurriculum, build_curriculum


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention)
# ---------------------------------------------------------------------------
def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-8, None)


def _quat_random(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniformly random unit quaternions, shape (n, 4)."""
    u = rng.random((n, 3))
    q = np.empty((n, 4), dtype=np.float64)
    q[:, 0] = np.sqrt(1.0 - u[:, 0]) * np.sin(2.0 * np.pi * u[:, 1])
    q[:, 1] = np.sqrt(1.0 - u[:, 0]) * np.cos(2.0 * np.pi * u[:, 1])
    q[:, 2] = np.sqrt(u[:, 0]) * np.sin(2.0 * np.pi * u[:, 2])
    q[:, 3] = np.sqrt(u[:, 0]) * np.cos(2.0 * np.pi * u[:, 2])
    return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z), broadcasting on last dim."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(a.shape[:-1] + (4,), dtype=a.dtype)
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular distance (radians) between two unit quaternions, shape (...)."""
    q1 = _quat_normalize(q1)
    q2 = _quat_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


class AllegroHandEnv:
    """Vectorized AllegroHand (16-DoF) in-hand cube reorientation environment."""

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    num_joints = 16
    hand_dof = 16
    obj_pose_dim = 7   # position (3) + quaternion (4)
    obj_vel_dim = 6    # linear (3) + angular (3)
    goal_dim = 4       # goal quaternion
    z_dim = 1

    def __init__(
        self,
        num_envs: int = 1,
        task: str = "reorientation",
        cfg: Optional[Any] = None,
        device: Optional[Any] = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.num_envs = int(num_envs)
        self.task = str(task).lower()
        self.cfg = cfg
        self.device = device
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        get = self._cfg_getter(cfg)

        # ---- observation / action dims -------------------------------
        self.obs_dim = (
            self.num_joints * 2          # q, q_dot
            + self.obj_pose_dim          # x_t
            + self.obj_vel_dim           # v_t, omega_t
            + self.goal_dim              # g_t
            + self.z_dim                 # z_t
        )
        self.action_dim = self.num_joints

        # ---- task parameters -----------------------------------------
        self.success_steps = int(get("success_steps", 30))
        self.episode_length = int(get("episode_length", 300))
        self.control_freq = float(get("control_freq", 20.0))
        self.dt = 1.0 / max(self.control_freq, 1e-6)

        # ---- reward weights ------------------------------------------
        self.w1 = float(get("reward_w1", 1.0))
        self.r_orientation_w = float(get("reward_orientation_w", 1.0))
        self.r_success_w = float(get("reward_success_w", 10.0))

        # ---- curriculum ----------------------------------------------
        self.curriculum: Optional[SuccessToleranceCurriculum] = build_curriculum(cfg)
        if self.curriculum is None:
            # Default tolerance for the in-hand task (radians).
            self.curriculum = SuccessToleranceCurriculum(
                initial_tolerance=float(get("curriculum_initial_tolerance", 0.5)),
                min_tolerance=float(get("curriculum_min_tolerance", 0.1)),
                decay=float(get("curriculum_decay", 0.9)),
                success_threshold=float(get("curriculum_success_threshold", 3.0)),
            )

        # ---- state buffers -------------------------------------------
        self.q = np.zeros((self.num_envs, self.num_joints), dtype=np.float64)
        self.q_dot = np.zeros((self.num_envs, self.num_joints), dtype=np.float64)
        self.obj_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_quat = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]), (self.num_envs, 1)
        ).astype(np.float64)
        self.obj_lin_vel = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_ang_vel = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.goal_quat = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]), (self.num_envs, 1)
        ).astype(np.float64)

        self.episode_step = np.zeros(self.num_envs, dtype=np.int64)
        self.success_counter = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_successes = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_reward = np.zeros(self.num_envs, dtype=np.float64)
        self._last_success = np.zeros(self.num_envs, dtype=bool)

        # Palm reference position (object is held near the palm).
        self.palm_pos = np.zeros((self.num_envs, 3), dtype=np.float64)

        self.reset()

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------
    @staticmethod
    def _cfg_getter(cfg: Optional[Any]):
        if cfg is None:
            return lambda key, default=None: default
        if isinstance(cfg, dict):
            return lambda key, default=None: cfg.get(key, default)

        def _get(key, default=None):
            return getattr(cfg, key, default)

        return _get

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if indices is None:
            idx = np.arange(self.num_envs)
        else:
            idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        n = idx.shape[0]
        if n == 0:
            return self._get_obs()

        # Random initial joint configuration near zero.
        self.q[idx] = self.rng.normal(0.0, 0.1, size=(n, self.num_joints))
        self.q_dot[idx] = 0.0

        # Object starts at the palm with a random orientation.
        self.palm_pos[idx] = self.rng.normal(0.0, 0.01, size=(n, 3))
        self.obj_pos[idx] = self.palm_pos[idx] + self.rng.normal(
            0.0, 0.005, size=(n, 3)
        )
        self.obj_quat[idx] = _quat_random(self.rng, n)
        self.obj_lin_vel[idx] = 0.0
        self.obj_ang_vel[idx] = 0.0

        # Goal orientation: a random rotation away from the initial one.
        self.goal_quat[idx] = _quat_random(self.rng, n)

        self.episode_step[idx] = 0
        self.success_counter[idx] = 0
        self.episode_successes[idx] = 0.0
        self.episode_reward[idx] = 0.0
        self._last_success[idx] = False

        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_envs, -1)
        actions = np.clip(actions, -1.0, 1.0)

        # ---- simplified hand/object dynamics -------------------------
        # Joint velocities track the commanded action with damping.
        self.q_dot = 0.8 * self.q_dot + 0.2 * actions
        self.q = self.q + self.dt * self.q_dot
        self.q = np.clip(self.q, -1.5, 1.5)

        # The object is held by the palm; finger motion induces a small
        # angular velocity that rotates the object toward the goal.
        finger_effect = np.mean(actions, axis=1, keepdims=True)  # (N, 1)
        # Error-driven rotation: rotate object toward goal.
        err_axis = self._orientation_error_axis()
        self.obj_ang_vel = 0.7 * self.obj_ang_vel + 0.3 * (
            err_axis + 0.1 * finger_effect
        )
        self.obj_ang_vel = np.clip(self.obj_ang_vel, -5.0, 5.0)

        # Integrate orientation: dq = 0.5 * omega_quat * q
        omega_quat = np.concatenate(
            [np.zeros((self.num_envs, 1)), self.obj_ang_vel], axis=1
        )
        dq = 0.5 * _quat_mul(omega_quat, self.obj_quat) * self.dt
        self.obj_quat = _quat_normalize(self.obj_quat + dq)

        # Object position stays near the palm (in-hand task).
        self.obj_pos = self.palm_pos + 0.5 * (self.obj_pos - self.palm_pos)
        self.obj_lin_vel = 0.5 * self.obj_lin_vel

        # ---- reward --------------------------------------------------
        reward = self._compute_reward()

        # ---- success detection ---------------------------------------
        success = self._check_success()
        self.success_counter = np.where(
            success, self.success_counter + 1, 0
        )
        newly_successful = self.success_counter >= self.success_steps
        self.episode_successes += newly_successful.astype(np.float64)
        reward = reward + self.r_success_w * newly_successful.astype(np.float64)
        self._last_success = newly_successful

        self.episode_reward += reward
        self.episode_step += 1

        # ---- termination ---------------------------------------------
        done = self.episode_step >= self.episode_length
        done = done | newly_successful

        info: Dict[str, Any] = {
            "successes": newly_successful.astype(np.float64),
            "episode_successes": self.episode_successes.copy(),
            "episode_reward": self.episode_reward.copy(),
            "tolerance": self.curriculum.get_tolerance(),
            "task": self.task,
        }

        # ---- curriculum update on episode end ------------------------
        if np.any(done):
            done_idx = np.where(done)[0]
            mean_succ = float(np.mean(self.episode_successes[done_idx]))
            self.curriculum.update(mean_succ)
            info["tolerance"] = self.curriculum.get_tolerance()
            self.reset(done_idx)

        obs = self._get_obs()
        return obs, reward, done.astype(np.float32), info

    # ------------------------------------------------------------------
    # Reward / success
    # ------------------------------------------------------------------
    def _orientation_error_axis(self) -> np.ndarray:
        """Rotation axis (in world frame) that reduces orientation error."""
        q_err = _quat_mul(self.goal_quat, _quat_conjugate(self.obj_quat))
        q_err = _quat_normalize(q_err)
        # Ensure shortest path (w >= 0).
        sign = np.where(q_err[:, 0:1] < 0, -1.0, 1.0)
        q_err = q_err * sign
        axis = q_err[:, 1:]
        angle = 2.0 * np.arccos(np.clip(q_err[:, 0], -1.0, 1.0))
        return axis * angle[:, None]

    def _compute_reward(self) -> np.ndarray:
        # Reach term: keep the object close to the palm.
        reach_dist = np.linalg.norm(self.obj_pos - self.palm_pos, axis=1)
        r_reach = np.exp(-10.0 * reach_dist)

        # Orientation term: shaped by angular distance to goal.
        ang_dist = _quat_angular_distance(self.obj_quat, self.goal_quat)
        r_orientation = np.exp(-2.0 * ang_dist)

        reward = self.w1 * r_reach + self.r_orientation_w * r_orientation
        return reward.astype(np.float64)

    def _check_success(self) -> np.ndarray:
        ang_dist = _quat_angular_distance(self.obj_quat, self.goal_quat)
        return ang_dist < self.curriculum.get_tolerance()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        z = (self.episode_step.astype(np.float64) / max(self.episode_length, 1))[
            :, None
        ]
        obs = np.concatenate(
            [
                self.q,               # (N, 16)
                self.q_dot,           # (N, 16)
                self.obj_pos,         # (N, 3)
                self.obj_quat,        # (N, 4)
                self.obj_lin_vel,     # (N, 3)
                self.obj_ang_vel,     # (N, 3)
                self.goal_quat,       # (N, 4)
                z,                    # (N, 1)
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # State (de)serialization
    # ------------------------------------------------------------------
    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "curriculum": self.curriculum.state_dict(),
            "episode_step": self.episode_step.copy(),
            "episode_successes": self.episode_successes.copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "curriculum" in state and state["curriculum"] is not None:
            self.curriculum.load_state_dict(state["curriculum"])
        if "episode_step" in state:
            self.episode_step = np.asarray(state["episode_step"], dtype=np.int64)
        if "episode_successes" in state:
            self.episode_successes = np.asarray(
                state["episode_successes"], dtype=np.float64
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.num_envs


__all__ = ["AllegroHandEnv"]
