"""AllegroKuka environment for SAPG.

Implements the Allegro 16-DoF hand + Kuka 7-DoF arm (23 joints) manipulation
tasks from the SAPG paper:
  - Regrasping: goal position g_t in R^3, success tolerance curriculum
    7.5cm -> 1cm, -10% when avg successes/episode > 3.
  - Throw: throw object into a bucket goal.
  - Reorientation: reorient object to goal pose g_t in R^7.

Observation o_t = [q, q_dot (R^23), x_t (R^7 pose), v_t, omega_t, g_t, z_t].

Reward = w1*r_reach + r_lift + r_target + r_success.
Success metric = number of successes per episode.

This is a CPU/GPU-vectorized reference implementation (numpy/torch) that
simulates the task dynamics in a simplified manner so the algorithm code can
be exercised end-to-end without IsaacGym. It exposes the same interface as the
GPU-parallel environments used in the paper.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .curriculum import SuccessToleranceCurriculum, build_curriculum


class AllegroKukaEnv:
    """Vectorized AllegroKuka manipulation environment.

    Simplified physics: the object is a point mass whose position/velocity is
    directly controlled by the hand/arm joints (a kinematic proxy). This lets
    the RL algorithm be trained end-to-end while matching the observation and
    reward structure of the paper.
    """

    def __init__(
        self,
        num_envs: int = 1,
        task: str = "regrasping",
        cfg: Optional[Any] = None,
        device: Optional[str] = None,
        seed: int = 0,
        **kwargs,
    ):
        self.num_envs = int(num_envs)
        self.task = task
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # --- Config helpers ---
        def _get(key, default):
            if cfg is None:
                return default
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        self.control_freq = _get("control_freq", 20)
        self.episode_length = _get("episode_length", 300)
        self.dt = 1.0 / self.control_freq

        # Joints: 16 Allegro hand + 7 Kuka arm = 23
        self.num_joints = 23
        self.hand_dof = 16
        self.arm_dof = 7

        # Object pose (position + quaternion) = 7 dims
        self.obj_pose_dim = 7
        # Object linear + angular velocity = 6 dims
        self.obj_vel_dim = 6

        # Goal dims depend on task
        if task == "reorientation":
            self.goal_dim = 7  # goal pose (quaternion)
        else:
            self.goal_dim = 3  # goal position

        # Extra scalar z_t (e.g., object height / grasp state)
        self.z_dim = 1

        # Observation dim: q(23) + q_dot(23) + x_t(7) + v_t(3) + omega_t(3) + g_t + z_t
        self.obs_dim = (
            self.num_joints * 2
            + self.obj_pose_dim
            + self.obj_vel_dim
            + self.goal_dim
            + self.z_dim
        )

        # Action dim: 23 joints
        self.action_dim = self.num_joints

        # Reward weights (defaults; paper leaves exact values unspecified)
        self.w1 = _get("reward_w1", 1.0)
        self.r_lift_w = _get("reward_lift_w", 1.0)
        self.r_target_w = _get("reward_target_w", 1.0)
        self.r_success_w = _get("reward_success_w", 1.0)

        # Success tolerance curriculum
        self.use_curriculum = _get("use_curriculum", True)
        self.curriculum = build_curriculum(cfg)
        self.tolerance = (
            self.curriculum.get_tolerance()
            if self.curriculum is not None
            else _get("curriculum_initial_tolerance", 0.075)
        )

        # Success steps required (K=30 for regrasping)
        self.success_steps = _get("success_steps", 30)

        # --- State buffers ---
        self.reset()

    # ------------------------------------------------------------------ #
    # Reset / step
    # ------------------------------------------------------------------ #
    def reset(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environment(s) and return observations."""
        if indices is None:
            n = self.num_envs
            self.q = self.rng.uniform(-0.1, 0.1, size=(n, self.num_joints))
            self.q_dot = np.zeros((n, self.num_joints))
            self.obj_pos = np.zeros((n, 3))
            self.obj_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
            self.obj_lin_vel = np.zeros((n, 3))
            self.obj_ang_vel = np.zeros((n, 3))
            self.z = np.zeros((n, 1))
            self.step_count = np.zeros(n, dtype=int)
            self.successes = np.zeros(n, dtype=int)
            self.success_streak = np.zeros(n, dtype=int)
            self.episode_reward = np.zeros(n)
            self.episode_successes = np.zeros(n, dtype=int)
            self._set_goals()
        else:
            idx = np.asarray(indices)
            self.q[idx] = self.rng.uniform(-0.1, 0.1, size=(len(idx), self.num_joints))
            self.q_dot[idx] = 0.0
            self.obj_pos[idx] = 0.0
            self.obj_quat[idx] = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(idx), 1))
            self.obj_lin_vel[idx] = 0.0
            self.obj_ang_vel[idx] = 0.0
            self.z[idx] = 0.0
            self.step_count[idx] = 0
            self.successes[idx] = 0
            self.success_streak[idx] = 0
            self.episode_reward[idx] = 0.0
            self.episode_successes[idx] = 0
            self._set_goals(indices=idx)

        return self._get_obs()

    def _set_goals(self, indices: Optional[np.ndarray] = None) -> None:
        n = self.num_envs if indices is None else len(indices)
        if self.task == "reorientation":
            # Random goal quaternion
            goal_quat = self._random_quat(n)
            self.goal = goal_quat
        else:
            # Goal position in a reachable region
            goal_pos = self.rng.uniform(-0.2, 0.2, size=(n, 3))
            goal_pos[:, 2] = self.rng.uniform(0.1, 0.4, size=n)
            self.goal = goal_pos

    def _random_quat(self, n: int) -> np.ndarray:
        """Uniform random unit quaternions (w,x,y,z)."""
        u1 = self.rng.uniform(0.0, 1.0, size=n)
        u2 = self.rng.uniform(0.0, 2.0 * np.pi, size=n)
        u3 = self.rng.uniform(0.0, 2.0 * np.pi, size=n)
        a = np.sqrt(1.0 - u1)
        b = np.sqrt(u1)
        quat = np.stack(
            [
                a * np.sin(u2),
                a * np.cos(u2),
                b * np.sin(u3),
                b * np.cos(u3),
            ],
            axis=-1,
        )
        return quat

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Apply actions, advance simulation, return (obs, reward, done, info)."""
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self.action_dim)
        actions = np.clip(actions, -1.0, 1.0)

        # Simple kinematic integration: joints move toward commanded action
        joint_scale = 0.5  # max joint velocity per step
        self.q_dot = joint_scale * actions
        self.q = self.q + self.q_dot * self.dt
        self.q = np.clip(self.q, -1.5, 1.5)

        # Object dynamics: object follows hand palm position (proxy for grasp)
        palm_pos = self._palm_position()
        palm_vel = (palm_pos - self.obj_pos) / max(self.dt, 1e-6)
        # Object tracks palm with some lag (grasp stiffness)
        grasp_stiffness = 5.0
        self.obj_lin_vel = grasp_stiffness * (palm_pos - self.obj_pos) + 0.1 * palm_vel
        self.obj_pos = self.obj_pos + self.obj_lin_vel * self.dt

        # Angular velocity from hand orientation change (proxy)
        self.obj_ang_vel = 0.5 * actions[:, :3]
        # Simple quaternion integration
        dq = 0.5 * self.dt * self.obj_ang_vel
        self.obj_quat = self._integrate_quat(self.obj_quat, dq)

        # z_t: object height (proxy for lift state)
        self.z = self.obj_pos[:, 2:3]

        # Compute reward
        reward = self._compute_reward()

        # Success detection
        success = self._check_success()
        self.successes += success.astype(int)
        self.success_streak = np.where(success, self.success_streak + 1, 0)
        self.episode_successes += success.astype(int)

        # Step count / done
        self.step_count += 1
        done = (self.step_count >= self.episode_length).astype(np.float32)

        # Curriculum update on episode end
        if self.curriculum is not None and done.any():
            mean_successes = float(self.episode_successes[done > 0].mean()) if (done > 0).any() else 0.0
            self.tolerance = self.curriculum.update(mean_successes)

        # Reset done envs
        reset_idx = np.where(done > 0)[0]
        if len(reset_idx) > 0:
            self.episode_reward[reset_idx] = 0.0
            self.episode_successes[reset_idx] = 0
            self.reset(indices=reset_idx)

        obs = self._get_obs()
        info = {
            "successes": self.successes.copy(),
            "episode_successes": self.episode_successes.copy(),
            "tolerance": self.tolerance,
            "task": self.task,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    # Reward / success
    # ------------------------------------------------------------------ #
    def _palm_position(self) -> np.ndarray:
        """Approximate palm position from arm joints (proxy)."""
        # Use first 3 arm joints as x,y,z offset from base
        base = np.array([0.0, 0.0, 0.5])
        palm = base + 0.3 * self.q[:, :3]
        return palm

    def _compute_reward(self) -> np.ndarray:
        """Compute per-env reward vector."""
        palm = self._palm_position()
        obj = self.obj_pos

        # Reach reward: distance from palm to object
        reach_dist = np.linalg.norm(palm - obj, axis=-1)
        r_reach = -reach_dist

        # Lift reward: object height above threshold
        lift_thresh = 0.2
        r_lift = np.clip(obj[:, 2] - lift_thresh, 0.0, None)

        # Target reward: distance from object to goal
        if self.task == "reorientation":
            # Quaternion distance
            target_dist = self._quat_distance(self.obj_quat, self.goal)
            r_target = -target_dist
        else:
            target_dist = np.linalg.norm(obj - self.goal, axis=-1)
            r_target = -target_dist

        # Success bonus
        success = self._check_success()
        r_success = success.astype(np.float32) * 10.0

        reward = (
            self.w1 * r_reach
            + self.r_lift_w * r_lift
            + self.r_target_w * r_target
            + self.r_success_w * r_success
        )
        self.episode_reward += reward
        return reward

    def _check_success(self) -> np.ndarray:
        """Return boolean array indicating which envs achieved success this step."""
        if self.task == "reorientation":
            dist = self._quat_distance(self.obj_quat, self.goal)
        else:
            dist = np.linalg.norm(self.obj_pos - self.goal, axis=-1)
        return (dist < self.tolerance).astype(bool)

    def _quat_distance(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Angular distance between two quaternions (batch)."""
        # q1, q2: (N,4) w,x,y,z
        dot = np.abs(np.sum(q1 * q2, axis=-1))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def _integrate_quat(self, quat: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Integrate quaternion by small rotation vector dq (N,3)."""
        # dq is angular velocity * dt
        angle = np.linalg.norm(dq, axis=-1, keepdims=True)
        axis = dq / (angle + 1e-8)
        # Rotation quaternion
        half = angle / 2.0
        rot = np.concatenate(
            [np.cos(half), axis * np.sin(half)], axis=-1
        )  # (N,4) w,x,y,z
        # Quaternion multiply: q_new = rot * q
        w1, x1, y1, z1 = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
        w2, x2, y2, z2 = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        out = np.stack([w, x, y, z], axis=-1)
        # Normalize
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        return out / (norm + 1e-8)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        """Assemble observation vector per paper spec."""
        obs = np.concatenate(
            [
                self.q,                      # (N,23)
                self.q_dot,                  # (N,23)
                self.obj_pos,                # (N,3)
                self.obj_quat,               # (N,4)
                self.obj_lin_vel,            # (N,3)
                self.obj_ang_vel,            # (N,3)
                self.goal,                   # (N,goal_dim)
                self.z,                      # (N,1)
            ],
            axis=-1,
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "curriculum": self.curriculum.state_dict() if self.curriculum else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "tolerance" in state:
            self.tolerance = state["tolerance"]
        if state.get("curriculum") and self.curriculum:
            self.curriculum.load_state_dict(state["curriculum"])

    def __repr__(self) -> str:
        return (
            f"AllegroKukaEnv(task={self.task}, num_envs={self.num_envs}, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim})"
        )
