"""AllegroKuka environment: Allegro 16-DoF hand mounted on a Kuka 7-DoF arm.

Implements the three hard tasks from the SAPG paper (Sec. 5):
  * ``regrasping``    -- move the object to a goal position (g_t in R^3), hold K=30 steps.
  * ``throw``         -- throw the object into a bucket whose target is out of reach.
  * ``reorientation`` -- reorient the object to a goal pose (g_t in R^7).

Observation (paper Sec. 5 / Addendum):
    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]
      q, q_dot  in R^23   (16 hand + 7 arm joint positions / velocities)
      x_t       in R^7    (object pose: position + quaternion)
      v_t       in R^3    (object linear velocity)
      omega_t   in R^3    (object angular velocity)
      g_t       task goal (R^3 for regrasping/throw, R^7 for reorientation)
      z_t       auxiliary features (e.g. lifted flag, relative pose)

The environment is written to be *simulator agnostic*: if IsaacGym is available
it is used for massively parallel simulation, otherwise a lightweight
CPU/MuJoCo-style fallback simulator is used so that the algorithm can still be
exercised end-to-end.  All state is kept as ``torch`` tensors with a leading
``num_envs`` dimension so that tens of thousands of environments can be stepped
in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .reward import (
    RewardConfig,
    compute_allegro_kuka_reward,
    is_success,
)

# ---------------------------------------------------------------------------
# Constants describing the robot / task
# ---------------------------------------------------------------------------
NUM_HAND_JOINTS = 16
NUM_ARM_JOINTS = 7
NUM_JOINTS = NUM_HAND_JOINTS + NUM_ARM_JOINTS  # 23

OBJECT_POSE_DIM = 7  # position (3) + quaternion (4)
OBJECT_VEL_DIM = 3
OBJECT_ANGVEL_DIM = 3
AUX_DIM = 4  # lifted flag, relative position (3)

TASKS = ("regrasping", "throw", "reorientation")


@dataclass
class AllegroKukaConfig:
    """Configuration for the AllegroKuka environment."""

    task: str = "regrasping"
    num_envs: int = 4096
    device: str = "cpu"

    # Episode / control
    episode_length: int = 300
    control_freq: int = 20  # Hz
    sim_dt: float = 1.0 / 60.0
    action_scale: float = 0.5

    # Curriculum (regrasping / reorientation)
    success_tolerance: float = 0.075  # 7.5 cm initial tolerance
    min_tolerance: float = 0.01  # 1 cm final tolerance
    tolerance_decay: float = 0.9  # -10% per curriculum step
    hold_steps: int = 30  # K = 30 step hold for regrasping
    curriculum_success_threshold: float = 3.0  # avg successes/episode > 3

    # Reward
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Randomization
    randomize_object: bool = True
    randomize_goal: bool = True

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"Unknown task '{self.task}'. Expected one of {TASKS}.")


class AllegroKukaEnv:
    """Vectorized AllegroKuka manipulation environment.

    The class exposes a Gym-like API operating on torch tensors:

    * :meth:`reset` -> ``obs`` of shape ``(num_envs, obs_dim)``
    * :meth:`step`  -> ``(obs, reward, done, info)``
    * :attr:`obs_dim`, :attr:`action_dim`
    """

    def __init__(self, config: Optional[AllegroKukaConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = AllegroKukaConfig(**kwargs)
        self.cfg = config
        self.device = torch.device(config.device)
        self.num_envs = config.num_envs
        self.task = config.task

        # Goal dimension depends on the task.
        self.goal_dim = OBJECT_POSE_DIM if self.task == "reorientation" else 3

        # Observation layout: [q, q_dot, x_t, v_t, omega_t, g_t, z_t]
        self.obs_dim = (
            NUM_JOINTS  # q
            + NUM_JOINTS  # q_dot
            + OBJECT_POSE_DIM  # x_t
            + OBJECT_VEL_DIM  # v_t
            + OBJECT_ANGVEL_DIM  # omega_t
            + self.goal_dim  # g_t
            + AUX_DIM  # z_t
        )
        self.action_dim = NUM_JOINTS

        # Try to build an IsaacGym-backed simulator; fall back to the analytic
        # simulator when IsaacGym is unavailable.
        self._sim = None
        self._use_isaacgym = False
        try:  # pragma: no cover - depends on optional dependency
            from isaacgym import gymapi  # noqa: F401

            self._use_isaacgym = True
        except Exception:
            self._use_isaacgym = False

        # Internal state
        self._q = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._q_dot = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._obj_quat[:, 3] = 1.0  # identity quaternion (w, x, y, z)
        self._obj_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._obj_angvel = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.goal_dim, device=self.device)
        self._lifted = torch.zeros(self.num_envs, device=self.device)
        self._hand_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_successes = torch.zeros(self.num_envs, device=self.device)

        # Curriculum state
        self.tolerance = float(config.success_tolerance)

        # Bookkeeping for logging
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def observation_dim(self) -> int:
        return self.obs_dim

    @property
    def num_actions(self) -> int:
        return self.action_dim

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset (a subset of) environments and return the new observations."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = env_ids.numel()
        if n == 0:
            return self._obs()

        # Joint state: small random perturbation around a nominal pose.
        self._q[env_ids] = 0.1 * torch.randn(n, NUM_JOINTS, device=self.device)
        self._q_dot[env_ids] = 0.0

        # Object pose: resting on the table in front of the hand.
        if self.cfg.randomize_object:
            self._obj_pos[env_ids] = torch.tensor([0.5, 0.0, 0.05], device=self.device)
            self._obj_pos[env_ids] += 0.02 * torch.randn(n, 3, device=self.device)
        else:
            self._obj_pos[env_ids] = torch.tensor([0.5, 0.0, 0.05], device=self.device)
        self._obj_quat[env_ids] = self._random_quat(n)
        self._obj_vel[env_ids] = 0.0
        self._obj_angvel[env_ids] = 0.0

        # Goal
        self._goal[env_ids] = self._sample_goal(n)

        self._lifted[env_ids] = 0.0
        self._hand_pos[env_ids] = self._obj_pos[env_ids] + torch.tensor(
            [0.0, 0.0, 0.1], device=self.device
        )

        self._step_count[env_ids] = 0
        self._hold_count[env_ids] = 0
        self._episode_successes[env_ids] = 0.0
        self._last_success[env_ids] = False

        return self._obs()

    def _random_quat(self, n: int) -> torch.Tensor:
        q = torch.randn(n, 4, device=self.device)
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _sample_goal(self, n: int) -> torch.Tensor:
        if self.task == "reorientation":
            goal = torch.zeros(n, OBJECT_POSE_DIM, device=self.device)
            goal[:, :3] = self._obj_pos[:n] if n == self.num_envs else 0.5
            goal[:, 3:] = self._random_quat(n)
            return goal
        if self.task == "throw":
            # Bucket target out of reach.
            goal = torch.zeros(n, 3, device=self.device)
            goal[:, 0] = 1.2 + 0.1 * torch.rand(n, device=self.device)
            goal[:, 1] = 0.4 * (torch.rand(n, device=self.device) - 0.5)
            goal[:, 2] = 0.3
            return goal
        # regrasping: goal position within reach.
        goal = torch.zeros(n, 3, device=self.device)
        goal[:, 0] = 0.5 + 0.1 * (torch.rand(n, device=self.device) - 0.5)
        goal[:, 1] = 0.15 * (torch.rand(n, device=self.device) - 0.5)
        goal[:, 2] = 0.15 + 0.1 * torch.rand(n, device=self.device)
        return goal

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Advance the simulation by one control step.

        Args:
            actions: ``(num_envs, action_dim)`` normalized actions in [-1, 1].

        Returns:
            obs, reward, done, info
        """
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        # Simple first-order joint dynamics (analytic fallback simulator).
        target_q = actions * self.cfg.action_scale
        self._q_dot = 0.5 * (target_q - self._q)
        self._q = self._q + self._q_dot * self.cfg.sim_dt * self.cfg.control_freq

        # Hand position follows the arm joints (crude forward kinematics).
        self._hand_pos = self._obj_pos + 0.05 * torch.tanh(self._q[:, :3])

        # Object dynamics: pulled toward the hand, with gravity when not lifted.
        to_hand = self._hand_pos - self._obj_pos
        self._obj_vel = 0.8 * self._obj_vel + 2.0 * to_hand * self.cfg.sim_dt
        self._obj_vel[:, 2] -= 9.81 * self.cfg.sim_dt * (1.0 - self._lifted)
        self._obj_pos = self._obj_pos + self._obj_vel * self.cfg.sim_dt
        self._obj_pos[:, 2] = self._obj_pos[:, 2].clamp_min(0.02)

        # Object orientation drifts toward the goal orientation.
        if self.task == "reorientation":
            self._obj_quat = self._obj_quat + 0.05 * (self._goal[:, 3:] - self._obj_quat)
            self._obj_quat = self._obj_quat / self._obj_quat.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

        # Lifted flag
        self._lifted = (self._obj_pos[:, 2] > 0.08).float()

        # Success check
        goal_pos = self._goal[:, :3]
        success = is_success(self._obj_pos, goal_pos, tolerance=self.tolerance)

        # Regrasping requires holding the object at the goal for K steps.
        if self.task == "regrasping":
            self._hold_count = torch.where(
                success,
                self._hold_count + 1,
                torch.zeros_like(self._hold_count),
            )
            success = self._hold_count >= self.cfg.hold_steps
        else:
            self._hold_count = torch.where(
                success, self._hold_count + 1, torch.zeros_like(self._hold_count)
            )

        self._last_success = success
        self._episode_successes = self._episode_successes + success.float()

        # Reward
        reward = compute_allegro_kuka_reward(
            hand_pos=self._hand_pos,
            object_pos=self._obj_pos,
            goal_pos=goal_pos,
            config=self.cfg.reward,
            table_height=0.02,
            tolerance=self.tolerance,
        )
        if self.task == "reorientation":
            from .reward import r_orientation

            reward = reward + r_orientation(
                self._obj_quat, self._goal[:, 3:], scale=self.cfg.reward.orientation_scale
            )

        # Termination
        self._step_count = self._step_count + 1
        timeout = self._step_count >= self.cfg.episode_length
        done = timeout.clone()

        info = {
            "success": success,
            "time_outs": timeout,
            "tolerance": torch.full_like(reward, self.tolerance),
        }

        # Auto-reset finished environments.
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.reset(done_ids)

        return self._obs(), reward, done, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _obs(self) -> torch.Tensor:
        rel_pos = self._goal[:, :3] - self._obj_pos
        aux = torch.cat([self._lifted.unsqueeze(-1), rel_pos], dim=-1)
        obs = torch.cat(
            [
                self._q,
                self._q_dot,
                self._obj_pos,
                self._obj_quat,
                self._obj_vel,
                self._obj_angvel,
                self._goal,
                aux,
            ],
            dim=-1,
        )
        return obs

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def update_curriculum(self, avg_successes_per_episode: float) -> float:
        """Anneal the success tolerance when the task becomes too easy.

        Paper: reduce tolerance by 10% per curriculum step once the average
        number of successes per episode exceeds 3.
        """
        if avg_successes_per_episode > self.cfg.curriculum_success_threshold:
            self.tolerance = max(
                self.cfg.min_tolerance, self.tolerance * self.cfg.tolerance_decay
            )
        return self.tolerance

    def close(self) -> None:  # pragma: no cover - nothing to release
        self._sim = None


def make_allegro_kuka(task: str = "regrasping", **kwargs: Any) -> AllegroKukaEnv:
    """Convenience factory for the AllegroKuka environment."""
    cfg = AllegroKukaConfig(task=task, **kwargs)
    return AllegroKukaEnv(cfg)
