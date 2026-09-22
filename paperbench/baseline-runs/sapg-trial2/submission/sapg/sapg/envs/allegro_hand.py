"""AllegroHand environment: 16-DoF in-hand cube reorientation.

This module implements a vectorized (torch tensor) Gym-like environment for the
AllegroHand 16-DoF in-hand cube reorientation task described in the SAPG paper
(Section 5 / Table 4). The task is an "easy" task in the paper's taxonomy: the
metric of interest is the *net episode reward* (not success rate), and the
baseline comparison is against PQL (Li et al., 2023).

Design notes
------------
* Simulator-agnostic: if IsaacGym is importable we set a flag and would use it;
  otherwise we fall back to a lightweight analytic first-order simulator so the
  code runs on CPU-only machines. The observation/reward/curriculum interfaces
  are identical regardless of backend.
* Fully vectorized over a leading ``num_envs`` dimension so it can be driven by
  tens of thousands of parallel environments (the SAPG scaling regime).
* Observation layout (mirrors ``shadow_hand.py``)::

      o_t = [ q, q_dot, x_t, q_obj, v_t, omega_t, g_t, z_t ]

  where ``q, q_dot in R^16`` (AllegroHand joints), ``x_t in R^3`` object
  position, ``q_obj in R^4`` object quaternion, ``v_t, omega_t in R^3`` object
  linear/angular velocity, ``g_t in R^4`` goal quaternion, and ``z_t in R^2``
  auxiliary features ``[success_flag, orientation_error]``.

* Reward: dense orientation error plus a sparse success bonus, via
  :func:`sapg.envs.reward.compute_reorientation_reward`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .reward import RewardConfig, compute_reorientation_reward, r_orientation

try:  # pragma: no cover - optional heavy dependency
    import isaacgym  # type: ignore  # noqa: F401

    _use_isaacgym = True
except Exception:  # pragma: no cover
    _use_isaacgym = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_JOINTS = 16
OBJECT_POS_DIM = 3
OBJECT_QUAT_DIM = 4
OBJECT_VEL_DIM = 3
OBJECT_ANGVEL_DIM = 3
AUX_DIM = 2
TASKS = ("reorientation",)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AllegroHandConfig:
    """Hyperparameters for the AllegroHand reorientation environment."""

    task: str = "reorientation"
    num_envs: int = 4096
    device: str = "cpu"

    # Episode / control
    episode_length: int = 100
    control_freq: int = 20
    sim_dt: float = 1.0 / 60.0
    action_scale: float = 0.5

    # Curriculum (success tolerance annealing)
    success_tolerance: float = 0.4  # radians-ish geodesic tolerance
    min_tolerance: float = 0.1
    tolerance_decay: float = 0.9
    curriculum_success_threshold: float = 3.0

    # Reward
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Randomization
    randomize_object: bool = True
    randomize_goal: bool = True

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(
                f"Unknown task '{self.task}'. Valid tasks: {TASKS}"
            )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class AllegroHandEnv:
    """Vectorized 16-DoF AllegroHand cube reorientation environment."""

    def __init__(self, config: Optional[AllegroHandConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = AllegroHandConfig(**kwargs)
        elif kwargs:
            # Allow overriding individual fields via kwargs.
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        self.config = config
        self.device = torch.device(config.device)
        self.num_envs = int(config.num_envs)
        self._use_isaacgym = _use_isaacgym

        # Joint state
        self.q = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self.q_dot = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

        # Object state
        self.object_pos = torch.zeros(self.num_envs, OBJECT_POS_DIM, device=self.device)
        self.object_quat = torch.zeros(self.num_envs, OBJECT_QUAT_DIM, device=self.device)
        self.object_quat[:, 3] = 1.0  # identity (w, x, y, z)
        self.object_vel = torch.zeros(self.num_envs, OBJECT_VEL_DIM, device=self.device)
        self.object_angvel = torch.zeros(self.num_envs, OBJECT_ANGVEL_DIM, device=self.device)

        # Goal quaternion
        self.goal_quat = torch.zeros(self.num_envs, OBJECT_QUAT_DIM, device=self.device)
        self.goal_quat[:, 3] = 1.0

        # Bookkeeping
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_flag = torch.zeros(self.num_envs, device=self.device)
        self.orientation_error = torch.zeros(self.num_envs, device=self.device)
        self.tolerance = float(config.success_tolerance)

        self.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def observation_dim(self) -> int:
        return (
            NUM_JOINTS
            + NUM_JOINTS
            + OBJECT_POS_DIM
            + OBJECT_QUAT_DIM
            + OBJECT_VEL_DIM
            + OBJECT_ANGVEL_DIM
            + OBJECT_QUAT_DIM
            + AUX_DIM
        )

    @property
    def num_actions(self) -> int:
        return NUM_JOINTS

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_quat(self, n: int) -> torch.Tensor:
        """Sample uniformly random unit quaternions (w, x, y, z)."""
        u = torch.rand(n, 3, device=self.device)
        q = torch.stack(
            [
                torch.sqrt(1.0 - u[:, 0]) * torch.sin(2.0 * math.pi * u[:, 1]),
                torch.sqrt(1.0 - u[:, 0]) * torch.cos(2.0 * math.pi * u[:, 1]),
                torch.sqrt(u[:, 0]) * torch.sin(2.0 * math.pi * u[:, 2]),
                torch.sqrt(u[:, 0]) * torch.cos(2.0 * math.pi * u[:, 2]),
            ],
            dim=-1,
        )
        # Reorder to (w, x, y, z)
        q = q[:, [3, 0, 1, 2]]
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _sample_goal(self, n: int) -> torch.Tensor:
        if self.config.randomize_goal:
            return self._random_quat(n)
        goal = torch.zeros(n, OBJECT_QUAT_DIM, device=self.device)
        goal[:, 3] = 1.0
        return goal

    def _obs(self) -> torch.Tensor:
        return torch.cat(
            [
                self.q,
                self.q_dot,
                self.object_pos,
                self.object_quat,
                self.object_vel,
                self.object_angvel,
                self.goal_quat,
                torch.stack([self.success_flag, self.orientation_error], dim=-1),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset (a subset of) environments and return the observation."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = int(env_ids.numel())
        if n == 0:
            return self._obs()

        # Reset joints to a small random pose near zero.
        self.q[env_ids] = 0.1 * torch.randn(n, NUM_JOINTS, device=self.device)
        self.q_dot[env_ids] = 0.0

        # Object pose: small random offset from origin, random orientation.
        if self.config.randomize_object:
            self.object_pos[env_ids] = 0.02 * torch.randn(n, OBJECT_POS_DIM, device=self.device)
            self.object_quat[env_ids] = self._random_quat(n)
        else:
            self.object_pos[env_ids] = 0.0
            self.object_quat[env_ids] = 0.0
            self.object_quat[env_ids, 3] = 1.0

        self.object_vel[env_ids] = 0.0
        self.object_angvel[env_ids] = 0.0

        self.goal_quat[env_ids] = self._sample_goal(n)

        self.episode_step[env_ids] = 0
        self.success_flag[env_ids] = 0.0
        self.orientation_error[env_ids] = r_orientation(
            self.object_quat[env_ids], self.goal_quat[env_ids]
        ).abs()

        return self._obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Advance the simulation by one control step.

        Args:
            actions: ``(num_envs, num_actions)`` joint position targets (scaled).

        Returns:
            ``(obs, reward, done, info)`` where ``done`` is a boolean tensor of
            shape ``(num_envs,)`` and ``info`` contains ``success`` and
            ``episode_reward`` bookkeeping.
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        actions = actions.to(self.device)

        # --- Fallback analytic dynamics -------------------------------------
        # Joint targets -> smoothed joint motion.
        target_q = self.config.action_scale * torch.tanh(actions)
        self.q_dot = 0.5 * (target_q - self.q) / max(self.config.sim_dt, 1e-6)
        self.q = self.q + self.q_dot * self.config.sim_dt

        # Hand motion couples to the object: crude but differentiable proxy.
        hand_effect = self.q_dot.mean(dim=-1, keepdim=True)  # (N, 1)
        self.object_angvel = 0.5 * self.object_angvel + 0.3 * hand_effect * torch.ones(
            self.num_envs, OBJECT_ANGVEL_DIM, device=self.device
        )
        self.object_vel = 0.9 * self.object_vel + 0.01 * hand_effect * torch.ones(
            self.num_envs, OBJECT_VEL_DIM, device=self.device
        )

        # Integrate object pose.
        self.object_pos = self.object_pos + self.object_vel * self.config.sim_dt
        self.object_quat = self._integrate_quat(
            self.object_quat, self.object_angvel, self.config.sim_dt
        )

        # --- Reward ----------------------------------------------------------
        reward = compute_reorientation_reward(
            self.object_quat,
            self.goal_quat,
            config=self.config.reward,
            success_tolerance=self.tolerance,
        )

        # --- Success / orientation error ------------------------------------
        self.orientation_error = r_orientation(
            self.object_quat, self.goal_quat
        ).abs()
        success = (self.orientation_error <= self.tolerance).float()
        self.success_flag = torch.clamp(self.success_flag + success, max=1.0)

        # --- Termination -----------------------------------------------------
        self.episode_step = self.episode_step + 1
        timeout = self.episode_step >= self.config.episode_length
        done = timeout

        info: Dict[str, Any] = {
            "success": success,
            "orientation_error": self.orientation_error,
            "time_outs": timeout,
        }

        # Auto-reset finished envs.
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            self.reset(done_ids)

        return self._obs(), reward, done, info

    # ------------------------------------------------------------------
    # Quaternion integration
    # ------------------------------------------------------------------
    @staticmethod
    def _integrate_quat(
        quat: torch.Tensor, angvel: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Integrate a quaternion (w, x, y, z) by angular velocity ``angvel``."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        wx, wy, wz = angvel[:, 0], angvel[:, 1], angvel[:, 2]
        half_dt = 0.5 * dt
        # dq = 0.5 * q * omega
        dw = -half_dt * (x * wx + y * wy + z * wz)
        dx = half_dt * (w * wx + y * wz - z * wy)
        dy = half_dt * (w * wy - x * wz + z * wx)
        dz = half_dt * (w * wz + x * wy - y * wx)
        new = torch.stack([w + dw, x + dx, y + dy, z + dz], dim=-1)
        return new / new.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def update_curriculum(self, avg_successes_per_episode: float) -> float:
        """Anneal the success tolerance when the policy is doing well.

        Mirrors the paper's curriculum: when the average number of successes
        per episode exceeds ``curriculum_success_threshold`` (default 3), the
        tolerance is reduced by ``tolerance_decay`` (default 10% per step),
        down to ``min_tolerance``.
        """
        if avg_successes_per_episode > self.config.curriculum_success_threshold:
            self.tolerance = max(
                self.config.min_tolerance,
                self.tolerance * self.config.tolerance_decay,
            )
        return self.tolerance

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:  # pragma: no cover - no resources to release
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_allegro_hand(task: str = "reorientation", **kwargs: Any) -> AllegroHandEnv:
    """Convenience factory for the AllegroHand environment."""
    config = AllegroHandConfig(task=task, **kwargs)
    return AllegroHandEnv(config)
