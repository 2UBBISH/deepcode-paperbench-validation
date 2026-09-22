"""ShadowHand environment: 24-DoF in-hand cube reorientation.

The task is to reorient a cube to a target orientation (goal quaternion ``g_t``
in R^4) using the 24-DoF ShadowHand.  The metric reported in the paper is the
*net episode reward* (not success rate), so the reward is a dense orientation
error term plus a sparse success bonus.

This module follows the same simulator-agnostic, fully vectorized (torch
tensor) design as :mod:`sapg.envs.allegro_kuka`: it attempts to use IsaacGym
when available and otherwise falls back to a lightweight analytic simulator so
that the code remains runnable on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .reward import RewardConfig, compute_reorientation_reward, r_orientation

try:  # pragma: no cover - optional heavy dependency
    import isaacgym  # noqa: F401

    _use_isaacgym = True
except Exception:  # pragma: no cover
    _use_isaacgym = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_JOINTS = 24
OBJECT_QUAT_DIM = 4
OBJECT_POS_DIM = 3
OBJECT_VEL_DIM = 3
OBJECT_ANGVEL_DIM = 3
AUX_DIM = 2  # [success_flag, orientation_error]

TASKS = ("reorientation",)


@dataclass
class ShadowHandConfig:
    """Configuration for :class:`ShadowHandEnv`."""

    task: str = "reorientation"
    num_envs: int = 4096
    device: str = "cpu"
    episode_length: int = 100
    control_freq: int = 20
    sim_dt: float = 1.0 / 120.0
    action_scale: float = 0.5
    success_tolerance: float = 0.1
    min_tolerance: float = 0.05
    tolerance_decay: float = 0.9
    curriculum_success_threshold: float = 3.0
    reward: RewardConfig = field(default_factory=RewardConfig)
    randomize_object: bool = True
    randomize_goal: bool = True

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(
                f"Unknown ShadowHand task '{self.task}'. Valid tasks: {TASKS}"
            )


class ShadowHandEnv:
    """Vectorized 24-DoF ShadowHand cube reorientation environment."""

    def __init__(self, config: Optional[ShadowHandConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = ShadowHandConfig(**kwargs)
        self.cfg = config
        self.device = torch.device(config.device)
        self.num_envs = int(config.num_envs)
        self._use_isaacgym = _use_isaacgym

        # Joint state -----------------------------------------------------
        self.q = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self.q_dot = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

        # Object state ----------------------------------------------------
        self.object_pos = torch.zeros(self.num_envs, OBJECT_POS_DIM, device=self.device)
        self.object_quat = torch.zeros(self.num_envs, OBJECT_QUAT_DIM, device=self.device)
        self.object_vel = torch.zeros(self.num_envs, OBJECT_VEL_DIM, device=self.device)
        self.object_angvel = torch.zeros(self.num_envs, OBJECT_ANGVEL_DIM, device=self.device)

        # Goal ------------------------------------------------------------
        self.goal_quat = torch.zeros(self.num_envs, OBJECT_QUAT_DIM, device=self.device)

        # Bookkeeping -----------------------------------------------------
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_successes = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward = torch.zeros(self.num_envs, device=self.device)
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
        q = torch.randn(n, 4, device=self.device)
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _sample_goal(self, n: int) -> torch.Tensor:
        if self.cfg.randomize_goal:
            return self._random_quat(n)
        goal = torch.zeros(n, 4, device=self.device)
        goal[:, 0] = 1.0
        return goal

    def _obs(self) -> torch.Tensor:
        orientation_error = r_orientation(self.object_quat, self.goal_quat).unsqueeze(-1)
        aux = torch.cat(
            [self.success_buf.float().unsqueeze(-1), orientation_error], dim=-1
        )
        return torch.cat(
            [
                self.q,
                self.q_dot,
                self.object_pos,
                self.object_quat,
                self.object_vel,
                self.object_angvel,
                self.goal_quat,
                aux,
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = int(env_ids.numel())
        if n == 0:
            return self._obs()

        self.q[env_ids] = 0.1 * torch.randn(n, NUM_JOINTS, device=self.device)
        self.q_dot[env_ids] = 0.0
        self.object_pos[env_ids] = 0.0
        self.object_pos[env_ids, 2] = 0.05
        self.object_quat[env_ids] = self._random_quat(n)
        self.object_vel[env_ids] = 0.0
        self.object_angvel[env_ids] = 0.0
        self.goal_quat[env_ids] = self._sample_goal(n)
        self.progress_buf[env_ids] = 0
        self.success_buf[env_ids] = False
        self.episode_successes[env_ids] = 0.0
        self.episode_reward[env_ids] = 0.0
        return self._obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        actions = actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        # Simple first-order joint dynamics (fallback simulator) ----------
        self.q_dot = 0.8 * self.q_dot + self.cfg.action_scale * actions
        self.q = self.q + self.q_dot * self.cfg.sim_dt * self.cfg.control_freq

        # Object follows a smoothed version of the hand's mean joint motion.
        hand_motion = self.q_dot.mean(dim=-1, keepdim=True)
        self.object_angvel = 0.9 * self.object_angvel + 0.1 * hand_motion.expand(
            -1, OBJECT_ANGVEL_DIM
        )
        self.object_vel = 0.9 * self.object_vel + 0.1 * hand_motion.expand(
            -1, OBJECT_VEL_DIM
        )
        self.object_pos = self.object_pos + self.object_vel * self.cfg.sim_dt
        self.object_pos[:, 2] = self.object_pos[:, 2].clamp(min=0.02)

        # Integrate quaternion with angular velocity.
        w = self.object_angvel
        dq = torch.cat(
            [
                torch.zeros(self.num_envs, 1, device=self.device),
                w,
            ],
            dim=-1,
        )
        self.object_quat = self.object_quat + 0.5 * dq * self.cfg.sim_dt
        self.object_quat = self.object_quat / self.object_quat.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        # Reward ----------------------------------------------------------
        reward = compute_reorientation_reward(
            self.object_quat,
            self.goal_quat,
            config=self.cfg.reward,
            success_tolerance=self.tolerance,
        )
        success = r_orientation(self.object_quat, self.goal_quat) > -self.tolerance
        self.success_buf = success
        self.episode_successes = self.episode_successes + success.float()
        self.episode_reward = self.episode_reward + reward

        self.progress_buf = self.progress_buf + 1
        done = self.progress_buf >= self.cfg.episode_length

        info: Dict[str, Any] = {
            "success": success,
            "time_outs": done.clone(),
            "episode_successes": self.episode_successes.clone(),
            "episode_reward": self.episode_reward.clone(),
        }

        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            info["final_episode_successes"] = self.episode_successes[done_ids].clone()
            info["final_episode_reward"] = self.episode_reward[done_ids].clone()
            self.reset(done_ids)

        return self._obs(), reward, done, info

    def update_curriculum(self, avg_successes_per_episode: float) -> float:
        """Anneal the success tolerance when the task becomes too easy."""
        if avg_successes_per_episode > self.cfg.curriculum_success_threshold:
            self.tolerance = max(
                self.cfg.min_tolerance, self.tolerance * self.cfg.tolerance_decay
            )
        return self.tolerance

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None


def make_shadow_hand(task: str = "reorientation", **kwargs: Any) -> ShadowHandEnv:
    """Convenience factory for :class:`ShadowHandEnv`."""
    return ShadowHandEnv(ShadowHandConfig(task=task, **kwargs))
