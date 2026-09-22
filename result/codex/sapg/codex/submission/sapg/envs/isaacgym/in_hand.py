"""The two easy in-hand reorientation tasks (ShadowHand / AllegroHand).

"Shadow Hand: We test on in-hand reorientation task of a cube using the 24-DoF
Shadow Hand.  The task is to attain a specified goal orientation (specified as
a quaternion) for the cube ``g_t in R^4``.  The reward is a combination of the
orientation error and a success bonus." (App. A)

The AllegroHand variant is identical except for the 16-DoF hand.  Both follow
the environments of Li et al. (2023); the net episode reward is the performance
metric (App. A).
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from ..base import StepResult, VecEnv
from .assets import DEFAULT_ASSET_ROOT, asset_path


class InHandReorientationTask(VecEnv):
    hand_key = "shadow_hand"
    num_hand_dofs = 24

    def __init__(self, cfg, device: str = "cuda:0", for_eval: bool = False) -> None:
        import isaacgym  # noqa: F401
        from isaacgym import gymapi, gymtorch  # noqa: F401

        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.cfg = cfg
        self.device = torch.device(device)
        env_cfg = cfg.get("env", {}) or {}

        self.num_envs = int(env_cfg.get("num_envs", 24576))
        self.episode_length = int(env_cfg.get("episode_length", 200))
        self.control_freq = int(env_cfg.get("control_freq", 4))
        self.asset_root = str(env_cfg.get("asset_root", DEFAULT_ASSET_ROOT))
        self.num_dofs = self.num_hand_dofs
        self.action_dim = self.num_hand_dofs
        self.obs_dim = 2 * self.num_dofs + 7 + 3 + 3 + 4 + 4
        self.rotation_scale = float(env_cfg.get("rotation_scale", 5.0))
        self.success_bonus = float(env_cfg.get("success_bonus", 10.0))
        self.success_threshold = float(env_cfg.get("success_threshold", 0.1))

        self._build_sim()
        self._allocate_buffers()
        self.reset()

    # ------------------------------------------------------------------ #
    def _build_sim(self) -> None:  # pragma: no cover - requires IsaacGym
        gymapi = self.gymapi
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 120.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.use_gpu = True
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        options = gymapi.AssetOptions()
        options.fix_base_link = True
        hand_asset = self.gym.load_asset(
            self.sim, os.path.dirname(asset_path(self.asset_root, self.hand_key, suite="in_hand")),
            os.path.basename(asset_path(self.asset_root, self.hand_key, suite="in_hand")), options,
        )
        block_options = gymapi.AssetOptions()
        block_options.fix_base_link = False
        block_options.density = 1000.0
        block_asset = self.gym.load_asset(
            self.sim, os.path.dirname(asset_path(self.asset_root, "object", suite="in_hand")),
            os.path.basename(asset_path(self.asset_root, "object", suite="in_hand")), block_options,
        )
        pose = gymapi.Transform()
        self.env_handles = []
        self.hand_handles, self.object_handles = [], []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            env = self.gym.create_env(
                self.sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), num_per_row
            )
            self.env_handles.append(env)
            self.hand_handles.append(self.gym.create_actor(env, hand_asset, pose, "hand", i, 0))
            self.object_handles.append(self.gym.create_actor(env, block_asset, pose, "object", i, 0))
        self.gym.prepare_sim(self.sim)

    def _allocate_buffers(self) -> None:  # pragma: no cover
        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.default_dof_pos = torch.zeros(self.num_dofs, device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_quat[:, 0] = 1.0
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)

    def _refresh_state(self) -> None:  # pragma: no cover
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

    # ------------------------------------------------------------------ #
    def _orientation_error(self) -> torch.Tensor:  # pragma: no cover
        dot = torch.abs((self.goal_quat * self.object_quat).sum(dim=-1)).clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    def _compute_observations(self) -> torch.Tensor:  # pragma: no cover
        return torch.cat(
            [
                self.dof_pos,
                self.dof_vel,
                self.object_pos,
                self.object_quat,
                self.object_lin_vel,
                self.object_ang_vel,
                self.goal_quat,
                self.actions,
            ],
            dim=-1,
        )

    def _compute_reward(self) -> torch.Tensor:  # pragma: no cover
        error = self._orientation_error()
        reward = torch.exp(-self.rotation_scale * error)
        reward = reward + self.success_bonus * (error < self.success_threshold).float()
        return reward

    # ------------------------------------------------------------------ #
    def reset(self) -> torch.Tensor:  # pragma: no cover
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh_state()
        return self._compute_observations()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:  # pragma: no cover
        n = env_ids.numel()
        self.dof_pos[env_ids] = self.default_dof_pos + 0.1 * torch.randn((n, self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.0
        self.object_pos[env_ids] = torch.zeros((n, 3), device=self.device)
        self.object_quat[env_ids] = torch.zeros((n, 4), device=self.device)
        self.object_quat[env_ids, 0] = 1.0
        self.goal_quat[env_ids] = _random_quaternion(n, self.device)
        self.progress_buf[env_ids] = 0

    def step(self, actions: torch.Tensor) -> StepResult:  # pragma: no cover
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self._apply_position_targets(self.default_dof_pos + 0.5 * self.actions)
        self._refresh_state()
        rewards = self._compute_reward()
        self.progress_buf += 1
        timeouts = (self.progress_buf >= self.episode_length).float()
        dones = timeouts.clone()
        reset_ids = (timeouts > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            self._refresh_state()
        success = (self._orientation_error() < self.success_threshold).float()
        return StepResult(
            obs=self._compute_observations(),
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            infos={"successes": success},
        )

    def _apply_position_targets(self, targets: torch.Tensor) -> None:  # pragma: no cover
        self.gym.set_dof_position_target_tensor(self.sim, self.gymtorch.unwrap_tensor(targets))
        for _ in range(self.control_freq):
            self.gym.simulate(self.sim)

    def episode_stats(self) -> Dict[str, float]:  # pragma: no cover
        return {}


class ShadowHandReorientation(InHandReorientationTask):
    hand_key = "shadow_hand"
    num_hand_dofs = 24


class AllegroHandReorientation(InHandReorientationTask):
    hand_key = "allegro_hand"
    num_hand_dofs = 16


def _random_quaternion(n: int, device) -> torch.Tensor:
    u = torch.rand((n, 3), device=device)
    q = torch.stack(
        [
            torch.sqrt(1 - u[:, 0]) * torch.sin(2 * np.pi * u[:, 1]),
            torch.sqrt(1 - u[:, 0]) * torch.cos(2 * np.pi * u[:, 1]),
            torch.sqrt(u[:, 0]) * torch.sin(2 * np.pi * u[:, 2]),
            torch.sqrt(u[:, 0]) * torch.cos(2 * np.pi * u[:, 2]),
        ],
        dim=-1,
    )
    return q
