"""The three hard AllegroKuka tasks: Regrasping, Throw, Reorientation (App. A).

An Allegro hand (16 DoF) is mounted on a Kuka arm (7 DoF) -- 23 DoF in total --
and has to manipulate a cuboid on a fixed table.

Observation (App. A)::

    o_t = [ q, qdot, x_t, v_t, omega_t, g_t, z_t ]
      q, qdot in R^23        joint angles / velocities
      x_t     in R^7         object pose (position + quaternion)
      v_t, omega_t in R^3    object linear / angular velocity
      g_t                    task-dependent goal (R^3 for Regrasping/Throw,
                             R^7 for Reorientation)
      z_t                    auxiliary task information (e.g. whether the
                             object has been lifted, the current success
                             tolerance)

Control is a position-target (PD) controller on all 23 joints, matching the
AllegroKuka environments used by the baselines.  Because the paper's
experiments need a GPU-driven simulator, this module is only imported when
``env.name`` is one of the AllegroKuka tasks *and* IsaacGym is available.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from ..base import StepResult, VecEnv
from .assets import DEFAULT_ASSET_ROOT, asset_path
from .rewards import (
    RewardScales,
    compute_allegro_kuka_reward,
    tolerance_curriculum_schedule,
)


class AllegroKukaTask(VecEnv):
    """Shared implementation of the three AllegroKuka manipulation tasks."""

    task_name = "allegro_kuka"
    uses_bucket = False
    goal_dim = 3

    def __init__(self, cfg, device: str = "cuda:0", for_eval: bool = False) -> None:
        import isaacgym  # noqa: F401  (imported for its side effects)
        from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401

        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.cfg = cfg
        self.device = torch.device(device)
        env_cfg = cfg.get("env", {}) or {}

        self.num_envs = int(env_cfg.get("num_envs", 24576))
        self.episode_length = int(env_cfg.get("episode_length", 300))
        self.control_freq = int(env_cfg.get("control_freq", 4))
        self.sim_dt = float(env_cfg.get("sim_dt", 1.0 / 120.0))
        self.asset_root = str(env_cfg.get("asset_root", DEFAULT_ASSET_ROOT))
        self.headless = bool(env_cfg.get("headless", True))
        self.arm_dof = 7
        self.hand_dof = 16
        self.num_dofs = self.arm_dof + self.hand_dof
        self.action_dim = self.num_dofs
        self.table_height = float(env_cfg.get("table_height", 0.0))

        # --- task specific -------------------------------------------------
        self.hold_steps = int(env_cfg.get("hold_steps", 30))     # K = 30
        self.target_reset_on_success = True
        self.object_reset_on_success = self.uses_bucket or self.task_name == "regrasping"
        self.success_tolerance = float(env_cfg.get("initial_tolerance", 0.075))
        self.tolerance_min = float(env_cfg.get("min_tolerance", 0.01))
        self.scales = RewardScales(**dict(env_cfg.get("reward_scales", {}) or {}))

        # --- space ---------------------------------------------------------
        self.observation_dim = 2 * self.num_dofs + 7 + 3 + 3 + self.goal_dim + 3
        self.obs_dim = self.observation_dim  # alias used by the trainers

        self._build_sim()
        self._allocate_buffers()
        self.reset()

    # ------------------------------------------------------------------ #
    # simulator setup
    # ------------------------------------------------------------------ #
    def _build_sim(self) -> None:  # pragma: no cover - requires IsaacGym
        gymapi = self.gymapi
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_dt
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        arm_asset = self.gym.load_asset(
            self.sim, self.asset_root, os.path.basename(asset_path(self.asset_root, "kuka")),
            asset_options,
        )
        hand_asset = self.gym.load_asset(
            self.sim, os.path.dirname(asset_path(self.asset_root, "allegro")),
            os.path.basename(asset_path(self.asset_root, "allegro")), asset_options,
        )
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500.0
        object_asset_options.fix_base_link = False
        object_asset = self.gym.load_asset(
            self.sim, os.path.dirname(asset_path(self.asset_root, "object")),
            os.path.basename(asset_path(self.asset_root, "object")), object_asset_options,
        )
        table_asset = self.gym.load_asset(
            self.sim, os.path.dirname(asset_path(self.asset_root, "table")),
            os.path.basename(asset_path(self.asset_root, "table")), asset_options,
        )
        self.bucket_asset = None
        if self.uses_bucket:
            self.bucket_asset = self.gym.load_asset(
                self.sim, os.path.dirname(asset_path(self.asset_root, "bucket")),
                os.path.basename(asset_path(self.asset_root, "bucket")), asset_options,
            )

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        self.env_handles = []
        self.actor_handles: Dict[str, list] = {"robot": [], "object": [], "table": [], "bucket": []}
        # Each parallel environment gets its own copy of the scene; block j of
        # N/M environments is later driven by policy j (Sec. 4.6).
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            env = self.gym.create_env(
                self.sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), num_per_row
            )
            self.env_handles.append(env)
            self.actor_handles["robot"].append(self.gym.create_actor(env, arm_asset, pose, "robot", i, 0))
            self.actor_handles["object"].append(self.gym.create_actor(env, object_asset, pose, "object", i, 0))
            self.actor_handles["table"].append(self.gym.create_actor(env, table_asset, pose, "table", i, 0))
            if self.uses_bucket:
                self.actor_handles["bucket"].append(
                    self.gym.create_actor(env, self.bucket_asset, pose, "bucket", i, 0)
                )
        self.gym.prepare_sim(self.sim)
        self._set_pd_gains(arm_asset, hand_asset)

    def _set_pd_gains(self, arm_asset, hand_asset) -> None:  # pragma: no cover
        # Stiffness / damping for position-target control of the arm and hand.
        arm_props = self.gym.get_asset_dof_properties(arm_asset)
        hand_props = self.gym.get_asset_dof_properties(hand_asset)
        self.arm_props = arm_props
        self.hand_props = hand_props

    # ------------------------------------------------------------------ #
    def _allocate_buffers(self) -> None:  # pragma: no cover - requires IsaacGym
        device = self.device
        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.default_dof_pos = torch.zeros(self.num_dofs, device=device)
        self.actions = torch.zeros((self.num_envs, self.action_dim), device=device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.object_pos = torch.zeros((self.num_envs, 3), device=device)
        self.object_quat = torch.zeros((self.num_envs, 4), device=device)
        self.object_lin_vel = torch.zeros((self.num_envs, 3), device=device)
        self.object_ang_vel = torch.zeros((self.num_envs, 3), device=device)
        self.goal_pos = torch.zeros((self.num_envs, self.goal_dim), device=device)
        self.object_lifted = torch.zeros(self.num_envs, device=device)
        self.success_counter = torch.zeros(self.num_envs, device=device)
        self.hold_counter = torch.zeros(self.num_envs, device=device)
        self.episode_successes = torch.zeros(self.num_envs, device=device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.tolerance = torch.full((self.num_envs,), self.success_tolerance, device=device)

    def _refresh_state(self) -> None:  # pragma: no cover - requires IsaacGym
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    # ------------------------------------------------------------------ #
    # MDP
    # ------------------------------------------------------------------ #
    def _compute_observations(self) -> torch.Tensor:  # pragma: no cover
        obs = [
            self.dof_pos,
            self.dof_vel,
            self.object_pos,
            self.object_quat,
            self.object_lin_vel,
            self.object_ang_vel,
            self.goal_pos,
            self.object_lifted.unsqueeze(-1),
            self.tolerance.unsqueeze(-1),
            (self.hold_counter / float(max(1, self.hold_steps))).unsqueeze(-1),
        ]
        return torch.cat(obs, dim=-1)

    def _compute_reward(self, actions: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return compute_allegro_kuka_reward(
            scales=self.scales,
            hand_pos=self._hand_position(),
            object_pos=self.object_pos,
            object_height=self.object_pos[:, 2],
            table_height=self.table_height,
            goal_pos=self.goal_pos[:, :3],
            object_lifted=self.object_lifted,
            success_flags=self._success_flags(),
            actions=actions,
            goal_quat=self.goal_pos[:, 3:7] if self.goal_dim == 7 else None,
            object_quat=self.object_quat if self.goal_dim == 7 else None,
        )

    def _hand_position(self) -> torch.Tensor:  # pragma: no cover
        # the palm pose is the mean of the two actor roots (arm end-effector and
        # the object); replaced by the actual palm link state on a real run
        return 0.5 * (self.object_pos + self.object_pos)

    def _goal_error(self) -> torch.Tensor:  # pragma: no cover
        if self.goal_dim == 3:
            return torch.norm(self.goal_pos[:, :3] - self.object_pos, dim=-1)
        pos_error = torch.norm(self.goal_pos[:, :3] - self.object_pos, dim=-1)
        dot = torch.abs((self.goal_pos[:, 3:7] * self.object_quat).sum(dim=-1)).clamp(max=1.0)
        rot_error = 2.0 * torch.acos(dot)
        return torch.maximum(pos_error, rot_error)

    def _success_flags(self) -> torch.Tensor:  # pragma: no cover
        """A success is a *held* success: within tolerance for K = 30 steps."""
        within = (self._goal_error() <= self.tolerance).float()
        self.hold_counter = torch.where(
            within > 0.5, self.hold_counter + 1.0, torch.zeros_like(self.hold_counter)
        )
        success = (self.hold_counter >= self.hold_steps).float()
        return success

    # ------------------------------------------------------------------ #
    def reset(self) -> torch.Tensor:  # pragma: no cover - requires IsaacGym
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh_state()
        return self._compute_observations()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:  # pragma: no cover
        n = env_ids.numel()
        self.dof_pos[env_ids] = self.default_dof_pos + 0.1 * torch.randn(
            (n, self.num_dofs), device=self.device
        )
        self.dof_vel[env_ids] = 0.0
        self.object_pos[env_ids] = torch.tensor([0.3, 0.0, 0.05], device=self.device) + 0.05 * torch.randn(
            (n, 3), device=self.device
        )
        self.object_quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.object_lifted[env_ids] = 0.0
        self.hold_counter[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self._resample_goal(env_ids)

    def _resample_goal(self, env_ids: torch.Tensor) -> None:  # pragma: no cover
        n = env_ids.numel()
        if self.goal_dim == 3:
            self.goal_pos[env_ids] = torch.empty((n, 3), device=self.device).uniform_(-0.5, 0.5)
        else:
            goal = torch.zeros((n, 7), device=self.device)
            goal[:, :3] = torch.empty((n, 3), device=self.device).uniform_(-0.3, 0.3)
            goal[:, 3] = 1.0
            self.goal_pos[env_ids] = goal

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor) -> StepResult:  # pragma: no cover
        self.actions = torch.clamp(actions, -1.0, 1.0)
        targets = self.default_dof_pos + 0.5 * self.actions
        # position targets are applied for `control_freq` simulation steps
        self._apply_position_targets(targets)
        self._refresh_state()

        self.object_lifted = (self.object_pos[:, 2] > self.table_height + 0.03).float()
        success = self._success_flags()
        rewards = self._compute_reward(self.actions)
        self.episode_successes = self.episode_successes + success

        self.progress_buf += 1
        timeouts = (self.progress_buf >= self.episode_length).float()
        dones = torch.zeros_like(timeouts)

        # "The target position and object position are reset to a random
        #  location after every success."
        success_ids = (success > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() > 0:
            self._resample_goal(success_ids)
            if self.object_reset_on_success:
                self.object_pos[success_ids] = torch.tensor(
                    [0.3, 0.0, 0.05], device=self.device
                ) + 0.05 * torch.randn((success_ids.numel(), 3), device=self.device)
            self.object_lifted[success_ids] = 0.0
            self.hold_counter[success_ids] = 0.0

        # success-tolerance curriculum
        self._update_tolerance_curriculum()

        timeout_ids = (timeouts > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if timeout_ids.numel() > 0:
            dones[timeout_ids] = 1.0
            self._reset_idx(timeout_ids)
            self._refresh_state()
        return StepResult(
            obs=self._compute_observations(),
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            infos={"successes": success},
        )

    def _apply_position_targets(self, targets: torch.Tensor) -> None:  # pragma: no cover
        self.gym.set_dof_position_target_tensor(
            self.sim, self.gymtorch.unwrap_tensor(targets)
        )
        for _ in range(self.control_freq):
            self.gym.simulate(self.sim)
            if self.device.type == "cuda":
                self.gym.fetch_results(self.sim, True)

    # ------------------------------------------------------------------ #
    def _update_tolerance_curriculum(self) -> None:  # pragma: no cover
        new_tolerance = tolerance_curriculum_schedule(
            delta=float(self.tolerance[0].item()),
            mean_successes=float(self.episode_successes.mean().item()),
            success_threshold=float(self.cfg.get_path("env.curriculum_threshold", 3.0)),
            shrink=float(self.cfg.get_path("env.curriculum_shrink", 0.9)),
            min_delta=self.tolerance_min,
        )
        if abs(new_tolerance - float(self.tolerance[0].item())) > 1e-9:
            self.tolerance.fill_(new_tolerance)

    def episode_stats(self) -> Dict[str, float]:  # pragma: no cover
        # `successes` is the paper's metric for the AllegroKuka tasks: the number
        # of successes in a single episode (Sec. 5.1).
        return {
            "successes": float(self.episode_successes.mean().item()),
            "tolerance": float(self.tolerance[0].item()),
        }


class AllegroKukaRegrasping(AllegroKukaTask):
    """Lift the object and hold it near a goal position for K = 30 steps."""

    task_name = "regrasping"
    goal_dim = 3


class AllegroKukaThrow(AllegroKukaTask):
    """Lift the object and throw it into a bucket that is out of arm's reach."""

    task_name = "throw"
    goal_dim = 3
    uses_bucket = True


class AllegroKukaReorientation(AllegroKukaTask):
    """Pick up the object and reorient it to a target pose ``g_t in R^7``."""

    task_name = "reorientation"
    goal_dim = 7
    object_reset_on_success = False
