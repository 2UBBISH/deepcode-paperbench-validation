"""AllegroHand 16-DoF in-hand reorientation task for SAPG.

This module implements the AllegroHand in-hand reorientation task described in
the SAPG paper (Section 5 / Table 4).  The task requires reorienting a cube to a
randomly sampled target orientation using a 16-DoF Allegro hand mounted on a
fixed wrist.

Observation layout (paper Section 4.2 / Addendum)::

    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]

where
    q       : hand joint positions            (16)
    q_dot   : hand joint velocities           (16)
    x_t     : object position                 (3)
    v_t     : object linear velocity          (3)
    omega_t : object angular velocity         (3)
    g_t     : goal orientation (quaternion)   (4)
    z_t     : auxiliary features              (16)

Total observation dimension: 16 + 16 + 3 + 3 + 3 + 4 + 16 = 61.

The reward follows the PQL (Li et al., 2023) decomposition used for in-hand
reorientation: orientation tracking, position tracking, velocity / angular
velocity / action penalties, plus a sparse success bonus.

A lazy IsaacGym simulation path is provided; if IsaacGym is unavailable (e.g.
CPU-only CI), the environment falls back to a pure-PyTorch analytic dynamics
model so that the full SAPG pipeline can be exercised end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

try:  # pragma: no cover - depends on the host machine
    from isaacgym import gymapi, gymtorch, gymutil  # type: ignore

    _HAS_ISAACGYM = True
except Exception:  # pragma: no cover
    gymapi = None  # type: ignore
    gymtorch = None  # type: ignore
    gymutil = None  # type: ignore
    _HAS_ISAACGYM = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HAND_DOF = 16
OBJECT_STATE_DIM = 13  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
GOAL_DIM = 4  # target orientation quaternion
AUX_DIM = 16  # auxiliary features (e.g. previous action / fingertip contacts)
OBS_DIM = HAND_DOF + HAND_DOF + 3 + 3 + 3 + GOAL_DIM + AUX_DIM  # = 61

TASKS = ("reorientation",)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AllegroHandRewardWeights:
    """Reward weights for the AllegroHand reorientation task (PQL-style)."""

    orientation: float = 1.0
    position: float = 0.5
    velocity: float = 0.01
    angular_velocity: float = 0.01
    action_penalty: float = 0.0001
    torque_penalty: float = 1e-5
    success: float = 10.0


@dataclass
class AllegroHandConfig:
    """Configuration for the AllegroHand reorientation task."""

    task: str = "reorientation"
    num_envs: int = 24576
    device: str = "cuda:0"
    headless: bool = True
    seed: int = 0

    # Episode / control
    episode_length: int = 200
    control_freq: int = 20
    dt: float = 1.0 / 60.0

    # Object / goal sampling
    object_pos_range: float = 0.02
    object_height: float = 0.60
    goal_orientation_range: float = 1.0

    # Curriculum
    use_curriculum: bool = True
    tolerance_start: float = 0.5
    tolerance_end: float = 0.1
    tolerance_decay: float = 0.90
    success_threshold: float = 3.0

    reward_weights: AllegroHandRewardWeights = field(
        default_factory=AllegroHandRewardWeights
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_quaternion(
    num: int, device: torch.device, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample uniformly random unit quaternions of shape ``(num, 4)``."""
    u1 = torch.rand(num, device=device, generator=generator)
    u2 = torch.rand(num, device=device, generator=generator)
    u3 = torch.rand(num, device=device, generator=generator)
    q1 = torch.sqrt(1.0 - u1) * torch.sin(2.0 * math.pi * u2)
    q2 = torch.sqrt(1.0 - u1) * torch.cos(2.0 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2.0 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2.0 * math.pi * u3)
    return torch.stack([q1, q2, q3, q4], dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions ``a`` and ``b`` (last dim = 4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (last dim = 4)."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def _quat_angle(q: torch.Tensor) -> torch.Tensor:
    """Rotation angle (radians) encoded by a unit quaternion."""
    w = q[..., 0].clamp(-1.0, 1.0)
    return 2.0 * torch.acos(w.abs())


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class AllegroHandTask:
    """Vectorized AllegroHand in-hand reorientation environment.

    The class exposes the SAPG env contract: ``num_envs``, ``obs_dim``,
    ``act_dim``, ``reset()``, ``step(actions)`` and ``close()``.
    """

    def __init__(
        self,
        config: Optional[AllegroHandConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = AllegroHandConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config
        self.task = config.task
        self.num_envs = int(config.num_envs)
        self.device = torch.device(config.device)
        self.obs_dim = OBS_DIM
        self.act_dim = HAND_DOF

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(config.seed))

        # Curriculum state
        self.tolerance = float(config.tolerance_start)
        self._success_count = 0.0
        self._episode_count = 0.0

        # Simulation handles (populated only when IsaacGym is available)
        self._sim = None
        self._envs = None
        self._actor_handles = None
        self._object_handles = None
        self._use_isaacgym = False

        if _HAS_ISAACGYM:
            try:  # pragma: no cover - requires GPU + IsaacGym
                self._build_sim()
                self._use_isaacgym = True
            except Exception:
                self._use_isaacgym = False

        # Analytic fallback state
        self._state = torch.zeros(self.num_envs, OBJECT_STATE_DIM, device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, HAND_DOF, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, HAND_DOF, device=self.device)
        self._goal_quat = torch.zeros(self.num_envs, GOAL_DIM, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, HAND_DOF, device=self.device)
        self._episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_actions(self) -> int:
        return self.act_dim

    @property
    def num_obs(self) -> int:
        return self.obs_dim

    @property
    def max_episode_length(self) -> int:
        return int(self.config.episode_length)

    # ------------------------------------------------------------------
    # IsaacGym construction (lazy)
    # ------------------------------------------------------------------
    def _build_sim(self) -> None:  # pragma: no cover - requires IsaacGym
        """Build the IsaacGym simulation for the AllegroHand task."""
        cfg = self.config
        self._sim = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.dt = cfg.dt
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True

        self._sim = self._sim.create_sim(
            compute_device=0,
            graphics_device=0,
            type=gymapi.SIM_PHYSX,
            params=sim_params,
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self._sim.add_ground(plane_params)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        hand_asset = self._sim.load_asset(
            "assets/allegro_hand_description/allegro_hand_description_right.urdf",
            asset_options,
        )

        obj_options = gymapi.AssetOptions()
        obj_options.density = 500.0
        object_asset = self._sim.create_box(
            self._sim, 0.05, 0.05, 0.05, obj_options
        )

        env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
        env_upper = gymapi.Vec3(0.5, 0.5, 1.0)

        self._envs = []
        self._actor_handles = []
        self._object_handles = []
        for _ in range(self.num_envs):
            env = self._sim.create_env(env_lower, env_upper, 1)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
            actor = self._sim.create_actor(env, hand_asset, pose, "hand", 0, 0)
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(0.0, 0.0, cfg.object_height)
            obj = self._sim.create_actor(env, object_asset, obj_pose, "object", 0, 1)
            self._envs.append(env)
            self._actor_handles.append(actor)
            self._object_handles.append(obj)

        self._sim.prepare_sim()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset all environments and return the initial observation."""
        cfg = self.config

        # Object pose
        pos = torch.zeros(self.num_envs, 3, device=self.device)
        pos[:, 0] = (
            torch.rand(self.num_envs, generator=self._generator) * 2.0 - 1.0
        ) * cfg.object_pos_range
        pos[:, 1] = (
            torch.rand(self.num_envs, generator=self._generator) * 2.0 - 1.0
        ) * cfg.object_pos_range
        pos[:, 2] = cfg.object_height

        quat = _random_quaternion(
            self.num_envs, torch.device("cpu"), self._generator
        ).to(self.device)

        self._state = torch.zeros(self.num_envs, OBJECT_STATE_DIM, device=self.device)
        self._state[:, 0:3] = pos
        self._state[:, 3:7] = quat

        # Goal orientation
        self._goal_quat = _random_quaternion(
            self.num_envs, torch.device("cpu"), self._generator
        ).to(self.device)

        # Joints
        self._joint_pos = (
            torch.rand(self.num_envs, HAND_DOF, generator=self._generator).to(
                self.device
            )
            * 2.0
            - 1.0
        ) * 0.1
        self._joint_vel = torch.zeros(self.num_envs, HAND_DOF, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, HAND_DOF, device=self.device)
        self._episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        return self._build_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Advance the simulation by one control step.

        Returns ``(obs, rewards, dones, infos)`` following the vectorized env
        convention used across SAPG.
        """
        actions = actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        actions = actions.clamp(-1.0, 1.0)

        if self._use_isaacgym:  # pragma: no cover - requires IsaacGym
            self._isaacgym_step(actions)
        else:
            self._analytic_step(actions)

        rewards = self._compute_reward(actions)
        self._prev_action = actions

        self._episode_step += 1
        dones = (self._episode_step >= self.max_episode_length).float()

        # Curriculum update
        self._update_curriculum(rewards, dones)

        obs = self._build_obs()

        infos: Dict[str, Any] = {
            "episode": {
                "r": rewards.clone(),
                "l": self._episode_step.float().clone(),
            },
            "tolerance": self.tolerance,
        }

        # Auto-reset finished environments
        if dones.any():
            done_mask = dones.bool()
            self._reset_envs(done_mask)

        return obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Analytic fallback dynamics
    # ------------------------------------------------------------------
    def _analytic_step(self, actions: torch.Tensor) -> None:
        """Damped analytic dynamics used when IsaacGym is unavailable."""
        # Joint dynamics: first-order response to the commanded action.
        self._joint_vel = 0.9 * self._joint_vel + 0.1 * actions
        self._joint_pos = self._joint_pos + self.config.dt * self._joint_vel
        self._joint_pos = self._joint_pos.clamp(-1.5, 1.5)

        # Object dynamics: coupling from the mean joint velocity + damping.
        coupling = 0.05 * self._joint_vel.mean(dim=-1, keepdim=True)
        self._state[:, 7:10] = 0.95 * self._state[:, 7:10] + coupling
        self._state[:, 10:13] = 0.95 * self._state[:, 10:13] + 0.5 * coupling
        self._state[:, 0:3] = self._state[:, 0:3] + self.config.dt * self._state[:, 7:10]

        # Integrate orientation from angular velocity (small-angle approx).
        omega = self._state[:, 10:13]
        dq = torch.zeros_like(self._state[:, 3:7])
        dq[:, 0] = 1.0
        dq[:, 1:4] = 0.5 * self.config.dt * omega
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self._state[:, 3:7] = _quat_mul(self._state[:, 3:7], dq)
        self._state[:, 3:7] = self._state[:, 3:7] / self._state[
            :, 3:7
        ].norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _isaacgym_step(self, actions: torch.Tensor) -> None:  # pragma: no cover
        """Step the real IsaacGym simulation (requires GPU + IsaacGym)."""
        # Placeholder: real implementation would set DOF targets, step the sim,
        # and refresh the root/object state tensors via gymtorch.
        self._analytic_step(actions)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, actions: torch.Tensor) -> torch.Tensor:
        """PQL-style reward decomposition for in-hand reorientation."""
        w = self.config.reward_weights

        obj_quat = self._state[:, 3:7]
        # Orientation error: angle between current and goal orientation.
        rel = _quat_mul(_quat_conjugate(obj_quat), self._goal_quat)
        angle_err = _quat_angle(rel)
        r_orientation = -angle_err

        # Position tracking: keep the object near the palm centre.
        target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos[:, 2] = self.config.object_height
        pos_err = (self._state[:, 0:3] - target_pos).pow(2).sum(dim=-1)
        r_position = -pos_err

        r_velocity = -self._state[:, 7:10].pow(2).sum(dim=-1)
        r_angular = -self._state[:, 10:13].pow(2).sum(dim=-1)
        r_action = -actions.pow(2).sum(dim=-1)

        reward = (
            w.orientation * r_orientation
            + w.position * r_position
            + w.velocity * r_velocity
            + w.angular_velocity * r_angular
            + w.action_penalty * r_action
        )

        # Sparse success bonus when within the current curriculum tolerance.
        success = (angle_err < self.tolerance).float()
        reward = reward + w.success * success

        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _build_obs(self) -> torch.Tensor:
        """Construct the observation ``o_t`` for all environments."""
        aux = torch.cat(
            [self._prev_action, self._joint_pos[:, : AUX_DIM - HAND_DOF]]
            if AUX_DIM > HAND_DOF
            else [self._prev_action],
            dim=-1,
        )
        if aux.shape[-1] < AUX_DIM:
            pad = torch.zeros(
                self.num_envs, AUX_DIM - aux.shape[-1], device=self.device
            )
            aux = torch.cat([aux, pad], dim=-1)
        aux = aux[:, :AUX_DIM]

        obs = torch.cat(
            [
                self._joint_pos,  # q
                self._joint_vel,  # q_dot
                self._state[:, 0:3],  # x_t
                self._state[:, 7:10],  # v_t
                self._state[:, 10:13],  # omega_t
                self._goal_quat,  # g_t
                aux,  # z_t
            ],
            dim=-1,
        )
        return obs

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def _update_curriculum(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> None:
        """Shrink the success tolerance when the policy succeeds often."""
        if not self.config.use_curriculum:
            return

        obj_quat = self._state[:, 3:7]
        rel = _quat_mul(_quat_conjugate(obj_quat), self._goal_quat)
        angle_err = _quat_angle(rel)
        successes = (angle_err < self.tolerance).float()

        self._success_count += float(successes.sum().item())
        self._episode_count += float(dones.sum().item())

        if self._episode_count >= 1.0:
            avg_successes = self._success_count / max(self._episode_count, 1.0)
            if avg_successes > self.config.success_threshold:
                self.tolerance = max(
                    self.config.tolerance_end,
                    self.tolerance * self.config.tolerance_decay,
                )
                self._success_count = 0.0
                self._episode_count = 0.0

    # ------------------------------------------------------------------
    # Partial reset
    # ------------------------------------------------------------------
    def _reset_envs(self, mask: torch.Tensor) -> None:
        """Reset the environments selected by ``mask``."""
        n = int(mask.sum().item())
        if n == 0:
            return

        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = (
            torch.rand(n, generator=self._generator) * 2.0 - 1.0
        ) * self.config.object_pos_range
        pos[:, 1] = (
            torch.rand(n, generator=self._generator) * 2.0 - 1.0
        ) * self.config.object_pos_range
        pos[:, 2] = self.config.object_height

        quat = _random_quaternion(n, torch.device("cpu"), self._generator).to(
            self.device
        )

        self._state[mask] = 0.0
        self._state[mask, 0:3] = pos
        self._state[mask, 3:7] = quat
        self._goal_quat[mask] = _random_quaternion(
            n, torch.device("cpu"), self._generator
        ).to(self.device)
        self._joint_pos[mask] = (
            torch.rand(n, HAND_DOF, generator=self._generator).to(self.device) * 2.0
            - 1.0
        ) * 0.1
        self._joint_vel[mask] = 0.0
        self._prev_action[mask] = 0.0
        self._episode_step[mask] = 0

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def get_task_info(self) -> Dict[str, Any]:
        """Return a summary of the task configuration."""
        return {
            "task": self.task,
            "num_envs": self.num_envs,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "episode_length": self.max_episode_length,
            "tolerance": self.tolerance,
            "use_isaacgym": self._use_isaacgym,
        }

    def close(self) -> None:
        """Release simulation resources."""
        self._sim = None
        self._envs = None
        self._actor_handles = None
        self._object_handles = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_allegrohand_env(
    task: str = "reorientation",
    num_envs: int = 24576,
    device: str = "cuda:0",
    headless: bool = True,
    seed: int = 0,
    **kwargs: Any,
) -> AllegroHandTask:
    """Create an :class:`AllegroHandTask` environment."""
    config = AllegroHandConfig(
        task=task,
        num_envs=num_envs,
        device=device,
        headless=headless,
        seed=seed,
        **kwargs,
    )
    return AllegroHandTask(config)


__all__ = [
    "AllegroHandTask",
    "AllegroHandConfig",
    "AllegroHandRewardWeights",
    "make_allegrohand_env",
    "OBS_DIM",
    "HAND_DOF",
    "TASKS",
]
