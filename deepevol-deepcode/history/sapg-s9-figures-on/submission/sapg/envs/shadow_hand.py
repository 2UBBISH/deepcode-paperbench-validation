"""ShadowHand 24-DoF in-hand reorientation task for SAPG.

Implements the ShadowHand in-hand reorientation task described in the SAPG
paper (Section 5 / Appendix).  The task requires reorienting a cube (or
object) to a randomly sampled goal orientation using a 24-DoF ShadowHand.

Observation layout (paper Section 4.2 / Addendum):
    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]

where
    q       : joint positions          (24)
    q_dot   : joint velocities         (24)
    x_t     : object pose (pos+quat)   (7)
    v_t     : object linear velocity   (3)
    omega_t : object angular velocity  (3)
    g_t     : goal orientation (quat)  (4)
    z_t     : auxiliary features       (aux)

The module provides a lazy IsaacGym simulation path plus a pure-PyTorch
analytic fallback so the pipeline can be exercised on CPU without IsaacGym.

Hyperparameters (Table 3 of the paper):
    - MLP 512x512x256x128, ELU
    - phi_j in R^16
    - horizon H = 8
    - learning rate 5e-4
    - clip eps = 0.1
    - mini-epochs = 5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
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
HAND_DOF = 24
OBJECT_STATE_DIM = 13  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
GOAL_DIM = 4  # goal quaternion
AUX_DIM = 16  # auxiliary features (fingertip positions, contacts, etc.)
OBS_DIM = HAND_DOF + HAND_DOF + OBJECT_STATE_DIM + GOAL_DIM + AUX_DIM  # = 81

TASKS = ("reorientation",)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ShadowHandRewardWeights:
    """Reward weights for the ShadowHand reorientation task.

    Follows the PQL (Li et al., 2023) style reward decomposition used for
    in-hand reorientation: orientation tracking + position tracking +
    velocity penalties + action penalty.
    """

    orientation: float = 1.0
    position: float = 0.5
    velocity: float = 0.01
    angular_velocity: float = 0.01
    action_penalty: float = 0.0001
    torque_penalty: float = 1e-5
    success: float = 10.0


@dataclass
class ShadowHandConfig:
    """Configuration for the ShadowHand in-hand reorientation task."""

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
    goal_orientation_range: float = 1.0  # full random quaternion

    # Curriculum (success tolerance on orientation error, radians)
    use_curriculum: bool = True
    tolerance_start: float = 0.5
    tolerance_end: float = 0.1
    tolerance_decay: float = 0.90
    success_threshold: float = 3.0

    reward_weights: ShadowHandRewardWeights = field(
        default_factory=ShadowHandRewardWeights
    )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class ShadowHandTask:
    """Vectorized ShadowHand 24-DoF in-hand reorientation environment.

    Exposes the SAPG env contract: ``num_envs``, ``obs_dim``, ``act_dim``,
    ``reset()``, ``step(actions)`` and ``close()``.
    """

    def __init__(
        self,
        config: Optional[ShadowHandConfig] = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = ShadowHandConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        self.config = config

        self.num_envs = int(config.num_envs)
        self.device = torch.device(config.device)
        self.obs_dim = OBS_DIM
        self.act_dim = HAND_DOF

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(config.seed))

        # Simulation handle (lazy)
        self._sim = None
        self._use_isaacgym = False

        # State tensors (analytic fallback / mirror of sim state)
        self._q = torch.zeros(self.num_envs, HAND_DOF, device=self.device)
        self._q_dot = torch.zeros(self.num_envs, HAND_DOF, device=self.device)
        self._obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._obj_quat[:, 0] = 1.0
        self._obj_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._obj_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_quat[:, 0] = 1.0
        self._aux = torch.zeros(self.num_envs, AUX_DIM, device=self.device)

        self._episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._successes_this_episode = torch.zeros(
            self.num_envs, device=self.device
        )
        self._tolerance = float(config.tolerance_start)

        self._build_sim()

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
    # Simulation construction
    # ------------------------------------------------------------------
    def _build_sim(self) -> None:
        """Attempt to build the IsaacGym simulation; fall back to analytic."""
        if not _HAS_ISAACGYM:
            self._use_isaacgym = False
            return
        try:  # pragma: no cover - requires GPU + IsaacGym
            self._sim = self._create_isaacgym_sim()
            self._use_isaacgym = True
        except Exception:
            self._sim = None
            self._use_isaacgym = False

    def _create_isaacgym_sim(self):  # pragma: no cover - requires IsaacGym
        """Create the IsaacGym simulation for ShadowHand reorientation.

        This is intentionally defensive: any failure propagates to the caller
        which falls back to the analytic dynamics.
        """
        cfg = self.config
        gym = gymapi
        sim_params = gym.SimParams()
        sim_params.dt = cfg.dt
        sim_params.substeps = 2
        sim_params.up_axis = gym.UP_AXIS_Z
        sim_params.gravity = gym.Vec3(0.0, 0.0, -9.81)

        sim = gym.create_sim(0, 0, gym.SIM_PHYSX, sim_params)
        if sim is None:
            raise RuntimeError("Failed to create IsaacGym sim")

        plane_params = gym.PlaneParams()
        plane_params.normal = gym.Vec3(0.0, 0.0, 1.0)
        gym.add_ground(sim, plane_params)

        asset_root = ""
        hand_asset = "shadow_hand"
        hand_options = gym.AssetOptions()
        hand_options.fix_base_link = True
        hand_options.collapse_fixed_joints = True
        hand_handle = gym.load_asset(sim, asset_root, hand_asset, hand_options)

        object_options = gym.AssetOptions()
        object_options.density = 400.0
        object_handle = gym.load_asset(
            sim, asset_root, "cube_multicolor", object_options
        )

        envs = []
        for i in range(self.num_envs):
            env = gym.create_env(sim, gym.Vec3(-1, -1, 0), gym.Vec3(1, 1, 1), 1)
            gym.create_actor(env, hand_handle, gym.Transform(), "hand", i, 0)
            gym.create_actor(env, object_handle, gym.Transform(), "object", i, 0)
            envs.append(env)

        gym.prepare_sim(sim)
        return {
            "gym": gym,
            "sim": sim,
            "envs": envs,
            "hand_handle": hand_handle,
            "object_handle": object_handle,
        }

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset all environments and return the initial observation."""
        cfg = self.config
        n = self.num_envs

        self._q = 0.1 * torch.randn(
            n, HAND_DOF, device=self.device, generator=self._generator
        )
        self._q_dot = torch.zeros(n, HAND_DOF, device=self.device)

        self._obj_pos = torch.zeros(n, 3, device=self.device)
        self._obj_pos[:, 0] = cfg.object_pos_range * (
            torch.rand(n, device=self.device, generator=self._generator) - 0.5
        )
        self._obj_pos[:, 1] = cfg.object_pos_range * (
            torch.rand(n, device=self.device, generator=self._generator) - 0.5
        )
        self._obj_pos[:, 2] = cfg.object_height

        self._obj_quat = self._random_quaternion(n)
        self._obj_lin_vel = torch.zeros(n, 3, device=self.device)
        self._obj_ang_vel = torch.zeros(n, 3, device=self.device)

        self._goal_quat = self._random_quaternion(n)
        self._aux = torch.zeros(n, AUX_DIM, device=self.device)

        self._episode_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self._successes_this_episode = torch.zeros(n, device=self.device)

        return self._get_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Advance the environment by one control step.

        Args:
            actions: (num_envs, act_dim) target joint positions.

        Returns:
            obs, rewards, dones, infos
        """
        actions = actions.to(self.device).clamp(-1.0, 1.0)
        if self._use_isaacgym:  # pragma: no cover - requires IsaacGym
            self._isaacgym_step(actions)
        else:
            self._analytic_step(actions)

        rewards = self._compute_rewards(actions)
        self._episode_step += 1
        dones = (self._episode_step >= self.max_episode_length).float()

        infos: Dict = {}
        if bool(dones.any()):
            done_mask = dones > 0.5
            infos["episode"] = {
                "r": rewards[done_mask].mean().item(),
                "l": self._episode_step[done_mask].float().mean().item(),
            }
            self._reset_done(done_mask)

        return self._get_obs(), rewards, dones, infos

    def close(self) -> None:
        if self._use_isaacgym and self._sim is not None:  # pragma: no cover
            try:
                self._sim["gym"].destroy_sim(self._sim["sim"])
            except Exception:
                pass
        self._sim = None
        self._use_isaacgym = False

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _analytic_step(self, actions: torch.Tensor) -> None:
        """Pure-PyTorch damped dynamics used when IsaacGym is unavailable."""
        target_q = actions * math.pi
        self._q_dot = 0.5 * (target_q - self._q)
        self._q = self._q + 0.1 * self._q_dot

        # Object follows a damped random walk influenced by the hand.
        hand_effect = 0.01 * self._q[:, :3]
        self._obj_lin_vel = 0.9 * self._obj_lin_vel + 0.05 * hand_effect
        self._obj_pos = self._obj_pos + 0.01 * self._obj_lin_vel
        self._obj_pos[:, 2] = torch.clamp(
            self._obj_pos[:, 2], min=0.55, max=0.75
        )

        ang_effect = 0.02 * self._q[:, 3:6]
        self._obj_ang_vel = 0.9 * self._obj_ang_vel + 0.05 * ang_effect
        self._obj_quat = self._integrate_quat(self._obj_quat, self._obj_ang_vel)

    def _isaacgym_step(self, actions: torch.Tensor) -> None:  # pragma: no cover
        """Step the real IsaacGym simulation (requires GPU)."""
        gym = self._sim["gym"]
        sim = self._sim["sim"]
        envs = self._sim["envs"]
        for i, env in enumerate(envs):
            gym.set_dof_target_position(
                env, self._sim["hand_handle"], actions[i].cpu().numpy()
            )
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        # State refresh would read tensors via gymtorch; kept minimal here.

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _compute_rewards(self, actions: torch.Tensor) -> torch.Tensor:
        w = self.config.reward_weights

        # Orientation tracking: 1 - |<q_obj, q_goal>| (quaternion distance).
        quat_dot = torch.abs((self._obj_quat * self._goal_quat).sum(dim=-1))
        quat_dot = quat_dot.clamp(max=1.0)
        orientation_err = 1.0 - quat_dot
        r_orientation = w.orientation * (1.0 - orientation_err)

        # Position tracking: keep object near the palm.
        pos_err = (self._obj_pos[:, 2] - self.config.object_height).pow(2)
        r_position = w.position * (1.0 - pos_err)

        r_vel = -w.velocity * self._obj_lin_vel.pow(2).sum(dim=-1)
        r_ang_vel = -w.angular_velocity * self._obj_ang_vel.pow(2).sum(dim=-1)
        r_action = -w.action_penalty * actions.pow(2).sum(dim=-1)

        success = (orientation_err < self._tolerance).float()
        r_success = w.success * success
        self._successes_this_episode = (
            self._successes_this_episode + success
        )

        reward = (
            r_orientation
            + r_position
            + r_vel
            + r_ang_vel
            + r_action
            + r_success
        )
        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> torch.Tensor:
        return torch.cat(
            [
                self._q,
                self._q_dot,
                self._obj_pos,
                self._obj_quat,
                self._obj_lin_vel,
                self._obj_ang_vel,
                self._goal_quat,
                self._aux,
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_quaternion(self, n: int) -> torch.Tensor:
        """Sample uniformly random unit quaternions (w, x, y, z)."""
        u = torch.rand(n, 3, device=self.device, generator=self._generator)
        q = torch.stack(
            [
                torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]),
                torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
                torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2]),
                torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2]),
            ],
            dim=-1,
        )
        # Reorder to (w, x, y, z)
        q = q[:, [3, 0, 1, 2]]
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _integrate_quat(quat: torch.Tensor, ang_vel: torch.Tensor) -> torch.Tensor:
        """Integrate a quaternion by an angular velocity (small-angle)."""
        dt = 0.01
        w, x, y, z = quat.unbind(dim=-1)
        wx, wy, wz = ang_vel.unbind(dim=-1)
        dq = torch.stack(
            [
                -0.5 * (x * wx + y * wy + z * wz),
                0.5 * (w * wx + y * wz - z * wy),
                0.5 * (w * wy - x * wz + z * wx),
                0.5 * (w * wz + x * wy - y * wx),
            ],
            dim=-1,
        )
        quat = quat + dt * dq
        return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _reset_done(self, done_mask: torch.Tensor) -> None:
        """Reset environments that finished an episode."""
        n_done = int(done_mask.sum().item())
        if n_done == 0:
            return
        idx = done_mask.nonzero(as_tuple=False).squeeze(-1)

        # Curriculum: shrink tolerance when mean successes/episode > threshold.
        if self.config.use_curriculum:
            mean_success = float(self._successes_this_episode[done_mask].mean())
            if mean_success > self.config.success_threshold:
                self._tolerance = max(
                    self.config.tolerance_end,
                    self._tolerance * self.config.tolerance_decay,
                )

        self._q[idx] = 0.1 * torch.randn(
            n_done, HAND_DOF, device=self.device, generator=self._generator
        )
        self._q_dot[idx] = 0.0
        self._obj_pos[idx, 0] = self.config.object_pos_range * (
            torch.rand(n_done, device=self.device, generator=self._generator) - 0.5
        )
        self._obj_pos[idx, 1] = self.config.object_pos_range * (
            torch.rand(n_done, device=self.device, generator=self._generator) - 0.5
        )
        self._obj_pos[idx, 2] = self.config.object_height
        self._obj_quat[idx] = self._random_quaternion(n_done)
        self._obj_lin_vel[idx] = 0.0
        self._obj_ang_vel[idx] = 0.0
        self._goal_quat[idx] = self._random_quaternion(n_done)
        self._aux[idx] = 0.0
        self._episode_step[idx] = 0
        self._successes_this_episode[idx] = 0.0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_task_info(self) -> Dict:
        return {
            "task": self.config.task,
            "num_envs": self.num_envs,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "episode_length": self.max_episode_length,
            "tolerance": self._tolerance,
            "use_isaacgym": self._use_isaacgym,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_shadowhand_env(
    task: str = "reorientation",
    num_envs: int = 24576,
    device: str = "cuda:0",
    headless: bool = True,
    seed: int = 0,
    **kwargs,
) -> ShadowHandTask:
    """Create a ShadowHand in-hand reorientation environment."""
    config = ShadowHandConfig(
        task=task,
        num_envs=num_envs,
        device=device,
        headless=headless,
        seed=seed,
        **kwargs,
    )
    return ShadowHandTask(config)


__all__ = [
    "ShadowHandTask",
    "ShadowHandConfig",
    "ShadowHandRewardWeights",
    "make_shadowhand_env",
    "OBS_DIM",
    "HAND_DOF",
    "TASKS",
]
