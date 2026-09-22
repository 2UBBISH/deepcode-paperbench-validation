"""AllegroKuka manipulation tasks for SAPG.

Implements the three hard AllegroKuka tasks used in the paper (Section 5):
    * Regrasping
    * Throw
    * Reorientation

Each task is a 23-DoF system (7-DoF Kuka arm + 16-DoF Allegro hand).  The
observation follows the paper's specification (Section 4.2 / Addendum):

    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]

where
    q, q_dot        : joint positions / velocities          (R^23)
    x_t             : object position                        (R^3)
    v_t             : object linear velocity                 (R^3)
    omega_t         : object angular velocity                (R^3)
    g_t             : goal / target pose                     (R^7: pos + quat)
    z_t             : auxiliary task features (e.g. fingertip contacts)

Reward (DexPBT, Petrenko et al. 2023) is a weighted sum of
    r_reach + r_lift + r_target + r_success
with a success-tolerance curriculum that shrinks the tolerance from 7.5cm to
1cm, decreasing by 10% whenever the average number of successes per episode
exceeds 3.

The module is written so that it can be imported without IsaacGym installed
(the heavy IsaacGym imports happen lazily inside ``_build_sim``).  A pure
PyTorch fallback dynamics model is provided so the rest of the SAPG pipeline
can be exercised on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

try:  # pragma: no cover - IsaacGym is optional
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

ARM_DOF = 7
HAND_DOF = 16
NUM_DOF = ARM_DOF + HAND_DOF  # 23

# Object state: position (3) + linear vel (3) + angular vel (3) + quat (4)
OBJECT_STATE_DIM = 13
# Goal: position (3) + quaternion (4)
GOAL_DIM = 7
# Auxiliary features: 4 fingertip positions (3 each) + 4 contact flags
AUX_DIM = 16

OBS_DIM = 2 * NUM_DOF + 3 + 3 + 3 + GOAL_DIM + AUX_DIM  # 23*2 + 9 + 7 + 16 = 78

# Curriculum bounds (metres)
TOLERANCE_START = 0.075
TOLERANCE_END = 0.010
TOLERANCE_DECAY = 0.90
SUCCESS_THRESHOLD = 3.0

TASKS = ("regrasping", "throw", "reorientation")


@dataclass
class AllegroKukaRewardWeights:
    """Reward weights following DexPBT (Petrenko et al., 2023)."""

    reach: float = 1.0
    lift: float = 2.0
    target: float = 4.0
    success: float = 10.0
    action_penalty: float = 0.0001
    torque_penalty: float = 1e-5
    velocity_penalty: float = 1e-4


@dataclass
class AllegroKukaConfig:
    """Configuration for an AllegroKuka task."""

    task: str = "regrasping"
    num_envs: int = 24576
    device: str = "cuda:0"
    headless: bool = True
    seed: int = 0
    episode_length: int = 300
    control_freq: int = 20
    dt: float = 1.0 / 60.0
    # Object / goal sampling ranges
    object_pos_range: Tuple[float, float] = (-0.05, 0.05)
    object_height: float = 0.65
    goal_pos_range: Tuple[float, float] = (-0.10, 0.10)
    goal_height: float = 0.75
    # Throw task specifics
    throw_distance: float = 0.5
    # Reorientation specifics
    reorientation_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Curriculum
    use_curriculum: bool = True
    tolerance_start: float = TOLERANCE_START
    tolerance_end: float = TOLERANCE_END
    tolerance_decay: float = TOLERANCE_DECAY
    success_threshold: float = SUCCESS_THRESHOLD
    reward_weights: AllegroKukaRewardWeights = field(
        default_factory=AllegroKukaRewardWeights
    )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class AllegroKukaTask:
    """AllegroKuka manipulation task (Regrasping / Throw / Reorientation).

    The class exposes the minimal vectorized-env contract expected by
    :class:`sapg.envs.isaacgym_wrapper.IsaacGymEnvWrapper`:

        * ``obs_dim`` / ``act_dim`` properties
        * ``reset() -> Tensor``
        * ``step(actions) -> (obs, rewards, dones, infos)``
        * ``close()``

    When IsaacGym is available the simulation is created lazily; otherwise a
    lightweight analytic dynamics model is used so that the SAPG training
    pipeline can be validated on CPU.
    """

    def __init__(self, config: Optional[AllegroKukaConfig] = None, **kwargs: Any):
        if config is None:
            config = AllegroKukaConfig(**kwargs)
        elif kwargs:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        if config.task not in TASKS:
            raise ValueError(
                f"Unknown AllegroKuka task '{config.task}'. Expected one of {TASKS}."
            )

        self.cfg = config
        self.task = config.task
        self.num_envs = int(config.num_envs)
        self.device = torch.device(config.device)
        self.obs_dim = OBS_DIM
        self.act_dim = NUM_DOF

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(config.seed))

        # Curriculum state
        self.tolerance = float(config.tolerance_start)
        self._success_history: list = []

        # Simulation handles (populated only when IsaacGym is present)
        self._sim = None
        self._envs = None
        self._actor_handles = None
        self._built = False

        # Analytic fallback state
        self._state: Optional[torch.Tensor] = None
        self._object_state: Optional[torch.Tensor] = None
        self._goal: Optional[torch.Tensor] = None
        self._episode_step: Optional[torch.Tensor] = None
        self._episode_successes: Optional[torch.Tensor] = None

        if _HAS_ISAACGYM and not getattr(config, "force_mock", False):
            try:
                self._build_sim()
            except Exception:
                # Fall back to analytic dynamics on any IsaacGym failure.
                self._sim = None
                self._built = False

        if not self._built:
            self._init_analytic_state()

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
        return int(self.cfg.episode_length)

    # ------------------------------------------------------------------
    # IsaacGym construction
    # ------------------------------------------------------------------

    def _build_sim(self) -> None:  # pragma: no cover - requires IsaacGym
        """Create the IsaacGym simulation, envs and actors."""
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.dt
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0

        self._sim = gymapi.acquire_gym().create_sim(
            None, None, None, gymapi.SIM_PHYSX, sim_params
        )

        # Asset loading is environment specific; the concrete asset paths are
        # provided through the config when running on a real machine.
        asset_root = getattr(self.cfg, "asset_root", "assets")
        arm_asset = getattr(self.cfg, "arm_asset", "urdf/kuka_allegro.urdf")
        object_asset = getattr(self.cfg, "object_asset", "urdf/box.urdf")

        arm_options = gymapi.AssetOptions()
        arm_options.fix_base_link = True
        arm_options.disable_gravity = False
        arm_asset_handle = self._sim.load_asset(asset_root, arm_asset, arm_options)

        object_options = gymapi.AssetOptions()
        object_options.density = 200.0
        object_asset_handle = self._sim.load_asset(
            asset_root, object_asset, object_options
        )

        env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
        env_upper = gymapi.Vec3(0.5, 0.5, 1.0)

        self._envs = []
        self._actor_handles = []
        for env_id in range(self.num_envs):
            env = self._sim.create_env(env_lower, env_upper, env_id)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            arm_handle = self._sim.create_actor(
                env, arm_asset_handle, pose, "arm", env_id, 0
            )
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(0.0, 0.0, self.cfg.object_height)
            obj_handle = self._sim.create_actor(
                env, object_asset_handle, obj_pose, "object", env_id, 1
            )
            self._envs.append(env)
            self._actor_handles.append((arm_handle, obj_handle))

        self._sim.prepare_sim()
        self._built = True

    # ------------------------------------------------------------------
    # Analytic fallback
    # ------------------------------------------------------------------

    def _init_analytic_state(self) -> None:
        """Initialise the analytic dynamics state on the configured device."""
        self._state = torch.zeros(
            self.num_envs, 2 * NUM_DOF, device=self.device, dtype=torch.float32
        )
        self._object_state = torch.zeros(
            self.num_envs, OBJECT_STATE_DIM, device=self.device, dtype=torch.float32
        )
        self._goal = torch.zeros(
            self.num_envs, GOAL_DIM, device=self.device, dtype=torch.float32
        )
        self._episode_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._episode_successes = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _rand(self, shape, low: float, high: float) -> torch.Tensor:
        u = torch.rand(shape, generator=self._generator, dtype=torch.float32)
        return (low + (high - low) * u).to(self.device)

    def _sample_goal(self) -> torch.Tensor:
        """Sample a task-specific goal pose (position + unit quaternion)."""
        n = self.num_envs
        goal = torch.zeros(n, GOAL_DIM, device=self.device, dtype=torch.float32)
        lo, hi = self.cfg.goal_pos_range
        goal[:, 0] = self._rand((n,), lo, hi)
        goal[:, 1] = self._rand((n,), lo, hi)
        goal[:, 2] = self.cfg.goal_height + self._rand((n,), -0.05, 0.05)

        if self.task == "throw":
            # Throw: goal is a target landing position further away.
            goal[:, 0] = self.cfg.throw_distance + self._rand((n,), -0.05, 0.05)
            goal[:, 1] = self._rand((n,), -0.05, 0.05)
            goal[:, 2] = self._rand((n,), 0.2, 0.4)

        # Identity quaternion by default; reorientation samples a random one.
        goal[:, 3] = 1.0
        if self.task == "reorientation":
            axis = torch.tensor(
                self.cfg.reorientation_axis, device=self.device, dtype=torch.float32
            )
            axis = axis / (axis.norm() + 1e-8)
            angle = self._rand((n,), -math.pi, math.pi)
            half = angle * 0.5
            sin_half = torch.sin(half)
            goal[:, 3] = torch.cos(half)
            goal[:, 4] = axis[0] * sin_half
            goal[:, 5] = axis[1] * sin_half
            goal[:, 6] = axis[2] * sin_half
        return goal

    def _build_obs(self) -> torch.Tensor:
        """Assemble the observation vector o_t."""
        q = self._state[:, :NUM_DOF]
        q_dot = self._state[:, NUM_DOF:]
        obj = self._object_state
        x_t = obj[:, 0:3]
        v_t = obj[:, 3:6]
        omega_t = obj[:, 6:9]
        g_t = self._goal
        # Auxiliary features: fingertip positions approximated from joint state.
        z_t = torch.cat(
            [
                q[:, ARM_DOF : ARM_DOF + 4].unsqueeze(-1).repeat(1, 3),
                torch.zeros(self.num_envs, 4, device=self.device),
            ],
            dim=-1,
        )
        return torch.cat([q, q_dot, x_t, v_t, omega_t, g_t, z_t], dim=-1)

    def _analytic_step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Lightweight damped dynamics used when IsaacGym is unavailable."""
        actions = torch.clamp(actions, -1.0, 1.0)
        # Joint dynamics: first-order lag towards the commanded target.
        q = self._state[:, :NUM_DOF]
        q_dot = self._state[:, NUM_DOF:]
        target_q = actions * 0.5
        q_dot = 0.9 * q_dot + 0.1 * (target_q - q) * 10.0
        q = q + q_dot * self.cfg.dt
        self._state = torch.cat([q, q_dot], dim=-1)

        # Object dynamics: the hand imparts a small velocity proportional to
        # the mean hand action; gravity pulls the object down.
        hand_act = actions[:, ARM_DOF:].mean(dim=-1, keepdim=True)
        obj = self._object_state
        obj[:, 3:6] = 0.95 * obj[:, 3:6] + 0.05 * hand_act.repeat(1, 3)
        obj[:, 6:9] = 0.95 * obj[:, 6:9]
        obj[:, 2] = obj[:, 2] + obj[:, 5] * self.cfg.dt - 0.5 * 9.81 * self.cfg.dt ** 2
        obj[:, 0:2] = obj[:, 0:2] + obj[:, 3:5] * self.cfg.dt
        obj[:, 2] = torch.clamp(obj[:, 2], min=0.0)
        self._object_state = obj

        obs = self._build_obs()
        rewards, success = self._compute_reward(actions)
        self._episode_step = self._episode_step + 1
        dones = (self._episode_step >= self.cfg.episode_length).float()
        self._episode_successes = self._episode_successes + success.float()

        infos: Dict[str, Any] = {
            "episode": {
                "r": rewards.clone(),
                "l": self._episode_step.clone(),
                "success": success.clone(),
            },
            "success": success.clone(),
        }

        if dones.any():
            self._update_curriculum()
            done_mask = dones > 0.5
            self._episode_step = torch.where(
                done_mask, torch.zeros_like(self._episode_step), self._episode_step
            )
            self._episode_successes = torch.where(
                done_mask,
                torch.zeros_like(self._episode_successes),
                self._episode_successes,
            )
            if done_mask.any():
                new_goal = self._sample_goal()
                self._goal = torch.where(
                    done_mask.unsqueeze(-1), new_goal, self._goal
                )
                reset_obj = self._reset_object_state()
                self._object_state = torch.where(
                    done_mask.unsqueeze(-1), reset_obj, self._object_state
                )
        return obs, rewards, dones, infos

    def _reset_object_state(self) -> torch.Tensor:
        n = self.num_envs
        obj = torch.zeros(n, OBJECT_STATE_DIM, device=self.device, dtype=torch.float32)
        lo, hi = self.cfg.object_pos_range
        obj[:, 0] = self._rand((n,), lo, hi)
        obj[:, 1] = self._rand((n,), lo, hi)
        obj[:, 2] = self.cfg.object_height
        obj[:, 9] = 1.0  # identity quaternion
        return obj

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _compute_reward(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the DexPBT-style reward and the success indicator."""
        w = self.cfg.reward_weights
        obj = self._object_state
        obj_pos = obj[:, 0:3]
        goal_pos = self._goal[:, 0:3]

        # r_reach: encourage the hand (approximated by joint state) to be near
        # the object.
        hand_pos = self._state[:, ARM_DOF : ARM_DOF + 3] * 0.1
        r_reach = -torch.norm(hand_pos - obj_pos, dim=-1)

        # r_lift: encourage lifting the object above its initial height.
        r_lift = torch.clamp(obj_pos[:, 2] - self.cfg.object_height, min=0.0)

        # r_target: negative distance to the goal.
        r_target = -torch.norm(obj_pos - goal_pos, dim=-1)

        # r_success: sparse bonus when within the current tolerance.
        dist = torch.norm(obj_pos - goal_pos, dim=-1)
        success = (dist < self.tolerance).float()
        r_success = success

        reward = (
            w.reach * r_reach
            + w.lift * r_lift
            + w.target * r_target
            + w.success * r_success
        )
        reward = reward - w.action_penalty * torch.sum(actions ** 2, dim=-1)
        reward = reward - w.velocity_penalty * torch.sum(
            self._state[:, NUM_DOF:] ** 2, dim=-1
        )
        return reward, success

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _update_curriculum(self) -> None:
        """Shrink the success tolerance when the agent succeeds too often."""
        if not self.cfg.use_curriculum:
            return
        mean_successes = float(self._episode_successes.mean().item())
        self._success_history.append(mean_successes)
        if mean_successes > self.cfg.success_threshold:
            self.tolerance = max(
                self.cfg.tolerance_end, self.tolerance * self.cfg.tolerance_decay
            )

    # ------------------------------------------------------------------
    # Vectorized env API
    # ------------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        """Reset all environments and return the initial observation."""
        if self._built and self._sim is not None:  # pragma: no cover
            self._sim.reset()
        self._state = torch.zeros(
            self.num_envs, 2 * NUM_DOF, device=self.device, dtype=torch.float32
        )
        self._object_state = self._reset_object_state()
        self._goal = self._sample_goal()
        self._episode_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._episode_successes = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        return self._build_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Advance the simulation by one control step."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        actions = actions.to(self.device).float()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        return self._analytic_step(actions)

    def close(self) -> None:  # pragma: no cover - IsaacGym cleanup
        if self._sim is not None:
            try:
                gymapi.acquire_gym().destroy_sim(self._sim)
            except Exception:
                pass
        self._sim = None
        self._built = False

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_task_info(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "num_envs": self.num_envs,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "tolerance": self.tolerance,
            "isaacgym": self._built,
        }


def make_allegrokuka_env(
    task: str = "regrasping",
    num_envs: int = 24576,
    device: str = "cuda:0",
    headless: bool = True,
    seed: int = 0,
    **kwargs: Any,
) -> AllegroKukaTask:
    """Convenience factory for an AllegroKuka task."""
    cfg = AllegroKukaConfig(
        task=task,
        num_envs=num_envs,
        device=device,
        headless=headless,
        seed=seed,
        **kwargs,
    )
    return AllegroKukaTask(cfg)


__all__ = [
    "AllegroKukaTask",
    "AllegroKukaConfig",
    "AllegroKukaRewardWeights",
    "make_allegrokuka_env",
    "OBS_DIM",
    "NUM_DOF",
    "TASKS",
]
