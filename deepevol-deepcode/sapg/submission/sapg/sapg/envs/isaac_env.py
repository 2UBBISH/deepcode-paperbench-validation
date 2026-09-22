"""IsaacGym parallel-simulation wrapper for the SAPG task suite.

This module provides the vectorised environment interface used by the SAPG
training loop (Algorithm 1, Section 4.6).  All five tasks of the paper
(Appendix A) are exposed through a single ``IsaacEnv`` object with the minimal
interface expected by ``sapg.algorithms.rollout``::

    obs = env.reset()                                  # [N, obs_dim]
    obs, reward, done, infos = env.step(actions)       # actions [N, action_dim]

The environments follow the *auto-reset* convention of GPU vectorised
simulators (IsaacGym, `rl_games`): when an environment terminates it is
immediately reset in place, the returned ``done`` flag is ``True`` for that
environment and the returned observation is the first observation of the new
episode.  ``sapg.algorithms.rollout`` masks the bootstrap value for those
environments.

Two backends are supported:

* ``IsaacGymParallelEnv`` -- the real GPU simulation, built through
  ``isaacgymenvs.tasks.isaacgym_task_map`` (the DexPBT / AllegroKuka task
  suite of Petrenko et al., 2023).  Used when ``isaacgym`` and
  ``isaacgymenvs`` are importable and the user did not disable them.
* ``SurrogateVectorEnv`` -- a dependency-free (pure ``torch``) batched
  surrogate implementing the observation layout, reward decomposition and
  success criteria described in Appendix A.  It exists so that the full SAPG
  code path (splitting blocks, collecting rollouts, importance-sampled updates)
  is runnable and testable without a GPU simulator.  It is a *surrogate*: the
  rigid-body simulator is replaced by light-weight kinematic bookkeeping.

Observation layout (Appendix A)::

    o_t = [q, qdot, x_t, v_t, omega_t, g_t, z_t]

with ``q, qdot in R^23`` for the Allegro-Kuka tasks (Allegro 16 DoF + Kuka
7 DoF), ``x_t in R^7`` the object pose (position + quaternion), ``v_t`` and
``omega_t`` the object linear/angular velocities in ``R^3``, ``g_t`` the
task-dependent goal (``R^3`` for regrasping/throw, ``R^7`` for reorientation,
``R^4`` quaternion for the in-hand reorientation tasks) and ``z_t`` auxiliary
information (e.g. whether the object has been lifted).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - exercised only on machines with IsaacGym installed
    import isaacgym  # noqa: F401  (must be imported before torch in some builds)

    _HAS_ISAACGYM = True
except Exception:  # pragma: no cover - the common case for CI / unit tests
    _HAS_ISAACGYM = False

try:  # pragma: no cover
    import isaacgymenvs  # noqa: F401

    _HAS_ISAACGYMENVS = True
except Exception:  # pragma: no cover
    _HAS_ISAACGYMENVS = False

HAS_ISAACGYM = _HAS_ISAACGYM and _HAS_ISAACGYMENVS

# ---------------------------------------------------------------------------
# Task metadata (Appendix A, Table 1 of the reproduction plan)
# ---------------------------------------------------------------------------

#: Number of actuated joints.
JOINT_DIMS: Dict[str, int] = {
    "regrasping": 23,
    "regrasp": 23,
    "throw": 23,
    "reorientation": 23,
    "allegro_kuka": 23,
    "shadow_hand": 24,
    "allegro_hand": 16,
}

#: Default goal-observation dimension.
GOAL_DIMS: Dict[str, int] = {
    "regrasping": 3,
    "regrasp": 3,
    "throw": 3,
    "reorientation": 7,
    "allegro_kuka": 3,
    "shadow_hand": 4,
    "allegro_hand": 4,
}

#: Task group / IsaacGym task name / task kind.
TASK_REGISTRY: Dict[str, Dict[str, str]] = {
    "regrasping": {"group": "allegro_kuka", "isaac_task": "AllegroKukaRegrasping", "kind": "reach"},
    "regrasp": {"group": "allegro_kuka", "isaac_task": "AllegroKukaRegrasping", "kind": "reach"},
    "throw": {"group": "allegro_kuka", "isaac_task": "AllegroKukaThrow", "kind": "throw"},
    "reorientation": {"group": "allegro_kuka", "isaac_task": "AllegroKukaReorientation", "kind": "orient"},
    "shadow_hand": {"group": "shadow_hand", "isaac_task": "ShadowHand", "kind": "hand"},
    "allegro_hand": {"group": "allegro_hand", "isaac_task": "AllegroHand", "kind": "hand"},
}

#: Object diameter / relative-pose dimension in ``x_t``.
OBJECT_POSE_DIM = 7

#: Auxiliary information ``z_t`` (1 = "object has been lifted" flag).
EXTRA_DIMS: Dict[str, int] = {
    "allegro_kuka": 1,
    "shadow_hand": 0,
    "allegro_hand": 0,
}


def task_group_of(task: str) -> str:
    """Return the ``allegro_kuka`` / ``shadow_hand`` / ``allegro_hand`` group."""
    key = str(task).lower()
    if key in TASK_REGISTRY:
        return TASK_REGISTRY[key]["group"]
    if key in ("allegro_kuka", "kuka"):
        return "allegro_kuka"
    if key in ("shadow", "shadowhand", "shadow_hand"):
        return "shadow_hand"
    if key in ("allegro", "allegrohand", "allegro_hand"):
        return "allegro_hand"
    raise ValueError(f"Unknown task {task!r}; known tasks: {sorted(TASK_REGISTRY)}")


def joint_dim_for(task: str) -> int:
    return JOINT_DIMS.get(str(task).lower(), JOINT_DIMS[task_group_of(task)])


def goal_dim_for(task: str) -> int:
    key = str(task).lower()
    if key in GOAL_DIMS:
        return GOAL_DIMS[key]
    return 4  # in-hand reorientation quaternion goal


def action_dim_for(task: str) -> int:
    """Action space dimension equals the number of actuated joints."""
    return joint_dim_for(task)


def obs_dim_for(task: str) -> int:
    """Observation dimension implied by Appendix A for the given task."""
    group = task_group_of(task)
    j = joint_dim_for(task)
    return 2 * j + OBJECT_POSE_DIM + 3 + 3 + goal_dim_for(task) + EXTRA_DIMS[group]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """Configuration of one vectorised environment instance.

    Mirrors the relevant fields of :class:`sapg.utils.config.SAPGConfig` so that
    ``make_env(config)`` can be called with either object.
    """

    task: str = "regrasping"
    num_envs: int = 24576
    device: str = "cuda:0"
    seed: int = 0

    # --- simulation -------------------------------------------------------
    use_isaacgym: bool = True
    headless: bool = True
    use_gpu_pipeline: bool = True
    substeps: int = 2
    control_dt: float = 1.0 / 60.0
    max_episode_length: int = 250
    randomize: bool = True
    observation_noise: float = 0.0
    clip_actions: float = 1.0
    clip_obs: float = 10.0

    # --- dimensions (None => derived from Appendix A) ---------------------
    obs_dim: Optional[int] = None
    action_dim: Optional[int] = None
    goal_dim: Optional[int] = None

    # --- task specific ----------------------------------------------------
    num_successes: int = 3              # curriculum trigger (Appendix A)
    success_tolerance: float = 0.075    # 7.5 cm, curriculum start
    min_success_tolerance: float = 0.01  # 1 cm, curriculum end
    hold_steps: int = 30                # K = 30 steps (regrasping)
    table_height: float = 0.0
    object_size: float = 0.05
    bucket_radius: float = 0.15
    reward_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_any(cls, config: Any = None, **overrides: Any) -> "EnvConfig":
        """Build an :class:`EnvConfig` from a SAPG config object / dict / kwargs."""
        data: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if config is not None and hasattr(config, key):
                data[key] = getattr(config, key)
            elif isinstance(config, dict) and key in config:
                data[key] = config[key]
        # ``env_name`` in the SAPG config encodes the task.
        if config is not None and not data.get("task"):
            data["task"] = getattr(config, "task", None) or getattr(config, "env_name", "regrasping")
        data.update({k: v for k, v in overrides.items() if v is not None})
        env_cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]
        if env_cfg.obs_dim is None:
            env_cfg.obs_dim = obs_dim_for(env_cfg.task)
        if env_cfg.action_dim is None:
            env_cfg.action_dim = action_dim_for(env_cfg.task)
        if env_cfg.goal_dim is None:
            # allow the config's obs_dim to override the default goal width
            base = 2 * joint_dim_for(env_cfg.task) + OBJECT_POSE_DIM + 6
            extra = EXTRA_DIMS[task_group_of(env_cfg.task)]
            derived = int(env_cfg.obs_dim) - base - extra
            env_cfg.goal_dim = derived if derived >= 1 else goal_dim_for(env_cfg.task)
        if not env_cfg.reward_weights:
            env_cfg.reward_weights = default_reward_weights(env_cfg.task)
        return env_cfg


def default_reward_weights(task: str) -> Dict[str, float]:
    """Reward decomposition of Appendix A (r_reach, r_lift, r_target, r_success).

    Weights follow the DexPBT / AllegroKuka codebase (Petrenko et al., 2023)
    which the paper's tasks are derived from.
    """
    group = task_group_of(task)
    if group == "allegro_kuka":
        weights = {
            "reach": 1.0,
            "lift": 1.0,
            "target": 1.0,
            "success": 5.0,
            "orient": 1.0,
            "drop": 0.0,
        }
        if str(task).lower() == "throw":
            weights.update({"target": 1.0, "success": 10.0})
        if str(task).lower() == "reorientation":
            weights.update({"orient": 2.0})
        return weights
    # in-hand reorientation: orientation error + success bonus
    return {"orient": 1.0, "success": 5.0, "reach": 0.0, "lift": 0.0, "target": 0.0, "drop": 0.0}


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention, as in IsaacGym)
# ---------------------------------------------------------------------------


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_angle_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between two unit quaternions, shape ``[...]``."""
    d = quat_mul(a, quat_conjugate(b)).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d[..., 0].clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def quat_random(n: int, device: Any, generator: Optional[torch.Generator] = None,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    u = torch.rand(n, 3, device=device, dtype=dtype, generator=generator)
    q = torch.stack(
        [
            torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]),
            torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
            torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2]),
            torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2]),
        ],
        dim=-1,
    )
    return quat_normalize(q)


# ---------------------------------------------------------------------------
# Real IsaacGym backend
# ---------------------------------------------------------------------------


def default_isaac_task_cfg(env_cfg: EnvConfig) -> Dict[str, Any]:
    """Default IsaacGym task configuration (port of the DexPBT configs)."""
    group = task_group_of(env_cfg.task)
    cfg: Dict[str, Any] = {
        "env": {
            "numEnvs": int(env_cfg.num_envs),
            "envSpacing": 1.5,
            "episodeLength": int(env_cfg.max_episode_length),
            "enableDebugVis": False,
            "clipObservations": float(env_cfg.clip_obs),
            "clipActions": float(env_cfg.clip_actions),
            "controlFrequencyInv": 2,
            "numSuccesses": int(env_cfg.num_successes),
            "successTolerance": float(env_cfg.success_tolerance),
            "printNumSuccesses": False,
            "maxConsecutiveSuccesses": int(env_cfg.hold_steps),
            "resetPositionNoise": 0.01 if env_cfg.randomize else 0.0,
            "resetOrientationNoise": 0.1 if env_cfg.randomize else 0.0,
        },
        "sim": {"dt": env_cfg.control_dt / max(1, env_cfg.substeps), "substeps": int(env_cfg.substeps),
                "up_axis": "z", "use_gpu_pipeline": bool(env_cfg.use_gpu_pipeline)},
        "task": {"randomTolerance": 0.0, "useRelativeGoal": False},
        "domain_randomization": {"randomize": bool(env_cfg.randomize)},
    }
    if group == "allegro_kuka":
        cfg["task"].update({"randomTolerance": 0.0, "useRelativeGoal": False})
        cfg["env"].update({"numSuccesses": int(env_cfg.num_successes)})
    return cfg


class IsaacGymParallelEnv:
    """Thin adapter around an IsaacGymEnvs ``VecTask``.

    Instantiated through ``isaacgymenvs.tasks.isaacgym_task_map`` following the
    ``create_rlgpu_env`` pattern used by the DexPBT / AllegroKuka codebases.
    """

    def __init__(self, env_cfg: EnvConfig):
        if not HAS_ISAACGYM:  # pragma: no cover - requires the GPU simulator
            raise ImportError(
                "isaacgym / isaacgymenvs not importable; use the surrogate backend "
                "(EnvConfig(use_isaacgym=False)) or install IsaacGym."
            )
        from isaacgymenvs.tasks import isaacgym_task_map  # type: ignore  # noqa: WPS433

        task = str(env_cfg.task).lower()
        isaac_task = TASK_REGISTRY.get(task, {}).get("isaac_task", "AllegroKukaRegrasping")
        if isaac_task not in isaacgym_task_map:  # pragma: no cover
            raise KeyError(
                f"IsaacGym task {isaac_task!r} not registered; available: "
                f"{sorted(isaacgym_task_map)}"
            )
        self._cfg = default_isaac_task_cfg(env_cfg)
        self._device = env_cfg.device
        self._gpu = torch.device(env_cfg.device)
        graphics_device_id = 0 if torch.cuda.is_available() and env_cfg.device.startswith("cuda") else -1
        self._task = isaacgym_task_map[isaac_task](
            self._cfg,
            self._device,
            self._device,
            graphics_device_id,
            bool(env_cfg.headless),
        )
        self.num_envs = int(self._task.num_envs)
        self.num_obs = int(self._task.num_obs)
        self.num_actions = int(self._task.num_actions)
        self.device = self._gpu
        self.env_cfg = env_cfg
        self._episode_returns = torch.zeros(self.num_envs, device=self._gpu)
        self._episode_lengths = torch.zeros(self.num_envs, device=self._gpu)

    # -- interface ---------------------------------------------------------
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        obs = self._task.reset(env_ids)
        if env_ids is None:
            self._episode_returns.zero_()
            self._episode_lengths.zero_()
        else:
            self._episode_returns[env_ids] = 0.0
            self._episode_lengths[env_ids] = 0.0
        return obs.to(self._gpu).float()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        actions = actions.to(self._gpu).float()
        obs, reward, done, info = self._task.step(actions)
        reward = reward.to(self._gpu).float().view(-1)
        done = done.to(self._gpu).float().view(-1)
        self._episode_returns += reward
        self._episode_lengths += 1.0
        info = dict(info or {})
        info.setdefault("episode_return", self._episode_returns.clone())
        info.setdefault("episode_length", self._episode_lengths.clone())
        if "successes" in info:  # hard-task success metric (Appendix A)
            info["successes"] = torch.as_tensor(info["successes"], device=self._gpu).float().view(-1)
        reset_mask = done > 0.0
        if reset_mask.any():
            self._episode_returns[reset_mask] = 0.0
            self._episode_lengths[reset_mask] = 0.0
        return obs.to(self._gpu).float(), reward, done, info

    def set_success_tolerance(self, tolerance: float) -> None:
        if hasattr(self._task, "success_tolerance"):
            try:
                self._task.success_tolerance = float(tolerance)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                pass

    def close(self) -> None:  # pragma: no cover
        try:
            self._task.__del__()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Surrogate (pure torch) backend
# ---------------------------------------------------------------------------


class SurrogateVectorEnv:
    """Batched, dependency-free surrogate of the five IsaacGym tasks.

    Implements the observation layout, reward decomposition and success
    criteria of Appendix A using light-weight kinematic bookkeeping instead of
    a rigid-body simulator, so that the SAPG training loop is exercisable
    without CUDA/IsaacGym.  The environment API is identical to
    :class:`IsaacGymParallelEnv`.
    """

    def __init__(self, env_cfg: EnvConfig):
        self.env_cfg = env_cfg
        self.task = str(env_cfg.task).lower()
        self.group = task_group_of(self.task)
        self.kind = TASK_REGISTRY.get(self.task, {}).get("kind", "reach")
        self.device = torch.device(env_cfg.device)
        self.num_envs = int(env_cfg.num_envs)
        self.joint_dim = joint_dim_for(self.task)
        self.goal_dim = int(env_cfg.goal_dim or goal_dim_for(self.task))
        self.extra_dim = EXTRA_DIMS[self.group]
        self.num_obs = (
            2 * self.joint_dim + OBJECT_POSE_DIM + 6 + self.goal_dim + self.extra_dim
        )
        self.num_actions = self.joint_dim
        self.dt = float(env_cfg.control_dt)
        self.max_episode_length = int(env_cfg.max_episode_length)
        self.success_tolerance = float(env_cfg.success_tolerance)
        self.hold_steps = int(env_cfg.hold_steps)
        self.weights = dict(env_cfg.reward_weights or default_reward_weights(self.task))
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(env_cfg.seed))
        self._joint_gain = 10.0
        self._grasp_radius = 0.12
        self._table_height = float(env_cfg.table_height)

        # Fixed (seeded) linear map q -> approximate hand centre (forward kinematics proxy).
        g = torch.Generator(device="cpu").manual_seed(int(env_cfg.seed) + 12345)
        W = torch.randn(self.joint_dim, 3, generator=g) / math.sqrt(self.joint_dim)
        self.register_buffer("fk_map", W.to(self.device))
        self.register_buffer("joint_scale", torch.full((self.joint_dim,), 0.6, device=self.device))

        # State
        self.q = torch.zeros(self.num_envs, self.joint_dim, device=self.device)
        self.qdot = torch.zeros(self.num_envs, self.joint_dim, device=self.device)
        self.obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.obj_lin = torch.zeros(self.num_envs, 3, device=self.device)
        self.obj_ang = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.lifted = torch.zeros(self.num_envs, device=self.device)
        self._hold_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self._episode_returns = torch.zeros(self.num_envs, device=self.device)
        self._episode_successes = torch.zeros(self.num_envs, device=self.device)
        self._episode_length_limit = torch.full(
            (self.num_envs,), float(self.max_episode_length), device=self.device
        )
        self.reset()

    # -- buffer registration helpers (keeps the class usable without nn.Module)
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    # -- sampling helpers --------------------------------------------------
    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.generator).to(self.device)

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.generator).to(self.device)

    def _sample_goal_pos(self, n: int) -> torch.Tensor:
        if self.kind == "throw":
            # bucket placed out of reach of the arm
            out = torch.stack(
                [
                    0.45 + 0.15 * self._rand(n),
                    -0.3 + 0.6 * self._rand(n),
                    0.05 + 0.10 * self._rand(n),
                ],
                dim=-1,
            )
            return out
        return torch.stack(
            [
                -0.15 + 0.30 * self._rand(n),
                -0.15 + 0.30 * self._rand(n),
                0.10 + 0.20 * self._rand(n),
            ],
            dim=-1,
        )

    def _sample_object_pose(self, n: int, env_ids: Optional[torch.Tensor] = None) -> None:
        pos = torch.stack(
            [
                -0.05 + 0.10 * self._rand(n),
                -0.05 + 0.10 * self._rand(n),
                torch.full((n,), self._table_height, device=self.device)
                + 0.5 * self.env_cfg.object_size,
            ],
            dim=-1,
        )
        quat = quat_random(n, self.device, self.generator)
        if env_ids is None:
            self.obj_pos, self.obj_quat = pos, quat
            self.obj_lin = torch.zeros(n, 3, device=self.device)
            self.obj_ang = torch.zeros(n, 3, device=self.device)
        else:
            self.obj_pos[env_ids] = pos
            self.obj_quat[env_ids] = quat
            self.obj_lin[env_ids] = 0.0
            self.obj_ang[env_ids] = 0.0

    # -- observations ------------------------------------------------------
    def _hand_centre(self) -> torch.Tensor:
        return 0.15 * torch.tanh(0.5 * (self.q / self.joint_scale)) @ self.fk_map

    @property
    def goal(self) -> torch.Tensor:
        """Task-dependent goal observation ``g_t``."""
        if self.goal_dim == 4:
            return self.goal_quat
        if self.goal_dim == 7:
            return torch.cat([self.goal_pos, self.goal_quat], dim=-1)
        return self.goal_pos[:, : self.goal_dim]

    def _build_obs(self) -> torch.Tensor:
        parts = [
            self.q,
            self.qdot,
            torch.cat([self.obj_pos, self.obj_quat], dim=-1),
            self.obj_lin,
            self.obj_ang,
            self.goal,
        ]
        if self.extra_dim:
            parts.append(self.lifted.unsqueeze(-1))
        obs = torch.cat(parts, dim=-1)
        if self.env_cfg.observation_noise > 0 and self.training_noise_enabled:
            obs = obs + float(self.env_cfg.observation_noise) * torch.randn_like(obs)
        if self.num_obs > obs.shape[-1]:  # pad if the caller asked for more
            obs = torch.cat(
                [obs, torch.zeros(self.num_envs, self.num_obs - obs.shape[-1], device=self.device)],
                dim=-1,
            )
        return obs[:, : self.num_obs].clamp(-self.env_cfg.clip_obs, self.env_cfg.clip_obs)

    # -- gym-like API ------------------------------------------------------
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.num_envs if env_ids is None else int(env_ids.numel())
        q = self._randn(n, self.joint_dim) * 0.1
        gobj = self._sample_goal_pos(n)
        gquat = quat_random(n, self.device, self.generator)
        if self.kind == "orient":
            # goal orientation reachable by the object
            gquat = self.obj_quat if env_ids is None else self.obj_quat[env_ids]
        if env_ids is None:
            self.q = q
            self.qdot = torch.zeros_like(q)
            self.goal_pos = gobj
            self.goal_quat = gquat
            self.lifted = torch.zeros(n, device=self.device)
            self._hold_count = torch.zeros(n, device=self.device)
            self._episode_lengths = torch.zeros(n, device=self.device)
            self._episode_returns = torch.zeros(n, device=self.device)
            self._episode_successes = torch.zeros(n, device=self.device)
            self._sample_object_pose(n)
        else:
            self.q[env_ids] = q
            self.qdot[env_ids] = 0.0
            self.goal_pos[env_ids] = gobj
            self.goal_quat[env_ids] = gquat
            self.lifted[env_ids] = 0.0
            self._hold_count[env_ids] = 0.0
            self._episode_lengths[env_ids] = 0.0
            self._episode_returns[env_ids] = 0.0
            self._episode_successes[env_ids] = 0.0
            self._sample_object_pose(n, env_ids)
        # randomised episode length truncation
        if self.env_cfg.randomize:
            limit = self.max_episode_length * (0.5 + 0.5 * self._rand(n))
        else:
            limit = torch.full((n,), float(self.max_episode_length), device=self.device)
        if env_ids is None:
            self._episode_length_limit = limit
        else:
            self._episode_length_limit[env_ids] = limit
        self.training_noise_enabled = True
        return self._build_obs()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        actions = torch.as_tensor(actions, device=self.device).float().view(self.num_envs, self.num_actions)
        actions = actions.clamp(-self.env_cfg.clip_actions, self.env_cfg.clip_actions)

        # --- joint servo tracking ----------------------------------------
        target_q = actions * self.joint_scale
        q_prev = self.q
        self.q = self.q + self.dt * self._joint_gain * (target_q - self.q)
        self.qdot = (self.q - q_prev) / max(self.dt, 1e-6)

        hand = self._hand_centre()
        dist_to_obj = (hand - self.obj_pos).norm(dim=-1)
        grasped = (dist_to_obj < self._grasp_radius).float()

        # --- object dynamics ---------------------------------------------
        if self.kind == "throw":
            self.obj_lin = self.obj_lin + self.dt * (
                grasped.unsqueeze(-1) * (hand - self.obj_pos) * 20.0
            )
            self.obj_lin[:, 2] -= 9.81 * self.dt * (1.0 - grasped)
            self.obj_pos = self.obj_pos + self.dt * self.obj_lin
        else:
            self.obj_pos = self.obj_pos + self.dt * grasped.unsqueeze(-1) * (
                (hand - self.obj_pos) * 10.0
            )
            self.obj_lin = self.dt * grasped.unsqueeze(-1) * (hand - self.obj_pos) * 10.0
        self.obj_pos[:, 2] = torch.clamp(self.obj_pos[:, 2], min=self._table_height)
        self.lifted = (self.obj_pos[:, 2] > self._table_height + 0.03).float()

        # object orientation follows the hand once grasped
        if self.kind in ("orient", "hand"):
            spin = self.dt * torch.tanh(actions[:, :4].mean(dim=-1, keepdim=True)) * grasped.unsqueeze(-1)
            dq = 0.5 * spin * self.obj_quat  # small first-order rotation step
            self.obj_quat = quat_normalize(self.obj_quat + self.dt * dq)
        else:
            self.obj_ang = 0.1 * grasped.unsqueeze(-1) * actions[:, :3]

        # --- task quantities ---------------------------------------------
        pos_err = (self.goal_pos - self.obj_pos).norm(dim=-1)
        orn_err = quat_angle_error(self.obj_quat, self.goal_quat)
        dist_hand = (self.goal_pos - hand).norm(dim=-1)

        if self.kind == "hand":
            in_place = (orn_err <= 0.3).float()
            success_now = (in_place * (dist_to_obj < self._grasp_radius).float())
        elif self.kind == "orient":
            success_now = (
                (pos_err <= self.success_tolerance) & (orn_err <= 0.3)
            ).float()
        elif self.kind == "throw":
            in_bucket = (pos_err <= self.env_cfg.bucket_radius) & (self.lifted > 0)
            success_now = in_bucket.float()
        else:  # regrasping
            success_now = (pos_err <= self.success_tolerance).float()

        # hold counter -> "success" requires K = hold_steps consecutive steps
        self._hold_count = self._hold_count + success_now
        self._hold_count = torch.where(
            success_now > 0, self._hold_count, torch.zeros_like(self._hold_count)
        )
        success = (self._hold_count >= self.hold_steps).float()

        # --- rewards (Appendix A decomposition) --------------------------
        w = self.weights
        r_reach = -float(w.get("reach", 0.0)) * dist_to_obj
        r_lift = float(w.get("lift", 0.0)) * self.lifted
        r_target = -float(w.get("target", 0.0)) * pos_err * self.lifted
        r_orient = -float(w.get("orient", 0.0)) * orn_err * (self.lifted if self.kind != "hand" else 1.0)
        r_success = float(w.get("success", 0.0)) * success
        reward = r_reach + r_lift + r_target + r_orient + r_success

        # --- episode bookkeeping -----------------------------------------
        self._episode_lengths += 1.0
        self._episode_returns += reward
        self._episode_successes += success

        done = (self._episode_lengths >= self._episode_length_limit).float()
        # on success in the hard tasks the goal/object are reset (Appendix A)
        resample = (success > 0) | (done > 0)
        if resample.any():
            ids = resample.nonzero(as_tuple=False).squeeze(-1)
            self._resample_targets(ids, done=done[ids] > 0)

        info: Dict[str, Any] = {
            "successes": self._episode_successes.clone(),
            "success": success,
            "episode_return": self._episode_returns.clone(),
            "episode_length": self._episode_lengths.clone(),
            "goal_distance": pos_err.detach(),
            "orientation_error": orn_err.detach(),
            "metrics": {
                "successes": self._episode_successes.mean(),
                "episode_return": self._episode_returns.mean(),
            },
        }

        # auto-reset terminated environments and return their first observation
        reset_ids = (done > 0).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.reset(reset_ids)

        return self._build_obs(), reward, done, info

    # -- helpers -----------------------------------------------------------
    def _resample_targets(self, env_ids: torch.Tensor, done: torch.Tensor) -> None:
        """Reset goal and object pose after a success (Algorithm: Appendix A)."""
        n = int(env_ids.numel())
        self.goal_pos[env_ids] = self._sample_goal_pos(n)
        if self.kind in ("orient", "hand"):
            self.goal_quat[env_ids] = quat_random(n, self.device, self.generator)
        successes = env_ids[self._episode_successes[env_ids] > 0]
        if successes.numel() > 0:
            # object is re-placed randomly after every success
            ids = successes[(torch.rand(len(successes), generator=self.generator) < 0.5).to(self.device)]
            if ids.numel() > 0:
                self._sample_object_pose(int(ids.numel()), ids)
        self._hold_count[env_ids] = 0.0

    def set_success_tolerance(self, tolerance: float) -> None:
        self.success_tolerance = float(tolerance)

    def close(self) -> None:
        return None

    training_noise_enabled: bool = True


# ---------------------------------------------------------------------------
# Unified facade
# ---------------------------------------------------------------------------


class IsaacEnv:
    """Unified vectorised-environment facade for the SAPG task suite.

    Dispatches to :class:`IsaacGymParallelEnv` when the GPU simulator is
    available (and ``use_isaacgym`` is set) and to :class:`SurrogateVectorEnv`
    otherwise.  Exposes the interface required by ``sapg.algorithms.rollout``.
    """

    def __init__(self, config: Any = None, **overrides: Any):
        self.env_cfg = EnvConfig.from_any(config, **overrides)
        want_isaac = bool(self.env_cfg.use_isaacgym) and not _env_flag("SAPG_FORCE_SURROGATE")
        if want_isaac and HAS_ISAACGYM:
            try:  # pragma: no cover - only on machines with IsaacGym
                self.backend = IsaacGymParallelEnv(self.env_cfg)
                self.backend_name = "isaacgym"
            except Exception as exc:  # pragma: no cover
                print(f"[sapg.envs] IsaacGym backend unavailable ({exc}); using surrogate.")
                self.backend = SurrogateVectorEnv(self.env_cfg)
                self.backend_name = "surrogate"
        else:
            self.backend = SurrogateVectorEnv(self.env_cfg)
            self.backend_name = "surrogate"

        self.num_envs = int(self.backend.num_envs)
        self.num_obs = int(self.backend.num_obs)
        self.num_actions = int(self.backend.num_actions)
        self.num_actions = int(self.backend.num_actions)
        self.device = getattr(self.backend, "device", torch.device("cpu"))
        self.task = self.env_cfg.task
        self.task_group = task_group_of(self.task)

    # -- properties -------------------------------------------------------- #
    @property
    def obs_dim(self) -> int:
        return self.num_obs

    @property
    def action_dim(self) -> int:
        return self.num_actions

    @property
    def success_tolerance(self) -> float:
        return float(getattr(self.backend, "success_tolerance", self.env_cfg.success_tolerance))

    # -- gym-like API ------------------------------------------------------ #
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.backend.reset(env_ids)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return self.backend.step(actions)

    def set_success_tolerance(self, tolerance: float) -> None:
        setter = getattr(self.backend, "set_success_tolerance", None)
        if callable(setter):
            setter(tolerance)
        self.env_cfg.success_tolerance = float(tolerance)

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()

    # -- convenience ------------------------------------------------------- #
    def sample_actions(self) -> torch.Tensor:
        return torch.rand(
            self.num_envs, self.num_actions, device=self.device
        ) * 2.0 - 1.0

    def info_snapshot(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "task_group": self.task_group,
            "backend": self.backend_name,
            "num_envs": self.num_envs,
            "num_obs": self.num_obs,
            "num_actions": self.num_actions,
            "success_tolerance": self.success_tolerance,
        }

    def __len__(self) -> int:
        return self.num_envs

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"IsaacEnv(task={self.task!r}, backend={self.backend_name!r}, "
            f"num_envs={self.num_envs}, obs_dim={self.num_obs}, "
            f"action_dim={self.num_actions})"
        )


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).lower() in ("1", "true", "yes", "on")


def make_isaac_env(task: str = "regrasping", config: Any = None, **overrides: Any) -> IsaacEnv:
    """Factory: build an :class:`IsaacEnv` for ``task`` (or from a SAPG config)."""
    if config is None:
        overrides.setdefault("task", task)
        return IsaacEnv(**overrides)
    if not overrides.get("task"):
        overrides["task"] = task if task else getattr(config, "task", "regrasping")
    return IsaacEnv(config, **overrides)


__all__ = [
    "HAS_ISAACGYM",
    "TASK_REGISTRY",
    "EnvConfig",
    "IsaacEnv",
    "IsaacGymParallelEnv",
    "SurrogateVectorEnv",
    "make_isaac_env",
    "obs_dim_for",
    "action_dim_for",
    "goal_dim_for",
    "joint_dim_for",
    "task_group_of",
    "default_reward_weights",
    "default_isaac_task_cfg",
    "quat_angle_error",
    "quat_mul",
    "quat_random",
]
