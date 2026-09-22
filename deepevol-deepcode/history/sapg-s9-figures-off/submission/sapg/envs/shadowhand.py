"""ShadowHand in-hand reorientation task (24-DoF) for SAPG.

Implements the "easy" ShadowHand reorientation task from the SAPG paper
(Sec 5.1, Appendix A).  The task follows Li et al. 2023 (PQL): a 24-DoF
ShadowHand must reorient an object to a target goal quaternion ``g_t`` in
R^4.  The reported metric is episode reward.

The environment is simulator-agnostic: it exposes the same
``reset`` / ``step`` / ``block_slice`` interface as
:mod:`sapg.envs.allegrokuka` and :mod:`sapg.envs.isaacgym_wrapper`.  When
IsaacGym is available it delegates to the real ShadowHand task; otherwise it
falls back to a lightweight analytic surrogate so the full SAPG training loop
can be exercised on CPU.

Observation layout (Appendix A):
    o_t = [q (24), q_dot (24), x (3), v (3), omega (3), g (4), z (aux)]
where ``q`` are joint positions, ``q_dot`` joint velocities, ``x`` object
position, ``v`` object linear velocity, ``omega`` object angular velocity,
``g`` the goal quaternion (w, x, y, z), and ``z`` auxiliary features.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
    from isaacgym import gymapi  # type: ignore

    HAS_ISAACGYM = True
except Exception:  # pragma: no cover - IsaacGym not installed
    gymapi = None  # type: ignore
    HAS_ISAACGYM = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_JOINTS = 24  # ShadowHand has 24 DoF
GOAL_DIM = 4  # goal quaternion (w, x, y, z)
AUX_DIM = 16  # auxiliary observation features (z_t)
OBJ_POS_DIM = 3
OBJ_VEL_DIM = 3
OBJ_ANGVEL_DIM = 3

# Reward shaping weights (following Li et al. 2023 / PQL conventions).
REWARD_POS_SCALE = 1.0
REWARD_ROT_SCALE = 1.0
REWARD_VEL_PENALTY = 0.01
SUCCESS_BONUS = 5.0
SUCCESS_ANGLE_TOL = 0.1  # radians (~5.7 deg) for a successful reorientation


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a batch of quaternions ``(w, x, y, z)`` to rotation matrices.

    Args:
        q: Tensor of shape ``[B, 4]``.

    Returns:
        Tensor of shape ``[B, 3, 3]``.
    """
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    b = q.shape[0]
    m = torch.zeros(b, 3, 3, device=q.device, dtype=q.dtype)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - w * z)
    m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y)
    m[:, 2, 1] = 2 * (y * z + w * x)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular distance (radians) between two batches of quaternions.

    Args:
        q1: Tensor of shape ``[B, 4]``.
        q2: Tensor of shape ``[B, 4]``.

    Returns:
        Tensor of shape ``[B]`` with the geodesic angle in ``[0, pi]``.
    """
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-8)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _random_quat(num: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample uniformly random unit quaternions of shape ``[num, 4]``."""
    u = torch.rand(num, 3, device=device, dtype=dtype)
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
    return q[:, [3, 0, 1, 2]]


class ShadowHandReorientation:
    """24-DoF ShadowHand in-hand reorientation task.

    Args:
        num_envs: Number of parallel environments (default 24576).
        num_blocks: Number of SAPG blocks (default 6).
        horizon: Steps per env instance per update (default 8).
        device: Torch device.
        seed: Optional RNG seed.
        use_isaacgym: Force/disable the IsaacGym backend.  ``None`` means
            auto-detect.
        **task_kwargs: Extra task-specific keyword arguments (ignored by the
            surrogate backend).
    """

    goal_dim = GOAL_DIM
    uses_curriculum = False

    def __init__(
        self,
        num_envs: int = 24576,
        num_blocks: int = 6,
        horizon: int = 8,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        use_isaacgym: Optional[bool] = None,
        **task_kwargs: Any,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_blocks = int(num_blocks)
        self.horizon = int(horizon)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.seed = seed
        self.task = "shadowhand"
        self.task_kwargs = task_kwargs

        if self.num_envs % self.num_blocks != 0:
            raise ValueError(
                f"num_envs ({self.num_envs}) must be divisible by num_blocks "
                f"({self.num_blocks})"
            )
        self.envs_per_block = self.num_envs // self.num_blocks

        self.obs_dim = 2 * NUM_JOINTS + OBJ_POS_DIM + OBJ_VEL_DIM + OBJ_ANGVEL_DIM + GOAL_DIM + AUX_DIM
        self.action_dim = NUM_JOINTS

        self._generator = torch.Generator(device=self.device)
        if seed is not None:
            self._generator.manual_seed(int(seed))

        # Decide backend.
        if use_isaacgym is None:
            use_isaacgym = HAS_ISAACGYM
        self._use_isaacgym = bool(use_isaacgym) and HAS_ISAACGYM
        self._isaac_env = None
        if self._use_isaacgym:
            self._isaac_env = self._build_isaac_env()

        # Surrogate state.
        self._q = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._q_dot = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._obj_pos = torch.zeros(self.num_envs, OBJ_POS_DIM, device=self.device)
        self._obj_vel = torch.zeros(self.num_envs, OBJ_VEL_DIM, device=self.device)
        self._obj_omega = torch.zeros(self.num_envs, OBJ_ANGVEL_DIM, device=self.device)
        self._obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._obj_quat[:, 0] = 1.0
        self._goal = torch.zeros(self.num_envs, GOAL_DIM, device=self.device)
        self._goal[:, 0] = 1.0
        self._aux = torch.zeros(self.num_envs, AUX_DIM, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Episode success bookkeeping (metric = episode reward, but we still
        # track successes for diagnostics).
        self._episode_successes = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_successes = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_reward = torch.zeros(self.num_envs, device=self.device)

        self.reset()

    # ------------------------------------------------------------------
    # Backend construction
    # ------------------------------------------------------------------
    def _build_isaac_env(self):  # pragma: no cover - requires IsaacGym
        """Attempt to construct the real IsaacGym ShadowHand task."""
        try:
            from isaacgymenvs.tasks.shadow_hand import ShadowHand  # type: ignore

            return ShadowHand(
                num_envs=self.num_envs,
                device=str(self.device),
                **self.task_kwargs,
            )
        except Exception:
            self._use_isaacgym = False
            return None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset all environments and return the initial observation."""
        if self._isaac_env is not None:  # pragma: no cover
            obs = self._isaac_env.reset()
            return self._to_tensor(obs)

        self._q = 0.1 * torch.randn(
            self.num_envs, NUM_JOINTS, device=self.device, generator=self._generator
        )
        self._q_dot = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._obj_pos = 0.05 * torch.randn(
            self.num_envs, OBJ_POS_DIM, device=self.device, generator=self._generator
        )
        self._obj_vel = torch.zeros(self.num_envs, OBJ_VEL_DIM, device=self.device)
        self._obj_omega = torch.zeros(self.num_envs, OBJ_ANGVEL_DIM, device=self.device)
        self._obj_quat = _random_quat(self.num_envs, self.device, self._q.dtype)
        self._goal = _random_quat(self.num_envs, self.device, self._q.dtype)
        self._aux = torch.zeros(self.num_envs, AUX_DIM, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_successes = torch.zeros(self.num_envs, device=self.device)
        self._episode_reward = torch.zeros(self.num_envs, device=self.device)
        return self._build_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Advance the environment by one step.

        Args:
            actions: Tensor of shape ``[num_envs, action_dim]``.

        Returns:
            ``(obs, rewards, dones, infos)`` where ``obs`` is ``[num_envs, obs_dim]``,
            ``rewards``/``dones`` are ``[num_envs]``, and ``infos`` contains
            ``successes`` and ``episode_successes``.
        """
        if self._isaac_env is not None:  # pragma: no cover
            obs, rew, done, info = self._isaac_env.step(actions)
            return self._to_tensor(obs), self._to_tensor(rew), self._to_tensor(done), info

        actions = actions.to(self.device)
        self._integrate(actions)
        rewards = self._compute_rewards()
        self._episode_reward += rewards

        self._step_count += 1
        done = self._step_count >= self.horizon

        # Track successes.
        ang_dist = _quat_distance(self._obj_quat, self._goal)
        success = (ang_dist < SUCCESS_ANGLE_TOL).float()
        self._episode_successes += success

        # Record finished episodes.
        self._last_episode_successes = torch.where(
            done, self._episode_successes, self._last_episode_successes
        )
        self._last_episode_reward = torch.where(
            done, self._episode_reward, self._last_episode_reward
        )

        obs = self._build_obs()
        info = {
            "successes": success,
            "episode_successes": self._last_episode_successes.clone(),
            "episode_reward": self._last_episode_reward.clone(),
            "time_outs": done.clone(),
        }

        # Auto-reset finished envs.
        if done.any():
            self._reset_done(done)

        return obs, rewards, done.float(), info

    def block_slice(self, block_idx: int) -> slice:
        """Return the contiguous env-index slice for SAPG block ``block_idx``."""
        start = block_idx * self.envs_per_block
        return slice(start, start + self.envs_per_block)

    def close(self) -> None:
        """Release backend resources."""
        if self._isaac_env is not None:  # pragma: no cover
            try:
                self._isaac_env.close()
            except Exception:
                pass
            self._isaac_env = None

    def __len__(self) -> int:
        return self.num_envs

    # ------------------------------------------------------------------
    # Surrogate dynamics
    # ------------------------------------------------------------------
    def _integrate(self, actions: torch.Tensor) -> None:
        """Lightweight analytic dynamics for the surrogate backend."""
        actions = torch.tanh(actions)
        self._q_dot = 0.9 * self._q_dot + 0.1 * actions
        self._q = self._q + 0.1 * self._q_dot

        # Object motion driven by the mean hand action (crude coupling).
        drive = actions.mean(dim=-1, keepdim=True)
        self._obj_pos = self._obj_pos + 0.01 * drive.expand(-1, OBJ_POS_DIM)
        self._obj_vel = 0.01 * drive.expand(-1, OBJ_VEL_DIM)

        # Rotate the object toward the goal by a small step proportional to
        # the action magnitude (surrogate for in-hand reorientation).
        ang_dist = _quat_distance(self._obj_quat, self._goal)
        step_ang = (0.05 * actions.abs().mean(dim=-1)).clamp(max=0.2)
        # Interpolate quaternion toward the goal.
        t = (step_ang / (ang_dist + 1e-6)).clamp(max=1.0).unsqueeze(-1)
        q_new = self._obj_quat + t * (self._goal - self._obj_quat)
        self._obj_quat = q_new / (q_new.norm(dim=-1, keepdim=True) + 1e-8)
        self._obj_omega = 0.1 * step_ang.unsqueeze(-1).expand(-1, OBJ_ANGVEL_DIM)

    def _compute_rewards(self) -> torch.Tensor:
        """Reorientation reward: negative angular distance + success bonus."""
        ang_dist = _quat_distance(self._obj_quat, self._goal)
        pos_pen = REWARD_POS_SCALE * self._obj_pos.pow(2).sum(dim=-1)
        vel_pen = REWARD_VEL_PENALTY * self._q_dot.pow(2).sum(dim=-1)
        success = (ang_dist < SUCCESS_ANGLE_TOL).float()
        reward = -REWARD_ROT_SCALE * ang_dist - pos_pen - vel_pen + SUCCESS_BONUS * success
        return reward

    def _build_obs(self) -> torch.Tensor:
        """Assemble the observation vector ``o_t``."""
        return torch.cat(
            [
                self._q,
                self._q_dot,
                self._obj_pos,
                self._obj_vel,
                self._obj_omega,
                self._goal,
                self._aux,
            ],
            dim=-1,
        )

    def _reset_done(self, done: torch.Tensor) -> None:
        """Reset the environments flagged by ``done``."""
        idx = done.nonzero(as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n == 0:
            return
        self._q[idx] = 0.1 * torch.randn(
            n, NUM_JOINTS, device=self.device, generator=self._generator
        )
        self._q_dot[idx] = 0.0
        self._obj_pos[idx] = 0.05 * torch.randn(
            n, OBJ_POS_DIM, device=self.device, generator=self._generator
        )
        self._obj_vel[idx] = 0.0
        self._obj_omega[idx] = 0.0
        self._obj_quat[idx] = _random_quat(n, self.device, self._q.dtype)
        self._goal[idx] = _random_quat(n, self.device, self._q.dtype)
        self._aux[idx] = 0.0
        self._step_count[idx] = 0
        self._episode_successes[idx] = 0.0
        self._episode_reward[idx] = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, x: Any) -> torch.Tensor:  # pragma: no cover
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, device=self.device)


__all__ = [
    "ShadowHandReorientation",
    "NUM_JOINTS",
    "GOAL_DIM",
    "HAS_ISAACGYM",
]
