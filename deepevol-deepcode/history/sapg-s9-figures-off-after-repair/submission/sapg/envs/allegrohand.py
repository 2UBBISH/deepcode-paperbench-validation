"""AllegroHand in-hand reorientation task for SAPG.

Implements the 16-DoF AllegroHand in-hand reorientation task (an "easy" task
from SAPG Sec 5.1 / Appendix A).  The environment exposes a simulator-agnostic
``reset`` / ``step`` / ``block_slice`` interface compatible with the SAPG
training loop (``sapg.rollout`` / ``sapg.algorithm``).

When IsaacGym is available the environment delegates to the real
``isaacgymenvs`` AllegroHand task (following Li et al. 2023 / PQL).  Otherwise a
lightweight analytic surrogate is used so the full SAPG loop can be exercised on
CPU for smoke tests.

Observation layout (Appendix A):
    o_t = [q(16), q_dot(16), x(3), v(3), omega(3), g(4), z(16)]
    -> obs_dim = 16 + 16 + 3 + 3 + 3 + 4 + 16 = 61

Metric: episode reward (per paper, Table 1).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
    from isaacgym import gymapi  # type: ignore

    HAS_ISAACGYM = True
except Exception:  # pragma: no cover
    gymapi = None  # type: ignore
    HAS_ISAACGYM = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_JOINTS = 16
GOAL_DIM = 4  # goal quaternion g_t in R^4
AUX_DIM = 16  # auxiliary latent z_t
OBJ_POS_DIM = 3
OBJ_VEL_DIM = 3
OBJ_ANGVEL_DIM = 3

# Reward weights (follow Li et al. 2023 / PQL conventions).
REWARD_POS_SCALE = 1.0
REWARD_ROT_SCALE = 1.0
REWARD_VEL_PENALTY = 0.01
SUCCESS_BONUS = 5.0
SUCCESS_ANGLE_TOL = 0.1  # radians

# Surrogate dynamics constants.
ACTION_SCALE = 0.5
JOINT_DAMPING = 0.1
OBJ_POS_SCALE = 0.02
OBJ_ROT_SCALE = 0.05


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``[B, 4]`` (w, x, y, z) to rotation matrices ``[B, 3, 3]``."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    b = q.shape[0]
    m = torch.empty((b, 3, 3), device=q.device, dtype=q.dtype)
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
    """Geodesic angular distance (radians) between two batches of quaternions."""
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-8)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _random_quat(num: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample uniform random unit quaternions ``[num, 4]``."""
    u1 = torch.rand(num, device=device, dtype=dtype)
    u2 = torch.rand(num, device=device, dtype=dtype)
    u3 = torch.rand(num, device=device, dtype=dtype)
    q = torch.stack(
        [
            torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2),
            torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2),
            torch.sqrt(u1) * torch.sin(2 * math.pi * u3),
            torch.sqrt(u1) * torch.cos(2 * math.pi * u3),
        ],
        dim=-1,
    )
    # Reorder to (w, x, y, z).
    return q[:, [3, 0, 1, 2]]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class AllegroHandReorientation:
    """16-DoF AllegroHand in-hand reorientation environment.

    Parameters
    ----------
    num_envs:
        Total number of parallel environments (default 24576).
    num_blocks:
        Number of SAPG blocks the envs are split into (default 6).
    horizon:
        Rollout horizon per update (default 8 for the hands).
    device:
        Torch device.
    seed:
        Optional RNG seed.
    use_isaacgym:
        Force IsaacGym on/off.  ``None`` auto-detects.
    """

    goal_dim = GOAL_DIM
    uses_curriculum = False
    task = "allegrohand"

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

        if self.num_envs % self.num_blocks != 0:
            raise ValueError(
                f"num_envs ({self.num_envs}) must be divisible by num_blocks ({self.num_blocks})"
            )
        self.envs_per_block = self.num_envs // self.num_blocks

        self.obs_dim = 2 * NUM_JOINTS + OBJ_POS_DIM + OBJ_VEL_DIM + OBJ_ANGVEL_DIM + GOAL_DIM + AUX_DIM
        self.action_dim = NUM_JOINTS

        if seed is not None:
            torch.manual_seed(seed)

        # Decide backend.
        if use_isaacgym is None:
            use_isaacgym = HAS_ISAACGYM
        self._use_isaacgym = bool(use_isaacgym and HAS_ISAACGYM)
        self._isaac_env = None
        if self._use_isaacgym:
            self._isaac_env = self._build_isaac_env(task_kwargs)

        # Surrogate state.
        self._q = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._q_dot = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._obj_pos = torch.zeros(self.num_envs, OBJ_POS_DIM, device=self.device)
        self._obj_vel = torch.zeros(self.num_envs, OBJ_VEL_DIM, device=self.device)
        self._obj_angvel = torch.zeros(self.num_envs, OBJ_ANGVEL_DIM, device=self.device)
        self._obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._obj_quat[:, 0] = 1.0
        self._goal = torch.zeros(self.num_envs, GOAL_DIM, device=self.device)
        self._goal[:, 0] = 1.0
        self._aux = torch.zeros(self.num_envs, AUX_DIM, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Episode statistics.
        self._episode_reward = torch.zeros(self.num_envs, device=self.device)
        self._episode_successes = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_reward = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_successes = torch.zeros(self.num_envs, device=self.device)

        self.reset()

    # ------------------------------------------------------------------
    # IsaacGym backend
    # ------------------------------------------------------------------
    def _build_isaac_env(self, task_kwargs: Dict[str, Any]):
        """Build the real IsaacGym AllegroHand task if available."""
        try:  # pragma: no cover - requires IsaacGym
            from isaacgymenvs.tasks.allegro_hand import AllegroHand  # type: ignore

            return AllegroHand(
                num_envs=self.num_envs,
                device=str(self.device),
                **task_kwargs,
            )
        except Exception:
            self._use_isaacgym = False
            return None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset all environments and return the initial observation."""
        if self._use_isaacgym and self._isaac_env is not None:  # pragma: no cover
            obs = self._isaac_env.reset()
            return self._to_tensor(obs)

        self._q = 0.1 * torch.randn(self.num_envs, NUM_JOINTS, device=self.device)
        self._q_dot = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._obj_pos = 0.05 * torch.randn(self.num_envs, OBJ_POS_DIM, device=self.device)
        self._obj_vel = torch.zeros(self.num_envs, OBJ_VEL_DIM, device=self.device)
        self._obj_angvel = torch.zeros(self.num_envs, OBJ_ANGVEL_DIM, device=self.device)
        self._obj_quat = _random_quat(self.num_envs, self.device, torch.float32)
        self._goal = _random_quat(self.num_envs, self.device, torch.float32)
        self._aux = torch.zeros(self.num_envs, AUX_DIM, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_reward = torch.zeros(self.num_envs, device=self.device)
        self._episode_successes = torch.zeros(self.num_envs, device=self.device)
        return self._build_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Advance the environment by one step.

        Returns ``(obs, rewards, dones, infos)`` where ``dones`` is horizon-based
        and ``infos`` contains ``successes`` and ``episode_successes``.
        """
        actions = self._to_tensor(actions)
        if self._use_isaacgym and self._isaac_env is not None:  # pragma: no cover
            obs, rew, done, info = self._isaac_env.step(actions)
            obs = self._to_tensor(obs)
            rew = self._to_tensor(rew).reshape(-1)
            done = self._to_tensor(done).reshape(-1).bool()
            info = dict(info) if isinstance(info, dict) else {}
            info.setdefault("successes", torch.zeros(self.num_envs, device=self.device))
            info.setdefault("episode_successes", torch.zeros(self.num_envs, device=self.device))
            return obs, rew, done, info

        self._integrate(actions)
        rewards, successes = self._compute_rewards()

        self._episode_reward += rewards
        self._episode_successes += successes
        self._step_count += 1

        done = self._step_count >= self.horizon
        if done.any():
            self._last_episode_reward = torch.where(
                done, self._episode_reward, self._last_episode_reward
            )
            self._last_episode_successes = torch.where(
                done, self._episode_successes, self._last_episode_successes
            )
            self._reset_done(done)

        obs = self._build_obs()
        info = {
            "successes": successes,
            "episode_successes": self._last_episode_successes.clone(),
            "episode_reward": self._last_episode_reward.clone(),
        }
        return obs, rewards, done, info

    def block_slice(self, block_idx: int) -> slice:
        """Return the contiguous env-index slice for SAPG block ``block_idx``."""
        if not 0 <= block_idx < self.num_blocks:
            raise IndexError(f"block_idx {block_idx} out of range [0, {self.num_blocks})")
        start = block_idx * self.envs_per_block
        return slice(start, start + self.envs_per_block)

    def close(self) -> None:
        if self._isaac_env is not None:  # pragma: no cover
            try:
                self._isaac_env.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return self.num_envs

    # ------------------------------------------------------------------
    # Surrogate dynamics
    # ------------------------------------------------------------------
    def _integrate(self, actions: torch.Tensor) -> None:
        """Lightweight analytic dynamics for CPU smoke tests."""
        actions = torch.tanh(actions)
        self._q_dot = (1.0 - JOINT_DAMPING) * self._q_dot + ACTION_SCALE * actions
        self._q = self._q + self._q_dot

        # Object moves toward the goal pose driven by the hand action magnitude.
        drive = actions.mean(dim=-1, keepdim=True)
        goal_dir = self._goal[:, 1:]  # (x, y, z) part of goal quaternion as a proxy direction
        self._obj_pos = self._obj_pos + OBJ_POS_SCALE * drive * goal_dir
        self._obj_vel = OBJ_POS_SCALE * drive * goal_dir

        # Rotate object quaternion toward goal.
        self._obj_quat = self._obj_quat + OBJ_ROT_SCALE * drive * (self._goal - self._obj_quat)
        self._obj_quat = self._obj_quat / (self._obj_quat.norm(dim=-1, keepdim=True) + 1e-8)

        self._obj_angvel = OBJ_ROT_SCALE * drive * (self._goal - self._obj_quat)[:, :3]

    def _compute_rewards(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-env reward and success indicator."""
        pos_dist = self._obj_pos.norm(dim=-1)
        ang_dist = _quat_distance(self._obj_quat, self._goal)
        vel_penalty = self._obj_vel.norm(dim=-1) + self._obj_angvel.norm(dim=-1)

        success = (ang_dist < SUCCESS_ANGLE_TOL).float()
        reward = (
            -REWARD_POS_SCALE * pos_dist
            - REWARD_ROT_SCALE * ang_dist
            - REWARD_VEL_PENALTY * vel_penalty
            + SUCCESS_BONUS * success
        )
        return reward, success

    def _build_obs(self) -> torch.Tensor:
        """Assemble the observation vector ``[q, q_dot, x, v, omega, g, z]``."""
        return torch.cat(
            [
                self._q,
                self._q_dot,
                self._obj_pos,
                self._obj_vel,
                self._obj_angvel,
                self._goal,
                self._aux,
            ],
            dim=-1,
        )

    def _reset_done(self, done: torch.Tensor) -> None:
        """Reset environments that finished their episode."""
        n = int(done.sum().item())
        if n == 0:
            return
        idx = done.nonzero(as_tuple=False).squeeze(-1)
        self._q[idx] = 0.1 * torch.randn(n, NUM_JOINTS, device=self.device)
        self._q_dot[idx] = 0.0
        self._obj_pos[idx] = 0.05 * torch.randn(n, OBJ_POS_DIM, device=self.device)
        self._obj_vel[idx] = 0.0
        self._obj_angvel[idx] = 0.0
        self._obj_quat[idx] = _random_quat(n, self.device, torch.float32)
        self._goal[idx] = _random_quat(n, self.device, torch.float32)
        self._aux[idx] = 0.0
        self._step_count[idx] = 0
        self._episode_reward[idx] = 0.0
        self._episode_successes[idx] = 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)


__all__ = [
    "AllegroHandReorientation",
    "NUM_JOINTS",
    "GOAL_DIM",
    "HAS_ISAACGYM",
]
