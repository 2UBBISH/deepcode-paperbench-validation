"""AllegroKuka (23-DoF arm + Allegro hand) manipulation tasks for SAPG.

Implements the three hard tasks from the paper (Sec 5.1, Appendix A):

  * Regrasping     -- hold the object near a goal position g_t in R^3 for K=30
                      consecutive steps, with a tolerance curriculum that decays
                      from 7.5cm to 1cm (-10% whenever the average number of
                      successes per episode exceeds 3).
  * Throw          -- throw the object into a bucket located at g_t in R^3.
  * Reorientation  -- reorient the object to a goal pose g_t in R^7 (position +
                      quaternion).

Observation (23-DoF, Appendix A):
    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]
where
    q       : joint positions of the arm+hand (23)
    q_dot   : joint velocities (23)
    x_t     : object position (3)
    v_t     : object linear velocity (3)
    omega_t : object angular velocity (3)
    g_t     : goal (3 for regrasping/throw, 7 for reorientation)
    z_t     : task-specific auxiliary features (e.g. object orientation, finger
              contacts, previous action)

Metric: successes per episode (AllegroKuka tasks).

These classes are written to be *simulator agnostic*: they expose the same
``reset()`` / ``step(actions)`` / ``block_slice(idx)`` interface used by
``envs.isaacgym_wrapper``.  When IsaacGym is available the physics is delegated
to a task-specific IsaacGym env; otherwise a lightweight analytic dynamics
model is used so the full SAPG training loop can be exercised on CPU.
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

NUM_JOINTS = 23  # 7-DoF Kuka arm + 16-DoF Allegro hand
REGRAAP_HOLD_STEPS = 30  # K in the paper
REGRAAP_TOL_START = 0.075  # 7.5 cm
REGRAAP_TOL_END = 0.01  # 1 cm
REGRAAP_TOL_DECAY = 0.9  # -10%
REGRAAP_SUCCESS_THRESHOLD = 3.0  # avg successes/episode to trigger decay


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a batch of quaternions [B, 4] (w, x, y, z) to rotation matrices."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    b = q.shape[0]
    m = torch.empty(b, 3, 3, device=q.device, dtype=q.dtype)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular distance between two batches of quaternions (radians)."""
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-8)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class AllegroKukaBase:
    """Shared machinery for the 23-DoF AllegroKuka tasks.

    Parameters
    ----------
    task : str
        One of ``regrasping``, ``throw``, ``reorientation``.
    num_envs : int
        Number of parallel environments (default 24576).
    num_blocks : int
        Number of SAPG blocks (default 6).
    horizon : int
        Episode length in steps (default 16 for AllegroKuka).
    device : torch.device, optional
    seed : int, optional
    """

    #: dimensionality of the goal vector g_t
    goal_dim: int = 3
    #: dimensionality of the auxiliary feature vector z_t
    aux_dim: int = 16
    #: whether the task uses the regrasping tolerance curriculum
    uses_curriculum: bool = False

    def __init__(
        self,
        task: str = "regrasping",
        num_envs: int = 24576,
        num_blocks: int = 6,
        horizon: int = 16,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        **task_kwargs: Any,
    ) -> None:
        self.task = task
        self.num_envs = int(num_envs)
        self.num_blocks = int(num_blocks)
        self.horizon = int(horizon)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.seed = seed

        if self.num_envs % self.num_blocks != 0:
            raise ValueError(
                f"num_envs ({self.num_envs}) must be divisible by num_blocks "
                f"({self.num_blocks})"
            )
        self.envs_per_block = self.num_envs // self.num_blocks

        # Observation layout: q(23) + q_dot(23) + x(3) + v(3) + omega(3)
        #                      + g(goal_dim) + z(aux_dim)
        self.obs_dim = 2 * NUM_JOINTS + 9 + self.goal_dim + self.aux_dim
        self.action_dim = NUM_JOINTS

        # Curriculum state (regrasping only)
        self.tolerance = REGRAAP_TOL_START
        self._success_history: list = []

        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(int(seed))

        # Internal state
        self._step_count = torch.zeros(self.num_envs, device=self.device)
        self._hold_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_successes = torch.zeros(self.num_envs, device=self.device)
        self._successes_this_episode = torch.zeros(self.num_envs, device=self.device)

        self._q = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._q_dot = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._obj_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._obj_omega = torch.zeros(self.num_envs, 3, device=self.device)
        self._obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._obj_quat[:, 0] = 1.0
        self._goal = torch.zeros(self.num_envs, self.goal_dim, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

        self._isaac_env = None
        if HAS_ISAACGYM and task_kwargs.pop("use_isaacgym", True):
            self._isaac_env = self._build_isaac_env(task_kwargs)

    # -- IsaacGym hook -----------------------------------------------------
    def _build_isaac_env(self, task_kwargs: Dict[str, Any]):
        """Attempt to build a real IsaacGym env; return None on failure."""
        try:  # pragma: no cover - requires IsaacGym
            from isaacgymenvs.tasks.allegro_kuka import (  # type: ignore
                AllegroKuka as _IsaacAllegroKuka,
            )

            return _IsaacAllegroKuka(
                task=self.task,
                num_envs=self.num_envs,
                device=self.device,
                **task_kwargs,
            )
        except Exception:
            return None

    # -- Goal sampling -----------------------------------------------------
    def _sample_goals(self, mask: Optional[torch.Tensor] = None) -> None:
        """Sample new goals g_t for the environments selected by ``mask``."""
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        n = int(mask.sum().item())
        if n == 0:
            return
        if self.task in ("regrasping", "throw"):
            goal = (torch.rand(n, 3, generator=self._generator) * 2 - 1) * 0.3
            goal = goal.to(self.device)
        else:  # reorientation: position + quaternion
            pos = (torch.rand(n, 3, generator=self._generator) * 2 - 1) * 0.1
            quat = torch.randn(n, 4, generator=self._generator)
            quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
            goal = torch.cat([pos, quat], dim=-1).to(self.device)
        self._goal[mask] = goal

    # -- Observation -------------------------------------------------------
    def _build_obs(self) -> torch.Tensor:
        z = torch.cat(
            [
                self._obj_quat,
                self._prev_action[:, : self.aux_dim - 4]
                if self.aux_dim > 4
                else self._prev_action[:, :0],
            ],
            dim=-1,
        )
        if z.shape[-1] < self.aux_dim:
            pad = torch.zeros(
                self.num_envs, self.aux_dim - z.shape[-1], device=self.device
            )
            z = torch.cat([z, pad], dim=-1)
        z = z[:, : self.aux_dim]
        obs = torch.cat(
            [
                self._q,
                self._q_dot,
                self._obj_pos,
                self._obj_vel,
                self._obj_omega,
                self._goal,
                z,
            ],
            dim=-1,
        )
        return obs

    # -- Dynamics ----------------------------------------------------------
    def _integrate(self, actions: torch.Tensor) -> None:
        """Analytic surrogate dynamics (used when IsaacGym is unavailable)."""
        actions = actions.clamp(-1.0, 1.0)
        # Joint dynamics: first-order lag toward the commanded action.
        self._q_dot = 0.5 * (actions - self._q) + 0.1 * self._q_dot
        self._q = self._q + 0.1 * self._q_dot

        # Object dynamics: the hand imparts a force proportional to the mean
        # finger action; gravity pulls the object down.
        finger_force = actions[:, 7:].mean(dim=-1, keepdim=True) * 0.05
        self._obj_vel = self._obj_vel + finger_force - 0.01
        self._obj_pos = self._obj_pos + 0.05 * self._obj_vel

        # Angular dynamics: small torque from finger asymmetry.
        torque = actions[:, 7:].mean(dim=-1, keepdim=True) * 0.02
        self._obj_omega = 0.9 * self._obj_omega + torque
        dq = 0.05 * self._obj_omega
        self._obj_quat = self._obj_quat + torch.cat(
            [torch.zeros_like(dq[:, :1]), dq], dim=-1
        )
        self._obj_quat = self._obj_quat / (
            self._obj_quat.norm(dim=-1, keepdim=True) + 1e-8
        )

    # -- Rewards -----------------------------------------------------------
    def _compute_rewards(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (rewards, successes) for the current state."""
        if self.task == "regrasping":
            dist = (self._obj_pos - self._goal).norm(dim=-1)
            within = dist < self.tolerance
            self._hold_count = torch.where(
                within, self._hold_count + 1, torch.zeros_like(self._hold_count)
            )
            success = self._hold_count >= REGRAAP_HOLD_STEPS
            reward = -dist - 0.01 * self._q_dot.pow(2).sum(-1)
            reward = reward + 5.0 * success.float()
        elif self.task == "throw":
            dist = (self._obj_pos - self._goal).norm(dim=-1)
            success = dist < 0.1
            reward = -dist + 10.0 * success.float()
        else:  # reorientation
            pos_dist = (self._obj_pos - self._goal[:, :3]).norm(dim=-1)
            ang_dist = _quat_distance(self._obj_quat, self._goal[:, 3:7])
            success = (pos_dist < 0.05) & (ang_dist < 0.3)
            reward = -pos_dist - ang_dist + 5.0 * success.float()
        return reward, success

    # -- Public API --------------------------------------------------------
    def reset(self) -> torch.Tensor:
        self._step_count.zero_()
        self._hold_count.zero_()
        self._successes_this_episode.zero_()
        self._q = 0.1 * torch.randn(
            self.num_envs, NUM_JOINTS, generator=self._generator
        ).to(self.device)
        self._q_dot.zero_()
        self._obj_pos = 0.05 * torch.randn(
            self.num_envs, 3, generator=self._generator
        ).to(self.device)
        self._obj_vel.zero_()
        self._obj_omega.zero_()
        self._obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._obj_quat[:, 0] = 1.0
        self._prev_action.zero_()
        self._sample_goals()
        return self._build_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self._isaac_env is not None:  # pragma: no cover - requires IsaacGym
            obs, rew, done, info = self._isaac_env.step(actions)
            return obs, rew, done, info

        actions = torch.as_tensor(actions, device=self.device)
        self._integrate(actions)
        self._prev_action = actions
        reward, success = self._compute_rewards()
        self._successes_this_episode += success.float()
        self._step_count += 1

        done = self._step_count >= self.horizon
        if done.any():
            self._record_episode_successes(done)
            self._reset_done(done)

        info = {
            "successes": success.float(),
            "episode_successes": self._successes_this_episode.clone(),
            "tolerance": torch.full_like(reward, self.tolerance),
        }
        return self._build_obs(), reward, done.float(), info

    def _reset_done(self, done: torch.Tensor) -> None:
        n = int(done.sum().item())
        if n == 0:
            return
        self._step_count[done] = 0
        self._hold_count[done] = 0
        self._successes_this_episode[done] = 0
        self._q[done] = 0.1 * torch.randn(
            n, NUM_JOINTS, generator=self._generator
        ).to(self.device)
        self._q_dot[done] = 0
        self._obj_pos[done] = 0.05 * torch.randn(
            n, 3, generator=self._generator
        ).to(self.device)
        self._obj_vel[done] = 0
        self._obj_omega[done] = 0
        self._obj_quat[done] = 0
        self._obj_quat[done, 0] = 1.0
        self._prev_action[done] = 0
        self._sample_goals(done)

    def _record_episode_successes(self, done: torch.Tensor) -> None:
        if not self.uses_curriculum:
            return
        finished = self._successes_this_episode[done]
        if finished.numel() == 0:
            return
        self._success_history.append(float(finished.mean().item()))
        if len(self._success_history) > 100:
            self._success_history.pop(0)
        avg = sum(self._success_history) / len(self._success_history)
        if avg > REGRAAP_SUCCESS_THRESHOLD and self.tolerance > REGRAAP_TOL_END:
            self.tolerance = max(
                REGRAAP_TOL_END, self.tolerance * REGRAAP_TOL_DECAY
            )

    def block_slice(self, block_idx: int) -> slice:
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


# ---------------------------------------------------------------------------
# Concrete tasks
# ---------------------------------------------------------------------------


class AllegroKukaRegrasping(AllegroKukaBase):
    """Regrasping: hold the object near g_t for K=30 steps (curriculum)."""

    goal_dim = 3
    uses_curriculum = True

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "regrasping")
        super().__init__(**kwargs)


class AllegroKukaThrow(AllegroKukaBase):
    """Throw: throw the object into a bucket at g_t."""

    goal_dim = 3
    uses_curriculum = False

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "throw")
        super().__init__(**kwargs)


class AllegroKukaReorientation(AllegroKukaBase):
    """Reorientation: reorient the object to pose g_t in R^7."""

    goal_dim = 7
    uses_curriculum = False

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "reorientation")
        super().__init__(**kwargs)


__all__ = [
    "AllegroKukaBase",
    "AllegroKukaRegrasping",
    "AllegroKukaThrow",
    "AllegroKukaReorientation",
    "NUM_JOINTS",
    "REGRAAP_HOLD_STEPS",
    "REGRAAP_TOL_START",
    "REGRAAP_TOL_END",
    "HAS_ISAACGYM",
]
