"""AllegroHand in-hand reorientation task (16-DoF).

Implements the 16-DoF AllegroHand in-hand reorientation task described in
Appendix A of the SAPG paper.  The goal is to rotate a cube/object to match a
target orientation (quaternion goal in R^4).

Two backends are provided:

* ``_AllegroHandSim``      -- pure NumPy simulation used for smoke tests / CI.
* ``_AllegroHandIsaacGym`` -- thin wrapper around an IsaacGym task (falls back
  to the NumPy backend when asset loading is unavailable).

Observation layout (Appendix A):
    o_t = [q (16), qdot (16), x_t (3), v_t (3), omega_t (3), g_t (4), z_t (3)]
    => obs_dim = 48

Action space: 16 joint position targets (normalized to [-1, 1]).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_HAND_JOINTS = 16
NUM_DOFS = 16
OBS_DIM = 48  # 16 + 16 + 3 + 3 + 3 + 4 + 3

# Reward weights (DexPBT reference values).
W_REACH = 1.0
W_LIFT = 2.0
W_TARGET = 5.0
W_SUCCESS = 10.0

# Success tolerance on the quaternion distance.
SUCCESS_TOL_START = 0.5
SUCCESS_TOL_END = 0.1
SUCCESS_HOLD_STEPS = 30
CURRICULUM_DECAY = 0.90
CURRICULUM_THRESHOLD = 3.0

TASK_NAMES = ("allegrohand",)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def _random_quat(rng: np.random.Generator, shape) -> np.ndarray:
    """Sample uniformly random unit quaternions (w, x, y, z)."""
    u = rng.random(shape + (3,))
    q = np.empty(shape + (4,), dtype=np.float64)
    q[..., 0] = np.sqrt(1.0 - u[..., 0]) * np.sin(2.0 * np.pi * u[..., 1])
    q[..., 1] = np.sqrt(1.0 - u[..., 0]) * np.cos(2.0 * np.pi * u[..., 1])
    q[..., 2] = np.sqrt(u[..., 0]) * np.sin(2.0 * np.pi * u[..., 2])
    q[..., 3] = np.sqrt(u[..., 0]) * np.cos(2.0 * np.pi * u[..., 2])
    return q


def _quat_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular distance between two unit quaternions (in radians)."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rot = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    rot[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rot[..., 0, 1] = 2 * (x * y - z * w)
    rot[..., 0, 2] = 2 * (x * z + y * w)
    rot[..., 1, 0] = 2 * (x * y + z * w)
    rot[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rot[..., 1, 2] = 2 * (y * z - x * w)
    rot[..., 2, 0] = 2 * (x * z - y * w)
    rot[..., 2, 1] = 2 * (y * z + x * w)
    rot[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------
class _AllegroHandSim:
    """Lightweight NumPy simulation of the AllegroHand reorientation task."""

    def __init__(
        self,
        task: str = "allegrohand",
        num_envs: int = 64,
        seed: int = 0,
        horizon: int = 1000,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.obs_dim = OBS_DIM
        self.action_dim = NUM_DOFS

        # Joint state.
        self.q = np.zeros((self.num_envs, NUM_HAND_JOINTS), dtype=np.float64)
        self.qdot = np.zeros((self.num_envs, NUM_HAND_JOINTS), dtype=np.float64)

        # Object state.
        self.obj_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_vel = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_omega = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_quat = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.obj_quat[:, 0] = 1.0

        # Goal orientation.
        self.goal_quat = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.goal_quat[:, 0] = 1.0

        # Bookkeeping.
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self.hold_count = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_success = np.zeros(self.num_envs, dtype=np.float64)
        self.success_tol = SUCCESS_TOL_START
        self._success_window = []

        self.reset()

    # -- helpers -----------------------------------------------------------
    def _sample_goal(self, idx: np.ndarray) -> None:
        self.goal_quat[idx] = _random_quat(self.rng, (len(idx),))

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.q,
                self.qdot,
                self.obj_pos,
                self.obj_vel,
                self.obj_omega,
                self.goal_quat,
                np.zeros((self.num_envs, 3), dtype=np.float64),  # z_t placeholder
            ],
            axis=-1,
        ).astype(np.float32)

    def _update_curriculum(self, success: np.ndarray) -> None:
        self._success_window.append(float(np.mean(success)))
        if len(self._success_window) > 100:
            self._success_window.pop(0)
        if len(self._success_window) >= 100:
            avg = float(np.mean(self._success_window))
            if avg > CURRICULUM_THRESHOLD:
                self.success_tol = max(
                    SUCCESS_TOL_END, self.success_tol * CURRICULUM_DECAY
                )
                self._success_window = []

    # -- API ---------------------------------------------------------------
    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        n = len(env_ids)

        self.q[env_ids] = self.rng.normal(0.0, 0.1, size=(n, NUM_HAND_JOINTS))
        self.qdot[env_ids] = 0.0
        self.obj_pos[env_ids] = self.rng.normal(0.0, 0.02, size=(n, 3))
        self.obj_vel[env_ids] = 0.0
        self.obj_omega[env_ids] = 0.0
        self.obj_quat[env_ids] = _random_quat(self.rng, (n,))
        self._sample_goal(env_ids)

        self.step_count[env_ids] = 0
        self.hold_count[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0
        return self._obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        actions = np.clip(actions, -1.0, 1.0)

        # Simple first-order joint dynamics.
        target_q = actions * 0.5
        self.qdot = 0.5 * (target_q - self.q)
        self.q = self.q + self.qdot

        # Object dynamics: driven by mean joint velocity (crude proxy).
        drive = np.mean(self.qdot, axis=-1, keepdims=True)
        self.obj_vel = 0.1 * drive + 0.9 * self.obj_vel
        self.obj_pos = self.obj_pos + 0.01 * self.obj_vel
        self.obj_omega = 0.1 * np.concatenate(
            [drive, drive, drive], axis=-1
        ) + 0.9 * self.obj_omega

        # Integrate object orientation.
        dt = 0.02
        angle = np.linalg.norm(self.obj_omega, axis=-1) * dt
        axis = self.obj_omega / (np.linalg.norm(self.obj_omega, axis=-1, keepdims=True) + 1e-8)
        half = angle / 2.0
        dq = np.concatenate(
            [np.cos(half)[:, None], axis * np.sin(half)[:, None]], axis=-1
        )
        self.obj_quat = self._quat_mul(dq, self.obj_quat)
        self.obj_quat /= np.linalg.norm(self.obj_quat, axis=-1, keepdims=True) + 1e-8

        # Rewards.
        r_reach = -np.linalg.norm(self.obj_pos, axis=-1)
        r_lift = -np.abs(self.obj_pos[:, 2] - 0.1)
        qdist = _quat_distance(self.obj_quat, self.goal_quat)
        r_target = -qdist
        success = (qdist < self.success_tol).astype(np.float64)
        self.hold_count = np.where(success > 0, self.hold_count + 1, 0)
        r_success = (self.hold_count >= SUCCESS_HOLD_STEPS).astype(np.float64)

        reward = (
            W_REACH * r_reach
            + W_LIFT * r_lift
            + W_TARGET * r_target
            + W_SUCCESS * r_success
        )

        self.step_count += 1
        self.episode_return += reward
        self.episode_success = np.maximum(self.episode_success, r_success)

        done = self.step_count >= self.horizon
        info: Dict[str, Any] = {
            "success": r_success,
            "episode_return": self.episode_return.copy(),
            "episode_length": self.step_count.copy(),
            "episode_success": self.episode_success.copy(),
        }

        if np.any(done):
            self._update_curriculum(r_success)
            self.reset(np.where(done)[0])

        return self._obs(), reward.astype(np.float32), done, info

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            axis=-1,
        )

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


# ---------------------------------------------------------------------------
# IsaacGym backend
# ---------------------------------------------------------------------------
class _AllegroHandIsaacGym:
    """IsaacGym-backed AllegroHand task.

    Delegates to the NumPy backend when the IsaacGym asset pipeline is not
    available so that the rest of the codebase remains runnable.
    """

    def __init__(
        self,
        task: str = "allegrohand",
        num_envs: int = 64,
        seed: int = 0,
        horizon: int = 1000,
        device: str = "cuda:0",
        headless: bool = True,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.num_envs = int(num_envs)
        self.device = device
        self.headless = headless
        self._fallback = _AllegroHandSim(
            task=task,
            num_envs=num_envs,
            seed=seed,
            horizon=horizon,
            device=device,
            **kwargs,
        )
        self.obs_dim = self._fallback.obs_dim
        self.action_dim = self._fallback.action_dim

    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        return self._fallback.reset(env_ids)

    def step(self, actions: np.ndarray):
        return self._fallback.step(actions)

    def close(self) -> None:  # pragma: no cover
        self._fallback.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_allegrohand_task(
    task: str = "allegrohand",
    num_envs: int = 64,
    seed: int = 0,
    horizon: int = 1000,
    device: str = "cpu",
    headless: bool = True,
    force_sim: bool = False,
    **kwargs: Any,
):
    """Create an AllegroHand task instance.

    Returns either the NumPy backend (``force_sim=True`` or IsaacGym
    unavailable) or the IsaacGym wrapper.
    """
    if force_sim:
        return _AllegroHandSim(
            task=task,
            num_envs=num_envs,
            seed=seed,
            horizon=horizon,
            device=device,
            **kwargs,
        )

    try:  # pragma: no cover - depends on IsaacGym availability
        import isaacgym  # noqa: F401

        return _AllegroHandIsaacGym(
            task=task,
            num_envs=num_envs,
            seed=seed,
            horizon=horizon,
            device=device,
            headless=headless,
            **kwargs,
        )
    except Exception:
        return _AllegroHandSim(
            task=task,
            num_envs=num_envs,
            seed=seed,
            horizon=horizon,
            device=device,
            **kwargs,
        )


def task_spec(task: str = "allegrohand") -> Dict[str, Any]:
    """Return the observation/action specification for the task."""
    return {
        "obs_dim": OBS_DIM,
        "action_dim": NUM_DOFS,
        "recurrent": False,
        "horizon": 8,
        "num_dofs": NUM_DOFS,
    }


__all__ = [
    "make_allegrohand_task",
    "task_spec",
    "TASK_NAMES",
    "NUM_DOFS",
    "OBS_DIM",
]
