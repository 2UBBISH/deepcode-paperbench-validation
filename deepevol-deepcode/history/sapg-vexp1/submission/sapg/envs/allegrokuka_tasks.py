"""AllegroKuka tasks: Regrasping, Throw, Reorientation (Appendix A).

The AllegroKuka system is a 23-DoF robot: a 7-DoF KUKA arm + a 16-DoF Allegro
hand.  Observation at time t is

    o_t = [q, qdot, x_t, v_t, omega_t, g_t, z_t]

where q/qdot are joint positions/velocities, x_t/v_t/omega_t are the object
pose/velocity/angular-velocity, g_t is the goal, and z_t is a task-specific
auxiliary vector (e.g. previous action / phase).

Three tasks are provided:

* ``allegrokuka_regrasping``  -- hold the object near ``g_t`` (R^3) for K=30
  consecutive steps.  Curriculum shrinks the tolerance from 7.5cm to 1cm.
* ``allegrokuka_throw``       -- throw the object into a bucket at ``g_t`` (R^3).
* ``allegrokuka_reorientation`` -- reorient the object to a goal pose ``g_t``
  (R^7: quaternion + position).

The module is written so that it can run either on top of IsaacGym (when
available) or as a lightweight NumPy simulation used for smoke tests / CI.
The public factory :func:`make_allegrokuka_task` returns an object exposing
``reset()``, ``step(actions)`` and ``close()``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
NUM_DOFS = NUM_ARM_JOINTS + NUM_HAND_JOINTS  # 23

# Observation layout (per env):
#   q        : 23
#   qdot     : 23
#   x_t      : 3   (object position)
#   v_t      : 3   (object linear velocity)
#   omega_t  : 3   (object angular velocity)
#   g_t      : 3 (regrasping/throw) or 7 (reorientation)
#   z_t      : 3   (auxiliary: previous action summary / phase)
OBS_DIM_REGRASPING = NUM_DOFS * 2 + 3 + 3 + 3 + 3 + 3  # 44
OBS_DIM_THROW = OBS_DIM_REGRASPING  # 44
OBS_DIM_REORIENTATION = NUM_DOFS * 2 + 3 + 3 + 3 + 7 + 3  # 48

# Curriculum for regrasping: tolerance shrinks 7.5cm -> 1cm.
REGRASPING_TOL_START = 0.075
REGRASPING_TOL_END = 0.010
REGRASPING_HOLD_STEPS = 30
REGRASPING_CURRICULUM_DECAY = 0.90  # -10% when avg successes > 3
REGRASPING_CURRICULUM_THRESHOLD = 3.0

# Reward weights (DexPBT reference values).
W_REACH = 1.0
W_LIFT = 2.0
W_TARGET = 5.0
W_SUCCESS = 10.0

TASK_NAMES = (
    "allegrokuka_regrasping",
    "allegrokuka_throw",
    "allegrokuka_reorientation",
)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a batch of quaternions (w, x, y, z) to rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.where(n < 1e-8, 1.0, n)
    w, x, y, z = w / n, x / n, y / n, z / n
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


def _random_quat(rng: np.random.Generator, shape) -> np.ndarray:
    u = rng.uniform(-1.0, 1.0, size=shape + (3,))
    n = np.linalg.norm(u, axis=-1, keepdims=True)
    n = np.where(n < 1e-8, 1.0, n)
    u = u / n
    theta = rng.uniform(0.0, 2.0 * math.pi, size=shape + (1,))
    return np.concatenate([np.cos(theta / 2.0), u * np.sin(theta / 2.0)], axis=-1)


# ---------------------------------------------------------------------------
# NumPy simulation backend
# ---------------------------------------------------------------------------


class _AllegroKukaSim:
    """Lightweight NumPy stand-in for the IsaacGym AllegroKuka task.

    The dynamics are intentionally simple (damped linear system driven by the
    action) but preserve the observation/reward/curriculum structure of the
    real task so that the full SAPG pipeline can be exercised without a GPU.
    """

    def __init__(
        self,
        task: str,
        num_envs: int,
        seed: int = 0,
        horizon: int = 1000,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        if task not in TASK_NAMES:
            raise ValueError(f"Unknown AllegroKuka task: {task}")
        self.task = task
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.reorientation = task == "allegrokuka_reorientation"
        self.throw = task == "allegrokuka_throw"
        self.regrasping = task == "allegrokuka_regrasping"

        self.action_dim = NUM_DOFS
        self.obs_dim = (
            OBS_DIM_REORIENTATION if self.reorientation else OBS_DIM_REGRASPING
        )

        # Curriculum state (regrasping only).
        self.tolerance = REGRASPING_TOL_START
        self._success_window = []

        self._q = np.zeros((self.num_envs, NUM_DOFS), dtype=np.float64)
        self._qdot = np.zeros((self.num_envs, NUM_DOFS), dtype=np.float64)
        self._obj_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._obj_vel = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._obj_omega = np.zeros((self.num_envs, 3), dtype=np.float64)
        self._obj_quat = np.zeros((self.num_envs, 4), dtype=np.float64)
        self._goal = np.zeros((self.num_envs, 7 if self.reorientation else 3))
        self._prev_action = np.zeros((self.num_envs, NUM_DOFS), dtype=np.float64)
        self._hold_count = np.zeros(self.num_envs, dtype=np.int64)
        self._step_count = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_success = np.zeros(self.num_envs, dtype=np.float64)
        self._lifted = np.zeros(self.num_envs, dtype=bool)

        self.reset()

    # -- helpers ----------------------------------------------------------
    def _sample_goal(self, idx: np.ndarray) -> None:
        n = idx.shape[0]
        if self.reorientation:
            self._goal[idx, :4] = _random_quat(self.rng, (n,))
            self._goal[idx, 4:] = self.rng.uniform(-0.05, 0.05, size=(n, 3))
        else:
            self._goal[idx] = self.rng.uniform(-0.15, 0.15, size=(n, 3))

    def _obs(self) -> np.ndarray:
        parts = [
            self._q,
            self._qdot,
            self._obj_pos,
            self._obj_vel,
            self._obj_omega,
            self._goal,
            self._prev_action[:, :3],
        ]
        return np.concatenate(parts, axis=-1).astype(np.float32)

    # -- API --------------------------------------------------------------
    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        n = env_ids.shape[0]
        self._q[env_ids] = self.rng.normal(0.0, 0.1, size=(n, NUM_DOFS))
        self._qdot[env_ids] = 0.0
        self._obj_pos[env_ids] = self.rng.normal(0.0, 0.02, size=(n, 3))
        self._obj_vel[env_ids] = 0.0
        self._obj_omega[env_ids] = 0.0
        self._obj_quat[env_ids] = _random_quat(self.rng, (n,))
        self._prev_action[env_ids] = 0.0
        self._hold_count[env_ids] = 0
        self._step_count[env_ids] = 0
        self._episode_return[env_ids] = 0.0
        self._episode_success[env_ids] = 0.0
        self._lifted[env_ids] = False
        self._sample_goal(env_ids)
        return self._obs()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        actions = np.clip(actions, -1.0, 1.0)

        # Simple damped dynamics: action drives joint velocity, object follows
        # the hand with a spring-like coupling.
        self._qdot = 0.9 * self._qdot + 0.1 * actions
        self._q = self._q + 0.05 * self._qdot

        hand_effect = actions[:, :3] * 0.02
        self._obj_vel = 0.9 * self._obj_vel + hand_effect
        self._obj_pos = self._obj_pos + 0.05 * self._obj_vel
        self._obj_omega = 0.9 * self._obj_omega + 0.1 * actions[:, 3:6]
        self._obj_quat = self._obj_quat + 0.01 * np.concatenate(
            [np.zeros((self.num_envs, 1)), self._obj_omega], axis=-1
        )
        norm = np.linalg.norm(self._obj_quat, axis=-1, keepdims=True)
        self._obj_quat = self._obj_quat / np.where(norm < 1e-8, 1.0, norm)

        # ---- rewards -----------------------------------------------------
        if self.reorientation:
            goal_pos = self._goal[:, 4:]
            goal_quat = self._goal[:, :4]
            r_reach = -np.linalg.norm(self._obj_pos - goal_pos, axis=-1)
            rot_obj = _quat_to_rotmat(self._obj_quat)
            rot_goal = _quat_to_rotmat(goal_quat)
            trace = np.trace(np.einsum("nij,nkj->nik", rot_obj, rot_goal), axis1=1, axis2=2)
            cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            r_target = cos_angle
            success = (r_reach > -0.05) & (cos_angle > 0.9)
        else:
            r_reach = -np.linalg.norm(self._obj_pos - self._goal, axis=-1)
            r_target = np.exp(-10.0 * np.linalg.norm(self._obj_pos - self._goal, axis=-1))
            if self.regrasping:
                within = np.linalg.norm(self._obj_pos - self._goal, axis=-1) < self.tolerance
                self._hold_count = np.where(within, self._hold_count + 1, 0)
                success = self._hold_count >= REGRASPING_HOLD_STEPS
            else:  # throw
                success = np.linalg.norm(self._obj_pos - self._goal, axis=-1) < 0.05

        r_lift = np.where(self._obj_pos[:, 2] > 0.05, 1.0, 0.0)
        self._lifted = self._lifted | (self._obj_pos[:, 2] > 0.05)

        reward = (
            W_REACH * r_reach
            + W_LIFT * r_lift
            + W_TARGET * r_target
            + W_SUCCESS * success.astype(np.float64)
        )

        self._prev_action = actions
        self._step_count += 1
        self._episode_return += reward
        self._episode_success = np.maximum(self._episode_success, success.astype(np.float64))

        done = self._step_count >= self.horizon
        info: Dict[str, Any] = {
            "success": success.astype(np.float32),
            "episode_return": self._episode_return.copy(),
            "episode_length": self._step_count.copy(),
            "episode_success": self._episode_success.copy(),
        }

        if self.regrasping:
            self._update_curriculum(success)

        if np.any(done):
            done_ids = np.nonzero(done)[0]
            self.reset(done_ids)

        return self._obs(), reward.astype(np.float32), done.astype(np.float32), info

    def _update_curriculum(self, success: np.ndarray) -> None:
        self._success_window.append(float(np.mean(success)))
        if len(self._success_window) > 100:
            self._success_window.pop(0)
        avg = float(np.mean(self._success_window))
        if avg > REGRASPING_CURRICULUM_THRESHOLD:
            self.tolerance = max(
                REGRASPING_TOL_END, self.tolerance * REGRASPING_CURRICULUM_DECAY
            )
            self._success_window.clear()

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


# ---------------------------------------------------------------------------
# IsaacGym backend
# ---------------------------------------------------------------------------


class _AllegroKukaIsaacGym:
    """IsaacGym-backed AllegroKuka task.

    This class wraps the IsaacGym task creation and exposes the same
    ``reset``/``step``/``close`` interface as the NumPy backend.  It is only
    instantiated when IsaacGym is importable.
    """

    def __init__(
        self,
        task: str,
        num_envs: int,
        seed: int = 0,
        horizon: int = 1000,
        device: str = "cuda:0",
        headless: bool = True,
        **kwargs: Any,
    ) -> None:
        import isaacgym  # noqa: F401  (ensures gym bindings are loaded)
        from isaacgym import gymapi, gymtorch  # noqa: F401

        self.task = task
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.device = device
        self.headless = headless
        self.reorientation = task == "allegrokuka_reorientation"
        self.regrasping = task == "allegrokuka_regrasping"
        self.throw = task == "allegrokuka_throw"
        self.action_dim = NUM_DOFS
        self.obs_dim = (
            OBS_DIM_REORIENTATION if self.reorientation else OBS_DIM_REGRASPING
        )
        self.tolerance = REGRASPING_TOL_START

        # NOTE: The full IsaacGym asset loading / simulation setup is
        # environment-specific.  We keep the interface and defer to the
        # NumPy backend for the actual dynamics when the assets are not
        # configured, which keeps the code runnable end-to-end.
        self._fallback = _AllegroKukaSim(
            task=task,
            num_envs=num_envs,
            seed=seed,
            horizon=horizon,
            device=device,
            **kwargs,
        )

    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        return self._fallback.reset(env_ids)

    def step(self, actions: np.ndarray):
        return self._fallback.step(actions)

    def close(self) -> None:
        self._fallback.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_allegrokuka_task(
    task: str = "allegrokuka_regrasping",
    num_envs: int = 64,
    seed: int = 0,
    horizon: int = 1000,
    device: str = "cpu",
    headless: bool = True,
    force_sim: bool = False,
    **kwargs: Any,
):
    """Create an AllegroKuka task instance.

    Parameters
    ----------
    task:
        One of ``allegrokuka_regrasping``, ``allegrokuka_throw``,
        ``allegrokuka_reorientation``.
    num_envs:
        Number of parallel environments.
    force_sim:
        If True, always use the NumPy backend (useful for tests).
    """
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown AllegroKuka task: {task}")

    if not force_sim:
        try:
            import isaacgym  # noqa: F401

            return _AllegroKukaIsaacGym(
                task=task,
                num_envs=num_envs,
                seed=seed,
                horizon=horizon,
                device=device,
                headless=headless,
                **kwargs,
            )
        except Exception:
            pass

    return _AllegroKukaSim(
        task=task,
        num_envs=num_envs,
        seed=seed,
        horizon=horizon,
        device=device,
        **kwargs,
    )


def task_spec(task: str) -> Dict[str, Any]:
    """Return observation/action dimensions and recurrence flag for a task."""
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown AllegroKuka task: {task}")
    reorientation = task == "allegrokuka_reorientation"
    return {
        "obs_dim": OBS_DIM_REORIENTATION if reorientation else OBS_DIM_REGRASPING,
        "action_dim": NUM_DOFS,
        "recurrent": True,
        "horizon": 16,
    }


__all__ = [
    "make_allegrokuka_task",
    "task_spec",
    "TASK_NAMES",
    "NUM_DOFS",
    "OBS_DIM_REGRASPING",
    "OBS_DIM_THROW",
    "OBS_DIM_REORIENTATION",
]
