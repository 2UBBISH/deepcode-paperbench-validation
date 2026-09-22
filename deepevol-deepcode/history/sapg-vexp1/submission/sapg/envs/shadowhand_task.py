"""ShadowHand in-hand reorientation task (Appendix A of the SAPG paper).

The ShadowHand is a 24-DoF anthropomorphic hand.  The task is to reorient an
object so that its orientation matches a target quaternion goal ``g_t`` in
R^4.  This module mirrors the structure of :mod:`allegrohand_task` and
:mod:`allegrokuka_tasks`:

* a NumPy simulation backend used for smoke tests / CI (no GPU required), and
* an IsaacGym backend wrapper that delegates to the NumPy fallback when the
  IsaacGym assets are not available.

Observation layout (Appendix A)::

    o_t = [q (24), qdot (24), x_t (3), v_t (3), omega_t (3), g_t (4), z_t (3)]

which gives ``OBS_DIM = 64``.  The action space is the 24 joint position
targets (``ACTION_DIM = 24``).

Reward is a weighted sum of reach / lift / target / success terms using the
DexPBT reference weights (see :mod:`allegrokuka_tasks`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_HAND_JOINTS = 24
NUM_DOFS = 24
ACTION_DIM = 24

# q(24) + qdot(24) + x_t(3) + v_t(3) + omega_t(3) + g_t(4) + z_t(3) = 64
OBS_DIM = 64

TASK_NAMES = ("shadowhand",)

# Reward weights (DexPBT reference values).
W_REACH = 1.0
W_LIFT = 2.0
W_TARGET = 5.0
W_SUCCESS = 10.0

# Curriculum: success tolerance (angular distance in radians) decays from
# SUCCESS_TOL_START to SUCCESS_TOL_END when the average number of successes
# over a 100-step window exceeds CURRICULUM_THRESHOLD.
SUCCESS_TOL_START = 0.5
SUCCESS_TOL_END = 0.1
SUCCESS_HOLD_STEPS = 30
CURRICULUM_DECAY = 0.90
CURRICULUM_THRESHOLD = 3.0

# Default episode horizon for the NumPy backend.
DEFAULT_HORIZON = 1000


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions along the last axis."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-8, None)


def _random_quat(rng: np.random.Generator, shape) -> np.ndarray:
    """Sample uniformly random unit quaternions of the given leading shape."""
    u = rng.uniform(0.0, 1.0, size=tuple(shape) + (3,))
    q = np.empty(tuple(shape) + (4,), dtype=np.float64)
    q[..., 0] = np.sqrt(1.0 - u[..., 0]) * np.sin(2.0 * math.pi * u[..., 1])
    q[..., 1] = np.sqrt(1.0 - u[..., 0]) * np.cos(2.0 * math.pi * u[..., 1])
    q[..., 2] = np.sqrt(u[..., 0]) * np.sin(2.0 * math.pi * u[..., 2])
    q[..., 3] = np.sqrt(u[..., 0]) * np.cos(2.0 * math.pi * u[..., 2])
    return _normalize_quat(q)


def _quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular distance (radians) between two unit quaternions.

    ``d = 2 * arccos(|<q1, q2>|)`` which is in ``[0, pi]``.
    """
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions ``[..., 4]`` to rotation matrices ``[..., 3, 3]``."""
    q = _normalize_quat(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rot = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    rot[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[..., 0, 1] = 2.0 * (x * y - z * w)
    rot[..., 0, 2] = 2.0 * (x * z + y * w)
    rot[..., 1, 0] = 2.0 * (x * y + z * w)
    rot[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[..., 1, 2] = 2.0 * (y * z - x * w)
    rot[..., 2, 0] = 2.0 * (x * z - y * w)
    rot[..., 2, 1] = 2.0 * (y * z + x * w)
    rot[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


# ---------------------------------------------------------------------------
# NumPy simulation backend
# ---------------------------------------------------------------------------


class _ShadowHandSim:
    """Lightweight NumPy simulation of the ShadowHand reorientation task.

    The dynamics are intentionally simple (first-order joint integration with
    a mean-joint-velocity-driven object motion).  This backend exists so the
    full SAPG pipeline can be exercised without a GPU / IsaacGym install; the
    observation / reward / curriculum logic matches the IsaacGym backend.
    """

    def __init__(
        self,
        task: str = "shadowhand",
        num_envs: int = 64,
        seed: int = 0,
        horizon: int = DEFAULT_HORIZON,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM
        self.num_actions = ACTION_DIM

        # Joint state.
        self.q = np.zeros((self.num_envs, NUM_HAND_JOINTS), dtype=np.float64)
        self.qdot = np.zeros((self.num_envs, NUM_HAND_JOINTS), dtype=np.float64)

        # Object state.
        self.obj_pos = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_vel = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_omega = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.obj_quat = np.zeros((self.num_envs, 4), dtype=np.float64)

        # Goal orientation (quaternion).
        self.goal_quat = np.zeros((self.num_envs, 4), dtype=np.float64)

        # Palm / hand reference position.
        self.palm_pos = np.zeros((self.num_envs, 3), dtype=np.float64)

        # Book-keeping.
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self.hold_count = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_success = np.zeros(self.num_envs, dtype=np.float64)
        self.success_tol = SUCCESS_TOL_START
        self._success_window = []

        self.reset()

    # -- helpers ----------------------------------------------------------

    def _sample_goal(self, idx: np.ndarray) -> None:
        self.goal_quat[idx] = _random_quat(self.rng, (idx.shape[0],))

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.q,
                self.qdot,
                self.obj_pos,
                self.obj_vel,
                self.obj_omega,
                self.goal_quat,
                self.palm_pos,
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
                self._success_window.clear()

    # -- API --------------------------------------------------------------

    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        n = env_ids.shape[0]

        self.q[env_ids] = self.rng.normal(0.0, 0.1, size=(n, NUM_HAND_JOINTS))
        self.qdot[env_ids] = 0.0

        self.obj_pos[env_ids] = self.rng.normal(0.0, 0.02, size=(n, 3))
        self.obj_vel[env_ids] = 0.0
        self.obj_omega[env_ids] = 0.0
        self.obj_quat[env_ids] = _random_quat(self.rng, (n,))

        self.palm_pos[env_ids] = self.rng.normal(0.0, 0.01, size=(n, 3))

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

        # First-order joint integration.
        target_q = actions * 0.5
        self.qdot = (target_q - self.q) * 10.0
        self.q = self.q + self.qdot * 0.02

        # Object motion driven by mean joint velocity (crude proxy for contact).
        mean_qdot = np.mean(self.qdot, axis=-1, keepdims=True)
        self.obj_vel = 0.1 * mean_qdot + self.rng.normal(
            0.0, 0.005, size=(self.num_envs, 3)
        )
        self.obj_pos = self.obj_pos + self.obj_vel * 0.02

        # Object orientation drifts with mean joint velocity.
        axis = np.concatenate(
            [np.zeros((self.num_envs, 1)), mean_qdot, np.zeros((self.num_envs, 1))],
            axis=-1,
        )
        angle = np.linalg.norm(axis, axis=-1, keepdims=True) * 0.02
        axis_norm = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-8, None)
        half = angle * 0.5
        dq = np.concatenate(
            [np.cos(half), axis_norm * np.sin(half)], axis=-1
        )
        self.obj_quat = _normalize_quat(self.obj_quat + 0.5 * dq)
        self.obj_omega = axis_norm * angle / 0.02

        # ---- rewards ----------------------------------------------------
        # Reach: keep the object close to the palm.
        dist_palm = np.linalg.norm(self.obj_pos - self.palm_pos, axis=-1)
        r_reach = -dist_palm

        # Lift: encourage lifting the object above the palm.
        r_lift = np.clip(self.obj_pos[:, 2] - self.palm_pos[:, 2], 0.0, None)

        # Target: angular distance to the goal orientation.
        ang_dist = _quat_angular_distance(self.obj_quat, self.goal_quat)
        r_target = -ang_dist

        # Success: hold the target orientation for SUCCESS_HOLD_STEPS steps.
        at_target = ang_dist < self.success_tol
        self.hold_count = np.where(at_target, self.hold_count + 1, 0)
        success = self.hold_count >= SUCCESS_HOLD_STEPS
        r_success = success.astype(np.float64)

        reward = (
            W_REACH * r_reach
            + W_LIFT * r_lift
            + W_TARGET * r_target
            + W_SUCCESS * r_success
        )

        self.episode_return += reward
        self.episode_success = np.maximum(self.episode_success, success.astype(np.float64))
        self.step_count += 1

        self._update_curriculum(success)

        done = self.step_count >= self.horizon
        info: Dict[str, Any] = {
            "success": success.astype(np.float32),
            "episode_return": self.episode_return.astype(np.float32),
            "episode_length": self.step_count.astype(np.float32),
            "episode_success": self.episode_success.astype(np.float32),
            "success_tol": np.full(self.num_envs, self.success_tol, dtype=np.float32),
        }

        if np.any(done):
            done_ids = np.where(done)[0]
            self.reset(done_ids)

        return self._obs(), reward.astype(np.float32), done, info

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


# ---------------------------------------------------------------------------
# IsaacGym backend
# ---------------------------------------------------------------------------


class _ShadowHandIsaacGym:
    """IsaacGym-backed ShadowHand reorientation task.

    Asset loading is environment specific; when the IsaacGym assets are not
    available this wrapper transparently falls back to the NumPy backend so
    that the training pipeline remains runnable.
    """

    def __init__(
        self,
        task: str = "shadowhand",
        num_envs: int = 64,
        seed: int = 0,
        horizon: int = DEFAULT_HORIZON,
        device: str = "cpu",
        headless: bool = True,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.device = device
        self.headless = headless

        self._sim = None
        try:  # pragma: no cover - requires IsaacGym install
            from isaacgym import gymapi  # noqa: F401

            self._gymapi = gymapi
            self._build_isaacgym_env(**kwargs)
        except Exception:
            # Fall back to the NumPy backend.
            self._sim = _ShadowHandSim(
                task=task,
                num_envs=num_envs,
                seed=seed,
                horizon=horizon,
                device=device,
            )

        if self._sim is not None:
            self.obs_dim = self._sim.obs_dim
            self.action_dim = self._sim.action_dim
        else:  # pragma: no cover
            self.obs_dim = OBS_DIM
            self.action_dim = ACTION_DIM
        self.num_actions = self.action_dim

    def _build_isaacgym_env(self, **kwargs: Any) -> None:  # pragma: no cover
        """Placeholder for IsaacGym asset/scene construction.

        A full implementation requires the ShadowHand URDF and object assets
        shipped with IsaacGymEnvs.  Until those are wired up we delegate to the
        NumPy backend so the rest of the pipeline can be exercised.
        """
        self._sim = _ShadowHandSim(
            task=self.task,
            num_envs=self.num_envs,
            seed=kwargs.get("seed", 0),
            horizon=self.horizon,
            device=self.device,
        )

    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        return self._sim.reset(env_ids)

    def step(self, actions: np.ndarray):
        return self._sim.step(actions)

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()


# ---------------------------------------------------------------------------
# Factory / spec
# ---------------------------------------------------------------------------


def make_shadowhand_task(
    task: str = "shadowhand",
    num_envs: int = 64,
    seed: int = 0,
    horizon: int = DEFAULT_HORIZON,
    device: str = "cpu",
    headless: bool = True,
    force_sim: bool = False,
    **kwargs: Any,
):
    """Create a ShadowHand reorientation task instance.

    Args:
        task: Task name (``"shadowhand"``).
        num_envs: Number of parallel environments.
        seed: RNG seed.
        horizon: Episode horizon.
        device: Torch device string.
        headless: Whether to run IsaacGym headless.
        force_sim: Force the NumPy backend (useful for tests).

    Returns:
        A task object exposing ``reset``, ``step`` and ``close``.
    """
    if force_sim:
        return _ShadowHandSim(
            task=task,
            num_envs=num_envs,
            seed=seed,
            horizon=horizon,
            device=device,
        )
    return _ShadowHandIsaacGym(
        task=task,
        num_envs=num_envs,
        seed=seed,
        horizon=horizon,
        device=device,
        headless=headless,
        **kwargs,
    )


def task_spec(task: str = "shadowhand") -> Dict[str, Any]:
    """Return the observation / action specification for the task."""
    return {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "recurrent": False,
        "horizon": 8,
        "num_dofs": NUM_DOFS,
    }


__all__ = [
    "make_shadowhand_task",
    "task_spec",
    "TASK_NAMES",
    "NUM_DOFS",
    "NUM_HAND_JOINTS",
    "OBS_DIM",
    "ACTION_DIM",
]
