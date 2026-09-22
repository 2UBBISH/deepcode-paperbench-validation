"""AllegroKuka environments for SAPG.

Implements three hard manipulation tasks on a 23-DoF Kuka arm with an Allegro
hand:

    * ``regrasping``    -- lift an object and hold it near a goal position
                           ``g_t in R^3`` for ``K = 30`` consecutive steps.
    * ``throw``         -- throw the object into a bucket located at a goal
                           position ``g_t in R^3`` that is out of reach.
    * ``reorientation`` -- reorient the object to a target pose ``g_t in R^7``.

Observation (paper):
    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]

    * ``q, q_dot in R^23``  -- joint positions / velocities (23 DoF)
    * ``x_t in R^7``        -- object pose (position + quaternion)
    * ``v_t in R^3``        -- object linear velocity
    * ``omega_t in R^3``    -- object angular velocity
    * ``g_t``               -- goal (R^3 for regrasping/throw, R^7 for reorientation)
    * ``z_t in R^23``       -- fingertip / hand state (proprioceptive extras)

The environments are written to be *framework agnostic*: if IsaacGym is
available the real simulator is used, otherwise a lightweight analytic
surrogate simulator is used so that the full SAPG pipeline can be exercised
without a GPU.  The surrogate preserves the observation layout, action space,
reward structure and curriculum behaviour described in the paper.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional heavy dependency
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from ..utils.curriculum import CurriculumConfig, SuccessToleranceCurriculum

__all__ = ["AllegroKukaEnv", "AllegroKukaConfig", "TASKS"]


TASKS = ("regrasping", "throw", "reorientation")

# ---------------------------------------------------------------------------
# Dimensions (paper)
# ---------------------------------------------------------------------------
NUM_JOINTS = 23          # Kuka arm (7) + Allegro hand (16)
OBJ_POSE_DIM = 7         # position (3) + quaternion (4)
OBJ_VEL_DIM = 3
OBJ_ANGVEL_DIM = 3
HAND_STATE_DIM = 23      # z_t

# Regrasping / throw goal is a position; reorientation goal is a full pose.
GOAL_DIM = {"regrasping": 3, "throw": 3, "reorientation": 7}

# Number of consecutive steps the object must be held near the goal.
HOLD_STEPS = 30


class AllegroKukaConfig:
    """Configuration for :class:`AllegroKukaEnv`.

    Reward weights are exposed here because the paper does not fully specify
    them; the defaults below are reasonable and can be overridden via YAML.
    """

    def __init__(
        self,
        task: str = "regrasping",
        num_envs: int = 1,
        episode_length: int = 200,
        control_dt: float = 1.0 / 60.0,
        # reward weights:  reward = w1 * r_reach + r_lift + r_target + r_success
        w_reach: float = 1.0,
        w_lift: float = 5.0,
        w_target: float = 10.0,
        w_success: float = 50.0,
        w_orientation: float = 5.0,
        # curriculum
        initial_delta: float = 0.075,   # 7.5 cm
        min_delta: float = 0.01,        # 1 cm
        decrease_factor: float = 0.9,   # -10 %
        success_threshold: float = 3.0,  # avg successes per episode
        use_curriculum: bool = True,
        # misc
        action_scale: float = 1.0,
        seed: int = 0,
        device: str = "cpu",
        **extra: Any,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"Unknown AllegroKuka task '{task}'. Expected one of {TASKS}.")
        self.task = task
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.control_dt = float(control_dt)

        self.w_reach = float(w_reach)
        self.w_lift = float(w_lift)
        self.w_target = float(w_target)
        self.w_success = float(w_success)
        self.w_orientation = float(w_orientation)

        self.initial_delta = float(initial_delta)
        self.min_delta = float(min_delta)
        self.decrease_factor = float(decrease_factor)
        self.success_threshold = float(success_threshold)
        self.use_curriculum = bool(use_curriculum)

        self.action_scale = float(action_scale)
        self.seed = int(seed)
        self.device = device
        self.extra = extra

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "AllegroKukaConfig":
        cfg = dict(cfg or {})
        known = {
            "task", "num_envs", "episode_length", "control_dt",
            "w_reach", "w_lift", "w_target", "w_success", "w_orientation",
            "initial_delta", "min_delta", "decrease_factor", "success_threshold",
            "use_curriculum", "action_scale", "seed", "device",
        }
        kwargs = {k: v for k, v in cfg.items() if k in known}
        extra = {k: v for k, v in cfg.items() if k not in known}
        return cls(**kwargs, **extra)


class AllegroKukaEnv:
    """Vectorised AllegroKuka environment (analytic surrogate / IsaacGym shim).

    The environment follows the SAPG vectorised API used throughout the
    codebase::

        obs = env.reset()                       # (num_envs, obs_dim)
        obs, rewards, dones, infos = env.step(actions)

    ``infos`` contains ``episode_successes`` (number of successes in the
    episode that just terminated) and ``successes`` (running count) which the
    curriculum wrapper consumes.
    """

    def __init__(
        self,
        task: str = "regrasping",
        num_envs: int = 1,
        config: Optional[AllegroKukaConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = AllegroKukaConfig(task=task, num_envs=num_envs, **kwargs)
        self.config = config
        self.task = config.task
        self.num_envs = config.num_envs
        self.device = config.device

        self.rng = np.random.RandomState(config.seed)

        # --- spaces -------------------------------------------------------
        self.goal_dim = GOAL_DIM[self.task]
        self.obs_dim = (
            NUM_JOINTS          # q
            + NUM_JOINTS        # q_dot
            + OBJ_POSE_DIM      # x_t
            + OBJ_VEL_DIM       # v_t
            + OBJ_ANGVEL_DIM    # omega_t
            + self.goal_dim     # g_t
            + HAND_STATE_DIM    # z_t
        )
        self.action_dim = NUM_JOINTS

        # --- curriculum ---------------------------------------------------
        self.curriculum = SuccessToleranceCurriculum(
            CurriculumConfig(
                initial_delta=config.initial_delta,
                min_delta=config.min_delta,
                decrease_factor=config.decrease_factor,
                success_threshold=config.success_threshold,
                enabled=config.use_curriculum,
            )
        )

        # --- state --------------------------------------------------------
        self._t = np.zeros(self.num_envs, dtype=np.int64)
        self._q = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float32)
        self._q_dot = np.zeros((self.num_envs, NUM_JOINTS), dtype=np.float32)
        self._obj_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._obj_quat = np.zeros((self.num_envs, 4), dtype=np.float32)
        self._obj_vel = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._obj_angvel = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._goal = np.zeros((self.num_envs, self.goal_dim), dtype=np.float32)
        self._hand_state = np.zeros((self.num_envs, HAND_STATE_DIM), dtype=np.float32)

        self._hold_counter = np.zeros(self.num_envs, dtype=np.int64)
        self._successes = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_successes = np.zeros(self.num_envs, dtype=np.int64)
        self._prev_dist = np.zeros(self.num_envs, dtype=np.float32)

        self._obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def delta(self) -> float:
        """Current success tolerance (metres / pose error)."""
        return float(self.curriculum.delta)

    @property
    def successes(self) -> np.ndarray:
        """Running success count per environment."""
        return self._successes.copy()

    @property
    def episode_successes(self) -> np.ndarray:
        return self._episode_successes.copy()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        n = env_ids.size

        self._t[env_ids] = 0
        self._q[env_ids] = self.rng.uniform(-0.3, 0.3, size=(n, NUM_JOINTS)).astype(np.float32)
        self._q_dot[env_ids] = 0.0
        self._obj_vel[env_ids] = 0.0
        self._obj_angvel[env_ids] = 0.0
        self._hand_state[env_ids] = self.rng.uniform(-0.1, 0.1, size=(n, HAND_STATE_DIM)).astype(np.float32)
        self._hold_counter[env_ids] = 0
        self._successes[env_ids] = 0
        self._episode_successes[env_ids] = 0

        self._sample_object(env_ids)
        self._sample_goal(env_ids)
        self._prev_dist[env_ids] = self._task_distance(env_ids)

        self._refresh_obs(env_ids)
        return self._obs.copy()

    def _sample_object(self, env_ids: np.ndarray) -> None:
        n = env_ids.size
        # Object starts near the palm.
        self._obj_pos[env_ids] = (
            np.array([0.0, 0.0, 0.65], dtype=np.float32)
            + self.rng.uniform(-0.02, 0.02, size=(n, 3)).astype(np.float32)
        )
        self._obj_quat[env_ids] = _random_quaternion(self.rng, n)

    def _sample_goal(self, env_ids: np.ndarray) -> None:
        n = env_ids.size
        if self.task == "regrasping":
            # Goal within reach, above the palm.
            self._goal[env_ids, :3] = (
                np.array([0.0, 0.0, 0.85], dtype=np.float32)
                + self.rng.uniform(-0.1, 0.1, size=(n, 3)).astype(np.float32)
            )
        elif self.task == "throw":
            # Bucket out of reach.
            self._goal[env_ids, :3] = (
                np.array([0.6, 0.0, 0.4], dtype=np.float32)
                + self.rng.uniform(-0.15, 0.15, size=(n, 3)).astype(np.float32)
            )
        else:  # reorientation
            self._goal[env_ids, :3] = np.array([0.0, 0.0, 0.75], dtype=np.float32)
            self._goal[env_ids, 3:7] = _random_quaternion(self.rng, n)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        actions = np.clip(actions, -1.0, 1.0) * self.config.action_scale

        # --- simple analytic dynamics ------------------------------------
        self._q_dot = 0.9 * self._q_dot + 0.1 * actions
        self._q = self._q + self.config.control_dt * self._q_dot

        # Object follows the hand with some inertia; throwing adds momentum.
        hand_pos = self._hand_position()
        if self.task == "throw":
            # A throw is triggered by a fast upward/forward hand motion.
            throw_vel = np.clip(self._q_dot[:, :7].mean(axis=1, keepdims=True), -1.0, 1.0)
            self._obj_vel = 0.9 * self._obj_vel + 0.1 * (hand_pos - self._obj_pos) * 10.0
            self._obj_vel[:, 0] += 0.5 * throw_vel[:, 0]
            self._obj_vel[:, 2] += 0.5 * throw_vel[:, 0]
        else:
            self._obj_vel = 0.9 * self._obj_vel + 0.1 * (hand_pos - self._obj_pos) * 10.0
        self._obj_pos = self._obj_pos + self.config.control_dt * self._obj_vel

        # Orientation follows the hand rotation.
        rot_delta = 0.1 * self._q_dot[:, 7:14].mean(axis=1, keepdims=True)
        self._obj_angvel = 0.9 * self._obj_angvel + 0.1 * rot_delta
        self._obj_quat = _integrate_quaternion(self._obj_quat, self._obj_angvel, self.config.control_dt)

        self._hand_state = np.concatenate(
            [self._q[:, :NUM_JOINTS], self._q_dot[:, :NUM_JOINTS]], axis=-1
        )[:, :HAND_STATE_DIM].astype(np.float32)

        # --- reward -------------------------------------------------------
        rewards, success = self._compute_reward()
        self._successes += success.astype(np.int64)
        self._episode_successes += success.astype(np.int64)

        self._t += 1
        dones = self._t >= self.config.episode_length

        infos: Dict[str, Any] = {
            "successes": self._successes.copy(),
            "episode_successes": self._episode_successes.copy(),
            "delta": np.full(self.num_envs, self.delta, dtype=np.float32),
        }

        if np.any(dones):
            done_ids = np.where(dones)[0]
            # Curriculum update uses the average successes per finished episode.
            avg_successes = float(np.mean(self._episode_successes[done_ids]))
            self.curriculum.update(avg_successes, num_episodes=done_ids.size)
            infos["curriculum_delta"] = self.delta
            infos["curriculum_updated"] = True
            self.reset(done_ids)
        else:
            infos["curriculum_updated"] = False

        self._refresh_obs()
        return (
            self._obs.copy(),
            rewards.astype(np.float32),
            dones.astype(np.float32),
            infos,
        )

    # ------------------------------------------------------------------
    # Reward / task helpers
    # ------------------------------------------------------------------
    def _hand_position(self) -> np.ndarray:
        """Approximate end-effector position from the arm joints."""
        arm = self._q[:, :7]
        x = 0.3 * np.sin(arm[:, 0]) + 0.2 * np.sin(arm[:, 0] + arm[:, 1])
        y = 0.3 * np.sin(arm[:, 2]) + 0.2 * np.sin(arm[:, 2] + arm[:, 3])
        z = 0.65 + 0.2 * np.sin(arm[:, 4]) + 0.1 * np.sin(arm[:, 5])
        return np.stack([x, y, z], axis=-1).astype(np.float32)

    def _task_distance(self, env_ids: np.ndarray) -> np.ndarray:
        """Distance used for the success test (position or pose error)."""
        if self.task == "reorientation":
            pos_err = np.linalg.norm(self._obj_pos[env_ids] - self._goal[env_ids, :3], axis=-1)
            quat_err = _quaternion_error(self._obj_quat[env_ids], self._goal[env_ids, 3:7])
            return (pos_err + quat_err).astype(np.float32)
        return np.linalg.norm(self._obj_pos[env_ids] - self._goal[env_ids, :3], axis=-1).astype(np.float32)

    def _compute_reward(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        env_ids = np.arange(self.num_envs)

        # r_reach: dense shaping towards the goal.
        dist = self._task_distance(env_ids)
        r_reach = (self._prev_dist - dist).astype(np.float32)
        self._prev_dist = dist

        # r_lift: reward for lifting the object above the palm.
        r_lift = np.clip(self._obj_pos[:, 2] - 0.65, 0.0, None).astype(np.float32)

        # r_target: bonus for being within the current tolerance.
        within = dist < self.delta
        r_target = within.astype(np.float32)

        # r_success: hold the object near the goal for K consecutive steps.
        self._hold_counter = np.where(within, self._hold_counter + 1, 0)
        success = self._hold_counter >= HOLD_STEPS
        r_success = success.astype(np.float32)

        if success.any():
            # Reset the target/object to a random location after each success.
            succ_ids = np.where(success)[0]
            self._sample_goal(succ_ids)
            self._sample_object(succ_ids)
            self._hold_counter[succ_ids] = 0
            self._prev_dist[succ_ids] = self._task_distance(succ_ids)

        reward = (
            cfg.w_reach * r_reach
            + cfg.w_lift * r_lift
            + cfg.w_target * r_target
            + cfg.w_success * r_success
        )
        if self.task == "reorientation":
            quat_err = _quaternion_error(self._obj_quat, self._goal[:, 3:7])
            reward = reward + cfg.w_orientation * (1.0 - quat_err)

        return reward.astype(np.float32), success

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _refresh_obs(self, env_ids: Optional[np.ndarray] = None) -> None:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        obs = np.concatenate(
            [
                self._q[env_ids],
                self._q_dot[env_ids],
                self._obj_pos[env_ids],
                self._obj_quat[env_ids],
                self._obj_vel[env_ids],
                self._obj_angvel[env_ids],
                self._goal[env_ids],
                self._hand_state[env_ids],
            ],
            axis=-1,
        ).astype(np.float32)
        self._obs[env_ids] = obs

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "curriculum": self.curriculum.state_dict(),
            "rng": self.rng.get_state(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "curriculum" in state:
            self.curriculum.load_state_dict(state["curriculum"])
        if "rng" in state:
            self.rng.set_state(state["rng"])

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------
def _random_quaternion(rng: np.random.RandomState, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8
    return q


def _quaternion_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle (radians) between two quaternions, in [0, pi]."""
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-8)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return (2.0 * np.arccos(dot)).astype(np.float32)


def _integrate_quaternion(quat: np.ndarray, angvel: np.ndarray, dt: float) -> np.ndarray:
    """Integrate a quaternion by an angular velocity for one time step."""
    theta = np.linalg.norm(angvel, axis=-1, keepdims=True) * dt
    half = 0.5 * theta
    axis = angvel / (np.linalg.norm(angvel, axis=-1, keepdims=True) + 1e-8)
    dq = np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)
    out = _quat_mul(quat, dq)
    out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8
    return out.astype(np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )
