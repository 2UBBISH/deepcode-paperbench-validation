"""Allegro Hand in-hand reorientation environment.

Implements the *easy* Allegro Hand task of the SAPG paper (Appendix A, part 2):

    "Allegro Hand: This is the same as the previous in-hand reorientation task
     but with the 16-DoF Allegro Hand instead."

so the task is the in-hand reorientation of a cube with the 16-DoF Allegro Hand
mounted in the palm, where the goal is a target orientation given as a quaternion
``g_t in R^4``.  The reward is a combination of the orientation error and a
success bonus, and the metric reported in the paper is the *net episode reward*
(rather than the success-count metric used by the hard Allegro-Kuka tasks).

Observation layout (following the joint task observation used by the paper,
``o_t = [q, qdot, x_t, v_t, omega_t, g_t]``):

    ==================  =====  ==============================================
    Field               Dim    Meaning
    ==================  =====  ==============================================
    ``q``               16     Allegro Hand joint angles
    ``qdot``            16     joint velocities
    ``object_pose``      7     cube pose (position + quaternion)
    ``object_lin_vel``   3     cube linear velocity
    ``object_ang_vel``   3     cube angular velocity
    ``goal_quat``        4     goal cube orientation (quaternion)
    ==================  =====  ==============================================

giving ``obs_dim = 49``.

The module is deliberately simulator agnostic: it wraps the uniform
``sapg.envs.isaac_env`` facade, so it works with the real IsaacGym backend when
available and with the pure-``torch`` surrogate otherwise.  ``torch`` is imported
lazily so the module can also be imported (and unit-tested) without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from .isaac_env import (  # noqa: F401  (re-exported for convenience)
    GOAL_DIMS,
    JOINT_DIMS,
    TASK_REGISTRY,
    IsaacEnv,
    action_dim_for,
    goal_dim_for,
    joint_dim_for,
    make_isaac_env,
    obs_dim_for,
    quat_angle_error,
    quat_conjugate,
    quat_mul,
    quat_normalize,
    quat_random,
)

try:  # pragma: no cover - torch is always present in the training environment
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


__all__ = [
    "ALLEGRO_HAND",
    "IN_HAND_REORIENTATION",
    "ALLEGRO_HAND_TASKS",
    "AllegroHandEnv",
    "AllegroHandRewardWeights",
    "SuccessTracker",
    "make_allegro_hand_env",
    "make_in_hand_reorientation_env",
    "make_allegro_hand_reorientation_env",
    "split_observation",
    "default_reward_weights_allegro_hand",
    "ALLEGRO_HAND_JOINTS",
    "GOAL_QUAT_DIM",
    "OBJECT_POSE_DIM",
    "OBJECT_VEL_DIM",
    "ORIENTATION_REWARD_WEIGHT",
    "SUCCESS_BONUS",
    "ROT_TOLERANCE_DEFAULT",
    "HOLD_STEPS_DEFAULT",
    "MAX_EPISODE_LENGTH_DEFAULT",
]


# ---------------------------------------------------------------------------
# Constants (Appendix A)
# ---------------------------------------------------------------------------

ALLEGRO_HAND = "allegro_hand"
IN_HAND_REORIENTATION = "in_hand_reorientation"
ALLEGRO_HAND_TASKS: Tuple[str, ...] = (ALLEGRO_HAND, IN_HAND_REORIENTATION)

ALLEGRO_HAND_JOINTS = 16
GOAL_QUAT_DIM = 4
OBJECT_POSE_DIM = 7
OBJECT_VEL_DIM = 3

#: reward is "a combination of the orientation error and a success bonus"
ORIENTATION_REWARD_WEIGHT = 1.0
SUCCESS_BONUS = 5.0

#: the easy tasks do not use the success-tolerance curriculum of the hard tasks,
#: but a rotation tolerance is still used to decide when a re-orientation counts
#: as an achieved goal.
ROT_TOLERANCE_DEFAULT = math.pi / 6.0
HOLD_STEPS_DEFAULT = 10
MAX_EPISODE_LENGTH_DEFAULT = 200


def default_reward_weights_allegro_hand() -> Dict[str, float]:
    """Paper default reward weights for the Allegro Hand reorientation task."""
    return {
        "orientation": ORIENTATION_REWARD_WEIGHT,
        "success": SUCCESS_BONUS,
        "reach": 0.0,
        "shaping_power": 1.0,
        "hold": 0.0,
    }


# ---------------------------------------------------------------------------
# Small tensor/float helpers (mirror the torch-free friendly style of
# shadow_hand.py so that both easy tasks can be used interchangeably)
# ---------------------------------------------------------------------------


def _is_tensor(x: Any) -> bool:
    return _HAS_TORCH and torch.is_tensor(x)


def _to_tensor(x: Any, device: Any = None, dtype: Any = None) -> Any:
    if not _HAS_TORCH:
        return x
    if torch.is_tensor(x):
        if device is not None or dtype is not None:
            return x.to(device=device or x.device, dtype=dtype or x.dtype)
        return x
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)


def _norm_last(x: Any, dim: int = -1) -> Any:
    if _is_tensor(x):
        return torch.linalg.vector_norm(x, dim=dim)
    return math.sqrt(sum(float(v) * float(v) for v in x))


def _sqrt(x: Any) -> Any:
    if _is_tensor(x):
        return torch.sqrt(x)
    return math.sqrt(float(x))


def _zeros(n: Any, device: Any = None, dtype: Any = None) -> Any:
    if _HAS_TORCH:
        if isinstance(n, int):
            return torch.zeros(n, device=device, dtype=dtype or torch.float32)
        return torch.zeros_like(
            n, device=device or getattr(n, "device", None), dtype=dtype or getattr(n, "dtype", None)
        )
    if isinstance(n, int):
        return [0.0] * n
    return [0.0 for _ in n]


def _quat_angle_to(quat: Any, goal: Any, eps: float = 1e-6) -> Any:
    """Angle (radians) between two quaternions, tolerating either convention."""
    try:
        return quat_angle_error(quat, goal)
    except Exception:
        pass
    if _is_tensor(quat):
        q = quat_normalize(quat)
        g = quat_normalize(goal)
        # q * conj(g) -> scalar part gives the rotation angle
        prod = quat_mul(q, quat_conjugate(g))
        w = torch.clamp(torch.abs(prod[..., 0]), max=1.0 - eps)
        return 2.0 * torch.acos(w)
    # list/float fallback
    q = list(quat)
    g = list(goal)
    dot = abs(sum(float(a) * float(b) for a, b in zip(q, g)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def _normalise_list(v: Any, eps: float = 1e-8) -> Any:
    if _is_tensor(v):
        return quat_normalize(v)
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    n = max(n, eps)
    return [float(x) / n for x in v]


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------


@dataclass
class AllegroHandRewardWeights:
    """Reward coefficients for the Allegro Hand reorientation task."""

    orientation: float = ORIENTATION_REWARD_WEIGHT
    success: float = SUCCESS_BONUS
    reach: float = 0.0
    shaping_power: float = 1.0
    hold: float = 0.0

    @classmethod
    def from_any(cls, weights: Any = None, task: str = ALLEGRO_HAND) -> "AllegroHandRewardWeights":
        base = default_reward_weights_allegro_hand()
        if weights is None:
            return cls(**base)
        if isinstance(weights, AllegroHandRewardWeights):
            return weights
        if hasattr(weights, "__dataclass_fields__"):
            base.update({k: getattr(weights, k) for k in base if hasattr(weights, k)})
        elif isinstance(weights, dict):
            base.update({k: v for k, v in weights.items() if k in base})
        elif hasattr(weights, "__dict__"):
            base.update({k: getattr(weights, k) for k in base if hasattr(weights, k)})
        return cls(**base)

    def as_dict(self) -> Dict[str, float]:
        return {
            "orientation": float(self.orientation),
            "success": float(self.success),
            "reach": float(self.reach),
            "shaping_power": float(self.shaping_power),
            "hold": float(self.hold),
        }


# ---------------------------------------------------------------------------
# Success tracking (K consecutive steps within tolerance)
# ---------------------------------------------------------------------------


class SuccessTracker:
    """Track per-env successes requiring ``hold_steps`` consecutive in-tolerance steps."""

    def __init__(self, num_envs: int, hold_steps: int = HOLD_STEPS_DEFAULT, device: Any = None):
        self.num_envs = int(num_envs)
        self.hold_steps = int(hold_steps) if hold_steps and hold_steps > 0 else 1
        self.device = device
        if _HAS_TORCH:
            self._counter = torch.zeros(self.num_envs, dtype=torch.long, device=device)
            self._active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        else:  # pragma: no cover
            self._counter = [0] * self.num_envs
            self._active = [False] * self.num_envs
        self.success_counts = _zeros(self.num_envs, device=device, dtype=torch.long if _HAS_TORCH else None)
        self.total_successes = 0

    # -- helpers ---------------------------------------------------------
    def _as_bool_mask(self, within_tolerance: Any) -> Any:
        if _is_tensor(within_tolerance):
            return within_tolerance.to(dtype=torch.bool)
        if isinstance(within_tolerance, (list, tuple)):
            return [bool(x) for x in within_tolerance]
        return bool(within_tolerance)

    def reset(self, env_ids: Any = None) -> None:
        """Reset the hold counters (and optionally per-env success counts)."""
        if env_ids is None:
            if _HAS_TORCH and _is_tensor(self._counter):
                self._counter.zero_()
                self._active.zero_()
                self.success_counts.zero_()
            else:  # pragma: no cover
                self._counter = [0] * self.num_envs
                self._active = [False] * self.num_envs
                self.success_counts = [0] * self.num_envs
            self.total_successes = 0
            return
        idx = env_ids
        if _HAS_TORCH and _is_tensor(self._counter):
            idx = idx.to(device=self._counter.device, dtype=torch.long) if _is_tensor(idx) else idx
            self._counter[idx] = 0
            self._active[idx] = False
            self.success_counts[idx] = 0
        else:  # pragma: no cover
            for i in idx:
                self._counter[i] = 0
                self._active[i] = False
                self.success_counts[i] = 0

    def update(self, within_tolerance: Any) -> Any:
        """Update hold counters and return the mask of newly achieved successes."""
        wt = self._as_bool_mask(within_tolerance)
        if _HAS_TORCH and _is_tensor(self._counter):
            wt = wt.to(device=self._counter.device)
            self._counter = torch.where(wt, self._counter + 1, torch.zeros_like(self._counter))
            newly = wt & (self._counter >= self.hold_steps) & (~self._active)
            if bool(newly.any()):
                self.success_counts = self.success_counts + newly.to(self.success_counts.dtype)
                self.total_successes += int(newly.sum().item())
            self._active = self._active | newly
            # once out of tolerance the environment may succeed again later
            self._active = self._active & wt
            return newly
        # pragma: no cover - torch-free fallback
        out = []
        for i, ok in enumerate(wt):
            self._counter[i] = self._counter[i] + 1 if ok else 0
            newly = bool(ok and self._counter[i] >= self.hold_steps and not self._active[i])
            if newly:
                self.success_counts[i] += 1
                self.total_successes += 1
            self._active[i] = bool(ok and (newly or self._active[i]))
            out.append(newly)
        return out

    def finalise_episode(self, dones: Any) -> Any:
        """Return per-env success counts for episodes that just terminated."""
        done_mask = self._as_bool_mask(dones)
        counts = self.success_counts
        if _HAS_TORCH and _is_tensor(counts):
            done = done_mask.to(device=counts.device)
            out = counts.clone()
            self.success_counts = torch.where(done, torch.zeros_like(counts), counts)
            self._counter = torch.where(done, torch.zeros_like(self._counter), self._counter)
            self._active = self._active & (~done)
            return out
        return counts  # pragma: no cover


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------


def split_observation(obs: Any, task: str = ALLEGRO_HAND) -> Dict[str, Any]:
    """Split the flat Allegro Hand observation into its named components."""
    joints = ALLEGRO_HAND_JOINTS
    result: Dict[str, Any] = {}
    if _is_tensor(obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        result["q"] = obs[..., :joints]
        result["qdot"] = obs[..., joints : 2 * joints]
        result["object_pose"] = obs[..., 2 * joints : 2 * joints + OBJECT_POSE_DIM]
        result["object_pos"] = obs[..., 2 * joints : 2 * joints + 3]
        result["object_quat"] = obs[..., 2 * joints + 3 : 2 * joints + OBJECT_POSE_DIM]
        base = 2 * joints + OBJECT_POSE_DIM
        result["object_lin_vel"] = obs[..., base : base + OBJECT_VEL_DIM]
        result["object_ang_vel"] = obs[..., base + 3 : base + 2 * OBJECT_VEL_DIM]
        base = base + 2 * OBJECT_VEL_DIM
        result["goal"] = obs[..., base : base + GOAL_QUAT_DIM]
        result["goal_quat"] = result["goal"]
        result["aux"] = obs[..., base + GOAL_QUAT_DIM :]
        result["lifted"] = result["object_pos"][..., 2] > 0.0
        return result

    # torch-free fallback -------------------------------------------------
    row = obs
    if isinstance(row, (list, tuple)) and row and isinstance(row[0], (list, tuple)):
        row = row[0]  # pragma: no cover
    vals = [float(v) for v in row]
    result["q"] = vals[:joints]
    result["qdot"] = vals[joints : 2 * joints]
    result["object_pose"] = vals[2 * joints : 2 * joints + OBJECT_POSE_DIM]
    result["object_pos"] = vals[2 * joints : 2 * joints + 3]
    result["object_quat"] = vals[2 * joints + 3 : 2 * joints + OBJECT_POSE_DIM]
    base = 2 * joints + OBJECT_POSE_DIM
    result["object_lin_vel"] = vals[base : base + OBJECT_VEL_DIM]
    result["object_ang_vel"] = vals[base + 3 : base + 2 * OBJECT_VEL_DIM]
    base = base + 2 * OBJECT_VEL_DIM
    result["goal"] = vals[base : base + GOAL_QUAT_DIM]
    result["goal_quat"] = result["goal"]
    result["aux"] = vals[base + GOAL_QUAT_DIM :]
    result["lifted"] = result["object_pos"][2] > 0.0
    return result


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------


class AllegroHandEnv:
    """Task wrapper for the 16-DoF Allegro Hand in-hand reorientation task."""

    task = ALLEGRO_HAND
    task_group = "allegro_hand"
    is_allegro_hand = True
    is_easy_task = True

    def __init__(
        self,
        task: str = ALLEGRO_HAND,
        num_envs: Optional[int] = None,
        env: Optional[Any] = None,
        config: Optional[Any] = None,
        device: Optional[Any] = None,
        hold_steps: int = HOLD_STEPS_DEFAULT,
        rot_tolerance: float = ROT_TOLERANCE_DEFAULT,
        reward_weights: Optional[Any] = None,
        use_curriculum: bool = False,
        max_episode_length: int = MAX_EPISODE_LENGTH_DEFAULT,
        seed: Optional[int] = None,
        **env_kwargs: Any,
    ):
        self.task = ALLEGRO_HAND
        self._config = config
        self._env = env
        self._env_kwargs = dict(env_kwargs)
        self._device = device if device is not None else getattr(config, "device", None)

        if num_envs is None:
            num_envs = getattr(config, "num_envs", None)
        if num_envs is None:
            num_envs = getattr(config, "num_envs", 1) or 1  # pragma: no cover
        self._num_envs = int(num_envs)

        if seed is None:
            seed = getattr(config, "seed", None)
        self._seed = seed

        self._hold_steps = int(hold_steps) if hold_steps else HOLD_STEPS_DEFAULT
        self._rot_tolerance = float(rot_tolerance)
        self._max_episode_length = int(max_episode_length)
        self.use_curriculum = bool(use_curriculum)

        self._weights = AllegroHandRewardWeights.from_any(reward_weights or config, task=self.task)

        self._tracker: Optional[SuccessTracker] = None
        self._episode_lengths = _zeros(self._num_envs, device=self._device, dtype=torch.long if _HAS_TORCH else None)
        self._episode_returns = _zeros(self._num_envs, device=self._device)
        self._last_obs: Any = None
        self._goal_generator = None

    # -- lazy environment construction ------------------------------------
    @property
    def env(self) -> Any:
        if self._env is None:
            overrides = dict(self._env_kwargs)
            overrides.setdefault("task", ALLEGRO_HAND)
            overrides.setdefault("num_envs", self._num_envs)
            self._env = make_isaac_env(task=ALLEGRO_HAND, config=self._config, **overrides)
            self._num_envs = int(getattr(self._env, "num_envs", self._num_envs) or self._num_envs)
        return self._env

    @property
    def num_envs(self) -> int:
        return int(self._num_envs)

    @property
    def obs_dim(self) -> int:
        return int(obs_dim_for(ALLEGRO_HAND))

    @property
    def action_dim(self) -> int:
        return int(action_dim_for(ALLEGRO_HAND))

    @property
    def goal_dim(self) -> int:
        return GOAL_QUAT_DIM

    @property
    def joint_dim(self) -> int:
        return ALLEGRO_HAND_JOINTS

    @property
    def success_tolerance(self) -> float:
        return float(self._rot_tolerance)

    @property
    def goal(self) -> Any:
        goal = getattr(self.env, "goal", None)
        if goal is not None:
            return goal
        return getattr(self.env, "_goal", None)  # pragma: no cover

    # -- success tracking --------------------------------------------------
    @property
    def tracker(self) -> SuccessTracker:
        if self._tracker is None:
            self._tracker = SuccessTracker(
                self._num_envs, hold_steps=self._hold_steps, device=self._device
            )
        return self._tracker

    def set_success_tolerance(self, tolerance: float) -> float:
        """Set the rotation tolerance used for the success criterion."""
        self._rot_tolerance = float(tolerance)
        if hasattr(self.env, "set_success_tolerance"):
            try:
                self.env.set_success_tolerance(self._rot_tolerance)
            except Exception:  # pragma: no cover
                pass
        return self._rot_tolerance

    set_rot_tolerance = set_success_tolerance

    def set_rot_tolerance(self, tolerance: float) -> float:  # noqa: F811 - explicit alias
        self._rot_tolerance = float(tolerance)
        if hasattr(self.env, "set_success_tolerance"):
            try:
                self.env.set_success_tolerance(self._rot_tolerance)
            except Exception:  # pragma: no cover
                pass
        return self._rot_tolerance

    # -- core API ----------------------------------------------------------
    def reset(self, env_ids: Any = None) -> Any:
        obs = self.env.reset(env_ids=env_ids) if env_ids is not None else self.env.reset()
        self._last_obs = obs
        if env_ids is None:
            if self._tracker is not None:
                self._tracker.reset()
            self._episode_lengths = _zeros(self._num_envs, device=self._device, dtype=torch.long if _HAS_TORCH else None)
            self._episode_returns = _zeros(self._num_envs, device=self._device)
        else:
            if self._tracker is not None:
                self._tracker.reset(env_ids)
            self._reset_episode_stats(env_ids)
        return obs

    def _reset_episode_stats(self, env_ids: Any) -> None:
        if _HAS_TORCH and _is_tensor(self._episode_lengths):
            self._episode_lengths[env_ids] = 0
            self._episode_returns[env_ids] = 0.0
        else:  # pragma: no cover
            for i in env_ids:
                self._episode_lengths[i] = 0
                self._episode_returns[i] = 0.0

    def step(self, actions: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """Step the underlying simulator and add task-level info."""
        obs, base_reward, done, info = self.env.step(actions)
        info = dict(info or {})
        self._last_obs = obs

        progress = self.measure_progress(obs)
        success_mask = self.tracker.update(progress["within_tolerance"])
        reward = self.compute_reward(progress, succeeded=success_mask, base_reward=base_reward)

        done = _to_tensor(done) if _is_tensor(done) or isinstance(done, (list, tuple)) else done
        # episode bookkeeping ------------------------------------------------
        counts = self.tracker.finalise_episode(done)
        if _HAS_TORCH and _is_tensor(self._episode_lengths):
            self._episode_lengths = self._episode_lengths + 1
            self._episode_returns = self._episode_returns + (
                reward if _is_tensor(reward) else _to_tensor(reward, device=self._episode_returns.device)
            )
            info["episode_stats_return"] = self._episode_returns.clone()
            info["episode_stats_length"] = (self._episode_lengths * 1.0).clone()
            done_mask = _to_tensor(done).to(dtype=torch.bool) if _is_tensor(done) else None
            if done_mask is not None and bool(done_mask.any()):
                info["episode_return"] = self._episode_returns[done_mask].mean()
                info["episode_length"] = self._episode_lengths[done_mask].float().mean()
                self._episode_returns = torch.where(
                    done_mask, torch.zeros_like(self._episode_returns), self._episode_returns
                )
                self._episode_lengths = torch.where(
                    done_mask, torch.zeros_like(self._episode_lengths), self._episode_lengths
                )
            success_count = float(counts.sum().item()) if _is_tensor(counts) else 0.0
            successes = float(success_mask.sum().item()) if _is_tensor(success_mask) else float(sum(success_mask))
        else:  # pragma: no cover - torch-free fallback
            success_count = float(sum(counts))
            successes = float(sum(success_mask))

        info.setdefault("successes", successes)
        info.setdefault("success_count", success_count)
        info.setdefault("within_tolerance", progress["within_tolerance"])
        info.setdefault("orientation_error", progress["orientation_error"])
        info.setdefault("success_tolerance", self._rot_tolerance)
        info.setdefault("task", self.task)
        info.setdefault("reward_orientation", progress["orientation_reward"])

        if self._max_episode_length > 0 and info.get("episode_length") is None:
            info["episode_length"] = float(self._max_episode_length)

        return obs, reward, done, info

    def measure_progress(self, obs: Any) -> Dict[str, Any]:
        """Compute the orientation error and the within-tolerance mask."""
        if obs is None:
            obs = self._last_obs
        parts = split_observation(obs, task=self.task)
        obj_quat = parts["object_quat"]
        goal = parts["goal_quat"]
        angle = _quat_angle_to(obj_quat, goal)
        if _is_tensor(angle):
            within = angle <= self._rot_tolerance
        else:  # pragma: no cover
            within = float(angle) <= self._rot_tolerance
        # orientation reward: 1 when perfectly aligned, 0 at a pi rotation
        if _is_tensor(angle):
            orientation_reward = 1.0 - angle / math.pi
        else:  # pragma: no cover
            orientation_reward = 1.0 - float(angle) / math.pi
        return {
            "orientation_error": angle,
            "orientation_reward": orientation_reward,
            "within_tolerance": within,
            "distance": angle,
            "goal_quat": goal,
        }

    def compute_reward(
        self,
        progress: Dict[str, Any],
        succeeded: Any = None,
        base_reward: Any = None,
    ) -> Any:
        """Paper reward: orientation error shaping + success bonus.

        The underlying simulator's own shaping terms are intentionally not double
        counted because ``isaac_env``'s surrogate already includes the same
        orientation-error term; only the explicit success bonus is added here.
        """
        orient = progress["orientation_reward"]
        w = self._weights
        if _is_tensor(orient):
            reward = w.orientation * orient
            if succeeded is not None:
                succ = succeeded.to(dtype=orient.dtype) if _is_tensor(succeeded) else _to_tensor(
                    succeeded, device=orient.device, dtype=orient.dtype
                )
                reward = reward + w.success * succ
            return reward
        # torch-free fallback --------------------------------------------------
        reward = w.orientation * float(orient)  # pragma: no cover
        if succeeded is not None:  # pragma: no cover
            reward += w.success * float(sum(succeeded) if isinstance(succeeded, (list, tuple)) else succeeded)
        return reward  # pragma: no cover

    # -- utilities ---------------------------------------------------------
    def sample_actions(self) -> Any:
        if hasattr(self.env, "sample_actions"):
            return self.env.sample_actions()
        if _HAS_TORCH:  # pragma: no cover
            return torch.zeros(self._num_envs, self.action_dim, device=self._device)
        return [[0.0] * self.action_dim for _ in range(self._num_envs)]  # pragma: no cover

    def info_snapshot(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {
            "task": self.task,
            "num_envs": self._num_envs,
            "rot_tolerance": self._rot_tolerance,
            "hold_steps": self._hold_steps,
            "reward_weights": self._weights.as_dict(),
        }
        if self._tracker is not None:
            snap["total_successes"] = self._tracker.total_successes
        if hasattr(self.env, "info_snapshot"):
            try:
                snap["env"] = self.env.info_snapshot()
            except Exception:  # pragma: no cover
                pass
        return snap

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:  # pragma: no cover
                pass

    def __len__(self) -> int:
        return self._num_envs

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AllegroHandEnv(task={self.task!r}, num_envs={self._num_envs}, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"rot_tolerance={self._rot_tolerance})"
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_allegro_hand_env(
    task: str = ALLEGRO_HAND,
    config: Optional[Any] = None,
    num_envs: Optional[int] = None,
    env: Optional[Any] = None,
    **kwargs: Any,
) -> AllegroHandEnv:
    """Create the Allegro Hand reorientation environment wrapper."""
    device = kwargs.pop("device", None)
    if device is None:
        device = getattr(config, "device", None)
    return AllegroHandEnv(
        task=ALLEGRO_HAND,
        num_envs=num_envs,
        env=env,
        config=config,
        device=device,
        **kwargs,
    )


# Aliases used by different entry points / configs.
make_in_hand_reorientation_env = make_allegro_hand_env
make_allegro_hand_reorientation_env = make_allegro_hand_env
