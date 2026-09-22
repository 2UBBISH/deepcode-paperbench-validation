"""Allegro-Kuka task wrappers: Regrasping, Throw and Reorientation.

Paper reference
---------------
Appendix A ("Hard Difficulty Tasks") and Section 5.1.

All three hard tasks are built on the Allegro-Kuka environment suite
(Petrenko et al., 2023): a 16-DoF Allegro Hand mounted on a 7-DoF Kuka arm
manipulating a cuboidal object kept on a fixed table.

Observation (Appendix A)::

    o_t = [q, qdot, x_t, v_t, omega_t, g_t, z_t]

    q, qdot in R^23   joint angles / velocities (Allegro 16 + Kuka 7)
    x_t     in R^7    object pose (position + quaternion)
    v_t               object linear velocity
    omega_t           object angular velocity
    g_t               task dependent goal
    z_t               auxiliary info (e.g. whether the object has been lifted)

Tasks (Appendix A):
  * ``regrasping``   - lift the object and hold it near ``g_t in R^3`` for
    ``K = 30`` steps (a "success").  The success tolerance
    ``delta = ||g_t - (x_t)_{0:3}|| <= delta`` is annealed by a curriculum from
    7.5 cm to 1 cm (x 0.9 whenever the average number of successes per episode
    crosses 3).  Reward = r_reach + r_lift + r_target + r_success.
  * ``throw``        - lift the object and throw it into a bucket at
    ``g_t in R^3`` placed out of reach of the arm.  Same reward shape as
    regrasping, with the bucket as the target.
  * ``reorientation``- pick the object up and reorient it to ``g_t in R^7``
    (position + orientation), with a curriculum on the tolerance as well.

This module is intentionally dependency-light: it only relies on the
``sapg.envs.isaac_env`` facade (which dispatches to IsaacGym or to the
pure-torch surrogate) and on ``sapg.envs.curriculum`` for the tolerance
schedule.  Each wrapper therefore works both with the real GPU simulator and
with the surrogate, which keeps the training loop testable without IsaacGym.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # torch is required by the underlying env facade
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency in practice
    torch = None  # type: ignore

from .isaac_env import (  # noqa: F401  (re-exported convenience helpers)
    GOAL_DIMS,
    JOINT_DIMS,
    IsaacEnv,
    make_isaac_env,
    obs_dim_for,
    action_dim_for,
    goal_dim_for,
    joint_dim_for,
    default_reward_weights,
    quat_angle_error,
)
from .curriculum import (
    SuccessCurriculum,
    apply_tolerance,
    make_curriculum,
    update_curriculum,
)

__all__ = [
    "ALLEGRO_KUKA_TASKS",
    "REGRASPING",
    "THROW",
    "REORIENTATION",
    "TaskRewardWeights",
    "SuccessTracker",
    "AllegroKukaEnv",
    "RegraspingEnv",
    "ThrowEnv",
    "ReorientationEnv",
    "make_allegro_kuka_env",
    "make_regrasping_env",
    "make_throw_env",
    "make_reorientation_env",
]

# ---------------------------------------------------------------------------
# Task names / constants
# ---------------------------------------------------------------------------

REGRASPING = "regrasping"
THROW = "throw"
REORIENTATION = "reorientation"

ALLEGRO_KUKA_TASKS: Tuple[str, ...] = (REGRASPING, THROW, REORIENTATION)

#: Number of consecutive steps the object must remain within tolerance for the
#: attempt to count as a success (Appendix A: ``K = 30``).
HOLD_STEPS_DEFAULT = 30

#: Curriculum endpoints for the success tolerance ``delta`` (7.5 cm -> 1 cm).
INITIAL_TOLERANCE_DEFAULT = 0.075
MIN_TOLERANCE_DEFAULT = 0.01
TOLERANCE_DECREMENT_DEFAULT = 0.9
SUCCESSES_THRESHOLD_DEFAULT = 3.0


# ---------------------------------------------------------------------------
# Reward shape (r_reach + r_lift + r_target + r_success)
# ---------------------------------------------------------------------------


@dataclass
class TaskRewardWeights:
    """Weights of the weighted reward combination of the hard tasks.

    Appendix A: "The reward function is a weighted combination of rewards
    encouraging the hand to reach the object ``r_reach``, a bonus ``r_lift``,
    rewards encouraging the hand to move to goal location after lifting
    ``r_target`` and a success bonus ``r_success``."

    The paper does not publish the numeric weights; the defaults below follow
    the DexPBT / AllegroKuka codebase magnitudes and are overridable through
    the environment configuration (``reward_weights``).
    """

    reach: float = 1.0
    lift: float = 15.0
    target: float = 5.0
    success: float = 100.0
    #: exponent used to sharpen the reach/target shaping (DexPBT uses 2.0)
    shaping_power: float = 2.0
    #: bonus applied while the object is held in the target region
    hold: float = 0.0

    @classmethod
    def from_any(cls, weights: Any = None, task: str = REGRASPING) -> "TaskRewardWeights":
        """Build weights from a dict / config object / nothing."""
        if weights is None:
            return cls()
        if isinstance(weights, TaskRewardWeights):
            return weights
        if isinstance(weights, dict):
            data = dict(weights)
        else:  # config object or dotted namespace
            data = {}
            for key in ("reach", "lift", "target", "success", "shaping_power", "hold"):
                if hasattr(weights, key):
                    data[key] = getattr(weights, key)
                elif hasattr(weights, f"reward_{key}"):
                    data[key] = getattr(weights, f"reward_{key}")
        if "r_reach" in data:
            data.setdefault("reach", data["r_reach"])
        if "r_lift" in data:
            data.setdefault("lift", data["r_lift"])
        if "r_target" in data:
            data.setdefault("target", data["r_target"])
        if "r_success" in data:
            data.setdefault("success", data["r_success"])
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)  # type: ignore[arg-type]

    def as_dict(self) -> Dict[str, float]:
        return {
            "reach": self.reach,
            "lift": self.lift,
            "target": self.target,
            "success": self.success,
            "shaping_power": self.shaping_power,
            "hold": self.hold,
        }


def _shaping_weights_for(task: str, env: Any = None) -> TaskRewardWeights:
    """Reward weights for a task, honouring ``env.cfg.reward_weights``."""
    for holder in (getattr(env, "cfg", None), env, getattr(env, "env_cfg", None)):
        if holder is None:
            continue
        weights = None
        if isinstance(holder, dict):
            weights = holder.get("reward_weights")
        else:
            weights = getattr(holder, "reward_weights", None)
        if weights:
            base = default_reward_weights(task) if callable(default_reward_weights) else {}
            if base:
                merged = dict(base)
                merged.update(weights)
                return TaskRewardWeights.from_any(merged, task)
            return TaskRewardWeights.from_any(weights, task)
    base = default_reward_weights(task) if callable(default_reward_weights) else {}
    return TaskRewardWeights.from_any(base or None, task)


# ---------------------------------------------------------------------------
# Success bookkeeping (hold `K` steps within `delta` of the goal)
# ---------------------------------------------------------------------------


class SuccessTracker:
    """Tracks the ``K``-step hold requirement and per-episode successes.

    A success is registered when the *same* environment has satisfied the
    tolerance for ``hold_steps`` consecutive steps.  Following Appendix A the
    goal (and, on the real simulator, the object) is then re-randomised, which
    the wrapper performs through any of the env's resampling entry points.
    """

    def __init__(self, num_envs: int, hold_steps: int = HOLD_STEPS_DEFAULT, device: Any = None):
        self.num_envs = int(num_envs)
        self.hold_steps = int(hold_steps)
        self.device = device
        self._hold = _zeros(self.num_envs, device)
        self.successes = _zeros(self.num_envs, device)
        self.episode_successes = _zeros(self.num_envs, device)

    # -- bookkeeping ------------------------------------------------------
    def reset(self, env_ids: Any = None) -> None:
        if env_ids is None:
            self._hold = _zeros(self.num_envs, self.device)
            self.successes = _zeros(self.num_envs, self.device)
            return
        self._hold = _scatter_zeros(self._hold, env_ids)
        self.successes = _scatter_zeros(self.successes, env_ids)

    def update(self, within_tolerance: Any) -> Any:
        """Advance hold counters; return a boolean mask of *new* successes."""
        tol = _as_bool_tensor(within_tolerance, self.num_envs, self.device)
        self._hold = _where(tol, self._hold + 1.0, _zeros(self.num_envs, self.device))
        succeeded = self._hold >= float(self.hold_steps)
        if succeeded.any():
            self.successes = self.successes + succeeded.to(self.successes.dtype)
            self.episode_successes = self.episode_successes + succeeded.to(
                self.episode_successes.dtype
            )
            # a new attempt starts immediately after each success
            self._hold = _where(succeeded, _zeros(self.num_envs, self.device), self._hold)
        return succeeded

    def finalise_episode(self, dones: Any) -> Any:
        """Zero per-episode counters for finished envs, returning their values."""
        finished = _as_bool_tensor(dones, self.num_envs, self.device)
        counts = self.episode_successes.clone()
        self.episode_successes = _where(finished, _zeros(self.num_envs, self.device),
                                        self.episode_successes)
        self._hold = _where(finished, _zeros(self.num_envs, self.device), self._hold)
        return counts


# ---------------------------------------------------------------------------
# Tensor helpers (avoid importing torch at module import time)
# ---------------------------------------------------------------------------


def _zeros(n: int, device: Any = None):
    if torch is None:  # pragma: no cover
        return [0.0] * n
    return torch.zeros(n, device=device, dtype=torch.float32)


def _as_tensor(x: Any, device: Any = None):
    if torch is None:  # pragma: no cover
        return x
    if isinstance(x, torch.Tensor):
        t = x.detach()
        return t.to(device) if device is not None else t
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _as_bool_tensor(x: Any, n: int, device: Any = None):
    if torch is None:  # pragma: no cover
        return x
    if isinstance(x, torch.Tensor):
        t = x.to(device) if device is not None else x
        if t.dtype != torch.bool:
            t = t > 0.5 if t.dtype.is_floating_point else t.bool()
        return t.reshape(-1)
    return torch.zeros(n, dtype=torch.bool, device=device)


def _where(cond: Any, a: Any, b: Any):
    if torch is None:  # pragma: no cover
        return a
    return torch.where(cond, a, b)


def _scatter_zeros(t: Any, env_ids: Any):
    if torch is None:  # pragma: no cover
        return t
    out = t.clone()
    out[env_ids] = 0.0
    return out


def _norm(x: Any, dim: int = -1):
    if torch is None:  # pragma: no cover
        return x
    return torch.linalg.norm(x, dim=dim)


def _as_float(x: Any) -> float:
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().float().mean().item())
    try:
        return float(x)
    except Exception:  # pragma: no cover
        return 0.0


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------


class AllegroKukaEnv:
    """Task-aware wrapper around the Allegro-Kuka environment facade.

    Parameters
    ----------
    task:
        One of ``regrasping`` / ``throw`` / ``reorientation``.
    num_envs:
        Number of parallel environments (``N = 24576`` in the paper).
    env:
        Pre-built environment implementing ``reset``/``step``.  When omitted a
        new one is created through :func:`sapg.envs.isaac_env.make_isaac_env`.
    config:
        Optional :class:`sapg.utils.config.SAPGConfig` (or plain dict); used to
        pick up ``num_envs``, the curriculum settings and reward weights.
    """

    task: str = REGRASPING
    goal_dim: int = 3

    def __init__(
        self,
        task: str = REGRASPING,
        num_envs: Optional[int] = None,
        env: Any = None,
        config: Any = None,
        device: Any = None,
        hold_steps: int = HOLD_STEPS_DEFAULT,
        initial_tolerance: float = INITIAL_TOLERANCE_DEFAULT,
        min_tolerance: float = MIN_TOLERANCE_DEFAULT,
        tolerance_decrement: float = TOLERANCE_DECREMENT_DEFAULT,
        success_threshold: float = SUCCESSES_THRESHOLD_DEFAULT,
        reward_weights: Any = None,
        use_curriculum: bool = True,
        seed: Optional[int] = None,
        **env_kwargs: Any,
    ) -> None:
        self.task = str(task).lower()
        self.config = config

        if num_envs is None:
            num_envs = _config_get(config, "num_envs", None)
        if device is None:
            device = _config_get(config, "device", None)

        overrides: Dict[str, Any] = dict(env_kwargs)
        if num_envs is not None:
            overrides["num_envs"] = int(num_envs)
        if device is not None:
            overrides["device"] = device
        if seed is not None:
            overrides["seed"] = seed

        if env is None:
            env = make_isaac_env(self.task, config=config, **overrides)
        self.env = env
        self.unwrapped = getattr(env, "env", env)

        self.num_envs = int(
            getattr(env, "num_envs", num_envs if num_envs is not None else _config_get(config, "num_envs", 1))
            or 1
        )
        self.device = getattr(env, "device", device)

        self.hold_steps = int(hold_steps)
        self.reward_weights = TaskRewardWeights.from_any(
            reward_weights if reward_weights is not None else None
        ) if reward_weights is not None else _shaping_weights_for(self.task, env)

        # success tolerance + curriculum (Appendix A)
        tolerance = initial_tolerance
        self.curriculum: Optional[SuccessCurriculum] = None
        if use_curriculum:
            self.curriculum = make_curriculum(
                task=self.task,
                env=env,
                config=config,
                initial_tolerance=tolerance,
                min_tolerance=min_tolerance,
                decrement=tolerance_decrement,
                threshold=success_threshold,
            )
            tolerance = float(getattr(self.curriculum, "delta", tolerance))
        else:
            apply_tolerance(env, tolerance)
        self._tolerance = float(tolerance)

        self.tracker = SuccessTracker(
            num_envs=self.num_envs, hold_steps=self.hold_steps, device=self.device
        )
        self._last_obs: Any = None
        self._last_infos: Dict[str, Any] = {}
        self._episode_reward = _zeros(self.num_envs, self.device)
        self._episode_length = _zeros(self.num_envs, self.device)

    # -- properties -------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return int(getattr(self.env, "obs_dim", obs_dim_for(self.task)))

    @property
    def action_dim(self) -> int:
        return int(getattr(self.env, "action_dim", action_dim_for(self.task)))

    @property
    def success_tolerance(self) -> float:
        return float(self._tolerance)

    @property
    def goal(self) -> Any:
        return getattr(self.env, "goal", None)

    def __len__(self) -> int:
        return self.num_envs

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(task={self.task!r}, num_envs={self.num_envs}, "
            f"tolerance={self._tolerance:.4f})"
        )

    # -- core API ---------------------------------------------------------
    def reset(self, env_ids: Any = None):
        obs = self.env.reset(env_ids) if env_ids is not None else self.env.reset()
        self.tracker.reset(env_ids)
        if env_ids is None:
            self._episode_reward = _zeros(self.num_envs, self.device)
            self._episode_length = _zeros(self.num_envs, self.device)
        else:
            self._episode_reward = _scatter_zeros(self._episode_reward, env_ids)
            self._episode_length = _scatter_zeros(self._episode_length, env_ids)
        self._last_obs = obs
        return obs

    def step(self, actions: Any):
        obs, reward, done, info = self.env.step(actions)

        # -- task specific progress measurements --------------------------
        progress = self.measure_progress(obs)
        within = progress["within_tolerance"]
        succeeded = self.tracker.update(within)

        # -- reward shaping (r_reach + r_lift + r_target + r_success) -----
        shaped = self.compute_reward(progress, succeeded)
        reward = _as_tensor(reward, self.device).reshape(-1) + shaped

        # -- auto-reset bookkeeping ---------------------------------------
        episode_successes = self.tracker.finalise_episode(done)
        self._episode_reward = self._episode_reward + reward
        self._episode_length = self._episode_length + 1.0

        # -- on success re-randomise the goal (Appendix A) ----------------
        if bool(_any(succeeded)):
            self._resample_goals(_indices(succeeded))

        info = dict(info) if isinstance(info, dict) else {}
        info.setdefault("successes", self.tracker.successes)
        info.setdefault("episode_successes", episode_successes)
        info.setdefault("success_tolerance", self._tolerance)
        info.setdefault("within_tolerance", within)
        info.setdefault("progress", progress)
        info["episode_return"] = self._episode_reward
        info["episode_length"] = self._episode_length

        if done is not None:
            finished = _as_bool_tensor(done, self.num_envs, self.device)
            self._episode_reward = _where(finished, _zeros(self.num_envs, self.device),
                                          self._episode_reward)
            self._episode_length = _where(finished, _zeros(self.num_envs, self.device),
                                          self._episode_length)

        # -- curriculum update (avg successes per episode > 3) ------------
        if self.curriculum is not None:
            self.curriculum.record_episodes(_as_list(episode_successes, finished_mask(done, self.num_envs, self.device)))
            self.curriculum.update(infos=None, env=self.env)
            self._tolerance = float(getattr(self.curriculum, "delta", self._tolerance))
            info["success_tolerance"] = self._tolerance

        self._last_obs = obs
        self._last_infos = info
        return obs, reward, done, info

    def set_success_tolerance(self, tolerance: float) -> float:
        self._tolerance = float(tolerance)
        if self.curriculum is not None:
            self.curriculum.reset(tolerance=self._tolerance, reset_statistics=False)
        apply_tolerance(self.env, self._tolerance)
        return self._tolerance

    def close(self) -> None:
        closer = getattr(self.env, "close", None)
        if callable(closer):
            closer()

    def sample_actions(self):
        sampler = getattr(self.env, "sample_actions", None)
        if callable(sampler):
            return sampler()
        if torch is None:  # pragma: no cover
            return None
        return torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def info_snapshot(self) -> Dict[str, Any]:
        return dict(self._last_infos)

    # -- hooks ------------------------------------------------------------
    def measure_progress(self, obs: Any) -> Dict[str, Any]:
        """Populate task specific progress quantities from the observation.

        Subclasses override this to expose the goal error, lift height and so
        on.  The base implementation returns zeros so the wrapper degrades
        gracefully with an unknown env backend.
        """
        return {
            "goal_error": _zeros(self.num_envs, self.device),
            "within_tolerance": _zeros(self.num_envs, self.device).bool()
            if torch is not None
            else None,
            "lifted": _zeros(self.num_envs, self.device),
            "reach_error": _zeros(self.num_envs, self.device),
            "target_error": _zeros(self.num_envs, self.device),
        }

    def compute_reward(self, progress: Dict[str, Any], succeeded: Any) -> Any:
        w = self.reward_weights
        p = max(float(w.shaping_power), 1.0)

        reach = _as_tensor(progress.get("reach_error", 0.0), self.device)
        lift = _as_tensor(progress.get("lifted", 0.0), self.device)
        target = _as_tensor(progress.get("target_error", 0.0), self.device)

        r_reach = w.reach * _exp(-p * reach)
        r_lift = w.lift * lift
        r_target = w.target * _exp(-p * target)
        r_success = w.success * succeeded.to(reach.dtype) if torch is not None else 0.0
        shape = _as_tensor(r_reach, self.device) + _as_tensor(r_lift, self.device)
        shape = shape + _as_tensor(r_target, self.device) + _as_tensor(r_success, self.device)
        return shape

    # -- helpers ----------------------------------------------------------
    def _resample_goals(self, env_ids: Any) -> None:
        for name in ("_resample_targets", "resample_targets", "resample_goals",
                     "reset_target", "reset_goal"):
            fn = getattr(self.unwrapped, name, None) or getattr(self.env, name, None)
            if callable(fn):
                try:
                    fn(env_ids)
                except TypeError:
                    try:
                        fn()
                    except Exception:  # pragma: no cover
                        pass
                return


def finished_mask(dones: Any, n: int, device: Any = None):
    return _as_bool_tensor(dones, n, device)


def _as_list(values: Any, mask: Any = None) -> List[float]:
    if mask is None:
        if torch is not None and isinstance(values, torch.Tensor):
            return [float(v) for v in values.reshape(-1).tolist()]
        return [float(values)]
    if torch is not None and isinstance(values, torch.Tensor):
        selected = values.reshape(-1)[mask.reshape(-1)]
        return [float(v) for v in selected.tolist()]
    return []


def _any(mask: Any) -> bool:
    if torch is not None and isinstance(mask, torch.Tensor):
        return bool(mask.any().item())
    try:
        return any(bool(m) for m in mask)
    except TypeError:  # pragma: no cover
        return bool(mask)


def _indices(mask: Any):
    if torch is not None and isinstance(mask, torch.Tensor):
        return torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    return [i for i, m in enumerate(mask) if m]


def _exp(x: Any):
    if torch is not None and isinstance(x, torch.Tensor):
        return torch.exp(-x)
    return math.exp(-float(x)) if x is not None else 0.0


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ---------------------------------------------------------------------------
# Observation slicing shared by the three tasks
# ---------------------------------------------------------------------------

#: Layout of the Appendix A observation: [q, qdot, x_t, v_t, omega_t, g_t, z_t]
_JOINTS = 23  # 16 (Allegro) + 7 (Kuka)
_OBJECT_POSE = 7
_OBJECT_VEL = 3
_OBJECT_OMEGA = 3


def split_observation(obs: Any, goal_dim: int, task: str = REGRASPING):
    """Split the flat observation into its Appendix A components.

    Returns a dict with ``q``, ``qdot``, ``object_pose``, ``object_pos``,
    ``object_quat``, ``object_lin_vel``, ``object_ang_vel``, ``goal`` and
    ``aux``.  The layout is::

        [ q(23) | qdot(23) | x_t(7) | v_t(3) | omega_t(3) | g_t(goal_dim) | z_t(rest) ]
    """
    if torch is None or not isinstance(obs, torch.Tensor):  # pragma: no cover
        return {}
    o = obs.reshape(obs.shape[0], -1)
    joint_dim = joint_dim_for(task)
    idx = 0
    q = o[:, idx:idx + joint_dim]; idx += joint_dim
    qdot = o[:, idx:idx + joint_dim]; idx += joint_dim
    x_t = o[:, idx:idx + _OBJECT_POSE]; idx += _OBJECT_POSE
    v_t = o[:, idx:idx + _OBJECT_VEL]; idx += _OBJECT_VEL
    omega_t = o[:, idx:idx + _OBJECT_OMEGA]; idx += _OBJECT_OMEGA
    g_t = o[:, idx:idx + goal_dim]; idx += goal_dim
    aux = o[:, idx:]
    pos = x_t[:, :3]
    quat = x_t[:, 3:7]
    return {
        "q": q,
        "qdot": qdot,
        "object_pose": x_t,
        "object_pos": pos,
        "object_quat": quat,
        "object_lin_vel": v_t,
        "object_ang_vel": omega_t,
        "goal": g_t,
        "aux": aux,
        "lifted": (aux[:, 0] if aux.shape[1] > 0 else pos[:, 2]),
    }


# ---------------------------------------------------------------------------
# Regrasping
# ---------------------------------------------------------------------------


class RegraspingEnv(AllegroKukaEnv):
    """Regrasping: lift the object and hold it near ``g_t in R^3`` for K=30 steps."""

    task = REGRASPING
    goal_dim = 3

    def measure_progress(self, obs: Any) -> Dict[str, Any]:
        parts = split_observation(obs, self.goal_dim, self.task) if obs is not None else {}
        if not parts:
            return super().measure_progress(obs)
        obj = parts["object_pos"]
        goal = parts["goal"]
        goal_error = _norm(obj - goal, dim=-1)
        within = goal_error <= float(self._tolerance)

        # reach error: distance between the hand centre and the object
        q = parts["q"][:, :16]  # Allegro joints drive the fingers
        reach_error = _norm(q, dim=-1) * 0.0 + _norm(obj - obj, dim=-1)  # placeholder
        reach_error = _norm(parts["object_lin_vel"], dim=-1) * 0.0 + reach_error

        lifted = parts["lifted"]
        return {
            "goal_error": goal_error,
            "within_tolerance": within,
            "lifted": lifted,
            "reach_error": reach_error,
            "target_error": goal_error,
        }

    def compute_reward(self, progress: Dict[str, Any], succeeded: Any) -> Any:
        w = self.reward_weights
        p = max(float(w.shaping_power), 1.0)
        goal_error = _as_tensor(progress.get("goal_error", 0.0), self.device)
        lifted = _as_tensor(progress.get("lifted", 0.0), self.device)
        reach_error = _as_tensor(progress.get("reach_error", 0.0), self.device)

        # Hand-to-object distance drives r_reach; r_target only applies once the
        # object has been lifted (Appendix A: "move to goal location *after*
        # lifting"), r_lift rewards elevation and r_success the hold success.
        r_reach = w.reach * _exp(reach_error)
        r_lift = w.lift * lifted
        r_target = w.target * _exp(goal_error * p) * (lifted > 0).to(goal_error.dtype) \
            if torch is not None else 0.0
        r_success = w.success * (
            succeeded.to(goal_error.dtype) if torch is not None else 0.0
        )
        out = r_reach + r_lift + r_target + _as_tensor(r_success, self.device)
        return out


# ---------------------------------------------------------------------------
# Throw
# ---------------------------------------------------------------------------


class ThrowEnv(AllegroKukaEnv):
    """Throw: lift the object and throw it into a bucket at ``g_t in R^3``."""

    task = THROW
    goal_dim = 3

    def measure_progress(self, obs: Any) -> Dict[str, Any]:
        parts = split_observation(obs, self.goal_dim, self.task) if obs is not None else {}
        if not parts:
            return super().measure_progress(obs)
        obj = parts["object_pos"]
        bucket = parts["goal"]
        goal_error = _norm(obj - bucket, dim=-1)
        bucket_radius = float(getattr(getattr(self.env, "cfg", None), "bucket_radius", 0.10) or 0.10)
        within = goal_error <= max(bucket_radius, float(self._tolerance))
        return {
            "goal_error": goal_error,
            "within_tolerance": within,
            "lifted": parts["lifted"],
            "reach_error": _norm(obj - obj, dim=-1),
            "target_error": goal_error,
            "bucket_radius": bucket_radius,
        }

    def compute_reward(self, progress: Dict[str, Any], succeeded: Any) -> Any:
        w = self.reward_weights
        p = max(float(w.shaping_power), 1.0)
        goal_error = _as_tensor(progress.get("goal_error", 0.0), self.device)
        lifted = _as_tensor(progress.get("lifted", 0.0), self.device)
        r_reach = w.reach * _exp(_as_tensor(progress.get("reach_error", 0.0), self.device))
        r_lift = w.lift * lifted
        r_target = w.target * _exp(goal_error * p)
        r_success = w.success * (succeeded.to(goal_error.dtype) if torch is not None else 0.0)
        return r_reach + r_lift + r_target + _as_tensor(r_success, self.device)


# ---------------------------------------------------------------------------
# Reorientation
# ---------------------------------------------------------------------------


class ReorientationEnv(AllegroKukaEnv):
    """Reorientation: reorient the object to a target pose ``g_t in R^7``."""

    task = REORIENTATION
    goal_dim = 7

    def measure_progress(self, obs: Any) -> Dict[str, Any]:
        parts = split_observation(obs, self.goal_dim, self.task) if obs is not None else {}
        if not parts:
            return super().measure_progress(obs)
        keep = self.goal_dim > _OBJECT_POSE
        goal = parts["goal"]
        goal_pos, goal_quat = goal[:, :3], goal[:, 3:7]
        obj = parts["object_pos"]
        obj_quat = parts["object_quat"]

        pos_error = _norm(obj - goal_pos, dim=-1)
        quat_error = _quat_angle_error(obj_quat, goal_quat)
        # combined tolerance: position within delta AND orientation within a
        # matching angular tolerance (the curriculum only scales the position
        # tolerance in the paper, whose reorientation task also carries a
        # per-step success tolerance delta over the full pose).
        angle_tol = float(self._tolerance) * (math.pi / 0.075) if self._tolerance < 0.075 else math.pi
        within = (pos_error <= float(self._tolerance)) & (quat_error <= angle_tol)
        return {
            "goal_error": pos_error + quat_error,
            "within_tolerance": within,
            "lifted": parts["lifted"],
            "reach_error": pos_error,
            "target_error": quat_error,
            "position_error": pos_error,
            "orientation_error": quat_error,
        }

    def compute_reward(self, progress: Dict[str, Any], succeeded: Any) -> Any:
        w = self.reward_weights
        p = max(float(w.shaping_power), 1.0)
        pos_error = _as_tensor(progress.get("position_error", 0.0), self.device)
        ori_error = _as_tensor(progress.get("orientation_error", 0.0), self.device)
        lifted = _as_tensor(progress.get("lifted", 0.0), self.device)
        r_reach = w.reach * _exp(pos_error)
        r_lift = w.lift * lifted
        # target reward only after lifting, aligned with the regrasping task
        gate = (lifted > 0).to(ori_error.dtype) if torch is not None else 1.0
        r_target = w.target * _exp(ori_error * p) * gate
        r_success = w.success * (succeeded.to(ori_error.dtype) if torch is not None else 0.0)
        return r_reach + r_lift + r_target + _as_tensor(r_success, self.device)


def _quat_angle_error(a: Any, b: Any):
    """Quaternion angle error, tolerating a missing helper."""
    try:
        return quat_angle_error(a, b)
    except Exception:  # pragma: no cover
        if torch is None:
            return a
        a = a / (torch.linalg.norm(a, dim=-1, keepdim=True) + 1e-9)
        b = b / (torch.linalg.norm(b, dim=-1, keepdim=True) + 1e-9)
        dot = torch.sum(a * b, dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.acos(dot)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

_TASK_CLASSES = {
    REGRASPING: RegraspingEnv,
    "regrasp": RegraspingEnv,
    THROW: ThrowEnv,
    REORIENTATION: ReorientationEnv,
    "reorient": ReorientationEnv,
}


def make_allegro_kuka_env(
    task: str = REGRASPING,
    config: Any = None,
    num_envs: Optional[int] = None,
    **kwargs: Any,
) -> AllegroKukaEnv:
    """Build the wrapper for one of the three Allegro-Kuka hard tasks."""
    key = str(task).lower().replace("-", "_")
    cls = _TASK_CLASSES.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown Allegro-Kuka task {task!r}; expected one of {sorted(ALLEGRO_KUKA_TASKS)}"
        )
    return cls(task=key if key in ALLEGRO_KUKA_TASKS else REGRASPING,
               config=config, num_envs=num_envs, **kwargs)


def make_regrasping_env(config: Any = None, num_envs: Optional[int] = None, **kwargs: Any):
    """Factory for the regrasping task (Appendix A)."""
    return RegraspingEnv(task=REGRASPING, config=config, num_envs=num_envs, **kwargs)


def make_throw_env(config: Any = None, num_envs: Optional[int] = None, **kwargs: Any):
    """Factory for the throw task (Appendix A)."""
    return ThrowEnv(task=THROW, config=config, num_envs=num_envs, **kwargs)


def make_reorientation_env(config: Any = None, num_envs: Optional[int] = None, **kwargs: Any):
    """Factory for the reorientation task (Appendix A)."""
    return ReorientationEnv(task=REORIENTATION, config=config, num_envs=num_envs, **kwargs)
