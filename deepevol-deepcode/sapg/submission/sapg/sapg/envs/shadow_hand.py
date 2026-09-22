"""Shadow Hand in-hand reorientation environment (SAPG paper, Appendix A / Sec. 5.1).

Paper specification (Appendix A, "Easy Difficulty Tasks"):

    "We test on in-hand reorientation task of a cube using the 24-DoF Shadow
     Hand (OpenAI et al., 2018).  The task is to attain a specified goal
     orientation (specified as a quaternion) for the cube g_t in R^4.  The
     reward is a combination of the orientation error and a success bonus."

    "the observation space consists of the joint angles and velocities q_t,
     qdot_t, object pose x_t and velocities v_t, omega_t"

    "we use the net episode reward as a performance metric for the ShadowHand
     and AllegroHand tasks."

So the observation is

    o_t = [q (24), qdot (24), x_t (7), v_t (3), omega_t (3), g_t (4)]
    obs_dim = 65

and the reward is a weighted combination of an orientation-error shaping term
and a binary success bonus (the cube's orientation is within ``rot_tolerance``
of the goal quaternion).

This module layers that task-specific observation parsing, reward shaping and
success tracking on top of the dependency-light :mod:`sapg.envs.isaac_env`
facade, exactly like :mod:`sapg.envs.allegro_kuka` does for the hard tasks.
The wrapper is simulator agnostic: it works with the real IsaacGym backend when
available and with the pure-torch surrogate otherwise, and it degrades
gracefully when ``torch`` is unavailable (list/float fallbacks) so that the
module can be imported in unit tests without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

try:  # torch is optional at import time (tests must run CPU-only / torch-less).
    import torch
except Exception:  # pragma: no cover - exercised only in torch-less environments
    torch = None  # type: ignore[assignment]

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

# ---------------------------------------------------------------------------
# Constants (Appendix A)
# ---------------------------------------------------------------------------

SHADOW_HAND = "shadow_hand"
IN_HAND_REORIENTATION = "in_hand_reorientation"
SHADOW_HAND_TASKS = (SHADOW_HAND, IN_HAND_REORIENTATION)

#: 24 actuated degrees of freedom (OpenAI et al., 2018).
SHADOW_HAND_JOINTS = 24
#: object pose quaternion lives in R^4 and the goal is a goal quaternion R^4.
GOAL_QUAT_DIM = 4
OBJECT_POSE_DIM = 7
OBJECT_VEL_DIM = 3

#: Appendix A reward = orientation error shaping + success bonus.
ORIENTATION_REWARD_WEIGHT = 1.0
SUCCESS_BONUS = 5.0
#: Success is the cube orientation being within this angular tolerance.
ROT_TOLERANCE_DEFAULT = math.pi / 6.0  # 30 degrees
#: Net episode reward is the paper's metric; episode length for logging.
MAX_EPISODE_LENGTH_DEFAULT = 200
#: Episode is considered a success when the orientation is held for K steps.
HOLD_STEPS_DEFAULT = 10


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------


@dataclass
class ShadowRewardWeights:
    """Weights of the Shadow Hand reward decomposition (Appendix A)."""

    orientation: float = ORIENTATION_REWARD_WEIGHT
    success: float = SUCCESS_BONUS
    reach: float = 0.0
    shaping_power: float = 1.0
    hold: float = 0.0

    @classmethod
    def from_any(cls, weights: Any, task: str = SHADOW_HAND) -> "ShadowRewardWeights":
        """Build from dict / dataclass / attribute object / ``None``."""
        if weights is None:
            return cls()
        if isinstance(weights, ShadowRewardWeights):
            return weights
        base = cls()
        if isinstance(weights, dict):
            data = dict(weights)
        else:
            keys = (
                "orientation",
                "success",
                "reach",
                "shaping_power",
                "hold",
                "orientation_weight",
                "success_bonus",
            )
            data = {}
            for key in keys:
                if hasattr(weights, key):
                    data[key] = getattr(weights, key)
        alias = {
            "orientation_weight": "orientation",
            "success_bonus": "success",
            "reach_weight": "reach",
        }
        for key, value in data.items():
            key = alias.get(key, key)
            if hasattr(base, key) and value is not None:
                setattr(base, key, float(value))
        return base

    def as_dict(self) -> Dict[str, float]:
        return {
            "orientation": float(self.orientation),
            "success": float(self.success),
            "reach": float(self.reach),
            "shaping_power": float(self.shaping_power),
            "hold": float(self.hold),
        }


def default_reward_weights_shadow() -> Dict[str, float]:
    """Paper-specified default reward weights for the Shadow Hand task."""
    return ShadowRewardWeights().as_dict()


# ---------------------------------------------------------------------------
# small tensor helpers (with list/torch-agnostic fallbacks)
# ---------------------------------------------------------------------------


def _is_tensor(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _to_tensor(x: Any, device: Any = None) -> Any:
    if torch is None:
        return x
    if isinstance(x, torch.Tensor):
        return x if device is None else x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _norm_last(x: Any) -> Any:
    """L2 norm along the last dimension (list fallback included)."""
    if _is_tensor(x):
        if x.numel() == 0:
            return x
        return torch.linalg.norm(x, dim=-1)
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        return [math.sqrt(sum(float(v) * float(v) for v in row)) for row in x]
    if isinstance(x, (list, tuple)):
        return math.sqrt(sum(float(v) * float(v) for v in x))
    return abs(float(x))


def _sqrt(x: Any, eps: float = 1e-8) -> Any:
    if _is_tensor(x):
        return torch.sqrt(torch.clamp(x, min=eps))
    return math.sqrt(max(float(x), eps))


def _zeros(n: int, device: Any = None) -> Any:
    if torch is None:
        return [0.0] * n
    return torch.zeros(n, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Success tracking (orientation held for K consecutive steps)
# ---------------------------------------------------------------------------


class SuccessTracker:
    """Tracks per-environment successes with a consecutive-step hold criterion.

    Mirrors :class:`sapg.envs.allegro_kuka.SuccessTracker` so the SAPG logging
    and curriculum code can treat both task families identically.
    """

    def __init__(self, num_envs: int, hold_steps: int = HOLD_STEPS_DEFAULT, device: Any = None):
        self.num_envs = int(num_envs)
        self.hold_steps = max(int(hold_steps), 1)
        self.device = device
        if torch is not None:
            self.count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
            self.ever = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        else:  # pragma: no cover
            self.count = [0] * self.num_envs
            self.ever = [False] * self.num_envs

    def reset(self, env_ids: Optional[Any] = None) -> None:
        if torch is not None:
            if env_ids is None:
                self.count.zero_()
                self.ever.zero_()
            else:
                idx = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
                self.count[idx] = 0
                self.ever[idx] = False
        else:  # pragma: no cover
            ids = range(self.num_envs) if env_ids is None else env_ids
            for i in ids:
                self.count[i] = 0
                self.ever[i] = False

    def update(self, within_tolerance: Any) -> Any:
        """Advance hold counters; return a boolean success mask for this step."""
        if torch is not None:
            ok = within_tolerance
            if not isinstance(ok, torch.Tensor):
                ok = torch.as_tensor(ok, device=self.device)
            ok = ok.to(self.count.device).bool().reshape(-1)
            self.count = torch.where(ok, self.count + 1, torch.zeros_like(self.count))
            success = self.count >= self.hold_steps
            self.ever = self.ever | success
            return success
        ok = list(within_tolerance)
        success = []
        for i in range(self.num_envs):
            if bool(ok[i]):
                self.count[i] += 1
            else:
                self.count[i] = 0
            s = self.count[i] >= self.hold_steps
            success.append(s)
            self.ever[i] = self.ever[i] or s
        return success

    def finalise_episode(self, dones: Any) -> Any:
        """Return per-env success counts for the episodes that just ended."""
        if torch is not None:
            done = dones
            if not isinstance(done, torch.Tensor):
                done = torch.as_tensor(done, device=self.device)
            done = done.to(self.ever.device).bool().reshape(-1)
            counts = torch.where(
                done,
                self.ever.long(),
                torch.zeros_like(self.ever, dtype=torch.long),
            )
            self.count = torch.where(done, torch.zeros_like(self.count), self.count)
            self.ever = self.ever & (~done)
            return counts
        done = list(dones)
        counts = []
        for i in range(self.num_envs):
            counts.append(int(self.ever[i]) if bool(done[i]) else 0)
            if bool(done[i]):
                self.count[i] = 0
                self.ever[i] = False
        return counts


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------


def split_observation(obs: Any, task: str = SHADOW_HAND) -> Dict[str, Any]:
    """Slice a flat Shadow Hand observation into its Appendix-A components.

    ``o_t = [q (24), qdot (24), x_t (7), v_t (3), omega_t (3), g_t (4)]``

    Handles ``[..., obs_dim]`` tensors as well as plain nested lists.
    """
    n = SHADOW_HAND_JOINTS
    if _is_tensor(obs):
        q = obs[..., :n]
        qdot = obs[..., n : 2 * n]
        object_pose = obs[..., 2 * n : 2 * n + OBJECT_POSE_DIM]
        object_pos = obs[..., 2 * n : 2 * n + 3]
        object_quat = obs[..., 2 * n + 3 : 2 * n + OBJECT_POSE_DIM]
        lin_vel = obs[..., 2 * n + OBJECT_POSE_DIM : 2 * n + OBJECT_POSE_DIM + OBJECT_VEL_DIM]
        ang_vel = obs[
            ..., 2 * n + OBJECT_POSE_DIM + OBJECT_VEL_DIM : 2 * n + OBJECT_POSE_DIM + 2 * OBJECT_VEL_DIM
        ]
        goal = obs[..., 2 * n + OBJECT_POSE_DIM + 2 * OBJECT_VEL_DIM :]
        return {
            "q": q,
            "qdot": qdot,
            "object_pose": object_pose,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": lin_vel,
            "object_ang_vel": ang_vel,
            "goal": goal,
            "goal_quat": goal,
            "aux": obs[..., 2 * n + OBJECT_POSE_DIM :],
            "lifted": _norm_last(ang_vel) * 0.0 + 1.0,
        }
    row = obs
    if isinstance(row, (list, tuple)) and row and isinstance(row[0], (list, tuple)):
        return {
            "q": [list(r[:n]) for r in row],
            "qdot": [list(r[n : 2 * n]) for r in row],
            "object_pose": [list(r[2 * n : 2 * n + OBJECT_POSE_DIM]) for r in row],
            "object_pos": [list(r[2 * n : 2 * n + 3]) for r in row],
            "object_quat": [list(r[2 * n + 3 : 2 * n + OBJECT_POSE_DIM]) for r in row],
            "object_lin_vel": [list(r[2 * n + OBJECT_POSE_DIM : 2 * n + OBJECT_POSE_DIM + 3]) for r in row],
            "object_ang_vel": [
                list(r[2 * n + OBJECT_POSE_DIM + 3 : 2 * n + OBJECT_POSE_DIM + 6]) for r in row
            ],
            "goal": [list(r[2 * n + OBJECT_POSE_DIM + 6 :]) for r in row],
            "goal_quat": [list(r[2 * n + OBJECT_POSE_DIM + 6 :]) for r in row],
            "aux": [list(r[2 * n + OBJECT_POSE_DIM :]) for r in row],
            "lifted": [1.0] * len(row),
        }
    seq = list(row)
    return {
        "q": seq[:n],
        "qdot": seq[n : 2 * n],
        "object_pose": seq[2 * n : 2 * n + OBJECT_POSE_DIM],
        "object_pos": seq[2 * n : 2 * n + 3],
        "object_quat": seq[2 * n + 3 : 2 * n + OBJECT_POSE_DIM],
        "object_lin_vel": seq[2 * n + OBJECT_POSE_DIM : 2 * n + OBJECT_POSE_DIM + 3],
        "object_ang_vel": seq[2 * n + OBJECT_POSE_DIM + 3 : 2 * n + OBJECT_POSE_DIM + 6],
        "goal": seq[2 * n + OBJECT_POSE_DIM + 6 :],
        "goal_quat": seq[2 * n + OBJECT_POSE_DIM + 6 :],
        "aux": seq[2 * n + OBJECT_POSE_DIM :],
        "lifted": 1.0,
    }


def _quat_angle_to(quat: Any, goal: Any, eps: float = 1e-6) -> Any:
    """Absolute rotation angle between two quaternions (``w, x, y, z``)."""
    if _is_tensor(quat) and _is_tensor(goal):
        q = quat_normalize(quat)
        g = quat_normalize(goal)
        dot = (q * g).sum(dim=-1).abs().clamp(max=1.0 - eps)
        return 2.0 * torch.acos(dot)
    if isinstance(quat, (list, tuple)) and quat and isinstance(quat[0], (list, tuple)):
        return [_quat_angle_to(q, g) for q, g in zip(quat, goal)]
    qn = _normalise_list(list(quat))
    gn = _normalise_list(list(goal))
    dot = min(1.0 - eps, abs(sum(a * b for a, b in zip(qn, gn))))
    return 2.0 * math.acos(dot)


def _normalise_list(v: Sequence[float], eps: float = 1e-8) -> list:
    norm = math.sqrt(sum(float(x) * float(x) for x in v)) + eps
    return [float(x) / norm for x in v]


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------


class ShadowHandEnv:
    """Shadow Hand in-hand reorientation wrapper (Appendix A, easy tasks).

    Options / hyper-parameters for this task group are given in Table 3
    (``phi_dim = 16``, MLP ``512x512x256x128`` ELU, ``clip_epsilon = 0.2``,
    ``horizon_length = 8``, ``mini_epochs = 5``).
    """

    is_shadow_hand = True
    task_group = "shadow_hand"

    def __init__(
        self,
        task: str = SHADOW_HAND,
        num_envs: Optional[int] = None,
        env: Optional[Any] = None,
        config: Optional[Any] = None,
        device: Optional[Any] = None,
        hold_steps: int = HOLD_STEPS_DEFAULT,
        rot_tolerance: float = ROT_TOLERANCE_DEFAULT,
        reward_weights: Optional[Any] = None,
        use_curriculum: bool = False,
        seed: Optional[int] = None,
        max_episode_length: int = MAX_EPISODE_LENGTH_DEFAULT,
        **env_kwargs: Any,
    ) -> None:
        self.task = SHADOW_HAND
        self.config = config
        self.device = device if device is not None else getattr(config, "device", None)
        self._num_envs_override = num_envs
        self._env = env
        self._owns_env = env is None
        self.hold_steps = max(int(hold_steps), 1)
        self.rot_tolerance = float(rot_tolerance)
        self.reward_weights = ShadowRewardWeights.from_any(reward_weights, self.task)
        # Easy tasks do not use the success-tolerance curriculum, but the flag
        # is accepted so that the shared SAPG training script can be reused.
        self.use_curriculum = bool(use_curriculum)
        self.seed = seed
        self.max_episode_length = int(max_episode_length)
        self._env_kwargs = dict(env_kwargs)
        self._done_count = 0

        self._tracker: Optional[SuccessTracker] = None
        self._episode_return: Any = None
        self._episode_length: Any = None
        if self._env is not None:
            self._init_bookkeeping()

    # -- lazy env construction -------------------------------------------------

    @property
    def env(self) -> Any:
        if self._env is None:
            kwargs = dict(self._env_kwargs)
            if self._num_envs_override is not None:
                kwargs["num_envs"] = self._num_envs_override
            if self.device is not None:
                kwargs.setdefault("device", self.device)
            if self.seed is not None:
                kwargs.setdefault("seed", self.seed)
            self._env = make_isaac_env(self.task, config=self.config, **kwargs)
            self._init_bookkeeping()
        return self._env

    def _init_bookkeeping(self) -> None:
        n = len(self.env)
        device = getattr(self.env, "device", None) or self.device
        self._tracker = SuccessTracker(n, hold_steps=self.hold_steps, device=device)
        if torch is not None:
            self._episode_return = torch.zeros(n, dtype=torch.float32, device=device)
            self._episode_length = torch.zeros(n, dtype=torch.long, device=device)
        else:  # pragma: no cover
            self._episode_return = [0.0] * n
            self._episode_length = [0] * n

    # -- dimensions ------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        if self._env is not None:
            return len(self._env)
        if self._num_envs_override is not None:
            return int(self._num_envs_override)
        cfg_n = getattr(self.config, "num_envs", None)
        return int(cfg_n) if cfg_n else 1

    @property
    def obs_dim(self) -> int:
        cfg = getattr(self.config, "obs_dim", None)
        if cfg:
            return int(cfg)
        return obs_dim_for(self.task)

    @property
    def action_dim(self) -> int:
        cfg = getattr(self.config, "action_dim", None)
        if cfg:
            return int(cfg)
        return action_dim_for(self.task)

    @property
    def goal_dim(self) -> int:
        return goal_dim_for(self.task)

    @property
    def joint_dim(self) -> int:
        return joint_dim_for(self.task)

    def __len__(self) -> int:
        return self.num_envs

    # -- core interface --------------------------------------------------------

    def reset(self, env_ids: Optional[Any] = None) -> Any:
        obs = self.env.reset(env_ids)
        if self._tracker is not None:
            self._tracker.reset(env_ids)
        if env_ids is None:
            if _is_tensor(self._episode_return):
                self._episode_return.zero_()
                self._episode_length.zero_()
            else:  # pragma: no cover
                self._episode_return = [0.0] * self.num_envs
                self._episode_length = [0] * self.num_envs
        else:
            if _is_tensor(self._episode_return):
                idx = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
                self._episode_return[idx] = 0.0
                self._episode_length[idx] = 0
        return obs

    def step(self, actions: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(actions)
        info = dict(info or {})

        progress = self.measure_progress(obs)
        succeeded = self._tracker.update(progress["within_tolerance"]) if self._tracker else None

        shaped = self.compute_reward(progress, succeeded, base_reward=reward)
        if _is_tensor(shaped):
            reward_t = shaped
        else:
            reward_t = _to_tensor(shaped, getattr(self.env, "device", None))

        # Episode bookkeeping ---------------------------------------------------
        if _is_tensor(self._episode_return) and _is_tensor(reward_t):
            self._episode_return = self._episode_return + reward_t
            self._episode_length = self._episode_length + 1
        else:  # pragma: no cover
            for i, (r, e) in enumerate(zip(reward_t, self._episode_return)):
                self._episode_return[i] = e + float(r)
                self._episode_length[i] += 1

        successes = None
        if self._tracker is not None and done is not None:
            successes = self._tracker.finalise_episode(done)

        episode_return = None
        episode_length = None
        if successes is not None and _is_tensor(self._episode_return):
            done_mask = _to_tensor(done, self._episode_return.device).bool().reshape(-1)
            episode_return = self._episode_return.clone()
            episode_length = self._episode_length.clone()
            info.setdefault("episode_return", self._episode_return[done_mask].detach().cpu())
            info.setdefault("episode_length", self._episode_length[done_mask].detach().cpu())
            info.setdefault("successes", successes[done_mask].detach().cpu())
            self._episode_return = torch.where(
                done_mask, torch.zeros_like(self._episode_return), self._episode_return
            )
            self._episode_length = torch.where(
                done_mask, torch.zeros_like(self._episode_length), self._episode_length
            )
            info["episode_stats_return"] = episode_return
            info["episode_stats_length"] = episode_length

        info.setdefault("successes", successes)
        info["within_tolerance"] = progress["within_tolerance"]
        info["orientation_error"] = progress["orientation_error"]
        info["success_tolerance"] = self.rot_tolerance
        info["task"] = self.task
        info["success_count"] = (
            float(successes.sum()) if _is_tensor(successes) else float(sum(1 for s in successes or [] if s))
        )
        return obs, reward_t, done, info

    # -- task hooks ------------------------------------------------------------

    def measure_progress(self, obs: Any) -> Dict[str, Any]:
        """Orientation error between the cube and the goal quaternion (Appendix A)."""
        parts = split_observation(obs, self.task)
        obj_quat = parts["object_quat"]
        goal_quat = parts["goal_quat"]

        if _is_tensor(obj_quat) and _is_tensor(goal_quat):
            valid = goal_quat.abs().sum(dim=-1) > 1e-6
            zero = torch.zeros((), dtype=goal_quat.dtype, device=goal_quat.device)
            g = torch.where(valid.unsqueeze(-1), goal_quat, obj_quat * 0 + 1.0)
            g = torch.where(
                valid.unsqueeze(-1),
                g,
                torch.cat([zero.new_ones(goal_quat.shape[:-1] + (1,)), goal_quat[..., 1:]], dim=-1),
            )
            angle = _quat_angle_to(obj_quat, g)
            within = angle <= self.rot_tolerance
            return {
                "orientation_error": angle,
                "orientation_reward": 1.0 - angle / math.pi,
                "within_tolerance": within,
                "distance": angle,
                "goal_quat": g,
            }
        if isinstance(obj_quat, list) and obj_quat and isinstance(obj_quat[0], (list, tuple)):
            errs = []
            within = []
            for o, g in zip(obj_quat, goal_quat):
                if sum(abs(float(v)) for v in g) < 1e-6:
                    g = [1.0, 0.0, 0.0, 0.0]
                e = _quat_angle_to(o, g)
                errs.append(e)
                within.append(e <= self.rot_tolerance)
            return {
                "orientation_error": errs,
                "orientation_reward": [1.0 - e / math.pi for e in errs],
                "within_tolerance": within,
                "distance": errs,
                "goal_quat": goal_quat,
            }
        g = goal_quat if sum(abs(float(v)) for v in goal_quat) > 1e-6 else [1.0, 0.0, 0.0, 0.0]
        e = _quat_angle_to(obj_quat, g)
        return {
            "orientation_error": e,
            "orientation_reward": 1.0 - e / math.pi,
            "within_tolerance": e <= self.rot_tolerance,
            "distance": e,
            "goal_quat": g,
        }

    def compute_reward(
        self,
        progress: Dict[str, Any],
        succeeded: Any = None,
        base_reward: Any = None,
    ) -> Any:
        """Orientation-error shaping + success bonus (Appendix A)."""
        w = self.reward_weights
        orient = progress.get("orientation_reward")
        if orient is None:
            orient = 1.0 - progress.get("orientation_error", 0.0) / math.pi
        if _is_tensor(orient):
            reward = w.orientation * orient.to(torch.float32)
        elif isinstance(orient, list):
            reward = [w.orientation * float(v) for v in orient]
        else:
            reward = w.orientation * float(orient)

        if succeeded is not None:
            if _is_tensor(succeeded):
                reward = reward + w.success * succeeded.to(reward.dtype)
            elif isinstance(succeeded, list):
                if isinstance(reward, list):
                    reward = [r + w.success * float(s) for r, s in zip(reward, succeeded)]
                else:
                    reward = reward + w.success * float(sum(1 for s in succeeded if s))
            else:
                reward = reward + w.success * float(succeeded)

        # Keep the underlying simulator's reward available for parity with the
        # IsaacGym backend but do not double count it: the surrogate already
        # includes the same shaping terms.
        if base_reward is not None and not _is_tensor(orient):
            pass
        return reward

    # -- curriculum / tolerance ------------------------------------------------

    @property
    def success_tolerance(self) -> float:
        return self.rot_tolerance

    def set_success_tolerance(self, tolerance: float) -> float:
        """Update the angular success tolerance (radians)."""
        self.rot_tolerance = float(tolerance)
        return self.rot_tolerance

    def set_rot_tolerance(self, tolerance: float) -> float:
        return self.set_success_tolerance(tolerance)

    # -- misc ------------------------------------------------------------------

    @property
    def goal(self) -> Any:
        g = getattr(self.env, "goal", None)
        if g is not None:
            return g
        return _zeros(max(self.num_envs, 1), self.device)

    def sample_actions(self) -> Any:
        if hasattr(self.env, "sample_actions"):
            return self.env.sample_actions()
        if torch is not None:
            return torch.zeros(self.num_envs, self.action_dim, device=self.device)
        return [[0.0] * self.action_dim for _ in range(self.num_envs)]  # pragma: no cover

    def info_snapshot(self) -> Dict[str, Any]:
        snap = {
            "task": self.task,
            "task_group": self.task_group,
            "num_envs": self.num_envs,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "goal_dim": self.goal_dim,
            "rot_tolerance": self.rot_tolerance,
            "hold_steps": self.hold_steps,
            "reward_weights": self.reward_weights.as_dict(),
        }
        if hasattr(self.env, "info_snapshot"):
            try:
                snap.update(self.env.info_snapshot())
            except Exception:  # pragma: no cover
                pass
        return snap

    def close(self) -> None:
        if self._owns_env and self._env is not None and hasattr(self._env, "close"):
            self._env.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ShadowHandEnv(task={self.task!r}, num_envs={self.num_envs}, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"goal_dim={self.goal_dim})"
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_shadow_hand_env(
    task: str = SHADOW_HAND,
    config: Optional[Any] = None,
    num_envs: Optional[int] = None,
    env: Optional[Any] = None,
    **kwargs: Any,
) -> ShadowHandEnv:
    """Create the Shadow Hand in-hand reorientation environment (Appendix A)."""
    if config is not None and "device" not in kwargs:
        device = getattr(config, "device", None)
        if device is not None:
            kwargs["device"] = device
    return ShadowHandEnv(task=task, num_envs=num_envs, env=env, config=config, **kwargs)


# Alias used by the task registry / training scripts.
make_in_hand_reorientation_env = make_shadow_hand_env


__all__ = [
    "SHADOW_HAND",
    "IN_HAND_REORIENTATION",
    "SHADOW_HAND_TASKS",
    "SHADOW_HAND_JOINTS",
    "GOAL_QUAT_DIM",
    "ROT_TOLERANCE_DEFAULT",
    "ShadowRewardWeights",
    "SuccessTracker",
    "ShadowHandEnv",
    "split_observation",
    "make_shadow_hand_env",
    "make_in_hand_reorientation_env",
    "default_reward_weights_shadow",
]
