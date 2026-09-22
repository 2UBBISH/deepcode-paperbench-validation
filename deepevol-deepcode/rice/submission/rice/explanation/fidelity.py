"""Fidelity evaluator for RICE explanations (Experiment I).

This module implements the fidelity score introduced by StateMask (Cheng et al.,
2023) and adopted by RICE (Sec. 4.1 "Evaluation Metrics"):

    fidelity = log(d / d_max) - log(l / L)

where

* ``L``      : length of the whole trajectory,
* ``l``      : width of the sliding window (``l = L x K``, ``K in {10,20,30,40}%``),
* ``d``      : the reward change caused by randomizing (masking) the actions in
               the selected window (``|R' - R|``),
* ``d_max``  : the maximum possible reward change for the environment
               (the maximum single-episode reward).

Pipeline (Sec. 4.1 + Experiment I, Sec. 4.2):

1. roll the frozen target policy ``pi`` for a full trajectory and record the
   step-level importance score of every visited state,
2. slide a window of width ``l`` over the trajectory and pick the window with
   the **highest average importance score**,
3. fast-forward to the start of that window, force the agent to take random
   actions for ``l`` steps ("masking"), then follow the frozen policy ``pi``
   again to the end of the episode, measuring the resulting return ``R'``,
4. compute ``d = |R' - R|`` and the fidelity score above.

Experiment I repeats this over ``500`` trajectories with ``3`` random seeds and
reports the mean and standard deviation for each ``K``.  The module also
provides the wall-clock efficiency comparison helpers used for Table 4 (mask
network training time under a fixed sample budget).

Everything is defensive: environments may be real gym/MuJoCo instances or one
of the lightweight fallbacks used for smoke testing, and the evaluator always
falls back to deterministic action replay when direct state restore fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Internal imports (defensive: the module stays importable in minimal setups)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from rice.utils.metrics import best_window, mean_std, window_length
except Exception:  # pragma: no cover
    best_window = None  # type: ignore
    mean_std = None  # type: ignore
    window_length = None  # type: ignore

try:
    from rice.utils.seeding import get_rng, seed_env, seed_from
except Exception:  # pragma: no cover
    def get_rng(seed=None):  # type: ignore
        return np.random.RandomState(seed)

    def seed_env(env, seed=None, rank=0):  # type: ignore
        return seed

    def seed_from(base_seed, *offsets):  # type: ignore
        return int(base_seed)

try:
    from rice.utils.logging import get_logger
except Exception:  # pragma: no cover
    def get_logger(*args, **kwargs):  # type: ignore
        import logging

        return logging.getLogger("rice")

try:
    from rice.explanation.importance import score_trajectory as _score_trajectory
except Exception:  # pragma: no cover
    _score_trajectory = None  # type: ignore

try:
    from rice.explanation.critical_state import (
        TrajectoryRollout,
        policy_action,
        roll_trajectory,
    )
except Exception:  # pragma: no cover
    TrajectoryRollout = None  # type: ignore
    roll_trajectory = None  # type: ignore

    def policy_action(policy, observation, deterministic=False):  # type: ignore
        if hasattr(policy, "predict"):
            return policy.predict(observation, deterministic=deterministic)[0]
        if hasattr(policy, "act"):
            return policy.act(observation, deterministic=deterministic)
        return policy(observation)

try:
    from rice.models.policies import sample_random_action
except Exception:  # pragma: no cover
    def sample_random_action(action_space=None, rng=None, **kwargs):  # type: ignore
        rng = rng or np.random
        if action_space is not None and hasattr(action_space, "sample"):
            return action_space.sample()
        dim = kwargs.get("dim", 1)
        return rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)

try:
    from rice.envs.make_env import d_max_for
except Exception:  # pragma: no cover
    def d_max_for(env_id, default=None):  # type: ignore
        return default

try:
    from rice.envs.reset_wrapper import get_env_state, set_env_state
except Exception:  # pragma: no cover
    def get_env_state(env):  # type: ignore
        return None

    def set_env_state(env, packed):  # type: ignore
        return False


__all__ = [
    "DEFAULT_K_VALUES",
    "DEFAULT_N_TRAJECTORIES",
    "DEFAULT_SEEDS",
    "FidelityConfig",
    "FidelityResult",
    "FidelityEvaluator",
    "TrajectoryFidelityDetail",
    "fidelity_score_from_change",
    "fidelity_from_rewards",
    "evaluate_fidelity",
    "evaluate_fidelity_multi_K",
    "evaluate_methods",
    "random_importance_scores",
    "training_time_reduction",
    "format_fidelity_table",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
#: Sliding-window ratios reported by Experiment I (Sec. 4.2).
DEFAULT_K_VALUES: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)

#: Number of trajectories used by Experiment I.
DEFAULT_N_TRAJECTORIES: int = 500

#: Random seeds used to report mean +- std (Sec. 4.2).
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: Fallback episode horizon when the environment does not report one.
DEFAULT_HORIZON: int = 1000

#: Guard on the "resume the policy" tail so a broken env cannot loop forever.
MAX_TAIL_FACTOR: int = 10


# --------------------------------------------------------------------------------------
# Low level helpers
# --------------------------------------------------------------------------------------
def _unpack_step(result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalise a ``env.step`` result to ``(obs, reward, terminated, truncated, info)``."""
    if not isinstance(result, (tuple, list)):
        return result, 0.0, False, False, {}
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, done, info = result
        terminated, truncated = bool(done), False
    elif len(result) == 3:
        obs, reward, info = result
        terminated, truncated = False, False
    else:  # pragma: no cover - unusual API
        obs = result[0]
        reward = float(result[1]) if len(result) > 1 else 0.0
        terminated = truncated = False
        info = {}
    reward = float(np.asarray(reward).reshape(-1)[0]) if np.ndim(reward) else float(reward)
    info = info if isinstance(info, dict) else {}
    return obs, reward, bool(terminated), bool(truncated), info


def _as_flat_obs(observation: Any) -> np.ndarray:
    """Flatten an observation (array / dict / scalar) to a 1-D float32 vector."""
    if isinstance(observation, dict):
        parts = [np.asarray(observation[k], dtype=np.float32).reshape(-1) for k in sorted(observation)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def _episode_horizon(env: Any, default: int = DEFAULT_HORIZON) -> int:
    """Best-effort resolution of the episode horizon ``T`` (used as ``L``)."""
    if env is None:
        return default
    for attr in ("rice_max_episode_steps", "_max_episode_steps", "max_episode_steps"):
        value = getattr(env, attr, None)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    spec = getattr(env, "rice_env_spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None):
        try:
            return int(spec.max_episode_steps)
        except (TypeError, ValueError):
            pass
    inner = getattr(env, "env", None)
    if inner is not None and inner is not env:
        return _episode_horizon(inner, default)
    return default


def _is_discrete(action_space: Any) -> bool:
    if action_space is None:
        return False
    if hasattr(action_space, "n") and not hasattr(action_space, "shape"):
        return True
    if type(action_space).__name__.lower().startswith("discrete"):
        return True
    return False


def random_importance_scores(n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Importance scores of the "Random" baseline explanation (uniform in [0, 1)).

    The paper's Random baseline "identifies critical steps by randomly selecting a
    visited state as the critical state" (Sec. 4.1), which is equivalent to using
    uninformative random importance scores when selecting the window.
    """
    rng = rng if rng is not None else np.random
    n = max(int(n), 0)
    if hasattr(rng, "random"):
        return np.asarray(rng.random(n), dtype=np.float64)
    return np.random.random(n)  # pragma: no cover


def fidelity_score_from_change(d: float, d_max: float, l: int, L: int, eps: float = 1e-12) -> float:
    """``log(d / d_max) - log(l / L)`` (StateMask fidelity, Sec. 4.1).

    A higher score indicates a higher-fidelity explanation.  Degenerate inputs
    (zero reward change / zero length) are ``eps``-clamped so the score stays
    finite and monotone.
    """
    d = abs(float(d))
    d_max = float(d_max)
    l = int(l)
    L = int(L)
    if d_max <= 0:
        d_max = max(d, eps)
    if L <= 0 or l <= 0:
        return float("nan")
    ratio_d = max(d / d_max, eps)
    ratio_l = max(min(l / L, 1.0), eps)
    return float(math.log(ratio_d) - math.log(ratio_l))


def fidelity_from_rewards(
    R: float,
    R_prime: float,
    d_max: float,
    l: int,
    L: int,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Convenience wrapper computing ``d`` from two episode returns."""
    d = abs(float(R_prime) - float(R))
    score = fidelity_score_from_change(d, d_max, l, L, eps=eps)
    return {"R": float(R), "R_prime": float(R_prime), "d": float(d), "d_max": float(d_max),
            "l": int(l), "L": int(L), "fidelity": score}


# --------------------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------------------
@dataclass
class TrajectoryFidelityDetail:
    """Per-trajectory fidelity bookkeeping."""

    trajectory_index: int = 0
    seed: Optional[int] = None
    L: int = 0
    l: int = 0
    K: float = 0.0
    window_start: int = 0
    window_end: int = 0
    window_importance: float = 0.0
    baseline_reward: float = 0.0
    randomized_reward: float = 0.0
    d: float = 0.0
    d_max: float = 0.0
    fidelity: float = float("nan")
    terminated: bool = False
    steps_taken: int = 0
    restore_mode: str = "none"
    scoring: str = "mask"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_index": self.trajectory_index,
            "seed": self.seed,
            "L": self.L,
            "l": self.l,
            "K": self.K,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window_importance": self.window_importance,
            "baseline_reward": self.baseline_reward,
            "randomized_reward": self.randomized_reward,
            "d": self.d,
            "d_max": self.d_max,
            "fidelity": self.fidelity,
            "terminated": self.terminated,
            "steps_taken": self.steps_taken,
            "restore_mode": self.restore_mode,
            "scoring": self.scoring,
        }


@dataclass
class FidelityResult:
    """Aggregated fidelity of one explanation method at one window ratio ``K``."""

    env_id: str = "default"
    K: float = 0.1
    scoring: str = "mask"
    n_trajectories: int = 0
    n_seeds: int = 0
    seeds: Tuple[int, ...] = ()
    mean: float = float("nan")
    std: float = float("nan")
    values: List[float] = field(default_factory=list)
    per_seed: Dict[int, float] = field(default_factory=dict)
    per_seed_std: Dict[int, float] = field(default_factory=dict)
    d_max: Optional[float] = None
    mean_window_length: float = float("nan")
    details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.mean

    def to_dict(self, include_details: bool = False) -> Dict[str, Any]:
        out = {
            "env_id": self.env_id,
            "K": self.K,
            "K_percent": round(100.0 * self.K, 4),
            "scoring": self.scoring,
            "n_trajectories": self.n_trajectories,
            "n_seeds": self.n_seeds,
            "seeds": list(self.seeds),
            "mean": self.mean,
            "std": self.std,
            "d_max": self.d_max,
            "mean_window_length": self.mean_window_length,
            "per_seed": {str(k): v for k, v in self.per_seed.items()},
            "per_seed_std": {str(k): v for k, v in self.per_seed_std.items()},
        }
        if include_details:
            out["details"] = list(self.details)
        return out

    def format(self, decimals: int = 3, percent: bool = True) -> str:
        label = f"K={100 * self.K:.0f}%" if percent else f"K={self.K}"
        return f"{self.env_id} {label}: {self.mean:.{decimals}f} +- {self.std:.{decimals}f}"


@dataclass
class FidelityConfig:
    """Configuration bundle mirroring the paper's Experiment-I setup."""

    env_id: str = "default"
    K_values: Tuple[float, ...] = DEFAULT_K_VALUES
    n_trajectories: int = DEFAULT_N_TRAJECTORIES
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    deterministic_policy: bool = True
    scoring: str = "mask"  # "mask" | "random"
    max_steps: Optional[int] = None
    window_metric: str = "mean"
    store_details: bool = False

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **kwargs) -> "FidelityConfig":
        cfg = dict(cfg or {})
        nested = cfg.get("fidelity") or cfg.get("explanation") or {}
        if isinstance(nested, dict):
            merged = {**nested, **{k: v for k, v in cfg.items() if k not in ("fidelity", "explanation")}}
        else:
            merged = cfg
        merged.update(kwargs)
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        clean = {k: v for k, v in merged.items() if k in fields and v is not None}
        if "K_values" in clean and isinstance(clean["K_values"], (list, tuple)):
            clean["K_values"] = tuple(float(k) for k in clean["K_values"])
        if "seeds" in clean and isinstance(clean["seeds"], (list, tuple)):
            clean["seeds"] = tuple(int(s) for s in clean["seeds"])
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "K_values": list(self.K_values),
            "n_trajectories": self.n_trajectories,
            "seeds": list(self.seeds),
            "deterministic_policy": self.deterministic_policy,
            "scoring": self.scoring,
            "max_steps": self.max_steps,
            "window_metric": self.window_metric,
        }


# --------------------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------------------
class FidelityEvaluator:
    """Computes the StateMask/RICE fidelity score for an explanation method.

    Parameters
    ----------
    env:
        Environment (gym-like) used to roll trajectories.  Restored in place.
    policy:
        Frozen target policy ``pi`` (SB3 ``predict``, native ``act`` or callable).
    mask_net:
        Trained mask network used to score state importance.  ``None`` disables
        mask scoring (the evaluator then uses the "Random" baseline, i.e. random
        importance scores, unless a custom ``scorer`` is supplied).
    env_id:
        Registry key of the environment (used for ``d_max`` lookup).
    d_max:
        Maximum single-episode reward; resolved via ``rice.envs.make_env.d_max_for``
        when omitted.
    scoring:
        ``"mask"`` (``P(keep)`` from the mask net), ``"random"`` (Random baseline)
        or ``"auto"``.
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        mask_net: Any = None,
        env_id: Optional[str] = None,
        d_max: Optional[float] = None,
        scoring: str = "auto",
        scorer: Optional[Callable[[Any, Sequence[Any]], np.ndarray]] = None,
        rng: Optional[np.random.RandomState] = None,
        logger: Any = None,
        deterministic_policy: bool = True,
        action_mode: str = "random",
        restore: str = "auto",
        max_tail_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.mask_net = mask_net
        self.env_id = env_id or self._infer_env_id(env) or "default"
        self.d_max = d_max
        self.scorer = scorer
        self.rng = rng if rng is not None else get_rng(seed)
        self.logger = logger if logger is not None else get_logger("rice.fidelity")
        self.deterministic_policy = deterministic_policy
        self.action_mode = action_mode
        self.restore = restore
        self.max_tail_steps = max_tail_steps
        self.discrete = _is_discrete(getattr(env, "action_space", None))

        if scoring == "auto":
            scoring = "mask" if (mask_net is not None or scorer is not None) else "random"
        self.scoring = scoring
        if self.scoring == "random":
            # Random baseline: uninformative importance scores.
            self.scorer = lambda observations, _rng=self.rng: random_importance_scores(
                len(observations), _rng
            )

        self.stats: Dict[str, Any] = {"trajectories": 0, "restore_direct": 0, "restore_replay": 0}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _infer_env_id(env: Any) -> Optional[str]:
        if env is None:
            return None
        for attr in ("rice_env_key", "env_id", "id"):
            value = getattr(env, attr, None)
            if isinstance(value, str):
                return value
        spec = getattr(env, "rice_env_spec", None)
        if spec is not None:
            return getattr(spec, "key", None) or getattr(spec, "env_id", None)
        inner = getattr(env, "env", None)
        if inner is not None and inner is not env:
            return FidelityEvaluator._infer_env_id(inner)
        return None

    @property
    def horizon(self) -> int:
        return _episode_horizon(self.env, DEFAULT_HORIZON)

    def resolve_d_max(self, fallback_reward: float = 0.0) -> float:
        """Max possible reward change ``d_max`` (max single-episode reward)."""
        if self.d_max is not None and float(self.d_max) > 0:
            return float(self.d_max)
        resolved = None
        try:
            resolved = d_max_for(self.env_id, default=None)
        except Exception:  # pragma: no cover
            resolved = None
        if resolved is not None and float(resolved) > 0:
            return float(resolved)
        # Last resort: use the magnitude of the observed return (never zero).
        return float(abs(fallback_reward)) or 1.0

    def _reset_env(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            try:
                seed_env(self.env, seed)
            except Exception:  # pragma: no cover
                pass
        result = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        if isinstance(result, (tuple, list)) and len(result) == 2:
            return result[0], result[1] if isinstance(result[1], dict) else {}
        return result, {}

    def _step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        return _unpack_step(self.env.step(action))

    def _random_action(self) -> Any:
        if self.scorer is not None and self.scoring == "random" and self.action_mode == "random":
            pass  # keep the branch explicit: random actions below
        return sample_random_action(
            action_space=getattr(self.env, "action_space", None),
            rng=self.rng,
            discrete=self.discrete,
        )

    # ------------------------------------------------------------- importance
    def importance_scores(self, observations: Sequence[Any]) -> np.ndarray:
        """Step-level importance scores for a trajectory.

        Uses the trained mask net (``P(a_t^m = 0 | s_t)``, the "keep" probability)
        when available, otherwise the Random baseline scores.
        """
        observations = list(observations)
        if len(observations) == 0:
            return np.zeros(0, dtype=np.float64)
        if self.scorer is not None:
            scores = np.asarray(self.scorer(self.mask_net, observations), dtype=np.float64)
            if scores.shape[0] == len(observations):
                return scores
        if self.mask_net is not None and _score_trajectory is not None:
            try:
                return np.asarray(_score_trajectory(self.mask_net, observations), dtype=np.float64)
            except Exception:  # pragma: no cover - fall through to random
                pass
        return random_importance_scores(len(observations), self.rng)

    # ---------------------------------------------------------------- rollout
    def rollout(
        self,
        policy: Any = None,
        seed: Optional[int] = None,
        length: Optional[int] = None,
        deterministic: Optional[bool] = None,
        collect_states: bool = True,
    ) -> Any:
        """Roll the frozen target policy for one (full) trajectory.

        Returns a :class:`rice.explanation.critical_state.TrajectoryRollout`.
        """
        policy = policy if policy is not None else self.policy
        length = int(length or self.horizon)
        deterministic = self.deterministic_policy if deterministic is None else deterministic
        if roll_trajectory is None:  # pragma: no cover - only without torch stack
            return self._simple_rollout(policy, seed, length, deterministic, collect_states)
        return roll_trajectory(
            self.env,
            policy,
            length=length,
            reset=True,
            deterministic=deterministic,
            seed=seed,
            collect_states=collect_states,
            stop_on_done=True,
        )

    def _simple_rollout(self, policy, seed, length, deterministic, collect_states):  # pragma: no cover
        obs, _ = self._reset_env(seed)
        observations, actions, rewards, next_observations, dones, states = [], [], [], [], [], []
        for _ in range(int(length)):
            action = policy_action(policy, obs, deterministic=deterministic)
            state = get_env_state(self.env) if collect_states else None
            next_obs, reward, terminated, truncated, _info = self._step(action)
            observations.append(obs)
            actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            rewards.append(reward)
            next_observations.append(next_obs)
            dones.append(bool(terminated or truncated))
            states.append(state)
            obs = next_obs
            if terminated or truncated:
                break
        payload = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "dones": dones,
            "states": states,
            "env_id": self.env_id,
            "length": len(observations),
            "terminated_early": bool(dones and dones[-1]),
        }
        if TrajectoryRollout is not None:
            return TrajectoryRollout(**payload)
        return _DictRollout(payload)

    # --------------------------------------------------------- window picking
    def select_window(self, importance: Sequence[float], K: float, L: Optional[int] = None) -> Tuple[int, int, float]:
        """Highest-average-importance window of width ``l = L x K`` (Sec. 4.1)."""
        importance = np.asarray(importance, dtype=np.float64).reshape(-1)
        L = int(L if L is not None else importance.shape[0])
        if L <= 0:
            return 0, 0, float("nan")
        if window_length is not None:
            l = int(window_length(L, K))
        else:  # pragma: no cover
            frac = K / 100.0 if K > 1 else K
            l = int(round(L * frac))
        l = max(1, min(l, L))
        if best_window is not None:
            try:
                start, end, score = best_window(importance, L=L, K=K)
                return int(start), int(end), float(score)
            except Exception:  # pragma: no cover
                pass
        # Local fallback: exact sliding-window mean over the valid range.
        n_windows = L - l + 1
        if n_windows <= 0:
            return 0, L, float(np.mean(importance)) if L else float("nan")
        cumsum = np.concatenate([[0.0], np.cumsum(importance)])
        means = (cumsum[l:] - cumsum[:-l]) / float(l)
        # NaN-safe: ignore all-NaN windows by treating them as -inf.
        if np.all(np.isnan(means)):
            start = 0
        else:
            start = int(np.nanargmax(means))
        return start, start + l, float(means[start])

    # ------------------------------------------------------ randomized replay
    def _prepare_at(
        self,
        baseline: Any,
        start: int,
        seed: Optional[int] = None,
        policy: Any = None,
    ) -> Tuple[Any, float, int, str]:
        """Position the env at (or just before) step ``start`` of the baseline.

        Returns ``(observation, pre_window_reward, steps_advanced, restore_mode)``.
        Direct simulator state injection is attempted first; deterministic action
        replay from a re-seeded reset is the universal fallback.
        """
        observations = list(getattr(baseline, "observations", []))
        actions = list(getattr(baseline, "actions", []))
        rewards = list(getattr(baseline, "rewards", []))
        states = list(getattr(baseline, "states", []) or [])
        start = int(start)
        policy = policy if policy is not None else self.policy

        # 1) direct simulator state injection (preferred: exact fast-forward)
        if self.restore in ("auto", "direct") and 0 <= start < len(states) and states[start] is not None:
            try:
                if set_env_state(self.env, states[start]):
                    obs = observations[start] if start < len(observations) else self.env.reset()[0]
                    return obs, float(np.sum(rewards[:start])), start, "direct"
            except Exception:  # pragma: no cover
                pass
        if self.restore == "direct":  # pragma: no cover
            obs = observations[start] if start < len(observations) else self.env.reset()[0]
            return obs, float(np.sum(rewards[:start])), start, "direct"

        # 2) deterministic action replay from a re-seeded reset (Go-Explore style)
        obs, _info = self._reset_env(seed)
        pre = 0.0
        advanced = 0
        for t in range(min(start, len(actions))):
            next_obs, reward, terminated, truncated, _info = self._step(actions[t])
            pre += reward
            advanced += 1
            obs = next_obs
            if terminated or truncated:
                break
        return obs, pre, advanced, "replay"

    def randomized_rollout(
        self,
        baseline: Any,
        start: int,
        l: int,
        seed: Optional[int] = None,
        policy: Any = None,
    ) -> Dict[str, Any]:
        """Fast-forward to ``start``, randomize ``l`` steps, then resume ``pi``."""
        policy = policy if policy is not None else self.policy
        obs, total, advanced, mode = self._prepare_at(baseline, start, seed=seed, policy=policy)
        l = max(int(l), 1)
        max_tail = self.max_tail_steps or (self.horizon * MAX_TAIL_FACTOR)
        steps = 0
        terminated = truncated = False

        # --- random (masked) window -------------------------------------------------
        for _ in range(l):
            action = self._random_action()
            obs, reward, terminated, truncated, _info = self._step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                break

        # --- resume the frozen target policy to the end of the episode --------------
        while not (terminated or truncated) and advanced + steps < max_tail:
            action = policy_action(policy, obs, deterministic=self.deterministic_policy)
            obs, reward, terminated, truncated, _info = self._step(action)
            total += reward
            steps += 1

        return {
            "return": float(total),
            "steps": int(steps),
            "restore_mode": mode,
            "terminated": bool(terminated or truncated),
        }

    # -------------------------------------------------------------- evaluation
    def evaluate_trajectory(
        self,
        K: float = 0.1,
        seed: Optional[int] = None,
        trajectory_index: int = 0,
        baseline: Any = None,
        importance: Optional[Sequence[float]] = None,
        policy: Any = None,
    ) -> TrajectoryFidelityDetail:
        """Compute the fidelity score of one trajectory for one window ratio ``K``."""
        policy = policy if policy is not None else self.policy
        baseline = baseline if baseline is not None else self.rollout(policy=policy, seed=seed)
        observations = list(getattr(baseline, "observations", []))
        rewards = list(getattr(baseline, "rewards", []))
        L = len(observations)
        detail = TrajectoryFidelityDetail(
            trajectory_index=int(trajectory_index),
            seed=seed,
            L=L,
            K=float(K),
            scoring=self.scoring,
        )
        if L == 0:
            detail.d_max = self.resolve_d_max(0.0)
            return detail

        if importance is None:
            importance = self.importance_scores(observations)
        start, end, avg_importance = self.select_window(importance, K, L=L)
        l = max(int(end - start), 1)
        R = float(np.sum(rewards))

        result = self.randomized_rollout(baseline, start, l, seed=seed, policy=policy)
        R_prime = float(result["return"])
        d_max = self.resolve_d_max(R)
        score = fidelity_score_from_change(abs(R_prime - R), d_max, l, L)

        detail.l = l
        detail.window_start = int(start)
        detail.window_end = int(end)
        detail.window_importance = float(avg_importance)
        detail.baseline_reward = R
        detail.randomized_reward = R_prime
        detail.d = abs(R_prime - R)
        detail.d_max = d_max
        detail.fidelity = score
        detail.terminated = bool(result["terminated"])
        detail.steps_taken = int(result["steps"])
        detail.restore_mode = result["restore_mode"]

        self.stats["trajectories"] += 1
        self.stats[f"restore_{detail.restore_mode}"] = self.stats.get(f"restore_{detail.restore_mode}", 0) + 1
        return detail

    def evaluate(
        self,
        K: float = 0.1,
        n_trajectories: int = DEFAULT_N_TRAJECTORIES,
        seeds: Sequence[int] = DEFAULT_SEEDS,
        progress: bool = False,
        progress_every: int = 50,
        store_details: bool = False,
    ) -> FidelityResult:
        """Run Experiment I for a single ``K``: ``n_trajectories`` x ``len(seeds)``."""
        values: List[float] = []
        per_seed: Dict[int, float] = {}
        per_seed_std: Dict[int, float] = {}
        details: List[Dict[str, Any]] = []
        window_lengths: List[int] = []
        d_max_seen: Optional[float] = None

        for seed in seeds:
            seed = int(seed)
            seed_values: List[float] = []
            for i in range(int(n_trajectories)):
                traj_seed = int(seed_from(seed, i))
                detail = self.evaluate_trajectory(K=K, seed=traj_seed, trajectory_index=i)
                detail_dict = detail.to_dict()
                seed_values.append(detail.fidelity)
                values.append(detail.fidelity)
                window_lengths.append(detail.l)
                d_max_seen = detail.d_max
                if store_details:
                    details.append(detail_dict)
                if progress and (i + 1) % max(int(progress_every), 1) == 0:
                    self.logger.info(
                        "[fidelity] env=%s K=%.0f%% seed=%d traj=%d/%d mean=%.4f",
                        self.env_id, 100 * K, seed, i + 1, n_trajectories, float(np.nanmean(seed_values)),
                    )
            per_seed[seed] = float(np.nanmean(seed_values)) if seed_values else float("nan")
            per_seed_std[seed] = float(np.nanstd(seed_values)) if seed_values else float("nan")

        if mean_std is not None:
            m, s = mean_std(values)
        else:  # pragma: no cover
            arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
            m, s = (float(arr.mean()), float(arr.std())) if arr.size else (float("nan"), float("nan"))

        return FidelityResult(
            env_id=self.env_id,
            K=float(K),
            scoring=self.scoring,
            n_trajectories=int(n_trajectories),
            n_seeds=len(list(seeds)),
            seeds=tuple(int(x) for x in seeds),
            mean=float(m),
            std=float(s),
            values=values,
            per_seed=per_seed,
            per_seed_std=per_seed_std,
            d_max=d_max_seen,
            mean_window_length=float(np.mean(window_lengths)) if window_lengths else float("nan"),
            details=details,
        )

    def evaluate_multi_K(
        self,
        K_values: Sequence[float] = DEFAULT_K_VALUES,
        n_trajectories: int = DEFAULT_N_TRAJECTORIES,
        seeds: Sequence[int] = DEFAULT_SEEDS,
        progress: bool = False,
        store_details: bool = False,
    ) -> Dict[float, FidelityResult]:
        """Experiment-I sweep over ``K in {10%, 20%, 30%, 40%}``."""
        return {
            float(K): self.evaluate(
                K=float(K),
                n_trajectories=n_trajectories,
                seeds=seeds,
                progress=progress,
                store_details=store_details,
            )
            for K in K_values
        }

    # Convenience alias kept for symmetry with other modules.
    score = importance_scores


# --------------------------------------------------------------------------------------
# Lightweight rollout container used when the critical_state module is unavailable
# --------------------------------------------------------------------------------------
class _DictRollout:  # pragma: no cover - only used as a fallback
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.__dict__.update(payload)

    def __len__(self) -> int:
        return len(self.observations)


# --------------------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------------------
def evaluate_fidelity(
    env: Any,
    policy: Any,
    mask_net: Any = None,
    K: float = 0.1,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    env_id: Optional[str] = None,
    d_max: Optional[float] = None,
    scoring: str = "auto",
    deterministic_policy: bool = True,
    progress: bool = False,
    store_details: bool = False,
    logger: Any = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> FidelityResult:
    """One-shot Experiment-I fidelity evaluation for a single ``K``."""
    evaluator = FidelityEvaluator(
        env=env,
        policy=policy,
        mask_net=mask_net,
        env_id=env_id,
        d_max=d_max,
        scoring=scoring,
        deterministic_policy=deterministic_policy,
        logger=logger,
        seed=seed,
        **kwargs,
    )
    return evaluator.evaluate(
        K=K, n_trajectories=n_trajectories, seeds=seeds, progress=progress, store_details=store_details
    )


def evaluate_fidelity_multi_K(
    env: Any,
    policy: Any,
    mask_net: Any = None,
    K_values: Sequence[float] = DEFAULT_K_VALUES,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    env_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[float, FidelityResult]:
    """Experiment-I sweep over all ``K`` values for one explanation method."""
    evaluator = FidelityEvaluator(env=env, policy=policy, mask_net=mask_net, env_id=env_id, **kwargs)
    return evaluator.evaluate_multi_K(
        K_values=K_values, n_trajectories=n_trajectories, seeds=seeds, progress=kwargs.get("progress", False)
    )


def evaluate_methods(
    env: Any,
    policy: Any,
    explanations: Dict[str, Any],
    K_values: Sequence[float] = DEFAULT_K_VALUES,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    env_id: Optional[str] = None,
    include_random: bool = False,
    progress: bool = False,
    **kwargs: Any,
) -> Dict[str, Dict[float, FidelityResult]]:
    """Evaluate several explanation methods (Ours / StateMask / Random).

    ``explanations`` maps a method name to a mask network (or ``None``, which
    yields the Random baseline).  ``include_random`` additionally evaluates the
    Random baseline with an independent random window draw.
    """
    results: Dict[str, Dict[float, FidelityResult]] = {}
    for name, mask_net in explanations.items():
        evaluator = FidelityEvaluator(
            env=env, policy=policy, mask_net=mask_net, env_id=env_id, logger=kwargs.pop("logger", None)
        )
        results[name] = evaluator.evaluate_multi_K(
            K_values=K_values, n_trajectories=n_trajectories, seeds=seeds, progress=progress
        )
    if include_random and "Random" not in results:
        evaluator = FidelityEvaluator(env=env, policy=policy, mask_net=None, env_id=env_id, scoring="random")
        results["Random"] = evaluator.evaluate_multi_K(
            K_values=K_values, n_trajectories=n_trajectories, seeds=seeds, progress=progress
        )
    return results


def training_time_reduction(
    baseline_times: Dict[str, float],
    ours_times: Dict[str, float],
) -> Dict[str, Any]:
    """Table-4 efficiency comparison: mask-net training time under a fixed budget.

    The paper reports that RICE's mask network takes ~16.8% less training time
    than StateMask's for a fixed number of training samples.  Returns per-env
    reductions plus the overall mean reduction (in percent).
    """
    per_env: Dict[str, Dict[str, float]] = {}
    reductions: List[float] = []
    for env_id, baseline in baseline_times.items():
        ours = ours_times.get(env_id)
        if ours is None or baseline <= 0:
            continue
        reduction = 100.0 * (float(baseline) - float(ours)) / float(baseline)
        per_env[env_id] = {"baseline": float(baseline), "ours": float(ours), "reduction_percent": reduction}
        reductions.append(reduction)
    mean_reduction = float(np.mean(reductions)) if reductions else float("nan")
    return {"per_env": per_env, "mean_reduction_percent": mean_reduction}


def format_fidelity_table(
    results: Dict[str, Dict[float, FidelityResult]],
    decimals: int = 3,
) -> Dict[str, Dict[str, str]]:
    """Render ``{method: {K: FidelityResult}}`` as ``{method: {K: 'mean +- std'}}``."""
    table: Dict[str, Dict[str, str]] = {}
    for method, per_k in results.items():
        table[method] = {
            f"{100 * float(K):.0f}%": f"{res.mean:.{decimals}f} +- {res.std:.{decimals}f}"
            for K, res in per_K.items()
        }
    return table
