"""Random explanation baseline for RICE (Stage 1).

Paper reference (Cheng et al., ICML 2024, *RICE: A Refining Scheme for
Reinforcement Learning with Explanation*), Section 4.1 "Baseline Explanation
Methods":

    "Additionally, we introduce 'Random' as a baseline explanation method.
     'Random' identifies critical steps by randomly selecting a visited state
     as the critical state."

In other words the Random explainer attaches an *uninformative* importance
score to every visited state and therefore returns a uniformly random state of
a trajectory as the "critical" state used to build the mixed initial state
distribution ``mu(s) = beta * d_rho^pihat(s) + (1 - beta) * rho(s)``
(Section 3.3 / Algorithm 2).  It is the lower-bound reference for
Experiment III ("fix refine = Ours, vary explanation in {Random, StateMask,
Ours}") and for the fidelity metric of Experiment I (Section 4.1).  C.3
confirms the expectation that the mask-network explanations (Ours / StateMask)
achieve *higher* fidelity than the Random explanation across all applications.

This module provides

* :func:`random_importance_scores` -- ``n`` i.i.d. uniform scores, the exact
  helper the fidelity evaluator uses for its ``"random"`` scoring branch;
* :class:`RandomImportanceScorer` -- a drop-in replacement for
  ``rice.explanation.importance.ImportanceScorer`` (same method surface:
  ``score``, ``score_trajectory``, ``most_important_index``,
  ``top_k_states``, ``trajectory_aggregate``, ``attach_to_wrapper``, ...);
* :class:`RandomStateSelector` -- a drop-in replacement for
  ``rice.explanation.critical_state.CriticalStateSelector`` that returns a
  uniformly random visited state instead of the mask-argmax state;
* :class:`RandomMaskNetwork` -- an optional duck-typed stand-in for
  ``MaskNetwork`` (random ``P(keep)``) so code paths that insist on a
  mask-net object can still run with the Random explanation;
* :class:`RandomExplanation` -- the umbrella object wiring the three together,
  which is what the experiment drivers / :func:`make_random_explanation`
  return.

Design notes
------------
* ``RandomExplanation.mask_net`` is ``None`` by default.  The rest of the
  code base treats ``mask_net=None`` as "uninformative / random importance"
  (see ``importance.score_observations`` and ``PPORefiner``), so the Random
  explanation automatically selects the random branch of scoring without any
  special-casing.  ``RandomExplanation.as_mask_network()`` materialises the
  eager :class:`RandomMaskNetwork` shim when an object is required.
* All randomness flows through a single ``numpy.random.RandomState`` so the
  ``500 trajectories x 3 seeds`` protocol of Experiment I is reproducible.
* The module is dependency-light: numpy is required, everything from the rest
  of ``rice`` (importance / critical-state helpers, torch) is imported
  defensively and has a local fallback so this file always imports.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    # config / containers
    "RandomExplanationConfig",
    # scorer + selector + shim + umbrella
    "RandomImportanceScorer",
    "RandomStateSelector",
    "RandomMaskNetwork",
    "RandomExplanation",
    # factories
    "make_random_explanation",
    "build_random_explanation",
    "random_explanation_for",
    # scoring helpers
    "random_importance_scores",
    "random_explanation_scores",
    "random_index",
    "random_indices",
    "random_critical_index",
    # critical-state helpers
    "random_critical_state",
    "identify_random_state",
    "select_random_states",
    "random_top_k_states",
    "describe_random_explanation",
    # constants
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SCORE_RANGE",
    "IMPORTANCE_MODES",
    "RANDOM_MODES",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE: int = 4096
DEFAULT_SCORE_RANGE: Tuple[float, float] = (0.0, 1.0)
RANDOM_MODES: Tuple[str, ...] = ("uniform", "hash")
IMPORTANCE_MODES: Tuple[str, ...] = ("mean", "max", "sum", "last", "first", "topk_mean")
DEFAULT_K: int = 1000


# ---------------------------------------------------------------------------
# Defensive imports from the rest of the project
# ---------------------------------------------------------------------------
_IMPORT_WARNINGS: List[str] = []

try:  # pragma: no cover - exercised implicitly
    from rice.utils.seeding import get_rng as _get_rng

    HAS_SEEDING = True
except Exception as _exc:  # pragma: no cover
    _IMPORT_WARNINGS.append(f"rice.utils.seeding: {_exc}")
    HAS_SEEDING = False

    def _get_rng(seed: Optional[int] = None) -> np.random.RandomState:  # type: ignore
        return np.random.RandomState(seed)


try:  # pragma: no cover
    from rice.utils.logging import get_logger as _get_logger

    HAS_LOGGING = True
except Exception as _exc:  # pragma: no cover
    _IMPORT_WARNINGS.append(f"rice.utils.logging: {_exc}")
    HAS_LOGGING = False

    import logging as _logging

    _LOGGERS: Dict[str, Any] = {}

    def _get_logger(name: str = "rice", out_dir: Optional[str] = None, level: Any = None):  # type: ignore
        logger = _LOGGERS.get(name)
        if logger is None:
            logger = _logging.getLogger(name)
            if not logger.handlers:
                handler = _logging.StreamHandler()
                handler.setFormatter(_logging.Formatter("[%(asctime)s] %(name)s: %(message)s"))
                logger.addHandler(handler)
            logger.setLevel(level if level is not None else _logging.INFO)
            _LOGGERS[name] = logger
        return logger


# ---- importance helpers ---------------------------------------------------
try:  # pragma: no cover
    from rice.explanation.importance import (  # type: ignore
        DEFAULT_BATCH_SIZE as _imp_batch_size,
        IMPORTANCE_MODES as _imp_modes,
        TrajectoryImportance as _imp_TrajectoryImportance,
        argmax_importance as _imp_argmax_importance,
        attach_scores_to_wrapper as _imp_attach_scores,
        extract_observations as _imp_extract_observations,
        rank_states as _imp_rank_states,
        summarize_importance as _imp_summarize_importance,
        top_k_indices as _imp_top_k_indices,
    )

    HAS_IMPORTANCE = True
except Exception as _exc:  # pragma: no cover
    _IMPORT_WARNINGS.append(f"rice.explanation.importance: {_exc}")
    HAS_IMPORTANCE = False
    _imp_batch_size = DEFAULT_BATCH_SIZE
    _imp_modes = IMPORTANCE_MODES
    _imp_TrajectoryImportance = None
    _imp_argmax_importance = None
    _imp_attach_scores = None
    _imp_extract_observations = None
    _imp_rank_states = None
    _imp_summarize_importance = None
    _imp_top_k_indices = None


# ---- critical-state helpers ----------------------------------------------
try:  # pragma: no cover
    from rice.explanation.critical_state import (  # type: ignore
        CriticalState as _cs_CriticalState,
        CriticalStateSelector as _cs_CriticalStateSelector,
        TrajectoryRollout as _cs_TrajectoryRollout,
        attach_critical_state as _cs_attach_critical_state,
        default_k as _cs_default_k,
        policy_action as _cs_policy_action,
        roll_trajectory as _cs_roll_trajectory,
    )

    HAS_CRITICAL_STATE = True
except Exception as _exc:  # pragma: no cover
    _IMPORT_WARNINGS.append(f"rice.explanation.critical_state: {_exc}")
    HAS_CRITICAL_STATE = False
    _cs_CriticalState = None
    _cs_CriticalStateSelector = None
    _cs_TrajectoryRollout = None
    _cs_attach_critical_state = None
    _cs_default_k = None
    _cs_policy_action = None
    _cs_roll_trajectory = None


# ---- env helpers (optional, only used for state snapshots) ---------------
try:  # pragma: no cover
    from rice.envs.reset_wrapper import get_env_state as _get_env_state

    HAS_RESET_WRAPPER = True
except Exception as _exc:  # pragma: no cover
    HAS_RESET_WRAPPER = False
    _get_env_state = None


# ===========================================================================
# Local fallbacks (used only when the real project modules are unavailable)
# ===========================================================================
@dataclass
class _FallbackTrajectoryImportance:
    """Minimal stand-in for ``rice.explanation.importance.TrajectoryImportance``."""

    scores: np.ndarray
    observations: Any = None
    indices: Optional[np.ndarray] = None
    mode: str = "mean"
    aggregate: float = float("nan")
    critical_index: int = 0

    def __len__(self) -> int:
        return int(len(self.scores))

    def top_k(self, k: int = 10) -> np.ndarray:
        return _local_top_k_indices(self.scores, k=k)

    def to_dict(self, include_scores: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": self.mode,
            "aggregate": float(self.aggregate),
            "critical_index": int(self.critical_index),
            "length": len(self),
        }
        if include_scores:
            payload["scores"] = np.asarray(self.scores, dtype=float).tolist()
        return payload


def _local_argmax_importance(scores: Sequence[float]) -> int:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0
    arr = np.where(np.isnan(arr), -np.inf, arr)
    return int(np.argmax(arr))


def _local_rank_states(scores: Sequence[float], descending: bool = True) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-arr, kind="mergesort") if descending else np.argsort(arr, kind="mergesort")
    return order.astype(int)


def _local_top_k_indices(scores: Sequence[float], k: int = 10) -> np.ndarray:
    order = _local_rank_states(scores, descending=True)
    if k is None or k < 0:
        return order
    return order[: int(k)]


def _local_min_max_normalize(scores: Sequence[float], eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < eps:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo + eps)


def _local_summarize_importance(scores: Sequence[float], threshold: Optional[float] = None) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    out: Dict[str, float] = {
        "n": float(arr.size),
        "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        "std": float(np.nanstd(arr)) if arr.size else float("nan"),
        "min": float(np.nanmin(arr)) if arr.size else float("nan"),
        "max": float(np.nanmax(arr)) if arr.size else float("nan"),
        "median": float(np.nanmedian(arr)) if arr.size else float("nan"),
        "argmax": float(_local_argmax_importance(arr)),
    }
    if threshold is not None:
        out["threshold"] = float(threshold)
        out["above_threshold"] = float(np.sum(arr >= threshold)) if arr.size else 0.0
    return out


def _local_extract_observations(trajectory: Any) -> np.ndarray:
    """Best-effort observation extraction from a trajectory-like container."""
    if trajectory is None:
        return np.zeros((0,), dtype=np.float32)
    for attr in ("observations", "obs", "observations_"):
        value = getattr(trajectory, attr, None)
        if value is not None:
            return np.asarray(value)
    if isinstance(trajectory, dict):
        for key in ("observations", "obs"):
            if key in trajectory:
                return np.asarray(trajectory[key])
    if isinstance(trajectory, np.ndarray):
        return trajectory
    try:
        return np.asarray(list(trajectory))
    except Exception:
        return np.zeros((0,), dtype=np.float32)


def _local_attach_scores_to_wrapper(wrapper: Any, scores: Sequence[float]) -> Any:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    try:
        snapshots = getattr(wrapper, "snapshots", None)
        if snapshots is not None:
            n = min(len(snapshots), arr.size)
            for i in range(n):
                try:
                    snapshots[i].score = float(arr[i])
                except Exception:
                    pass
        setattr(wrapper, "last_importance_scores", arr)
    except Exception:
        pass
    return wrapper


@dataclass
class _FallbackCriticalState:
    """Minimal stand-in for ``rice.explanation.critical_state.CriticalState``."""

    index: int = 0
    observation: Any = None
    state: Any = None
    actions: List[Any] = field(default_factory=list)
    score: float = float("nan")
    importance_scores: Optional[np.ndarray] = None
    trajectory_length: int = 0
    env_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def restore_payload(self) -> Any:
        if self.state is None:
            return None
        kind = "none"
        if isinstance(self.state, dict):
            kind = str(self.state.get("kind", "attr"))
        return {"kind": kind, "state": self.state}

    def to_dict(self, include_scores: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "index": int(self.index),
            "score": float(self.score),
            "trajectory_length": int(self.trajectory_length),
            "env_id": self.env_id,
            "metadata": dict(self.metadata),
        }
        if include_scores and self.importance_scores is not None:
            payload["importance_scores"] = np.asarray(self.importance_scores, dtype=float).tolist()
        return payload


def _local_policy_action(policy: Any, observation: Any, deterministic: bool = False) -> np.ndarray:
    if policy is None:
        raise ValueError("A policy is required to obtain an action.")
    if hasattr(policy, "predict"):
        try:
            action, _ = policy.predict(observation, deterministic=deterministic)
            return np.asarray(action)
        except TypeError:
            action, _ = policy.predict(observation)
            return np.asarray(action)
    if hasattr(policy, "act"):
        return np.asarray(policy.act(observation, deterministic=deterministic))
    return np.asarray(policy(observation))


def _local_unpack_step(result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    if not isinstance(result, (tuple, list)):
        raise TypeError(f"Unsupported step result type: {type(result)!r}")
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), False, dict(info or {})
    raise ValueError(f"Unsupported step tuple of length {len(result)}")


def _local_unpack_reset(result: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(result, (tuple, list)) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result, {}


def _local_default_k(env: Any = None, fallback: int = DEFAULT_K) -> int:
    for attr in ("rice_max_episode_steps", "_max_episode_steps", "max_episode_steps", "_max_steps"):
        value = getattr(env, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    spec = getattr(env, "spec", None)
    if spec is not None:
        value = getattr(spec, "max_episode_steps", None)
        if value:
            return int(value)
    inner = getattr(env, "env", None)
    if inner is not None and inner is not env:
        return _local_default_k(inner, fallback=fallback)
    return int(fallback)


@dataclass
class _LocalRollout:
    """Fallback rollout container mirroring ``TrajectoryRollout``."""

    observations: Any = None
    actions: Any = None
    rewards: Any = None
    next_observations: Any = None
    dones: Any = None
    infos: List[Dict[str, Any]] = field(default_factory=list)
    states: List[Any] = field(default_factory=list)
    env_id: str = "default"
    length: int = 0
    terminated_early: bool = False

    def __len__(self) -> int:
        try:
            return int(len(self.observations))
        except Exception:
            return int(self.length)

    @property
    def importance_observations(self) -> Any:
        return self.observations

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        payload = {
            "env_id": self.env_id,
            "length": int(self.length),
            "terminated_early": bool(self.terminated_early),
        }
        if include_arrays:
            payload["observations"] = np.asarray(self.observations).tolist()
            payload["actions"] = np.asarray(self.actions).tolist()
            payload["rewards"] = np.asarray(self.rewards, dtype=float).tolist()
        return payload


def _local_roll_trajectory(
    env: Any,
    policy: Any,
    length: Optional[int] = None,
    reset: bool = True,
    deterministic: bool = False,
    seed: Optional[int] = None,
    collect_states: bool = True,
    stop_on_done: bool = True,
) -> Any:
    """Compact rollout used only when ``critical_state.roll_trajectory`` is missing."""
    K = int(length) if length else _local_default_k(env)
    if reset:
        if seed is not None and hasattr(env, "reset"):
            try:
                obs, info = _local_unpack_reset(env.reset(seed=seed))
            except TypeError:
                obs, info = _local_unpack_reset(env.reset())
        else:
            obs, info = _local_unpack_reset(env.reset())
    else:
        obs, info = None, {}

    observations: List[Any] = []
    actions: List[Any] = []
    rewards: List[float] = []
    next_observations: List[Any] = []
    dones: List[bool] = []
    infos: List[Dict[str, Any]] = []
    states: List[Any] = []
    terminated_early = False

    current_obs = obs
    steps = 0
    for _ in range(K):
        if current_obs is None:
            break
        observations.append(np.asarray(current_obs).copy())
        if collect_states and HAS_RESET_WRAPPER and _get_env_state is not None:
            try:
                states.append(_get_env_state(env))
            except Exception:
                states.append(None)
        try:
            action = _local_policy_action(policy, current_obs, deterministic=deterministic)
        except Exception:
            action = np.zeros((0,))
        actions.append(np.asarray(action))
        try:
            step_result = env.step(action)
        except Exception:
            break
        next_obs, reward, terminated, truncated, step_info = _local_unpack_step(step_result)
        rewards.append(float(reward))
        next_observations.append(np.asarray(next_obs).copy())
        dones.append(bool(terminated or truncated))
        infos.append(step_info)
        steps += 1
        current_obs = next_obs
        if (terminated or truncated) and stop_on_done:
            terminated_early = True
            break

    payload = {
        "observations": np.asarray(observations) if observations else np.zeros((0,), dtype=np.float32),
        "actions": np.asarray(actions) if actions else np.zeros((0,), dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(next_observations)
        if next_observations
        else np.zeros((0,), dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "infos": infos,
        "states": states,
        "env_id": getattr(env, "rice_env_key", "default"),
        "length": steps,
        "terminated_early": terminated_early,
    }
    if HAS_CRITICAL_STATE and _cs_TrajectoryRollout is not None:
        try:
            return _cs_TrajectoryRollout(**payload)
        except Exception:
            pass
    return _LocalRollout(**payload)


# Resolve the public names used internally.
DEFAULT_BATCH_SIZE = int(_imp_batch_size) if HAS_IMPORTANCE else DEFAULT_BATCH_SIZE
IMPORTANCE_MODES = tuple(_imp_modes) if HAS_IMPORTANCE else IMPORTANCE_MODES
TrajectoryImportance = _imp_TrajectoryImportance if HAS_IMPORTANCE else _FallbackTrajectoryImportance
CriticalState = _cs_CriticalState if HAS_CRITICAL_STATE else _FallbackCriticalState

argmax_importance = _imp_argmax_importance if HAS_IMPORTANCE else _local_argmax_importance
rank_states = _imp_rank_states if HAS_IMPORTANCE else _local_rank_states
top_k_indices = _imp_top_k_indices if HAS_IMPORTANCE else _local_top_k_indices
summarize_importance = _imp_summarize_importance if HAS_IMPORTANCE else _local_summarize_importance
extract_observations = _imp_extract_observations if HAS_IMPORTANCE else _local_extract_observations
attach_scores_to_wrapper = _imp_attach_scores if HAS_IMPORTANCE else _local_attach_scores_to_wrapper
min_max_normalize = _local_min_max_normalize

policy_action = _cs_policy_action if HAS_CRITICAL_STATE else _local_policy_action
roll_trajectory = _cs_roll_trajectory if HAS_CRITICAL_STATE else _local_roll_trajectory
default_k = _cs_default_k if HAS_CRITICAL_STATE else _local_default_k
attach_critical_state = _cs_attach_critical_state

_unpack_step = _local_unpack_step
_unpack_reset = _local_unpack_reset


# ===========================================================================
# Small RNG helpers (support RandomState and Generator)
# ===========================================================================
def _as_rng(rng: Any = None, seed: Optional[int] = None) -> Any:
    """Return a usable RNG (``RandomState``/``Generator``); never ``None``."""
    if rng is not None:
        return rng
    try:
        return _get_rng(seed)
    except Exception:
        return np.random.RandomState(seed)


def _rng_uniform(rng: Any, size: Any = None) -> np.ndarray:
    """Draws from ``U(0, 1)`` for both ``RandomState`` and ``Generator``."""
    try:
        return np.asarray(rng.uniform(0.0, 1.0, size), dtype=float)
    except TypeError:
        return np.asarray(rng.random_sample(size), dtype=float)


def _rng_randint(rng: Any, low: int, high: int, size: Any = None) -> np.ndarray:
    """Draws integers in ``[low, high)`` for both RNG flavours."""
    if hasattr(rng, "integers"):
        return np.asarray(rng.integers(low, high, size=size), dtype=int)
    return np.asarray(rng.randint(low, high, size=size), dtype=int)


def _count_states(states: Any, default: Optional[int] = None) -> int:
    """Number of states implied by a container (array/list/scalar/None)."""
    if states is None:
        return int(default or 0)
    if isinstance(states, (int, np.integer)):
        return int(states)
    if isinstance(states, float):
        return int(states)
    try:
        return int(len(states))
    except TypeError:
        arr = np.asarray(states)
        return int(arr.size) if arr.ndim == 0 else int(len(arr))


def _is_torch_tensor(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("torch") and hasattr(value, "detach")


def _batch_shape(observations: Any) -> Tuple[int, ...]:
    """Leading (batch) shape of an observation container."""
    if _is_torch_tensor(observations):
        shape = tuple(observations.shape)
        return tuple(shape[:-1]) if len(shape) > 1 else ()
    arr = np.asarray(observations)
    return tuple(arr.shape[:-1]) if arr.ndim > 1 else ()


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class RandomExplanationConfig:
    """Configuration for the Random explanation baseline.

    Parameters
    ----------
    env_id:
        Application / gym id the explanation is used for (bookkeeping only).
    seed:
        Seed used when no external RNG is supplied.
    mode:
        ``"uniform"`` -- independent uniform score per visited state;
        ``"hash"``    -- deterministic pseudo-score derived from the state
        index (still uninformative, but identical across calls).
    score_range:
        ``(low, high)`` for the uniform scores.
    batch_size:
        Chunk size kept for interface parity with ``ImportanceScorer``
        (unused by the random scorer).
    topk_fraction:
        Fraction used by the ``"topk_mean"`` aggregation mode.
    """

    env_id: str = "default"
    seed: Optional[int] = None
    mode: str = "uniform"
    score_range: Tuple[float, float] = DEFAULT_SCORE_RANGE
    batch_size: int = DEFAULT_BATCH_SIZE
    topk_fraction: float = 0.1
    deterministic_tie_break: bool = True
    attach_scores: bool = True
    with_mask_shim: bool = False

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in RANDOM_MODES:
            self.mode = "uniform"
        try:
            low, high = float(self.score_range[0]), float(self.score_range[1])
        except Exception:
            low, high = DEFAULT_SCORE_RANGE
        self.score_range = (low, high)
        if self.topk_fraction <= 0:
            self.topk_fraction = 0.1

    # -- (de)serialisation -------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "RandomExplanationConfig":
        """Build a config from a (possibly nested) YAML-derived dict."""
        data: Dict[str, Any] = {}
        if isinstance(cfg, dict):
            for section in ("random", "explanation", "baseline", "random_explanation"):
                sub = cfg.get(section)
                if isinstance(sub, dict):
                    data.update(sub)
            for key in (
                "env_id",
                "seed",
                "mode",
                "score_range",
                "batch_size",
                "topk_fraction",
                "deterministic_tie_break",
                "attach_scores",
                "with_mask_shim",
            ):
                if key in cfg and not isinstance(cfg[key], dict):
                    data[key] = cfg[key]
            if "range" in cfg and "score_range" not in data:
                data["score_range"] = cfg["range"]
            if "random_mode" in cfg and "mode" not in data:
                data["mode"] = cfg["random_mode"]
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k in allowed and v is not None}
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "seed": self.seed,
            "mode": self.mode,
            "score_range": list(self.score_range),
            "batch_size": int(self.batch_size),
            "topk_fraction": float(self.topk_fraction),
            "deterministic_tie_break": bool(self.deterministic_tie_break),
            "attach_scores": bool(self.attach_scores),
            "with_mask_shim": bool(self.with_mask_shim),
        }


# ===========================================================================
# RandomMaskNetwork -- duck-typed MaskNetwork shim
# ===========================================================================
class RandomMaskNetwork:
    """Uninformative (random) stand-in for ``rice.explanation.MaskNetwork``.

    It exposes the same *scoring* surface as :class:`MaskNetwork`
    (``logits``, ``probabilities``, ``keep_probability``, ``score``, ...) but
    draws ``P(keep) ~ U(0, 1)`` instead of using a learnt network, i.e. exactly
    the "Random" baseline explanation of Section 4.1.  It is **not trainable**
    (``parameters()`` is empty) and is only materialised on demand by
    :meth:`RandomExplanation.as_mask_network`.
    """

    is_random: bool = True
    num_logits: int = 2
    keep_index: int = 0
    blind_index: int = 1
    env_key: str = "default"

    def __init__(
        self,
        obs_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = (64, 64),
        env_key: str = "default",
        seed: Optional[int] = None,
        rng: Any = None,
        mode: str = "uniform",
        low: float = 0.0,
        high: float = 1.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.env_key = env_key
        self.device = device
        self.mode = str(mode).lower() if str(mode).lower() in RANDOM_MODES else "uniform"
        self.low = float(low)
        self.high = float(high)
        self.rng = _as_rng(rng, seed)
        self.seed = seed
        self._call_index = 0
        self.training = False

    # -- internal ----------------------------------------------------------
    def _rand(self, observations: Any, shape: Tuple[int, ...]) -> Any:
        if _is_torch_tensor(observations):
            import torch  # local import: torch is optional for this module

            if self.mode == "hash":
                size = int(np.prod(shape)) if shape else 1
                values = np.linspace(self.low, self.high, max(size, 2))[:size]
                tensor = torch.tensor(values, dtype=torch.float32, device=observations.device)
                return tensor.reshape(shape) if shape else tensor[0]
            return torch.rand(shape, dtype=torch.float32, device=observations.device) * (
                self.high - self.low
            ) + self.low
        if self.mode == "hash":
            size = int(np.prod(shape)) if shape else 1
            values = np.linspace(self.low, self.high, max(size, 2))[:size]
            return values.reshape(shape) if shape else float(values[0])
        values = _rng_uniform(self.rng, shape if shape else None)
        if not shape:
            values = np.asarray(values).reshape(())
        return values * (self.high - self.low) + self.low

    def _keep(self, observations: Any) -> Any:
        self._call_index += 1
        return self._rand(observations, _batch_shape(observations))

    # -- scoring surface ---------------------------------------------------
    def keep_probability(self, observations: Any) -> Any:
        """``P(a_t^m = 0 | s_t)`` -- uniformly random (the baseline explanation)."""
        return self._keep(observations)

    def blind_probability(self, observations: Any) -> Any:
        return 1.0 - self._keep(observations)

    def score(self, observations: Any) -> Any:
        return self.keep_probability(observations)

    def probabilities(self, observations: Any) -> Any:
        keep = self.keep_probability(observations)
        if _is_torch_tensor(keep):
            import torch

            return torch.stack([keep, 1.0 - keep], dim=-1)
        arr = np.asarray(keep, dtype=float)
        return np.stack([arr, 1.0 - arr], axis=-1)

    def logits(self, observations: Any) -> Any:
        """Two-logit view of the random probabilities (``[0, log p/(1-p)]``)."""
        raw = self.keep_probability(observations)
        if _is_torch_tensor(raw):
            import torch

            clamped = torch.clamp(raw, 1e-6, 1 - 1e-6)
            diff = torch.log(clamped) - torch.log1p(-clamped)
            return torch.stack([torch.zeros_like(diff), diff], dim=-1)
        keep = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
        diff = np.log(keep) - np.log1p(-keep)
        zeros = np.zeros_like(diff)
        return np.stack([zeros, diff], axis=-1)

    def distribution(self, observations: Any) -> Any:
        return self.probabilities(observations)

    def forward(self, observations: Any) -> Any:  # pragma: no cover - parity only
        return self.logits(observations)

    def __call__(self, observations: Any) -> Any:  # pragma: no cover - parity only
        return self.logits(observations)

    def sample(self, observations: Any, deterministic: bool = False) -> np.ndarray:
        keep = np.asarray(self.keep_probability(observations), dtype=float)
        if deterministic:
            masks = (keep < 0.5).astype(np.int64)
        else:
            draws = _rng_uniform(self.rng, keep.shape if keep.shape else None)
            masks = (np.asarray(draws) >= keep).astype(np.int64)
        return masks

    greedy = sample

    def act(self, observations: Any, deterministic: bool = False) -> np.ndarray:
        return self.sample(observations, deterministic=deterministic)

    def evaluate_actions(self, observations: Any, masks: Any) -> Tuple[Any, Any, Any]:
        """Interface parity only -- a random mask is never optimised."""
        n = _count_states(masks)
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    # -- torch-module parity (no-op) ---------------------------------------
    def to(self, *args: Any, **kwargs: Any) -> "RandomMaskNetwork":
        return self

    def cpu(self) -> "RandomMaskNetwork":
        return self

    def eval(self) -> "RandomMaskNetwork":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "RandomMaskNetwork":
        self.training = bool(mode)
        return self

    def parameters(self, recurse: bool = True) -> List[Any]:
        return []

    def named_parameters(self, recurse: bool = True) -> List[Any]:
        return []

    def get_parameters(self) -> List[Any]:
        return []

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"random": True, "env_key": self.env_key, "mode": self.mode}

    def load_state_dict(self, state: Any, strict: bool = False) -> Any:
        return state

    def extra_repr(self) -> str:
        return f"random=True, env_key={self.env_key}, mode={self.mode}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RandomMaskNetwork({self.extra_repr()})"


# ===========================================================================
# RandomImportanceScorer
# ===========================================================================
class RandomImportanceScorer:
    """Random (uninformative) state-importance scorer.

    Drop-in replacement for :class:`rice.explanation.importance.ImportanceScorer`
    used as the "Random" explanation baseline: every visited state receives an
    i.i.d. uniform score in ``config.score_range``.

    Parameters
    ----------
    seed, rng:
        Reproducibility control (``rng`` wins when both are given).
    mode:
        ``"uniform"`` (fresh draws) or ``"hash"`` (deterministic pseudo-scores).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        rng: Any = None,
        low: Optional[float] = None,
        high: Optional[float] = None,
        mode: str = "uniform",
        batch_size: int = DEFAULT_BATCH_SIZE,
        topk_frac: float = 0.1,
        mask_net: Any = None,
        config: Optional[RandomExplanationConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            allowed = set(RandomExplanationConfig.__dataclass_fields__)  # type: ignore[attr-defined]
            config = RandomExplanationConfig.from_dict(
                kwargs.pop("cfg", None),
                seed=seed,
                mode=mode,
                batch_size=batch_size,
                topk_fraction=topk_frac,
                **{k: v for k, v in kwargs.items() if k in allowed},
            )
        self.config = config
        self.rng = _as_rng(rng, seed if seed is not None else config.seed)
        self.seed = seed
        self.low = float(config.score_range[0] if low is None else low)
        self.high = float(config.score_range[1] if high is None else high)
        self.mode = config.mode
        self.batch_size = int(config.batch_size)
        self.topk_frac = float(config.topk_fraction)
        self._mask_net = mask_net  # normally None (=> "random" scoring branch)
        self.calls = 0

    # -- properties --------------------------------------------------------
    @property
    def is_random(self) -> bool:
        return True

    @property
    def mask_net(self) -> Any:
        """Always ``None`` unless explicitly supplied: signals the uninformative
        explanation to consumers (``importance`` / ``PPORefiner``)."""
        return self._mask_net

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "kind": "random",
            "mode": self.mode,
            "score_range": [self.low, self.high],
            "calls": int(self.calls),
            "seed": self.seed,
        }

    # -- scoring -----------------------------------------------------------
    def draw(self, n: int) -> np.ndarray:
        """``n`` i.i.d. random importance scores."""
        n = int(max(0, n))
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if self.mode == "hash":
            values = np.linspace(self.low, self.high, max(n, 2))[:n]
        else:
            values = _rng_uniform(self.rng, n) * (self.high - self.low) + self.low
        return np.asarray(values, dtype=np.float32)

    def score(self, observations: Any) -> np.ndarray:
        self.calls += 1
        return self.draw(_count_states(observations))

    score_batch = score

    def score_observations(self, observations: Any) -> np.ndarray:
        return self.score(observations)

    def score_trajectory(self, trajectory: Any) -> Any:
        """Score a whole trajectory (one random score per visited state)."""
        observations = extract_observations(trajectory)
        scores = self.score(observations)
        aggregate = float(np.mean(scores)) if scores.size else float("nan")
        critical_index = int(argmax_importance(scores)) if scores.size else 0
        indices = np.arange(scores.size, dtype=int)
        try:
            return TrajectoryImportance(
                scores=scores,
                observations=observations,
                indices=indices,
                mode="mean",
                aggregate=aggregate,
                critical_index=critical_index,
            )
        except Exception:  # pragma: no cover - container mismatch
            return _FallbackTrajectoryImportance(
                scores=scores,
                observations=observations,
                indices=indices,
                mode="mean",
                aggregate=aggregate,
                critical_index=critical_index,
            )

    def most_important_index(self, trajectory: Any) -> int:
        scores = self.score_trajectory(trajectory).scores
        return int(argmax_importance(scores)) if len(scores) else 0

    def most_important_state(self, trajectory: Any) -> Tuple[int, Any]:
        idx = self.most_important_index(trajectory)
        observations = extract_observations(trajectory)
        try:
            return idx, observations[idx]
        except Exception:
            return idx, None

    def top_k_states(self, trajectory: Any, k: int = 10) -> Tuple[np.ndarray, Any]:
        scores = self.score_trajectory(trajectory).scores
        idx = top_k_indices(scores, k=k)
        observations = extract_observations(trajectory)
        try:
            return idx, observations[idx]
        except Exception:
            return idx, None

    def ranked_observations(self, trajectory: Any) -> Tuple[np.ndarray, Any]:
        scores = self.score_trajectory(trajectory).scores
        idx = rank_states(scores, descending=True)
        observations = extract_observations(trajectory)
        try:
            return idx, observations[idx]
        except Exception:
            return idx, None

    def trajectory_aggregate(self, trajectory: Any, mode: Optional[str] = None) -> float:
        scores = np.asarray(self.score_trajectory(trajectory).scores, dtype=float)
        mode = (mode or "mean").lower()
        if scores.size == 0:
            return float("nan")
        if mode == "mean":
            return float(np.mean(scores))
        if mode == "max":
            return float(np.max(scores))
        if mode == "sum":
            return float(np.sum(scores))
        if mode == "last":
            return float(scores[-1])
        if mode == "first":
            return float(scores[0])
        if mode == "topk_mean":
            k = max(1, int(round(self.topk_frac * scores.size)))
            return float(np.mean(np.sort(scores)[-k:]))
        return float(np.mean(scores))

    def summary(self, trajectory: Any) -> Dict[str, Any]:
        scores = np.asarray(self.score_trajectory(trajectory).scores, dtype=float)
        payload = summarize_importance(scores)
        payload["kind"] = "random"
        payload["mode"] = self.mode
        return payload

    def attach_to_wrapper(self, wrapper: Any, trajectory: Any = None) -> Any:
        scores = (
            np.asarray(self.score_trajectory(trajectory).scores, dtype=float)
            if trajectory is not None
            else np.asarray(getattr(wrapper, "last_importance_scores", np.zeros(0)), dtype=float)
        )
        try:
            return attach_scores_to_wrapper(wrapper, scores)
        except Exception:
            return wrapper

    def reset_statistics(self) -> None:
        self.calls = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "RandomImportanceScorer", **self.stats}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RandomImportanceScorer(mode={self.mode}, range=({self.low}, {self.high}))"


# ===========================================================================
# RandomStateSelector
# ===========================================================================
class RandomStateSelector:
    """Pick a **randomly chosen visited state** as the critical state.

    Drop-in replacement for
    ``rice.explanation.critical_state.CriticalStateSelector`` implementing the
    Random baseline: roll the frozen policy for ``K`` steps, then choose one of
    the visited states uniformly at random (Section 4.1).
    """

    def __init__(
        self,
        env: Any = None,
        mask_net: Any = None,
        policy: Any = None,
        K: Optional[int] = None,
        scorer: Optional[RandomImportanceScorer] = None,
        deterministic_policy: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        rng: Any = None,
        seed: Optional[int] = None,
        attach_scores: bool = True,
        cache: bool = False,
        config: Optional[RandomExplanationConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.env = env
        self.policy = policy
        self.K = K
        self.deterministic_policy = bool(deterministic_policy)
        self.batch_size = int(batch_size)
        self.device = device
        self.attach_scores = bool(attach_scores)
        self.cache = bool(cache)
        self.config = config or RandomExplanationConfig(seed=seed, batch_size=batch_size)
        self.rng = _as_rng(rng, seed if seed is not None else self.config.seed)
        self.seed = seed
        self.scorer = scorer or RandomImportanceScorer(rng=self.rng, config=self.config)
        self._mask_net: Any = mask_net  # normally None
        self.critical_states: List[Any] = []
        self.last_critical_state: Any = None
        self.history: List[Dict[str, Any]] = []

    # -- properties --------------------------------------------------------
    @property
    def is_random(self) -> bool:
        return True

    @property
    def mask_net(self) -> Any:
        """``None`` by default: the Random explanation is uninformative."""
        return self._mask_net

    @property
    def env_id(self) -> str:
        return getattr(self.env, "rice_env_key", self.config.env_id)

    @property
    def statistics(self) -> Dict[str, Any]:
        lengths = [h.get("trajectory_length", 0) for h in self.history]
        indices = [h.get("index", 0) for h in self.history]
        return {
            "kind": "random",
            "n_rollouts": len(self.history),
            "mean_trajectory_length": float(np.mean(lengths)) if lengths else 0.0,
            "mean_critical_index": float(np.mean(indices)) if indices else 0.0,
        }

    stats = statistics

    # -- scoring / rollouts ------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)

    def _resolve_k(self, K: Optional[int] = None) -> int:
        value = K if K is not None else self.K
        if value is None:
            return int(default_k(self.env, fallback=DEFAULT_K))
        value = float(value)
        if 0.0 < value < 1.0:
            horizon = int(default_k(self.env, fallback=DEFAULT_K))
            return max(1, int(round(value * horizon)))
        return int(value)

    def rollout(
        self,
        policy: Any = None,
        K: Optional[int] = None,
        reset: bool = True,
        seed: Optional[int] = None,
    ) -> Any:
        policy = policy if policy is not None else self.policy
        length = self._resolve_k(K)
        return roll_trajectory(
            self.env,
            policy,
            length=length,
            reset=reset,
            deterministic=self.deterministic_policy,
            seed=seed,
            collect_states=True,
        )

    # -- selection ---------------------------------------------------------
    def select(
        self,
        policy: Any = None,
        K: Optional[int] = None,
        reset: bool = True,
        seed: Optional[int] = None,
        rollout: Any = None,
    ) -> Any:
        """Return the randomly selected critical state of a fresh rollout."""
        if rollout is None:
            rollout = self.rollout(policy=policy, K=K, reset=reset, seed=seed)
        return random_critical_state(
            rollout,
            rng=self.rng,
            scorer=self.scorer,
            env_id=self.env_id,
            attach_scores=self.attach_scores,
            env=self.env,
        )

    def select_top_k(
        self,
        k: int = 10,
        policy: Any = None,
        K: Optional[int] = None,
        reset: bool = True,
        seed: Optional[int] = None,
        rollout: Any = None,
    ) -> List[Any]:
        if rollout is None:
            rollout = self.rollout(policy=policy, K=K, reset=reset, seed=seed)
        return random_top_k_states(
            rollout,
            k=k,
            rng=self.rng,
            env_id=self.env_id,
            attach_scores=self.attach_scores,
            env=self.env,
        )

    def record(self, critical: Any, rollout: Any = None) -> Any:
        self.history.append(
            {
                "index": int(getattr(critical, "index", 0)),
                "trajectory_length": int(getattr(critical, "trajectory_length", 0)),
                "score": float(getattr(critical, "score", float("nan"))),
            }
        )
        self.last_critical_state = critical
        self.critical_states.append(critical)
        if self.cache and self.env is not None and attach_critical_state is not None:
            try:
                attach_critical_state(self.env, critical)
            except Exception:
                pass
        return critical

    # -- restore -----------------------------------------------------------
    def reset_to(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        env = env if env is not None else self.env
        critical = critical if critical is not None else self.last_critical_state
        if env is None or critical is None:
            raise ValueError("reset_to requires both an env and a critical state.")
        payload = getattr(critical, "restore_payload", None)
        if payload is not None and hasattr(env, "reset_to_state"):
            try:
                return _unpack_reset(env.reset_to_state(payload, **kwargs))
            except Exception:
                pass
        for method in ("reset_to", "reset_to_critical"):
            fn = getattr(env, method, None)
            if callable(fn):
                try:
                    return _unpack_reset(fn(critical, **kwargs))
                except TypeError:
                    try:
                        return _unpack_reset(fn(critical))
                    except Exception:
                        continue
                except Exception:
                    continue
        return _unpack_reset(env.reset())

    def reset_to_critical(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.reset_to(env=env, critical=critical, **kwargs)

    def summary(self) -> Dict[str, Any]:
        payload = self.statistics
        payload["scorer"] = self.scorer.stats
        return payload

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RandomStateSelector(env_id={self.env_id!r}, K={self.K})"


# ===========================================================================
# RandomExplanation -- umbrella object
# ===========================================================================
class RandomExplanation:
    """The "Random" explanation baseline, packaged for the RICE pipelines.

    Usage::

        explanation = make_random_explanation(env, policy, env_id="hopper", seed=7)
        critical = explanation.identify(K=None)          # random visited state
        scores = explanation.score(rollout)              # uniform scores

        # Stage 2 / fidelity: the Random explanation is the uninformative one
        refiner = PPORefiner(env, policy, mask_net=explanation.mask_net)  # None
        result = FidelityEvaluator(env, policy, scorer=explanation.scorer)
    """

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        env_id: str = "default",
        seed: Optional[int] = None,
        config: Optional[RandomExplanationConfig] = None,
        rng: Any = None,
        K: Optional[int] = None,
        deterministic_policy: bool = False,
        with_mask_shim: Optional[bool] = None,
        scorer: Optional[RandomImportanceScorer] = None,
        selector: Optional[RandomStateSelector] = None,
        **kwargs: Any,
    ) -> None:
        allowed = set(RandomExplanationConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        self.env = env
        self.policy = policy
        self.config = config or RandomExplanationConfig.from_dict(
            kwargs.pop("cfg", None),
            env_id=env_id if env_id != "default" else None,
            seed=seed,
            **{k: v for k, v in kwargs.items() if k in allowed},
        )
        if env_id != "default":
            self.config.env_id = env_id
        self.env_id = env_id or self.config.env_id
        self.seed = seed if seed is not None else self.config.seed
        self.rng = _as_rng(rng, self.seed)
        self.scorer = scorer or RandomImportanceScorer(rng=self.rng, config=self.config, seed=self.seed)
        self.selector = selector or RandomStateSelector(
            env=env,
            policy=policy,
            K=K,
            scorer=self.scorer,
            deterministic_policy=deterministic_policy,
            rng=self.rng,
            seed=self.seed,
            config=self.config,
        )
        self._mask_net: Any = None
        self._shim_net: Optional[RandomMaskNetwork] = None
        want_shim = self.config.with_mask_shim if with_mask_shim is None else bool(with_mask_shim)
        if want_shim:
            self.as_mask_network()

    # -- properties --------------------------------------------------------
    @property
    def is_random(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Random"

    @property
    def mask_net(self) -> Any:
        """``None`` (uninformative) unless a mask shim was requested."""
        return self._mask_net

    @property
    def mask_network(self) -> Any:
        return self.mask_net

    @property
    def statistics(self) -> Dict[str, Any]:
        payload = {
            "kind": "random",
            "name": "Random",
            "env_id": self.env_id,
            "scorer": self.scorer.stats,
        }
        payload.update({f"selector/{k}": v for k, v in self.selector.statistics.items()})
        return payload

    stats = statistics

    # -- mask shim ---------------------------------------------------------
    def as_mask_network(self, **kwargs: Any) -> RandomMaskNetwork:
        """Materialise (and cache) the random ``MaskNetwork`` shim."""
        if self._shim_net is None:
            self._shim_net = RandomMaskNetwork(
                obs_dim=kwargs.pop("obs_dim", getattr(self.env, "rice_obs_dim", None)),
                env_key=self.env_id,
                seed=self.seed,
                rng=self.rng,
                mode=self.config.mode,
                low=self.config.score_range[0],
                high=self.config.score_range[1],
                **kwargs,
            )
            self._mask_net = self._shim_net
        return self._shim_net

    # -- explanation API ---------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)

    importance_scores = score

    def score_trajectory(self, trajectory: Any) -> Any:
        return self.scorer.score_trajectory(trajectory)

    def rank(self, trajectory: Any) -> np.ndarray:
        scores = np.asarray(self.score_trajectory(trajectory).scores, dtype=float)
        return rank_states(scores, descending=True)

    def top_k_states(self, trajectory: Any, k: int = 10) -> Tuple[np.ndarray, Any]:
        return self.scorer.top_k_states(trajectory, k=k)

    def rollout(self, policy: Any = None, K: Optional[int] = None, **kwargs: Any) -> Any:
        return self.selector.rollout(policy=policy, K=K, **kwargs)

    def select(self, policy: Any = None, K: Optional[int] = None, **kwargs: Any) -> Any:
        critical = self.selector.select(policy=policy, K=K, **kwargs)
        return self.selector.record(critical)

    def select_top_k(self, k: int = 10, policy: Any = None, K: Optional[int] = None, **kwargs: Any) -> List[Any]:
        """Top-``k`` frontier states -- with random scores these are random states."""
        rollout = kwargs.pop("rollout", None)
        if rollout is None:
            rollout = self.rollout(policy=policy, K=K, **kwargs)
        return random_top_k_states(
            rollout,
            k=k,
            rng=self.rng,
            env_id=self.env_id,
            attach_scores=self.config.attach_scores,
            env=self.env,
        )

    def identify(self, env: Any = None, policy: Any = None, K: Optional[int] = None, **kwargs: Any) -> Any:
        """Convenience wrapper around :func:`identify_random_state`."""
        env = env if env is not None else self.env
        policy = policy if policy is not None else self.policy
        return identify_random_state(
            env,
            policy,
            K=K,
            rng=self.rng,
            env_id=self.env_id,
            deterministic=self.selector.deterministic_policy,
            attach_scores=self.config.attach_scores,
            **kwargs,
        )

    identify_critical_state = identify

    def reset_to(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.selector.reset_to(env=env, critical=critical, **kwargs)

    def attach_to_wrapper(self, wrapper: Any, trajectory: Any = None) -> Any:
        return self.scorer.attach_to_wrapper(wrapper, trajectory=trajectory)

    def summary(self) -> Dict[str, Any]:
        payload = self.statistics
        payload["config"] = self.config.to_dict()
        return payload

    def __call__(self, trajectory: Any) -> np.ndarray:
        """Calling the explainer scores a trajectory (handy for ``map``/lambda glue)."""
        return self.scorer.score_trajectory(trajectory).scores

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RandomExplanation(env_id={self.env_id!r}, mode={self.config.mode})"


# ===========================================================================
# Module-level helper functions
# ===========================================================================
def random_importance_scores(
    n: Any = None,
    rng: Any = None,
    low: float = 0.0,
    high: float = 1.0,
    seed: Optional[int] = None,
    size: Optional[int] = None,
    scores: Any = None,
    uniform: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """``n`` i.i.d. uniform importance scores in ``[low, high)``.

    Signature-compatible with the fidelity evaluator's helper
    (``random_importance_scores(n, rng=None)``); also accepts a container as
    ``n`` (e.g. a trajectory's observations), in which case its length is used.
    """
    if size is not None and n is None:
        n = size
    count = _count_states(n, default=0)
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    local_rng = _as_rng(rng, seed)
    values = _rng_uniform(local_rng, count)
    return np.asarray(values * (float(high) - float(low)) + float(low), dtype=np.float32)


# Alias used by the experiment drivers / fidelity module.
random_explanation_scores = random_importance_scores


def random_index(n: Any = None, rng: Any = None, seed: Optional[int] = None, **kwargs: Any) -> int:
    """Uniformly random index in ``[0, n)`` (0 when ``n <= 0``)."""
    count = _count_states(n, default=0)
    if count <= 0:
        return 0
    local_rng = _as_rng(rng, seed)
    return int(_rng_randint(local_rng, 0, count))


def random_indices(
    n: Any = None,
    k: int = 10,
    rng: Any = None,
    seed: Optional[int] = None,
    replace: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """``k`` random visited-state indices (sampled without replacement by default)."""
    count = _count_states(n, default=0)
    if count <= 0 or k is None or k <= 0:
        return np.zeros(0, dtype=int)
    local_rng = _as_rng(rng, seed)
    k = int(min(k, count) if not replace else k)
    if replace:
        return np.asarray(_rng_randint(local_rng, 0, count, size=k), dtype=int)
    try:
        return np.asarray(local_rng.choice(count, size=k, replace=False), dtype=int)
    except Exception:
        return np.asarray(_rng_randint(local_rng, 0, count, size=k), dtype=int)


def random_critical_index(
    scores: Any = None,
    n: Any = None,
    rng: Any = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> int:
    """Random critical-step index, either from a score vector or an explicit length."""
    if scores is not None:
        count = _count_states(scores, default=0)
        if count > 0:
            return random_index(count, rng=rng, seed=seed)
    return random_index(_count_states(n, default=0), rng=rng, seed=seed)


def random_critical_state(
    rollout: Any,
    rng: Any = None,
    seed: Optional[int] = None,
    scorer: Optional[RandomImportanceScorer] = None,
    env_id: str = "default",
    attach_scores: bool = True,
    env: Any = None,
    index: Optional[int] = None,
    scores: Any = None,
    **kwargs: Any,
) -> Any:
    """Build the Random baseline's critical state from a rollout.

    The state is chosen **uniformly at random** among the visited states of
    ``rollout`` (Section 4.1: "randomly selecting a visited state as the
    critical state").
    """
    observations = extract_observations(rollout)
    n = _count_states(observations, default=0)
    local_rng = _as_rng(rng, seed)
    if scores is None:
        scores = random_importance_scores(n, rng=local_rng)
    else:
        scores = np.asarray(scores, dtype=float)
    idx = int(index) if index is not None else random_index(n, rng=local_rng)

    observation = None
    try:
        observation = observations[idx]
    except Exception:
        observation = None

    states = getattr(rollout, "states", None)
    if states is None and isinstance(rollout, dict):
        states = rollout.get("states")
    state = None
    states_kind = "none"
    if states is not None:
        try:
            state = states[idx]
            if isinstance(state, dict):
                states_kind = str(state.get("kind", "attr"))
            elif state is not None:
                states_kind = "sim"
        except Exception:
            state = None
    if state is None and env is not None and HAS_RESET_WRAPPER and _get_env_state is not None:
        try:  # last resort: snapshot the *current* env state (approximate)
            state = _get_env_state(env)
            states_kind = "current"
        except Exception:
            state = None

    actions = getattr(rollout, "actions", None)
    if actions is None and isinstance(rollout, dict):
        actions = rollout.get("actions")
    try:
        action_list = [np.asarray(a) for a in list(actions)[: idx + 1]]
    except Exception:
        action_list = []

    length = int(n)
    metadata: Dict[str, Any] = {
        "explanation": "random",
        "random_index": int(idx),
        "n_visited_states": int(length),
        "state_kind": states_kind,
        "terminated_early": bool(getattr(rollout, "terminated_early", False)),
    }
    rewards = getattr(rollout, "rewards", None)
    if rewards is None and isinstance(rollout, dict):
        rewards = rollout.get("rewards")
    if rewards is not None:
        try:
            metadata["trajectory_return"] = float(np.sum(np.asarray(rewards, dtype=float)))
        except Exception:
            pass

    score = float(scores[idx]) if scores is not None and idx < len(scores) else float("nan")

    try:
        return CriticalState(
            index=int(idx),
            observation=observation,
            state=state,
            actions=action_list,
            score=score,
            importance_scores=np.asarray(scores, dtype=float) if attach_scores else None,
            trajectory_length=int(length),
            env_id=env_id,
            metadata=metadata,
        )
    except Exception:  # pragma: no cover - container mismatch
        return _FallbackCriticalState(
            index=int(idx),
            observation=observation,
            state=state,
            actions=action_list,
            score=score,
            importance_scores=np.asarray(scores, dtype=float) if attach_scores else None,
            trajectory_length=int(length),
            env_id=env_id,
            metadata=metadata,
        )


def random_top_k_states(
    rollout: Any,
    k: int = 10,
    rng: Any = None,
    seed: Optional[int] = None,
    env_id: str = "default",
    attach_scores: bool = True,
    env: Any = None,
    **kwargs: Any,
) -> List[Any]:
    """``k`` random visited states of ``rollout``, returned as critical states."""
    observations = extract_observations(rollout)
    n = _count_states(observations, default=0)
    local_rng = _as_rng(rng, seed)
    scores = random_importance_scores(n, rng=local_rng)
    idxs = random_indices(n, k=k, rng=local_rng)
    return [
        random_critical_state(
            rollout,
            scorer=None,
            env_id=env_id,
            attach_scores=attach_scores,
            env=env,
            index=int(i),
            scores=scores,
        )
        for i in idxs
    ]


def identify_random_state(
    env: Any,
    policy: Any,
    K: Optional[int] = None,
    mask_net: Any = None,
    rng: Any = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    reset: bool = True,
    return_rollout: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    attach_scores: bool = True,
    env_id: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Roll ``pi`` for ``K`` steps and return a random visited state.

    Drop-in analogue of
    ``rice.explanation.critical_state.identify_critical_state`` implementing the
    Random explanation baseline.
    """
    length = K
    if length is None:
        length = int(default_k(env, fallback=DEFAULT_K))
    else:
        value = float(length)
        if 0.0 < value < 1.0:
            length = max(1, int(round(value * int(default_k(env, fallback=DEFAULT_K)))))
        else:
            length = int(value)

    rollout = roll_trajectory(
        env,
        policy,
        length=int(length),
        reset=reset,
        deterministic=deterministic,
        seed=seed,
        collect_states=True,
    )
    key = env_id or getattr(env, "rice_env_key", "default")
    critical = random_critical_state(
        rollout,
        rng=rng,
        seed=seed,
        env_id=key,
        attach_scores=attach_scores,
        env=env,
    )
    if return_rollout:
        return critical, rollout
    return critical


def select_random_states(
    env: Any,
    policy: Any,
    n: int = 1,
    K: Optional[int] = None,
    seed: Optional[int] = None,
    rng: Any = None,
    deterministic: bool = False,
    env_id: Optional[str] = None,
    **kwargs: Any,
) -> List[Any]:
    """``n`` independent rollouts, each contributing one random critical state."""
    local_rng = _as_rng(rng, seed)
    out: List[Any] = []
    for _ in range(int(max(0, n))):
        out.append(
            identify_random_state(
                env,
                policy,
                K=K,
                rng=local_rng,
                seed=None,
                deterministic=deterministic,
                env_id=env_id,
                **kwargs,
            )
        )
    return out


def make_random_explanation(
    env: Any = None,
    policy: Any = None,
    env_id: str = "default",
    seed: Optional[int] = None,
    config: Optional[Any] = None,
    K: Optional[int] = None,
    deterministic_policy: bool = False,
    rng: Any = None,
    **kwargs: Any,
) -> RandomExplanation:
    """Factory for the Random explanation baseline.

    Accepts either a :class:`RandomExplanationConfig` or a raw (nested) config
    dict as ``config``, mirroring the other RICE factories.
    """
    if isinstance(config, dict):
        config = RandomExplanationConfig.from_dict(config)
    elif config is not None and not isinstance(config, RandomExplanationConfig):
        config = RandomExplanationConfig.from_dict(getattr(config, "__dict__", None))
    return RandomExplanation(
        env=env,
        policy=policy,
        env_id=env_id,
        seed=seed,
        config=config,
        rng=rng,
        K=K,
        deterministic_policy=deterministic_policy,
        **kwargs,
    )


build_random_explanation = make_random_explanation


def random_explanation_for(env_id: str = "default", **kwargs: Any) -> RandomExplanation:
    """Build a Random explanation pre-configured for a given application id."""
    kwargs.setdefault("env_id", env_id)
    return make_random_explanation(**kwargs)


def describe_random_explanation(explanation: Any = None) -> str:
    """One-line human-readable description (logging helper)."""
    if explanation is None:
        return "Random explanation: uniformly random visited state as the critical state."
    env_id = getattr(explanation, "env_id", getattr(getattr(explanation, "config", None), "env_id", "default"))
    mode = getattr(getattr(explanation, "config", None), "mode", "uniform")
    return (
        f"Random explanation (env_id={env_id!r}, mode={mode}): "
        "uninformative importance scores; critical state = uniformly random visited state."
    )
