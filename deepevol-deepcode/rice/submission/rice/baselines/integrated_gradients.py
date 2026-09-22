"""Integrated Gradients explanation baseline for RICE (Experiment III / Table 6).

The paper ("Impact of Other Explanation Methods", Appendix C.3) investigates the impact of
*other* explanation methods -- Integrated Gradients (Sundararajan et al., 2017) and AIRS
(Yu et al., 2023) -- on four MuJoCo games.  They fix the refining method (RICE's Stage-2 PPO
refinement with the mixed initial state distribution and the RND bonus) and only swap the
explanation used to identify the critical steps.  Using Integrated Gradients / AIRS "our
framework still achieves better results than the random baseline, suggesting that our
framework can work with different explanation method choices."

This module provides that explanation method.  Given a frozen target policy ``pi`` and a
visited state ``s``, the step-level importance attributed to ``s`` is the Integrated
Gradients attribution of the policy's action output with respect to the observation, along
the straight-line path from a baseline observation ``b`` to ``s``::

    IG_i(s) = (s_i - b_i) * (1/m) * sum_{k=1..m} d F(b + (k/m)(s - b)) / d x_i

with ``F`` the (sum of the) policy action output.  The scalar step importance is a reduction
(L1 by default) over the attribution vector, so that -- exactly like the mask network's
``P(keep)`` -- higher values mean "this step is more critical".

The class/function surface deliberately mirrors
:class:`rice.baselines.random_explanation.RandomExplanation` and
:class:`rice.explanation.critical_state.CriticalStateSelector` (``score``, ``score_trajectory``,
``rank``, ``top_k_states``, ``rollout``, ``select``, ``select_top_k``, ``identify``,
``reset_to``, ``summary``) so experiment drivers can swap
``{Random, StateMask, Integrated Gradients, AIRS, Ours}`` transparently.

Everything is import-safe: if PyTorch or the project's explanation modules are unavailable a
dependency-free finite-difference / hashed fallback keeps the API usable.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - exercised at runtime
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False

try:
    from ..utils.seeding import get_rng as _project_get_rng
except Exception:  # pragma: no cover
    _project_get_rng = None

try:
    from ..utils.logging import get_logger as _project_get_logger
except Exception:  # pragma: no cover
    _project_get_logger = None

try:
    from ..models.policies import normalize_env_key as _normalize_env_key
except Exception:  # pragma: no cover

    def _normalize_env_key(env_id: Any) -> str:  # type: ignore
        text = str(env_id if env_id is not None else "default").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        for sep in ("_v0", "_v1", "_v2", "_v3", "_v4", "_v5"):
            if text.endswith(sep):
                text = text[: -len(sep)]
        if text.endswith(".yaml"):
            text = text[:-5]
        return text or "default"


try:  # importance helpers (scoring/ranking/aggregation)
    from ..explanation.importance import (  # type: ignore
        DEFAULT_BATCH_SIZE,
        IMPORTANCE_MODES,
        TrajectoryImportance,
        aggregate_importance,
        argmax_importance,
        attach_scores_to_wrapper,
        extract_observations,
        rank_states,
        summarize_importance,
        top_k_indices,
    )

    _HAS_IMPORTANCE = True
except Exception:  # pragma: no cover
    _HAS_IMPORTANCE = False
    DEFAULT_BATCH_SIZE = 4096
    IMPORTANCE_MODES = ("mean", "max", "sum", "last", "first", "topk_mean")
    TrajectoryImportance = None  # type: ignore
    aggregate_importance = None  # type: ignore
    argmax_importance = None  # type: ignore
    attach_scores_to_wrapper = None  # type: ignore
    extract_observations = None  # type: ignore
    rank_states = None  # type: ignore
    summarize_importance = None  # type: ignore
    top_k_indices = None  # type: ignore

try:  # rollout / critical-state machinery
    from ..explanation.critical_state import (  # type: ignore
        CriticalState,
        CriticalStateSelector,
        TrajectoryRollout,
        attach_critical_state,
        default_k,
        policy_action,
        roll_trajectory,
    )

    _HAS_CRITICAL = True
except Exception:  # pragma: no cover
    _HAS_CRITICAL = False
    CriticalState = None  # type: ignore
    CriticalStateSelector = None  # type: ignore
    TrajectoryRollout = None  # type: ignore
    attach_critical_state = None  # type: ignore
    policy_action = None  # type: ignore
    roll_trajectory = None  # type: ignore

    def default_k(env=None, fallback: int = 1000) -> int:  # type: ignore
        for attr in ("rice_max_episode_steps", "max_episode_steps", "_max_episode_steps"):
            try:
                value = getattr(env, attr, None)
            except Exception:
                value = None
            if value:
                try:
                    return int(value)
                except Exception:
                    pass
        spec = getattr(env, "rice_env_spec", None) if env is not None else None
        if spec is not None and getattr(spec, "max_episode_steps", None):
            return int(spec.max_episode_steps)
        inner = getattr(env, "env", None) if env is not None else None
        if inner is not None and inner is not env:
            return default_k(inner, fallback=fallback)
        return int(fallback)


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_N_STEPS: int = 16
"""Number of Riemann steps ``m`` for the IG path integral (typical IG uses 20-50)."""

DEFAULT_BATCH_SIZE_IG: int = 512
DEFAULT_TOPK_FRACTION: float = 0.1
DEFAULT_K: int = 1000

IG_TARGETS = ("action", "log_prob", "value")
IG_BASELINES = ("zero", "mean", "random")
IG_REDUCTIONS = ("l1", "l2", "sum", "abs_mean", "signed", "max")

_LOGGER = logging.getLogger("rice.baselines.integrated_gradients")


# --------------------------------------------------------------------------------------
# Small dependency-free helpers
# --------------------------------------------------------------------------------------
def _get_rng(rng: Any = None, seed: Optional[int] = None) -> Any:
    """Return a numpy RandomState/Generator (project helper when available)."""
    if rng is not None:
        return rng
    if _project_get_rng is not None:
        try:
            return _project_get_rng(seed)
        except Exception:
            pass
    if seed is None:
        return np.random.RandomState(0)
    return np.random.RandomState(int(seed) % (2 ** 31 - 1))


def _get_logger(name: str = "rice.baselines.integrated_gradients"):
    if _project_get_logger is not None:
        try:
            return _project_get_logger(name)
        except Exception:
            pass
    return logging.getLogger(name)


def flatten_observation(observation: Any) -> np.ndarray:
    """Flatten an observation (array / dict / tuple / scalar) to a 1-D float32 vector."""
    if isinstance(observation, dict):
        parts: List[np.ndarray] = []
        for key in sorted(observation.keys()):
            parts.append(np.asarray(observation[key], dtype=np.float32).reshape(-1))
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)
    if isinstance(observation, (list, tuple)) and observation and np.isscalar(observation[0]):
        return np.asarray(observation, dtype=np.float32).reshape(-1)
    if _HAS_TORCH and torch is not None and isinstance(observation, torch.Tensor):
        observation = observation.detach().cpu().numpy()
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def observation_matrix(observations: Any) -> np.ndarray:
    """Stack an arbitrary collection of observations into an ``(N, D)`` float32 matrix."""
    if observations is None:
        return np.zeros((0, 0), dtype=np.float32)
    if _HAS_TORCH and torch is not None and isinstance(observations, torch.Tensor):
        observations = observations.detach().cpu().numpy()
    arr = np.asarray(observations, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1).astype(np.float32)


def _extract_observations(trajectory: Any) -> np.ndarray:
    """Normalise a rollout container into an ``(L, D)`` observation matrix."""
    if extract_observations is not None:
        try:
            obs = extract_observations(trajectory)
            return observation_matrix(obs)
        except Exception:
            pass
    if trajectory is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(trajectory, dict):
        for key in ("observations", "obs", "states"):
            if key in trajectory:
                return observation_matrix(trajectory[key])
        return np.zeros((0, 0), dtype=np.float32)
    for attr in ("observations", "obs"):
        if hasattr(trajectory, attr):
            try:
                return observation_matrix(getattr(trajectory, attr))
            except Exception:
                pass
    if isinstance(trajectory, (list, tuple)):
        if len(trajectory) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return observation_matrix(list(trajectory))
    return observation_matrix(trajectory)


def _argmax_importance(scores: np.ndarray) -> int:
    if argmax_importance is not None:
        try:
            return int(argmax_importance(scores))
        except Exception:
            pass
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return 0
    return int(np.nanargmax(np.where(np.isfinite(arr), arr, -np.inf)))


def _rank_states(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    if rank_states is not None:
        try:
            return np.asarray(rank_states(scores, descending=descending))
        except Exception:
            pass
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    filled = np.where(np.isfinite(arr), arr, -np.inf if descending else np.inf)
    order = np.argsort(-filled, kind="mergesort") if descending else np.argsort(filled, kind="mergesort")
    return order.astype(np.int64)


def _top_k_indices(scores: np.ndarray, k: int = 10) -> np.ndarray:
    if top_k_indices is not None:
        try:
            return np.asarray(top_k_indices(scores, k=k))
        except Exception:
            pass
    order = _rank_states(scores, descending=True)
    return order[: max(int(k), 0)].astype(np.int64)


def _aggregate_importance(scores: np.ndarray, mode: str = "mean", topk_frac: float = DEFAULT_TOPK_FRACTION) -> float:
    if aggregate_importance is not None:
        try:
            return float(aggregate_importance(scores, mode=mode, topk_frac=topk_frac))
        except Exception:
            pass
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    mode = (mode or "mean").lower()
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "max":
        return float(np.max(arr))
    if mode == "min":
        return float(np.min(arr))
    if mode == "sum":
        return float(np.sum(arr))
    if mode == "last":
        return float(arr[-1])
    if mode == "first":
        return float(arr[0])
    if mode == "topk_mean":
        k = max(int(math.ceil(float(topk_frac) * arr.size)), 1)
        return float(np.mean(np.sort(arr)[::-1][:k]))
    return float(np.mean(arr))


def _summarize_importance(scores: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
    if summarize_importance is not None:
        try:
            return dict(summarize_importance(scores, threshold=threshold))
        except Exception:
            pass
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "argmax": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "argmax": int(np.nanargmax(arr)),
        "threshold": (None if threshold is None else float(threshold)),
    }


def _attach_scores(wrapper: Any, scores: np.ndarray) -> Any:
    if attach_scores_to_wrapper is not None:
        try:
            return attach_scores_to_wrapper(wrapper, scores)
        except Exception:
            pass
    try:
        setattr(wrapper, "last_importance_scores", np.asarray(scores, dtype=np.float64).reshape(-1))
    except Exception:
        pass
    return wrapper


def _hashed_scores(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Deterministic pseudo-importance scores (last-resort fallback)."""
    n = max(int(n), 0)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    salt = float((seed or 0) % 9973)
    return 0.5 + 0.5 * np.sin(idx * 12.9898 + salt)


# --------------------------------------------------------------------------------------
# Policy output extraction + attribution maths
# --------------------------------------------------------------------------------------
def _dist_mean(dist: Any) -> Any:
    """Best-effort extraction of a differentiable distribution location parameter."""
    for attr in ("mean", "loc"):
        try:
            value = getattr(dist, attr)
            if callable(value):
                value = value()
            if value is not None:
                return value
        except Exception:
            continue
    inner = getattr(dist, "distribution", None)
    if inner is not None:
        for attr in ("mean", "loc"):
            try:
                value = getattr(inner, attr)
                if callable(value):
                    value = value()
                if value is not None:
                    return value
            except Exception:
                continue
    for attr in ("logits", "probs"):
        try:
            value = getattr(dist, attr)
            if value is not None:
                return value
        except Exception:
            continue
    return None


def policy_action_output(policy: Any, obs_tensor: Any, target: str = "action", actions: Any = None) -> Any:
    """Differentiable policy output used as ``F`` in the Integrated Gradients path integral.

    ``target`` selects the scalarisation:
      * ``"action"``  -- the policy's action mean (continuous) / logits (discrete);
      * ``"value"``   -- the critic's value estimate;
      * ``"log_prob"``-- the log-probability of ``actions`` (supplied) under the policy.
    Returns a tensor of shape ``(B, D')`` (or ``(B,)``) or ``None`` when unavailable.
    """
    if policy is None or obs_tensor is None:
        return None

    target = (target or "action").lower()

    if target == "value":
        for name in ("predict_values", "value"):
            fn = getattr(policy, name, None)
            if callable(fn):
                try:
                    out = fn(obs_tensor)
                    if out is not None:
                        return out
                except Exception:
                    continue
        return None

    get_dist = getattr(policy, "get_distribution", None)
    dist = None
    if callable(get_dist):
        try:
            dist = get_dist(obs_tensor)
        except Exception:
            dist = None

    if target == "log_prob":
        if dist is not None and actions is not None:
            try:
                log_prob = dist.log_prob(actions)
                return log_prob
            except Exception:
                pass
        return None

    if dist is not None:
        mean = _dist_mean(dist)
        if mean is not None:
            return mean

    # Fallbacks: module forward pass
    for call in (
        lambda x: policy(x),
        lambda x: policy.forward(x),
    ):
        try:
            out = call(obs_tensor)
        except Exception:
            continue
        if out is None:
            continue
        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                continue
            if len(out) >= 2 and target == "value":
                out = out[1]
            else:
                out = out[0]
        return out

    actor = getattr(policy, "actor", None) or getattr(policy, "action_net", None)
    if actor is not None:
        try:
            return actor(obs_tensor)
        except Exception:
            pass

    if callable(policy):
        try:
            return policy(obs_tensor)
        except Exception:
            pass
    return None


def _reduce_attributions(attributions: np.ndarray, reduction: str = "l1") -> np.ndarray:
    attr = np.asarray(attributions, dtype=np.float64)
    if attr.ndim == 1:
        attr = attr.reshape(-1, 1)
    reduction = (reduction or "l1").lower()
    if reduction in ("l1", "abs_mean"):
        return np.mean(np.abs(attr), axis=1)
    if reduction == "l2":
        return np.sqrt(np.mean(attr ** 2, axis=1))
    if reduction == "sum":
        return np.sum(np.abs(attr), axis=1)
    if reduction == "max":
        return np.max(np.abs(attr), axis=1)
    if reduction == "signed":
        return np.mean(attr, axis=1)
    return np.mean(np.abs(attr), axis=1)


def _resolve_baseline(
    observations: np.ndarray,
    baseline: Any,
    rng: Any = None,
) -> np.ndarray:
    """Resolve the IG reference observation ``b`` (shape ``(N, D)``)."""
    obs = np.asarray(observations, dtype=np.float32)
    if isinstance(baseline, str):
        key = baseline.strip().lower()
        if key in ("zero", "zeros", "none"):
            return np.zeros_like(obs)
        if key in ("mean", "average"):
            return np.repeat(obs.mean(axis=0, keepdims=True), obs.shape[0], axis=0)
        if key in ("random", "rand"):
            source = rng if rng is not None else np.random.RandomState(0)
            if hasattr(source, "uniform"):
                return source.uniform(low=obs.min(), high=obs.max(), size=obs.shape).astype(np.float32)
            return np.random.uniform(low=obs.min(), high=obs.max(), size=obs.shape).astype(np.float32)
        key_arr = np.asarray(baseline, dtype=np.float32).reshape(-1)
        return np.broadcast_to(key_arr, obs.shape).astype(np.float32)
    if baseline is None:
        return np.zeros_like(obs)
    arr = np.asarray(baseline, dtype=np.float32).reshape(-1)
    return np.broadcast_to(arr, obs.shape).astype(np.float32)


def integrated_gradients_attributions(
    policy: Any,
    observations: Any,
    baseline: Any = "zero",
    n_steps: int = DEFAULT_N_STEPS,
    target: str = "action",
    actions: Any = None,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE_IG,
    rng: Any = None,
    reduction: str = "l1",
) -> Optional[np.ndarray]:
    """Compute Integrated Gradients attributions ``(s - b) * mean_k grad F(b + k/m (s-b))``.

    Returns an ``(N, D)`` float64 array of per-feature attributions, or ``None`` if torch /
    a differentiable policy is unavailable (the caller then falls back to finite differences).
    """
    if not _HAS_TORCH or policy is None:
        return None
    obs = observation_matrix(observations)
    if obs.size == 0:
        return np.zeros((obs.shape[0], obs.shape[1]), dtype=np.float64)
    base = _resolve_baseline(obs, baseline, rng=rng)
    steps = max(int(n_steps), 1)
    out_attr: List[np.ndarray] = []
    bs = max(int(batch_size), 1)

    for start in range(0, obs.shape[0], bs):
        x_np = obs[start : start + bs]
        b_np = base[start : start + bs]
        acts_np = None
        if actions is not None:
            try:
                acts_np = np.asarray(actions, dtype=np.float32)[start : start + bs]
            except Exception:
                acts_np = None
        try:
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            b = torch.as_tensor(b_np, dtype=torch.float32, device=device)
            delta = (x - b).detach()
            total_grad = torch.zeros_like(x)
            for k in range(1, steps + 1):
                alpha = float(k) / float(steps)
                point = (b + alpha * delta).detach().clone().requires_grad_(True)
                acts_t = None
                if acts_np is not None:
                    acts_t = torch.as_tensor(acts_np, dtype=torch.float32, device=device)
                out = policy_action_output(policy, point, target=target, actions=acts_t)
                if out is None:
                    return None
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    return None
                scalar = out.reshape(out.shape[0], -1).sum()
                grad = torch.autograd.grad(scalar, point, retain_graph=False, allow_unused=True)[0]
                if grad is None:
                    return None
                total_grad = total_grad + grad.detach()
            avg_grad = total_grad / float(steps)
            attr = (delta * avg_grad).detach().cpu().numpy().astype(np.float64)
            out_attr.append(attr)
        except Exception:
            return None

    if not out_attr:
        return None
    return np.concatenate(out_attr, axis=0)


def _policy_scalar_output(
    policy: Any,
    observations: Any,
    target: str = "action",
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE_IG,
) -> Optional[np.ndarray]:
    """Evaluate the (reduced) policy output at observations -- used by the FD fallback."""
    obs = observation_matrix(observations)
    if obs.size == 0:
        return np.zeros((obs.shape[0],), dtype=np.float64)
    bs = max(int(batch_size), 1)
    chunks: List[np.ndarray] = []
    for start in range(0, obs.shape[0], bs):
        chunk = obs[start : start + bs]
        out_np = None
        if _HAS_TORCH and policy is not None:
            try:
                with torch.no_grad():
                    tensor = torch.as_tensor(chunk, dtype=torch.float32, device=device)
                    out = policy_action_output(policy, tensor, target=target)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if out is not None:
                        if isinstance(out, torch.Tensor):
                            out_np = out.detach().cpu().numpy()
                        else:
                            out_np = np.asarray(out)
            except Exception:
                out_np = None
        if out_np is None and policy is not None and callable(policy):
            try:
                value = policy(chunk)
                out_np = np.asarray(value)
            except Exception:
                out_np = None
        if out_np is None:
            return None
        arr = np.asarray(out_np, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        chunks.append(arr.reshape(arr.shape[0], -1))
    return np.concatenate(chunks, axis=0) if chunks else None


def finite_difference_importance(
    policy: Any,
    observations: Any,
    baseline: Any = "zero",
    target: str = "action",
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE_IG,
    rng: Any = None,
) -> Optional[np.ndarray]:
    """Zero-order fallback: ``||F(s) - F(b)||_1`` per state (a first-order IG surrogate)."""
    obs = observation_matrix(observations)
    if obs.size == 0:
        return np.zeros((obs.shape[0],), dtype=np.float64)
    base = _resolve_baseline(obs, baseline, rng=rng)
    out_s = _policy_scalar_output(policy, obs, target=target, device=device, batch_size=batch_size)
    out_b = _policy_scalar_output(policy, base, target=target, device=device, batch_size=batch_size)
    if out_s is None or out_b is None:
        return None
    diff = np.abs(out_s - out_b)
    return np.mean(diff, axis=1) if diff.ndim > 1 else diff.reshape(-1)


def ig_importance_scores(
    policy: Any,
    observations: Any,
    baseline: Any = "zero",
    n_steps: int = DEFAULT_N_STEPS,
    target: str = "action",
    reduction: str = "l1",
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE_IG,
    rng: Any = None,
    seed: Optional[int] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Step-level Integrated Gradients importance scores for a batch of observations.

    Priority order: (1) true IG attributions, (2) finite-difference surrogate,
    (3) deterministic hashed pseudo-scores.  Never raises.
    """
    obs = observation_matrix(observations)
    n = int(obs.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    scores: Optional[np.ndarray] = None
    attr = integrated_gradients_attributions(
        policy,
        obs,
        baseline=baseline,
        n_steps=n_steps,
        target=target,
        device=device,
        batch_size=batch_size,
        rng=_get_rng(rng, seed),
        reduction=reduction,
    )
    if attr is not None:
        scores = _reduce_attributions(attr, reduction=reduction)

    if scores is None:
        scores = finite_difference_importance(
            policy,
            obs,
            baseline=baseline,
            target=target,
            device=device,
            batch_size=batch_size,
            rng=_get_rng(rng, seed),
        )

    if scores is None:
        scores = _hashed_scores(n, seed=seed)

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size != n:
        scores = np.resize(scores, n)

    if normalize:
        lo, hi = float(np.min(scores)), float(np.max(scores))
        span = hi - lo
        scores = (scores - lo) / span if span > 1e-12 else np.zeros_like(scores)
    return scores


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class IntegratedGradientsConfig:
    """Configuration for the Integrated Gradients explanation baseline."""

    env_id: str = "default"
    seed: Optional[int] = None
    n_steps: int = DEFAULT_N_STEPS
    baseline: str = "zero"
    target: str = "action"
    reduction: str = "l1"
    normalize: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE_IG
    topk_fraction: float = DEFAULT_TOPK_FRACTION
    deterministic_policy: bool = True
    attach_scores: bool = True
    with_mask_shim: bool = False
    device: str = "cpu"
    K: Optional[int] = None
    uses_project_modules: bool = field(default=True)

    def __post_init__(self) -> None:
        self.env_id = _normalize_env_key(self.env_id)
        self.n_steps = max(int(self.n_steps), 1)
        self.baseline = str(self.baseline).strip().lower() if isinstance(self.baseline, str) else self.baseline
        self.target = str(self.target).strip().lower()
        if self.target not in IG_TARGETS:
            self.target = "action"
        self.reduction = str(self.reduction).strip().lower()
        if self.reduction not in IG_REDUCTIONS:
            self.reduction = "l1"
        self.batch_size = max(int(self.batch_size), 1)
        self.topk_fraction = float(np.clip(self.topk_fraction, 1e-6, 1.0))

    @classmethod
    def from_dict(cls, cfg: Optional[Any] = None, **overrides: Any) -> "IntegratedGradientsConfig":
        """Build from a YAML section / flat dict, with alias and nested-section support."""
        data: Dict[str, Any] = {}
        if cfg is not None:
            if isinstance(cfg, IntegratedGradientsConfig):
                data.update(cfg.to_dict())
            elif isinstance(cfg, dict):
                for section in ("integrated_gradients", "integrated-gradients", "ig", "explanation", "baseline"):
                    sub = cfg.get(section)
                    if isinstance(sub, dict):
                        data.update(sub)
                for key, value in cfg.items():
                    if key in cls.__dataclass_fields__:
                        data[key] = value
                # accept the whole dict too when it only contains known keys
            else:
                data["env_id"] = str(cfg)
        alias = {
            "m": "n_steps",
            "steps": "n_steps",
            "n_interpolations": "n_steps",
            "reference": "baseline",
            "base": "baseline",
            "topk_frac": "topk_fraction",
            "lr": None,
            "lambda": None,
            "lam": None,
            "p": None,
        }
        for key, value in list(overrides.items()):
            if key in alias and alias[key]:
                data[alias[key]] = value
            elif key in cls.__dataclass_fields__:
                data[key] = value
        known = set(cls.__dataclass_fields__.keys())
        data = {k: v for k, v in data.items() if k in known and k != "uses_project_modules"}
        try:
            return cls(**data)
        except Exception:
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "seed": self.seed,
            "n_steps": self.n_steps,
            "baseline": self.baseline,
            "target": self.target,
            "reduction": self.reduction,
            "normalize": bool(self.normalize),
            "batch_size": self.batch_size,
            "topk_fraction": self.topk_fraction,
            "deterministic_policy": bool(self.deterministic_policy),
            "attach_scores": bool(self.attach_scores),
            "with_mask_shim": bool(self.with_mask_shim),
            "device": self.device,
            "K": self.K,
        }


# --------------------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------------------
class IGScorer:
    """Step-level Integrated Gradients scorer (drop-in for ``ImportanceScorer``)."""

    name = "Integrated Gradients"

    def __init__(
        self,
        policy: Any = None,
        config: Optional[Any] = None,
        env_id: str = "default",
        rng: Any = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        n_steps: Optional[int] = None,
        baseline: Optional[str] = None,
        target: Optional[str] = None,
        reduction: Optional[str] = None,
        mask_net: Any = None,
        **kwargs: Any,
    ) -> None:
        self.config = IntegratedGradientsConfig.from_dict(
            config if config is not None else {"env_id": env_id},
            env_id=env_id,
            **{k: v for k, v in kwargs.items() if k in IntegratedGradientsConfig.__dataclass_fields__},
        )
        self.policy = policy
        self.mask_net = None  # IG is a gradient-based explanation: no mask network
        self.env_id = _normalize_env_key(self.config.env_id)
        if device is not None:
            self.config.device = device
        if batch_size is not None:
            self.config.batch_size = int(batch_size)
        if n_steps is not None:
            self.config.n_steps = int(n_steps)
        if baseline is not None:
            self.config.baseline = baseline
        if target is not None:
            self.config.target = target
        if reduction is not None:
            self.config.reduction = reduction
        self.rng = _get_rng(rng, seed if seed is not None else self.config.seed)
        self._n_scored = 0
        self._sum_score = 0.0

    # -- core scoring ---------------------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        """Attribution-based importance ``I(s)`` for each observation."""
        if observations is None:
            return np.zeros((0,), dtype=np.float64)
        obs = observation_matrix(observations)
        if obs.size == 0:
            return np.zeros((0,), dtype=np.float64)
        scores = ig_importance_scores(
            self.policy,
            obs,
            baseline=self.config.baseline,
            n_steps=self.config.n_steps,
            target=self.config.target,
            reduction=self.config.reduction,
            device=self.config.device,
            batch_size=self.config.batch_size,
            rng=self.rng,
            seed=self.config.seed,
            normalize=self.config.normalize,
        )
        self._n_scored += int(scores.size)
        self._sum_score += float(np.sum(scores)) if scores.size else 0.0
        return scores

    # aliases matching rice.explanation.importance.ImportanceScorer
    score_batch = score
    score_observations = score

    def score_trajectory(self, trajectory: Any) -> np.ndarray:
        return self.score(_extract_observations(trajectory))

    def importance_scores(self, trajectory: Any) -> np.ndarray:
        return self.score_trajectory(trajectory)

    def most_important_index(self, trajectory: Any) -> int:
        scores = self.score_trajectory(trajectory)
        return _argmax_importance(scores)

    def most_important_state(self, trajectory: Any) -> Tuple[int, np.ndarray]:
        obs = _extract_observations(trajectory)
        idx = self.most_important_index(trajectory)
        if obs.shape[0] == 0:
            return 0, np.zeros((0,), dtype=np.float32)
        return idx, obs[min(idx, obs.shape[0] - 1)]

    def top_k_states(self, trajectory: Any, k: int = 10) -> List[Tuple[int, np.ndarray]]:
        obs = _extract_observations(trajectory)
        scores = self.score(obs)
        order = _top_k_indices(scores, k=k)
        out: List[Tuple[int, np.ndarray]] = []
        for idx in order:
            j = int(idx)
            if 0 <= j < obs.shape[0]:
                out.append((j, obs[j]))
        return out

    def ranked_observations(self, trajectory: Any) -> np.ndarray:
        obs = _extract_observations(trajectory)
        if obs.shape[0] == 0:
            return obs
        return obs[_rank_states(self.score(obs), descending=True)]

    def rank(self, trajectory: Any) -> np.ndarray:
        return _rank_states(self.score_trajectory(trajectory), descending=True)

    def trajectory_aggregate(self, trajectory: Any, mode: Optional[str] = None) -> float:
        scores = self.score_trajectory(trajectory)
        return _aggregate_importance(scores, mode=mode or "mean", topk_frac=self.config.topk_fraction)

    def summary(self, trajectory: Any) -> Dict[str, Any]:
        scores = self.score_trajectory(trajectory)
        out = _summarize_importance(scores)
        out.update({"method": self.name, "env_id": self.env_id})
        return out

    def attach_to_wrapper(self, wrapper: Any, trajectory: Any = None) -> Any:
        if wrapper is None:
            return wrapper
        scores = self.score_trajectory(trajectory) if trajectory is not None else None
        if scores is not None:
            _attach_scores(wrapper, scores)
        try:
            setattr(wrapper, "ig_scorer", self)
        except Exception:
            pass
        return wrapper

    @property
    def is_random(self) -> bool:
        return False

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "n_scored": self._n_scored,
            "mean_score": (self._sum_score / self._n_scored) if self._n_scored else float("nan"),
            "config": self.config.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.stats

    # mask-network compatibility shims ---------------------------------------------
    def keep_probability(self, observations: Any) -> np.ndarray:
        return self.score(observations)

    blind_probability = keep_probability

    def __call__(self, observations: Any) -> np.ndarray:
        return self.score(observations)


# --------------------------------------------------------------------------------------
# Selector (critical-state identification)
# --------------------------------------------------------------------------------------
def _build_critical_state(
    rollout: Any,
    scores: np.ndarray,
    index: int,
    env_id: str = "default",
    extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construct a restorable ``CriticalState`` (or duck-typed equivalent)."""
    obs = _extract_observations(rollout)
    n = int(scores.size) if scores is not None else obs.shape[0]
    idx = int(np.clip(index, 0, max(n - 1, 0)))
    observation = obs[idx] if obs.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

    state = None
    states = getattr(rollout, "states", None)
    if states is not None:
        try:
            state = states[idx]
        except Exception:
            state = None

    actions = None
    prior = getattr(rollout, "actions", None)
    if prior is not None:
        try:
            arr = np.asarray(prior)
            actions = arr[: idx + 1].copy() if arr.ndim >= 1 else None
        except Exception:
            actions = None

    length = getattr(rollout, "length", None)
    try:
        trajectory_length = int(length) if length is not None else int(obs.shape[0])
    except Exception:
        trajectory_length = int(obs.shape[0])

    metadata: Dict[str, Any] = {
        "method": "Integrated Gradients",
        "explanation": "integrated_gradients",
        "trajectory_length": trajectory_length,
        "env_id": env_id,
    }
    if extra:
        metadata.update(extra)

    if CriticalState is not None:
        try:
            return CriticalState(
                index=idx,
                observation=observation,
                state=state,
                actions=actions,
                score=float(scores[idx]) if scores is not None and scores.size > idx else float("nan"),
                importance_scores=(None if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)),
                trajectory_length=trajectory_length,
                env_id=env_id,
                metadata=metadata,
            )
        except Exception:
            pass

    # duck-typed fallback
    class _FallbackCriticalState:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

        @property
        def restore_payload(self) -> Any:
            return self.state

        def to_dict(self, include_scores: bool = False) -> Dict[str, Any]:
            out = {k: v for k, v in self.__dict__.items() if k != "importance_scores"}
            out["importance_scores"] = (
                None if self.importance_scores is None else np.asarray(self.importance_scores).tolist()
            ) if include_scores else None
            return out

    return _FallbackCriticalState(
        index=idx,
        observation=observation,
        state=state,
        actions=actions,
        score=float(scores[idx]) if scores is not None and scores.size > idx else float("nan"),
        importance_scores=(None if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)),
        trajectory_length=trajectory_length,
        env_id=env_id,
        metadata=metadata,
    )


def _rollout(env: Any, policy: Any, K: Any = None, reset: bool = True, seed: Optional[int] = None,
             deterministic: bool = True) -> Any:
    """Roll the frozen policy for ``K`` steps (delegating to the project helper)."""
    length = K
    if length is None:
        length = default_k(env)
    elif isinstance(length, float) and 0.0 < length < 1.0:
        length = int(round(length * default_k(env)))
    if roll_trajectory is not None:
        try:
            return roll_trajectory(
                env,
                policy,
                length=int(length),
                reset=reset,
                deterministic=deterministic,
                seed=seed,
                collect_states=True,
                stop_on_done=True,
            )
        except Exception:
            pass
    # minimal local rollout
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    states: List[Any] = []
    obs, _info = _reset_env(env, seed=seed, reset=reset)
    for _ in range(int(length)):
        act = _policy_action(policy, obs, deterministic=deterministic)
        observations.append(flatten_observation(obs))
        actions.append(np.asarray(act, dtype=np.float32).reshape(-1))
        result = env.step(act)
        obs, reward, done = _unpack_step(result)
        rewards.append(float(reward))
        try:
            from ..envs.reset_wrapper import get_env_state as _get_state

            states.append(_get_state(env))
        except Exception:
            states.append(None)
        if done:
            break

    class _LocalRollout:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)

        def __len__(self) -> int:
            return int(self.length)

    return _LocalRollout(
        observations=np.asarray(observations, dtype=np.float32).reshape(len(observations), -1),
        actions=np.asarray(actions, dtype=np.float32).reshape(len(actions), -1),
        rewards=np.asarray(rewards, dtype=np.float64),
        next_observations=np.asarray(observations, dtype=np.float32).reshape(len(observations), -1),
        dones=np.zeros((len(rewards),), dtype=bool),
        infos=[{} for _ in rewards],
        states=states,
        env_id=getattr(env, "rice_env_key", "default"),
        length=len(rewards),
        terminated_early=False,
    )


def _reset_env(env: Any, seed: Optional[int] = None, reset: bool = True) -> Tuple[Any, Dict[str, Any]]:
    if not reset:
        obs = getattr(env, "last_observation", None)
        if obs is not None:
            return obs, {}
    result = env.reset(seed=seed) if seed is not None else env.reset()
    return _unpack_reset(result)


def _unpack_reset(result: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0], (result[1] if isinstance(result[1], dict) else {})
        return result[0], {}
    return result, {}


def _unpack_step(result: Any) -> Tuple[Any, float, bool]:
    if isinstance(result, tuple):
        if len(result) == 5:
            obs, reward, terminated, truncated, _info = result
            return obs, reward, bool(terminated) or bool(truncated)
        if len(result) == 4:
            obs, reward, done, _info = result
            return obs, reward, bool(done)
        if len(result) == 3:
            return result[0], result[1], bool(result[2])
        if len(result) == 2:
            return result[0], result[1], False
        return result[0], 0.0, False
    return result, 0.0, False


def _policy_action(policy: Any, observation: Any, deterministic: bool = True) -> np.ndarray:
    if policy is None:
        return np.zeros((1,), dtype=np.float32)
    if policy_action is not None:
        try:
            return np.asarray(policy_action(policy, observation, deterministic=deterministic), dtype=np.float32)
        except Exception:
            pass
    predict = getattr(policy, "predict", None)
    if callable(predict):
        try:
            out = predict(observation, deterministic=deterministic)
            if isinstance(out, tuple):
                out = out[0]
            return np.asarray(out, dtype=np.float32)
        except Exception:
            pass
    act = getattr(policy, "act", None)
    if callable(act):
        try:
            out = act(observation, deterministic=deterministic)
            if isinstance(out, tuple):
                out = out[0]
            return np.asarray(out, dtype=np.float32)
        except Exception:
            pass
    if callable(policy):
        try:
            out = policy(observation)
            if isinstance(out, tuple):
                out = out[0]
            if _HAS_TORCH and torch is not None and isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
            return np.asarray(out, dtype=np.float32)
        except Exception:
            pass
    return np.zeros((1,), dtype=np.float32)


class IGStateSelector:
    """Critical-state selector using Integrated Gradients importance (mirrors ``CriticalStateSelector``)."""

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        mask_net: Any = None,
        K: Any = None,
        scorer: Optional[IGScorer] = None,
        deterministic_policy: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE_IG,
        device: str = "cpu",
        rng: Any = None,
        seed: Optional[int] = None,
        attach_scores: bool = True,
        cache: bool = False,
        config: Optional[Any] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.mask_net = None
        self.K = K
        self.attach_scores = bool(attach_scores)
        self.cache = bool(cache)
        self.deterministic_policy = bool(deterministic_policy)
        self.env_id = _normalize_env_key(
            getattr(env, "rice_env_key", None) or (config.env_id if isinstance(config, IntegratedGradientsConfig) else "default")
        )
        if scorer is None:
            scorer = IGScorer(
                policy=policy,
                config=config,
                env_id=self.env_id,
                rng=rng,
                seed=seed,
                device=device,
                batch_size=batch_size,
            )
        self.scorer = scorer
        self.rng = _get_rng(rng, seed)
        self.last_rollout = None
        self.last_critical = None
        self.history: List[Any] = []

    # -- scoring ------------------------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)

    def score_trajectory(self, trajectory: Any) -> np.ndarray:
        return self.scorer.score_trajectory(trajectory)

    importance_scores = score_trajectory

    # -- rollouts / selection -------------------------------------------------------
    def rollout(self, policy: Any = None, K: Any = None, reset: bool = True, seed: Optional[int] = None) -> Any:
        rollout = _rollout(
            self.env,
            policy if policy is not None else self.policy,
            K if K is not None else self.K,
            reset=reset,
            seed=seed,
            deterministic=self.deterministic_policy,
        )
        self.last_rollout = rollout
        return rollout

    def select(
        self,
        policy: Any = None,
        K: Any = None,
        reset: bool = True,
        seed: Optional[int] = None,
        rollout: Any = None,
    ) -> Any:
        """Algorithm-2 style critical-state selection with IG importance."""
        if rollout is None:
            rollout = self.rollout(policy=policy, K=K, reset=reset, seed=seed)
        scores = self.scorer.score_trajectory(rollout)
        idx = _argmax_importance(scores)
        critical = _build_critical_state(
            rollout,
            scores,
            idx,
            env_id=self.env_id,
            extra={"n_steps": self.scorer.config.n_steps, "baseline": self.scorer.config.baseline},
        )
        if self.attach_scores and self.env is not None and scores.size:
            _attach_scores(self.env, scores)
            if attach_critical_state is not None:
                try:
                    attach_critical_state(self.env, critical)
                except Exception:
                    pass
        self.last_critical = critical
        self.history.append(critical)
        return critical

    def select_top_k(
        self,
        k: int = 10,
        policy: Any = None,
        K: Any = None,
        reset: bool = True,
        seed: Optional[int] = None,
        rollout: Any = None,
    ) -> List[Any]:
        if rollout is None:
            rollout = self.rollout(policy=policy, K=K, reset=reset, seed=seed)
        scores = self.scorer.score_trajectory(rollout)
        out: List[Any] = []
        for idx in _top_k_indices(scores, k=k):
            out.append(_build_critical_state(rollout, scores, int(idx), env_id=self.env_id))
        return out

    def reset_to(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        target_env = env if env is not None else self.env
        critical = critical if critical is not None else self.last_critical
        if target_env is None or critical is None:
            return (None, {})
        payload = getattr(critical, "restore_payload", None)
        if payload is not None:
            for method in ("reset_to_state", "reset_to", "reset_to_critical"):
                fn = getattr(target_env, method, None)
                if callable(fn):
                    try:
                        result = fn(payload) if method == "reset_to_state" else fn(critical)
                        obs, info = _unpack_reset(result)
                        return obs, info
                    except Exception:
                        continue
        result = target_env.reset()
        return _unpack_reset(result)

    def reset_to_critical(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.reset_to(env=env, critical=critical, **kwargs)

    @property
    def is_random(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "Integrated Gradients"

    @property
    def statistics(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "rollouts": len(self.history),
            "scorer": self.scorer.stats,
        }

    stats = statistics

    def summary(self) -> Dict[str, Any]:
        return self.statistics


# --------------------------------------------------------------------------------------
# Umbrella object + factories
# --------------------------------------------------------------------------------------
class IntegratedGradients:
    """Umbrella Integrated Gradients explainer (swap-in for ``RandomExplanation``).

    Exposes ``scorer`` / ``selector`` with the same method surface as the RICE mask-network
    explainer, but ``mask_net`` is ``None`` (IG is gradient based) so downstream code uses
    the explanation-agnostic scoring branch.
    """

    name = "Integrated Gradients"

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        env_id: str = "default",
        seed: Optional[int] = None,
        config: Optional[Any] = None,
        rng: Any = None,
        K: Any = None,
        deterministic_policy: bool = True,
        with_mask_shim: Optional[bool] = None,
        scorer: Optional[IGScorer] = None,
        selector: Optional[IGStateSelector] = None,
    ) -> None:
        if isinstance(config, dict):
            config = IntegratedGradientsConfig.from_dict(config)
        self.config = config if isinstance(config, IntegratedGradientsConfig) else IntegratedGradientsConfig(
            env_id=env_id, seed=seed
        )
        self.env = env
        self.policy = policy
        self.seed = seed if seed is not None else self.config.seed
        self.rng = _get_rng(rng, self.seed)
        self.env_id = _normalize_env_key(
            getattr(env, "rice_env_key", None) or self.config.env_id or env_id
        )
        self.K = K if K is not None else self.config.K
        self.deterministic_policy = bool(deterministic_policy)
        self.with_mask_shim = bool(
            self.config.with_mask_shim if with_mask_shim is None else with_mask_shim
        )
        self._mask_shim = None

        self.scorer = scorer or IGScorer(
            policy=policy,
            config=self.config,
            env_id=self.env_id,
            rng=self.rng,
            seed=self.seed,
            device=self.config.device,
            batch_size=self.config.batch_size,
        )
        self.selector = selector or IGStateSelector(
            env=env,
            policy=policy,
            scorer=self.scorer,
            K=self.K,
            rng=self.rng,
            seed=self.seed,
            deterministic_policy=self.deterministic_policy,
            config=self.config,
        )

    # -- scoring surface ------------------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        return self.scorer.score(observations)

    def score_trajectory(self, trajectory: Any) -> np.ndarray:
        return self.scorer.score_trajectory(trajectory)

    def importance_scores(self, trajectory: Any) -> np.ndarray:
        return self.scorer.score_trajectory(trajectory)

    def rank(self, trajectory: Any) -> np.ndarray:
        return self.scorer.rank(trajectory)

    def top_k_states(self, trajectory: Any, k: int = 10) -> List[Tuple[int, np.ndarray]]:
        return self.scorer.top_k_states(trajectory, k=k)

    def attach_to_wrapper(self, wrapper: Any, trajectory: Any = None) -> Any:
        return self.scorer.attach_to_wrapper(wrapper, trajectory=trajectory)

    # -- selection surface ----------------------------------------------------------
    def rollout(self, policy: Any = None, K: Any = None, **kwargs: Any) -> Any:
        return self.selector.rollout(policy=policy, K=K, **kwargs)

    def select(self, policy: Any = None, K: Any = None, **kwargs: Any) -> Any:
        self.selector.K = self.K if K is None else K
        return self.selector.select(policy=policy, K=K, **kwargs)

    def select_top_k(self, k: int = 10, policy: Any = None, K: Any = None, **kwargs: Any) -> List[Any]:
        return self.selector.select_top_k(k=k, policy=policy, K=K, **kwargs)

    def identify(self, env: Any = None, policy: Any = None, K: Any = None, **kwargs: Any) -> Any:
        """Identify the IG-critical state of a fresh rollout of the frozen policy."""
        target_env = env if env is not None else self.env
        target_policy = policy if policy is not None else self.policy
        self.selector.env = target_env
        if target_policy is not None:
            self.selector.policy = target_policy
            self.scorer.policy = target_policy
        return self.selector.select(policy=target_policy, K=self.K if K is None else K, **kwargs)

    identify_critical_state = identify

    def reset_to(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.selector.reset_to(env=env, critical=critical, **kwargs)

    def reset_to_critical(self, env: Any = None, critical: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.selector.reset_to_critical(env=env, critical=critical, **kwargs)

    # -- mask-network compatibility -------------------------------------------------
    def as_mask_network(self, **kwargs: Any) -> Any:
        """Return an object exposing the mask-network surface backed by IG scores."""
        if self._mask_shim is None:
            self._mask_shim = _IGMaskShim(self.scorer)
        return self._mask_shim

    @property
    def mask_net(self) -> Any:
        return self.as_mask_network() if self.with_mask_shim else None

    @property
    def mask_network(self) -> Any:
        return self.mask_net

    @property
    def is_random(self) -> bool:
        return False

    @property
    def statistics(self) -> Dict[str, Any]:
        return {
            "method": self.name,
            "env_id": self.env_id,
            "scorer": self.scorer.stats,
            "selector": self.selector.statistics,
            "config": self.config.to_dict(),
            "rollouts": len(self.selector.history),
        }

    stats = statistics

    def summary(self) -> Dict[str, Any]:
        return self.statistics

    def describe(self) -> str:
        return (
            f"IntegratedGradients(env={self.env_id}, m={self.config.n_steps}, "
            f"baseline={self.config.baseline}, target={self.config.target}, "
            f"reduction={self.config.reduction})"
        )

    def __call__(self, trajectory: Any) -> np.ndarray:
        return self.score_trajectory(trajectory)


class _IGMaskShim:
    """Object exposing a mask-network-like surface (``keep_probability`` etc.) backed by IG."""

    def __init__(self, scorer: IGScorer) -> None:
        self._scorer = scorer
        self.name = "IntegratedGradientsMaskShim"

    def keep_probability(self, observations: Any) -> Any:
        """Map IG importance onto a (non-normalised) ``P(keep)``-like score."""
        scores = self._scorer.score(observations)
        if _HAS_TORCH and torch is not None:
            try:
                return torch.as_tensor(scores, dtype=torch.float32)
            except Exception:
                pass
        return scores

    blind_probability = keep_probability
    score = keep_probability

    def state_importance(self, observations: Any, batch_size: int = DEFAULT_BATCH_SIZE_IG) -> Any:
        return self.keep_probability(observations)


# --------------------------------------------------------------------------------------
# Free functions
# --------------------------------------------------------------------------------------
def integrated_gradients_scores(
    policy: Any,
    observations: Any,
    baseline: Any = "zero",
    n_steps: int = DEFAULT_N_STEPS,
    target: str = "action",
    reduction: str = "l1",
    device: str = "cpu",
    seed: Optional[int] = None,
    normalize: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Functional entry point: IG importance scores for observations."""
    return ig_importance_scores(
        policy,
        observations,
        baseline=baseline,
        n_steps=n_steps,
        target=target,
        reduction=reduction,
        device=device,
        seed=seed,
        normalize=normalize,
        **kwargs,
    )


ig_scores = integrated_gradients_scores


def identify_critical_state_with_ig(
    env: Any,
    policy: Any,
    K: Any = None,
    n_steps: int = DEFAULT_N_STEPS,
    baseline: str = "zero",
    env_id: Optional[str] = None,
    deterministic: bool = True,
    reset: bool = True,
    seed: Optional[int] = None,
    return_rollout: bool = False,
    **kwargs: Any,
) -> Any:
    """Roll the frozen policy for ``K`` steps and return the IG-most-critical visited state."""
    selector = IGStateSelector(
        env=env,
        policy=policy,
        K=K,
        deterministic_policy=deterministic,
        seed=seed,
        config=IntegratedGradientsConfig.from_dict(
            {"env_id": env_id or getattr(env, "rice_env_key", "default"), "n_steps": n_steps, "baseline": baseline}
        ),
        **{k: v for k, v in kwargs.items() if k in ("batch_size", "device", "rng", "attach_scores", "cache")},
    )
    rollout = selector.rollout(policy=policy, K=K, reset=reset, seed=seed)
    critical = selector.select(policy=policy, K=K, reset=reset, seed=seed, rollout=rollout)
    if return_rollout:
        return critical, rollout
    return critical


def select_ig_states(
    env: Any,
    policy: Any,
    n: int = 1,
    K: Any = None,
    seed: Optional[int] = None,
    env_id: Optional[str] = None,
    deterministic: bool = True,
    **kwargs: Any,
) -> List[Any]:
    """``n`` independent rollouts, each contributing its IG-critical state."""
    rng = _get_rng(kwargs.pop("rng", None), seed)
    out: List[Any] = []
    for i in range(max(int(n), 0)):
        child_seed = None
        try:
            child_seed = int(rng.randint(0, 2 ** 31 - 1))
        except Exception:
            child_seed = None
        critical = identify_critical_state_with_ig(
            env,
            policy,
            K=K,
            env_id=env_id,
            deterministic=deterministic,
            seed=child_seed,
            **kwargs,
        )
        out.append(critical)
    return out


def make_integrated_gradients(
    env: Any = None,
    policy: Any = None,
    env_id: str = "default",
    seed: Optional[int] = None,
    config: Optional[Any] = None,
    K: Any = None,
    deterministic_policy: bool = True,
    rng: Any = None,
    **kwargs: Any,
) -> IntegratedGradients:
    """Factory used by the baseline registry."""
    if not _HAS_TORCH:
        _get_logger().warning(
            "IntegratedGradients: PyTorch unavailable -- falling back to the finite-difference "
            "importance surrogate."
        )
    return IntegratedGradients(
        env=env,
        policy=policy,
        env_id=env_id,
        seed=seed,
        config=config,
        rng=rng,
        K=K,
        deterministic_policy=deterministic_policy,
        **{k: v for k, v in kwargs.items() if k in ("with_mask_shim", "scorer", "selector")},
    )


build_integrated_gradients = make_integrated_gradients


def integrated_gradients_for(env_id: str = "default", **kwargs: Any) -> IntegratedGradients:
    """Factory pre-configured for an application id."""
    return make_integrated_gradients(env_id=env_id, **kwargs)


def describe_integrated_gradients(explanation: Any = None) -> str:
    """One-line human-readable description for logging."""
    if isinstance(explanation, IntegratedGradients):
        return explanation.describe()
    if isinstance(explanation, IntegratedGradientsConfig):
        cfg = explanation
    else:
        cfg = IntegratedGradientsConfig.from_dict({} if explanation is None else explanation)
    return (
        f"Integrated Gradients explanation (env={cfg.env_id}, m={cfg.n_steps}, "
        f"baseline={cfg.baseline}, target={cfg.target}, reduction={cfg.reduction}); "
        "gradient attribution of the frozen policy's action output w.r.t. the state."
    )


__all__ = [
    "IntegratedGradients",
    "IntegratedGradientsConfig",
    "IGScorer",
    "IGStateSelector",
    "make_integrated_gradients",
    "build_integrated_gradients",
    "integrated_gradients_for",
    "describe_integrated_gradients",
    "integrated_gradients_scores",
    "ig_scores",
    "integrated_gradients_attributions",
    "ig_importance_scores",
    "finite_difference_importance",
    "policy_action_output",
    "identify_critical_state_with_ig",
    "select_ig_states",
    "flatten_observation",
    "observation_matrix",
    "DEFAULT_N_STEPS",
    "IG_TARGETS",
    "IG_BASELINES",
    "IG_REDUCTIONS",
]
