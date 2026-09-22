"""State-importance scoring for the RICE explanation stage (Stage 1).

Paper reference (Section 3.3 "Step-level Explanation"):

    "By applying this resolved mask to each state, we will be able to assess the
     state importance (i.e., the probability of mask network outputting "0") at
     any time step."

So the importance of a visited state ``s_t`` under the trained mask network
``\\tilde{\\pi}_\\theta`` is simply::

    I(s_t) = P(a_t^m = 0 | s_t) = 1 - P(a_t^m = 1 | s_t)          ("keep")

i.e. the probability that the mask *keeps* the target agent's action at that
step.  A high score means "blinding the agent here would change the outcome",
i.e. the step is important for the final reward.

This module is deliberately *only* about scoring: it turns raw observations
(or a ``ResetWrapper`` trajectory / snapshot archive) into step-level importance
scores, and offers the usual aggregation/ranking/normalisation helpers that the
downstream consumers need:

* ``rice/explanation/critical_state.py`` -- rolls out the frozen policy for a
  length-K trajectory and picks the ``argmax``-importance state as the
  exploration frontier (Algorithm 2).
* ``rice/explanation/fidelity.py``      -- sliding-window (width ``l = L x K``)
  average-importance search used by Experiment I.
* ``rice/refining/mixed_init.py``       -- resets to the identified critical
  state with probability ``p``.

The heavy lifting (batched forward passes, softmax over the two mask logits) is
done by :func:`rice.explanation.mask_network.state_importance`; this module adds
the trajectory-aware API on top and never mutates the mask network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from rice.explanation.mask_network import (
    BLIND_INDEX,
    KEEP_INDEX,
    NUM_MASK_ACTIONS,
    MaskNetwork,
    flatten_observation,
    state_importance,
)
from rice.utils.logging import get_logger

logger = get_logger("rice.explanation.importance")

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "IMPORTANCE_MODES",
    "ImportanceScorer",
    "TrajectoryImportance",
    "aggregate_importance",
    "argmax_importance",
    "attach_scores_to_wrapper",
    "extract_observations",
    "importance_mode",
    "min_max_normalize",
    "most_important_index",
    "normalize_scores",
    "rank_states",
    "score_batch",
    "score_observations",
    "score_trajectory",
    "score_trajectory_summary",
    "softmax_normalize",
    "summarize_importance",
    "top_k_indices",
    "trajectory_importance",
]


DEFAULT_BATCH_SIZE = 4096
"""Chunk size used for batched mask-network forward passes."""

IMPORTANCE_MODES: Tuple[str, ...] = ("mean", "max", "sum", "last", "first", "topk_mean")
"""Supported trajectory-level aggregation modes."""


# ---------------------------------------------------------------------------
# Observation extraction helpers
# ---------------------------------------------------------------------------
def _to_obs_list(observations: Any) -> List[Any]:
    """Normalise many possible trajectory containers into a list of observations.

    Accepts: a list/tuple of observations, a numpy array of shape ``(T, obs_dim)``,
    a ``ResetWrapper``, an object exposing ``observations`` / ``trajectory`` /
    ``obs`` / ``states``, or a list of ``StateSnapshot`` objects.
    """
    if observations is None:
        return []

    # ResetWrapper-ish containers: expose `.observations`
    for attr in ("observations", "obs", "states", "trajectory", "trajectories"):
        if not isinstance(observations, (list, tuple, np.ndarray)) and hasattr(observations, attr):
            candidate = getattr(observations, attr)
            if candidate is not None:
                return _to_obs_list(candidate)

    if isinstance(observations, np.ndarray):
        if observations.ndim <= 1:
            return [observations]
        return [observations[i] for i in range(observations.shape[0])]

    if isinstance(observations, (list, tuple)):
        out: List[Any] = []
        for item in observations:
            # StateSnapshot support
            if hasattr(item, "observation") and not isinstance(item, np.ndarray):
                out.append(getattr(item, "observation"))
            else:
                out.append(item)
        return out

    # A single observation
    return [observations]


def extract_observations(trajectory: Any) -> Union[np.ndarray, List[Any]]:
    """Return the observation sequence contained in ``trajectory``.

    ``trajectory`` may be a list of observations, a ``(T, obs_dim)`` array, a
    ``ResetWrapper``/``Logger``-like object exposing ``observations``, or a list
    of ``StateSnapshot``s.  Returns an ``np.ndarray`` when the observations are
    uniformly stackable, otherwise the raw list.
    """
    obs_list = _to_obs_list(trajectory)
    if not obs_list:
        return np.zeros((0,), dtype=np.float32)
    try:
        return np.asarray([flatten_observation(o) for o in obs_list], dtype=np.float32)
    except Exception:  # pragma: no cover - dict/pixel observations etc.
        return obs_list


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------
def score_observations(
    mask_net: Optional[MaskNetwork],
    observations: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
) -> np.ndarray:
    """Score observations with the mask network -> ``P(a^m = 0 | s)`` in ``[0, 1]``.

    Parameters
    ----------
    mask_net:
        Trained mask network (``\\tilde{\\pi}_\\theta``).  If ``None`` the scores
        degenerate to all ones (equivalent to "every state is important"), which
        keeps the baselines (e.g. Random) runnable without a mask net.
    observations:
        Any container accepted by :func:`extract_observations`.
    batch_size:
        Forward-pass chunk size.
    device:
        Optional device override; defaults to the mask network's device.

    Returns
    -------
    np.ndarray of shape ``(T,)`` with ``P(keep)`` per step.
    """
    obs = extract_observations(observations)
    obs_array = np.asarray(obs, dtype=np.float32) if not isinstance(obs, list) else obs
    n = len(obs_array) if not isinstance(obs_array, np.ndarray) else obs_array.shape[0]

    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    if mask_net is None:
        return np.ones((n,), dtype=np.float32)

    scores = state_importance(mask_net, obs_array, batch_size=batch_size)
    return np.asarray(scores, dtype=np.float32).reshape(-1)


def score_batch(
    mask_net: Optional[MaskNetwork],
    observations: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Alias of :func:`score_observations` (explicit batch-scoring name)."""
    return score_observations(mask_net, observations, batch_size=batch_size)


def score_trajectory(
    mask_net: Optional[MaskNetwork],
    trajectory: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Score every state of a trajectory produced by the frozen policy ``\\pi``.

    ``trajectory`` may be:
      * a ``ResetWrapper`` (its ``observations`` are used),
      * an ``(T, obs_dim)`` array,
      * a list of observations or ``StateSnapshot`` objects.

    Returns an ``np.ndarray`` of ``P(keep)`` values, one per visited state.
    """
    return score_observations(mask_net, trajectory, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Aggregation / ranking
# ---------------------------------------------------------------------------
def importance_mode(mode: Optional[str]) -> str:
    """Validate and normalise an aggregation mode string."""
    if mode is None:
        return "mean"
    m = str(mode).strip().lower()
    if m not in IMPORTANCE_MODES:
        raise ValueError(f"Unknown importance mode {mode!r}; expected one of {IMPORTANCE_MODES}")
    return m


def aggregate_importance(scores: Sequence[float], mode: str = "mean", topk_frac: float = 0.1) -> float:
    """Aggregate step-level scores into a single trajectory-level importance.

    Modes: ``mean``, ``max``, ``sum``, ``last``, ``first``, ``topk_mean``
    (mean of the highest ``topk_frac`` fraction of steps).
    """
    arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return float("nan")
    m = importance_mode(mode)
    if m == "mean":
        return float(arr.mean())
    if m == "max":
        return float(arr.max())
    if m == "sum":
        return float(arr.sum())
    if m == "last":
        return float(arr[-1])
    if m == "first":
        return float(arr[0])
    # topk_mean
    frac = float(np.clip(topk_frac, 1e-6, 1.0))
    k = max(1, int(round(frac * arr.size)))
    part = np.sort(arr)[::-1][:k]
    return float(part.mean())


def trajectory_importance(
    mask_net: Optional[MaskNetwork],
    trajectory: Any,
    mode: str = "mean",
    batch_size: int = DEFAULT_BATCH_SIZE,
    topk_frac: float = 0.1,
) -> float:
    """Convenience: score ``trajectory`` and aggregate it to a single number."""
    scores = score_trajectory(mask_net, trajectory, batch_size=batch_size)
    return aggregate_importance(scores, mode=mode, topk_frac=topk_frac)


def argmax_importance(scores: Sequence[float]) -> int:
    """Index of the most important step (ties broken by the earliest step).

    The paper's critical-state rule is ``argmax`` of the mask-based score, so we
    use ``np.argmax`` semantics (first maximum) for deterministic seeds.
    """
    arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return -1
    if not np.isfinite(arr).any():
        return 0
    return int(np.nanargmax(arr))


def most_important_index(scores: Sequence[float]) -> int:
    """Alias of :func:`argmax_importance` (readable name for Algorithm 2)."""
    return argmax_importance(scores)


def rank_states(scores: Sequence[float], descending: bool = True) -> np.ndarray:
    """Return step indices ordered by importance.

    Default order is ``descending`` (highest importance first).  Ties are broken
    by ascending step index thanks to NumPy's stable sort with ``mergesort``.
    """
    arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    filled = np.where(np.isfinite(arr), arr, -np.inf)
    order = np.argsort(-filled if descending else filled, kind="mergesort")
    return order.astype(np.int64)


def top_k_indices(scores: Sequence[float], k: int = 10) -> np.ndarray:
    """Decreasing-importance step indices, clipped to ``k`` (or all if ``k<=0``)."""
    order = rank_states(list(scores), descending=True)
    if order.size == 0:
        return order
    if k is None or k <= 0:
        return order
    return order[: min(int(k), order.size)]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def min_max_normalize(scores: Sequence[float], eps: float = 1e-8) -> np.ndarray:
    """Rescale scores to ``[0, 1]``; constant inputs map to all zeros."""
    arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= eps:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo + eps)


def softmax_normalize(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Temperature-softmax of the scores (sums to 1).  Useful as sampling weights."""
    arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    t = max(float(temperature), 1e-8)
    z = (arr - float(np.nanmax(arr))) / t
    e = np.exp(z)
    denom = e.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(arr, 1.0 / arr.size)
    return e / denom


def normalize_scores(scores: Sequence[float], method: str = "minmax", **kwargs) -> np.ndarray:
    """Dispatch to ``minmax``, ``softmax`` or ``none`` normalisation."""
    m = str(method or "none").strip().lower()
    if m in ("minmax", "min_max", "min-max"):
        return min_max_normalize(scores, **kwargs)
    if m == "softmax":
        return softmax_normalize(scores, **kwargs)
    return np.asarray(list(scores), dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize_importance(scores: Sequence[float], threshold: Optional[float] = None) -> Dict[str, float]:
    """Descriptive statistics of a step-level importance vector."""
    arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "argmax": -1.0,
            "median": float("nan"),
            "above_threshold": 0.0,
        }
    out = {
        "n": float(arr.size),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "argmax": float(argmax_importance(arr)),
        "median": float(np.nanmedian(arr)),
    }
    thr = 0.5 if threshold is None else float(threshold)
    out["above_threshold"] = float(np.sum(arr > thr))
    out["threshold"] = thr
    return out


def score_trajectory_summary(
    mask_net: Optional[MaskNetwork],
    trajectory: Any,
    mode: str = "mean",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Score a trajectory and return both the raw scores and their summary."""
    scores = score_trajectory(mask_net, trajectory, batch_size=batch_size)
    summary = summarize_importance(scores)
    summary["aggregate"] = aggregate_importance(scores, mode=mode)
    summary["mode"] = importance_mode(mode)
    summary["scores"] = scores
    return summary


def attach_scores_to_wrapper(wrapper: Any, scores: Sequence[float]) -> Any:
    """Attach step-level scores to a ``ResetWrapper``'s snapshots (in place).

    Algorithm 2 needs to correlate the critical state found by the mask with a
    *restorable* snapshot; this back-fills ``StateSnapshot.score`` so that
    ``wrapper.best_snapshot(scores=...)`` / ``reset_to_critical()`` work.
    """
    arr = np.asarray(list(scores), dtype=np.float32).reshape(-1)
    snapshots = getattr(wrapper, "snapshots", None)
    if snapshots is None:
        inner = getattr(wrapper, "env", None)
        snapshots = getattr(inner, "snapshots", None)
    if snapshots is None:
        return wrapper

    n = min(len(snapshots), arr.size)
    for i in range(n):
        snap = snapshots[i]
        if hasattr(snap, "score"):
            snap.score = float(arr[i])
        elif isinstance(snap, dict):
            snap["score"] = float(arr[i])
    try:
        wrapper.last_importance_scores = arr
    except Exception:  # pragma: no cover - duck-typed wrappers
        pass
    return wrapper


# ---------------------------------------------------------------------------
# Object-oriented facade
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryImportance:
    """Container bundling a trajectory's step-level scores with its metadata."""

    scores: np.ndarray
    observations: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    mode: str = "mean"
    aggregate: float = float("nan")
    critical_index: int = -1

    def __len__(self) -> int:
        return int(self.scores.size)

    def top_k(self, k: int = 10) -> np.ndarray:
        """Top-k step indices by importance (descending)."""
        return top_k_indices(self.scores, k=k)

    def to_dict(self, include_scores: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "mode": self.mode,
            "aggregate": float(self.aggregate),
            "critical_index": int(self.critical_index),
            "n": int(self.scores.size),
            "summary": summarize_importance(self.scores),
        }
        if include_scores:
            d["scores"] = np.asarray(self.scores, dtype=np.float32).tolist()
            if self.indices is not None:
                d["indices"] = np.asarray(self.indices).astype(int).tolist()
        return d


@dataclass
class ImportanceScorer:
    """Score states / trajectories with a trained mask network.

    Example
    -------
    >>> scorer = ImportanceScorer(mask_net)
    >>> traj = scorer.score_trajectory(observations)
    >>> traj.critical_index          # argmax state importance
    >>> scorer.most_important_state(observations)

    Parameters
    ----------
    mask_net:
        Trained ``\\tilde{\\pi}_\\theta`` (or ``None`` -> uniform scores).
    batch_size:
        Chunk size for batched forward passes.
    mode:
        Default trajectory-level aggregation mode.
    normalize:
        Optional normalisation applied to returned scores (``"minmax"`` /
        ``"softmax"`` / ``None``); ``None`` keeps the raw ``P(keep)`` values,
        which is what the fidelity score and the ``argmax`` selection require.
    """

    mask_net: Optional[MaskNetwork] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    mode: str = "mean"
    normalize: Optional[str] = None
    _stats: Dict[str, float] = field(default_factory=dict, repr=False)

    # -- scoring ----------------------------------------------------------
    def score(self, observations: Any) -> np.ndarray:
        """Score a batch of observations -> ``P(keep)``."""
        scores = score_observations(self.mask_net, observations, batch_size=self.batch_size)
        if self.normalize:
            scores = normalize_scores(scores, method=self.normalize)
        return scores

    def score_trajectory(self, trajectory: Any) -> TrajectoryImportance:
        """Score a whole trajectory (list/array/``ResetWrapper``)."""
        obs = extract_observations(trajectory)
        scores = self.score(obs)
        agg = aggregate_importance(scores, mode=self.mode)
        idx = argmax_importance(scores)
        return TrajectoryImportance(
            scores=scores,
            observations=obs if isinstance(obs, np.ndarray) else None,
            indices=np.arange(len(scores), dtype=np.int64),
            mode=importance_mode(self.mode),
            aggregate=agg,
            critical_index=idx,
        )

    # -- selection --------------------------------------------------------
    def most_important_state(self, trajectory: Any) -> Tuple[int, Any]:
        """Return ``(index, observation)`` of the highest-importance state."""
        obs = _to_obs_list(trajectory)
        scores = self.score(obs)
        idx = argmax_importance(scores)
        if idx < 0:
            return -1, None
        return idx, obs[idx]

    def most_important_index(self, trajectory: Any) -> int:
        """Index (within ``trajectory``) of the argmax-importance state."""
        return argmax_importance(self.score(_to_obs_list(trajectory)))

    def top_k_states(self, trajectory: Any, k: int = 10) -> List[Tuple[int, Any]]:
        """``[(index, observation), ...]`` ordered by decreasing importance."""
        obs = _to_obs_list(trajectory)
        scores = self.score(obs)
        return [(int(i), obs[int(i)]) for i in top_k_indices(scores, k=k)]

    def ranked_observations(self, trajectory: Any) -> List[Any]:
        """Observations sorted by decreasing importance (fidelity's top-K use)."""
        obs = _to_obs_list(trajectory)
        scores = self.score(obs)
        return [obs[int(i)] for i in rank_states(scores, descending=True)]

    # -- aggregation ------------------------------------------------------
    def trajectory_aggregate(self, trajectory: Any, mode: Optional[str] = None) -> float:
        """Aggregate a trajectory's importance (default: this scorer's mode)."""
        scores = self.score(_to_obs_list(trajectory))
        return aggregate_importance(scores, mode=mode or self.mode)

    def summary(self, trajectory: Any) -> Dict[str, Any]:
        """Full summary (statistics + aggregate + critical index) of a trajectory."""
        traj = self.score_trajectory(trajectory)
        out = summarize_importance(traj.scores)
        out.update(
            {
                "aggregate": float(traj.aggregate),
                "mode": traj.mode,
                "critical_index": int(traj.critical_index),
            }
        )
        self._stats = out
        return out

    # -- integration helpers ---------------------------------------------
    def attach_to_wrapper(self, wrapper: Any, trajectory: Any = None) -> Any:
        """Score ``trajectory`` (defaults to ``wrapper.observations``) and attach."""
        obs = extract_observations(wrapper if trajectory is None else trajectory)
        return attach_scores_to_wrapper(wrapper, self.score(obs))

    @property
    def stats(self) -> Dict[str, float]:
        """Statistics from the most recent :meth:`summary` call."""
        return dict(self._stats)


def make_scorer(mask_net: Optional[MaskNetwork], **kwargs) -> ImportanceScorer:
    """Convenience factory for :class:`ImportanceScorer`."""
    return ImportanceScorer(mask_net=mask_net, **kwargs)
