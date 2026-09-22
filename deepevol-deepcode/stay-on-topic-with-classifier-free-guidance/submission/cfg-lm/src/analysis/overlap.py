"""Section 5.2 / Appendix E -- CFG's relation to instruction tuning.

This module implements the second half of the Section 5 analysis of
*Stay on Topic with Classifier-Free Guidance*: a token-level comparison of the
vocabulary distributions produced by

* ``cfg``        -- the guided distribution ``P_cfg(y|x)`` at gamma = 1.5,
* ``vanilla``    -- the vanilla prompted distribution ``P(y|x)`` (gamma = 1),
* ``unprompted`` -- the unprompted distribution ``P(x)`` (gamma = 0 / empty prefix),
* ``instruct``   -- an instruction-tuned model ``P_instruct(y|x)``.

The paper measures the *minimum* top-p = 90% nucleus (the smallest set of tokens
whose cumulative probability reaches 0.9) at every completion step, then reports

1. the fraction of that set which is shared between two distributions
   (:func:`overlap_fraction`, :func:`top_p_token_overlap`).  Figure 18b shows CFG
   shares *roughly 50%* of the tokens in its top-p = 0.9 nucleus with the vanilla
   prompted model, and that Instruction-Tuned and CFG distributions are "largely
   not overlapping" (Fig. 19 also shows vanilla prompting is *more* similar to
   instruction-tuning than CFG is);
2. Spearman rank correlations ``r_s`` between the token-overlap signal of CFG and
   that of an instruction-tuned model, which reach ``r_s > .7`` for harder /
   longer prompts (Table 5 and Appendix E Tables 12-14).

Everything in the pure-math layer is NumPy/SciPy only, so the analysis is
unit-testable on CPU without a model; ``torch`` (needed for the actual forward
passes) and ``scipy`` (exact ``spearmanr``) are optional imports.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # package-relative import first, plain ``src.`` fallback for scripts
    from ..cfg.logits import log_softmax, softmax  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    try:
        from src.cfg.logits import log_softmax, softmax  # type: ignore
    except Exception:  # pragma: no cover
        log_softmax = None  # type: ignore
        softmax = None  # type: ignore

try:  # exact scipy implementation of the paper-cited rank correlation
    from scipy.stats import spearmanr as _scipy_spearmanr  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_spearmanr = None  # type: ignore
    _HAS_SCIPY = False

try:  # torch is only needed for the real model path
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants (paper values)
# --------------------------------------------------------------------------- #

#: Nucleus threshold used for the overlap analysis in Section 5.2 / Figure 18b.
TOP_P = 0.9

#: Guidance strength used for the CFG distribution throughout Section 5.
ANALYSIS_GAMMA = 1.5

#: Figure 18b: "CFG shares roughly 50% of the tokens in top-p = 0.9 as the
#: vanilla P(y | x) model."
CFG_VANILLA_OVERLAP = 0.5

#: Section 5.2 / Appendix E: "significant spearman correlations of r_s > .7
#: between Instruction-Tuned models and CFG" (particularly for longer prompts).
SPEARMAN_THRESHOLD = 0.7

#: Tolerance accepted when comparing a measured overlap against the paper anchor.
OVERLAP_TOLERANCE = 0.1

#: The four distributions compared in Figure 18 / Appendix E.
OVERLAP_MODES: Tuple[str, ...] = ("cfg", "vanilla", "unprompted", "instruct")

#: Mode -> gamma mapping (mirrors ``entropy.MODE_GAMMAS``).
MODE_GAMMAS: Dict[str, float] = {
    "cfg": ANALYSIS_GAMMA,
    "vanilla": 1.0,
    "unprompted": 0.0,
    "instruct": 1.0,
}

#: Display labels for the plain-text report.
MODE_LABELS: Dict[str, str] = {
    "cfg": "CFG (gamma=1.5)",
    "vanilla": "Vanilla P(y|x)",
    "unprompted": "Unprompted P(x)",
    "instruct": "Instruction-tuned",
}


# --------------------------------------------------------------------------- #
# Low-level numeric helpers
# --------------------------------------------------------------------------- #


def _as_numpy(x: Any) -> np.ndarray:
    """Detach/convert a tensor-like object to a float NumPy array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):  # pragma: no cover - torch path
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable NumPy softmax (reuses :mod:`logits` when possible)."""
    if softmax is not None and _HAS_TORCH and isinstance(x, torch.Tensor):  # pragma: no cover
        return _as_numpy(softmax(x, axis=axis))
    arr = _as_numpy(x).astype(np.float64)
    arr = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(arr)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _last_row(logits: Any) -> np.ndarray:
    """Return the ``[vocab]`` next-token logits from a forward-pass output."""
    arr = _as_numpy(logits)
    if arr.ndim == 3:  # [batch, seq, vocab]
        arr = arr[0, -1]
    elif arr.ndim == 2:  # [seq, vocab] or [batch, vocab]
        arr = arr[-1]
    return np.asarray(arr, dtype=np.float64).ravel()


def guided_distribution(
    logits_cond: Any,
    logits_uncond: Optional[Any] = None,
    gamma: float = ANALYSIS_GAMMA,
    temperature: float = 1.0,
) -> np.ndarray:
    """The post-CFG vocabulary distribution of Section 2.2 (Eq. 7).

    ``p = softmax(uncond + gamma * (cond - uncond))`` computed in logit space
    *before* any softmax, with ``gamma = 1`` reducing exactly to ``cond`` and
    ``gamma = 0`` to ``uncond``.
    """
    cond = _last_row(logits_cond)
    if logits_uncond is None or float(gamma) == 1.0:
        guided = cond
    else:
        uncond = _last_row(logits_uncond)
        if float(gamma) == 0.0:
            guided = uncond
        else:
            guided = uncond + float(gamma) * (cond - uncond)
    if temperature is not None and float(temperature) != 1.0:
        guided = guided / float(temperature)
    return _softmax_np(guided, axis=-1)


# backwards/forwards compatible private aliases
_guided_distribution = guided_distribution


# --------------------------------------------------------------------------- #
# Top-p (nucleus) token sets
# --------------------------------------------------------------------------- #


def top_p_token_set(probs: Any, top_p: float = TOP_P) -> frozenset:
    """Smallest nucleus ``top_p``: tokens covering ``top_p`` of the mass.

    Sorts descending, cumulates, and keeps the tokens up to and including the
    first one that crosses ``top_p`` (HuggingFace ``TopPLogitsWarper``
    semantics), so the set is the *minimum* nucleus and always non-empty.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    if p.size == 0:
        return frozenset()
    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    csum = np.cumsum(sorted_p)
    k = int(np.searchsorted(csum, float(top_p), side="left"))
    k = min(k + 1, sorted_p.size)  # include the crossing token
    k = max(k, 1)
    return frozenset(order[:k].tolist())


def top_p_token_sets(probs: Any, top_p: float = TOP_P) -> List[frozenset]:
    """Vectorised :func:`top_p_token_set` over ``[..., vocab]`` probabilities."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim == 1:
        return [top_p_token_set(arr, top_p=top_p)]
    flat = arr.reshape(-1, arr.shape[-1])
    return [top_p_token_set(row, top_p=top_p) for row in flat]


def top_p_token_count(probs: Any, top_p: float = TOP_P) -> np.ndarray:
    """Number of tokens in the minimum nucleus of every distribution."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim == 1:
        return np.asarray([len(top_p_token_set(arr, top_p=top_p))], dtype=np.int64)
    flat = arr.reshape(-1, arr.shape[-1])
    return np.asarray([len(top_p_token_set(row, top_p=top_p)) for row in flat], dtype=np.int64)


def overlap_fraction(
    set_a: Iterable[int],
    set_b: Iterable[int],
    denominator: str = "first",
) -> float:
    """Fraction of ``set_a`` (or the min/union) that is shared with ``set_b``.

    ``denominator`` selects the normalisation:

    * ``"first"``  -> ``|a & b| / |a|``  (the paper's phrasing: "CFG shares
      roughly 50% of the tokens in top-p = 0.9 as the vanilla model");
    * ``"second"`` -> ``|a & b| / |b|``;
    * ``"min"``    -> ``|a & b| / min(|a|, |b|)`` (symmetric containment);
    * ``"union"``  -> Jaccard index ``|a & b| / |a | b|``.
    """
    a = set(int(t) for t in set_a)
    b = set(int(t) for t in set_b)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if denominator == "first":
        denom = len(a)
    elif denominator == "second":
        denom = len(b)
    elif denominator == "min":
        denom = min(len(a), len(b))
    elif denominator == "union":
        denom = len(a | b)
    else:
        raise ValueError(
            f"unknown denominator {denominator!r}; expected one of "
            "'first', 'second', 'min', 'union'"
        )
    return float(inter) / float(denom)


def jaccard(set_a: Iterable[int], set_b: Iterable[int]) -> float:
    """Jaccard index between two token sets."""
    return overlap_fraction(set_a, set_b, denominator="union")


def intersection_size(set_a: Iterable[int], set_b: Iterable[int]) -> int:
    """``|a & b|`` for two token sets (the raw token count in Figure 18b)."""
    return len(set(int(t) for t in set_a) & set(int(t) for t in set_b))


def top_p_token_overlap(
    probs_a: Any,
    probs_b: Any,
    top_p: float = TOP_P,
    denominator: str = "first",
) -> float:
    """Top-p overlap between two (post-CFG) vocabulary distributions."""
    return overlap_fraction(
        top_p_token_set(probs_a, top_p=top_p),
        top_p_token_set(probs_b, top_p=top_p),
        denominator=denominator,
    )


# --------------------------------------------------------------------------- #
# Rank correlation (Spearman) -- Section 5.2 / Appendix E
# --------------------------------------------------------------------------- #


def rankdata(x: Sequence[float]) -> np.ndarray:
    """Average-rank transform (ties get the mean of their ranks)."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = arr.size
    if n == 0:
        return arr
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_arr = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation (``nan`` for degenerate inputs)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if n < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation ``r_s`` (exact ``scipy.stats.spearmanr`` when
    SciPy is installed, otherwise the rank/Pearson identity)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if n < 2:
        return float("nan")
    if _HAS_SCIPY:  # pragma: no cover - depends on environment
        try:
            rho = _scipy_spearmanr(a, b).correlation
            return float(rho)
        except Exception:  # fall through to the manual implementation
            pass
    return pearson(rankdata(a), rankdata(b))


def spearman_with_pvalue(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """``(r_s, p)`` for two equal-length sequences."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if n < 3:
        return spearman_correlation(a, b), float("nan")
    if _HAS_SCIPY:  # pragma: no cover
        try:
            res = _scipy_spearmanr(a, b)
            return float(res.correlation), float(res.pvalue)
        except Exception:
            pass
    rho = pearson(rankdata(a), rankdata(b))
    # large-sample t-approximation, only used without SciPy
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        return rho, 0.0 if abs(rho) >= 1.0 else float("nan")
    t = rho * math.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho))
    # two-sided p via normal approximation of Student's t
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return rho, float(p)


# --------------------------------------------------------------------------- #
# Per-token / per-sample statistics containers
# --------------------------------------------------------------------------- #


@dataclass
class TopPSetStats:
    """Per-completion-token minimum top-p nuclei for one mode.

    Fields
    ------
    sets:
        One ``frozenset`` of token ids per completion token (flattened over
        samples).
    name, gamma, top_p:
        Bookkeeping / book-keeping for reports.
    sizes:
        Nucleus size per token (``|top-p = 0.9|``, cf. Appendix E).
    positions:
        Index of each token *within its completion* (for the Figure 19
        word-index analysis); ``None`` when not tracked.
    sample_ids:
        Index of the P3 datapoint each token came from.
    per_entropy:
        Optional per-token entropy of the same distribution.
    """

    sets: List[frozenset] = field(default_factory=list)
    name: str = "cfg"
    gamma: float = ANALYSIS_GAMMA
    top_p: float = TOP_P
    sizes: Optional[np.ndarray] = None
    positions: Optional[np.ndarray] = None
    sample_ids: Optional[np.ndarray] = None
    per_entropy: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.sets = [frozenset(int(t) for t in s) for s in self.sets]
        if self.sizes is None:
            self.sizes = np.asarray([len(s) for s in self.sets], dtype=np.int64)
        else:
            self.sizes = np.asarray(self.sizes, dtype=np.int64)
        if self.positions is not None:
            self.positions = np.asarray(self.positions, dtype=np.int64)
        if self.sample_ids is not None:
            self.sample_ids = np.asarray(self.sample_ids, dtype=np.int64)
        if self.per_entropy is not None:
            self.per_entropy = np.asarray(self.per_entropy, dtype=np.float64)

    # -- descriptive statistics ------------------------------------------- #
    def __len__(self) -> int:
        return len(self.sets)

    @property
    def n_tokens(self) -> int:
        return len(self.sets)

    @property
    def mean_size(self) -> float:
        if not self.sets:
            return float("nan")
        return float(np.mean(self.sizes))

    @property
    def std_size(self) -> float:
        if not self.sets:
            return float("nan")
        return float(np.std(self.sizes))

    @property
    def mean_entropy(self) -> float:
        if self.per_entropy is None or self.per_entropy.size == 0:
            return float("nan")
        return float(np.mean(self.per_entropy))

    def by_sample(self) -> Dict[int, List[frozenset]]:
        """Group token sets by originating datapoint."""
        out: Dict[int, List[frozenset]] = {}
        for i, s in enumerate(self.sets):
            sid = int(self.sample_ids[i]) if self.sample_ids is not None else 0
            out.setdefault(sid, []).append(s)
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gamma": float(self.gamma),
            "top_p": float(self.top_p),
            "n_tokens": self.n_tokens,
            "mean_size": self.mean_size,
            "std_size": self.std_size,
            "mean_entropy": self.mean_entropy,
        }


@dataclass
class OverlapStats:
    """Per-token top-p overlap of one mode against a reference mode.

    ``per_token[i]`` is :func:`overlap_fraction` of the ``i``-th completion token
    of ``name`` against the corresponding token of ``reference``.
    """

    per_token: np.ndarray = field(default_factory=lambda: np.zeros(0))
    name: str = "cfg"
    reference: str = "vanilla"
    gamma: float = ANALYSIS_GAMMA
    top_p: float = TOP_P
    denominator: str = "first"
    sizes_a: Optional[np.ndarray] = None
    sizes_b: Optional[np.ndarray] = None
    positions: Optional[np.ndarray] = None
    sample_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.per_token = np.asarray(self.per_token, dtype=np.float64).ravel()
        for attr in ("sizes_a", "sizes_b", "positions", "sample_ids"):
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, np.asarray(val))

    def __len__(self) -> int:
        return int(self.per_token.size)

    @property
    def n_tokens(self) -> int:
        return int(self.per_token.size)

    @property
    def mean(self) -> float:
        if self.per_token.size == 0:
            return float("nan")
        return float(np.mean(self.per_token))

    @property
    def std(self) -> float:
        if self.per_token.size == 0:
            return float("nan")
        return float(np.std(self.per_token))

    @property
    def median(self) -> float:
        if self.per_token.size == 0:
            return float("nan")
        return float(np.median(self.per_token))

    @property
    def min(self) -> float:
        return float(np.min(self.per_token)) if self.per_token.size else float("nan")

    @property
    def max(self) -> float:
        return float(np.max(self.per_token)) if self.per_token.size else float("nan")

    @property
    def sem(self) -> float:
        n = self.per_token.size
        if n < 2:
            return float("nan")
        return float(np.std(self.per_token, ddof=1) / math.sqrt(n))

    @property
    def fraction_below_half(self) -> float:
        """Share of tokens whose overlap is under 0.5 (Figure 18b tail)."""
        if self.per_token.size == 0:
            return float("nan")
        return float(np.mean(self.per_token < 0.5))

    def by_index(self, max_index: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Mean/std/count of overlap per *position within the completion*.

        This is the per-word-index curve of Figure 19 (Appendix E).
        """
        if self.positions is None or self.per_token.size == 0:
            return {
                "index": np.zeros(0, dtype=np.int64),
                "mean": np.zeros(0),
                "std": np.zeros(0),
                "count": np.zeros(0, dtype=np.int64),
            }
        pos = np.asarray(self.positions, dtype=np.int64)
        if max_index is not None:
            mask = pos <= int(max_index)
            pos = pos[mask]
        vals = self.per_token[: pos.size] if max_index is not None else self.per_token
        # recompute mask consistently
        if max_index is not None:
            full_pos = np.asarray(self.positions, dtype=np.int64)
            vals = self.per_token[full_pos <= int(max_index)]
        idx = np.unique(pos)
        means = np.asarray([vals[pos == i].mean() for i in idx], dtype=np.float64)
        stds = np.asarray(
            [vals[pos == i].std() if (pos == i).sum() > 1 else 0.0 for i in idx],
            dtype=np.float64,
        )
        counts = np.asarray([int((pos == i).sum()) for i in idx], dtype=np.int64)
        return {"index": idx, "mean": means, "std": stds, "count": counts}

    def by_sample(self) -> Dict[int, float]:
        """Mean overlap per originating datapoint (P3 subset item)."""
        if self.sample_ids is None or self.per_token.size == 0:
            return {}
        out: Dict[int, List[float]] = {}
        for sid, val in zip(np.asarray(self.sample_ids, dtype=np.int64), self.per_token):
            out.setdefault(int(sid), []).append(float(val))
        return {k: float(np.mean(v)) for k, v in out.items()}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "reference": self.reference,
            "gamma": float(self.gamma),
            "top_p": float(self.top_p),
            "denominator": self.denominator,
            "n_tokens": self.n_tokens,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "sem": self.sem,
            "fraction_below_half": self.fraction_below_half,
        }


@dataclass
class OverlapComparison:
    """All top-p overlap measurements of one Section 5.2 comparison run."""

    stats: Dict[str, OverlapStats] = field(default_factory=dict)
    mode_stats: Dict[str, TopPSetStats] = field(default_factory=dict)
    reference: str = "vanilla"
    n: int = 0
    gamma: float = ANALYSIS_GAMMA
    top_p: float = TOP_P
    spearman_vs_instruct: Dict[str, float] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        """Headline number: CFG (or primary mode) vs reference top-p overlap.

        Figure 18b anchor: ``approx 0.5`` for CFG vs the vanilla prompted model.
        """
        if "cfg" in self.stats:
            return self.stats["cfg"].mean
        for name, st in self.stats.items():
            if name != self.reference:
                return st.mean
        return float("nan")

    @property
    def delta(self) -> float:
        """``mean - CFG_VANILLA_OVERLAP`` (how far from the paper anchor)."""
        return self.mean - CFG_VANILLA_OVERLAP

    def overlap_of(self, mode: str) -> Optional[OverlapStats]:
        return self.stats.get(mode)

    def check_against_paper(self, tolerance: float = OVERLAP_TOLERANCE) -> Dict[str, Any]:
        """Verify the Figure 18b anchor (CFG/vanilla top-p overlap ~ 50%)."""
        mean = self.mean
        within = bool(np.isfinite(mean) and abs(mean - CFG_VANILLA_OVERLAP) <= tolerance)
        out: Dict[str, Any] = {
            "measured_overlap": mean,
            "expected_overlap": CFG_VANILLA_OVERLAP,
            "tolerance": float(tolerance),
            "within_tolerance": within,
            "reference": self.reference,
            "n_tokens": int(self.stats["cfg"].n_tokens) if "cfg" in self.stats else 0,
        }
        # Appendix E: vanilla prompting is *more* similar to instruction-tuning
        # than CFG is.
        if "instruct" in self.stats:
            out["overlap_vs_instruct"] = {
                m: st.mean for m, st in self.stats.items() if m != "instruct"
            }
        if "instruct" in self.stats:
            out["vanilla_closer_to_instruct_than_cfg"] = bool(
                "vanilla" in self.stats
                and np.isfinite(self.stats["vanilla"].mean)
                and np.isfinite(self.stats["instruct"].mean)
                and self.stats["vanilla"].mean > self.stats["instruct"].mean
            )
        if self.spearman_vs_instruct:
            out["spearman_vs_instruct"] = dict(self.spearman_vs_instruct)
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "n": int(self.n),
            "gamma": float(self.gamma),
            "top_p": float(self.top_p),
            "mean_overlap": self.mean,
            "stats": {k: v.as_dict() for k, v in self.stats.items()},
            "mode_stats": {k: v.as_dict() for k, v in self.mode_stats.items()},
            "spearman_vs_instruct": dict(self.spearman_vs_instruct),
        }


# --------------------------------------------------------------------------- #
# Overlap between a pair of per-token top-p set sequences
# --------------------------------------------------------------------------- #


def compare_token_sets(
    sets_a: Sequence[frozenset],
    sets_b: Sequence[frozenset],
    name: str = "cfg",
    reference: str = "vanilla",
    gamma: float = ANALYSIS_GAMMA,
    top_p: float = TOP_P,
    denominator: str = "first",
    positions: Optional[Sequence[int]] = None,
    sample_ids: Optional[Sequence[int]] = None,
) -> OverlapStats:
    """Per-token top-p overlap between two aligned token-set sequences."""
    n = min(len(sets_a), len(sets_b))
    per_token = np.asarray(
        [
            overlap_fraction(sets_a[i], sets_b[i], denominator=denominator)
            for i in range(n)
        ],
        dtype=np.float64,
    )
    return OverlapStats(
        per_token=per_token,
        name=name,
        reference=reference,
        gamma=gamma,
        top_p=top_p,
        denominator=denominator,
        sizes_a=np.asarray([len(s) for s in sets_a[:n]], dtype=np.int64),
        sizes_b=np.asarray([len(s) for s in sets_b[:n]], dtype=np.int64),
        positions=None if positions is None else np.asarray(positions[:n], dtype=np.int64),
        sample_ids=None if sample_ids is None else np.asarray(sample_ids[:n], dtype=np.int64),
    )


def overlap_series(
    per_sample_overlaps: Sequence[Sequence[float]],
    positions: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, np.ndarray]:
    """Aggregate overlap curves by word index (Figure 19, Appendix E)."""
    if positions is None:
        lengths = [len(s) for s in per_sample_overlaps]
        positions = [list(range(n)) for n in lengths]
    buckets: Dict[int, List[float]] = {}
    for values, pos in zip(per_sample_overlaps, positions):
        for i, v in zip(pos, values):
            if i is None or not np.isfinite(v):
                continue
            buckets.setdefault(int(i), []).append(float(v))
    idx = np.asarray(sorted(buckets), dtype=np.int64)
    mean = np.asarray([np.mean(buckets[i]) for i in idx], dtype=np.float64) if idx.size else np.zeros(0)
    std = np.asarray([np.std(buckets[i]) for i in idx], dtype=np.float64) if idx.size else np.zeros(0)
    count = np.asarray([len(buckets[i]) for i in idx], dtype=np.int64) if idx.size else np.zeros(0, dtype=np.int64)
    return {"index": idx, "mean": mean, "std": std, "count": count}


# --------------------------------------------------------------------------- #
# Spearman difficulty correlations (Section 5.2 / Appendix E)
# --------------------------------------------------------------------------- #


def spearman_difficulty(
    metrics_a: Sequence[float],
    metrics_b: Sequence[float],
) -> Tuple[float, float]:
    """Spearman ``r_s`` (and p-value) between two per-prompt difficulty signals.

    The paper correlates CFG behaviour with instruction-tuned behaviour per
    datapoint ("harder phrases ... harder phrases for Instruction-Tuned models
    are typically where CFG and Instruction-Tuned models align").
    """
    return spearman_with_pvalue(metrics_a, metrics_b)


def difficulty_correlation(
    cfg_metrics: Sequence[float],
    instruct_metrics: Sequence[float],
    prompt_lengths: Optional[Sequence[int]] = None,
    min_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Rank correlation of CFG vs instruction-tuned difficulty signals.

    When ``prompt_lengths`` is supplied, the correlation is additionally reported
    for the subset of prompts longer than ``min_length`` (default: the median
    length), where Appendix E finds the effect "particularly pronounced".
    """
    a = np.asarray(cfg_metrics, dtype=np.float64).ravel()
    b = np.asarray(instruct_metrics, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    r_s, p = spearman_difficulty(a, b)
    out: Dict[str, Any] = {
        "spearman": r_s,
        "p_value": p,
        "n": int(n),
        "above_threshold": bool(np.isfinite(r_s) and r_s > SPEARMAN_THRESHOLD),
        "threshold": SPEARMAN_THRESHOLD,
    }
    if prompt_lengths is not None:
        lengths = np.asarray(prompt_lengths, dtype=np.float64).ravel()[:n]
        cutoff = float(min_length) if min_length is not None else float(np.median(lengths))
        mask = lengths >= cutoff
        if mask.sum() >= 3:
            r_long, p_long = spearman_difficulty(a[mask], b[mask])
            out["long_prompt_spearman"] = r_long
            out["long_prompt_p_value"] = p_long
            out["long_prompt_n"] = int(mask.sum())
            out["long_prompt_cutoff"] = cutoff
            out["long_prompt_above_threshold"] = bool(
                np.isfinite(r_long) and r_long > SPEARMAN_THRESHOLD
            )
    return out


def spearman_by_length_bin(
    cfg_metrics: Sequence[float],
    instruct_metrics: Sequence[float],
    prompt_lengths: Sequence[int],
    n_bins: int = 4,
) -> List[Dict[str, Any]]:
    """``r_s`` per prompt-length bin (shows the upward trend with length)."""
    a = np.asarray(cfg_metrics, dtype=np.float64).ravel()
    b = np.asarray(instruct_metrics, dtype=np.float64).ravel()
    lengths = np.asarray(prompt_lengths, dtype=np.float64).ravel()
    n = min(a.size, b.size, lengths.size)
    a, b, lengths = a[:n], b[:n], lengths[:n]
    if n == 0:
        return []
    n_bins = max(1, int(n_bins))
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    out: List[Dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lengths >= lo) & (lengths < hi) if hi < edges[-1] else (lengths >= lo) & (lengths <= hi)
        if mask.sum() < 3:
            continue
        r_s, p = spearman_difficulty(a[mask], b[mask])
        out.append(
            {
                "length_lo": float(lo),
                "length_hi": float(hi),
                "n": int(mask.sum()),
                "spearman": r_s,
                "p_value": p,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Appendix E tables (12/13/14)
# --------------------------------------------------------------------------- #


def dataset_similarity_table(
    records: Iterable[Dict[str, Any]],
    key: str = "overlap",
    dataset_key: str = "dataset",
    n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Table 12: per-P3-subset CFG <-> instruction-tuned similarity ranking.

    ``records`` are dicts with ``dataset`` and an overlap/similarity field; the
    result is sorted most-similar first and annotated as similar / dissimilar
    relative to the paper anchor.
    """
    buckets: Dict[str, List[float]] = {}
    for rec in records:
        name = str(rec.get(dataset_key, "unknown"))
        val = rec.get(key)
        if val is None or not np.isfinite(val):
            continue
        buckets.setdefault(name, []).append(float(val))
    table = [
        {
            "dataset": k,
            "mean_overlap": float(np.mean(v)),
            "n": len(v),
            "similar": bool(np.mean(v) > CFG_VANILLA_OVERLAP),
        }
        for k, v in buckets.items()
    ]
    table.sort(key=lambda d: d["mean_overlap"], reverse=True)
    if n is not None:
        table = table[: int(n)]
    return table


def most_least_similar_examples(
    records: Iterable[Dict[str, Any]],
    key: str = "overlap",
    prompt_key: str = "prompt",
    n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Tables 13/14: the examples with the most / least CFG<->instruct overlap.

    Appendix E: the least overlapping examples are "vague, open-ended
    questions", the most overlapping ones are "longer, more complex questions".
    """
    rows = [
        {
            prompt_key: rec.get(prompt_key, ""),
            "overlap": float(rec.get(key, float("nan"))),
            "length": int(rec.get("length", len(str(rec.get(prompt_key, "")).split()))),
        }
        for rec in records
        if rec.get(key) is not None and np.isfinite(rec.get(key))
    ]
    rows.sort(key=lambda d: d["overlap"], reverse=True)
    return {"most_similar": rows[: int(n)], "least_similar": rows[-int(n):][::-1]}


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def format_overlap_table(
    comparison: OverlapComparison,
    modes: Sequence[str] = OVERLAP_MODES,
) -> str:
    """Plain-text Table-style rendering of an :class:`OverlapComparison`."""
    lines = [
        f"Top-p = {comparison.top_p:g} token overlap vs {comparison.reference!r} "
        f"(gamma_cfg={comparison.gamma:g}, n={comparison.n})",
        f"{'mode':<22}{'mean':>8}{'std':>8}{'median':>8}{'n_tokens':>10}",
        "-" * 56,
    ]
    for mode in modes:
        st = comparison.stats.get(mode)
        if st is None:
            continue
        lines.append(
            f"{MODE_LABELS.get(mode, mode):<22}"
            f"{st.mean:>8.3f}{st.std:>8.3f}{st.median:>8.3f}{st.n_tokens:>10d}"
        )
    if comparison.spearman_vs_instruct:
        lines.append("")
        lines.append("Spearman r_s vs instruction-tuned per datapoint:")
        for mode, rho in comparison.spearman_vs_instruct.items():
            flag = " (> .7)" if np.isfinite(rho) and rho > SPEARMAN_THRESHOLD else ""
            lines.append(f"  {MODE_LABELS.get(mode, mode):<22}{rho:>8.3f}{flag}")
    lines.append("")
    lines.append(
        f"Figure 18b anchor: CFG/vanilla overlap ~ {CFG_VANILLA_OVERLAP:.2f} "
        f"(measured {comparison.mean:.3f})"
    )
    return "\n".join(lines)


def summarize_overlap(stats: OverlapStats) -> str:
    """One-line textual summary of an :class:`OverlapStats` record."""
    return (
        f"{stats.name} vs {stats.reference}: mean top-p={stats.top_p:g} overlap "
        f"{stats.mean:.3f} +/- {stats.sem:.3f} over {stats.n_tokens} tokens "
        f"(median {stats.median:.3f}, {stats.fraction_below_half:.1%} below 0.5)"
    )


def overlap_report(comparison: OverlapComparison) -> Dict[str, Any]:
    """Dictionary summary suitable for JSON dumping by ``scripts/run_analysis.py``."""
    rep = comparison.as_dict()
    rep["paper_check"] = comparison.check_against_paper()
    rep["formatted"] = format_overlap_table(comparison)
    return rep


def check_against_paper(
    comparison: OverlapComparison,
    tolerance: float = OVERLAP_TOLERANCE,
) -> Dict[str, Any]:
    """Convenience wrapper around :meth:`OverlapComparison.check_against_paper`."""
    return comparison.check_against_paper(tolerance=tolerance)


# --------------------------------------------------------------------------- #
# Model-driven driver (Section 5 protocol on the P3 sample)
# --------------------------------------------------------------------------- #


class OverlapAnalyzer:
    """Drives a :class:`~src.cfg.model_wrapper.CFGModelWrapper` for Section 5.2.

    Parameters
    ----------
    model_wrapper:
        Base model wrapped for dual-context (conditional/unconditional) passes.
    gamma:
        CFG strength for the ``cfg`` mode (default 1.5, as in Section 5).
    top_p:
        Nucleus threshold (default 0.9).
    max_new_tokens:
        Completion budget per prompt.
    temperature:
        Applied after the CFG combination when computing the distribution.
    unconditional_mode:
        ``"empty_prefix"`` (Table 13/14 default) or ``"last_prompt_token"``.
    instruct_wrapper:
        Optional second wrapper (same tokenizer) for ``P_instruct(y|x)``.
    """

    def __init__(
        self,
        model_wrapper: Any,
        gamma: float = ANALYSIS_GAMMA,
        top_p: float = TOP_P,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        unconditional_mode: str = "empty_prefix",
        instruct_wrapper: Optional[Any] = None,
        device: Optional[Any] = None,
        seed: int = 0,
        batch_size: int = 1,
    ) -> None:
        self.model_wrapper = model_wrapper
        self.gamma = float(gamma)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.unconditional_mode = unconditional_mode
        self.instruct_wrapper = instruct_wrapper
        self.device = device
        self.seed = int(seed)
        self.batch_size = int(batch_size)

    # -- wrapper bookkeeping ----------------------------------------------- #
    def _wrap(self, mode: str, wrapper: Optional[Any] = None) -> Any:
        if mode == "instruct":
            w = wrapper or self.instruct_wrapper or self.model_wrapper
        else:
            w = wrapper or self.model_wrapper
        if self.unconditional_mode is not None and hasattr(w, "unconditional_mode"):
            w.unconditional_mode = self.unconditional_mode
        return w

    @staticmethod
    def _encode(wrapper: Any, prompt: str) -> Any:
        if hasattr(wrapper, "encode"):
            return wrapper.encode(prompt, return_tensors="pt")
        raise AttributeError("model wrapper must expose an `encode` method")

    @staticmethod
    def _dual(wrapper: Any, input_ids: Any, prompt_length: Optional[int],
              negative_input_ids: Any = None) -> Tuple[Any, Any]:
        """Call ``dual_logits(..., only_last=True)`` with a signature fallback."""
        try:
            dual = wrapper.dual_logits(
                input_ids,
                prompt_length=prompt_length,
                only_last=True,
                negative_input_ids=negative_input_ids,
            )
        except TypeError:  # pragma: no cover - older wrapper signature
            dual = wrapper.dual_logits(input_ids, only_last=True)
        return dual.cond, dual.uncond

    # -- per-prompt token sets --------------------------------------------- #
    def token_sets_for_prompt(
        self,
        prompt: str,
        mode: str = "cfg",
        wrapper: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        prompt_length: Optional[int] = None,
        return_tokens: bool = False,
    ) -> Union[List[frozenset], Tuple[List[frozenset], List[int]]]:
        """Minimum top-p token sets for every completion step of one prompt.

        ``mode`` selects the distribution:

        * ``cfg``        -> Eq. 7 with ``gamma = self.gamma``;
        * ``vanilla``    -> ``gamma = 1`` (the conditional distribution only);
        * ``unprompted`` -> ``gamma = 0`` (the unconditional ``P(x)`` pass);
        * ``instruct``   -> ``gamma = 1`` on ``instruct_wrapper``.
        """
        w = self._wrap(mode, wrapper)
        gamma = float(MODE_GAMMAS.get(mode, self.gamma))
        if mode == "cfg":
            gamma = self.gamma
        budget = int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens)
        gen = None
        if _HAS_TORCH:  # pragma: no cover - torch path
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.seed if seed is None else seed))

        enc = self._encode(w, prompt)
        input_ids = enc["input_ids"] if isinstance(enc, dict) else enc
        ctx_len = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(input_ids)
        plen = int(prompt_length) if prompt_length is not None else ctx_len

        neg_ids = None
        if negative_prompt:
            neg_enc = self._encode(w, negative_prompt)
            neg_ids = neg_enc["input_ids"] if isinstance(neg_enc, dict) else neg_enc

        sets: List[frozenset] = []
        tokens: List[int] = []
        device = getattr(w, "device", None)

        with torch.no_grad() if _HAS_TORCH else _nullcontext():  # pragma: no cover
            for _ in range(budget):
                cond, uncond = self._dual(w, input_ids, plen, neg_ids)
                probs = guided_distribution(
                    cond,
                    uncond if gamma != 1.0 and mode != "vanilla" else None,
                    gamma=gamma,
                    temperature=self.temperature,
                )
                sets.append(top_p_token_set(probs, top_p=self.top_p))
                if do_sample and _HAS_TORCH:  # pragma: no cover
                    tok = int(torch.multinomial(torch.as_tensor(probs), 1, generator=gen).item())
                else:
                    tok = int(np.argmax(probs))
                tokens.append(tok)
                step = torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device) \
                    if _HAS_TORCH else None
                if step is None:  # pragma: no cover - numpy fallback
                    break
                input_ids = torch.cat([input_ids, step], dim=-1)
                eos = getattr(w, "eos_token_id", None)
                if eos is not None and tok == int(eos):
                    break
                if device is None and hasattr(input_ids, "to"):
                    input_ids = input_ids.to(w.device) if getattr(w, "device", None) else input_ids
        return (sets, tokens) if return_tokens else sets

    # -- aggregation -------------------------------------------------------- #
    def measure(
        self,
        prompts: Sequence[str],
        mode: str = "cfg",
        wrapper: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        progress: bool = False,
        sample_ids: Optional[Sequence[int]] = None,
        prompt_length: Optional[int] = None,
    ) -> TopPSetStats:
        """Collect :class:`TopPSetStats` for a list of prompts in one mode."""
        all_sets: List[frozenset] = []
        sizes: List[int] = []
        positions: List[int] = []
        sids: List[int] = []
        for i, prompt in enumerate(prompts):
            if progress:
                logger.info("[overlap] %s %d/%d", mode, i + 1, len(prompts))
            sets, toks = self.token_sets_for_prompt(  # type: ignore[misc]
                prompt,
                mode=mode,
                wrapper=wrapper,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                seed=None if seed is None else int(seed) + i,
                negative_prompt=negative_prompt,
                prompt_length=prompt_length,
                return_tokens=True,
            )
            all_sets.extend(sets)
            sizes.extend(len(s) for s in sets)
            positions.extend(range(len(sets)))
            sids.extend([int(sample_ids[i]) if sample_ids is not None else i] * len(sets))
        return TopPSetStats(
            sets=all_sets,
            name=mode,
            gamma=float(MODE_GAMMAS.get(mode, self.gamma)),
            top_p=self.top_p,
            sizes=np.asarray(sizes, dtype=np.int64),
            positions=np.asarray(positions, dtype=np.int64),
            sample_ids=np.asarray(sids, dtype=np.int64),
        )

    def compare(
        self,
        prompts: Sequence[str],
        modes: Sequence[str] = ("cfg", "vanilla", "unprompted"),
        reference: str = "vanilla",
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        seed: Optional[int] = None,
        progress: bool = False,
        denominator: str = "first",
        include_instruct: bool = False,
        prompt_length: Optional[int] = None,
        prompt_lengths: Optional[Sequence[int]] = None,
    ) -> OverlapComparison:
        """Run the full Section 5.2 comparison over ``prompts``.

        Returns an :class:`OverlapComparison` whose ``stats`` maps each
        non-reference mode to its per-token top-p overlap with ``reference``
        (``cfg`` vs ``vanilla`` is the Figure 18b number).
        """
        mode_list = list(modes)
        if include_instruct and "instruct" not in mode_list:
            mode_list.append("instruct")
        mode_stats: Dict[str, TopPSetStats] = {}
        for mode in mode_list:
            mode_stats[mode] = self.measure(
                prompts,
                mode=mode,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                seed=seed,
                progress=progress,
                prompt_length=prompt_length,
            )
        if reference not in mode_stats:
            raise ValueError(f"reference mode {reference!r} was not measured; got {mode_list}")

        ref = mode_stats[reference]
        stats: Dict[str, OverlapStats] = {}
        for mode, ms in mode_stats.items():
            if mode == reference:
                continue
            # re-derive positions/sample ids aligned with the min length
            stats[mode] = compare_token_sets(
                ms.sets,
                ref.sets,
                name=mode,
                reference=reference,
                gamma=float(MODE_GAMMAS.get(mode, self.gamma)),
                top_p=self.top_p,
                denominator=denominator,
                positions=ms.positions,
                sample_ids=ms.sample_ids,
            )

        spearman: Dict[str, float] = {}
        if "instruct" in mode_stats:
            instruct = mode_stats["instruct"]
            for mode in ("cfg", "vanilla"):
                if mode in mode_stats:
                    rho = _spearman_between_modes(mode_stats[mode], instruct)
                    if np.isfinite(rho):
                        spearman[mode] = rho

        return OverlapComparison(
            stats=stats,
            mode_stats=mode_stats,
            reference=reference,
            n=len(prompts),
            gamma=self.gamma,
            top_p=self.top_p,
            spearman_vs_instruct=spearman,
        )

    # -- convenience: distributions rather than sets ------------------------ #
    def distributions_for_prompt(
        self,
        prompt: str,
        mode: str = "cfg",
        wrapper: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        prompt_length: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Post-CFG vocabulary distributions per completion step (for reuse)."""
        w = self._wrap(mode, wrapper)
        gamma = self.gamma if mode == "cfg" else float(MODE_GAMMAS.get(mode, self.gamma))
        budget = int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens)
        enc = self._encode(w, prompt)
        input_ids = enc["input_ids"] if isinstance(enc, dict) else enc
        ctx_len = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(input_ids)
        plen = int(prompt_length) if prompt_length is not None else ctx_len
        neg_ids = None
        if negative_prompt:
            ne = self._encode(w, negative_prompt)
            neg_ids = ne["input_ids"] if isinstance(ne, dict) else ne

        dists: List[np.ndarray] = []
        gen = None
        if _HAS_TORCH:  # pragma: no cover
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.seed if seed is None else seed))
        for _ in range(budget):
            cond, uncond = self._dual(w, input_ids, plen, neg_ids)
            probs = guided_distribution(
                cond,
                None if mode == "vanilla" else uncond,
                gamma=gamma,
                temperature=self.temperature,
            )
            dists.append(probs)
            if do_sample and _HAS_TORCH:  # pragma: no cover
                tok = int(torch.multinomial(torch.as_tensor(probs), 1, generator=gen).item())
            else:
                tok = int(np.argmax(probs))
            if not _HAS_TORCH:  # pragma: no cover
                break
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device),
                ],
                dim=-1,
            )
            eos = getattr(w, "eos_token_id", None)
            if eos is not None and tok == int(eos):
                break
        return dists


class _nullcontext:  # pragma: no cover - tiny stdlib-free null context manager
    """Minimal ``contextlib.nullcontext`` stand-in used when torch is absent."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


def _spearman_between_modes(stats_a: TopPSetStats, stats_b: TopPSetStats) -> float:
    """Per-datapoint Spearman ``r_s`` of mean top-p nucleus size between modes.

    Mirrors Section 5.2's claim: "harder phrases for Instruction-Tuned models are
    typically where CFG and Instruction-Tuned models align: significant spearman
    correlations of r_s > .7 between Instruction-Tuned models and CFG".
    """
    by_a = stats_a.by_sample()
    by_b = stats_b.by_sample()
    keys = sorted(set(by_a) & set(by_b))
    if len(keys) < 2:
        return float("nan")
    a = np.asarray([np.mean([len(s) for s in by_a[k]]) for k in keys], dtype=np.float64)
    b = np.asarray([np.mean([len(s) for s in by_b[k]]) for k in keys], dtype=np.float64)
    return spearman_correlation(a, b)


def per_sample_entropy_proxy(stats: TopPSetStats) -> Dict[int, float]:
    """Per-datapoint difficulty proxy: mean top-p nucleus *size*.

    Larger nuclei = flatter, higher-entropy, "easier"/more open-ended prompts
    (Appendix E); used for the CFG-vs-instruction-tuned Spearman analysis when
    the entropy analyzer's records are not available.
    """
    return {k: float(np.mean([len(s) for s in v])) for k, v in stats.by_sample().items()}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def _demo() -> None:  # pragma: no cover - manual smoke test
    rng = np.random.default_rng(0)
    logits = rng.normal(size=1000)
    logits[3] += 6.0
    p_a = _softmax_np(logits)
    p_b = _softmax_np(logits + 0.5)  # slightly sharper -> strong overlap
    p_c = _softmax_np(rng.normal(size=1000))  # unrelated

    same = overlap_fraction(top_p_token_set(p_a), top_p_token_set(p_a))
    assert abs(same - 1.0) < 1e-9, same
    near = top_p_token_overlap(p_a, p_b)
    assert near > 0.5, near
    far = top_p_token_overlap(p_a, p_c)
    assert far < near

    x = np.arange(20, dtype=np.float64)
    assert abs(spearman_correlation(x, x ** 3) - 1.0) < 1e-9
    r_s, p = spearman_with_pvalue(x, -x)
    assert r_s < -0.99

    sets = [top_p_token_set(p_a) for _ in range(5)]
    stats = compare_token_sets(sets, sets, name="cfg", reference="vanilla",
                               positions=[0, 1, 2, 3, 4], sample_ids=[0, 0, 1, 1, 1])
    assert abs(stats.mean - 1.0) < 1e-9
    assert len(stats.by_index()["index"]) == 5
    assert len(stats.by_sample()) == 2
    logger.info("%s", summarize_overlap(stats))
    logger.info("overlap demo OK")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _demo()
