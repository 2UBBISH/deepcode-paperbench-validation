"""Perplexity analysis for Classifier-Free Guidance (Section 5.2 / Appendix E).

This module implements the perplexity half of the Section 5 "CFG's Relation to
Instruction Tuning" analysis of *Stay on Topic with Classifier-Free Guidance*.

The key protocol detail (paper Section 5.2 + Appendix E): perplexity is computed
**on the continuation only** -- the prompt tokens contribute no log-likelihood to
the score.  Concretely, for a prompt ``x`` and its continuation ``y``::

    PPL(y | x) = exp( - (1/n) * sum_{i=1..n} log p(y_i | x, y_<i) )

where ``p`` is either the vanilla prompted distribution ``P(y|x)``, the
Classifier-Free Guided distribution::

    p_cfg(y_i | ...) = softmax( logits_uncond + gamma * (logits_cond - logits_uncond) )

(Eq. 7), or the instruction-tuned model's distribution ``P_instruct(y|x)``.

The paper reports (Figure 5) that per-datapoint perplexities correlate strongly
between CFG and the vanilla prompted model (r ~= 0.94) and moderately between the
instruction-tuned model and CFG (r ~= 0.70), i.e. CFG re-weights difficulty in a
way that is related to but distinct from instruction-tuning.

Design
------
As with ``analysis/entropy.py`` and ``analysis/overlap.py``, the module is split
into

* a **pure-math, dependency-light layer** (NumPy only; SciPy optional) that is
  fully unit-testable on CPU without any model, and
* a thin **driver** (``PerplexityAnalyzer``) that only touches torch / a
  :class:`~src.cfg.model_wrapper.CFGModelWrapper` when one is supplied.

If ``torch``/``transformers`` are unavailable the module still imports cleanly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies (guarded so the pure-math layer always imports)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover - optional
    from ..cfg.logits import log_softmax as _log_softmax

    _HAS_LOGITS = True
except Exception:  # pragma: no cover
    try:
        from src.cfg.logits import log_softmax as _log_softmax  # type: ignore

        _HAS_LOGITS = True
    except Exception:
        _log_softmax = None  # type: ignore
        _HAS_LOGITS = False

try:  # pragma: no cover - reuse the Section 5.2 statistics helpers
    from .overlap import pearson as _pearson_impl, rankdata as _rankdata_impl

    _HAS_OVERLAP = True
except Exception:  # pragma: no cover
    try:
        from src.analysis.overlap import (  # type: ignore
            pearson as _pearson_impl,
            rankdata as _rankdata_impl,
        )

        _HAS_OVERLAP = True
    except Exception:
        _pearson_impl = None  # type: ignore
        _rankdata_impl = None  # type: ignore
        _HAS_OVERLAP = False


# ---------------------------------------------------------------------------
# Constants (mirror the paper's Section 5 anchors and shared sweep grids)
# ---------------------------------------------------------------------------
ANALYSIS_GAMMA: float = 1.5
CFG_PPL_VANILLA_CORR: float = 0.94
CFG_PPL_INSTRUCT_CORR: float = 0.70
PPL_TOLERANCE: float = 0.10
PPL_MODES: Tuple[str, ...] = ("cfg", "vanilla", "instruct")
MODE_GAMMAS: Dict[str, float] = {"cfg": ANALYSIS_GAMMA, "vanilla": 1.0, "instruct": 1.0}
MODE_LABELS: Dict[str, str] = {
    "cfg": r"PPL_cfg (gamma=1.5)",
    "vanilla": "PPL(y|x) vanilla",
    "unprompted": "PPL(x) unprompted",
    "instruct": r"PPL_instruct(y|x)",
}
#: modes whose perplexity is anchored in the paper's Figure 5 correlation table
CORRELATION_PAIRS: Tuple[Tuple[str, str, float], ...] = (
    ("cfg", "vanilla", CFG_PPL_VANILLA_CORR),
    ("instruct", "cfg", CFG_PPL_INSTRUCT_CORR),
)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _to_numpy(x: Any) -> np.ndarray:
    """Detach/convert a torch tensor or array-like into a float64 numpy array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
        return x.detach().to("cpu", dtype=torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def _guided_logits(
    logits_cond: Any,
    logits_uncond: Optional[Any] = None,
    gamma: float = ANALYSIS_GAMMA,
) -> np.ndarray:
    """Eq. 7 in logit space: ``uncond + gamma * (cond - uncond)``.

    ``gamma == 1`` returns the conditional logits unchanged; ``gamma == 0``
    returns the unconditional logits.  ``None`` unconditional logits fall back to
    the conditional ones (vanilla behaviour).
    """
    cond = _to_numpy(logits_cond)
    if logits_uncond is None or float(gamma) == 1.0:
        return cond
    uncond = _to_numpy(logits_uncond)
    gamma = float(gamma)
    if gamma == 0.0:
        return uncond
    return uncond + gamma * (cond - uncond)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax over ``axis`` (numpy)."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _log_softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable log-softmax over ``axis`` (numpy)."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def log_probs_from_logits(
    logits: np.ndarray,
    target_ids: Optional[Sequence[int]] = None,
    axis: int = -1,
) -> np.ndarray:
    """Log-softmax over the vocabulary, optionally gathering ``target_ids``.

    Parameters
    ----------
    logits:
        ``[..., vocab]`` (or ``[vocab]``) array of raw logits.
    target_ids:
        If given, returns ``[..., len(target_ids)]`` log-probabilities gathered at
        those vocabulary indices (broadcast along the leading axes).

    Returns
    -------
    np.ndarray
        Log-probabilities (never ``-inf`` for the gathered entries).
    """
    logits = _to_numpy(logits)
    lp = _log_softmax_np(logits, axis=axis)
    if target_ids is None:
        return lp
    targets = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    if lp.ndim == 1:
        return lp[targets]
    return np.take_along_axis(lp, targets.reshape((1,) * (lp.ndim - 1) + (-1,)), axis=-1)


def token_perplexities(logprobs: Any) -> np.ndarray:
    """Per-token perplexity ``exp(-log p)`` for a sequence of log-probabilities."""
    lp = _to_numpy(logprobs).reshape(-1)
    lp = lp[np.isfinite(lp)]
    return np.exp(-lp)


def perplexity_from_logprobs(logprobs: Any, ignore_prompt: bool = True) -> float:
    """Perplexity of a token sequence from its per-token log-probabilities.

    ``PPL = exp( - (1/n) sum_i log p_i )``.

    ``ignore_prompt`` documents (and enforces) the paper's convention that only
    *continuation* log-likelihoods are used: callers must pass log-probs that
    already correspond to continuation tokens.  Setting it to ``False`` merely
    permits prompt log-probs to be included for a non-paper-faithful baseline.

    Returns ``inf`` for an empty sequence.
    """
    del ignore_prompt  # documented convention; the caller controls the input
    lp = _to_numpy(logprobs).reshape(-1)
    lp = lp[np.isfinite(lp)]
    if lp.size == 0:
        return float("inf")
    return float(np.exp(-np.mean(lp)))


def mean_nll(logprobs: Any) -> float:
    """Mean negative log-likelihood ``-(1/n) sum_i log p_i`` (log-perplexity)."""
    lp = _to_numpy(logprobs).reshape(-1)
    lp = lp[np.isfinite(lp)]
    if lp.size == 0:
        return float("inf")
    return float(-np.mean(lp))


# ---------------------------------------------------------------------------
# Correlation helpers (reuse overlap.py when available; else local fallback)
# ---------------------------------------------------------------------------
def rankdata(x: Any) -> np.ndarray:
    """Average-ranks of ``x`` (ties share the mean rank)."""
    if _rankdata_impl is not None:
        return np.asarray(_rankdata_impl(_to_numpy(x)), dtype=np.float64)
    arr = _to_numpy(x).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    # average ties
    uniq, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    if uniq.size != arr.size:
        idx = np.argsort(ranks, kind="mergesort")
        boundaries = np.concatenate(
            ([0], np.cumsum(counts))
        )
        grouped = ranks[idx]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            grouped[start:stop] = grouped[start:stop].mean()
        ranks[idx] = grouped
    return ranks


def pearson(x: Any, y: Any) -> float:
    """Pearson correlation coefficient (``nan`` for degenerate inputs)."""
    if _pearson_impl is not None:
        return float(_pearson_impl(_to_numpy(x), _to_numpy(y)))
    a = _to_numpy(x).reshape(-1)
    b = _to_numpy(y).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    a_std = a.std()
    b_std = b.std()
    if a_std == 0 or b_std == 0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (a_std * b_std))


def spearman_correlation(x: Any, y: Any) -> float:
    """Spearman rank correlation (Pearson on average-ranks)."""
    a = _to_numpy(x).reshape(-1)
    b = _to_numpy(y).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def regression_slope(x: Any, y: Any) -> Tuple[float, float]:
    """Ordinary least-squares ``(slope, intercept)`` of ``y`` on ``x``."""
    a = _to_numpy(x).reshape(-1)
    b = _to_numpy(y).reshape(-1)
    if a.size != b.size or a.size < 2 or a.std() == 0:
        return (float("nan"), float("nan"))
    slope = float(np.cov(a, b, bias=True)[0, 1] / a.var())
    return slope, float(b.mean() - slope * a.mean())


def correlation_table(
    per_sample: Dict[str, Sequence[float]],
    pairs: Sequence[Tuple[str, str, float]] = CORRELATION_PAIRS,
    min_samples: int = 2,
) -> Dict[str, Dict[str, float]]:
    """Pearson + Spearman correlations for the paper's Figure 5 pairs.

    Parameters
    ----------
    per_sample:
        ``{mode: [ppl_per_datapoint, ...]}`` -- all modes must be aligned on the
        same datapoints (same order / length).
    pairs:
        Iterable of ``(mode_a, mode_b, paper_reference)`` triples.

    Returns
    -------
    dict
        ``{"<a>_vs_<b>": {"pearson": r, "spearman": rho, "n": k, "paper": ref}}``.
    """
    out: Dict[str, Dict[str, float]] = {}
    for a, b, reference in pairs:
        if a not in per_sample or b not in per_sample:
            continue
        va = _to_numpy(per_sample[a]).reshape(-1)
        vb = _to_numpy(per_sample[b]).reshape(-1)
        n = int(min(va.size, vb.size))
        if n < min_samples:
            continue
        va, vb = va[:n], vb[:n]
        keep = np.isfinite(va) & np.isfinite(vb)
        va, vb = va[keep], vb[keep]
        if va.size < min_samples:
            continue
        key = f"{a}_vs_{b}"
        out[key] = {
            "pearson": pearson(va, vb),
            "spearman": spearman_correlation(va, vb),
            "n": int(va.size),
            "paper": float(reference),
            "delta": float(pearson(va, vb) - reference),
        }
    return out


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class PerplexityStats:
    """Per-datapoint perplexities of one mode (continuation tokens only).

    Attributes
    ----------
    per_sample:
        ``PPL`` of each datapoint's continuation.
    n_tokens:
        Number of scored continuation tokens per datapoint.
    per_sample_nll:
        Mean negative log-likelihood ``= log(PPL)`` per datapoint.
    sample_ids / names:
        Bookkeeping so CFG/vanilla/instruct records can be aligned.
    """

    per_sample: List[float] = field(default_factory=list)
    n_tokens: List[int] = field(default_factory=list)
    per_sample_nll: List[float] = field(default_factory=list)
    sample_ids: List[Any] = field(default_factory=list)
    name: str = "cfg"
    gamma: float = ANALYSIS_GAMMA

    # -- basic accessors ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.per_sample)

    @property
    def n_datapoints(self) -> int:
        return len(self.per_sample)

    @property
    def total_tokens(self) -> int:
        return int(np.sum(self.n_tokens)) if self.n_tokens else 0

    @property
    def mean_ppl(self) -> float:
        """Mean of per-datapoint perplexities (the table's headline number)."""
        if not self.per_sample:
            return float("nan")
        return float(np.nanmean(np.asarray(self.per_sample, dtype=np.float64)))

    @property
    def median_ppl(self) -> float:
        if not self.per_sample:
            return float("nan")
        return float(np.nanmedian(np.asarray(self.per_sample, dtype=np.float64)))

    @property
    def std_ppl(self) -> float:
        if len(self.per_sample) < 2:
            return float("nan")
        return float(np.nanstd(np.asarray(self.per_sample, dtype=np.float64), ddof=1))

    @property
    def sem(self) -> float:
        if len(self.per_sample) < 2:
            return float("nan")
        return self.std_ppl / math.sqrt(len(self.per_sample))

    @property
    def mean_nll(self) -> float:
        """Mean per-datapoint log-perplexity (used for the regressions)."""
        if not self.per_sample_nll:
            return float("nan")
        return float(np.nanmean(np.asarray(self.per_sample_nll, dtype=np.float64)))

    @property
    def dataset_ppl(self) -> float:
        """Corpus-level perplexity: ``exp(total_nll / total_tokens)``."""
        if not self.per_sample_nll or not self.n_tokens:
            return self.mean_ppl
        nll = np.asarray(self.per_sample_nll, dtype=np.float64)
        nt = np.asarray(self.n_tokens, dtype=np.float64)
        denom = nt.sum()
        if denom <= 0:
            return float("nan")
        return float(np.exp(float((nll * nt).sum() / denom)))

    # -- conversion --------------------------------------------------------
    def by_sample(self) -> Dict[Any, float]:
        """``{sample_id: ppl}`` (falls back to positional indices)."""
        ids = self.sample_ids or list(range(len(self.per_sample)))
        return {i: float(p) for i, p in zip(ids, self.per_sample)}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gamma": float(self.gamma),
            "n_datapoints": self.n_datapoints,
            "total_tokens": self.total_tokens,
            "mean_ppl": self.mean_ppl,
            "median_ppl": self.median_ppl,
            "std_ppl": self.std_ppl,
            "sem": self.sem,
            "dataset_ppl": self.dataset_ppl,
            "mean_nll": self.mean_nll,
            "per_sample": [float(p) for p in self.per_sample],
            "n_tokens": [int(t) for t in self.n_tokens],
        }


@dataclass
class PerplexityComparison:
    """Aggregate of a Section 5.2 perplexity run across modes."""

    stats: Dict[str, PerplexityStats] = field(default_factory=dict)
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reference: str = "vanilla"
    gamma: float = ANALYSIS_GAMMA
    temperature: float = 1.0

    def __len__(self) -> int:
        return len(self.stats)

    @property
    def mean_ppl(self) -> Dict[str, float]:
        return {k: s.mean_ppl for k, s in self.stats.items()}

    def ppl_of(self, mode: str) -> float:
        return self.stats[mode].mean_ppl if mode in self.stats else float("nan")

    def correlation(self, pair: str) -> float:
        """Pearson correlation for a ``"a_vs_b"`` key (``nan`` if absent)."""
        entry = self.correlations.get(pair, {})
        return float(entry.get("pearson", float("nan")))

    def check_against_paper(self, tolerance: float = PPL_TOLERANCE) -> Dict[str, Any]:
        """Compare measured correlations to Figure 5's anchors."""
        checks: Dict[str, Any] = {}
        for a, b, reference in CORRELATION_PAIRS:
            key = f"{a}_vs_{b}"
            measured = self.correlation(key)
            checks[key] = {
                "measured": measured,
                "paper": float(reference),
                "within_tolerance": bool(
                    np.isfinite(measured) and abs(measured - reference) <= tolerance
                ),
            }
        checks["all_within_tolerance"] = bool(
            all(v["within_tolerance"] for k, v in checks.items() if k != "all_within_tolerance")
        )
        return checks

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "gamma": float(self.gamma),
            "temperature": float(self.temperature),
            "mean_ppl": self.mean_ppl,
            "correlations": self.correlations,
            "stats": {k: s.as_dict() for k, s in self.stats.items()},
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class PerplexityAnalyzer:
    """Section 5.2 perplexity driver over a :class:`CFGModelWrapper`.

    Perplexity is measured on the **continuation only**: the prompt's tokens
    never contribute log-likelihood.  Modes map to gammas exactly as in the
    entropy/overlap analyses: ``cfg`` -> ``self.gamma`` (1.5), ``vanilla`` -> 1.0,
    ``unprompted`` -> 0.0 (the continuation is scored conditioned on nothing but
    the BOS/empty prefix), ``instruct`` -> 1.0 but using ``instruct_wrapper``.
    """

    def __init__(
        self,
        model_wrapper: Any,
        gamma: float = ANALYSIS_GAMMA,
        temperature: float = 1.0,
        unconditional_mode: str = "empty_prefix",
        instruct_wrapper: Any = None,
        device: Any = None,
        max_length: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.wrapper = model_wrapper
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.unconditional_mode = unconditional_mode
        self.instruct_wrapper = instruct_wrapper
        self.device = device if device is not None else getattr(model_wrapper, "device", None)
        self.max_length = max_length
        self.negative_prompt = negative_prompt
        self.seed = seed

    # -- internals ---------------------------------------------------------
    def _wrapper_for(self, mode: str) -> Any:
        if mode == "instruct":
            if self.instruct_wrapper is None:
                raise ValueError("mode='instruct' requires instruct_wrapper")
            return self.instruct_wrapper
        return self.wrapper

    def _gamma_for(self, mode: str, gamma: Optional[float]) -> float:
        if gamma is not None:
            return float(gamma)
        if mode == "cfg":
            return self.gamma
        if mode == "vanilla":
            return 1.0
        if mode == "instruct":
            return 1.0
        if mode in ("unprompted", "uncond", "unconditional"):
            return 0.0
        return 1.0

    def _encode(self, wrapper: Any, text: str) -> List[int]:
        ids = wrapper.encode(text, add_special_tokens=False)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def _dual(self, wrapper: Any, input_ids: List[int], prompt_length: int,
              negative_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(cond_logits, uncond_logits)`` of shape ``[T, vocab]``."""
        if _HAS_TORCH:
            device = getattr(wrapper, "device", None)
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            neg = None
            if negative_ids is not None:
                neg = torch.tensor([negative_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                try:
                    dual = wrapper.dual_logits(
                        ids,
                        prompt_length=prompt_length,
                        only_last=False,
                        negative_input_ids=neg,
                    )
                except TypeError:
                    dual = wrapper.dual_logits(ids, prompt_length=prompt_length)
            cond = dual.cond[0]
            uncond = dual.uncond[0] if dual.uncond is not None else None
            return _to_numpy(cond), (_to_numpy(uncond) if uncond is not None else None)
        raise RuntimeError("torch is required to run the model path")

    # -- public API --------------------------------------------------------
    def logprobs_for_pair(
        self,
        prompt: str,
        continuation: str,
        mode: str = "cfg",
        gamma: Optional[float] = None,
        wrapper: Any = None,
        negative_prompt: Optional[str] = None,
    ) -> np.ndarray:
        """Per-token continuation log-probs of ``continuation`` given ``prompt``.

        The prompt tokens are *never* scored (Section 5.2 convention).  The
        returned array has one entry per continuation token.
        """
        model = wrapper if wrapper is not None else self._wrapper_for(mode)
        prompt_ids = self._encode(model, prompt)
        cont_ids = self._encode(model, continuation)
        if not cont_ids:
            return np.asarray([], dtype=np.float64)
        input_ids = prompt_ids + cont_ids
        if self.max_length is not None:
            # left-truncate the prompt so the continuation always fits
            overflow = len(input_ids) - int(self.max_length)
            if overflow > 0:
                drop = min(overflow, len(prompt_ids))
                prompt_ids = prompt_ids[drop:]
                input_ids = prompt_ids + cont_ids
        prompt_length = len(prompt_ids)

        neg_ids = self._encode(model, negative_prompt) if negative_prompt else None
        cond_logits, uncond_logits = self._dual(model, input_ids, prompt_length, neg_ids)

        g = self._gamma_for(mode, gamma)
        guided = _guided_logits(cond_logits, uncond_logits, g)
        if self.temperature and self.temperature != 1.0:
            guided = guided / float(self.temperature)
        logprobs = _log_softmax_np(guided)

        # position prompt_length-1+i predicts continuation token i
        start = max(prompt_length - 1, 0)
        score_positions = np.arange(start, start + len(cont_ids))
        score_positions = score_positions[score_positions < logprobs.shape[0]]
        targets = np.asarray(cont_ids[: score_positions.size], dtype=np.int64)
        if score_positions.size == 0:
            return np.asarray([], dtype=np.float64)
        return logprobs[score_positions, targets]

    def perplexity(
        self,
        prompt: str,
        continuation: str,
        mode: str = "cfg",
        gamma: Optional[float] = None,
        wrapper: Any = None,
        negative_prompt: Optional[str] = None,
    ) -> float:
        """Continuation-only perplexity of one ``(prompt, continuation)`` pair."""
        lp = self.logprobs_for_pair(
            prompt,
            continuation,
            mode=mode,
            gamma=gamma,
            wrapper=wrapper,
            negative_prompt=negative_prompt,
        )
        return perplexity_from_logprobs(lp)

    def measure(
        self,
        records: Sequence[Any],
        mode: str = "cfg",
        gamma: Optional[float] = None,
        wrapper: Any = None,
        negative_prompt: Optional[str] = None,
        max_datapoints: Optional[int] = None,
        sample_ids: Optional[Sequence[Any]] = None,
        progress: bool = False,
    ) -> PerplexityStats:
        """Compute per-datapoint continuation perplexities for ``records``.

        Each record may be a :class:`~src.data.p3_sampler.P3Sample`, a dict with
        ``inputs``/``targets`` (P3 field names) or ``prompt``/``continuation``,
        or a plain ``(prompt, continuation)`` tuple.
        """
        g = self._gamma_for(mode, gamma)
        stats = PerplexityStats(
            name=mode,
            gamma=g,
            sample_ids=list(sample_ids) if sample_ids is not None else [],
        )
        for i, record in enumerate(records):
            if max_datapoints is not None and i >= max_datapoints:
                break
            prompt, continuation = _split_record(record)
            lp = self.logprobs_for_pair(
                prompt,
                continuation,
                mode=mode,
                gamma=gamma,
                wrapper=wrapper,
                negative_prompt=negative_prompt,
            )
            stats.per_sample.append(perplexity_from_logprobs(lp))
            stats.n_tokens.append(int(lp.size))
            stats.per_sample_nll.append(mean_nll(lp))
            if not stats.sample_ids:
                stats.sample_ids.append(i)
            if progress and (i + 1) % 10 == 0:
                logger.info("[perplexity:%s] %d/%d datapoints", mode, i + 1, len(records))
        return stats

    def compare(
        self,
        records: Sequence[Any],
        modes: Sequence[str] = PPL_MODES,
        reference: str = "vanilla",
        max_datapoints: Optional[int] = None,
        include_instruct: bool = True,
        progress: bool = False,
    ) -> PerplexityComparison:
        """Measure every mode on the same datapoints and correlate the PPLs."""
        modes = tuple(m for m in modes if m != "instruct" or include_instruct)
        comparison = PerplexityComparison(
            reference=reference, gamma=self.gamma, temperature=self.temperature
        )
        per_sample: Dict[str, List[float]] = {}
        for mode in modes:
            if mode == "instruct" and self.instruct_wrapper is None:
                logger.info("skipping mode 'instruct' (no instruct_wrapper)")
                continue
            stats = self.measure(
                records,
                mode=mode,
                max_datapoints=max_datapoints,
                progress=progress,
            )
            comparison.stats[mode] = stats
            per_sample[mode] = stats.per_sample
        comparison.correlations = correlation_table(per_sample)
        return comparison


def _split_record(record: Any) -> Tuple[str, str]:
    """Extract ``(prompt, continuation)`` from a P3 sample / dict / tuple."""
    if isinstance(record, tuple) and len(record) == 2:
        return str(record[0]), str(record[1])
    if isinstance(record, dict):
        prompt = record.get("prompt", record.get("inputs", record.get("inputs_pretokenized", "")))
        cont = record.get(
            "continuation", record.get("targets", record.get("targets_pretokenized", record.get("target", "")))
        )
        return str(prompt), str(cont)
    for attr_p, attr_t in (
        ("inputs", "targets"),
        ("prompt", "continuation"),
        ("question", "answer"),
    ):
        if hasattr(record, attr_p) and hasattr(record, attr_t):
            return str(getattr(record, attr_p)), str(getattr(record, attr_t))
    raise TypeError(f"cannot extract (prompt, continuation) from {type(record)!r}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def perplexity_report(comparison: PerplexityComparison) -> Dict[str, Any]:
    """JSON-serialisable Section 5.2 report (means + correlations + anchors)."""
    return {
        "mean_ppl": comparison.mean_ppl,
        "correlations": comparison.correlations,
        "check_against_paper": comparison.check_against_paper(),
        "n_datapoints": {k: s.n_datapoints for k, s in comparison.stats.items()},
        "totals": {
            "gamma": comparison.gamma,
            "reference": comparison.reference,
        },
    }


def check_against_paper(
    comparison: PerplexityComparison, tolerance: float = PPL_TOLERANCE
) -> Dict[str, Any]:
    """Convenience wrapper around :meth:`PerplexityComparison.check_against_paper`."""
    return comparison.check_against_paper(tolerance=tolerance)


def format_ppl_table(
    comparison: PerplexityComparison, modes: Sequence[str] = PPL_MODES
) -> str:
    """Plain-text rendering of the Section 5.2 perplexity / correlation table."""
    lines = ["Mode                 Mean PPL     Median     Std     N datapoints  N tokens"]
    lines.append("-" * 80)
    for mode in modes:
        stats = comparison.stats.get(mode)
        if stats is None:
            continue
        lines.append(
            f"{MODE_LABELS.get(mode, mode):<20} "
            f"{stats.mean_ppl:>9.3f} {stats.median_ppl:>10.3f} {stats.std_ppl:>8.3f} "
            f"{stats.n_datapoints:>12d} {stats.total_tokens:>10d}"
        )
    lines.append("")
    lines.append("Correlations (per-datapoint PPL)")
    lines.append("-" * 80)
    for key, entry in comparison.correlations.items():
        lines.append(
            f"{key:<24} pearson={entry['pearson']:+.3f}  spearman={entry['spearman']:+.3f}  "
            f"n={entry['n']:<5d} paper~{entry['paper']:.2f}"
        )
    if not comparison.correlations:
        lines.append("(none -- at least two aligned modes are required)")
    return "\n".join(lines)


def summarize_ppl(stats: PerplexityStats) -> str:
    """One-line textual summary of a :class:`PerplexityStats` record."""
    return (
        f"{stats.name} (gamma={stats.gamma:g}): mean PPL={stats.mean_ppl:.3f} "
        f"median={stats.median_ppl:.3f} over {stats.n_datapoints} datapoints "
        f"({stats.total_tokens} continuation tokens)"
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _demo() -> bool:
    """Pure-numpy identities + correlation behaviour; returns True on success."""
    # 1) uniform distribution over V tokens -> PPL = V
    V = 8
    logits = np.zeros(V)
    lp = log_probs_from_logits(logits)
    assert abs(perplexity_from_logprobs(lp) - V) < 1e-9, "uniform PPL must equal V"

    # 2) deterministic (one-hot) distribution -> PPL ~ 1
    peaked = np.array([50.0] + [0.0] * (V - 1))
    lp2 = log_probs_from_logits(peaked)
    assert perplexity_from_logprobs(lp2) < 1.0001, "one-hot PPL must be ~1"

    # 3) gathering log-probs at target ids matches the softmax entries
    probs = _softmax(logits)
    gathered = log_probs_from_logits(logits, target_ids=[0, 3])
    assert np.allclose(np.exp(gathered), [probs[0], probs[3]]), "gather mismatch"

    # 4) Eq. 7 identities
    cond = np.array([[1.0, 2.0, 3.0]])
    uncond = np.array([[0.5, 0.0, -1.0]])
    assert np.allclose(_guided_logits(cond, uncond, 1.0), cond)
    assert np.allclose(_guided_logits(cond, uncond, 0.0), uncond)
    assert np.allclose(_guided_logits(cond, uncond, 2.0), uncond + 2.0 * (cond - uncond))

    # 5) CFG sharpening reduces the (vanilla) continuation perplexity
    step = np.array([3.0, 1.0, 0.0, -1.0, -2.0])
    base = np.array([1.5, 1.0, 0.5, 0.0, -0.5])
    target = [0, 0, 1, 1, 2]
    lp_vanilla = log_probs_from_logits(_guided_logits(step, base, 1.0), target_ids=target)
    lp_cfg = log_probs_from_logits(_guided_logits(step, base, 1.5), target_ids=target)
    assert perplexity_from_logprobs(lp_cfg) < perplexity_from_logprobs(lp_vanilla)

    # 6) correlation identities
    a = np.arange(10, dtype=np.float64)
    assert abs(pearson(a, a) - 1.0) < 1e-9
    assert abs(spearman_correlation(a, 2 * a + 1) - 1.0) < 1e-9
    assert abs(spearman_correlation(a, a[::-1]) + 1.0) < 1e-9
    tbl = correlation_table({"cfg": a, "vanilla": a, "instruct": a[::-1]})
    assert abs(tbl["cfg_vs_vanilla"]["pearson"] - 1.0) < 1e-9
    assert abs(tbl["instruct_vs_cfg"]["spearman"] + 1.0) < 1e-9

    # 7) PPL = exp(mean NLL) and corpus-level aggregation
    stats = PerplexityStats(
        per_sample=[2.0, 4.0],
        n_tokens=[1, 3],
        per_sample_nll=[math.log(2.0), math.log(4.0)],
        name="cfg",
    )
    assert abs(stats.mean_ppl - 3.0) < 1e-9
    assert abs(stats.dataset_ppl - math.exp((math.log(2.0) * 1 + math.log(4.0) * 3) / 4)) < 1e-9

    # 8) record parsing
    assert _split_record(("p", "c")) == ("p", "c")
    assert _split_record({"inputs": "p", "targets": "c"}) == ("p", "c")

    logger.info("perplexity.py self-test passed")
    return True


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _demo()
    print("OK")
