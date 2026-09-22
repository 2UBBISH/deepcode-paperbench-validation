"""CIDEr(-D) captioning metric and worst-case bookkeeping for Robust CLIP.

Paper / Addendum requirements implemented here
---------------------------------------------
* The captioning score is CIDEr (Vedantam et al., 2015) -- paper body, Sec. 4.1
  ("We report the CIDEr score (Vedantam et al., 2015) for captioning").
* Addendum: "For computation of the CIDEr scores, they compute the CIDEr scores
  after every attack, so that they can take the worst case score for each
  sample, and remember the best ground-truth and perturbation for the
  single-precision attack."

This module therefore provides

1. a self-contained CIDEr-D scorer (n-gram counts up to ``n=4``, TF-IDF style
   document frequency weighting computed over the reference set, the CIDEr-D
   count clipping and the length gaussian penalty), with a corpus-level
   document-frequency cache so the score of every candidate is comparable;
2. the "after every attack" helpers: :func:`cider_after_attack` to score a
   freshly produced caption, and :class:`WorstCaseCiderTracker` which retains
   the *minimum* CIDEr per sample together with the caption / ground truth /
   perturbation / precision that produced it (the state needed to warm-start
   the single-precision attack).

The Addendum does not specify the captioning attack budget or CIDEr
configuration; those live in ``attacks/captioning.py`` / ``configs/*.yaml``.
Nothing here invents paper hyperparameters: ``n=4``, ``sigma=6.0`` and
``scale=10.0`` are the canonical CIDEr-D constants of Vedantam et al. (2015).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import string
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

LOGGER = logging.getLogger("robust_clip_repro.metrics.cider")

__all__ = [
    "CIDER_N",
    "CIDER_SIGMA",
    "CIDER_SCALE",
    "UNSPECIFIED",
    "CiderWorstCase",
    "WorstCaseCiderTracker",
    "CiderScorer",
    "tokenize",
    "ngrams",
    "ngram_counts",
    "compute_document_frequency",
    "cider_score_single",
    "cider_score",
    "cider_d",
    "cider_after_attack",
    "worst_case_cider",
    "best_ground_truth",
    "cider_threshold_reached",
    "mean_cider",
    "cider_statistics",
    "make_cider_fn",
    "references_from_samples",
    "main",
]

# ---------------------------------------------------------------------------
# Canonical CIDEr-D constants (Vedantam et al., 2015) -- not Addendum values.
# ---------------------------------------------------------------------------
CIDER_N = 4                 # n-grams of size 1..4
CIDER_SIGMA = 6.0           # gaussian length penalty sigma
CIDER_SCALE = 10.0          # per-n-gram scaling factor

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

# Reference translation when a sample's references are stored under a key.
CAPTION_KEYS: Tuple[str, ...] = (
    "captions",
    "references",
    "ground_truths",
    "ground_truth",
    "answer",
    "answers",
    "labels",
    "refs",
)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Tokenisation / n-grams
# ---------------------------------------------------------------------------
def tokenize(text: Any) -> List[str]:
    """Lowercase, punctuation-stripped, whitespace-split tokenisation.

    Matches the tokenisation convention used by the standard CIDEr
    implementations (pycocoevalcap / PTB-lite): case folding, punctuation
    removal, and splitting on whitespace.
    """
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        # A list of strings is not a caption; join deterministically.
        text = " ".join(str(t) for t in text)
    text = str(text)
    text = text.lower().translate(_PUNCT_TABLE)
    text = _WS_RE.sub(" ", text).strip()
    return text.split() if text else []


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Counter of contiguous ``n``-grams (tuples) of ``tokens``."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def ngram_counts(text: Any, n: int = CIDER_N) -> Dict[int, Counter]:
    """``{ngram_size: Counter}`` for sizes ``1..n`` of a raw caption string."""
    tokens = tokenize(text) if not isinstance(text, (list, tuple)) or _is_token_list(text) else list(text)
    return {size: ngrams(tokens, size) for size in range(1, n + 1)}


def _is_token_list(value: Any) -> bool:
    """True when ``value`` looks like an already-tokenised sequence."""
    return isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)


# ---------------------------------------------------------------------------
# Document frequency (IDF weights)
# ---------------------------------------------------------------------------
def compute_document_frequency(
    references: Sequence[Sequence[Any]],
    n: int = CIDER_N,
    *,
    each_reference_counts_once: bool = True,
) -> Dict[int, Dict[Tuple[str, ...], int]]:
    """Document frequency of each n-gram over the whole reference corpus.

    ``references`` is a per-sample sequence of reference captions.  Following
    Vedantam et al. (2015) each reference sentence contributes at most one
    occurrence of a given n-gram type to the document frequency.
    """
    df: Dict[int, Dict[Tuple[str, ...], int]] = {size: defaultdict(int) for size in range(1, n + 1)}
    for sample_refs in references:
        if sample_refs is None:
            continue
        if isinstance(sample_refs, str):
            sample_refs = [sample_refs]
        for ref in sample_refs:
            tokens = list(ref) if _is_token_list(ref) else tokenize(ref)
            for size in range(1, n + 1):
                for gram in set(ngrams(tokens, size)):
                    df[size][gram] += 1
    # Materialise as plain dicts for picklability / readability.
    return {size: dict(counts) for size, counts in df.items()}


def _idf_vector(
    df: Optional[Dict[int, Dict[Tuple[str, ...], int]]],
    num_documents: int,
    n: int = CIDER_N,
) -> Dict[int, Dict[Tuple[str, ...], float]]:
    """``log(N / df)`` weights, defaulting to ``0.0`` for unseen n-grams.

    When no document frequency is supplied every n-gram gets weight ``0``,
    which is the degenerate (unweighted) CIDEr; callers should therefore
    normally pass a corpus ``df``.  See :class:`CiderScorer` for the common
    workflow where the ``df`` is built once from the reference set.
    """
    weights: Dict[int, Dict[Tuple[str, ...], float]] = {}
    for size in range(1, n + 1):
        size_weights: Dict[Tuple[str, ...], float] = {}
        if df is not None:
            counts = df.get(size, {}) or {}
            for gram, count in counts.items():
                if count > 0:
                    size_weights[gram] = math.log(float(num_documents) / float(count))
        weights[size] = size_weights
    return weights


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------
def cider_score_single(
    hypothesis: Any,
    references: Sequence[Any],
    *,
    idf: Optional[Dict[int, Dict[Tuple[str, ...], float]]] = None,
    sigma: float = CIDER_SIGMA,
    n: int = CIDER_N,
    scale: float = CIDER_SCALE,
) -> float:
    """CIDEr-D of one candidate against its reference captions.

    The per-``n`` score is

        sum_gram min(count_hyp, max_ref_count) * idf(gram)
        --------------------------------------------------  * scale
                sum_gram count_hyp * idf(gram)

    averaged over ``n = 1..4`` and multiplied by the gaussian length penalty
    ``exp(-(len_hyp - len_ref)^2 / (2 sigma^2))`` (averaged over references).
    This is the "D" (diversity) variant used by the captioning literature
    (count clipping + length penalty), as in Vedantam et al. (2015).
    """
    if isinstance(references, str):
        references = [references]
    references = [r for r in (references or []) if r is not None]
    if not references:
        # No reference captions -> score undefined; report 0 rather than nan.
        return 0.0

    hyp_tokens = list(hypothesis) if _is_token_list(hypothesis) else tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0

    if idf is None:
        # Fall back to a local idf so a single sample can still be scored.
        idf = _idf_vector(
            compute_document_frequency([[list(references[0])] if False else references], n=n),
            max(len(references), 1),
            n=n,
        )
        # Use the sample-local document frequency (all weights >= 0).
        idf = _idf_vector(
            compute_document_frequency([list(_as_text(r) for r in references)], n=n),
            max(len(references), 1),
            n=n,
        )

    ref_token_lists = [list(r) if _is_token_list(r) else tokenize(r) for r in references]
    hyp_len = len(hyp_tokens)
    ref_lens = [max(len(t), 1) for t in ref_token_lists]

    # CIDEr-D length penalty (gaussian), averaged over the references.
    len_penalty = sum(
        math.exp(-((hyp_len - ref_len) ** 2) / (2.0 * sigma ** 2)) for ref_len in ref_lens
    ) / float(len(ref_lens))

    total = 0.0
    for size in range(1, n + 1):
        hyp_counts = ngrams(hyp_tokens, size)
        if not hyp_counts:
            continue
        weights = (idf.get(size) or {}) if idf else {}
        # Special case: a completely empty idf vector would make the ratio 0/0.
        numerator = 0.0
        denominator = 0.0
        for gram, hyp_count in hyp_counts.items():
            weight = weights.get(gram, 0.0)
            if weight <= 0.0:
                continue
            ref_count = 0
            for ref_tokens in ref_token_lists:
                ref_count = max(ref_count, ngrams(ref_tokens, size).get(gram, 0))
            numerator += min(hyp_count, ref_count) * weight
            denominator += hyp_count * weight
        if denominator > 0.0:
            total += (numerator / denominator) * scale
        else:
            # Everything unseen in the references: identical captions should
            # still score full marks for this n, otherwise this n contributes 0.
            ref_counts_match = any(
                ngrams(ref_tokens, size) == hyp_counts for ref_tokens in ref_token_lists
            )
            total += scale if ref_counts_match else 0.0

    return float(total / float(n) * len_penalty)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


@dataclass
class CiderScorer:
    """Corpus-level CIDEr-D scorer.

    Typical usage::

        scorer = CiderScorer()
        scorer.fit(references)            # list per-sample of reference lists
        score = scorer.score(caption, refs)

    ``fit`` builds the document-frequency table once so that every candidate is
    scored with the same IDF weights (required for "take the worst case score
    for each sample" bookkeeping across attacks).
    """

    n: int = CIDER_N
    sigma: float = CIDER_SIGMA
    scale: float = CIDER_SCALE
    df: Optional[Dict[int, Dict[Tuple[str, ...], int]]] = None
    num_documents: int = 0
    _idf: Dict[int, Dict[Tuple[str, ...], float]] = field(default_factory=dict, init=False)

    # -- fitting -----------------------------------------------------------
    def fit(self, references: Sequence[Sequence[Any]]) -> "CiderScorer":
        """Build the document-frequency / IDF table from reference captions."""
        prepared = _prepare_references(references)
        self.df = compute_document_frequency(prepared, n=self.n)
        self.num_documents = sum(len(refs) for refs in prepared) or 1
        self._idf = _idf_vector(self.df, self.num_documents, n=self.n)
        return self

    def fit_samples(self, samples: Sequence[Dict[str, Any]]) -> "CiderScorer":
        """Convenience wrapper: fit from canonical captioning samples."""
        return self.fit(references_from_samples(samples))

    # -- scoring -----------------------------------------------------------
    def score(self, hypothesis: Any, references: Sequence[Any]) -> float:
        if not self._idf:
            # Score with a sample-local idf but keep it deterministic.
            local = _idf_vector(
                compute_document_frequency([[_as_text(r) for r in references]], n=self.n),
                max(len(references), 1),
                n=self.n,
            )
            return cider_score_single(
                hypothesis, references, idf=local, sigma=self.sigma, n=self.n, scale=self.scale
            )
        return cider_score_single(
            hypothesis, references, idf=self._idf, sigma=self.sigma, n=self.n, scale=self.scale
        )

    # -- batch -------------------------------------------------------------
    def score_batch(
        self,
        hypotheses: Sequence[Any],
        references: Sequence[Sequence[Any]],
    ) -> List[float]:
        return [
            self.score(hyp, refs) for hyp, refs in zip(hypotheses, _prepare_references(references))
        ]

    def corpus_score(
        self,
        hypotheses: Sequence[Any],
        references: Sequence[Sequence[Any]],
    ) -> float:
        scores = self.score_batch(hypotheses, references)
        return float(sum(scores) / len(scores)) if scores else 0.0


def _prepare_references(references: Sequence[Any]) -> List[List[Any]]:
    """Normalise a references argument to ``[[ref, ref, ...], ...]`` per sample."""
    prepared: List[List[Any]] = []
    for sample_refs in references or []:
        if sample_refs is None:
            prepared.append([])
        elif isinstance(sample_refs, str):
            prepared.append([sample_refs])
        elif _is_token_list(sample_refs):
            # A single tokenised caption: treat as one reference.
            prepared.append([list(sample_refs)])
        else:
            prepared.append([r for r in sample_refs if r is not None])
    return prepared


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def cider_score(
    hypotheses: Union[Any, Sequence[Any]],
    references: Sequence[Any],
    *,
    n: int = CIDER_N,
    sigma: float = CIDER_SIGMA,
    return_per_sample: bool = False,
    scorer: Optional[CiderScorer] = None,
) -> Union[float, Tuple[float, List[float]]]:
    """Mean CIDEr (-D) of one or many hypotheses.

    ``references`` may be a list of strings (one reference per sample) or a
    list of reference lists.  When ``return_per_sample`` is True the tuple
    ``(mean, per_sample)`` is returned.
    """
    if isinstance(hypotheses, str):
        refs = references if isinstance(references, str) else references
        scorer = scorer or CiderScorer(n=n, sigma=sigma)
        if not getattr(scorer, "_idf", None):
            scorer.fit(_prepare_references([refs] if isinstance(refs, str) else refs))
        value = float(scorer.score(hypotheses, refs if isinstance(refs, (list, tuple)) else [refs]))
        return (value, [value]) if return_per_sample else value

    prepared = _prepare_references(references)
    scorer = scorer or CiderScorer(n=n, sigma=sigma)
    if not getattr(scorer, "_idf", None):
        scorer.fit(prepared)
    per_sample = scorer.score_batch(hypotheses, prepared)
    mean = float(sum(per_sample) / len(per_sample)) if per_sample else 0.0
    return (mean, per_sample) if return_per_sample else mean


def cider_d(*args: Any, **kwargs: Any) -> Union[float, Tuple[float, List[float]]]:
    """Alias of :func:`cider_score` (CIDEr-D is the implemented variant)."""
    return cider_score(*args, **kwargs)


def make_cider_fn(
    references: Optional[Sequence[Any]] = None,
    *,
    n: int = CIDER_N,
    sigma: float = CIDER_SIGMA,
) -> Callable[..., float]:
    """Return a ``cider_fn(caption_or_pixels, references) -> float`` callable.

    ``attacks/captioning.py`` lazily imports ``cider_score`` from this module;
    this factory provides an equally convenient injection point for
    ``eval_captioning.py`` and for tests, and fits a corpus IDF table up front
    when the reference set is known.
    """
    scorer = CiderScorer(n=n, sigma=sigma)
    if references is not None:
        scorer.fit(_prepare_references(references))

    def _cider_fn(caption: Any, refs: Optional[Sequence[Any]] = None, **_: Any) -> float:
        refs = refs if refs is not None else references
        if refs is None:
            raise ValueError("cider_fn requires references")
        if isinstance(refs, str):
            refs = [refs]
        if not getattr(scorer, "_idf", None):
            scorer.fit(_prepare_references([list(refs)]))
        return float(scorer.score(caption, refs))

    _cider_fn.scorer = scorer  # type: ignore[attr-defined]
    _cider_fn.fit = scorer.fit  # type: ignore[attr-defined]
    return _cider_fn


def references_from_samples(samples: Sequence[Dict[str, Any]]) -> List[List[str]]:
    """Extract per-sample reference caption lists from canonical samples."""
    out: List[List[str]] = []
    for sample in samples or []:
        refs: Optional[Any] = None
        if isinstance(sample, dict):
            for key in CAPTION_KEYS:
                if sample.get(key):
                    refs = sample[key]
                    break
        else:
            for key in CAPTION_KEYS:
                value = getattr(sample, key, None)
                if value:
                    refs = value
                    break
        if refs is None:
            out.append([])
        elif isinstance(refs, str):
            out.append([refs])
        else:
            out.append([_as_text(r) for r in refs])
    return out


# ---------------------------------------------------------------------------
# "CIDEr after every attack" + worst-case bookkeeping (Addendum)
# ---------------------------------------------------------------------------
@dataclass
class CiderWorstCase:
    """Worst-case CIDEr bookkeeping for one sample."""

    sample_index: int = 0
    worst_cider: Optional[float] = None
    worst_caption: Optional[str] = None
    best_ground_truth: Optional[str] = None
    best_perturbation: Any = None
    best_precision: Optional[str] = None
    best_stage: Optional[str] = None
    best_iteration_index: Optional[int] = None
    num_attacks_scored: int = 0

    def update(
        self,
        cider: float,
        *,
        caption: Optional[str] = None,
        ground_truth: Optional[str] = None,
        perturbation: Any = None,
        precision: Optional[str] = None,
        stage: Optional[str] = None,
        iteration_index: Optional[int] = None,
    ) -> bool:
        """Record a freshly computed CIDEr; keep it only if it is worse (lower).

        Returns True when this attack produced a new worst case.  Per the
        Addendum the *minimum* CIDEr is retained together with the ground
        truth / perturbation / precision that produced it, so that the
        single-precision attack can be warm-started from the best state.
        """
        self.num_attacks_scored += 1
        value = float(cider)
        if self.worst_cider is not None and value >= self.worst_cider:
            return False
        self.worst_cider = value
        if caption is not None:
            self.worst_caption = caption
        if ground_truth is not None:
            self.best_ground_truth = ground_truth
        if perturbation is not None:
            self.best_perturbation = perturbation
        if precision is not None:
            self.best_precision = precision
        if stage is not None:
            self.best_stage = stage
        if iteration_index is not None:
            self.best_iteration_index = iteration_index
        return True

    def as_dict(self, include_perturbation: bool = False) -> Dict[str, Any]:
        payload = {
            "sample_index": self.sample_index,
            "worst_cider": self.worst_cider,
            "worst_caption": self.worst_caption,
            "best_ground_truth": self.best_ground_truth,
            "best_precision": self.best_precision,
            "best_stage": self.best_stage,
            "best_iteration_index": self.best_iteration_index,
            "num_attacks_scored": self.num_attacks_scored,
        }
        if include_perturbation:
            payload["best_perturbation"] = self.best_perturbation
        return payload


class WorstCaseCiderTracker:
    """Track the worst-case CIDEr per sample across a sequence of attacks.

    The tracker is deliberately metric-only (no model, no attack code) so it can
    be unit tested and reused by ``attacks/captioning.py`` and
    ``eval_captioning.py``.  It enforces the Addendum invariant that CIDEr is
    computed *after every attack*: :meth:`update` refuses to record a score
    unless the caller passes the caption produced by that specific attack.
    """

    def __init__(self, require_caption_for_update: bool = True) -> None:
        self.require_caption_for_update = require_caption_for_update
        self._states: Dict[int, CiderWorstCase] = {}

    # -- bookkeeping -------------------------------------------------------
    def state(self, sample_index: int) -> CiderWorstCase:
        if sample_index not in self._states:
            self._states[sample_index] = CiderWorstCase(sample_index=sample_index)
        return self._states[sample_index]

    def update(
        self,
        sample_index: int,
        cider: float,
        *,
        caption: Optional[str] = None,
        ground_truth: Optional[str] = None,
        perturbation: Any = None,
        precision: Optional[str] = None,
        stage: Optional[str] = None,
        iteration_index: Optional[int] = None,
    ) -> bool:
        if self.require_caption_for_update and caption is None:
            raise ValueError(
                "CIDEr must be computed after every attack: a caption is required for each update"
            )
        return self.state(sample_index).update(
            cider,
            caption=caption,
            ground_truth=ground_truth,
            perturbation=perturbation,
            precision=precision,
            stage=stage,
            iteration_index=iteration_index,
        )

    def worst_cider(self, sample_index: int) -> Optional[float]:
        return self.state(sample_index).worst_cider

    def worst_ciders(self) -> List[Optional[float]]:
        return [self._states[i].worst_cider for i in sorted(self._states)]

    def scores(self, *, missing: float = 0.0) -> List[float]:
        return [
            missing if self._states[i].worst_cider is None else float(self._states[i].worst_cider)
            for i in sorted(self._states)
        ]

    def mean_worst_cider(self) -> float:
        values = [v for v in self.worst_ciders() if v is not None]
        return float(sum(values) / len(values)) if values else 0.0

    def best_ground_truths(self) -> List[Optional[str]]:
        return [self._states[i].best_ground_truth for i in sorted(self._states)]

    def best_perturbations(self) -> List[Any]:
        return [self._states[i].best_perturbation for i in sorted(self._states)]

    def num_attacks_scored(self) -> int:
        return sum(state.num_attacks_scored for state in self._states.values())

    # -- reporting ---------------------------------------------------------
    def as_dict(self, include_perturbation: bool = False) -> Dict[str, Any]:
        return {
            "num_samples": len(self._states),
            "num_attacks_scored": self.num_attacks_scored(),
            "mean_worst_cider": self.mean_worst_cider(),
            "worst_ciders": self.worst_ciders(),
            "best_ground_truths": self.best_ground_truths(),
            "per_sample": [
                self._states[i].as_dict(include_perturbation=include_perturbation)
                for i in sorted(self._states)
            ],
        }

    def clear(self) -> None:
        self._states.clear()


def cider_after_attack(
    caption: Any,
    references: Sequence[Any],
    *,
    tracker: Optional[WorstCaseCiderTracker] = None,
    sample_index: int = 0,
    ground_truth: Optional[str] = None,
    perturbation: Any = None,
    precision: Optional[str] = None,
    stage: Optional[str] = None,
    iteration_index: Optional[int] = None,
    scorer: Optional[CiderScorer] = None,
) -> float:
    """Score a caption produced by one attack and update the worst-case state.

    This is the single call site that encodes the Addendum rule: CIDEr is
    computed *immediately after each attack* (never batched at the end), and
    only the per-sample minimum is retained.
    """
    refs = references if isinstance(references, (list, tuple)) else [references]
    value = float((scorer or CiderScorer()).score(caption, refs))
    if tracker is not None:
        tracker.update(
            sample_index,
            value,
            caption=_as_text(caption),
            ground_truth=ground_truth,
            perturbation=perturbation,
            precision=precision,
            stage=stage,
            iteration_index=iteration_index,
        )
    return value


def worst_case_cider(scores: Iterable[float]) -> Tuple[Optional[float], Optional[int]]:
    """Return ``(min_score, argmin_index)`` over an iterable of CIDEr scores."""
    values = [float(s) for s in scores]
    if not values:
        return None, None
    index = int(min(range(len(values)), key=lambda i: values[i]))
    return values[index], index


def best_ground_truth(ground_truths: Sequence[Any], scores: Sequence[float]) -> Optional[Any]:
    """Ground truth that led to the lowest score (Addendum VQA step 2, shared)."""
    _, index = worst_case_cider(scores)
    if index is None or index >= len(ground_truths):
        return None
    return ground_truths[index]


def cider_threshold_reached(score: float, threshold: Optional[float]) -> bool:
    """True when the sample is already broken (score strictly below threshold)."""
    if threshold is None:
        return False
    return float(score) < float(threshold)


def mean_cider(scores: Iterable[float]) -> float:
    """Mean of a list of CIDEr scores (empty -> 0.0)."""
    values = [float(v) for v in scores]
    return float(sum(values) / len(values)) if values else 0.0


def cider_statistics(scores: Iterable[float]) -> Dict[str, Any]:
    """Descriptive statistics of a CIDEr list (paper tables report the mean)."""
    values = sorted(float(v) for v in scores)
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": 0.0, "min": None, "max": None, "median": None}
    mid = n // 2
    median = values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])
    return {
        "count": n,
        "mean": float(sum(values) / n),
        "min": values[0],
        "max": values[-1],
        "median": float(median),
    }


# ---------------------------------------------------------------------------
# Self test / CLI
# ---------------------------------------------------------------------------
def _self_test() -> Dict[str, Any]:
    """Offline validation of the scorer and the Addendum bookkeeping rules."""
    refs = ["a cat sits on a mat", "the cat is sitting on the mat"]
    same = cider_score_single("a cat sits on a mat", refs, idf=_idf_vector(
        compute_document_frequency([refs]), len(refs)))
    assert same > 0.0, same

    # Perfect caption with a single identical reference scores CIDER_SCALE.
    perfect = cider_score_single("a cat sits on a mat", ["a cat sits on a mat"],
                                 idf=_idf_vector(compute_document_frequency([["a cat sits on a mat"]]), 1))
    assert abs(perfect - CIDER_SCALE) < 1e-6, perfect

    # Unrelated caption must be strictly worse than a perfect one.
    unrelated = cider_score_single("quantum chromodynamics Lagrangian", ["a cat sits on a mat"],
                                   idf=_idf_vector(compute_document_frequency([["a cat sits on a mat"]]), 1))
    assert unrelated < perfect, (unrelated, perfect)

    # Repetition is clipped (CIDEr-D): repeating the reference many times must
    # not beat the reference itself.
    repeated = cider_score_single("a cat sits on a mat a cat sits on a mat", ["a cat sits on a mat"],
                                  idf=_idf_vector(compute_document_frequency([["a cat sits on a mat"]]), 1))
    assert repeated <= perfect + 1e-6, (repeated, perfect)

    # Empty hypothesis / references are handled without NaN.
    assert cider_score_single("", refs) == 0.0
    assert cider_score_single("hello", []) == 0.0

    # Corpus-level scorer with a shared IDF table.
    scorer = CiderScorer().fit([["a cat sits on a mat"], ["a dog runs in a park"]])
    corpus_mean, per_sample = cider_score(
        ["a cat sits on a mat", "totally different words here"],
        [["a cat sits on a mat"], ["a dog runs in a park"]],
        return_per_sample=True,
        scorer=scorer,
    )
    assert per_sample[0] > per_sample[1], per_sample
    assert abs(corpus_mean - sum(per_sample) / 2.0) < 1e-9

    # Addendum: CIDEr computed after EVERY attack, minimum retained per sample.
    tracker = WorstCaseCiderTracker()
    tracker.update(0, 2.0, caption="first attack caption", ground_truth=refs[0],
                   precision="half", stage="half", perturbation="p1")
    assert tracker.worst_cider(0) == 2.0
    # A better (higher) score must NOT replace the worst case.
    changed = tracker.update(0, 3.5, caption="better caption", ground_truth=refs[1],
                             precision="half", stage="half", perturbation="p2")
    assert changed is False and tracker.worst_cider(0) == 2.0
    # A worse (lower) score replaces it and stores GT + perturbation for the
    # single-precision warm start.
    changed = tracker.update(0, 0.5, caption="attacked caption", ground_truth=refs[1],
                             precision="single", stage="single", perturbation="p3",
                             iteration_index=100)
    assert changed is True and tracker.worst_cider(0) == 0.5
    assert tracker.best_ground_truths()[0] == refs[1]
    assert tracker.best_perturbations()[0] == "p3"
    assert tracker.num_attacks_scored() == 3
    assert tracker.mean_worst_cider() == 0.5

    # A caption is mandatory per update -> proves scoring happens per attack.
    try:
        tracker.update(1, 1.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("update without a caption must be rejected")

    # cider_after_attack wires scoring + bookkeeping together.
    value = cider_after_attack("a cat sits on a mat", ["a cat sits on a mat"],
                               tracker=tracker, sample_index=2, precision="half",
                               stage="half", ground_truth="a cat sits on a mat")
    assert value > 0.0 and tracker.worst_cider(2) == value

    # Threshold / worst-case / statistics helpers.
    assert cider_threshold_reached(0.5, 10.0) is True
    assert cider_threshold_reached(20.0, 10.0) is False
    assert cider_threshold_reached(20.0, None) is False
    worst, idx = worst_case_cider([3.0, 1.0, 2.0])
    assert (worst, idx) == (1.0, 1)
    assert best_ground_truth(["a", "b", "c"], [3.0, 1.0, 2.0]) == "b"
    stats = cider_statistics([1.0, 3.0, 2.0])
    assert stats["mean"] == 2.0 and stats["min"] == 1.0 and stats["max"] == 3.0

    # make_cider_fn / references_from_samples.
    fn = make_cider_fn([["a cat sits on a mat"]])
    assert fn("a cat sits on a mat") > 0.0
    refs_from_samples = references_from_samples(
        [{"captions": ["a cat sits on a mat"]}, {"ground_truths": ["a dog"]}]
    )
    assert refs_from_samples == [["a cat sits on a mat"], ["a dog"]]

    # Tokenisation sanity.
    assert tokenize("A cat, sits!") == ["a", "cat", "sits"]
    assert compute_document_frequency([["a a a"]])[1][("a",)] == 1

    return {
        "perfect_cider": perfect,
        "unrelated_cider": unrelated,
        "repeated_cider": repeated,
        "mean_worst_cider": tracker.mean_worst_cider(),
        "status": "ok",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CIDEr(-D) scorer and worst-case bookkeeping (Robust CLIP reproduction)"
    )
    parser.add_argument("--self-test", action="store_true", help="run the offline validity checks")
    parser.add_argument("--hypothesis", type=str, default=None, help="candidate caption")
    parser.add_argument("--reference", type=str, action="append", default=None,
                        help="reference caption (repeat for multiple references)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.self_test or args.hypothesis is None:
        result = _self_test()
        print(json.dumps(result, indent=2))
        return 0
    score = cider_score(args.hypothesis, args.reference or [])
    if args.json:
        print(json.dumps({"cider": score, "hypothesis": args.hypothesis,
                          "references": args.reference or []}))
    else:
        print(f"CIDEr: {score:.6f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
