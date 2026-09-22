"""Positive/negative sample buffers and the SEL / outcome-supervision rules.

Paper reference
---------------
BBox-Adapter Section 3.4 ("Online Adaptation"), Eq. (5) and Eq. (6), plus the
``Initialization`` paragraph and Section 4.1 ("Settings").

The two sample sets that the ranking-based NCE loss Eq. (2)/(3) consumes are
maintained *per query*:

* positive samples are drawn from the real data distribution
  ``y_+ ~ p_data(y | x)``  (ground truth, human preference, or GPT-4 feedback),
* negative samples are drawn from the adapter's own adapted inference
  ``y_- ~ p_theta(y | x)``.

Initialization (Section 3.4)
----------------------------
For each input query ``x_i`` we prompt the black-box LLM to generate ``K``
responses ``{y_{i,j}}_{j=1}^K``.  The best response (per ground truth or
human/AI feedback) becomes the initial positive sample::

    y_{i+}^{(0)} = y_{i,k} = SEL({y_{i,j}}_{j=1}^{K})

and the remaining candidates serve as the initial negative cases::

    y_{i-}^{(0)} = {y_{i,j} | j != k}_{j=1}^{K}.

When no ground truth exists, the initial negatives can alternatively come from
``p_{theta_0}`` with a randomly initialized adapter.

Update rules (Section 3.4, Eq. 5/6)
-----------------------------------
Given ``M`` freshly sampled candidates from the current adapted inference
``p_{theta_t}``::

    y_{i+}^{(t)} = SEL(y_{i+}^{(t-1)}, {yhat_{i,m}}_{m=1}^{M})      (Eq. 5)
    y_{i-}^{(t)} = {yhat_{i,m} | yhat_{i,m} != y_{i+}^{(t)}}        (Eq. 6)

Note that Eq. (6) is evaluated on the *newly sampled* candidate set, so the
negative set is refreshed every outer iteration, while the positive set is
cumulative through ``SEL(previous positive, new candidates)``.

Outcome supervision (Section 4.1, "Settings")
--------------------------------------------
Applied in *all* settings: any adapted inference whose final answer matches the
training-set (i.e. current positive) answers joins the positives, everything
else stays/joins the negatives.

Nothing in this module talks to the black-box LLM's logprobs, hidden states or
gradients: candidates are plain text strings obtained through the text-only
proposal API.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..data.answer_extraction import (
    extract_final_answer,
    normalize_answer,
)
from ..feedback.ai_feedback import (
    parse_feedback_index,
    select_by_ai_feedback,
    split_positives_negatives as _split_helpers,
)

logger = logging.getLogger(__name__)

__all__ = [
    # config / records
    "BufferConfig",
    "QuerySamples",
    "SampleBuffer",
    # selectors
    "ground_truth_select",
    "ai_feedback_select",
    "make_selector",
    "SEL_MODES",
    "SEL_MODE_ALIASES",
    # initialization
    "initial_query_samples",
    "initialize_buffer",
    # updates
    "update_positive",
    "update_negatives",
    "update_query_samples",
    "apply_outcome_supervision",
    # helpers
    "deduplicate",
    "candidate_key",
]


# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

SEL_MODE_GROUND_TRUTH = "ground_truth"
SEL_MODE_AI_FEEDBACK = "ai_feedback"
SEL_MODE_COMBINED = "combined"
SEL_MODE_RANDOM = "random"

#: The three positive-sample sources evaluated in Section 4.1 (plus a random
#: control used by unit tests / untrained-adapter sanity checks).
SEL_MODES: Tuple[str, ...] = (
    SEL_MODE_GROUND_TRUTH,
    SEL_MODE_AI_FEEDBACK,
    SEL_MODE_COMBINED,
    SEL_MODE_RANDOM,
)

SEL_MODE_ALIASES: Dict[str, str] = {
    "": SEL_MODE_GROUND_TRUTH,
    "default": SEL_MODE_GROUND_TRUTH,
    "gt": SEL_MODE_GROUND_TRUTH,
    "groundtruth": SEL_MODE_GROUND_TRUTH,
    "ground-truth": SEL_MODE_GROUND_TRUTH,
    "ground_truth": SEL_MODE_GROUND_TRUTH,
    "truth": SEL_MODE_GROUND_TRUTH,
    "supervised": SEL_MODE_GROUND_TRUTH,
    "ai": SEL_MODE_AI_FEEDBACK,
    "ai_feedback": SEL_MODE_AI_FEEDBACK,
    "ai-feedback": SEL_MODE_AI_FEEDBACK,
    "feedback": SEL_MODE_AI_FEEDBACK,
    "gpt4": SEL_MODE_AI_FEEDBACK,
    "gpt_4": SEL_MODE_AI_FEEDBACK,
    "hf": SEL_MODE_AI_FEEDBACK,
    "human": SEL_MODE_AI_FEEDBACK,
    "human_feedback": SEL_MODE_AI_FEEDBACK,
    "combined": SEL_MODE_COMBINED,
    "both": SEL_MODE_COMBINED,
    "mix": SEL_MODE_COMBINED,
    "gt+ai": SEL_MODE_COMBINED,
    "random": SEL_MODE_RANDOM,
}


def _normalize_mode(mode: Optional[str]) -> str:
    """Normalize a SEL mode string (lowercase, spaces -> underscores)."""
    if mode is None:
        return SEL_MODE_GROUND_TRUTH
    key = str(mode).strip().lower().replace(" ", "_")
    if key in SEL_MODE_ALIASES:
        return SEL_MODE_ALIASES[key]
    raise ValueError(
        f"Unknown SEL mode {mode!r}; expected one of {SEL_MODES} "
        f"(aliases: {sorted(SEL_MODE_ALIASES)})"
    )


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------


@dataclass
class BufferConfig:
    """Configuration of the positive/negative sample buffers.

    Parameters
    ----------
    dataset:
        Dataset name (drives answer extraction, cf. ``data/dataset_specs.py``).
    sel_mode:
        One of ``{"ground_truth", "ai_feedback", "combined", "random"}``
        (Section 4.1 "Settings").
    k_init:
        ``K`` in Section 3.4 Initialization: number of responses initially
        prompted from the black-box LLM per query for the positive/negative sets.
    m_candidates:
        ``M`` in Algorithm 1 / Eq. (1): number of candidates sampled from the
        current adapted inference per outer iteration.
    max_positives / max_negatives:
        Optional caps on the stored sets (memory control). ``None`` = unbounded.
    deduplicate:
        Drop text-duplicate candidates when refreshing the sets.
    outcome_supervision:
        Enable the Section 4.1 outcome-supervision rule in all settings.
    ai_feedback_n_ranked:
        Number of ranked answers requested from the rater on TruthfulQA.
    answer_type / choices:
        Forwarded to ``data.answer_extraction`` for final-answer parsing.
    seed:
        Seed used by the random selector and any tie-breaking.
    """

    dataset: Optional[str] = None
    sel_mode: str = SEL_MODE_GROUND_TRUTH
    k_init: int = 5
    m_candidates: int = 5
    max_positives: Optional[int] = None
    max_negatives: Optional[int] = None
    deduplicate: bool = True
    outcome_supervision: bool = True
    ai_feedback_n_ranked: int = 5
    answer_type: Optional[str] = None
    choices: Optional[Sequence[str]] = None
    prompt_key: Optional[str] = None
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sel_mode = _normalize_mode(self.sel_mode)
        if self.k_init < 1:
            raise ValueError("k_init must be >= 1")
        if self.m_candidates < 1:
            raise ValueError("m_candidates must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sel_mode": self.sel_mode,
            "k_init": self.k_init,
            "m_candidates": self.m_candidates,
            "max_positives": self.max_positives,
            "max_negatives": self.max_negatives,
            "deduplicate": self.deduplicate,
            "outcome_supervision": self.outcome_supervision,
            "ai_feedback_n_ranked": self.ai_feedback_n_ranked,
            "answer_type": self.answer_type,
            "choices": list(self.choices) if self.choices is not None else None,
            "prompt_key": self.prompt_key,
            "seed": self.seed,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BufferConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in dict(data).items() if k in known}
        return cls(**kwargs)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def candidate_key(text: Any) -> str:
    """Stable identity of a text candidate (whitespace-insensitive hash)."""
    if text is None:
        return ""
    norm = re.sub(r"\s+", " ", str(text)).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def deduplicate(candidates: Sequence[Any]) -> List[Any]:
    """Drop duplicate candidates preserving order (first occurrence wins)."""
    seen = set()
    out: List[Any] = []
    for cand in candidates:
        key = candidate_key(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _extracted(candidate: Any, answer_type: Optional[str], choices=None):
    """Extract the final answer from a text candidate (may be ``None``)."""
    if answer_type is None:
        return None
    try:
        return extract_final_answer(candidate, answer_type, choices=choices)
    except Exception:  # pragma: no cover - defensive
        return None


def _answers_match(a: Any, b: Any, answer_type: Optional[str]) -> bool:
    """Compare two already-extracted answers with dataset-aware semantics."""
    if a is None or b is None:
        return False
    # ``normalize_answer`` handles numeric tolerance / int coercion / casing.
    try:
        return normalize_answer(a, answer_type) == normalize_answer(b, answer_type)
    except Exception:  # pragma: no cover - defensive
        return str(a).strip().lower() == str(b).strip().lower()


# ----------------------------------------------------------------------------
# SEL (the positive-sample selection function)
# ----------------------------------------------------------------------------


class _SelectResult(tuple):
    """Internal container: (index, method, meta)."""

    __slots__ = ()

    @property
    def index(self) -> int:
        return self[0]

    @property
    def method(self) -> str:
        return self[1]

    @property
    def meta(self) -> Dict[str, Any]:
        return self[2]


def ground_truth_select(
    candidates: Sequence[Any],
    *,
    gold: Any,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    tie_break: str = "first",
    pool: Optional[Sequence[Any]] = None,
) -> int:
    """``SEL`` under the Ground-Truth setting (Section 4.1).

    Returns the index (into ``candidates``) of the candidate whose extracted
    final answer matches the dataset solution ``gold``.  ``pool`` is an optional
    additional list (e.g. the *previous* positive sample, per Eq. 5) that is
    considered first for backward compatibility of the positive set.

    ``-1`` is returned when no candidate matches the ground truth.
    """
    if gold is None:
        return -1
    gold_answer = gold
    if answer_type is not None:
        extracted_gold = _extracted(gold, answer_type, choices)
        if extracted_gold is not None:
            gold_answer = extracted_gold
        elif isinstance(gold, str):
            gold_answer = gold

    def _matches(cand: Any) -> bool:
        pred = _extracted(cand, answer_type, choices)
        if pred is not None and _answers_match(pred, gold_answer, answer_type):
            return True
        # Fallback for callers that pass already-extracted answers.
        return _answers_match(cand, gold_answer, answer_type)

    hits = [i for i, c in enumerate(candidates) if _matches(c)]
    if not hits:
        if pool:
            pool_hits = [i for i, c in enumerate(pool) if _matches(c)]
            if pool_hits:
                # Keep the previous positive sample: Eq. (5) selects among
                # {previous positive} U {new candidates}; we only signal it via
                # a negative index convention of "keep".
                return -2
        return -1
    if tie_break == "last":
        return hits[-1]
    return hits[0]


def ai_feedback_select(
    question: str,
    candidates: Sequence[Any],
    *,
    dataset: Optional[str] = None,
    rater: Any = None,
    ai_feedback: Any = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    n_ranked: int = 5,
    return_result: bool = False,
):
    """``SEL`` under the AI-Feedback setting: GPT-4 simulates human preference.

    Delegates prompt building + parsing to ``feedback/ai_feedback.py``
    (Appendix G criteria / Appendix J prompt formats).  Returns the 0-based
    index into ``candidates`` (or ``-1`` when the rater could not decide).
    """
    if not candidates:
        return -1
    try:
        out = select_by_ai_feedback(
            question,
            list(candidates),
            dataset=dataset,
            rater=rater,
            ai_feedback=ai_feedback,
            answer_type=answer_type,
            return_result=True,
        )
    except Exception as exc:  # pragma: no cover - offline / no credentials
        logger.warning("AI feedback selection failed (%s); falling back to -1", exc)
        return -1
    idx = getattr(out, "index", None)
    if idx is None:
        idx = out if isinstance(out, int) else -1
    if idx is None or idx < 0:
        return -1
    return int(idx)


def combined_select(
    question: str,
    candidates: Sequence[Any],
    *,
    gold: Any = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
    rater: Any = None,
    ai_feedback: Any = None,
    n_ranked: int = 5,
) -> int:
    """``SEL`` under the Combined setting: ground truth augmented with AI feedback.

    Ground truth takes precedence when a candidate matches it; otherwise we fall
    back to the GPT-4 preference (Section 4.1, setting (3)).
    """
    gt_idx = ground_truth_select(
        candidates,
        gold=gold,
        answer_type=answer_type,
        choices=choices,
    )
    if gt_idx >= 0:
        return gt_idx
    return ai_feedback_select(
        question,
        candidates,
        dataset=dataset,
        rater=rater,
        ai_feedback=ai_feedback,
        answer_type=answer_type,
        choices=choices,
        n_ranked=n_ranked,
    )


def random_select(candidates: Sequence[Any], *, seed: int = 0) -> int:
    """Random SEL baseline (used for the untrained-adapter sanity check)."""
    if not candidates:
        return -1
    import random

    return random.Random(seed).randrange(len(candidates))


def make_selector(
    mode: str = SEL_MODE_GROUND_TRUTH,
    *,
    config: Optional[BufferConfig] = None,
    rater: Any = None,
    ai_feedback: Any = None,
) -> Callable[..., int]:
    """Build a ``SEL(question, candidates, ...) -> int`` callable for ``mode``.

    The returned callable accepts ``(question, candidates, **kwargs)`` and merges
    any explicit kwargs with the values pinned by ``config``.  ``-1`` means
    "no positive sample could be selected".
    """
    mode = _normalize_mode(mode or (config.sel_mode if config else None))
    cfg = config or BufferConfig(sel_mode=mode)

    if mode == SEL_MODE_GROUND_TRUTH:

        def _sel(question, candidates, **kw):
            return ground_truth_select(
                candidates,
                gold=kw.get("gold"),
                answer_type=kw.get("answer_type", cfg.answer_type),
                choices=kw.get("choices", cfg.choices),
            )

        return _sel

    if mode == SEL_MODE_AI_FEEDBACK:

        def _sel(question, candidates, **kw):
            return ai_feedback_select(
                question,
                candidates,
                dataset=kw.get("dataset", cfg.dataset),
                rater=kw.get("rater", rater),
                ai_feedback=kw.get("ai_feedback", ai_feedback),
                answer_type=kw.get("answer_type", cfg.answer_type),
                choices=kw.get("choices", cfg.choices),
                n_ranked=kw.get("n_ranked", cfg.ai_feedback_n_ranked),
            )

        return _sel

    if mode == SEL_MODE_COMBINED:

        def _sel(question, candidates, **kw):
            return combined_select(
                question,
                candidates,
                gold=kw.get("gold"),
                answer_type=kw.get("answer_type", cfg.answer_type),
                choices=kw.get("choices", cfg.choices),
                dataset=kw.get("dataset", cfg.dataset),
                rater=kw.get("rater", rater),
                ai_feedback=kw.get("ai_feedback", ai_feedback),
                n_ranked=kw.get("n_ranked", cfg.ai_feedback_n_ranked),
            )

        return _sel

    def _sel(question, candidates, **kw):
        return random_select(candidates, seed=kw.get("seed", cfg.seed))

    return _sel


# ----------------------------------------------------------------------------
# per-query record
# ----------------------------------------------------------------------------


@dataclass
class QuerySamples:
    """Positive/negative sample sets of one query ``x_i`` (Section 3.4)."""

    question: str
    uid: Optional[str] = None
    positive: List[str] = field(default_factory=list)
    negatives: List[str] = field(default_factory=list)
    gold: Any = None
    answer_type: Optional[str] = None
    choices: Optional[Sequence[str]] = None
    dataset: Optional[str] = None
    #: how the *current* positive sample was obtained
    sel_method: Optional[str] = None
    sel_index: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def y_plus(self) -> Optional[str]:
        """The current positive sample ``y_{i+}^{(t)}`` (last selected)."""
        return self.positive[-1] if self.positive else None

    @property
    def y_minus(self) -> List[str]:
        """The current negative set ``y_{i-}^{(t)}``."""
        return list(self.negatives)

    @property
    def positive_answer(self) -> Any:
        """Extracted final answer of the current positive sample."""
        return _extracted(self.y_plus, self.answer_type, self.choices)

    @property
    def is_empty(self) -> bool:
        return not self.positive and not self.negatives

    def add_positive(self, text: str, *, prepend_from_front: bool = False) -> None:
        """Append a positive sample, moving it to the front if requested."""
        if text is None:
            return
        if text in self.positive:
            if prepend_from_front:
                self.positive.remove(text)
            else:
                return
        if prepend_from_front:
            self.positive.insert(0, text)
        else:
            self.positive.append(text)

    def add_negative(self, text: str) -> None:
        if text is None:
            return
        if text not in self.negatives:
            self.negatives.append(text)

    def positives_cap(self, cap: Optional[int]) -> None:
        """Keep only the most recent ``cap`` positive samples."""
        if cap is not None and cap > 0 and len(self.positive) > cap:
            self.positive = self.positive[-cap:]

    def negatives_cap(self, cap: Optional[int]) -> None:
        """Keep only the most recent ``cap`` negative samples."""
        if cap is not None and cap > 0 and len(self.negatives) > cap:
            self.negatives = self.negatives[-cap:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "question": self.question,
            "positive": list(self.positive),
            "negatives": list(self.negatives),
            "gold": self.gold,
            "answer_type": self.answer_type,
            "dataset": self.dataset,
            "sel_method": self.sel_method,
            "sel_index": self.sel_index,
            "meta": dict(self.meta),
        }


# ----------------------------------------------------------------------------
# Eq. (5) / Eq. (6): the update rules
# ----------------------------------------------------------------------------


def update_positive(
    samples: QuerySamples,
    candidates: Sequence[str],
    *,
    selector: Callable[..., int],
    config: Optional[BufferConfig] = None,
    question: Optional[str] = None,
) -> Optional[str]:
    """Eq. (5): ``y_{i+}^{(t)} = SEL(y_{i+}^{(t-1)}, {yhat_{i,m}}_{m=1}^{M})``.

    ``SEL`` is asked to choose among the union of the previous positive sample
    and the freshly sampled candidates (which is exactly the paper's argument
    list).  When the selector picks the previous positive, the set is unchanged
    beyond we still record it as the newest entry.
    """
    cfg = config or BufferConfig()
    if not candidates and not samples.positive:
        return None

    pool: List[str] = []
    if samples.y_plus is not None:
        pool.append(samples.y_plus)
    pool.extend(candidates)
    if cfg.deduplicate:
        union = deduplicate(pool)
    else:
        union = list(pool)
    if not union:
        return samples.y_plus

    prev = samples.y_plus
    kwargs = dict(
        gold=samples.gold,
        answer_type=samples.answer_type or cfg.answer_type,
        choices=samples.choices or cfg.choices,
        dataset=samples.dataset or cfg.dataset,
    )
    idx = -1
    try:
        idx = int(selector(question if question is not None else samples.question, union, **kwargs))
    except TypeError:
        # selector that only accepts candidates
        try:
            idx = int(selector(union, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SEL failed for %s: %s", samples.uid, exc)
            idx = -1
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("SEL failed for %s: %s", samples.uid, exc)
        idx = -1

    samples.sel_index = idx
    if idx is None or idx < 0:
        # No new positive selected: keep the previous positive sample
        # (Section 3.4 keeps y_+^{(t-1)} when no better candidate is found).
        samples.sel_method = samples.sel_method or "keep"
        if prev is not None:
            samples.positive = [prev] + [p for p in samples.positive if p != prev]
            samples.positives_cap(cfg.max_positives)
        return samples.y_plus

    chosen = union[idx]
    # Record the new positive (front of the list = current positive).
    rest = [p for p in samples.positive if p != chosen]
    samples.positive = [chosen] + rest
    samples.sel_method = getattr(selector, "__name__", None) or f"sel[{cfg.sel_mode}]"
    if prev is not None and chosen == prev:
        samples.sel_method = "keep"
    samples.positives_cap(cfg.max_positives)
    return samples.y_plus


def update_negatives(
    samples: QuerySamples,
    candidates: Sequence[str],
    *,
    config: Optional[BufferConfig] = None,
    add_previous: bool = False,
    keep_initial: bool = True,
) -> List[str]:
    """Eq. (6): ``y_{i-}^{(t)} = {yhat_{i,m} | yhat_{i,m} != y_{i+}^{(t)}}``.

    The remaining candidates of the *newly sampled* set become the negative
    set; the selected positive sample is explicitly excluded.  ``keep_initial``
    retains the initial negatives (Section 3.4 Initialization) alongside the
    refreshed ones, which stabilizes the early iterations.
    """
    cfg = config or BufferConfig()
    positive = samples.y_plus
    pos_answer = samples.positive_answer

    out: List[str] = list(samples.negatives) if keep_initial else []
    if add_previous and samples.y_plus is not None:
        out.append(samples.y_plus)

    for cand in candidates:
        if cand is None:
            continue
        if positive is not None and candidate_key(cand) == candidate_key(positive):
            continue
        if cfg.deduplicate:
            out = deduplicate(out)
        if candidate_key(cand) in {candidate_key(x) for x in out}:
            continue
        out.append(cand)

    # Never let the selected positive leak into the negative set.
    if positive is not None:
        out = [c for c in out if candidate_key(c) != candidate_key(positive)]
    if pos_answer is not None and cfg.extra.get("exclude_answer_matches", False):
        out = [
            c
            for c in out
            if not _answers_match(
                _extracted(c, samples.answer_type, samples.choices),
                pos_answer,
                samples.answer_type,
            )
        ]
    if cfg.deduplicate:
        out = deduplicate(out)
    samples.negatives = out
    samples.negatives_cap(cfg.max_negatives)
    return samples.negatives


def update_query_samples(
    samples: QuerySamples,
    candidates: Sequence[str],
    *,
    selector: Callable[..., int],
    config: Optional[BufferConfig] = None,
    question: Optional[str] = None,
    outcome_candidates: Optional[Sequence[str]] = None,
) -> QuerySamples:
    """One full outer-iteration refresh of a query's sets (Algorithm 1, t-loop).

    Performs Eq. (5) then Eq. (6), and optionally applies outcome supervision
    (Section 4.1) to an additional batch of inferences.
    """
    cfg = config or BufferConfig()
    update_positive(samples, candidates, selector=selector, config=cfg, question=question)
    update_negatives(samples, candidates, config=cfg)
    if cfg.outcome_supervision and outcome_candidates:
        apply_outcome_supervision(samples, outcome_candidates, config=cfg)
    return samples


def apply_outcome_supervision(
    samples: QuerySamples,
    inferences: Sequence[str],
    *,
    config: Optional[BufferConfig] = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
) -> QuerySamples:
    """Section 4.1 outcome supervision: match final answers to the positives.

    "Those inferences that align with the training set answers are treated as
    additional positive samples, while all others are considered negative."

    The reference answers are the extracted final answers of the *existing*
    positive set (the training-set answers).
    """
    cfg = config or BufferConfig()
    atype = answer_type or samples.answer_type or cfg.answer_type
    ch = choices if choices is not None else (samples.choices or cfg.choices)

    pos_answers = []
    for p in samples.positive:
        a = _extracted(p, atype, ch)
        if a is not None:
            pos_answers.append(a)
    if not pos_answers and samples.gold is not None:
        # Fall back to the ground truth when the positive set is not yet filled.
        g = _extracted(samples.gold, atype, ch)
        pos_answers.append(g if g is not None else samples.gold)

    for cand in inferences:
        if cand is None:
            continue
        pred = _extracted(cand, atype, ch)
        matched = any(_answers_match(pred, a, atype) for a in pos_answers)
        if matched:
            samples.add_positive(cand)
        else:
            samples.add_negative(cand)
    samples.positives_cap(cfg.max_positives)
    samples.negatives_cap(cfg.max_negatives)
    if cfg.deduplicate:
        samples.positive = deduplicate(samples.positive)
        samples.negatives = [
            c
            for c in deduplicate(samples.negatives)
            if c not in set(samples.positive)
        ]
    return samples


# ----------------------------------------------------------------------------
# initialization (Section 3.4 "Initialization")
# ----------------------------------------------------------------------------


def initial_query_samples(
    question: str,
    candidates: Sequence[str],
    *,
    selector: Callable[..., int],
    gold: Any = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
    uid: Optional[str] = None,
    config: Optional[BufferConfig] = None,
    fallback_index: Optional[int] = 0,
) -> QuerySamples:
    """Build the ``t = 0`` sample sets from ``K`` prompted responses.

    ``{y_{i,j}}_{j=1}^{K}`` are the K responses prompted from the black-box LLM;
    the selected one becomes ``y_{i+}^{(0)}`` and the other ``K-1`` are the
    initial negatives ``y_{i-}^{(0)}`` (Section 3.4 Initialization).

    When ``SEL`` fails (e.g. no candidate matches the ground truth), the
    ``fallback_index`` candidate is used as positive (``None`` disables the
    fallback and leaves the positive set empty, as in the AI-Feedback setting
    where the rater must decide).
    """
    cfg = config or BufferConfig()
    cands = deduplicate(candidates) if cfg.deduplicate else list(candidates)
    samples = QuerySamples(
        question=question,
        uid=uid,
        gold=gold,
        answer_type=answer_type or cfg.answer_type,
        choices=choices if choices is not None else cfg.choices,
        dataset=dataset or cfg.dataset,
    )
    kwargs = dict(
        gold=gold,
        answer_type=samples.answer_type,
        choices=samples.choices,
        dataset=samples.dataset,
    )
    idx = -1
    try:
        idx = int(selector(question, cands, **kwargs))
    except TypeError:
        try:
            idx = int(selector(cands, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("initial SEL failed: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("initial SEL failed: %s", exc)

    if (idx is None or idx < 0) and fallback_index is not None and cands:
        if 0 <= fallback_index < len(cands):
            idx = fallback_index
            samples.meta["fallback"] = True

    if idx is not None and 0 <= idx < len(cands):
        chosen = cands[idx]
        samples.positive = [chosen]
        samples.negatives = [c for j, c in enumerate(cands) if j != idx]
        samples.sel_index = idx
        samples.sel_method = cfg.sel_mode
    else:
        samples.negatives = list(cands)
        samples.sel_index = -1
        samples.sel_method = cfg.sel_mode
    samples.positives_cap(cfg.max_positives)
    samples.negatives_cap(cfg.max_negatives)
    return samples


def initialize_buffer(
    questions: Sequence[str],
    candidate_lists: Sequence[Sequence[str]],
    *,
    selector: Callable[..., int],
    golds: Optional[Sequence[Any]] = None,
    answer_types: Optional[Sequence[Optional[str]]] = None,
    choices_list: Optional[Sequence[Optional[Sequence[str]]]] = None,
    dataset: Optional[str] = None,
    uids: Optional[Sequence[str]] = None,
    config: Optional[BufferConfig] = None,
    fallback_index: Optional[int] = 0,
) -> "SampleBuffer":
    """Initialize a whole :class:`SampleBuffer` from prompted candidate lists."""
    buf = SampleBuffer(config=config, dataset=dataset)
    n = len(questions)
    for i, q in enumerate(questions):
        golds_i = golds[i] if golds is not None and i < len(golds) else None
        at = answer_types[i] if answer_types is not None and i < len(answer_types) else None
        ch = choices_list[i] if choices_list is not None and i < len(choices_list) else None
        uid = uids[i] if uids is not None and i < len(uids) else None
        buf.add(
            initial_query_samples(
                q,
                candidate_lists[i],
                selector=selector,
                gold=golds_i,
                answer_type=at,
                choices=ch,
                dataset=dataset,
                uid=uid,
                config=config,
                fallback_index=fallback_index,
            )
        )
    return buf


# ----------------------------------------------------------------------------
# buffer container
# ----------------------------------------------------------------------------


class SampleBuffer:
    """Container of per-query positive/negative sets (dynamic in the t-loop).

    Keys are stable per-query identifiers (``uid`` when given, otherwise the
    question text).  The class is intentionally dataset-agnostic: all answer
    semantics come from ``data/answer_extraction.py``.
    """

    def __init__(
        self,
        config: Optional[BufferConfig] = None,
        *,
        dataset: Optional[str] = None,
        selector: Optional[Callable[..., int]] = None,
        rater: Any = None,
        ai_feedback: Any = None,
    ) -> None:
        self.config = config or BufferConfig(dataset=dataset)
        if dataset is not None:
            self.config.dataset = dataset
            if self.config.prompt_key is None:
                try:
                    from ..data.dataset_specs import get_spec

                    self.config.prompt_key = get_spec(dataset).prompt_key
                    if self.config.answer_type is None:
                        self.config.answer_type = get_spec(dataset).answer_type
                except Exception:  # pragma: no cover - unknown dataset name
                    pass
        self.selector = selector or make_selector(
            self.config.sel_mode, config=self.config, rater=rater, ai_feedback=ai_feedback
        )
        self.store: Dict[str, QuerySamples] = {}
        self.order: List[str] = []
        self.t = 0
        self.history: List[Dict[str, Any]] = []

    # -- dict-like access --------------------------------------------------
    @staticmethod
    def key_of(question: str, uid: Optional[str] = None) -> str:
        return uid if uid else hashlib.sha1(question.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.order)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, QuerySamples):
            return key.uid in self.store or self.key_of(key.question) in self.store
        if key in self.store:
            return True
        return self.key_of(str(key)) in self.store

    def __getitem__(self, key: Any) -> QuerySamples:
        if isinstance(key, int):
            return self.store[self.order[key]]
        if key in self.store:
            return self.store[key]
        k = self.key_of(str(key))
        if k in self.store:
            return self.store[k]
        raise KeyError(key)

    def __iter__(self):
        for k in self.order:
            yield self.store[k]

    def get(self, key: Any, default: Any = None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        return list(self.order)

    def values(self) -> List[QuerySamples]:
        return [self.store[k] for k in self.order]

    def items(self):
        return [(k, self.store[k]) for k in self.order]

    # -- mutation ----------------------------------------------------------
    def add(self, samples: QuerySamples) -> str:
        key = self.key_of(samples.question, samples.uid)
        samples.uid = samples.uid or key
        if key not in self.store:
            self.order.append(key)
        self.store[key] = samples
        return key

    def add_initial(
        self,
        question: str,
        candidates: Sequence[str],
        *,
        gold: Any = None,
        answer_type: Optional[str] = None,
        choices: Optional[Sequence[str]] = None,
        uid: Optional[str] = None,
        fallback_index: Optional[int] = 0,
    ) -> QuerySamples:
        samples = initial_query_samples(
            question,
            candidates,
            selector=self.selector,
            gold=gold,
            answer_type=answer_type or self.config.answer_type,
            choices=choices or self.config.choices,
            dataset=self.config.dataset,
            uid=uid,
            config=self.config,
            fallback_index=fallback_index,
        )
        self.add(samples)
        return samples

    def update(
        self,
        key: Any,
        candidates: Sequence[str],
        *,
        outcome_candidates: Optional[Sequence[str]] = None,
    ) -> QuerySamples:
        """Apply Eq. (5) + Eq. (6) (+ outcome supervision) for one query."""
        samples = self[key]
        update_query_samples(
            samples,
            candidates,
            selector=self.selector,
            config=self.config,
            question=samples.question,
            outcome_candidates=outcome_candidates,
        )
        return samples

    def refresh(
        self,
        candidates_by_key: Dict[Any, Sequence[str]],
        *,
        outcome_by_key: Optional[Dict[Any, Sequence[str]]] = None,
    ) -> Dict[str, QuerySamples]:
        """Refresh many queries at once (one outer iteration of Algorithm 1)."""
        updated: Dict[str, QuerySamples] = {}
        for key, cands in candidates_by_key.items():
            if key not in self:
                continue
            outcome = None
            if outcome_by_key is not None:
                outcome = outcome_by_key.get(key)
            s = self.update(key, cands, outcome_candidates=outcome)
            updated[s.uid or self.key_of(s.question)] = s
        return updated

    def advance(self) -> int:
        """Close one outer iteration ``t`` (Algorithm 1) and record stats."""
        self.history.append({"t": self.t, **self.stats()})
        self.t += 1
        return self.t

    # -- supervision helpers ----------------------------------------------
    def apply_outcome_supervision(
        self, key: Any, inferences: Sequence[str]
    ) -> QuerySamples:
        samples = self[key]
        return apply_outcome_supervision(samples, inferences, config=self.config)

    # -- statistics --------------------------------------------------------
    def n_queries(self) -> int:
        return len(self.order)

    def n_with_positive(self) -> int:
        return sum(1 for s in self if s.positive)

    def n_with_negatives(self) -> int:
        return sum(1 for s in self if s.negatives)

    def total_positives(self) -> int:
        return sum(len(s.positive) for s in self)

    def total_negatives(self) -> int:
        return sum(len(s.negatives) for s in self)

    def mean_negatives(self) -> float:
        if not self.order:
            return 0.0
        return self.total_negatives() / len(self.order)

    def coverage(self) -> float:
        """Fraction of queries that have both a positive and >=1 negative."""
        if not self.order:
            return 0.0
        ok = sum(1 for s in self if s.positive and s.negatives)
        return ok / len(self.order)

    def stats(self) -> Dict[str, Any]:
        return {
            "n_queries": self.n_queries(),
            "n_with_positive": self.n_with_positive(),
            "n_with_negatives": self.n_with_negatives(),
            "total_positives": self.total_positives(),
            "total_negatives": self.total_negatives(),
            "mean_negatives": self.mean_negatives(),
            "coverage": self.coverage(),
            "sel_mode": self.config.sel_mode,
        }

    # -- training-set extraction ------------------------------------------
    def contrastive_sets(self) -> Tuple[List[str], List[str], List[List[str]]]:
        """Return ``(questions, positives, negatives)`` for the NCE loss.

        Only queries carrying both a positive and at least one negative are
        emitted, mirroring Eq. (2)'s requirement of one positive plus its
        negatives per contrastive set.
        """
        qs: List[str] = []
        pos: List[str] = []
        negs: List[List[str]] = []
        for s in self:
            if not s.positive or not s.negatives:
                continue
            qs.append(s.question)
            pos.append(s.y_plus)
            negs.append([n for n in s.negatives if n != s.y_plus])
        return qs, pos, negs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "config": self.config.to_dict(),
            "samples": [s.to_dict() for s in self],
            "history": list(self.history),
        }

    @classmethod
    def from_questions(
        cls,
        questions: Sequence[str],
        golds: Optional[Sequence[Any]] = None,
        *,
        dataset: Optional[str] = None,
        config: Optional[BufferConfig] = None,
        selector: Optional[Callable[..., int]] = None,
    ) -> "SampleBuffer":
        """Empty buffer pre-populated with the questions (and golds)."""
        buf = cls(config=config, dataset=dataset, selector=selector)
        for i, q in enumerate(questions):
            gold = golds[i] if golds is not None and i < len(golds) else None
            buf.add(
                QuerySamples(
                    question=q,
                    gold=gold,
                    answer_type=buf.config.answer_type,
                    choices=buf.config.choices,
                    dataset=buf.config.dataset,
                )
            )
        return buf


# ----------------------------------------------------------------------------
# self test (offline, no network / no torch)
# ----------------------------------------------------------------------------


def _self_test() -> Dict[str, Any]:
    """Dependency-free sanity checks for Eq. (5), Eq. (6) and supervision."""
    from ..data.answer_extraction import extract_final_answer

    cfg = BufferConfig(dataset="strategyqa", sel_mode="ground_truth", k_init=4)
    assert cfg.answer_type == "yesno", cfg.answer_type

    rater_calls = {"n": 0}

    def fake_rater(prompt, **kw):
        rater_calls["n"] += 1
        return "Best Answer and Explanation:\nAnswer 2"

    # -- initialization (K=4 prompted responses) ---------------------------
    cands = [
        "Yes, because Paris is in France.\n#### Yes.",
        "No, Paris is not in France.\n#### No.",
        "Yes.\n#### Yes.",
        "No.\n#### No.",
    ]
    sel = make_selector("ground_truth", config=cfg)
    s = initial_query_samples(
        "Is Paris the capital of France?",
        cands,
        selector=sel,
        gold="Yes",
        answer_type="yesno",
        config=cfg,
    )
    assert s.y_plus == cands[0], s.y_plus
    assert len(s.negatives) == 3 and cands[0] not in s.negatives, s.negatives
    assert s.positive_answer == "Yes"

    # -- Eq. (5): AI feedback picks candidate #2 of the union --------------
    cfg_ai = BufferConfig(dataset="strategyqa", sel_mode="ai_feedback", m_candidates=3)
    new_cands = ["Maybe.\n#### No.", "Paris is indeed the capital.\n#### Yes.", "Unsure."]
    sel_ai = make_selector("ai_feedback", config=cfg_ai, rater=fake_rater)
    updated = update_query_samples(
        s, new_cands, selector=sel_ai, config=cfg_ai, outcome_candidates=None
    )
    # union = [prev_positive] + new_cands -> index 2 == new_cands[1]
    assert updated.y_plus == new_cands[1], (updated.y_plus, new_cands)
    assert updated.negatives == new_cands[0::2], updated.negatives
    assert rater_calls["n"] >= 1

    # -- positive must never appear in the negative set -------------------
    assert all(candidate_key(c) != candidate_key(updated.y_plus) for c in updated.negatives)

    # -- Eq. (6) shape: negatives come from the new sample set -------------
    s2 = QuerySamples(
        question="q", positive=["P"], negatives=["old"], answer_type="free", gold="P"
    )
    update_negatives(s2, ["P", "N1", "N2"], config=BufferConfig())
    assert "P" not in s2.negatives and "N1" in s2.negatives and "N2" in s2.negatives

    # -- outcome supervision (Section 4.1) --------------------------------
    s3 = QuerySamples(
        question="q",
        positive=["correct\n#### 7"],
        answer_type="numeric",
        gold="7",
    )
    apply_outcome_supervision(s3, ["also 7?\n#### 7", "wrong\n#### 9"], config=BufferConfig())
    assert "also 7?\n#### 7" in s3.positive
    assert "wrong\n#### 9" in s3.negatives

    # -- buffer container --------------------------------------------------
    buf = SampleBuffer(config=BufferConfig(dataset="strategyqa"), selector=sel)
    for i, (q, gold, cs) in enumerate(
        [
            ("Is Paris in France?", "Yes", cands),
            ("Is the sky green?", "No", ["No.\n#### No.", "Yes.\n#### Yes.", "No."]),
        ]
    ):
        buf.add_initial(q, cs, gold=gold)
    assert buf.n_queries() == 2
    assert buf.n_with_positive() == 2
    assert buf.coverage() == 1.0
    qs_list, pos_list, neg_list = buf.contrastive_sets()
    assert len(qs_list) == 2 and len(pos_list) == 2 and len(neg_list) == 2
    assert all(p not in n for p, n in zip(pos_list, neg_list))
    buf.update(qs_list[0], ["A.\n#### Yes.", "B.\n#### No."])
    buf.advance()
    assert buf.t == 1 and buf.history

    # -- mode aliases ------------------------------------------------------
    for alias, canon in [("gt", "ground_truth"), ("ai", "ai_feedback"), ("combined", "combined")]:
        assert _normalize_mode(alias) == canon
    try:
        _normalize_mode("nonsense")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown SEL mode must raise")
    assert _normalize_mode(None) == "ground_truth"

    # -- combined setting falls back to AI when GT is absent ---------------
    cfg_c = BufferConfig(sel_mode="combined")
    sel_c = make_selector("combined", config=cfg_c, rater=fake_rater)
    idx = sel_c("q", ["zzz", "yyy"], gold="nomatch", answer_type="free")
    assert idx == 1, idx

    # -- no logprob / hidden-state access anywhere: text in, text out ------
    assert isinstance(s.y_plus, str)

    out = {
        "init_positive": s.y_plus,
        "init_negatives": len(s.negatives),
        "updated_positive": updated.y_plus,
        "updated_negatives": updated.negatives,
        "buffer_stats": buf.stats(),
    }
    print("[buffers._self_test] OK")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _self_test()
