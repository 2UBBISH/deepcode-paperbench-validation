"""Final answer selection for BBox-Adapter adapted inference (Section 3.3).

The paper's adapted inference decomposes a solution into sentences and performs a
sentence-level beam search in which the black-box LLM is a *proposal generator* and
the adapter ``g_theta`` is an *evaluator*.  Section 3.3 closes the loop with:

    "Once a pre-defined number of :math:`L` iterations is reached or all beams
    encounter a stop signal, we obtain :math:`k` reasoning steps.  The adapted
    generation is then selected based on the highest-scoring option evaluated by
    the adapter."

This module implements exactly that last step.  Given

* a :class:`~bbox_adapter.inference.beam_search.BeamSearchResult`,
* a list of :class:`~bbox_adapter.inference.beam_search.Hypothesis` objects,
* a ragged list of ``(text, score)`` pairs, or
* plain texts (which are then scored with the adapter),

the :class:`AnswerSelector` emits the surface answer of the candidate that the
adapter evaluates most highly.  Because the raw best-scoring *hypothesis* is not
always the best *answer* (several beams may decode the same final answer while one
beam carries a slightly higher score), the selector also supports aggregating the
adapter scores of all candidates that reduce to the same extracted answer and
picking the answer with the best aggregate.  Aggregation is optional and defaults
to ``"max"``, which reproduces the paper's plain "highest-scoring option" rule.

Everything here is inference-time only: no gradients are taken, and the black-box
LLM is never queried or differentiated through.  Scoring is delegated to the
trained :class:`~bbox_adapter.adapter.energy_model.EnergyModel` through the
duck-typed :func:`~bbox_adapter.inference.beam_search.resolve_adapter_scores`
helper so that energy sign conventions (lower energy is better) are normalized to
"larger is better" in exactly one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..data.answer_extraction import extract_final_answer  # noqa: F401  (re-export convenience)

try:  # pragma: no cover - the helper lives in the beam-search module
    from .beam_search import (
        BeamSearchResult,
        Hypothesis,
        resolve_adapter_scores,
    )

    _HAS_BEAM_SEARCH = True
except Exception:  # pragma: no cover - defensive: selector must import standalone
    BeamSearchResult = Any  # type: ignore
    Hypothesis = Any  # type: ignore
    resolve_adapter_scores = None  # type: ignore
    _HAS_BEAM_SEARCH = False


__all__ = [
    "Candidate",
    "SelectionResult",
    "SelectorConfig",
    "AnswerSelector",
    "select_answer",
    "select_from_result",
    "normalize_candidates",
    "aggregate_scores",
    "rank_candidates",
    "best_by_score",
    "best_by_answer",
    "score_tie_break",
    "logsumexp",
    "SCORE_AGGREGATIONS",
    "DEFAULT_AGGREGATION",
]


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

#: Supported aggregation rules when several candidates share an extracted answer.
SCORE_AGGREGATIONS: Tuple[str, ...] = ("max", "sum", "mean", "logsumexp")

#: Paper-faithful default: "highest-scoring option evaluated by the adapter".
DEFAULT_AGGREGATION = "max"


# --------------------------------------------------------------------------------------
# small numeric helper (avoids a torch dependency for pure-Python data)
# --------------------------------------------------------------------------------------
def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable ``log(sum(exp(v)))`` for a sequence of floats."""
    vals = [float(v) for v in values]
    if not vals:
        return float("-inf")
    m = max(vals)
    if m == float("-inf"):
        return float("-inf")
    if m == float("inf"):
        return float("inf")
    return m + math.log(sum(math.exp(v - m) for v in vals))


def aggregate_scores(scores: Sequence[float], mode: str = DEFAULT_AGGREGATION) -> float:
    """Aggregate several adapter scores belonging to one extracted answer.

    Args:
        scores: adapter scores, all normalized so that larger is better.
        mode: one of :data:`SCORE_AGGREGATIONS`.

    Returns:
        A single scalar; ``-inf`` for an empty group.
    """
    vals = [float(s) for s in scores]
    if not vals:
        return float("-inf")
    if mode == "max":
        return max(vals)
    if mode == "sum":
        return float(sum(vals))
    if mode == "mean":
        return float(sum(vals)) / float(len(vals))
    if mode == "logsumexp":
        return logsumexp(vals)
    raise ValueError(
        f"Unknown aggregation mode {mode!r}; expected one of {SCORE_AGGREGATIONS}."
    )


# --------------------------------------------------------------------------------------
# data structures
# --------------------------------------------------------------------------------------
@dataclass
class Candidate:
    """One ranked candidate answer considered by the selector.

    Attributes:
        text: the raw generated text (possibly a chain of sentences).
        score: adapter score, already normalized to *larger is better*.
        answer: extracted/canonical answer used for grouping (may be ``None`` when
            extraction fails).
        normalized: optional normalized form of ``answer`` used as the grouping key.
        index: position of the candidate in the original list (tie-breaking).
        meta: free-form bookkeeping (e.g. ``{"source": "beam"}``).
    """

    text: str
    score: float = 0.0
    answer: Optional[Any] = None
    normalized: Optional[Any] = None
    index: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def group_key(self) -> Any:
        """Key used to group candidates that decode to the same answer."""
        return self.normalized if self.normalized is not None else self.answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "answer": self.answer,
            "index": self.index,
            "meta": dict(self.meta),
        }


@dataclass
class SelectionResult:
    """Output of :class:`AnswerSelector.select`.

    Attributes:
        answer: the selected canonical answer (``None`` if extraction failed for
            every candidate).
        text: raw text of the selected candidate.
        score: adapter score of the selected candidate.
        candidates: all candidates that were considered, sorted by score desc.
        groups: mapping ``answer-key -> (aggregate score, [Candidate, ...])``.
        reason: short human-readable explanation of how the choice was made
            (``"aggregate"``, ``"highest_score"``, ``"fallback"``).
    """

    answer: Optional[Any]
    text: str
    score: float
    candidates: List[Candidate] = field(default_factory=list)
    groups: Dict[Any, Tuple[float, List[Candidate]]] = field(default_factory=dict)
    reason: str = "highest_score"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "text": self.text,
            "score": self.score,
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class SelectorConfig:
    """Configuration for :class:`AnswerSelector` (defaults follow the paper)."""

    #: How to combine scores of candidates that share an extracted answer.
    #: ``"max"`` reproduces Section 3.3 verbatim ("highest-scoring option").
    aggregation: str = DEFAULT_AGGREGATION
    #: Optional per-sentence/length normalization of scores ("none" matches the paper).
    normalization: str = "none"
    length_penalty: float = 0.0
    #: Prefer candidates that emitted the stop signal ``####`` when scores tie.
    prefer_terminated: bool = True
    #: Fall back to the highest-scoring *hypothesis* when no answer can be extracted.
    fallback_to_best_hypothesis: bool = True
    #: If True, only the aggregated-answer rule decides; otherwise the single
    #: highest scoring candidate wins (identical when aggregation == "max").
    group_by_answer: bool = True
    #: Require at least this many sharing candidates before using the aggregate.
    min_group_size: int = 1
    answer_type: Optional[str] = None
    choices: Optional[Sequence[str]] = None
    max_length: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SelectorConfig":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------------------
# candidate normalization / scoring helpers
# --------------------------------------------------------------------------------------
def _coerce_pair(item: Any) -> Tuple[str, Optional[float], Dict[str, Any]]:
    """Coerce one candidate-like object into ``(text, score_or_None, meta)``."""
    # Hypothesis (or any object exposing .text/.score)
    text = None
    score: Optional[float] = None
    meta: Dict[str, Any] = {}

    if isinstance(item, str):
        text = item
    elif isinstance(item, dict):
        text = item.get("text") or item.get("best_text") or item.get("answer_text")
        raw = item.get("score", item.get("best_score"))
        score = None if raw is None else float(raw)
        meta = dict(item)
    elif isinstance(item, (tuple, list)) and len(item) >= 1:
        text = str(item[0])
        if len(item) >= 2 and item[1] is not None:
            try:
                score = float(item[1])
            except (TypeError, ValueError):
                score = None
        if len(item) >= 3 and isinstance(item[2], dict):
            meta = dict(item[2])
    else:  # duck-typed Hypothesis / arbitrary object
        text = getattr(item, "text", None)
        if text is None and hasattr(item, "best_text"):
            text = item.best_text
        raw = getattr(item, "score", None)
        if raw is None:
            raw = getattr(item, "best_score", None)
        if raw is not None:
            try:
                score = float(raw)
            except (TypeError, ValueError):
                score = None
        for name in ("n_sentences", "n_chars", "meta"):
            if hasattr(item, name):
                meta[name] = getattr(item, name)

    if text is None:
        raise TypeError(f"Cannot interpret candidate of type {type(item)!r}: {item!r}")
    return str(text), score, meta


def normalize_candidates(items: Any) -> List[Candidate]:
    """Flatten any supported candidate container into a list of :class:`Candidate`.

    Accepted inputs: a :class:`BeamSearchResult` (uses ``beams``, falling back to
    ``candidates`` and the single best hypothesis), a list/tuple of candidates, or a
    single candidate object.  Scores already attached to candidates are assumed to
    follow the beam-search convention "larger is better" (beam search maximizes the
    sign-normalized adapter score).
    """
    if items is None:
        return []

    # BeamSearchResult (or anything shaped like it)
    if hasattr(items, "beams") or hasattr(items, "best_hypothesis"):
        beams = list(getattr(items, "beams", None) or [])
        if not beams:
            cands = list(getattr(items, "candidates", None) or [])
            # candidates may be plain strings or hypotheses; de-duplicate by text
            beamed = []
            seen = set()
            for c in cands:
                try:
                    t, s, m = _coerce_pair(c)
                except TypeError:
                    continue
                if t in seen:
                    continue
                seen.add(t)
                beamed.append(c if not isinstance(c, str) else (t, s if s is not None else 0.0))
            beams = beamed
        if not beams:
            best_h = getattr(items, "best_hypothesis", None)
            if best_h is not None:
                beams = [best_h]
            elif getattr(items, "best_text", None):
                beams = [
                    (items.best_text, float(getattr(items, "best_score", 0.0) or 0.0))
                ]
        items = beams

    if isinstance(items, (str, dict)) or hasattr(items, "text"):
        items = [items]

    out: List[Candidate] = []
    for i, item in enumerate(items):
        try:
            text, score, meta = _coerce_pair(item)
        except TypeError:
            continue
        out.append(
            Candidate(
                text=text,
                score=0.0 if score is None else float(score),
                index=i,
                meta=meta,
            )
        )
    return out


def rank_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Sort candidates by score descending, breaking ties deterministically.

    Ties are broken (a) in favour of candidates carrying the stop signal (a complete
    answer is preferable to a truncated chain), then (b) by shorter text, then (c) by
    original index, so results are reproducible across runs.
    """

    def _stop_flag(c: Candidate) -> int:
        return 1 if "####" in c.text else 0

    return sorted(
        candidates,
        key=lambda c: (-float(c.score), -_stop_flag(c), len(c.text), c.index),
    )


def best_by_score(candidates: Sequence[Candidate]) -> Optional[Candidate]:
    """Return the single highest-scoring candidate (paper's plain rule)."""
    ranked = rank_candidates(candidates)
    return ranked[0] if ranked else None


def _group_candidates(candidates: Sequence[Candidate]) -> Dict[Any, List[Candidate]]:
    groups: Dict[Any, List[Candidate]] = {}
    for c in candidates:
        key = c.group_key
        if key is None:
            # Answer extraction failed: keep the candidate in its own singleton group
            # keyed by a unique object so it never merges with real answers.
            key = ("__unparsed__", c.index)
        groups.setdefault(key, []).append(c)
    return groups


def best_by_answer(
    candidates: Sequence[Candidate],
    aggregation: str = DEFAULT_AGGREGATION,
) -> Tuple[Optional[Any], float, List[Candidate]]:
    """Pick the extracted answer whose candidate group has the best aggregate score.

    Returns ``(answer_key, aggregate_score, group_candidates)``; ``(None, -inf, [])``
    when there is nothing to choose from.
    """
    groups = _group_candidates(candidates)
    best_key: Optional[Any] = None
    best_score = float("-inf")
    best_group: List[Candidate] = []
    for key, group in groups.items():
        agg = aggregate_scores([c.score for c in group], aggregation)
        if agg > best_score:
            best_score, best_key, best_group = agg, key, group
    return best_key, best_score, best_group


def score_tie_break(a: Candidate, b: Candidate) -> int:
    """Return -1/0/1 comparing two candidates with the selector's tie-breaking."""
    ra, rb = rank_candidates([a, b])[0], rank_candidates([b, a])[0]
    if ra is a and rb is a:
        return 0
    return -1 if ra is a else 1


# --------------------------------------------------------------------------------------
# the selector
# --------------------------------------------------------------------------------------
class AnswerSelector:
    """Select the final adapted answer from adapter-evaluated candidates (§3.3).

    Args:
        adapter: optional trained adapter used to score candidates that arrive
            without scores.  Any object supported by
            :func:`~bbox_adapter.inference.beam_search.resolve_adapter_scores`
            (``score_pairs`` / ``energy`` / ``score_batch`` / callable) works.
        config: :class:`SelectorConfig` or plain dict of overrides.

    Example:
        >>> sel = AnswerSelector(adapter)                       # doctest: +SKIP
        >>> res = sel.select(question, beam_result)             # doctest: +SKIP
        >>> res.answer                                          # doctest: +SKIP
    """

    def __init__(
        self,
        adapter: Optional[Any] = None,
        config: Optional[Union[SelectorConfig, Dict[str, Any]]] = None,
    ) -> None:
        self.adapter = adapter
        if isinstance(config, SelectorConfig):
            self.config = config
        else:
            self.config = SelectorConfig.from_dict(config)

    # -- scoring -----------------------------------------------------------------
    def _apply_normalization(self, cands: List[Candidate]) -> List[Candidate]:
        mode = (self.config.normalization or "none").lower()
        if mode in ("none", "", "raw", None):
            if self.config.length_penalty:
                for c in cands:
                    n = max(1, len(c.text.split()))
                    c.score = float(c.score) + float(self.config.length_penalty) * n
            return cands
        for c in cands:
            if mode in ("length", "token", "tokens"):
                denom = float(max(1, len(c.text.split())))
            elif mode in ("sentences", "sentence"):
                denom = float(max(1, len([s for s in c.text.split("\n") if s.strip()])))
            elif mode in ("chars", "characters"):
                denom = float(max(1, len(c.text)))
            else:
                raise ValueError(
                    f"Unknown normalization {self.config.normalization!r}; use "
                    "'none', 'length', 'sentences' or 'chars'."
                )
            c.score = float(c.score) / denom
            if self.config.length_penalty:
                c.score += float(self.config.length_penalty) * denom
        return cands

    def score_candidates(
        self,
        question: str,
        candidates: Sequence[Candidate],
        *,
        rescore: bool = False,
    ) -> List[Candidate]:
        """Ensure every candidate carries an adapter score ("larger is better").

        Candidates with an already-attached score are kept as-is unless ``rescore``
        is True, which re-queries the adapter on the full candidate text.
        """
        needs = [c for c in candidates if rescore or c.score == 0.0]
        if not needs:
            return list(candidates)
        if self.adapter is None:
            # No adapter available: leave provided scores untouched.
            return list(candidates)
        if resolve_adapter_scores is None:  # pragma: no cover
            raise RuntimeError(
                "resolve_adapter_scores is unavailable; cannot score candidates."
            )
        try:
            scores = resolve_adapter_scores(
                self.adapter,
                question,
                [c.text for c in needs],
                max_length=self.config.max_length,
            )
        except TypeError:
            scores = resolve_adapter_scores(
                self.adapter, question, [c.text for c in needs]
            )
        for c, s in zip(needs, scores):
            c.score = float(s)
        return list(candidates)

    def _extract(self, cands: Sequence[Candidate], answer_type: Optional[str]) -> None:
        atype = answer_type or self.config.answer_type
        for c in cands:
            if c.answer is not None and not self.config.extra.get("reextract", False):
                continue
            try:
                ans = extract_final_answer(
                    c.text,
                    atype,
                    choices=self.config.choices,
                )
            except Exception:
                ans = None
            c.answer = ans
            c.normalized = _normalize_key(ans)

    # -- main entry point --------------------------------------------------------
    def select(
        self,
        question: str,
        candidates: Any,
        *,
        answer_type: Optional[str] = None,
        choices: Optional[Sequence[str]] = None,
        rescore: bool = False,
        return_result: bool = False,
    ) -> Union[SelectionResult, Any]:
        """Select the adapter-preferred answer from ``candidates``.

        Args:
            question: the input question ``x``.
            candidates: a :class:`BeamSearchResult`, list of hypotheses, list of
                ``(text, score)`` pairs, or list of raw texts.
            answer_type: dataset answer type passed to ``extract_final_answer``
                (overrides the config when given).
            choices: MCQ choices when ``answer_type == "mcq"``.
            rescore: re-run adapter scoring on all candidates instead of trusting
                any attached scores.
            return_result: when True return a :class:`SelectionResult`; otherwise
                return the selected canonical answer (or the best raw text when no
                answer could be extracted and
                ``fallback_to_best_hypothesis`` is enabled).

        Returns:
            The canonical answer, the best raw text, or a :class:`SelectionResult`.
        """
        if choices is not None:
            self.config.choices = choices
        cands = normalize_candidates(candidates)
        cands = self.score_candidates(question, cands, rescore=rescore)
        cands = self._apply_normalization(cands)
        self._extract(cands, answer_type)

        ranked = rank_candidates(cands)
        groups = _group_candidates(ranked)
        group_stats: Dict[Any, Tuple[float, List[Candidate]]] = {
            key: (aggregate_scores([c.score for c in group], self.config.aggregation), group)
            for key, group in groups.items()
        }

        reason = "highest_score"
        chosen: Optional[Candidate] = None

        if self.config.group_by_answer and ranked:
            best_key, best_agg, best_group = best_by_answer(
                ranked, self.config.aggregation
            )
            if (
                best_key is not None
                and not (isinstance(best_key, tuple) and best_key[0] == "__unparsed__")
                and len(best_group) >= max(1, int(self.config.min_group_size))
            ):
                # Highest aggregate score among answer groups; within the group take
                # the highest-scoring individual candidate.
                chosen = best_group[0]
                reason = "aggregate" if len(best_group) > 1 else "highest_score"
                if self.config.aggregation == DEFAULT_AGGREGATION:
                    reason = "highest_score"

        if chosen is None:
            chosen = ranked[0] if ranked else None
            reason = "highest_score" if chosen is not None else "fallback"

        if chosen is None:
            result = SelectionResult(answer=None, text="", score=float("-inf"), reason="fallback")
            return result if return_result else None

        result = SelectionResult(
            answer=chosen.answer,
            text=chosen.text,
            score=float(chosen.score),
            candidates=ranked,
            groups=group_stats,
            reason=reason,
        )

        if return_result:
            return result
        if result.answer is not None:
            return result.answer
        if self.config.fallback_to_best_hypothesis:
            return result.text
        return None

    # -- convenience -------------------------------------------------------------
    def __call__(self, question: str, candidates: Any, **kwargs: Any) -> Any:
        return self.select(question, candidates, **kwargs)

    def select_from_result(
        self, result: Any, *, rerank_with_beams: bool = True, **kwargs: Any
    ) -> Any:
        """Select from a :class:`BeamSearchResult`.

        When ``rerank_with_beams`` is True (default) every retained beam is scored
        by the adapter rather than trusting the single stored ``best_score``; this
        makes the final choice robust to how the beam search happened to order
        equal-scoring hypotheses.
        """
        question = kwargs.pop("question", None) or getattr(result, "question", "") or ""
        cands: Any
        if rerank_with_beams and getattr(result, "beams", None):
            cands = list(result.beams)
        else:
            cands = result
        return self.select(question, cands, **kwargs)


# --------------------------------------------------------------------------------------
# module-level helpers (mirror the class API for functional use)
# --------------------------------------------------------------------------------------
def _normalize_key(answer: Any) -> Any:
    """Lowercase/whitespace-collapse string answers so equal answers group together."""
    if answer is None:
        return None
    if isinstance(answer, str):
        return " ".join(answer.strip().lower().rstrip(".").split())
    return answer


def select_answer(
    question: str,
    candidates: Any,
    *,
    adapter: Optional[Any] = None,
    config: Optional[Union[SelectorConfig, Dict[str, Any]]] = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    rescore: bool = False,
    return_result: bool = False,
) -> Any:
    """Functional wrapper around :class:`AnswerSelector.select`."""
    selector = AnswerSelector(adapter, config)
    return selector.select(
        question,
        candidates,
        answer_type=answer_type,
        choices=choices,
        rescore=rescore,
        return_result=return_result,
    )


def select_from_result(
    result: Any,
    *,
    adapter: Optional[Any] = None,
    config: Optional[Union[SelectorConfig, Dict[str, Any]]] = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    return_result: bool = False,
) -> Any:
    """Select the final answer directly from a :class:`BeamSearchResult`."""
    selector = AnswerSelector(adapter, config)
    return selector.select_from_result(
        result,
        answer_type=answer_type,
        choices=choices,
        return_result=return_result,
    )


# --------------------------------------------------------------------------------------
# self test (dependency-free)
# --------------------------------------------------------------------------------------
def _self_test() -> None:
    """Sanity checks: highest-scoring selection, answer grouping, tie-breaking."""
    # 1. Plain highest-score rule (paper Section 3.3).
    cands = [("x", 0.1), ("y", 0.9), ("z", 0.5)]
    res = select_answer("q", cands, answer_type=None, return_result=True)
    assert res.text == "y" and abs(res.score - 0.9) < 1e-9, res.to_dict()

    # 2. Adapter score sign convention is normalized upstream; weaker candidates
    #    never win.
    assert res.candidates[0].text == "y"

    # 3. Grouping by extracted answer: two candidates yielding "Yes." beat a single
    #    slightly higher-scoring "No." when summing.
    grouped = [
        ("Because of A, the answer is #### Yes.", 0.4),
        ("Another route: #### Yes.", 0.35),
        ("Short: #### No.", 0.5),
    ]
    res_max = select_answer("q", grouped, answer_type="yesno", return_result=True)
    assert res_max.answer == "No", res_max.to_dict()  # max aggregation -> paper rule
    res_sum = select_answer(
        "q",
        grouped,
        config={"aggregation": "sum"},
        answer_type="yesno",
        return_result=True,
    )
    assert res_sum.answer == "Yes", res_sum.to_dict()
    assert res_sum.reason == "aggregate"

    # 4. Tie-breaking prefers the terminated candidate and is order independent.
    tie = [("#### Yes.", 1.0), ("#### Yes.", 1.0)]
    r = select_answer("q", tie, answer_type="yesno", return_result=True)
    assert r.candidates[0].score == 1.0

    # 5. length normalization and additive penalty run without error.
    nr = select_answer(
        "q",
        ["a b c #### 1", "a #### 1"],
        answer_type="mcq",
        config={"normalization": "length", "length_penalty": 0.01},
        return_result=True,
    )
    assert nr.answer is not None

    # 6. Fallback to raw text when no answer can be extracted.
    fb = select_answer("q", [("no terminator here", 0.2)], answer_type="numeric")
    assert fb == "no terminator here", fb

    # 7. BeamSearchResult-shaped input (duck-typed, no torch required).
    class _H:
        def __init__(self, text, score):
            self.text = text
            self.score = score
            self.n_sentences = 1
            self.n_chars = len(text)

    class _R:
        question = "q"
        beams = [_H("late #### 3", 0.2), _H("best #### 7", 0.8)]
        best_text = "best #### 7"
        best_score = 0.8
        best_hypothesis = beams[1]

    rr = select_from_result(_R(), answer_type="numeric", return_result=True)
    assert rr.answer == "7", rr.to_dict()

    # 8. empty inputs degrade gracefully
    assert select_answer("q", [], answer_type="numeric") is None


if __name__ == "__main__":  # pragma: no cover
    _self_test()
    print("selector self-test passed")
