"""Unit tests for :mod:`bbox_adapter.training.buffers`.

These tests exercise the positive/negative sample buffers that implement the
SEL selection function and the Eq. (5) / Eq. (6) update rules of BBox-Adapter
(paper Section 3.4, Appendix G), plus outcome supervision, contrastive-set
extraction (the input of the ranking-NCE objective Eq. (2)) and config
serialization.

Design notes
------------
* The tests are **pure Python** (no torch / transformers / datasets / network),
  so they can run offline and inside CI.
* Where the buffer module is intentionally permissive about shapes (a
  ``QuerySamples.y_plus`` may be stored as a scalar or as a list of strings, and
  the selector built by ``make_selector`` may be invoked with a couple of
  different calling conventions), the helpers ``_as_list`` and
  ``_call_selector`` absorb that ambiguity so the assertions still check the
  *semantics* required by the paper rather than an incidental API detail.

Run with pytest::

    pytest tests/test_buffers.py -v

or directly::

    python tests/test_buffers.py
"""

from __future__ import annotations

import inspect
import os
import sys
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# import plumbing (allow `python tests/test_buffers.py` from the repo root)
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # pragma: no cover - import error is reported by the tests below
    from bbox_adapter.training import buffers as B
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "could not import bbox_adapter.training.buffers; run these tests from "
        "the repository root ({}): {}".format(_ROOT, exc)
    )

try:
    from bbox_adapter.data.answer_extraction import (
        extract_final_answer,
        is_correct,
    )
except Exception:  # pragma: no cover - fall back to local shims
    def extract_final_answer(text, answer_type, **kw):  # type: ignore
        return text

    def is_correct(pred, gold, answer_type, **kw):  # type: ignore
        return str(pred).strip() == str(gold).strip()


# ---------------------------------------------------------------------------
# tolerant helpers
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> List[Any]:
    """Normalize a possibly-scalar positive/negative store into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, dict):
        return list(value.keys())
    try:
        return list(value)
    except TypeError:
        return [value]


def _val(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Return ``obj.name``, calling it when it is a method."""
    attr = getattr(obj, name)
    return attr(*args, **kwargs) if callable(attr) else attr


def _call_selector(
    selector: Callable[..., Any],
    candidates: Sequence[str],
    *,
    question: Optional[str] = None,
    gold: Any = None,
    answer_type: Optional[str] = None,
    choices: Any = None,
) -> Any:
    """Call a selector produced by :func:`make_selector` robustly.

    ``ground_truth_select`` is candidate-first, ``ai_feedback_select`` is
    question-first; the factory returns a wrapper for both, so we try the
    plausible conventions in order and report the first one that works.
    """
    try:
        params = set(inspect.signature(selector).parameters)
    except (TypeError, ValueError):  # builtins / lambdas without signature
        params = set()

    attempts: List[Callable[[], Any]] = []
    if "candidates" in params:
        attempts.append(
            lambda: selector(
                candidates=candidates,
                question=question,
                gold=gold,
                answer_type=answer_type,
                choices=choices,
            )
        )
    attempts.extend(
        [
            lambda: selector(
                candidates,
                gold=gold,
                answer_type=answer_type,
                choices=choices,
                question=question,
            ),
            lambda: selector(
                question,
                candidates,
                gold=gold,
                answer_type=answer_type,
                choices=choices,
            ),
            lambda: selector(candidates, gold=gold, answer_type=answer_type),
            lambda: selector(candidates, gold=gold),
            lambda: selector(question, candidates),
            lambda: selector(candidates),
        ]
    )

    last_exc: Optional[BaseException] = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:  # wrong calling convention -> try the next one
            last_exc = exc
    raise AssertionError(
        "could not call selector {!r} with any known convention: {}".format(
            selector, last_exc
        )
    )


def _numeric_eq(text: Any, target: Any = 7) -> bool:
    """True when ``text`` parses to the numeric answer ``target``."""
    try:
        return bool(is_correct(extract_final_answer(text, "numeric"), target, "numeric"))
    except Exception:
        return str(text).strip() == str(target).strip()


# ---------------------------------------------------------------------------
# shared fixtures (plain functions: no pytest fixtures required)
# ---------------------------------------------------------------------------
GSM_QUESTION = "Weng earns $12 an hour. How much did she earn working 5 hours?"

def _gsm_candidates() -> List[str]:
    # distinct texts *and* distinct numeric answers, so text-dedup and
    # answer-dedup policies give the same partition.
    return [
        "She earns 12 per hour.\n#### The answer is 7",
        "She earns 12 per hour.\n#### The answer is 8",
        "She earns 12 per hour.\n#### The answer is 6",
    ]


def _gsm_config(**kwargs: Any) -> Any:
    params = dict(dataset="gsm8k", answer_type="numeric", sel_mode="ground_truth")
    params.update(kwargs)
    try:
        return B.BufferConfig(**params)
    except TypeError:
        params.pop("answer_type", None)
        return B.BufferConfig(**params)


def _fixed_selector(index: int = 0) -> Callable[..., int]:
    """A selector that always selects ``index`` regardless of call style."""

    def selector(*args: Any, **kwargs: Any) -> int:
        return index

    return selector


# ---------------------------------------------------------------------------
# 1. candidate identity
# ---------------------------------------------------------------------------
def test_candidate_key_is_whitespace_insensitive() -> None:
    a = B.candidate_key("####  Yes. ")
    b = B.candidate_key("#### Yes.")
    c = B.candidate_key("#### No.")
    assert isinstance(a, str) and a
    assert a == b, "candidate_key must ignore surrounding/inner whitespace"
    assert a != c
    # identical strings give identical keys (stable hashing)
    assert B.candidate_key("x") == B.candidate_key("x")


def test_deduplicate_preserves_order() -> None:
    out = _as_list(B.deduplicate(["a", "b", "a", "c", "b"]))
    assert out == ["a", "b", "c"], out
    assert _as_list(B.deduplicate([])) == []


# ---------------------------------------------------------------------------
# 2. SEL: ground-truth selection
# ---------------------------------------------------------------------------
def test_ground_truth_select_numeric() -> None:
    cands = [
        "First attempt.\n#### The answer is 12",
        "Second attempt.\n#### The answer is 9",
        "Third attempt.\n#### The answer is 9",
    ]
    idx = B.ground_truth_select(cands, gold="9", answer_type="numeric")
    assert idx == 1, "ties must break to the first matching candidate: {}".format(idx)

    idx0 = B.ground_truth_select(cands, gold="12", answer_type="numeric")
    assert idx0 == 0

    # gold may be given verbatim in the dataset's answer format
    idx_term = B.ground_truth_select(cands, gold="#### The answer is 12", answer_type="numeric")
    assert idx_term == 0


def test_ground_truth_select_yesno() -> None:
    cands = ["Because of X.\n#### Yes.", "Because of Y.\n#### No."]
    assert B.ground_truth_select(cands, gold="No", answer_type="yesno") == 1
    assert B.ground_truth_select(cands, gold="Yes", answer_type="yesno") == 0


def test_ground_truth_select_mcq() -> None:
    cands = ["reasoning a\n#### 1", "reasoning b\n#### 2", "reasoning c\n#### 3"]
    expected = [
        i
        for i, cand in enumerate(cands)
        if is_correct(extract_final_answer(cand, "mcq"), 2, "mcq")
    ]
    assert expected == [1], "test fixture is inconsistent with answer_extraction"
    assert B.ground_truth_select(cands, gold=2, answer_type="mcq") == 1


def test_ground_truth_select_no_match_returns_valid_index() -> None:
    cands = ["a\n#### The answer is 3", "b\n#### The answer is 4"]
    idx = B.ground_truth_select(cands, gold="999", answer_type="numeric")
    assert isinstance(idx, int)
    assert idx == -1 or 0 <= idx < len(cands)


# ---------------------------------------------------------------------------
# 3. SEL: random baseline + selector factory
# ---------------------------------------------------------------------------
def test_random_select_is_seed_deterministic() -> None:
    cands = _gsm_candidates()
    first = B.random_select(cands, seed=0)
    second = B.random_select(cands, seed=0)
    assert first == second
    assert 0 <= first < len(cands)
    # a different seed is still a valid index
    other = B.random_select(cands, seed=1234)
    assert 0 <= other < len(cands)


def test_make_selector_ground_truth() -> None:
    cands = _gsm_candidates()
    selector = B.make_selector("ground_truth", config=_gsm_config())
    assert callable(selector)
    idx = _call_selector(
        selector, cands, question=GSM_QUESTION, gold="8", answer_type="numeric"
    )
    assert idx == 1, "ground-truth SEL must pick the candidate matching the gold answer"


def test_make_selector_modes_exist() -> None:
    for mode in ("ground_truth", "random", "ai_feedback", "combined"):
        selector = B.make_selector(mode, config=_gsm_config(sel_mode=mode))
        assert callable(selector), mode
    # SEL mode constants of the three paper settings
    assert B.SEL_MODE_GROUND_TRUTH in B.SEL_MODES
    assert B.SEL_MODE_AI_FEEDBACK in B.SEL_MODES
    assert B.SEL_MODE_COMBINED in B.SEL_MODES


def test_sel_mode_aliases_point_at_canonical_modes() -> None:
    aliases = getattr(B, "SEL_MODE_ALIASES", None)
    assert aliases, "SEL_MODE_ALIASES must be defined"
    keys, values = set(aliases.keys()), set(aliases.values())
    for canonical in (B.SEL_MODE_GROUND_TRUTH, B.SEL_MODE_AI_FEEDBACK, B.SEL_MODE_COMBINED):
        assert canonical in keys or canonical in values, canonical


def test_combined_select_prefers_ground_truth() -> None:
    cands = _gsm_candidates()
    idx = B.combined_select(
        GSM_QUESTION, cands, gold="6", answer_type="numeric", dataset="gsm8k"
    )
    assert idx == 2, "combined SEL must use the ground truth when it is available"


# ---------------------------------------------------------------------------
# 4. initialization (t = 0): one positive (SEL) + the K-1 remaining negatives
# ---------------------------------------------------------------------------
def test_initial_query_samples_structure() -> None:
    cands = _gsm_candidates()
    samples = B.initial_query_samples(
        GSM_QUESTION,
        cands,
        selector=_fixed_selector(0),
        gold="7",
        answer_type="numeric",
        dataset="gsm8k",
        config=_gsm_config(),
    )
    positives = _as_list(samples.y_plus)
    negatives = _as_list(samples.y_minus)

    assert positives == [cands[0]], "the SEL-selected candidate must be the positive"
    assert set(negatives) == {cands[1], cands[2]}, negatives
    assert len(negatives) == len(cands) - 1
    assert not samples.is_empty
    assert samples.positive_answer is not None


def test_initial_query_samples_respects_selector_choice() -> None:
    cands = _gsm_candidates()
    samples = B.initial_query_samples(
        GSM_QUESTION,
        cands,
        selector=_fixed_selector(2),
        gold="6",
        answer_type="numeric",
        dataset="gsm8k",
    )
    positives = _as_list(samples.y_plus)
    negatives = _as_list(samples.y_minus)
    assert positives == [cands[2]]
    assert cands[2] not in negatives
    assert set(negatives) == {cands[0], cands[1]}


def test_initialize_buffer_multiple_queries() -> None:
    questions = ["q one?", "q two?"]
    candidate_lists = [
        _gsm_candidates(),
        ["x\n#### The answer is 1", "y\n#### The answer is 2", "z\n#### The answer is 3"],
    ]
    buf = B.initialize_buffer(
        questions,
        candidate_lists,
        selector=_fixed_selector(0),
        golds=["7", "1"],
        answer_types=["numeric", "numeric"],
        dataset="gsm8k",
        config=_gsm_config(),
    )

    assert len(buf) == 2
    assert _val(buf, "n_queries") == 2
    assert _val(buf, "n_with_positive") == 2
    assert _val(buf, "total_positives") >= 2
    assert _val(buf, "total_negatives") >= 2

    qs, pos, neg = buf.contrastive_sets()
    assert len(qs) == len(pos) == len(neg) >= 1
    for question, positive, negatives in zip(qs, pos, neg):
        assert isinstance(positive, str) and positive
        neg_list = _as_list(negatives)
        assert len(neg_list) >= 1
        assert positive not in neg_list, "a positive must never also be a negative"
        assert question in questions


def test_buffer_lookup_and_stats() -> None:
    questions = ["q one?", "q two?"]
    candidate_lists = [
        _gsm_candidates(),
        ["x\n#### The answer is 1", "y\n#### The answer is 2", "z\n#### The answer is 3"],
    ]
    buf = B.initialize_buffer(
        questions,
        candidate_lists,
        selector=_fixed_selector(0),
        golds=["7", "1"],
        answer_types=["numeric", "numeric"],
        dataset="gsm8k",
    )
    stats = _val(buf, "stats")
    assert isinstance(stats, dict)
    coverage = _val(buf, "coverage")
    assert 0.0 <= float(coverage) <= 1.0
    dumped = buf.to_dict()
    assert isinstance(dumped, dict)
    assert len(dumped) > 0


# ---------------------------------------------------------------------------
# 5. Eq. (5) / Eq. (6) buffer refresh
# ---------------------------------------------------------------------------
def test_update_query_samples_follows_eq5_and_eq6() -> None:
    cands = _gsm_candidates()
    cfg = _gsm_config()
    samples = B.initial_query_samples(
        GSM_QUESTION,
        cands,
        selector=_fixed_selector(0),
        gold="7",
        answer_type="numeric",
        dataset="gsm8k",
        config=cfg,
    )

    new_candidates = [
        "better reasoning\n#### The answer is 7",   # correct -> new positive
        "wrong reasoning\n#### The answer is 3",    # negative
        "other reasoning\n#### The answer is 4",    # negative
    ]
    updated = B.update_query_samples(
        samples,
        new_candidates,
        selector=_fixed_selector(0),
        config=cfg,
        question=GSM_QUESTION,
    )

    positives = _as_list(updated.y_plus)
    negatives = _as_list(updated.y_minus)

    # Eq. (5): the SEL-chosen candidate is admitted to the positive set.
    assert new_candidates[0] in positives, positives
    # Eq. (6): y_- = {y_hat_m | y_hat_m != y_+}; no positive may leak into negatives.
    for positive in positives:
        assert positive not in negatives
    assert new_candidates[1] in negatives
    assert new_candidates[2] in negatives


def test_update_positive_and_update_negatives_helpers() -> None:
    cands = _gsm_candidates()
    cfg = _gsm_config()
    samples = B.initial_query_samples(
        GSM_QUESTION,
        cands,
        selector=_fixed_selector(0),
        gold="7",
        answer_type="numeric",
        dataset="gsm8k",
        config=cfg,
    )

    new_candidates = [
        "better reasoning\n#### The answer is 7",
        "wrong reasoning\n#### The answer is 3",
        "other reasoning\n#### The answer is 4",
    ]

    new_positive = B.update_positive(
        samples, new_candidates, selector=_fixed_selector(0), config=cfg, question=GSM_QUESTION
    )
    if new_positive is not None:
        assert new_positive in new_candidates, new_positive

    negatives = _as_list(B.update_negatives(samples, new_candidates, config=cfg))
    assert negatives, "negative store must not become empty"
    for positive in _as_list(samples.y_plus):
        assert positive not in negatives, "Eq. (6): positives are excluded from negatives"


# ---------------------------------------------------------------------------
# 6. outcome supervision
# ---------------------------------------------------------------------------
def test_outcome_supervision_splits_by_final_answer() -> None:
    cfg = _gsm_config(outcome_supervision=True)
    cands = _gsm_candidates()
    samples = B.initial_query_samples(
        GSM_QUESTION,
        cands,
        selector=_fixed_selector(0),
        gold="7",
        answer_type="numeric",
        dataset="gsm8k",
        config=cfg,
    )

    inferences = [
        "long chain of thought\n#### The answer is 7",  # matches -> positive
        "long chain of thought\n#### The answer is 5",  # mismatch -> negative
    ]
    updated = B.apply_outcome_supervision(
        samples, inferences, config=cfg, answer_type="numeric"
    )

    positives = _as_list(updated.y_plus)
    negatives = _as_list(updated.y_minus)

    assert any(_numeric_eq(p, 7) for p in positives), positives
    assert any(_numeric_eq(n, 5) for n in negatives), negatives
    for positive in positives:
        assert positive not in negatives


def test_outcome_supervision_can_be_disabled() -> None:
    cfg = _gsm_config(outcome_supervision=False)
    samples = B.initial_query_samples(
        GSM_QUESTION,
        _gsm_candidates(),
        selector=_fixed_selector(0),
        gold="7",
        answer_type="numeric",
        dataset="gsm8k",
        config=cfg,
    )
    before_pos = list(_as_list(samples.y_plus))
    before_neg = list(_as_list(samples.y_minus))
    updated = B.apply_outcome_supervision(
        samples, ["x\n#### The answer is 7"], config=cfg, answer_type="numeric"
    )
    # with the flag off the helper must not drop existing samples
    assert set(before_pos) <= set(_as_list(updated.y_plus))
    assert set(before_neg) <= set(_as_list(updated.y_minus))


# ---------------------------------------------------------------------------
# 7. configuration round-trip
# ---------------------------------------------------------------------------
def test_buffer_config_roundtrip() -> None:
    cfg = B.BufferConfig(
        dataset="strategyqa",
        sel_mode="ai_feedback",
        k_init=6,
        m_candidates=4,
        max_positives=8,
        max_negatives=32,
        outcome_supervision=False,
        seed=3,
    )
    payload = cfg.to_dict()
    assert isinstance(payload, dict)
    assert payload["dataset"] == "strategyqa"
    assert payload["sel_mode"] == "ai_feedback"

    restored = B.BufferConfig.from_dict(payload)
    assert restored.dataset == "strategyqa"
    assert restored.sel_mode == "ai_feedback"
    assert restored.k_init == 6
    assert restored.m_candidates == 4
    assert restored.max_negatives == 32
    assert restored.to_dict()["outcome_supervision"] == cfg.outcome_supervision


def test_buffer_config_defaults() -> None:
    cfg = B.BufferConfig()
    assert cfg.deduplicate is True
    assert int(cfg.k_init) >= 1
    assert int(cfg.max_negatives) >= 1
    assert cfg.sel_mode in B.SEL_MODES


# ---------------------------------------------------------------------------
# 8. end-to-end online-adaptation-style loop on the buffer (no LLM, no torch)
# ---------------------------------------------------------------------------
def test_buffer_survives_multiple_refresh_rounds() -> None:
    cfg = _gsm_config()
    questions = ["q one?"]
    buf = B.initialize_buffer(
        questions,
        [_gsm_candidates()],
        selector=_fixed_selector(0),
        golds=["7"],
        answer_types=["numeric"],
        dataset="gsm8k",
        config=cfg,
    )

    for round_index in range(3):
        samples = buf[questions[0]]
        candidates = [
            "round {} reasoning\n#### The answer is 7".format(round_index),
            "round {} reasoning\n#### The answer is 2".format(round_index),
            "round {} reasoning\n#### The answer is 9".format(round_index),
        ]
        updated = B.update_query_samples(
            samples,
            candidates,
            selector=_fixed_selector(0),
            config=cfg,
            question=questions[0],
        )
        buf.update(questions[0], updated)

        positive = _as_list(updated.y_plus)[0]
        negatives = _as_list(updated.y_minus)
        assert positive not in negatives
        assert len(negatives) >= 1

    qs, pos, neg = buf.contrastive_sets()
    assert len(qs) == len(pos) == len(neg)
    assert all(isinstance(p, str) and p for p in pos)


# ---------------------------------------------------------------------------
# runner (dependency-free)
# ---------------------------------------------------------------------------
def _all_tests() -> List[Tuple[str, Callable[[], None]]]:
    names = sorted(n for n in list(globals()) if n.startswith("test_"))
    return [(name, globals()[name]) for name in names]


def run_all(verbose: bool = True) -> Dict[str, Any]:
    """Run every ``test_*`` function; return a summary dict."""
    results: Dict[str, Any] = {"passed": [], "failed": [], "errors": {}}
    for name, fn in _all_tests():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report, don't abort
            results["failed"].append(name)
            results["errors"][name] = "{}: {}".format(type(exc).__name__, exc)
            if verbose:
                print("FAIL {}: {}".format(name, exc))
                traceback.print_exc()
        else:
            results["passed"].append(name)
            if verbose:
                print("ok   {}".format(name))
    if verbose:
        print(
            "\n{}/{} buffer tests passed".format(
                len(results["passed"]), len(results["passed"]) + len(results["failed"])
            )
        )
    return results


if __name__ == "__main__":
    summary = run_all()
    sys.exit(1 if summary["failed"] else 0)
