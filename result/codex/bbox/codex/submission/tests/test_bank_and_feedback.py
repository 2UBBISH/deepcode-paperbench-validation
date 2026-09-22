"""Positive/negative sample updates (Eqs. 5-6) and outcome supervision."""

from __future__ import annotations

from bbox_adapter.online.bank import SampleBank, SampleEntry, answers_match
from bbox_adapter.online.feedback import GroundTruthFeedback


def make_entry() -> SampleEntry:
    return SampleEntry(
        key="q1",
        question="What is 2 + 2?",
        gold_answer="4",
        gold_solution="Two plus two equals four.\n#### 4",
        choices=None,
    )


def test_initialisation_splits_candidates():
    bank = SampleBank("gsm8k", positive_source="ground_truth", outcome_supervision=True)
    bank.add(make_entry())
    candidates = {"q1": ["reasoning\n#### 4", "reasoning\n#### 5", "reasoning\n#### 6"]}
    bank.initialize(candidates, selector=GroundTruthFeedback("gsm8k"))
    entry = bank.get("q1")
    assert entry.positive == entry.gold_solution  # ground-truth stays positive
    assert "reasoning\n#### 4" in entry.positives()
    assert set(entry.negatives) == {"reasoning\n#### 5", "reasoning\n#### 6"}


def test_update_replaces_negatives_and_keeps_excluded_positive():
    bank = SampleBank("gsm8k", positive_source="ground_truth", outcome_supervision=True)
    bank.add(make_entry())
    bank.initialize({"q1": ["reasoning\n#### 5"]}, selector=GroundTruthFeedback("gsm8k"))
    entry = bank.update(
        bank.get("q1"),
        ["reasoning\n#### 4", "reasoning\n#### 7"],
        selector=GroundTruthFeedback("gsm8k"),
    )
    assert entry.negatives == ["reasoning\n#### 7"]
    assert "reasoning\n#### 4" in entry.extra_positives
    assert entry.positive not in entry.negatives


def test_without_outcome_supervision_every_other_candidate_is_negative():
    bank = SampleBank("gsm8k", positive_source="ground_truth", outcome_supervision=False)
    bank.add(make_entry())
    bank.initialize({"q1": ["a\n#### 4", "b\n#### 4"]}, selector=GroundTruthFeedback("gsm8k"))
    entry = bank.get("q1")
    assert entry.extra_positives == []
    assert entry.negatives == ["b\n#### 4"]


def test_ai_feedback_setting_uses_preferred_candidate_as_positive():
    bank = SampleBank("strategyqa", positive_source="ai_feedback", outcome_supervision=False)
    entry = SampleEntry(key="q", question="Is it?", gold_answer="Yes", gold_solution="Because.\n#### Yes")
    bank.add(entry)
    bank.initialize({"q": ["bad\n#### No", "good\n#### Yes"]}, selector=GroundTruthFeedback("strategyqa"))
    # ground-truth selector falls back to matching candidates; positive becomes
    # the selected candidate rather than the gold solution in this setting.
    assert bank.get("q").positive == "good\n#### Yes"


def test_answers_match_numeric_equivalence():
    assert answers_match("gsm8k", "#### 18.0", "18")
    assert not answers_match("gsm8k", "#### 18", "19")
    assert answers_match("strategyqa", "#### yes", "Yes")
