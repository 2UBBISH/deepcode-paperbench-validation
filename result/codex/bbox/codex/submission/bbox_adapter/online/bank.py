"""Positive / negative sample banks and the SEL(.) operators of Section 3.4.

The paper maintains two sets for every training question:

* ``y_+`` (positive) -- ground-truth solutions, preferred candidates selected by
  human/AI feedback, or the union of both ("combined" setting);
* ``y_-`` (negative) -- the remaining candidates sampled from the adapted
  inference ``p_theta`` (Eq. 6).

Outcome supervision is applied on top of every setting: candidates whose final
answer agrees with the training answer become additional positive samples while
all remaining candidates are negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..data.answer_extraction import extract_answer


def answers_match(dataset: str, text: str, gold: str, num_choices: Optional[int] = None) -> bool:
    """Whether the final answer of ``text`` agrees with the training answer."""

    if not gold:
        return False
    predicted = extract_answer(dataset, text, num_choices)
    if predicted is None:
        return False
    if dataset.lower() == "gsm8k":
        from ..data.answer_extraction import normalize_number

        left, right = normalize_number(predicted), normalize_number(gold)
        if left is None or right is None:
            return False
        return abs(float(left) - float(right)) < 1e-6
    return predicted.strip().lower() == str(gold).strip().lower()


@dataclass
class SampleEntry:
    """All supervision available for one training question."""

    key: str
    question: str
    gold_answer: str = ""
    gold_solution: str = ""
    choices: Optional[List[str]] = None
    positive: str = ""
    negatives: List[str] = field(default_factory=list)
    extra_positives: List[str] = field(default_factory=list)
    initial_candidates: List[str] = field(default_factory=list)

    @property
    def num_choices(self) -> Optional[int]:
        return len(self.choices) if self.choices else None

    def positives(self) -> List[str]:
        values = [self.positive] + list(self.extra_positives)
        seen, unique = set(), []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                unique.append(value)
        return unique


class OutcomeSupervision:
    """Split candidates into positives / negatives with the training answer."""

    def __init__(self, dataset: str, enabled: bool = True) -> None:
        self.dataset = dataset
        self.enabled = enabled

    def split(self, entry: SampleEntry, candidates: Sequence[str]):
        if not self.enabled or not entry.gold_answer:
            return [], list(candidates)
        positives, negatives = [], []
        for candidate in candidates:
            if answers_match(self.dataset, candidate, entry.gold_answer, entry.num_choices):
                positives.append(candidate)
            else:
                negatives.append(candidate)
        return positives, negatives


class SampleBank:
    """Keeps ``y_+`` / ``y_-`` for every training question (Algorithm 1)."""

    def __init__(self, dataset: str, positive_source: str = "ground_truth",
                 outcome_supervision: bool = True) -> None:
        self.dataset = dataset
        self.positive_source = positive_source
        self.entries: Dict[str, SampleEntry] = {}
        self.supervision = OutcomeSupervision(dataset, enabled=outcome_supervision)

    # ------------------------------------------------------------------ setup
    def add(self, entry: SampleEntry) -> None:
        self.entries[entry.key] = entry

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries.values())

    def get(self, key: str) -> SampleEntry:
        return self.entries[key]

    # -------------------------------------------------------- initialization
    def initialize(self, candidates_per_question: Dict[str, List[str]],
                   selector) -> None:
        """Set ``y_+^(0)`` and ``y_-^(0)`` (Section 3.4, Initialization).

        ``selector`` is a callable ``(entry, candidates) -> (positive, preferred)``
        implementing ``SEL(.)`` with ground-truth or AI feedback.
        """

        for entry in self.entries.values():
            candidates = list(candidates_per_question.get(entry.key, []))
            entry.initial_candidates = candidates
            preferred, all_preferred = selector(entry, candidates)
            selected = preferred or (candidates[0] if candidates else "")
            if self.positive_source in {"ground_truth", "combined"} and entry.gold_solution:
                entry.positive = entry.gold_solution
            else:
                entry.positive = selected
            outcome_positives, negatives = self.supervision.split(entry, candidates)
            if self.positive_source == "combined":
                entry.extra_positives = _dedupe(
                    list(all_preferred) + outcome_positives, exclude={entry.positive}
                )
            else:
                entry.extra_positives = _dedupe(outcome_positives, exclude={entry.positive})
            # ``y_-^(0) = {y_j | j != k}``: every candidate except the selected
            # one (and except the candidates promoted to positives by outcome
            # supervision) is a negative sample.
            excluded = set(entry.positives())
            if selected:
                excluded.add(selected)
            remaining = [c for c in candidates if c not in excluded]
            entry.negatives = _dedupe(remaining) or _dedupe(negatives)

    # -------------------------------------------------------------- updating
    def update(self, entry: SampleEntry, candidates: Sequence[str], selector) -> SampleEntry:
        """Eq. (5): ``y_+^(t) = SEL(y_+^(t-1), {y_hat_m})`` and Eq. (6)."""

        preferred, all_preferred = selector(entry, candidates)
        if self.positive_source in {"ground_truth", "combined"} and entry.gold_solution:
            # The ground-truth positive stays constant through the loop.
            entry.positive = entry.gold_solution
        elif preferred:
            entry.positive = preferred
        outcome_positives, _ = self.supervision.split(entry, candidates)
        if self.positive_source == "combined":
            entry.extra_positives = _dedupe(
                list(entry.extra_positives) + list(all_preferred) + outcome_positives,
                exclude={entry.positive},
            )
        else:
            entry.extra_positives = _dedupe(outcome_positives, exclude={entry.positive})
        remaining = [c for c in candidates if c not in set(entry.positives())]
        # Eq. (6): the newly sampled candidates become the negative set.  If the
        # sampler only produced positives (rare) we keep the previous negatives.
        entry.negatives = _dedupe(remaining) or entry.negatives
        return entry

    # ------------------------------------------------------------- training
    def to_training_examples(self, max_negatives: int = 4):
        from ..adapter.base import TrainingExample

        examples = []
        for entry in self.entries.values():
            positives = entry.positives()
            negatives = entry.negatives
            if not positives:
                continue
            if not negatives:
                # A negative sample from p_theta is required by Eq. (3); fall
                # back to the model's own initial generations.
                negatives = [c for c in entry.initial_candidates if c not in set(positives)]
            if not negatives:
                continue
            examples.append(
                TrainingExample(
                    key=entry.key,
                    question=entry.question,
                    positive=entry.positive or positives[0],
                    negatives=negatives[:max_negatives],
                    extra_positives=[p for p in positives if p != (entry.positive or positives[0])],
                )
            )
        return examples

    def statistics(self) -> Dict[str, float]:
        num_pos = sum(len(entry.positives()) for entry in self.entries.values())
        num_neg = sum(len(entry.negatives) for entry in self.entries.values())
        count = max(1, len(self.entries))
        return {
            "num_questions": len(self.entries),
            "positives_per_question": num_pos / count,
            "negatives_per_question": num_neg / count,
        }


def _dedupe(values: Sequence[str], exclude: Optional[set] = None) -> List[str]:
    exclude = exclude or set()
    seen, unique = set(), []
    for value in values:
        if not value or value in seen or value in exclude:
            continue
        seen.add(value)
        unique.append(value)
    return unique
