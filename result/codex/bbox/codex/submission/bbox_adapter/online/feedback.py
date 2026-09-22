"""``SEL(.)``: selecting positives with ground-truth or AI feedback.

Ground-Truth setting: the answer of a candidate is compared to the training
answer.  AI Feedback setting: an advanced LLM (gpt-4) simulates human
preference with the criteria of Appendix G (coherency, reasonability,
correctness, format) and the prompts of Appendix J.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..data.prompts import build_ai_feedback_prompt
from .bank import SampleEntry, answers_match


class GroundTruthFeedback:
    """Select the candidate whose final answer matches the training answer."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset

    def __call__(self, entry: SampleEntry, candidates: Sequence[str]):
        matches = [
            candidate
            for candidate in candidates
            if answers_match(self.dataset, candidate, entry.gold_answer, entry.num_choices)
        ]
        if matches:
            return matches[0], matches
        return (candidates[0] if candidates else ""), []


@dataclass
class AIFeedback:
    """gpt-4 based rater implementing the selection criteria of Appendix G."""

    llm: object
    dataset: str
    num_selected: int = 1
    max_candidates: int = 8

    def _parse(self, response: str, num_candidates: int) -> List[int]:
        # Prompts ask for "Candidate Answer _"; TruthfulQA asks for a top-5
        # ranking.  Parse both, in order of appearance.
        order: List[int] = []
        for match in re.finditer(r"[Cc]andidate\s*[Aa]nswer\s*(\d+)", response or ""):
            index = int(match.group(1)) - 1
            if 0 <= index < num_candidates and index not in order:
                order.append(index)
        if not order:
            for match in re.finditer(r"^\s*(?:The\s+)?(?:\d+(?:st|nd|rd|th)|Best)\b.*?(\d+)",
                                     response or "", flags=re.MULTILINE):
                index = int(match.group(1)) - 1
                if 0 <= index < num_candidates and index not in order:
                    order.append(index)
        return order

    def __call__(self, entry: SampleEntry, candidates: Sequence[str]):
        if not candidates:
            return "", []
        candidates = list(candidates)[: self.max_candidates]
        prompt = build_ai_feedback_prompt(
            self.dataset, entry.question, candidates, entry.choices
        )
        response = self.llm.generate_one(prompt, temperature=0.0, top_p=1.0)
        order = self._parse(response, len(candidates))
        if not order:
            return candidates[0], [candidates[0]]
        selected = [candidates[i] for i in order[: self.num_selected]]
        return selected[0], selected


class CombinedFeedback:
    """Ground-truth positives augmented with AI preferred candidates."""

    def __init__(self, dataset: str, ground_truth: GroundTruthFeedback, ai: AIFeedback) -> None:
        self.dataset = dataset
        self.ground_truth = ground_truth
        self.ai = ai

    def __call__(self, entry: SampleEntry, candidates: Sequence[str]):
        gt_positive, gt_all = self.ground_truth(entry, candidates)
        ai_positive, ai_all = self.ai(entry, candidates)
        preferred = gt_positive or ai_positive
        merged: List[str] = []
        for candidate in list(gt_all) + list(ai_all):
            if candidate not in merged:
                merged.append(candidate)
        return preferred, merged


def build_feedback(positive_source: str, dataset: str, llm=None):
    """Factory for the three settings of Section 4.1."""

    ground_truth = GroundTruthFeedback(dataset)
    if positive_source == "ground_truth":
        return ground_truth
    if llm is None:
        raise ValueError("The AI feedback setting requires a gpt-4 client")
    ai = AIFeedback(llm=llm, dataset=dataset,
                    num_selected=5 if dataset == "truthfulqa" else 1)
    if positive_source == "ai_feedback":
        return ai
    if positive_source == "combined":
        return CombinedFeedback(dataset, ground_truth, ai)
    raise ValueError(f"Unknown positive_source {positive_source!r}")
