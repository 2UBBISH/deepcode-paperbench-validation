"""VQA accuracy (VQAv2, TextVQA) as used in Sec. 4.1 of the paper.

Both VQAv2 and TextVQA come with ten human answers per question and are scored
*with the official metric*

.. math::

    \\mathrm{acc}(\\hat{a}) = \\frac{1}{|\\mathcal{A}|}
        \\sum_{a \\in \\mathcal{A}_{unique}} \\min\\Big(\\frac{\\#\\{a\\}}{3}, 1\\Big),

i.e. an answer gets full credit as soon as three annotators agree with it, and
the answers are compared after the official normalization of the respective
benchmark (articles / punctuation are removed and numbers are written as digits).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Sequence

__all__ = [
    "normalize_answer",
    "textvqa_normalize",
    "vqa_accuracy",
    "VQAAccuracy",
    "TextVQAAccuracy",
    "most_frequent_answers",
]

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "its": "it's",
    "itd": "it'd",
    "lets": "let's",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyd": "they'd",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wheres": "where's",
    "whos": "who's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youre": "you're",
    "youve": "you've",
}
_DIGITS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_answer(answer: str) -> str:
    """Official VQA answer normalization."""
    answer = str(answer).lower().strip()
    answer = answer.replace("\n", " ").replace("\t", " ")
    answer = _ARTICLES.sub(" ", answer)
    answer = answer.translate(_PUNCT_TABLE)
    tokens = []
    for token in answer.split():
        token = _CONTRACTIONS.get(token, token)
        token = _DIGITS.get(token, token)
        tokens.append(token)
    return " ".join(tokens).strip()


def textvqa_normalize(answer: str) -> str:
    """TextVQA normalization: like VQA, but keeps a few time / digit patterns."""
    answer = str(answer).lower().strip()
    answer = answer.replace(",", "").replace("?", "").replace("!", "").replace(".", "")
    answer = re.sub(r"\b(\d+)\s*(am|pm)\b", r"\1\2", answer)
    answer = re.sub(r"\s+", " ", answer)
    return normalize_answer(answer)


def vqa_accuracy(prediction: str, ground_truths: Sequence[str], normalizer=normalize_answer) -> float:
    """Official VQA accuracy of a single prediction (``min(count / 3, 1)`` averaged)."""
    if len(ground_truths) == 0:
        return 0.0
    normalized_prediction = normalizer(prediction)
    if normalized_prediction == "":
        return 0.0
    counts = Counter(normalizer(gt) for gt in ground_truths)
    score = 0.0
    for answer, count in counts.items():
        if answer == normalized_prediction:
            score += min(count / 3.0, 1.0)
    return score / len(ground_truths)


def most_frequent_answers(answers: Sequence[str], k: int = 5) -> List[str]:
    """The ``k`` most frequent ground truth answers (App. B.6 for the VQA attacks)."""
    order: List[str] = []
    counts: Counter = Counter()
    for answer in answers:
        normalized = normalize_answer(answer)
        if normalized not in counts:
            order.append(normalized)
        counts[normalized] += 1
    return sorted(order, key=lambda a: (-counts[a], order.index(a)))[:k]


class VQAAccuracy:
    """Per-sample / aggregated VQA accuracy (Sec. 4.1)."""

    name = "vqa_accuracy"
    normalizer = staticmethod(normalize_answer)

    def score(self, prediction: str, ground_truths: Sequence[str]) -> float:
        return vqa_accuracy(prediction, ground_truths, normalizer=self.normalizer)

    def score_batch(self, predictions: Sequence[str], ground_truths: Sequence[Sequence[str]]) -> List[float]:
        return [self.score(pred, gts) for pred, gts in zip(predictions, ground_truths)]

    def aggregate(self, scores: Sequence[float]) -> float:
        return 100.0 * sum(scores) / max(len(scores), 1)


class TextVQAAccuracy(VQAAccuracy):
    """TextVQA uses its own answer normalization."""

    name = "textvqa_accuracy"
    normalizer = staticmethod(textvqa_normalize)
