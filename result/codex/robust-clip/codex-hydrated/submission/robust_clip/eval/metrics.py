"""Task metrics: VQA accuracy (Antol et al., 2015) and POPE F1."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence


_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
    "hes": "he's", "im": "i'm", "isnt": "isn't", "its": "it's",
    "itd": "it'd", "lets": "let's", "mightnt": "mightn't", "mustnt": "mustn't",
    "shant": "shan't", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's",
    "theyd": "they'd", "theyre": "they're", "theyve": "they've",
    "wasnt": "wasn't", "werent": "weren't", "whats": "what's",
    "wheres": "where's", "whos": "who's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all",
    "youre": "you're", "youve": "you've",
}

_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = list(string.punctuation)


def process_vqa_answer(answer: str, ascii_only: bool = False) -> str:
    """The official VQA answer normalisation."""
    answer = answer.replace("\n", " ").replace("\t", " ").strip().lower()
    answer = _PERIOD_STRIP.sub("", answer, re.UNICODE)
    for punct in _PUNCT:
        if (punct + " " in answer or " " + punct in answer) or punct in _ARTICLES:
            answer = answer.replace(punct, "")
    answer = "".join(ch for ch in answer if ord(ch) < 128) if ascii_only else answer
    answer = answer.replace(",", "")
    answer = " ".join([_CONTRACTIONS.get(w, w) for w in answer.split()])
    answer = " ".join(w for w in answer.split() if w not in _ARTICLES)
    return answer.strip()


def vqa_score(prediction: str, answers: Sequence[str]) -> float:
    """VQA accuracy (Antol et al., 2015) of one prediction against 10 answers.

    ``acc = min(#annotators that gave the predicted answer / 3, 1)``.
    """
    if not answers:
        return 0.0
    prediction = process_vqa_answer(prediction)
    matching = sum(1 for answer in answers if process_vqa_answer(answer) == prediction)
    return min(1.0, matching / 3.0)


def vqa_accuracy(predictions: Sequence[str], answers: Sequence[Sequence[str]]) -> List[float]:
    return [vqa_score(p, a) for p, a in zip(predictions, answers)]


def vqa_accuracy_percentage(predictions: Sequence[str], answers: Sequence[Sequence[str]]) -> float:
    scores = vqa_accuracy(predictions, answers)
    return 100.0 * sum(scores) / max(1, len(scores))


# --------------------------------------------------------------------------- #
#                                    POPE                                      #
# --------------------------------------------------------------------------- #
def _parse_yes_no(text: str) -> str:
    """Extract the model's yes/no decision from its free-form answer."""
    text = text.strip().lower()
    # POPE prompts LLaVA to answer with a single word; fall back to a scan.
    for token in re.findall(r"[a-z']+", text):
        if token in ("yes", "no"):
            return token
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "other"


def pope_f1(predictions: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    """F1 of the "yes" class, as reported in Table 5 of the paper."""
    tp = fp = fn = tn = 0
    for prediction, label in zip(predictions, labels):
        pred = _parse_yes_no(prediction)
        gold = str(label).strip().lower()
        if pred == "yes" and gold == "yes":
            tp += 1
        elif pred == "yes" and gold == "no":
            fp += 1
        elif pred == "no" and gold == "yes":
            fn += 1
        elif pred == "no" and gold == "no":
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "yes": tp + fn,
        "no": tn + fp,
    }
