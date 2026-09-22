"""Wrappers around the external classifiers used in Table 4.

The paper compares CFG against FUDGE, an external-classifier control method
(Yang & Klein, 2021):

* sentiment control on IMDB (Maas et al., 2011), prompt ``"That was a good
  movie!"``, guidance classifier
  ``bhadresh-savani/distilbert-base-uncased-emotion`` and evaluation
  classifier ``stevhliu/my_awesome_model``;
* toxicity control on Jigsaw (cjadams et al., 2017), prompt ``"Don't be
  mean"``, with ``unitary/toxic-bert`` used both for guidance and evaluation.

The metric of Table 4 is the ``%`` increase in the classification likelihood
of the desired label ("positive" for sentiment, "not toxic" for toxicity)
relative to the vanilla model's generations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ClassifierConfig:
    name: str
    positive_labels: Tuple[str, ...]  # labels that count as "desired"


SENTIMENT_GUIDANCE = ClassifierConfig(
    "bhadresh-savani/distilbert-base-uncased-emotion", ("joy", "love", "surprise")
)
SENTIMENT_EVAL = ClassifierConfig("stevhliu/my_awesome_model", ("POSITIVE", "LABEL_1", "positive"))
TOXICITY = ClassifierConfig("unitary/toxic-bert", ("non-toxic", "LABEL_0", "not toxic", "neutral"))


class TextClassifier:
    """A thin, dependency-light wrapper around a HF sequence classifier.

    ``probability(text)`` returns the probability mass the classifier assigns
    to the "desired" labels, which is the quantity the paper increases.
    """

    def __init__(self, config: ClassifierConfig, device: Optional[str] = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.name).to(self.device)
        self.model.eval()
        self.label_names = [
            str(self.model.config.id2label.get(i, i)).lower()
            for i in range(self.model.config.num_labels)
        ]
        self.positive_idx = [
            i
            for i, label in enumerate(self.label_names)
            if any(target.lower() in label for target in config.positive_labels)
        ]
        if not self.positive_idx:
            # Fall back to the last index (typical binary convention) so the
            # experiment still runs, and report it.
            self.positive_idx = [self.model.config.num_labels - 1]

    @torch.no_grad()
    def probability(self, texts: Sequence[str]) -> List[float]:
        """Probability of the desired label for each text."""
        if not texts:
            return []
        enc = self.tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        logits = self.model(**enc).logits.float()
        probs = F.softmax(logits, dim=-1)
        return probs[:, self.positive_idx].sum(dim=-1).tolist()

    def as_reward(self) -> Callable[[Sequence[str]], List[float]]:
        return self.probability


def percent_increase(baseline: Sequence[float], guided: Sequence[float]) -> float:
    """``%`` increase in mean desired-label probability, as in Table 4."""
    import numpy as np

    base = float(np.mean(baseline)) if baseline else float("nan")
    new = float(np.mean(guided)) if guided else float("nan")
    if base == 0:
        return float("nan")
    return (new - base) / base
