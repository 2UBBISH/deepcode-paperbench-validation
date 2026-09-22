"""TriviaQA substring-match scoring (Appendix C.1).

The CFG paper (Section 3.1 / Appendix C.1) evaluates TriviaQA with substring
matching rather than exact matching:

    "We run TriviaQA based on the LLaMA (Touvron et al., 2023) methodology,
     however we perform substring match rather than exact match. This stems
     from manual analysis which showed that exact matching disqualified
     answers like "Mark Twain" (with quotes) or His name is Mark Twain
     instead of the exact Mark Twain."

Implementation notes
--------------------
Both the reference answer(s) and the model prediction are normalized
(lower-cased, surrounding quotes/whitespace/punctuation stripped, articles and
repeated whitespace collapsed) and then compared with *substring containment*
in either direction.  A prediction is correct if any of the (normalized)
reference answers occurs inside the (normalized) prediction.

The module is deliberately dependency-free (standard library only) so that the
zero-shot sweep in ``scripts/run_zero_shot.py`` and the LM-Evaluation-Harness
shim in ``src/eval/harness_cfg.py`` can use it on a pure-CPU install.

References
----------
* Paper Section 3.1, Appendix C.1 (TriviaQA substring-match clarification).
* ``triviaqa`` official evaluation (Joshi et al., 2017) for the normalization
  conventions (``answer.normalized_value``, alias lists).
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_answer",
    "strip_surrounding_quotes",
    "substring_match",
    "substring_match_any",
    "triviaqa_score",
    "normalize_references",
    "TriviaQAScorer",
    "TriviaQAResult",
    "QUOTE_CHARS",
    "ARTICLES",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Characters treated as quotes when stripping a prediction's extremities.
QUOTE_CHARS = "\"'`\u2018\u2019\u201c\u201d\u00ab\u00bb"

#: Leading articles dropped during normalization (SQuAD/TriviaQA convention).
ARTICLES = {"a", "an", "the"}

#: Punctuation removed by the standard normalizer.
_PUNCT_TABLE = {ord(ch): " " for ch in string.punctuation}

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the)\s+")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def strip_surrounding_quotes(text: str) -> str:
    """Remove surrounding quotation marks / whitespace / trailing punctuation.

    Handles raw ``"Mark Twain"`` (the exact failure mode described in
    Appendix C.1) as well as curly quotes and multi-character wrappers.
    """
    if not text:
        return ""
    out = text.strip()
    # Drop trailing sentence punctuation before stripping quotes.
    out = out.rstrip(".,;:!?")
    # Repeatedly peel matching quote pairs from both ends.
    changed = True
    while changed and len(out) >= 2:
        changed = False
        if out[0] in QUOTE_CHARS and out[-1] in QUOTE_CHARS:
            out = out[1:-1].strip()
            changed = True
        elif out[0] in QUOTE_CHARS:
            out = out[1:].strip()
            changed = True
        elif out[-1] in QUOTE_CHARS:
            out = out[:-1].strip()
            changed = True
        out = out.rstrip(".,;:!?").strip()
    return out


def normalize_answer(text: Optional[Union[str, int, float]]) -> str:
    """Normalize an answer string for substring comparison.

    Steps (SQuAD/TriviaQA-style, plus leniency for generated text):

    1. ``None`` -> ``""``; non-strings are cast to ``str``.
    2. Extract the content of a final ``\\boxed{...}`` if present.
    3. Take the content after a leading answer phrase such as
       ``"the answer is"`` / ``"answer:"`` when present.
    4. Strip surrounding quotes and trailing punctuation.
    5. Lower-case.
    6. Remove punctuation, collapse whitespace, drop leading articles.

    Parameters
    ----------
    text:
        Raw reference answer or model prediction.

    Returns
    -------
    str
        The normalized string (possibly empty).
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    out = text.strip()
    if not out:
        return ""

    boxed = _BOXED_RE.findall(out)
    if boxed:
        out = boxed[-1]

    lowered = out.lower()
    # Look for an explicit answer phrase and keep what follows it.
    for phrase in ("the answer is", "answer is", "answer:"):
        idx = lowered.rfind(phrase)
        if idx != -1:
            candidate = out[idx + len(phrase):].strip()
            if candidate:
                out = candidate
                lowered = out.lower()
            break

    out = strip_surrounding_quotes(out)
    out = out.lower().translate(_PUNCT_TABLE)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    # TriviaQA answers are usually short; dropping a leading article makes
    # "the godfather" and "godfather" equivalent.
    out = _LEADING_ARTICLE_RE.sub("", out).strip()
    return out


def normalize_references(references: Union[str, Sequence[Any], Dict[str, Any], None]) -> List[str]:
    """Coerce many reference shapes into a flat list of normalized strings.

    Accepts:
      * a plain string,
      * a list/tuple of strings (TriviaQA alias list),
      * a dict as produced by the ``trivia_qa`` dataset, e.g.
        ``{"value": ..., "aliases": [...], "normalized_value": ...}``,
      * a dict with question/answer metadata such as ``{"answer": ...}``.
    """
    if references is None:
        return []

    raw: List[Any] = []

    def _collect(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            raw.append(obj)
        elif isinstance(obj, dict):
            for key in ("answer", "value", "normalized_value", "label", "target"):
                if key in obj:
                    raw.append(obj[key])
            for key in ("aliases", "acceptable_answers", "answers"):
                value = obj.get(key)
                if isinstance(value, (list, tuple)):
                    raw.extend(list(value))
                elif isinstance(value, str):
                    raw.append(value)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _collect(item)
        else:
            raw.append(obj)

    _collect(references)

    normalized: List[str] = []
    for item in raw:
        value = normalize_answer(item)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _contains_longest(prediction_norm: str, reference_norm: str) -> bool:
    """True if ``prediction_norm`` contains ``reference_norm`` as a substring.

    Falls back to the longest whitespace-delimited span of the reference when
    the full reference is not present (so ``"the capital of France is Paris"``
    matches the reference ``"paris france"``).
    """
    if not prediction_norm or not reference_norm:
        return False
    if reference_norm in prediction_norm:
        return True
    if prediction_norm in reference_norm:
        # Prediction is a shorter but complete form of the answer.
        return True
    # Word-boundary containment (avoids "par" matching inside "paris").
    if re.search(r"(?<![0-9a-z])" + re.escape(reference_norm) + r"(?![0-9a-z])", prediction_norm):
        return True
    return False


def substring_match(prediction: Optional[str], reference: Union[str, Sequence[Any], None]) -> bool:
    """Substring-match a prediction against one (or many) reference answers.

    Both sides are normalized with :func:`normalize_answer`; the match
    succeeds if any normalized reference is a substring of the normalized
    prediction (or vice versa, allowing short-form answers).
    """
    prediction_norm = normalize_answer(prediction)
    if not prediction_norm:
        return False

    references_norm = normalize_references(reference)
    if not references_norm:
        return False

    if isinstance(reference, str):
        references_norm.append(prediction_norm)

    for ref in references_norm:
        if _contains_longest(prediction_norm, ref):
            return True
    return False


def substring_match_any(
    prediction: Optional[str],
    references: Iterable[Union[str, Sequence[Any]]],
) -> Tuple[bool, Optional[int]]:
    """Return ``(matched, index)`` for the first matching reference group."""
    for i, reference in enumerate(references):
        if substring_match(prediction, reference):
            return True, i
    return False, None


@dataclass
class TriviaQAResult:
    """Per-example TriviaQA scoring record (as a dict via :meth:`as_dict`)."""

    question_id: Optional[str]
    prediction: str
    reference: List[str]
    correct: bool
    matched_reference: Optional[str] = None
    normalized_prediction: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "prediction": self.prediction,
            "reference": list(self.reference),
            "correct": bool(self.correct),
            "matched_reference": self.matched_reference,
            "normalized_prediction": self.normalized_prediction,
        }


def triviaqa_score(
    predictions: Union[str, Sequence[str]],
    references: Union[Any, Sequence[Any], None] = None,
    question_ids: Optional[Sequence[Optional[str]]] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """Score TriviaQA predictions with substring matching.

    Flexible signature:

    * ``triviaqa_score(prediction, reference)`` -> ``1.0`` / ``0.0`` for a
      single example.
    * ``triviaqa_score(predictions, references)`` -> mean accuracy (float)
      over all examples.
    * With ``return_details=True`` -> dict with ``accuracy``, ``n``,
      ``n_correct`` and the per-example ``details``.

    Parameters
    ----------
    predictions:
        One prediction string, or a sequence of prediction strings.
    references:
        Matching reference(s): a single answer/alias list (for one prediction)
        or a sequence of references (one per prediction).
    question_ids:
        Optional ids carried into the per-example details.
    return_details:
        When ``True`` return the full scoring dictionary instead of the
        scalar accuracy.
    """
    single = isinstance(predictions, str)
    if single:
        is_correct = substring_match(predictions, references)
        if not return_details:
            return 1.0 if is_correct else 0.0
        refs = normalize_references(references)
        detail = TriviaQAResult(
            question_id=question_ids[0] if question_ids else None,
            prediction=predictions,
            reference=refs,
            correct=is_correct,
            matched_reference=next(
                (r for r in refs if _contains_longest(normalize_answer(predictions), r)), None
            ),
            normalized_prediction=normalize_answer(predictions),
        )
        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "n": 1,
            "n_correct": int(is_correct),
            "details": [detail.as_dict()],
        }

    preds = list(predictions)
    n = len(preds)
    if n == 0:
        empty = {"accuracy": float("nan"), "n": 0, "n_correct": 0, "details": []}
        return empty if return_details else float("nan")

    # Decide whether `references` is a per-example sequence or one shared answer.
    per_example: List[Any]
    if references is None:
        per_example = [None] * n
    elif isinstance(references, str):
        per_example = [references] * n
    elif isinstance(references, dict):
        per_example = [references] * n
    else:
        ref_list = list(references)
        if len(ref_list) == n:
            per_example = ref_list
        else:
            # Ambiguous: treat as a single alias list shared by all predictions.
            per_example = [references] * n

    details: List[TriviaQAResult] = []
    n_correct = 0
    for i, (pred, ref) in enumerate(zip(preds, per_example)):
        is_correct = substring_match(pred, ref)
        n_correct += int(is_correct)
        refs = normalize_references(ref)
        details.append(
            TriviaQAResult(
                question_id=question_ids[i] if question_ids is not None and i < len(question_ids) else None,
                prediction=pred,
                reference=refs,
                correct=is_correct,
                matched_reference=next(
                    (r for r in refs if _contains_longest(normalize_answer(pred), r)), None
                ),
                normalized_prediction=normalize_answer(pred),
            )
        )

    accuracy = n_correct / n
    if not return_details:
        return accuracy
    return {
        "accuracy": accuracy,
        "n": n,
        "n_correct": n_correct,
        "details": [d.as_dict() for d in details],
    }


@dataclass
class TriviaQAScorer:
    """Stateful TriviaQA scorer mirroring the LM-Eval-Harness ``process_results``.

    Usage
    -----
    >>> scorer = TriviaQAScorer()
    >>> scorer.add("Paris", ["paris", "Paris, France"])
    >>> scorer.accuracy
    1.0
    """

    #: When True (paper default) use substring matching; else exact normalized match.
    substring: bool = True
    #: Minimum normalized reference length considered reliable for matching.
    min_reference_len: int = 1
    results: List[TriviaQAResult] = field(default_factory=list)

    # -- scoring -----------------------------------------------------------
    def score(self, prediction: Optional[str], reference: Any) -> bool:
        """Score one (prediction, reference) pair and record the result."""
        refs = [
            r for r in normalize_references(reference) if len(r) >= self.min_reference_len
        ]
        if self.substring:
            correct = substring_match(prediction, refs if refs else reference)
        else:
            correct = normalize_answer(prediction) in set(refs) if refs else False
        matched = None
        if correct:
            pred_norm = normalize_answer(prediction)
            matched = next((r for r in refs if _contains_longest(pred_norm, r)), pred_norm)
        self.results.append(
            TriviaQAResult(
                question_id=None,
                prediction=prediction or "",
                reference=refs,
                correct=correct,
                matched_reference=matched,
                normalized_prediction=normalize_answer(prediction),
            )
        )
        return correct

    def add(
        self,
        prediction: Optional[str],
        reference: Any,
        question_id: Optional[str] = None,
    ) -> bool:
        """Alias of :meth:`score` with an optional ``question_id``."""
        correct = self.score(prediction, reference)
        if question_id is not None:
            self.results[-1].question_id = question_id
        return correct

    def add_batch(
        self,
        predictions: Sequence[Optional[str]],
        references: Sequence[Any],
        question_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> List[bool]:
        """Score a batch; returns the per-example correctness flags."""
        out: List[bool] = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            qid = question_ids[i] if question_ids is not None and i < len(question_ids) else None
            out.append(self.add(pred, ref, question_id=qid))
        return out

    # -- aggregation -------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else float("nan")

    def summary(self) -> Dict[str, Any]:
        """Dict with ``accuracy``, ``n``, ``n_correct`` and per-example details."""
        return {
            "accuracy": self.accuracy,
            "n": self.n,
            "n_correct": self.n_correct,
            "details": [r.as_dict() for r in self.results],
        }

    def reset(self) -> None:
        self.results = []


# ---------------------------------------------------------------------------
# LM-Eval-Harness compatibility helpers
# ---------------------------------------------------------------------------


def prepare_harness_references(doc: Dict[str, Any], field: str = "answer") -> List[str]:
    """Extract normalized TriviaQA references from a harness *doc*.

    The ``triviaqa`` harness task stores the answer under ``"answer"`` with the
    supporting aliases under ``"answer"``/``"aliases"`` depending on the dataset
    revision; this helper accepts either shape.
    """
    if not isinstance(doc, dict):
        return normalize_references(doc)
    if field in doc:
        return normalize_references(doc[field])
    return normalize_references(doc)
