"""Chain-of-Thought (CoT) evaluation utilities.

Reproduces the paired curves of Figure 2 (GSM8K) and Figure 17 (AQuA) from
"Stay on Topic with Classifier-Free Guidance" (§3.2 / Appendix C.5):

    ...using CFG increases the percentage of CoT resulting in valid, parsable
    answers. For low guidance strengths, model performances increase. However,
    for gamma > 1.5, the quality of reasoning chains degrade, and overall the
    performances drop (§3.2).

For every generation two independent questions are answered:

(a) **Is the chain valid / parsable?**  I.e. does the generated reasoning chain
    terminate in the task's answer marker (``####`` for GSM8K, ``The answer is
    <letter>`` for AQuA) and is the emitted answer itself parsable?
(b) **Is the answer correct?**  I.e. does the extracted final numeric / letter
    answer match the gold answer?

The aggregate of (a) over the evaluation set gives the *% invalid chains* curve
(bottom panels of Figures 2 / 17); the aggregate of (b) gives the *accuracy*
curve (top panels).

Task conventions (Appendix C.5)
-------------------------------
GSM8K (Cobbe et al. 2021):
    Few-shot prompt from Wang et al. 2023 (Self-Consistency, 8-shot).  The chain
    is a natural-language solution terminating in a line ``#### <number>``.
    The final answer is the numeric literal following the marker.

AQuA (Ling et al. 2017):
    Multiple-choice algebra word problems with options (A)-(E).  The chain
    terminates in ``The answer is <letter>``.  The final answer is the letter.

``parse_chain`` / ``extract_answer`` are deliberately lenient about surface form
(whitespace, ``\\boxed{7}``, ``$72``, ``(C)``, trailing punctuation) so that a
*valid* chain is one a human would grade, not one that matches a rigid regex.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = [
    "CoTConfig",
    "CoTResult",
    "GSM8K_ANSWER_MARKER",
    "AQUA_ANSWER_MARKER",
    "extract_answer",
    "extract_gold_answer",
    "parse_chain",
    "is_valid_chain",
    "answers_match",
    "score_generation",
    "evaluate_cot",
    "accuracy_vs_gamma",
    "invalid_rate_vs_gamma",
    "cot_curves",
    "aggregate_by_gamma",
    "to_float",
    "normalize_numeric",
    "normalize_letter",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

GSM8K_ANSWER_MARKER = "####"
"""GSM8K chains terminate in a line beginning with this marker."""

AQUA_ANSWER_MARKER = "The answer is"
"""AQuA chains terminate in this phrase followed by a choice letter."""

#: Phrases that indicate a completed chain even without the task's own marker
#: (some models answer GSM8K with ``The answer is 42``).
ANSWER_PHRASES: Tuple[str, ...] = (
    GSM8K_ANSWER_MARKER,
    AQUA_ANSWER_MARKER,
    "The final answer is",
    "Final answer:",
    "Answer:",
    "the answer is",
)

#: AQuA multiple-choice letters.
CHOICE_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E")

#: Tolerance for comparing two numeric answers parsed from text.
NUMERIC_TOL = 1e-4

# Regexes -------------------------------------------------------------------
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
#: ``#### 42`` / ``####42`` / ``#### $42`` / ``#### \boxed{42}``
_GSM8K_AFTER_MARKER_RE = re.compile(
    r"#{2,}\s*(?:\$+|\\boxed\{|\\\(|\(|\s)*(?P<num>-?\d[\d,]*\.?\d*)",
)
#: ``The answer is C`` / ``answer is (C)`` / ``Answer: C)``
_AQUA_AFTER_PHRASE_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(?:option\s*)?[\s\(\[]*"
    r"(?P<letter>[A-Ea-e])\b",
    re.IGNORECASE,
)
_BOXED_LETTER_RE = re.compile(r"\(\s*(?P<letter>[A-Ea-e])\s*\)")
_BOXED_NUM_RE = re.compile(r"\\boxed\{\s*(?P<num>-?\d[\d,]*\.?\d*)\s*\}")
_CURRENCY_STRIP_RE = re.compile(r"[$,]")


# --------------------------------------------------------------------------- #
# Configuration / result containers
# --------------------------------------------------------------------------- #


@dataclass
class CoTConfig:
    """Task description used by the parsers.

    Attributes:
        task: ``"gsm8k"`` (numeric answer, ``####`` marker) or ``"aqua"``
            (multiple-choice letter, ``The answer is <letter>``).
        answer_marker: Substring terminating a valid chain; defaulted from
            ``task`` when left as ``None``.
        require_marker: If True, a chain is only valid when the terminating marker
            is present.  If False, any chain from which a parsable answer can be
            extracted counts as valid (lenient mode).
        max_answer_length: Maximum characters after the marker still attributed
            to the answer (guards against runaway chains).
    """

    task: str = "gsm8k"
    answer_marker: Optional[str] = None
    require_marker: bool = True
    max_answer_length: int = 64

    def __post_init__(self) -> None:
        self.task = (self.task or "gsm8k").lower().strip()
        if self.task in ("gsm", "gsm-8k", "gsm8k"):
            self.task = "gsm8k"
        elif self.task in ("aqua", "aqua-rat", "a-qua"):
            self.task = "aqua"
        if self.answer_marker is None:
            self.answer_marker = (
                GSM8K_ANSWER_MARKER if self.task == "gsm8k" else AQUA_ANSWER_MARKER
            )

    @property
    def is_multiple_choice(self) -> bool:
        return self.task == "aqua"


@dataclass
class CoTResult:
    """Parsed representation of a single chain-of-thought generation.

    Attributes:
        text: Raw generated completion.
        answer: Extracted final answer (``str``) or ``None`` if unparsable.
        valid: Whether the chain terminates in a valid, parsable answer.
        has_marker: Whether the task's answer marker is present at all.
        n_reasoning_steps: Rough count of arithmetic reasoning steps before the
            marker (used for the qualitative chain comparisons of Tables 15/16).
        chain: The reasoning portion (everything before the marker).
        reason: Human-readable explanation for an invalid chain.
    """

    text: str
    answer: Optional[str] = None
    valid: bool = False
    has_marker: bool = False
    n_reasoning_steps: int = 0
    chain: str = ""
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "answer": self.answer,
            "valid": self.valid,
            "has_marker": self.has_marker,
            "n_reasoning_steps": self.n_reasoning_steps,
            "chain": self.chain,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #


def normalize_numeric(value: Union[str, float, int]) -> Optional[float]:
    """Parse ``value`` into a float, tolerating ``$``/``,`` and LaTeX wrappers.

    Returns ``None`` when no numeric literal can be found.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    boxed = _BOXED_NUM_RE.search(text)
    if boxed:
        text = boxed.group("num")
    else:
        text = _CURRENCY_STRIP_RE.sub("", text).strip().rstrip(".")
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:  # pragma: no cover - defensive
        return None


def normalize_letter(value: Union[str, int]) -> Optional[str]:
    """Return the single uppercase choice letter contained in ``value``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if 0 <= value < len(CHOICE_LETTERS):
            return CHOICE_LETTERS[value]
        return None
    match = re.search(r"[A-Ea-e]", str(value).strip())
    if not match:
        return None
    return match.group(0).upper()


def to_float(value: Any) -> Optional[float]:
    """Alias of :func:`normalize_numeric`."""
    return normalize_numeric(value)


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #


def _extract_gsm8k_answer(text: str, max_len: int = 64) -> Optional[str]:
    """Extract the numeric answer following the ``####`` marker."""
    matches = list(_GSM8K_AFTER_MARKER_RE.finditer(text))
    if matches:
        return matches[-1].group("num").rstrip(".")
    # Lenient fallback: marker present but separated from the number by prose.
    idx = text.rfind(GSM8K_ANSWER_MARKER)
    if idx != -1:
        tail = text[idx + len(GSM8K_ANSWER_MARKER) : idx + len(GSM8K_ANSWER_MARKER) + max_len]
        number = normalize_numeric(tail)
        if number is not None:
            return str(number)
    # Last resort: a boxed number anywhere.
    boxed = _BOXED_NUM_RE.findall(text)
    if boxed:
        return boxed[-1]
    return None


def _extract_aqua_answer(text: str, max_len: int = 64) -> Optional[str]:
    """Extract the choice letter of an AQuA chain."""
    matches = list(_AQUA_AFTER_PHRASE_RE.finditer(text))
    if matches:
        return matches[-1].group("letter").upper()
    tail = text[-max(200, max_len) :]
    boxed = _BOXED_LETTER_RE.findall(tail)
    if boxed:
        return boxed[-1].upper()
    return None


def extract_answer(
    text: str,
    task: Union[str, CoTConfig] = "gsm8k",
    **config_kwargs: Any,
) -> Optional[str]:
    """Extract the final answer of a CoT generation.

    Args:
        text: Raw generated completion (reasoning chain + answer).
        task: ``"gsm8k"`` or ``"aqua"`` (or a pre-built :class:`CoTConfig`).
        **config_kwargs: Forwarded to :class:`CoTConfig` when ``task`` is a str.

    Returns:
        The extracted answer as a ``str`` (numeric literal for GSM8K, uppercase
        letter for AQuA), or ``None`` when no answer can be parsed.
    """
    cfg = task if isinstance(task, CoTConfig) else CoTConfig(task=task, **config_kwargs)
    if text is None:
        return None
    if cfg.is_multiple_choice:
        return _extract_aqua_answer(text, cfg.max_answer_length)
    return _extract_gsm8k_answer(text, cfg.max_answer_length)


def extract_gold_answer(
    reference: Union[str, Dict[str, Any], int, float],
    task: Union[str, CoTConfig] = "gsm8k",
) -> Optional[str]:
    """Normalise a dataset reference into the same space as :func:`extract_answer`.

    Accepts the raw reference string (GSM8K: reference solution ending in
    ``#### <n>``; AQuA: the letter) or a dataset record dict carrying an
    ``"answer"`` / ``"correct"`` / ``"target"`` / ``"label"`` field.
    """
    cfg = task if isinstance(task, CoTConfig) else CoTConfig(task=task)
    if isinstance(reference, dict):
        for key in ("answer", "correct", "target", "label"):
            if reference.get(key) is not None:
                reference = reference[key]
                break
    if reference is None or isinstance(reference, bool):
        return None
    if isinstance(reference, (int, float)):
        return normalize_letter(int(reference)) if cfg.is_multiple_choice else str(reference)
    if cfg.is_multiple_choice:
        return normalize_letter(str(reference))
    return _extract_gsm8k_answer(str(reference), cfg.max_answer_length)


# --------------------------------------------------------------------------- #
# Chain parsing / validity
# --------------------------------------------------------------------------- #


def parse_chain(
    text: str,
    task: Union[str, CoTConfig] = "gsm8k",
    **config_kwargs: Any,
) -> CoTResult:
    """Parse a CoT generation into a :class:`CoTResult`.

    A chain terminates in a valid answer when

    1. the task answer marker (``####`` / ``The answer is``) is present, and
    2. a parsable answer literal follows it (or, when ``require_marker=False``,
       can be found anywhere in the text).

    Chains that are empty, marker-less, or whose tail is unparsable are invalid;
    ``reason`` records which condition failed.
    """
    cfg = task if isinstance(task, CoTConfig) else CoTConfig(task=task, **config_kwargs)

    if text is None:
        return CoTResult(text="", valid=False, reason="empty generation")

    raw = str(text).strip()
    if not raw:
        return CoTResult(text=raw, valid=False, reason="empty generation")

    marker_idx = -1
    matched_marker = cfg.answer_marker
    for marker in (cfg.answer_marker, GSM8K_ANSWER_MARKER, AQUA_ANSWER_MARKER):
        if not marker:
            continue
        idx = raw.rfind(marker)
        if idx > marker_idx:
            marker_idx = idx
            matched_marker = marker
    has_marker = marker_idx >= 0

    chain = raw[:marker_idx].strip() if has_marker else raw
    answer = extract_answer(raw, cfg)
    valid = answer is not None
    if valid and cfg.require_marker and not has_marker:
        valid = False

    reason = ""
    if not valid:
        reason = (
            "no answer marker found"
            if not has_marker
            else f"marker '{matched_marker}' present but answer unparsable"
        )

    return CoTResult(
        text=raw,
        answer=answer,
        valid=bool(valid),
        has_marker=has_marker,
        n_reasoning_steps=_count_reasoning_steps(chain),
        chain=chain,
        reason=reason,
    )


def is_valid_chain(
    text: str,
    task: Union[str, CoTConfig] = "gsm8k",
    **config_kwargs: Any,
) -> bool:
    """Convenience wrapper: ``parse_chain(...).valid``."""
    return parse_chain(text, task, **config_kwargs).valid


_STEP_RE = re.compile(r"(?:=|\\times|\\cdot|\\div|\+|-|\d+\s*%)")


def _count_reasoning_steps(chain: str) -> int:
    """Rough number of arithmetic reasoning steps in a CoT chain.

    Counts equation-like lines, falling back to sentence-level counting when no
    line contains an operator.  Used only for qualitative chain comparisons.
    """
    if not chain:
        return 0
    lines = [ln.strip() for ln in chain.splitlines() if ln.strip()]
    counted = sum(1 for line in lines if _STEP_RE.search(line))
    if counted == 0:
        counted = sum(
            1 for sent in re.split(r"(?<=[.!?])\s+", chain) if _STEP_RE.search(sent)
        )
    return counted


# --------------------------------------------------------------------------- #
# Answer comparison / generation scoring
# --------------------------------------------------------------------------- #


def answers_match(
    prediction: Optional[str],
    gold: Optional[str],
    task: Union[str, CoTConfig] = "gsm8k",
) -> bool:
    """Compare a predicted answer against the gold answer for the given task."""
    if prediction is None or gold is None:
        return False
    cfg = task if isinstance(task, CoTConfig) else CoTConfig(task=task)
    if cfg.is_multiple_choice:
        pred_letter = normalize_letter(prediction)
        gold_letter = normalize_letter(gold)
        return pred_letter is not None and pred_letter == gold_letter
    pred_num = normalize_numeric(prediction)
    gold_num = normalize_numeric(gold)
    if pred_num is None or gold_num is None:
        return str(prediction).strip() == str(gold).strip()
    return abs(pred_num - gold_num) <= NUMERIC_TOL * max(1.0, abs(gold_num))


def score_generation(
    text: str,
    reference: Union[str, Dict[str, Any]],
    task: Union[str, CoTConfig] = "gsm8k",
) -> Dict[str, Any]:
    """Score one generation, returning validity, correctness, and the parse.

    Keys: ``valid``, ``correct``, ``answer``, ``gold``, ``n_reasoning_steps``,
    ``reason``, ``text``, ``chain``, ``has_marker``.
    """
    cfg = task if isinstance(task, CoTConfig) else CoTConfig(task=task)
    parsed = parse_chain(text, cfg)
    gold = extract_gold_answer(reference, cfg)
    correct = bool(parsed.valid and answers_match(parsed.answer, gold, cfg))
    out = parsed.as_dict()
    out.update({"correct": correct, "gold": gold})
    return out


# --------------------------------------------------------------------------- #
# Dataset-level evaluation
# --------------------------------------------------------------------------- #


def evaluate_cot(
    generations: Sequence[str],
    references: Sequence[Union[str, Dict[str, Any]]],
    task: Union[str, CoTConfig] = "gsm8k",
    gammas: Optional[Sequence[float]] = None,
    return_per_example: bool = True,
) -> Dict[str, Any]:
    """Aggregate CoT accuracy and invalid-chain rate over a generation set.

    This is the function behind the paired curves of Figures 2 and 17.

    Args:
        generations: One generated chain per example (already truncated by the
            generation loop at the answer marker / token budget).
        references: Gold reference per example (GSM8K reference solution or AQuA
            letter / dataset record).
        task: ``"gsm8k"`` or ``"aqua"``.
        gammas: Optional guidance strength per generation.  When provided,
            per-gamma aggregates are returned under ``"by_gamma"``.
        return_per_example: Attach the per-example scores to the result.

    Returns:
        Dict with ``accuracy``, ``invalid_rate``, ``mean_accuracy``,
        ``mean_invalid_rate``, ``n``, ``n_valid``, ``n_correct`` and optionally
        ``per_example`` / ``by_gamma``.
    """
    cfg = task if isinstance(task, CoTConfig) else CoTConfig(task=task)

    if len(generations) != len(references):
        raise ValueError(
            f"generations ({len(generations)}) and references "
            f"({len(references)}) must have the same length"
        )

    per_example: List[Dict[str, Any]] = []
    n_correct = 0
    n_valid = 0
    for text, ref in zip(generations, references):
        scored = score_generation(text, ref, cfg)
        per_example.append(scored)
        n_valid += int(scored["valid"])
        n_correct += int(scored["correct"])

    n = len(per_example)
    accuracy = (n_correct / n) if n else 0.0
    invalid_rate = (1.0 - n_valid / n) if n else 0.0

    result: Dict[str, Any] = {
        "task": cfg.task,
        "n": n,
        "n_correct": n_correct,
        "n_valid": n_valid,
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        # Aliases consumed by the plotting scripts.
        "mean_accuracy": accuracy,
        "mean_invalid_rate": invalid_rate,
    }

    if return_per_example:
        result["per_example"] = per_example

    if gammas is not None:
        if len(gammas) != n:
            raise ValueError("gammas must align with generations")
        result["by_gamma"] = aggregate_by_gamma(per_example, gammas)

    return result


def aggregate_by_gamma(
    per_example: Sequence[Dict[str, Any]],
    gammas: Sequence[float],
) -> Dict[float, Dict[str, float]]:
    """Group already-scored examples by gamma and average the metrics."""
    buckets: Dict[float, List[Dict[str, Any]]] = {}
    for scored, gamma in zip(per_example, gammas):
        buckets.setdefault(float(gamma), []).append(scored)

    out: Dict[float, Dict[str, float]] = {}
    for gamma, rows in buckets.items():
        n = len(rows)
        out[gamma] = {
            "n": float(n),
            "accuracy": sum(int(r["correct"]) for r in rows) / n,
            "invalid_rate": sum(1 - int(r["valid"]) for r in rows) / n,
            "mean_reasoning_steps": sum(r.get("n_reasoning_steps", 0) for r in rows) / n,
        }
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# Curve helpers (Figure 2 / 17)
# --------------------------------------------------------------------------- #


def _points_from_groups(
    groups: Dict[float, Dict[str, float]], key: str
) -> Tuple[List[float], List[float]]:
    gammas = sorted(groups)
    return gammas, [groups[g][key] for g in gammas]


def _ensure_groups(
    generations_or_groups: Union[Sequence[str], Dict[float, Dict[str, float]]],
    references: Optional[Sequence[Union[str, Dict[str, Any]]]],
    gammas: Optional[Sequence[float]],
    task: Union[str, CoTConfig],
) -> Dict[float, Dict[str, float]]:
    """Coerce either a generation set (with gammas) or a group dict into groups."""
    if isinstance(generations_or_groups, dict):
        return generations_or_groups
    if references is None or gammas is None:
        raise ValueError(
            "references and gammas are required when passing raw generations"
        )
    evaluation = evaluate_cot(
        generations_or_groups, references, task, gammas, return_per_example=True
    )
    return evaluation["by_gamma"]


def accuracy_vs_gamma(
    generations_or_groups: Union[Sequence[str], Dict[float, Dict[str, float]]],
    references: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
    gammas: Optional[Sequence[float]] = None,
    task: Union[str, CoTConfig] = "gsm8k",
) -> Tuple[List[float], List[float]]:
    """Return ``(gammas, accuracy)`` -- the top panel of Figure 2 / 17."""
    groups = _ensure_groups(generations_or_groups, references, gammas, task)
    return _points_from_groups(groups, "accuracy")


def invalid_rate_vs_gamma(
    generations_or_groups: Union[Sequence[str], Dict[float, Dict[str, float]]],
    references: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
    gammas: Optional[Sequence[float]] = None,
    task: Union[str, CoTConfig] = "gsm8k",
) -> Tuple[List[float], List[float]]:
    """Return ``(gammas, invalid_rate)`` -- the bottom panel of Figure 2 / 17."""
    groups = _ensure_groups(generations_or_groups, references, gammas, task)
    return _points_from_groups(groups, "invalid_rate")


def cot_curves(
    generations: Sequence[str],
    references: Sequence[Union[str, Dict[str, Any]]],
    gammas: Sequence[float],
    task: Union[str, CoTConfig] = "gsm8k",
) -> Dict[str, Dict[str, Any]]:
    """Build both curves of Figure 2 / 17 in one call.

    Returns ``{"accuracy": {"gamma": [...], "value": [...]},
    "invalid_rate": {"gamma": [...], "value": [...]}, "groups": {...}}`` with
    gammas sorted ascending -- directly plottable as the top (accuracy) and
    bottom (invalid %) panels.
    """
    groups = _ensure_groups(generations, references, gammas, task)
    acc_g, acc_v = _points_from_groups(groups, "accuracy")
    inv_g, inv_v = _points_from_groups(groups, "invalid_rate")
    return {
        "accuracy": {"gamma": acc_g, "value": acc_v},
        "invalid_rate": {"gamma": inv_g, "value": inv_v},
        "groups": groups,
    }


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    gsm8k_gens = [
        "She has 5 apples and buys 3 more, so 5 + 3 = 8.\n#### 8",
        "5 + 3 = 9\n#### 9",
        "I do not know",
        "The total is $12\n#### $12",
    ]
    gsm8k_golds = ["#### 8", "#### 8", "#### 8", "#### 12"]
    print("GSM8K:", evaluate_cot(gsm8k_gens, gsm8k_golds, "gsm8k", gammas=[1.0, 1.5, 1.5, 2.0]))

    aqua_gens = [
        "We solve x = 3, therefore the answer is B.",
        "Hmm, no idea.",
        "The answer is (C)",
    ]
    aqua_golds = ["B", "B", "B"]
    print("AQuA:", evaluate_cot(aqua_gens, aqua_golds, "aqua"))
