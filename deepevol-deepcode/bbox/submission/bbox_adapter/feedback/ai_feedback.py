"""AI feedback (GPT-4 rater) for BBox-Adapter positive-sample selection (SEL).

Paper grounding
---------------
* Section 3.4 (Online Adaptation): when ground truth is unavailable the positive
  sample is obtained by asking an advanced LLM (GPT-4) to simulate human
  preference over the ``K`` responses sampled from the black-box LLM:

      y_{i+}^{(0)} = y_{i,k} = SEL({y_{i,j}}_{j=1..K})

  and, inside each iteration ``t``, the positive set is refreshed from the
  previous positive plus the newly sampled candidates::

      y_{i+}^{(t)} = SEL(y_{i+}^{(t-1)}, {y_hat_{i,m}}_{m=1..M})

  The remaining candidates become negatives (Eq. 6).

* Section 4.1 (Settings, "AI Feedback"): an advanced LLM (gpt-4) is used to
  simulate human preference and the most preferred candidate is selected as the
  positive sample; candidate answers come from the adapted inference
  ``p_{theta_t}``.

* Appendix G (selection criteria): (1) Coherency, (2) Reasonability,
  (3) Correctness, (4) Format.  All four criteria are passed to the rater
  (see ``..llm.prompts.RATER_CRITERIA``).

* Appendix J (prompt design): the best-answer rater output format
  ("Best Answer and Explanation:") for StrategyQA / GSM8K / ScienceQA, and the
  ranked top-5 format for TruthfulQA.

This module only builds rater prompts (via ``..llm.prompts``), calls a rater
object exposing ``rate`` / ``rate_one`` / ``generate`` (a
``blackbox_client.GPT4Rater`` by default) and parses the free-form response back
into a candidate index.  Nothing here requests token probabilities, hidden
states or gradients from the black-box LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------------------
# Prompt-side imports (defensive: prompts.py is the contract for the rater format).
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import guard keeps this module importable in isolation
    from ..llm.prompts import (  # type: ignore
        BEST_ANSWER_OUTPUT_FORMAT,
        RANKING_OUTPUT_FORMAT,
        RATER_CRITERIA,
        build_ai_feedback_prompt,
        build_truthfulqa_ranking_prompt,
        format_candidate_block,
    )
except Exception:  # pragma: no cover
    BEST_ANSWER_OUTPUT_FORMAT = "Best Answer and Explanation:"
    RANKING_OUTPUT_FORMAT = "Ranked Answers:"
    RATER_CRITERIA = (
        "Coherency: The answer should present logical step-by-step reasoning that is "
        "coherent and directly related to the question.",
        "Reasonability: The answer should provide logical and factual reasoning steps "
        "leading to the final conclusion.",
        "Correctness: The final answer should be correct.",
        "Format: Each reasoning step should be in a separate sentence, ending with a "
        "definitive answer.",
    )

    def format_candidate_block(candidates, *, start_index: int = 1, include_reasoning: bool = True) -> str:  # type: ignore
        lines = []
        for offset, cand in enumerate(candidates):
            idx = start_index + offset
            text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
            lines.append(f"Answer {idx}: {text}\n")
        return "\n".join(lines)

    def build_ai_feedback_prompt(question, candidates, *, dataset=None, ranked=False, n_ranked=5, criteria=None, extra_instructions=None):  # type: ignore
        header = "Ranked Answers:" if ranked else "Best Answer and Explanation:"
        return (
            f"Question: {question}\n\n"
            + format_candidate_block(candidates)
            + "\nPlease follow the criteria below to select the best answer.\n"
            + "\n".join(f"- {c}" for c in (criteria or RATER_CRITERIA))
            + f"\nOutput in the format:\n{header}\n"
        )

    def build_truthfulqa_ranking_prompt(question, candidates, *, n_ranked: int = 5):  # type: ignore
        return build_ai_feedback_prompt(question, candidates, dataset="truthfulqa", ranked=True, n_ranked=n_ranked)


try:  # pragma: no cover
    from ..data.answer_extraction import extract_final_answer  # type: ignore
except Exception:  # pragma: no cover
    def extract_final_answer(text, answer_type=None, *, choices=None, n_choices=None):  # type: ignore
        return text


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Dataset for which the paper uses the ranked top-5 rater prompt (Appendix J).
RANKED_DATASETS: Tuple[str, ...] = ("truthfulqa", "truthful_qa", "truthful")

#: Number of candidates ranked by the rater for TruthfulQA (paper: top-5).
DEFAULT_N_RANKED = 5

#: Marker written by the rater before its verdict (Appendix J output format).
BEST_ANSWER_MARKER = "best answer and explanation"
RANKED_MARKER = "ranked answers"

#: Regular expressions considered, in order, when parsing "Answer <k>" verdicts.
_INDEX_PATTERNS: Tuple[str, ...] = (
    r"best\s+answer\s*(?:is|:|=|-)?\s*(?:answer\s*)?(?:#|no\.?|number)?\s*\(?(\d+)\)?",
    r"(?:answer|option|candidate)\s*(?:#|no\.?|number)?\s*(\d+)\s*(?:is\s*)?(?:the\s*)?best",
    r"best\s*(?:answer|option|candidate)\s*(?:#|no\.?|number)?\s*(\d+)",
    r"(?:i\s+)?(?:choose|select|pick)\s*(?:answer|option|candidate)?\s*(?:#|no\.?|number)?\s*\(?(\d+)\)?",
    r"answer\s*(?:#|no\.?|number)?\s*(\d+)",
    r"\(\s*(\d+)\s*\)",
    r"^\s*(\d+)[.,:;]?\s*$",
)

#: Pattern capturing explicit ranked instructions such as "Rank 1: Answer 3".
_RANK_LINE_PATTERN = re.compile(
    r"rank\s*(\d+)\s*[:.\-]?\s*(?:answer\s*)?(?:#|no\.?|number)?\s*(\d+)", re.IGNORECASE
)

_INT_PATTERN = re.compile(r"\d+")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class AIFeedbackConfig:
    """Configuration of the gpt-4 feedback rater (Appendix G / J, Section 4.1).

    Parameters
    ----------
    dataset:
        Dataset name; TruthfulQA switches to the ranked top-``n_ranked`` prompt.
    n_ranked:
        Number of candidates ranked per rater call in the ranked (TruthfulQA) mode.
        The paper ranks the top-5 answers.
    criteria:
        The four Appendix-G criteria handed to the rater.
    prompt_style:
        ``"auto"`` (ranked for TruthfulQA, best-answer otherwise), ``"best"`` or
        ``"ranked"``.
    temperature:
        Rater sampling temperature. The paper uses gpt-4 as a *simulator* of human
        preference; greedy decoding (0.0) is the sensible default.
    max_candidates:
        Optional cap on the number of candidates shown to the rater.
    tie_break:
        How to resolve several indices in the rater response: ``"first"`` (default),
        ``"last"``, ``"lowest"``, ``"highest"``.
    retries:
        Number of extra attempts if the response cannot be parsed.
    fallback:
        Offline behaviour when no rater is available or parsing fails:
        ``"majority"`` (most frequent extracted final answer, first candidate wins),
        ``"first"`` or ``"raise"``.  Not specified by the paper -> chosen default.
    deduplicate:
        Drop byte-identical duplicate candidate texts before rating.
    """

    dataset: Optional[str] = None
    n_ranked: int = DEFAULT_N_RANKED
    criteria: Sequence[str] = field(default_factory=lambda: tuple(RATER_CRITERIA))
    prompt_style: str = "auto"
    temperature: float = 0.0
    max_candidates: Optional[int] = None
    tie_break: str = "first"
    retries: int = 2
    fallback: str = "majority"
    deduplicate: bool = False
    answer_type: Optional[str] = None
    max_len: int = 512
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_ranked": self.n_ranked,
            "criteria": list(self.criteria),
            "prompt_style": self.prompt_style,
            "temperature": self.temperature,
            "max_candidates": self.max_candidates,
            "tie_break": self.tie_break,
            "retries": self.retries,
            "fallback": self.fallback,
            "deduplicate": self.deduplicate,
            "answer_type": self.answer_type,
            "max_len": self.max_len,
            "seed": self.seed,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AIFeedbackConfig":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = data.pop("extra", {}) or {}
        for key in list(data):
            if key not in known:
                extra[key] = data.pop(key)
        cfg = cls(**data)
        cfg.extra = dict(extra)
        return cfg

    @property
    def ranked(self) -> bool:
        """Whether the ranked (TruthfulQA top-5) rater prompt is used."""
        if self.prompt_style == "ranked":
            return True
        if self.prompt_style == "best":
            return False
        return _is_ranked_dataset(self.dataset)


@dataclass
class FeedbackSelection:
    """Result of one AI-feedback selection (SEL) call."""

    index: int = 0
    text: str = ""
    ranking: List[int] = field(default_factory=list)
    n_candidates: int = 0
    parsed: bool = False
    method: str = "ai_feedback"  # "ai_feedback" | "fallback_majority" | "fallback_first"
    dataset: Optional[str] = None
    criteria: List[str] = field(default_factory=list)
    explanation: str = ""
    raw_response: str = ""
    prompt: str = ""
    answer: Any = None
    attempts: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "ranking": list(self.ranking),
            "n_candidates": self.n_candidates,
            "parsed": self.parsed,
            "method": self.method,
            "dataset": self.dataset,
            "criteria": list(self.criteria),
            "explanation": self.explanation,
            "raw_response": self.raw_response,
            "answer": self.answer,
            "attempts": self.attempts,
            "meta": dict(self.meta),
        }

    def __int__(self) -> int:  # convenience: int(selection) -> index
        return int(self.index)


# --------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------


def _is_ranked_dataset(dataset: Optional[str]) -> bool:
    if not dataset:
        return False
    key = str(dataset).lower().replace("-", "").replace("_", "").replace(" ", "")
    return any(key == d.replace("_", "") for d in RANKED_DATASETS) or key.startswith("truthful")


def _section(response: str, marker: str) -> Optional[str]:
    """Return the text that follows ``marker`` (case-insensitive), else ``None``."""
    if not response:
        return None
    low = response.lower()
    pos = low.find(marker.lower())
    if pos < 0:
        return None
    return response[pos + len(marker):]


def _indices_in(text: str, n_candidates: int, *, allow_numbers: bool = False) -> List[int]:
    """All 1-based candidate indices mentioned in ``text`` (order preserved)."""
    out: List[int] = []
    for match in _INT_PATTERN.finditer(text or ""):
        value = int(match.group(0))
        if 1 <= value <= n_candidates:
            out.append(value)
    if not out and allow_numbers and n_candidates >= 1:
        pass
    return out


def _dedupe(values: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _choose(candidates: Sequence[int], policy: str) -> int:
    if not candidates:
        raise ValueError("empty candidate index list")
    if policy == "last":
        return int(candidates[-1])
    if policy == "lowest":
        return int(min(candidates))
    if policy == "highest":
        return int(max(candidates))
    return int(candidates[0])  # "first"


def parse_best_answer(
    response: Optional[str],
    n_candidates: int,
    *,
    tie_break: str = "first",
) -> Optional[int]:
    """Parse the rater's "Best Answer and Explanation:" verdict into a 1-based index.

    Handles the Appendix-J output format as well as the many free-form variants a
    chat rater produces ("Answer 3 is the best", "(2)", "Best answer: Answer 1",
    a bare "4", ...).  Returns ``None`` when no valid index is found.
    """
    if not response or n_candidates <= 0:
        return None

    verdict = _section(response, BEST_ANSWER_MARKER)
    if verdict is None:
        verdict = _section(response, "best answer")

    search_spaces = [verdict] if verdict else []
    search_spaces.append(response)

    for space in search_spaces:
        if not space:
            continue
        for pattern in _INDEX_PATTERNS:
            found = [
                int(m.group(1))
                for m in re.finditer(pattern, space, flags=re.IGNORECASE | re.MULTILINE)
                if 1 <= int(m.group(1)) <= n_candidates
            ]
            if found:
                return _choose(found, tie_break)

    # last resort: the first in-range integer mentioned
    loose = _indices_in(search_spaces[0] if search_spaces else response, n_candidates)
    if loose:
        return _choose(loose, tie_break)
    return None


def parse_ranked_answers(
    response: Optional[str],
    n_candidates: int,
    *,
    n_ranked: int = DEFAULT_N_RANKED,
) -> List[int]:
    """Parse the TruthfulQA ranked top-5 rater response into ordered 1-based indices."""
    if not response or n_candidates <= 0:
        return []

    body = _section(response, RANKED_MARKER)
    if body is None:
        body = _section(response, "ranked")
    if body is None:
        body = _section(response, BEST_ANSWER_MARKER) or response

    ranked: List[int] = []
    for match in _RANK_LINE_PATTERN.finditer(body):
        idx = int(match.group(2))
        if 1 <= idx <= n_candidates:
            ranked.append(idx)
    ranked = _dedupe(ranked)

    if not ranked:
        # fall back: "Answer k" mentions, then any in-range integers in order
        ranked = _dedupe(
            [
                int(m.group(1))
                for m in re.finditer(r"answer\s*(?:#|no\.?|number)?\s*(\d+)", body, flags=re.IGNORECASE)
                if 1 <= int(m.group(1)) <= n_candidates
            ]
        )
    if not ranked:
        ranked = _dedupe(_indices_in(body, n_candidates))
    if not ranked and body is not response:
        ranked = _dedupe(
            [
                int(m.group(1))
                for m in re.finditer(r"answer\s*(?:#|no\.?|number)?\s*(\d+)", response, flags=re.IGNORECASE)
                if 1 <= int(m.group(1)) <= n_candidates
            ]
        )

    return ranked[: max(1, n_ranked)]


# Backwards/forwards-compatible alias used by training/buffers.py.
def parse_feedback_index(
    response: Optional[str],
    n_candidates: int,
    *,
    ranked: bool = False,
    tie_break: str = "first",
    n_ranked: int = DEFAULT_N_RANKED,
) -> Optional[int]:
    """Parse a rater response into the 1-based index of the preferred candidate."""
    if ranked:
        order = parse_ranked_answers(response, n_candidates, n_ranked=n_ranked)
        if order:
            return int(order[0])
        return parse_best_answer(response, n_candidates, tie_break=tie_break)
    return parse_best_answer(response, n_candidates, tie_break=tie_break)


def extract_explanation(response: Optional[str]) -> str:
    """Text of the rater response after the verdict marker (for logging/auditing)."""
    if not response:
        return ""
    body = _section(response, BEST_ANSWER_MARKER) or _section(response, RANKED_MARKER)
    return (body or "").strip()


# --------------------------------------------------------------------------------------
# Offline fallback (not specified by the paper: used only when no rater is available)
# --------------------------------------------------------------------------------------


def heuristic_select(
    candidates: Sequence[Any],
    *,
    question: Optional[str] = None,
    dataset: Optional[str] = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    policy: str = "majority",
) -> FeedbackSelection:
    """Dependency-free stand-in for the gpt-4 rater.

    ``policy="majority"`` picks the candidate whose *extracted final answer* is the
    most frequent among the candidates (ties broken by first appearance), which is a
    reasonable surrogate for human preference when no rater is configured.
    ``policy="first"`` simply returns candidate 1.
    """
    texts = [_candidate_text(c) for c in candidates]
    n = len(texts)
    if n == 0:
        return FeedbackSelection(index=-1, text="", n_candidates=0, parsed=False, method="empty_fallback", dataset=dataset)

    if policy == "raise":
        raise ValueError("AI feedback parsing failed and fallback policy is 'raise'")

    if policy == "majority":
        keys: List[Any] = []
        for text in texts:
            try:
                keys.append(
                    extract_final_answer(text, answer_type, choices=choices) if answer_type else text.strip()
                )
            except Exception:
                keys.append(text.strip())
        counts: Dict[Any, int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        best_key = max(counts.items(), key=lambda kv: kv[1])[0]
        idx = keys.index(best_key)
        return FeedbackSelection(
            index=idx,
            text=texts[idx],
            ranking=[idx],
            n_candidates=n,
            parsed=False,
            method="fallback_majority",
            dataset=dataset,
        )

    return FeedbackSelection(
        index=0,
        text=texts[0],
        ranking=[0],
        n_candidates=n,
        parsed=False,
        method="fallback_first",
        dataset=dataset,
    )


def _candidate_text(candidate: Any) -> str:
    """Normalize a candidate (str / dict / object) into its text."""
    if candidate is None:
        return ""
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        for key in ("text", "answer", "completion", "content", "solution"):
            if key in candidate and candidate[key] is not None:
                return str(candidate[key])
        return str(candidate)
    for attr in ("text", "answer", "completion", "content"):
        if hasattr(candidate, attr):
            return str(getattr(candidate, attr))
    return str(candidate)


def _candidate_meta(candidate: Any) -> Dict[str, Any]:
    if isinstance(candidate, dict):
        return dict(candidate)
    meta: Dict[str, Any] = {}
    for attr in ("score", "answer", "index", "meta", "dataset", "uid"):
        if hasattr(candidate, attr):
            try:
                meta[attr] = getattr(candidate, attr)
            except Exception:  # pragma: no cover
                pass
    return meta


# --------------------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------------------


class AIFeedback:
    """gpt-4-simulated human preference used as ``SEL(.)`` in Section 3.4.

    Parameters
    ----------
    rater:
        Any object exposing ``rate_one(prompt, ...)``, ``rate(prompt, n=1, ...)``,
        ``generate_one(prompt, ...)`` or ``generate(prompt, n=1, ...)``.  Defaults to
        ``blackbox_client.build_rater()`` (a ``GPT4Rater``); pass ``None`` together with
        ``allow_build_rater=False`` to run fully offline (heuristic fallback).
    config:
        :class:`AIFeedbackConfig` or plain dict.
    dataset:
        Dataset name; used for prompt selection (TruthfulQA -> ranked top-5).
    prompt_builder:
        Optional callable ``(question, candidates) -> str`` overriding the
        Appendix-J prompt construction (useful for tests).
    """

    def __init__(
        self,
        rater: Optional[Any] = None,
        config: Optional[Union[AIFeedbackConfig, Dict[str, Any]]] = None,
        *,
        dataset: Optional[str] = None,
        prompt_builder: Optional[Callable[[str, List[str]], str]] = None,
        allow_build_rater: bool = True,
    ) -> None:
        if isinstance(config, AIFeedbackConfig):
            self.config = config
        else:
            self.config = AIFeedbackConfig.from_dict(config)
        if dataset is not None:
            self.config.dataset = dataset
        self.prompt_builder = prompt_builder

        if rater is None and allow_build_rater:
            try:  # pragma: no cover - requires Azure credentials
                from ..llm.blackbox_client import build_rater  # type: ignore

                rater = build_rater()
            except Exception as exc:  # pragma: no cover
                logger.warning("GPT-4 rater unavailable (%s); falling back to heuristics", exc)
                rater = None
        self.rater = rater

        self.n_calls = 0
        self.n_parsed = 0
        self.n_failed = 0
        self.n_fallback = 0

    # -- properties -------------------------------------------------------------------
    @property
    def ranked(self) -> bool:
        return bool(self.config.ranked)

    @property
    def criteria(self) -> List[str]:
        return list(self.config.criteria)

    # -- prompt construction -----------------------------------------------------------
    def build_prompt(self, question: str, candidates: Sequence[Any]) -> str:
        """Build the Appendix-J rater prompt for ``question``/``candidates``."""
        texts = [_candidate_text(c) for c in candidates]
        if self.prompt_builder is not None:
            return self.prompt_builder(question, texts)
        if self.ranked:
            return build_truthfulqa_ranking_prompt(question, texts, n_ranked=self.config.n_ranked)
        return build_ai_feedback_prompt(
            question,
            texts,
            dataset=self.config.dataset,
            ranked=False,
            criteria=self.config.criteria,
        )

    # -- rater plumbing ----------------------------------------------------------------
    def _call_rater(self, prompt: str, *, attempt: int = 0) -> str:
        """One rater call; accepts the several method names our clients expose."""
        if self.rater is None:
            raise RuntimeError("no AI-feedback rater configured")

        temperature = self.config.temperature
        if attempt > 0:  # nudge decoding when the previous response did not parse
            temperature = max(temperature, 0.3)
        max_len = int(self.config.max_len)
        kwargs: Dict[str, Any] = {"temperature": temperature, "max_len": max_len}
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed + attempt

        for name in ("rate_one", "generate_one"):
            fn = getattr(self.rater, name, None)
            if callable(fn):
                return str(fn(prompt, **kwargs))
        for name in ("rate", "generate"):
            fn = getattr(self.rater, name, None)
            if callable(fn):
                out = fn(prompt, n=1, **kwargs)
                if isinstance(out, str):
                    return out
                if isinstance(out, (list, tuple)) and out:
                    return str(out[0])
                texts = getattr(out, "texts", None)
                if texts:
                    return str(texts[0])
                return str(out)
        raise TypeError("rater object exposes none of rate_one/rate/generate_one/generate")

    # -- selection ---------------------------------------------------------------------
    def select(
        self,
        question: str,
        candidates: Sequence[Any],
        *,
        dataset: Optional[str] = None,
        answer_type: Optional[str] = None,
        choices: Optional[Sequence[str]] = None,
        return_result: bool = False,
    ) -> Union[int, FeedbackSelection]:
        """Run ``SEL`` on ``candidates`` and return the index of the preferred answer.

        Returns a 0-based index into ``candidates`` (use ``return_result=True`` for the
        full :class:`FeedbackSelection`, which also carries the ranking for TruthfulQA).
        """
        if dataset is not None:
            self.config.dataset = dataset
        if answer_type is not None:
            self.config.answer_type = answer_type

        texts = [_candidate_text(c) for c in candidates]
        if self.config.deduplicate:
            seen: Dict[str, int] = {}
            keep: List[int] = []
            for i, t in enumerate(texts):
                if t not in seen:
                    seen[t] = i
                    keep.append(i)
            texts = [texts[i] for i in keep]
        else:
            keep = list(range(len(texts)))

        if self.config.max_candidates is not None:
            texts = texts[: self.config.max_candidates]
            keep = keep[: self.config.max_candidates]

        n = len(texts)
        if n == 0:
            result = FeedbackSelection(
                index=-1, text="", n_candidates=0, parsed=False, method="empty", dataset=self.config.dataset
            )
            return result if return_result else -1

        if self.rater is None:
            fb = heuristic_select(
                texts,
                question=question,
                dataset=self.config.dataset,
                answer_type=self.config.answer_type,
                choices=choices,
                policy=self.config.fallback,
            )
            self.n_fallback += 1
            idx = keep[fb.index] if 0 <= fb.index < len(keep) else 0
            result = FeedbackSelection(
                index=idx,
                text=_candidate_text(candidates[idx]),
                ranking=[idx],
                n_candidates=len(candidates),
                parsed=False,
                method=fb.method,
                dataset=self.config.dataset,
                criteria=self.criteria,
            )
            return result if return_result else idx

        prompt = self.build_prompt(question, texts)
        last_response = ""
        attempts = max(1, int(self.config.retries) + 1)
        parsed_index: Optional[int] = None
        ranking: List[int] = []

        for attempt in range(attempts):
            try:
                last_response = self._call_rater(prompt, attempt=attempt)
            except Exception as exc:  # pragma: no cover - network/backend failure
                logger.warning("AI feedback rater call failed: %s", exc)
                last_response = ""
            self.n_calls += 1

            if self.ranked:
                ranking = parse_ranked_answers(last_response, n, n_ranked=self.config.n_ranked)
                parsed_index = ranking[0] if ranking else None
            else:
                parsed_index = parse_best_answer(last_response, n, tie_break=self.config.tie_break)
            if parsed_index is not None:
                break

        if parsed_index is not None:
            self.n_parsed += 1
            local_idx = int(parsed_index) - 1  # rater numbering is 1-based
            local_ranking = [int(i) - 1 for i in ranking] if ranking else [local_idx]
        else:
            self.n_failed += 1
            if self.config.fallback == "raise":
                raise ValueError(
                    f"could not parse AI feedback response: {last_response!r}"
                )
            fb = heuristic_select(
                texts,
                question=question,
                dataset=self.config.dataset,
                answer_type=self.config.answer_type,
                choices=choices,
                policy=self.config.fallback,
            )
            self.n_fallback += 1
            local_idx = fb.index
            local_ranking = [fb.index]

        idx = keep[local_idx] if 0 <= local_idx < len(keep) else keep[0]
        global_ranking = [keep[i] for i in local_ranking if 0 <= i < len(keep)]
        answer = None
        try:
            answer = extract_final_answer(
                _candidate_text(candidates[idx]), self.config.answer_type, choices=choices
            )
        except Exception:  # pragma: no cover
            answer = None

        result = FeedbackSelection(
            index=idx,
            text=_candidate_text(candidates[idx]),
            ranking=global_ranking or [idx],
            n_candidates=len(candidates),
            parsed=parsed_index is not None,
            method="ai_feedback" if parsed_index is not None else f"fallback_{self.config.fallback}",
            dataset=self.config.dataset,
            criteria=self.criteria,
            explanation=extract_explanation(last_response),
            raw_response=last_response,
            prompt=prompt,
            answer=answer,
            attempts=self.n_calls,
            meta={"local_index": local_idx, "candidate_meta": _candidate_meta(candidates[idx])},
        )
        return result if return_result else idx

    # ---------------------------------------------------------------------------------
    def select_from_generations(
        self,
        question: str,
        generations: Sequence[str],
        *,
        dataset: Optional[str] = None,
        answer_type: Optional[str] = None,
        choices: Optional[Sequence[str]] = None,
        return_result: bool = False,
    ) -> Union[int, FeedbackSelection]:
        """Convenience wrapper mirroring ``SEL({y_{i,j}}_{j=1..K})`` from Section 3.4."""
        return self.select(
            question,
            list(generations),
            dataset=dataset,
            answer_type=answer_type,
            choices=choices,
            return_result=return_result,
        )

    def select_batch(
        self,
        items: Sequence[Tuple[str, Sequence[Any]]],
        **kwargs: Any,
    ) -> List[FeedbackSelection]:
        """Run SEL over ``(question, candidates)`` pairs, returning full results."""
        out: List[FeedbackSelection] = []
        for question, candidates in items:
            result = self.select(question, candidates, return_result=True, **kwargs)
            assert isinstance(result, FeedbackSelection)
            out.append(result)
        return out

    def rank_candidates(
        self,
        question: str,
        candidates: Sequence[Any],
        **kwargs: Any,
    ) -> List[int]:
        """Return the full preference ranking (TruthfulQA top-5; single best otherwise)."""
        result = self.select(question, candidates, return_result=True, **kwargs)
        assert isinstance(result, FeedbackSelection)
        return list(result.ranking)

    def __call__(self, question: str, candidates: Sequence[Any], **kwargs: Any):
        return self.select(question, candidates, **kwargs)

    def stats(self) -> Dict[str, Any]:
        return {
            "calls": self.n_calls,
            "parsed": self.n_parsed,
            "failed": self.n_failed,
            "fallback": self.n_fallback,
            "rater": type(self.rater).__name__ if self.rater is not None else None,
            "ranked": self.ranked,
            "dataset": self.config.dataset,
        }


# --------------------------------------------------------------------------------------
# Functional API (used by training/buffers.py and the experiment scripts)
# --------------------------------------------------------------------------------------


def build_ai_feedback(
    dataset: Optional[str] = None,
    rater: Optional[Any] = None,
    config: Optional[Union[AIFeedbackConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> AIFeedback:
    """Factory: build an :class:`AIFeedback` selector for ``dataset``."""
    return AIFeedback(rater, config, dataset=dataset, **kwargs)


def select_by_ai_feedback(
    question: str,
    candidates: Sequence[Any],
    *,
    dataset: Optional[str] = None,
    rater: Optional[Any] = None,
    config: Optional[Union[AIFeedbackConfig, Dict[str, Any]]] = None,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[int, FeedbackSelection]:
    """One-shot ``SEL`` with AI feedback (Eq. 5 of Section 3.4)."""
    feedback = build_ai_feedback(dataset=dataset, rater=rater, config=config, **kwargs)
    return feedback.select(question, candidates, dataset=dataset, return_result=return_result)


def split_positives_negatives(
    candidates: Sequence[Any],
    positive_index: int,
    *,
    deduplicate: bool = True,
) -> Tuple[Any, List[Any]]:
    """Apply Eq. (6): positives = selected answer, negatives = all other candidates.

    Returns ``(positive, negatives)`` with the selected candidate removed from the
    negative list; identical texts (if any) are dropped when ``deduplicate=True``.
    """
    if positive_index < 0 or positive_index >= len(candidates):
        raise IndexError(f"positive_index {positive_index} out of range for {len(candidates)} candidates")
    positive = candidates[positive_index]
    pos_text = _candidate_text(positive)
    negatives: List[Any] = []
    for i, cand in enumerate(candidates):
        if i == positive_index:
            continue
        if deduplicate and _candidate_text(cand) == pos_text:
            continue
        negatives.append(cand)
    return positive, negatives


def criteria_summary() -> Dict[str, str]:
    """The four Appendix-G criteria as a name -> description mapping."""
    out: Dict[str, str] = {}
    for item in RATER_CRITERIA:
        text = str(item)
        if ":" in text:
            name, desc = text.split(":", 1)
            out[name.strip().lower()] = desc.strip()
        else:  # pragma: no cover
            out[text.strip().lower()] = text.strip()
    return out


# --------------------------------------------------------------------------------------
# Self test (no network): `python -m bbox_adapter.feedback.ai_feedback`
# --------------------------------------------------------------------------------------


def _self_test() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    # parsing ---------------------------------------------------------------
    checks["best_plain"] = parse_best_answer("Best Answer and Explanation: Answer 2 is best.", 3) == 2
    checks["best_bare"] = parse_best_answer("Best answer: 3", 4) == 3
    checks["best_paren"] = parse_best_answer("Reasoning...\nThe best is (1)", 3) == 1
    checks["best_bad"] = parse_best_answer("I cannot decide.", 3) is None
    checks["ranked"] = parse_ranked_answers(
        "Ranked Answers:\nRank 1: Answer 3\nRank 2: Answer 1\nRank 3: Answer 2", 3
    ) == [3, 1, 2]
    checks["ranked_marker"] = parse_ranked_answers(
        "Ranked Answers: Answer 2, then Answer 1", 2, n_ranked=5
    ) == [2, 1]
    checks["index_helper"] = parse_feedback_index("Best Answer: Answer 4", 5) == 4
    checks["index_ranked_helper"] = parse_feedback_index(
        "Ranked Answers:\nRank 1: Answer 2", 3, ranked=True
    ) == 2
    checks["criteria"] = len(criteria_summary()) == 4

    class _FakeRater:
        def __init__(self, response: str):
            self.response = response
            self.calls = 0

        def rate_one(self, prompt, **kwargs):
            self.calls += 1
            return self.response

    # best-answer selection ------------------------------------------------
    cands = ["wrong reasoning\n#### No.", "good reasoning\n#### Yes.", "odd\n#### Yes."]
    fb = AIFeedback(_FakeRater("Best Answer and Explanation: Answer 2 is best."), {"dataset": "strategyqa"},
                    allow_build_rater=False)
    sel = fb.select("Is x true?", cands, return_result=True, answer_type="yesno")
    assert isinstance(sel, FeedbackSelection)
    checks["select_index"] = sel.index == 1
    checks["select_parsed"] = sel.parsed and sel.method == "ai_feedback"
    checks["select_answer"] = sel.answer == "Yes"
    pos, negs = split_positives_negatives(cands, sel.index)
    checks["eq6"] = pos == cands[1] and len(negs) == 2

    # ranked (TruthfulQA) selection ---------------------------------------
    fb_r = AIFeedback(
        _FakeRater("Ranked Answers:\nRank 1: Answer 2\nRank 2: Answer 1"),
        {"dataset": "truthfulqa", "n_ranked": 5},
        allow_build_rater=False,
    )
    sel_r = fb_r.select("q", ["a", "b"], return_result=True)
    assert isinstance(sel_r, FeedbackSelection)
    checks["ranked_select"] = sel_r.index == 1 and sel_r.ranking[:2] == [1, 0]

    # parsing failure -> offline fallback ---------------------------------
    fb_f = AIFeedback(_FakeRater("mmm"), {"dataset": "gsm8k", "retries": 0}, allow_build_rater=False)
    idx_f = fb_f.select("q", ["#### 5", "#### 5", "#### 9"], answer_type="numeric")
    checks["fallback_majority"] = idx_f == 0 and fb_f.n_failed == 1

    # no rater at all ------------------------------------------------------
    fb_none = AIFeedback(None, {"dataset": "gsm8k"}, allow_build_rater=False)
    checks["offline"] = fb_none.select("q", ["#### 1", "#### 2"]) == 0
    checks["payload_text_only"] = "logprob" not in fb.build_prompt("q", cands).lower()

    return checks


if __name__ == "__main__":  # pragma: no cover
    import json

    results = _self_test()
    print(json.dumps(results, indent=2))
    assert all(results.values()), f"self-test failures: {[k for k, v in results.items() if not v]}"
    print("ai_feedback self-test: OK")
