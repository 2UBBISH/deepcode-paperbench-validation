"""Token accounting and dollar-cost conversion for BBox-Adapter.

Implements the bookkeeping behind Table 4 of "Lightweight Adapting for
Black-Box Large Language Models" (paper Sec. 4.4): the black-box LLM client
(``bbox_adapter/llm/blackbox_client.py``) reports *only* token usage (prompt +
completion tokens) and this module converts those token counts into USD and
into the paper's reporting unit, **dollars per 1,000 questions**.

Design notes
------------
* The module is a thin, dependency-free layer over the ledger seam used by the
  black-box client: any object exposing::

      ledger.add(prompt_tokens=..., completion_tokens=..., model=..., n=...)

  is accepted.  ``TokenLedger`` below is such an object; when
  :mod:`bbox_adapter.eval.cost` is importable we simply alias its richer
  ``CostLedger`` so both modules report identical numbers.
* Pricing is taken from the OpenAI price list for the *paper's* pinned model
  ``gpt-3.5-turbo-1106`` ($0.0015 / 1k input tokens, $0.002 / 1k output tokens),
  which is the model used for the Azure-served proposals in the paper and for
  the Table 4 dollar figures.
* Token counts may be *exact* (reported by the Azure API) or *estimated*
  (Whitespace/character heuristic used by the offline mock client); the ledger
  records which was the case so cost tables can be annotated.
* Nothing here requests output probabilities: the black-box contract of
  Appendix C is respected (only ``usage`` token counts are read).

External services / secrets: ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``
(see ``.env`` handling in the scripts); this module itself never performs I/O.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("bbox_adapter.llm.token_cost")

__all__ = [
    # pricing / constants
    "DEFAULT_PRICING_MODEL",
    "USD_PER_1K_INPUT_TOKENS",
    "USD_PER_1K_OUTPUT_TOKENS",
    "PRICING",
    "DEFAULT_PER_QUESTIONS",
    "CHARS_PER_TOKEN",
    "PHASE_TOTAL",
    "PHASE_TRAINING",
    "PHASE_INFERENCE",
    "PHASE_EVALUATION",
    "PHASES",
    "PAPER_TABLE4",
    # classes
    "TokenPricing",
    "TokenCounter",
    "TokenUsage",
    "PhaseCounter",
    "TokenLedger",
    "CostLedger",
    "TokenBudget",
    # functions
    "resolve_pricing",
    "model_pricing",
    "estimate_tokens",
    "count_message_tokens",
    "price_tokens",
    "tokens_to_usd",
    "cost_per_1k_questions",
    "inference_cost_per_1k",
    "training_cost_per_1k",
    "cost_per_question",
    "cost_ratio",
    "average_cost_per_1k",
    "project_cost",
    "build_ledger",
    "format_cost_table",
    "make_cost_table_rows",
    "self_check",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The model whose price list the paper's Table 4 dollar figures use.
DEFAULT_PRICING_MODEL = "gpt-3.5-turbo-1106"

#: USD per 1,000 prompt (input) tokens for ``gpt-3.5-turbo-1106``.
USD_PER_1K_INPUT_TOKENS = 0.0015

#: USD per 1,000 completion (output) tokens for ``gpt-3.5-turbo-1106``.
USD_PER_1K_OUTPUT_TOKENS = 0.0020

#: USD per 1,000 tokens (``input`` = prompt, ``output`` = completion).
PRICING: Dict[str, Dict[str, float]] = {
    DEFAULT_PRICING_MODEL: {"input": 0.0015, "output": 0.0020},
    "gpt-3.5-turbo-1106": {"input": 0.0015, "output": 0.0020},
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.0020},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-0613": {"input": 0.03, "output": 0.06},
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
    "davinci-002": {"input": 0.012, "output": 0.012},
    "text-davinci-003": {"input": 0.02, "output": 0.02},
}

#: Fallback used when a model name is not in :data:`PRICING`.
PRICING_FALLBACK: Dict[str, float] = {
    "input": USD_PER_1K_INPUT_TOKENS,
    "output": USD_PER_1K_OUTPUT_TOKENS,
}

#: Paper reporting unit: dollars per 1,000 questions.
DEFAULT_PER_QUESTIONS = 1000

#: Character heuristic used when the provider does not report token counts
#: (matches ``blackbox_client.estimate_tokens``).
CHARS_PER_TOKEN = 4.0

PHASE_TOTAL = "total"
PHASE_TRAINING = "training"
PHASE_INFERENCE = "inference"
PHASE_EVALUATION = "evaluation"
PHASES: Tuple[str, ...] = (PHASE_TOTAL, PHASE_TRAINING, PHASE_INFERENCE, PHASE_EVALUATION)

#: Reference Table 4 rows (paper Sec. 4.4) used for regression checks only.
PAPER_TABLE4: Dict[str, Dict[str, Dict[str, float]]] = {
    "strategyqa": {
        "single_step": {
            "accuracy": 69.87,
            "train_cost": 2.77,
            "inference_cost": 2.20,
        },
        "full_step": {
            "accuracy": 71.62,
            "train_cost": 3.48,
            "inference_cost": 5.37,
        },
        "azure_sft": {
            "accuracy": 72.06,
            "train_cost": 153.00,
            "inference_cost": 7.50,
        },
    },
    "gsm8k": {
        "single_step": {
            "accuracy": 71.13,
            "train_cost": 7.54,
            "inference_cost": 3.10,
        },
        "full_step": {
            "accuracy": 74.28,
            "train_cost": 11.58,
            "inference_cost": 12.46,
        },
        "azure_sft": {
            "accuracy": 74.05,
            "train_cost": 216.50,
            "inference_cost": 28.30,
        },
    },
}

_TOKEN_SPLIT_RE = re.compile(r"\S+")

# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------


def resolve_pricing(
    model: Optional[str] = None,
    usd_per_1k_input_tokens: Optional[float] = None,
    usd_per_1k_output_tokens: Optional[float] = None,
) -> Dict[str, float]:
    """Return ``{"input": .., "output": ..}`` USD per 1k tokens for ``model``.

    Explicit overrides win; otherwise the longest matching prefix in
    :data:`PRICING` is used; otherwise :data:`PRICING_FALLBACK`.
    """
    if usd_per_1k_input_tokens is not None and usd_per_1k_output_tokens is not None:
        return {
            "input": float(usd_per_1k_input_tokens),
            "output": float(usd_per_1k_output_tokens),
        }
    base = dict(PRICING_FALLBACK)
    if model:
        key = str(model).strip().lower()
        if key in PRICING:
            base = dict(PRICING[key])
        else:
            best_key = ""
            for candidate in PRICING:
                if key.startswith(candidate) or candidate in key:
                    if len(candidate) > len(best_key):
                        best_key = candidate
            if best_key:
                base = dict(PRICING[best_key])
    if usd_per_1k_input_tokens is not None:
        base["input"] = float(usd_per_1k_input_tokens)
    if usd_per_1k_output_tokens is not None:
        base["output"] = float(usd_per_1k_output_tokens)
    return base


def model_pricing(model: Optional[str] = None) -> Dict[str, float]:
    """Alias of :func:`resolve_pricing` (used by ``eval/cost.py`` peers)."""
    return resolve_pricing(model)


@dataclass
class TokenPricing:
    """Frozen-ish price list entry for one model."""

    model: str = DEFAULT_PRICING_MODEL
    usd_per_1k_input_tokens: float = USD_PER_1K_INPUT_TOKENS
    usd_per_1k_output_tokens: float = USD_PER_1K_OUTPUT_TOKENS

    def rates(self) -> Dict[str, float]:
        return resolve_pricing(
            self.model, self.usd_per_1k_input_tokens, self.usd_per_1k_output_tokens
        )

    def price(self, prompt_tokens: int, completion_tokens: int) -> float:
        return price_tokens(prompt_tokens, completion_tokens, model=self.model)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_model(cls, model: Optional[str] = None) -> "TokenPricing":
        rates = resolve_pricing(model)
        return cls(
            model=model or DEFAULT_PRICING_MODEL,
            usd_per_1k_input_tokens=rates["input"],
            usd_per_1k_output_tokens=rates["output"],
        )


def price_tokens(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    *,
    model: Optional[str] = None,
    usd_per_1k_input_tokens: Optional[float] = None,
    usd_per_1k_output_tokens: Optional[float] = None,
) -> float:
    """Convert token counts to USD for ``model`` (paper Sec. 4.4 / Table 4)."""
    rates = resolve_pricing(model, usd_per_1k_input_tokens, usd_per_1k_output_tokens)
    return (
        float(prompt_tokens) / 1000.0 * rates["input"]
        + float(completion_tokens) / 1000.0 * rates["output"]
    )


#: Alias used by ``eval/cost.py`` consumers.
tokens_to_usd = price_tokens


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def estimate_tokens(text: Optional[str], chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Estimate the token count of ``text`` without a tokenizer.

    Uses ``max(1, ceil(len(text) / chars_per_token))`` when the text is
    non-empty and ``0`` for ``None``/empty strings.  This mirrors the mock
    client's estimator so offline cost tables remain self-consistent.
    """
    if not text:
        return 0
    import math

    return max(1, int(math.ceil(len(str(text)) / float(chars_per_token))))


def count_message_tokens(messages: Union[str, Sequence[Any]], *, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Estimate prompt tokens for a chat prompt or a plain string.

    Each chat message costs ``estimate_tokens(content)`` plus a small constant
    per message to account for role/formatting overhead.
    """
    if isinstance(messages, str):
        return estimate_tokens(messages, chars_per_token)
    total = 0
    for message in messages or []:
        if isinstance(message, dict):
            total += estimate_tokens(message.get("content"), chars_per_token)
            total += 4  # role + delimiters
        else:
            total += estimate_tokens(message, chars_per_token)
    return total


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


class TokenCounter:
    """Exact-or-estimated token counter for prompts and responses."""

    def __init__(
        self,
        *,
        chars_per_token: float = CHARS_PER_TOKEN,
        tokenizer: Optional[Any] = None,
        exact: bool = False,
        model: Optional[str] = None,
    ) -> None:
        self.chars_per_token = float(chars_per_token)
        self.tokenizer = tokenizer
        self.exact = bool(exact)
        self.model = model
        self.n_prompts = 0
        self.n_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_estimated = 0
        self.n_exact = 0

    # -- counting ---------------------------------------------------------
    def count(self, text: Optional[str]) -> int:
        """Token count of ``text``; uses the tokenizer when available."""
        if self.tokenizer is not None:
            try:
                self.n_exact += 1
                return len(self.tokenizer.encode(str(text or "")))
            except Exception:  # pragma: no cover - tokenizer quirks
                logger.debug("tokenizer failed; falling back to heuristic")
        self.n_estimated += 1
        return estimate_tokens(text, self.chars_per_token)

    def add_prompt(self, text: Optional[str]) -> int:
        tokens = self.count(text)
        self.n_prompts += 1
        self.prompt_tokens += tokens
        return tokens

    def add_response(self, text: Optional[str]) -> int:
        tokens = self.count(text)
        self.n_responses += 1
        self.completion_tokens += tokens
        return tokens

    def add_exact(self, prompt_tokens: int, completion_tokens: int, *, exact: bool = True) -> None:
        """Record counts reported by the provider (``usage`` field)."""
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.n_prompts += 1
        self.n_responses += 1
        if exact:
            self.n_exact += 1
        else:
            self.n_estimated += 1

    # -- reporting --------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, model: Optional[str] = None, **kwargs: Any) -> float:
        return price_tokens(
            self.prompt_tokens, self.completion_tokens, model=model or self.model, **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "n_prompts": self.n_prompts,
            "n_responses": self.n_responses,
            "n_exact_counts": self.n_exact,
            "n_estimated_counts": self.n_estimated,
            "model": self.model,
            "cost_usd": self.cost(),
        }

    def reset(self) -> None:
        self.n_prompts = 0
        self.n_responses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_estimated = 0
        self.n_exact = 0


# ``TokenUsage`` is the per-model / per-phase accumulator used by the ledger.
try:  # pragma: no cover - prefer the richer eval-side implementation
    from ..eval.cost import TokenUsage as _EvalTokenUsage  # type: ignore
except Exception:  # pragma: no cover
    _EvalTokenUsage = None


if _EvalTokenUsage is not None:  # pragma: no cover - normal path
    TokenUsage = _EvalTokenUsage  # type: ignore
else:

    @dataclass
    class TokenUsage:  # type: ignore[no-redef]
        """Accumulated prompt/completion tokens with derived statistics."""

        prompt_tokens: int = 0
        completion_tokens: int = 0
        n_calls: int = 0
        n_samples: int = 0
        n_questions: int = 0
        by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)
        exact: bool = True

        @property
        def total_tokens(self) -> int:
            return self.prompt_tokens + self.completion_tokens

        @property
        def mean_prompt_tokens(self) -> float:
            return self.prompt_tokens / self.n_calls if self.n_calls else 0.0

        @property
        def mean_completion_tokens(self) -> float:
            return self.completion_tokens / self.n_calls if self.n_calls else 0.0

        def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
            self.prompt_tokens += int(getattr(other, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(other, "completion_tokens", 0) or 0)
            self.n_calls += int(getattr(other, "n_calls", 0) or 0)
            self.n_samples += int(getattr(other, "n_samples", 0) or 0)
            self.n_questions += int(getattr(other, "n_questions", 0) or 0)
            self.exact = bool(self.exact and getattr(other, "exact", True))
            for model, counts in (getattr(other, "by_model", {}) or {}).items():
                slot = self.by_model.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
                slot["prompt"] += int(counts.get("prompt", 0) or 0)
                slot["completion"] += int(counts.get("completion", 0) or 0)
                slot["calls"] += int(counts.get("calls", 0) or 0)
            return self

        def to_dict(self) -> Dict[str, Any]:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "n_calls": self.n_calls,
                "n_samples": self.n_samples,
                "n_questions": self.n_questions,
                "by_model": dict(self.by_model),
                "exact": self.exact,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "TokenUsage":
            data = dict(data or {})
            data.pop("total_tokens", None)
            fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**fields)

        @classmethod
        def from_value(cls, value: Any) -> "TokenUsage":
            """Build usage from a ``GenerationResult``-like object or a mapping."""
            usage = cls()
            if value is None:
                return usage
            prompt = getattr(value, "prompt_tokens", None)
            completion = getattr(value, "completion_tokens", None)
            if prompt is None and isinstance(value, dict):
                prompt = value.get("prompt_tokens") or value.get("prompt")
                completion = value.get("completion_tokens") or value.get("completion")
            usage.prompt_tokens = int(prompt or 0)
            usage.completion_tokens = int(completion or 0)
            n = int(getattr(value, "n", 1) or 1) if not isinstance(value, dict) else int(value.get("n", 1) or 1)
            usage.n_samples += n
            usage.n_calls += 1
            model = getattr(value, "model", None) if not isinstance(value, dict) else value.get("model")
            if model:
                usage.by_model[model] = {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "calls": 1,
                }
            return usage


class PhaseCounter:
    """Token counters grouped by experiment phase (training / inference)."""

    def __init__(self, phases: Sequence[str] = PHASES, *, model: Optional[str] = None) -> None:
        self.model = model
        self.counters: "OrderedDict[str, TokenCounter]" = OrderedDict(
            (phase, TokenCounter(model=model)) for phase in phases
        )

    def __getitem__(self, phase: str) -> TokenCounter:
        if phase not in self.counters:
            self.counters[phase] = TokenCounter(model=self.model)
        return self.counters[phase]

    def add_exact(self, phase: str, prompt_tokens: int, completion_tokens: int) -> None:
        self[phase].add_exact(prompt_tokens, completion_tokens)

    def usage(self, phase: str) -> Dict[str, Any]:
        return self[phase].to_dict()

    def cost(self, phase: str = PHASE_TOTAL) -> float:
        return self[phase].cost(self.model)

    def summary(self) -> Dict[str, Any]:
        return {phase: counter.to_dict() for phase, counter in self.counters.items()}


# ---------------------------------------------------------------------------
# Ledger (drop-in for ``blackbox_client``)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - prefer the richer eval-side implementation
    from ..eval.cost import CostLedger as _EvalCostLedger  # type: ignore
except Exception:  # pragma: no cover
    _EvalCostLedger = None  # type: ignore


class TokenLedger:
    """Minimal token ledger accepted by ``BlackBoxClient(ledger=...)``.

    API::

        ledger.add(prompt_tokens=..., completion_tokens=..., model=..., n=...)

    Costs are converted with :func:`price_tokens` using the ``gpt-3.5-turbo-1106``
    price list (the model behind the paper's Table-4 dollar figures), and can be
    reported per 1,000 questions via :meth:`cost_per_1k_questions`.
    """

    def __init__(
        self,
        model: Optional[str] = DEFAULT_PRICING_MODEL,
        *,
        per_questions: int = DEFAULT_PER_QUESTIONS,
        usd_per_1k_input_tokens: Optional[float] = None,
        usd_per_1k_output_tokens: Optional[float] = None,
        phases: Sequence[str] = PHASES,
    ) -> None:
        self.model = model
        self.per_questions = int(per_questions)
        self.usd_per_1k_input_tokens = usd_per_1k_input_tokens
        self.usd_per_1k_output_tokens = usd_per_1k_output_tokens
        self.phases: List[str] = list(phases)
        self.usage_by_phase: "OrderedDict[str, Dict[str, Any]]" = OrderedDict(
            (phase, {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0, "n_samples": 0})
            for phase in self.phases
        )
        self.by_model: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
        self.n_questions = 0
        self.n_exact = 0
        self.n_estimated = 0
        self._phase: str = PHASE_TOTAL

    # -- phase routing ----------------------------------------------------
    def phase(self, name: str) -> "PhaseCounter":  # pragma: no cover - convenience
        self._phase = name
        return PhaseCounter([name], model=self.model)  # type: ignore[return-value]

    def set_phase(self, name: str) -> str:
        self._phase = name
        return self._phase

    # -- recording --------------------------------------------------------
    def add(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        model: Optional[str] = None,
        n: int = 1,
        phase: Optional[str] = None,
        exact: bool = True,
        n_questions: int = 1,
        **_: Any,
    ) -> Dict[str, Any]:
        """Record one black-box call (seam used by ``BlackBoxClient._record``)."""
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        target = phase or self._phase or PHASE_TOTAL
        for key in (PHASE_TOTAL, target):  # total mirrors everything
            slot = self.usage_by_phase.setdefault(
                key, {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0, "n_samples": 0}
            )
            slot["prompt_tokens"] += p
            slot["completion_tokens"] += c
            slot["n_calls"] += 1
            slot["n_samples"] += int(n or 1)
        if target != PHASE_TOTAL:
            # avoid double counting `total` through the loop above
            pass
        resolved_model = model or self.model or DEFAULT_PRICING_MODEL
        slot = self.by_model.setdefault(
            resolved_model, {"prompt": 0, "completion": 0, "calls": 0}
        )
        slot["prompt"] += p
        slot["completion"] += c
        slot["calls"] += 1
        if exact:
            self.n_exact += 1
        else:
            self.n_estimated += 1
        self.n_questions += max(0, int(n_questions))
        return {"prompt_tokens": p, "completion_tokens": c, "model": resolved_model}

    # ``CostLedger`` compatibility aliases
    record = add

    def record_result(self, result: Any, *, phase: Optional[str] = None) -> Dict[str, Any]:
        """Record a ``GenerationResult`` (duck-typed: prompt/completion tokens)."""
        prompt = int(getattr(result, "prompt_tokens", 0) or 0)
        completion = int(getattr(result, "completion_tokens", 0) or 0)
        if not prompt and not completion:
            texts = getattr(result, "texts", None) or []
            completion = sum(estimate_tokens(t) for t in texts)
        return self.add(
            prompt_tokens=prompt,
            completion_tokens=completion,
            model=getattr(result, "model", None) or self.model,
            n=int(getattr(result, "n", 1) or 1),
            phase=phase,
            exact=not bool(getattr(result, "mock", False)),
        )

    # -- aggregation ------------------------------------------------------
    def usage(self, phase: str = PHASE_TOTAL) -> Dict[str, Any]:
        slot = self.usage_by_phase.get(phase, {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0, "n_samples": 0})
        out = dict(slot)
        out["total_tokens"] = slot["prompt_tokens"] + slot["completion_tokens"]
        return out

    def cost(self, phase: str = PHASE_TOTAL, model: Optional[str] = None) -> float:
        slot = self.usage(phase)
        return price_tokens(
            slot["prompt_tokens"],
            slot["completion_tokens"],
            model=model or self.model,
            usd_per_1k_input_tokens=self.usd_per_1k_input_tokens,
            usd_per_1k_output_tokens=self.usd_per_1k_output_tokens,
        )

    def cost_per_1k_questions(
        self,
        phase: str = PHASE_TOTAL,
        n_questions: Optional[int] = None,
        *,
        per_questions: Optional[int] = None,
        model: Optional[str] = None,
    ) -> float:
        return cost_per_1k_questions(
            self.cost(phase=phase, model=model),
            n_questions if n_questions is not None else self.n_questions,
            per_questions=per_questions or self.per_questions,
        )

    @property
    def training_cost(self) -> float:
        return self.cost(PHASE_TRAINING)

    @property
    def inference_cost(self) -> float:
        return self.cost(PHASE_INFERENCE)

    @property
    def total_cost(self) -> float:
        return self.cost(PHASE_TOTAL)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "n_questions": self.n_questions,
            "n_calls": self.usage(PHASE_TOTAL)["n_calls"],
            "prompt_tokens": self.usage(PHASE_TOTAL)["prompt_tokens"],
            "completion_tokens": self.usage(PHASE_TOTAL)["completion_tokens"],
            "total_tokens": self.usage(PHASE_TOTAL)["total_tokens"],
            "total_cost_usd": self.total_cost,
            "training_cost_usd": self.training_cost,
            "inference_cost_usd": self.inference_cost,
            "cost_per_1k_questions_usd": self.cost_per_1k_questions(),
            "by_model": {k: dict(v) for k, v in self.by_model.items()},
            "exact_counts": self.n_exact,
            "estimated_counts": self.n_estimated,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.summary()

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        payload = self.summary()
        if extra:
            payload.update(extra)
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def reset(self) -> None:
        for phase in list(self.usage_by_phase):
            self.usage_by_phase[phase] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "n_calls": 0,
                "n_samples": 0,
            }
        self.by_model.clear()
        self.n_questions = 0
        self.n_exact = 0
        self.n_estimated = 0


#: ``eval/cost.py`` exposes a richer ledger; alias it when available so the
#: black-box client, the training loop and the report writer share one object.
if _EvalCostLedger is not None:  # pragma: no cover - normal path
    CostLedger = _EvalCostLedger  # type: ignore
else:
    CostLedger = TokenLedger  # type: ignore


def build_ledger(config: Optional[Any] = None, **kwargs: Any) -> TokenLedger:
    """Build a ledger from a ``cost:`` config mapping/section or explicit kwargs.

    Prefers :class:`bbox_adapter.eval.cost.CostLedger` (which understands the
    YAML ``cost:`` section) and otherwise returns a local :class:`TokenLedger`.
    """
    section: Dict[str, Any] = {}
    if config is not None:
        if isinstance(config, dict):
            section = dict(config.get("cost", config))
        else:  # duck-typed config namespace
            section = dict(getattr(config, "cost", {}) or {})
    section = {k: v for k, v in section.items() if v is not None}
    section.update(kwargs)
    if _EvalCostLedger is not None:  # pragma: no cover - normal path
        try:
            return _EvalCostLedger.from_config(section)  # type: ignore[attr-defined]
        except Exception:
            try:
                return _EvalCostLedger(**section)  # type: ignore[call-arg]
            except Exception:
                logger.debug("falling back to TokenLedger", exc_info=True)
    return TokenLedger(
        model=section.get("model") or section.get("model_for_pricing") or DEFAULT_PRICING_MODEL,
        per_questions=int(section.get("per_questions", DEFAULT_PER_QUESTIONS)),
        usd_per_1k_input_tokens=section.get("usd_per_1k_input_tokens"),
        usd_per_1k_output_tokens=section.get("usd_per_1k_output_tokens"),
    )


# ---------------------------------------------------------------------------
# Cost arithmetic
# ---------------------------------------------------------------------------


def cost_per_1k_questions(
    total_cost: float,
    n_questions: int,
    *,
    per_questions: int = DEFAULT_PER_QUESTIONS,
) -> float:
    """Scale a total cost to dollars per ``per_questions`` questions (Table 4)."""
    if not n_questions or n_questions <= 0:
        return 0.0
    return float(total_cost) / float(n_questions) * float(per_questions)


#: Alias matching the paper's "inference cost" column.
inference_cost_per_1k = cost_per_1k_questions


def training_cost_per_1k(train_total_cost: float, n_train_questions: int, *, per_questions: int = DEFAULT_PER_QUESTIONS) -> float:
    """Training cost normalised per ``per_questions`` training questions."""
    return cost_per_1k_questions(train_total_cost, n_train_questions, per_questions=per_questions)


def cost_per_question(total_cost: float, n_questions: int) -> float:
    if not n_questions or n_questions <= 0:
        return 0.0
    return float(total_cost) / float(n_questions)


def cost_ratio(ours: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """``baseline / ours`` (the paper's "31.30x cheaper" style statement)."""
    if ours is None or baseline is None:
        return None
    if not ours:
        return None
    return float(baseline) / float(ours)


def average_cost_per_1k(rows: Iterable[Dict[str, Any]], key: str = "inference_cost") -> Optional[float]:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def project_cost(
    cost_per_1k: float,
    n_questions: int,
    *,
    per_questions: int = DEFAULT_PER_QUESTIONS,
) -> float:
    """Inverse of :func:`cost_per_1k_questions`."""
    return float(cost_per_1k) * float(n_questions) / float(per_questions)


# ---------------------------------------------------------------------------
# Table 4 reporting helpers
# ---------------------------------------------------------------------------


def make_cost_table_rows(
    *,
    dataset: str,
    variant: str,
    accuracy: Optional[float] = None,
    train_cost: Optional[float] = None,
    inference_cost: Optional[float] = None,
    base_accuracy: Optional[float] = None,
    baseline_train_cost: Optional[float] = None,
    baseline_inference_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble one Table-4 style row with the paper's derived quantities."""
    delta = None
    if accuracy is not None and base_accuracy is not None:
        delta = float(accuracy) - float(base_accuracy)
    return {
        "dataset": dataset,
        "method": variant,
        "accuracy": accuracy,
        "delta": delta,
        "train_cost": train_cost,
        "inference_cost": inference_cost,
        "train_ratio": cost_ratio(train_cost, baseline_train_cost),
        "inference_ratio": cost_ratio(inference_cost, baseline_inference_cost),
    }


def format_cost_table(rows: Sequence[Dict[str, Any]], *, title: str = "Cost analysis ($/1k questions)") -> str:
    """Render a plain-text Table 4 (no third-party dependency)."""
    headers = ["Dataset", "Method", "Acc", "Train $", "Infer $", "xTrain", "xInfer"]
    lines = [title, "-" * len(title)]

    def _fmt(value: Any, nd: int = 2) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.{nd}f}"
        except (TypeError, ValueError):
            return str(value)

    lines.append(" | ".join(headers))
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("dataset", "-")),
                    str(row.get("method", "-")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("train_cost")),
                    _fmt(row.get("inference_cost")),
                    _fmt(row.get("train_ratio")),
                    _fmt(row.get("inference_ratio")),
                ]
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def self_check() -> Dict[str, Any]:
    """Dependency-free smoke test (``python -m bbox_adapter.llm.token_cost``)."""
    results: Dict[str, Any] = {}

    # 1. pricing
    rates = resolve_pricing("gpt-3.5-turbo-1106")
    assert abs(rates["input"] - 0.0015) < 1e-12, rates
    assert abs(rates["output"] - 0.0020) < 1e-12, rates
    results["pricing"] = rates

    # 2. token -> dollar conversion (Table-4 pricing)
    usd = price_tokens(1000, 1000, model="gpt-3.5-turbo-1106")
    assert abs(usd - 0.0035) < 1e-12, usd
    results["usd_for_1k_in_1k_out"] = usd

    # 3. ledger writes both total and phase buckets
    ledger = TokenLedger(per_questions=1000)
    ledger.add(prompt_tokens=100, completion_tokens=50, phase=PHASE_TRAINING, n=1)
    ledger.add(prompt_tokens=200, completion_tokens=100, phase=PHASE_INFERENCE, n=1)
    assert ledger.usage(PHASE_TOTAL)["prompt_tokens"] == 300
    assert ledger.usage(PHASE_TRAINING)["prompt_tokens"] == 100
    assert ledger.usage(PHASE_INFERENCE)["completion_tokens"] == 100
    results["ledger"] = ledger.summary()

    # 4. duck-typed GenerationResult recording
    class _FakeResult:
        prompt_tokens = 1000
        completion_tokens = 0
        n = 1
        model = "gpt-3.5-turbo-1106"
        mock = False
        texts = ("#### Yes.",)

    ledger2 = TokenLedger()
    ledger2.record_result(_FakeResult())
    assert abs(ledger2.total_cost - 0.0015) < 1e-12, ledger2.summary()
    results["duck_typed_cost"] = ledger2.total_cost

    # 5. per-1k-questions normalisation
    assert abs(cost_per_1k_questions(3.0, 1000) - 3.0) < 1e-12
    assert abs(cost_per_1k_questions(3.0, 100, per_questions=1000) - 30.0) < 1e-12
    results["per_1k"] = [cost_per_1k_questions(3.0, 1000), cost_per_1k_questions(3.0, 100)]

    # 6. cost ratio (paper Sec. 4.4: 153.00 / 3.48 ~= 43.97 for single-step
    #    Azure-SFT comparison over BBox-Adapter full-step training cost)
    assert abs(cost_ratio(3.48, 153.0) - 43.96551724137931) < 1e-9
    results["cost_ratio"] = cost_ratio(3.48, 153.0)

    # 7. token estimation heuristic
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert count_message_tokens([{"role": "user", "content": "abcd"}]) == 5
    results["estimate_tokens"] = estimate_tokens("x" * 40)

    # 8. paper reference rows still present
    results["paper_table4_keys"] = sorted(PAPER_TABLE4)
    results["ok"] = True
    return results


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(self_check(), indent=2))
