"""Cost accounting for BBox-Adapter (paper Section 4.4, Table 4).

The paper reports, for every method, the performance (accuracy %) together with

* **Training Cost ($)** -- the total dollar amount spent while adapting, and
* **Inference Cost ($) / 1k Q** -- dollars needed to answer one thousand questions.

Table 4 (:math:`\\S4.4`, reproduced verbatim below) is the reference target:

============  ==================  ====================  ==================  ====================
              StrategyQA                                GSM8K
-------------  ------------------  --------------------  ------------------  --------------------
Method         Acc.(%)  Train($)   Infer($)/1kQ          Acc.(%)  Train($)   Infer($)/1kQ
============  ==================  ====================  ==================  ====================
gpt-3.5-turbo  66.59     -         0.41                  67.51    -          1.22
Azure-SFT      76.86     153.00    7.50                  69.94    216.50     28.30
BBox (single)  69.87     2.77      2.20                  71.13    7.54       3.10
BBox (full)    71.62     3.48      5.37                  74.28    11.58      12.46
============  ==================  ====================  ==================  ====================

Quoting the paper: *"the inference cost was calculated by aggregating the total token
consumption statistics provided by Azure API and subsequently applying the cost per
token (gpt-3.5-turbo-1106) as specified in the OpenAI official documentation."*
Training cost is computed the same way over the whole adaptation run (candidate
generation for the initial positive/negative sets plus every online-adaptation
iteration).

This module therefore provides

1. :data:`PRICING` / :func:`model_pricing` -- per-1k-token prices (gpt-3.5-turbo-1106 is
   the pricing model used by the paper for the *cost table*, Section 4.4),
2. :class:`CostLedger` -- a drop-in ledger for
   :class:`bbox_adapter.llm.blackbox_client.BlackBoxClient` (the client calls
   ``ledger.add(prompt_tokens=..., completion_tokens=..., model=..., n=...)``), with
   per-phase (training / inference) accumulation,
3. :class:`ExperimentCost` -- one row of Table 4 with accuracy, training cost,
   inference cost per 1k questions, plus delta (%) and "times less than SFT" ratios,
4. estimators used for dry runs / projections, and
5. the paper's Table 4 reference numbers together with
   :func:`compare_cost_to_paper` for regression checks.

Everything here is pure-python (no torch / no network): it only consumes token
statistics returned by the text-only black-box clients.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pricing (OpenAI official documentation, gpt-3.5-turbo-1106 -- Section 4.4)
# --------------------------------------------------------------------------- #

DEFAULT_PRICING_MODEL = "gpt-3.5-turbo-1106"
USD_PER_1K_INPUT_TOKENS = 0.0015
USD_PER_1K_OUTPUT_TOKENS = 0.0020

#: USD per 1k tokens, keyed by model name.  ``input`` == prompt tokens,
#: ``output`` == completion tokens.
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-3.5-turbo-1106": {"input": 0.0015, "output": 0.0020},
    "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.0020},
    "gpt-3.5-turbo-16k": {"input": 0.0030, "output": 0.0040},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.0020},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.0020},
    "davinci-002": {"input": 0.0020, "output": 0.0020},
    "gpt-4-32k": {"input": 0.060, "output": 0.120},
    "gpt-4": {"input": 0.030, "output": 0.060},
}

#: Fallback rule when a model is unknown (matches the paper's pricing model).
PRICING_FALLBACK = {"input": USD_PER_1K_INPUT_TOKENS, "output": USD_PER_1K_OUTPUT_TOKENS}

#: Phase names used by the ledger.
PHASE_TOTAL = "total"
PHASE_TRAINING = "training"
PHASE_INFERENCE = "inference"
DEFAULT_PHASE = PHASE_TOTAL
PHASES = (PHASE_TOTAL, PHASE_TRAINING, PHASE_INFERENCE)

#: Number of questions the "cost per 1k Q" unit refers to (Table 4 header).
DEFAULT_PER_QUESTIONS = 1000


def _normalize_model_name(model: Optional[str]) -> str:
    """Lower-case / strip whitespace of a (possibly Azure-deployment) model name."""
    if not model:
        return DEFAULT_PRICING_MODEL
    return str(model).strip().lower()


def model_pricing(model: Optional[str] = None) -> Dict[str, float]:
    """Return ``{"input": usd_per_1k, "output": usd_per_1k}`` for ``model``.

    Resolution order: exact match, longest matching prefix among known models,
    then :data:`PRICING_FALLBACK` (gpt-3.5-turbo-1106 prices, the paper's choice).
    """
    name = _normalize_model_name(model)
    if name in PRICING:
        return dict(PRICING[name])
    best_key = ""
    for key in PRICING:
        if name.startswith(key) and len(key) > len(best_key):
            best_key = key
    if best_key:
        return dict(PRICING[best_key])
    return dict(PRICING_FALLBACK)


def resolve_pricing(
    model: Optional[str] = None,
    usd_per_1k_input_tokens: Optional[float] = None,
    usd_per_1k_output_tokens: Optional[float] = None,
) -> Dict[str, float]:
    """Combine a model lookup with explicit price overrides (explicit wins)."""
    price = model_pricing(model)
    if usd_per_1k_input_tokens is not None:
        price["input"] = float(usd_per_1k_input_tokens)
    if usd_per_1k_output_tokens is not None:
        price["output"] = float(usd_per_1k_output_tokens)
    return price


def tokens_to_usd(
    prompt_tokens: float = 0,
    completion_tokens: float = 0,
    *,
    model: Optional[str] = None,
    usd_per_1k_input_tokens: Optional[float] = None,
    usd_per_1k_output_tokens: Optional[float] = None,
    per_1k: float = 1000.0,
) -> float:
    """Convert raw Azure token statistics into dollars.

    ``cost = prompt_tokens/1k * price_in + completion_tokens/1k * price_out``
    exactly as described in Section 4.4.
    """
    price = resolve_pricing(model, usd_per_1k_input_tokens, usd_per_1k_output_tokens)
    scale = float(per_1k) if per_1k else 1000.0
    return (
        float(prompt_tokens) / scale * price["input"]
        + float(completion_tokens) / scale * price["output"]
    )


#: Alias used by scripts: price a token bundle.
price_tokens = tokens_to_usd


# --------------------------------------------------------------------------- #
# Token usage containers
# --------------------------------------------------------------------------- #


@dataclass
class TokenUsage:
    """Accumulated token statistics for one phase (or a whole run)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0
    n_samples: int = 0
    n_questions: int = 0
    models: Dict[str, int] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------- #
    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    @property
    def empty(self) -> bool:
        return self.n_calls == 0 and self.total_tokens == 0

    @property
    def mean_prompt_tokens(self) -> float:
        return self.prompt_tokens / self.n_calls if self.n_calls else 0.0

    @property
    def mean_completion_tokens(self) -> float:
        return self.completion_tokens / self.n_calls if self.n_calls else 0.0

    # -- arithmetic ------------------------------------------------------- #
    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += int(getattr(other, "prompt_tokens", 0))
        self.completion_tokens += int(getattr(other, "completion_tokens", 0))
        self.n_calls += int(getattr(other, "n_calls", 0))
        self.n_samples += int(getattr(other, "n_samples", 0))
        self.n_questions += int(getattr(other, "n_questions", 0))
        for model, count in (getattr(other, "models", None) or {}).items():
            self.models[model] = self.models.get(model, 0) + int(count)
        return self

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        clone = self.copy()
        clone += other
        return clone

    def copy(self) -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=int(self.prompt_tokens),
            completion_tokens=int(self.completion_tokens),
            n_calls=int(self.n_calls),
            n_samples=int(self.n_samples),
            n_questions=int(self.n_questions),
            models=dict(self.models),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
            "n_calls": int(self.n_calls),
            "n_samples": int(self.n_samples),
            "n_questions": int(self.n_questions),
            "mean_prompt_tokens": round(self.mean_prompt_tokens, 3),
            "mean_completion_tokens": round(self.mean_completion_tokens, 3),
        }
        if self.models:
            data["models"] = dict(self.models)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenUsage":
        data = data or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            n_calls=int(data.get("n_calls", 0)),
            n_samples=int(data.get("n_samples", 0)),
            n_questions=int(data.get("n_questions", 0)),
            models=dict(data.get("models", {}) or {}),
        )

    @classmethod
    def from_value(cls, value: Any) -> "TokenUsage":
        """Best-effort coercion of a GenerationResult / dict / tuple / numbers."""
        if value is None:
            return cls()
        if isinstance(value, TokenUsage):
            return value.copy()
        if isinstance(value, Mapping):
            return cls(
                prompt_tokens=int(value.get("prompt_tokens", value.get("prompt", 0)) or 0),
                completion_tokens=int(
                    value.get("completion_tokens", value.get("completion", 0)) or 0
                ),
                n_calls=int(value.get("n_calls", 1) or 0),
                n_samples=int(value.get("n_samples", value.get("n", 0)) or 0),
                n_questions=int(value.get("n_questions", 0) or 0),
            )
        if isinstance(value, (tuple, list)):
            if len(value) >= 2:
                return cls(
                    prompt_tokens=int(value[0] or 0),
                    completion_tokens=int(value[1] or 0),
                    n_calls=1,
                )
            if len(value) == 1:
                return cls(completion_tokens=int(value[0] or 0), n_calls=1)
            return cls()
        # duck-typed objects (GenerationResult from llm/blackbox_client.py)
        prompt = getattr(value, "prompt_tokens", 0) or 0
        completion = getattr(value, "completion_tokens", 0) or 0
        n_samples = getattr(value, "n", 0) or 0
        if not n_samples:
            texts = getattr(value, "texts", None)
            n_samples = len(texts) if texts is not None else 0
        model = getattr(value, "model", None)
        usage = cls(
            prompt_tokens=int(prompt),
            completion_tokens=int(completion),
            n_calls=1,
            n_samples=int(n_samples),
        )
        if model:
            usage.models[str(model)] = usage.n_samples or 1
        return usage


@dataclass
class CostConfig:
    """Pricing / reporting configuration (mirrors ``cost:`` in ``configs/*.yaml``)."""

    model: str = DEFAULT_PRICING_MODEL
    usd_per_1k_input_tokens: Optional[float] = None
    usd_per_1k_output_tokens: Optional[float] = None
    per_questions: int = DEFAULT_PER_QUESTIONS
    currency: str = "USD"
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def pricing(self) -> Dict[str, float]:
        return resolve_pricing(
            self.model, self.usd_per_1k_input_tokens, self.usd_per_1k_output_tokens
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "usd_per_1k_input_tokens": self.usd_per_1k_input_tokens,
            "usd_per_1k_output_tokens": self.usd_per_1k_output_tokens,
            "per_questions": self.per_questions,
            "currency": self.currency,
            "enabled": self.enabled,
            "pricing": self.pricing(),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "CostConfig":
        data = data or {}
        known = {
            "model",
            "usd_per_1k_input_tokens",
            "usd_per_1k_output_tokens",
            "per_questions",
            "currency",
            "enabled",
        }
        kwargs = {k: v for k, v in data.items() if k in known and v is not None}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs) if kwargs else cls()
        if extra:
            cfg.extra = dict(extra)
        return cfg

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "CostConfig":
        """Build from a full experiment config (uses its ``cost`` sub-mapping)."""
        if not config:
            return cls()
        section = config.get("cost") if "cost" in config else config
        return cls.from_dict(section if isinstance(section, Mapping) else None)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "CostConfig":
        """Accept either direct field names or a nested ``cost=...`` mapping."""
        nested = kwargs.pop("cost", None)
        if isinstance(nested, Mapping):
            base = cls.from_dict(nested)
            for key, value in kwargs.items():
                if hasattr(base, key) and value is not None:
                    setattr(base, key, value)
            return base
        return cls.from_dict({k: v for k, v in kwargs.items() if v is not None})


# --------------------------------------------------------------------------- #
# Ledger (the seam consumed by llm/blackbox_client.py)
# --------------------------------------------------------------------------- #


class PhaseView:
    """Context-manager view routing :meth:`CostLedger.add` calls to one phase.

    Usage::

        ledger = CostLedger()
        client = build_client("gpt-3.5-turbo", ledger=ledger)
        with ledger.phase("training"):
            ...            # every client call is booked as training cost
        with ledger.phase("inference"):
            ...
    """

    def __init__(self, ledger: "CostLedger", phase: str):
        self.ledger = ledger
        self.phase = phase

    def __enter__(self) -> "PhaseView":
        self.ledger.push_phase(self.phase)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ledger.pop_phase()
        return False

    # convenience passthroughs so a view can be used like a ledger
    def add(self, *args: Any, **kwargs: Any) -> TokenUsage:
        kwargs.setdefault("phase", self.phase)
        return self.ledger.add(*args, **kwargs)

    def record(self, *args: Any, **kwargs: Any) -> TokenUsage:
        kwargs.setdefault("phase", self.phase)
        return self.ledger.record(*args, **kwargs)

    def usage(self) -> TokenUsage:
        return self.ledger.usage(self.phase)

    def cost(self, **kwargs: Any) -> float:
        return self.ledger.cost(phase=self.phase, **kwargs)

    def reset(self) -> None:
        self.ledger.reset(self.phase)


class CostLedger:
    """Accumulates Azure token statistics and converts them to dollars.

    Drop-in ``ledger`` for :class:`bbox_adapter.llm.blackbox_client.BlackBoxClient`,
    which calls ``ledger.add(prompt_tokens=..., completion_tokens=..., model=..., n=...)``
    for every completion request (no logprobs, text-only).
    """

    def __init__(
        self,
        config: Optional[Union[CostConfig, Mapping[str, Any]]] = None,
        *,
        model: Optional[str] = None,
        usd_per_1k_input_tokens: Optional[float] = None,
        usd_per_1k_output_tokens: Optional[float] = None,
        per_questions: Optional[int] = None,
        phases: Optional[Sequence[str]] = None,
    ):
        if isinstance(config, Mapping):
            self.config = CostConfig.from_dict(config)
        elif isinstance(config, CostConfig):
            self.config = config
        else:
            self.config = CostConfig()
        if model:
            self.config.model = model
        if usd_per_1k_input_tokens is not None:
            self.config.usd_per_1k_input_tokens = usd_per_1k_input_tokens
        if usd_per_1k_output_tokens is not None:
            self.config.usd_per_1k_output_tokens = usd_per_1k_output_tokens
        if per_questions:
            self.config.per_questions = int(per_questions)

        self._usage: "OrderedDict[str, TokenUsage]" = OrderedDict()
        for phase in phases or PHASES:
            self._usage[phase] = TokenUsage()
        self._events: List[Dict[str, Any]] = []
        self._phase_stack: List[str] = []
        self.started_at: float = time.time()

    # -- phase handling --------------------------------------------------- #
    def push_phase(self, phase: str) -> None:
        phase = self._normalize_phase(phase)
        self._usage.setdefault(phase, TokenUsage())
        self._phase_stack.append(phase)

    def pop_phase(self) -> Optional[str]:
        return self._phase_stack.pop() if self._phase_stack else None

    def phase(self, name: str) -> PhaseView:
        return PhaseView(self, self._normalize_phase(name))

    @property
    def active_phase(self) -> str:
        return self._phase_stack[-1] if self._phase_stack else DEFAULT_PHASE

    @staticmethod
    def _normalize_phase(phase: Optional[str]) -> str:
        if not phase:
            return DEFAULT_PHASE
        return str(phase).strip().lower().replace(" ", "_")

    def phases(self) -> List[str]:
        return list(self._usage.keys())

    # -- recording -------------------------------------------------------- #
    def add(
        self,
        prompt_tokens: float = 0,
        completion_tokens: float = 0,
        *,
        model: Optional[str] = None,
        n: int = 1,
        phase: Optional[str] = None,
        n_questions: int = 0,
        tag: Optional[str] = None,
        **kwargs: Any,
    ) -> TokenUsage:
        """Book one API request.  Signature matches the black-box client seam."""
        phase_name = self._normalize_phase(phase or self.active_phase)
        usage = self._usage.setdefault(phase_name, TokenUsage())
        usage.prompt_tokens += int(round(float(prompt_tokens or 0)))
        usage.completion_tokens += int(round(float(completion_tokens or 0)))
        usage.n_calls += 1
        usage.n_samples += int(n or 0)
        usage.n_questions += int(n_questions or 0)
        model_name = model or self.config.model
        usage.models[model_name] = usage.models.get(model_name, 0) + int(n or 0)

        # the ``total`` bucket always mirrors every call
        if phase_name != PHASE_TOTAL:
            total = self._usage.setdefault(PHASE_TOTAL, TokenUsage())
            total.prompt_tokens += int(round(float(prompt_tokens or 0)))
            total.completion_tokens += int(round(float(completion_tokens or 0)))
            total.n_calls += 1
            total.n_samples += int(n or 0)
            total.n_questions += int(n_questions or 0)
            total.models[model_name] = total.models.get(model_name, 0) + int(n or 0)

        self._events.append(
            {
                "phase": phase_name,
                "model": model_name,
                "prompt_tokens": int(round(float(prompt_tokens or 0))),
                "completion_tokens": int(round(float(completion_tokens or 0))),
                "n": int(n or 0),
                "t": round(time.time() - self.started_at, 3),
                "tag": tag,
            }
        )
        return usage

    # aliases used by different client versions
    record_tokens = add
    add_usage = add

    def record(
        self,
        result: Any,
        *,
        phase: Optional[str] = None,
        model: Optional[str] = None,
        n: Optional[int] = None,
        n_questions: int = 0,
        tag: Optional[str] = None,
    ) -> TokenUsage:
        """Book a ``GenerationResult`` (or dict/tuple/TokenUsage)."""
        usage = TokenUsage.from_value(result)
        if model is None:
            model = getattr(result, "model", None) or self.config.model
        return self.add(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=model,
            n=int(usage.n_samples if usage.n_samples else (n or 1)),
            phase=phase,
            n_questions=n_questions,
            tag=tag,
        )

    record_result = record

    def extend(
        self, results: Iterable[Any], *, phase: Optional[str] = None, **kwargs: Any
    ) -> TokenUsage:
        for result in results or []:
            self.record(result, phase=phase, **kwargs)
        return self.usage(phase)

    def __call__(self, *args: Any, **kwargs: Any) -> TokenUsage:
        return self.add(*args, **kwargs)

    # -- reporting -------------------------------------------------------- #
    def usage(self, phase: Optional[str] = None) -> TokenUsage:
        if phase is None:
            return self._usage.setdefault(PHASE_TOTAL, TokenUsage()).copy()
        return self._usage.setdefault(self._normalize_phase(phase), TokenUsage()).copy()

    #: alias
    tokens = usage

    def cost(
        self,
        phase: Optional[str] = None,
        *,
        model: Optional[str] = None,
        usd_per_1k_input_tokens: Optional[float] = None,
        usd_per_1k_output_tokens: Optional[float] = None,
    ) -> float:
        """Dollar cost of a phase (default: the grand total)."""
        usage = self.usage(phase)
        return tokens_to_usd(
            usage.prompt_tokens,
            usage.completion_tokens,
            model=model or self.config.model,
            usd_per_1k_input_tokens=(
                usd_per_1k_input_tokens
                if usd_per_1k_input_tokens is not None
                else self.config.usd_per_1k_input_tokens
            ),
            usd_per_1k_output_tokens=(
                usd_per_1k_output_tokens
                if usd_per_1k_output_tokens is not None
                else self.config.usd_per_1k_output_tokens
            ),
        )

    def cost_per_1k_questions(
        self,
        n_questions: int,
        *,
        phase: Optional[str] = None,
        per_questions: Optional[int] = None,
        **kwargs: Any,
    ) -> float:
        return cost_per_1k_questions(
            self.cost(phase, **kwargs),
            n_questions,
            per_questions=per_questions or self.config.per_questions,
        )

    def training_cost(self) -> float:
        return self.cost(PHASE_TRAINING)

    def inference_cost(self) -> float:
        return self.cost(PHASE_INFERENCE)

    def reset(self, phase: Optional[str] = None) -> None:
        if phase is None:
            for name in list(self._usage.keys()):
                self._usage[name] = TokenUsage()
            self._events.clear()
            self._phase_stack.clear()
            self.started_at = time.time()
        else:
            self._usage[self._normalize_phase(phase)] = TokenUsage()

    def merge(self, other: "CostLedger", *, scale: float = 1.0) -> "CostLedger":
        """Accumulate another ledger (optionally scaled, e.g. for multi-seed runs)."""
        if not isinstance(other, CostLedger):
            raise TypeError(f"expected CostLedger, got {type(other)!r}")
        for phase, usage in other._usage.items():
            target = self._usage.setdefault(phase, TokenUsage())
            target.prompt_tokens += int(round(usage.prompt_tokens * scale))
            target.completion_tokens += int(round(usage.completion_tokens * scale))
            target.n_calls += int(round(usage.n_calls * scale))
            target.n_samples += int(round(usage.n_samples * scale))
            target.n_questions += int(round(usage.n_questions * scale))
            for model, count in usage.models.items():
                target.models[model] = target.models.get(model, 0) + int(round(count * scale))
        return self

    def __add__(self, other: "CostLedger") -> "CostLedger":
        clone = CostLedger(self.config)
        clone.merge(self)
        clone.merge(other)
        return clone

    def snapshot(self) -> Dict[str, Any]:
        return {phase: usage.copy() for phase, usage in self._usage.items()}

    def n_calls(self, phase: Optional[str] = None) -> int:
        return self.usage(phase).n_calls

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "pricing": self.config.pricing(),
            "per_questions": self.config.per_questions,
            "phases": {
                phase: {**usage.to_dict(), "cost": round(self.cost(phase), 6)}
                for phase, usage in self._usage.items()
            },
            "total_cost": round(self.cost(), 6),
            "training_cost": round(self.cost(PHASE_TRAINING), 6),
            "inference_cost": round(self.cost(PHASE_INFERENCE), 6),
            "n_calls": self.n_calls(),
            "elapsed_seconds": round(time.time() - self.started_at, 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.summary()

    # -- factory helpers -------------------------------------------------- #
    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> "CostLedger":
        ledger = cls(CostConfig.from_config(config))
        for key, value in kwargs.items():
            if hasattr(ledger.config, key) and value is not None:
                setattr(ledger.config, key, value)
        return ledger

    def save(self, path: str, extra: Optional[Mapping[str, Any]] = None) -> str:
        payload = self.summary()
        if extra:
            payload.update(dict(extra))
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return path


#: Alias: the ledger is also a *token* ledger for `llm/token_cost.py` consumers.
TokenLedger = CostLedger


# --------------------------------------------------------------------------- #
# Cost arithmetic
# --------------------------------------------------------------------------- #


def cost_per_1k_questions(
    total_cost: float, n_questions: int, *, per_questions: int = DEFAULT_PER_QUESTIONS
) -> float:
    """Scale a measured total cost to the paper's "$ / 1k Q" unit."""
    if not n_questions or n_questions <= 0:
        raise ValueError("n_questions must be a positive integer")
    return float(total_cost) * float(per_questions) / float(n_questions)


def cost_per_question(total_cost: float, n_questions: int) -> float:
    if not n_questions or n_questions <= 0:
        raise ValueError("n_questions must be a positive integer")
    return float(total_cost) / float(n_questions)


#: aliases matching Table 4 column names
inference_cost_per_1k = cost_per_1k_questions


def training_cost_per_1k(train_total_cost: float, n_train_questions: int) -> float:
    """Training cost expressed per 1k *training* questions (Table 4 quoting unit)."""
    return cost_per_1k_questions(train_total_cost, n_train_questions)


def times_less(ours: float, baseline: float) -> Optional[float]:
    """``baseline / ours`` -- "N times less cost than SFT" (Section 4.4)."""
    if ours is None or baseline is None:
        return None
    if ours <= 0:
        return float("inf") if baseline > 0 else None
    return float(baseline) / float(ours)


#: aliases
cost_ratio = times_less
ratio = times_less


def speedup_ratio(ours: float, baseline: float) -> Optional[float]:
    """Alias of :func:`times_less` (kept for script readability)."""
    return times_less(ours, baseline)


def average(values: Iterable[Optional[float]], *, skip_none: bool = True) -> Optional[float]:
    """Mean over datasets, mirroring how the paper averages cost ratios."""
    items = [float(v) for v in values if (v is not None or not skip_none)]
    if not items:
        return None
    return sum(items) / len(items)


# --------------------------------------------------------------------------- #
# Estimators (dry runs / projections before spending Azure credits)
# --------------------------------------------------------------------------- #


def estimate_inference_cost_per_1k(
    prompt_tokens_per_question: float,
    completion_tokens_per_question: float,
    *,
    n_per_question: int = 1,
    per_questions: int = DEFAULT_PER_QUESTIONS,
    model: Optional[str] = None,
    usd_per_1k_input_tokens: Optional[float] = None,
    usd_per_1k_output_tokens: Optional[float] = None,
) -> float:
    """Project inference $ / 1k questions from per-question token counts.

    ``n_per_question`` is the number of black-box calls needed per question
    (1 for CoT / single-step candidate generation, ``beam_size * steps`` for the
    full-step adapted inference of Table 4).
    """
    n = max(int(n_per_question), 1)
    per_question = tokens_to_usd(
        prompt_tokens_per_question * n,
        completion_tokens_per_question * n,
        model=model,
        usd_per_1k_input_tokens=usd_per_1k_input_tokens,
        usd_per_1k_output_tokens=usd_per_1k_output_tokens,
    )
    return per_question * float(per_questions)


def estimate_candidate_sampling_cost(
    n_questions: int,
    *,
    n_calls_per_question: int = 1,
    prompt_tokens: float = 0.0,
    completion_tokens: float = 0.0,
    model: Optional[str] = None,
    usd_per_1k_input_tokens: Optional[float] = None,
    usd_per_1k_output_tokens: Optional[float] = None,
) -> float:
    """Cost of prompting the black-box LLM for ``n_calls_per_question`` candidates.

    Used to project the training cost of Table 4: the initial positive set needs
    ``K_init`` calls per training question, and each online iteration needs
    ``M`` more (Section 3.4, Algorithm 1).
    """
    if n_questions <= 0:
        raise ValueError("n_questions must be positive")
    calls = int(n_questions) * max(int(n_calls_per_question), 0)
    return tokens_to_usd(
        prompt_tokens * calls,
        completion_tokens * calls,
        model=model,
        usd_per_1k_input_tokens=usd_per_1k_input_tokens,
        usd_per_1k_output_tokens=usd_per_1k_output_tokens,
    )


def estimate_training_cost(
    n_train_questions: int,
    *,
    n_iterations: int = 4,
    n_candidates: int = 5,
    k_init: int = 5,
    prompt_tokens: float = 0.0,
    completion_tokens: float = 0.0,
    model: Optional[str] = None,
    usd_per_1k_input_tokens: Optional[float] = None,
    usd_per_1k_output_tokens: Optional[float] = None,
) -> float:
    """Project the adaptation (training) cost of one full BBox-Adapter run.

    Calls per training question = ``k_init + n_iterations * n_candidates``
    (initial candidate pool + one refreshed pool per online iteration), matching
    Algorithm 1 / Section 3.4.  Adapter updates themselves are free of API cost.
    """
    calls_per_question = int(k_init) + int(n_iterations) * int(n_candidates)
    return estimate_candidate_sampling_cost(
        n_train_questions,
        n_calls_per_question=calls_per_question,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        usd_per_1k_input_tokens=usd_per_1k_input_tokens,
        usd_per_1k_output_tokens=usd_per_1k_output_tokens,
    )


def estimate_azure_sft_cost(
    n_train_questions: int,
    *,
    epochs: int = 3,
    prompt_tokens: float = 0.0,
    completion_tokens: float = 0.0,
    n_test_questions: int = DEFAULT_PER_QUESTIONS,
    test_prompt_tokens: float = 0.0,
    test_completion_tokens: float = 0.0,
    model: str = "gpt-3.5-turbo",
    training_price_multiplier: float = 1.0,
    usd_per_1k_input_tokens: Optional[float] = None,
    usd_per_1k_output_tokens: Optional[float] = None,
) -> Dict[str, float]:
    """Project Azure-SFT training/inference cost (baseline rows of Table 4).

    Azure fine-tuning bills per training token (and per epoch); inference then
    uses an hourly hosting rate.  Both components are here approximated with the
    same per-1k-token prices scaled by ``training_price_multiplier`` so the
    accounting stays transparent and reproducible from token statistics alone.
    """
    train_cost = tokens_to_usd(
        prompt_tokens * n_train_questions * max(int(epochs), 1),
        completion_tokens * n_train_questions * max(int(epochs), 1),
        model=model,
        usd_per_1k_input_tokens=usd_per_1k_input_tokens,
        usd_per_1k_output_tokens=usd_per_1k_output_tokens,
    ) * float(training_price_multiplier)
    infer_cost = estimate_inference_cost_per_1k(
        test_prompt_tokens,
        test_completion_tokens,
        model=model,
        usd_per_1k_input_tokens=usd_per_1k_input_tokens,
        usd_per_1k_output_tokens=usd_per_1k_output_tokens,
    ) * (float(n_test_questions) / float(DEFAULT_PER_QUESTIONS))
    return {"training_cost": train_cost, "inference_cost": inference_cost}


# --------------------------------------------------------------------------- #
# One Table-4 row
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentCost:
    """A single Table 4 row: accuracy + training cost + inference cost per 1k Q."""

    dataset: Optional[str] = None
    method: str = "bbox_adapter"
    variant: Optional[str] = None  # "single_step" | "full_step" | None
    accuracy: Optional[float] = None
    training_cost: float = 0.0
    inference_cost_per_1k: float = 0.0
    n_train_questions: Optional[int] = None
    n_questions: int = DEFAULT_PER_QUESTIONS
    base_accuracy: Optional[float] = None
    baseline_training_cost: Optional[float] = None
    baseline_inference_cost_per_1k: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    config: Optional[CostConfig] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------- #
    @property
    def accuracy_delta(self) -> Optional[float]:
        """Absolute percentage-point improvement over the base model (paper's Δ%)."""
        if self.accuracy is None or self.base_accuracy is None:
            return None
        return float(self.accuracy) - float(self.base_accuracy)

    @property
    def training_cost_ratio(self) -> Optional[float]:
        """"N times less training cost than SFT" (Section 4.4)."""
        if self.baseline_training_cost is None:
            return None
        return times_less(self.training_cost, self.baseline_training_cost)

    @property
    def inference_cost_ratio(self) -> Optional[float]:
        if self.baseline_inference_cost_per_1k is None:
            return None
        return times_less(self.inference_cost_per_1k, self.baseline_inference_cost_per_1k)

    #: script-friendly aliases
    training_ratio = training_cost_ratio
    inference_ratio = inference_cost_ratio

    @property
    def training_cost_per_1k(self) -> Optional[float]:
        if not self.n_train_questions:
            return None
        return training_cost_per_1k(self.training_cost, self.n_train_questions)

    def cost_efficiency(self) -> Optional[float]:
        """Accuracy delta per training dollar (higher is better); diagnostic only."""
        delta = self.accuracy_delta
        if delta is None or self.training_cost <= 0:
            return None
        return delta / self.training_cost

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "dataset": self.dataset,
            "method": self.method,
            "variant": self.variant,
            "accuracy": self.accuracy,
            "accuracy_delta": self.accuracy_delta,
            "base_accuracy": self.base_accuracy,
            "training_cost": round(float(self.training_cost), 4),
            "inference_cost_per_1k": round(float(self.inference_cost_per_1k), 4),
            "n_questions": self.n_questions,
            "n_train_questions": self.n_train_questions,
            "baseline_training_cost": self.baseline_training_cost,
            "baseline_inference_cost_per_1k": self.baseline_inference_cost_per_1k,
            "baseline_accuracy": self.baseline_accuracy,
            "training_cost_ratio": _round_opt(self.training_cost_ratio),
            "inference_cost_ratio": _round_opt(self.inference_cost_ratio),
        }
        if self.config is not None:
            data["config"] = self.config.to_dict()
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    def row(self) -> Dict[str, Any]:
        """Table-4 style short row (used by :func:`format_cost_table`)."""
        return {
            "method": self.method,
            "dataset": self.dataset,
            "accuracy": self.accuracy,
            "training_cost": round(float(self.training_cost), 2),
            "inference_cost_per_1k": round(float(self.inference_cost_per_1k), 2),
            "delta": _round_opt(self.accuracy_delta),
        }

    # -- constructors ----------------------------------------------------- #
    @classmethod
    def from_ledgers(
        cls,
        training_ledger: Optional[CostLedger],
        inference_ledger: Optional[CostLedger],
        *,
        n_questions: int = DEFAULT_PER_QUESTIONS,
        n_train_questions: Optional[int] = None,
        accuracy: Optional[float] = None,
        base_accuracy: Optional[float] = None,
        baseline_training_cost: Optional[float] = None,
        baseline_inference_cost_per_1k: Optional[float] = None,
        dataset: Optional[str] = None,
        method: str = "bbox_adapter",
        variant: Optional[str] = None,
        per_questions: int = DEFAULT_PER_QUESTIONS,
        **kwargs: Any,
    ) -> "ExperimentCost":
        """Build a row from measured token ledgers (the real reproduction path)."""
        training_cost = training_ledger.cost() if training_ledger is not None else 0.0
        infer_cost = inference_ledger.cost() if inference_ledger is not None else 0.0
        config = None
        for ledger in (training_ledger, inference_ledger):
            if ledger is not None:
                config = ledger.config
                break
        return cls(
            dataset=dataset,
            method=method,
            variant=variant,
            accuracy=accuracy,
            training_cost=float(training_cost or 0.0),
            inference_cost_per_1k=cost_per_1k_questions(
                infer_cost, n_questions, per_questions=per_questions
            ),
            n_train_questions=n_train_questions,
            n_questions=int(n_questions),
            base_accuracy=base_accuracy,
            baseline_training_cost=baseline_training_cost,
            baseline_inference_cost_per_1k=baseline_inference_cost_per_1k,
            config=config,
            extra=dict(kwargs),
        )

    @classmethod
    def from_paper(cls, dataset: str, method: str) -> "ExperimentCost":
        """Reconstruct a Table 4 row from the paper's reported numbers."""
        row = paper_table4_row(dataset, method)
        if row is None:
            raise KeyError(f"no Table 4 row for dataset={dataset!r} method={method!r}")
        base = PAPER_TABLE4.get(_norm_dataset(dataset), {}).get("gpt-3.5-turbo", {})
        sft = PAPER_TABLE4.get(_norm_dataset(dataset), {}).get("azure_sft", {})
        return cls(
            dataset=_norm_dataset(dataset),
            method=method,
            variant=row.get("variant"),
            accuracy=row.get("accuracy"),
            training_cost=float(row.get("training_cost") or 0.0),
            inference_cost_per_1k=float(row.get("inference_cost_per_1k") or 0.0),
            n_questions=int(row.get("n_questions", DEFAULT_PER_QUESTIONS)),
            base_accuracy=base.get("accuracy"),
            baseline_training_cost=sft.get("training_cost"),
            baseline_inference_cost_per_1k=sft.get("inference_cost_per_1k"),
            baseline_accuracy=sft.get("accuracy"),
            extra={"source": "paper_table4"},
        )


def _round_opt(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isinf(value)):
        return value
    return round(float(value), ndigits)


# --------------------------------------------------------------------------- #
# Paper reference numbers (Table 4, Section 4.4)
# --------------------------------------------------------------------------- #

#: Training cost ($, whole run), inference cost ($/1k questions), accuracy (%).
PAPER_TABLE4: Dict[str, Dict[str, Dict[str, Any]]] = {
    "strategyqa": {
        "gpt-3.5-turbo": {
            "accuracy": 66.59,
            "training_cost": None,
            "inference_cost_per_1k": 0.41,
        },
        "azure_sft": {
            "accuracy": 76.86,
            "training_cost": 153.00,
            "inference_cost_per_1k": 7.50,
        },
        "bbox_adapter_single_step": {
            "accuracy": 69.87,
            "training_cost": 2.77,
            "inference_cost_per_1k": 2.20,
            "variant": "single_step",
        },
        "bbox_adapter_full_step": {
            "accuracy": 71.62,
            "training_cost": 3.48,
            "inference_cost_per_1k": 5.37,
            "variant": "full_step",
        },
    },
    "gsm8k": {
        "gpt-3.5-turbo": {
            "accuracy": 67.51,
            "training_cost": None,
            "inference_cost_per_1k": 1.22,
        },
        "azure_sft": {
            "accuracy": 69.94,
            "training_cost": 216.50,
            "inference_cost_per_1k": 28.30,
        },
        "bbox_adapter_single_step": {
            "accuracy": 71.13,
            "training_cost": 7.54,
            "inference_cost_per_1k": 3.10,
            "variant": "single_step",
        },
        "bbox_adapter_full_step": {
            "accuracy": 74.28,
            "training_cost": 11.58,
            "inference_cost_per_1k": 12.46,
            "variant": "full_step",
        },
    },
}

#: Section 4.4 headline ratios: "N times less cost than SFT" (averaged over
#: StrategyQA and GSM8K -- reproducible from :data:`PAPER_TABLE4`).
PAPER_COST_RATIOS: Dict[str, Dict[str, float]] = {
    "single_step": {"training": 41.97, "inference": 6.27},
    "full_step": {"training": 31.30, "inference": 1.84},
}

#: Section 4.4 headline performance gains over the base model, averaged over the
#: two datasets (Section 4.4: "+3.45%" single-step, "+5.90%" full-step).
PAPER_PERFORMANCE_GAINS: Dict[str, float] = {
    "single_step": 3.45,
    "full_step": 5.90,
}

#: Section 4.4 says BBox-Adapter (full-step) is this much cheaper to train than
#: Azure-SFT on StrategyQA / GSM8K respectively (re-derived from the table).
PAPER_METHOD_ALIASES = {
    "bbox": "bbox_adapter_full_step",
    "bbox_adapter": "bbox_adapter_full_step",
    "bbox_adapter_full": "bbox_adapter_full_step",
    "full": "bbox_adapter_full_step",
    "full_step": "bbox_adapter_full_step",
    "fullstep": "bbox_adapter_full_step",
    "single": "bbox_adapter_single_step",
    "single_step": "bbox_adapter_single_step",
    "singlestep": "bbox_adapter_single_step",
    "sft": "azure_sft",
    "azure": "azure_sft",
    "azure-sft": "azure_sft",
    "base": "gpt-3.5-turbo",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

DATASET_ALIASES = {
    "strategyqa": "strategyqa",
    "strategy_qa": "strategyqa",
    "sq": "strategyqa",
    "gsm8k": "gsm8k",
    "gsm": "gsm8k",
}


def _norm_dataset(dataset: Optional[str]) -> str:
    if not dataset:
        return ""
    key = str(dataset).strip().lower().replace("-", "").replace("_", "")
    for alias, canonical in DATASET_ALIASES.items():
        if alias.replace("_", "").replace("-", "") == key:
            return canonical
    return str(dataset).strip().lower()


def _norm_method(method: Optional[str]) -> str:
    if not method:
        return "bbox_adapter_full_step"
    key = str(method).strip().lower().replace(" ", "_")
    return PAPER_METHOD_ALIASES.get(key, key)


def paper_table4_row(dataset: str, method: str) -> Optional[Dict[str, Any]]:
    """Look up a Table 4 cell (dataset, method) in the paper's reported numbers."""
    return PAPER_TABLE4.get(_norm_dataset(dataset), {}).get(_norm_method(method))


def cost_ratios_from_table(
    variant: str = "full_step",
    *,
    datasets: Sequence[str] = ("strategyqa", "gsm8k"),
    table: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Re-derive the Section 4.4 "times less than SFT" ratios from Table 4.

    The paper averages the per-dataset ratios: e.g. full-step training gives
    ``mean(153.00/3.48, 216.50/11.58) = 31.30`` and inference gives
    ``mean(7.50/5.37, 28.30/12.46) = 1.84`` -- matching Section 4.4 verbatim.
    """
    table = table or PAPER_TABLE4
    key = "bbox_adapter_single_step" if "single" in str(variant) else "bbox_adapter_full_step"
    train_ratios, infer_ratios = [], []
    for dataset in datasets:
        cell = table.get(_norm_dataset(dataset), {})
        ours, sft = cell.get(key), cell.get("azure_sft")
        if not ours or not sft:
            continue
        if ours.get("training_cost") and sft.get("training_cost"):
            train_ratios.append(sft["training_cost"] / ours["training_cost"])
        if ours.get("inference_cost_per_1k") and sft.get("inference_cost_per_1k"):
            infer_ratios.append(sft["inference_cost_per_1k"] / ours["inference_cost_per_1k"])
    return {
        "training": average(train_ratios) or 0.0,
        "inference": average(infer_ratios) or 0.0,
    }


def performance_gain_from_table(
    variant: str = "full_step",
    *,
    datasets: Sequence[str] = ("strategyqa", "gsm8k"),
    table: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    """Re-derive Section 4.4's average accuracy gain over the base model."""
    table = table or PAPER_TABLE4
    key = "bbox_adapter_single_step" if "single" in str(variant) else "bbox_adapter_full_step"
    deltas = []
    for dataset in datasets:
        cell = table.get(_norm_dataset(dataset), {})
        ours, base = cell.get(key), cell.get("gpt-3.5-turbo")
        if not ours or not base:
            continue
        if ours.get("accuracy") is not None and base.get("accuracy") is not None:
            deltas.append(float(ours["accuracy"]) - float(base["accuracy"]))
    return average(deltas)


def compare_cost_to_paper(
    dataset: str,
    method: str = "bbox_adapter_full_step",
    *,
    training_cost: Optional[float] = None,
    inference_cost_per_1k: Optional[float] = None,
    accuracy: Optional[float] = None,
    tolerance: float = 0.5,
    relative_tolerance: float = 0.25,
) -> Dict[str, Any]:
    """Compare measured costs against Table 4 for reproducibility reporting.

    ``tolerance`` is absolute (USD) for training cost, ``relative_tolerance`` is
    applied to the inference cost per 1k questions (the paper's numbers were
    produced with Azure token statistics, which are deployment dependent).
    """
    reference = paper_table4_row(dataset, method)
    if reference is None:
        raise KeyError(f"no Table 4 row for dataset={dataset!r} method={method!r}")
    result: Dict[str, Any] = {
        "dataset": _norm_dataset(dataset),
        "method": _norm_method(method),
        "reference": dict(reference),
    }
    checks = [
        ("accuracy", accuracy, reference.get("accuracy"), tolerance),
        ("training_cost", training_cost, reference.get("training_cost"), tolerance),
        (
            "inference_cost_per_1k",
            inference_cost_per_1k,
            reference.get("inference_cost_per_1k"),
            None,
        ),
    ]
    for name, measured, expected, tol in checks:
        if measured is None:
            continue
        entry: Dict[str, Any] = {"measured": float(measured), "expected": expected}
        if expected is None:
            entry["within_tolerance"] = None
        else:
            entry["difference"] = float(measured) - float(expected)
            if name == "inference_cost_per_1k":
                limit = abs(float(expected)) * float(relative_tolerance)
            else:
                limit = abs(float(tol)) if tol is not None else 0.5
            entry["within_tolerance"] = abs(entry["difference"]) <= limit
            entry["limit"] = limit
        result[name] = entry
    return result


def reference_table(table: str = "table4") -> Dict[str, Any]:
    """Return a copy of one of the paper's reference tables."""
    name = str(table or "table4").strip().lower()
    if name in ("table4", "cost", "4"):
        return json.loads(json.dumps(PAPER_TABLE4))
    if name in ("ratios", "cost_ratios"):
        return json.loads(json.dumps(PAPER_COST_RATIOS))
    if name in ("gains", "performance", "performance_gains"):
        return json.loads(json.dumps(PAPER_PERFORMANCE_GAINS))
    raise KeyError(f"unknown reference table: {table!r}")


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def format_cost_table(rows: Iterable[Any], *, title: str = "") -> str:
    """Render Table-4 style rows (dicts or :class:`ExperimentCost`)."""
    materialized: List[Dict[str, Any]] = []
    for row in rows or []:
        if isinstance(row, ExperimentCost):
            materialized.append(row.row())
        elif isinstance(row, Mapping):
            materialized.append(dict(row))
    if not materialized:
        return title or ""

    datasets = []
    for row in materialized:
        ds = _norm_dataset(row.get("dataset")) or "?"
        if ds not in datasets:
            datasets.append(ds)

    header = ["Method"] + [f"{ds}" for ds in datasets] + ["Inference $/1k Q"]
    lines = []
    if title:
        lines.append(title)
    lines.append("  ".join(header))
    lines.append("-" * max(len("  ".join(header)), 40))
    by_dataset: Dict[str, List[Dict[str, Any]]] = {ds: [] for ds in datasets}
    for row in materialized:
        by_dataset.setdefault(_norm_dataset(row.get("dataset")) or "?", []).append(row)
    methods: List[str] = []
    for row in materialized:
        method = str(row.get("method", "?"))
        if method not in methods:
            methods.append(method)
    for method in methods:
        cells = [method]
        for ds in datasets:
            entry = next((r for r in by_dataset.get(ds, []) if str(r.get("method")) == method), None)
            if entry is None:
                cells.append("-")
            else:
                acc = entry.get("accuracy")
                delta = entry.get("delta")
                if acc is None:
                    cells.append("-")
                elif delta is None:
                    cells.append(f"{float(acc):.2f}")
                else:
                    cells.append(f"{float(acc):.2f} ({float(delta):+.2f})")
        infer = [
            float(r["inference_cost_per_1k"])
            for r in materialized
            if str(r.get("method")) == method and r.get("inference_cost_per_1k") is not None
        ]
        cells.append(f"{sum(infer)/len(infer):.2f}" if infer else "-")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def save_cost_report(path: str, payload: Any) -> str:
    """Persist a cost report (list of dicts, ExperimentCost, or ledger summary)."""
    if isinstance(payload, CostLedger):
        data: Any = payload.summary()
    elif isinstance(payload, ExperimentCost):
        data = payload.to_dict()
    elif isinstance(payload, Iterable) and not isinstance(payload, (Mapping, str, bytes)):
        data = [
            item.to_dict() if isinstance(item, ExperimentCost) else dict(item)
            for item in payload
        ]
    else:
        data = payload
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return path


def summarize_costs(rows: Iterable[ExperimentCost]) -> Dict[str, Any]:
    """Aggregate rows into the paper's headline statistics (Section 4.4)."""
    rows = list(rows)
    return {
        "n_rows": len(rows),
        "mean_inference_cost_per_1k": average(r.inference_cost_per_1k for r in rows),
        "mean_training_cost": average(r.training_cost for r in rows),
        "mean_accuracy_delta": average(r.accuracy_delta for r in rows),
    }


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #


def _self_test() -> Dict[str, Any]:
    """Dependency-free checks of pricing, ledger accounting and Table 4 targets."""
    checks: Dict[str, Any] = {}

    # 1) token -> dollar conversion (Section 4.4 pricing).
    cost = tokens_to_usd(1_000_000, 1_000_000, model="gpt-3.5-turbo-1106")
    expected = 1000 * 0.0015 + 1000 * 0.0020
    assert abs(cost - expected) < 1e-9, (cost, expected)
    checks["pricing_1M_tokens_usd"] = round(cost, 4)

    # unknown model falls back to the paper's pricing model
    assert abs(tokens_to_usd(1000, 0, model="unknown-deployment") - 0.0015) < 1e-12
    checks["pricing_fallback"] = True

    # 2) ledger records per phase and the total mirrors everything.
    ledger = CostLedger(CostConfig(model="gpt-3.5-turbo-1106"))
    with ledger.phase(PHASE_TRAINING):
        ledger.add(prompt_tokens=1000, completion_tokens=200, model="gpt-3.5-turbo", n=5)
    with ledger.phase(PHASE_INFERENCE):
        ledger.add(prompt_tokens=2000, completion_tokens=400, model="gpt-3.5-turbo", n=7)
    assert ledger.usage(PHASE_TRAINING).n_calls == 1
    assert ledger.usage(PHASE_INFERENCE).n_samples == 7
    assert ledger.usage().prompt_tokens == 3000
    assert abs(
        ledger.cost() - (ledger.cost(PHASE_TRAINING) + ledger.cost(PHASE_INFERENCE))
    ) < 1e-12
    checks["ledger_cost_usd"] = round(ledger.cost(), 6)
    checks["ledger_phases"] = ledger.phases()

    # 3) record() accepts GenerationResult-like objects (client seam).
    class _FakeResult:
        prompt_tokens = 500
        completion_tokens = 100
        n = 3
        model = "gpt-3.5-turbo"

    ledger2 = CostLedger()
    ledger2.record(_FakeResult(), phase=PHASE_INFERENCE)
    assert ledger2.usage(PHASE_INFERENCE).prompt_tokens == 500
    checks["record_duck_typed"] = True

    # 4) cost per 1k questions scaling + ratio helper.
    assert abs(cost_per_1k_questions(2.2, 1000) - 2.2) < 1e-12
    assert abs(cost_per_1k_questions(1.1, 500) - 2.2) < 1e-12
    assert abs(times_less(3.48, 153.00) - 43.9655) < 1e-3
    checks["cost_per_1k_scaling"] = True

    # 5) Section 4.4 ratios and gains must be reproducible from Table 4.
    ratios_full = cost_ratios_from_table("full_step")
    ratios_single = cost_ratios_from_table("single_step")
    assert abs(ratios_full["training"] - PAPER_COST_RATIOS["full_step"]["training"]) < 0.5, ratios_full
    assert abs(ratios_full["inference"] - PAPER_COST_RATIOS["full_step"]["inference"]) < 0.05, ratios_full
    assert abs(ratios_single["training"] - PAPER_COST_RATIOS["single_step"]["training"]) < 0.5, ratios_single
    assert abs(ratios_single["inference"] - PAPER_COST_RATIOS["single_step"]["inference"]) < 0.05, ratios_single
    gains_full = performance_gain_from_table("full_step")
    gains_single = performance_gain_from_table("single_step")
    assert abs(gains_full - PAPER_PERFORMANCE_GAINS["full_step"]) < 0.05, gains_full
    assert abs(gains_single - PAPER_PERFORMANCE_GAINS["single_step"]) < 0.05, gains_single
    checks["table4_ratios"] = {
        "full_step": {k: round(v, 3) for k, v in ratios_full.items()},
        "single_step": {k: round(v, 3) for k, v in ratios_single.items()},
    }
    checks["table4_gains"] = {
        "full_step": round(gains_full, 3),
        "single_step": round(gains_single, 3),
    }

    # 6) ExperimentCost row mirrors Table 4 and knows its SFT baseline.
    row = ExperimentCost.from_paper("strategyqa", "bbox_adapter_full_step")
    assert abs(row.accuracy - 71.62) < 1e-9
    assert abs(row.accuracy_delta - 5.03) < 1e-9
    assert abs(row.training_cost_ratio - 43.9655) < 1e-3
    assert abs(row.inference_cost_ratio - 1.3966) < 1e-3
    checks["example_row"] = row.to_dict()

    # 7) Estimators.
    est = estimate_inference_cost_per_1k(500, 200, n_per_question=18, model="gpt-3.5-turbo-1106")
    manual = (500 * 18 / 1000 * 0.0015 + 200 * 18 / 1000 * 0.0020) * 1000
    assert abs(est - manual) < 1e-9, (est, manual)
    train_est = estimate_training_cost(
        2059, n_iterations=4, n_candidates=5, k_init=5, prompt_tokens=200, completion_tokens=80
    )
    assert train_est > 0
    checks["estimated_train_cost_usd"] = round(train_est, 4)

    # 8) compare_to_paper accepts the reference values themselves.
    cmp = compare_cost_to_paper(
        "gsm8k",
        "bbox_adapter_full_step",
        training_cost=11.58,
        inference_cost_per_1k=12.46,
        accuracy=74.28,
    )
    assert cmp["accuracy"]["within_tolerance"] is True
    assert cmp["training_cost"]["within_tolerance"] is True
    assert cmp["inference_cost_per_1k"]["within_tolerance"] is True
    checks["compare_to_paper"] = "ok"

    # 9) Table rendering + JSON persistence (no-op path without output dir).
    text = format_cost_table(
        [ExperimentCost.from_paper("strategyqa", "gpt-3.5-turbo"),
         ExperimentCost.from_paper("gsm8k", "bbox_adapter_full_step")],
        title="Table 4 (reproduced)",
    )
    assert "Table 4" in text and "Inference $/1k Q" in text
    checks["table_rows"] = len(text.splitlines())

    # 10) Config round trip.
    cfg = CostConfig.from_config(
        {"cost": {"model_for_pricing": "gpt-3.5-turbo-1106", "per_questions": 1000}}
    )
    assert cfg.per_questions == 1000
    checks["config_round_trip"] = cfg.to_dict()

    return checks


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(_self_test(), indent=2, default=str))
