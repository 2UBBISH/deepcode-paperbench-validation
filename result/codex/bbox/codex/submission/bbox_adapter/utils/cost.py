"""Token accounting and cost estimation (Table 4, Section 4.4).

Prices are USD per 1K tokens.  The paper reports the StrategyQA / GSM8K costs
with the ``gpt-3.5-turbo-1106`` price of the OpenAI official documentation,
therefore that is the default entry of :data:`PRICING`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Pricing:
    input_per_1k: float
    output_per_1k: float


# USD / 1K tokens.  gpt-3.5-turbo-1106: $0.001 input, $0.002 output.
PRICING: Dict[str, Pricing] = {
    "gpt-3.5-turbo": Pricing(0.001, 0.002),
    "gpt-3.5-turbo-1106": Pricing(0.001, 0.002),
    "gpt-3.5-turbo-0613": Pricing(0.0015, 0.002),
    "gpt-4": Pricing(0.03, 0.06),
    "gpt-4-0613": Pricing(0.03, 0.06),
    "davinci-002": Pricing(0.002, 0.002),
}


@dataclass
class UsageTracker:
    """Accumulates token usage so that Table 4 can be reproduced."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_calls: int = 0
    per_call_prompt_tokens: list = field(default_factory=list)
    per_call_completion_tokens: list = field(default_factory=list)

    def add(self, prompt_tokens: int, completion_tokens: int, n_calls: int = 1) -> None:
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        self.num_calls += int(n_calls)
        self.per_call_prompt_tokens.append(int(prompt_tokens))
        self.per_call_completion_tokens.append(int(completion_tokens))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, model: str) -> float:
        price = PRICING.get(model) or PRICING.get(model.split("-20")[0])
        if price is None:
            raise KeyError(
                f"No pricing entry for {model!r}; add it to bbox_adapter.utils.cost.PRICING"
            )
        return (
            self.prompt_tokens / 1000.0 * price.input_per_1k
            + self.completion_tokens / 1000.0 * price.output_per_1k
        )

    def cost_per_1k_questions(self, model: str, num_questions: int) -> float:
        if num_questions <= 0:
            return float("nan")
        return self.cost(model) / num_questions * 1000.0

    def cost_or_none(self, model: str) -> Optional[float]:
        """Cost in USD, or ``None`` when the model has no pricing entry.

        Local models (Mixtral, the mock client used by the smoke tests) cost
        nothing per token, so they return ``0.0``.
        """

        if model.startswith("mock"):
            return 0.0
        try:
            return self.cost(model)
        except KeyError:
            return None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "num_calls": self.num_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def __add__(self, other: "UsageTracker") -> "UsageTracker":
        merged = UsageTracker()
        merged.prompt_tokens = self.prompt_tokens + other.prompt_tokens
        merged.completion_tokens = self.completion_tokens + other.completion_tokens
        merged.num_calls = self.num_calls + other.num_calls
        merged.per_call_prompt_tokens = (
            list(self.per_call_prompt_tokens) + list(other.per_call_prompt_tokens)
        )
        merged.per_call_completion_tokens = (
            list(self.per_call_completion_tokens) + list(other.per_call_completion_tokens)
        )
        return merged

    def __iadd__(self, other: "UsageTracker") -> "UsageTracker":
        merged = self + other
        self.prompt_tokens = merged.prompt_tokens
        self.completion_tokens = merged.completion_tokens
        self.num_calls = merged.num_calls
        self.per_call_prompt_tokens = merged.per_call_prompt_tokens
        self.per_call_completion_tokens = merged.per_call_completion_tokens
        return self
