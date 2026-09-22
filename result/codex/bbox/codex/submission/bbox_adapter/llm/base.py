"""Common interface of the black-box LLM clients."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..utils.cost import UsageTracker


@dataclass
class Generation:
    """One generation returned by a black-box LLM."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class LLMResult:
    """All generations produced for a single prompt."""

    prompt: str
    generations: List[Generation]

    @property
    def texts(self) -> List[str]:
        return [g.text for g in self.generations]

    @property
    def best(self) -> str:
        return self.generations[0].text if self.generations else ""


class BlackBoxLLM(abc.ABC):
    """Base class of every black-box LLM used as a proposal generator."""

    def __init__(self, name: str, max_new_tokens: int = 512) -> None:
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.usage = UsageTracker()

    # ------------------------------------------------------------------ API
    @abc.abstractmethod
    def _generate(
        self,
        prompts: Sequence[str],
        n: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        stop: Optional[Sequence[str]],
        system_prompt: Optional[str],
        seed: Optional[int],
    ) -> List[LLMResult]:
        """Return ``n`` generations for each prompt."""

    def generate(
        self,
        prompts: Sequence[str],
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[LLMResult]:
        """Generate text for ``prompts`` while accumulating token usage."""

        if isinstance(prompts, str):  # convenience: accept a bare string
            prompts = [prompts]
        prompts = list(prompts)
        if not prompts:
            return []
        results = self._generate(
            prompts=prompts,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            stop=stop,
            system_prompt=system_prompt,
            seed=seed,
        )
        for result in results:
            for generation in result.generations:
                self.usage.add(generation.prompt_tokens, generation.completion_tokens)
        return results

    def generate_texts(self, prompts: Sequence[str], **kwargs) -> List[List[str]]:
        return [r.texts for r in self.generate(prompts, **kwargs)]

    def generate_one(self, prompt: str, **kwargs) -> str:
        return self.generate([prompt], n=1, **kwargs)[0].best

    def reset_usage(self) -> None:
        self.usage = UsageTracker()


def retry_call(fn, max_retries: int = 6, base_delay: float = 1.0, exceptions=(Exception,)):
    """Call ``fn`` with exponential back-off (network flakiness of the APIs)."""

    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except exceptions as exc:  # pragma: no cover - requires live API
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(base_delay * (2 ** attempt))
    raise last_error  # type: ignore[misc]
