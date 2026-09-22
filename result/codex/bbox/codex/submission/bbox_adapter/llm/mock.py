"""A deterministic in-process "black-box LLM" used by the unit tests.

It mimics the qualitative behaviour the adapter must exploit: the sampled
solutions contain a mixture of correct and incorrect reasoning traces, and the
distribution can be skewed by the prompt so that tests can check that the
adapter learns to prefer the target-domain answers.
"""

from __future__ import annotations

import hashlib
import random
from typing import List, Optional, Sequence

from .base import BlackBoxLLM, Generation, LLMResult


class MockLLM(BlackBoxLLM):
    def __init__(
        self,
        name: str = "mock-llm",
        max_new_tokens: int = 512,
        vocabulary: Optional[Sequence[str]] = None,
        sentence_pool: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name=name, max_new_tokens=max_new_tokens)
        self.vocabulary = list(vocabulary or ["alpha", "beta", "gamma"])
        self.sentence_pool = list(
            sentence_pool
            or [
                "We first restate the question.",
                "We compute the intermediate quantity.",
                "We combine the intermediate results.",
            ]
        )

    def _rng(self, prompt: str, seed: Optional[int]) -> random.Random:
        digest = hashlib.sha256(f"{prompt}|{seed}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _generate(self, prompts, n, temperature, top_p, max_new_tokens, stop,
                  system_prompt, seed) -> List[LLMResult]:
        results: List[LLMResult] = []
        for prompt in prompts:
            rng = self._rng(prompt, seed)
            generations: List[Generation] = []
            for _ in range(n):
                chunk = rng.choice(self.sentence_pool)
                token = rng.choice(self.vocabulary)
                # Emulate a single-sentence continuation: the adapter's beam
                # search stops at the newline it requested.
                text = f"{chunk} #### {token}"
                generations.append(
                    Generation(
                        text=text,
                        prompt_tokens=len(prompt.split()),
                        completion_tokens=len(text.split()),
                    )
                )
            results.append(LLMResult(prompt=prompt, generations=generations))
        return results
