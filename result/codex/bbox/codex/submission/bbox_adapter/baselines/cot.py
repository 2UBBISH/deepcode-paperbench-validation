"""Chain-of-Thought baseline: the un-adapted black-box LLM.

"(1) Chain-of-Thoughts (CoT) (Wei et al., 2022) represents the performance of
the LLM without any adaptation."  Every baseline and BBOX-ADAPTER use the CoT
prompt of Appendix J.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..data.loaders import QAExample
from ..data.prompts import build_generator_prompt
from ..llm.base import BlackBoxLLM


class CoTBaseline:
    def __init__(
        self,
        llm: BlackBoxLLM,
        dataset: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        num_samples: int = 1,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.dataset = dataset
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_samples = num_samples
        self.system_prompt = system_prompt

    def generate(self, examples: Sequence[QAExample]) -> List[List[str]]:
        prompts = [
            build_generator_prompt(self.dataset, ex.question, ex.choices) for ex in examples
        ]
        results = self.llm.generate(
            prompts,
            n=self.num_samples,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            system_prompt=self.system_prompt,
        )
        return [result.texts for result in results]

    def answer(self, example: QAExample) -> str:
        return self.generate([example])[0][0]


def run_cot(llm, dataset: str, examples: Sequence[QAExample], **kwargs) -> List[List[str]]:
    return CoTBaseline(llm, dataset, **kwargs).generate(examples)
