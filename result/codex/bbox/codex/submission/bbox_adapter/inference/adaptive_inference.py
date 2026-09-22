"""High level interface used by the experiments.

Two inference variants are reported in the paper (Section 4.4):

* ``full``   -- sentence level beam search (Section 3.3), the default;
* ``single`` -- "a simplified approach wherein the base model generates a set of
  complete answers in a single step and the adapter then selects the best answer
  from these candidates as the final response".
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..data.prompts import build_generator_prompt, format_choices
from ..data.answer_extraction import extract_answer
from ..llm.base import BlackBoxLLM
from .beam_search import AdaptedBeamSearch, BeamSearchResult


def score_candidates(adapter, question: str, candidates: Sequence[str]) -> List[float]:
    """Adapter scores ``g_theta(x, y)`` for a list of candidates."""

    if adapter is None or not candidates:
        return [0.0 for _ in candidates]
    scores = adapter.score_batch([question] * len(candidates), list(candidates))
    return list(scores.tolist()) if hasattr(scores, "tolist") else list(scores)


class AdaptedInference:
    """Adapted inference ``p_theta`` for a single black-box LLM."""

    def __init__(
        self,
        llm: BlackBoxLLM,
        adapter,
        dataset: str,
        beam_size: int = 3,
        num_samples_per_beam: int = 3,
        max_sentence_steps: int = 24,
        max_new_tokens: int = 64,
        max_solution_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        mode: str = "full",
        num_single_step_candidates: int = 5,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.adapter = adapter
        self.dataset = dataset
        self.mode = mode
        self.max_solution_tokens = max_solution_tokens
        self.num_single_step_candidates = num_single_step_candidates
        self.system_prompt = system_prompt
        self.beam_search = AdaptedBeamSearch(
            llm=llm,
            adapter=adapter,
            dataset=dataset,
            beam_size=beam_size,
            num_samples_per_beam=num_samples_per_beam,
            max_sentence_steps=max_sentence_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    # ------------------------------------------------------------------ util
    @staticmethod
    def _suffix(choices: Optional[Sequence[str]]) -> Optional[str]:
        return format_choices(choices) if choices else None

    def _prompt(self, question: str, choices: Optional[Sequence[str]]) -> str:
        return build_generator_prompt(self.dataset, question, choices)

    # ----------------------------------------------------------- single step
    def single_step_sampling(self, question: str, choices: Optional[Sequence[str]] = None,
                             num_candidates: Optional[int] = None) -> List[str]:
        prompt = self._prompt(question, choices)
        n = num_candidates or self.num_single_step_candidates
        results = self.llm.generate(
            [prompt],
            n=n,
            temperature=self.beam_search.temperature,
            top_p=self.beam_search.top_p,
            max_new_tokens=self.max_solution_tokens,
            system_prompt=self.system_prompt,
        )
        return results[0].texts

    def single_step(self, question: str, choices: Optional[Sequence[str]] = None,
                    num_candidates: Optional[int] = None) -> str:
        candidates = self.single_step_sampling(question, choices, num_candidates)
        if not candidates:
            return ""
        scoring_question = question
        if choices:
            scoring_question = f"{question}\n{format_choices(choices)}"
        scores = score_candidates(self.adapter, scoring_question, candidates)
        best = max(range(len(candidates)), key=lambda index: scores[index])
        return candidates[best]

    # ------------------------------------------------------------- full step
    def full_step(self, question: str, choices: Optional[Sequence[str]] = None) -> str:
        return self.beam_search.answer(question, choices)

    def sample_candidates(self, question: str, choices: Optional[Sequence[str]] = None,
                          num_candidates: int = 5) -> List[str]:
        """Candidates sampled from the adapted inference (Eq. 5/6 of Section 3.4)."""

        return self.beam_search.sample_candidates(question, choices, num_candidates)

    # ------------------------------------------------------------------ main
    def answer(self, question: str, choices: Optional[Sequence[str]] = None,
               num_candidates: Optional[int] = None) -> str:
        if self.mode == "single":
            return self.single_step(question, choices, num_candidates)
        return self.full_step(question, choices)

    def answer_with_details(self, question: str, choices: Optional[Sequence[str]] = None):
        """Return the answer together with the ranked beam hypotheses."""

        if self.mode == "single":
            candidates = self.single_step_sampling(question, choices)
            scoring_question = question
            if choices:
                scoring_question = f"{question}\n{format_choices(choices)}"
            scores = score_candidates(self.adapter, scoring_question, candidates)
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            result = BeamSearchResult(beams=[])
            best = candidates[order[0]] if order else ""
            return best, [(candidates[i], scores[i]) for i in order]
        result = self.beam_search.search(question)
        return result.best_text, list(zip(result.texts, result.scores()))

    def final_answer(self, question: str, choices: Optional[Sequence[str]] = None) -> Optional[str]:
        text = self.answer(question, choices)
        return extract_answer(self.dataset, text, len(choices) if choices else None)
