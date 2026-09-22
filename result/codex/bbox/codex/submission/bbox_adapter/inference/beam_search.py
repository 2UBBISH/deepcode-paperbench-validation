"""Sentence level beam search: the LLM proposes, the adapter evaluates.

Section 3.3: "For a beam size of ``k``, at each step ``l``, we generate ``n``
samples of ``s^l`` based on ``p_LLM(s^l | x, s^{1:l-1})`` for each beam.  This
results in ``n k`` candidate chain hypotheses of ``s^{1:l}``, forming the
candidate set ``C``.  We then select the top-``k`` beams with the highest scores
``g_theta(s^{1:l}, x)`` given by the adapter."

The paper's implementation is faithful to this description: the black-box LLM
is queried for exactly one further sentence (the generation stops at the
newline that separates reasoning steps) and the adapter scores the complete
prefix ``(x, s^{1:l})``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..data.prompts import build_next_sentence_prompt
from ..llm.base import BlackBoxLLM


@dataclass
class Beam:
    sentences: List[str] = field(default_factory=list)
    score: float = float("-inf")
    finished: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.sentences)

    def extend(self, sentence: str, stop_marker: str = "####") -> "Beam":
        sentence = (sentence or "").strip()
        sentences = self.sentences + ([sentence] if sentence else [])
        return Beam(
            sentences=sentences,
            score=self.score,
            finished=self.finished or stop_marker in sentence,
        )


@dataclass
class BeamSearchResult:
    beams: List[Beam]

    @property
    def best(self) -> Beam:
        return self.beams[0]

    @property
    def best_text(self) -> str:
        return self.beams[0].text if self.beams else ""

    @property
    def texts(self) -> List[str]:
        return [beam.text for beam in self.beams]

    def scores(self) -> List[float]:
        return [beam.score for beam in self.beams]


class AdaptedBeamSearch:
    """Beam search over sentence level hypotheses scored by the adapter."""

    def __init__(
        self,
        llm: BlackBoxLLM,
        adapter,
        dataset: str,
        beam_size: int = 3,
        num_samples_per_beam: int = 3,
        max_sentence_steps: int = 24,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_marker: str = "####",
        system_prompt: Optional[str] = None,
        scoring_question_suffix: Optional[str] = None,
        prefer_finished: bool = True,
    ) -> None:
        self.llm = llm
        self.adapter = adapter
        self.dataset = dataset
        self.beam_size = beam_size
        self.num_samples_per_beam = num_samples_per_beam
        self.max_sentence_steps = max_sentence_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_marker = stop_marker
        self.system_prompt = system_prompt
        self.scoring_question_suffix = scoring_question_suffix
        # A hypothesis that has not produced a final answer yet cannot be
        # graded; when at least one beam is complete we therefore restrict the
        # final selection to the complete hypotheses (still ranked by g_theta).
        self.prefer_finished = prefer_finished

    # ------------------------------------------------------------------ utils
    def _scoring_question(self, question: str) -> str:
        if self.scoring_question_suffix:
            return f"{question}\n{self.scoring_question_suffix}"
        return question

    def _score(self, question: str, beams: Sequence[Beam]) -> List[float]:
        if not beams or self.adapter is None:
            return [0.0 for _ in beams]
        texts = [beam.text for beam in beams]
        scores = self.adapter.score_batch([self._scoring_question(question)] * len(texts), texts)
        if hasattr(scores, "tolist"):
            return list(scores.tolist())
        return list(scores)

    # ------------------------------------------------------------------- run
    def search(self, question: str, num_return_beams: Optional[int] = None,
               beam_size: Optional[int] = None) -> BeamSearchResult:
        """Run the beam search and return the surviving beams, best first."""

        original_beam_size = self.beam_size
        if beam_size is not None:
            # Algorithm 1 samples ``M`` candidates with beam size ``M``.
            self.beam_size = beam_size
        try:
            return self._search(question, num_return_beams)
        finally:
            self.beam_size = original_beam_size

    def _search(self, question: str, num_return_beams: Optional[int] = None) -> BeamSearchResult:
        beams: List[Beam] = [Beam(sentences=[], score=0.0, finished=False)]
        for _ in range(self.max_sentence_steps):
            if all(beam.finished for beam in beams):
                break
            prompts: List[str] = []
            owners: List[int] = []
            for index, beam in enumerate(beams):
                if beam.finished:
                    continue
                prompts.append(
                    build_next_sentence_prompt(
                        self.dataset,
                        question,
                        beam.text,
                        choices=getattr(self, "_choices", None),
                    )
                )
                owners.append(index)
            if not prompts:
                break

            results = self.llm.generate(
                prompts,
                n=self.num_samples_per_beam,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                stop=["\n"],
                system_prompt=self.system_prompt,
            )

            candidates: List[Beam] = []
            for owner, result in zip(owners, results):
                for text in result.texts:
                    candidates.append(beams[owner].extend(text, self.stop_marker))
            # Keep the finished beams that were not expanded in this round.
            candidates.extend(beam for beam in beams if beam.finished)
            if not candidates:
                break

            scores = self._score(question, candidates)
            for beam, score in zip(candidates, scores):
                beam.score = float(score)

            # Finished hypotheses are ranked together with the unfinished ones
            # so that a complete answer is never discarded in favour of a
            # partial prefix with a higher energy.
            candidates.sort(key=lambda beam: beam.score, reverse=True)
            beams = candidates[: self.beam_size]

        beams = sorted(beams, key=lambda beam: beam.score, reverse=True)
        if self.prefer_finished and any(beam.finished for beam in beams):
            finished = [beam for beam in beams if beam.finished]
            unfinished = [beam for beam in beams if not beam.finished]
            beams = finished + unfinished
        if num_return_beams is not None:
            beams = beams[:num_return_beams]
        return BeamSearchResult(beams=beams)

    def answer(self, question: str, choices: Optional[Sequence[str]] = None) -> str:
        self._choices = choices
        result = self.search(question)
        return result.best_text

    def sample_candidates(
        self,
        question: str,
        choices: Optional[Sequence[str]] = None,
        num_candidates: Optional[int] = None,
    ) -> List[str]:
        """Draw ``M`` candidates from the adapted inference ``p_theta``."""

        self._choices = choices
        num_candidates = num_candidates or self.beam_size
        result = self.search(
            question,
            num_return_beams=num_candidates,
            beam_size=max(self.beam_size, num_candidates),
        )
        return result.texts
