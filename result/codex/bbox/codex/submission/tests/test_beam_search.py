"""Sentence level beam search (Section 3.3)."""

from __future__ import annotations

from typing import List, Sequence

from bbox_adapter.inference.beam_search import AdaptedBeamSearch
from bbox_adapter.llm.mock import MockLLM


class StubScorer:
    """Prefers hypotheses that contain the word ``target``."""

    def __init__(self, rewards=None) -> None:
        self.rewards = rewards or {}

    def score_batch(self, questions: Sequence[str], answers: Sequence[str]) -> List[float]:
        return [self.rewards.get(answer, float(answer.count("target"))) for answer in answers]


def test_beam_search_returns_adapter_preferred_hypothesis():
    llm = MockLLM(vocabulary=["correct"], sentence_pool=["step one", "target step"])
    adapter = StubScorer()
    search = AdaptedBeamSearch(
        llm=llm, adapter=adapter, dataset="gsm8k", beam_size=3,
        num_samples_per_beam=3, max_sentence_steps=4, max_new_tokens=16,
    )
    result = search.search("What is 2 + 2?")
    assert result.beams, "the beam search must return at least one hypothesis"
    assert "target" in result.best_text
    assert result.best.score >= min(beam.score for beam in result.beams)


def test_beam_search_stops_when_all_beams_finish():
    llm = MockLLM(vocabulary=["x"], sentence_pool=["done #### x"])
    search = AdaptedBeamSearch(
        llm=llm, adapter=StubScorer(), dataset="gsm8k", beam_size=2,
        num_samples_per_beam=2, max_sentence_steps=5, max_new_tokens=8,
    )
    result = search.search("q")
    assert all(beam.finished for beam in result.beams)
    # One generation round is enough because the first sentence already ends
    # with the stop marker.
    assert len(llm.usage.per_call_prompt_tokens) <= 2


def test_prefer_finished_ranks_complete_hypotheses_first():
    class ScriptedLLM(MockLLM):
        def _generate(self, prompts, n, temperature, top_p, max_new_tokens, stop,
                      system_prompt, seed):
            from bbox_adapter.llm.base import Generation, LLMResult

            results = []
            for prompt in prompts:
                if "step" in prompt:
                    texts = ["final #### 1"]
                else:
                    texts = ["partial reasoning"]
                results.append(
                    LLMResult(
                        prompt=prompt,
                        generations=[Generation(text=text) for text in texts[:n]],
                    )
                )
            return results

    scorer = StubScorer(rewards={"final #### 1": 0.0, "partial reasoning": 5.0})
    search = AdaptedBeamSearch(
        llm=ScriptedLLM(), adapter=scorer, dataset="gsm8k", beam_size=1,
        num_samples_per_beam=1, max_sentence_steps=2, prefer_finished=True,
    )
    result = search.search("q")
    assert result.best.finished
