"""Single-step vs. full-step adapted inference and plug-and-play."""

from __future__ import annotations

from typing import List, Sequence

from bbox_adapter.inference.adaptive_inference import AdaptedInference, score_candidates
from bbox_adapter.llm.mock import MockLLM


class RewardScorer:
    """Deterministic adapter stub: higher energy for answers containing 'good'."""

    def __init__(self, rewards=None) -> None:
        self.rewards = rewards or {}

    def score_batch(self, questions: Sequence[str], answers: Sequence[str]) -> List[float]:
        return [self.rewards.get(answer, float(answer.count("good"))) for answer in answers]


def test_single_step_picks_highest_energy_candidate():
    class FixedLLM(MockLLM):
        def _generate(self, prompts, n, temperature, top_p, max_new_tokens, stop,
                      system_prompt, seed):
            from bbox_adapter.llm.base import Generation, LLMResult

            texts = ["bad answer #### 1", "good answer #### 2", "bad answer #### 3"]
            return [LLMResult(prompt=p, generations=[Generation(text=t) for t in texts[:n]])
                    for p in prompts]

    inference = AdaptedInference(
        llm=FixedLLM(), adapter=RewardScorer(), dataset="gsm8k",
        mode="single", num_single_step_candidates=3,
    )
    assert "good" in inference.answer("q")


def test_full_step_uses_sentence_level_beam_search():
    llm = MockLLM(vocabulary=["7"], sentence_pool=["good step #### 7"])
    inference = AdaptedInference(
        llm=llm, adapter=RewardScorer(), dataset="gsm8k", beam_size=2,
        num_samples_per_beam=2, max_sentence_steps=3, mode="full",
    )
    answer = inference.answer("q")
    assert answer.endswith("#### 7")
    # Beam search stops as soon as the hypotheses carry the final answer.
    assert len(llm.usage.per_call_prompt_tokens) < 10


def test_same_adapter_plugs_into_another_llm():
    """Table 3 mechanism: the adapter is independent of the LLM's parameters."""

    adapter = RewardScorer()
    outputs = []
    for llm in (MockLLM(sentence_pool=["good step #### 1"]),
                MockLLM(sentence_pool=["good step #### 1"], name="other-llm")):
        inference = AdaptedInference(
            llm=llm, adapter=adapter, dataset="gsm8k", beam_size=1,
            num_samples_per_beam=1, max_sentence_steps=2, mode="full",
        )
        outputs.append(inference.answer("q"))
    assert outputs[0] == outputs[1]


def test_score_candidates_handles_missing_adapter():
    assert score_candidates(None, "q", ["a", "b"]) == [0.0, 0.0]
    assert score_candidates(RewardScorer(), "q", ["good", "bad"]) == [1.0, 0.0]
