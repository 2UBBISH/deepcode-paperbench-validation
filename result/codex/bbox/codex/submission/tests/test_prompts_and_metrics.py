"""Prompts of Appendix J and the answer extraction of the four datasets."""

from __future__ import annotations

from bbox_adapter.data.answer_extraction import extract_answer
from bbox_adapter.data.metrics import correctness
from bbox_adapter.data.prompts import (
    AI_FEEDBACK_PROMPTS,
    build_ai_feedback_prompt,
    build_generator_prompt,
)


def test_generator_prompt_contains_question_and_shots():
    prompt = build_generator_prompt("strategyqa", "Was X a part of Y?")
    assert "Was X a part of Y?" in prompt
    assert "Karachi was a part of Alexander the Great's success?" in prompt  # 2-shot
    gsm = build_generator_prompt("gsm8k", "How many apples?")
    assert gsm.count("#### The answer is") >= 4  # 4-shot prompt
    scienceqa = build_generator_prompt("scienceqa", "Which?", ["a", "b"])
    assert "Choices:" in scienceqa and "0: a" in scienceqa


def test_ai_feedback_prompts_include_candidates():
    prompt = build_ai_feedback_prompt("gsm8k", "How many?", ["#### 3", "#### 4"])
    assert "Candidate Answer 1" in prompt and "Candidate Answer 2" in prompt
    for dataset, template in AI_FEEDBACK_PROMPTS.items():
        assert "coherency" in template.lower() or "truthful" in template.lower()


def test_answer_extraction_per_dataset():
    assert extract_answer("gsm8k", "some reasoning\n#### The answer is 1,200") == "1200"
    assert extract_answer("strategyqa", "reasoning\n#### No.") == "No"
    assert extract_answer("scienceqa", "reasoning\n#### 2", 4) == "2"
    assert extract_answer("truthfulqa", "Watermelon seeds pass through.").startswith("Watermelon")


def test_correctness_matching():
    assert correctness("gsm8k", "#### 18", "18")
    assert correctness("gsm8k", "#### $18.0", "18")
    assert not correctness("gsm8k", "#### 19", "18")
    assert correctness("strategyqa", "#### yes", "Yes")
    assert correctness("scienceqa", "#### 1", "1", 3)
    assert not correctness("scienceqa", "#### 2", "1", 3)
