"""Cost accounting of Table 4."""

from __future__ import annotations

from bbox_adapter.utils.cost import Pricing, UsageTracker


def test_cost_uses_input_and_output_prices():
    usage = UsageTracker()
    usage.add(prompt_tokens=1000, completion_tokens=1000)
    # gpt-3.5-turbo-1106: $0.001 / 1k input and $0.002 / 1k output
    assert abs(usage.cost("gpt-3.5-turbo-1106") - 0.003) < 1e-12


def test_cost_per_1k_questions_scales_with_the_split():
    usage = UsageTracker()
    usage.add(prompt_tokens=500, completion_tokens=500)
    cost_on_100 = usage.cost_per_1k_questions("gpt-3.5-turbo", 100)
    assert abs(cost_on_100 - usage.cost("gpt-3.5-turbo") * 10) < 1e-12


def test_local_and_mock_models_are_free():
    usage = UsageTracker()
    usage.add(prompt_tokens=10, completion_tokens=10)
    assert usage.cost_or_none("mock-llm") == 0.0
    assert usage.cost_or_none("mistralai/Mixtral-8x7B-v0.1") is None


def test_usage_tracker_accumulates():
    left = UsageTracker()
    left.add(10, 5)
    right = UsageTracker()
    right.add(1, 2)
    total = left + right
    assert (total.prompt_tokens, total.completion_tokens, total.num_calls) == (11, 7, 2)
