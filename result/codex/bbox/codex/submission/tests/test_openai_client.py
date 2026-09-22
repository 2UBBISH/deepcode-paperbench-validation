"""The Azure/OpenAI client wiring, exercised with a stubbed SDK.

The reproduction environment has the real credentials; here we replace the SDK
object so that the request construction (messages, ``n``, ``stop``, system
prompt), the response parsing and the token accounting are still tested.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from bbox_adapter.llm.openai_client import OpenAIClient


class StubCompletions:
    def __init__(self, texts: List[str], usage: Dict[str, int]) -> None:
        self.texts = texts
        self.usage = usage
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    text=text,
                    finish_reason="stop",
                )
                for text in self.texts
            ],
            usage=SimpleNamespace(**self.usage),
        )


class StubSDK:
    def __init__(self, texts: List[str], usage: Dict[str, int]) -> None:
        completions = StubCompletions(texts, usage)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


def make_client(name: str, texts: List[str], cache_dir=None):
    client = OpenAIClient(name=name, provider="azure", cache_dir=cache_dir)
    sdk = StubSDK(texts, {"prompt_tokens": 12, "completion_tokens": 6})
    client._client = sdk  # bypass the credential check
    return client, sdk


def test_chat_client_parses_responses_and_counts_tokens():
    client, sdk = make_client("gpt-3.5-turbo", ["#### 4", "#### 5"])
    results = client.generate(["Q: what is 2 + 2?"], n=2, temperature=1.0, max_new_tokens=32,
                              stop=["\n"], system_prompt="be helpful")
    assert results[0].texts == ["#### 4", "#### 5"]
    assert client.usage.prompt_tokens == 12
    assert client.usage.completion_tokens == 6
    assert client.usage.num_calls == 2  # one accounting record per generation
    request = sdk.chat.completions.calls[0]
    assert request["model"] == "gpt-3.5-turbo"
    assert request["messages"][0] == {"role": "system", "content": "be helpful"}
    assert request["messages"][1]["role"] == "user"
    assert request["stop"] == ["\n"] and request["max_tokens"] == 32


def test_completion_client_is_used_for_davinci():
    client, sdk = make_client("davinci-002", ["Karachi is in Pakistan #### Yes"])
    text = client.generate_one("Q: Karachi?")
    assert "Karachi" in text
    assert sdk.completions.calls, "davinci-002 must use the completions endpoint"
    assert "messages" not in sdk.completions.calls[0]


def test_responses_are_cached_on_disk(tmp_path):
    cache_dir = str(tmp_path / "cache")
    client, sdk = make_client("gpt-3.5-turbo", ["cached answer"], cache_dir=cache_dir)
    client.generate_one("Q", temperature=0.0)
    assert len(sdk.chat.completions.calls) == 1

    # A fresh client with the same cache must not hit the API again.
    second, second_sdk = make_client("gpt-3.5-turbo", ["should not be used"], cache_dir=cache_dir)
    assert second.generate_one("Q", temperature=0.0) == "cached answer"
    assert second_sdk.chat.completions.calls == []


def test_price_lookup_for_known_models():
    from bbox_adapter.utils.cost import PRICING

    assert PRICING["gpt-3.5-turbo-1106"].input_per_1k == 0.001
    assert PRICING["davinci-002"].output_per_1k == 0.002
