"""C4: the Paratera provider — thinking off on the wire, guard on the way back, retries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from apps.v2.agent.paper2code.config import EventLog
from apps.v2.agent.paper2code.provider import ParateraProvider, ThinkingNotDisabled


def _completion(content="hi", tool_calls=None, finish="stop", reasoning_tokens=0, extra_message=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if extra_message:
        message.update(extra_message)
    return {
        "id": "x",
        "model": "DeepSeek-V4-Flash",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


class _Server:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body)


def _provider(server: _Server, tmp_path: Path, **kwargs) -> ParateraProvider:
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    provider = ParateraProvider(
        model="DeepSeek-V4-Flash",
        base_url="https://llmapi.example/v1",
        api_key="k",
        log_dir=tmp_path / "llm",
        events=EventLog(tmp_path / "events.jsonl"),
        transport=httpx.MockTransport(server.handler),
        sleep=_sleep,
        **kwargs,
    )
    provider._test_sleeps = sleeps  # type: ignore[attr-defined]
    return provider


def test_request_carries_thinking_disabled_and_logs_call(tmp_path: Path) -> None:
    server = _Server([(200, _completion("hello"))])
    provider = _provider(server, tmp_path)
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}], max_tokens=64))
    body = server.requests[0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 64
    assert body["model"] == "DeepSeek-V4-Flash"
    assert "tools" not in body
    assert response.content == "hello"
    assert response.finish_reason == "stop"
    assert response.usage["reasoning_tokens"] == 0
    log = json.loads((tmp_path / "llm" / "00001.json").read_text())
    assert log["reasoning_tokens"] == 0
    assert log["attempts"] == 1
    assert "Authorization" not in json.dumps(log)
    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    assert event["kind"] == "llm.call"
    assert event["seq"] == 1


def test_reasoning_tokens_abort_the_run(tmp_path: Path) -> None:
    provider = _provider(_Server([(200, _completion("x", reasoning_tokens=3))]), tmp_path)
    with pytest.raises(ThinkingNotDisabled):
        asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}]))
    provider = _provider(_Server([(200, _completion("x", extra_message={"reasoning_content": "let me think"}))]), tmp_path)
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}]))
    assert response.content == "x"
    assert response.usage["reasoning_tokens"] == 0
    assert response.usage["reasoning_content_chars"] == len("let me think")


def test_reasoning_content_aborts_only_in_strict_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER2CODE_STRICT_REASONING_CONTENT", "1")
    provider = _provider(_Server([(200, _completion("x", extra_message={"reasoning_content": "let me think"}))]), tmp_path)
    with pytest.raises(ThinkingNotDisabled):
        asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}]))


def test_429_twice_then_success_with_standard_delays(tmp_path: Path) -> None:
    server = _Server([(429, {"error": "slow down"}), (503, {"error": "busy"}), (200, _completion("ok"))])
    provider = _provider(server, tmp_path)
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}], retry_mode="standard"))
    assert response.content == "ok"
    assert provider._test_sleeps == [1.0, 2.0]
    log = json.loads((tmp_path / "llm" / "00001.json").read_text())
    assert log["attempts"] == 3
    assert len(log["errors"]) == 2


def test_standard_mode_gives_up_after_three_retries(tmp_path: Path) -> None:
    server = _Server([(500, {})] * 4)
    provider = _provider(server, tmp_path)
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}], retry_mode="standard"))
    assert response.finish_reason == "error"
    assert "HTTP 500" in (response.content or "")
    assert provider._test_sleeps == [1.0, 2.0, 4.0]


def test_persistent_mode_uses_env_delays_and_identical_error_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPCODE_CHAT_RETRY_DELAYS", "5,7")
    monkeypatch.setenv("DEEPCODE_PERSISTENT_MAX_DELAY", "6")
    monkeypatch.setenv("DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT", "3")
    server = _Server([(502, {"e": 1})] * 10)
    provider = _provider(server, tmp_path)
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}], retry_mode="persistent"))
    assert response.finish_reason == "error"
    assert provider._test_sleeps == [5.0, 6.0]  # third identical error stops before a third sleep
    assert len(server.requests) == 3


def test_non_retryable_status_returns_error_immediately(tmp_path: Path) -> None:
    server = _Server([(401, {"error": "bad key"})])
    provider = _provider(server, tmp_path)
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}], retry_mode="persistent"))
    assert response.finish_reason == "error"
    assert response.error_status_code == 401
    assert provider._test_sleeps == []


def test_length_passes_through_and_tool_calls_parse(tmp_path: Path) -> None:
    calls = [{"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": '{"file_path": "a.py", "content": "x"}'}}]
    server = _Server([(200, _completion("partial", finish="length")), (200, _completion(None, tool_calls=calls, finish="tool_calls"))])
    provider = _provider(server, tmp_path)
    first = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}]))
    assert first.finish_reason == "length"
    assert first.content == "partial"
    second = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}]))
    assert second.should_execute_tools
    assert second.tool_calls[0].name == "write_file"
    assert second.tool_calls[0].arguments == {"file_path": "a.py", "content": "x"}
    assert server.requests[1]["tools"][0]["function"]["name"] == "write_file"


def test_invalid_tool_arguments_become_error_response(tmp_path: Path) -> None:
    calls = [{"id": "c", "type": "function", "function": {"name": "write_file", "arguments": "{not json"}}]
    provider = _provider(_Server([(200, _completion(None, tool_calls=calls, finish="tool_calls"))]), tmp_path)
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}]))
    assert response.finish_reason == "error"
    assert "invalid JSON" in (response.content or "")


def test_streaming_aggregates_chunks(tmp_path: Path) -> None:
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": '{"a":'}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' 1}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 0}}},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=body.encode(), headers={"Content-Type": "text/event-stream"})

    provider = ParateraProvider(model="m", base_url="https://x.example/v1", api_key="k", stream=True, transport=httpx.MockTransport(handler))
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}]))
    assert response.content == "Hello"
    assert response.tool_calls[0].arguments == {"a": 1}
    assert response.tool_calls[0].id == "c1"
    assert response.usage["total_tokens"] == 3


def test_call_numbering_continues_across_processes(tmp_path: Path) -> None:
    (tmp_path / "llm").mkdir()
    (tmp_path / "llm" / "00007.json").write_text("{}")
    provider = _provider(_Server([(200, _completion("again"))]), tmp_path)
    asyncio.run(provider.chat_with_retry([{"role": "user", "content": "hi"}]))
    assert (tmp_path / "llm" / "00008.json").exists()


def test_thinking_enabled_sends_enabled_and_accepts_reasoning_tokens(tmp_path: Path) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("hi", reasoning_tokens=37, extra_message={"reasoning_content": "let me think"}))

    provider = ParateraProvider(model="deepseek-flash", base_url="https://api.deepseek.com/v1", api_key="k", log_dir=tmp_path, transport=httpx.MockTransport(handler), thinking="enabled")
    response = asyncio.run(provider.chat_with_retry([{"role": "user", "content": "x"}], retry_mode="standard"))
    assert seen[0]["thinking"] == {"type": "enabled"}
    assert response.usage["reasoning_tokens"] == 37  # no ThinkingNotDisabled: the caliber is thinking on
    assert response.usage["reasoning_content_chars"] == len("let me think")
    with pytest.raises(Exception, match="thinking must be"):
        ParateraProvider(model="m", base_url="https://x", api_key="k", thinking="maybe")
