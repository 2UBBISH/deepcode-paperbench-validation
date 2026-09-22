"""PLAN-3 item 4b: the loopback chat endpoint the experiment agent's clients talk to."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from apps.v2.agent.paper2code.llm_loopback import CHAT_PATH, LoopbackServer
from apps.v2.agent.paper2code.provider import ThinkingNotDisabled
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMResponse


class _Provider:
    """What the loopback needs from a provider: ``chat_with_retry`` + ``aclose``; records every call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = False
        self.behaviour = "ok"

    async def chat_with_retry(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.behaviour == "thinking":
            raise ThinkingNotDisabled("reply reports 12 reasoning tokens")
        if self.behaviour == "error":
            return LLMResponse(content="HTTP 503: upstream busy", finish_reason="error", error_status_code=503)
        if self.behaviour == "boom":
            raise RuntimeError("loop exploded")
        return LLMResponse(
            content='{"thought": "install", "action_type": "SHELL_COMMAND", "content": {"command": "pip install torch"}}',
            finish_reason="stop",
            usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150, "reasoning_tokens": 0},
        )

    async def aclose(self):
        self.closed = True


@pytest.fixture
def server(tmp_path: Path):
    provider = _Provider()
    srv = LoopbackServer(lambda: provider, model="DeepSeek-V4-Flash", audit_path=tmp_path / "llm" / "rsa" / "loopback.jsonl", max_tokens_cap=32768)
    with srv:
        yield srv, provider
    assert provider.closed


def _post(srv: LoopbackServer, payload: dict, *, token: str | None = None, path: str = CHAT_PATH) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token if token is not None else srv.token}"}
    return httpx.post(f"http://127.0.0.1:{srv.port}{path}", json=payload, headers=headers, timeout=10)


def test_setupx_style_request_goes_through_the_provider_with_response_format_dropped(server, tmp_path: Path) -> None:
    srv, provider = server
    payload = {
        "model": "whatever-setupx-was-told",
        "messages": [{"role": "system", "content": "be an env agent"}, {"role": "user", "content": "state"}],
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    r = _post(srv, payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "DeepSeek-V4-Flash"
    assert body["choices"][0]["message"]["content"].startswith('{"thought"')
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150, "reasoning_tokens": 0}
    call = provider.calls[0]
    assert call["messages"] == payload["messages"]
    assert call["model"] == "DeepSeek-V4-Flash"
    assert call["max_tokens"] == 4096
    assert call["temperature"] is None
    assert "response_format" not in call
    audit = [json.loads(line) for line in (tmp_path / "llm" / "rsa" / "loopback.jsonl").read_text().splitlines()]
    assert audit[0]["dropped"] == ["response_format"]
    assert audit[0]["requested_model"] == "whatever-setupx-was-told"
    assert audit[0]["status"] == 200


def test_rsa_style_request_keeps_temperature_and_caps_max_tokens(server) -> None:
    srv, provider = server
    r = _post(srv, {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 200000, "temperature": 0.0})
    assert r.status_code == 200
    assert provider.calls[0]["max_tokens"] == 32768
    assert provider.calls[0]["temperature"] == 0.0


def test_bad_token_and_bad_path_and_bad_body(server) -> None:
    srv, provider = server
    assert _post(srv, {"messages": [{"role": "user", "content": "x"}]}, token="nope").status_code == 401
    assert _post(srv, {"messages": [{"role": "user", "content": "x"}]}, path="/v1/embeddings").status_code == 404
    assert _post(srv, {"messages": []}).status_code == 400
    assert _post(srv, {"model": "m"}).status_code == 400
    assert provider.calls == []


def test_provider_error_answers_502_so_rsa_retries(server) -> None:
    srv, provider = server
    provider.behaviour = "error"
    r = _post(srv, {"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 502
    assert "upstream busy" in r.json()["error"]["message"]
    provider.behaviour = "boom"
    assert _post(srv, {"messages": [{"role": "user", "content": "x"}]}).status_code == 502


def test_caliber_violation_answers_400_so_nobody_retries(server, tmp_path: Path) -> None:
    srv, provider = server
    provider.behaviour = "thinking"
    r = _post(srv, {"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "thinking_not_disabled"
    audit = json.loads((tmp_path / "llm" / "rsa" / "loopback.jsonl").read_text().splitlines()[-1])
    assert audit["finish_reason"] == "caliber_violation"


def test_target_is_what_setupx_env_needs(server) -> None:
    srv, _ = server
    target = srv.target()
    assert target.base_url == f"http://127.0.0.1:{srv.port}/v1"
    assert target.model_id == "DeepSeek-V4-Flash"
    assert target.api_key == srv.token
    assert target.provider == "paper2code-loopback"
    assert target.redacted()["api_key_chars"] == len(target.api_key)
    assert "api_key" not in target.redacted()
    assert srv.token not in json.dumps(target.redacted())


def test_from_run_logs_under_llm_rsa(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("P2C_TEST_KEY", "k-test")
    run = SimpleNamespace(model="DeepSeek-V4-Flash", provider_base_url="https://example.invalid/v1", provider_key_env="P2C_TEST_KEY", provider_stream=False)
    paths = SimpleNamespace(llm_dir=tmp_path / "llm")
    srv = LoopbackServer.from_run(run, paths)
    with srv:
        assert srv.audit_path == tmp_path / "llm" / "rsa" / "loopback.jsonl"
        assert srv.provider._log_dir == tmp_path / "llm" / "rsa"
        assert srv.provider._model == "DeepSeek-V4-Flash"
        assert srv.model == "DeepSeek-V4-Flash"  # no experiment_model on the run: the phase model
    # S2: the loopback answers as run.json's experiment_model, whatever the phase model is
    run2 = SimpleNamespace(model="DeepSeek-V4-Flash", experiment_model="DeepSeek-V4-Flash-Vision-Exp", provider_base_url="https://example.invalid/v1", provider_key_env="P2C_TEST_KEY", provider_stream=False)
    srv2 = LoopbackServer.from_run(run2, paths)
    with srv2:
        assert srv2.model == "DeepSeek-V4-Flash-Vision-Exp"
        assert srv2.target().model_id == "DeepSeek-V4-Flash-Vision-Exp"
