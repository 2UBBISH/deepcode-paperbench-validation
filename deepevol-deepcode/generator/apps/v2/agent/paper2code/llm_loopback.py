"""A loopback ``/v1/chat/completions`` so the experiment agent's model calls go through the line's provider.

RSA's compiler (``rsa/compiler/llm.py``) and SetupX (``setupx/src/llm_engine.py``) speak bare
OpenAI-shaped HTTP to whatever ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``OPENAI_MODEL`` name;
in the product an agent shell forwards those to the Gateway. In CLI mode the line stands in
for the shell: an HTTP server on 127.0.0.1 that turns each request into a
``ParateraProvider.chat_with_retry`` call. That is what keeps the caliber (PLAN-3 §0 "租约与端点",
``adr/0002``): ``thinking: disabled`` on every request, the reasoning-token guard, one log per
call under ``llm/rsa/``, retries by the line's policy, and the vendor key never written to disk
— ``.env.small`` gets this server's address and a per-run token instead.

Two deliberate deviations from a transparent proxy, both recorded per request in
``llm/rsa/loopback.jsonl``:

* ``response_format`` is dropped. SetupX's ``json_mode`` sends ``{"type": "json_object"}``, and
  Paratera's V4-Flash returns scrambled text on requests that carry it (PITFALLS D/E, the same
  fault that forced the judge's parser onto V4-Pro). SetupX's prompts already demand JSON and it
  retries once on a parse failure, which is what the product path relies on too.
* ``model`` is always the run's ``experiment_model`` (``run.json``; the same
  ``DeepSeek-V4-Flash-Vision-Exp`` as the phase model for now, PLAN-3 §0 "模型（全部）"); a different
  requested name is recorded, not honoured.

``stream`` is refused (neither client streams). A provider-level failure after the line's own
retries answers 502 so RSA's client retries on its schedule; a caliber violation
(``ThinkingNotDisabled``) answers 400 so nobody retries it.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from loguru import logger

from apps.v2.agent.paper2code.provider import ParateraProvider, ThinkingNotDisabled

DROPPED_KEYS = ("response_format", "stream", "stream_options", "tools", "tool_choice", "n", "logprobs")
CHAT_PATH = "/v1/chat/completions"
REQUEST_TIMEOUT_S = 15 * 60


@dataclass(frozen=True, slots=True)
class LoopbackTarget:
    """What ``run_experiment_on_machine(llm_target=…)`` needs (main's ``setupx_env.LlmTarget`` shape)."""

    provider: str
    model_id: str
    base_url: str
    api_key: str

    def redacted(self) -> dict[str, Any]:
        # the key never leaves this process: only its length is logged (and the egress guard reads a literal
        # ``"api_key": <value>`` as a credential payload, so the field is named for what it is)
        return {"provider": self.provider, "model_id": self.model_id, "base_url": self.base_url, "api_key_chars": len(self.api_key)}


class LoopbackServer:
    """Owns a thread with its own event loop (the provider's httpx client lives there) and the HTTP server."""

    def __init__(
        self,
        provider_factory: Callable[[], Any],
        *,
        model: str,
        audit_path: Path | None = None,
        max_tokens_cap: int | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        request_timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self._provider_factory = provider_factory
        self.model = model
        self.audit_path = audit_path
        self.max_tokens_cap = max_tokens_cap
        self.token = secrets.token_urlsafe(24)
        self._host, self._port = host, port
        self._timeout = request_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._provider: Any = None
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._audit_lock = threading.Lock()
        self.requests = 0

    # -- lifecycle ---------------------------------------------------------------------------

    @classmethod
    def from_run(cls, run: Any, paths: Any, *, events: Callable[..., Any] | None = None, **kwargs: Any) -> "LoopbackServer":
        log_dir = paths.llm_dir / "rsa"
        return cls(
            lambda: ParateraProvider.from_run(run, paths, events=events, log_dir=log_dir),
            model=str(getattr(run, "experiment_model", None) or run.model),
            audit_path=log_dir / "loopback.jsonl",
            **kwargs,
        )

    def start(self) -> "LoopbackServer":
        if self._server is not None:
            return self
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, name="p2c-loopback-loop", daemon=True)
        self._loop_thread.start()
        self._provider = asyncio.run_coroutine_threadsafe(self._make_provider(), self._loop).result(timeout=30)
        server = self
        server_cls = ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                server._handle(self)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("loopback: " + fmt, *args)

        self._server = server_cls((self._host, self._port), Handler)
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(target=self._server.serve_forever, name="p2c-loopback-http", daemon=True)
        self._server_thread.start()
        logger.info("loopback chat endpoint on {} for model {} (logs under {})", self.base_url, self.model, self.audit_path.parent if self.audit_path else "-")
        return self

    async def _make_provider(self) -> Any:
        return self._provider_factory()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._loop is not None:
            if self._provider is not None and hasattr(self._provider, "aclose"):
                try:
                    asyncio.run_coroutine_threadsafe(self._provider.aclose(), self._loop).result(timeout=30)
                except Exception as exc:  # closing must never mask the run's own outcome
                    logger.debug("loopback provider close: {}", exc)
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=10)
            self._loop.close()
            self._loop = None
        self._provider = None

    def __enter__(self) -> "LoopbackServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- what the callers need -----------------------------------------------------------------

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("loopback server not started")
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self.port}/v1"

    def target(self) -> LoopbackTarget:
        return LoopbackTarget(provider="paper2code-loopback", model_id=self.model, base_url=self.base_url, api_key=self.token)

    @property
    def provider(self) -> Any:
        return self._provider

    # -- one request ----------------------------------------------------------------------------

    def _handle(self, http: BaseHTTPRequestHandler) -> None:
        started = time.time()
        if http.path.rstrip("/") not in {CHAT_PATH, CHAT_PATH.removeprefix("/v1")}:
            self._reply(http, 404, {"error": {"message": f"unknown path {http.path}; only {CHAT_PATH}"}})
            return
        auth = http.headers.get("Authorization", "")
        if not secrets.compare_digest(auth.removeprefix("Bearer ").strip(), self.token):
            self._reply(http, 401, {"error": {"message": "bad loopback token"}})
            return
        try:
            length = int(http.headers.get("Content-Length") or 0)
            payload = json.loads(http.rfile.read(length) or b"{}")
            messages = payload["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
        except (ValueError, KeyError, TypeError) as exc:
            self._reply(http, 400, {"error": {"message": f"bad request: {exc}"}})
            return
        dropped = sorted(k for k in DROPPED_KEYS if k in payload)
        requested_model = payload.get("model")
        max_tokens = payload.get("max_tokens")
        if isinstance(max_tokens, int) and self.max_tokens_cap:
            max_tokens = min(max_tokens, self.max_tokens_cap)
        temperature = payload.get("temperature")
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._provider.chat_with_retry(
                messages,
                model=self.model,
                max_tokens=max_tokens if isinstance(max_tokens, int) else None,
                temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
            ),
            self._loop,
        )
        try:
            response = future.result(timeout=self._timeout)
        except ThinkingNotDisabled as exc:
            self._audit(started, dropped, requested_model, status=400, finish_reason="caliber_violation", usage={})
            self._reply(http, 400, {"error": {"message": f"caliber violation: {exc}", "type": "thinking_not_disabled"}})
            return
        except Exception as exc:
            self._audit(started, dropped, requested_model, status=502, finish_reason="exception", usage={})
            self._reply(http, 502, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            return
        self.requests += 1
        if response.finish_reason == "error":
            self._audit(started, dropped, requested_model, status=502, finish_reason="error", usage=dict(response.usage))
            self._reply(http, 502, {"error": {"message": response.content or "provider error", "status": response.error_status_code}})
            return
        body = {
            "id": f"loopback-{self.requests}",
            "object": "chat.completion",
            "created": int(started),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content or ""},
                    "finish_reason": response.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": int(response.usage.get("prompt_tokens", 0)),
                "completion_tokens": int(response.usage.get("completion_tokens", 0)),
                "total_tokens": int(response.usage.get("total_tokens", 0)),
                "reasoning_tokens": int(response.usage.get("reasoning_tokens", 0)),
            },
        }
        self._audit(started, dropped, requested_model, status=200, finish_reason=response.finish_reason, usage=body["usage"])
        self._reply(http, 200, body)

    def _audit(self, started: float, dropped: list[str], requested_model: Any, *, status: int, finish_reason: str, usage: dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        line = {
            "ts": started,
            "duration_s": round(time.time() - started, 3),
            "status": status,
            "finish_reason": finish_reason,
            "dropped": dropped,
            "requested_model": requested_model,
            "model": self.model,
            "usage": usage,
        }
        try:
            with self._audit_lock:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("loopback audit write failed: {}", exc)

    @staticmethod
    def _reply(http: BaseHTTPRequestHandler, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        http.send_response(status)
        http.send_header("Content-Type", "application/json")
        http.send_header("Content-Length", str(len(data)))
        http.end_headers()
        http.wfile.write(data)


__all__ = ["CHAT_PATH", "DROPPED_KEYS", "LoopbackServer", "LoopbackTarget"]
