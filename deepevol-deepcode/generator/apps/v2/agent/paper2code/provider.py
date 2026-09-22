"""``LLMProvider`` for Paratera's OpenAI-compatible endpoint, over httpx.

Every model call of a run goes through one :class:`ParateraProvider`.
It sends ``thinking: {"type": "disabled"}`` on every request (the only
form Paratera honours; ``enable_thinking: false`` is ignored) and refuses
to continue if the reply reports any reasoning tokens or reasoning content:
the run's caliber is "V4-Flash, thinking off", and a single thinking call
would silently change what is being measured.

Retry policy follows the validation repo's env knobs (PLAN.md §6):
``standard`` = three retries at 1/2/4 s; ``persistent`` = the
``DEEPCODE_CHAT_RETRY_DELAYS`` cycle capped by ``DEEPCODE_PERSISTENT_MAX_DELAY``
until the same error repeats ``DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT``
times. Only network errors, 5xx, 429 and empty replies are retried.

Every call is written to ``llm/<seq>.json`` (request without the key,
response, usage, timing, attempts) and announced as an ``llm.call`` event.

Deferred (PLAN.md §8): a ``GatewayProvider`` over
``apps.v2.agent.gateway_clients.GatewayInvocationHttpClient`` for the
product's billing and admission path. The seam stays as it is.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from apps.v2.agent.paper2code.figures import redact_images
from apps.v2.agent_engine.paper2code.seams.llm_runtime import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)

THINKING_DISABLED: dict[str, Any] = {"type": "disabled"}
STANDARD_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)
DEFAULT_REQUEST_TIMEOUT_S = 600.0
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class ThinkingNotDisabled(RuntimeError):
    """The endpoint returned reasoning tokens although thinking was disabled."""


class ProviderConfigError(RuntimeError):
    """Missing key or unusable base URL; a programmer/operator error, not a model failure."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_delays(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name, "")
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                values.append(float(part))
            except ValueError:
                return default
    return tuple(values) or default


class RetryPolicy:
    """Delay sequence + stop rule for one ``chat_with_retry`` call."""

    def __init__(self, mode: str) -> None:
        self.mode = "persistent" if mode == "persistent" else "standard"
        if self.mode == "standard":
            self._delays: tuple[float, ...] = STANDARD_RETRY_DELAYS
            self.max_delay = max(self._delays)
            self.identical_limit = len(self._delays) + 1
            self.max_attempts: int | None = len(self._delays) + 1
        else:
            self._delays = _env_delays("DEEPCODE_CHAT_RETRY_DELAYS", (10.0, 30.0, 60.0, 180.0, 300.0))
            self.max_delay = float(_env_int("DEEPCODE_PERSISTENT_MAX_DELAY", 900))
            self.identical_limit = _env_int("DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT", 30)
            self.max_attempts = None
        self._identical = 0
        self._last_error: str | None = None
        self.attempts = 0

    def note_failure(self, error: str) -> bool:
        """Record a failure; return True when the caller may retry."""
        self.attempts += 1
        if error == self._last_error:
            self._identical += 1
        else:
            self._identical = 1
            self._last_error = error
        if self.max_attempts is not None and self.attempts >= self.max_attempts:
            return False
        return self._identical < self.identical_limit

    def delay(self) -> float:
        index = self.attempts - 1
        if self.mode == "standard":
            return self._delays[min(index, len(self._delays) - 1)]
        return min(self._delays[index % len(self._delays)], self.max_delay)


class ParateraProvider(LLMProvider):
    provider_name = "paratera"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        stream: bool = False,
        log_dir: Path | None = None,
        events: Callable[..., Any] | None = None,
        timeout_s: float | None = None,
        generation: GenerationSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        thinking: str = "disabled",
    ) -> None:
        super().__init__(generation)
        if thinking not in ("disabled", "enabled"):
            raise ProviderConfigError(f"thinking must be disabled or enabled, got {thinking!r}")
        self._thinking = thinking
        if not api_key:
            raise ProviderConfigError("provider api key is empty")
        if not base_url.startswith(("http://", "https://")):
            raise ProviderConfigError(f"unusable provider base_url {base_url!r}")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._stream = stream
        self._log_dir = log_dir
        self._events = events
        self._timeout_s = timeout_s or float(_env_int("DEEPCODE_OPENAI_REQUEST_TIMEOUT_S", int(DEFAULT_REQUEST_TIMEOUT_S)))
        self._sleep = sleep or asyncio.sleep
        self._seq = itertools.count(self._next_seq(log_dir))
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout_s, connect=30.0),
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        self.calls = 0
        self.total_usage: dict[str, int] = {}

    @classmethod
    def from_run(cls, run: Any, paths: Any, *, events: Callable[..., Any] | None = None, log_dir: Path | None = None, **kwargs: Any) -> "ParateraProvider":
        """``log_dir`` defaults to the run's ``llm/``; a second provider in the same run (the loopback
        endpoint the experiment agent talks to) logs under a sub-directory so the two sequences never collide."""
        key = os.environ.get(run.provider_key_env, "")
        if not key:
            raise ProviderConfigError(
                f"environment variable {run.provider_key_env} is not set; pass it with --env-file"
            )
        return cls(
            model=run.model,
            base_url=run.provider_base_url,
            api_key=key,
            stream=bool(run.provider_stream),
            thinking=getattr(run, "thinking", "disabled"),
            log_dir=log_dir if log_dir is not None else paths.llm_dir,
            events=events,
            **kwargs,
        )

    # -- LLMProvider -----------------------------------------------------------

    def get_default_model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload = self.build_payload(
            messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        policy = RetryPolicy(retry_mode)
        seq = next(self._seq)
        started = time.time()
        errors: list[str] = []
        while True:
            outcome = await self._attempt(payload)
            if outcome.response is not None:
                self.calls += 1
                self._merge_usage(outcome.response.usage)
                self._write_log(seq, payload, outcome, started, errors)
                return outcome.response
            error = outcome.error or "unknown error"
            errors.append(error)
            if not outcome.retryable or not policy.note_failure(error):
                logger.warning("LLM call {} failed for good after {} attempt(s): {}", seq, policy.attempts, error)
                final = LLMResponse(
                    content=error,
                    finish_reason="error",
                    error_status_code=outcome.status,
                    error_kind="provider",
                    error_should_retry=False,
                )
                self._write_log(seq, payload, outcome, started, errors, final=final)
                return final
            wait = policy.delay()
            note = f"LLM call {seq} attempt {policy.attempts} failed ({error[:160]}); retrying in {wait:.0f}s"
            logger.warning(note)
            if on_retry_wait is not None:
                try:
                    await on_retry_wait(note)
                except Exception as exc:
                    logger.debug("on_retry_wait raised: {}", exc)
            await self._sleep(wait)

    # -- request/response -------------------------------------------------------

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "max_tokens": int(max_tokens if max_tokens is not None else self.generation.max_tokens),
            "temperature": float(temperature if temperature is not None else self.generation.temperature),
            "thinking": {"type": self._thinking},  # the only form Paratera / DeepSeek honour
            "stream": bool(self._stream),
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format:  # JSON mode (DeepSeek honours {"type": "json_object"} with thinking on; 09-20 probe)
            payload["response_format"] = response_format
        if self._stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    class _Outcome:
        __slots__ = ("attempts_raw", "error", "response", "retryable", "status")

        def __init__(self) -> None:
            self.response: LLMResponse | None = None
            self.error: str | None = None
            self.retryable = False
            self.status: int | None = None
            self.attempts_raw: dict[str, Any] | None = None

    async def _attempt(self, payload: dict[str, Any]) -> "ParateraProvider._Outcome":
        outcome = self._Outcome()
        try:
            if payload.get("stream"):
                raw = await self._post_stream(payload)
            else:
                http = await self._client.post("/chat/completions", json=payload)
                outcome.status = http.status_code
                if http.status_code >= 400:
                    outcome.error = f"HTTP {http.status_code}: {http.text[:400]}"
                    outcome.retryable = http.status_code in RETRYABLE_STATUS
                    return outcome
                raw = http.json()
        except httpx.HTTPError as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            outcome.retryable = True
            return outcome
        except (ValueError, KeyError) as exc:
            outcome.error = f"malformed provider response: {exc}"
            outcome.retryable = True
            return outcome
        outcome.attempts_raw = raw
        try:
            response = self.parse_completion(raw)
        except ThinkingNotDisabled:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            outcome.error = f"malformed provider response: {exc}"
            outcome.retryable = True
            return outcome
        if not response.tool_calls and not (response.content or "").strip() and response.finish_reason not in {"error", "length"}:
            outcome.error = "empty response from provider"
            outcome.retryable = True
            return outcome
        outcome.response = response
        return outcome

    async def _post_stream(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Aggregate an SSE stream into one completion dict."""
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        async with self._client.stream("POST", "/chat/completions", json=payload) as http:
            if http.status_code >= 400:
                body = await http.aread()
                raise httpx.HTTPStatusError(
                    f"HTTP {http.status_code}: {body[:400]!r}", request=http.request, response=http
                )
            async for line in http.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    for tc in delta.get("tool_calls") or []:
                        index = int(tc.get("index", 0))
                        slot = tool_calls.setdefault(
                            index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return {
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage or {},
            "_streamed": True,
        }

    def parse_completion(self, raw: dict[str, Any]) -> LLMResponse:
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage_raw = raw.get("usage") or {}
        details = usage_raw.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens") or 0)
        reasoning_content = message.get("reasoning_content") or message.get("reasoning")
        reasoning_chars = len(reasoning_content) if isinstance(reasoning_content, str) else 0
        if reasoning_tokens != 0 and self._thinking == "disabled":
            raise ThinkingNotDisabled(
                f"model {raw.get('model', self._model)!r} returned reasoning_tokens={reasoning_tokens}; "
                "the run's caliber requires thinking off"
            )
        if reasoning_chars and self._thinking == "disabled":
            # Paratera's DeepSeek-V4-Flash sometimes fills reasoning_content while billing zero
            # reasoning tokens (sapg C9, 2026-09-17). The caliber is defined on the token count
            # (PLAN.md §0); the text is recorded (llm/<seq>.json reasoning_content_chars) and
            # only PAPER2CODE_STRICT_REASONING_CONTENT=1 turns it into an abort.
            if os.environ.get("PAPER2CODE_STRICT_REASONING_CONTENT") == "1":
                raise ThinkingNotDisabled(
                    f"model {raw.get('model', self._model)!r} returned {reasoning_chars} chars of reasoning_content"
                )
            logger.warning("model returned {} chars of reasoning_content with reasoning_tokens=0", reasoning_chars)
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
            "reasoning_tokens": reasoning_tokens,
            "reasoning_content_chars": reasoning_chars,
        }
        finish_reason = choice.get("finish_reason") or "stop"
        content = message.get("content")
        calls: list[ToolCallRequest] = []
        for index, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments")
            if isinstance(args_raw, dict):
                args = args_raw
            else:
                try:
                    args = json.loads(args_raw or "{}")
                except json.JSONDecodeError as exc:
                    return LLMResponse(
                        content=f"tool call {fn.get('name')!r} carried invalid JSON arguments: {exc}",
                        finish_reason="error",
                        usage=usage,
                        error_kind="tool_arguments",
                    )
                if not isinstance(args, dict):
                    args = {"value": args}
            calls.append(
                ToolCallRequest(
                    id=str(tc.get("id") or f"call_{index}"),
                    name=str(fn.get("name") or ""),
                    arguments=args,
                )
            )
        if calls and finish_reason != "error":
            finish_reason = "tool_calls"
        return LLMResponse(content=content, tool_calls=calls, finish_reason=finish_reason, usage=usage)

    # -- bookkeeping ------------------------------------------------------------

    @staticmethod
    def _next_seq(log_dir: Path | None) -> int:
        """Continue numbering after the calls an earlier process logged in ``log_dir``."""
        if log_dir is None or not log_dir.is_dir():
            return 1
        existing = [int(p.stem) for p in log_dir.glob("*.json") if p.stem.isdigit()]
        return (max(existing) + 1) if existing else 1

    def _merge_usage(self, usage: dict[str, int]) -> None:
        for key, value in usage.items():
            self.total_usage[key] = self.total_usage.get(key, 0) + int(value)

    def _write_log(
        self,
        seq: int,
        payload: dict[str, Any],
        outcome: "ParateraProvider._Outcome",
        started: float,
        errors: list[str],
        *,
        final: LLMResponse | None = None,
    ) -> None:
        response = final or outcome.response
        record = {
            "seq": seq,
            "ts": started,
            "duration_s": round(time.time() - started, 3),
            "model": payload.get("model"),
            "base_url": self._base_url,
            "thinking": self._thinking,
            "request": {k: v for k, v in payload.items() if k != "messages"},
            "messages": redact_images(payload.get("messages")),
            "raw_response": outcome.attempts_raw,
            "finish_reason": response.finish_reason if response else None,
            "usage": dict(response.usage) if response else {},
            "reasoning_tokens": (response.usage.get("reasoning_tokens", 0) if response else None),
            "attempts": len(errors) + (1 if outcome.response is not None else 0),
            "errors": errors,
        }
        if self._log_dir is not None:
            try:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                (self._log_dir / f"{seq:05d}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
                )
            except OSError as exc:
                logger.warning("could not write llm log {}: {}", seq, exc)
        if self._events is not None:
            self._events(
                "llm.call",
                seq=seq,
                model=record["model"],
                finish_reason=record["finish_reason"],
                usage=record["usage"],
                duration_s=record["duration_s"],
                attempts=record["attempts"],
                error=(errors[-1] if errors and outcome.response is None else None),
            )


__all__ = [
    "THINKING_DISABLED",
    "ParateraProvider",
    "ProviderConfigError",
    "RetryPolicy",
    "ThinkingNotDisabled",
]
