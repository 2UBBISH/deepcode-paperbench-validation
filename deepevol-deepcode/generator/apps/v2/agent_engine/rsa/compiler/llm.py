"""A small OpenAI-compatible client for the Compiler.

Deliberately not SetupX's. `src/llm_engine.py:124-127` hardcodes
`max_tokens: 4096` and nothing in the codebase inspects `finish_reason`. For the
setup loop that is survivable -- a truncated action is a bad action and the loop
takes another step. For the Compiler it is not: a criterion truncated mid-line is
frozen, and then presents downstream as a wide FAIL rather than as an error,
which is precisely the failure shape that is hardest to notice.

So this client asserts `finish_reason == "stop"`, and records token usage
(SetupX reads the `usage` block and discards it at llm_engine.py:140-151) because
the Router's budget terminal state needs a live number.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx


class LLMError(RuntimeError):
    pass


class TruncatedError(LLMError):
    """The model stopped because it ran out of room, not because it was done."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0

    def add(self, block: dict) -> None:
        self.calls += 1
        self.prompt_tokens += int(block.get("prompt_tokens") or 0)
        self.completion_tokens += int(block.get("completion_tokens") or 0)
        details = block.get("prompt_tokens_details") or {}
        self.cached_tokens += int(details.get("cached_tokens") or 0)

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cny(self) -> float:
        """Priced with harness/cost.py's rates: 1.00 / 0.02 / 2.00 CNY per M."""
        fresh = max(self.prompt_tokens - self.cached_tokens, 0)
        return round(fresh / 1e6 * 1.00 + self.cached_tokens / 1e6 * 0.02
                     + self.completion_tokens / 1e6 * 2.00, 4)


@dataclass
class LLM:
    base_url: str
    api_key: str
    model: str
    # Keep room for the compiler prompt on providers whose context window is
    # 8192 tokens (the gemma endpoint rejects 8192 output tokens plus any input).
    # SetupX itself uses 4096, and a complete criterion JSON fits comfortably.
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: float = 300.0
    retries: int = 4
    usage: Usage = field(default_factory=Usage)

    @staticmethod
    def from_env() -> "LLM":
        """Read the endpoint SetupX is configured with.

        Call inside `rsa.setupx_interop.setupx_configured(...)` so `.env.local`
        has already been written and `.env` cannot override it -- outside that
        block the values are whatever a previous crashed run left behind.
        """
        base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", "")
        if not (base and model):
            raise LLMError(
                "OPENAI_BASE_URL / OPENAI_MODEL are unset. The Compiler reads the "
                "same endpoint SetupX uses; configure it with "
                "rsa.setupx_interop.setupx_configured(backend=...)."
            )
        return LLM(base_url=base, api_key=key, model=model)

    def chat(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last = ""
        for attempt in range(self.retries):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                               headers=headers, timeout=self.timeout)
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code != 200:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:600]}")

            body = r.json()
            choice = (body.get("choices") or [{}])[0]
            self.usage.add(body.get("usage") or {})
            reason = choice.get("finish_reason")
            content = (choice.get("message") or {}).get("content") or ""
            if reason == "length":
                raise TruncatedError(
                    f"the model hit max_tokens ({payload['max_tokens']}) and the reply "
                    "is incomplete. A truncated criterion freezes as a wide FAIL "
                    "rather than an error, so it is refused here."
                )
            if not content.strip():
                last = f"empty reply (finish_reason={reason})"
                time.sleep(min(2 ** attempt, 20))
                continue
            return content
        raise LLMError(f"no usable reply after {self.retries} attempts: {last}")

    def json(self, system: str, user: str, **kw) -> dict:
        return extract_json(self.chat(system, user, **kw))


_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_json(text: str) -> dict:
    """Recover a JSON object from a chatty reply.

    Same ladder SetupX uses in three places (verifier/prosecutor/judge each carry
    their own copy): strip reasoning tags, try the whole string, then a fenced
    block, then the outermost braces. Copied rather than imported so this package
    does not depend on SetupX being importable.
    """
    s = _THINK.sub("", text).strip()
    for candidate in _json_candidates(s):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise LLMError(f"no JSON object in the reply; first 600 chars:\n{text[:600]}")


def _json_candidates(s: str):
    yield s
    for m in _FENCE.finditer(s):
        yield m.group(1).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        yield s[start:end + 1]
