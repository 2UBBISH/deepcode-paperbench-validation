"""Count tokens as they are spent, so the budget terminal state is real.

SetupX receives the `usage` block on every reply and drops it
(`llm_engine.py:140-151`), so there is no live number to enforce a budget
against. The harness solves this with a local proxy the run is pointed at; that
works, but it adds a process, a port, and a `.env.local` that a crashed run
leaves behind aimed at a dead endpoint -- which is exactly the stale file sitting
in the tree today.

Since the loop is driven in-process here, the cheaper answer is to wrap the one
`httpx.Client.post` the client already owns. No proxy, no port, and the numbers
are the provider's own rather than an estimate.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# harness/cost.py's rates, in CNY per million tokens.
PRICE_INPUT = 1.00
PRICE_CACHED = 0.02
PRICE_OUTPUT = 2.00


@dataclass
class TokenMeter:
    prompt: int = 0
    cached: int = 0
    completion: int = 0
    calls: int = 0
    by_role: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: dict, role: str = "setup") -> None:
        if not usage:
            return
        with self._lock:
            self.calls += 1
            self.prompt += int(usage.get("prompt_tokens") or 0)
            self.completion += int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            self.cached += int(details.get("cached_tokens") or 0)
            n = int(usage.get("total_tokens") or 0) or (
                int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
            self.by_role[role] = self.by_role.get(role, 0) + n

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def cny(self) -> float:
        fresh = max(self.prompt - self.cached, 0)
        return round(fresh / 1e6 * PRICE_INPUT + self.cached / 1e6 * PRICE_CACHED
                     + self.completion / 1e6 * PRICE_OUTPUT, 4)

    def to_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt,
                "cached_tokens": self.cached, "completion_tokens": self.completion,
                "total_tokens": self.total, "cny": self.cny(),
                "by_role": dict(self.by_role)}


def attach(openai_client, meter: TokenMeter, role: str = "setup") -> None:
    """Meter one `OpenAICompatibleClient` in place.

    Wraps the httpx client's `post`, which is the single point every agent's LLM
    traffic passes through (`llm_engine.py:133`). Idempotent: attaching twice
    would otherwise double-count every call.
    """
    inner = getattr(openai_client, "_client", None)
    if inner is None or getattr(inner, "_rsa_metered", False):
        return

    original = inner.post

    def post(*args, **kwargs):
        response = original(*args, **kwargs)
        try:
            meter.add((response.json() or {}).get("usage") or {}, role)
        except Exception:
            # Metering must never be able to fail a run. A reply that is not JSON
            # is the client's problem to report, not the meter's.
            pass
        return response

    inner.post = post
    inner._rsa_metered = True
