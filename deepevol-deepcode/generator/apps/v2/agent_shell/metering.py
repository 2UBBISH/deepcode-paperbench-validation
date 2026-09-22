"""In-process model metering for lines that call the Gateway directly.

The chat graph, the paper/report/presentation lines and the research runner
build LangChain models over the run's Gateway backend rather than speaking
HTTP to the shell.  They still get the shell's accounting: the shell hands
them a :class:`MeteredGatewayBackend`, a transparent proxy whose ``invoke``
(and ``stream`` / ``astream`` when the backend has them) records the usage
the Gateway reports into the same :class:`~.ledger.ShellUsageLedger` the
HTTP face uses.  Everything else (evidence, attributes) passes straight
through, so the durable checkpointer and the lines see the backend unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from .ledger import ShellUsageLedger


def _usage_of(response: Any) -> Mapping[str, Any] | None:
    if isinstance(response, Mapping):
        usage = response.get("usage_metadata")
    else:
        usage = getattr(response, "usage_metadata", None)
    return usage if isinstance(usage, Mapping) else None


def _normalized(usage: Mapping[str, Any]) -> dict[str, int]:
    out = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        try:
            out[key] = int(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            out[key] = 0
    if not out["total_tokens"]:
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


class MeteredGatewayBackend:
    """``GatewayChatBackend`` proxy that books every call into the ledger."""

    def __init__(self, backend: Any, ledger: ShellUsageLedger) -> None:
        if not callable(getattr(backend, "invoke", None)):
            raise ValueError("Gateway backend must provide invoke()")
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_ledger", ledger)

    @property
    def wrapped(self) -> Any:
        return self._backend

    def invoke(self, request: Any) -> Any:
        try:
            response = self._backend.invoke(request)
        except Exception:
            self._ledger.record_llm(None, failed=True)
            raise
        usage = _usage_of(response)
        self._ledger.record_llm(_normalized(usage) if usage is not None else None)
        return response

    def _stream(self, method: Any) -> Any:
        def stream(request: Any) -> Iterator[Any]:
            booked = False
            try:
                for chunk in method(request):
                    usage = _usage_of(chunk)
                    if usage is not None and not booked:
                        self._ledger.record_llm(_normalized(usage))
                        booked = True
                    yield chunk
            except Exception:
                if not booked:
                    self._ledger.record_llm(None, failed=True)
                raise
            if not booked:
                self._ledger.record_llm(None)

        return stream

    def _astream(self, method: Any) -> Any:
        async def astream(request: Any) -> AsyncIterator[Any]:
            booked = False
            try:
                async for chunk in method(request):
                    usage = _usage_of(chunk)
                    if usage is not None and not booked:
                        self._ledger.record_llm(_normalized(usage))
                        booked = True
                    yield chunk
            except Exception:
                if not booked:
                    self._ledger.record_llm(None, failed=True)
                raise
            if not booked:
                self._ledger.record_llm(None)

        return astream

    def __getattr__(self, name: str) -> Any:
        # ``stream``/``astream`` exist on the proxy only when the wrapped
        # backend has them: LangChain probes with getattr and falls back to a
        # single chunk otherwise, and that fallback must stay reachable.
        value = getattr(self._backend, name)
        if name == "stream" and callable(value):
            return self._stream(value)
        if name == "astream" and callable(value):
            return self._astream(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._backend, name, value)

    def __repr__(self) -> str:
        return f"MeteredGatewayBackend({self._backend!r})"


__all__ = ["MeteredGatewayBackend"]
