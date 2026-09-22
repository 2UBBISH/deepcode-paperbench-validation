"""Single-attempt HTTP admission. Callers retain their domain error semantics."""

from __future__ import annotations

import hashlib
import asyncio
from collections.abc import Awaitable, Callable

import httpx

from .circuit import CircuitPolicy, CircuitRegistry


HTTP_CIRCUITS = CircuitRegistry(CircuitPolicy.from_env())


class HTTPDependencyProtocolError(ValueError):
    """A dependency returned a successful HTTP response with an invalid payload."""


async def guarded_async_request(
    dependency: str,
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    timeout_seconds: float = 30.0,
    circuits: CircuitRegistry = HTTP_CIRCUITS,
) -> httpx.Response:
    """Bound an async buffered request; cancellation closes HTTPX async I/O.

    Do not wrap to_thread/SDK futures: cancelling a future does not stop its
    underlying thread. Streaming callers must hold their own permit to close.
    """
    if not 0 < timeout_seconds <= 120:
        raise ValueError("async dependency timeout must be in (0, 120]")
    permit = circuits.acquire(hashlib.sha256(dependency.encode()).hexdigest())
    try:
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await send()
        except (httpx.TransportError, TimeoutError, HTTPDependencyProtocolError):
            permit.finish("failure")
            raise
        permit.finish("failure" if response.status_code in {408, 429} or response.status_code >= 500 else "success")
        return response
    finally:
        permit.finish()


def guarded_request(
    dependency: str,
    send: Callable[[], httpx.Response],
    *,
    circuits: CircuitRegistry = HTTP_CIRCUITS,
) -> httpx.Response:
    # Hash configured route identities; never export URLs, credentials or users.
    key = hashlib.sha256(dependency.encode()).hexdigest()
    permit = circuits.acquire(key)
    try:
        try:
            response = send()
        except (httpx.TransportError, HTTPDependencyProtocolError):
            permit.finish("failure")
            raise
        permit.finish("failure" if response.status_code in {408, 429} or response.status_code >= 500 else "success")
        return response
    finally:
        permit.finish()
