"""Synchronous HTTPX facade over cancellable async network I/O.

One event-loop thread per transport, no executor thread per request. Deadlines
span connection acquisition, headers and every body byte, including slow drip
responses. Cancelling HTTPX async I/O closes the active connection. Never retries.
"""
from __future__ import annotations

import asyncio
import math
from contextvars import ContextVar
from threading import Event, Lock, Thread
import time
from typing import Any

import httpx


_CALLER_EGRESS_DEADLINE: ContextVar[float | None] = ContextVar(
    "deepevol_http_caller_egress_deadline",
    default=None,
)


class _CallerEgressDeadlineScope:
    def __init__(self, deadline_monotonic: float) -> None:
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("HTTP caller deadline must be finite")
        self.deadline = float(deadline_monotonic)
        self.token: Any | None = None

    def __enter__(self) -> None:
        if self.token is not None:
            raise RuntimeError("HTTP caller deadline scope cannot be reused")
        current = _CALLER_EGRESS_DEADLINE.get()
        effective = self.deadline if current is None else min(self.deadline, current)
        self.token = _CALLER_EGRESS_DEADLINE.set(effective)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self.token is None:
            raise RuntimeError("HTTP caller deadline scope was not entered")
        _CALLER_EGRESS_DEADLINE.reset(self.token)
        return False


def http_egress_deadline(deadline_monotonic: float) -> _CallerEgressDeadlineScope:
    """Bound every nested non-renewable V2 HTTP transport to one caller deadline."""

    return _CallerEgressDeadlineScope(deadline_monotonic)


def http_egress_deadline_limit() -> float | None:
    """Return the active caller deadline for integration checks and local waits."""

    return _CALLER_EGRESS_DEADLINE.get()


class HTTPReadCancelled(Exception):
    """Local cancellation, deliberately not a dependency transport failure."""


class DeadlineHTTPTransport(httpx.BaseTransport):
    def __init__(self, *, timeout_seconds: float = 300, transport=None, limits=None, renewable: bool = False):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("HTTP total timeout must be positive and finite")
        self.timeout_seconds = timeout_seconds
        self.renewable = renewable
        self._transport = transport or httpx.AsyncHTTPTransport(
            retries=0, trust_env=False, limits=limits or httpx.Limits(
                max_connections=16, max_keepalive_connections=8,
            ),
        )
        self._lock = Lock()
        self._closed = False
        self._loop = None
        self._thread = None

    def _run(self, coroutine):
        with self._lock:
            if self._closed:
                coroutine.close()
                raise RuntimeError("HTTP transport is closed")
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = Thread(target=self._loop.run_forever, name="provider-http-io", daemon=True)
                self._thread.start()
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    @staticmethod
    async def _before(operation, deadline, cancel_event=None):
        if time.monotonic() >= deadline:
            raise httpx.ReadTimeout("Provider HTTP total deadline exceeded")
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                if cancel_event is None:
                    return await operation()
                if cancel_event.is_set():
                    raise HTTPReadCancelled("HTTP read cancelled")
                pending = asyncio.create_task(operation())
                try:
                    while not pending.done():
                        await asyncio.wait({pending}, timeout=0.05)
                        if pending.done():
                            break
                        if cancel_event.is_set():
                            raise HTTPReadCancelled("HTTP read cancelled")
                    return await pending
                finally:
                    if not pending.done():
                        pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
        except TimeoutError:
            raise httpx.ReadTimeout("Provider HTTP total deadline exceeded") from None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        budget = request.extensions.get("total_timeout_seconds", self.timeout_seconds)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            raise ValueError("HTTP total timeout must be positive and finite")
        deadline = time.monotonic() + min(budget, self.timeout_seconds)
        absolute = request.extensions.get("egress_deadline_monotonic")
        caller_absolute = _CALLER_EGRESS_DEADLINE.get()
        if caller_absolute is not None:
            if absolute is not None and (
                isinstance(absolute, bool)
                or not isinstance(absolute, (int, float))
                or not math.isfinite(absolute)
            ):
                raise ValueError("invalid absolute HTTP deadline")
            absolute = (
                caller_absolute
                if absolute is None
                else min(float(absolute), caller_absolute)
            )
        if absolute is not None:
            if self.renewable:
                raise ValueError("absolute HTTP deadlines cannot be renewed")
            if isinstance(absolute, bool) or not isinstance(absolute, (int, float)) or not math.isfinite(absolute):
                raise ValueError("invalid absolute HTTP deadline")
            deadline = min(deadline, absolute)
        cancel_event = request.extensions.get("read_cancel_event")
        if cancel_event is not None and not isinstance(cancel_event, Event):
            raise ValueError("read cancellation requires threading.Event")
        outgoing = httpx.Request(
            request.method, request.url, headers=request.headers,
            content=request.read(), extensions=request.extensions,
        )
        response = self._run(self._before(lambda: self._transport.handle_async_request(outgoing), deadline, cancel_event))
        stream = _DeadlineStream(self, response.stream, deadline, cancel_event)
        extensions = {key: value for key, value in response.extensions.items() if key != "network_stream"}
        if self.renewable:
            extensions["renew_stream_deadline"] = lambda: stream.renew(min(budget, self.timeout_seconds))
        return httpx.Response(response.status_code, headers=response.headers, stream=stream, extensions=extensions)

    async def _shutdown(self):
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self._transport.aclose()
        await asyncio.sleep(0)
        await asyncio.get_running_loop().shutdown_asyncgens()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
            if loop is None:
                # No operation has created a network resource yet.
                return
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
        try:
            future.result()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()


class _DeadlineStream(httpx.SyncByteStream):
    def __init__(self, owner: DeadlineHTTPTransport, stream: Any, deadline: float, cancel_event=None):
        self.owner = owner
        self.stream = stream
        self.deadline = deadline
        self.cancel_event = cancel_event
        self.closed = False
        self.expired = False
        self._closing = None
        self._timer = owner._run(self._arm())

    async def _arm(self):
        self._timer = asyncio.create_task(self._expire())
        return self._timer

    async def _expire(self):
        while time.monotonic() < self.deadline:
            if self.cancel_event is not None and self.cancel_event.is_set():
                break
            remaining = max(0, self.deadline - time.monotonic())
            await asyncio.sleep(min(0.05, remaining) if self.cancel_event is not None else remaining)
        self.expired = self.cancel_event is None or not self.cancel_event.is_set()
        try:
            await self._aclose()
        except Exception:
            # The consumer will see the deadline; don't leave an unobserved
            # task exception if an abandoned response also fails during close.
            pass

    def renew(self, seconds):
        self.owner._run(self._renew(seconds))

    async def _renew(self, seconds):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise HTTPReadCancelled("HTTP read cancelled")
        if self.closed or self.expired or time.monotonic() >= self.deadline:
            raise httpx.ReadTimeout("HTTP frame deadline exceeded")
        self._timer.cancel()
        self.deadline = time.monotonic() + seconds
        await self._arm()

    async def _aclose(self):
        if self._closing is None:
            self.closed = True
            if asyncio.current_task() is not self._timer:
                self._timer.cancel()
            self._closing = asyncio.create_task(self.stream.aclose())
        # A consumer closing concurrently with the timer waits for the same
        # close operation; cancellation must not interrupt socket cleanup.
        await asyncio.shield(self._closing)

    def __iter__(self):
        iterator = self.stream.__aiter__()
        try:
            while True:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise HTTPReadCancelled("HTTP read cancelled")
                if self.expired:
                    raise httpx.ReadTimeout("Provider HTTP total deadline exceeded")
                try:
                    yield self.owner._run(self.owner._before(lambda: anext(iterator), self.deadline, self.cancel_event))
                except StopAsyncIteration:
                    return
                except httpx.TransportError:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        raise HTTPReadCancelled("HTTP read cancelled") from None
                    if self.expired:
                        raise httpx.ReadTimeout("Provider HTTP total deadline exceeded") from None
                    raise
        finally:
            self.close()

    def close(self):
        try:
            self.owner._run(self._aclose())
        except RuntimeError:
            if not self.owner._closed:
                raise
