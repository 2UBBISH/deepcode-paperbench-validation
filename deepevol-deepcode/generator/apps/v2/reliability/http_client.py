"""HTTPX client with bounded dependency admission, including streamed bodies.

Use for internal service adapters. Billable Provider egress must continue to
use Gateway's pre-SEND_INTENT admission instead of this HTTP-level guard.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

import httpx

from .circuit import DependencyUnavailable, Permit
from .http_dependency import HTTP_CIRCUITS
from .deadline_transport import DeadlineHTTPTransport


class DependencyOperation(Enum):
    """Finite internal operation identities; never derive from request input."""

    MEMOS_SEARCH = "memos-search"
    MEMOS_ADD = "memos-add"


class DependencyHTTPUnavailable(httpx.ConnectError):
    """Proven pre-send refusal, preserving HTTPX callers' failure handling."""

    def __init__(self, error: DependencyUnavailable, request: httpx.Request) -> None:
        super().__init__(error.reason, request=request)
        self.reason = error.reason
        self.retry_after = error.retry_after


class _Stream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, permit: Permit, outcome: str) -> None:
        self.stream = stream
        self.permit = permit
        self.outcome = outcome
        self.finished_reading = False
        self.closed = False
        self.validating = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self.stream
            self.finished_reading = True
        except httpx.TransportError:
            self.outcome = "failure"
            try:
                self.close()
            except Exception:
                pass
            raise

    def mark_failed(self) -> None:
        if not self.closed or self.validating:
            self.outcome = "failure"

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.stream.close()
        finally:
            # Cancellation before EOF is neither a provider success nor failure.
            if not self.validating:
                self._finish()

    def _finish(self) -> None:
        self.permit.finish(self.outcome if self.finished_reading or self.outcome == "failure" else "neutral")

    @contextmanager
    def validation(self):
        self.validating = True
        try:
            yield
        except Exception:
            self.mark_failed()
            raise
        finally:
            self.validating = False
            if self.closed:
                self._finish()


class DependencyHTTPClient(httpx.Client):
    def __init__(self, *, dependency: str, circuits=HTTP_CIRCUITS, owns_circuits: bool = False, coordinator_read_fallback_paths=frozenset(), **kwargs: Any) -> None:
        if not dependency or len(dependency) > 128:
            raise ValueError("dependency label must be bounded")
        self.dependency = dependency
        self.circuits = circuits
        self._owns_circuits = owns_circuits
        self.coordinator_read_fallback_paths = frozenset(coordinator_read_fallback_paths)
        kwargs.setdefault("limits", httpx.Limits(max_connections=16, max_keepalive_connections=8))
        kwargs.setdefault("timeout", httpx.Timeout(15.0, connect=3.0, pool=1.0))
        super().__init__(**kwargs)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._owns_circuits:
                self._owns_circuits = False
                self.circuits.close()

    def send(self, request: httpx.Request, *, stream: bool = False, **kwargs: Any) -> httpx.Response:
        # No query, authorization header, caller id or object path enters a key.
        origin = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        operation = request.extensions.get("dependency_operation")
        if operation is not None and not isinstance(operation, DependencyOperation):
            raise ValueError("dependency operation must be a fixed DependencyOperation")
        identity = f"{self.dependency}\0{origin}\0{request.method}"
        if operation is not None:
            identity += f"\0{operation.value}"
        key = hashlib.sha256(identity.encode()).hexdigest()
        request.extensions.pop("dependency_coordinator_fallback", None)
        try:
            permit = self.circuits.acquire(key)
        except DependencyUnavailable as exc:
            if not (exc.reason == "DEPENDENCY_COORDINATOR_UNAVAILABLE" and request.method == "GET"
                    and not request.url.query and request.url.path in self.coordinator_read_fallback_paths):
                raise DependencyHTTPUnavailable(exc, request) from None
            from .emergency_read import EMERGENCY_READS

            try:
                permit = EMERGENCY_READS.acquire(key)
            except DependencyUnavailable as refused:
                raise DependencyHTTPUnavailable(refused, request) from None
            request.extensions["dependency_coordinator_fallback"] = True
        handed_off = False
        try:
            if getattr(permit, "egress_deadline_monotonic", None) is not None:
                transport = self._transport_for_url(request.url)
                redirects = kwargs.get("follow_redirects")
                if not isinstance(redirects, bool):
                    redirects = self.follow_redirects
                if redirects:
                    raise DependencyUnavailable("DEPENDENCY_SHARED_REDIRECT_UNSUPPORTED", 1)
                if not isinstance(transport, DeadlineHTTPTransport) or transport.renewable:
                    raise DependencyUnavailable("DEPENDENCY_SHARED_DEADLINE_UNSUPPORTED", 1)
                deadline = permit.validate()
                existing = request.extensions.get("egress_deadline_monotonic")
                # The transport validates caller-provided deadline types. Never
                # lengthen a caller's existing operation budget.
                if existing is not None:
                    if isinstance(existing, bool) or not isinstance(existing, (int, float)) or not math.isfinite(existing):
                        raise ValueError("invalid absolute HTTP deadline")
                    deadline = min(deadline, existing)
                request.extensions["egress_deadline_monotonic"] = deadline
            response = super().send(request, stream=stream, **kwargs)
            if request.extensions.get("dependency_coordinator_fallback"):
                response.extensions["dependency_coordinator_fallback"] = True
            outcome = "failure" if response.status_code in {408, 429} or response.status_code >= 500 else "success"
            if stream and not response.is_closed:
                guarded = _Stream(response.stream, permit, outcome)
                response.stream = guarded
                response.extensions["dependency_response_failure"] = guarded.mark_failed
                response.extensions["dependency_response_validation"] = guarded.validation
                handed_off = True
            else:
                permit.finish(outcome)
            return response
        except DependencyUnavailable as exc:
            raise DependencyHTTPUnavailable(exc, request) from None
        except httpx.TransportError:
            permit.finish("failure")
            raise
        finally:
            if not handed_off:
                permit.finish()


def mark_response_failure(response: httpx.Response) -> None:
    """Mark a streaming peer protocol violation before closing its response.

    Plain HTTP clients have no dependency permit. Cancellation alone remains
    neutral; adapters call this only for an invalid successful peer response.
    """
    mark = response.extensions.get("dependency_response_failure")
    if callable(mark):
        mark()
