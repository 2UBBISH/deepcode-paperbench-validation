"""Bounded reads for durable event acknowledgement envelopes."""

from typing import Any
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from uuid import UUID

from apps.common.v2_ids import parse_typed_id

import httpx
from .control_response import build_control_http_client
from .http_client import DependencyHTTPClient, mark_response_failure


def build_event_http_client(dependency: str) -> DependencyHTTPClient:
    return build_control_http_client(dependency, timeout_seconds=15)


def post_event(client: httpx.Client, url: str, *, validate_success: Callable[[httpx.Response], bool] | None = None, validation_statuses: frozenset[int] = frozenset({200}), **kwargs: Any) -> httpx.Response:
    headers = dict(kwargs.pop("headers", {}))
    headers["Accept-Encoding"] = "identity"
    with client.stream("POST", url, headers=headers, **kwargs) as response:
        validation = response.extensions.get("dependency_response_validation")
        with validation() if validation is not None else nullcontext():
            raw = bytearray()
            compressed = response.headers.get("content-encoding", "identity").strip().lower() != "identity"
            if compressed:
                if response.status_code == 200:
                    mark_response_failure(response)
                    raise httpx.ReadError("Event acknowledgement must be uncompressed", request=response.request)
            else:
                for chunk in response.iter_bytes():
                    if len(raw) + len(chunk) > 64 * 1024:
                        if response.status_code == 200:
                            mark_response_failure(response)
                            raise httpx.ReadError("Event acknowledgement exceeds limit", request=response.request)
                        raw.clear()
                        break
                    raw.extend(chunk)
            bounded_headers = dict(response.headers)
            bounded_headers.pop("content-encoding", None)
            bounded_headers.pop("content-length", None)
            rebuilt = httpx.Response(
                response.status_code, headers=bounded_headers, content=bytes(raw), request=response.request
            )
            if response.status_code in validation_statuses and validate_success is not None and not validate_success(rebuilt):
                mark_response_failure(response)
            return rebuilt


def valid_event_ack(response: httpx.Response, event_id: UUID) -> bool:
    """Count malformed success receipts before releasing the circuit permit."""
    try:
        document = response.json()
        if not isinstance(document, Mapping):
            return False
        state = document.get("effect_state")
        if response.status_code == 409 and state != "DEFERRED_GAP":
            return True  # Ordinary conflict rejections are not dependency outages.
        sequence = document.get("expected_next_seq")
        return (
            isinstance(state, str)
            and state in {"APPLIED", "ALREADY_RECORDED", "DEFERRED_GAP"}
            and isinstance(sequence, int) and not isinstance(sequence, bool) and sequence > 0
            and parse_typed_id(document["event_id"], expected_prefix="evt") == event_id
        )
    except (KeyError, TypeError, ValueError):
        return False
