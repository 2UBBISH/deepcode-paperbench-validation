"""Bounded control receipts; callers own the surrounding stream context."""

from contextlib import nullcontext

import httpx
import os
from urllib.parse import urlparse

from .http_client import DependencyHTTPClient, mark_response_failure
from .deadline_transport import DeadlineHTTPTransport


def read_control_response(
    response: httpx.Response, *, maximum_bytes: int, accepted_statuses: frozenset[int] = frozenset({200}), validate_success=None
) -> httpx.Response:
    validation = response.extensions.get("dependency_response_validation")
    with validation() if callable(validation) and validate_success is not None else nullcontext():
        success = response.status_code in accepted_statuses
        raw = bytearray()
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in {"", "identity"}:
            if success:
                mark_response_failure(response)
                raise httpx.ReadError("Control response compression is unsupported", request=response.request)
        else:
            for chunk in response.iter_bytes():
                if len(chunk) > maximum_bytes - len(raw):
                    if success:
                        mark_response_failure(response)
                        raise httpx.ReadError("Control response exceeds its byte limit", request=response.request)
                    raw.clear()
                    break
                raw.extend(chunk)
        headers = dict(response.headers)
        headers.pop("content-length", None)
        headers.pop("content-encoding", None)
        # A discarded error body must retain its status and cannot resemble success.
        body = bytes(raw) if raw or success else b"{}"
        rebuilt = httpx.Response(response.status_code, headers=headers, content=body, request=response.request)
        if success and validate_success is not None:
            validate_success(rebuilt)
        return rebuilt


def build_control_http_client(dependency: str, *, timeout_seconds: float, coordinator_read_fallback_paths=frozenset(), limits: httpx.Limits | None = None) -> DependencyHTTPClient:
    shared_url = os.environ.get("DEEPEVOL_HTTP_CIRCUIT_REDIS_URL", "").strip()
    circuits = None
    if shared_url:
        parsed = urlparse(shared_url)
        if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
            raise ValueError("invalid HTTP circuit Redis URL")
        from redis import Redis
        from redis.backoff import NoBackoff
        from redis.retry import Retry
        from .redis_circuit import RedisCircuitRegistry
        from .coordinator_client import ProtectedCoordinatorClient
        from .circuit import CircuitPolicy

        namespace = os.environ.get("DEEPEVOL_HTTP_CIRCUIT_NAMESPACE", "http-v2")
        if not namespace or len(namespace) > 128 or not namespace.isascii() or not all(c.isalnum() or c in "-_" for c in namespace):
            raise ValueError("invalid HTTP circuit namespace")
        policy = CircuitPolicy.from_env()
        redis = Redis.from_url(shared_url, socket_timeout=0.5, socket_connect_timeout=0.5,
                              max_connections=2, retry=Retry(NoBackoff(), 0), retry_on_error=[],
                              health_check_interval=0)
        circuits = RedisCircuitRegistry(ProtectedCoordinatorClient(redis, identity=shared_url), namespace=namespace, policy=policy, lease_seconds=900)
        # Missing state must enter quarantine, never reset on runtime startup.
    try:
        client = DependencyHTTPClient(
            dependency=dependency,
            coordinator_read_fallback_paths=coordinator_read_fallback_paths,
            trust_env=False,
            timeout=httpx.Timeout(timeout_seconds, connect=3, pool=1),
            transport=DeadlineHTTPTransport(timeout_seconds=timeout_seconds, limits=limits),
            **({"limits": limits} if limits is not None else {}),
            **({"circuits": circuits, "owns_circuits": True} if circuits is not None else {}),
        )
        if circuits is not None:
            from .control_metrics import register_shared_client

            register_shared_client(client, shared_url + "\0" + namespace)
        return client
    except BaseException:
        if circuits is not None:
            circuits.close()
        raise
