"""Optional shared circuit registries for Agent research dependencies."""

from __future__ import annotations

import atexit
import math
import os
from threading import Lock
import time
from urllib.parse import urlparse

from apps.v2.reliability.circuit import CircuitRegistry, DependencyUnavailable


_REGISTRIES: dict[str, CircuitRegistry] = {}
_REGISTRIES_LOCK = Lock()
_METRICS_SCRAPE_LOCK = Lock()


def optional_shared_circuit(
    local: CircuitRegistry,
    *,
    scope: str,
    lease_seconds: float,
) -> CircuitRegistry:
    """Return a Redis registry when explicitly configured, otherwise ``local``.

    Each scope has its own Redis namespace because RedisCircuitRegistry locks a
    namespace to one policy. Construction never initializes or resets state.
    """

    if (
        not scope
        or len(scope) > 48
        or not scope.isascii()
        or not all(character.isalnum() or character in "-_" for character in scope)
    ):
        raise ValueError("invalid research circuit scope")
    if not math.isfinite(lease_seconds) or not 30 <= lease_seconds <= 900:
        raise ValueError("research circuit lease must be between 30 and 900 seconds")
    shared_url = os.environ.get("DEEPEVOL_RESEARCH_CIRCUIT_REDIS_URL", "").strip()
    if not shared_url:
        _register(scope, local)
        return local
    parsed = urlparse(shared_url)
    if (
        parsed.scheme not in {"redis", "rediss"}
        or not parsed.hostname
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid research circuit Redis URL")
    base_namespace = os.environ.get(
        "DEEPEVOL_RESEARCH_CIRCUIT_NAMESPACE", "research-v2"
    ).strip()
    namespace = f"{base_namespace}-{scope}"
    if (
        not base_namespace
        or len(namespace) > 128
        or not namespace.isascii()
        or not all(character.isalnum() or character in "-_" for character in namespace)
    ):
        raise ValueError("invalid research circuit namespace")

    from redis import Redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    from apps.v2.reliability.coordinator_client import ProtectedCoordinatorClient
    from apps.v2.reliability.redis_circuit import RedisCircuitRegistry

    client = Redis.from_url(
        shared_url,
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
        max_connections=2,
        retry=Retry(NoBackoff(), 0),
        retry_on_error=[],
        health_check_interval=0,
    )
    try:
        registry = RedisCircuitRegistry(
            ProtectedCoordinatorClient(client, identity=shared_url),
            namespace=namespace,
            policy=local.policy,
            lease_seconds=lease_seconds,
        )
        _register(scope, registry)
        return registry
    except BaseException:
        client.close()
        raise


def _register(scope: str, registry: CircuitRegistry) -> None:
    with _REGISTRIES_LOCK:
        existing = _REGISTRIES.get(scope)
        if existing is not None and existing is not registry:
            close = getattr(existing, "close", None)
            if callable(close):
                close()
        _REGISTRIES[scope] = registry


def close_research_circuits() -> None:
    with _REGISTRIES_LOCK:
        values = list(_REGISTRIES.values())
        _REGISTRIES.clear()
    closed: set[int] = set()
    for registry in values:
        if id(registry) in closed:
            continue
        closed.add(id(registry))
        close = getattr(registry, "close", None)
        if callable(close):
            close()


def research_circuit_metrics() -> str:
    with _REGISTRIES_LOCK:
        entries = tuple(sorted(_REGISTRIES.items()))
    body = f"deepevol_research_circuit_scopes {len(entries)}\n"
    if len(entries) > 16 or not _METRICS_SCRAPE_LOCK.acquire(blocking=False):
        return body + "deepevol_research_circuit_scrape_available 0\n"
    complete = True
    deadline = time.monotonic() + 1.5
    try:
        for scope, registry in entries:
            if time.monotonic() >= deadline:
                complete = False
                break
            body += _registry_metrics(scope, registry)
        return body + (
            f"deepevol_research_circuit_scrape_available {1 if complete else 0}\n"
        )
    finally:
        _METRICS_SCRAPE_LOCK.release()


def _registry_metrics(scope: str, registry: CircuitRegistry) -> str:
    body = ""
    labels = f'{{scope="{scope}"}}'
    is_shared = registry.__class__.__name__ == "RedisCircuitRegistry"
    body += f"deepevol_research_circuit_shared{labels} {1 if is_shared else 0}\n"
    try:
        state = registry.snapshot()
    except DependencyUnavailable:
        if is_shared:
            body += f"deepevol_research_circuit_coordinator_available{labels} 0\n"
        return body
    if is_shared:
        body += f"deepevol_research_circuit_coordinator_available{labels} 1\n"
        body += (
            f"deepevol_research_circuit_recovering{labels} "
            f"{state.get('recovering', 0)}\n"
        )
    for field in (
        "keys",
        "open",
        "half_open",
        "in_flight",
        "rejected_total",
        "opened_total",
    ):
        body += f"deepevol_research_circuit_{field}{labels} {state[field]}\n"
    return body


atexit.register(close_research_circuits)
