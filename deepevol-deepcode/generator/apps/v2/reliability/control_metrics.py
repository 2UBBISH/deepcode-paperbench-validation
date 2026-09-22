"""HTTP breaker telemetry, with shared coordinators deduplicated per scrape."""

import hashlib
from threading import Lock
from weakref import WeakKeyDictionary

from .circuit import DependencyUnavailable
from .http_dependency import HTTP_CIRCUITS

_CLIENTS = WeakKeyDictionary()
_CLIENTS_LOCK = Lock()
_SCRAPE_LOCK = Lock()
_FIELDS = ("keys", "open", "half_open", "in_flight", "rejected_total", "opened_total")


def register_shared_client(client, identity: str) -> None:
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    with _CLIENTS_LOCK:
        _CLIENTS[client] = digest


def control_http_metrics() -> str:
    from .emergency_read import EMERGENCY_READS

    body = HTTP_CIRCUITS.prometheus_metrics(component="http") + EMERGENCY_READS.metrics()
    with _CLIENTS_LOCK:
        # Several pools commonly target the same coordinator. Its counters
        # describe shared state, so neither query nor sum them per client.
        registries = {identity: client.circuits for client, identity in list(_CLIENTS.items()) if not client.is_closed}
    body += f"deepevol_http_shared_coordinators {len(registries)}\n"
    if not registries:
        return body
    if len(registries) > 4 or not _SCRAPE_LOCK.acquire(blocking=False):
        return body + "deepevol_http_shared_scrape_available 0\n"
    try:
        body += "deepevol_http_shared_scrape_available 1\n"
        for identity, registry in sorted(registries.items()):
            labels = f'{{coordinator="{identity}"}}'
            try:
                state = registry.snapshot()
            except DependencyUnavailable:
                body += f"deepevol_http_shared_coordinator_available{labels} 0\n"
                continue
            body += f"deepevol_http_shared_coordinator_available{labels} 1\n"
            for field in _FIELDS:
                body += f"deepevol_http_shared_circuit_{field}{labels} {state[field]}\n"
        return body
    finally:
        _SCRAPE_LOCK.release()
