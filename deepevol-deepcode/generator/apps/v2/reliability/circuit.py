"""Process-local admission control; never retries or hides an external effect.

Permits span the entire operation (including streaming). No thread waits for a
permit. Epoch fencing prevents late successes from closing a newer circuit.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from threading import Lock
from collections.abc import Callable


class DependencyUnavailable(RuntimeError):
    """Admission refused before any external request was sent."""

    def __init__(self, reason: str, retry_after: float) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = max(1, math.ceil(retry_after))


@dataclass(frozen=True)
class CircuitPolicy:
    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    max_in_flight: int = 8
    max_keys: int = 1024

    @classmethod
    def from_env(cls) -> CircuitPolicy:
        return cls(
            failure_threshold=int(os.environ.get("DEEPEVOL_CIRCUIT_FAILURE_THRESHOLD", "5")),
            recovery_seconds=float(os.environ.get("DEEPEVOL_CIRCUIT_RECOVERY_SECONDS", "30")),
            max_in_flight=int(os.environ.get("DEEPEVOL_CIRCUIT_MAX_IN_FLIGHT", "8")),
            max_keys=int(os.environ.get("DEEPEVOL_CIRCUIT_MAX_KEYS", "1024")),
        )

    def __post_init__(self) -> None:
        for value in (self.failure_threshold, self.max_in_flight, self.max_keys):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("circuit limits must be positive integers")
        if not math.isfinite(self.recovery_seconds) or self.recovery_seconds <= 0:
            raise ValueError("circuit recovery time must be positive and finite")


@dataclass
class _State:
    failures: int = 0
    in_flight: int = 0
    epoch: int = 0
    open_until: float | None = None
    probing: bool = False


class Permit:
    def __init__(self, registry: CircuitRegistry, key: str, epoch: int, probe: bool) -> None:
        self.registry = registry
        self.key = key
        self.epoch = epoch
        self.probe = probe
        self.finished = False

    def finish(self, outcome: str = "neutral") -> None:
        """Release once: success, failure, or neutral (cancelled/not sent)."""
        if outcome not in {"success", "failure", "neutral"}:
            raise ValueError("invalid circuit outcome")
        self.registry._finish(self, outcome)


class CircuitRegistry:
    def __init__(self, policy: CircuitPolicy = CircuitPolicy(), *, clock: Callable[[], float] = time.monotonic):
        self.policy = policy
        self.clock = clock
        self._lock = Lock()
        self._states: dict[str, _State] = {}
        self._rejected = 0
        self._opened = 0

    def acquire(self, key: str) -> Permit:
        if not isinstance(key, str) or not key or len(key) > 2048:
            raise ValueError("invalid dependency key")
        with self._lock:
            now = self.clock()
            state = self._states.get(key)
            if state is None:
                if len(self._states) >= self.policy.max_keys:
                    # Only evict fully healthy idle entries; never forget an outage.
                    idle = next((k for k, s in self._states.items()
                                 if not s.in_flight and s.open_until is None and not s.failures), None)
                    if idle is None:
                        self._reject("DEPENDENCY_REGISTRY_FULL", 1)
                    del self._states[idle]
                state = self._states[key] = _State()
            if state.open_until is not None and (now < state.open_until or state.probing):
                self._reject("DEPENDENCY_CIRCUIT_OPEN", state.open_until - now)
            if state.in_flight >= self.policy.max_in_flight:
                self._reject("DEPENDENCY_CAPACITY_EXHAUSTED", 1)
            probe = state.open_until is not None
            if probe:
                state.probing = True
            state.in_flight += 1
            return Permit(self, key, state.epoch, probe)

    def check_available(self, key: str) -> None:
        """Read-only routing hint; dispatch must still acquire its own permit."""
        if not isinstance(key, str) or not key or len(key) > 2048:
            raise ValueError("invalid dependency key")
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return
            now = self.clock()
            if state.open_until is not None and (now < state.open_until or state.probing):
                raise DependencyUnavailable("DEPENDENCY_CIRCUIT_OPEN", state.open_until - now)
            if state.in_flight >= self.policy.max_in_flight:
                raise DependencyUnavailable("DEPENDENCY_CAPACITY_EXHAUSTED", 1)

    def _reject(self, reason: str, delay: float) -> None:
        self._rejected += 1
        raise DependencyUnavailable(reason, delay)

    def _finish(self, permit: Permit, outcome: str) -> None:
        with self._lock:
            if permit.finished:
                return
            permit.finished = True
            state = self._states[permit.key]
            state.in_flight -= 1
            if permit.epoch != state.epoch:
                return
            if outcome == "failure":
                state.failures += 1
                if permit.probe or state.failures >= self.policy.failure_threshold:
                    state.open_until = self.clock() + self.policy.recovery_seconds
                    state.probing = False
                    state.epoch += 1
                    self._opened += 1
            elif outcome == "success":
                state.failures = 0
                if permit.probe:
                    state.open_until = None
                    state.probing = False
                    state.epoch += 1
            elif permit.probe:
                state.probing = False
                state.open_until = self.clock() + self.policy.recovery_seconds

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "keys": len(self._states),
                "open": sum(s.open_until is not None for s in self._states.values()),
                "half_open": sum(s.probing for s in self._states.values()),
                "in_flight": sum(s.in_flight for s in self._states.values()),
                "rejected_total": self._rejected,
                "opened_total": self._opened,
            }

    def prometheus_metrics(self, *, component: str | None = None) -> str:
        if component not in {None, "gateway", "http", "s3", "stream", "optional-telemetry-http", "optional-telemetry-span"}:
            raise ValueError("unknown circuit metrics component")
        labels = "" if component is None else f'{{component="{component}"}}'
        return "".join(f"deepevol_dependency_circuit_{key}{labels} {value}\n" for key, value in self.snapshot().items())


def provider_circuit_key(provider: str, account: str, model: str, resource_kind: str) -> str:
    """One shared key definition for model selection and actual dispatch."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps([provider, account, model, resource_kind]).encode()).hexdigest()
