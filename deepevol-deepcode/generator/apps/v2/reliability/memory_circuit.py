"""Memory-specific adapter over the atomic shared Redis circuit registry."""

from __future__ import annotations

from contextvars import ContextVar
from threading import Lock
from typing import Any

from apps.v2.memory.circuit_breaker import CircuitSnapshot, CircuitState
from apps.v2.memory.errors import MemoryErrorCode, MemorySystemError

from .circuit import DependencyUnavailable


class SharedMemoryCircuitBreaker:
    """Expose the Memory breaker protocol while retaining per-call Redis permits."""

    def __init__(self, registry: Any, *, key: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("shared memory circuit key is required")
        self._registry = registry
        self._key = key
        self._permit: ContextVar[Any | None] = ContextVar(
            f"deepevol_memory_permit_{id(self)}",
            default=None,
        )
        self._latch_lock = Lock()
        self._local_latched = False

    def before_call(self) -> None:
        with self._latch_lock:
            if self._local_latched:
                raise MemorySystemError(
                    MemoryErrorCode.CIRCUIT_OPEN,
                    "memory provider safety latch is open",
                    retryable=False,
                )
        if self._permit.get() is not None:
            raise RuntimeError("shared memory circuit call is already active in this context")
        try:
            permit = self._registry.acquire(self._key)
        except DependencyUnavailable as exc:
            raise MemorySystemError(
                MemoryErrorCode.CIRCUIT_OPEN,
                "shared memory provider circuit rejected admission",
                retryable=exc.reason != "DEPENDENCY_CIRCUIT_LATCHED",
                cause=exc,
            ) from exc
        self._permit.set(permit)

    def record_success(self, *, reset_failures: bool = True) -> None:
        self._finish("success" if reset_failures else "neutral")

    def record_failure(self, error: MemorySystemError) -> None:
        self._finish("failure" if error.counts_toward_breaker else "neutral")

    def record_aborted(self) -> None:
        self._finish("neutral")

    def force_open(self) -> None:
        with self._latch_lock:
            self._local_latched = True
        try:
            self._registry.force_open(self._key, latched=True)
        except DependencyUnavailable:
            # Local fail-closed state remains effective while Redis is unavailable.
            pass

    def snapshot(self) -> CircuitSnapshot:
        with self._latch_lock:
            local_latched = self._local_latched
        try:
            state = self._registry.key_snapshot(self._key)
        except DependencyUnavailable as exc:
            recovering = exc.reason == "DEPENDENCY_COORDINATOR_RECOVERING"
            return CircuitSnapshot(
                state=CircuitState.OPEN,
                failure_count=0,
                opened_at=None,
                latched=local_latched,
                shared=True,
                coordinator_available=recovering,
                recovering=recovering,
            )
        latched = local_latched or bool(state.get("latched", 0))
        if bool(state.get("half_open", 0)):
            circuit_state = CircuitState.HALF_OPEN
        elif bool(state.get("open", 0)) or latched:
            circuit_state = CircuitState.OPEN
        else:
            circuit_state = CircuitState.CLOSED
        return CircuitSnapshot(
            state=circuit_state,
            failure_count=max(0, int(state.get("failures", 0))),
            opened_at=None,
            latched=latched,
            shared=True,
            coordinator_available=True,
            recovering=False,
        )

    def close(self) -> None:
        self._registry.close()

    def _finish(self, outcome: str) -> None:
        permit = self._permit.get()
        if permit is None:
            return
        self._permit.set(None)
        permit.finish(outcome)


__all__ = ["SharedMemoryCircuitBreaker"]
