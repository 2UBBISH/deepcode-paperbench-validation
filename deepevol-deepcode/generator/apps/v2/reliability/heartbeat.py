"""A narrow synchronous lease-heartbeat runner with strict fence checks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread, current_thread

from .models import Lease


class HeartbeatFailed(RuntimeError):
    """The background heartbeat stopped after a renewal failure."""


class HeartbeatInvariantError(RuntimeError):
    """A renewal response changed the lease's immutable fence."""


class HeartbeatRunner:
    """Renew one lease serially until stopped or the first renewal fails.

    The callback owns network/database timeouts. The runner deliberately never
    overlaps renewals and never accepts a changed operation, holder, generation,
    token hash, acquisition time, or non-increasing expiry.
    """

    def __init__(
        self,
        lease: Lease,
        renew: Callable[[Lease], Lease],
        *,
        interval: timedelta,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        thread_name: str = "deepevol-v2-reliability-heartbeat",
    ) -> None:
        if not isinstance(lease, Lease):
            raise TypeError("heartbeat lease must be a Lease")
        if not callable(renew) or not callable(clock):
            raise TypeError("heartbeat renew and clock must be callable")
        if interval <= timedelta(0):
            raise ValueError("heartbeat interval must be positive")
        lease_duration = lease.duration
        if lease_duration is not None and interval >= lease_duration:
            raise ValueError("heartbeat interval must be below the known lease duration")
        if not thread_name or len(thread_name.encode("utf-8")) > 160:
            raise ValueError("heartbeat thread_name is empty or oversized")
        self._lease = lease
        self._renew = renew
        self._interval_seconds = interval.total_seconds()
        self._clock = clock
        self._thread_name = thread_name
        self._stop = Event()
        self._lock = Lock()
        self._beat_lock = Lock()
        self._thread: Thread | None = None
        self._started = False
        self._error: BaseException | None = None

    @property
    def lease(self) -> Lease:
        with self._lock:
            return self._lease

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def running(self) -> bool:
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("heartbeat runner is single-use and already started")
            self._started = True
            self._thread = Thread(target=self._run, name=self._thread_name, daemon=True)
            thread = self._thread
        thread.start()

    def stop(self, *, timeout_seconds: float = 5.0) -> bool:
        if timeout_seconds < 0:
            raise ValueError("heartbeat stop timeout cannot be negative")
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is None or thread is current_thread():
            return True
        thread.join(timeout_seconds)
        return not thread.is_alive()

    def beat(self) -> Lease:
        """Perform one synchronous renewal, primarily for deterministic adapters/tests."""

        with self._beat_lock:
            current = self.lease
            now = self._clock()
            if not current.is_active(now):
                raise HeartbeatInvariantError("heartbeat cannot renew an inactive lease")
            renewed = self._renew(current)
            if not isinstance(renewed, Lease):
                raise HeartbeatInvariantError("heartbeat renewal returned a non-Lease value")
            immutable_before = (
                current.holder,
                current.generation,
                current.token_hash,
                current.acquired_at,
                current.operation_id,
            )
            immutable_after = (
                renewed.holder,
                renewed.generation,
                renewed.token_hash,
                renewed.acquired_at,
                renewed.operation_id,
            )
            if immutable_after != immutable_before:
                raise HeartbeatInvariantError("heartbeat renewal changed the lease fence")
            if renewed.expires_at <= current.expires_at:
                raise HeartbeatInvariantError("heartbeat renewal did not extend the lease")
            if not renewed.is_active(self._clock()):
                raise HeartbeatInvariantError("heartbeat renewal returned an inactive lease")
            with self._lock:
                self._lease = renewed
            return renewed

    def raise_if_failed(self) -> None:
        error = self.error
        if error is not None:
            raise HeartbeatFailed(f"lease heartbeat failed with {type(error).__name__}") from error

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self.beat()
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                self._stop.set()
                return

    def __enter__(self) -> "HeartbeatRunner":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


__all__ = ["HeartbeatFailed", "HeartbeatInvariantError", "HeartbeatRunner"]
