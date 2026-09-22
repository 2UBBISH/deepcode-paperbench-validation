"""Immutable reliability value objects shared by future V2 adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from .canonical import canonical_json_sha256


def _require_text(value: str, name: str, maximum_bytes: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        raise ValueError(f"{name} must contain 1 to {maximum_bytes} bounded UTF-8 bytes")


def _require_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _wire_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class OperationIdentity:
    """Stable, minimal identity for one logical operation across attempts.

    The three public fields are also the complete canonical identity document.
    Callers that start with a UUID may pass it directly; it is normalized to
    its lowercase string representation so equivalent inputs hash identically.
    """

    operation_id: str | UUID
    idempotency_key: str
    request_sha256: str

    def __post_init__(self) -> None:
        operation_id = str(self.operation_id) if isinstance(self.operation_id, UUID) else self.operation_id
        _require_text(operation_id, "operation_id", 256)
        _require_text(self.idempotency_key, "idempotency_key", 512)
        _require_digest(self.request_sha256, "request_sha256")
        object.__setattr__(self, "operation_id", operation_id)

    def as_document(self) -> dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "operation_id": self.operation_id,
            "request_sha256": self.request_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.as_document())


@dataclass(frozen=True, slots=True)
class Lease:
    """A generation-fenced lease that stores only a token digest."""

    holder: str
    generation: int
    token_hash: str
    expires_at: datetime
    acquired_at: datetime | None = None
    operation_id: str | UUID | None = None

    def __post_init__(self) -> None:
        _require_text(self.holder, "holder", 256)
        if not isinstance(self.generation, int) or isinstance(self.generation, bool) or self.generation < 1:
            raise ValueError("lease generation must be a positive integer")
        _require_digest(self.token_hash, "token_hash")
        _require_aware(self.expires_at, "expires_at")
        if self.acquired_at is not None:
            _require_aware(self.acquired_at, "acquired_at")
        if self.acquired_at is not None and self.expires_at <= self.acquired_at:
            raise ValueError("lease expiry must be after acquisition")
        if self.operation_id is not None:
            operation_id = str(self.operation_id) if isinstance(self.operation_id, UUID) else self.operation_id
            _require_text(operation_id, "operation_id", 256)
            object.__setattr__(self, "operation_id", operation_id)

    @property
    def duration(self) -> timedelta | None:
        if self.acquired_at is None:
            return None
        return self.expires_at - self.acquired_at

    def is_active(self, now: datetime) -> bool:
        _require_aware(now, "now")
        return (self.acquired_at is None or self.acquired_at <= now) and now < self.expires_at

    def remaining(self, now: datetime) -> timedelta:
        _require_aware(now, "now")
        return max(timedelta(0), self.expires_at - now)

    def renewed(self, *, expires_at: datetime) -> "Lease":
        _require_aware(expires_at, "expires_at")
        if expires_at <= self.expires_at:
            raise ValueError("a lease renewal must extend the current expiry")
        return replace(self, expires_at=expires_at)


class FailureClass(StrEnum):
    INTERNAL_PERSISTENCE_TRANSIENT = "INTERNAL_PERSISTENCE_TRANSIENT"
    NETWORK_PRE_EFFECT = "NETWORK_PRE_EFFECT"
    RATE_LIMITED_PRE_EFFECT = "RATE_LIMITED_PRE_EFFECT"
    RESOURCE_CAPACITY = "RESOURCE_CAPACITY"
    SAFE_TIMEOUT = "SAFE_TIMEOUT"
    LEASE_LOST = "LEASE_LOST"
    WAIT_EXTERNAL = "WAIT_EXTERNAL"
    KNOWN_RESULT_COMMIT_PENDING = "KNOWN_RESULT_COMMIT_PENDING"
    POST_EFFECT_UNKNOWN = "POST_EFFECT_UNKNOWN"
    AUTHORIZATION_STALE = "AUTHORIZATION_STALE"
    COMPATIBILITY_MISMATCH = "COMPATIBILITY_MISMATCH"
    VALIDATION = "VALIDATION"
    SECURITY = "SECURITY"
    BUG = "BUG"
    CANCELLED = "CANCELLED"
    PERMANENT = "PERMANENT"


class RetryDisposition(StrEnum):
    RETRY = "RETRY"
    WAIT = "WAIT"
    RECONCILE = "RECONCILE"
    BLOCKED = "BLOCKED"
    TERMINAL = "TERMINAL"
    EXHAUSTED = "EXHAUSTED"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Auditable output of one versioned retry-policy evaluation."""

    policy_version: str
    failure_class: FailureClass
    disposition: RetryDisposition
    attempt: int
    max_attempts: int | None
    first_failure_at: datetime
    decided_at: datetime
    deadline_at: datetime | None
    next_attempt_at: datetime | None
    retry_after_at: datetime | None = None
    retry_after_honored: bool = False
    exhausted_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.policy_version, "policy_version", 64)
        if not isinstance(self.failure_class, FailureClass):
            raise TypeError("failure_class must be a FailureClass")
        if not isinstance(self.disposition, RetryDisposition):
            raise TypeError("disposition must be a RetryDisposition")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if self.max_attempts is not None and (
            not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool) or self.max_attempts < 0
        ):
            raise ValueError("max_attempts must be non-negative or None")
        _require_aware(self.first_failure_at, "first_failure_at")
        _require_aware(self.decided_at, "decided_at")
        if self.first_failure_at > self.decided_at:
            raise ValueError("first_failure_at cannot be after decided_at")
        for name, value in (
            ("deadline_at", self.deadline_at),
            ("next_attempt_at", self.next_attempt_at),
            ("retry_after_at", self.retry_after_at),
        ):
            if value is not None:
                _require_aware(value, name)
        scheduled = self.disposition in {RetryDisposition.RETRY, RetryDisposition.WAIT}
        if scheduled != (self.next_attempt_at is not None):
            raise ValueError("only RETRY and WAIT decisions carry next_attempt_at")
        if self.next_attempt_at is not None and self.next_attempt_at < self.decided_at:
            raise ValueError("next_attempt_at cannot precede the decision")
        if self.retry_after_honored and self.retry_after_at is None:
            raise ValueError("an honored Retry-After requires retry_after_at")
        if self.disposition is RetryDisposition.EXHAUSTED:
            _require_text(self.exhausted_reason or "", "exhausted_reason", 128)
        elif self.exhausted_reason is not None:
            raise ValueError("only an exhausted decision carries exhausted_reason")

    @property
    def delay(self) -> timedelta | None:
        if self.next_attempt_at is None:
            return None
        return self.next_attempt_at - self.decided_at

    @property
    def should_retry(self) -> bool:
        return self.disposition is RetryDisposition.RETRY

    @property
    def requires_reconciliation(self) -> bool:
        return self.disposition is RetryDisposition.RECONCILE

    @property
    def attempt_number(self) -> int:
        """Compatibility alias for adapters written before the locked schema."""

        return self.attempt

    def as_document(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "deadline_at": _wire_time(self.deadline_at),
            "decided_at": _wire_time(self.decided_at),
            "disposition": self.disposition.value,
            "exhausted_reason": self.exhausted_reason,
            "failure_class": self.failure_class.value,
            "first_failure_at": _wire_time(self.first_failure_at),
            "max_attempts": self.max_attempts,
            "next_attempt_at": _wire_time(self.next_attempt_at),
            "policy_version": self.policy_version,
            "retry_after_at": _wire_time(self.retry_after_at),
            "retry_after_honored": self.retry_after_honored,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.as_document())


__all__ = [
    "FailureClass",
    "Lease",
    "OperationIdentity",
    "RetryDecision",
    "RetryDisposition",
]
