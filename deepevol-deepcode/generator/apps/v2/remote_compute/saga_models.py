"""Value objects for the Product-owned Remote Compute operation Saga."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any
from uuid import UUID

from apps.common.v2_ids import derive_uuid7, format_typed_id


REMOTE_COMPUTE_OPERATION_JSON_MAX_BYTES = 65_536


class RemoteComputeOperationKind(StrEnum):
    PROVISION = "PROVISION"
    ACTIVATE = "ACTIVATE"
    FINISH = "FINISH"
    RELEASE = "RELEASE"
    COMPENSATE = "COMPENSATE"


class RemoteComputeOperationStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    PROVIDER_PENDING = "PROVIDER_PENDING"
    PROVIDER_SUCCEEDED = "PROVIDER_SUCCEEDED"
    COMPENSATION_PENDING = "COMPENSATION_PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    COMPENSATED = "COMPENSATED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in {
            RemoteComputeOperationStatus.SUCCEEDED,
            RemoteComputeOperationStatus.FAILED,
            RemoteComputeOperationStatus.COMPENSATED,
            RemoteComputeOperationStatus.CANCELLED,
        }


class RemoteComputeEffectPhase(StrEnum):
    INTENT_PERSISTED = "INTENT_PERSISTED"
    EFFECT_MAY_HAVE_OCCURRED = "EFFECT_MAY_HAVE_OCCURRED"
    PROVIDER_ACCEPTED = "PROVIDER_ACCEPTED"
    RECEIPT_PERSISTED = "RECEIPT_PERSISTED"
    NO_EFFECT_RECEIPT_PERSISTED = "NO_EFFECT_RECEIPT_PERSISTED"
    COMPENSATION_REQUIRED = "COMPENSATION_REQUIRED"
    COMPENSATION_RECEIPT_PERSISTED = "COMPENSATION_RECEIPT_PERSISTED"


class RemoteComputeFailureCertainty(StrEnum):
    CONFIRMED_NO_EFFECT = "CONFIRMED_NO_EFFECT"
    EFFECT_MAY_HAVE_OCCURRED = "EFFECT_MAY_HAVE_OCCURRED"


def canonical_operation_json(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise ValueError("Remote Compute operation JSON must be an object")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Remote Compute operation JSON is not canonical") from exc
    if len(encoded) > REMOTE_COMPUTE_OPERATION_JSON_MAX_BYTES:
        raise ValueError("Remote Compute operation JSON exceeds 65536 bytes")
    return encoded


def operation_json_sha256(value: Mapping[str, Any]) -> bytes:
    return sha256(canonical_operation_json(value)).digest()


def stable_provider_operation_id(
    operation_id: UUID,
    operation_kind: RemoteComputeOperationKind,
) -> str:
    """Derive the Provider idempotency key solely from durable operation identity."""

    format_typed_id("op", operation_id)
    digest = sha256(
        b"deepevol:remote-compute-operation:v1\0"
        + operation_id.bytes
        + operation_kind.value.encode("ascii")
    ).hexdigest()
    return f"rcop_{digest}"


def stable_remote_compute_binding_id(operation_id: UUID) -> UUID:
    """Derive the Product binding identity before Billing or Provider effects."""

    format_typed_id("op", operation_id)
    return derive_uuid7(
        operation_id,
        b"deepevol:remote-compute-binding:v1",
    )


def stable_remote_compute_release_operation_id(provision_operation_id: UUID) -> UUID:
    """Derive the RELEASE operation identity from the PROVISION it undoes.

    A run-scoped machine (experiment Agent lease) is released by the Agent
    once the run no longer needs it; deriving the identity from the provision
    receipt makes the release request idempotent across Agent retries without
    a second identity to persist.
    """

    format_typed_id("op", provision_operation_id)
    return derive_uuid7(
        provision_operation_id,
        b"deepevol:remote-compute-release:v1",
    )


def _bounded_text(value: str, name: str, maximum: int) -> None:
    if not value or len(value.encode("utf-8")) > maximum or "\x00" in value:
        raise ValueError(f"{name} must contain 1 to {maximum} safe UTF-8 bytes")


def _digest(value: bytes, name: str) -> None:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError(f"{name} must contain a binary SHA-256 digest")


def _aware(value: datetime | None, name: str) -> None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class RemoteComputeOperationCreate:
    operation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    operation_kind: RemoteComputeOperationKind
    provider: str
    idempotency_key: str
    request: Mapping[str, Any]
    request_sha256: bytes = field(repr=False)
    provider_operation_id: str
    resource_id: UUID | None = None
    binding_id: UUID | None = None
    rid: UUID | None = None
    compensates_operation_id: UUID | None = None
    max_attempts: int = 5
    next_attempt_at: datetime | None = None

    def __post_init__(self) -> None:
        format_typed_id("op", self.operation_id)
        _bounded_text(self.provider, "provider", 128)
        _bounded_text(self.idempotency_key, "idempotency_key", 512)
        _bounded_text(self.provider_operation_id, "provider_operation_id", 256)
        _digest(self.request_sha256, "request_sha256")
        if operation_json_sha256(self.request) != self.request_sha256:
            raise ValueError("request_sha256 does not match canonical operation request")
        expected_provider_id = stable_provider_operation_id(
            self.operation_id,
            self.operation_kind,
        )
        if self.provider_operation_id != expected_provider_id:
            raise ValueError("provider_operation_id is not derived from operation identity")
        if isinstance(self.max_attempts, bool) or not 1 <= self.max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        _aware(self.next_attempt_at, "next_attempt_at")
        if (self.operation_kind is RemoteComputeOperationKind.COMPENSATE) != (
            self.compensates_operation_id is not None
        ):
            raise ValueError("only COMPENSATE operations identify a compensated operation")

    @classmethod
    def from_request(
        cls,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_kind: RemoteComputeOperationKind,
        provider: str,
        idempotency_key: str,
        request: Mapping[str, Any],
        resource_id: UUID | None = None,
        binding_id: UUID | None = None,
        rid: UUID | None = None,
        compensates_operation_id: UUID | None = None,
        max_attempts: int = 5,
        next_attempt_at: datetime | None = None,
    ) -> "RemoteComputeOperationCreate":
        return cls(
            operation_id=operation_id,
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            operation_kind=operation_kind,
            provider=provider,
            idempotency_key=idempotency_key,
            request=request,
            request_sha256=operation_json_sha256(request),
            provider_operation_id=stable_provider_operation_id(
                operation_id,
                operation_kind,
            ),
            resource_id=resource_id,
            binding_id=binding_id,
            rid=rid,
            compensates_operation_id=compensates_operation_id,
            max_attempts=max_attempts,
            next_attempt_at=next_attempt_at,
        )


@dataclass(frozen=True, slots=True)
class RemoteComputeOperation:
    operation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    operation_kind: RemoteComputeOperationKind
    provider: str
    idempotency_key: str
    request: Mapping[str, Any]
    request_sha256: bytes = field(repr=False)
    provider_operation_id: str
    resource_id: UUID | None
    binding_id: UUID | None
    rid: UUID | None
    compensates_operation_id: UUID | None
    status: RemoteComputeOperationStatus
    effect_phase: RemoteComputeEffectPhase
    attempt_count: int
    max_attempts: int
    next_attempt_at: datetime
    lease_generation: int
    lease_owner: str | None
    heartbeat_at: datetime | None
    lease_expires_at: datetime | None
    provider_request_ref: str | None
    provider_status: str | None
    result: Mapping[str, Any] | None
    result_sha256: bytes | None = field(repr=False)
    error_code: str | None
    error_summary: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        format_typed_id("op", self.operation_id)
        _bounded_text(self.provider, "provider", 128)
        _bounded_text(self.idempotency_key, "idempotency_key", 512)
        _bounded_text(self.provider_operation_id, "provider_operation_id", 256)
        _digest(self.request_sha256, "request_sha256")
        if operation_json_sha256(self.request) != self.request_sha256:
            raise ValueError("request_sha256 does not match canonical operation request")
        if self.provider_operation_id != stable_provider_operation_id(
            self.operation_id,
            self.operation_kind,
        ):
            raise ValueError("provider_operation_id is not derived from operation identity")
        if (self.operation_kind is RemoteComputeOperationKind.COMPENSATE) != (
            self.compensates_operation_id is not None
        ):
            raise ValueError("only COMPENSATE operations identify a compensated operation")
        if self.result_sha256 is not None:
            _digest(self.result_sha256, "result_sha256")
        if (self.result is None) != (self.result_sha256 is None):
            raise ValueError("result and result_sha256 must be present together")
        if self.result is not None and operation_json_sha256(self.result) != self.result_sha256:
            raise ValueError("result_sha256 does not match canonical Provider result")
        if not 0 <= self.attempt_count <= self.max_attempts:
            raise ValueError("attempt_count is outside the operation retry budget")
        if self.lease_generation < 0:
            raise ValueError("lease_generation cannot be negative")
        for name in (
            "next_attempt_at",
            "heartbeat_at",
            "lease_expires_at",
            "created_at",
            "updated_at",
            "completed_at",
        ):
            _aware(getattr(self, name), name)
        leased = self.status is RemoteComputeOperationStatus.CLAIMED
        lease_values = (self.lease_owner, self.heartbeat_at, self.lease_expires_at)
        if leased != all(value is not None for value in lease_values) or (
            not leased and any(value is not None for value in lease_values)
        ):
            raise ValueError("CLAIMED status and lease fields must agree")
        if self.status.terminal != (self.completed_at is not None):
            raise ValueError("terminal status and completed_at must agree")
        _validate_status_phase(
            status=self.status,
            effect_phase=self.effect_phase,
            has_provider_receipt=self.provider_request_ref is not None and self.result is not None,
        )


@dataclass(frozen=True, slots=True)
class RemoteComputeOperationLease:
    operation: RemoteComputeOperation
    token: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.operation.status is not RemoteComputeOperationStatus.CLAIMED:
            raise ValueError("operation lease requires CLAIMED status")
        if len(self.token) != 32:
            raise ValueError("operation lease token must contain 32 random bytes")

    @property
    def generation(self) -> int:
        return self.operation.lease_generation

    @property
    def operation_id(self) -> UUID:
        return self.operation.operation_id

    @property
    def lease_owner(self) -> str:
        assert self.operation.lease_owner is not None
        return self.operation.lease_owner


@dataclass(frozen=True, slots=True)
class RemoteComputeOperationStatusView:
    operation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    operation_kind: RemoteComputeOperationKind
    provider: str
    provider_operation_id: str
    resource_id: UUID | None
    binding_id: UUID | None
    rid: UUID | None
    compensates_operation_id: UUID | None
    status: RemoteComputeOperationStatus
    effect_phase: RemoteComputeEffectPhase
    attempt_count: int
    max_attempts: int
    next_attempt_at: datetime
    provider_request_ref: str | None
    provider_status: str | None
    error_code: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        format_typed_id("op", self.operation_id)
        _bounded_text(self.provider, "provider", 128)
        _bounded_text(self.provider_operation_id, "provider_operation_id", 256)
        if self.provider_operation_id != stable_provider_operation_id(
            self.operation_id,
            self.operation_kind,
        ):
            raise ValueError("provider_operation_id is not derived from operation identity")
        if (self.operation_kind is RemoteComputeOperationKind.COMPENSATE) != (
            self.compensates_operation_id is not None
        ):
            raise ValueError("only COMPENSATE operations identify a compensated operation")
        for name in ("next_attempt_at", "created_at", "updated_at", "completed_at"):
            _aware(getattr(self, name), name)
        if self.status.terminal != (self.completed_at is not None):
            raise ValueError("terminal status and completed_at must agree")
        _validate_status_phase(
            status=self.status,
            effect_phase=self.effect_phase,
            # The status view deliberately omits result bytes. The database
            # constraint, not the polling projection, proves receipt presence.
            has_provider_receipt=True,
        )


def _validate_status_phase(
    *,
    status: RemoteComputeOperationStatus,
    effect_phase: RemoteComputeEffectPhase,
    has_provider_receipt: bool,
) -> None:
    required = {
        RemoteComputeOperationStatus.PENDING: {RemoteComputeEffectPhase.INTENT_PERSISTED},
        RemoteComputeOperationStatus.PROVIDER_PENDING: {
            RemoteComputeEffectPhase.PROVIDER_ACCEPTED
        },
        RemoteComputeOperationStatus.PROVIDER_SUCCEEDED: {
            RemoteComputeEffectPhase.RECEIPT_PERSISTED,
            RemoteComputeEffectPhase.COMPENSATION_RECEIPT_PERSISTED,
        },
        RemoteComputeOperationStatus.RECONCILIATION_REQUIRED: {
            RemoteComputeEffectPhase.EFFECT_MAY_HAVE_OCCURRED
        },
        RemoteComputeOperationStatus.COMPENSATION_PENDING: {
            RemoteComputeEffectPhase.COMPENSATION_REQUIRED
        },
        RemoteComputeOperationStatus.SUCCEEDED: {
            RemoteComputeEffectPhase.RECEIPT_PERSISTED,
            RemoteComputeEffectPhase.COMPENSATION_RECEIPT_PERSISTED,
        },
        RemoteComputeOperationStatus.COMPENSATED: {
            RemoteComputeEffectPhase.COMPENSATION_RECEIPT_PERSISTED
        },
        RemoteComputeOperationStatus.CANCELLED: {
            RemoteComputeEffectPhase.INTENT_PERSISTED
        },
        RemoteComputeOperationStatus.FAILED: {
            RemoteComputeEffectPhase.INTENT_PERSISTED,
            RemoteComputeEffectPhase.NO_EFFECT_RECEIPT_PERSISTED,
        },
    }
    allowed = required.get(status)
    if allowed is not None and effect_phase not in allowed:
        raise ValueError(f"{status.value} status is inconsistent with effect phase")
    if status in {
        RemoteComputeOperationStatus.PROVIDER_SUCCEEDED,
        RemoteComputeOperationStatus.SUCCEEDED,
    } and not has_provider_receipt:
        raise ValueError(f"{status.value} status requires a durable Provider receipt")
    if (
        effect_phase is RemoteComputeEffectPhase.NO_EFFECT_RECEIPT_PERSISTED
        and not has_provider_receipt
    ):
        raise ValueError("explicit no-effect result requires a durable Provider receipt")


@dataclass(frozen=True, slots=True)
class RemoteComputeOperationAccepted:
    operation: RemoteComputeOperationStatusView
    status_path: str
    replayed: bool
    http_status: int = 202
    retry_after_seconds: int = 2

    def __post_init__(self) -> None:
        if self.http_status != 202 or self.retry_after_seconds < 1:
            raise ValueError("operation acceptance must describe HTTP 202 polling")
        if not self.status_path.startswith("/"):
            raise ValueError("status_path must be an absolute application path")


__all__ = [
    "REMOTE_COMPUTE_OPERATION_JSON_MAX_BYTES",
    "RemoteComputeEffectPhase",
    "RemoteComputeFailureCertainty",
    "RemoteComputeOperation",
    "RemoteComputeOperationAccepted",
    "RemoteComputeOperationCreate",
    "RemoteComputeOperationKind",
    "RemoteComputeOperationLease",
    "RemoteComputeOperationStatus",
    "RemoteComputeOperationStatusView",
    "canonical_operation_json",
    "operation_json_sha256",
    "stable_provider_operation_id",
    "stable_remote_compute_binding_id",
    "stable_remote_compute_release_operation_id",
]
