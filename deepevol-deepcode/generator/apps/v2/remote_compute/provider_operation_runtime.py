"""Provider-neutral durable execution for Agent Remote Compute effects."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID

from apps.v2.reliability import (
    FailureClass,
    HeartbeatRunner,
    Lease,
    RetryDisposition,
    decide_retry,
)

from .models import RemoteComputeCommand
from .provider_operations import (
    ProviderOperationLease,
    ProviderOperationRecord,
    ProviderOperationStatus,
    ProviderResolution,
    ProviderResolutionState,
    PsycopgProviderOperationRepository,
)


class ProviderOperationRuntimeError(RuntimeError):
    code = "REMOTE_PROVIDER_OPERATION_RUNTIME_ERROR"


class ProviderOperationInProgress(ProviderOperationRuntimeError):
    code = "REMOTE_PROVIDER_OPERATION_IN_PROGRESS"


class ProviderOperationResultUnknown(ProviderOperationRuntimeError):
    code = "REMOTE_PROVIDER_RESULT_UNKNOWN"


class ProviderPreEffectRetryableError(ProviderOperationRuntimeError):
    """A Provider adapter proved that no mutating request left this process."""

    code = "REMOTE_PROVIDER_PRE_EFFECT_UNAVAILABLE"

    def __init__(self, code: str | None = None, *, retry_after: object = None) -> None:
        super().__init__(code or self.code)
        self.code = code or self.code
        self.retry_after = retry_after


class ProviderOperationCrashPoint(StrEnum):
    """Crash boundaries for the test-only V2 reliability harness."""

    AFTER_OPERATION_COMMIT = "AFTER_OPERATION_COMMIT"
    AFTER_APPLY_INTENT = "AFTER_APPLY_INTENT"
    AFTER_EFFECT_STARTED = "AFTER_EFFECT_STARTED"
    AFTER_PROVIDER_RESPONSE = "AFTER_PROVIDER_RESPONSE"
    AFTER_APPLY_RECEIPT = "AFTER_APPLY_RECEIPT"
    AFTER_RECONCILE_RESPONSE = "AFTER_RECONCILE_RESPONSE"
    AFTER_RECONCILE_RECEIPT = "AFTER_RECONCILE_RECEIPT"
    AFTER_COMPENSATION_INTENT = "AFTER_COMPENSATION_INTENT"
    AFTER_COMPENSATION_RESPONSE = "AFTER_COMPENSATION_RESPONSE"
    AFTER_COMPENSATION_RECEIPT = "AFTER_COMPENSATION_RECEIPT"


ProviderOperationFaultInjector = Callable[[ProviderOperationCrashPoint], None]


@dataclass(frozen=True, slots=True)
class ProviderApplyOutcome:
    operation_id: UUID
    status: str
    effect: Mapping[str, Any]
    error_code: str | None
    replayed: bool


class DurableRemoteComputeProvider(Protocol):
    """Every adapter exposes apply plus non-mutating reconcile and compensation."""

    def apply(
        self,
        command: RemoteComputeCommand,
        *,
        secret: str,
        target_secret: str | None = None,
    ) -> Mapping[str, Any]: ...

    def reconcile(
        self,
        operation: ProviderOperationRecord,
        *,
        secret: str,
        target_secret: str | None = None,
    ) -> ProviderResolution: ...

    def compensate(
        self,
        operation: ProviderOperationRecord,
        *,
        secret: str,
        target_secret: str | None = None,
    ) -> Mapping[str, Any]: ...


class DurableProviderOperationExecutor:
    """Persist intent before egress and never re-apply an unknown effect."""

    def __init__(
        self,
        repository: PsycopgProviderOperationRepository,
        provider: DurableRemoteComputeProvider,
        *,
        holder: str | None = None,
        lease_ttl: timedelta = timedelta(seconds=120),
        heartbeat_interval: timedelta = timedelta(seconds=20),
        heartbeat_stop_timeout_seconds: float = 5.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        fault_injector: ProviderOperationFaultInjector | None = None,
        retry_random_source: Callable[[], float] | None = None,
    ) -> None:
        if heartbeat_interval * 3 >= lease_ttl:
            raise ValueError("heartbeat interval must be below one third of lease TTL")
        if heartbeat_stop_timeout_seconds <= 0:
            raise ValueError("heartbeat stop timeout must be positive")
        self.repository = repository
        self.provider = provider
        self.holder = holder or f"{socket.gethostname()}:{os.getpid()}"
        self.lease_ttl = lease_ttl
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_stop_timeout_seconds = heartbeat_stop_timeout_seconds
        self.clock = clock
        self.fault_injector = fault_injector
        self.retry_random_source = retry_random_source

    def execute_apply(
        self,
        command: RemoteComputeCommand,
        *,
        command_event_id: UUID,
        request: Mapping[str, Any],
        secret: str,
        target_secret: str | None = None,
    ) -> ProviderApplyOutcome:
        operation = self.repository.create_or_get(
            command,
            command_event_id=command_event_id,
            request=request,
            request_sha256=command.request_fingerprint,
        )
        self._fault(ProviderOperationCrashPoint.AFTER_OPERATION_COMMIT)
        prior = _terminal_outcome(operation, replayed=True)
        if prior is not None:
            return prior
        if operation.status in {
            ProviderOperationStatus.RESULT_UNKNOWN,
            ProviderOperationStatus.AMBIGUOUS,
            ProviderOperationStatus.RECONCILING,
            ProviderOperationStatus.MANUAL_REVIEW,
        }:
            raise ProviderOperationResultUnknown(self._unknown_message(operation))
        lease = self.repository.claim_apply(
            resource_tid=command.resource_tid,
            owner_uid=command.owner_uid,
            operation_id=command.command_id,
            holder=self.holder,
            lease_ttl=self.lease_ttl,
        )
        if lease is None:
            current = self.repository.get(
                resource_tid=command.resource_tid,
                owner_uid=command.owner_uid,
                operation_id=command.command_id,
            )
            if current is None:
                raise ProviderOperationInProgress("Provider operation disappeared during claim")
            prior = _terminal_outcome(current, replayed=True)
            if prior is not None:
                return prior
            if current.status in {
                ProviderOperationStatus.RESULT_UNKNOWN,
                ProviderOperationStatus.AMBIGUOUS,
                ProviderOperationStatus.RECONCILING,
                ProviderOperationStatus.MANUAL_REVIEW,
            }:
                raise ProviderOperationResultUnknown(self._unknown_message(current))
            raise ProviderOperationInProgress("Provider operation has a live executor")

        self.repository.persist_apply_intent(
            lease,
            {
                "action": command.action.value,
                "external_resource_id": command.external_resource_id,
                "idempotency_key": command.idempotency_key,
                "provider": command.provider,
                "request_sha256": command.request_fingerprint,
            },
        )
        self._fault(ProviderOperationCrashPoint.AFTER_APPLY_INTENT)
        self.repository.mark_effect_started(lease)
        self._fault(ProviderOperationCrashPoint.AFTER_EFFECT_STARTED)
        try:
            effect = self._with_heartbeat(
                lease,
                lambda: dict(
                    self.provider.apply(
                        command,
                        secret=secret,
                        target_secret=target_secret,
                    )
                ),
            )
            self._fault(ProviderOperationCrashPoint.AFTER_PROVIDER_RESPONSE)
        except Exception as exc:
            code = _error_code(exc)
            if _is_known_terminal(exc):
                operation = self.repository.complete_apply_failure(
                    lease,
                    error_code=code,
                    evidence={"provider_rejected": True},
                )
                return _terminal_outcome(operation, replayed=False) or ProviderApplyOutcome(
                    command.command_id, "FAILED", {}, code, False
                )
            try:
                self.repository.mark_result_unknown(lease, code)
            except Exception:
                # A replacement generation may already own reconciliation.
                pass
            raise ProviderOperationResultUnknown(code) from exc
        operation = self.repository.complete_apply(lease, effect)
        self._fault(ProviderOperationCrashPoint.AFTER_APPLY_RECEIPT)
        return _terminal_outcome(operation, replayed=False) or ProviderApplyOutcome(
            command.command_id, "APPLIED", effect, None, False
        )

    def reconcile_once(
        self,
        credentials: Callable[[ProviderOperationRecord], tuple[str, str | None]],
    ) -> ProviderOperationRecord | None:
        lease = self.repository.claim_next_reconciliation(
            holder=self.holder,
            lease_ttl=self.lease_ttl,
        )
        if lease is None:
            return None
        try:
            secret, target_secret = credentials(lease.operation)
            resolution = self._with_heartbeat(
                lease,
                lambda: self.provider.reconcile(
                    lease.operation,
                    secret=secret,
                    target_secret=target_secret,
                ),
            )
            if not isinstance(resolution, ProviderResolution):
                raise TypeError("Provider reconciliation returned an invalid result")
            self._fault(ProviderOperationCrashPoint.AFTER_RECONCILE_RESPONSE)
        except Exception as exc:
            resolution = ProviderResolution(
                ProviderResolutionState.UNKNOWN,
                {"runtime_error": type(exc).__name__},
                _error_code(exc),
            )
        operation = self.repository.resolve_reconciliation(lease, resolution)
        self._fault(ProviderOperationCrashPoint.AFTER_RECONCILE_RECEIPT)
        return operation

    def compensate_once(
        self,
        credentials: Callable[[ProviderOperationRecord], tuple[str, str | None]],
    ) -> ProviderOperationRecord | None:
        lease = self.repository.claim_next_compensation(
            holder=self.holder,
            lease_ttl=self.lease_ttl,
        )
        if lease is None:
            return None
        secret, target_secret = credentials(lease.operation)
        self.repository.persist_compensation_intent(
            lease,
            {
                "idempotency_key": f"{lease.operation.idempotency_key}:compensate",
                "provider_receipt_sha256": lease.operation.provider_receipt_sha256,
            },
        )
        self._fault(ProviderOperationCrashPoint.AFTER_COMPENSATION_INTENT)
        try:
            receipt = self._with_heartbeat(
                lease,
                lambda: dict(
                    self.provider.compensate(
                        lease.operation,
                        secret=secret,
                        target_secret=target_secret,
                    )
                ),
            )
            self._fault(ProviderOperationCrashPoint.AFTER_COMPENSATION_RESPONSE)
        except Exception as exc:
            if isinstance(exc, ProviderPreEffectRetryableError):
                now = self.clock()
                decision = decide_retry(
                    FailureClass.NETWORK_PRE_EFFECT,
                    attempt=lease.operation.compensation_attempts,
                    first_failure_at=now,
                    now=now,
                    retry_after=exc.retry_after,
                    random_source=self.retry_random_source,
                )
                if (
                    decision.disposition is RetryDisposition.RETRY
                    and decision.next_attempt_at is not None
                ):
                    return self.repository.defer_compensation_retry(
                        lease,
                        error_code=exc.code,
                        next_attempt_at=decision.next_attempt_at,
                    )
            return self.repository.mark_compensation_ambiguous(
                lease,
                error_code=_error_code(exc),
            )
        operation = self.repository.complete_compensation(lease, receipt)
        self._fault(ProviderOperationCrashPoint.AFTER_COMPENSATION_RECEIPT)
        return operation

    def _fault(self, point: ProviderOperationCrashPoint) -> None:
        if self.fault_injector is not None:
            self.fault_injector(point)

    def _with_heartbeat(
        self,
        operation_lease: ProviderOperationLease,
        call: Callable[[], Any],
    ) -> Any:
        acquired_at = operation_lease.operation.heartbeat_at or self.clock()
        lease = Lease(
            holder=operation_lease.holder,
            generation=operation_lease.generation,
            token_hash=operation_lease.token_hash,
            acquired_at=acquired_at,
            expires_at=operation_lease.expires_at,
            operation_id=operation_lease.operation.operation_id,
        )

        def renew(current: Lease) -> Lease:
            renewed = self.repository.heartbeat(
                operation_lease,
                lease_ttl=self.lease_ttl,
            )
            return current.renewed(expires_at=renewed.expires_at)

        heartbeat = HeartbeatRunner(
            lease,
            renew,
            interval=self.heartbeat_interval,
            clock=self.clock,
            thread_name=f"remote-provider-operation-{operation_lease.operation.operation_id}",
        )
        heartbeat.start()
        error: BaseException | None = None
        result: Any = None
        try:
            result = call()
        except BaseException as exc:
            error = exc
        stopped = heartbeat.stop(timeout_seconds=self.heartbeat_stop_timeout_seconds)
        if not stopped or heartbeat.error is not None:
            raise ProviderOperationResultUnknown("Provider operation heartbeat lost")
        if error is not None:
            raise error
        return result

    @staticmethod
    def _unknown_message(operation: ProviderOperationRecord) -> str:
        return operation.last_error_code or "REMOTE_PROVIDER_RESULT_UNKNOWN"


def _terminal_outcome(
    operation: ProviderOperationRecord,
    *,
    replayed: bool,
) -> ProviderApplyOutcome | None:
    if operation.status is ProviderOperationStatus.SUCCEEDED:
        return ProviderApplyOutcome(
            operation.operation_id,
            "APPLIED",
            dict(operation.provider_receipt or {}),
            None,
            replayed,
        )
    if operation.status is ProviderOperationStatus.FAILED:
        return ProviderApplyOutcome(
            operation.operation_id,
            "FAILED",
            dict(operation.provider_receipt or {}),
            operation.last_error_code or "REMOTE_PROVIDER_REJECTED",
            replayed,
        )
    return None


def _error_code(error: BaseException) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code and len(code.encode("utf-8")) <= 256:
        return code
    return "REMOTE_PROVIDER_RUNTIME_ERROR"


def _is_known_terminal(error: BaseException) -> bool:
    return hasattr(error, "retryable") and error.retryable is False


__all__ = [
    "DurableProviderOperationExecutor",
    "DurableRemoteComputeProvider",
    "ProviderApplyOutcome",
    "ProviderOperationCrashPoint",
    "ProviderOperationFaultInjector",
    "ProviderOperationInProgress",
    "ProviderOperationResultUnknown",
    "ProviderOperationRuntimeError",
    "ProviderPreEffectRetryableError",
]
