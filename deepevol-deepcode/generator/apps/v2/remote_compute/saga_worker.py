"""Lease-fenced Product worker for Remote Compute operation Sagas.

The worker never calls a Provider directly.  Its effect client is an
authenticated Product-to-Agent boundary whose implementation must persist its
own Agent-side intent before Provider egress.  Product records an ambiguity
boundary before invoking that client and never re-applies an uncertain effect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, TypeVar, cast
from uuid import UUID

from apps.v2.reliability import (
    FailureClass,
    HeartbeatRunner,
    Lease,
    RetryDisposition,
    decide_retry,
)

from .saga_models import (
    RemoteComputeEffectPhase,
    RemoteComputeFailureCertainty,
    RemoteComputeOperation,
    RemoteComputeOperationLease,
)
from .saga_service import RemoteComputeSagaStore


class RemoteComputeSagaCrashPoint(StrEnum):
    """Product-side process boundaries exercised by the chaos harness."""

    AFTER_CLAIM = "AFTER_CLAIM"
    AFTER_AGENT_PREPARATION = "AFTER_AGENT_PREPARATION"
    AFTER_EFFECT_BOUNDARY = "AFTER_EFFECT_BOUNDARY"
    AFTER_AGENT_RESPONSE = "AFTER_AGENT_RESPONSE"
    AFTER_PROVIDER_RECEIPT = "AFTER_PROVIDER_RECEIPT"
    AFTER_LOCAL_FINALIZATION = "AFTER_LOCAL_FINALIZATION"
    AFTER_LOCAL_COMPLETION = "AFTER_LOCAL_COMPLETION"


RemoteComputeSagaFaultInjector = Callable[[RemoteComputeSagaCrashPoint], None]
_T = TypeVar("_T")
_MISSING = object()


class RemoteComputeEffectDisposition(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RemoteComputeEffectOutcome:
    disposition: RemoteComputeEffectDisposition
    provider_request_ref: str | None = None
    provider_status: str | None = None
    result: Mapping[str, Any] | None = None
    error_code: str | None = None
    retry_after: timedelta = timedelta(seconds=2)

    def __post_init__(self) -> None:
        if self.disposition in {
            RemoteComputeEffectDisposition.SUCCEEDED,
            RemoteComputeEffectDisposition.PENDING,
            RemoteComputeEffectDisposition.REJECTED,
        } and (not self.provider_request_ref or not self.provider_status):
            raise ValueError("known Provider outcomes require request identity and status")
        if self.disposition in {
            RemoteComputeEffectDisposition.SUCCEEDED,
            RemoteComputeEffectDisposition.REJECTED,
        } and self.result is None:
            raise ValueError("terminal Provider outcomes require durable evidence")
        if self.disposition is RemoteComputeEffectDisposition.PENDING and (
            self.retry_after < timedelta(seconds=1)
        ):
            raise ValueError("pending Provider outcomes require a positive poll delay")
        if self.disposition in {
            RemoteComputeEffectDisposition.REJECTED,
            RemoteComputeEffectDisposition.UNKNOWN,
        } and not self.error_code:
            raise ValueError("failed Provider outcomes require a stable error code")


@dataclass(frozen=True, slots=True)
class RemoteComputeEffectPreparation:
    """Agent receipt proving intent exists and Provider egress has not begun."""

    operation_id: UUID
    provider_operation_id: str
    dispatch_ref: str

    def __post_init__(self) -> None:
        if not self.provider_operation_id or len(self.provider_operation_id) > 256:
            raise ValueError("prepared Provider operation identity is invalid")
        if not self.dispatch_ref or len(self.dispatch_ref) > 512:
            raise ValueError("prepared dispatch reference is invalid")


class RemoteComputeSagaEffectClient(Protocol):
    def prepare(
        self,
        operation: RemoteComputeOperation,
    ) -> RemoteComputeEffectPreparation: ...

    def apply(
        self,
        operation: RemoteComputeOperation,
        preparation: RemoteComputeEffectPreparation,
    ) -> RemoteComputeEffectOutcome: ...

    def reconcile(
        self,
        operation: RemoteComputeOperation,
    ) -> RemoteComputeEffectOutcome: ...


class RemoteComputeSagaLocalFinalizer(Protocol):
    """Adopt a durable Provider receipt into Product-owned local state."""

    def finalize(self, operation: RemoteComputeOperation) -> None: ...


class RemoteComputePreEffectFailure(RuntimeError):
    """The effect client proved no request bytes crossed its Agent boundary."""

    def __init__(
        self,
        code: str,
        *,
        failure_class: FailureClass = FailureClass.NETWORK_PRE_EFFECT,
        retry_after: object = None,
    ) -> None:
        if failure_class not in {
            FailureClass.NETWORK_PRE_EFFECT,
            FailureClass.RATE_LIMITED_PRE_EFFECT,
            FailureClass.RESOURCE_CAPACITY,
            FailureClass.INTERNAL_PERSISTENCE_TRANSIENT,
        }:
            raise ValueError("pre-effect failure class is not safely retryable")
        super().__init__(code)
        self.code = code
        self.failure_class = failure_class
        self.retry_after = retry_after


class RemoteComputeSagaWorker:
    """Process at most one Product operation while preserving effect certainty."""

    def __init__(
        self,
        store: RemoteComputeSagaStore,
        effect_client: RemoteComputeSagaEffectClient,
        *,
        holder: str,
        heartbeat_interval: timedelta = timedelta(seconds=20),
        heartbeat_stop_timeout_seconds: float = 5.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        retry_random_source: Callable[[], float] | None = None,
        fault_injector: RemoteComputeSagaFaultInjector | None = None,
        local_finalizer: RemoteComputeSagaLocalFinalizer | None = None,
    ) -> None:
        if not holder or len(holder.encode("utf-8")) > 256:
            raise ValueError("Remote Compute Saga worker holder is invalid")
        if heartbeat_interval <= timedelta(0):
            raise ValueError("Remote Compute Saga heartbeat interval must be positive")
        if heartbeat_stop_timeout_seconds <= 0:
            raise ValueError("Remote Compute heartbeat stop timeout must be positive")
        required_store_methods = (
            "claim_next",
            "heartbeat",
            "mark_effect_started",
            "mark_provider_pending",
            "record_provider_success",
            "record_provider_rejection",
            "record_failure",
            "complete",
        )
        missing = tuple(
            name
            for name in required_store_methods
            if not callable(getattr(store, name, None))
        )
        if missing:
            raise TypeError(
                "Remote Compute Saga worker store is incomplete: "
                + ", ".join(missing)
            )
        self.store = store
        self.effect_client = effect_client
        self.holder = holder
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_stop_timeout_seconds = heartbeat_stop_timeout_seconds
        self.clock = clock
        self.retry_random_source = retry_random_source
        self.fault_injector = fault_injector
        self.local_finalizer = local_finalizer

    def process_one(self) -> RemoteComputeOperation | None:
        lease = self.store.claim_next(lease_owner=self.holder)
        if lease is None:
            return None
        self._fault(RemoteComputeSagaCrashPoint.AFTER_CLAIM)
        operation = lease.operation
        if operation.effect_phase in {
            RemoteComputeEffectPhase.RECEIPT_PERSISTED,
            RemoteComputeEffectPhase.COMPENSATION_RECEIPT_PERSISTED,
        }:
            if self.local_finalizer is not None:
                self._with_heartbeat(
                    lease,
                    lambda: self._finalize_local(operation),
                )
            self._fault(RemoteComputeSagaCrashPoint.AFTER_LOCAL_FINALIZATION)
            completed = self.store.complete(lease)
            self._fault(RemoteComputeSagaCrashPoint.AFTER_LOCAL_COMPLETION)
            return completed

        if operation.effect_phase is RemoteComputeEffectPhase.INTENT_PERSISTED:
            effect_started = False

            def prepare_and_apply() -> RemoteComputeEffectOutcome:
                nonlocal effect_started
                preparation = self.effect_client.prepare(operation)
                self._fault(RemoteComputeSagaCrashPoint.AFTER_AGENT_PREPARATION)
                if (
                    not isinstance(preparation, RemoteComputeEffectPreparation)
                    or preparation.operation_id != operation.operation_id
                    or preparation.provider_operation_id
                    != operation.provider_operation_id
                    or preparation.dispatch_ref
                    != (
                        "/internal/v2/remote-compute/operations/"
                        f"{operation.provider_operation_id}"
                    )
                ):
                    raise ValueError("REMOTE_COMPUTE_PREPARATION_INVALID")
                self.store.mark_effect_started(lease)
                effect_started = True
                self._fault(RemoteComputeSagaCrashPoint.AFTER_EFFECT_BOUNDARY)
                return self.effect_client.apply(operation, preparation)

            try:
                outcome = self._with_heartbeat(lease, prepare_and_apply)
            except RemoteComputePreEffectFailure as exc:
                if effect_started:
                    return self._record_unknown(lease, exc.code)
                return self._record_pre_effect_failure(lease, operation, exc)
            except Exception as exc:
                if effect_started:
                    return self._record_unknown(lease, _exception_code(exc))
                return self.store.record_failure(
                    lease,
                    certainty=RemoteComputeFailureCertainty.CONFIRMED_NO_EFFECT,
                    retryable=False,
                    error_code=_exception_code(exc),
                    error_summary="Agent intent preparation failed before Provider egress",
                )
        elif operation.effect_phase in {
            RemoteComputeEffectPhase.EFFECT_MAY_HAVE_OCCURRED,
            RemoteComputeEffectPhase.PROVIDER_ACCEPTED,
        }:
            try:
                outcome = self._with_heartbeat(
                    lease,
                    lambda: self.effect_client.reconcile(operation),
                )
            except Exception as exc:
                return self._record_unknown(lease, _exception_code(exc))
        else:
            return self.store.record_failure(
                lease,
                certainty=RemoteComputeFailureCertainty.EFFECT_MAY_HAVE_OCCURRED,
                retryable=False,
                error_code="REMOTE_COMPUTE_EFFECT_PHASE_UNKNOWN",
                error_summary="Claimed operation requires reconciliation",
            )
        self._fault(RemoteComputeSagaCrashPoint.AFTER_AGENT_RESPONSE)
        persisted = self._persist_outcome(lease, outcome)
        if outcome.disposition is RemoteComputeEffectDisposition.SUCCEEDED:
            self._fault(RemoteComputeSagaCrashPoint.AFTER_PROVIDER_RECEIPT)
        return persisted

    def _record_pre_effect_failure(
        self,
        lease: RemoteComputeOperationLease,
        operation: RemoteComputeOperation,
        exc: RemoteComputePreEffectFailure,
    ) -> RemoteComputeOperation:
        now = self.clock()
        decision = decide_retry(
            exc.failure_class,
            attempt=operation.attempt_count,
            first_failure_at=operation.created_at,
            now=now,
            retry_after=exc.retry_after,
            random_source=self.retry_random_source,
        )
        return self.store.record_failure(
            lease,
            certainty=RemoteComputeFailureCertainty.CONFIRMED_NO_EFFECT,
            retryable=decision.disposition is RetryDisposition.RETRY,
            retry_after=(
                timedelta(0)
                if decision.next_attempt_at is None
                else max(timedelta(0), decision.next_attempt_at - now)
            ),
            error_code=exc.code,
            error_summary="Agent preparation confirmed no Provider effect",
        )

    def _persist_outcome(
        self,
        lease: RemoteComputeOperationLease,
        outcome: RemoteComputeEffectOutcome,
    ) -> RemoteComputeOperation:
        if not isinstance(outcome, RemoteComputeEffectOutcome):
            return self._record_unknown(lease, "REMOTE_COMPUTE_EFFECT_RESPONSE_INVALID")
        if outcome.disposition is RemoteComputeEffectDisposition.UNKNOWN:
            return self._record_unknown(
                lease,
                outcome.error_code or "REMOTE_COMPUTE_PROVIDER_RESULT_UNKNOWN",
            )
        assert outcome.provider_request_ref is not None
        assert outcome.provider_status is not None
        if outcome.disposition is RemoteComputeEffectDisposition.PENDING:
            return self.store.mark_provider_pending(
                lease,
                provider_request_ref=outcome.provider_request_ref,
                provider_status=outcome.provider_status,
                retry_after=outcome.retry_after,
            )
        assert outcome.result is not None
        if outcome.disposition is RemoteComputeEffectDisposition.REJECTED:
            return self.store.record_provider_rejection(
                lease,
                provider_request_ref=outcome.provider_request_ref,
                provider_status=outcome.provider_status,
                result=outcome.result,
                error_code=outcome.error_code or "REMOTE_COMPUTE_PROVIDER_REJECTED",
                error_summary="Provider returned conclusive no-effect evidence",
            )
        return self.store.record_provider_success(
            lease,
            provider_request_ref=outcome.provider_request_ref,
            provider_status=outcome.provider_status,
            result=outcome.result,
        )

    def _record_unknown(
        self,
        lease: RemoteComputeOperationLease,
        error_code: str,
    ) -> RemoteComputeOperation:
        return self.store.record_failure(
            lease,
            certainty=RemoteComputeFailureCertainty.EFFECT_MAY_HAVE_OCCURRED,
            retryable=False,
            error_code=error_code[:128] or "REMOTE_COMPUTE_PROVIDER_RESULT_UNKNOWN",
            error_summary="Provider effect result is unknown; reconciliation is required",
        )

    def _with_heartbeat(
        self,
        operation_lease: RemoteComputeOperationLease,
        call: Callable[[], _T],
    ) -> _T:
        operation = operation_lease.operation
        assert operation.lease_owner is not None
        assert operation.heartbeat_at is not None
        assert operation.lease_expires_at is not None
        lease = Lease(
            holder=operation.lease_owner,
            generation=operation.lease_generation,
            token_hash=sha256(operation_lease.token).hexdigest(),
            acquired_at=operation.heartbeat_at,
            expires_at=operation.lease_expires_at,
            operation_id=operation.operation_id,
        )

        def renew(current: Lease) -> Lease:
            renewed = self.store.heartbeat(operation_lease)
            assert renewed.lease_expires_at is not None
            return current.renewed(expires_at=renewed.lease_expires_at)

        heartbeat = HeartbeatRunner(
            lease,
            renew,
            interval=self.heartbeat_interval,
            clock=self.clock,
            thread_name=f"remote-compute-saga-{operation.operation_id}",
        )
        heartbeat.start()
        result: _T | object = _MISSING
        error: BaseException | None = None
        try:
            result = call()
        except BaseException as exc:
            error = exc
        stopped = heartbeat.stop(timeout_seconds=self.heartbeat_stop_timeout_seconds)
        if not stopped or heartbeat.error is not None:
            raise RuntimeError("REMOTE_COMPUTE_SAGA_HEARTBEAT_LOST")
        if error is not None:
            raise error
        if result is _MISSING:
            raise RuntimeError("REMOTE_COMPUTE_SAGA_CALL_RETURNED_NO_RESULT")
        return cast(_T, result)

    def _finalize_local(
        self,
        operation: RemoteComputeOperation,
    ) -> Mapping[str, Any] | None:
        assert self.local_finalizer is not None
        self.local_finalizer.finalize(operation)
        return None

    def _fault(self, point: RemoteComputeSagaCrashPoint) -> None:
        if self.fault_injector is not None:
            self.fault_injector(point)


def _exception_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code[:128]
    return f"REMOTE_COMPUTE_{type(exc).__name__.upper()}"[:128]


__all__ = [
    "RemoteComputeEffectDisposition",
    "RemoteComputeEffectOutcome",
    "RemoteComputeEffectPreparation",
    "RemoteComputePreEffectFailure",
    "RemoteComputeSagaCrashPoint",
    "RemoteComputeSagaEffectClient",
    "RemoteComputeSagaFaultInjector",
    "RemoteComputeSagaLocalFinalizer",
    "RemoteComputeSagaWorker",
]
