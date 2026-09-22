"""Agent-side preparation boundary for Product Remote Compute Sagas."""

from __future__ import annotations

import logging

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Protocol
from uuid import UUID

from apps.v2.reliability import HeartbeatRunner, Lease

from .provider_operations import (
    ProviderOperationLease,
    ProviderOperationRecord,
    ProviderOperationStatus,
    ProviderResolution,
    ProviderResolutionState,
)
from .provider_operation_runtime import (
    ProviderOperationCrashPoint,
    ProviderOperationFaultInjector,
)
from .provisioning import ProvisionedRemoteCompute, RemoteComputeProvisioningError
from .saga_models import RemoteComputeOperation, RemoteComputeOperationKind
from .saga_worker import (
    RemoteComputeEffectDisposition,
    RemoteComputeEffectOutcome,
    RemoteComputeEffectPreparation,
)


logger = logging.getLogger(__name__)


class AgentSagaPreparationStore(Protocol):
    def prepare_saga_operation(
        self,
        *,
        operation_id,
        resource_tid,
        owner_uid,
        rid,
        resource_id,
        operation_kind,
        provider,
        provider_operation_id,
        request,
        request_sha256,
    ) -> tuple[ProviderOperationRecord, bool]: ...

    def get(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
    ) -> ProviderOperationRecord | None: ...

    def claim_apply(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
        holder: str,
        lease_ttl: timedelta,
    ) -> ProviderOperationLease | None: ...

    def heartbeat(
        self,
        lease: ProviderOperationLease,
        *,
        lease_ttl: timedelta,
    ) -> ProviderOperationLease: ...

    def persist_apply_intent(
        self,
        lease: ProviderOperationLease,
        intent: Mapping[str, Any],
    ) -> object: ...

    def mark_effect_started(
        self,
        lease: ProviderOperationLease,
    ) -> ProviderOperationRecord: ...

    def complete_apply(
        self,
        lease: ProviderOperationLease,
        receipt: Mapping[str, Any],
    ) -> ProviderOperationRecord: ...

    def complete_apply_failure(
        self,
        lease: ProviderOperationLease,
        *,
        error_code: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> ProviderOperationRecord: ...

    def mark_result_unknown(
        self,
        lease: ProviderOperationLease,
        error_code: str,
    ) -> ProviderOperationRecord: ...

    def claim_saga_reconciliation(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
        holder: str,
        lease_ttl: timedelta,
    ) -> ProviderOperationLease | None: ...

    def complete_saga_reconciliation(
        self,
        lease: ProviderOperationLease,
        receipt: Mapping[str, Any],
    ) -> ProviderOperationRecord: ...

    def resolve_reconciliation(
        self,
        lease: ProviderOperationLease,
        resolution: ProviderResolution,
    ) -> ProviderOperationRecord: ...


class AgentSagaProvisioner(Protocol):
    def provision(
        self,
        *,
        resource_id: UUID,
        target: str,
        spec: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute: ...

    def activate(
        self,
        *,
        resource_id: UUID,
        resource: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute: ...

    def reconcile_provision(
        self,
        *,
        resource_id: UUID,
        target: str,
        spec: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute | None: ...

    def reconcile_activate(
        self,
        *,
        resource_id: UUID,
        resource: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute | None: ...

    def reconcile_release(
        self,
        provisioned: ProvisionedRemoteCompute,
    ) -> bool | None: ...

    def rollback(self, provisioned: ProvisionedRemoteCompute) -> None: ...


class AgentRemoteComputeSagaNotFound(RuntimeError):
    code = "REMOTE_COMPUTE_OPERATION_NOT_FOUND"


class _ConclusiveNoEffect(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class AgentRemoteComputeSagaPreparationService:
    """Persist Agent intent and return a non-dispatching receipt.

    Calling this service never claims the operation and never invokes a
    provisioner or Provider.  That guarantee is what permits Product to retry
    preparation before it persists its own ambiguity boundary.
    """

    def __init__(
        self,
        store: AgentSagaPreparationStore,
        *,
        fault_injector: ProviderOperationFaultInjector | None = None,
    ) -> None:
        self.store = store
        self.fault_injector = fault_injector

    def prepare(
        self,
        operation: RemoteComputeOperation,
    ) -> tuple[RemoteComputeEffectPreparation, bool]:
        if operation.rid is None or operation.resource_id is None:
            raise ValueError("Remote Compute Saga preparation requires run and resource")
        return self.prepare_fields(
            operation_id=operation.operation_id,
            resource_tid=operation.resource_tid,
            owner_uid=operation.owner_uid,
            rid=operation.rid,
            resource_id=operation.resource_id,
            operation_kind=operation.operation_kind,
            provider=operation.provider,
            provider_operation_id=operation.provider_operation_id,
            request=operation.request,
            request_sha256=operation.request_sha256,
        )

    def prepare_fields(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
        rid: UUID,
        resource_id: UUID,
        operation_kind: RemoteComputeOperationKind,
        provider: str,
        provider_operation_id: str,
        request: Mapping[str, Any],
        request_sha256: bytes,
    ) -> tuple[RemoteComputeEffectPreparation, bool]:
        prepared, replayed = self.store.prepare_saga_operation(
            operation_id=operation_id,
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            rid=rid,
            resource_id=resource_id,
            operation_kind=operation_kind,
            provider=provider,
            provider_operation_id=provider_operation_id,
            request=request,
            request_sha256=request_sha256.hex(),
        )
        self._fault(ProviderOperationCrashPoint.AFTER_OPERATION_COMMIT)
        if (
            prepared.operation_id != operation_id
            or prepared.resource_tid != resource_tid
            or prepared.owner_uid != owner_uid
            or prepared.rid != rid
            or prepared.resource_id != resource_id
            or prepared.idempotency_key != provider_operation_id
            or prepared.status.value != "PENDING"
            or prepared.effect_phase != "PRE_EFFECT"
            or prepared.claim_kind is not None
        ):
            raise RuntimeError("REMOTE_COMPUTE_PREPARATION_RECEIPT_INVALID")
        return (
            RemoteComputeEffectPreparation(
                operation_id=prepared.operation_id,
                provider_operation_id=prepared.idempotency_key,
                dispatch_ref=(
                    "/internal/v2/remote-compute/operations/"
                    f"{provider_operation_id}"
                ),
            ),
            replayed,
        )

    def _fault(self, point: ProviderOperationCrashPoint) -> None:
        if self.fault_injector is not None:
            self.fault_injector(point)


class AgentRemoteComputeSagaExecutor:
    """Execute an already-prepared Agent operation with a durable receipt."""

    def __init__(
        self,
        store: AgentSagaPreparationStore,
        provisioner: AgentSagaProvisioner,
        *,
        holder: str,
        lease_ttl: timedelta = timedelta(seconds=120),
        heartbeat_interval: timedelta = timedelta(seconds=20),
        heartbeat_stop_timeout_seconds: float = 5.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        fault_injector: ProviderOperationFaultInjector | None = None,
    ) -> None:
        if not holder or len(holder.encode("utf-8")) > 256:
            raise ValueError("Agent Remote Compute Saga holder is invalid")
        if heartbeat_interval * 3 >= lease_ttl:
            raise ValueError("heartbeat interval must be below one third of lease TTL")
        if heartbeat_stop_timeout_seconds <= 0:
            raise ValueError("heartbeat stop timeout must be positive")
        self.store = store
        self.provisioner = provisioner
        self.holder = holder
        self.lease_ttl = lease_ttl
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_stop_timeout_seconds = heartbeat_stop_timeout_seconds
        self.clock = clock
        self.fault_injector = fault_injector

    def apply(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
        preparation: RemoteComputeEffectPreparation,
    ) -> RemoteComputeEffectOutcome:
        operation = self._required_operation(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            operation_id=operation_id,
        )
        if (
            preparation.operation_id != operation_id
            or preparation.provider_operation_id != operation.idempotency_key
            or preparation.dispatch_ref
            != (
                "/internal/v2/remote-compute/operations/"
                f"{operation.idempotency_key}"
            )
        ):
            raise ValueError("REMOTE_COMPUTE_PREPARATION_INVALID")
        existing = _operation_outcome(operation)
        if existing is not None:
            return existing
        if operation.status is ProviderOperationStatus.PROCESSING:
            return _pending_outcome(operation)
        if operation.status is not ProviderOperationStatus.PENDING:
            return _unknown_outcome(
                operation,
                operation.last_error_code or "REMOTE_COMPUTE_PROVIDER_RESULT_UNKNOWN",
            )
        lease = self.store.claim_apply(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            operation_id=operation_id,
            holder=self.holder,
            lease_ttl=self.lease_ttl,
        )
        if lease is None:
            current = self._required_operation(
                resource_tid=resource_tid,
                owner_uid=owner_uid,
                operation_id=operation_id,
            )
            return _operation_outcome(current) or _pending_outcome(current)

        self.store.persist_apply_intent(
            lease,
            {
                "action": lease.operation.action,
                "idempotency_key": lease.operation.idempotency_key,
                "provider": lease.operation.provider,
                "request_sha256": lease.operation.request_sha256,
            },
        )
        self._fault(ProviderOperationCrashPoint.AFTER_APPLY_INTENT)
        self.store.mark_effect_started(lease)
        self._fault(ProviderOperationCrashPoint.AFTER_EFFECT_STARTED)
        try:
            receipt = self._with_heartbeat(
                lease,
                lambda: self._dispatch(lease.operation),
            )
            assert receipt is not None
            self._fault(ProviderOperationCrashPoint.AFTER_PROVIDER_RESPONSE)
        except _ConclusiveNoEffect as exc:
            failed = self.store.complete_apply_failure(
                lease,
                error_code=exc.code,
                evidence={"effect": "NONE", "reason": exc.code},
            )
            return _operation_outcome(failed) or _unknown_outcome(failed, exc.code)
        except Exception as exc:
            code = _exception_code(exc)
            logger.warning(
                "remote compute %s apply failed with an unknown effect: %s",
                lease.operation.action,
                code,
                exc_info=True,
            )
            try:
                unknown = self.store.mark_result_unknown(lease, code)
            except Exception:
                unknown = lease.operation
            return _unknown_outcome(unknown, code)
        try:
            completed = self.store.complete_apply(lease, receipt)
            self._fault(ProviderOperationCrashPoint.AFTER_APPLY_RECEIPT)
        except Exception as exc:
            code = _exception_code(exc)
            try:
                unknown = self.store.mark_result_unknown(lease, code)
            except Exception:
                unknown = lease.operation
            return _unknown_outcome(unknown, code)
        return _operation_outcome(completed) or _unknown_outcome(
            completed,
            "REMOTE_COMPUTE_AGENT_RECEIPT_INVALID",
        )

    def reconcile(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
    ) -> RemoteComputeEffectOutcome:
        operation = self._required_operation(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            operation_id=operation_id,
        )
        existing = _operation_outcome(operation)
        if existing is not None:
            self._fault(ProviderOperationCrashPoint.AFTER_RECONCILE_RESPONSE)
            return existing
        lease = self.store.claim_saga_reconciliation(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            operation_id=operation_id,
            holder=self.holder,
            lease_ttl=self.lease_ttl,
        )
        if lease is None:
            current = self._required_operation(
                resource_tid=resource_tid,
                owner_uid=owner_uid,
                operation_id=operation_id,
            )
            outcome = (
                _pending_outcome(current)
                if current.status in {
                    ProviderOperationStatus.PENDING,
                    ProviderOperationStatus.PROCESSING,
                    ProviderOperationStatus.RECONCILING,
                }
                else _unknown_outcome(
                    current,
                    current.last_error_code
                    or "REMOTE_COMPUTE_PROVIDER_RESULT_UNKNOWN",
                )
            )
            self._fault(ProviderOperationCrashPoint.AFTER_RECONCILE_RESPONSE)
            return outcome
        try:
            receipt = self._with_heartbeat(
                lease,
                lambda: self._reconcile_dispatch(lease.operation),
            )
            self._fault(ProviderOperationCrashPoint.AFTER_RECONCILE_RESPONSE)
        except Exception as exc:
            deferred = self.store.resolve_reconciliation(
                lease,
                ProviderResolution(
                    ProviderResolutionState.UNKNOWN,
                    {"runtime_error": type(exc).__name__},
                    _exception_code(exc),
                ),
            )
            return _unknown_outcome(
                deferred,
                deferred.last_error_code or _exception_code(exc),
            )
        if receipt is None:
            deferred = self.store.resolve_reconciliation(
                lease,
                ProviderResolution(
                    ProviderResolutionState.UNKNOWN,
                    {"stable_identity_found": False},
                    "REMOTE_COMPUTE_PROVIDER_RESULT_UNKNOWN",
                ),
            )
            return _unknown_outcome(
                deferred,
                deferred.last_error_code or "REMOTE_COMPUTE_PROVIDER_RESULT_UNKNOWN",
            )
        completed = self.store.complete_saga_reconciliation(lease, receipt)
        self._fault(ProviderOperationCrashPoint.AFTER_RECONCILE_RECEIPT)
        outcome = _operation_outcome(completed) or _unknown_outcome(
            completed,
            "REMOTE_COMPUTE_AGENT_RECEIPT_INVALID",
        )
        return outcome

    def _reconcile_dispatch(
        self,
        operation: ProviderOperationRecord,
    ) -> Mapping[str, Any] | None:
        try:
            kind = RemoteComputeOperationKind(operation.action)
        except ValueError as exc:
            raise _ConclusiveNoEffect(
                "REMOTE_COMPUTE_OPERATION_KIND_UNSUPPORTED"
            ) from exc
        if kind is RemoteComputeOperationKind.PROVISION:
            target = operation.request.get("target")
            spec = operation.request.get("spec")
            if not isinstance(target, str) or not target or not isinstance(spec, Mapping):
                raise _ConclusiveNoEffect("REMOTE_COMPUTE_REQUEST_INVALID")
            reconciled = self.provisioner.reconcile_provision(
                resource_id=operation.resource_id,
                target=target,
                spec=spec,
            )
            return None if reconciled is None else reconciled.internal_dict()
        if kind is RemoteComputeOperationKind.ACTIVATE:
            resource = operation.request.get("resource")
            if not isinstance(resource, Mapping):
                raise _ConclusiveNoEffect("REMOTE_COMPUTE_REQUEST_INVALID")
            reconciled = self.provisioner.reconcile_activate(
                resource_id=operation.resource_id,
                resource=resource,
            )
            return None if reconciled is None else reconciled.internal_dict()
        if kind in {
            RemoteComputeOperationKind.RELEASE,
            RemoteComputeOperationKind.COMPENSATE,
        }:
            try:
                provisioned = ProvisionedRemoteCompute.from_dict(
                    operation.request.get("provisioned")
                )
            except ValueError as exc:
                raise _ConclusiveNoEffect("REMOTE_COMPUTE_REQUEST_INVALID") from exc
            released = self.provisioner.reconcile_release(provisioned)
            if released is not True:
                return None
            return {
                "released": True,
                "confirmed_not_found": True,
                "resource_id": str(operation.resource_id),
            }
        return None

    def _required_operation(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
    ) -> ProviderOperationRecord:
        operation = self.store.get(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            operation_id=operation_id,
        )
        if operation is None:
            raise AgentRemoteComputeSagaNotFound(
                "Agent Remote Compute Saga operation does not exist"
            )
        return operation

    def _dispatch(self, operation: ProviderOperationRecord) -> Mapping[str, Any]:
        try:
            kind = RemoteComputeOperationKind(operation.action)
        except ValueError as exc:
            raise _ConclusiveNoEffect("REMOTE_COMPUTE_OPERATION_KIND_UNSUPPORTED") from exc
        request = operation.request
        if kind is RemoteComputeOperationKind.PROVISION:
            target = request.get("target")
            spec = request.get("spec")
            if not isinstance(target, str) or not target or not isinstance(spec, Mapping):
                raise _ConclusiveNoEffect("REMOTE_COMPUTE_REQUEST_INVALID")
            return self.provisioner.provision(
                resource_id=operation.resource_id,
                target=target,
                spec=spec,
            ).internal_dict()
        if kind is RemoteComputeOperationKind.ACTIVATE:
            resource = request.get("resource")
            if not isinstance(resource, Mapping):
                raise _ConclusiveNoEffect("REMOTE_COMPUTE_REQUEST_INVALID")
            return self.provisioner.activate(
                resource_id=operation.resource_id,
                resource=resource,
            ).internal_dict()
        if kind in {
            RemoteComputeOperationKind.RELEASE,
            RemoteComputeOperationKind.COMPENSATE,
        }:
            try:
                provisioned = ProvisionedRemoteCompute.from_dict(
                    request.get("provisioned")
                )
            except ValueError as exc:
                raise _ConclusiveNoEffect("REMOTE_COMPUTE_REQUEST_INVALID") from exc
            self.provisioner.rollback(provisioned)
            return {
                "released": True,
                "resource_id": str(operation.resource_id),
            }
        raise _ConclusiveNoEffect("REMOTE_COMPUTE_OPERATION_KIND_UNSUPPORTED")

    def _with_heartbeat(
        self,
        operation_lease: ProviderOperationLease,
        call: Callable[[], Mapping[str, Any] | None],
    ) -> Mapping[str, Any] | None:
        operation = operation_lease.operation
        assert operation.heartbeat_at is not None
        assert operation.claim_expires_at is not None
        lease = Lease(
            holder=operation_lease.holder,
            generation=operation_lease.generation,
            token_hash=sha256(operation_lease.token).hexdigest(),
            acquired_at=operation.heartbeat_at,
            expires_at=operation.claim_expires_at,
            operation_id=operation.operation_id,
        )

        def renew(current: Lease) -> Lease:
            renewed = self.store.heartbeat(
                operation_lease,
                lease_ttl=self.lease_ttl,
            )
            return current.renewed(expires_at=renewed.expires_at)

        heartbeat = HeartbeatRunner(
            lease,
            renew,
            interval=self.heartbeat_interval,
            clock=self.clock,
            thread_name=f"agent-remote-compute-saga-{operation.operation_id}",
        )
        heartbeat.start()
        result: Mapping[str, Any] | None = None
        error: BaseException | None = None
        try:
            result = call()
        except BaseException as exc:
            error = exc
        stopped = heartbeat.stop(timeout_seconds=self.heartbeat_stop_timeout_seconds)
        if not stopped or heartbeat.error is not None:
            raise RuntimeError("REMOTE_COMPUTE_AGENT_HEARTBEAT_LOST")
        if error is not None:
            raise error
        return result

    def _fault(self, point: ProviderOperationCrashPoint) -> None:
        if self.fault_injector is not None:
            self.fault_injector(point)


def _operation_outcome(
    operation: ProviderOperationRecord,
) -> RemoteComputeEffectOutcome | None:
    if operation.status is ProviderOperationStatus.SUCCEEDED:
        if operation.provider_receipt is None:
            return _unknown_outcome(
                operation,
                "REMOTE_COMPUTE_AGENT_RECEIPT_INVALID",
            )
        return RemoteComputeEffectOutcome(
            RemoteComputeEffectDisposition.SUCCEEDED,
            provider_request_ref=operation.idempotency_key,
            provider_status=operation.status.value,
            result=operation.provider_receipt,
        )
    if operation.status is ProviderOperationStatus.FAILED:
        return RemoteComputeEffectOutcome(
            RemoteComputeEffectDisposition.REJECTED,
            provider_request_ref=operation.idempotency_key,
            provider_status=operation.status.value,
            result=operation.provider_receipt or {"effect": "NONE"},
            error_code=operation.last_error_code or "REMOTE_COMPUTE_PROVIDER_REJECTED",
        )
    return None


def _pending_outcome(operation: ProviderOperationRecord) -> RemoteComputeEffectOutcome:
    return RemoteComputeEffectOutcome(
        RemoteComputeEffectDisposition.PENDING,
        provider_request_ref=operation.idempotency_key,
        provider_status=operation.status.value,
        retry_after=timedelta(seconds=2),
    )


def _unknown_outcome(
    operation: ProviderOperationRecord,
    code: str,
) -> RemoteComputeEffectOutcome:
    return RemoteComputeEffectOutcome(
        RemoteComputeEffectDisposition.UNKNOWN,
        provider_request_ref=operation.idempotency_key,
        provider_status=operation.status.value,
        error_code=code[:128],
    )


def _exception_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code[:128]
    if isinstance(exc, RemoteComputeProvisioningError):
        return exc.code[:128]
    return f"REMOTE_COMPUTE_{type(exc).__name__.upper()}"[:128]


__all__ = [
    "AgentRemoteComputeSagaExecutor",
    "AgentRemoteComputeSagaNotFound",
    "AgentRemoteComputeSagaPreparationService",
    "AgentSagaPreparationStore",
    "AgentSagaProvisioner",
]
