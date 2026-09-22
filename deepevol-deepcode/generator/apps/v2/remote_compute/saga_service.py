"""Provider-neutral application service for Remote Compute operation Sagas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Protocol
from uuid import UUID

from apps.common.v2_ids import format_typed_id

from .saga_models import (
    RemoteComputeFailureCertainty,
    RemoteComputeOperation,
    RemoteComputeOperationAccepted,
    RemoteComputeOperationCreate,
    RemoteComputeOperationLease,
    RemoteComputeOperationStatus,
    RemoteComputeOperationStatusView,
)
from .saga_postgres import RemoteComputeOperationNotFound


class RemoteComputeSagaStore(Protocol):
    """Product-only durable boundary; deliberately contains no Provider call."""

    def create_or_get(
        self,
        request: RemoteComputeOperationCreate,
    ) -> tuple[RemoteComputeOperation, bool]: ...

    def claim_next(self, *, lease_owner: str) -> RemoteComputeOperationLease | None: ...

    def heartbeat(self, lease: RemoteComputeOperationLease) -> RemoteComputeOperation: ...

    def require_live(self, lease: RemoteComputeOperationLease) -> None: ...

    def mark_effect_started(
        self,
        lease: RemoteComputeOperationLease,
    ) -> RemoteComputeOperation: ...

    def mark_provider_pending(
        self,
        lease: RemoteComputeOperationLease,
        *,
        provider_request_ref: str,
        provider_status: str,
        retry_after: timedelta,
    ) -> RemoteComputeOperation: ...

    def record_provider_success(
        self,
        lease: RemoteComputeOperationLease,
        *,
        provider_request_ref: str,
        provider_status: str,
        result: Mapping[str, object],
    ) -> RemoteComputeOperation: ...

    def record_provider_rejection(
        self,
        lease: RemoteComputeOperationLease,
        *,
        provider_request_ref: str,
        provider_status: str,
        result: Mapping[str, object],
        error_code: str,
        error_summary: str,
    ) -> RemoteComputeOperation: ...

    def record_failure(
        self,
        lease: RemoteComputeOperationLease,
        *,
        certainty: RemoteComputeFailureCertainty,
        retryable: bool,
        error_code: str,
        error_summary: str,
        retry_after: timedelta = timedelta(0),
    ) -> RemoteComputeOperation: ...

    def complete(self, lease: RemoteComputeOperationLease) -> RemoteComputeOperation: ...

    def cancel(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
    ) -> RemoteComputeOperation: ...

    def get_status(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
    ) -> RemoteComputeOperationStatusView | None: ...

    def list_statuses(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        statuses: Sequence[RemoteComputeOperationStatus] = (),
        limit: int = 100,
    ) -> tuple[RemoteComputeOperationStatusView, ...]: ...


class RemoteComputeSagaService:
    """Orchestration facade shared by the live HTTP and worker adapters.

    Returning an explicit 202 envelope makes the durable polling contract
    usable before any Provider adapter is wired.  Provider execution remains
    outside this service so an external effect can never be hidden inside a
    Product transaction.
    """

    def __init__(self, store: RemoteComputeSagaStore) -> None:
        self.store = store

    def submit(self, request: RemoteComputeOperationCreate) -> RemoteComputeOperationAccepted:
        _operation, replayed = self.store.create_or_get(request)
        status = self.get_status(
            operation_id=request.operation_id,
            resource_tid=request.resource_tid,
            owner_uid=request.owner_uid,
        )
        return RemoteComputeOperationAccepted(
            operation=status,
            status_path=self.status_path(request.operation_id),
            replayed=replayed,
        )

    def get_status(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
    ) -> RemoteComputeOperationStatusView:
        status = self.store.get_status(
            operation_id=operation_id,
            resource_tid=resource_tid,
            owner_uid=owner_uid,
        )
        if status is None:
            raise RemoteComputeOperationNotFound("Remote Compute operation does not exist")
        return status

    def list_statuses(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        statuses: Sequence[RemoteComputeOperationStatus] = (),
        limit: int = 100,
    ) -> tuple[RemoteComputeOperationStatusView, ...]:
        return self.store.list_statuses(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            statuses=statuses,
            limit=limit,
        )

    def cancel(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
    ) -> RemoteComputeOperationStatusView:
        self.store.cancel(
            operation_id=operation_id,
            resource_tid=resource_tid,
            owner_uid=owner_uid,
        )
        return self.get_status(
            operation_id=operation_id,
            resource_tid=resource_tid,
            owner_uid=owner_uid,
        )

    @staticmethod
    def status_path(operation_id: UUID) -> str:
        return (
            "/internal/v2/remote-compute/operations/"
            f"{format_typed_id('op', operation_id)}"
        )


__all__ = ["RemoteComputeSagaService", "RemoteComputeSagaStore"]
