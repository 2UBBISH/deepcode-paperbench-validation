"""Idempotent Product-local adoption of durable Remote Compute receipts.

This boundary deliberately runs after the Agent Provider receipt is durable.
It may be repeated after a process crash, but it can never dispatch a Provider
operation.  The resource catalog and run binding are the local effect receipts.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from apps.common.v2_ids import InvalidTypedId, parse_typed_id

from .lifecycle import RemoteComputeRunLifecycle
from .models import RemoteComputeResource, RemoteWorkspaceBinding
from .provisioning import ProvisionedRemoteCompute
from .saga_models import RemoteComputeOperation, RemoteComputeOperationKind


class ProductRemoteComputeCatalog(Protocol):
    def get_resource(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        resource_id: UUID,
    ) -> RemoteComputeResource | None: ...

    def upsert_provisioned_resource(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        provisioned: ProvisionedRemoteCompute,
    ) -> tuple[RemoteComputeResource, bool]: ...

    def finalize_provider_release(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        resource_id: UUID,
        binding_id: UUID | None = None,
    ) -> bool: ...


class ProductRemoteComputeLifecycle(Protocol):
    def start(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        sid: UUID,
        rid: UUID,
        resource_id: UUID,
        binding_id: UUID | None = None,
        estimated_credits: int = 0,
        resource_policy: Mapping[str, int] | None = None,
        expires_at: datetime | None = None,
        started_at: datetime | None = None,
    ) -> tuple[RemoteWorkspaceBinding, bool]: ...


class RemoteComputeSagaFinalizationError(RuntimeError):
    code = "REMOTE_COMPUTE_LOCAL_FINALIZATION_INVALID"


class ProductRemoteComputeSagaFinalizer:
    """Adopt PROVISION/ACTIVATE receipts using stable Product identities."""

    def __init__(
        self,
        catalog: ProductRemoteComputeCatalog,
        lifecycle: ProductRemoteComputeLifecycle | RemoteComputeRunLifecycle,
    ) -> None:
        self.catalog = catalog
        self.lifecycle = lifecycle

    def finalize(self, operation: RemoteComputeOperation) -> None:
        if operation.operation_kind in {
            RemoteComputeOperationKind.RELEASE,
            RemoteComputeOperationKind.COMPENSATE,
        }:
            self._finalize_release(operation)
            return
        if operation.operation_kind not in {
            RemoteComputeOperationKind.PROVISION,
            RemoteComputeOperationKind.ACTIVATE,
        }:
            raise RemoteComputeSagaFinalizationError(
                "operation kind has no registered Product finalizer"
            )
        if (
            operation.result is None
            or operation.resource_id is None
            or operation.binding_id is None
            or operation.rid is None
        ):
            raise RemoteComputeSagaFinalizationError(
                "operation receipt or Product identity is incomplete"
            )
        try:
            provisioned = ProvisionedRemoteCompute.from_dict(operation.result)
        except (TypeError, ValueError) as exc:
            raise RemoteComputeSagaFinalizationError(
                "Provider receipt is not a provisioned resource"
            ) from exc
        if (
            provisioned.resource_id != operation.resource_id
            or provisioned.provider != operation.provider
        ):
            raise RemoteComputeSagaFinalizationError(
                "Provider receipt differs from the admitted Product identity"
            )

        sid, estimated_credits, resource_policy, expires_at = _adoption_context(
            operation.request
        )
        resource = self.catalog.get_resource(
            resource_tid=operation.resource_tid,
            owner_uid=operation.owner_uid,
            resource_id=operation.resource_id,
        )
        if resource is None:
            resource, _resource_replayed = self.catalog.upsert_provisioned_resource(
                resource_tid=operation.resource_tid,
                owner_uid=operation.owner_uid,
                provisioned=provisioned,
            )
        if (
            resource.resource_id != operation.resource_id
            or resource.provider != operation.provider
            or resource.external_resource_id != provisioned.external_resource_id
        ):
            raise RemoteComputeSagaFinalizationError(
                "Product resource receipt differs from the operation"
            )
        binding, _binding_replayed = self.lifecycle.start(
            resource_tid=operation.resource_tid,
            owner_uid=operation.owner_uid,
            sid=sid,
            rid=operation.rid,
            resource_id=operation.resource_id,
            binding_id=operation.binding_id,
            estimated_credits=estimated_credits,
            resource_policy=resource_policy,
            expires_at=expires_at,
        )
        if (
            binding.binding_id != operation.binding_id
            or binding.resource_id != operation.resource_id
            or binding.resource_tid != operation.resource_tid
            or binding.owner_uid != operation.owner_uid
            or binding.sid != sid
            or binding.rid != operation.rid
        ):
            raise RemoteComputeSagaFinalizationError(
                "Product binding receipt differs from the operation"
            )

    def _finalize_release(self, operation: RemoteComputeOperation) -> None:
        if operation.resource_id is None or operation.result is None:
            raise RemoteComputeSagaFinalizationError(
                "release receipt or Product resource identity is incomplete"
            )
        if operation.result.get("released") is not True:
            raise RemoteComputeSagaFinalizationError(
                "release receipt does not prove Provider release"
            )
        receipt_resource_id = operation.result.get("resource_id")
        if receipt_resource_id not in {
            str(operation.resource_id),
            None,
        }:
            raise RemoteComputeSagaFinalizationError(
                "release receipt differs from the Product resource identity"
            )
        self.catalog.finalize_provider_release(
            resource_tid=operation.resource_tid,
            owner_uid=operation.owner_uid,
            resource_id=operation.resource_id,
            binding_id=operation.binding_id,
        )


def _adoption_context(
    request: Mapping[str, Any],
) -> tuple[UUID, int, dict[str, int], datetime | None]:
    raw = request.get("product_adoption")
    if not isinstance(raw, Mapping):
        raise RemoteComputeSagaFinalizationError(
            "operation has no Product adoption context"
        )
    try:
        sid = parse_typed_id(raw.get("sid"), expected_prefix="sid")
    except (InvalidTypedId, TypeError) as exc:
        raise RemoteComputeSagaFinalizationError(
            "Product adoption session identity is invalid"
        ) from exc
    estimated_credits = raw.get("estimated_credits")
    if (
        isinstance(estimated_credits, bool)
        or not isinstance(estimated_credits, int)
        or estimated_credits < 0
    ):
        raise RemoteComputeSagaFinalizationError(
            "Product adoption credit estimate is invalid"
        )
    policy_raw = raw.get("resource_policy")
    if not isinstance(policy_raw, Mapping) or any(
        not isinstance(key, str)
        or not key
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in policy_raw.items()
    ):
        raise RemoteComputeSagaFinalizationError(
            "Product adoption resource policy is invalid"
        )
    expires_raw = raw.get("expires_at")
    if expires_raw is None:
        expires_at = None
    elif isinstance(expires_raw, str):
        try:
            expires_at = datetime.fromisoformat(expires_raw)
        except ValueError as exc:
            raise RemoteComputeSagaFinalizationError(
                "Product adoption expiry is invalid"
            ) from exc
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise RemoteComputeSagaFinalizationError(
                "Product adoption expiry is invalid"
            )
    else:
        raise RemoteComputeSagaFinalizationError(
            "Product adoption expiry is invalid"
        )
    return sid, estimated_credits, dict(policy_raw), expires_at


__all__ = [
    "ProductRemoteComputeSagaFinalizer",
    "RemoteComputeSagaFinalizationError",
]
