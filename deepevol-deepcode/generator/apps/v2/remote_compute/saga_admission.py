"""Product Billing admission for Remote Compute Saga operations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from psycopg import Connection

from apps.common.v2_ids import InvalidTypedId, parse_typed_id

from .models import RemoteComputePricingContext, RemoteWorkspaceBinding
from .saga_models import (
    RemoteComputeOperation,
    RemoteComputeOperationCreate,
    RemoteComputeOperationKind,
)
from .saga_postgres import PsycopgRemoteComputeSagaRepository


class RemoteComputeAdmissionBilling(Protocol):
    def authorize_remote_compute(
        self,
        connection: Connection[Any],
        binding: RemoteWorkspaceBinding,
        pricing: RemoteComputePricingContext,
    ) -> None: ...


class RemoteComputeSagaAdmissionError(RuntimeError):
    code = "REMOTE_COMPUTE_BILLING_ADMISSION_INVALID"


class ProductRemoteComputeSagaAdmission:
    """Authorize one stable operation before it becomes visible to workers."""

    def __init__(self, billing: RemoteComputeAdmissionBilling) -> None:
        self.billing = billing

    def admit(
        self,
        connection: Connection[Any],
        request: RemoteComputeOperationCreate,
    ) -> None:
        if request.operation_kind not in {
            RemoteComputeOperationKind.PROVISION,
            RemoteComputeOperationKind.ACTIVATE,
        }:
            raise RemoteComputeSagaAdmissionError(
                "only allocation operations require this admission path"
            )
        if (
            request.resource_id is None
            or request.binding_id is None
            or request.rid is None
        ):
            raise RemoteComputeSagaAdmissionError(
                "allocation operation has no stable Product identities"
            )
        adoption = _required_mapping(request.request, "product_adoption")
        sid = _typed(adoption.get("sid"), "sid")
        started_at = _aware_datetime(adoption.get("started_at"), "started_at")
        expires_at = _optional_aware_datetime(adoption.get("expires_at"), "expires_at")
        if expires_at is not None and expires_at <= started_at:
            raise RemoteComputeSagaAdmissionError(
                "allocation expiry must follow admission time"
            )
        estimated_credits = _non_negative_int(
            adoption.get("estimated_credits"),
            "estimated_credits",
        )
        pricing_raw = _required_mapping(adoption, "pricing")
        pricing = RemoteComputePricingContext(
            provider=_required_text(pricing_raw.get("provider"), "provider", 128),
            billing_mode=_required_text(
                pricing_raw.get("billing_mode"),
                "billing_mode",
                32,
            ),
            cpu_cores=_non_negative_int(pricing_raw.get("cpu_cores"), "cpu_cores"),
        )
        if pricing.provider.strip().lower() != request.provider.strip().lower():
            raise RemoteComputeSagaAdmissionError(
                "frozen pricing provider differs from operation provider"
            )

        run = connection.execute(
            """
            SELECT sid, payer_tid, billing_account_id, state, deleted_at
              FROM conversation.runs
             WHERE rid = %(rid)s
               AND resource_tid = %(resource_tid)s
               AND owner_uid = %(owner_uid)s
             FOR UPDATE
            """,
            {
                "rid": request.rid,
                "resource_tid": request.resource_tid,
                "owner_uid": request.owner_uid,
            },
        ).fetchone()
        if (
            run is None
            or run["deleted_at"] is not None
            or run["sid"] != sid
            or str(run["state"]) in {"SUCCEEDED", "FAILED", "CANCELLED"}
            or run["payer_tid"] is None
            or run["billing_account_id"] is None
        ):
            raise RemoteComputeSagaAdmissionError(
                "allocation run has no active payer authority"
            )
        # The run row lock serializes competing selections for one run.  The
        # current intent is already visible in this transaction; reject any
        # other non-failed operation before a second credit hold or Provider
        # exposure can be created.
        competing = connection.execute(
            """
            SELECT operation_id
              FROM ops.remote_compute_operations
             WHERE resource_tid = %(resource_tid)s
               AND owner_uid = %(owner_uid)s
               AND rid = %(rid)s
               AND operation_id <> %(operation_id)s
               AND status NOT IN ('FAILED', 'CANCELLED', 'COMPENSATED')
             LIMIT 1
            """,
            {
                "operation_id": request.operation_id,
                "rid": request.rid,
                "resource_tid": request.resource_tid,
                "owner_uid": request.owner_uid,
            },
        ).fetchone()
        if competing is not None:
            raise RemoteComputeSagaAdmissionError(
                "run already has an active Remote Compute selection"
            )
        binding = RemoteWorkspaceBinding(
            binding_id=request.binding_id,
            resource_id=request.resource_id,
            resource_tid=request.resource_tid,
            owner_uid=request.owner_uid,
            sid=sid,
            rid=request.rid,
            lease_generation=1,
            status="ADMISSION_PENDING",
            remote_root="/pending",
            secret_ref="pending/provider-receipt",
            secret_version="pending",
            created_at=started_at,
            expires_at=expires_at,
            payer_tid=run["payer_tid"],
            billing_account_id=run["billing_account_id"],
            hourly_price_credits=_non_negative_int(
                pricing_raw.get("hourly_price_credits"),
                "hourly_price_credits",
            ),
            estimated_credits=estimated_credits,
            started_at=started_at,
            resource_policy=_resource_policy(adoption.get("resource_policy")),
        )
        self.billing.authorize_remote_compute(connection, binding, pricing)


class ProductRemoteComputeSagaSubmitter:
    def __init__(
        self,
        repository: PsycopgRemoteComputeSagaRepository,
        admission: ProductRemoteComputeSagaAdmission,
    ) -> None:
        self.repository = repository
        self.admission = admission

    def submit(
        self,
        request: RemoteComputeOperationCreate,
    ) -> tuple[RemoteComputeOperation, bool]:
        return self.repository.create_or_get_admitted(
            request,
            admission=self.admission.admit,
        )

    def get(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
    ) -> RemoteComputeOperation | None:
        """Read the original intent before rebuilding any mutable catalog input."""

        return self.repository.get(
            operation_id=operation_id,
            resource_tid=resource_tid,
            owner_uid=owner_uid,
        )


def _required_mapping(parent: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = parent.get(field)
    if not isinstance(value, Mapping):
        raise RemoteComputeSagaAdmissionError(f"{field} must be an object")
    return value


def _typed(value: object, prefix: str):
    try:
        return parse_typed_id(value, expected_prefix=prefix)
    except (InvalidTypedId, TypeError) as exc:
        raise RemoteComputeSagaAdmissionError(
            f"{prefix} identity is invalid"
        ) from exc


def _required_text(value: object, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > maximum
        or "\x00" in value
    ):
        raise RemoteComputeSagaAdmissionError(f"{field} is invalid")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RemoteComputeSagaAdmissionError(f"{field} is invalid")
    return value


def _aware_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise RemoteComputeSagaAdmissionError(f"{field} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RemoteComputeSagaAdmissionError(f"{field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RemoteComputeSagaAdmissionError(f"{field} is invalid")
    return parsed


def _optional_aware_datetime(value: object, field: str) -> datetime | None:
    return None if value is None else _aware_datetime(value, field)


def _resource_policy(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or len(value) > 32 or any(
        not isinstance(key, str)
        or not key
        or len(key.encode("utf-8")) > 128
        or isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        for key, item in value.items()
    ):
        raise RemoteComputeSagaAdmissionError("resource_policy is invalid")
    return dict(value)


__all__ = [
    "ProductRemoteComputeSagaAdmission",
    "ProductRemoteComputeSagaSubmitter",
    "RemoteComputeSagaAdmissionError",
]
