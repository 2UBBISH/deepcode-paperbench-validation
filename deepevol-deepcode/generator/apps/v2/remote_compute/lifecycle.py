"""Run-scoped Remote Compute lifecycle without a parallel billing authority."""

from __future__ import annotations

import logging

import hashlib
import math
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol
from uuid import UUID

from .models import (
    RemoteComputePricingContext,
    RemoteComputeSettlement,
    RemoteWorkspaceBinding,
)
from .postgres import (
    PsycopgRemoteComputeRepository,
    RemoteComputeConflict,
    RemoteComputeNotFound,
)


logger = logging.getLogger(__name__)


class BillingSettlementPort(Protocol):
    """Billing-owned settlement boundary for the unified V2 ledgers."""

    def authorize_remote_compute(
        self,
        connection: Any,
        binding: RemoteWorkspaceBinding,
        pricing: RemoteComputePricingContext,
    ) -> None: ...

    def settle_remote_compute(
        self,
        *,
        binding: RemoteWorkspaceBinding,
        seconds: int,
        ended_at: datetime,
        outcome: str,
        idempotency_key: str,
        external_resource_id: str = "",
    ) -> RemoteComputeSettlement: ...


class RemoteComputeSettlementUnavailable(RuntimeError):
    """Settlement did not commit and the finish operation may be retried."""


class RemoteComputeAuthorizationUnavailable(RuntimeError):
    """The start transaction could not obtain a durable Billing hold."""


class RemoteComputeInsufficientCredits(RuntimeError):
    def __init__(self, *, required_credits: int, available_credits: int) -> None:
        super().__init__("remote compute preflight balance is insufficient")
        self.required_credits = required_credits
        self.available_credits = available_credits


class RemoteComputeRunLifecycle:
    def __init__(
        self,
        repository: PsycopgRemoteComputeRepository,
        billing: BillingSettlementPort | None,
    ) -> None:
        self.repository = repository
        self.billing = billing

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
    ) -> tuple[RemoteWorkspaceBinding, bool]:
        if self.billing is None:
            raise RemoteComputeAuthorizationUnavailable(
                "remote compute Billing authorization adapter is unavailable"
            )
        try:
            return self.repository.start_run_binding(
                resource_tid=resource_tid,
                owner_uid=owner_uid,
                sid=sid,
                rid=rid,
                resource_id=resource_id,
                binding_id=binding_id,
                estimated_credits=estimated_credits,
                resource_policy=resource_policy,
                expires_at=expires_at,
                started_at=started_at or datetime.now(UTC),
                authorize=self.billing.authorize_remote_compute,
            )
        except (RemoteComputeInsufficientCredits, RemoteComputeConflict):
            raise
        except (ValueError, TypeError):
            raise
        except Exception as exc:
            raise RemoteComputeAuthorizationUnavailable(
                "remote compute Billing authorization failed"
            ) from exc

    def finish(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        rid: UUID,
        ended_at: datetime,
        outcome: str,
    ) -> tuple[RemoteWorkspaceBinding, bool]:
        if ended_at.tzinfo is None or ended_at.utcoffset() is None:
            raise ValueError("remote compute end time must be timezone-aware")
        normalized_outcome = outcome.strip().upper()
        if normalized_outcome not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            raise ValueError("remote compute outcome must be a terminal run state")
        binding = self.repository.get_run_binding(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            rid=rid,
        )
        if binding is None:
            # A run that never rented a machine has nothing to settle; the
            # Agent's terminal hook relies on the 404 this maps to.
            raise RemoteComputeNotFound("remote compute run has no binding")
        if binding.charge_id is not None:
            return binding, True
        started_at = binding.started_at or binding.created_at
        if started_at is None or ended_at < started_at:
            raise RemoteComputeConflict("remote compute finish precedes its start")
        seconds = max(1, math.ceil((ended_at - started_at).total_seconds()))
        if self.billing is None:
            raise RemoteComputeSettlementUnavailable(
                "remote compute Billing settlement adapter is unavailable"
            )
        stable = (
            f"remote-compute:{binding.binding_id}:"
            f"lease:{binding.lease_generation}:run:{binding.rid}"
        )
        idempotency_key = "rcsettle_" + hashlib.sha256(stable.encode("ascii")).hexdigest()
        # The Billing command role cannot read ops.remote_compute_resources;
        # the provider instance id (usage evidence) is resolved here, under the
        # coordinator role, and handed over.
        resource = self.repository.get_resource(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            resource_id=binding.resource_id,
        )
        external_resource_id = "" if resource is None else str(resource.external_resource_id or "")
        try:
            settlement = self.billing.settle_remote_compute(
                binding=binding,
                seconds=seconds,
                ended_at=ended_at,
                outcome=normalized_outcome,
                idempotency_key=idempotency_key,
                external_resource_id=external_resource_id,
            )
        except RemoteComputeSettlementUnavailable:
            raise
        except Exception as exc:
            logger.exception("remote compute settlement failed rid=%s", rid)
            raise RemoteComputeSettlementUnavailable(
                "remote compute Billing settlement failed"
            ) from exc
        return self.repository.finalize_run_binding(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            binding_id=binding.binding_id,
            lease_generation=binding.lease_generation,
            ended_at=ended_at,
            actual_seconds=seconds,
            charge_id=settlement.charge_id,
            actual_credits=settlement.charged_credits,
        )


__all__ = [
    "BillingSettlementPort",
    "RemoteComputeAuthorizationUnavailable",
    "RemoteComputeInsufficientCredits",
    "RemoteComputeRunLifecycle",
    "RemoteComputeSettlementUnavailable",
]
