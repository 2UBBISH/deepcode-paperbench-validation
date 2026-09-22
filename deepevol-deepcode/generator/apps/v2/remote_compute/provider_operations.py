"""Durable Agent authority for Remote Compute Provider effects and transfers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from secrets import token_bytes
from typing import Any, Iterator
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from apps.common.v2_ids import format_typed_id, uuid7
from apps.v2.database import AGENT_DATABASE, activate_runtime_role, require_business_schema

from .models import RemoteComputeCommand
from .saga_models import RemoteComputeOperationKind, stable_provider_operation_id


class ProviderOperationStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    RESULT_UNKNOWN = "RESULT_UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    RECONCILING = "RECONCILING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    COMPENSATION_PENDING = "COMPENSATION_PENDING"
    COMPENSATING = "COMPENSATING"
    COMPENSATED = "COMPENSATED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class ProviderResolutionState(StrEnum):
    APPLIED = "APPLIED"
    NOT_APPLIED = "NOT_APPLIED"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"


class TransferItemStatus(StrEnum):
    DISCOVER = "DISCOVER"
    COPY_PARTIAL = "COPY_PARTIAL"
    VERIFY = "VERIFY"
    ATOMIC_RENAME = "ATOMIC_RENAME"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class ProviderOperationError(RuntimeError):
    code = "PROVIDER_OPERATION_ERROR"


class ProviderOperationConflict(ProviderOperationError):
    code = "PROVIDER_OPERATION_IDEMPOTENCY_CONFLICT"


class ProviderOperationNotFound(ProviderOperationError):
    code = "PROVIDER_OPERATION_NOT_FOUND"


class ProviderOperationStaleLease(ProviderOperationError):
    code = "PROVIDER_OPERATION_STALE_LEASE"


class ProviderOperationInvalidTransition(ProviderOperationError):
    code = "PROVIDER_OPERATION_INVALID_TRANSITION"


@dataclass(frozen=True, slots=True)
class ProviderResolution:
    state: ProviderResolutionState
    evidence: Mapping[str, Any]
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ProviderResolutionState):
            raise TypeError("state must be a ProviderResolutionState")
        _json_sha256(self.evidence)
        if self.error_code is not None:
            _bounded(self.error_code, "error_code", 256)


@dataclass(frozen=True, slots=True)
class ProviderOperationRecord:
    operation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    rid: UUID | None
    command_event_id: UUID
    resource_id: UUID
    target_resource_id: UUID | None
    action: str
    provider: str
    external_resource_id: str
    idempotency_key: str
    request: Mapping[str, Any]
    request_sha256: str
    status: ProviderOperationStatus
    effect_phase: str
    attempt_count: int
    reconciliation_attempts: int
    compensation_attempts: int
    next_attempt_at: datetime
    claim_kind: str | None
    claim_holder: str | None
    claim_generation: int
    claim_token_hash: str | None
    claim_expires_at: datetime | None
    heartbeat_at: datetime | None
    provider_receipt: Mapping[str, Any] | None
    provider_receipt_sha256: str | None
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    retention_until: datetime


@dataclass(frozen=True, slots=True)
class ProviderOperationLease:
    operation: ProviderOperationRecord
    kind: str
    holder: str
    generation: int
    token_hash: str
    expires_at: datetime
    token: bytes = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ProviderOperationStepRecord:
    step_id: UUID
    operation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    step_key: str
    step_kind: str
    step_seq: int
    claim_generation: int
    payload: Mapping[str, Any]
    payload_sha256: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TransferItemCreate:
    transfer_id: UUID
    operation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    item_key: str
    source_ref: str
    target_ref: str
    size_bytes: int
    part_size_bytes: int


@dataclass(frozen=True, slots=True)
class TransferItemRecord:
    transfer_id: UUID
    operation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    item_key: str
    source_ref: str
    target_ref: str
    size_bytes: int
    part_size_bytes: int
    status: TransferItemStatus
    offset_bytes: int
    source_sha256: str | None
    final_sha256: str | None
    claim_holder: str | None
    claim_generation: int
    claim_token_hash: str | None
    claim_expires_at: datetime | None
    heartbeat_at: datetime | None
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    source_deleted_at: datetime | None


@dataclass(frozen=True, slots=True)
class TransferItemLease:
    item: TransferItemRecord
    holder: str
    generation: int
    token_hash: str
    expires_at: datetime
    token: bytes = field(repr=False, compare=False)


class PsycopgProviderOperationRepository:
    """Short-transaction repository with generation/token fencing on every mutation."""

    def __init__(
        self,
        connection_factory: Callable[[], Connection[Any]],
        *,
        runtime_role: str = "agent_ops_dispatcher",
        enforce_release_gate: bool = True,
        token_factory: Callable[[], bytes] = lambda: token_bytes(32),
    ) -> None:
        if not callable(connection_factory) or not callable(token_factory):
            raise TypeError("connection and token factories must be callable")
        self.connection_factory = connection_factory
        self.runtime_role = runtime_role
        self.enforce_release_gate = enforce_release_gate
        self.token_factory = token_factory

    def verify_schema(self) -> None:
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT n.nspname || '.' || c.relname AS relation_name
                FROM pg_catalog.pg_class AS c
                JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                WHERE c.relkind = 'r' AND n.nspname = 'agent_ops'
                  AND c.relname = ANY(%s)
                """,
                (["provider_operations", "provider_operation_steps", "transfer_items"],),
            ).fetchall()
        actual = {str(row["relation_name"]) for row in rows}
        expected = {
            "agent_ops.provider_operations",
            "agent_ops.provider_operation_steps",
            "agent_ops.transfer_items",
        }
        if actual != expected:
            raise ProviderOperationError(f"Provider operation schema is incomplete: {sorted(expected - actual)}")

    def create_or_get(
        self,
        command: RemoteComputeCommand,
        *,
        command_event_id: UUID,
        request: Mapping[str, Any],
        request_sha256: str,
    ) -> ProviderOperationRecord:
        _json_sha256(request)
        _digest(request_sha256, "request_sha256")
        idempotency_key = command.idempotency_key.strip()
        _bounded(idempotency_key, "idempotency_key", 512)
        params = {
            "operation_id": command.command_id,
            "resource_tid": command.resource_tid,
            "owner_uid": command.owner_uid,
            "rid": command.rid,
            "command_event_id": command_event_id,
            "resource_id": command.resource_id,
            "target_resource_id": command.target_resource_id,
            "action": command.action.value,
            "provider": command.provider,
            "external_resource_id": command.external_resource_id,
            "idempotency_key": idempotency_key,
            "request_jsonb": Jsonb(dict(request)),
            "request_sha256": request_sha256,
        }
        with self._transaction(command.resource_tid, command.owner_uid) as connection:
            connection.execute(
                """
                INSERT INTO agent_ops.provider_operations (
                  operation_id, resource_tid, owner_uid, rid, command_event_id,
                  resource_id, target_resource_id, action, provider,
                  external_resource_id, idempotency_key, request_jsonb, request_sha256
                ) VALUES (
                  %(operation_id)s, %(resource_tid)s, %(owner_uid)s, %(rid)s,
                  %(command_event_id)s, %(resource_id)s, %(target_resource_id)s,
                  %(action)s, %(provider)s, %(external_resource_id)s,
                  %(idempotency_key)s, %(request_jsonb)s, %(request_sha256)s
                ) ON CONFLICT DO NOTHING
                """,
                params,
            )
            row = connection.execute(
                """
                SELECT * FROM agent_ops.provider_operations
                WHERE resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s
                  AND (operation_id = %(operation_id)s OR idempotency_key = %(idempotency_key)s)
                ORDER BY (operation_id = %(operation_id)s) DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        if row is None:
            raise ProviderOperationConflict("Provider operation identity collided")
        operation = _operation_from_row(row)
        if (
            operation.operation_id != command.command_id
            or operation.command_event_id != command_event_id
            or operation.idempotency_key != idempotency_key
            or operation.request_sha256 != request_sha256
        ):
            raise ProviderOperationConflict("Provider operation replay changed immutable input")
        return operation

    def prepare_saga_operation(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
        rid: UUID | None,
        resource_id: UUID,
        operation_kind: RemoteComputeOperationKind,
        provider: str,
        provider_operation_id: str,
        request: Mapping[str, Any],
        request_sha256: str,
    ) -> tuple[ProviderOperationRecord, bool]:
        """Persist a Product Saga intent without allowing Provider egress."""

        if not isinstance(operation_kind, RemoteComputeOperationKind):
            raise TypeError("operation_kind must be a RemoteComputeOperationKind")
        expected_provider_operation_id = stable_provider_operation_id(
            operation_id,
            operation_kind,
        )
        if provider_operation_id != expected_provider_operation_id:
            raise ProviderOperationConflict(
                "Provider operation identity is not derived from Product operation"
            )
        _bounded(provider, "provider", 128)
        _bounded(provider_operation_id, "provider_operation_id", 256)
        actual_sha256 = _json_sha256(request)
        _digest(request_sha256, "request_sha256")
        if actual_sha256 != request_sha256:
            raise ProviderOperationConflict("Product Saga request digest does not match")
        external_resource_id = request.get("external_resource_id", "")
        if not isinstance(external_resource_id, str) or len(
            external_resource_id.encode("utf-8")
        ) > 512:
            raise ValueError("external_resource_id is invalid")
        params = {
            "operation_id": operation_id,
            "resource_tid": resource_tid,
            "owner_uid": owner_uid,
            "rid": rid,
            # A Product Saga operation is also its immutable command envelope.
            "command_event_id": operation_id,
            "resource_id": resource_id,
            "target_resource_id": None,
            "action": operation_kind.value,
            "provider": provider,
            "external_resource_id": external_resource_id,
            "idempotency_key": provider_operation_id,
            "request_jsonb": Jsonb(dict(request)),
            "request_sha256": request_sha256,
        }
        with self._transaction(resource_tid, owner_uid) as connection:
            row = connection.execute(
                """
                INSERT INTO agent_ops.provider_operations (
                  operation_id, resource_tid, owner_uid, rid, command_event_id,
                  resource_id, target_resource_id, action, provider,
                  external_resource_id, idempotency_key, request_jsonb, request_sha256
                ) VALUES (
                  %(operation_id)s, %(resource_tid)s, %(owner_uid)s, %(rid)s,
                  %(command_event_id)s, %(resource_id)s, %(target_resource_id)s,
                  %(action)s, %(provider)s, %(external_resource_id)s,
                  %(idempotency_key)s, %(request_jsonb)s, %(request_sha256)s
                ) ON CONFLICT DO NOTHING
                RETURNING *
                """,
                params,
            ).fetchone()
            replayed = row is None
            if row is None:
                row = connection.execute(
                    """
                    SELECT * FROM agent_ops.provider_operations
                    WHERE resource_tid = %(resource_tid)s
                      AND owner_uid = %(owner_uid)s
                      AND (operation_id = %(operation_id)s
                           OR idempotency_key = %(idempotency_key)s)
                    ORDER BY (operation_id = %(operation_id)s) DESC
                    LIMIT 1
                    """,
                    params,
                ).fetchone()
        if row is None:
            raise ProviderOperationConflict("Product Saga intent collided")
        operation = _operation_from_row(row)
        expected = (
            operation_id,
            resource_tid,
            owner_uid,
            rid,
            operation_id,
            resource_id,
            operation_kind.value,
            provider,
            external_resource_id,
            provider_operation_id,
            dict(request),
            request_sha256,
        )
        actual = (
            operation.operation_id,
            operation.resource_tid,
            operation.owner_uid,
            operation.rid,
            operation.command_event_id,
            operation.resource_id,
            operation.action,
            operation.provider,
            operation.external_resource_id,
            operation.idempotency_key,
            dict(operation.request),
            operation.request_sha256,
        )
        if actual != expected:
            raise ProviderOperationConflict(
                "Product Saga replay changed immutable Provider intent"
            )
        return operation, replayed

    def get(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
    ) -> ProviderOperationRecord | None:
        with self._transaction(resource_tid, owner_uid) as connection:
            row = connection.execute(
                """SELECT * FROM agent_ops.provider_operations
                WHERE resource_tid = %s AND owner_uid = %s AND operation_id = %s""",
                (resource_tid, owner_uid, operation_id),
            ).fetchone()
        return None if row is None else _operation_from_row(row)

    def claim_apply(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
        holder: str,
        lease_ttl: timedelta = timedelta(seconds=120),
    ) -> ProviderOperationLease | None:
        return self._claim(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            operation_id=operation_id,
            holder=holder,
            lease_ttl=lease_ttl,
            kind="APPLY",
        )

    def heartbeat(
        self,
        lease: ProviderOperationLease,
        *,
        lease_ttl: timedelta = timedelta(seconds=120),
    ) -> ProviderOperationLease:
        seconds = _lease_seconds(lease_ttl)
        with self._transaction(lease.operation.resource_tid, lease.operation.owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET heartbeat_at = now(), claim_expires_at = now() + (%(seconds)s * interval '1 second'),
                    updated_at = now()
                WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
                  AND owner_uid = %(owner_uid)s AND claim_kind = %(kind)s
                  AND claim_holder = %(holder)s AND claim_generation = %(generation)s
                  AND claim_token_hash = %(token_hash)s AND claim_expires_at > now()
                  AND status IN ('PROCESSING','RECONCILING','COMPENSATING')
                RETURNING *
                """,
                {**_operation_lease_params(lease), "seconds": seconds},
            ).fetchone()
        if row is None:
            raise ProviderOperationStaleLease("Provider operation heartbeat lost authority")
        operation = _operation_from_row(row)
        assert operation.claim_expires_at is not None
        return ProviderOperationLease(
            operation,
            lease.kind,
            lease.holder,
            lease.generation,
            lease.token_hash,
            operation.claim_expires_at,
            lease.token,
        )

    def persist_apply_intent(
        self,
        lease: ProviderOperationLease,
        intent: Mapping[str, Any],
    ) -> ProviderOperationStepRecord:
        return self._step_transition(
            lease,
            step_key="APPLY:INTENT",
            step_kind="APPLY_INTENT",
            payload=intent,
            required_status="PROCESSING",
            required_phase="PRE_EFFECT",
            set_phase="INTENT_PERSISTED",
        )[0]

    def mark_effect_started(self, lease: ProviderOperationLease) -> ProviderOperationRecord:
        with self._transaction(lease.operation.resource_tid, lease.operation.owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET effect_phase = 'EFFECT_MAY_HAVE_OCCURRED', updated_at = now()
                WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
                  AND owner_uid = %(owner_uid)s AND status = 'PROCESSING'
                  AND effect_phase = 'INTENT_PERSISTED' AND claim_kind = 'APPLY'
                  AND claim_holder = %(holder)s AND claim_generation = %(generation)s
                  AND claim_token_hash = %(token_hash)s AND claim_expires_at > now()
                RETURNING *
                """,
                _operation_lease_params(lease),
            ).fetchone()
        if row is None:
            raise ProviderOperationStaleLease("Provider effect exposure lost authority")
        return _operation_from_row(row)

    def complete_apply(
        self,
        lease: ProviderOperationLease,
        receipt: Mapping[str, Any],
    ) -> ProviderOperationRecord:
        _step, operation = self._step_transition(
            lease,
            step_key="APPLY:RECEIPT",
            step_kind="APPLY_RECEIPT",
            payload=receipt,
            required_status="PROCESSING",
            required_phase="EFFECT_MAY_HAVE_OCCURRED",
            set_phase="RECEIPT_PERSISTED",
            terminal_status="SUCCEEDED",
        )
        return operation

    def complete_apply_failure(
        self,
        lease: ProviderOperationLease,
        *,
        error_code: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> ProviderOperationRecord:
        _bounded(error_code, "error_code", 256)
        receipt = {
            "error_code": error_code,
            "evidence": dict(evidence or {}),
        }
        _step, operation = self._step_transition(
            lease,
            step_key="APPLY:RECEIPT",
            step_kind="APPLY_RECEIPT",
            payload=receipt,
            required_status="PROCESSING",
            required_phase="EFFECT_MAY_HAVE_OCCURRED",
            set_phase="RECEIPT_PERSISTED",
            terminal_status="FAILED",
            last_error_code=error_code,
        )
        return operation

    def mark_result_unknown(
        self,
        lease: ProviderOperationLease,
        error_code: str,
    ) -> ProviderOperationRecord:
        _bounded(error_code, "error_code", 256)
        with self._transaction(lease.operation.resource_tid, lease.operation.owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET status = 'RESULT_UNKNOWN', last_error_code = %(error_code)s,
                    next_attempt_at = now(),
                    claim_kind = NULL, claim_holder = NULL, claim_token_hash = NULL,
                    claim_expires_at = NULL, heartbeat_at = NULL, updated_at = now()
                WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
                  AND owner_uid = %(owner_uid)s AND status = 'PROCESSING'
                  AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED' AND claim_kind = 'APPLY'
                  AND claim_holder = %(holder)s AND claim_generation = %(generation)s
                  AND claim_token_hash = %(token_hash)s
                RETURNING *
                """,
                {**_operation_lease_params(lease), "error_code": error_code},
            ).fetchone()
        if row is None:
            raise ProviderOperationStaleLease("Provider unknown transition lost authority")
        return _operation_from_row(row)

    def claim_next_reconciliation(
        self,
        *,
        holder: str,
        lease_ttl: timedelta = timedelta(seconds=120),
        max_attempts: int = 8,
    ) -> ProviderOperationLease | None:
        return self._claim_next_recovery(
            holder=holder,
            lease_ttl=lease_ttl,
            max_attempts=max_attempts,
            kind="RECONCILE",
        )

    def claim_saga_reconciliation(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
        holder: str,
        lease_ttl: timedelta = timedelta(seconds=120),
        max_attempts: int = 8,
    ) -> ProviderOperationLease | None:
        """Claim one Product Saga unknown without entering the legacy command path."""

        _bounded(holder, "holder", 256)
        if not 1 <= max_attempts <= 100:
            raise ValueError("recovery max_attempts is invalid")
        seconds = _lease_seconds(lease_ttl)
        token = self._token()
        token_hash = hashlib.sha256(token).hexdigest()
        params = {
            "resource_tid": resource_tid,
            "owner_uid": owner_uid,
            "operation_id": operation_id,
            "holder": holder,
            "token_hash": token_hash,
            "seconds": seconds,
            "max_attempts": max_attempts,
        }
        saga_actions = "('PROVISION','ACTIVATE','FINISH','COMPENSATE')"
        with self._transaction(resource_tid, owner_uid) as connection:
            connection.execute(
                f"""
                UPDATE agent_ops.provider_operations
                   SET status = CASE
                         WHEN effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                           THEN 'RESULT_UNKNOWN'
                         ELSE 'PENDING'
                       END,
                       claim_kind = NULL, claim_holder = NULL,
                       claim_token_hash = NULL, claim_expires_at = NULL,
                       heartbeat_at = NULL, next_attempt_at = now(), updated_at = now()
                 WHERE operation_id = %(operation_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND action IN {saga_actions}
                   AND status = 'PROCESSING'
                   AND claim_expires_at <= now()
                """,
                params,
            )
            connection.execute(
                f"""
                UPDATE agent_ops.provider_operations
                   SET status = 'AMBIGUOUS', claim_kind = NULL,
                       claim_holder = NULL, claim_token_hash = NULL,
                       claim_expires_at = NULL, heartbeat_at = NULL,
                       next_attempt_at = now(), updated_at = now()
                 WHERE operation_id = %(operation_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND action IN {saga_actions}
                   AND status = 'RECONCILING'
                   AND claim_expires_at <= now()
                """,
                params,
            )
            connection.execute(
                f"""
                UPDATE agent_ops.provider_operations
                   SET status = 'MANUAL_REVIEW',
                       last_error_code = 'REMOTE_RECOVERY_BUDGET_EXHAUSTED',
                       updated_at = now()
                 WHERE operation_id = %(operation_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND action IN {saga_actions}
                   AND status IN ('RESULT_UNKNOWN','AMBIGUOUS')
                   AND reconciliation_attempts >= %(max_attempts)s
                """,
                params,
            )
            row = connection.execute(
                f"""
                UPDATE agent_ops.provider_operations
                   SET status = 'RECONCILING', claim_kind = 'RECONCILE',
                       claim_holder = %(holder)s,
                       claim_generation = claim_generation + 1,
                       claim_token_hash = %(token_hash)s, heartbeat_at = now(),
                       claim_expires_at = now()
                         + (%(seconds)s * interval '1 second'),
                       reconciliation_attempts = reconciliation_attempts + 1,
                       updated_at = now()
                 WHERE operation_id = %(operation_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND action IN {saga_actions}
                   AND status IN ('RESULT_UNKNOWN','AMBIGUOUS')
                   AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                   AND next_attempt_at <= now()
                   AND reconciliation_attempts < %(max_attempts)s
                RETURNING *
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        operation = _operation_from_row(row)
        assert operation.claim_expires_at is not None
        return ProviderOperationLease(
            operation,
            "RECONCILE",
            holder,
            operation.claim_generation,
            token_hash,
            operation.claim_expires_at,
            token,
        )

    def complete_saga_reconciliation(
        self,
        lease: ProviderOperationLease,
        receipt: Mapping[str, Any],
    ) -> ProviderOperationRecord:
        """Persist query-only Saga adoption without requiring a command inbox."""

        if lease.operation.action not in {
            "PROVISION",
            "ACTIVATE",
            "FINISH",
            "COMPENSATE",
        }:
            raise ProviderOperationInvalidTransition(
                "only Product Saga operations use Saga reconciliation"
            )
        return self._step_transition(
            lease,
            step_key="SAGA_RECONCILIATION:RECEIPT",
            step_kind="RECONCILIATION_EVIDENCE",
            payload=receipt,
            required_status="RECONCILING",
            required_phase="EFFECT_MAY_HAVE_OCCURRED",
            set_phase="RECEIPT_PERSISTED",
            terminal_status="SUCCEEDED",
        )[1]

    def resolve_reconciliation(
        self,
        lease: ProviderOperationLease,
        resolution: ProviderResolution,
    ) -> ProviderOperationRecord:
        if resolution.state in {
            ProviderResolutionState.APPLIED,
            ProviderResolutionState.NOT_APPLIED,
            ProviderResolutionState.FAILED,
        }:
            return self._resolve_terminal_reconciliation(lease, resolution)

        payload = {
            "error_code": resolution.error_code,
            "evidence": dict(resolution.evidence),
            "resolution": resolution.state.value,
        }
        target = "MANUAL_REVIEW" if lease.operation.reconciliation_attempts >= 7 else "AMBIGUOUS"
        self._write_step_only(
            lease,
            step_key=f"RECONCILIATION:{lease.generation}",
            step_kind="RECONCILIATION_EVIDENCE",
            payload=payload,
        )
        with self._transaction(lease.operation.resource_tid, lease.operation.owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET status = %(target)s, last_error_code = %(error_code)s,
                    next_attempt_at = now() + interval '30 seconds',
                    claim_kind = NULL, claim_holder = NULL, claim_token_hash = NULL,
                    claim_expires_at = NULL, heartbeat_at = NULL, updated_at = now(),
                    completed_at = CASE WHEN %(target)s = 'FAILED' THEN now() ELSE NULL END
                WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
                  AND owner_uid = %(owner_uid)s AND status = 'RECONCILING'
                  AND claim_kind = 'RECONCILE' AND claim_holder = %(holder)s
                  AND claim_generation = %(generation)s AND claim_token_hash = %(token_hash)s
                RETURNING *
                """,
                {
                    **_operation_lease_params(lease),
                    "target": target,
                    "error_code": resolution.error_code or "REMOTE_PROVIDER_RESULT_UNKNOWN",
                },
            ).fetchone()
        if row is None:
            raise ProviderOperationStaleLease("Provider reconciliation lost authority")
        return _operation_from_row(row)

    def _resolve_terminal_reconciliation(
        self,
        lease: ProviderOperationLease,
        resolution: ProviderResolution,
    ) -> ProviderOperationRecord:
        """Commit Provider evidence and the Product-facing receipt atomically.

        The original command consumer deliberately parks its delivery inbox row
        in ``AMBIGUOUS`` after an uncertain egress.  A terminal Provider
        reconciliation is therefore not complete until that outer inbox is
        terminal and its result outbox event exists in the *same* transaction.
        This prevents a crash after Provider adoption from stranding Product in
        a permanent pending state, while never replaying the Provider effect.
        """

        succeeded = resolution.state is ProviderResolutionState.APPLIED
        terminal_status = "SUCCEEDED" if succeeded else "FAILED"
        result_status = "APPLIED" if succeeded else "FAILED"
        error_code = None
        if not succeeded:
            error_code = resolution.error_code or (
                "REMOTE_PROVIDER_EFFECT_NOT_APPLIED"
                if resolution.state is ProviderResolutionState.NOT_APPLIED
                else "REMOTE_PROVIDER_REJECTED"
            )
        evidence = dict(resolution.evidence)
        step_payload = {
            "error_code": error_code,
            "evidence": evidence,
            "resolution": resolution.state.value,
        }
        params = {
            **_operation_lease_params(lease),
            "step_id": uuid7(),
            "step_key": f"RECONCILIATION:{lease.generation}",
            "step_kind": "RECONCILIATION_EVIDENCE",
            "payload_jsonb": Jsonb(step_payload),
            "payload_sha256": _json_sha256(step_payload),
            "terminal_status": terminal_status,
            "receipt_jsonb": Jsonb(evidence),
            "receipt_sha256": _json_sha256(evidence),
            "error_code": error_code,
        }
        with self._transaction(
            lease.operation.resource_tid,
            lease.operation.owner_uid,
        ) as connection:
            self._insert_or_get_step(connection, params)
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                   SET status = %(terminal_status)s,
                       effect_phase = 'RECEIPT_PERSISTED',
                       provider_receipt_jsonb = %(receipt_jsonb)s,
                       provider_receipt_sha256 = %(receipt_sha256)s,
                       last_error_code = %(error_code)s,
                       claim_kind = NULL, claim_holder = NULL,
                       claim_token_hash = NULL, claim_expires_at = NULL,
                       heartbeat_at = NULL, completed_at = now(), updated_at = now()
                 WHERE operation_id = %(operation_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND status = 'RECONCILING'
                   AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                   AND claim_kind = 'RECONCILE'
                   AND claim_holder = %(holder)s
                   AND claim_generation = %(generation)s
                   AND claim_token_hash = %(token_hash)s
                   AND claim_expires_at > now()
                RETURNING *
                """,
                params,
            ).fetchone()
            if row is None:
                raise ProviderOperationStaleLease(
                    "Provider terminal reconciliation lost lease authority"
                )
            operation = _operation_from_row(row)
            self._settle_reconciled_delivery(
                connection,
                operation,
                result_status=result_status,
                error_code=error_code,
                effect=evidence,
            )
        return operation

    @staticmethod
    def _settle_reconciled_delivery(
        connection: Connection[Any],
        operation: ProviderOperationRecord,
        *,
        result_status: str,
        error_code: str | None,
        effect: Mapping[str, Any],
    ) -> None:
        inbox = connection.execute(
            """
            SELECT event_id, source_aggregate_id, producer_seq, resource_tid,
                   owner_uid, rid, correlation_id, payload_jsonb, effect_status
              FROM agent_ops.delivery_inbox
             WHERE source_service = 'PRODUCT_API'
               AND destination_service = 'AGENT_EXECUTION'
               AND event_id = %(command_event_id)s
               AND effect_type = 'APPLY_REMOTE_COMPUTE_COMMAND'
             FOR UPDATE
            """,
            {"command_event_id": operation.command_event_id},
        ).fetchone()
        if inbox is None:
            raise ProviderOperationConflict(
                "Provider reconciliation has no authoritative command inbox"
            )
        if str(inbox["effect_status"]) not in {"PROCESSING", "AMBIGUOUS"}:
            raise ProviderOperationConflict(
                "Provider reconciliation command inbox is not recoverable"
            )
        if (
            inbox["source_aggregate_id"] != operation.resource_id
            or inbox["resource_tid"] != operation.resource_tid
            or inbox["owner_uid"] != operation.owner_uid
            or inbox["rid"] != operation.rid
            or inbox["correlation_id"] != operation.operation_id
        ):
            raise ProviderOperationConflict(
                "Provider reconciliation command authority disagrees with operation"
            )

        command_payload = dict(inbox["payload_jsonb"])
        expected_identity = {
            "command_id": format_typed_id("op", operation.operation_id),
            "resource_id": format_typed_id("rcres", operation.resource_id),
            "action": operation.action,
            "rid": (
                None
                if operation.rid is None
                else format_typed_id("rid", operation.rid)
            ),
            "target_resource_id": (
                None
                if operation.target_resource_id is None
                else format_typed_id("rcres", operation.target_resource_id)
            ),
        }
        if any(command_payload.get(key) != value for key, value in expected_identity.items()):
            raise ProviderOperationConflict(
                "Provider reconciliation payload identity disagrees with operation"
            )
        required_result_fields = {
            "command_generation",
            "expected_resource_version",
            "binding_id",
            "lease_generation",
        }
        if not required_result_fields.issubset(command_payload):
            raise ProviderOperationConflict(
                "Provider reconciliation command payload is incomplete"
            )

        result_payload = {
            "command_id": command_payload["command_id"],
            "resource_id": command_payload["resource_id"],
            "action": command_payload["action"],
            "command_generation": command_payload["command_generation"],
            "expected_resource_version": command_payload[
                "expected_resource_version"
            ],
            "rid": command_payload["rid"],
            "binding_id": command_payload["binding_id"],
            "lease_generation": command_payload["lease_generation"],
            "target_resource_id": command_payload["target_resource_id"],
            "status": result_status,
            "error_code": error_code,
            "effect_result": dict(effect),
        }
        result_sha256 = _json_sha256(result_payload)
        result_event_id = uuid7()
        inbox_status = "APPLIED" if result_status == "APPLIED" else "REJECTED_TERMINAL"
        updated = connection.execute(
            """
            UPDATE agent_ops.delivery_inbox
               SET effect_status = %(inbox_status)s,
                   attempts = attempts + 1,
                   processed_at = now(), last_error_code = %(error_code)s,
                   claim_holder = NULL, claim_token_hash = NULL,
                   claim_expires_at = NULL
             WHERE source_service = 'PRODUCT_API'
               AND destination_service = 'AGENT_EXECUTION'
               AND event_id = %(command_event_id)s
               AND effect_type = 'APPLY_REMOTE_COMPUTE_COMMAND'
               AND effect_status IN ('PROCESSING','AMBIGUOUS')
            """,
            {
                "command_event_id": operation.command_event_id,
                "inbox_status": inbox_status,
                "error_code": error_code,
            },
        ).rowcount
        if updated != 1:
            raise ProviderOperationStaleLease(
                "Provider reconciliation lost command inbox authority"
            )
        connection.execute(
            """
            INSERT INTO agent_ops.delivery_outbox (
                event_id, source_service, destination_service,
                source_aggregate_type, source_aggregate_id, producer_seq,
                resource_tid, owner_uid, payer_tid, rid, event_type,
                schema_version, occurred_at, correlation_id,
                causation_event_id, payload_jsonb, payload_sha256, status
            ) VALUES (
                %(event_id)s, 'AGENT_EXECUTION', 'PRODUCT_API',
                'remote_compute_resource', %(resource_id)s, %(producer_seq)s,
                %(resource_tid)s, %(owner_uid)s, NULL, %(rid)s,
                'REMOTE_COMPUTE_COMMAND_RESULT', '2.0.0', now(),
                %(operation_id)s, %(command_event_id)s, %(payload)s,
                %(payload_sha256)s, 'PENDING'
            )
            """,
            {
                "event_id": result_event_id,
                "resource_id": operation.resource_id,
                "producer_seq": int(inbox["producer_seq"]),
                "resource_tid": operation.resource_tid,
                "owner_uid": operation.owner_uid,
                "rid": operation.rid,
                "operation_id": operation.operation_id,
                "command_event_id": operation.command_event_id,
                "payload": Jsonb(result_payload),
                "payload_sha256": result_sha256,
            },
        )

    def request_compensation(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
    ) -> ProviderOperationRecord:
        with self._transaction(resource_tid, owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET status = 'COMPENSATION_PENDING', completed_at = NULL,
                    effect_phase = 'PRE_EFFECT', next_attempt_at = now(), updated_at = now()
                WHERE resource_tid = %s AND owner_uid = %s AND operation_id = %s
                  AND status = 'SUCCEEDED'
                RETURNING *
                """,
                (resource_tid, owner_uid, operation_id),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """SELECT * FROM agent_ops.provider_operations
                    WHERE resource_tid = %s AND owner_uid = %s AND operation_id = %s""",
                    (resource_tid, owner_uid, operation_id),
                ).fetchone()
        if row is None:
            raise ProviderOperationNotFound("Provider operation is absent")
        operation = _operation_from_row(row)
        if operation.status not in {
            ProviderOperationStatus.COMPENSATION_PENDING,
            ProviderOperationStatus.COMPENSATING,
            ProviderOperationStatus.COMPENSATED,
        }:
            raise ProviderOperationInvalidTransition("Provider operation cannot be compensated")
        return operation

    def claim_next_compensation(
        self,
        *,
        holder: str,
        lease_ttl: timedelta = timedelta(seconds=120),
        max_attempts: int = 20,
    ) -> ProviderOperationLease | None:
        return self._claim_next_recovery(
            holder=holder,
            lease_ttl=lease_ttl,
            max_attempts=max_attempts,
            kind="COMPENSATE",
        )

    def persist_compensation_intent(
        self,
        lease: ProviderOperationLease,
        intent: Mapping[str, Any],
    ) -> ProviderOperationStepRecord:
        return self._step_transition(
            lease,
            step_key="COMPENSATION:INTENT",
            step_kind="COMPENSATION_INTENT",
            payload=intent,
            required_status="COMPENSATING",
            required_phase="PRE_EFFECT",
            set_phase="EFFECT_MAY_HAVE_OCCURRED",
        )[0]

    def complete_compensation(
        self,
        lease: ProviderOperationLease,
        receipt: Mapping[str, Any],
    ) -> ProviderOperationRecord:
        return self._step_transition(
            lease,
            step_key="COMPENSATION:RECEIPT",
            step_kind="COMPENSATION_RECEIPT",
            payload=receipt,
            required_status="COMPENSATING",
            required_phase="EFFECT_MAY_HAVE_OCCURRED",
            set_phase="RECEIPT_PERSISTED",
            terminal_status="COMPENSATED",
        )[1]

    def mark_compensation_ambiguous(
        self,
        lease: ProviderOperationLease,
        *,
        error_code: str,
    ) -> ProviderOperationRecord:
        _bounded(error_code, "error_code", 256)
        with self._transaction(lease.operation.resource_tid, lease.operation.owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET status = 'MANUAL_REVIEW', last_error_code = %(error_code)s,
                    claim_kind = NULL, claim_holder = NULL, claim_token_hash = NULL,
                    claim_expires_at = NULL, heartbeat_at = NULL, updated_at = now()
                WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
                  AND owner_uid = %(owner_uid)s AND status = 'COMPENSATING'
                  AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                  AND claim_kind = 'COMPENSATE' AND claim_holder = %(holder)s
                  AND claim_generation = %(generation)s AND claim_token_hash = %(token_hash)s
                RETURNING *
                """,
                {**_operation_lease_params(lease), "error_code": error_code},
            ).fetchone()
        if row is None:
            raise ProviderOperationStaleLease("Provider compensation unknown lost authority")
        return _operation_from_row(row)

    def defer_compensation_retry(
        self,
        lease: ProviderOperationLease,
        *,
        error_code: str,
        next_attempt_at: datetime,
    ) -> ProviderOperationRecord:
        """Release a compensation claim only after a proven pre-egress failure."""

        _bounded(error_code, "error_code", 256)
        if next_attempt_at.tzinfo is None or next_attempt_at.utcoffset() is None:
            raise ValueError("next_attempt_at must be timezone-aware")
        with self._transaction(lease.operation.resource_tid, lease.operation.owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET status = 'COMPENSATION_PENDING', effect_phase = 'PRE_EFFECT',
                    next_attempt_at = %(next_attempt_at)s,
                    last_error_code = %(error_code)s,
                    claim_kind = NULL, claim_holder = NULL, claim_token_hash = NULL,
                    claim_expires_at = NULL, heartbeat_at = NULL, updated_at = now()
                WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
                  AND owner_uid = %(owner_uid)s AND status = 'COMPENSATING'
                  AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                  AND claim_kind = 'COMPENSATE' AND claim_holder = %(holder)s
                  AND claim_generation = %(generation)s AND claim_token_hash = %(token_hash)s
                  AND claim_expires_at > now()
                RETURNING *
                """,
                {
                    **_operation_lease_params(lease),
                    "error_code": error_code,
                    "next_attempt_at": next_attempt_at,
                },
            ).fetchone()
        if row is None:
            raise ProviderOperationStaleLease("Provider compensation retry lost authority")
        return _operation_from_row(row)

    def create_transfer_item(self, command: TransferItemCreate) -> TransferItemRecord:
        _bounded(command.item_key, "item_key", 512)
        _bounded(command.source_ref, "source_ref", 4096)
        _bounded(command.target_ref, "target_ref", 4096)
        if command.size_bytes < 0 or not 1 <= command.part_size_bytes <= 64 * 1024 * 1024:
            raise ValueError("transfer size or part size is invalid")
        params = {
            "transfer_id": command.transfer_id,
            "operation_id": command.operation_id,
            "resource_tid": command.resource_tid,
            "owner_uid": command.owner_uid,
            "item_key": command.item_key,
            "source_ref": command.source_ref,
            "target_ref": command.target_ref,
            "size_bytes": command.size_bytes,
            "part_size_bytes": command.part_size_bytes,
        }
        with self._transaction(command.resource_tid, command.owner_uid) as connection:
            connection.execute(
                """
                INSERT INTO agent_ops.transfer_items (
                  transfer_id, operation_id, resource_tid, owner_uid, item_key,
                  source_ref, target_ref, size_bytes, part_size_bytes
                ) VALUES (
                  %(transfer_id)s, %(operation_id)s, %(resource_tid)s, %(owner_uid)s,
                  %(item_key)s, %(source_ref)s, %(target_ref)s, %(size_bytes)s,
                  %(part_size_bytes)s
                ) ON CONFLICT DO NOTHING
                """,
                params,
            )
            row = connection.execute(
                """
                SELECT * FROM agent_ops.transfer_items
                WHERE resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s
                  AND operation_id = %(operation_id)s AND item_key = %(item_key)s
                """,
                params,
            ).fetchone()
        if row is None:
            raise ProviderOperationConflict("transfer identity collided")
        item = _transfer_from_row(row)
        if (
            item.transfer_id != command.transfer_id
            or item.source_ref != command.source_ref
            or item.target_ref != command.target_ref
            or item.size_bytes != command.size_bytes
            or item.part_size_bytes != command.part_size_bytes
        ):
            raise ProviderOperationConflict("transfer replay changed immutable input")
        return item

    def get_transfer_item(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        transfer_id: UUID,
    ) -> TransferItemRecord | None:
        with self._transaction(resource_tid, owner_uid) as connection:
            row = connection.execute(
                """SELECT * FROM agent_ops.transfer_items
                WHERE resource_tid = %s AND owner_uid = %s AND transfer_id = %s""",
                (resource_tid, owner_uid, transfer_id),
            ).fetchone()
        return None if row is None else _transfer_from_row(row)

    def list_transfer_items(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
    ) -> tuple[TransferItemRecord, ...]:
        with self._transaction(resource_tid, owner_uid) as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_ops.transfer_items
                 WHERE resource_tid = %s AND owner_uid = %s AND operation_id = %s
                 ORDER BY item_key, transfer_id
                """,
                (resource_tid, owner_uid, operation_id),
            ).fetchall()
        return tuple(_transfer_from_row(row) for row in rows)

    def list_steps(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
    ) -> tuple[ProviderOperationStepRecord, ...]:
        with self._transaction(resource_tid, owner_uid) as connection:
            rows = connection.execute(
                """SELECT * FROM agent_ops.provider_operation_steps
                WHERE resource_tid = %s AND owner_uid = %s AND operation_id = %s
                ORDER BY step_seq""",
                (resource_tid, owner_uid, operation_id),
            ).fetchall()
        return tuple(_step_from_row(row) for row in rows)

    def claim_transfer_item(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        transfer_id: UUID,
        holder: str,
        lease_ttl: timedelta = timedelta(seconds=120),
    ) -> TransferItemLease | None:
        _bounded(holder, "holder", 256)
        seconds = _lease_seconds(lease_ttl)
        token = self._token()
        token_hash = hashlib.sha256(token).hexdigest()
        params = {
            "resource_tid": resource_tid,
            "owner_uid": owner_uid,
            "transfer_id": transfer_id,
            "holder": holder,
            "token_hash": token_hash,
            "seconds": seconds,
        }
        with self._transaction(resource_tid, owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.transfer_items
                SET claim_holder = %(holder)s, claim_generation = claim_generation + 1,
                    claim_token_hash = %(token_hash)s, heartbeat_at = now(),
                    claim_expires_at = now() + (%(seconds)s * interval '1 second'),
                    updated_at = now()
                WHERE resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s
                  AND transfer_id = %(transfer_id)s
                  AND status IN ('DISCOVER','COPY_PARTIAL','VERIFY','ATOMIC_RENAME')
                  AND (claim_holder IS NULL OR claim_expires_at <= now())
                RETURNING *
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        item = _transfer_from_row(row)
        assert item.claim_expires_at is not None
        return TransferItemLease(item, holder, item.claim_generation, token_hash, item.claim_expires_at, token)

    def record_copy_progress(
        self,
        lease: TransferItemLease,
        offset_bytes: int,
    ) -> TransferItemRecord:
        if not 0 <= offset_bytes <= lease.item.size_bytes:
            raise ValueError("transfer offset is outside the source size")
        return self._transfer_transition(
            lease,
            """
            status = 'COPY_PARTIAL', offset_bytes = %(offset_bytes)s,
            updated_at = now()
            """,
            extra_predicate="AND status IN ('DISCOVER','COPY_PARTIAL') AND offset_bytes <= %(offset_bytes)s",
            values={"offset_bytes": offset_bytes},
        )

    def heartbeat_transfer(
        self,
        lease: TransferItemLease,
        *,
        lease_ttl: timedelta = timedelta(seconds=120),
    ) -> TransferItemLease:
        seconds = _lease_seconds(lease_ttl)
        with self._transaction(lease.item.resource_tid, lease.item.owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.transfer_items
                SET heartbeat_at = now(),
                    claim_expires_at = now() + (%(seconds)s * interval '1 second'),
                    updated_at = now()
                WHERE transfer_id = %(transfer_id)s AND operation_id = %(operation_id)s
                  AND resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s
                  AND claim_holder = %(holder)s AND claim_generation = %(generation)s
                  AND claim_token_hash = %(token_hash)s AND claim_expires_at > now()
                  AND status IN ('DISCOVER','COPY_PARTIAL','VERIFY','ATOMIC_RENAME')
                RETURNING *
                """,
                {**_transfer_lease_params(lease), "seconds": seconds},
            ).fetchone()
        if row is None:
            raise ProviderOperationStaleLease("transfer heartbeat lost authority")
        item = _transfer_from_row(row)
        assert item.claim_expires_at is not None
        return TransferItemLease(
            item,
            lease.holder,
            lease.generation,
            lease.token_hash,
            item.claim_expires_at,
            lease.token,
        )

    def begin_verify(self, lease: TransferItemLease, source_sha256: str) -> TransferItemRecord:
        _digest(source_sha256, "source_sha256")
        return self._transfer_transition(
            lease,
            "status = 'VERIFY', source_sha256 = %(source_sha256)s, updated_at = now()",
            extra_predicate="AND status = 'COPY_PARTIAL' AND offset_bytes = size_bytes",
            values={"source_sha256": source_sha256},
        )

    def record_verified(self, lease: TransferItemLease, final_sha256: str) -> TransferItemRecord:
        _digest(final_sha256, "final_sha256")
        return self._transfer_transition(
            lease,
            "status = 'ATOMIC_RENAME', final_sha256 = %(final_sha256)s, updated_at = now()",
            extra_predicate="AND status = 'VERIFY' AND source_sha256 = %(final_sha256)s",
            values={"final_sha256": final_sha256},
        )

    def complete_atomic_rename(self, lease: TransferItemLease) -> TransferItemRecord:
        return self._transfer_transition(
            lease,
            """
            status = 'COMPLETED', completed_at = now(), updated_at = now(),
            claim_holder = NULL, claim_token_hash = NULL,
            claim_expires_at = NULL, heartbeat_at = NULL
            """,
            extra_predicate="AND status = 'ATOMIC_RENAME' AND final_sha256 = source_sha256",
        )

    def record_transfer_integrity_failure(
        self,
        lease: TransferItemLease,
        *,
        error_code: str,
    ) -> TransferItemRecord:
        _bounded(error_code, "error_code", 256)
        return self._transfer_transition(
            lease,
            """
            status = CASE WHEN status IN ('VERIFY','ATOMIC_RENAME')
                          THEN 'MANUAL_REVIEW' ELSE 'FAILED' END,
            last_error_code = %(error_code)s, updated_at = now(),
            claim_holder = NULL, claim_token_hash = NULL,
            claim_expires_at = NULL, heartbeat_at = NULL
            """,
            extra_predicate=(
                "AND status IN "
                "('DISCOVER','COPY_PARTIAL','VERIFY','ATOMIC_RENAME')"
            ),
            values={"error_code": error_code},
        )

    def mark_source_deleted(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        transfer_id: UUID,
    ) -> TransferItemRecord:
        with self._transaction(resource_tid, owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.transfer_items
                SET source_deleted_at = coalesce(source_deleted_at, now()), updated_at = now()
                WHERE resource_tid = %s AND owner_uid = %s AND transfer_id = %s
                  AND status = 'COMPLETED' AND final_sha256 = source_sha256
                RETURNING *
                """,
                (resource_tid, owner_uid, transfer_id),
            ).fetchone()
        if row is None:
            raise ProviderOperationInvalidTransition("source cannot be deleted before verified rename")
        return _transfer_from_row(row)

    def _claim(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        operation_id: UUID,
        holder: str,
        lease_ttl: timedelta,
        kind: str,
    ) -> ProviderOperationLease | None:
        _bounded(holder, "holder", 256)
        seconds = _lease_seconds(lease_ttl)
        token = self._token()
        token_hash = hashlib.sha256(token).hexdigest()
        params = {
            "resource_tid": resource_tid,
            "owner_uid": owner_uid,
            "operation_id": operation_id,
            "holder": holder,
            "token_hash": token_hash,
            "seconds": seconds,
        }
        with self._transaction(resource_tid, owner_uid) as connection:
            connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET status = CASE WHEN effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                                  THEN 'RESULT_UNKNOWN' ELSE 'PENDING' END,
                    claim_kind = NULL, claim_holder = NULL, claim_token_hash = NULL,
                    claim_expires_at = NULL, heartbeat_at = NULL, updated_at = now()
                WHERE resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s
                  AND operation_id = %(operation_id)s AND status = 'PROCESSING'
                  AND claim_expires_at <= now()
                """,
                params,
            )
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET status = 'PROCESSING', effect_phase = 'PRE_EFFECT',
                    claim_kind = 'APPLY', claim_holder = %(holder)s,
                    claim_generation = claim_generation + 1,
                    claim_token_hash = %(token_hash)s, heartbeat_at = now(),
                    claim_expires_at = now() + (%(seconds)s * interval '1 second'),
                    attempt_count = attempt_count + 1, updated_at = now()
                WHERE resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s
                  AND operation_id = %(operation_id)s AND status = 'PENDING'
                  AND next_attempt_at <= now()
                RETURNING *
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        operation = _operation_from_row(row)
        assert operation.claim_expires_at is not None
        return ProviderOperationLease(
            operation,
            kind,
            holder,
            operation.claim_generation,
            token_hash,
            operation.claim_expires_at,
            token,
        )

    def _claim_next_recovery(
        self,
        *,
        holder: str,
        lease_ttl: timedelta,
        max_attempts: int,
        kind: str,
    ) -> ProviderOperationLease | None:
        _bounded(holder, "holder", 256)
        if not 1 <= max_attempts <= 100:
            raise ValueError("recovery max_attempts is invalid")
        seconds = _lease_seconds(lease_ttl)
        token = self._token()
        token_hash = hashlib.sha256(token).hexdigest()
        counter = "reconciliation_attempts" if kind == "RECONCILE" else "compensation_attempts"
        source_states = "('RESULT_UNKNOWN','AMBIGUOUS')" if kind == "RECONCILE" else "('COMPENSATION_PENDING')"
        target_status = "RECONCILING" if kind == "RECONCILE" else "COMPENSATING"
        # The generic command reconciler only understands RemoteComputeAction.
        # Product Saga actions (PROVISION/ACTIVATE/FINISH/COMPENSATE) have a
        # separate query-by-stable-operation recovery path and must never be
        # claimed here as an unsupported legacy command.
        action_scope = (
            "action IN "
            "('POWER_ON','POWER_OFF','RELEASE','CLEANUP','MIGRATE','APPLY_QUOTA')"
        )
        params = {
            "holder": holder,
            "token_hash": token_hash,
            "seconds": seconds,
            "max_attempts": max_attempts,
        }
        with self._transaction() as connection:
            if kind == "RECONCILE":
                connection.execute(
                    """
                    UPDATE agent_ops.provider_operations
                    SET status = 'AMBIGUOUS', claim_kind = NULL, claim_holder = NULL,
                        claim_token_hash = NULL, claim_expires_at = NULL,
                        heartbeat_at = NULL, next_attempt_at = now(), updated_at = now()
                    WHERE status = 'RECONCILING' AND claim_expires_at <= now()
                      AND action IN (
                        'POWER_ON','POWER_OFF','RELEASE','CLEANUP','MIGRATE','APPLY_QUOTA'
                      )
                    """
                )
            else:
                connection.execute(
                    """
                    UPDATE agent_ops.provider_operations
                    SET status = 'COMPENSATION_PENDING', claim_kind = NULL,
                        claim_holder = NULL, claim_token_hash = NULL,
                        claim_expires_at = NULL, heartbeat_at = NULL,
                        next_attempt_at = now(), updated_at = now()
                    WHERE status = 'COMPENSATING' AND effect_phase = 'PRE_EFFECT'
                      AND claim_expires_at <= now()
                      AND action IN (
                        'POWER_ON','POWER_OFF','RELEASE','CLEANUP','MIGRATE','APPLY_QUOTA'
                      )
                    """
                )
                connection.execute(
                    """
                    UPDATE agent_ops.provider_operations
                    SET status = 'MANUAL_REVIEW',
                        last_error_code = 'REMOTE_COMPENSATION_RESULT_UNKNOWN',
                        claim_kind = NULL, claim_holder = NULL,
                        claim_token_hash = NULL, claim_expires_at = NULL,
                        heartbeat_at = NULL, updated_at = now()
                    WHERE status = 'COMPENSATING'
                      AND effect_phase <> 'PRE_EFFECT'
                      AND claim_expires_at <= now()
                      AND action IN (
                        'POWER_ON','POWER_OFF','RELEASE','CLEANUP','MIGRATE','APPLY_QUOTA'
                      )
                    """
                )
            connection.execute(
                f"""
                UPDATE agent_ops.provider_operations
                SET status = 'MANUAL_REVIEW', last_error_code = 'REMOTE_RECOVERY_BUDGET_EXHAUSTED',
                    updated_at = now()
                WHERE status IN {source_states} AND {counter} >= %(max_attempts)s
                  AND {action_scope}
                """,
                params,
            )
            row = connection.execute(
                f"""
                WITH candidate AS (
                  SELECT operation_id FROM agent_ops.provider_operations
                  WHERE status IN {source_states} AND next_attempt_at <= now()
                    AND {counter} < %(max_attempts)s
                    AND {action_scope}
                  ORDER BY next_attempt_at, created_at, operation_id
                  FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE agent_ops.provider_operations AS operation
                SET status = '{target_status}', claim_kind = '{kind}',
                    claim_holder = %(holder)s, claim_generation = claim_generation + 1,
                    claim_token_hash = %(token_hash)s, heartbeat_at = now(),
                    claim_expires_at = now() + (%(seconds)s * interval '1 second'),
                    {counter} = {counter} + 1, updated_at = now()
                FROM candidate WHERE operation.operation_id = candidate.operation_id
                RETURNING operation.*
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        operation = _operation_from_row(row)
        assert operation.claim_expires_at is not None
        return ProviderOperationLease(
            operation,
            kind,
            holder,
            operation.claim_generation,
            token_hash,
            operation.claim_expires_at,
            token,
        )

    def _step_transition(
        self,
        lease: ProviderOperationLease,
        *,
        step_key: str,
        step_kind: str,
        payload: Mapping[str, Any],
        required_status: str,
        required_phase: str,
        set_phase: str,
        terminal_status: str | None = None,
        last_error_code: str | None = None,
        receipt_payload: Mapping[str, Any] | None = None,
    ) -> tuple[ProviderOperationStepRecord, ProviderOperationRecord]:
        payload_sha256 = _json_sha256(payload)
        effective_receipt = payload if receipt_payload is None else receipt_payload
        receipt_sha256 = _json_sha256(effective_receipt)
        params = {
            **_operation_lease_params(lease),
            "step_id": uuid7(),
            "step_key": step_key,
            "step_kind": step_kind,
            "payload_jsonb": Jsonb(dict(payload)),
            "payload_sha256": payload_sha256,
            "required_status": required_status,
            "required_phase": required_phase,
            "set_phase": set_phase,
            "terminal_status": terminal_status,
            "last_error_code": last_error_code,
            "receipt_jsonb": Jsonb(dict(effective_receipt)),
            "receipt_sha256": receipt_sha256,
        }
        with self._transaction(lease.operation.resource_tid, lease.operation.owner_uid) as connection:
            step = self._insert_or_get_step(connection, params)
            row = connection.execute(
                """
                UPDATE agent_ops.provider_operations
                SET status = CASE WHEN %(terminal_status)s::text IS NULL THEN status
                                  ELSE %(terminal_status)s::text END,
                    effect_phase = %(set_phase)s,
                    provider_receipt_jsonb = CASE WHEN %(terminal_status)s::text IS NULL
                      THEN provider_receipt_jsonb ELSE %(receipt_jsonb)s END,
                    provider_receipt_sha256 = CASE WHEN %(terminal_status)s::text IS NULL
                      THEN provider_receipt_sha256 ELSE %(receipt_sha256)s END,
                    last_error_code = CASE WHEN %(terminal_status)s::text IS NULL
                      THEN last_error_code ELSE %(last_error_code)s::text END,
                    claim_kind = CASE WHEN %(terminal_status)s::text IS NULL THEN claim_kind ELSE NULL END,
                    claim_holder = CASE WHEN %(terminal_status)s::text IS NULL THEN claim_holder ELSE NULL END,
                    claim_token_hash = CASE WHEN %(terminal_status)s::text IS NULL THEN claim_token_hash ELSE NULL END,
                    claim_expires_at = CASE WHEN %(terminal_status)s::text IS NULL THEN claim_expires_at ELSE NULL END,
                    heartbeat_at = CASE WHEN %(terminal_status)s::text IS NULL THEN heartbeat_at ELSE NULL END,
                    completed_at = CASE WHEN %(terminal_status)s::text IS NULL THEN completed_at ELSE now() END,
                    updated_at = now()
                WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
                  AND owner_uid = %(owner_uid)s AND status = %(required_status)s
                  AND effect_phase = %(required_phase)s AND claim_kind = %(kind)s
                  AND claim_holder = %(holder)s AND claim_generation = %(generation)s
                  AND claim_token_hash = %(token_hash)s AND claim_expires_at > now()
                RETURNING *
                """,
                params,
            ).fetchone()
            if row is None:
                raise ProviderOperationStaleLease("Provider operation step lost lease authority")
        return step, _operation_from_row(row)

    def _write_step_only(
        self,
        lease: ProviderOperationLease,
        *,
        step_key: str,
        step_kind: str,
        payload: Mapping[str, Any],
    ) -> ProviderOperationStepRecord:
        params = {
            **_operation_lease_params(lease),
            "step_id": uuid7(),
            "step_key": step_key,
            "step_kind": step_kind,
            "payload_jsonb": Jsonb(dict(payload)),
            "payload_sha256": _json_sha256(payload),
        }
        with self._transaction(lease.operation.resource_tid, lease.operation.owner_uid) as connection:
            live = connection.execute(
                """
                SELECT 1 FROM agent_ops.provider_operations
                WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
                  AND owner_uid = %(owner_uid)s AND claim_kind = %(kind)s
                  AND claim_holder = %(holder)s AND claim_generation = %(generation)s
                  AND claim_token_hash = %(token_hash)s AND claim_expires_at > now()
                FOR UPDATE
                """,
                params,
            ).fetchone()
            if live is None:
                raise ProviderOperationStaleLease("Provider operation evidence lost lease authority")
            return self._insert_or_get_step(connection, params)

    @staticmethod
    def _insert_or_get_step(connection: Connection[Any], params: Mapping[str, Any]) -> ProviderOperationStepRecord:
        connection.execute(
            """
            INSERT INTO agent_ops.provider_operation_steps (
              step_id, operation_id, resource_tid, owner_uid, step_key, step_kind,
              step_seq, claim_generation, payload_jsonb, payload_sha256
            ) SELECT %(step_id)s, %(operation_id)s, %(resource_tid)s, %(owner_uid)s,
                     %(step_key)s, %(step_kind)s,
                     coalesce(max(step_seq), 0) + 1, %(generation)s,
                     %(payload_jsonb)s, %(payload_sha256)s
              FROM agent_ops.provider_operation_steps
             WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
               AND owner_uid = %(owner_uid)s
            ON CONFLICT DO NOTHING
            """,
            params,
        )
        row = connection.execute(
            """
            SELECT * FROM agent_ops.provider_operation_steps
            WHERE operation_id = %(operation_id)s AND resource_tid = %(resource_tid)s
              AND owner_uid = %(owner_uid)s AND step_key = %(step_key)s
            """,
            params,
        ).fetchone()
        if row is None:
            raise ProviderOperationConflict("Provider operation step identity collided")
        step = _step_from_row(row)
        if step.step_kind != params["step_kind"] or step.payload_sha256 != params["payload_sha256"]:
            raise ProviderOperationConflict("Provider operation step replay changed content")
        return step

    def _transfer_transition(
        self,
        lease: TransferItemLease,
        assignments: str,
        *,
        extra_predicate: str,
        values: Mapping[str, Any] | None = None,
    ) -> TransferItemRecord:
        params = {**_transfer_lease_params(lease), **dict(values or {})}
        with self._transaction(lease.item.resource_tid, lease.item.owner_uid) as connection:
            row = connection.execute(
                f"""
                UPDATE agent_ops.transfer_items SET {assignments}
                WHERE transfer_id = %(transfer_id)s AND operation_id = %(operation_id)s
                  AND resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s
                  AND claim_holder = %(holder)s AND claim_generation = %(generation)s
                  AND claim_token_hash = %(token_hash)s AND claim_expires_at > now()
                  {extra_predicate}
                RETURNING *
                """,
                params,
            ).fetchone()
        if row is None:
            raise ProviderOperationStaleLease("transfer mutation lost lease authority or state precondition")
        return _transfer_from_row(row)

    def _token(self) -> bytes:
        token = self.token_factory()
        if not isinstance(token, bytes) or len(token) < 16:
            raise ValueError("Provider operation token factory must return at least 16 bytes")
        return token

    @contextmanager
    def _transaction(
        self,
        resource_tid: UUID | None = None,
        owner_uid: UUID | None = None,
    ) -> Iterator[Connection[Any]]:
        if (resource_tid is None) != (owner_uid is None):
            raise ValueError("resource_tid and owner_uid must be set together")
        with self.connection_factory() as connection, connection.transaction():
            activate_runtime_role(connection, expected_role=self.runtime_role)
            if self.enforce_release_gate:
                require_business_schema(connection, AGENT_DATABASE)
            connection.row_factory = dict_row
            if resource_tid is not None:
                connection.execute("SELECT set_config('app.resource_tid', %s, true)", (str(resource_tid),))
                connection.execute("SELECT set_config('app.owner_uid', %s, true)", (str(owner_uid),))
            yield connection


def _operation_lease_params(lease: ProviderOperationLease) -> dict[str, Any]:
    return {
        "operation_id": lease.operation.operation_id,
        "resource_tid": lease.operation.resource_tid,
        "owner_uid": lease.operation.owner_uid,
        "kind": lease.kind,
        "holder": lease.holder,
        "generation": lease.generation,
        "token_hash": lease.token_hash,
    }


def _transfer_lease_params(lease: TransferItemLease) -> dict[str, Any]:
    return {
        "transfer_id": lease.item.transfer_id,
        "operation_id": lease.item.operation_id,
        "resource_tid": lease.item.resource_tid,
        "owner_uid": lease.item.owner_uid,
        "holder": lease.holder,
        "generation": lease.generation,
        "token_hash": lease.token_hash,
    }


def _operation_from_row(row: Mapping[str, Any]) -> ProviderOperationRecord:
    receipt = row["provider_receipt_jsonb"]
    return ProviderOperationRecord(
        operation_id=row["operation_id"],
        resource_tid=row["resource_tid"],
        owner_uid=row["owner_uid"],
        rid=row["rid"],
        command_event_id=row["command_event_id"],
        resource_id=row["resource_id"],
        target_resource_id=row["target_resource_id"],
        action=str(row["action"]),
        provider=str(row["provider"]),
        external_resource_id=str(row["external_resource_id"]),
        idempotency_key=str(row["idempotency_key"]),
        request=dict(row["request_jsonb"]),
        request_sha256=str(row["request_sha256"]),
        status=ProviderOperationStatus(row["status"]),
        effect_phase=str(row["effect_phase"]),
        attempt_count=int(row["attempt_count"]),
        reconciliation_attempts=int(row["reconciliation_attempts"]),
        compensation_attempts=int(row["compensation_attempts"]),
        next_attempt_at=row["next_attempt_at"],
        claim_kind=row["claim_kind"],
        claim_holder=row["claim_holder"],
        claim_generation=int(row["claim_generation"]),
        claim_token_hash=row["claim_token_hash"],
        claim_expires_at=row["claim_expires_at"],
        heartbeat_at=row["heartbeat_at"],
        provider_receipt=None if receipt is None else dict(receipt),
        provider_receipt_sha256=row["provider_receipt_sha256"],
        last_error_code=row["last_error_code"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        retention_until=row["retention_until"],
    )


def _step_from_row(row: Mapping[str, Any]) -> ProviderOperationStepRecord:
    return ProviderOperationStepRecord(
        step_id=row["step_id"],
        operation_id=row["operation_id"],
        resource_tid=row["resource_tid"],
        owner_uid=row["owner_uid"],
        step_key=str(row["step_key"]),
        step_kind=str(row["step_kind"]),
        step_seq=int(row["step_seq"]),
        claim_generation=int(row["claim_generation"]),
        payload=dict(row["payload_jsonb"]),
        payload_sha256=str(row["payload_sha256"]),
        created_at=row["created_at"],
    )


def _transfer_from_row(row: Mapping[str, Any]) -> TransferItemRecord:
    return TransferItemRecord(
        transfer_id=row["transfer_id"],
        operation_id=row["operation_id"],
        resource_tid=row["resource_tid"],
        owner_uid=row["owner_uid"],
        item_key=str(row["item_key"]),
        source_ref=str(row["source_ref"]),
        target_ref=str(row["target_ref"]),
        size_bytes=int(row["size_bytes"]),
        part_size_bytes=int(row["part_size_bytes"]),
        status=TransferItemStatus(row["status"]),
        offset_bytes=int(row["offset_bytes"]),
        source_sha256=row["source_sha256"],
        final_sha256=row["final_sha256"],
        claim_holder=row["claim_holder"],
        claim_generation=int(row["claim_generation"]),
        claim_token_hash=row["claim_token_hash"],
        claim_expires_at=row["claim_expires_at"],
        heartbeat_at=row["heartbeat_at"],
        last_error_code=row["last_error_code"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        source_deleted_at=row["source_deleted_at"],
    )


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise ValueError("Provider operation JSON exceeds 64 KiB")
    return hashlib.sha256(encoded).hexdigest()


def _bounded(value: str, name: str, maximum: int) -> None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} is empty or oversized")


def _digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _lease_seconds(value: timedelta) -> float:
    seconds = value.total_seconds()
    if not 1 <= seconds <= 3600:
        raise ValueError("lease TTL must be between 1 and 3600 seconds")
    return seconds


__all__ = [
    "ProviderOperationConflict",
    "ProviderOperationError",
    "ProviderOperationInvalidTransition",
    "ProviderOperationLease",
    "ProviderOperationNotFound",
    "ProviderOperationRecord",
    "ProviderOperationStaleLease",
    "ProviderOperationStatus",
    "ProviderOperationStepRecord",
    "ProviderResolution",
    "ProviderResolutionState",
    "PsycopgProviderOperationRepository",
    "TransferItemCreate",
    "TransferItemLease",
    "TransferItemRecord",
    "TransferItemStatus",
]
