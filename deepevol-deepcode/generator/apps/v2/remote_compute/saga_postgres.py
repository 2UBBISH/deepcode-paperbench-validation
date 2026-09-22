"""Product PostgreSQL authority for durable Remote Compute operations.

The repository persists intent before a Provider call and fences every worker
write with a generation, holder, token digest, and live database lease.  It
does not import an Agent repository or a Provider client: those are separate
protocol boundaries and cannot participate in this database transaction.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from hashlib import sha256
from typing import Any
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from apps.v2.database import PRODUCT_DATABASE, activate_runtime_role, require_business_schema

from .saga_models import (
    RemoteComputeEffectPhase,
    RemoteComputeFailureCertainty,
    RemoteComputeOperation,
    RemoteComputeOperationCreate,
    RemoteComputeOperationKind,
    RemoteComputeOperationLease,
    RemoteComputeOperationStatus,
    RemoteComputeOperationStatusView,
    operation_json_sha256,
)


ConnectionFactory = Callable[[], Connection[Any]]


class RemoteComputeSagaError(RuntimeError):
    """Base error for the Product-owned Remote Compute operation Saga."""


class RemoteComputeOperationNotFound(RemoteComputeSagaError):
    """The tenant-scoped operation does not exist."""


class RemoteComputeOperationConflict(RemoteComputeSagaError):
    """An operation identity or idempotency key was reused for another request."""


class RemoteComputeOperationStaleLease(RemoteComputeSagaError):
    """The caller no longer owns the live operation generation."""


class RemoteComputeOperationInvalidTransition(RemoteComputeSagaError):
    """The requested transition is unsafe for the durable effect phase."""


_OPERATION_COLUMNS = """
operation_id, resource_tid, owner_uid, resource_id, binding_id, rid,
       operation_kind, compensates_operation_id, provider, idempotency_key,
       request_jsonb, request_sha256, provider_operation_id, status,
       effect_phase, attempt_count, max_attempts, next_attempt_at,
       lease_generation, lease_owner, heartbeat_at, lease_expires_at,
       provider_request_ref, provider_status, result_jsonb, result_sha256,
       error_code, error_summary, created_at, updated_at, completed_at
"""

_SELECT_OPERATION = (
    f"SELECT {_OPERATION_COLUMNS} FROM ops.remote_compute_operations"
)

_SELECT_STATUS = """
SELECT operation_id, resource_tid, owner_uid, operation_kind, provider,
       provider_operation_id, resource_id, binding_id, rid,
       compensates_operation_id, status, effect_phase, attempt_count,
       max_attempts, next_attempt_at, provider_request_ref, provider_status,
       error_code, created_at, updated_at, completed_at
  FROM ops.remote_compute_operation_status_v
"""

_LIVE_LEASE = """
operation_id = %(operation_id)s
AND resource_tid = %(resource_tid)s
AND owner_uid = %(owner_uid)s
AND status = 'CLAIMED'
AND lease_generation = %(lease_generation)s
AND lease_owner = %(lease_owner)s
AND lease_token_hash = %(lease_token_hash)s
AND lease_expires_at > pg_catalog.clock_timestamp()
"""

class PsycopgRemoteComputeSagaRepository:
    """Short-transaction operation Saga repository in the Product database."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        runtime_role: str = "product_ops_runtime",
        lease_ttl: timedelta = timedelta(minutes=2),
        recovery_batch_size: int = 100,
        enforce_release_gate: bool = True,
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("Remote Compute Saga connection factory is required")
        if lease_ttl <= timedelta(seconds=5):
            raise ValueError("lease_ttl must exceed five seconds")
        if not 1 <= recovery_batch_size <= 1000:
            raise ValueError("recovery_batch_size must be between 1 and 1000")
        self._factory = connection_factory
        self.runtime_role = runtime_role
        self.lease_ttl = lease_ttl
        self.recovery_batch_size = recovery_batch_size
        self.enforce_release_gate = enforce_release_gate

    def probe(self) -> str:
        with self._transaction() as connection:
            return require_business_schema(connection, PRODUCT_DATABASE)

    def create_or_get(
        self,
        request: RemoteComputeOperationCreate,
    ) -> tuple[RemoteComputeOperation, bool]:
        """Persist an operation intent or return an identical durable replay."""

        return self._create_or_get(request, admission=None)

    def create_or_get_admitted(
        self,
        request: RemoteComputeOperationCreate,
        *,
        admission: Callable[
            [Connection[Any], RemoteComputeOperationCreate],
            None,
        ],
    ) -> tuple[RemoteComputeOperation, bool]:
        """Persist only if Product Billing admission succeeds in this transaction.

        The callback receives the already role-activated Product connection.
        Its writes and the operation intent commit or roll back together, so a
        worker can never claim an operation whose price/credit hold is absent.
        Replays invoke the callback again and therefore require an idempotent
        Billing receipt keyed by the stable operation/binding identity.
        """

        if not callable(admission):
            raise TypeError("Remote Compute Saga admission callback is required")
        return self._create_or_get(request, admission=admission)

    def _create_or_get(
        self,
        request: RemoteComputeOperationCreate,
        *,
        admission: Callable[
            [Connection[Any], RemoteComputeOperationCreate],
            None,
        ]
        | None,
    ) -> tuple[RemoteComputeOperation, bool]:

        with self._transaction(
            resource_tid=request.resource_tid,
            owner_uid=request.owner_uid,
        ) as connection:
            if request.compensates_operation_id is not None:
                parent = connection.execute(
                    f"{_SELECT_OPERATION} WHERE operation_id = %(parent_id)s "
                    "AND resource_tid = %(resource_tid)s "
                    "AND owner_uid = %(owner_uid)s FOR UPDATE",
                    {
                        "parent_id": request.compensates_operation_id,
                        "resource_tid": request.resource_tid,
                        "owner_uid": request.owner_uid,
                    },
                ).fetchone()
                if parent is None:
                    raise RemoteComputeOperationNotFound(
                        "compensated Remote Compute operation does not exist"
                    )
                if str(parent["status"]) not in {
                    RemoteComputeOperationStatus.PROVIDER_SUCCEEDED.value,
                    RemoteComputeOperationStatus.RECONCILIATION_REQUIRED.value,
                    RemoteComputeOperationStatus.COMPENSATION_PENDING.value,
                }:
                    raise RemoteComputeOperationInvalidTransition(
                        "operation is not eligible for compensation"
                    )

            row = connection.execute(
                f"""
                INSERT INTO ops.remote_compute_operations (
                    operation_id, resource_tid, owner_uid, resource_id,
                    binding_id, rid, operation_kind,
                    compensates_operation_id, provider, idempotency_key,
                    request_jsonb, request_sha256, provider_operation_id,
                    status, effect_phase, max_attempts, next_attempt_at
                ) VALUES (
                    %(operation_id)s, %(resource_tid)s, %(owner_uid)s,
                    %(resource_id)s, %(binding_id)s, %(rid)s,
                    %(operation_kind)s, %(compensates_operation_id)s,
                    %(provider)s, %(idempotency_key)s, %(request_jsonb)s,
                    %(request_sha256)s, %(provider_operation_id)s,
                    'PENDING', 'INTENT_PERSISTED', %(max_attempts)s,
                    COALESCE(%(next_attempt_at)s, pg_catalog.clock_timestamp())
                )
                ON CONFLICT DO NOTHING
                RETURNING {_OPERATION_COLUMNS}
                """,
                _create_params(request),
            ).fetchone()
            replayed = row is None
            if row is None:
                # INSERT .. ON CONFLICT has already waited for any concurrent
                # creator.  Coordinator admission intentionally has no UPDATE
                # privilege, so its replay read must not request a row lock;
                # the idempotent Billing receipt serializes the callback's own
                # effect.  Worker-owned creation and compensation retain the
                # lock used by their state transition below.
                replay_lock = "" if admission is not None else " FOR UPDATE"
                row = connection.execute(
                    f"{_SELECT_OPERATION} WHERE "
                    "operation_id = %(operation_id)s OR "
                    "(resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s "
                    " AND idempotency_key = %(idempotency_key)s) OR "
                    "(provider = %(provider)s "
                    " AND provider_operation_id = %(provider_operation_id)s) OR "
                    "(resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s "
                    " AND compensates_operation_id IS NOT DISTINCT FROM "
                    "     %(compensates_operation_id)s "
                    " AND %(compensates_operation_id)s IS NOT NULL)"
                    f"{replay_lock}",
                    _create_params(request),
                ).fetchone()
            if row is None:
                raise RemoteComputeSagaError("operation insert was not observable")
            _assert_same_request(row, request)

            if admission is not None:
                admission(connection, request)

            if request.compensates_operation_id is not None:
                updated = connection.execute(
                    """
                    UPDATE ops.remote_compute_operations
                       SET status = 'COMPENSATION_PENDING',
                           effect_phase = 'COMPENSATION_REQUIRED',
                           updated_at = pg_catalog.clock_timestamp()
                     WHERE operation_id = %(parent_id)s
                       AND resource_tid = %(resource_tid)s
                       AND owner_uid = %(owner_uid)s
                       AND status IN (
                           'PROVIDER_SUCCEEDED',
                           'RECONCILIATION_REQUIRED',
                           'COMPENSATION_PENDING'
                       )
                    """,
                    {
                        "parent_id": request.compensates_operation_id,
                        "resource_tid": request.resource_tid,
                        "owner_uid": request.owner_uid,
                    },
                )
                if updated.rowcount != 1:
                    raise RemoteComputeOperationInvalidTransition(
                        "compensation parent changed concurrently"
                    )
            return _operation(row), replayed

    def claim_next(self, *, lease_owner: str) -> RemoteComputeOperationLease | None:
        _safe_text(lease_owner, "lease_owner", 256)
        raw_token = secrets.token_bytes(32)
        with self._transaction() as connection:
            claimed = connection.execute(
                "SELECT claimed_operation_id, claimed_resource_tid, claimed_owner_uid "
                "FROM ops.claim_next_remote_compute_operation("
                "%(lease_owner)s, %(lease_token_hash)s, %(lease_seconds)s::bigint, "
                "%(recovery_batch_size)s::bigint)",
                {
                    "lease_owner": lease_owner,
                    "lease_token_hash": sha256(raw_token).digest(),
                    "lease_seconds": int(self.lease_ttl.total_seconds()),
                    "recovery_batch_size": self.recovery_batch_size,
                },
            ).fetchone()
            if claimed is None:
                return None
            connection.execute(
                "SELECT set_config('app.resource_tid', %s, true), "
                "set_config('app.uid', %s, true)",
                (
                    str(claimed["claimed_resource_tid"]),
                    str(claimed["claimed_owner_uid"]),
                ),
            )
            row = connection.execute(
                f"{_SELECT_OPERATION} WHERE operation_id = %(operation_id)s "
                "AND resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s",
                {
                    "operation_id": claimed["claimed_operation_id"],
                    "resource_tid": claimed["claimed_resource_tid"],
                    "owner_uid": claimed["claimed_owner_uid"],
                },
            ).fetchone()
            if row is None:
                raise RemoteComputeSagaError("claimed operation was not observable")
        return RemoteComputeOperationLease(operation=_operation(row), token=raw_token)

    def heartbeat(self, lease: RemoteComputeOperationLease) -> RemoteComputeOperation:
        return self._lease_update(
            lease,
            """
            UPDATE ops.remote_compute_operations
               SET heartbeat_at = pg_catalog.clock_timestamp(),
                   lease_expires_at = pg_catalog.clock_timestamp()
                                      + (%(lease_seconds)s * INTERVAL '1 second'),
                   updated_at = pg_catalog.clock_timestamp()
             WHERE {fence}
            RETURNING *
            """,
            {"lease_seconds": int(self.lease_ttl.total_seconds())},
        )

    def require_live(self, lease: RemoteComputeOperationLease) -> None:
        with self._transaction(
            resource_tid=lease.operation.resource_tid,
            owner_uid=lease.operation.owner_uid,
        ) as connection:
            row = connection.execute(
                f"SELECT operation_id FROM ops.remote_compute_operations WHERE {_LIVE_LEASE}",
                _lease_params(lease),
            ).fetchone()
        if row is None:
            raise RemoteComputeOperationStaleLease("Remote Compute operation lease is stale")

    def mark_effect_started(
        self,
        lease: RemoteComputeOperationLease,
    ) -> RemoteComputeOperation:
        """Commit the ambiguity boundary immediately before the Provider call."""

        return self._lease_update(
            lease,
            """
            UPDATE ops.remote_compute_operations
               SET effect_phase = 'EFFECT_MAY_HAVE_OCCURRED',
                   updated_at = pg_catalog.clock_timestamp()
             WHERE {fence}
               AND effect_phase = 'INTENT_PERSISTED'
            RETURNING *
            """,
        )

    def mark_provider_pending(
        self,
        lease: RemoteComputeOperationLease,
        *,
        provider_request_ref: str,
        provider_status: str,
        retry_after: timedelta,
    ) -> RemoteComputeOperation:
        _safe_text(provider_request_ref, "provider_request_ref", 512)
        _safe_text(provider_status, "provider_status", 128)
        if retry_after < timedelta(seconds=1):
            raise ValueError("Provider poll retry_after must be at least one second")
        return self._lease_update(
            lease,
            """
            UPDATE ops.remote_compute_operations
               SET status = 'PROVIDER_PENDING',
                   effect_phase = 'PROVIDER_ACCEPTED',
                   provider_request_ref = %(provider_request_ref)s,
                   provider_status = %(provider_status)s,
                   next_attempt_at = pg_catalog.clock_timestamp()
                                     + (%(retry_after_seconds)s * INTERVAL '1 second'),
                   lease_generation = lease_generation + 1,
                   lease_owner = NULL, lease_token_hash = NULL,
                   heartbeat_at = NULL, lease_expires_at = NULL,
                   updated_at = pg_catalog.clock_timestamp()
             WHERE {fence}
               AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
            RETURNING *
            """,
            {
                "provider_request_ref": provider_request_ref,
                "provider_status": provider_status,
                "retry_after_seconds": int(retry_after.total_seconds()),
            },
        )

    def record_provider_success(
        self,
        lease: RemoteComputeOperationLease,
        *,
        provider_request_ref: str,
        provider_status: str,
        result: Mapping[str, Any],
    ) -> RemoteComputeOperation:
        _safe_text(provider_request_ref, "provider_request_ref", 512)
        _safe_text(provider_status, "provider_status", 128)
        result_hash = operation_json_sha256(result)
        phase = (
            RemoteComputeEffectPhase.COMPENSATION_RECEIPT_PERSISTED.value
            if lease.operation.operation_kind is RemoteComputeOperationKind.COMPENSATE
            else RemoteComputeEffectPhase.RECEIPT_PERSISTED.value
        )
        return self._lease_update(
            lease,
            """
            UPDATE ops.remote_compute_operations
               SET status = 'PROVIDER_SUCCEEDED', effect_phase = %(effect_phase)s,
                   provider_request_ref = %(provider_request_ref)s,
                   provider_status = %(provider_status)s,
                   result_jsonb = %(result_jsonb)s,
                   result_sha256 = %(result_sha256)s,
                   next_attempt_at = pg_catalog.clock_timestamp(),
                   lease_generation = lease_generation + 1,
                   lease_owner = NULL, lease_token_hash = NULL,
                   heartbeat_at = NULL, lease_expires_at = NULL,
                   error_code = NULL, error_summary = NULL,
                   updated_at = pg_catalog.clock_timestamp()
             WHERE {fence}
               AND effect_phase IN ('EFFECT_MAY_HAVE_OCCURRED', 'PROVIDER_ACCEPTED')
            RETURNING *
            """,
            {
                "effect_phase": phase,
                "provider_request_ref": provider_request_ref,
                "provider_status": provider_status,
                "result_jsonb": Jsonb(dict(result)),
                "result_sha256": result_hash,
            },
        )

    def record_provider_rejection(
        self,
        lease: RemoteComputeOperationLease,
        *,
        provider_request_ref: str,
        provider_status: str,
        result: Mapping[str, Any],
        error_code: str,
        error_summary: str,
    ) -> RemoteComputeOperation:
        """Persist a Provider's conclusive no-effect receipt as terminal evidence.

        This path is deliberately separate from pre-egress failures. Once the
        send boundary was crossed, a caller may only avoid reconciliation when
        the Provider returned durable evidence that no effect occurred.
        """

        _safe_text(provider_request_ref, "provider_request_ref", 512)
        _safe_text(provider_status, "provider_status", 128)
        _safe_text(error_code, "error_code", 128)
        _safe_text(error_summary, "error_summary", 2048)
        result_hash = operation_json_sha256(result)
        return self._lease_update(
            lease,
            """
            UPDATE ops.remote_compute_operations
               SET status = 'FAILED',
                   effect_phase = 'NO_EFFECT_RECEIPT_PERSISTED',
                   provider_request_ref = %(provider_request_ref)s,
                   provider_status = %(provider_status)s,
                   result_jsonb = %(result_jsonb)s,
                   result_sha256 = %(result_sha256)s,
                   lease_generation = lease_generation + 1,
                   lease_owner = NULL, lease_token_hash = NULL,
                   heartbeat_at = NULL, lease_expires_at = NULL,
                   error_code = %(error_code)s,
                   error_summary = %(error_summary)s,
                   completed_at = pg_catalog.clock_timestamp(),
                   updated_at = pg_catalog.clock_timestamp()
             WHERE {fence}
               AND effect_phase IN ('EFFECT_MAY_HAVE_OCCURRED', 'PROVIDER_ACCEPTED')
            RETURNING *
            """,
            {
                "provider_request_ref": provider_request_ref,
                "provider_status": provider_status,
                "result_jsonb": Jsonb(dict(result)),
                "result_sha256": result_hash,
                "error_code": error_code,
                "error_summary": error_summary,
            },
        )
    def record_failure(
        self,
        lease: RemoteComputeOperationLease,
        *,
        certainty: RemoteComputeFailureCertainty,
        retryable: bool,
        error_code: str,
        error_summary: str,
        retry_after: timedelta = timedelta(0),
    ) -> RemoteComputeOperation:
        _safe_text(error_code, "error_code", 128)
        _safe_text(error_summary, "error_summary", 2048)
        if retry_after < timedelta(0):
            raise ValueError("retry_after cannot be negative")
        if certainty is RemoteComputeFailureCertainty.CONFIRMED_NO_EFFECT:
            status_sql = (
                "CASE WHEN %(retryable)s AND attempt_count < max_attempts "
                "THEN 'PENDING' ELSE 'FAILED' END"
            )
            completed_sql = (
                "CASE WHEN %(retryable)s AND attempt_count < max_attempts "
                "THEN NULL ELSE pg_catalog.clock_timestamp() END"
            )
            required_phase = "AND effect_phase = 'INTENT_PERSISTED'"
        else:
            status_sql = "'RECONCILIATION_REQUIRED'"
            completed_sql = "NULL"
            required_phase = (
                "AND effect_phase IN ('EFFECT_MAY_HAVE_OCCURRED', 'PROVIDER_ACCEPTED')"
            )
        return self._lease_update(
            lease,
            f"""
            UPDATE ops.remote_compute_operations
               SET status = {status_sql},
                   effect_phase = CASE
                     WHEN %(certainty)s = 'EFFECT_MAY_HAVE_OCCURRED'
                       THEN 'EFFECT_MAY_HAVE_OCCURRED'
                     ELSE effect_phase
                   END,
                   next_attempt_at = pg_catalog.clock_timestamp()
                                     + (%(retry_after_seconds)s * INTERVAL '1 second'),
                   lease_generation = lease_generation + 1,
                   lease_owner = NULL, lease_token_hash = NULL,
                   heartbeat_at = NULL, lease_expires_at = NULL,
                   error_code = %(error_code)s,
                   error_summary = %(error_summary)s,
                   completed_at = {completed_sql},
                   updated_at = pg_catalog.clock_timestamp()
             WHERE {{fence}}
               {required_phase}
            RETURNING *
            """,
            {
                "retryable": retryable,
                "certainty": certainty.value,
                "retry_after_seconds": int(retry_after.total_seconds()),
                "error_code": error_code,
                "error_summary": error_summary,
            },
        )

    def complete(self, lease: RemoteComputeOperationLease) -> RemoteComputeOperation:
        """Finalize Product-local state only after a durable Provider receipt."""

        with self._transaction(
            resource_tid=lease.operation.resource_tid,
            owner_uid=lease.operation.owner_uid,
        ) as connection:
            row = connection.execute(
                f"""
                UPDATE ops.remote_compute_operations
                   SET status = 'SUCCEEDED', completed_at = pg_catalog.clock_timestamp(),
                       lease_generation = lease_generation + 1,
                       lease_owner = NULL, lease_token_hash = NULL,
                       heartbeat_at = NULL, lease_expires_at = NULL,
                       error_code = NULL, error_summary = NULL,
                       updated_at = pg_catalog.clock_timestamp()
                 WHERE {_LIVE_LEASE}
                   AND effect_phase IN (
                       'RECEIPT_PERSISTED', 'COMPENSATION_RECEIPT_PERSISTED'
                   )
                RETURNING *
                """,
                _lease_params(lease),
            ).fetchone()
            if row is None:
                raise RemoteComputeOperationStaleLease(
                    "completion lost its operation generation or receipt fence"
                )
            operation = _operation(row)
            if operation.operation_kind is RemoteComputeOperationKind.COMPENSATE:
                parent = connection.execute(
                    """
                    UPDATE ops.remote_compute_operations
                       SET status = 'COMPENSATED',
                           effect_phase = 'COMPENSATION_RECEIPT_PERSISTED',
                           completed_at = pg_catalog.clock_timestamp(),
                           updated_at = pg_catalog.clock_timestamp()
                     WHERE operation_id = %(parent_id)s
                       AND resource_tid = %(resource_tid)s
                       AND owner_uid = %(owner_uid)s
                       AND status = 'COMPENSATION_PENDING'
                    RETURNING operation_id
                    """,
                    {
                        "parent_id": operation.compensates_operation_id,
                        "resource_tid": operation.resource_tid,
                        "owner_uid": operation.owner_uid,
                    },
                ).fetchone()
                if parent is None:
                    raise RemoteComputeOperationInvalidTransition(
                        "compensation completion lost its parent transition"
                    )
            return operation

    def cancel(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
    ) -> RemoteComputeOperation:
        """Cancel only an unclaimed intent; accepted effects require compensation."""

        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            row = connection.execute(
                """
                UPDATE ops.remote_compute_operations
                   SET status = 'CANCELLED',
                       completed_at = pg_catalog.clock_timestamp(),
                       updated_at = pg_catalog.clock_timestamp()
                 WHERE operation_id = %(operation_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND status = 'PENDING'
                   AND effect_phase = 'INTENT_PERSISTED'
                RETURNING *
                """,
                {
                    "operation_id": operation_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).fetchone()
            if row is None:
                existing = connection.execute(
                    f"{_SELECT_OPERATION} WHERE operation_id = %(operation_id)s "
                    "AND resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s",
                    {
                        "operation_id": operation_id,
                        "resource_tid": resource_tid,
                        "owner_uid": owner_uid,
                    },
                ).fetchone()
                if existing is None:
                    raise RemoteComputeOperationNotFound(
                        "Remote Compute operation does not exist"
                    )
                raise RemoteComputeOperationInvalidTransition(
                    "operation can no longer be cancelled without compensation"
                )
        return _operation(row)

    def get(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
    ) -> RemoteComputeOperation | None:
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            row = connection.execute(
                f"{_SELECT_OPERATION} WHERE operation_id = %(operation_id)s "
                "AND resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s",
                {
                    "operation_id": operation_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).fetchone()
        return None if row is None else _operation(row)

    def find_run_provision_operation(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        rid: UUID,
        resource_id: UUID,
    ) -> RemoteComputeOperation | None:
        """The succeeded PROVISION that created ``resource_id`` for this run.

        Only a machine this run provisioned may be released by the run's
        Agent; an ACTIVATE of a pre-existing pool resource returns ``None`` so
        the caller refuses rather than deleting a user's standing server.
        """

        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            row = connection.execute(
                f"{_SELECT_OPERATION} WHERE resource_tid = %(resource_tid)s "
                "AND owner_uid = %(owner_uid)s AND rid = %(rid)s "
                "AND resource_id = %(resource_id)s "
                "AND operation_kind = 'PROVISION' AND status = 'SUCCEEDED' "
                "ORDER BY created_at DESC, operation_id DESC LIMIT 1",
                {
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "rid": rid,
                    "resource_id": resource_id,
                },
            ).fetchone()
        return None if row is None else _operation(row)

    def get_status(
        self,
        *,
        operation_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
    ) -> RemoteComputeOperationStatusView | None:
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            row = connection.execute(
                f"{_SELECT_STATUS} WHERE operation_id = %(operation_id)s "
                "AND resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s",
                {
                    "operation_id": operation_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).fetchone()
        return None if row is None else _status_view(row)

    def list_statuses(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        statuses: Sequence[RemoteComputeOperationStatus] = (),
        limit: int = 100,
    ) -> tuple[RemoteComputeOperationStatusView, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        status_values = [status.value for status in statuses] or None
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            rows = connection.execute(
                f"{_SELECT_STATUS} WHERE resource_tid = %(resource_tid)s "
                "AND owner_uid = %(owner_uid)s "
                "AND (%(statuses)s::text[] IS NULL OR status = ANY(%(statuses)s::text[])) "
                "ORDER BY created_at DESC, operation_id DESC LIMIT %(limit)s",
                {
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "statuses": status_values,
                    "limit": limit,
                },
            ).fetchall()
        return tuple(_status_view(row) for row in rows)

    def _lease_update(
        self,
        lease: RemoteComputeOperationLease,
        statement: str,
        extra: Mapping[str, Any] | None = None,
    ) -> RemoteComputeOperation:
        params = _lease_params(lease)
        if extra:
            params.update(extra)
        with self._transaction(
            resource_tid=lease.operation.resource_tid,
            owner_uid=lease.operation.owner_uid,
        ) as connection:
            row = connection.execute(statement.format(fence=_LIVE_LEASE), params).fetchone()
        if row is None:
            raise RemoteComputeOperationStaleLease(
                "Remote Compute operation lease or effect phase is stale"
            )
        return _operation(row)

    @contextmanager
    def _transaction(
        self,
        *,
        resource_tid: UUID | None = None,
        owner_uid: UUID | None = None,
    ) -> Iterator[Connection[Any]]:
        if (resource_tid is None) != (owner_uid is None):
            raise ValueError("resource_tid and owner_uid must be supplied together")
        with self._factory() as connection, connection.transaction():
            activate_runtime_role(connection, expected_role=self.runtime_role)
            if self.enforce_release_gate:
                require_business_schema(connection, PRODUCT_DATABASE)
            if resource_tid is not None and owner_uid is not None:
                connection.execute(
                    "SELECT set_config('app.resource_tid', %s, true), "
                    "set_config('app.uid', %s, true)",
                    (str(resource_tid), str(owner_uid)),
                )
            connection.row_factory = dict_row
            yield connection


def _create_params(request: RemoteComputeOperationCreate) -> dict[str, Any]:
    return {
        "operation_id": request.operation_id,
        "resource_tid": request.resource_tid,
        "owner_uid": request.owner_uid,
        "resource_id": request.resource_id,
        "binding_id": request.binding_id,
        "rid": request.rid,
        "operation_kind": request.operation_kind.value,
        "compensates_operation_id": request.compensates_operation_id,
        "provider": request.provider,
        "idempotency_key": request.idempotency_key,
        "request_jsonb": Jsonb(dict(request.request)),
        "request_sha256": request.request_sha256,
        "provider_operation_id": request.provider_operation_id,
        "max_attempts": request.max_attempts,
        "next_attempt_at": request.next_attempt_at,
    }


def _assert_same_request(
    row: Mapping[str, Any],
    request: RemoteComputeOperationCreate,
) -> None:
    expected = {
        "operation_id": request.operation_id,
        "resource_tid": request.resource_tid,
        "owner_uid": request.owner_uid,
        "resource_id": request.resource_id,
        "binding_id": request.binding_id,
        "rid": request.rid,
        "operation_kind": request.operation_kind.value,
        "compensates_operation_id": request.compensates_operation_id,
        "provider": request.provider,
        "idempotency_key": request.idempotency_key,
        "request_sha256": request.request_sha256,
        "provider_operation_id": request.provider_operation_id,
        "max_attempts": request.max_attempts,
    }
    for name, value in expected.items():
        stored = row[name]
        if name == "request_sha256":
            stored = bytes(stored)
        if stored != value:
            raise RemoteComputeOperationConflict(
                "Remote Compute operation idempotency replay changed its request"
            )


def _lease_params(lease: RemoteComputeOperationLease) -> dict[str, Any]:
    return {
        "operation_id": lease.operation.operation_id,
        "resource_tid": lease.operation.resource_tid,
        "owner_uid": lease.operation.owner_uid,
        "lease_generation": lease.generation,
        "lease_owner": lease.lease_owner,
        "lease_token_hash": sha256(lease.token).digest(),
    }


def _operation(row: Mapping[str, Any]) -> RemoteComputeOperation:
    result = row["result_jsonb"]
    return RemoteComputeOperation(
        operation_id=row["operation_id"],
        resource_tid=row["resource_tid"],
        owner_uid=row["owner_uid"],
        operation_kind=RemoteComputeOperationKind(str(row["operation_kind"])),
        provider=str(row["provider"]),
        idempotency_key=str(row["idempotency_key"]),
        request=dict(row["request_jsonb"]),
        request_sha256=bytes(row["request_sha256"]),
        provider_operation_id=str(row["provider_operation_id"]),
        resource_id=row["resource_id"],
        binding_id=row["binding_id"],
        rid=row["rid"],
        compensates_operation_id=row["compensates_operation_id"],
        status=RemoteComputeOperationStatus(str(row["status"])),
        effect_phase=RemoteComputeEffectPhase(str(row["effect_phase"])),
        attempt_count=int(row["attempt_count"]),
        max_attempts=int(row["max_attempts"]),
        next_attempt_at=row["next_attempt_at"],
        lease_generation=int(row["lease_generation"]),
        lease_owner=None if row["lease_owner"] is None else str(row["lease_owner"]),
        heartbeat_at=row["heartbeat_at"],
        lease_expires_at=row["lease_expires_at"],
        provider_request_ref=(
            None if row["provider_request_ref"] is None else str(row["provider_request_ref"])
        ),
        provider_status=None if row["provider_status"] is None else str(row["provider_status"]),
        result=None if result is None else dict(result),
        result_sha256=(
            None if row["result_sha256"] is None else bytes(row["result_sha256"])
        ),
        error_code=None if row["error_code"] is None else str(row["error_code"]),
        error_summary=None if row["error_summary"] is None else str(row["error_summary"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _status_view(row: Mapping[str, Any]) -> RemoteComputeOperationStatusView:
    return RemoteComputeOperationStatusView(
        operation_id=row["operation_id"],
        resource_tid=row["resource_tid"],
        owner_uid=row["owner_uid"],
        operation_kind=RemoteComputeOperationKind(str(row["operation_kind"])),
        provider=str(row["provider"]),
        provider_operation_id=str(row["provider_operation_id"]),
        resource_id=row["resource_id"],
        binding_id=row["binding_id"],
        rid=row["rid"],
        compensates_operation_id=row["compensates_operation_id"],
        status=RemoteComputeOperationStatus(str(row["status"])),
        effect_phase=RemoteComputeEffectPhase(str(row["effect_phase"])),
        attempt_count=int(row["attempt_count"]),
        max_attempts=int(row["max_attempts"]),
        next_attempt_at=row["next_attempt_at"],
        provider_request_ref=(
            None if row["provider_request_ref"] is None else str(row["provider_request_ref"])
        ),
        provider_status=None if row["provider_status"] is None else str(row["provider_status"]),
        error_code=None if row["error_code"] is None else str(row["error_code"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _safe_text(value: str, name: str, maximum: int) -> None:
    if not value or "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must contain 1 to {maximum} safe UTF-8 bytes")


__all__ = [
    "PsycopgRemoteComputeSagaRepository",
    "RemoteComputeOperationConflict",
    "RemoteComputeOperationInvalidTransition",
    "RemoteComputeOperationNotFound",
    "RemoteComputeOperationStaleLease",
    "RemoteComputeSagaError",
]
