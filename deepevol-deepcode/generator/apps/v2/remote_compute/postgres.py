"""Product PostgreSQL repository for remote-compute authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from apps.common.v2_ids import uuid7
from apps.v2.database import PRODUCT_DATABASE, activate_runtime_role, require_business_schema

from .models import (
    RemoteComputeAdminRecord,
    RemoteComputeAction,
    RemoteComputeCommandReceipt,
    RemoteComputeCommandState,
    RemoteComputePricingContext,
    RemoteComputeResource,
    RemoteComputeResultEvent,
    RemoteWorkspaceBinding,
    RemoteWorkspaceDescriptor,
)
from .provisioning import ProvisionedRemoteCompute


class RemoteComputeRepositoryError(RuntimeError):
    """A remote-compute catalog invariant failed."""


class RemoteComputeConflict(RemoteComputeRepositoryError):
    """A command lost an optimistic resource or run lease fence."""


class RemoteComputeNotFound(RemoteComputeRepositoryError):
    """A tenant-scoped Remote Compute aggregate does not exist."""


class PsycopgRemoteComputeRepository:
    """Read and bind remote resources using Product RLS.

    The coordinator role is used for the command/read boundary because the
    actual public resource-selection command is still owned by the Product
    admission coordinator.  It is granted only the contracted columns.
    """

    def __init__(
        self,
        connection_factory: Callable[[], Connection[Any]],
        *,
        runtime_role: str = "product_coordinator_runtime",
        enforce_release_gate: bool = True,
    ) -> None:
        self.connection_factory = connection_factory
        self.runtime_role = runtime_role
        self.enforce_release_gate = enforce_release_gate

    def resolve_workspace(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        sid: UUID,
        rid: UUID | None = None,
    ) -> RemoteWorkspaceDescriptor | None:
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            row = connection.execute(
                """
                SELECT b.binding_id, b.resource_id, b.resource_tid, b.owner_uid,
                       b.sid, b.rid, b.lease_generation, b.status,
                       b.remote_root, b.secret_ref, b.secret_version,
                       b.expires_at, r.access_host, r.access_port,
                       r.access_username, r.provider
                  FROM ops.remote_compute_run_bindings AS b
                  JOIN ops.remote_compute_resources AS r
                    ON r.resource_id = b.resource_id
                   AND r.resource_tid = b.resource_tid
                   AND r.owner_uid = b.owner_uid
                 WHERE b.resource_tid = %(resource_tid)s
                   AND b.owner_uid = %(owner_uid)s
                   AND b.sid = %(sid)s
                   AND (%(rid)s IS NULL OR b.rid = %(rid)s)
                   AND b.status IN ('ACTIVE', 'RUNNING')
                   AND r.status IN ('ACTIVE', 'IDLE', 'BUSY', 'QUEUED')
                   AND r.deleted_at IS NULL
                   AND (b.expires_at IS NULL OR b.expires_at > clock_timestamp())
                 ORDER BY b.created_at DESC
                 LIMIT 1
                """,
                {"resource_tid": resource_tid, "owner_uid": owner_uid, "sid": sid, "rid": rid},
            ).fetchone()
        if row is None:
            return None
        return RemoteWorkspaceDescriptor(
            binding_id=row["binding_id"],
            resource_id=row["resource_id"],
            resource_tid=row["resource_tid"],
            owner_uid=row["owner_uid"],
            sid=row["sid"],
            rid=row["rid"],
            lease_generation=int(row["lease_generation"]),
            status=str(row["status"]),
            host=str(row["access_host"]),
            port=int(row["access_port"]),
            username=str(row["access_username"]),
            secret_ref=str(row["secret_ref"]),
            secret_version=str(row["secret_version"]),
            remote_root=str(row["remote_root"]),
            provider=str(row["provider"]),
            connection_name=_connection_name(row["resource_id"]),
            expires_at=row["expires_at"],
        )

    def get_resource(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        resource_id: UUID,
    ) -> RemoteComputeResource | None:
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            row = connection.execute(
                """
                SELECT resource_id, resource_tid, owner_uid, name, provider,
                       status, access_host, access_port, access_username,
                       secret_ref, secret_version, remote_root,
                       accelerator_type, accelerator_count, vram_gb,
                       billing_mode, hourly_price_credits, allocation_note,
                       region, instance_type,
                       cpu_cores, memory_gb, storage_gb, external_resource_id,
                       powered_off_at, command_generation, version,
                       created_at, updated_at, deleted_at
                 FROM ops.remote_compute_resources
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND deleted_at IS NULL
                """,
                {
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).fetchone()
        if row is None:
            return None
        return _resource_from_row(row)

    def list_resources(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        include_deleted: bool = False,
    ) -> tuple[RemoteComputeResource, ...]:
        """List the caller's canonical Product resource rows.

        The resource and owner predicates are deliberately mandatory.  This
        method is a public compatibility read, not an administrative scan;
        callers must never be able to turn it into a cross-tenant query.
        """

        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            rows = connection.execute(
                """
                SELECT resource_id, resource_tid, owner_uid, name, provider,
                       status, access_host, access_port, access_username,
                       secret_ref, secret_version, remote_root,
                       accelerator_type, accelerator_count, vram_gb,
                       billing_mode, hourly_price_credits, allocation_note,
                       region, instance_type,
                       cpu_cores, memory_gb, storage_gb, external_resource_id,
                       powered_off_at, command_generation, version,
                       created_at, updated_at, deleted_at
                  FROM ops.remote_compute_resources
                 WHERE resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND (%(include_deleted)s OR deleted_at IS NULL)
                 ORDER BY created_at DESC, resource_id DESC
                """,
                {
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "include_deleted": include_deleted,
                },
            ).fetchall()
        return tuple(_resource_from_row(row) for row in rows)

    def upsert_provisioned_resource(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        provisioned: ProvisionedRemoteCompute,
        updated_at: datetime | None = None,
    ) -> tuple[RemoteComputeResource, bool]:
        """Commit one Agent-provisioned resource without changing its authority.

        The Product-selected ``resource_id`` is the idempotency key.  A replay
        may refresh connection metadata for the same provider object, but it
        can never retarget that ID, claim another owner's provider object, or
        overwrite an in-use resource.
        """

        now = updated_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("remote compute update time must be timezone-aware")
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            existing = connection.execute(
                """
                SELECT resource_id, resource_tid, owner_uid, provider,
                       external_resource_id, status
                  FROM ops.remote_compute_resources
                 WHERE resource_id = %(resource_id)s
                 FOR UPDATE
                """,
                {"resource_id": provisioned.resource_id},
            ).fetchone()
            if existing is not None:
                if (
                    existing["resource_tid"] != resource_tid
                    or existing["owner_uid"] != owner_uid
                    or str(existing["provider"]) != provisioned.provider
                    or str(existing["external_resource_id"])
                    != provisioned.external_resource_id
                ):
                    raise RemoteComputeConflict(
                        "remote compute resource id belongs to another provider object"
                    )
                if str(existing["status"]) not in {"ACTIVE", "IDLE", "STOPPED", "ERROR"}:
                    raise RemoteComputeConflict(
                        "remote compute resource cannot be refreshed while in use"
                    )
            collision = connection.execute(
                """
                SELECT resource_id
                  FROM ops.remote_compute_resources
                 WHERE resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND provider = %(provider)s
                   AND external_resource_id = %(external_resource_id)s
                   AND resource_id <> %(resource_id)s
                   AND deleted_at IS NULL
                 LIMIT 1
                 FOR UPDATE
                """,
                {
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "provider": provisioned.provider,
                    "external_resource_id": provisioned.external_resource_id,
                    "resource_id": provisioned.resource_id,
                },
            ).fetchone()
            if collision is not None:
                raise RemoteComputeConflict(
                    "remote compute provider object is already registered"
                )
            values = {
                "resource_id": provisioned.resource_id,
                "resource_tid": resource_tid,
                "owner_uid": owner_uid,
                "name": provisioned.name,
                "provider": provisioned.provider,
                "status": provisioned.status,
                "access_host": provisioned.access_host,
                "access_port": provisioned.access_port,
                "access_username": provisioned.access_username,
                "secret_ref": provisioned.secret_ref,
                "secret_version": provisioned.secret_version,
                "remote_root": provisioned.remote_root,
                "accelerator_type": provisioned.accelerator_type,
                "accelerator_count": provisioned.accelerator_count,
                "vram_gb": provisioned.vram_gb,
                "billing_mode": provisioned.billing_mode,
                "hourly_price_credits": provisioned.hourly_price_credits,
                "region": provisioned.region,
                "instance_type": provisioned.instance_type,
                "cpu_cores": provisioned.cpu_cores,
                "memory_gb": provisioned.memory_gb,
                "storage_gb": provisioned.storage_gb,
                "external_resource_id": provisioned.external_resource_id,
                "updated_at": now,
            }
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO ops.remote_compute_resources (
                        resource_id, resource_tid, owner_uid, name, provider,
                        status, access_host, access_port, access_username,
                        secret_ref, secret_version, remote_root,
                        accelerator_type, accelerator_count, vram_gb,
                        billing_mode, hourly_price_credits, region,
                        instance_type, cpu_cores, memory_gb, storage_gb,
                        external_resource_id, created_at, updated_at
                    ) VALUES (
                        %(resource_id)s, %(resource_tid)s, %(owner_uid)s,
                        %(name)s, %(provider)s, %(status)s, %(access_host)s,
                        %(access_port)s, %(access_username)s, %(secret_ref)s,
                        %(secret_version)s, %(remote_root)s,
                        %(accelerator_type)s, %(accelerator_count)s, %(vram_gb)s,
                        %(billing_mode)s, %(hourly_price_credits)s, %(region)s,
                        %(instance_type)s, %(cpu_cores)s, %(memory_gb)s,
                        %(storage_gb)s, %(external_resource_id)s,
                        %(updated_at)s, %(updated_at)s
                    )
                    """,
                    values,
                )
                replayed = False
            else:
                updated = connection.execute(
                    """
                    UPDATE ops.remote_compute_resources
                       SET name = %(name)s, status = %(status)s,
                           access_host = %(access_host)s,
                           access_port = %(access_port)s,
                           access_username = %(access_username)s,
                           secret_ref = %(secret_ref)s,
                           secret_version = %(secret_version)s,
                           remote_root = %(remote_root)s,
                           accelerator_type = %(accelerator_type)s,
                           accelerator_count = %(accelerator_count)s,
                           vram_gb = %(vram_gb)s,
                           billing_mode = %(billing_mode)s,
                           hourly_price_credits = %(hourly_price_credits)s,
                           region = %(region)s,
                           instance_type = %(instance_type)s,
                           cpu_cores = %(cpu_cores)s,
                           memory_gb = %(memory_gb)s,
                           storage_gb = %(storage_gb)s,
                           powered_off_at = NULL,
                           version = version + 1,
                           updated_at = %(updated_at)s,
                           deleted_at = NULL
                     WHERE resource_id = %(resource_id)s
                       AND resource_tid = %(resource_tid)s
                       AND owner_uid = %(owner_uid)s
                       AND provider = %(provider)s
                       AND external_resource_id = %(external_resource_id)s
                       AND status IN ('ACTIVE','IDLE','STOPPED','ERROR')
                    """,
                    values,
                ).rowcount
                if updated != 1:
                    raise RemoteComputeConflict(
                        "remote compute resource refresh lost its fence"
                    )
                replayed = True
            row = connection.execute(
                """
                SELECT resource_id, resource_tid, owner_uid, name, provider,
                       status, access_host, access_port, access_username,
                       secret_ref, secret_version, remote_root,
                       accelerator_type, accelerator_count, vram_gb,
                       billing_mode, hourly_price_credits, allocation_note,
                       region, instance_type, cpu_cores, memory_gb, storage_gb,
                       external_resource_id, powered_off_at,
                       command_generation, version, created_at, updated_at,
                       deleted_at
                  FROM ops.remote_compute_resources
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                """,
                values,
            ).fetchone()
            if row is None:
                raise RemoteComputeConflict("remote compute resource write was not visible")
            return _resource_from_row(row), replayed

    def retire_unbound_provisioned_resource(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        provisioned: ProvisionedRemoteCompute,
        retired_at: datetime | None = None,
    ) -> bool:
        """Hide a provider allocation that was rolled back before binding."""

        now = retired_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("remote compute retirement time must be timezone-aware")
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            changed = connection.execute(
                """
                UPDATE ops.remote_compute_resources AS resource
                   SET status = 'DELETED', deleted_at = %(retired_at)s,
                       updated_at = %(retired_at)s, version = version + 1
                 WHERE resource.resource_id = %(resource_id)s
                   AND resource.resource_tid = %(resource_tid)s
                   AND resource.owner_uid = %(owner_uid)s
                   AND resource.provider = %(provider)s
                   AND resource.external_resource_id = %(external_resource_id)s
                   AND resource.deleted_at IS NULL
                   AND resource.status IN ('ACTIVE','IDLE','STOPPED','ERROR')
                   AND NOT EXISTS (
                       SELECT 1
                         FROM ops.remote_compute_run_bindings AS binding
                        WHERE binding.resource_id = resource.resource_id
                          AND binding.resource_tid = resource.resource_tid
                          AND binding.owner_uid = resource.owner_uid
                   )
                """,
                {
                    "retired_at": now,
                    "resource_id": provisioned.resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "provider": provisioned.provider,
                    "external_resource_id": provisioned.external_resource_id,
                },
            ).rowcount
        return changed == 1

    def finalize_provider_release(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        resource_id: UUID,
        binding_id: UUID | None = None,
        released_at: datetime | None = None,
    ) -> bool:
        """Project a proven Provider release without repeating the effect.

        A still-active, unsettled binding is a hard conflict: Provider release
        must not silently bypass the unified usage/charge path.  Replays after
        the binding and resource reached their local terminal states are no-op
        successes, including compensation before a resource was catalogued.
        """

        now = released_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("remote compute release time must be timezone-aware")
        with self._transaction(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
        ) as connection:
            unsettled = connection.execute(
                """
                SELECT binding_id
                  FROM ops.remote_compute_run_bindings
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND (%(binding_id)s IS NULL OR binding_id = %(binding_id)s)
                   AND status IN ('ACTIVE','RUNNING','MIGRATING')
                   AND charge_id IS NULL
                 LIMIT 1
                 FOR UPDATE
                """,
                {
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "binding_id": binding_id,
                },
            ).fetchone()
            if unsettled is not None:
                raise RemoteComputeConflict(
                    "remote compute Provider release precedes Billing settlement"
                )
            binding_count = connection.execute(
                """
                UPDATE ops.remote_compute_run_bindings
                   SET status = 'RELEASED', ended_at = coalesce(ended_at, %(at)s)
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND (%(binding_id)s IS NULL OR binding_id = %(binding_id)s)
                   AND status IN ('ACTIVE','RUNNING','MIGRATING')
                   AND charge_id IS NOT NULL
                """,
                {
                    "at": now,
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "binding_id": binding_id,
                },
            ).rowcount
            resource_count = connection.execute(
                """
                UPDATE ops.remote_compute_resources
                   SET status = 'RELEASED', deleted_at = coalesce(deleted_at, %(at)s),
                       updated_at = %(at)s, version = version + 1
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND status <> 'RELEASED'
                """,
                {
                    "at": now,
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).rowcount
        return binding_count > 0 or resource_count > 0


    def list_bindings(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        sid: UUID | None = None,
        rid: UUID | None = None,
        include_terminal: bool = True,
    ) -> tuple[RemoteWorkspaceBinding, ...]:
        """List run bindings from the canonical Product ops table only."""

        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            rows = connection.execute(
                """
                SELECT binding_id, resource_id, resource_tid, owner_uid,
                       sid, rid, lease_generation, status, remote_root,
                       secret_ref, secret_version, created_at, expires_at,
                       payer_tid, billing_account_id, hourly_price_credits,
                       estimated_credits, actual_seconds, actual_credits,
                       charge_id, started_at, ended_at, resource_policy_jsonb
                  FROM ops.remote_compute_run_bindings
                 WHERE resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND (%(sid)s::uuid IS NULL OR sid = %(sid)s)
                   AND (%(rid)s::uuid IS NULL OR rid = %(rid)s)
                   AND (%(include_terminal)s::boolean OR status IN ('ACTIVE', 'RUNNING'))
                 ORDER BY created_at DESC, binding_id DESC
                """,
                {
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "sid": sid,
                    "rid": rid,
                    "include_terminal": include_terminal,
                },
            ).fetchall()
        return tuple(
            RemoteWorkspaceBinding(
                binding_id=row["binding_id"],
                resource_id=row["resource_id"],
                resource_tid=row["resource_tid"],
                owner_uid=row["owner_uid"],
                sid=row["sid"],
                rid=row["rid"],
                lease_generation=int(row["lease_generation"]),
                status=str(row["status"]),
                remote_root=str(row["remote_root"]),
                secret_ref=str(row["secret_ref"]),
                secret_version=str(row["secret_version"] or ""),
                created_at=row.get("created_at"),
                expires_at=row.get("expires_at"),
                payer_tid=row.get("payer_tid"),
                billing_account_id=row.get("billing_account_id"),
                hourly_price_credits=int(row.get("hourly_price_credits") or 0),
                estimated_credits=int(row.get("estimated_credits") or 0),
                actual_seconds=int(row.get("actual_seconds") or 0),
                actual_credits=int(row.get("actual_credits") or 0),
                charge_id=row.get("charge_id"),
                started_at=row.get("started_at"),
                ended_at=row.get("ended_at"),
                resource_policy=dict(row.get("resource_policy_jsonb") or {}),
            )
            for row in rows
        )

    def start_run_binding(
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
        authorize: Callable[
            [Connection[Any], RemoteWorkspaceBinding, RemoteComputePricingContext],
            None,
        ]
        | None = None,
    ) -> tuple[RemoteWorkspaceBinding, bool]:
        """Atomically claim one existing Product resource for a new run.

        Product remains the allocation and payer authority.  The Agent never
        supplies payer identifiers or connection fields; they are copied from
        the admitted run and canonical resource rows while both are locked.
        """

        if estimated_credits < 0:
            raise ValueError("remote compute estimate cannot be negative")
        policy = dict(resource_policy or {})
        if any(
            not isinstance(key, str)
            or not key
            or len(key.encode("utf-8")) > 128
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in policy.items()
        ):
            raise ValueError("remote compute resource policy is invalid")
        if len(_canonical_json(policy)) > 8192:
            raise ValueError("remote compute resource policy is too large")
        now = started_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("remote compute start time must be timezone-aware")
        if expires_at is not None and (
            expires_at.tzinfo is None
            or expires_at.utcoffset() is None
            or expires_at <= now
        ):
            raise ValueError("remote compute expiry must be after start time")

        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            run = connection.execute(
                """
                SELECT sid, payer_tid, billing_account_id, state, deleted_at
                  FROM conversation.runs
                 WHERE rid = %(rid)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                 FOR UPDATE
                """,
                {"rid": rid, "resource_tid": resource_tid, "owner_uid": owner_uid},
            ).fetchone()
            if run is None or run["deleted_at"] is not None:
                raise RemoteComputeNotFound("remote compute run was not found")
            if run["sid"] != sid:
                raise RemoteComputeConflict("remote compute run session does not match")
            if str(run["state"]) in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                raise RemoteComputeConflict("remote compute run is terminal")

            existing = connection.execute(
                """
                SELECT binding.binding_id, binding.resource_id,
                       binding.resource_tid, binding.owner_uid, binding.sid,
                       binding.rid, binding.lease_generation, binding.status,
                       binding.remote_root, binding.secret_ref,
                       binding.secret_version, binding.created_at,
                       binding.expires_at, binding.payer_tid,
                       binding.billing_account_id,
                       binding.hourly_price_credits,
                       binding.estimated_credits, binding.actual_seconds,
                       binding.actual_credits, binding.charge_id,
                       binding.started_at, binding.ended_at,
                       binding.resource_policy_jsonb,
                       resource.provider AS resource_provider,
                       resource.billing_mode AS resource_billing_mode,
                       resource.cpu_cores AS resource_cpu_cores
                  FROM ops.remote_compute_run_bindings AS binding
                  JOIN ops.remote_compute_resources AS resource
                    ON resource.resource_id = binding.resource_id
                   AND resource.resource_tid = binding.resource_tid
                   AND resource.owner_uid = binding.owner_uid
                 WHERE binding.resource_tid = %(resource_tid)s
                   AND binding.owner_uid = %(owner_uid)s
                   AND binding.rid = %(rid)s
                 ORDER BY binding.lease_generation DESC
                 LIMIT 1
                 FOR UPDATE OF binding
                """,
                {"resource_tid": resource_tid, "owner_uid": owner_uid, "rid": rid},
            ).fetchone()
            if existing is not None and str(existing["status"]) in {"ACTIVE", "RUNNING", "MIGRATING"}:
                if existing["resource_id"] != resource_id:
                    raise RemoteComputeConflict("remote compute run already uses another resource")
                if binding_id is not None and existing["binding_id"] != binding_id:
                    raise RemoteComputeConflict(
                        "remote compute run binding identity differs"
                    )
                binding = _binding_from_row(existing)
                if authorize is not None:
                    authorize(
                        connection,
                        binding,
                        RemoteComputePricingContext(
                            provider=str(existing["resource_provider"]),
                            billing_mode=str(existing["resource_billing_mode"]),
                            cpu_cores=int(existing["resource_cpu_cores"] or 0),
                        ),
                    )
                return binding, True

            resource = connection.execute(
                """
                SELECT resource_id, status, remote_root, secret_ref,
                       secret_version, hourly_price_credits, provider,
                       billing_mode, cpu_cores, version
                  FROM ops.remote_compute_resources
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND deleted_at IS NULL
                 FOR UPDATE
                """,
                {
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).fetchone()
            if resource is None:
                raise RemoteComputeNotFound("remote compute resource was not found")
            if str(resource["status"]) not in {"ACTIVE", "IDLE"}:
                raise RemoteComputeConflict("remote compute resource is unavailable")
            occupied = connection.execute(
                """
                SELECT 1
                  FROM ops.remote_compute_run_bindings
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND status IN ('ACTIVE','RUNNING','MIGRATING')
                 LIMIT 1
                """,
                {
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).fetchone()
            if occupied is not None:
                raise RemoteComputeConflict("remote compute resource is already bound")

            generation = 1 if existing is None else int(existing["lease_generation"]) + 1
            binding_id = binding_id or uuid7()
            effective_estimate = estimated_credits
            if effective_estimate == 0 and int(resource["hourly_price_credits"] or 0) > 0:
                # V1 preflight defaults to a bounded 30-minute estimate when
                # the caller has no stronger workload estimate.
                effective_estimate = (int(resource["hourly_price_credits"]) + 1) // 2
            row = connection.execute(
                """
                INSERT INTO ops.remote_compute_run_bindings (
                    binding_id, resource_id, resource_tid, owner_uid, sid, rid,
                    lease_generation, status, remote_root, secret_ref,
                    secret_version, created_at, expires_at, payer_tid,
                    billing_account_id, hourly_price_credits, estimated_credits,
                    actual_seconds, actual_credits, started_at,
                    resource_policy_jsonb
                ) VALUES (
                    %(binding_id)s, %(resource_id)s, %(resource_tid)s,
                    %(owner_uid)s, %(sid)s, %(rid)s, %(generation)s, 'RUNNING',
                    %(remote_root)s, %(secret_ref)s, %(secret_version)s,
                    %(started_at)s, %(expires_at)s, %(payer_tid)s,
                    %(billing_account_id)s, %(hourly_price_credits)s,
                    %(estimated_credits)s, 0, 0, %(started_at)s, %(policy)s
                )
                RETURNING binding_id, resource_id, resource_tid, owner_uid, sid,
                          rid, lease_generation, status, remote_root, secret_ref,
                          secret_version, created_at, expires_at, payer_tid,
                          billing_account_id, hourly_price_credits,
                          estimated_credits, actual_seconds, actual_credits,
                          charge_id, started_at, ended_at, resource_policy_jsonb
                """,
                {
                    "binding_id": binding_id,
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "sid": sid,
                    "rid": rid,
                    "generation": generation,
                    "remote_root": resource["remote_root"],
                    "secret_ref": resource["secret_ref"],
                    "secret_version": resource["secret_version"],
                    "started_at": now,
                    "expires_at": expires_at,
                    "payer_tid": run["payer_tid"],
                    "billing_account_id": run["billing_account_id"],
                    "hourly_price_credits": resource["hourly_price_credits"],
                    "estimated_credits": effective_estimate,
                    "policy": Jsonb(policy),
                },
            ).fetchone()
            updated = connection.execute(
                """
                UPDATE ops.remote_compute_resources
                   SET status = 'BUSY', version = version + 1,
                       updated_at = %(started_at)s
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND version = %(version)s
                   AND status IN ('ACTIVE','IDLE')
                """,
                {
                    "started_at": now,
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "version": resource["version"],
                },
            ).rowcount
            if row is None or updated != 1:
                raise RemoteComputeConflict("remote compute resource allocation lost its fence")
            binding = _binding_from_row(row)
            if authorize is not None:
                authorize(
                    connection,
                    binding,
                    RemoteComputePricingContext(
                        provider=str(resource["provider"]),
                        billing_mode=str(resource["billing_mode"]),
                        cpu_cores=int(resource["cpu_cores"] or 0),
                    ),
                )
            return binding, False

    def get_run_binding(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        rid: UUID,
    ) -> RemoteWorkspaceBinding | None:
        bindings = self.list_bindings(
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            rid=rid,
            include_terminal=True,
        )
        return bindings[0] if bindings else None

    def get_run_cancel_context(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        rid: UUID,
    ) -> Mapping[str, Any] | None:
        """Read the minimum canonical context needed to reuse run cancel."""

        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            row = connection.execute(
                """
                SELECT sid, version, state, desired_state
                  FROM conversation.runs
                 WHERE rid = %(rid)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND deleted_at IS NULL
                """,
                {"rid": rid, "resource_tid": resource_tid, "owner_uid": owner_uid},
            ).fetchone()
        return None if row is None else dict(row)

    def finalize_run_binding(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        binding_id: UUID,
        lease_generation: int,
        ended_at: datetime,
        actual_seconds: int,
        charge_id: UUID,
        actual_credits: int,
    ) -> tuple[RemoteWorkspaceBinding, bool]:
        """Attach one Billing-owned settlement and release the Product lease."""

        if ended_at.tzinfo is None or ended_at.utcoffset() is None:
            raise ValueError("remote compute end time must be timezone-aware")
        if actual_seconds < 0 or actual_credits < 0 or lease_generation < 1:
            raise ValueError("remote compute settlement values are invalid")
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            existing = connection.execute(
                """
                SELECT binding_id, resource_id, resource_tid, owner_uid, sid, rid,
                       lease_generation, status, remote_root, secret_ref,
                       secret_version, created_at, expires_at, payer_tid,
                       billing_account_id, hourly_price_credits,
                       estimated_credits, actual_seconds, actual_credits,
                       charge_id, started_at, ended_at, resource_policy_jsonb
                  FROM ops.remote_compute_run_bindings
                 WHERE binding_id = %(binding_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND lease_generation = %(lease_generation)s
                 FOR UPDATE
                """,
                {
                    "binding_id": binding_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "lease_generation": lease_generation,
                },
            ).fetchone()
            if existing is None:
                raise RemoteComputeNotFound("remote compute run binding was not found")
            if existing["charge_id"] is not None:
                if (
                    existing["charge_id"] != charge_id
                    or int(existing["actual_seconds"] or 0) != actual_seconds
                    or int(existing["actual_credits"] or 0) != actual_credits
                ):
                    raise RemoteComputeConflict("remote compute settlement replay differs")
                return _binding_from_row(existing), True
            started_at = existing["started_at"] or existing["created_at"]
            if started_at is None or ended_at < started_at:
                raise RemoteComputeConflict("remote compute settlement precedes its start")
            row = connection.execute(
                """
                UPDATE ops.remote_compute_run_bindings
                   SET status = 'RELEASED', ended_at = %(ended_at)s,
                       actual_seconds = %(actual_seconds)s,
                       actual_credits = %(actual_credits)s,
                       charge_id = %(charge_id)s
                 WHERE binding_id = %(binding_id)s
                   AND lease_generation = %(lease_generation)s
                   AND charge_id IS NULL
                RETURNING binding_id, resource_id, resource_tid, owner_uid, sid,
                          rid, lease_generation, status, remote_root, secret_ref,
                          secret_version, created_at, expires_at, payer_tid,
                          billing_account_id, hourly_price_credits,
                          estimated_credits, actual_seconds, actual_credits,
                          charge_id, started_at, ended_at, resource_policy_jsonb
                """,
                {
                    "ended_at": ended_at,
                    "actual_seconds": actual_seconds,
                    "actual_credits": actual_credits,
                    "charge_id": charge_id,
                    "binding_id": binding_id,
                    "lease_generation": lease_generation,
                },
            ).fetchone()
            if row is None:
                raise RemoteComputeConflict("remote compute binding finish lost its fence")
            connection.execute(
                """
                UPDATE ops.remote_compute_resources AS resource
                   SET status = 'IDLE', version = version + 1,
                       updated_at = %(ended_at)s
                 WHERE resource.resource_id = %(resource_id)s
                   AND resource.resource_tid = %(resource_tid)s
                   AND resource.owner_uid = %(owner_uid)s
                   AND resource.status = 'BUSY'
                   AND NOT EXISTS (
                       SELECT 1 FROM ops.remote_compute_run_bindings AS active
                        WHERE active.resource_id = resource.resource_id
                          AND active.resource_tid = resource.resource_tid
                          AND active.owner_uid = resource.owner_uid
                          AND active.status IN ('ACTIVE','RUNNING','MIGRATING')
                   )
                """,
                {
                    "ended_at": ended_at,
                    "resource_id": row["resource_id"],
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            )
            return _binding_from_row(row), False

    def request_command(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        resource_id: UUID,
        action: RemoteComputeAction,
        idempotency_key: str,
        request_fingerprint: bytes,
        rid: UUID | None = None,
        binding_id: UUID | None = None,
        expected_lease_generation: int | None = None,
        target_resource_id: UUID | None = None,
        cleanup_items: Sequence[Mapping[str, str]] = (),
        resource_policy: Mapping[str, int] | None = None,
        requested_at: datetime | None = None,
    ) -> RemoteComputeCommandReceipt:
        """Commit an idempotent control intent and Product outbox row atomically."""

        if not idempotency_key or len(idempotency_key.encode("utf-8")) > 512:
            raise ValueError("remote compute idempotency key is invalid")
        if len(request_fingerprint) != 32:
            raise ValueError("remote compute request fingerprint must be SHA-256")
        now = requested_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("remote compute command time must be timezone-aware")
        payload_detail = {
            "cleanup_items": [dict(item) for item in cleanup_items],
            "resource_policy": dict(resource_policy or {}),
        }
        if len(_canonical_json(payload_detail)) > 32 * 1024:
            raise ValueError("remote compute command detail is too large")

        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            existing = connection.execute(
                """
                SELECT event_id AS command_id, source_aggregate_id AS resource_id,
                       payload_jsonb->>'action' AS action,
                       CASE WHEN status = 'REJECTED_TERMINAL' THEN 'FAILED'
                            ELSE 'PENDING' END AS status,
                       (payload_jsonb->>'command_generation')::bigint AS command_generation,
                       NULL::bigint AS result_resource_version,
                       NULL::text AS result_resource_status,
                       decode(payload_jsonb->>'request_fingerprint', 'hex') AS request_fingerprint,
                       '{}'::jsonb AS result_jsonb, last_error_code AS error_code
                  FROM ops.delivery_outbox
                 WHERE source_service = 'PRODUCT_API'
                   AND destination_service = 'AGENT_EXECUTION'
                   AND event_type = 'REMOTE_COMPUTE_COMMAND_REQUESTED'
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND payload_jsonb->>'idempotency_key' = %(idempotency_key)s
                 FOR UPDATE
                """,
                {
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "idempotency_key": idempotency_key,
                },
            ).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] != request_fingerprint:
                    raise RemoteComputeConflict("remote compute idempotency key was reused")
                return _receipt_from_command_row(existing, replayed=True)

            resource = connection.execute(
                """
                SELECT resource_id, status, version, command_generation,
                       provider, external_resource_id, access_host, access_port,
                       access_username, remote_root, secret_ref, secret_version
                  FROM ops.remote_compute_resources
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND deleted_at IS NULL
                 FOR UPDATE
                """,
                {
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).fetchone()
            if resource is None:
                raise RemoteComputeNotFound("remote compute resource was not found")
            _validate_action_state(action, str(resource["status"]))
            unresolved = connection.execute(
                """
                SELECT 1
                  FROM ops.delivery_outbox AS command
                 WHERE command.source_service = 'PRODUCT_API'
                   AND command.destination_service = 'AGENT_EXECUTION'
                   AND command.event_type = 'REMOTE_COMPUTE_COMMAND_REQUESTED'
                   AND command.source_aggregate_id = %(resource_id)s
                   AND command.resource_tid = %(resource_tid)s
                   AND command.owner_uid = %(owner_uid)s
                   AND NOT EXISTS (
                       SELECT 1 FROM ops.delivery_inbox AS result
                        WHERE result.source_service = 'AGENT_EXECUTION'
                          AND result.event_type = 'REMOTE_COMPUTE_COMMAND_RESULT'
                          AND result.correlation_id = command.event_id
                          AND result.effect_status IN ('APPLIED', 'REJECTED_TERMINAL')
                   )
                 LIMIT 1
                """,
                {
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                },
            ).fetchone()
            if unresolved is not None:
                raise RemoteComputeConflict(
                    "remote compute resource already has an unfinished command"
                )

            binding = None
            if action in {RemoteComputeAction.MIGRATE, RemoteComputeAction.APPLY_QUOTA} and rid is None:
                raise ValueError(f"{action.value} requires a run binding")
            if action is RemoteComputeAction.MIGRATE and target_resource_id is None:
                raise ValueError("MIGRATE requires a target resource")
            if rid is not None:
                binding = connection.execute(
                    """
                    SELECT binding_id, rid, lease_generation, status
                      FROM ops.remote_compute_run_bindings
                     WHERE resource_tid = %(resource_tid)s
                       AND owner_uid = %(owner_uid)s
                       AND rid = %(rid)s
                       AND (%(binding_id)s IS NULL OR binding_id = %(binding_id)s)
                     ORDER BY lease_generation DESC
                     LIMIT 1 FOR UPDATE
                    """,
                    {
                        "resource_tid": resource_tid,
                        "owner_uid": owner_uid,
                        "rid": rid,
                        "binding_id": binding_id,
                    },
                ).fetchone()
                if binding is None:
                    raise RemoteComputeNotFound("remote compute run binding was not found")
                if (
                    expected_lease_generation is not None
                    and int(binding["lease_generation"]) != expected_lease_generation
                ):
                    raise RemoteComputeConflict("remote compute run lease is stale")

            target = None
            if target_resource_id is not None:
                target = connection.execute(
                    """
                    SELECT resource_id, status, access_host, access_port,
                           access_username, remote_root, secret_ref, secret_version
                      FROM ops.remote_compute_resources
                     WHERE resource_id = %(target_resource_id)s
                       AND resource_tid = %(resource_tid)s
                       AND owner_uid = %(owner_uid)s
                       AND deleted_at IS NULL
                     FOR UPDATE
                    """,
                    {
                        "target_resource_id": target_resource_id,
                        "resource_tid": resource_tid,
                        "owner_uid": owner_uid,
                    },
                ).fetchone()
                if target is None:
                    raise RemoteComputeNotFound("remote compute migration target was not found")
                if target_resource_id == resource_id:
                    raise RemoteComputeConflict("remote compute migration target is the source")
                if str(target["status"]) not in {"ACTIVE", "IDLE"}:
                    raise RemoteComputeConflict("remote compute migration target is unavailable")

            command_id = uuid7()
            generation = int(resource["command_generation"]) + 1
            next_status = _pending_resource_status(action, str(resource["status"]))
            updated = connection.execute(
                """
                UPDATE ops.remote_compute_resources
                   SET status = %(status)s,
                       command_generation = %(generation)s,
                       version = version + 1,
                       updated_at = %(requested_at)s
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND version = %(expected_version)s
                RETURNING version
                """,
                {
                    "status": next_status,
                    "generation": generation,
                    "requested_at": now,
                    "resource_id": resource_id,
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "expected_version": resource["version"],
                },
            ).fetchone()
            if updated is None:
                raise RemoteComputeConflict("remote compute resource version changed")

            payload = _command_payload(
                command_id=command_id,
                resource_id=resource_id,
                action=action,
                generation=generation,
                resource_version=int(updated["version"]),
                rid=rid,
                binding_id=None if binding is None else binding["binding_id"],
                lease_generation=(
                    None if binding is None else int(binding["lease_generation"])
                ),
                target_resource_id=target_resource_id,
                provider=str(resource["provider"]),
                external_resource_id=str(resource["external_resource_id"]),
                access_host=str(resource["access_host"]),
                access_port=int(resource["access_port"]),
                access_username=str(resource["access_username"]),
                remote_root=str(resource["remote_root"]),
                secret_ref=str(resource["secret_ref"]),
                secret_version=str(resource["secret_version"] or ""),
                target=target,
                detail=payload_detail,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
            )
            self._append_command_outbox(
                connection,
                command_id=command_id,
                resource_id=resource_id,
                resource_tid=resource_tid,
                owner_uid=owner_uid,
                rid=rid,
                occurred_at=now,
                payload=payload,
            )
            return RemoteComputeCommandReceipt(
                command_id=command_id,
                resource_id=resource_id,
                action=action,
                state=RemoteComputeCommandState.PENDING,
                command_generation=generation,
                resource_version=int(updated["version"]),
                resource_status=next_status,
            )

    def get_command(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        command_id: UUID,
    ) -> RemoteComputeCommandReceipt | None:
        with self._transaction(resource_tid=resource_tid, owner_uid=owner_uid) as connection:
            row = connection.execute(
                """
                SELECT outbox.event_id AS command_id,
                       outbox.source_aggregate_id AS resource_id,
                       outbox.payload_jsonb->>'action' AS action,
                       CASE WHEN receipt.effect_status = 'REJECTED_TERMINAL'
                              THEN 'FAILED'
                            WHEN receipt.effect_status = 'APPLIED' THEN 'APPLIED'
                            WHEN outbox.status = 'REJECTED_TERMINAL' THEN 'FAILED'
                            ELSE 'PENDING' END AS status,
                       (outbox.payload_jsonb->>'command_generation')::bigint
                           AS command_generation,
                       resource.version AS result_resource_version,
                       resource.status AS result_resource_status,
                       '{}'::jsonb AS result_jsonb,
                       coalesce(receipt.last_error_code, outbox.last_error_code)
                           AS error_code
                  FROM ops.delivery_outbox AS outbox
                  JOIN ops.remote_compute_resources AS resource
                    ON resource.resource_id = outbox.source_aggregate_id
                   AND resource.resource_tid = outbox.resource_tid
                   AND resource.owner_uid = outbox.owner_uid
                  LEFT JOIN ops.delivery_inbox AS receipt
                    ON receipt.source_service = 'AGENT_EXECUTION'
                   AND receipt.event_type = 'REMOTE_COMPUTE_COMMAND_RESULT'
                   AND receipt.correlation_id = outbox.event_id
                 WHERE outbox.resource_tid = %(resource_tid)s
                   AND outbox.owner_uid = %(owner_uid)s
                   AND outbox.event_id = %(command_id)s
                   AND outbox.event_type = 'REMOTE_COMPUTE_COMMAND_REQUESTED'
                """,
                {
                    "resource_tid": resource_tid,
                    "owner_uid": owner_uid,
                    "command_id": command_id,
                },
            ).fetchone()
        return None if row is None else _receipt_from_command_row(row)

    def apply_result(self, event: RemoteComputeResultEvent) -> RemoteComputeCommandReceipt:
        """Project one Agent result after command, resource and lease fencing."""

        with self._transaction(
            resource_tid=event.resource_tid,
            owner_uid=event.owner_uid,
        ) as connection:
            receipt = connection.execute(
                """
                SELECT payload_sha256
                  FROM ops.effect_receipts
                 WHERE consumer_name = 'REMOTE_COMPUTE_RESULT'
                   AND source_service = 'AGENT_EXECUTION'
                   AND event_id = %(event_id)s
                """,
                {"event_id": event.event_id},
            ).fetchone()
            if receipt is not None:
                if bytes(receipt["payload_sha256"]) != event.payload_sha256:
                    raise RemoteComputeConflict("remote compute result event was reused")
                command = self._command_row(connection, event)
                return RemoteComputeCommandReceipt(
                    command_id=event.command_id,
                    resource_id=event.resource_id,
                    action=event.action,
                    state=(
                        RemoteComputeCommandState.APPLIED
                        if event.status == "APPLIED"
                        else RemoteComputeCommandState.FAILED
                    ),
                    command_generation=event.command_generation,
                    resource_version=int(command.get("result_resource_version") or 0),
                    resource_status=str(command.get("result_resource_status") or ""),
                    replayed=True,
                    error_code=event.error_code,
                    result=dict(event.effect_result or {}),
                )

            existing_inbox = connection.execute(
                """
                SELECT payload_sha256, effect_status, producer_seq
                  FROM ops.delivery_inbox
                 WHERE source_service = 'AGENT_EXECUTION'
                   AND event_id = %(event_id)s
                 FOR UPDATE
                """,
                {"event_id": event.event_id},
            ).fetchone()
            if existing_inbox is not None and bytes(
                existing_inbox["payload_sha256"]
            ) != event.payload_sha256:
                raise RemoteComputeConflict("remote compute result event was reused")
            watermark = connection.execute(
                """
                SELECT coalesce(max(producer_seq), 0) AS applied_seq
                  FROM ops.effect_receipts
                 WHERE consumer_name = 'REMOTE_COMPUTE_RESULT'
                   AND source_service = 'AGENT_EXECUTION'
                   AND source_aggregate_type = 'remote_compute_resource'
                   AND source_aggregate_id = %(resource_id)s
                   AND effect_status IN ('APPLIED', 'REJECTED_TERMINAL')
                """,
                {"resource_id": event.resource_id},
            ).fetchone()
            applied_seq = int(watermark["applied_seq"] if watermark else 0)
            if event.producer_seq <= applied_seq:
                raise RemoteComputeConflict(
                    "remote compute result is behind its compact receipt watermark"
                )
            if event.producer_seq > applied_seq + 1:
                if existing_inbox is None:
                    self._insert_result_inbox(
                        connection,
                        event,
                        effect_status="DEFERRED_GAP",
                    )
                elif existing_inbox["effect_status"] != "DEFERRED_GAP":
                    raise RemoteComputeConflict(
                        "remote compute result inbox is in an incompatible state"
                    )
                return RemoteComputeCommandReceipt(
                    command_id=event.command_id,
                    resource_id=event.resource_id,
                    action=event.action,
                    state=RemoteComputeCommandState.PENDING,
                    command_generation=event.command_generation,
                    resource_version=event.expected_resource_version,
                    resource_status="",
                    delivery_state="DEFERRED_GAP",
                )
            if existing_inbox is not None and existing_inbox["effect_status"] != "DEFERRED_GAP":
                raise RemoteComputeConflict(
                    "remote compute result inbox is in an incompatible state"
                )

            command = self._command_row(connection, event)
            payload = dict(command["payload_jsonb"])
            if (
                payload.get("action") != event.action.value
                or int(payload.get("command_generation") or 0) != event.command_generation
                or int(payload.get("expected_resource_version") or 0)
                != event.expected_resource_version
                or event.causation_event_id != command["command_id"]
            ):
                raise RemoteComputeConflict("remote compute result disagrees with command")
            resource = connection.execute(
                """
                SELECT status, version, command_generation
                  FROM ops.remote_compute_resources
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                 FOR UPDATE
                """,
                {
                    "resource_id": event.resource_id,
                    "resource_tid": event.resource_tid,
                    "owner_uid": event.owner_uid,
                },
            ).fetchone()
            if resource is None:
                raise RemoteComputeNotFound("remote compute result resource was not found")
            if (
                int(resource["command_generation"]) != event.command_generation
                or int(resource["version"]) != event.expected_resource_version
            ):
                raise RemoteComputeConflict("remote compute result lost its resource fence")

            binding = None
            if event.rid is not None:
                binding = connection.execute(
                    """
                    SELECT * FROM ops.remote_compute_run_bindings
                     WHERE resource_tid = %(resource_tid)s
                       AND owner_uid = %(owner_uid)s AND rid = %(rid)s
                       AND (%(binding_id)s IS NULL OR binding_id = %(binding_id)s)
                     ORDER BY lease_generation DESC LIMIT 1 FOR UPDATE
                    """,
                    {
                        "resource_tid": event.resource_tid,
                        "owner_uid": event.owner_uid,
                        "rid": event.rid,
                        "binding_id": event.binding_id,
                    },
                ).fetchone()
                if binding is None or int(binding["lease_generation"]) != event.lease_generation:
                    raise RemoteComputeConflict("remote compute result lost its run lease fence")

            if event.status == "APPLIED":
                final_status = self._apply_success(
                    connection,
                    event,
                    binding,
                    current_status=str(resource["status"]),
                )
            else:
                final_status = "ERROR"
            updated = connection.execute(
                """
                UPDATE ops.remote_compute_resources
                   SET status = %(status)s,
                       powered_off_at = CASE
                           WHEN %(status)s = 'STOPPED' THEN %(occurred_at)s
                           WHEN %(status)s IN ('ACTIVE','IDLE') THEN NULL
                           ELSE powered_off_at END,
                       deleted_at = CASE WHEN %(status)s = 'RELEASED'
                                         THEN %(occurred_at)s ELSE deleted_at END,
                       version = version + 1, updated_at = %(occurred_at)s
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND version = %(expected_version)s
                   AND command_generation = %(generation)s
                RETURNING version, status
                """,
                {
                    "status": final_status,
                    "occurred_at": event.occurred_at,
                    "resource_id": event.resource_id,
                    "resource_tid": event.resource_tid,
                    "owner_uid": event.owner_uid,
                    "expected_version": event.expected_resource_version,
                    "generation": event.command_generation,
                },
            ).fetchone()
            if updated is None:
                raise RemoteComputeConflict("remote compute result projection was fenced out")
            effect_sha = hashlib.sha256(
                event.command_id.bytes + event.payload_sha256
            ).digest()
            connection.execute(
                """
                INSERT INTO ops.effect_receipts (
                    consumer_name, source_service, event_id, payload_sha256,
                    source_aggregate_type, source_aggregate_id, producer_seq,
                    resource_tid, owner_uid, payer_tid, rid, effect_type,
                    effect_idempotency_key, effect_status, effect_sha256,
                    applied_at, retention_class, retain_until
                ) VALUES (
                    'REMOTE_COMPUTE_RESULT', 'AGENT_EXECUTION', %(event_id)s,
                    %(payload_sha256)s, 'remote_compute_resource', %(resource_id)s,
                    %(producer_seq)s, %(resource_tid)s, %(owner_uid)s, NULL, %(rid)s,
                    'PROJECT_REMOTE_COMPUTE_RESULT', %(effect_key)s,
                    %(effect_status)s,
                    %(effect_sha)s, %(occurred_at)s, 'DELIVERY_PENDING',
                    %(occurred_at)s + interval '10 years'
                )
                """,
                {
                    "event_id": event.event_id,
                    "payload_sha256": event.payload_sha256,
                    "resource_id": event.resource_id,
                    "producer_seq": event.producer_seq,
                    "resource_tid": event.resource_tid,
                    "owner_uid": event.owner_uid,
                    "rid": event.rid,
                    "effect_key": f"remote-compute:{event.command_id}",
                    "effect_status": (
                        "APPLIED"
                        if event.status == "APPLIED"
                        else "REJECTED_TERMINAL"
                    ),
                    "effect_sha": effect_sha,
                    "occurred_at": event.occurred_at,
                },
            )
            terminal_status = (
                "APPLIED" if event.status == "APPLIED" else "REJECTED_TERMINAL"
            )
            if existing_inbox is None:
                self._insert_result_inbox(
                    connection,
                    event,
                    effect_status=terminal_status,
                )
            else:
                updated_inbox = connection.execute(
                    """
                    UPDATE ops.delivery_inbox
                       SET effect_status = %(effect_status)s,
                           deferred_until = NULL,
                           processed_at = %(occurred_at)s,
                           last_error_code = %(error_code)s
                     WHERE source_service = 'AGENT_EXECUTION'
                       AND event_id = %(event_id)s
                       AND effect_status = 'DEFERRED_GAP'
                    """,
                    {
                        "event_id": event.event_id,
                        "effect_status": terminal_status,
                        "occurred_at": event.occurred_at,
                        "error_code": event.error_code,
                    },
                ).rowcount
                if updated_inbox != 1:
                    raise RemoteComputeConflict(
                        "remote compute deferred result was fenced out"
                    )
            return RemoteComputeCommandReceipt(
                command_id=event.command_id,
                resource_id=event.resource_id,
                action=event.action,
                state=(
                    RemoteComputeCommandState.APPLIED
                    if event.status == "APPLIED"
                    else RemoteComputeCommandState.FAILED
                ),
                command_generation=event.command_generation,
                resource_version=int(updated["version"]),
                resource_status=str(updated["status"]),
                error_code=event.error_code,
                result=dict(event.effect_result or {}),
            )

    @staticmethod
    def _insert_result_inbox(
        connection: Connection[Any],
        event: RemoteComputeResultEvent,
        *,
        effect_status: str,
    ) -> None:
        deferred = effect_status == "DEFERRED_GAP"
        connection.execute(
            """
            INSERT INTO ops.delivery_inbox (
                source_service, event_id, destination_service, event_type,
                source_aggregate_type, source_aggregate_id, producer_seq,
                resource_tid, owner_uid, payer_tid, rid,
                billing_account_id, schema_version, occurred_at,
                correlation_id, causation_event_id, payload_jsonb,
                payload_sha256, effect_type, effect_idempotency_key,
                effect_status, deferred_until, processed_at, last_error_code,
                received_at, next_attempt_at, expires_at
            ) VALUES (
                'AGENT_EXECUTION', %(event_id)s, 'PRODUCT_API',
                'REMOTE_COMPUTE_COMMAND_RESULT', 'remote_compute_resource',
                %(resource_id)s, %(producer_seq)s, %(resource_tid)s,
                %(owner_uid)s, NULL, %(rid)s, NULL, '2.0.0',
                %(occurred_at)s, %(command_id)s, %(causation_event_id)s,
                %(payload)s, %(payload_sha256)s,
                'PROJECT_REMOTE_COMPUTE_RESULT', %(effect_key)s,
                %(effect_status)s, %(deferred_until)s, %(processed_at)s,
                %(error_code)s, %(occurred_at)s, %(occurred_at)s,
                %(occurred_at)s + interval '14 days'
            )
            """,
            {
                "event_id": event.event_id,
                "resource_id": event.resource_id,
                "producer_seq": event.producer_seq,
                "resource_tid": event.resource_tid,
                "owner_uid": event.owner_uid,
                "rid": event.rid,
                "occurred_at": event.occurred_at,
                "command_id": event.command_id,
                "causation_event_id": event.causation_event_id,
                "payload": Jsonb(dict(event.payload)),
                "payload_sha256": event.payload_sha256,
                "effect_key": f"remote-compute:{event.command_id}",
                "effect_status": effect_status,
                "deferred_until": (
                    event.occurred_at + timedelta(minutes=5) if deferred else None
                ),
                "processed_at": None if deferred else event.occurred_at,
                "error_code": event.error_code,
            },
        )

    @staticmethod
    def _command_row(
        connection: Connection[Any], event: RemoteComputeResultEvent
    ) -> Mapping[str, Any]:
        row = connection.execute(
            """
            SELECT event_id AS command_id, source_aggregate_id AS resource_id,
                   payload_jsonb->>'action' AS action, payload_jsonb,
                   'PENDING'::text AS status,
                   (payload_jsonb->>'command_generation')::bigint AS command_generation,
                   NULL::bigint AS result_resource_version,
                   NULL::text AS result_resource_status,
                   '{}'::jsonb AS result_jsonb, last_error_code AS error_code
              FROM ops.delivery_outbox
             WHERE event_id = %(command_id)s
               AND resource_tid = %(resource_tid)s AND owner_uid = %(owner_uid)s
               AND source_aggregate_id = %(resource_id)s
               AND event_type = 'REMOTE_COMPUTE_COMMAND_REQUESTED'
             FOR UPDATE
            """,
            {
                "command_id": event.command_id,
                "resource_tid": event.resource_tid,
                "owner_uid": event.owner_uid,
                "resource_id": event.resource_id,
            },
        ).fetchone()
        if row is None:
            raise RemoteComputeNotFound("remote compute command was not found")
        return row

    @staticmethod
    def _apply_success(
        connection: Connection[Any],
        event: RemoteComputeResultEvent,
        binding: Mapping[str, Any] | None,
        *,
        current_status: str,
    ) -> str:
        if event.action is RemoteComputeAction.POWER_ON:
            return "IDLE"
        if event.action is RemoteComputeAction.POWER_OFF:
            return "STOPPED"
        if event.action is RemoteComputeAction.RELEASE:
            connection.execute(
                """
                UPDATE ops.remote_compute_run_bindings
                   SET status = 'RELEASED', ended_at = coalesce(ended_at, %(at)s)
                 WHERE resource_id = %(resource_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND status IN ('ACTIVE','RUNNING','MIGRATING')
                """,
                {
                    "at": event.occurred_at,
                    "resource_id": event.resource_id,
                    "resource_tid": event.resource_tid,
                    "owner_uid": event.owner_uid,
                },
            )
            return "RELEASED"
        if event.action is RemoteComputeAction.APPLY_QUOTA and binding is not None:
            connection.execute(
                """
                UPDATE ops.remote_compute_run_bindings
                   SET resource_policy_jsonb = %(policy)s
                 WHERE binding_id = %(binding_id)s AND lease_generation = %(lease)s
                """,
                {
                    "policy": Jsonb(dict(event.effect_result or {}).get("resource_policy", {})),
                    "binding_id": binding["binding_id"],
                    "lease": binding["lease_generation"],
                },
            )
            return current_status
        if event.action is RemoteComputeAction.MIGRATE and binding is not None:
            if event.target_resource_id is None:
                raise RemoteComputeConflict("remote compute migration result has no target")
            target = connection.execute(
                """
                SELECT remote_root, secret_ref, secret_version
                  FROM ops.remote_compute_resources
                 WHERE resource_id = %(target_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s AND deleted_at IS NULL
                 FOR UPDATE
                """,
                {
                    "target_id": event.target_resource_id,
                    "resource_tid": event.resource_tid,
                    "owner_uid": event.owner_uid,
                },
            ).fetchone()
            if target is None:
                raise RemoteComputeNotFound("remote compute migration target disappeared")
            connection.execute(
                """
                UPDATE ops.remote_compute_run_bindings
                   SET status = 'RELEASED', ended_at = coalesce(ended_at, %(at)s)
                 WHERE binding_id = %(binding_id)s AND lease_generation = %(lease)s
                """,
                {
                    "at": event.occurred_at,
                    "binding_id": binding["binding_id"],
                    "lease": binding["lease_generation"],
                },
            )
            connection.execute(
                """
                INSERT INTO ops.remote_compute_run_bindings (
                    binding_id, resource_id, resource_tid, owner_uid, sid, rid,
                    lease_generation, status, remote_root, secret_ref,
                    secret_version, created_at, expires_at, payer_tid,
                    billing_account_id, hourly_price_credits, estimated_credits,
                    actual_seconds, actual_credits, resource_policy_jsonb
                ) VALUES (
                    %(binding_id)s, %(target_id)s, %(resource_tid)s, %(owner_uid)s,
                    %(sid)s, %(rid)s, %(lease)s + 1, 'ACTIVE', %(remote_root)s,
                    %(secret_ref)s, %(secret_version)s, %(at)s, %(expires_at)s,
                    %(payer_tid)s, %(billing_account_id)s, %(hourly_price_credits)s,
                    %(estimated_credits)s, 0, 0, %(policy)s
                )
                """,
                {
                    "binding_id": uuid7(),
                    "target_id": event.target_resource_id,
                    "resource_tid": event.resource_tid,
                    "owner_uid": event.owner_uid,
                    "sid": binding["sid"],
                    "rid": binding["rid"],
                    "lease": binding["lease_generation"],
                    "remote_root": target["remote_root"],
                    "secret_ref": target["secret_ref"],
                    "secret_version": target["secret_version"],
                    "at": event.occurred_at,
                    "expires_at": binding["expires_at"],
                    "payer_tid": binding["payer_tid"],
                    "billing_account_id": binding["billing_account_id"],
                    "hourly_price_credits": binding["hourly_price_credits"],
                    "estimated_credits": binding["estimated_credits"],
                    "policy": Jsonb(dict(binding["resource_policy_jsonb"] or {})),
                },
            )
            target_updated = connection.execute(
                """
                UPDATE ops.remote_compute_resources
                   SET status = 'BUSY', version = version + 1,
                       updated_at = %(at)s
                 WHERE resource_id = %(target_id)s
                   AND resource_tid = %(resource_tid)s
                   AND owner_uid = %(owner_uid)s
                   AND status IN ('ACTIVE','IDLE')
                   AND deleted_at IS NULL
                """,
                {
                    "at": event.occurred_at,
                    "target_id": event.target_resource_id,
                    "resource_tid": event.resource_tid,
                    "owner_uid": event.owner_uid,
                },
            ).rowcount
            if target_updated != 1:
                raise RemoteComputeConflict(
                    "remote compute migration target is no longer available"
                )
            return "IDLE"
        if event.action is RemoteComputeAction.CLEANUP:
            return current_status
        raise RemoteComputeConflict("remote compute result action was not applicable")

    @staticmethod
    def _append_command_outbox(
        connection: Connection[Any],
        *,
        command_id: UUID,
        resource_id: UUID,
        resource_tid: UUID,
        owner_uid: UUID,
        rid: UUID | None,
        occurred_at: datetime,
        payload: Mapping[str, Any],
    ) -> None:
        stream = connection.execute(
            """
            INSERT INTO ops.delivery_streams (
                source_service, destination_service, source_aggregate_type,
                source_aggregate_id, next_producer_seq, version
            ) VALUES ('PRODUCT_API', 'AGENT_EXECUTION', 'remote_compute_resource',
                      %(resource_id)s, 2, 1)
            ON CONFLICT (source_service, destination_service,
                         source_aggregate_type, source_aggregate_id)
            DO UPDATE SET next_producer_seq = ops.delivery_streams.next_producer_seq + 1,
                          version = ops.delivery_streams.version + 1,
                          updated_at = now()
            RETURNING next_producer_seq - 1 AS producer_seq
            """,
            {"resource_id": resource_id},
        ).fetchone()
        if stream is None:
            raise RemoteComputeRepositoryError("remote compute command sequence allocation failed")
        encoded = _canonical_json(payload)
        connection.execute(
            """
            INSERT INTO ops.delivery_outbox (
                event_id, source_service, destination_service, event_type,
                source_aggregate_type, source_aggregate_id, producer_seq,
                resource_tid, owner_uid, payer_tid, rid, billing_account_id,
                schema_version, occurred_at, correlation_id,
                causation_event_id, payload_jsonb, payload_sha256, status
            ) VALUES (
                %(event_id)s, 'PRODUCT_API', 'AGENT_EXECUTION',
                'REMOTE_COMPUTE_COMMAND_REQUESTED', 'remote_compute_resource',
                %(resource_id)s, %(producer_seq)s, %(resource_tid)s,
                %(owner_uid)s, NULL, %(rid)s, NULL, '2.0.0',
                %(occurred_at)s, %(command_id)s, NULL, %(payload)s,
                %(payload_sha256)s, 'PENDING'
            )
            """,
            {
                "event_id": command_id,
                "resource_id": resource_id,
                "producer_seq": stream["producer_seq"],
                "resource_tid": resource_tid,
                "owner_uid": owner_uid,
                "rid": rid,
                "occurred_at": occurred_at,
                "command_id": command_id,
                "payload": Jsonb(dict(payload)),
                "payload_sha256": hashlib.sha256(encoded).digest(),
            },
        )

    @contextmanager
    def _transaction(self, *, resource_tid: UUID, owner_uid: UUID) -> Iterator[Connection[Any]]:
        with self.connection_factory() as connection, connection.transaction():
            activate_runtime_role(connection, expected_role=self.runtime_role)
            if self.enforce_release_gate:
                require_business_schema(connection, PRODUCT_DATABASE)
            connection.execute(
                "SELECT set_config('app.resource_tid', %s, true), "
                "set_config('app.uid', %s, true)",
                (str(resource_tid), str(owner_uid)),
            )
            connection.row_factory = dict_row
            yield connection


class PsycopgRemoteComputeAdminRepository:
    """Narrow super-admin adapter over audited Product PostgreSQL functions."""

    def __init__(
        self,
        connection_factory: Callable[[], Connection[Any]],
        *,
        runtime_role: str = "product_ops_runtime",
        enforce_release_gate: bool = True,
    ) -> None:
        self.connection_factory = connection_factory
        self.runtime_role = runtime_role
        self.enforce_release_gate = enforce_release_gate

    def list_resources(
        self,
        *,
        actor_uid: UUID,
        page: int,
        page_size: int,
    ) -> tuple[tuple[RemoteComputeAdminRecord, ...], int]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("remote compute admin pagination is invalid")
        with self._transaction() as connection:
            self._verify_super_admin(connection, actor_uid)
            rows = connection.execute(
                "SELECT * FROM ops.list_remote_compute_admin_resources(%s, %s)",
                (page_size, (page - 1) * page_size),
            ).fetchall()
            total = int(rows[0]["total_count"]) if rows else 0
            resource_rows = [row for row in rows if row.get("resource_id") is not None]
            owner_ids = list({row["owner_uid"] for row in resource_rows})
            owner_rows = (
                ()
                if not owner_ids
                else connection.execute(
                    """
                    SELECT usr.uid, usr.display_name, usr.account_type,
                           account.login_phone_e164, account.login_email_normalized
                      FROM identity.users AS usr
                      JOIN identity.accounts AS account
                        ON account.account_id = usr.account_id
                     WHERE usr.uid = ANY(%s)
                    """,
                    (owner_ids,),
                ).fetchall()
            )
            owners = {row["uid"]: row for row in owner_rows}
        return (
            tuple(
                RemoteComputeAdminRecord(
                    resource=_admin_resource_from_row(row),
                    owner_display_name=_optional_owner_text(owners, row["owner_uid"], "display_name"),
                    owner_account_type=_optional_owner_text(owners, row["owner_uid"], "account_type"),
                    owner_phone=_optional_owner_text(owners, row["owner_uid"], "login_phone_e164"),
                    owner_email=_optional_owner_text(owners, row["owner_uid"], "login_email_normalized"),
                )
                for row in resource_rows
            ),
            total,
        )

    def update_resource(
        self,
        *,
        actor_uid: UUID,
        operation_id: UUID,
        resource_id: UUID,
        status: str,
        billing_mode: str,
        hourly_price_credits: int,
        allocation_note: str,
    ) -> RemoteComputeResource:
        if status not in {"IDLE", "BUSY", "QUEUED", "STOPPED"}:
            raise ValueError("remote compute admin status is invalid")
        if billing_mode not in {"reserved", "hourly"}:
            raise ValueError("remote compute admin billing mode is invalid")
        if not 0 <= hourly_price_credits <= 10_000_000_000_000:
            raise ValueError("remote compute admin hourly price is invalid")
        note = allocation_note.strip()
        if len(note) > 160 or "\x00" in note:
            raise ValueError("remote compute admin allocation note is invalid")
        with self._transaction() as connection:
            self._verify_super_admin(connection, actor_uid)
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"remote-compute-admin:{operation_id}",),
            )
            row = connection.execute(
                "SELECT * FROM ops.update_remote_compute_admin_resource(%s,%s,%s,%s,%s)",
                (resource_id, status, billing_mode, hourly_price_credits, note),
            ).fetchone()
            if row is None:
                raise RemoteComputeNotFound("remote compute resource does not exist")
            resource = _admin_resource_from_row(row)
            self._record_admin_audit(
                connection,
                actor_uid=actor_uid,
                operation_id=operation_id,
                resource=resource,
            )
        return resource

    @staticmethod
    def _verify_super_admin(connection: Connection[Any], actor_uid: UUID) -> None:
        row = connection.execute(
            "SELECT is_super_admin FROM identity.users WHERE uid = %s",
            (actor_uid,),
        ).fetchone()
        if row is None or row["is_super_admin"] is not True:
            raise PermissionError("remote compute administration requires a super administrator")

    @staticmethod
    def _record_admin_audit(
        connection: Connection[Any],
        *,
        actor_uid: UUID,
        operation_id: UUID,
        resource: RemoteComputeResource,
    ) -> None:
        context = {
            "actor_uid": str(actor_uid),
            "operation_id": str(operation_id),
            "policy": "remote-compute-super-admin-v2",
        }
        evidence = {
            "allocation_note": resource.allocation_note,
            "billing_mode": resource.billing_mode,
            "hourly_price_credits": resource.hourly_price_credits,
            "operation_id": str(operation_id),
            "status": resource.status,
            "version": resource.version,
        }
        context_bytes = json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
        evidence_bytes = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        connection.execute(
            """
            INSERT INTO ops.audit_events (
              actor_uid, actor_service, actor_context_sha256, action,
              target_type, target_id, target_tid, decision, policy_version,
              evidence_jsonb, evidence_sha256
            ) VALUES (
              %(actor_uid)s, 'PRODUCT_API', %(context_sha256)s,
              'admin.remote_compute_resource.update', 'REMOTE_COMPUTE_RESOURCE',
              %(target_id)s, %(target_tid)s, 'EFFECT_RECORDED',
              'remote-compute-super-admin-v2', %(evidence)s, %(evidence_sha256)s
            )
            """,
            {
                "actor_uid": actor_uid,
                "context_sha256": hashlib.sha256(context_bytes).digest(),
                "target_id": str(resource.resource_id),
                "target_tid": resource.resource_tid,
                "evidence": Jsonb(evidence),
                "evidence_sha256": hashlib.sha256(evidence_bytes).digest(),
            },
        )

    @contextmanager
    def _transaction(self) -> Iterator[Connection[Any]]:
        with self.connection_factory() as connection, connection.transaction():
            activate_runtime_role(connection, expected_role=self.runtime_role)
            if self.enforce_release_gate:
                require_business_schema(connection, PRODUCT_DATABASE)
            connection.row_factory = dict_row
            yield connection


def _connection_name(resource_id: UUID) -> str:
    return f"remote_{resource_id}"


def _resource_from_row(row: Mapping[str, Any]) -> RemoteComputeResource:
    return RemoteComputeResource(
        resource_id=row["resource_id"],
        resource_tid=row["resource_tid"],
        owner_uid=row["owner_uid"],
        name=str(row["name"]),
        provider=str(row["provider"]),
        status=str(row["status"]),
        access_host=str(row["access_host"]),
        access_port=int(row["access_port"]),
        access_username=str(row["access_username"]),
        secret_ref=str(row["secret_ref"]),
        secret_version=str(row["secret_version"] or ""),
        remote_root=str(row["remote_root"]),
        accelerator_type=str(row["accelerator_type"] or ""),
        accelerator_count=int(row["accelerator_count"] or 0),
        vram_gb=int(row["vram_gb"] or 0),
        billing_mode=str(row["billing_mode"] or "reserved"),
        hourly_price_credits=int(row["hourly_price_credits"] or 0),
        allocation_note=str(row.get("allocation_note") or ""),
        region=str(row.get("region") or ""),
        instance_type=str(row.get("instance_type") or ""),
        cpu_cores=int(row.get("cpu_cores") or 0),
        memory_gb=int(row.get("memory_gb") or 0),
        storage_gb=int(row.get("storage_gb") or 0),
        external_resource_id=str(row.get("external_resource_id") or ""),
        powered_off_at=row.get("powered_off_at"),
        command_generation=int(row.get("command_generation") or 0),
        version=int(row.get("version") or 1),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        deleted_at=row.get("deleted_at"),
    )


def _admin_resource_from_row(row: Mapping[str, Any]) -> RemoteComputeResource:
    return RemoteComputeResource(
        resource_id=row["resource_id"],
        resource_tid=row["resource_tid"],
        owner_uid=row["owner_uid"],
        name=str(row["name"]),
        provider=str(row["provider"]),
        status=str(row["status"]),
        access_host="",
        access_port=22,
        access_username="",
        secret_ref="",
        secret_version="",
        remote_root="/",
        accelerator_type=str(row.get("accelerator_type") or ""),
        accelerator_count=int(row.get("accelerator_count") or 0),
        vram_gb=int(row.get("vram_gb") or 0),
        billing_mode=str(row.get("billing_mode") or "reserved"),
        hourly_price_credits=int(row.get("hourly_price_credits") or 0),
        allocation_note=str(row.get("allocation_note") or ""),
        region=str(row.get("region") or ""),
        instance_type=str(row.get("instance_type") or ""),
        cpu_cores=int(row.get("cpu_cores") or 0),
        memory_gb=int(row.get("memory_gb") or 0),
        storage_gb=int(row.get("storage_gb") or 0),
        version=int(row.get("version") or 1),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _optional_owner_text(
    owners: Mapping[UUID, Mapping[str, Any]],
    owner_uid: UUID,
    key: str,
) -> str | None:
    owner = owners.get(owner_uid)
    if owner is None or owner.get(key) in (None, ""):
        return None
    return str(owner[key])


def _binding_from_row(row: Mapping[str, Any]) -> RemoteWorkspaceBinding:
    return RemoteWorkspaceBinding(
        binding_id=row["binding_id"],
        resource_id=row["resource_id"],
        resource_tid=row["resource_tid"],
        owner_uid=row["owner_uid"],
        sid=row["sid"],
        rid=row["rid"],
        lease_generation=int(row["lease_generation"]),
        status=str(row["status"]),
        remote_root=str(row["remote_root"]),
        secret_ref=str(row["secret_ref"]),
        secret_version=str(row["secret_version"] or ""),
        created_at=row.get("created_at"),
        expires_at=row.get("expires_at"),
        payer_tid=row.get("payer_tid"),
        billing_account_id=row.get("billing_account_id"),
        hourly_price_credits=int(row.get("hourly_price_credits") or 0),
        estimated_credits=int(row.get("estimated_credits") or 0),
        actual_seconds=int(row.get("actual_seconds") or 0),
        actual_credits=int(row.get("actual_credits") or 0),
        charge_id=row.get("charge_id"),
        started_at=row.get("started_at"),
        ended_at=row.get("ended_at"),
        resource_policy=dict(row.get("resource_policy_jsonb") or {}),
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pending_resource_status(action: RemoteComputeAction, current: str) -> str:
    return {
        RemoteComputeAction.POWER_ON: "POWERING_ON",
        RemoteComputeAction.POWER_OFF: "POWERING_OFF",
        RemoteComputeAction.RELEASE: "RELEASING",
        RemoteComputeAction.CLEANUP: current,
        RemoteComputeAction.MIGRATE: "MIGRATING",
        RemoteComputeAction.APPLY_QUOTA: current,
    }[action]


def _validate_action_state(action: RemoteComputeAction, status: str) -> None:
    if status in {"DELETED", "RELEASING"}:
        raise RemoteComputeConflict("remote compute resource is terminal")
    if action is RemoteComputeAction.POWER_ON and status not in {"STOPPED", "IDLE"}:
        raise RemoteComputeConflict("remote compute resource is not stopped")
    if action in {RemoteComputeAction.POWER_OFF, RemoteComputeAction.RELEASE} and status in {
        "STOPPED",
        "RELEASED",
    }:
        raise RemoteComputeConflict("remote compute resource is already stopped")
    if action in {RemoteComputeAction.CLEANUP, RemoteComputeAction.MIGRATE} and status not in {
        "ACTIVE",
        "IDLE",
        "BUSY",
    }:
        raise RemoteComputeConflict("remote compute resource is not available")


def _command_payload(
    *,
    command_id: UUID,
    resource_id: UUID,
    action: RemoteComputeAction,
    generation: int,
    resource_version: int,
    rid: UUID | None,
    binding_id: UUID | None,
    lease_generation: int | None,
    target_resource_id: UUID | None,
    provider: str,
    external_resource_id: str,
    access_host: str,
    access_port: int,
    access_username: str,
    remote_root: str,
    secret_ref: str,
    secret_version: str,
    target: Mapping[str, Any] | None,
    detail: Mapping[str, Any],
    idempotency_key: str,
    request_fingerprint: bytes,
) -> dict[str, Any]:
    from apps.common.v2_ids import format_typed_id

    return {
        "command_id": format_typed_id("op", command_id),
        "resource_id": format_typed_id("rcres", resource_id),
        "action": action.value,
        "command_generation": generation,
        "expected_resource_version": resource_version,
        "rid": None if rid is None else format_typed_id("rid", rid),
        "binding_id": None if binding_id is None else format_typed_id("rcbind", binding_id),
        "lease_generation": lease_generation,
        "target_resource_id": (
            None if target_resource_id is None else format_typed_id("rcres", target_resource_id)
        ),
        "provider": provider,
        "external_resource_id": external_resource_id,
        "access_host": access_host,
        "access_port": access_port,
        "access_username": access_username,
        "remote_root": remote_root,
        "secret_ref": secret_ref,
        "secret_version": secret_version,
        "target_access_host": None if target is None else str(target["access_host"]),
        "target_access_port": None if target is None else int(target["access_port"]),
        "target_access_username": None if target is None else str(target["access_username"]),
        "target_remote_root": None if target is None else str(target["remote_root"]),
        "target_secret_ref": None if target is None else str(target["secret_ref"]),
        "target_secret_version": None if target is None else str(target["secret_version"] or ""),
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint.hex(),
        "cleanup_items": list(detail.get("cleanup_items") or []),
        "resource_policy": dict(detail.get("resource_policy") or {}),
    }


def _receipt_from_command_row(
    row: Mapping[str, Any], *, replayed: bool = False
) -> RemoteComputeCommandReceipt:
    return RemoteComputeCommandReceipt(
        command_id=row["command_id"],
        resource_id=row["resource_id"],
        action=RemoteComputeAction(str(row["action"])),
        state=RemoteComputeCommandState(str(row["status"])),
        command_generation=int(row["command_generation"]),
        resource_version=int(row.get("result_resource_version") or 0),
        resource_status=str(row.get("result_resource_status") or ""),
        replayed=replayed,
        error_code=row.get("error_code"),
        result=dict(row.get("result_jsonb") or {}),
    )


__all__ = [
    "PsycopgRemoteComputeRepository",
    "RemoteComputeConflict",
    "RemoteComputeNotFound",
    "RemoteComputeRepositoryError",
]
