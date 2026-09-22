"""Fail-closed runtime inventory for durable V2 work.

Each process reads exactly one authoritative PostgreSQL database through its
full-estate cutover-verifier principal.  A Product process never opens an
Agent connection (and vice versa), so there is no cross-database transaction
or hidden distributed snapshot.  Prometheus combines the labelled snapshots.

PostgreSQL metrics are committed as one in-memory snapshot only after every
query succeeds inside an explicit read-only transaction.  Provider and object
storage orphan counts cannot be inferred from catalog rows, so they enter via
bounded, timestamped inventory evidence written atomically by their scanners.
Missing evidence is unavailable, never a fabricated zero.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from apps.v2.database import (
    AGENT_DATABASE,
    PRODUCT_DATABASE,
    DatabaseSpec,
    activate_runtime_role,
    require_business_schema,
)


METRIC_PREFIX = "deepevol_v2_reliability_"
EXTERNAL_INVENTORY_SCHEMA_VERSION = "deepevol-v2-reliability-external-inventory-v1"
DEFAULT_MAX_SNAPSHOT_AGE = timedelta(seconds=120)
MAX_EXTERNAL_INVENTORY_BYTES = 16 * 1024
MAX_COUNTER_VALUE = 2**53 - 1


class InventoryDatabase(StrEnum):
    PRODUCT = "product"
    AGENT = "agent"

    @property
    def spec(self) -> DatabaseSpec:
        return PRODUCT_DATABASE if self is InventoryDatabase.PRODUCT else AGENT_DATABASE

    @property
    def verifier_role(self) -> str:
        return f"{self.value}_cutover_verifier"

    @property
    def external_source(self) -> str:
        return "object_storage" if self is InventoryDatabase.PRODUCT else "provider"


@dataclass(frozen=True, slots=True)
class MetricSample:
    name: str
    value: float
    labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.startswith(METRIC_PREFIX) or not self.name.replace("_", "").isalnum():
            raise ValueError("reliability metric name is invalid")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("reliability metric values must be finite")
        normalized = tuple(sorted((str(key), str(item)) for key, item in self.labels))
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("reliability metric labels must be unique")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "labels", normalized)


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    database: InventoryDatabase
    source: str
    observed_at: datetime
    samples: tuple[MetricSample, ...]

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("inventory observed_at must be timezone-aware")
        if not self.source or len(self.source.encode("utf-8")) > 64:
            raise ValueError("inventory source is invalid")
        labels = {"database": self.database.value, "source": self.source}
        for sample in self.samples:
            if not labels.items() <= dict(sample.labels).items():
                raise ValueError("every inventory sample must carry its database and source")


class InventoryProbe(Protocol):
    database: InventoryDatabase
    source: str

    def collect(self) -> InventorySnapshot: ...


_AGENT_FIELDS = (
    "snapshot_timestamp_seconds",
    "unresolved_unknown_effects",
    "unknown_oldest_age_seconds",
    "outbox_oldest_age_seconds",
    "delivery_dlq_entries",
    "delivery_incidents_open",
    "lease_overdue_entries",
    "processing_overdue_entries",
    "retry_due_overdue_entries",
    "checkpoint_recovery_failures_total",
)


_PRODUCT_FIELDS = (
    "snapshot_timestamp_seconds",
    "outbox_oldest_age_seconds",
    "delivery_dlq_entries",
    "delivery_incidents_open",
    "lease_overdue_entries",
    "processing_overdue_entries",
    "retry_due_overdue_entries",
    "quota_hold_oldest_age_seconds",
    "quota_drift_quota_exceeds_catalog_bytes",
    "quota_drift_catalog_exceeds_quota_bytes",
    "invoice_unmatched_overdue_micros_cny",
    "checkpoint_recovery_failures_total",
)


AGENT_INVENTORY_SQL = """
WITH clock AS (
    SELECT clock_timestamp() AS observed_at
),
unknown_effects AS (
    SELECT opened_at
      FROM llm_runtime.reconciliation_cases
     WHERE status <> 'RESOLVED'
    UNION ALL
    SELECT created_at
      FROM agent_ops.provider_operations
     WHERE status IN ('RESULT_UNKNOWN','AMBIGUOUS','MANUAL_REVIEW')
),
lease_overdue AS (
    SELECT attempt_id::text AS item_id
      FROM execution.attempts, clock
     WHERE status IN ('ACQUIRED','RUNNING','WAITING_SHORT')
       AND lease_expires_at <= clock.observed_at - make_interval(secs => %(run_lease_ttl)s)
    UNION ALL
    SELECT attempt_id::text
      FROM execution.worker_task_attempts, clock
     WHERE status = 'RUNNING'
       AND lease_expires_at <= clock.observed_at - make_interval(secs => %(run_lease_ttl)s)
    UNION ALL
    SELECT job_id::text
      FROM agent_ops.scheduler_jobs, clock
     WHERE status = 'CLAIMED'
       AND lease_expires_at <= clock.observed_at - make_interval(secs => %(run_lease_ttl)s)
),
processing_overdue AS (
    SELECT event_id::text AS item_id
      FROM agent_ops.delivery_inbox, clock
     WHERE effect_status = 'PROCESSING'
       AND claim_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT event_id::text
      FROM agent_ops.delivery_outbox, clock
     WHERE status = 'CLAIMED'
       AND claim_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT operation_id::text
      FROM agent_ops.provider_operations, clock
     WHERE status IN ('PROCESSING','RECONCILING','COMPENSATING')
       AND claim_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT transfer_id::text
      FROM agent_ops.transfer_items, clock
     WHERE claim_expires_at IS NOT NULL
       AND claim_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT case_id::text
      FROM llm_runtime.reconciliation_cases, clock
     WHERE status = 'CLAIMED'
       AND claim_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
),
past_due_retry AS (
    SELECT event_id::text AS item_id
      FROM agent_ops.delivery_outbox, clock
     WHERE status = 'PENDING' AND attempts > 0 AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT event_id::text
      FROM agent_ops.delivery_inbox, clock
     WHERE effect_status = 'RETRYABLE_FAILED' AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT rid::text
      FROM execution.runs, clock
     WHERE status = 'WAITING_RETRY' AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT task_id::text
      FROM execution.worker_tasks, clock
     WHERE status = 'WAITING_RETRY' AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT operation_id::text
      FROM agent_ops.provider_operations, clock
     WHERE status IN ('PENDING','COMPENSATION_PENDING')
       AND attempt_count > 0 AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT job_id::text
      FROM agent_ops.scheduler_jobs, clock
     WHERE status = 'READY' AND attempts > 0 AND next_run_at <= clock.observed_at
),
checkpoint_failures AS (
    SELECT attempt_id::text AS item_id
      FROM execution.attempts
     WHERE failure_class = 'COMPATIBILITY_MISMATCH'
        OR error_code LIKE '%%CHECKPOINT%%'
        OR error_code LIKE '%%FINGERPRINT%%'
    UNION ALL
    SELECT attempt_id::text
      FROM execution.worker_task_attempts
     WHERE failure_class = 'COMPATIBILITY_MISMATCH'
        OR error_code LIKE '%%CHECKPOINT%%'
        OR error_code LIKE '%%FINGERPRINT%%'
)
SELECT
    extract(epoch FROM clock.observed_at)::double precision,
    (SELECT count(*) FROM unknown_effects)::bigint,
    coalesce((SELECT greatest(0, extract(epoch FROM (clock.observed_at - min(opened_at))))
                FROM unknown_effects), 0)::double precision,
    coalesce((SELECT greatest(0, extract(epoch FROM (clock.observed_at - min(created_at))))
                FROM agent_ops.delivery_outbox
               WHERE status IN ('PENDING','CLAIMED','DEAD_LETTERED')), 0)::double precision,
    ((SELECT count(*) FROM agent_ops.delivery_outbox WHERE status = 'DEAD_LETTERED')
      + (SELECT count(*) FROM agent_ops.delivery_inbox WHERE effect_status = 'SECURITY_DLQ'))::bigint,
    ((SELECT count(*) FROM agent_ops.delivery_incidents
       WHERE status = 'OPEN')
      + (SELECT count(*) FROM agent_ops.delivery_inbox
          WHERE effect_status IN ('SECURITY_DLQ','REJECTED_TERMINAL','AMBIGUOUS')))::bigint,
    (SELECT count(*) FROM lease_overdue)::bigint,
    (SELECT count(*) FROM processing_overdue)::bigint,
    (SELECT count(*) FROM past_due_retry)::bigint,
    (SELECT count(*) FROM checkpoint_failures)::bigint
FROM clock
"""


PRODUCT_INVENTORY_SQL = """
WITH clock AS (
    SELECT clock_timestamp() AS observed_at
),
lease_overdue AS (
    SELECT upload_session_id::text AS item_id
      FROM asset.multipart_uploads, clock
     WHERE status IN ('VERIFYING','QUOTA_COMMIT_PENDING','WAITING_RECONCILIATION')
       AND finalize_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT migration_job_id::text
      FROM asset.object_migration_jobs, clock
     WHERE lease_expires_at IS NOT NULL
       AND status NOT IN ('COMPLETED','FAILED')
       AND lease_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT extraction_id::text
      FROM knowledge.extractions, clock
     WHERE status = 'RUNNING'
       AND claim_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT operation_id::text
      FROM ops.remote_compute_operations, clock
     WHERE status = 'CLAIMED'
       AND lease_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT scheduler_job_id::text
      FROM ops.jobs, clock
     WHERE worker_id IS NOT NULL
       AND claim_expires_at <= clock.observed_at - make_interval(secs => lease_ttl_seconds)
),
processing_overdue AS (
    SELECT event_id::text AS item_id
      FROM ops.delivery_inbox, clock
     WHERE effect_status = 'PROCESSING'
       AND claim_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
    UNION ALL
    SELECT event_id::text
      FROM ops.delivery_outbox, clock
     WHERE status = 'CLAIMED'
       AND claim_expires_at <= clock.observed_at - make_interval(secs => %(operation_lease_ttl)s)
),
past_due_retry AS (
    SELECT event_id::text AS item_id
      FROM ops.delivery_outbox, clock
     WHERE status = 'PENDING' AND attempts > 0 AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT event_id::text
      FROM ops.delivery_inbox, clock
     WHERE effect_status = 'RETRYABLE_FAILED' AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT migration_job_id::text
      FROM asset.object_migration_jobs, clock
     WHERE status = 'WAITING_RETRY' AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT extraction_id::text
      FROM knowledge.extractions, clock
     WHERE status = 'PENDING' AND attempt_count > 0 AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT operation_id::text
      FROM ops.remote_compute_operations, clock
     WHERE status IN ('PENDING','PROVIDER_PENDING','COMPENSATION_PENDING','RECONCILIATION_REQUIRED')
       AND attempt_count > 0 AND next_attempt_at <= clock.observed_at
    UNION ALL
    SELECT cleanup_key
      FROM ops.memory_cleanup_tasks, clock
     WHERE status IN ('PENDING','FAILED','UNCERTAIN')
       AND attempt_count > 0 AND retryable AND next_attempt_at <= clock.observed_at
),
quota_holds AS (
    SELECT reservation.created_at AS held_at
      FROM billing.storage_quota_reservations AS reservation
     WHERE reservation.status = 'HELD'
       AND NOT EXISTS (
           SELECT 1 FROM asset.multipart_uploads AS upload
            WHERE upload.quota_reservation_id = reservation.reservation_id
              AND upload.status = 'WAITING_RECONCILIATION'
       )
    UNION ALL
    SELECT admission.created_at
      FROM billing.run_billing_admissions AS admission
     WHERE admission.admission_state IN ('RESERVED','AUTHORIZED')
       AND admission.egress_state <> 'RESULT_UNKNOWN'
),
catalog_bytes AS (
    SELECT resource_tid, coalesce(sum(verified_size_bytes), 0)::bigint AS committed_bytes
      FROM asset.asset_versions
     WHERE verified_size_bytes IS NOT NULL
       AND status IN ('AVAILABLE','CORRUPT','DELETING','DELETED')
     GROUP BY resource_tid
),
quota_bytes AS (
    SELECT resource_tid, coalesce(sum(committed_bytes), 0)::bigint AS committed_bytes
      FROM billing.storage_quota_accounts
     GROUP BY resource_tid
),
quota_drift AS (
    SELECT coalesce(quota.resource_tid, catalog.resource_tid) AS resource_tid,
           coalesce(quota.committed_bytes, 0) - coalesce(catalog.committed_bytes, 0) AS drift_bytes
      FROM quota_bytes AS quota
      FULL OUTER JOIN catalog_bytes AS catalog USING (resource_tid)
),
latest_invoice_lines AS (
    SELECT DISTINCT ON (
               provider_key, provider_original_invoice_id_hash, provider_original_line_id_hash
           )
           invoice_line_id, amount_micros_cny, created_at
      FROM billing.provider_invoice_lines
     ORDER BY provider_key, provider_original_invoice_id_hash,
              provider_original_line_id_hash, revision DESC
),
latest_line_matches AS (
    SELECT DISTINCT ON (match.invoice_line_id)
           match.invoice_line_id, match.classification
      FROM billing.provider_reconciliation_matches AS match
      JOIN billing.provider_reconciliation_runs AS run
        ON run.reconciliation_run_id = match.reconciliation_run_id
     ORDER BY match.invoice_line_id, run.completed_at DESC, match.reconciliation_match_id DESC
),
checkpoint_failures AS (
    SELECT extraction_id::text AS item_id
      FROM knowledge.extractions
     WHERE status = 'FAILED'
       AND (error_code LIKE '%%CHECKPOINT%%' OR error_code LIKE '%%FINGERPRINT%%')
    UNION ALL
    SELECT migration_job_id::text
      FROM asset.object_migration_jobs
     WHERE status = 'FAILED'
       AND (error_code LIKE '%%CHECKPOINT%%' OR error_code LIKE '%%FINGERPRINT%%')
)
SELECT
    extract(epoch FROM clock.observed_at)::double precision,
    coalesce((SELECT greatest(0, extract(epoch FROM (clock.observed_at - min(created_at))))
                FROM ops.delivery_outbox
               WHERE status IN ('PENDING','CLAIMED','DEAD_LETTERED')), 0)::double precision,
    ((SELECT count(*) FROM ops.delivery_outbox WHERE status = 'DEAD_LETTERED')
      + (SELECT count(*) FROM ops.delivery_inbox WHERE effect_status = 'SECURITY_DLQ'))::bigint,
    ((SELECT count(*) FROM ops.delivery_outbox
       WHERE status IN ('DEAD_LETTERED','REJECTED_TERMINAL'))
      + (SELECT count(*) FROM ops.delivery_inbox
          WHERE effect_status IN ('SECURITY_DLQ','REJECTED_TERMINAL')))::bigint,
    (SELECT count(*) FROM lease_overdue)::bigint,
    (SELECT count(*) FROM processing_overdue)::bigint,
    (SELECT count(*) FROM past_due_retry)::bigint,
    coalesce((SELECT greatest(0, extract(epoch FROM (clock.observed_at - min(held_at))))
                FROM quota_holds), 0)::double precision,
    coalesce((SELECT sum(drift_bytes) FILTER (WHERE drift_bytes > 0) FROM quota_drift), 0)::bigint,
    coalesce((SELECT sum(drift_bytes) FILTER (WHERE drift_bytes < 0) FROM quota_drift), 0)::bigint,
    coalesce((SELECT sum(line.amount_micros_cny)
                FROM latest_invoice_lines AS line
                LEFT JOIN latest_line_matches AS match USING (invoice_line_id)
               WHERE (match.classification IS NULL OR match.classification = 'UNMATCHED')
                 AND line.created_at <= clock.observed_at - make_interval(secs => %(invoice_window)s)), 0)::bigint,
    (SELECT count(*) FROM checkpoint_failures)::bigint
FROM clock
"""


class PostgresInventoryProbe:
    """Collect one complete database snapshot through a read-only principal."""

    source = "postgresql"

    def __init__(
        self,
        database: InventoryDatabase,
        connection_factory: Callable[[], Any],
        *,
        run_lease_ttl: timedelta = timedelta(seconds=300),
        operation_lease_ttl: timedelta = timedelta(seconds=120),
        invoice_matching_window: timedelta = timedelta(hours=24),
    ) -> None:
        if run_lease_ttl <= timedelta(0) or operation_lease_ttl <= timedelta(0):
            raise ValueError("inventory lease TTLs must be positive")
        if invoice_matching_window <= timedelta(0):
            raise ValueError("invoice matching window must be positive")
        self.database = database
        self._connection_factory = connection_factory
        self._parameters = {
            "invoice_window": int(invoice_matching_window.total_seconds()),
            "operation_lease_ttl": int(operation_lease_ttl.total_seconds()),
            "run_lease_ttl": int(run_lease_ttl.total_seconds()),
        }

    def collect(self) -> InventorySnapshot:
        sql = PRODUCT_INVENTORY_SQL if self.database is InventoryDatabase.PRODUCT else AGENT_INVENTORY_SQL
        fields = _PRODUCT_FIELDS if self.database is InventoryDatabase.PRODUCT else _AGENT_FIELDS
        with self._connection_factory() as connection, connection.transaction():
            connection.execute("SET TRANSACTION READ ONLY")
            activate_runtime_role(connection, expected_role=self.database.verifier_role)
            read_only = _row_value(connection.execute("SHOW transaction_read_only").fetchone(), 0)
            if str(read_only).lower() != "on":
                raise RuntimeError("reliability inventory transaction is not read-only")
            require_business_schema(connection, self.database.spec)
            row = connection.execute(sql, self._parameters).fetchone()
            if row is None:
                raise RuntimeError("reliability inventory query returned no snapshot")
        values = {field: _row_value(row, index) for index, field in enumerate(fields)}
        observed_at = datetime.fromtimestamp(float(values.pop("snapshot_timestamp_seconds")), UTC)
        labels = (("database", self.database.value), ("source", self.source))
        samples: list[MetricSample] = []
        for name, value in values.items():
            if name == "quota_drift_quota_exceeds_catalog_bytes":
                samples.append(
                    MetricSample(
                        METRIC_PREFIX + "quota_drift_bytes",
                        float(value),
                        (*labels, ("direction", "quota_exceeds_catalog")),
                    )
                )
            elif name == "quota_drift_catalog_exceeds_quota_bytes":
                samples.append(
                    MetricSample(
                        METRIC_PREFIX + "quota_drift_bytes",
                        float(value),
                        (*labels, ("direction", "catalog_exceeds_quota")),
                    )
                )
            else:
                samples.append(MetricSample(METRIC_PREFIX + name, float(value), labels))
        return InventorySnapshot(self.database, self.source, observed_at, tuple(samples))


class ExternalInventoryFileProbe:
    """Read one scanner's complete, atomically replaced orphan inventory."""

    def __init__(
        self,
        database: InventoryDatabase,
        path: Path,
        *,
        source: str | None = None,
    ) -> None:
        self.database = database
        self.source = source or database.external_source
        if self.source != database.external_source:
            raise ValueError("external inventory source does not match its database")
        self.path = path

    def collect(self) -> InventorySnapshot:
        document = _read_external_document(self.path)
        metric_key = (
            "orphan_storage_objects"
            if self.source == "object_storage"
            else "orphan_compute_resources"
        )
        expected_keys = {
            "complete",
            "database",
            metric_key,
            "observed_at",
            "schema_version",
            "source",
        }
        if set(document) != expected_keys:
            raise ValueError("external inventory evidence has unexpected or missing fields")
        if document["schema_version"] != EXTERNAL_INVENTORY_SCHEMA_VERSION:
            raise ValueError("external inventory schema version is unsupported")
        if document["database"] != self.database.value or document["source"] != self.source:
            raise ValueError("external inventory authority binding changed")
        if document["complete"] is not True:
            raise ValueError("external inventory is incomplete")
        observed_at = _parse_observed_at(document["observed_at"])
        count = document[metric_key]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_COUNTER_VALUE:
            raise ValueError("external inventory orphan count is invalid")
        labels = (("database", self.database.value), ("source", self.source))
        return InventorySnapshot(
            self.database,
            self.source,
            observed_at,
            (MetricSample(METRIC_PREFIX + metric_key, float(count), labels),),
        )


class DisabledProviderInventoryProbe:
    """Publish a fresh zero compute inventory only for a fenced-off Provider surface.

    A disabled deployment has no Provider API to query.  Its authoritative
    evidence is therefore the deployment boundary itself: admission is false
    and no remote-compute Provider configuration is present in the process.
    Enabled deployments must use an external Provider inventory file instead.
    """

    database = InventoryDatabase.AGENT
    source = InventoryDatabase.AGENT.external_source

    def __init__(
        self,
        environment: Mapping[str, str],
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        admission = environment.get(
            "DEEPEVOL_V2_REMOTE_COMPUTE_ADMISSION_ENABLED", ""
        ).strip().lower()
        if admission != "false":
            raise ValueError(
                "disabled Provider inventory requires remote-compute admission=false"
            )
        forbidden = sorted(
            name
            for name, value in environment.items()
            if value.strip()
            and (
                name.startswith("DEEPEVOL_V2_REMOTE_COMPUTE_AUTODL_")
                or name.startswith("DEEPEVOL_V2_REMOTE_COMPUTE_ALIYUN_")
                or name
                in {
                    "DEEPEVOL_V2_PRODUCT_REMOTE_COMPUTE_URL",
                    "DEEPEVOL_V2_PRODUCT_REMOTE_COMPUTE_TOKEN",
                    "DEEPEVOL_V2_REMOTE_COMPUTE_SECRETS_FILE_HOST",
                    "DEEPEVOL_V2_REMOTE_COMPUTE_SECRETS_ROOT_HOST",
                }
            )
        )
        if forbidden:
            raise ValueError(
                "disabled Provider inventory rejects remote-compute configuration"
            )
        self._clock = clock

    def collect(self) -> InventorySnapshot:
        observed_at = self._clock()
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("disabled Provider inventory clock must be timezone-aware")
        labels = (("database", self.database.value), ("source", self.source))
        return InventorySnapshot(
            self.database,
            self.source,
            observed_at.astimezone(UTC),
            (MetricSample(METRIC_PREFIX + "orphan_compute_resources", 0, labels),),
        )


def write_external_inventory(
    path: Path,
    *,
    database: InventoryDatabase,
    orphan_count: int,
    observed_at: datetime,
    complete: bool = True,
) -> None:
    """Atomically publish one scanner result without following a target symlink."""

    if not path.is_absolute():
        raise ValueError("external inventory path must be absolute")
    if isinstance(orphan_count, bool) or not isinstance(orphan_count, int):
        raise ValueError("external inventory orphan count must be an integer")
    if not 0 <= orphan_count <= MAX_COUNTER_VALUE:
        raise ValueError("external inventory orphan count is out of range")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("external inventory observed_at must be timezone-aware")
    if not isinstance(complete, bool):
        raise ValueError("external inventory completeness must be boolean")
    source = database.external_source
    metric_key = (
        "orphan_storage_objects"
        if source == "object_storage"
        else "orphan_compute_resources"
    )
    document = {
        "complete": complete,
        "database": database.value,
        metric_key: orphan_count,
        "observed_at": observed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "schema_version": EXTERNAL_INVENTORY_SCHEMA_VERSION,
        "source": source,
    }
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") + b"\n"
    if len(payload) > MAX_EXTERNAL_INVENTORY_BYTES:
        raise ValueError("external inventory payload exceeds its byte limit")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.part")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise ValueError("external inventory target cannot be a symlink")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SourceHealth:
    database: InventoryDatabase
    source: str
    available: bool
    stale: bool
    observed_at: datetime | None
    failure_count: int


@dataclass(frozen=True, slots=True)
class InventoryReport:
    collected_at: datetime
    snapshots: tuple[InventorySnapshot, ...]
    sources: tuple[SourceHealth, ...]

    @property
    def ready(self) -> bool:
        return bool(self.sources) and all(item.available and not item.stale for item in self.sources)


class ReliabilityInventoryCollector:
    """Cache last-known evidence while exposing current source health honestly."""

    def __init__(
        self,
        probes: Sequence[InventoryProbe],
        *,
        max_snapshot_age: timedelta = DEFAULT_MAX_SNAPSHOT_AGE,
        future_tolerance: timedelta = timedelta(seconds=5),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not probes:
            raise ValueError("at least one reliability inventory probe is required")
        identities = {(probe.database, probe.source) for probe in probes}
        if len(identities) != len(probes):
            raise ValueError("reliability inventory probes must have unique identities")
        databases = {probe.database for probe in probes}
        if len(databases) != 1:
            raise ValueError("one collector process may read exactly one database")
        if max_snapshot_age <= timedelta(0) or future_tolerance < timedelta(0):
            raise ValueError("inventory time bounds are invalid")
        self.database = next(iter(databases))
        self._probes = tuple(probes)
        self._max_snapshot_age = max_snapshot_age
        self._future_tolerance = future_tolerance
        self._clock = clock
        self._cache: dict[tuple[InventoryDatabase, str], InventorySnapshot] = {}
        self._failures = dict.fromkeys(identities, 0)
        self._counter_floor: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._lock = threading.Lock()

    def collect(self) -> InventoryReport:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("inventory clock must return timezone-aware values")
        now = now.astimezone(UTC)
        availability: dict[tuple[InventoryDatabase, str], bool] = {}
        for probe in self._probes:
            identity = (probe.database, probe.source)
            try:
                snapshot = probe.collect()
                if snapshot.database is not probe.database or snapshot.source != probe.source:
                    raise ValueError("inventory probe returned a different authority")
                observed_at = snapshot.observed_at.astimezone(UTC)
                if observed_at > now + self._future_tolerance:
                    raise ValueError("inventory evidence timestamp is in the future")
                snapshot = self._preserve_counter_monotonicity(snapshot)
            except Exception:
                with self._lock:
                    self._failures[identity] += 1
                availability[identity] = False
            else:
                with self._lock:
                    self._cache[identity] = snapshot
                availability[identity] = True

        with self._lock:
            snapshots = tuple(self._cache[key] for key in sorted(self._cache, key=_identity_sort_key))
            failures = dict(self._failures)
        by_identity = {(item.database, item.source): item for item in snapshots}
        sources = []
        for identity in sorted(availability, key=_identity_sort_key):
            snapshot = by_identity.get(identity)
            age = None if snapshot is None else now - snapshot.observed_at.astimezone(UTC)
            sources.append(
                SourceHealth(
                    database=identity[0],
                    source=identity[1],
                    available=availability[identity],
                    stale=age is None or age > self._max_snapshot_age,
                    observed_at=None if snapshot is None else snapshot.observed_at,
                    failure_count=failures[identity],
                )
            )
        return InventoryReport(now, snapshots, tuple(sources))

    def _preserve_counter_monotonicity(self, snapshot: InventorySnapshot) -> InventorySnapshot:
        samples = []
        with self._lock:
            for sample in snapshot.samples:
                if sample.name != METRIC_PREFIX + "checkpoint_recovery_failures_total":
                    samples.append(sample)
                    continue
                key = (sample.name, sample.labels)
                floor = max(self._counter_floor.get(key, 0.0), sample.value)
                self._counter_floor[key] = floor
                samples.append(MetricSample(sample.name, floor, sample.labels))
        return InventorySnapshot(
            snapshot.database,
            snapshot.source,
            snapshot.observed_at,
            tuple(samples),
        )


_HELP = {
    "checkpoint_recovery_failures_total": "Durable checkpoint recovery failures observed in retained attempt history.",
    "delivery_dlq_entries": "Delivery rows currently held in a durable dead-letter state.",
    "delivery_incidents_open": "Delivery rows conservatively requiring an audited disposition.",
    "invoice_unmatched_overdue_micros_cny": "Overdue Provider invoice value without an accepted exact match or allocation.",
    "lease_overdue_entries": "Durable leases still active beyond their takeover window.",
    "orphan_compute_resources": "Provider-observed billable compute resources without durable ownership evidence.",
    "orphan_storage_objects": "Object-store entries unowned after the configured GC grace period.",
    "outbox_oldest_age_seconds": "Age of the oldest undelivered transactional outbox event.",
    "processing_overdue_entries": "Expired processing claims not reclaimed within the takeover window.",
    "quota_drift_bytes": "Signed Asset catalog versus committed storage quota drift.",
    "quota_hold_oldest_age_seconds": "Age of the oldest non-unknown storage or billing hold.",
    "retry_due_overdue_entries": "Durable retries whose next-attempt time has elapsed.",
    "snapshot_timestamp_seconds": "Database or external inventory observation time.",
    "unknown_oldest_age_seconds": "Age of the oldest unresolved external effect.",
    "unresolved_unknown_effects": "Unresolved RESULT_UNKNOWN or AMBIGUOUS external effects.",
    "v1_fallback_enabled": "Whether this V2-only runtime can route work to a V1 implementation.",
}


def prometheus_metrics(report: InventoryReport) -> str:
    """Render a bounded Prometheus text snapshot, including unavailable sources."""

    samples: list[MetricSample] = []
    for snapshot in report.snapshots:
        labels = (("database", snapshot.database.value), ("source", snapshot.source))
        samples.append(
            MetricSample(
                METRIC_PREFIX + "snapshot_timestamp_seconds",
                snapshot.observed_at.timestamp(),
                labels,
            )
        )
        samples.extend(snapshot.samples)
    for source in report.sources:
        labels = (("database", source.database.value), ("source", source.source))
        samples.extend(
            (
                MetricSample(METRIC_PREFIX + "source_available", float(source.available), labels),
                MetricSample(METRIC_PREFIX + "source_stale", float(source.stale), labels),
                MetricSample(
                    METRIC_PREFIX + "source_collection_failures_total",
                    float(source.failure_count),
                    labels,
                ),
            )
        )

    # This exporter is part of the V2-only runtime and has no V1 dispatch
    # dependency.  Publishing the invariant as a continuously scraped series
    # lets the seven-day release observation prove that the running estate,
    # rather than a hand-authored report, kept fallback disabled.
    databases = sorted({source.database.value for source in report.sources})
    for database in databases:
        samples.append(
            MetricSample(
                METRIC_PREFIX + "v1_fallback_enabled",
                0.0,
                (("database", database), ("source", "runtime")),
            )
        )

    lines: list[str] = []
    names = sorted({sample.name for sample in samples})
    for name in names:
        suffix = name.removeprefix(METRIC_PREFIX)
        help_text = _HELP.get(suffix)
        if help_text is None:
            help_text = {
                "source_available": "Whether the authoritative source completed its latest collection.",
                "source_collection_failures_total": "Failed authoritative inventory collections in this process.",
                "source_stale": "Whether the last complete source snapshot is missing or stale.",
            }[suffix]
        metric_type = "counter" if suffix.endswith("_total") else "gauge"
        lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"))
        for sample in sorted((item for item in samples if item.name == name), key=lambda item: item.labels):
            label_text = ",".join(
                f'{key}="{_escape_label(value)}"' for key, value in sample.labels
            )
            rendered_value = _render_number(sample.value)
            lines.append(f"{name}{{{label_text}}} {rendered_value}")
    return "\n".join(lines) + "\n"


def _read_external_document(path: Path) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= MAX_EXTERNAL_INVENTORY_BYTES:
            raise ValueError("external inventory must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(MAX_EXTERNAL_INVENTORY_BYTES + 1)
        if len(payload) > MAX_EXTERNAL_INVENTORY_BYTES:
            raise ValueError("external inventory exceeds its byte limit")
    finally:
        os.close(descriptor)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("external inventory is not canonical JSON data") from exc
    if not isinstance(document, dict):
        raise ValueError("external inventory must be a JSON object")
    return document


def _parse_observed_at(value: Any) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("external inventory observed_at is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("external inventory observed_at is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("external inventory observed_at must include an offset")
    return parsed.astimezone(UTC)


def _row_value(row: Any, position: int) -> Any:
    if isinstance(row, Mapping):
        return tuple(row.values())[position]
    return row[position]


def _identity_sort_key(identity: tuple[InventoryDatabase, str]) -> tuple[str, str]:
    return identity[0].value, identity[1]


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _render_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else format(value, ".15g")


__all__ = [
    "AGENT_INVENTORY_SQL",
    "DEFAULT_MAX_SNAPSHOT_AGE",
    "EXTERNAL_INVENTORY_SCHEMA_VERSION",
    "PRODUCT_INVENTORY_SQL",
    "DisabledProviderInventoryProbe",
    "ExternalInventoryFileProbe",
    "InventoryDatabase",
    "InventoryReport",
    "InventorySnapshot",
    "MetricSample",
    "PostgresInventoryProbe",
    "ReliabilityInventoryCollector",
    "SourceHealth",
    "prometheus_metrics",
    "write_external_inventory",
]
