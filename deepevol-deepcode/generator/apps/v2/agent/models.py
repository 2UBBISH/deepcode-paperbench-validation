"""Immutable commands and records for the contract-driven V2 Agent core."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping
from uuid import UUID

from apps.v2.run_authorization import RunAuthorizationClaims
from apps.v2.model_fallback import ChatFallbackPolicy


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    WAITING_RETRY = "WAITING_RETRY"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    WAITING_RECONCILIATION = "WAITING_RECONCILIATION"
    RECOVERING = "RECOVERING"
    FINALIZING = "FINALIZING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunAcquisitionKind(StrEnum):
    INITIAL = "INITIAL"
    RECOVERY = "RECOVERY"
    FAIL_NO_CHECKPOINT = "FAIL_NO_CHECKPOINT"


class AttemptStatus(StrEnum):
    ACQUIRED = "ACQUIRED"
    RUNNING = "RUNNING"
    WAITING_SHORT = "WAITING_SHORT"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    DEFERRED = "DEFERRED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    LOST = "LOST"
    CANCELLED = "CANCELLED"


class LogicalStepStatus(StrEnum):
    PLANNED = "PLANNED"
    DISPATCHED = "DISPATCHED"
    RESULT_AVAILABLE = "RESULT_AVAILABLE"
    CONSUMED = "CONSUMED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class InvocationStatus(StrEnum):
    CREATED = "CREATED"
    AUTHORIZED = "AUTHORIZED"
    DISPATCHING = "DISPATCHING"
    AWAITING_RESULT = "AWAITING_RESULT"
    WAITING_RECONCILIATION = "WAITING_RECONCILIATION"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProviderAttemptState(StrEnum):
    PLANNED = "PLANNED"
    AUTHORIZATION_REJECTED = "AUTHORIZATION_REJECTED"
    ALLOCATED = "ALLOCATED"
    ABORTED_BEFORE_EXPOSURE = "ABORTED_BEFORE_EXPOSURE"
    EXPOSURE_PENDING = "EXPOSURE_PENDING"
    EXPOSURE_OPEN = "EXPOSURE_OPEN"
    SEND_INTENT = "SEND_INTENT"
    IN_FLIGHT = "IN_FLIGHT"
    STREAMING = "STREAMING"
    SUCCEEDED = "SUCCEEDED"
    FAILED_CONFIRMED = "FAILED_CONFIRMED"
    RESULT_UNKNOWN = "RESULT_UNKNOWN"
    CANCELLED_CONFIRMED = "CANCELLED_CONFIRMED"


class OutcomeCertainty(StrEnum):
    NOT_SENT = "NOT_SENT"
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


class UsageStatus(StrEnum):
    NOT_AVAILABLE = "NOT_AVAILABLE"
    PARTIAL = "PARTIAL"
    FINAL = "FINAL"
    MISSING_USAGE = "MISSING_USAGE"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


class ResourceKind(StrEnum):
    LLM = "LLM"
    MEDIA = "MEDIA"


class Modality(StrEnum):
    LANGUAGE = "language"
    IMAGE = "image"

    @property
    def resource_kind(self) -> ResourceKind:
        return ResourceKind.LLM if self is Modality.LANGUAGE else ResourceKind.MEDIA


class UsageFactKind(StrEnum):
    PARTIAL = "PARTIAL"
    FINAL = "FINAL"
    CORRECTION = "CORRECTION"


class UsageAccuracy(StrEnum):
    PROVIDER_REPORTED = "PROVIDER_REPORTED"
    STATEMENT_CONFIRMED = "STATEMENT_CONFIRMED"
    RECONCILED = "RECONCILED"


class RunInboxOutcome(StrEnum):
    APPLIED = "APPLIED"
    REPLAYED = "REPLAYED"
    DEFERRED_GAP = "DEFERRED_GAP"
    SECURITY_DLQ = "SECURITY_DLQ"


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class RunRequested:
    rid: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    billing_account_id: UUID
    initiated_by_uid: UUID
    sid: UUID
    command_event_id: UUID
    authorization_id: UUID
    authorization_version: int
    authorization_claims_sha256: str
    authorization_signing_key_id: str
    authorization_signature: bytes = field(repr=False)
    authorization_claims: RunAuthorizationClaims
    pricing_snapshot_set_sha256: str
    input_manifest_sha256: str
    context_manifest_sha256: str
    workflow_build_id: str
    checkpoint_schema_version: str
    environment_fingerprint: str
    recovery_epoch: int
    control_generation: int

    def __post_init__(self) -> None:
        if self.authorization_version < 1 or self.recovery_epoch < 0 or self.control_generation < 0:
            raise ValueError("run authorization and control generations are invalid")
        _require_digest(self.authorization_claims_sha256, "authorization_claims_sha256")
        _require_digest(self.pricing_snapshot_set_sha256, "pricing_snapshot_set_sha256")
        _require_digest(self.input_manifest_sha256, "input_manifest_sha256")
        _require_digest(self.context_manifest_sha256, "context_manifest_sha256")
        _require_digest(self.environment_fingerprint, "environment_fingerprint")
        if not self.authorization_signing_key_id or not self.authorization_signature:
            raise ValueError("run authorization signature evidence is required")
        if len(self.authorization_signature) != 64:
            raise ValueError("run authorization must use a 64-byte Ed25519 signature")
        if not self.workflow_build_id or len(self.workflow_build_id.encode("utf-8")) > 512:
            raise ValueError("workflow_build_id must contain 1 to 512 bytes")
        if (
            not self.checkpoint_schema_version
            or len(self.checkpoint_schema_version) > 160
            or len(self.checkpoint_schema_version.encode("utf-8")) > 160
        ):
            raise ValueError("checkpoint_schema_version exceeds its frozen bounded-key limit")

    @property
    def authorization_scope(self) -> str:
        return self.authorization_claims.authorization_scope

    @property
    def workload_kind(self) -> str:
        """Return the signed workload kind carried by the authorization.

        ``RunRequested`` is the Agent-side transport model while the
        authorization claims are the source of truth.  Exposing the two
        identity fields here keeps repository/test callers from reaching
        through the model and preserves the existing request contract.
        """

        return self.authorization_claims.workload_kind

    @property
    def workload_id(self) -> UUID:
        """Return the canonical workload identity from the signed claims."""

        # ``RunAuthorizationClaims`` normalizes a missing workload id to
        # ``rid`` during construction, so this cast is safe at runtime.
        assert self.authorization_claims.workload_id is not None
        return self.authorization_claims.workload_id

    @property
    def authorization_claims_jsonb(self) -> dict[str, Any]:
        return self.authorization_claims.as_wire()

    @property
    def pricing_snapshot_refs_jsonb(self) -> list[dict[str, Any]]:
        return [item.as_wire() for item in self.authorization_claims.pricing_snapshot_refs]

    @property
    def authorization_valid_from(self) -> datetime:
        return self.authorization_claims.valid_from

    @property
    def authorization_expires_at(self) -> datetime:
        return self.authorization_claims.expires_at


@dataclass(frozen=True, slots=True)
class RunCancellationRequested:
    rid: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    billing_account_id: UUID
    requested_by_uid: UUID
    cancellation_version: int

    def __post_init__(self) -> None:
        if self.cancellation_version < 1:
            raise ValueError("cancellation_version must be positive")
        if self.requested_by_uid != self.owner_uid:
            raise ValueError("V2 cancellation requester must be the private run owner")


@dataclass(frozen=True, slots=True)
class RunRecord:
    request: RunRequested
    status: RunStatus
    current_attempt_no: int
    lease_generation: int
    created: bool = False


@dataclass(frozen=True, slots=True)
class RunInboxResult:
    outcome: RunInboxOutcome
    applied_producer_seq: int
    run: RunRecord | None = None

    def __post_init__(self) -> None:
        if self.applied_producer_seq < 0:
            raise ValueError("applied_producer_seq cannot be negative")
        if self.outcome in {RunInboxOutcome.APPLIED, RunInboxOutcome.REPLAYED}:
            if self.run is None:
                raise ValueError("an applied or replayed run inbox result requires its run")
        elif self.run is not None:
            raise ValueError("a deferred or security run inbox result cannot expose a run")


@dataclass(frozen=True, slots=True)
class RunCancellationRecord:
    rid: UUID
    status: RunStatus
    desired_state: str
    latest_cancellation_version: int

    def __post_init__(self) -> None:
        if self.desired_state != "CANCELLED":
            raise ValueError("an applied cancellation must request CANCELLED")
        if self.latest_cancellation_version < 1:
            raise ValueError("an applied cancellation requires a positive version")


@dataclass(frozen=True, slots=True)
class RunCancellationInboxResult:
    outcome: RunInboxOutcome
    applied_producer_seq: int
    cancellation: RunCancellationRecord | None = None

    def __post_init__(self) -> None:
        if self.applied_producer_seq < 0:
            raise ValueError("applied_producer_seq cannot be negative")
        if self.outcome in {RunInboxOutcome.APPLIED, RunInboxOutcome.REPLAYED}:
            if self.cancellation is None:
                raise ValueError("an applied or replayed cancellation result requires its projection")
        elif self.cancellation is not None:
            raise ValueError("a deferred or security cancellation result cannot expose a projection")


@dataclass(frozen=True, slots=True)
class RunClaimContext:
    resource_tid: UUID
    owner_uid: UUID


@dataclass(frozen=True, slots=True)
class RunnableRun:
    resource_tid: UUID
    owner_uid: UUID
    rid: UUID
    run_status: RunStatus
    acquisition_kind: RunAcquisitionKind
    recovery_epoch: int
    latest_checkpoint_id: UUID | None
    # Product's intent at claim time. A CANCELLED run is claimed only so the
    # worker can run the cancel path; it never resumes work.
    desired_state: str = "RUNNING"

    def __post_init__(self) -> None:
        if self.recovery_epoch < 0:
            raise ValueError("runnable run recovery_epoch must be non-negative")
        if self.desired_state not in {"RUNNING", "CANCELLED"}:
            raise ValueError("runnable run desired_state must be RUNNING or CANCELLED")
        if self.desired_state == "CANCELLED" and self.run_status is RunStatus.QUEUED:
            raise ValueError("a queued cancellation is terminal before any claim")
        if self.acquisition_kind is RunAcquisitionKind.INITIAL:
            # The repository also admits a verified input-only retry with no
            # registered work. It performs the durable evidence checks before
            # constructing this checkpoint-free acquisition descriptor.
            if self.run_status not in {RunStatus.QUEUED, RunStatus.WAITING_RETRY} or self.latest_checkpoint_id is not None:
                raise ValueError("initial acquisition requires a queued or retrying run without a checkpoint")
        elif self.acquisition_kind is RunAcquisitionKind.RECOVERY:
            if (
                self.run_status
                not in {
                    RunStatus.RUNNING,
                    RunStatus.RECOVERING,
                    RunStatus.FINALIZING,
                    RunStatus.WAITING_RETRY,
                }
                or self.latest_checkpoint_id is None
            ):
                raise ValueError("recovery acquisition requires a nonterminal checkpoint")
        elif self.run_status not in {
            RunStatus.RUNNING,
            RunStatus.RECOVERING,
            RunStatus.WAITING_RETRY,
        } or self.latest_checkpoint_id is not None:
            raise ValueError("no-checkpoint failure requires a started run without recovery state")


@dataclass(frozen=True, slots=True)
class ClaimedRunnableRun:
    candidate: RunnableRun
    authority: "LeaseAuthority"
    environment_snapshot_id: UUID

    def __post_init__(self) -> None:
        if self.candidate.rid != self.authority.rid:
            raise ValueError("claimed candidate and lease must belong to the same run")
        if self.candidate.resource_tid != self.authority.resource_tid:
            raise ValueError("claimed candidate and lease must share a resource boundary")
        if self.candidate.owner_uid != self.authority.owner_uid:
            raise ValueError("claimed candidate and lease must share an owner")
        if self.environment_snapshot_id != self.authority.attempt_id:
            raise ValueError("claimed environment snapshot must be bound to the attempt")


@dataclass(frozen=True, slots=True)
class ExecutionRunContext:
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    rid: UUID
    sid: UUID
    initiated_by_uid: UUID
    billing_account_id: UUID
    authorization_id: UUID
    authorization_version: int
    authorization_claims_sha256: str
    authorization_expires_at: datetime
    pricing_snapshot_refs: tuple[dict[str, Any], ...]
    input_manifest_sha256: str
    context_manifest_sha256: str
    workflow_build_id: str
    checkpoint_schema_version: str
    environment_fingerprint: str
    recovery_epoch: int
    control_generation: int
    desired_state: str

    def __post_init__(self) -> None:
        if self.authorization_version < 1:
            raise ValueError("execution context authorization_version must be positive")
        if self.recovery_epoch < 0 or self.control_generation < 0:
            raise ValueError("execution context generations must be non-negative")
        if self.authorization_expires_at.tzinfo is None:
            raise ValueError("execution context authorization expiry must be timezone-aware")
        for name in (
            "authorization_claims_sha256",
            "input_manifest_sha256",
            "context_manifest_sha256",
            "environment_fingerprint",
        ):
            _require_digest(getattr(self, name), name)
        if not self.pricing_snapshot_refs:
            raise ValueError("execution context requires a pricing snapshot reference")
        if self.desired_state not in {"RUNNING", "CANCELLED"}:
            raise ValueError("execution context desired_state is invalid")
@dataclass(frozen=True, slots=True)
class RuntimeBlobRegistration:
    runtime_blob_id: UUID
    resource_tid: UUID
    rid: UUID
    attempt_id: UUID
    lease_generation: int
    kind: str
    backend: str
    backend_key: str
    key_version: str
    compression: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.lease_generation < 1 or self.size_bytes < 0:
            raise ValueError("runtime blob fence and size must be positive")
        _require_digest(self.sha256, "runtime_blob.sha256")
        for name in ("kind", "backend", "backend_key", "key_version", "compression"):
            value = getattr(self, name)
            if not value or len(value.encode("utf-8")) > 1024:
                raise ValueError(f"runtime blob {name} is empty or oversized")


@dataclass(frozen=True, slots=True)
class RuntimeBlobRecord:
    registration: RuntimeBlobRegistration
    created: bool


@dataclass(frozen=True, slots=True)
class WorkspaceManifest:
    sha256: str
    byte_count: int
    inode_count: int

    def __post_init__(self) -> None:
        _require_digest(self.sha256, "workspace_manifest.sha256")
        if self.byte_count < 0 or self.inode_count < 0:
            raise ValueError("workspace manifest counts cannot be negative")


@dataclass(frozen=True, slots=True)
class WorkspaceCatalogRecord:
    workspace_id: UUID
    relative_workspace_key: str
    manifest: WorkspaceManifest
    status: str
    created: bool = False

    def __post_init__(self) -> None:
        if (
            not self.relative_workspace_key
            or self.relative_workspace_key.startswith("/")
            or ".." in self.relative_workspace_key.split("/")
            or len(self.relative_workspace_key.encode("utf-8")) > 1024
        ):
            raise ValueError("workspace catalog key must be a bounded relative path")
        if self.status not in {
            "ACTIVE",
            "SEALED",
            "ELIGIBLE",
            "QUARANTINED",
            "DELETING",
            "DELETED",
        }:
            raise ValueError("workspace catalog status is invalid")


@dataclass(frozen=True, slots=True)
class TraceStreamRecord:
    trace_stream_id: UUID
    trace_kind: str
    status: str
    event_count: int
    byte_count: int
    expires_at: datetime
    created: bool = False

    def __post_init__(self) -> None:
        if not self.trace_kind or len(self.trace_kind.encode("utf-8")) > 128:
            raise ValueError("trace kind must contain 1 to 128 UTF-8 bytes")
        if self.status not in {"OPEN", "SEALED", "CORRUPT", "DELETING", "DELETED"}:
            raise ValueError("trace stream status is invalid")
        if self.event_count < 0 or self.byte_count < 0:
            raise ValueError("trace stream counts cannot be negative")
        if self.expires_at.tzinfo is None:
            raise ValueError("trace stream expiry must be timezone-aware")


@dataclass(frozen=True, slots=True)
class TraceSegmentRecord:
    trace_segment_id: UUID
    trace_stream_id: UUID
    runtime_blob_id: UUID
    segment_no: int
    first_seq: int
    last_seq: int
    event_count: int
    byte_count: int
    status: str
    created: bool = False

    def __post_init__(self) -> None:
        if self.segment_no < 1 or self.first_seq < 1 or self.last_seq < self.first_seq:
            raise ValueError("trace segment sequence is invalid")
        if self.event_count != self.last_seq - self.first_seq + 1:
            raise ValueError("trace segment event count does not match its sequence range")
        if self.byte_count < 0:
            raise ValueError("trace segment byte count cannot be negative")
        if self.status not in {"OPEN", "SEALED", "CORRUPT", "DELETING", "DELETED"}:
            raise ValueError("trace segment status is invalid")


@dataclass(frozen=True, slots=True)
class LeaseAuthority:
    resource_tid: UUID
    owner_uid: UUID
    rid: UUID
    attempt_id: UUID
    attempt_no: int
    worker_id: str
    generation: int
    token: str = field(repr=False)
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class LogicalLlmStepPlan:
    logical_step_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    rid: UUID
    billing_account_id: UUID
    workflow_node_key: str
    logical_ordinal: int
    semantic_version: str
    invocation_id: UUID
    request_sha256: str

    def __post_init__(self) -> None:
        if self.logical_ordinal < 1 or not self.semantic_version:
            raise ValueError("logical step ordinals and versions are invalid")
        _require_digest(self.request_sha256, "request_sha256")


@dataclass(frozen=True, slots=True)
class InvocationDispatchClaim:
    resource_tid: UUID
    rid: UUID
    logical_step_id: UUID
    invocation_id: UUID
    attempt_id: UUID
    lease_generation: int
    capability_key_id: str
    capability_jti: UUID
    capability_claims_sha256: str
    capability_expires_at: datetime
    control_generation: int

    def __post_init__(self) -> None:
        if self.lease_generation < 1 or self.control_generation < 0:
            raise ValueError("dispatch claim generations are invalid")
        if not self.capability_key_id or len(self.capability_key_id.encode("utf-8")) > 256:
            raise ValueError("dispatch claim capability key id is invalid")
        _require_digest(self.capability_claims_sha256, "capability_claims_sha256")
        if self.capability_expires_at.tzinfo is None:
            raise ValueError("capability_expires_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class InvocationDispatchAuthority:
    """Verified Gateway authority with no plaintext worker lease credential."""

    resource_tid: UUID
    owner_uid: UUID
    rid: UUID
    logical_step_id: UUID
    invocation_id: UUID
    attempt_id: UUID
    lease_generation: int
    control_generation: int
    recovery_epoch: int
    request_sha256: str
    capability_key_id: str
    capability_jti: UUID
    capability_claims_sha256: str
    capability_expires_at: datetime

    def __post_init__(self) -> None:
        if self.lease_generation < 1 or self.control_generation < 0 or self.recovery_epoch < 0:
            raise ValueError("invocation authority generations are invalid")
        if not self.capability_key_id or len(self.capability_key_id.encode("utf-8")) > 256:
            raise ValueError("invocation authority capability key id is invalid")
        _require_digest(self.request_sha256, "request_sha256")
        _require_digest(self.capability_claims_sha256, "capability_claims_sha256")
        if self.capability_expires_at.tzinfo is None:
            raise ValueError("capability_expires_at must be timezone-aware")


class InvocationGrantKind(StrEnum):
    READ = "READ"
    DISPATCH = "DISPATCH"


class InvocationGrantStatus(StrEnum):
    ACTIVE = "ACTIVE"
    CONSUMED = "CONSUMED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class InvocationGrantCredential:
    """Short-lived database authority, separate from invocation identity.

    ``token_hash`` is already a one-way digest.  It remains hidden from repr so
    logs cannot accidentally disclose the value needed to consume a DISPATCH
    grant.
    """

    grant_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    rid: UUID
    invocation_id: UUID
    grant_kind: InvocationGrantKind
    subject_worker_id: str
    grant_jti: UUID
    claims_sha256: str
    token_hash: str = field(repr=False)
    expires_at: datetime
    generation: int = 1
    status: InvocationGrantStatus = InvocationGrantStatus.ACTIVE

    def __post_init__(self) -> None:
        if not self.subject_worker_id or len(self.subject_worker_id.encode("utf-8")) > 256:
            raise ValueError("invocation grant subject worker id is invalid")
        _require_digest(self.claims_sha256, "invocation_grant.claims_sha256")
        _require_digest(self.token_hash, "invocation_grant.token_hash")
        if self.generation < 1:
            raise ValueError("invocation grant generation must be positive")
        if self.expires_at.tzinfo is None:
            raise ValueError("invocation grant expiry must be timezone-aware")


@dataclass(frozen=True, slots=True)
class GatewayQueueClaim:
    """Generation-fenced ownership of one invocation awaiting dispatch."""

    invocation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    rid: UUID
    claim_generation: int
    gateway_instance_id: str
    token: str = field(repr=False)
    expires_at: datetime

    def __post_init__(self) -> None:
        if self.claim_generation < 1:
            raise ValueError("Gateway queue claim generation must be positive")
        if not self.gateway_instance_id or len(self.gateway_instance_id.encode("utf-8")) > 256:
            raise ValueError("Gateway queue claim holder is invalid")
        if not self.token:
            raise ValueError("Gateway queue claim token is empty")
        if self.expires_at.tzinfo is None:
            raise ValueError("Gateway queue claim expiry must be timezone-aware")


@dataclass(frozen=True, slots=True)
class GatewayDispatchCandidate:
    """Receiver-local locator and verified authority receipt for one queued call."""

    authority: InvocationDispatchAuthority
    invocation: InvocationRecord
    queue_claim: GatewayQueueClaim | None = None

    def __post_init__(self) -> None:
        request = self.invocation.request
        authority = self.authority
        if (
            request.resource_tid,
            request.owner_uid,
            request.rid,
            request.logical_step_id,
            request.invocation_id,
            request.request_sha256,
        ) != (
            authority.resource_tid,
            authority.owner_uid,
            authority.rid,
            authority.logical_step_id,
            authority.invocation_id,
            authority.request_sha256,
        ):
            raise ValueError("Gateway dispatch candidate changed immutable authority bindings")


@dataclass(frozen=True, slots=True)
class InvocationRequested:
    invocation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    rid: UUID
    logical_step_id: UUID
    actor_uid: UUID
    billing_account_id: UUID
    caller_service: str
    idempotency_key: str
    authorization_id: UUID
    authorization_version: int
    authorization_claims_sha256: str
    capability_key_id: str
    capability_jti: UUID
    capability_claims_sha256: str
    recovery_epoch: int
    control_generation: int
    resource_kind: ResourceKind
    requested_model: str
    resolved_model: str
    routing_policy_version: str
    request_trace_object_id: UUID
    request_sha256: str
    raw_retention_until: datetime
    parent_invocation_id: UUID | None = None
    workflow_node_key: str = "AGENT_EXECUTION"
    logical_ordinal: int = 1
    semantic_version: str = "gateway-v2-direct-model-v1"

    @property
    def modality(self) -> Modality:
        return Modality.LANGUAGE if self.resource_kind is ResourceKind.LLM else Modality.IMAGE

    def __post_init__(self) -> None:
        if self.authorization_version < 1 or self.recovery_epoch < 0 or self.control_generation < 0:
            raise ValueError("invocation authorization and control generations are invalid")
        if not self.workflow_node_key or self.logical_ordinal < 1 or not self.semantic_version:
            raise ValueError("invocation logical identity is invalid")
        if self.raw_retention_until.tzinfo is None:
            raise ValueError("raw_retention_until must be timezone-aware")
        _require_digest(self.authorization_claims_sha256, "authorization_claims_sha256")
        _require_digest(self.capability_claims_sha256, "capability_claims_sha256")
        _require_digest(self.request_sha256, "request_sha256")


@dataclass(frozen=True, slots=True)
class InvocationRecord:
    request: InvocationRequested
    status: InvocationStatus
    created: bool = False
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class InvocationStatusRecord:
    invocation_id: UUID
    rid: UUID
    logical_step_id: UUID
    status: InvocationStatus
    resolved_model: str
    final_provider_attempt_id: UUID | None
    response_sha256: str | None
    usage_status: UsageStatus
    version: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    usage_metrics: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class InvocationResultRecord:
    invocation_id: UUID
    status: InvocationStatus
    response_trace_object_id: UUID
    response_sha256: str
    media_type: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ResultConsumptionContext:
    resource_tid: UUID
    owner_uid: UUID


@dataclass(frozen=True, slots=True)
class ResultConsumptionReceipt:
    invocation_id: UUID
    checkpoint_id: UUID
    verified_at: datetime
    created: bool


@dataclass(frozen=True, slots=True)
class DispatchAuthorityContext:
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    billing_account_id: UUID
    actor_uid: UUID
    rid: UUID
    logical_step_id: UUID
    invocation_id: UUID
    authorization_id: UUID
    authorization_version: int
    authorization_claims_sha256: str
    capability_jti: UUID
    capability_claims_sha256: str
    recovery_epoch: int
    control_generation: int
    request_sha256: str
    pricing_snapshot_refs: tuple[dict[str, Any], ...]
    chat_fallback_policy: ChatFallbackPolicy | None = None

    def __post_init__(self) -> None:
        if self.authorization_version < 1 or self.recovery_epoch < 0 or self.control_generation < 0:
            raise ValueError("dispatch authority authorization and generations are invalid")
        _require_digest(self.authorization_claims_sha256, "authorization_claims_sha256")
        _require_digest(self.capability_claims_sha256, "capability_claims_sha256")
        _require_digest(self.request_sha256, "request_sha256")
        if not self.pricing_snapshot_refs:
            raise ValueError("dispatch authority requires signed pricing snapshot refs")

        if self.chat_fallback_policy is not None:
            if not isinstance(self.chat_fallback_policy, ChatFallbackPolicy):
                raise ValueError("invalid dispatch fallback policy")
            for route in self.chat_fallback_policy.routes:
                matches = [item for item in self.pricing_snapshot_refs
                           if item.get("resource_kind") == "LLM" and item.get("resource_key") == route.resource_key]
                if len(matches) != 1:
                    raise ValueError("dispatch fallback route lacks unique signed pricing")


@dataclass(frozen=True, slots=True)
class ProviderCapabilitySnapshot:
    provider: str
    modality: Modality
    profile_version: str
    profile_sha256: str
    supported_capabilities: frozenset[str]


@dataclass(frozen=True, slots=True)
class ProviderAttemptRequested:
    provider_attempt_id: UUID
    invocation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    billing_account_id: UUID
    rid: UUID
    logical_step_id: UUID
    attempt_no: int
    retry_of_attempt_id: UUID | None
    provider: str
    provider_account_ref: str
    model: str
    resource_kind: ResourceKind
    authorization_id: UUID
    authorization_version: int
    authorization_claims_sha256: str
    capability_key_id: str
    capability_jti: UUID
    capability_claims_sha256: str
    actor_uid: UUID
    recovery_epoch: int
    control_generation: int
    created_by_attempt_id: UUID
    created_by_lease_generation: int
    request_trace_object_id: UUID
    request_sha256: str
    pricing_snapshot_id: UUID
    forecast_credits: int
    exposure_event_id: UUID
    required_capabilities: tuple[str, ...] = ()
    provider_idempotency_key: str | None = None
    provider_configuration_sha256: str = ""
    authorized_max_output_tokens: int | None = None
    provider_deadline_seconds: int | None = None

    @property
    def modality(self) -> Modality:
        return Modality.LANGUAGE if self.resource_kind is ResourceKind.LLM else Modality.IMAGE

    def __post_init__(self) -> None:
        if self.provider_deadline_seconds is not None and (
            isinstance(self.provider_deadline_seconds, bool) or not isinstance(self.provider_deadline_seconds, int)
            or not 1 <= self.provider_deadline_seconds <= 120
        ):
            raise ValueError("invalid provider deadline")
        if self.authorized_max_output_tokens is not None and (
            isinstance(self.authorized_max_output_tokens, bool) or not isinstance(self.authorized_max_output_tokens, int)
            or not 1 <= self.authorized_max_output_tokens <= 10**12
        ):
            raise ValueError("invalid authorized output budget")
        if self.provider_configuration_sha256:
            _require_digest(self.provider_configuration_sha256, "provider_configuration_sha256")
        if self.attempt_no < 1 or self.authorization_version < 1:
            raise ValueError("provider attempt ordinals and authorization version are invalid")
        if self.created_by_lease_generation < 1 or self.recovery_epoch < 0 or self.control_generation < 0:
            raise ValueError("provider attempt execution and control generations are invalid")
        if self.forecast_credits < 0:
            raise ValueError("provider-attempt forecast cannot be negative")
        if not self.capability_key_id or len(self.capability_key_id.encode("utf-8")) > 256:
            raise ValueError("provider-attempt capability key id is invalid")
        _require_digest(self.authorization_claims_sha256, "authorization_claims_sha256")
        _require_digest(self.capability_claims_sha256, "capability_claims_sha256")
        _require_digest(self.request_sha256, "request_sha256")


@dataclass(frozen=True, slots=True)
class PreparedProviderAttempt:
    request: ProviderAttemptRequested
    capability_snapshot: ProviderCapabilitySnapshot


@dataclass(frozen=True, slots=True)
class ProviderAttemptRecord:
    request: ProviderAttemptRequested
    state: ProviderAttemptState
    outcome_certainty: OutcomeCertainty
    usage_status: UsageStatus
    created: bool = False


@dataclass(frozen=True, slots=True)
class GatewayClaim:
    provider_attempt_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    rid: UUID
    invocation_id: UUID
    logical_step_id: UUID
    claim_generation: int
    gateway_instance_id: str
    token: str = field(repr=False)
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ExpiredProviderGatewayClaim:
    """A newly fenced recovery claim for one abandoned Provider boundary."""

    claim: GatewayClaim
    state: ProviderAttemptState

    def __post_init__(self) -> None:
        if self.state not in {
            ProviderAttemptState.EXPOSURE_OPEN,
            ProviderAttemptState.SEND_INTENT,
            ProviderAttemptState.IN_FLIGHT,
            ProviderAttemptState.STREAMING,
        }:
            raise ValueError("expired Provider claim has no safe recovery disposition")

    @property
    def crossed_send_intent(self) -> bool:
        return self.state is not ProviderAttemptState.EXPOSURE_OPEN


@dataclass(frozen=True, slots=True)
class UsageMetrics:
    resource_kind: ResourceKind
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    media_kind: str | None = None
    operation_count: int | None = None
    output_count: int | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None

    @property
    def modality(self) -> Modality:
        return Modality.LANGUAGE if self.resource_kind is ResourceKind.LLM else Modality.IMAGE

    def __post_init__(self) -> None:
        token_values = (
            self.input_tokens,
            self.cached_input_tokens,
            self.cache_write_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.total_tokens,
        )
        media_values = (
            self.operation_count,
            self.output_count,
            self.width,
            self.height,
            self.duration_ms,
        )
        if self.resource_kind is ResourceKind.LLM:
            if any(value is None for value in token_values) or any(value is not None for value in media_values):
                raise ValueError("confirmed LLM usage requires every contracted token dimension")
            assert all(value is not None for value in token_values)
            if (
                any(value < 0 for value in token_values)
                or self.cached_input_tokens > self.input_tokens
                or self.cache_write_tokens > self.input_tokens - self.cached_input_tokens
                or self.reasoning_tokens > self.output_tokens
            ):
                raise ValueError("LLM usage metrics are inconsistent")
            if self.total_tokens < self.input_tokens + self.output_tokens:
                raise ValueError("LLM total_tokens is below input plus output")
            if self.media_kind is not None:
                raise ValueError("LLM usage cannot include a media kind")
        else:
            if (
                self.operation_count is None
                or self.output_count is None
                or self.media_kind not in {"IMAGE", "AUDIO", "VIDEO", "OTHER"}
            ):
                raise ValueError("confirmed media usage requires kind, operation_count and output_count")
            if any(value is not None for value in token_values):
                raise ValueError("media usage cannot include token dimensions")
            if self.operation_count < 1 or self.output_count < 0 or any(
                value is not None and value < 0 for value in (self.width, self.height, self.duration_ms)
            ):
                raise ValueError("media usage metrics are inconsistent")
            if (self.width is None) != (self.height is None):
                raise ValueError("media width and height must be supplied together")
            if any(value is not None and value == 0 for value in (self.width, self.height)):
                raise ValueError("media width and height must be positive when present")


@dataclass(frozen=True, slots=True)
class FinalUsage:
    usage_id: UUID
    provider_attempt_id: UUID
    invocation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    billing_account_id: UUID
    rid: UUID
    metrics: UsageMetrics
    evidence_trace: TraceObjectRegistration
    usage_source: str
    accuracy: UsageAccuracy
    captured_at: datetime
    outbox_event_id: UUID
    revision_no: int = 1
    fact_kind: UsageFactKind = UsageFactKind.FINAL
    corrects_usage_id: UUID | None = None
    provider_request_id: str | None = None
    provider_response_id: str | None = None
    provider_trace_id: str | None = None
    provider_job_id: str | None = None
    provider_reported_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.revision_no < 1 or self.captured_at.tzinfo is None:
            raise ValueError("usage revision and captured_at are invalid")
        if (self.fact_kind is UsageFactKind.CORRECTION) != (self.corrects_usage_id is not None):
            raise ValueError("usage correction must bind exactly one corrected fact")
        if self.usage_source not in {"PROVIDER_RESPONSE", "PROVIDER_STATEMENT", "CONTROLLED_RECONCILIATION"}:
            raise ValueError("unknown usage source")
        if self.evidence_trace.provider_attempt_id != self.provider_attempt_id:
            raise ValueError("usage evidence trace is not bound to the Provider attempt")

    @property
    def evidence_trace_object_id(self) -> UUID:
        return self.evidence_trace.trace_object_id

    @property
    def evidence_sha256(self) -> str:
        return self.evidence_trace.sha256


@dataclass(frozen=True, slots=True)
class ProviderResult:
    provider_response_id: str | None
    response_trace: TraceObjectRegistration
    provider_request_id: str | None = None
    provider_trace_id: str | None = None
    provider_job_id: str | None = None
    http_status: int | None = None
    usage: FinalUsage | None = None
    missing_usage_outbox_event_id: UUID | None = None

    @property
    def response_trace_object_id(self) -> UUID:
        return self.response_trace.trace_object_id

    @property
    def response_sha256(self) -> str:
        return self.response_trace.sha256


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    attempt: ProviderAttemptRequested
    authority: InvocationDispatchAuthority
    gateway_instance_id: str
    transport_payload: bytes = field(repr=False)
    result_unknown_outbox_event_id: UUID
    provider_failure_outbox_event_id: UUID


@dataclass(frozen=True, slots=True)
class AbandonedProviderAttemptRecovery:
    resource_tid: UUID
    owner_uid: UUID
    provider_attempt_id: UUID
    outbox_event_id: UUID
    error_code: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if not self.error_code:
            raise ValueError("abandoned Provider recovery requires a stable error code")
        if self.observed_at.tzinfo is None:
            raise ValueError("abandoned Provider recovery observed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class KnownResultRecoveryTarget:
    provider_attempt_id: UUID
    invocation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    rid: UUID
    billing_account_id: UUID
    resource_kind: ResourceKind
    state: ProviderAttemptState
    known_result_receipt_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class KnownResultDatabaseClaim:
    """A DB-only recovery claim, or a stable replay of an accepted receipt."""

    target: KnownResultRecoveryTarget
    receipt_sha256: str
    claim: GatewayClaim | None
    replayed: bool

    def __post_init__(self) -> None:
        _require_digest(self.receipt_sha256, "known_result_receipt_sha256")
        if self.replayed != (self.claim is None):
            raise ValueError("known-result claim replay shape is invalid")


@dataclass(frozen=True, slots=True)
class ProductExposureCommand:
    provider_attempt_id: UUID
    invocation_id: UUID
    rid: UUID
    authorization_id: UUID
    authorization_version: int
    authorization_claims_sha256: bytes
    billing_account_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    actor_uid: UUID
    recovery_epoch: int
    pricing_snapshot_id: UUID
    resource_kind: str
    provider_key: str
    model_key: str
    pricing_resource_key: str
    capability_profile_version: str
    capability_profile_sha256: bytes
    forecast_credits: int


@dataclass(frozen=True, slots=True)
class ProductExposureReceipt:
    provider_attempt_id: UUID
    rid: UUID
    status: str
    created: bool


@dataclass(frozen=True, slots=True)
class TraceObjectRegistration:
    trace_object_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    rid: UUID
    invocation_id: UUID
    provider_attempt_id: UUID | None
    kind: str
    backend: str
    storage_uri: str
    sha256: str
    size_bytes: int
    content_type: str
    encryption_key_ref: str
    classification: str
    retention_until: datetime

    def __post_init__(self) -> None:
        _require_digest(self.sha256, "trace_object.sha256")
        if self.size_bytes < 0 or self.retention_until.tzinfo is None:
            raise ValueError("trace object size and retention are invalid")
        if self.classification not in {"CONFIDENTIAL", "RESTRICTED"}:
            raise ValueError("trace object classification is invalid")


@dataclass(frozen=True, slots=True)
class ResultConsumptionAck:
    resource_tid: UUID
    owner_uid: UUID
    invocation_id: UUID
    rid: UUID
    logical_step_id: UUID
    checkpoint_id: UUID
    response_sha256: str
    source_event_id: UUID
    consumed_at: datetime

    def __post_init__(self) -> None:
        _require_digest(self.response_sha256, "response_sha256")
        if self.consumed_at.tzinfo is None:
            raise ValueError("consumed_at must be timezone-aware")


class ReconciliationDecision(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED_CONFIRMED = "FAILED_CONFIRMED"
    CANCELLED_CONFIRMED = "CANCELLED_CONFIRMED"


class ReconciliationCaseStatus(StrEnum):
    OPEN = "OPEN"
    CLAIMED = "CLAIMED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True, slots=True)
class ReconciliationCaseRecord:
    """Secret-free operator projection for one unknown Provider outcome."""

    case_id: UUID
    operation_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    rid: UUID
    invocation_id: UUID
    provider_attempt_id: UUID
    unknown_event_id: UUID
    reason_code: str
    status: ReconciliationCaseStatus
    priority: int
    claim_holder: str | None
    claim_generation: int
    claim_expires_at: datetime | None
    evidence_sha256: str | None
    evidence_blob_id: UUID | None
    decision: ReconciliationDecision | None
    resolution_actor_type: str | None
    resolution_actor_id: str | None
    resolution_event_id: UUID | None
    opened_at: datetime
    updated_at: datetime
    resolved_at: datetime | None

    def __post_init__(self) -> None:
        if not self.reason_code or len(self.reason_code.encode("utf-8")) > 256:
            raise ValueError("reconciliation reason code is invalid")
        if not -1000 <= self.priority <= 1000 or self.claim_generation < 0:
            raise ValueError("reconciliation priority or generation is invalid")
        for name in ("claim_expires_at", "opened_at", "updated_at", "resolved_at"):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware")
        if self.evidence_sha256 is not None:
            _require_digest(self.evidence_sha256, "evidence_sha256")
        if self.status is ReconciliationCaseStatus.CLAIMED:
            if (
                self.claim_generation < 1
                or self.claim_expires_at is None
                or not self.claim_holder
            ):
                raise ValueError("claimed reconciliation case requires a live generation")
        elif self.claim_expires_at is not None or self.claim_holder is not None:
            raise ValueError("unclaimed reconciliation case cannot expose a claim expiry")
        resolution_values = (
            self.evidence_sha256,
            self.decision,
            self.resolution_actor_type,
            self.resolution_actor_id,
            self.resolution_event_id,
            self.resolved_at,
        )
        if self.status is ReconciliationCaseStatus.RESOLVED:
            if not all(value is not None for value in resolution_values):
                raise ValueError("resolved reconciliation case requires a complete receipt")
        elif any(value is not None for value in resolution_values):
            raise ValueError("unresolved reconciliation case cannot contain a final receipt")


@dataclass(frozen=True, slots=True)
class ReconciliationCaseClaim:
    case: ReconciliationCaseRecord
    holder: str
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.case.status is not ReconciliationCaseStatus.CLAIMED:
            raise ValueError("reconciliation claim requires CLAIMED status")
        if not self.holder or len(self.holder.encode("utf-8")) > 256:
            raise ValueError("reconciliation claim holder is invalid")
        if len(self.token) != 64 or any(ch not in "0123456789abcdef" for ch in self.token):
            raise ValueError("reconciliation claim token must be 32-byte lowercase hex")


@dataclass(frozen=True, slots=True)
class ReconciliationEvidenceCommand:
    command_id: UUID
    provider_attempt_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    decision: ReconciliationDecision
    evidence_ref: str
    evidence_sha256: str
    evidence_trace: TraceObjectRegistration
    decided_by: str
    occurred_at: datetime
    outbox_event_id: UUID
    result: ProviderResult | None = None
    missing_usage_outbox_event_id: UUID | None = None

    def __post_init__(self) -> None:
        _require_digest(self.evidence_sha256, "evidence_sha256")
        if self.occurred_at.tzinfo is None:
            raise ValueError("reconciliation occurred_at must be timezone-aware")
        if not self.evidence_ref or not self.decided_by:
            raise ValueError("reconciliation evidence reference and principal are required")
        trace = self.evidence_trace
        if (
            trace.resource_tid != self.resource_tid
            or trace.owner_uid != self.owner_uid
            or trace.provider_attempt_id != self.provider_attempt_id
            or trace.kind != "PROVIDER_RECONCILIATION_EVIDENCE"
            or trace.sha256 != self.evidence_sha256
        ):
            raise ValueError("reconciliation evidence trace does not match the command")
        if self.decision is ReconciliationDecision.SUCCEEDED and self.result is None:
            raise ValueError("successful reconciliation requires the observed Provider result")
        result_missing_id = self.result.missing_usage_outbox_event_id if self.result is not None else None
        if (
            result_missing_id is not None
            and self.missing_usage_outbox_event_id is not None
            and result_missing_id != self.missing_usage_outbox_event_id
        ):
            raise ValueError("reconciliation has conflicting missing-usage event ids")
        has_usage = self.result is not None and self.result.usage is not None
        if has_usage == (self.effective_missing_usage_outbox_event_id is not None):
            raise ValueError("reconciliation requires exactly one final usage fact or missing-usage event id")

    @property
    def effective_missing_usage_outbox_event_id(self) -> UUID | None:
        if self.missing_usage_outbox_event_id is not None:
            return self.missing_usage_outbox_event_id
        return self.result.missing_usage_outbox_event_id if self.result is not None else None


__all__ = [name for name in globals() if not name.startswith("_")]
