"""Agent consumer for Product-owned Remote Compute lifecycle commands."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from secrets import token_bytes
from typing import Any, Iterator, Protocol
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from apps.common.v2_ids import InvalidTypedId, format_typed_id, parse_typed_id, uuid7
from apps.v2.database import AGENT_DATABASE, activate_runtime_role, require_business_schema
from apps.v2.reliability import HeartbeatRunner, Lease

from .models import RemoteComputeAction, RemoteComputeCommand, RemoteComputeResultEvent
from .provider_operation_runtime import (
    DurableProviderOperationExecutor,
    ProviderOperationInProgress,
    ProviderOperationResultUnknown,
)
from .secrets import FileRemoteComputeSecretResolver, RemoteComputeSecretError


class RemoteComputeCommandError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RemoteComputeEffectResult:
    command_id: UUID
    replayed: bool
    effect_state: str
    applied_producer_seq: int


@dataclass(frozen=True, slots=True)
class _InboxClaim:
    state: str
    generation: int | None = None
    token_hash: str | None = None
    token: bytes | None = field(default=None, repr=False, compare=False)
    acquired_at: datetime | None = None
    expires_at: datetime | None = None


class RemoteComputeProvider(Protocol):
    def apply(
        self,
        command: RemoteComputeCommand,
        *,
        secret: str,
        target_secret: str | None = None,
    ) -> Mapping[str, Any]: ...


def decode_remote_compute_command(raw: Mapping[str, Any]) -> RemoteComputeCommand:
    expected_envelope = {
        "event_id", "source_service", "destination_service", "event_type",
        "resource_tid", "owner_uid", "payer_tid", "rid", "source_aggregate_type",
        "source_aggregate_id", "producer_seq", "schema_version", "occurred_at",
        "correlation_id", "causation_event_id", "payload_sha256", "payload",
    }
    if set(raw) != expected_envelope:
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    if (
        raw.get("source_service") != "PRODUCT_API"
        or raw.get("destination_service") != "AGENT_EXECUTION"
        or raw.get("event_type") != "REMOTE_COMPUTE_COMMAND_REQUESTED"
        or raw.get("source_aggregate_type") != "remote_compute_resource"
        or raw.get("schema_version") != "2.0.0"
    ):
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    expected_payload = {
        "command_id", "resource_id", "action", "command_generation",
        "expected_resource_version", "rid", "binding_id", "lease_generation",
        "target_resource_id", "cleanup_items", "resource_policy",
        "idempotency_key", "request_fingerprint", "provider",
        "external_resource_id", "access_host", "access_port", "access_username",
        "remote_root", "secret_ref", "secret_version", "target_access_host",
        "target_access_port", "target_access_username", "target_remote_root",
        "target_secret_ref", "target_secret_version",
    }
    if set(payload) != expected_payload:
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    encoded = json.dumps(
        dict(payload), allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != raw.get("payload_sha256"):
        raise RemoteComputeCommandError("EVENT_HASH_MISMATCH")

    def typed(value: Any, prefix: str) -> UUID:
        try:
            return parse_typed_id(value, expected_prefix=prefix)
        except InvalidTypedId as exc:
            raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED") from exc

    try:
        action = RemoteComputeAction(str(payload["action"]))
        command_id = typed(payload["command_id"], "op")
        resource_id = typed(payload["resource_id"], "rcres")
        resource_tid = typed(raw["resource_tid"], "tid")
        owner_uid = typed(raw["owner_uid"], "uid")
        generation = int(payload["command_generation"])
        expected_version = int(payload["expected_resource_version"])
        rid = None if payload["rid"] is None else typed(payload["rid"], "rid")
        binding_id = (
            None if payload["binding_id"] is None else typed(payload["binding_id"], "rcbind")
        )
        target = (
            None if payload["target_resource_id"] is None
            else typed(payload["target_resource_id"], "rcres")
        )
        lease = None if payload["lease_generation"] is None else int(payload["lease_generation"])
        access_port = int(payload["access_port"])
        target_port = 22 if payload["target_access_port"] is None else int(payload["target_access_port"])
    except (TypeError, ValueError) as exc:
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED") from exc
    if command_id != typed(raw["correlation_id"], "op") or resource_id != typed(
        raw["source_aggregate_id"], "rcres"
    ):
        raise RemoteComputeCommandError("EVENT_AUTHORITY_MISMATCH")
    if generation < 1 or expected_version < 1 or (lease is not None and lease < 1):
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    if not 1 <= access_port <= 65535 or not 1 <= target_port <= 65535:
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    cleanup = payload["cleanup_items"]
    policy = payload["resource_policy"]
    if not isinstance(cleanup, list) or not isinstance(policy, Mapping):
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    idempotency_key = payload["idempotency_key"]
    request_fingerprint = payload["request_fingerprint"]
    if (
        not isinstance(idempotency_key, str)
        or not idempotency_key
        or idempotency_key != idempotency_key.strip()
        or len(idempotency_key.encode("utf-8")) > 512
        or not isinstance(request_fingerprint, str)
        or len(request_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in request_fingerprint)
    ):
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    occurred = datetime.fromisoformat(str(raw["occurred_at"]).replace("Z", "+00:00"))
    return RemoteComputeCommand(
        command_id=command_id,
        resource_id=resource_id,
        resource_tid=resource_tid,
        owner_uid=owner_uid,
        action=action,
        command_generation=generation,
        expected_resource_version=expected_version,
        producer_seq=int(raw["producer_seq"]),
        rid=rid,
        binding_id=binding_id,
        lease_generation=lease,
        target_resource_id=target,
        idempotency_key=idempotency_key,
        request_fingerprint=request_fingerprint,
        provider=str(payload["provider"]),
        external_resource_id=str(payload["external_resource_id"]),
        access_host=str(payload["access_host"]),
        access_port=access_port,
        access_username=str(payload["access_username"]),
        remote_root=str(payload["remote_root"]),
        secret_ref=str(payload["secret_ref"]),
        secret_version=str(payload["secret_version"]),
        target_access_host=str(payload["target_access_host"] or ""),
        target_access_port=target_port,
        target_access_username=str(payload["target_access_username"] or ""),
        target_remote_root=str(payload["target_remote_root"] or ""),
        target_secret_ref=str(payload["target_secret_ref"] or ""),
        target_secret_version=str(payload["target_secret_version"] or ""),
        cleanup_items=tuple(dict(item) for item in cleanup if isinstance(item, Mapping)),
        resource_policy={str(key): int(value) for key, value in policy.items()},
        requested_at=occurred,
    )


def decode_remote_compute_result(raw: Mapping[str, Any]) -> RemoteComputeResultEvent:
    expected_envelope = {
        "event_id", "source_service", "destination_service", "event_type",
        "resource_tid", "owner_uid", "payer_tid", "rid", "source_aggregate_type",
        "source_aggregate_id", "producer_seq", "schema_version", "occurred_at",
        "correlation_id", "causation_event_id", "payload_sha256", "payload",
    }
    if set(raw) != expected_envelope or (
        raw.get("source_service") != "AGENT_EXECUTION"
        or raw.get("destination_service") != "PRODUCT_API"
        or raw.get("event_type") != "REMOTE_COMPUTE_COMMAND_RESULT"
        or raw.get("source_aggregate_type") != "remote_compute_resource"
        or raw.get("schema_version") != "2.0.0"
        or raw.get("payer_tid") is not None
    ):
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    payload = raw.get("payload")
    expected_payload = {
        "command_id", "resource_id", "action", "command_generation",
        "expected_resource_version", "rid", "binding_id", "lease_generation",
        "target_resource_id", "status", "error_code", "effect_result",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_payload:
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    encoded = json.dumps(
        dict(payload), allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != raw.get("payload_sha256"):
        raise RemoteComputeCommandError("EVENT_HASH_MISMATCH")

    def typed(value: Any, prefix: str) -> UUID:
        try:
            return parse_typed_id(value, expected_prefix=prefix)
        except InvalidTypedId as exc:
            raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED") from exc

    try:
        event_id = typed(raw["event_id"], "evt")
        command_id = typed(payload["command_id"], "op")
        resource_id = typed(payload["resource_id"], "rcres")
        resource_tid = typed(raw["resource_tid"], "tid")
        owner_uid = typed(raw["owner_uid"], "uid")
        source_id = typed(raw["source_aggregate_id"], "rcres")
        correlation = typed(raw["correlation_id"], "op")
        rid = None if payload["rid"] is None else typed(payload["rid"], "rid")
        envelope_rid = None if raw["rid"] is None else typed(raw["rid"], "rid")
        binding_id = (
            None if payload["binding_id"] is None else typed(payload["binding_id"], "rcbind")
        )
        target_id = (
            None
            if payload["target_resource_id"] is None
            else typed(payload["target_resource_id"], "rcres")
        )
        causation = typed(raw["causation_event_id"], "evt")
        action = RemoteComputeAction(str(payload["action"]))
        generation = int(payload["command_generation"])
        expected_version = int(payload["expected_resource_version"])
        producer_seq = int(raw["producer_seq"])
        lease = None if payload["lease_generation"] is None else int(payload["lease_generation"])
        occurred_at = datetime.fromisoformat(str(raw["occurred_at"]).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED") from exc
    if (
        source_id != resource_id
        or correlation != command_id
        or envelope_rid != rid
        or generation < 1
        or producer_seq < 1
        or expected_version < 1
        or causation == event_id
        or (lease is not None and lease < 1)
        or occurred_at.tzinfo is None
    ):
        raise RemoteComputeCommandError("EVENT_AUTHORITY_MISMATCH")
    status = str(payload["status"])
    error_code = payload["error_code"]
    effect_result = payload["effect_result"]
    if (
        status not in {"APPLIED", "FAILED"}
        or (status == "APPLIED" and error_code is not None)
        or not isinstance(effect_result, Mapping)
    ):
        raise RemoteComputeCommandError("EVENT_SCHEMA_UNSUPPORTED")
    return RemoteComputeResultEvent(
        event_id=event_id,
        command_id=command_id,
        resource_id=resource_id,
        resource_tid=resource_tid,
        owner_uid=owner_uid,
        action=action,
        command_generation=generation,
        expected_resource_version=expected_version,
        producer_seq=producer_seq,
        occurred_at=occurred_at,
        payload_sha256=bytes.fromhex(digest),
        payload=dict(payload),
        rid=rid,
        binding_id=binding_id,
        lease_generation=lease,
        target_resource_id=target_id,
        status=status,
        error_code=None if error_code is None else str(error_code),
        effect_result=dict(effect_result),
        causation_event_id=causation,
    )


class PsycopgRemoteComputeResultConsumer:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def consume(self, raw: Mapping[str, Any]) -> RemoteComputeEffectResult:
        event = decode_remote_compute_result(raw)
        receipt = self.repository.apply_result(event)
        return RemoteComputeEffectResult(
            command_id=receipt.command_id,
            replayed=receipt.replayed,
            effect_state=(
                "ALREADY_RECORDED"
                if receipt.replayed
                else receipt.delivery_state
            ),
            applied_producer_seq=(
                event.producer_seq - 1
                if receipt.delivery_state == "DEFERRED_GAP"
                else event.producer_seq
            ),
        )


class PsycopgRemoteComputeCommandConsumer:
    """Claim, apply, and receipt a provider effect with durable inbox fencing."""

    def __init__(
        self,
        connection_factory: Callable[[], Connection[Any]],
        provider: RemoteComputeProvider,
        secrets: FileRemoteComputeSecretResolver,
        *,
        runtime_role: str = "agent_ops_dispatcher",
        enforce_release_gate: bool = True,
        claim_holder: str | None = None,
        claim_ttl_seconds: int = 120,
        heartbeat_interval_seconds: int = 20,
        claim_token_factory: Callable[[], bytes] = lambda: token_bytes(32),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        provider_operations: DurableProviderOperationExecutor | None = None,
    ) -> None:
        if not 1 <= claim_ttl_seconds <= 3600:
            raise ValueError("claim_ttl_seconds must be between 1 and 3600")
        if (
            heartbeat_interval_seconds < 1
            or heartbeat_interval_seconds * 3 >= claim_ttl_seconds
        ):
            raise ValueError("heartbeat interval must be below one third of claim TTL")
        self.connection_factory = connection_factory
        self.provider = provider
        self.secrets = secrets
        self.runtime_role = runtime_role
        self.enforce_release_gate = enforce_release_gate
        self.claim_holder = claim_holder or f"{socket.gethostname()}:{os.getpid()}"
        self.claim_ttl_seconds = claim_ttl_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.claim_token_factory = claim_token_factory
        self.clock = clock
        self.provider_operations = provider_operations

    def consume(self, raw: Mapping[str, Any]) -> RemoteComputeEffectResult:
        command = decode_remote_compute_command(raw)
        event_id = parse_typed_id(str(raw["event_id"]), expected_prefix="evt")
        producer_seq = int(raw["producer_seq"])
        payload_sha256 = str(raw["payload_sha256"])
        claim = self._record_inbox(
            command, event_id=event_id, producer_seq=producer_seq,
            payload_sha256=payload_sha256, payload=raw["payload"],
        )
        if claim.state == "APPLIED":
            return RemoteComputeEffectResult(command.command_id, True, "ALREADY_RECORDED", producer_seq)
        if claim.state in {"PROCESSING", "RECEIVED", "RETRYABLE_FAILED"}:
            raise RemoteComputeCommandError("REMOTE_COMMAND_IN_PROGRESS", retryable=True)
        if claim.state == "AMBIGUOUS":
            raise RemoteComputeCommandError("REMOTE_PROVIDER_RESULT_UNKNOWN")
        if claim.state == "SECURITY_DLQ":
            raise RemoteComputeCommandError("SECURITY_DLQ")
        if claim.state == "DEFERRED_GAP":
            return RemoteComputeEffectResult(command.command_id, False, "DEFERRED_GAP", producer_seq)
        if (
            claim.state != "CLAIMED"
            or claim.generation is None
            or claim.token_hash is None
            or claim.acquired_at is None
            or claim.expires_at is None
        ):
            raise RemoteComputeCommandError("REMOTE_COMMAND_EFFECT_FENCED")
        try:
            secret = self.secrets.resolve(command.secret_ref, version=command.secret_version)
            target_secret = (
                None
                if not command.target_secret_ref
                else self.secrets.resolve(
                    command.target_secret_ref,
                    version=command.target_secret_version,
                )
            )
        except RemoteComputeSecretError as exc:
            self._mark_retryable(command, event_id, claim, exc.code)
            raise RemoteComputeCommandError(exc.code, retryable=True) from exc

        # The legacy adapter needs the inbox exposure fence before it calls the
        # Provider.  The durable adapter owns a stronger intent/effect/receipt
        # fence in provider_operations, so its inbox row deliberately remains
        # PRE_EFFECT until that operation has either produced a receipt or an
        # explicit unknown result.  A crash before operation intent can then
        # safely reclaim the inbox command instead of stranding a no-effect
        # delivery as AMBIGUOUS.
        durable_provider_operation = self.provider_operations is not None
        if not durable_provider_operation:
            self._mark_exposed(command, event_id, claim)
        heartbeat = self._heartbeat(command, event_id, claim)
        heartbeat.start()
        effect: dict[str, Any] | None = None
        provider_error: Exception | None = None
        try:
            if self.provider_operations is None:
                effect = dict(
                    self.provider.apply(
                        command,
                        secret=secret,
                        target_secret=target_secret,
                    )
                )
            else:
                outcome = self.provider_operations.execute_apply(
                    command,
                    command_event_id=event_id,
                    request=raw["payload"],
                    secret=secret,
                    target_secret=target_secret,
                )
                effect = dict(outcome.effect)
                if outcome.status == "FAILED":
                    provider_error = RemoteComputeCommandError(
                        outcome.error_code or "REMOTE_PROVIDER_REJECTED"
                    )
        except Exception as exc:
            provider_error = exc
        finally:
            heartbeat_stopped = heartbeat.stop()
        if heartbeat.error is not None or not heartbeat_stopped:
            if durable_provider_operation:
                try:
                    self._mark_exposed(command, event_id, claim)
                except RemoteComputeCommandError:
                    pass
            try:
                self._mark_ambiguous(
                    command,
                    event_id,
                    claim,
                    "REMOTE_CLAIM_HEARTBEAT_LOST",
                )
            except RemoteComputeCommandError:
                # A replacement generation may already have fenced this
                # worker. Either way, the Provider result is not safe to use.
                pass
            raise RemoteComputeCommandError("REMOTE_PROVIDER_RESULT_UNKNOWN")
        if isinstance(provider_error, ProviderOperationInProgress):
            # Retrying the inbox command is safe: execute_apply will only
            # attach to/replay the stable provider operation and cannot emit a
            # second Provider request.
            self._mark_retryable(command, event_id, claim, provider_error.code)
            raise RemoteComputeCommandError(
                "REMOTE_COMMAND_IN_PROGRESS", retryable=True
            ) from provider_error
        if durable_provider_operation:
            self._mark_exposed(command, event_id, claim)
        if isinstance(provider_error, ProviderOperationResultUnknown):
            self._mark_ambiguous(
                command,
                event_id,
                claim,
                provider_error.code,
            )
            raise RemoteComputeCommandError(
                "REMOTE_PROVIDER_RESULT_UNKNOWN"
            ) from provider_error
        if isinstance(provider_error, RemoteComputeCommandError):
            if provider_error.retryable:
                self._mark_ambiguous(command, event_id, claim, provider_error.code)
                raise RemoteComputeCommandError(
                    "REMOTE_PROVIDER_RESULT_UNKNOWN"
                ) from provider_error
            self._mark_completed(
                command,
                event_id,
                claim,
                status="FAILED",
                error_code=provider_error.code,
                effect={},
            )
            return RemoteComputeEffectResult(command.command_id, False, "APPLIED", producer_seq)
        if provider_error is not None:
            self._mark_ambiguous(
                command,
                event_id,
                claim,
                "REMOTE_PROVIDER_UNAVAILABLE",
            )
            raise RemoteComputeCommandError(
                "REMOTE_PROVIDER_RESULT_UNKNOWN"
            ) from provider_error
        assert effect is not None
        self._mark_completed(
            command,
            event_id,
            claim,
            status="APPLIED",
            error_code=None,
            effect=effect,
        )
        return RemoteComputeEffectResult(command.command_id, False, "APPLIED", producer_seq)

    def _record_inbox(
        self,
        command: RemoteComputeCommand,
        *,
        event_id: UUID,
        producer_seq: int,
        payload_sha256: str,
        payload: Any,
    ) -> _InboxClaim:
        claim_token = self.claim_token_factory()
        if not isinstance(claim_token, bytes) or not claim_token:
            raise ValueError("claim token factory must return non-empty bytes")
        token_hash = hashlib.sha256(claim_token).hexdigest()
        with self._transaction(command) as connection:
            row = connection.execute(
                """
                SELECT effect_status, payload_sha256 FROM agent_ops.delivery_inbox
                 WHERE source_service = 'PRODUCT_API' AND event_id = %s FOR UPDATE
                """,
                (event_id,),
            ).fetchone()
            if row is not None:
                if row["payload_sha256"] != payload_sha256:
                    raise RemoteComputeCommandError("SECURITY_DLQ")
                prior_status = str(row["effect_status"])
                if prior_status in {"APPLIED", "REJECTED_TERMINAL"}:
                    return _InboxClaim("APPLIED")
                if prior_status == "PROCESSING":
                    ambiguous = connection.execute(
                        """
                        UPDATE agent_ops.delivery_inbox
                           SET effect_status = 'AMBIGUOUS', attempts = attempts + 1,
                               next_attempt_at = now(),
                               claim_holder = NULL, claim_token_hash = NULL,
                               claim_expires_at = NULL,
                               last_error_code = 'REMOTE_PROVIDER_RESULT_UNKNOWN'
                         WHERE source_service = 'PRODUCT_API' AND event_id = %s
                           AND effect_status = 'PROCESSING'
                           AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                           AND claim_expires_at <= now()
                        """,
                        (event_id,),
                    ).rowcount
                    if ambiguous == 1:
                        return _InboxClaim("AMBIGUOUS")
                if prior_status in {"RECEIVED", "RETRYABLE_FAILED", "PROCESSING"}:
                    claimed = connection.execute(
                        """
                        UPDATE agent_ops.delivery_inbox
                           SET effect_status = 'PROCESSING', next_attempt_at = now(),
                               effect_phase = 'PRE_EFFECT',
                               claim_holder = %(claim_holder)s,
                               claim_token_hash = %(claim_token_hash)s,
                               claim_generation = claim_generation + 1,
                               claim_expires_at = now() +
                                   (%(claim_ttl_seconds)s * interval '1 second')
                         WHERE source_service = 'PRODUCT_API'
                           AND event_id = %(event_id)s
                           AND (
                               effect_status = 'RECEIVED'
                               OR (effect_status = 'RETRYABLE_FAILED'
                                   AND next_attempt_at <= now())
                               OR (effect_status = 'PROCESSING'
                                   AND effect_phase = 'PRE_EFFECT'
                                   AND claim_expires_at <= now())
                           )
                        RETURNING claim_generation, claim_expires_at,
                                  now() AS claimed_at
                        """,
                        {
                            "event_id": event_id,
                            "claim_holder": self.claim_holder,
                            "claim_token_hash": token_hash,
                            "claim_ttl_seconds": self.claim_ttl_seconds,
                        },
                    ).fetchone()
                    if claimed is not None:
                        return _InboxClaim(
                            "CLAIMED",
                            int(claimed["claim_generation"]),
                            token_hash,
                            claim_token,
                            claimed["claimed_at"],
                            claimed["claim_expires_at"],
                        )
                return _InboxClaim(prior_status)
            previous = connection.execute(
                """
                SELECT coalesce(max(producer_seq), 0) AS applied_seq
                  FROM agent_ops.delivery_inbox
                 WHERE source_service = 'PRODUCT_API'
                   AND destination_service = 'AGENT_EXECUTION'
                   AND source_aggregate_type = 'remote_compute_resource'
                   AND source_aggregate_id = %s
                   AND effect_status IN ('APPLIED', 'REJECTED_TERMINAL')
                """,
                (command.resource_id,),
            ).fetchone()
            prior_seq = int(previous["applied_seq"])
            status = "PROCESSING" if producer_seq == prior_seq + 1 else "DEFERRED_GAP"
            connection.execute(
                """
                INSERT INTO agent_ops.delivery_inbox (
                    source_service, event_id, destination_service, event_type,
                    source_aggregate_type, source_aggregate_id, producer_seq,
                    prior_applied_seq, resource_tid, owner_uid, payer_tid, rid,
                    schema_version, occurred_at, correlation_id,
                    causation_event_id, payload_jsonb, payload_sha256,
                    effect_type, effect_idempotency_key, effect_status,
                    processed_at
                ) VALUES (
                    'PRODUCT_API', %(event_id)s, 'AGENT_EXECUTION',
                    'REMOTE_COMPUTE_COMMAND_REQUESTED', 'remote_compute_resource',
                    %(resource_id)s, %(producer_seq)s, %(prior_seq)s,
                    %(resource_tid)s, %(owner_uid)s, NULL, %(rid)s, '2.0.0',
                    %(occurred_at)s, %(command_id)s, NULL, %(payload)s,
                    %(payload_sha256)s, 'APPLY_REMOTE_COMPUTE_COMMAND',
                    %(effect_key)s, 'RECEIVED', NULL
                )
                """,
                {
                    "event_id": event_id,
                    "resource_id": command.resource_id,
                    "producer_seq": producer_seq,
                    "prior_seq": prior_seq,
                    "resource_tid": command.resource_tid,
                    "owner_uid": command.owner_uid,
                    "rid": command.rid,
                    "occurred_at": command.requested_at,
                    "command_id": command.command_id,
                    "payload": Jsonb(dict(payload)),
                    "payload_sha256": payload_sha256,
                    "effect_key": f"remote-compute:{command.command_id}",
                },
            )
            if status == "DEFERRED_GAP":
                connection.execute(
                    """
                    UPDATE agent_ops.delivery_inbox
                       SET effect_status = 'DEFERRED_GAP'
                     WHERE source_service = 'PRODUCT_API' AND event_id = %s
                       AND effect_status = 'RECEIVED'
                    """,
                    (event_id,),
                )
                return _InboxClaim(status)
            claimed = connection.execute(
                """
                UPDATE agent_ops.delivery_inbox
                   SET effect_status = 'PROCESSING',
                       effect_phase = 'PRE_EFFECT',
                       claim_holder = %(claim_holder)s,
                       claim_token_hash = %(claim_token_hash)s,
                       claim_generation = claim_generation + 1,
                       claim_expires_at = now() +
                           (%(claim_ttl_seconds)s * interval '1 second')
                 WHERE source_service = 'PRODUCT_API' AND event_id = %(event_id)s
                   AND effect_status = 'RECEIVED'
                RETURNING claim_generation, claim_expires_at,
                          now() AS claimed_at
                """,
                {
                    "event_id": event_id,
                    "claim_holder": self.claim_holder,
                    "claim_token_hash": token_hash,
                    "claim_ttl_seconds": self.claim_ttl_seconds,
                },
            ).fetchone()
            if claimed is None:
                raise RemoteComputeCommandError("REMOTE_COMMAND_EFFECT_FENCED")
            return _InboxClaim(
                "CLAIMED",
                int(claimed["claim_generation"]),
                token_hash,
                claim_token,
                claimed["claimed_at"],
                claimed["claim_expires_at"],
            )

    def _heartbeat(
        self,
        command: RemoteComputeCommand,
        event_id: UUID,
        claim: _InboxClaim,
    ) -> HeartbeatRunner:
        assert claim.generation is not None
        assert claim.token_hash is not None
        assert claim.acquired_at is not None
        assert claim.expires_at is not None
        lease = Lease(
            holder=self.claim_holder,
            generation=claim.generation,
            token_hash=claim.token_hash,
            acquired_at=claim.acquired_at,
            expires_at=claim.expires_at,
            operation_id=event_id,
        )
        return HeartbeatRunner(
            lease,
            lambda current: self._renew_claim(command, event_id, current),
            interval=timedelta(seconds=self.heartbeat_interval_seconds),
            clock=self.clock,
            thread_name=f"remote-compute-heartbeat-{event_id}",
        )

    def _renew_claim(
        self,
        command: RemoteComputeCommand,
        event_id: UUID,
        lease: Lease,
    ) -> Lease:
        with self._transaction(command) as connection:
            row = connection.execute(
                """
                UPDATE agent_ops.delivery_inbox
                   SET claim_expires_at = now() +
                       (%(claim_ttl_seconds)s * interval '1 second')
                 WHERE source_service = 'PRODUCT_API'
                   AND event_id = %(event_id)s
                   AND effect_status = 'PROCESSING'
                   AND effect_phase IN ('PRE_EFFECT','EFFECT_MAY_HAVE_OCCURRED')
                   AND claim_holder = %(claim_holder)s
                   AND claim_generation = %(claim_generation)s
                   AND claim_token_hash = %(claim_token_hash)s
                   AND claim_expires_at > now()
                RETURNING claim_expires_at
                """,
                {
                    "event_id": event_id,
                    "claim_holder": lease.holder,
                    "claim_generation": lease.generation,
                    "claim_token_hash": lease.token_hash,
                    "claim_ttl_seconds": self.claim_ttl_seconds,
                },
            ).fetchone()
        if row is None:
            raise RemoteComputeCommandError("REMOTE_COMMAND_EFFECT_FENCED")
        return lease.renewed(expires_at=row["claim_expires_at"])

    def _mark_exposed(
        self,
        command: RemoteComputeCommand,
        event_id: UUID,
        claim: _InboxClaim,
    ) -> None:
        with self._transaction(command) as connection:
            updated = connection.execute(
                """
                UPDATE agent_ops.delivery_inbox
                   SET effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                 WHERE source_service = 'PRODUCT_API' AND event_id = %(event_id)s
                   AND effect_status = 'PROCESSING'
                   AND effect_phase = 'PRE_EFFECT'
                   AND claim_generation = %(claim_generation)s
                   AND claim_token_hash = %(claim_token_hash)s
                   AND claim_expires_at > now()
                """,
                {
                    "event_id": event_id,
                    "claim_generation": claim.generation,
                    "claim_token_hash": claim.token_hash,
                },
            ).rowcount
            if updated != 1:
                raise RemoteComputeCommandError("REMOTE_COMMAND_EFFECT_FENCED")

    def _mark_retryable(
        self,
        command: RemoteComputeCommand,
        event_id: UUID,
        claim: _InboxClaim,
        code: str,
    ) -> None:
        with self._transaction(command) as connection:
            updated = connection.execute(
                """
                UPDATE agent_ops.delivery_inbox
                   SET effect_status = 'RETRYABLE_FAILED', attempts = attempts + 1,
                       next_attempt_at = now() + interval '5 seconds',
                       claim_holder = NULL, claim_token_hash = NULL,
                       claim_expires_at = NULL, last_error_code = %(code)s
                 WHERE source_service = 'PRODUCT_API' AND event_id = %(event_id)s
                   AND effect_status = 'PROCESSING'
                   AND effect_phase = 'PRE_EFFECT'
                   AND claim_generation = %(claim_generation)s
                   AND claim_token_hash = %(claim_token_hash)s
                """,
                {
                    "code": code,
                    "event_id": event_id,
                    "claim_generation": claim.generation,
                    "claim_token_hash": claim.token_hash,
                },
            ).rowcount
            if updated != 1:
                raise RemoteComputeCommandError("REMOTE_COMMAND_EFFECT_FENCED")

    def _mark_ambiguous(
        self,
        command: RemoteComputeCommand,
        event_id: UUID,
        claim: _InboxClaim,
        code: str,
    ) -> None:
        with self._transaction(command) as connection:
            updated = connection.execute(
                """
                UPDATE agent_ops.delivery_inbox
                   SET effect_status = 'AMBIGUOUS', attempts = attempts + 1,
                       next_attempt_at = now(), claim_holder = NULL,
                       claim_token_hash = NULL, claim_expires_at = NULL,
                       last_error_code = %(code)s
                 WHERE source_service = 'PRODUCT_API' AND event_id = %(event_id)s
                   AND effect_status = 'PROCESSING'
                   AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                   AND claim_generation = %(claim_generation)s
                   AND claim_token_hash = %(claim_token_hash)s
                """,
                {
                    "code": code,
                    "event_id": event_id,
                    "claim_generation": claim.generation,
                    "claim_token_hash": claim.token_hash,
                },
            ).rowcount
            if updated != 1:
                raise RemoteComputeCommandError("REMOTE_COMMAND_EFFECT_FENCED")

    def _mark_completed(
        self,
        command: RemoteComputeCommand,
        event_id: UUID,
        claim: _InboxClaim,
        *,
        status: str,
        error_code: str | None,
        effect: Mapping[str, Any],
    ) -> None:
        # The canonical inbound payload and its hash remain immutable.  The
        # provider result is emitted as a separate Agent event by the runtime.
        if status not in {"APPLIED", "FAILED"}:
            raise ValueError("remote compute completion status is invalid")
        result = dict(effect)
        result_bytes = json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(result_bytes) > 16 * 1024:
            raise RemoteComputeCommandError("REMOTE_EFFECT_RESULT_TOO_LARGE")
        result_event_id = uuid7()
        result_payload = {
            "command_id": format_typed_id("op", command.command_id),
            "resource_id": format_typed_id("rcres", command.resource_id),
            "action": command.action.value,
            "command_generation": command.command_generation,
            "expected_resource_version": command.expected_resource_version,
            "rid": None if command.rid is None else format_typed_id("rid", command.rid),
            "binding_id": (
                None
                if command.binding_id is None
                else format_typed_id("rcbind", command.binding_id)
            ),
            "lease_generation": command.lease_generation,
            "target_resource_id": (
                None
                if command.target_resource_id is None
                else format_typed_id("rcres", command.target_resource_id)
            ),
            "status": status,
            "error_code": error_code,
            "effect_result": result,
        }
        payload_bytes = json.dumps(
            result_payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with self._transaction(command) as connection:
            updated = connection.execute(
                """
                UPDATE agent_ops.delivery_inbox
                   SET effect_status = %(inbox_status)s, attempts = attempts + 1,
                       processed_at = now(), last_error_code = %(error_code)s,
                       claim_holder = NULL, claim_token_hash = NULL,
                       claim_expires_at = NULL
                 WHERE source_service = 'PRODUCT_API' AND event_id = %(event_id)s
                   AND effect_status = 'PROCESSING'
                   AND effect_phase = 'EFFECT_MAY_HAVE_OCCURRED'
                   AND claim_generation = %(claim_generation)s
                   AND claim_token_hash = %(claim_token_hash)s
                """,
                {
                    "inbox_status": (
                        "APPLIED" if status == "APPLIED" else "REJECTED_TERMINAL"
                    ),
                    "error_code": error_code,
                    "event_id": event_id,
                    "claim_generation": claim.generation,
                    "claim_token_hash": claim.token_hash,
                },
            ).rowcount
            if updated != 1:
                raise RemoteComputeCommandError("REMOTE_COMMAND_EFFECT_FENCED")
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
                    %(command_id)s, %(command_event_id)s, %(payload)s,
                    %(payload_sha256)s, 'PENDING'
                )
                ON CONFLICT (event_id) DO NOTHING
                """,
                {
                    "event_id": result_event_id,
                    "resource_id": command.resource_id,
                    "producer_seq": command.producer_seq,
                    "resource_tid": command.resource_tid,
                    "owner_uid": command.owner_uid,
                    "rid": command.rid,
                    "command_id": command.command_id,
                    "command_event_id": event_id,
                    "payload": Jsonb(result_payload),
                    "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                },
            )

    @contextmanager
    def _transaction(self, command: RemoteComputeCommand) -> Iterator[Connection[Any]]:
        with self.connection_factory() as connection, connection.transaction():
            activate_runtime_role(connection, expected_role=self.runtime_role)
            if self.enforce_release_gate:
                require_business_schema(connection, AGENT_DATABASE)
            connection.row_factory = dict_row
            connection.execute("SELECT set_config('app.resource_tid', %s, true)", (str(command.resource_tid),))
            connection.execute("SELECT set_config('app.owner_uid', %s, true)", (str(command.owner_uid),))
            yield connection


__all__ = [
    "PsycopgRemoteComputeCommandConsumer",
    "PsycopgRemoteComputeResultConsumer",
    "RemoteComputeCommandError",
    "RemoteComputeEffectResult",
    "RemoteComputeProvider",
    "decode_remote_compute_command",
    "decode_remote_compute_result",
]
