"""Canonical cross-service RUN_EXECUTION authorization claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import re
from typing import Any
from uuid import UUID

from apps.v2.model_fallback import ChatFallbackPolicy

from apps.common.v2_ids import InvalidTypedId, format_typed_id, parse_typed_id


RUN_EXECUTION_SCOPE = "RUN_EXECUTION"
CONTINUE_AND_SETTLE = "CONTINUE_AND_SETTLE"
CONVERSATION_WORKLOAD = "CONVERSATION"
# Workload identity remains part of the signed authorization and database
# join key.  Product V2 currently admits conversation runs only; the former
# Knowledge workload was retired and is deliberately not represented here.
_WORKLOAD_KINDS = frozenset({CONVERSATION_WORKLOAD})
_RESOURCE_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}:.+$")
_CLAIM_FIELDS = frozenset(
    {
        "authorization_id",
        "authorization_version",
        "authorization_scope",
        "rid",
        "billing_account_id",
        "resource_tid",
        "owner_uid",
        "payer_tid",
        "payer_membership_id",
        "payer_membership_version",
        "actor_uid",
        "sid",
        "workload_kind",
        "workload_id",
        "pricing_snapshot_refs",
        "pricing_snapshot_set_sha256",
        "input_manifest_sha256",
        "context_manifest_sha256",
        "workflow_build_id",
        "checkpoint_schema_version",
        "environment_fingerprint",
        "max_estimated_credits",
        "estimator_version",
        "overage_policy",
        "recovery_epoch",
        "control_generation",
        "valid_from",
        "expires_at",
    }
)
_FALLBACK_CLAIM_FIELDS = _CLAIM_FIELDS | {"chat_fallback_policy"}
_LEGACY_CLAIM_FIELDS = _CLAIM_FIELDS - {"workload_kind", "workload_id"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class PricingSnapshotRef:
    pricing_snapshot_id: UUID
    resource_kind: str
    resource_key: str

    def __post_init__(self) -> None:
        if self.resource_kind not in {"LLM", "MEDIA", "STORAGE"}:
            raise ValueError("pricing snapshot resource_kind is unsupported")
        if (
            _RESOURCE_KEY.fullmatch(self.resource_key) is None
            or len(self.resource_key.encode("utf-8")) > 640
        ):
            raise ValueError("pricing snapshot resource_key is not canonical")

    def as_wire(self) -> dict[str, Any]:
        return {
            "pricing_snapshot_id": format_typed_id("psnap", self.pricing_snapshot_id),
            "resource_kind": self.resource_kind,
            "resource_key": self.resource_key,
        }

    @classmethod
    def from_wire(cls, value: Any) -> PricingSnapshotRef:
        if not isinstance(value, Mapping) or set(value) != {
            "pricing_snapshot_id",
            "resource_kind",
            "resource_key",
        }:
            raise ValueError("pricing snapshot ref must contain exactly id, kind and key")
        return cls(
            pricing_snapshot_id=_typed(value["pricing_snapshot_id"], "psnap", "pricing_snapshot_id"),
            resource_kind=_string(value["resource_kind"], "resource_kind", max_bytes=32),
            resource_key=_string(value["resource_key"], "resource_key", max_bytes=640),
        )


def canonical_pricing_refs(
    values: Sequence[PricingSnapshotRef],
) -> tuple[PricingSnapshotRef, ...]:
    refs = tuple(sorted(values, key=lambda item: item.pricing_snapshot_id.bytes))
    if not refs or len(refs) > 32:
        raise ValueError("authorization requires 1 to 32 pricing snapshot refs")
    if len({item.pricing_snapshot_id for item in refs}) != len(refs):
        raise ValueError("pricing snapshot ids must be unique")
    if len({(item.resource_kind, item.resource_key) for item in refs}) != len(refs):
        raise ValueError("one authorization cannot bind multiple snapshots to one resource key")
    return refs


def pricing_snapshot_set_sha256(values: Sequence[PricingSnapshotRef]) -> bytes:
    refs = canonical_pricing_refs(values)
    return hashlib.sha256(canonical_json_bytes([item.as_wire() for item in refs])).digest()


@dataclass(frozen=True, slots=True)
class RunAuthorizationClaims:
    authorization_id: UUID
    authorization_version: int
    rid: UUID
    billing_account_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    payer_tid: UUID
    payer_membership_id: UUID | None
    payer_membership_version: int | None
    actor_uid: UUID
    sid: UUID
    pricing_snapshot_refs: tuple[PricingSnapshotRef, ...]
    input_manifest_sha256: str
    context_manifest_sha256: str
    workflow_build_id: str
    checkpoint_schema_version: str
    environment_fingerprint: str
    max_estimated_credits: int
    estimator_version: str
    recovery_epoch: int
    control_generation: int
    valid_from: datetime
    expires_at: datetime
    authorization_scope: str = RUN_EXECUTION_SCOPE
    overage_policy: str = CONTINUE_AND_SETTLE
    workload_kind: str = CONVERSATION_WORKLOAD
    workload_id: UUID | None = None
    chat_fallback_policy: ChatFallbackPolicy | None = None
    wire_contract_version: str = field(default="2.1.0", repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pricing_snapshot_refs",
            canonical_pricing_refs(self.pricing_snapshot_refs),
        )
        if self.authorization_version < 1:
            raise ValueError("authorization_version must be positive")
        if self.authorization_scope != RUN_EXECUTION_SCOPE:
            raise ValueError("authorization_scope must be RUN_EXECUTION")
        if self.workload_kind not in _WORKLOAD_KINDS:
            raise ValueError("workload_kind is unsupported")
        if self.workload_id is None:
            object.__setattr__(self, "workload_id", self.rid)
        if self.workload_kind == CONVERSATION_WORKLOAD and self.workload_id != self.rid:
            raise ValueError("conversation workload_id must equal rid")
        if self.wire_contract_version not in {"2.0.0", "2.1.0", "2.2.0"}:
            raise ValueError("authorization wire contract version is unsupported")
        if self.wire_contract_version == "2.0.0" and (
            self.workload_kind != CONVERSATION_WORKLOAD or self.workload_id != self.rid
        ):
            raise ValueError("legacy authorization claims can only describe a conversation run")
        if self.chat_fallback_policy is not None:
            if self.wire_contract_version != "2.2.0" or self.workload_kind != CONVERSATION_WORKLOAD:
                raise ValueError("chat fallback requires conversation authorization v2.2")
            if not isinstance(self.chat_fallback_policy, ChatFallbackPolicy):
                raise ValueError("invalid chat fallback policy")
            for route in self.chat_fallback_policy.routes:
                self.pricing_for(resource_kind="LLM", resource_key=route.resource_key)
        elif self.wire_contract_version == "2.2.0":
            raise ValueError("authorization v2.2 requires an explicit fallback policy")
        if self.overage_policy != CONTINUE_AND_SETTLE:
            raise ValueError("authorization overage policy is unsupported")
        if self.max_estimated_credits <= 0:
            raise ValueError("max_estimated_credits must be positive")
        if self.recovery_epoch < 0 or self.control_generation < 0:
            raise ValueError("authorization generations cannot be negative")
        if (self.payer_membership_id is None) != (self.payer_membership_version is None):
            raise ValueError("payer membership identity and version must be present together")
        if self.payer_membership_version is not None and self.payer_membership_version < 1:
            raise ValueError("payer_membership_version must be positive")
        for name in (
            "input_manifest_sha256",
            "context_manifest_sha256",
            "environment_fingerprint",
        ):
            _digest(getattr(self, name), name)
        for name, maximum in (
            ("workflow_build_id", 512),
            ("checkpoint_schema_version", 160),
            ("estimator_version", 512),
        ):
            _string(getattr(self, name), name, max_bytes=maximum)
        _aware(self.valid_from, "valid_from")
        _aware(self.expires_at, "expires_at")
        if self.expires_at <= self.valid_from:
            raise ValueError("authorization expiry must be after valid_from")

    @property
    def pricing_snapshot_set_sha256(self) -> bytes:
        return pricing_snapshot_set_sha256(self.pricing_snapshot_refs)

    def pricing_for(self, *, resource_kind: str, resource_key: str) -> PricingSnapshotRef:
        matches = [
            item
            for item in self.pricing_snapshot_refs
            if (item.resource_kind, item.resource_key) == (resource_kind, resource_key)
        ]
        if len(matches) != 1:
            raise ValueError("authorization has no unique pricing snapshot for the requested resource")
        return matches[0]

    def as_wire(self) -> dict[str, Any]:
        document = {
            "authorization_id": format_typed_id("authz", self.authorization_id),
            "authorization_version": self.authorization_version,
            "authorization_scope": self.authorization_scope,
            "rid": format_typed_id("rid", self.rid),
            "billing_account_id": format_typed_id("bacc", self.billing_account_id),
            "resource_tid": format_typed_id("tid", self.resource_tid),
            "owner_uid": format_typed_id("uid", self.owner_uid),
            "payer_tid": format_typed_id("tid", self.payer_tid),
            "payer_membership_id": (
                None
                if self.payer_membership_id is None
                else format_typed_id("mbr", self.payer_membership_id)
            ),
            "payer_membership_version": self.payer_membership_version,
            "actor_uid": format_typed_id("uid", self.actor_uid),
            "sid": format_typed_id("sid", self.sid),
            "workload_kind": self.workload_kind,
            "workload_id": format_typed_id("op", self.workload_id),
            "pricing_snapshot_refs": [item.as_wire() for item in self.pricing_snapshot_refs],
            "pricing_snapshot_set_sha256": self.pricing_snapshot_set_sha256.hex(),
            "input_manifest_sha256": self.input_manifest_sha256,
            "context_manifest_sha256": self.context_manifest_sha256,
            "workflow_build_id": self.workflow_build_id,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "environment_fingerprint": self.environment_fingerprint,
            "max_estimated_credits": self.max_estimated_credits,
            "estimator_version": self.estimator_version,
            "overage_policy": self.overage_policy,
            "recovery_epoch": self.recovery_epoch,
            "control_generation": self.control_generation,
            "valid_from": self.valid_from.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }
        if self.wire_contract_version == "2.0.0":
            document.pop("workload_kind")
            document.pop("workload_id")
        if self.chat_fallback_policy is not None:
            document["chat_fallback_policy"] = self.chat_fallback_policy.as_wire()
        return document

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_wire())

    @classmethod
    def from_wire(cls, value: Any) -> RunAuthorizationClaims:
        if not isinstance(value, Mapping) or set(value) not in {
            _CLAIM_FIELDS,
            _LEGACY_CLAIM_FIELDS,
            _FALLBACK_CLAIM_FIELDS,
        }:
            raise ValueError("authorization claims contain missing or unknown fields")
        legacy = set(value) == _LEGACY_CLAIM_FIELDS
        refs_raw = value["pricing_snapshot_refs"]
        if not isinstance(refs_raw, list):
            raise ValueError("pricing_snapshot_refs must be an array")
        claims = cls(
            authorization_id=_typed(value["authorization_id"], "authz", "authorization_id"),
            authorization_version=_integer(value["authorization_version"], "authorization_version", 1),
            authorization_scope=_string(value["authorization_scope"], "authorization_scope", max_bytes=64),
            rid=_typed(value["rid"], "rid", "rid"),
            billing_account_id=_typed(value["billing_account_id"], "bacc", "billing_account_id"),
            resource_tid=_typed(value["resource_tid"], "tid", "resource_tid"),
            owner_uid=_typed(value["owner_uid"], "uid", "owner_uid"),
            payer_tid=_typed(value["payer_tid"], "tid", "payer_tid"),
            payer_membership_id=(
                None
                if value["payer_membership_id"] is None
                else _typed(value["payer_membership_id"], "mbr", "payer_membership_id")
            ),
            payer_membership_version=(
                None
                if value["payer_membership_version"] is None
                else _integer(value["payer_membership_version"], "payer_membership_version", 1)
            ),
            actor_uid=_typed(value["actor_uid"], "uid", "actor_uid"),
            sid=_typed(value["sid"], "sid", "sid"),
            workload_kind=(
                CONVERSATION_WORKLOAD
                if legacy
                else _string(value["workload_kind"], "workload_kind", max_bytes=64)
            ),
            workload_id=(
                _typed(value["rid"], "rid", "rid")
                if legacy
                else _typed(value["workload_id"], "op", "workload_id")
            ),
            wire_contract_version="2.0.0" if legacy else ("2.2.0" if "chat_fallback_policy" in value else "2.1.0"),
            chat_fallback_policy=(ChatFallbackPolicy.from_wire(value["chat_fallback_policy"])
                                  if "chat_fallback_policy" in value else None),
            pricing_snapshot_refs=tuple(PricingSnapshotRef.from_wire(item) for item in refs_raw),
            input_manifest_sha256=_digest(value["input_manifest_sha256"], "input_manifest_sha256"),
            context_manifest_sha256=_digest(
                value["context_manifest_sha256"], "context_manifest_sha256"
            ),
            workflow_build_id=_string(value["workflow_build_id"], "workflow_build_id", max_bytes=512),
            checkpoint_schema_version=_string(
                value["checkpoint_schema_version"], "checkpoint_schema_version", max_bytes=160
            ),
            environment_fingerprint=_digest(
                value["environment_fingerprint"], "environment_fingerprint"
            ),
            max_estimated_credits=_integer(
                value["max_estimated_credits"], "max_estimated_credits", 1
            ),
            estimator_version=_string(value["estimator_version"], "estimator_version", max_bytes=512),
            overage_policy=_string(value["overage_policy"], "overage_policy", max_bytes=64),
            recovery_epoch=_integer(value["recovery_epoch"], "recovery_epoch", 0),
            control_generation=_integer(value["control_generation"], "control_generation", 0),
            valid_from=_timestamp(value["valid_from"], "valid_from"),
            expires_at=_timestamp(value["expires_at"], "expires_at"),
        )
        supplied_set_hash = _digest(
            value["pricing_snapshot_set_sha256"], "pricing_snapshot_set_sha256"
        )
        if supplied_set_hash != claims.pricing_snapshot_set_sha256.hex():
            raise ValueError("pricing_snapshot_set_sha256 does not match signed refs")
        if dict(value) != claims.as_wire():
            raise ValueError("authorization claims are not in canonical wire form")
        return claims


def _typed(value: Any, prefix: str, name: str) -> UUID:
    try:
        return parse_typed_id(value, expected_prefix=prefix)
    except InvalidTypedId as exc:
        raise ValueError(f"{name} is not a canonical {prefix}_ id") from exc


def _string(value: Any, name: str, *, max_bytes: int) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} must contain 1 to {max_bytes} UTF-8 bytes")
    return value


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    _aware(parsed, name)
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must use canonical datetime.isoformat() form")
    return parsed


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


__all__ = [
    "CONTINUE_AND_SETTLE",
    "CONVERSATION_WORKLOAD",
    "RUN_EXECUTION_SCOPE",
    "PricingSnapshotRef",
    "RunAuthorizationClaims",
    "canonical_json_bytes",
    "canonical_pricing_refs",
    "pricing_snapshot_set_sha256",
]
