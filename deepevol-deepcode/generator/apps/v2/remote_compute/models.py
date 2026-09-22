"""Transport-neutral remote-compute value objects.

These objects are intentionally smaller than the legacy ORM model.  In
particular, no value object has an ``access_secret`` field: credentials are
resolved in Agent memory from a deployment-owned secret reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping
from uuid import UUID

from apps.common.v2_ids import InvalidTypedId, format_typed_id, parse_typed_id


class RemoteComputeAction(StrEnum):
    POWER_ON = "POWER_ON"
    POWER_OFF = "POWER_OFF"
    RELEASE = "RELEASE"
    CLEANUP = "CLEANUP"
    MIGRATE = "MIGRATE"
    APPLY_QUOTA = "APPLY_QUOTA"


class RemoteComputeCommandState(StrEnum):
    PENDING = "PENDING"
    APPLIED = "APPLIED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class RemoteComputeResource:
    resource_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    name: str
    provider: str
    status: str
    access_host: str
    access_port: int
    access_username: str
    secret_ref: str
    secret_version: str
    remote_root: str
    accelerator_type: str = ""
    accelerator_count: int = 0
    vram_gb: int = 0
    billing_mode: str = "reserved"
    hourly_price_credits: int = 0
    allocation_note: str = ""
    region: str = ""
    instance_type: str = ""
    cpu_cores: int = 0
    memory_gb: int = 0
    storage_gb: int = 0
    external_resource_id: str = ""
    powered_off_at: datetime | None = None
    command_generation: int = 0
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RemoteComputeAdminRecord:
    """Redacted cross-tenant resource projection for the existing admin UI."""

    resource: RemoteComputeResource
    owner_display_name: str | None = None
    owner_account_type: str | None = None
    owner_phone: str | None = None
    owner_email: str | None = None


@dataclass(frozen=True, slots=True)
class RemoteWorkspaceBinding:
    binding_id: UUID
    resource_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    sid: UUID
    rid: UUID
    lease_generation: int
    status: str
    remote_root: str
    secret_ref: str
    secret_version: str
    created_at: datetime | None = None
    expires_at: datetime | None = None
    payer_tid: UUID | None = None
    billing_account_id: UUID | None = None
    hourly_price_credits: int = 0
    estimated_credits: int = 0
    actual_seconds: int = 0
    actual_credits: int = 0
    charge_id: UUID | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    resource_policy: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class RemoteComputePricingContext:
    """Non-secret provider facts frozen when Product allocates a resource."""

    provider: str
    billing_mode: str
    cpu_cores: int

    def __post_init__(self) -> None:
        if not self.provider or len(self.provider.encode("utf-8")) > 128:
            raise ValueError("remote compute provider is invalid")
        if self.billing_mode not in {"reserved", "hourly"}:
            raise ValueError("remote compute billing mode is invalid")
        if self.cpu_cores < 0:
            raise ValueError("remote compute CPU count cannot be negative")


@dataclass(frozen=True, slots=True)
class RemoteComputeSettlement:
    """Billing-owned result attached to one completed resource lease."""

    charge_id: UUID
    charged_credits: int
    replayed: bool = False

    def __post_init__(self) -> None:
        if self.charged_credits < 0:
            raise ValueError("remote compute charged credits cannot be negative")


@dataclass(frozen=True, slots=True)
class RemoteEnvironmentDescriptor:
    """Agent-only connection descriptor for a bounded environment scan."""

    resource_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    host: str
    port: int
    username: str
    secret_ref: str
    secret_version: str
    remote_root: str

    @classmethod
    def from_resource(cls, resource: "RemoteComputeResource") -> "RemoteEnvironmentDescriptor":
        return cls(
            resource_id=resource.resource_id,
            resource_tid=resource.resource_tid,
            owner_uid=resource.owner_uid,
            host=resource.access_host,
            port=resource.access_port,
            username=resource.access_username,
            secret_ref=resource.secret_ref,
            secret_version=resource.secret_version,
            remote_root=resource.remote_root,
        )

    def internal_dict(self) -> dict[str, object]:
        return {
            "resource_id": format_typed_id("rcres", self.resource_id),
            "resource_tid": format_typed_id("tid", self.resource_tid),
            "owner_uid": format_typed_id("uid", self.owner_uid),
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "secret_ref": self.secret_ref,
            "secret_version": self.secret_version,
            "remote_root": self.remote_root,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object] | object) -> "RemoteEnvironmentDescriptor":
        if not isinstance(raw, Mapping) or set(raw) != {
            "resource_id",
            "resource_tid",
            "owner_uid",
            "host",
            "port",
            "username",
            "secret_ref",
            "secret_version",
            "remote_root",
        }:
            raise ValueError("invalid remote environment descriptor")

        def text(name: str, maximum: int, *, allow_empty: bool = False) -> str:
            value = raw.get(name)
            if (
                not isinstance(value, str)
                or (not allow_empty and not value)
                or value != value.strip()
                or len(value.encode("utf-8")) > maximum
                or "\x00" in value
            ):
                raise ValueError(f"invalid remote environment field: {name}")
            return value

        port = raw.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
            raise ValueError("invalid remote environment port")
        root = text("remote_root", 4096)
        if not root.startswith("/") or root == "/":
            raise ValueError("remote environment root must be a confined absolute path")
        try:
            return cls(
                resource_id=parse_typed_id(text("resource_id", 64), expected_prefix="rcres"),
                resource_tid=parse_typed_id(text("resource_tid", 64), expected_prefix="tid"),
                owner_uid=parse_typed_id(text("owner_uid", 64), expected_prefix="uid"),
                host=text("host", 255),
                port=port,
                username=text("username", 255),
                secret_ref=text("secret_ref", 512),
                secret_version=text("secret_version", 256, allow_empty=True),
                remote_root=root,
            )
        except InvalidTypedId as exc:
            raise ValueError("invalid remote environment typed ID") from exc


@dataclass(frozen=True, slots=True)
class RemoteComputeCommand:
    command_id: UUID
    resource_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    action: RemoteComputeAction
    command_generation: int
    expected_resource_version: int
    producer_seq: int = 1
    rid: UUID | None = None
    binding_id: UUID | None = None
    lease_generation: int | None = None
    target_resource_id: UUID | None = None
    idempotency_key: str = ""
    request_fingerprint: str = ""
    provider: str = ""
    external_resource_id: str = ""
    access_host: str = ""
    access_port: int = 22
    access_username: str = ""
    remote_root: str = ""
    secret_ref: str = ""
    secret_version: str = ""
    target_access_host: str = ""
    target_access_port: int = 22
    target_access_username: str = ""
    target_remote_root: str = ""
    target_secret_ref: str = ""
    target_secret_version: str = ""
    cleanup_items: tuple[Mapping[str, str], ...] = ()
    resource_policy: Mapping[str, int] | None = None
    requested_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RemoteComputeCommandReceipt:
    command_id: UUID
    resource_id: UUID
    action: RemoteComputeAction
    state: RemoteComputeCommandState
    command_generation: int
    resource_version: int
    resource_status: str
    replayed: bool = False
    delivery_state: str = "APPLIED"
    error_code: str | None = None
    result: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RemoteComputeResultEvent:
    event_id: UUID
    command_id: UUID
    resource_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    action: RemoteComputeAction
    command_generation: int
    expected_resource_version: int
    producer_seq: int
    occurred_at: datetime
    payload_sha256: bytes
    payload: Mapping[str, Any]
    rid: UUID | None = None
    binding_id: UUID | None = None
    lease_generation: int | None = None
    target_resource_id: UUID | None = None
    status: str = "APPLIED"
    error_code: str | None = None
    effect_result: Mapping[str, Any] | None = None
    causation_event_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class RemoteWorkspaceDescriptor:
    """Least-privilege connection descriptor returned to Agent.

    ``secret_ref`` is an opaque lookup key.  The secret value never crosses
    the Product HTTP boundary and is never serialized by this class.
    """

    binding_id: UUID
    resource_id: UUID
    resource_tid: UUID
    owner_uid: UUID
    sid: UUID
    rid: UUID
    lease_generation: int
    status: str
    host: str
    port: int
    username: str
    secret_ref: str
    secret_version: str
    remote_root: str
    provider: str
    connection_name: str
    expires_at: datetime | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object] | object) -> "RemoteWorkspaceDescriptor":
        """Parse the internal wire form without accepting arbitrary IDs.

        Product and Agent use typed UUIDs at the HTTP boundary.  Keeping the
        parser here means every consumer applies the same canonical-ID and
        descriptor checks before a connection is attempted.
        """

        if not isinstance(raw, dict):
            raise TypeError("remote workspace descriptor must be an object")

        def text(name: str, *, required: bool = True, maximum: int = 512) -> str:
            value = raw.get(name)
            if not required and value is None:
                return ""
            if not isinstance(value, str) or (required and not value) or len(value) > maximum:
                raise ValueError(f"invalid remote workspace field: {name}")
            return value

        def typed(name: str, prefix: str) -> UUID:
            try:
                return parse_typed_id(text(name), expected_prefix=prefix)
            except InvalidTypedId as exc:
                raise ValueError(f"invalid remote workspace field: {name}") from exc

        port = raw.get("port")
        generation = raw.get("lease_generation")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
            raise ValueError("invalid remote workspace port")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("invalid remote workspace lease generation")
        expires = raw.get("expires_at")
        expires_at: datetime | None
        if expires is None:
            expires_at = None
        elif isinstance(expires, str):
            try:
                expires_at = datetime.fromisoformat(expires)
            except ValueError as exc:
                raise ValueError("invalid remote workspace expiry") from exc
            if expires_at.tzinfo is None or expires_at.utcoffset() is None:
                raise ValueError("remote workspace expiry must include a timezone")
        else:
            raise ValueError("invalid remote workspace expiry")
        host = text("host", maximum=255)
        root = text("remote_root", maximum=4096)
        if not root.startswith("/") or "\x00" in root:
            raise ValueError("remote workspace root must be absolute")
        status = text("status", maximum=32)
        if status not in {"ACTIVE", "RUNNING", "RELEASED", "EXPIRED"}:
            raise ValueError("invalid remote workspace status")
        return cls(
            binding_id=typed("binding_id", "rcbind"),
            resource_id=typed("resource_id", "rcres"),
            resource_tid=typed("resource_tid", "tid"),
            owner_uid=typed("owner_uid", "uid"),
            sid=typed("sid", "sid"),
            rid=typed("rid", "rid"),
            lease_generation=generation,
            status=status,
            host=host,
            port=port,
            username=text("username", maximum=256),
            secret_ref=text("secret_ref", maximum=512),
            secret_version=text("secret_version", required=False, maximum=128),
            remote_root=root,
            provider=text("provider", maximum=128),
            connection_name=text("connection_name", maximum=256),
            expires_at=expires_at,
        )

    def public_dict(self) -> dict[str, object]:
        """Return the bounded internal-wire representation without secrets."""

        return {
            "binding_id": format_typed_id("rcbind", self.binding_id),
            "resource_id": format_typed_id("rcres", self.resource_id),
            "resource_tid": format_typed_id("tid", self.resource_tid),
            "owner_uid": format_typed_id("uid", self.owner_uid),
            "sid": format_typed_id("sid", self.sid),
            "rid": format_typed_id("rid", self.rid),
            "lease_generation": self.lease_generation,
            "status": self.status,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "secret_ref": self.secret_ref,
            "secret_version": self.secret_version,
            "remote_root": self.remote_root,
            "provider": self.provider,
            "connection_name": self.connection_name,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


__all__ = [
    "RemoteComputeAction",
    "RemoteComputeCommand",
    "RemoteComputeCommandReceipt",
    "RemoteComputeCommandState",
    "RemoteComputeResource",
    "RemoteComputeResultEvent",
    "RemoteWorkspaceBinding",
    "RemoteWorkspaceDescriptor",
]
