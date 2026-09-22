"""Provider-neutral V2 Remote Compute provisioning contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from apps.common.v2_ids import InvalidTypedId, format_typed_id, parse_typed_id


class RemoteComputeProvisioningError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class ProvisionedRemoteCompute:
    resource_id: UUID
    name: str
    provider: str
    status: str
    access_host: str
    access_port: int
    access_username: str
    secret_ref: str
    secret_version: str
    remote_root: str
    external_resource_id: str
    accelerator_type: str = ""
    accelerator_count: int = 0
    vram_gb: int = 0
    billing_mode: str = "hourly"
    hourly_price_credits: int = 0
    region: str = ""
    instance_type: str = ""
    cpu_cores: int = 0
    memory_gb: int = 0
    storage_gb: int = 0

    def __post_init__(self) -> None:
        format_typed_id("rcres", self.resource_id)
        for field, maximum in (
            (self.name, 200),
            (self.provider, 128),
            (self.access_host, 255),
            (self.access_username, 256),
            (self.secret_ref, 512),
            (self.secret_version, 256),
            (self.remote_root, 4096),
            (self.external_resource_id, 512),
            (self.region, 128),
            (self.instance_type, 256),
        ):
            if len(field.encode("utf-8")) > maximum or "\x00" in field:
                raise ValueError("provisioned remote compute text is invalid")
        if not all(
            (
                self.name,
                self.provider,
                self.access_host,
                self.access_username,
                self.secret_ref,
                self.secret_version,
                self.remote_root,
                self.external_resource_id,
            )
        ):
            raise ValueError("provisioned remote compute identity is incomplete")
        if self.status not in {"ACTIVE", "IDLE"}:
            raise ValueError("provisioned remote compute status is invalid")
        if self.billing_mode not in {"reserved", "hourly"}:
            raise ValueError("provisioned remote compute billing mode is invalid")
        if not self.remote_root.startswith("/") or self.remote_root == "/":
            raise ValueError("provisioned remote root must be confined")
        if not 1 <= self.access_port <= 65_535:
            raise ValueError("provisioned remote compute port is invalid")
        if any(
            value < 0
            for value in (
                self.accelerator_count,
                self.vram_gb,
                self.hourly_price_credits,
                self.cpu_cores,
                self.memory_gb,
                self.storage_gb,
            )
        ):
            raise ValueError("provisioned remote compute capacity is invalid")

    def internal_dict(self) -> dict[str, object]:
        return {
            "resource_id": format_typed_id("rcres", self.resource_id),
            "name": self.name,
            "provider": self.provider,
            "status": self.status,
            "access_host": self.access_host,
            "access_port": self.access_port,
            "access_username": self.access_username,
            "secret_ref": self.secret_ref,
            "secret_version": self.secret_version,
            "remote_root": self.remote_root,
            "external_resource_id": self.external_resource_id,
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "vram_gb": self.vram_gb,
            "billing_mode": self.billing_mode,
            "hourly_price_credits": self.hourly_price_credits,
            "region": self.region,
            "instance_type": self.instance_type,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "storage_gb": self.storage_gb,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | object) -> "ProvisionedRemoteCompute":
        fields = {
            "resource_id",
            "name",
            "provider",
            "status",
            "access_host",
            "access_port",
            "access_username",
            "secret_ref",
            "secret_version",
            "remote_root",
            "external_resource_id",
            "accelerator_type",
            "accelerator_count",
            "vram_gb",
            "billing_mode",
            "hourly_price_credits",
            "region",
            "instance_type",
            "cpu_cores",
            "memory_gb",
            "storage_gb",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError("provisioned remote compute response is malformed")
        try:
            resource_id = parse_typed_id(raw["resource_id"], expected_prefix="rcres")
            text_fields = {
                name: raw[name]
                for name in (
                    "name",
                    "provider",
                    "status",
                    "access_host",
                    "access_username",
                    "secret_ref",
                    "secret_version",
                    "remote_root",
                    "external_resource_id",
                    "accelerator_type",
                    "billing_mode",
                    "region",
                    "instance_type",
                )
                if isinstance(raw[name], str)
            }
            if len(text_fields) != 13:
                raise ValueError("provisioned remote compute text fields are invalid")
            integer_fields = {
                name: raw[name]
                for name in (
                    "access_port",
                    "accelerator_count",
                    "vram_gb",
                    "hourly_price_credits",
                    "cpu_cores",
                    "memory_gb",
                    "storage_gb",
                )
                if isinstance(raw[name], int) and not isinstance(raw[name], bool)
            }
            if len(integer_fields) != 7:
                raise ValueError("provisioned remote compute numeric fields are invalid")
            return cls(
                resource_id=resource_id,
                name=text_fields["name"],
                provider=text_fields["provider"],
                status=text_fields["status"],
                access_host=text_fields["access_host"],
                access_port=integer_fields["access_port"],
                access_username=text_fields["access_username"],
                secret_ref=text_fields["secret_ref"],
                secret_version=text_fields["secret_version"],
                remote_root=text_fields["remote_root"],
                external_resource_id=text_fields["external_resource_id"],
                accelerator_type=text_fields["accelerator_type"],
                accelerator_count=integer_fields["accelerator_count"],
                vram_gb=integer_fields["vram_gb"],
                billing_mode=text_fields["billing_mode"],
                hourly_price_credits=integer_fields["hourly_price_credits"],
                region=text_fields["region"],
                instance_type=text_fields["instance_type"],
                cpu_cores=integer_fields["cpu_cores"],
                memory_gb=integer_fields["memory_gb"],
                storage_gb=integer_fields["storage_gb"],
            )
        except (InvalidTypedId, KeyError, TypeError, ValueError) as exc:
            raise ValueError("provisioned remote compute response is invalid") from exc


__all__ = [
    "ProvisionedRemoteCompute",
    "RemoteComputeProvisioningError",
]
