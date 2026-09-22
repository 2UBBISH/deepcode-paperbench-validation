"""Agent-owned provider discovery and provisioning for strict V2."""

from __future__ import annotations

import logging

import hashlib
import math
import os
import re
import secrets
import shlex
import socket
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from uuid import UUID

from apps.common.billing_constants import CREDITS_PER_CNY
from apps.v2.reliability.circuit import CircuitPolicy, CircuitRegistry, DependencyUnavailable

from .autodl_config import AutoDLRuntimeConfig
from .provisioning import ProvisionedRemoteCompute, RemoteComputeProvisioningError
from .secrets import FileRemoteComputeSecretStore, RemoteComputeSecretError

try:  # pragma: no cover - image build verifies the dependency
    import paramiko
except ImportError:  # pragma: no cover
    paramiko = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class AutoDLProvisioningClient(Protocol):
    def gpu_stock(self, region: str, filters: Mapping[str, Any]) -> list[dict[str, Any]]: ...
    def list_instances(self, *, page: int = 1, page_size: int = 100) -> list[dict[str, Any]]: ...
    def create_instance(self, body: Mapping[str, Any]) -> str: ...
    def get_status(self, instance_uuid: str) -> str: ...
    def get_snapshot(self, instance_uuid: str) -> dict[str, Any]: ...
    def power_on(self, instance_uuid: str, *, start_command: str | None = None) -> None: ...
    def power_off(self, instance_uuid: str) -> None: ...
    def release(self, instance_uuid: str) -> None: ...


class AliyunProvisioningClient(Protocol):
    def api(self, action: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...
    def start_instance(self, instance_id: str) -> None: ...
    def stop_instance(self, instance_id: str) -> None: ...
    def delete_instance(self, instance_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AliyunProvisioningConfig:
    region_id: str
    vswitch_id: str
    security_group_id: str
    image_id: str
    image_family: str = "acs:ubuntu_22_04_x64"
    default_instance_type: str = "ecs.c7.large"
    system_disk_size_gb: int = 40
    internet_max_bandwidth_out: int = 10
    provision_wait_seconds: int = 180
    cpu_options: tuple[Mapping[str, Any], ...] = ()
    # GPU SKUs (experiment Agent). Each row: instance_type, accelerator_type,
    # accelerator_count, vram_gb (per card), cpu_cores, memory_gb, optional
    # image_id (a baked image with Docker + NVIDIA toolkit + CloudMonitor) and
    # storage_gb.  Unlike CPU rows there is no built-in default catalog: a
    # GPU SKU must be one the target region can actually sell.
    gpu_options: tuple[Mapping[str, Any], ...] = ()
    # Zone for DescribeAvailableResource stock checks and CreateInstance.
    # Empty keeps the vswitch's own zone (Aliyun infers it).
    zone_id: str = ""

    @property
    def enabled(self) -> bool:
        return bool(
            self.region_id
            and self.vswitch_id
            and self.security_group_id
            and (self.image_id or self.image_family)
        )

    def normalized_cpu_options(self) -> tuple[dict[str, int | str], ...]:
        source: Sequence[Mapping[str, Any]] = self.cpu_options or _DEFAULT_ALIYUN_CPU_OPTIONS
        options: list[dict[str, int | str]] = []
        for item in source:
            instance_type = str(item.get("instance_type") or "").strip()
            if not instance_type:
                continue
            options.append(
                {
                    "instance_type": instance_type,
                    "cpu_cores": max(1, _integer(item.get("cpu_cores"), 1)),
                    "memory_gb": max(1, _integer(item.get("memory_gb"), 1)),
                    "storage_gb": max(
                        20,
                        _integer(item.get("storage_gb"), self.system_disk_size_gb),
                    ),
                    "hourly_price_credits": max(
                        0,
                        _integer(item.get("hourly_price_credits"), 0),
                    ),
                }
            )
        return tuple(options)

    def normalized_gpu_options(self) -> tuple[dict[str, Any], ...]:
        options: list[dict[str, Any]] = []
        for item in self.gpu_options:
            instance_type = str(item.get("instance_type") or "").strip()
            accelerator = str(item.get("accelerator_type") or "").strip()
            if not instance_type or not accelerator:
                continue
            options.append(
                {
                    "instance_type": instance_type,
                    "accelerator_type": accelerator,
                    "accelerator_count": max(1, _integer(item.get("accelerator_count"), 1)),
                    "vram_gb": max(0, _integer(item.get("vram_gb"), 0)),
                    "cpu_cores": max(1, _integer(item.get("cpu_cores"), 1)),
                    "memory_gb": max(1, _integer(item.get("memory_gb"), 1)),
                    "storage_gb": max(
                        20,
                        _integer(item.get("storage_gb"), self.system_disk_size_gb),
                    ),
                    "hourly_price_credits": max(
                        0,
                        _integer(item.get("hourly_price_credits"), 0),
                    ),
                    "image_id": str(item.get("image_id") or "").strip(),
                    "zone_id": str(item.get("zone_id") or "").strip() or self.zone_id,
                }
            )
        return tuple(options)


_DEFAULT_ALIYUN_CPU_OPTIONS: tuple[Mapping[str, Any], ...] = (
    {"instance_type": "ecs.c7.large", "cpu_cores": 2, "memory_gb": 4},
    {"instance_type": "ecs.c7.xlarge", "cpu_cores": 4, "memory_gb": 8},
    {"instance_type": "ecs.c7.2xlarge", "cpu_cores": 8, "memory_gb": 16},
    {"instance_type": "ecs.c7.4xlarge", "cpu_cores": 16, "memory_gb": 32},
    {"instance_type": "ecs.c7.8xlarge", "cpu_cores": 32, "memory_gb": 64},
)
_GPU_SPEC_RULES: tuple[tuple[str, str], ...] = (
    ("vgpu-48", "v-48g"),
    ("4090d", "4090D"),
    ("4090", "v-48g"),
    ("5090", "5090-p"),
    ("4080", "v-32g-p"),
    ("h800", "h800"),
    ("3090", "v-48g-350w"),
)
_GPU_VRAM = {
    "5090": 32,
    "4090d": 24,
    "4090": 24,
    "3090": 24,
    "4080": 16,
    "h800": 80,
    "a100": 80,
    "a40": 48,
}


HOST_KEY_CIRCUITS = CircuitRegistry(CircuitPolicy.from_env())


class FileKnownHostsStore:
    """Trust a just-provisioned provider address and publish it atomically."""

    def __init__(self, path: Path, *, timeout_seconds: float = 10.0) -> None:
        if paramiko is None:
            raise RuntimeError("paramiko is required for Remote Compute host keys")
        if not path.is_absolute() or path.is_symlink() or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("known_hosts path and timeout are invalid")
        self.path = path
        self.timeout = timeout_seconds
        self._lock = RLock()

    def trust(self, host: str, port: int, *, fresh: bool = False) -> None:
        """Learn ``host``'s key.  ``fresh`` marks an address the provider just
        issued to an instance we created in this very call: a key already on
        file for it belongs to a machine that no longer exists (providers
        recycle public IPs within hours), so it is replaced instead of
        raising REMOTE_HOST_KEY_CHANGED.  Adopting an existing resource keeps
        the strict comparison."""

        if not host or not 1 <= port <= 65_535:
            raise RemoteComputeProvisioningError("REMOTE_HOST_KEY_TARGET_INVALID")
        circuit_key = hashlib.sha256(f"{host.lower()}\0{port}".encode()).hexdigest()
        try:
            permit = HOST_KEY_CIRCUITS.acquire(circuit_key)
        except DependencyUnavailable:
            raise RemoteComputeProvisioningError("REMOTE_HOST_KEY_CIRCUIT_OPEN", retryable=True) from None
        connection = None
        transport = None
        outcome = "neutral"
        deadline = time.monotonic() + self.timeout
        try:
            connection = socket.create_connection((host, port), timeout=self.timeout)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("host key deadline exceeded")
            transport = paramiko.Transport(connection)
            transport.banner_timeout = remaining
            transport.start_client(timeout=remaining)
            key = transport.get_remote_server_key()
            if time.monotonic() >= deadline:
                raise TimeoutError("host key deadline exceeded")
            outcome = "success"
        except (OSError, paramiko.SSHException) as exc:
            outcome = "failure"
            raise RemoteComputeProvisioningError(
                "REMOTE_HOST_KEY_UNAVAILABLE",
                retryable=True,
            ) from exc
        finally:
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            permit.finish(outcome)
        marker = host if port == 22 else f"[{host}]:{port}"
        with self._lock:
            keys = paramiko.HostKeys()
            if self.path.exists():
                try:
                    keys.load(str(self.path))
                except (OSError, paramiko.SSHException) as exc:
                    raise RemoteComputeProvisioningError("REMOTE_HOST_KEY_STORE_INVALID") from exc
            existing = keys.lookup(marker)
            if existing is not None and key.get_name() in existing:
                if existing[key.get_name()] == key:
                    return
                if not fresh:
                    raise RemoteComputeProvisioningError("REMOTE_HOST_KEY_CHANGED")
                # A recycled address: drop every stale key for this marker.
                if marker in keys:
                    del keys[marker]
            keys.add(marker, key.get_name(), key)
            self._publish(keys)

    def _publish(self, keys: Any) -> None:
        parent = self.path.parent
        if parent.is_symlink():
            raise RemoteComputeProvisioningError("REMOTE_HOST_KEY_STORE_INVALID")
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                dir=parent,
            )
            os.close(descriptor)
            try:
                os.chmod(temporary_name, 0o600)
                keys.save(temporary_name)
                with open(temporary_name, "rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary_name, self.path)
                directory_descriptor = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
        except OSError as exc:
            raise RemoteComputeProvisioningError("REMOTE_HOST_KEY_STORE_UNAVAILABLE") from exc


class StrictRemoteComputeProvisioner:
    def __init__(
        self,
        *,
        autodl_factory: Callable[[], AutoDLProvisioningClient],
        autodl_config: Callable[[], AutoDLRuntimeConfig],
        aliyun_factory: Callable[[], AliyunProvisioningClient],
        aliyun_config: AliyunProvisioningConfig,
        secrets_store: FileRemoteComputeSecretStore,
        known_hosts: FileKnownHostsStore,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._autodl_factory = autodl_factory
        self._autodl_config = autodl_config
        self._aliyun_factory = aliyun_factory
        self._aliyun_config = aliyun_config
        self._secrets = secrets_store
        self._known_hosts = known_hosts
        self._sleep = sleeper
        self._monotonic = monotonic

    def catalog(self, requirements: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        options: list[Mapping[str, Any]] = []
        options.extend(self._autodl_options(requirements))
        options.extend(self._aliyun_gpu_options(requirements))
        options.extend(self._aliyun_options(requirements))
        return tuple(options[:16])

    def provision(
        self,
        *,
        resource_id: UUID,
        target: str,
        spec: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute:
        provider = str(spec.get("provider") or "").strip().lower()
        if provider == "autodl" or target.startswith("cloud_gpu:autodl:"):
            return self._provision_autodl(resource_id, spec)
        if provider == "aliyun_ecs" or target.startswith(("cloud_cpu:aliyun:", "cloud_gpu:aliyun:")):
            return self._provision_aliyun(resource_id, spec)
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_UNSUPPORTED")

    def reconcile_provision(
        self,
        *,
        resource_id: UUID,
        target: str,
        spec: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute | None:
        """Query a stable Provider identity without creating or starting anything."""

        provider = str(spec.get("provider") or "").strip().lower()
        instance_name = _stable_instance_name(resource_id)
        if provider == "autodl" or target.startswith("cloud_gpu:autodl:"):
            client = self._autodl_factory()
            instance_uuid = self._find_autodl_instance(client, instance_name)
            if not instance_uuid or client.get_status(instance_uuid) != "running":
                return None
            snapshot = client.get_snapshot(instance_uuid)
            return self._autodl_result(resource_id, snapshot, spec, instance_uuid)
        if provider == "aliyun_ecs" or target.startswith("cloud_cpu:aliyun:"):
            client = self._aliyun_factory()
            instance_id = self._find_aliyun_instance(client, instance_name)
            if not instance_id:
                return None
            instance = self._aliyun_instance(client, instance_id)
            if str(instance.get("Status") or "") != "Running":
                return None
            host = _aliyun_public_ip(instance)
            if not host:
                return None
            secret_ref = f"remote-compute/aliyun/{resource_id}"
            password = self._secrets.resolve(secret_ref)
            version = hashlib.sha256(password.encode("utf-8")).hexdigest()[:24]
            self._known_hosts.trust(host, 22)
            selected = next(
                (
                    row
                    for row in self._aliyun_config.normalized_cpu_options()
                    if row["instance_type"]
                    == str(instance.get("InstanceType") or spec.get("instance_type") or "")
                ),
                {},
            )
            instance_type = str(
                instance.get("InstanceType") or spec.get("instance_type") or ""
            )
            return ProvisionedRemoteCompute(
                resource_id=resource_id,
                name=f"Aliyun ECS {instance_type} {str(resource_id)[:8]}",
                provider="aliyun_ecs",
                status="IDLE",
                access_host=host,
                access_port=22,
                access_username="root",
                secret_ref=secret_ref,
                secret_version=version,
                remote_root=_remote_root(resource_id),
                external_resource_id=instance_id,
                billing_mode="hourly",
                hourly_price_credits=int(
                    spec.get("hourly_price_credits")
                    or selected.get("hourly_price_credits")
                    or 0
                ),
                region=self._aliyun_config.region_id,
                instance_type=instance_type,
                cpu_cores=int(spec.get("cpu_cores") or selected.get("cpu_cores") or 0),
                memory_gb=int(spec.get("memory_gb") or selected.get("memory_gb") or 0),
                storage_gb=int(
                    spec.get("storage_gb")
                    or selected.get("storage_gb")
                    or self._aliyun_config.system_disk_size_gb
                ),
            )
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_UNSUPPORTED")

    def reconcile_activate(
        self,
        *,
        resource_id: UUID,
        resource: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute | None:
        """Adopt an activation only when a read proves the resource is running."""

        provider = str(resource.get("provider") or "").strip().lower()
        external_id = str(resource.get("external_resource_id") or "").strip()
        if not external_id:
            raise RemoteComputeProvisioningError("REMOTE_EXTERNAL_ID_MISSING")
        if provider in {"autodl", "autodl_pro"}:
            client = self._autodl_factory()
            if client.get_status(external_id) != "running":
                return None
            return self._autodl_result(
                resource_id,
                client.get_snapshot(external_id),
                resource,
                external_id,
            )
        if provider in {"aliyun", "aliyun_ecs"}:
            client = self._aliyun_factory()
            instance = self._aliyun_instance(client, external_id)
            if str(instance.get("Status") or "") != "Running":
                return None
            host = _aliyun_public_ip(instance)
            if not host:
                return None
            port = int(resource.get("access_port") or 22)
            self._known_hosts.trust(host, port)
            return ProvisionedRemoteCompute(
                resource_id=resource_id,
                name=str(resource.get("name") or f"Aliyun ECS {external_id}"),
                provider="aliyun_ecs",
                status="IDLE",
                access_host=host,
                access_port=port,
                access_username=str(resource.get("access_username") or "root"),
                secret_ref=str(resource.get("secret_ref") or ""),
                secret_version=str(resource.get("secret_version") or ""),
                remote_root=str(
                    resource.get("remote_root") or _remote_root(resource_id)
                ),
                external_resource_id=external_id,
                accelerator_type=str(resource.get("accelerator_type") or ""),
                accelerator_count=int(resource.get("accelerator_count") or 0),
                vram_gb=int(resource.get("vram_gb") or 0),
                billing_mode=str(resource.get("billing_mode") or "hourly"),
                hourly_price_credits=int(
                    resource.get("hourly_price_credits") or 0
                ),
                region=str(resource.get("region") or self._aliyun_config.region_id),
                instance_type=str(resource.get("instance_type") or ""),
                cpu_cores=int(resource.get("cpu_cores") or 0),
                memory_gb=int(resource.get("memory_gb") or 0),
                storage_gb=int(resource.get("storage_gb") or 0),
            )
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_UNSUPPORTED")

    def reconcile_release(self, provisioned: ProvisionedRemoteCompute) -> bool | None:
        """Return true only when Provider inventory proves the resource is absent."""

        provider = provisioned.provider.strip().lower()
        if provider in {"autodl", "autodl_pro"}:
            found = self._find_autodl_instance(
                self._autodl_factory(),
                _stable_instance_name(provisioned.resource_id),
            )
            return True if not found else None
        if provider in {"aliyun", "aliyun_ecs"}:
            instance = self._aliyun_instance(
                self._aliyun_factory(),
                provisioned.external_resource_id,
            )
            return True if not instance else None
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_UNSUPPORTED")

    def activate(
        self,
        *,
        resource_id: UUID,
        resource: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute:
        provider = str(resource.get("provider") or "").strip().lower()
        external_id = str(resource.get("external_resource_id") or "").strip()
        if not external_id:
            raise RemoteComputeProvisioningError("REMOTE_EXTERNAL_ID_MISSING")
        if provider in {"autodl", "autodl_pro"}:
            client = self._autodl_factory()
            status = client.get_status(external_id)
            if status != "running":
                client.power_on(external_id)
            self._wait_autodl(client, external_id)
            snapshot = client.get_snapshot(external_id)
            return self._autodl_result(resource_id, snapshot, resource, external_id)
        if provider in {"aliyun", "aliyun_ecs"}:
            client = self._aliyun_factory()
            instance = self._aliyun_instance(client, external_id)
            if str(instance.get("Status") or "") != "Running":
                client.start_instance(external_id)
                instance = self._wait_aliyun(client, external_id, {"Running"})
            host = _aliyun_public_ip(instance)
            if not host:
                raise RemoteComputeProvisioningError("REMOTE_PROVIDER_ACCESS_UNAVAILABLE")
            port = int(resource.get("access_port") or 22)
            self._known_hosts.trust(host, port)
            return ProvisionedRemoteCompute(
                resource_id=resource_id,
                name=str(resource.get("name") or f"Aliyun ECS {external_id}"),
                provider="aliyun_ecs",
                status="IDLE",
                access_host=host,
                access_port=port,
                access_username=str(resource.get("access_username") or "root"),
                secret_ref=str(resource.get("secret_ref") or ""),
                secret_version=str(resource.get("secret_version") or ""),
                remote_root=str(resource.get("remote_root") or _remote_root(resource_id)),
                external_resource_id=external_id,
                accelerator_type=str(resource.get("accelerator_type") or ""),
                accelerator_count=int(resource.get("accelerator_count") or 0),
                vram_gb=int(resource.get("vram_gb") or 0),
                billing_mode=str(resource.get("billing_mode") or "hourly"),
                hourly_price_credits=int(resource.get("hourly_price_credits") or 0),
                region=str(resource.get("region") or self._aliyun_config.region_id),
                instance_type=str(resource.get("instance_type") or ""),
                cpu_cores=int(resource.get("cpu_cores") or 0),
                memory_gb=int(resource.get("memory_gb") or 0),
                storage_gb=int(resource.get("storage_gb") or 0),
            )
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_UNSUPPORTED")

    def rollback(self, provisioned: ProvisionedRemoteCompute) -> None:
        """Remove an uncommitted provider allocation and its Agent-only secret.

        Product calls this only when provisioning succeeded on Agent but the
        canonical resource row or run binding could not be committed.  Replays
        are intentionally harmless.
        """

        provider = provisioned.provider.strip().lower()
        try:
            if provider in {"autodl", "autodl_pro"}:
                self._release_autodl(
                    self._autodl_factory(),
                    provisioned.external_resource_id,
                )
            elif provider in {"aliyun", "aliyun_ecs"}:
                if not self._delete_aliyun(
                    self._aliyun_factory(),
                    provisioned.external_resource_id,
                ):
                    # Let the RELEASE saga retry instead of reporting a machine
                    # that is still billing as released.
                    raise RemoteComputeProvisioningError(
                        "REMOTE_PROVIDER_ACCESS_UNAVAILABLE", retryable=True
                    )
            else:
                raise RemoteComputeProvisioningError("REMOTE_PROVIDER_UNSUPPORTED")
        finally:
            try:
                self._secrets.delete(provisioned.secret_ref)
            except RemoteComputeSecretError as exc:
                raise RemoteComputeProvisioningError(
                    "REMOTE_SECRET_DELETE_FAILED",
                    retryable=True,
                ) from exc

    def _autodl_options(self, requirements: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        config = self._autodl_config()
        if not config.api_key or not config.image_uuid:
            return []
        filters = {
            "cuda_v_from": config.cuda_v_from,
            "cuda_v_to": config.cuda_v_to,
            "memory_size_from": max(
                1,
                math.ceil(int(requirements.get("min_memory_mb") or 0) / 1024),
            ),
            "memory_size_to": config.memory_size_to,
            "cpu_num_from": max(1, int(requirements.get("min_cpu_cores") or 1)),
            "cpu_num_to": config.cpu_num_to,
            "price_from": config.price_from,
            "price_to": config.price_to,
        }
        client = self._autodl_factory()
        options: list[Mapping[str, Any]] = []
        minimum_vram = int(requirements.get("min_vram_gb") or 0)
        for region in config.regions:
            try:
                stock_rows = client.gpu_stock(region, filters)
            except Exception:
                continue
            for row in stock_rows:
                for gpu_name, raw_stock in row.items():
                    if not isinstance(raw_stock, Mapping):
                        continue
                    idle = _integer(raw_stock.get("idle_gpu_num"), 0)
                    total = _integer(raw_stock.get("total_gpu_num"), 0)
                    spec_uuid = _gpu_spec_uuid(str(gpu_name))
                    vram = _gpu_vram(str(gpu_name))
                    if idle <= 0 or not spec_uuid or (minimum_vram and vram < minimum_vram):
                        continue
                    price_units = int(config.gpu_price_units.get(str(gpu_name), 0))
                    hourly_credits = math.ceil(
                        max(price_units, 0)
                        * config.credits_per_cny
                        / config.autodl_price_units_per_cny
                    )
                    options.append(
                        {
                            "id": f"autodl:{region}:{gpu_name}",
                            "target": f"cloud_gpu:autodl:{region}:{gpu_name}",
                            "provider": "autodl",
                            "name": f"AutoDL {gpu_name} ({region})",
                            "spec": f"{gpu_name} x1 / {vram or '?'} GB VRAM / {region}",
                            "region": region,
                            "gpu_name": str(gpu_name),
                            "gpu_spec_uuid": spec_uuid,
                            "accelerator_type": str(gpu_name),
                            "accelerator_count": 1,
                            "vram_gb": vram,
                            "cpu_cores": max(1, int(requirements.get("min_cpu_cores") or 1)),
                            "memory_gb": max(
                                1,
                                math.ceil(int(requirements.get("min_memory_mb") or 0) / 1024),
                            ),
                            "storage_gb": max(config.system_disk_expand_gb, 0),
                            "hourly_price_credits": hourly_credits,
                            "autodl_price_units": price_units,
                            "available_now": True,
                            "status": "rentable",
                            "status_label": f"空闲 {idle}/{total}",
                            "idle_gpu_num": idle,
                            "total_gpu_num": total,
                        }
                    )
        options.sort(
            key=lambda item: (
                int(item.get("hourly_price_credits") or 0),
                -int(item.get("idle_gpu_num") or 0),
            )
        )
        return options[:8]

    def _aliyun_options(self, requirements: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        config = self._aliyun_config
        if not config.enabled:
            return []
        client = self._aliyun_factory()
        options: list[Mapping[str, Any]] = []
        minimum_cpu = int(requirements.get("min_cpu_cores") or 1)
        minimum_memory = math.ceil(int(requirements.get("min_memory_mb") or 0) / 1024)
        for row in config.normalized_cpu_options():
            cpu_cores = int(row["cpu_cores"])
            memory_gb = int(row["memory_gb"])
            if cpu_cores < minimum_cpu or memory_gb < minimum_memory:
                continue
            hourly_credits = int(row["hourly_price_credits"])
            try:
                price = client.api(
                    "DescribePrice",
                    {
                        "ResourceType": "instance",
                        "PriceUnit": "Hour",
                        "Period": "1",
                        "InstanceType": row["instance_type"],
                        "InstanceChargeType": "PostPaid",
                        "SystemDisk.Category": "cloud_essd",
                        "SystemDisk.Size": row["storage_gb"],
                        "InternetChargeType": "PayByTraffic",
                        "InternetMaxBandwidthOut": config.internet_max_bandwidth_out,
                        "ImageId": config.image_id or None,
                    },
                )
                provider_cny = _aliyun_hourly_price(price)
                if provider_cny > 0:
                    hourly_credits = max(hourly_credits, math.ceil(provider_cny * CREDITS_PER_CNY))
            except Exception:
                provider_cny = 0.0
            instance_type = str(row["instance_type"])
            options.append(
                {
                    "id": f"aliyun:{config.region_id}:{instance_type}",
                    "target": f"cloud_cpu:aliyun:{config.region_id}:{instance_type}",
                    "provider": "aliyun_ecs",
                    "name": f"Aliyun ECS {instance_type}",
                    "spec": f"{cpu_cores} vCPU / {memory_gb} GB RAM / {config.region_id}",
                    "region": config.region_id,
                    "instance_type": instance_type,
                    "cpu_cores": cpu_cores,
                    "memory_gb": memory_gb,
                    "storage_gb": int(row["storage_gb"]),
                    "hourly_price_credits": hourly_credits,
                    "provider_hourly_price_cny": provider_cny,
                    "image_id": config.image_id,
                    "image_family": config.image_family,
                    "os_type": "linux",
                    "execution_backend": "ssh",
                    "available_now": True,
                    "status": "rentable",
                    "status_label": "可按量创建",
                }
            )
        return options[:8]

    def _aliyun_gpu_options(self, requirements: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        """Aliyun GPU SKUs from the configured catalog, priced and stock-checked.

        Only offered when the caller asked for GPU (``gpu_recommended`` or a
        VRAM floor): CPU-only workloads must not see a 40 CNY/h card.  Price is
        DescribePrice for the SKU + its image; purchasability is
        DescribeAvailableResource when a zone is known — an unavailable SKU is
        still listed but ``available_now`` is False so the experiment tiering
        skips it (a sold-out A100 is a hard failure, not a degradation).
        """

        config = self._aliyun_config
        wants_gpu = bool(requirements.get("gpu_recommended")) or int(requirements.get("min_vram_gb") or 0) > 0
        if not config.enabled or not wants_gpu:
            return []
        rows = config.normalized_gpu_options()
        if not rows:
            return []
        client = self._aliyun_factory()
        minimum_vram = int(requirements.get("min_vram_gb") or 0)
        minimum_memory = math.ceil(int(requirements.get("min_memory_mb") or 0) / 1024)
        options: list[Mapping[str, Any]] = []
        for row in rows:
            total_vram = int(row["vram_gb"]) * int(row["accelerator_count"])
            if total_vram < minimum_vram or int(row["memory_gb"]) < minimum_memory:
                continue
            instance_type = str(row["instance_type"])
            image_id = str(row["image_id"] or config.image_id)
            hourly_credits = int(row["hourly_price_credits"])
            try:
                price = client.api(
                    "DescribePrice",
                    {
                        "ResourceType": "instance",
                        "PriceUnit": "Hour",
                        "Period": "1",
                        "InstanceType": instance_type,
                        "InstanceChargeType": "PostPaid",
                        "SystemDisk.Category": "cloud_essd",
                        "SystemDisk.Size": row["storage_gb"],
                        "InternetChargeType": "PayByTraffic",
                        "InternetMaxBandwidthOut": config.internet_max_bandwidth_out,
                        "ImageId": image_id or None,
                    },
                )
                provider_cny = _aliyun_hourly_price(price)
                if provider_cny > 0:
                    hourly_credits = max(hourly_credits, math.ceil(provider_cny * CREDITS_PER_CNY))
            except Exception:
                provider_cny = 0.0
            available = provider_cny > 0 or hourly_credits > 0
            zone_id = str(row["zone_id"] or "")
            if available and zone_id:
                try:
                    available = _aliyun_zone_has_stock(client, zone_id=zone_id, instance_type=instance_type)
                except Exception:
                    # A failed stock probe must not hide the SKU: CreateInstance
                    # is the final arbiter either way.
                    available = True
            count = int(row["accelerator_count"])
            cards = f"{count}x " if count > 1 else ""
            options.append(
                {
                    "id": f"aliyun:{config.region_id}:{instance_type}",
                    "target": f"cloud_gpu:aliyun:{config.region_id}:{instance_type}",
                    "provider": "aliyun_ecs",
                    "name": f"Aliyun ECS {instance_type}",
                    "spec": (
                        f"{cards}{row['accelerator_type']} {total_vram}G / "
                        f"{row['cpu_cores']} vCPU / {row['memory_gb']} GB RAM / {config.region_id}"
                    ),
                    "region": config.region_id,
                    "zone_id": zone_id,
                    "instance_type": instance_type,
                    "accelerator_type": str(row["accelerator_type"]),
                    "accelerator_count": count,
                    # Total VRAM (the Product option contract); the experiment
                    # tiering divides by accelerator_count for the per-card view.
                    "vram_gb": total_vram,
                    "cpu_cores": int(row["cpu_cores"]),
                    "memory_gb": int(row["memory_gb"]),
                    "storage_gb": int(row["storage_gb"]),
                    "hourly_price_credits": hourly_credits,
                    "provider_hourly_price_cny": provider_cny,
                    "image_id": image_id,
                    "image_family": config.image_family,
                    "os_type": "linux",
                    "execution_backend": "ssh",
                    "available_now": available,
                    "status": "rentable" if available else "unavailable",
                    "status_label": "可按量创建" if available else "该可用区暂无库存",
                }
            )
        return options[:8]

    def _provision_autodl(
        self,
        resource_id: UUID,
        spec: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute:
        config = self._autodl_config()
        if not config.api_key or not config.image_uuid:
            raise RemoteComputeProvisioningError("REMOTE_PROVIDER_NOT_CONFIGURED")
        gpu_name = str(spec.get("gpu_name") or spec.get("accelerator_type") or "").strip()
        spec_uuid = str(spec.get("gpu_spec_uuid") or _gpu_spec_uuid(gpu_name)).strip()
        region = str(spec.get("region") or (config.regions[0] if config.regions else "")).strip()
        if not gpu_name or not spec_uuid or not region:
            raise RemoteComputeProvisioningError("REMOTE_PROVISIONING_SPEC_INVALID")
        client = self._autodl_factory()
        instance_uuid = ""
        created_here = False
        instance_name = _stable_instance_name(resource_id)
        try:
            body: dict[str, Any] = {
                "req_gpu_amount": max(1, int(spec.get("accelerator_count") or 1)),
                "gpu_spec_uuid": spec_uuid,
                "image_uuid": config.image_uuid,
                "cuda_v_from": config.cuda_v_from,
                "expand_system_disk_by_gb": config.system_disk_expand_gb,
                "instance_name": instance_name,
                "data_center_list": [region],
            }
            if config.start_cmd:
                body["start_command"] = config.start_cmd
            instance_uuid = self._find_autodl_instance(client, instance_name)
            if not instance_uuid:
                instance_uuid = client.create_instance(body)
                created_here = True
            status = client.get_status(instance_uuid)
            if status in {"shutdown", "stopped"}:
                client.power_on(instance_uuid, start_command=config.start_cmd or None)
            self._wait_autodl(client, instance_uuid)
            snapshot = client.get_snapshot(instance_uuid)
            return self._autodl_result(resource_id, snapshot, spec, instance_uuid)
        except RemoteComputeProvisioningError:
            if created_here:
                self._release_autodl(client, instance_uuid)
            raise
        except Exception as exc:
            if created_here:
                self._release_autodl(client, instance_uuid)
            raise RemoteComputeProvisioningError(
                "REMOTE_PROVISIONING_FAILED",
                retryable=True,
            ) from exc

    def _autodl_result(
        self,
        resource_id: UUID,
        snapshot: Mapping[str, Any],
        spec: Mapping[str, Any],
        instance_uuid: str,
    ) -> ProvisionedRemoteCompute:
        ssh_command = str(snapshot.get("ssh_command") or "").strip()
        password = str(snapshot.get("root_password") or "")
        host, port, username = _ssh_connection(ssh_command)
        secret_ref = f"remote-compute/autodl/{instance_uuid}"
        if not password:
            try:
                password = self._secrets.resolve(secret_ref)
            except RemoteComputeSecretError as exc:
                raise RemoteComputeProvisioningError(
                    "REMOTE_PROVIDER_ACCESS_UNAVAILABLE"
                ) from exc
        version = hashlib.sha256(password.encode("utf-8")).hexdigest()[:24]
        try:
            self._secrets.put(secret_ref, password, version=version)
        except RemoteComputeSecretError as exc:
            raise RemoteComputeProvisioningError("REMOTE_SECRET_PERSIST_FAILED") from exc
        try:
            self._known_hosts.trust(host, port)
        except Exception:
            self._secrets.delete(secret_ref)
            raise
        config = self._autodl_config()
        gpu_name = str(
            snapshot.get("gpu_name")
            or spec.get("gpu_name")
            or spec.get("accelerator_type")
            or "GPU"
        )
        price_units = _autodl_price_units(snapshot, spec, config)
        hourly_credits = math.ceil(
            price_units * config.credits_per_cny / config.autodl_price_units_per_cny
        )
        return ProvisionedRemoteCompute(
            resource_id=resource_id,
            name=f"AutoDL Pro {gpu_name} {str(resource_id)[:8]}",
            provider="autodl",
            status="IDLE",
            access_host=host,
            access_port=port,
            access_username=username,
            secret_ref=secret_ref,
            secret_version=version,
            remote_root=_remote_root(resource_id),
            external_resource_id=instance_uuid,
            accelerator_type=gpu_name,
            accelerator_count=max(
                1,
                int(snapshot.get("gpu_num") or spec.get("accelerator_count") or 1),
            ),
            vram_gb=_gpu_vram(gpu_name) or int(spec.get("vram_gb") or 0),
            billing_mode="hourly",
            hourly_price_credits=hourly_credits,
            region=str(snapshot.get("data_center") or spec.get("region") or ""),
            instance_type=gpu_name,
            cpu_cores=int(snapshot.get("cpu_num") or spec.get("cpu_cores") or 0),
            memory_gb=_gib(snapshot.get("memory_size")) or int(spec.get("memory_gb") or 0),
            storage_gb=(
                _gib(snapshot.get("system_init_disk_size"))
                or int(spec.get("storage_gb") or 0)
                or config.system_disk_expand_gb
            ),
        )

    def _provision_aliyun(
        self,
        resource_id: UUID,
        spec: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute:
        config = self._aliyun_config
        if not config.enabled:
            raise RemoteComputeProvisioningError("REMOTE_PROVIDER_NOT_CONFIGURED")
        instance_type = str(
            spec.get("instance_type") or config.default_instance_type
        ).strip()
        if not instance_type:
            raise RemoteComputeProvisioningError("REMOTE_PROVISIONING_SPEC_INVALID")
        client = self._aliyun_factory()
        instance_id = ""
        created_here = False
        instance_name = _stable_instance_name(resource_id)
        secret_ref = f"remote-compute/aliyun/{resource_id}"
        secret_created_here = False
        try:
            try:
                password = self._secrets.resolve(secret_ref)
            except RemoteComputeSecretError:
                password = _root_password()
                version = hashlib.sha256(password.encode("utf-8")).hexdigest()[:24]
                self._secrets.put(secret_ref, password, version=version)
                secret_created_here = True
            version = hashlib.sha256(password.encode("utf-8")).hexdigest()[:24]
            gpu_row = next(
                (
                    row
                    for row in config.normalized_gpu_options()
                    if row["instance_type"] == instance_type
                ),
                {},
            )
            # ``snapshot_image_id`` is the experiment upgrade path: a machine
            # image taken of the previous (smaller) machine so the environment
            # survives the SKU change.  The catalog never produces this key, so
            # it cannot be overwritten by the option's own image_id.
            image_id = str(
                spec.get("snapshot_image_id")
                or spec.get("image_id")
                or gpu_row.get("image_id")
                or config.image_id
            ).strip()
            if not image_id:
                image_id = _resolve_aliyun_image(client, config.image_family)
            zone_id = str(spec.get("zone_id") or gpu_row.get("zone_id") or config.zone_id or "").strip()
            disk_size = max(
                config.system_disk_size_gb,
                _integer(spec.get("storage_gb"), 0),
                _integer(gpu_row.get("storage_gb"), 0),
            )
            instance_id = self._find_aliyun_instance(client, instance_name)
            if not instance_id:
                create_params: dict[str, Any] = {
                    "ImageId": image_id,
                    "InstanceType": instance_type,
                    "InstanceName": instance_name,
                    "HostName": _aliyun_host_name(resource_id),
                    "Password": password,
                    "SecurityGroupId": config.security_group_id,
                    "VSwitchId": config.vswitch_id,
                    "InstanceChargeType": "PostPaid",
                    "InternetChargeType": "PayByTraffic",
                    "InternetMaxBandwidthOut": config.internet_max_bandwidth_out,
                    "SystemDisk.Category": "cloud_essd",
                    "SystemDisk.Size": disk_size,
                    "IoOptimized": "optimized",
                }
                if zone_id:
                    create_params["ZoneId"] = zone_id
                response = client.api("CreateInstance", create_params)
                instance_id = str(response.get("InstanceId") or "").strip()
                created_here = bool(instance_id)
            if not instance_id:
                raise RemoteComputeProvisioningError("REMOTE_PROVIDER_RESPONSE_INVALID")
            instance = self._aliyun_instance(client, instance_id)
            if str(instance.get("Status") or "") != "Running":
                if str(instance.get("Status") or "") != "Stopped":
                    self._wait_aliyun(client, instance_id, {"Stopped"})
                client.start_instance(instance_id)
                instance = self._wait_aliyun(client, instance_id, {"Running"})
            host = _aliyun_public_ip(instance)
            if not host:
                allocated = client.api(
                    "AllocatePublicIpAddress",
                    {"InstanceId": instance_id},
                )
                host = str(allocated.get("IpAddress") or "").strip()
            if not host:
                raise RemoteComputeProvisioningError("REMOTE_PROVIDER_ACCESS_UNAVAILABLE")
            # ECS reports Running well before sshd answers; keep asking for the
            # host key until it does (bounded by the provisioning wait).
            self._trust_when_sshd_is_up(host, 22)
            selected = next(
                (
                    row
                    for row in config.normalized_cpu_options()
                    if row["instance_type"] == instance_type
                ),
                {},
            )
            selected = selected or gpu_row
            hourly_credits = int(spec.get("hourly_price_credits") or selected.get("hourly_price_credits") or 0)
            accelerator_count = int(spec.get("accelerator_count") or gpu_row.get("accelerator_count") or 0)
            return ProvisionedRemoteCompute(
                resource_id=resource_id,
                name=f"Aliyun ECS {instance_type} {str(resource_id)[:8]}",
                provider="aliyun_ecs",
                status="IDLE",
                access_host=host,
                access_port=22,
                access_username="root",
                secret_ref=secret_ref,
                secret_version=version,
                remote_root=_remote_root(resource_id),
                external_resource_id=instance_id,
                accelerator_type=str(spec.get("accelerator_type") or gpu_row.get("accelerator_type") or ""),
                accelerator_count=accelerator_count,
                vram_gb=int(spec.get("vram_gb") or (int(gpu_row.get("vram_gb") or 0) * accelerator_count)),
                billing_mode="hourly",
                hourly_price_credits=hourly_credits,
                region=config.region_id,
                instance_type=instance_type,
                cpu_cores=int(spec.get("cpu_cores") or selected.get("cpu_cores") or 0),
                memory_gb=int(spec.get("memory_gb") or selected.get("memory_gb") or 0),
                storage_gb=int(spec.get("storage_gb") or selected.get("storage_gb") or disk_size),
            )
        except RemoteComputeProvisioningError:
            self._undo_aliyun_provision(
                client, instance_id, created_here=created_here,
                secret_ref=secret_ref, secret_created_here=secret_created_here,
            )
            raise
        except Exception as exc:
            self._undo_aliyun_provision(
                client, instance_id, created_here=created_here,
                secret_ref=secret_ref, secret_created_here=secret_created_here,
            )
            raise RemoteComputeProvisioningError(
                "REMOTE_PROVISIONING_FAILED",
                retryable=True,
            ) from exc

    def _undo_aliyun_provision(
        self,
        client: AliyunProvisioningClient,
        instance_id: str,
        *,
        created_here: bool,
        secret_ref: str,
        secret_created_here: bool,
    ) -> None:
        """Best-effort rollback of a failed rent.

        The secret is only dropped once the machine is really gone: if the
        delete fails the instance keeps billing, and reconciliation can only
        adopt it while its credentials still exist.
        """

        if not created_here:
            return
        if self._delete_aliyun(client, instance_id) and secret_created_here:
            self._secrets.delete(secret_ref)

    def _trust_when_sshd_is_up(self, host: str, port: int) -> None:
        deadline = self._monotonic() + max(self._aliyun_config.provision_wait_seconds, 1)
        while True:
            try:
                # The instance was created moments ago: its address is fresh.
                self._known_hosts.trust(host, port, fresh=True)
                return
            except RemoteComputeProvisioningError as exc:
                if exc.code != "REMOTE_HOST_KEY_UNAVAILABLE" or self._monotonic() >= deadline:
                    raise
            self._sleep(5)

    def _wait_autodl(self, client: AutoDLProvisioningClient, instance_uuid: str) -> None:
        deadline = self._monotonic() + max(self._autodl_config().provision_wait_seconds, 1)
        status = client.get_status(instance_uuid)
        while status != "running" and self._monotonic() < deadline:
            self._sleep(3)
            status = client.get_status(instance_uuid)
        if status != "running":
            raise RemoteComputeProvisioningError(
                "REMOTE_PROVIDER_START_TIMEOUT",
                retryable=True,
            )

    @staticmethod
    def _find_autodl_instance(
        client: AutoDLProvisioningClient,
        instance_name: str,
    ) -> str:
        matches: list[str] = []
        for page in range(1, 6):
            rows = client.list_instances(page=page, page_size=100)
            for row in rows:
                name = str(
                    row.get("instance_name")
                    or row.get("name")
                    or row.get("InstanceName")
                    or ""
                ).strip()
                if name != instance_name:
                    continue
                instance_uuid = str(
                    row.get("instance_uuid")
                    or row.get("uuid")
                    or row.get("id")
                    or ""
                ).strip()
                if instance_uuid:
                    matches.append(instance_uuid)
            if len(rows) < 100:
                break
        unique = tuple(dict.fromkeys(matches))
        if len(unique) > 1:
            raise RemoteComputeProvisioningError("REMOTE_PROVISIONING_IDEMPOTENCY_CONFLICT")
        return unique[0] if unique else ""

    @staticmethod
    def _find_aliyun_instance(
        client: AliyunProvisioningClient,
        instance_name: str,
    ) -> str:
        payload = client.api("DescribeInstances", {"InstanceName": instance_name})
        instances = payload.get("Instances")
        rows = instances.get("Instance") if isinstance(instances, Mapping) else None
        matches = [
            str(row.get("InstanceId") or "").strip()
            for row in rows
            if isinstance(row, Mapping)
            and str(row.get("InstanceName") or "").strip() == instance_name
            and str(row.get("InstanceId") or "").strip()
        ] if isinstance(rows, list) else []
        unique = tuple(dict.fromkeys(matches))
        if len(unique) > 1:
            raise RemoteComputeProvisioningError("REMOTE_PROVISIONING_IDEMPOTENCY_CONFLICT")
        return unique[0] if unique else ""

    def _wait_aliyun(
        self,
        client: AliyunProvisioningClient,
        instance_id: str,
        statuses: set[str],
    ) -> Mapping[str, Any]:
        deadline = self._monotonic() + max(self._aliyun_config.provision_wait_seconds, 1)
        instance = self._aliyun_instance(client, instance_id)
        while str(instance.get("Status") or "") not in statuses and self._monotonic() < deadline:
            self._sleep(3)
            instance = self._aliyun_instance(client, instance_id)
        if str(instance.get("Status") or "") not in statuses:
            raise RemoteComputeProvisioningError(
                "REMOTE_PROVIDER_START_TIMEOUT",
                retryable=True,
            )
        return instance

    @staticmethod
    def _aliyun_instance(
        client: AliyunProvisioningClient,
        instance_id: str,
    ) -> Mapping[str, Any]:
        payload = client.api("DescribeInstances", {"InstanceIds": [instance_id]})
        instances = payload.get("Instances")
        rows = instances.get("Instance") if isinstance(instances, Mapping) else None
        return dict(rows[0]) if isinstance(rows, list) and rows and isinstance(rows[0], Mapping) else {}

    @staticmethod
    def _release_autodl(client: AutoDLProvisioningClient, instance_uuid: str) -> None:
        if not instance_uuid:
            return
        try:
            client.power_off(instance_uuid)
        except Exception:
            pass
        try:
            client.release(instance_uuid)
        except Exception:
            pass

    def _delete_aliyun(self, client: AliyunProvisioningClient, instance_id: str) -> bool:
        """Delete an instance we created; returns whether the provider accepted.

        ECS refuses DeleteInstance while the machine is still Initializing /
        Starting, and a swallowed refusal used to leave a billing orphan, so
        wait for a deletable status and say what happened.
        """

        if not instance_id:
            return True
        deadline = self._monotonic() + max(self._aliyun_config.provision_wait_seconds, 1)
        while True:
            try:
                client.delete_instance(instance_id)
                return True
            except Exception as exc:
                status = ""
                try:
                    status = str(self._aliyun_instance(client, instance_id).get("Status") or "")
                except Exception:
                    pass
                if status in {"Pending", "Starting", "Stopping"} and self._monotonic() < deadline:
                    self._sleep(5)
                    continue
                logger.error(
                    "aliyun instance %s could not be deleted (status=%s): %s",
                    instance_id, status or "?", exc,
                )
                return False


def _ssh_connection(command: str) -> tuple[str, int, str]:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_ACCESS_UNAVAILABLE") from exc
    if not parts or parts[0] != "ssh":
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_ACCESS_UNAVAILABLE")
    port = 22
    destination = ""
    index = 1
    while index < len(parts):
        token = parts[index]
        if token in {"-p", "-l", "-i", "-o"}:
            if index + 1 >= len(parts):
                raise RemoteComputeProvisioningError("REMOTE_PROVIDER_ACCESS_UNAVAILABLE")
            if token == "-p":
                port = _integer(parts[index + 1], 0)
            elif token == "-l":
                destination = f"{parts[index + 1]}@{destination}"
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        destination = token
        index += 1
    if "@" in destination:
        username, host = destination.rsplit("@", 1)
    else:
        username, host = "root", destination
    if not host or not username or not 1 <= port <= 65_535:
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_ACCESS_UNAVAILABLE")
    return host, port, username


def _aliyun_host_name(resource_id: UUID) -> str:
    """ECS HostName: letters/digits/single hyphens only, no leading/trailing hyphen.

    ``deepevol-<uuid prefix>`` where the UUID prefix itself contains hyphens
    would yield ``--``, which Aliyun rejects (InvalidHostName.Malformed) —
    seen on the V1 line, where the random id sometimes carried ``-``.
    """

    compact = "".join(ch for ch in str(resource_id) if ch.isalnum())[:12]
    return f"deepevol-{compact}"


def _aliyun_zone_has_stock(client: AliyunProvisioningClient, *, zone_id: str, instance_type: str) -> bool:
    payload = client.api(
        "DescribeAvailableResource",
        {
            "DestinationResource": "InstanceType",
            "InstanceChargeType": "PostPaid",
            "ZoneId": zone_id,
            "InstanceType": instance_type,
        },
    )
    zones = payload.get("AvailableZones") or {}
    rows = zones.get("AvailableZone") if isinstance(zones, Mapping) else zones
    if not isinstance(rows, list):
        return True
    for zone in rows:
        if not isinstance(zone, Mapping) or str(zone.get("ZoneId") or "") != zone_id:
            continue
        resources = zone.get("AvailableResources") or {}
        items = resources.get("AvailableResource") if isinstance(resources, Mapping) else resources
        for item in items or []:
            supported = (item or {}).get("SupportedResources") or {}
            entries = supported.get("SupportedResource") if isinstance(supported, Mapping) else supported
            for entry in entries or []:
                if str((entry or {}).get("Value") or "") != instance_type:
                    continue
                # Status: Available | SoldOut; StatusCategory: WithStock |
                # ClosedWithStock | WithoutStock | ClosedWithoutStock.
                status = str(entry.get("Status") or "Available").lower()
                category = str(entry.get("StatusCategory") or "WithStock").lower()
                return status == "available" and category in {"withstock", "closedwithstock"}
        return False
    return True


def _stable_instance_name(resource_id: UUID) -> str:
    return f"deepevol-{resource_id!s}"


def _gpu_spec_uuid(gpu_name: str) -> str:
    lowered = gpu_name.strip().lower()
    return next((spec for token, spec in _GPU_SPEC_RULES if token in lowered), "")


def _gpu_vram(gpu_name: str) -> int:
    lowered = gpu_name.strip().lower()
    return next((vram for token, vram in _GPU_VRAM.items() if token in lowered), 0)


def _autodl_price_units(
    snapshot: Mapping[str, Any],
    spec: Mapping[str, Any],
    config: AutoDLRuntimeConfig,
) -> float:
    payg = snapshot.get("payg_price")
    if payg not in {None, ""}:
        try:
            return max(float(payg) / 1000, 0.0)
        except (TypeError, ValueError):
            pass
    raw = spec.get("autodl_price_units")
    if raw not in {None, ""}:
        try:
            return max(float(raw), 0.0)
        except (TypeError, ValueError):
            pass
    name = str(spec.get("gpu_name") or spec.get("accelerator_type") or "")
    return float(config.gpu_price_units.get(name, 0))


def _aliyun_hourly_price(payload: Mapping[str, Any]) -> float:
    info = payload.get("PriceInfo")
    price = info.get("Price") if isinstance(info, Mapping) else None
    if not isinstance(price, Mapping):
        return 0.0
    for key in ("TradePrice", "DiscountPrice", "OriginalPrice", "StandardPrice", "Price"):
        try:
            value = float(price.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def _resolve_aliyun_image(client: AliyunProvisioningClient, family: str) -> str:
    payload = client.api(
        "DescribeImages",
        {
            "ImageOwnerAlias": "system",
            "OSType": "linux",
            "Architecture": "x86_64",
            "ImageFamily": family,
            "PageSize": 100,
        },
    )
    images = payload.get("Images")
    rows = images.get("Image") if isinstance(images, Mapping) else None
    candidates = [dict(item) for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []
    candidates.sort(key=lambda item: str(item.get("CreationTime") or ""), reverse=True)
    image_id = str((candidates[0] if candidates else {}).get("ImageId") or "").strip()
    if not image_id:
        raise RemoteComputeProvisioningError("REMOTE_PROVIDER_IMAGE_UNAVAILABLE")
    return image_id


def _aliyun_public_ip(instance: Mapping[str, Any]) -> str:
    public = instance.get("PublicIpAddress")
    values = public.get("IpAddress") if isinstance(public, Mapping) else None
    if isinstance(values, list):
        return next((str(value).strip() for value in values if str(value).strip()), "")
    eip = instance.get("EipAddress")
    return str(eip.get("IpAddress") or "").strip() if isinstance(eip, Mapping) else ""


def _root_password(length: int = 30) -> str:
    """Aliyun ECS caps ``Password`` at 30 characters (InvalidPassword.Malformed
    above that); AutoDL accepts the same alphabet."""

    if not 8 <= length <= 30:
        raise ValueError("remote compute root password must be 8..30 characters")
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%^&*_-+="
    while True:
        value = "".join(secrets.choice(alphabet) for _ in range(length))
        if all(
            re.search(pattern, value)
            for pattern in (r"[A-Z]", r"[a-z]", r"[0-9]", r"[^A-Za-z0-9]")
        ):
            return value


def _remote_root(resource_id: UUID) -> str:
    return f"/root/deepevol/{resource_id}"


def _gib(value: Any) -> int:
    try:
        size = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return math.ceil(size / (1024**3)) if size > 0 else 0


def _integer(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


__all__ = [
    "AliyunProvisioningConfig",
    "FileKnownHostsStore",
    "StrictRemoteComputeProvisioner",
]
