"""One rented Aliyun ECS machine per run: create, reach, authorize a key, delete.

Adapted from feature/reproduction-c's canary lease
(``paper_reproduction_agent/canary/aliyun_lease.py`` @ 02853e7a). Two
changes the plan called for: errors are :class:`LeaseError` (the vendored
orchestrator's own class) instead of the reproduction line's
``ManifestViolation``; the ECS client is main's
``apps.v2.remote_compute.providers.AliyunControlClient`` extended with the
calls the canary needed (``RunInstances``, ``DescribeInstances``,
``AllocatePublicIpAddress``, Cloud Assistant ``RunCommand`` /
``DescribeInvocationResults``, ``DescribePrice``) on top of its generic
signed ``api``. Its ``_api`` is overridden so the ECS error code and
message survive — main's version folds every API error into one opaque
``REMOTE_PROVIDER_REJECTED``, which makes a failed ``RunInstances``
undebuggable.

The orchestrator (``ExperimentLease``), the release policy and remote_relay are main's own
(``apps.v2.agent_engine.experiment`` / ``apps.v2.agent_engine.remote_relay``); the line carried
copies of all three until PLAN-3 item 4a, when step 10 moved onto main's experiment agent and the
copies went (they had already fallen behind main by a bring-up retry and a status-retry fix).

The record lives in ``<run_dir>/lease.json``; the generated root password
in ``<run_dir>/secrets/remote-compute.json`` (0600) through main's
``FileRemoteComputeSecretStore``; the per-run SSH keypair next to it.

Credentials come from the environment (``--env-file``): either
``ALIYUN_*`` or ``DEEPEVOL_API_ALIYUN_*`` spellings are accepted.
Nothing here is touched unless the run was initialised with
``--compute aliyun``.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import hmac
import json
import secrets
import shlex
import string
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from apps.v2.agent_engine.experiment.lease import ExperimentLease, LeaseError, LeaseEvent, LeaseHandle, LeaseState
from apps.v2.remote_compute.providers import AliyunControlClient, _aliyun_parameter, _aliyun_quote
from apps.v2.remote_compute.secrets import SECRET_STORE_SCHEMA_VERSION, FileRemoteComputeSecretStore

CPU_TIERS = {"enough": "ecs.c7.xlarge", "comfortable": "ecs.c7.2xlarge"}
#: GPU machines the line can rent (PLAN-3 §2.3 S6): instance type → (GPU name, VRAM GiB per GPU, GPU count).
#: cn-hongkong, all in stock on 2026-09-18; hourly prices come from DescribePrice at decision time
#: (T4 ≈ 8 CNY/h, A10 ≈ 17 CNY/h). They boot the GPU image (``DEEPEVOL_API_ALIYUN_GPU_IMAGE_ID``, baked in S5).
GPU_CATALOG: dict[str, tuple[str, int, int]] = {
    "ecs.gn6i-c4g1.xlarge": ("T4", 16, 1),
    "ecs.gn6i-c8g1.2xlarge": ("T4", 16, 1),
    "ecs.gn7i-c8g1.2xlarge": ("A10", 24, 1),
    "ecs.gn7i-c16g1.4xlarge": ("A10", 24, 1),
    "ecs.gn7i-c32g1.8xlarge": ("A10", 24, 1),
    "ecs.gn7i-c32g1.16xlarge": ("A10", 24, 2),
}
#: the tier keys a run may name before any code exists (``compute_tier``): a GPU tier means "prefer the GPU
#: plan's tier of that name"; the smallest GPU machine stands in until the compute phase decides
GPU_TIER_KEYS = ("gpu-economy", "gpu-standard", "gpu-insurance", "gpu-speed")
SMALLEST_GPU_TYPE = "ecs.gn6i-c4g1.xlarge"
INSTANCE_CAPACITY = {
    "ecs.c7.large": (2, 4),
    "ecs.c7.xlarge": (4, 8),
    "ecs.c7.2xlarge": (8, 16),
    "ecs.c7.4xlarge": (16, 32),
    "ecs.c7.8xlarge": (32, 64),
    "ecs.gn6i-c4g1.xlarge": (4, 15),
    "ecs.gn6i-c8g1.2xlarge": (8, 31),
    "ecs.gn7i-c8g1.2xlarge": (8, 30),
    "ecs.gn7i-c16g1.4xlarge": (16, 60),
    "ecs.gn7i-c32g1.8xlarge": (32, 188),
    "ecs.gn7i-c32g1.16xlarge": (64, 376),
}


def is_gpu_type(instance_type: str) -> bool:
    return instance_type in GPU_CATALOG or instance_type.startswith("ecs.gn")
INSTANCE_PREFIX = "p2c-"
SECRET_REF = "paper2code-aliyun"
READY_TIMEOUT_SECONDS = 600.0
BRING_UP_BACKOFF = 15.0
# DeleteInstance is refused with ``IncorrectInstanceStatus.Initializing`` while Aliyun still counts the
# instance as initialising — real machine 2026-09-17: sshd up and the bootstrap done 60 s after
# RunInstances, DeleteInstance refused for minutes after that. main's orchestrator retries a release
# three times with 2/4/6 s backoff, which is far too short for this, so the backend itself waits.
DELETE_RETRY_SECONDS = 10.0
DELETE_RETRY_BUDGET_SECONDS = 300.0
API_VERSION = "2014-05-26"

# The Docker-only slice of reproduction-c's deploy/experiment-images/bootstrap.sh, for an image that is not
# Docker-ready (lease.json then records bootstrap_required). Aliyun mirrors because download.docker.com is reset
# from mainland machines; registry mirrors only when Docker Hub is unreachable.
MINIMAL_BOOTSTRAP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt_get() { apt-get -o DPkg::Lock::Timeout=600 "$@"; }
systemctl stop unattended-upgrades 2>/dev/null || true
if ! command -v docker >/dev/null 2>&1; then
  apt_get -qq update
  apt_get -qq install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $CODENAME stable" > /etc/apt/sources.list.d/docker.list
  apt_get -qq update
  apt_get -qq install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi
mkdir -p /etc/docker
if curl -sS -m 8 -o /dev/null https://registry-1.docker.io/v2/ 2>/dev/null; then
  [ -f /etc/docker/daemon.json ] || echo '{}' > /etc/docker/daemon.json
else
  printf '%s\n' '{"registry-mirrors": ["https://dockerproxy.net", "https://docker.m.daocloud.io", "https://docker.1ms.run"]}' > /etc/docker/daemon.json
fi
systemctl enable docker
systemctl restart docker
sleep 3
docker version --format '{{.Server.Version}}'
"""


def instance_type_for(tier: str) -> str:
    if tier in CPU_TIERS:
        return CPU_TIERS[tier]
    if tier in GPU_TIER_KEYS:
        return SMALLEST_GPU_TYPE
    if tier.startswith("ecs."):
        return tier
    raise LeaseError(f"unknown compute tier {tier!r}; expected one of {sorted(CPU_TIERS)}, {list(GPU_TIER_KEYS)} or an ecs.* type")


def job_limits(instance_type: str) -> dict[str, float]:
    cores, gib = INSTANCE_CAPACITY.get(instance_type, (1, 2))
    return {"cpus": float(cores), "memory_mib": max(1024, gib * 1024 - 1024)}


def _password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    body = "".join(secrets.choice(alphabet) for _ in range(length - 3))
    return body + "aZ9!"


def _env(*names: str, default: str = "") -> str:
    import os

    for name in names:
        value = os.environ.get(name, "")
        if value and value.strip():
            return value.strip()
    return default


@dataclasses.dataclass(slots=True)
class AliyunSettings:
    access_key_id: str
    access_key_secret: str
    region_id: str
    vswitch_id: str
    security_group_id: str
    image_id: str
    zone_id: str = ""
    system_disk_size_gb: int = 40
    internet_max_bandwidth_out: int = 100
    #: the GPU machine image (driver + Container Toolkit on top of the CPU image, S5); empty = no GPU tier rentable
    gpu_image_id: str = ""

    def image_for(self, instance_type: str) -> str:
        """The image a machine of ``instance_type`` boots: the GPU image for GPU types (required), else the CPU one."""
        if is_gpu_type(instance_type):
            if not self.gpu_image_id:
                raise LeaseError(f"{instance_type} is a GPU type but no GPU image is configured (ALIYUN_GPU_IMAGE_ID / DEEPEVOL_API_ALIYUN_GPU_IMAGE_ID)")
            return self.gpu_image_id
        return self.image_id

    @classmethod
    def from_env(cls) -> "AliyunSettings":
        def pick(suffix: str, default: str = "") -> str:
            return _env(f"ALIYUN_{suffix}", f"DEEPEVOL_API_ALIYUN_{suffix}", default=default)

        settings = cls(
            access_key_id=pick("ACCESS_KEY_ID"),
            access_key_secret=pick("ACCESS_KEY_SECRET"),
            region_id=pick("REGION_ID", "cn-hangzhou"),
            vswitch_id=pick("VSWITCH_ID"),
            security_group_id=pick("SECURITY_GROUP_ID"),
            image_id=pick("IMAGE_ID"),
            gpu_image_id=pick("GPU_IMAGE_ID"),
            zone_id=pick("ZONE_ID"),
            system_disk_size_gb=int(pick("SYSTEM_DISK_SIZE_GB", "40") or 40),
            internet_max_bandwidth_out=int(pick("INTERNET_MAX_BANDWIDTH_OUT", "100") or 100),
        )
        missing = [
            name
            for name, value in (
                ("ACCESS_KEY_ID", settings.access_key_id),
                ("ACCESS_KEY_SECRET", settings.access_key_secret),
                ("VSWITCH_ID", settings.vswitch_id),
                ("SECURITY_GROUP_ID", settings.security_group_id),
                ("IMAGE_ID", settings.image_id),
            )
            if not value
        ]
        if missing:
            raise LeaseError(
                "Aliyun settings incomplete; set ALIYUN_<name> or DEEPEVOL_API_ALIYUN_<name> for: " + ", ".join(missing)
            )
        return settings


class EcsClient(AliyunControlClient):
    """main's control client plus the calls a lease needs; API errors keep their code and message."""

    RETRY_ATTEMPTS = 3

    def __init__(self, settings: AliyunSettings, *, timeout_seconds: float = 30.0) -> None:
        super().__init__(
            access_key_id=settings.access_key_id,
            access_key_secret=settings.access_key_secret,
            region_id=settings.region_id,
            timeout_seconds=timeout_seconds,
        )
        self.settings = settings

    # -- transport ------------------------------------------------------------------

    TRANSPORT_RETRY_ATTEMPTS = 4

    def _api(self, action: str, request_params: Mapping[str, Any]) -> Mapping[str, Any]:
        """Retry transport-level failures for every action.

        A DeleteInstance that died on ``[SSL: UNEXPECTED_EOF_WHILE_READING]`` once left a machine
        billing (sapg rehearsal, 2026-09-17; the canary hit the same on park). The ECS write actions
        this line uses are idempotent for our purpose: a repeat of a completed one returns an API
        error (not a transport error), which the caller interprets.
        """
        attempts = self.TRANSPORT_RETRY_ATTEMPTS
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._api_once(action, request_params)
            except httpx.HTTPError as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(10 * (attempt + 1))
        raise LeaseError(f"Aliyun ECS {action}: network failure: {type(last).__name__}: {last}") from last

    def instance_exists(self, instance_id: str) -> bool:
        return bool(self.describe_instance(instance_id))

    def _api_once(self, action: str, request_params: Mapping[str, Any]) -> Mapping[str, Any]:
        params: dict[str, Any] = {
            "Action": action,
            "Version": API_VERSION,
            "Format": "JSON",
            "AccessKeyId": self._access_key_id,
            "SignatureMethod": "HMAC-SHA1",
            "Timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "SignatureVersion": "1.0",
            "SignatureNonce": str(uuid.uuid4()),
            **{key: _aliyun_parameter(value) for key, value in request_params.items() if value is not None and value != ""},
        }
        canonical = "&".join(f"{_aliyun_quote(k)}={_aliyun_quote(v)}" for k, v in sorted(params.items()))
        string_to_sign = f"GET&%2F&{_aliyun_quote(canonical)}"
        digest = hmac.new(f"{self._access_key_secret}&".encode(), string_to_sign.encode(), hashlib.sha1).digest()
        params["Signature"] = base64.b64encode(digest).decode("ascii")
        with httpx.Client(timeout=self._timeout, trust_env=False) as client:
            response = client.get(self._endpoint, params=params)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            raise LeaseError(f"Aliyun ECS {action}: non-object response (HTTP {response.status_code})")
        if response.status_code >= 400 or payload.get("Code"):
            code = str(payload.get("Code") or response.status_code)
            message = str(payload.get("Message") or response.reason_phrase or "")
            raise LeaseError(f"Aliyun ECS {action} failed: {code}: {message}")
        return payload

    # -- calls ------------------------------------------------------------------------

    def run_instance(self, *, name: str, instance_type: str, password: str, image_id: str | None = None) -> str:
        """``image_id`` overrides the image the settings pick for ``instance_type`` (GPU types boot the GPU image)."""
        params: dict[str, Any] = {
            "ImageId": image_id or self.settings.image_for(instance_type),
            "InstanceType": instance_type,
            "InstanceName": name,
            "HostName": name[:64],
            "Password": password,
            "SecurityGroupId": self.settings.security_group_id,
            "VSwitchId": self.settings.vswitch_id,
            "InstanceChargeType": "PostPaid",
            "InternetChargeType": "PayByTraffic",
            "InternetMaxBandwidthOut": self.settings.internet_max_bandwidth_out,
            "SystemDisk.Category": "cloud_essd",
            "SystemDisk.Size": self.settings.system_disk_size_gb,
            "IoOptimized": "optimized",
            "Amount": 1,
            "ClientToken": str(uuid.uuid4()),
        }
        if self.settings.zone_id:
            params["ZoneId"] = self.settings.zone_id
        data = self.api("RunInstances", params)
        ids = data.get("InstanceIdSets", {}).get("InstanceIdSet") if isinstance(data.get("InstanceIdSets"), dict) else None
        if not ids:
            raise LeaseError("Aliyun ECS RunInstances returned no InstanceId")
        return str(ids[0])

    def describe_instance(self, instance_id: str) -> dict[str, Any]:
        data = self.api("DescribeInstances", {"InstanceIds": json.dumps([instance_id])})
        instances = data.get("Instances") if isinstance(data.get("Instances"), dict) else {}
        items = instances.get("Instance") if isinstance(instances.get("Instance"), list) else []
        return dict(items[0]) if items and isinstance(items[0], dict) else {}

    def wait_for_status(self, instance_id: str, statuses: set[str], *, timeout_seconds: int = 180) -> dict[str, Any]:
        deadline = time.monotonic() + max(timeout_seconds, 0)
        instance: dict[str, Any] = {}
        while True:
            instance = self.describe_instance(instance_id)
            if str(instance.get("Status") or "") in statuses or time.monotonic() >= deadline:
                return instance
            time.sleep(3)

    def allocate_public_ip(self, instance_id: str) -> str:
        data = self.api("AllocatePublicIpAddress", {"InstanceId": instance_id})
        return str(data.get("IpAddress") or "").strip()

    def run_command(self, *, instance_id: str, command: str, timeout_seconds: int = 600, name: str = "paper2code") -> str:
        data = self.api(
            "RunCommand",
            {
                "Type": "RunShellScript",
                "CommandContent": base64.b64encode(command.encode()).decode("ascii"),
                "ContentEncoding": "Base64",
                "InstanceId.1": instance_id,
                "Name": name,
                "Timeout": timeout_seconds,
            },
        )
        invoke_id = str(data.get("InvokeId") or data.get("CommandId") or "").strip()
        if not invoke_id:
            raise LeaseError("Aliyun ECS RunCommand returned no InvokeId")
        return invoke_id

    def wait_for_command(self, *, instance_id: str, invoke_id: str, timeout_seconds: int = 180) -> dict[str, Any]:
        deadline = time.monotonic() + max(timeout_seconds, 0)
        latest: dict[str, Any] = {}
        while True:
            data = self.api("DescribeInvocationResults", {"InstanceId": instance_id, "InvokeId": invoke_id})
            invocation = data.get("Invocation") if isinstance(data.get("Invocation"), dict) else {}
            results = invocation.get("InvocationResults") if isinstance(invocation.get("InvocationResults"), dict) else {}
            items = results.get("InvocationResult") if isinstance(results.get("InvocationResult"), list) else []
            if items:
                latest = dict(items[0])
                status = str(latest.get("InvocationStatus") or latest.get("Status") or "")
                if status in {"Success", "Finished", "Failed", "Stopped", "Timeout", "Cancelled"}:
                    return latest
            if time.monotonic() >= deadline:
                return latest
            time.sleep(3)

    def in_stock(self, instance_type: str) -> bool | None:
        """Whether the zone can start ``instance_type`` now (DescribeAvailableResource); None when the API does not say."""
        try:
            data = self.api(
                "DescribeAvailableResource",
                {"DestinationResource": "InstanceType", "InstanceType": instance_type, "ZoneId": self.settings.zone_id or None,
                 "InstanceChargeType": "PostPaid", "IoOptimized": "optimized"},
            )
        except LeaseError:
            return None
        zones = (data.get("AvailableZones") or {}).get("AvailableZone") or [] if isinstance(data.get("AvailableZones"), dict) else []
        for zone in zones:
            for resource in ((zone.get("AvailableResources") or {}).get("AvailableResource") or []):
                for supported in ((resource.get("SupportedResources") or {}).get("SupportedResource") or []):
                    if str(supported.get("Value") or "") == instance_type:
                        return str(supported.get("Status") or "") == "Available" and str(supported.get("StatusCategory") or "") != "WithoutStock"
        return False

    def machine_probe(self, instance_type: str) -> dict[str, Any]:
        """What the compute review point shows per machine: stock and hourly price (``compute.estimate(probe=…)``)."""
        return {"in_stock": self.in_stock(instance_type), "hourly_price_cny": self.hourly_price_cny(instance_type) or None}

    def hourly_price_cny(self, instance_type: str) -> float:
        try:
            data = self.api(
                "DescribePrice",
                {
                    "ResourceType": "instance",
                    "InstanceType": instance_type,
                    "ImageId": self.settings.image_id,
                    "SystemDisk.Category": "cloud_essd",
                    "SystemDisk.Size": self.settings.system_disk_size_gb,
                    "PriceUnit": "Hour",
                },
            )
            price = data.get("PriceInfo", {}).get("Price", {}) if isinstance(data.get("PriceInfo"), dict) else {}
            return float(price.get("TradePrice") or 0.0)
        except (LeaseError, ValueError, TypeError):
            return 0.0


def public_ip_from_instance(instance: dict[str, Any]) -> str:
    public_ip = instance.get("PublicIpAddress") if isinstance(instance.get("PublicIpAddress"), dict) else {}
    ips = public_ip.get("IpAddress") if isinstance(public_ip.get("IpAddress"), list) else []
    if ips:
        return str(ips[0] or "").strip()
    eip = instance.get("EipAddress") if isinstance(instance.get("EipAddress"), dict) else {}
    return str(eip.get("IpAddress") or "").strip() if eip.get("IpAddress") else ""


# ---------------------------------------------------------------------------
# record + backend
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LeaseRecord:
    instance_id: str
    instance_type: str
    image_id: str
    region: str
    name: str
    host: str
    port: int
    username: str
    secret_ref: str
    secret_version: str
    hourly_price_cny: float
    created_at: str
    hard_cap: str
    state: str  # provisioning | running | released | failed
    bootstrap_required: bool = False
    released_at: str | None = None
    events: list = dataclasses.field(default_factory=list)
    #: what the experiment-agent path installed on first contact (machine_bootstrap.BootstrapResult.record())
    bootstrap: dict | None = None

    @property
    def access_url(self) -> str:
        return f"ssh -p {self.port} {self.username}@{self.host}"

    def redacted(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if k != "events"}


class AliyunLeaseBackend:
    """``LeaseBackend`` for the vendored orchestrator, against :class:`EcsClient`."""

    def __init__(
        self,
        client: Any,
        *,
        secrets_store: FileRemoteComputeSecretStore,
        clock: Callable[[], datetime],
        on_record: Callable[[LeaseRecord], None] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self.client, self.secrets, self.clock, self.on_record = client, secrets_store, clock, on_record
        self.last: LeaseRecord | None = None
        self._sleep = sleep or asyncio.sleep

    async def provision(self, *, run_id: str, spec: dict[str, Any]) -> LeaseHandle:
        return await asyncio.to_thread(self._provision_sync, run_id, spec)

    def _provision_sync(self, run_id: str, spec: dict[str, Any]) -> LeaseHandle:
        instance_type = spec["instance_type"]
        password = _password()
        version = sha256(password.encode()).hexdigest()[:16]
        self.secrets.put(SECRET_REF, password, version=version)
        hourly = float(self.client.hourly_price_cny(instance_type) or 0.0)
        name = f"{INSTANCE_PREFIX}{run_id[:24]}"
        instance_id = self.client.run_instance(name=name, instance_type=instance_type, password=password)
        self.last = LeaseRecord(
            instance_id=instance_id, instance_type=instance_type, image_id=self.client.settings.image_for(instance_type),
            region=self.client.settings.region_id, name=name, host="", port=22, username="root",
            secret_ref=SECRET_REF, secret_version=version, hourly_price_cny=hourly,
            created_at=self.clock().isoformat(), hard_cap=spec["hard_cap"], state="provisioning",
        )
        if self.on_record is not None:
            self.on_record(self.last)
        try:
            instance = self.client.wait_for_status(instance_id, {"Running"}, timeout_seconds=int(spec.get("wait_seconds", 300)))
            if str(instance.get("Status") or "") != "Running":
                raise LeaseError(f"instance {instance_id} did not reach Running (status {instance.get('Status')!r})")
            host = public_ip_from_instance(instance) or self.client.allocate_public_ip(instance_id)
            if not host:
                host = public_ip_from_instance(self.client.describe_instance(instance_id))
            if not host:
                raise LeaseError(f"instance {instance_id} has no public IP")
        except Exception:
            try:
                self.client.wait_for_status(instance_id, {"Stopped", "Running"}, timeout_seconds=120)
                self.client.delete_instance(instance_id)
            except Exception as cleanup:
                logger.warning("could not delete half-provisioned instance {}: {}", instance_id, cleanup)
            raise
        self.last.host = host
        if self.on_record is not None:
            self.on_record(self.last)
        return LeaseHandle(usage_id=instance_id, resource_id=instance_id, access_url=self.last.access_url,
                           username="root", password=password, spec=dict(spec))

    async def release(self, *, run_id: str, usage_id: str, reason: str) -> None:
        deadline = time.monotonic() + DELETE_RETRY_BUDGET_SECONDS
        while True:
            try:
                await asyncio.to_thread(self.client.delete_instance, usage_id)
                return
            except LeaseError as exc:
                if "IncorrectInstanceStatus" not in str(exc) or time.monotonic() >= deadline:
                    raise
                logger.warning("DeleteInstance {} refused while the instance settles ({}); retrying in {:.0f}s", usage_id, str(exc)[-80:], DELETE_RETRY_SECONDS)
                await self._sleep(DELETE_RETRY_SECONDS)

    def authorize_key(self, instance_id: str, public_key: str) -> None:
        command = (
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh && grep -qxF " + shlex.quote(public_key)
            + " /root/.ssh/authorized_keys 2>/dev/null || echo " + shlex.quote(public_key)
            + " >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys"
        )
        invoke_id = self.client.run_command(instance_id=instance_id, command=command, timeout_seconds=120, name="paper2code-authorize-key")
        result = self.client.wait_for_command(instance_id=instance_id, invoke_id=invoke_id, timeout_seconds=180)
        status = str(result.get("InvocationStatus") or result.get("Status") or "")
        exit_code = str(result.get("ExitCode") if result.get("ExitCode") is not None else "0")
        if status not in {"Success", "Finished"} or exit_code not in {"0", "None"}:
            raise LeaseError(f"authorizing the SSH key failed: {status} {str(result.get('ErrorInfo') or '')[:200]}")


def _runtime_factory(handle: LeaseHandle) -> Any:
    from apps.v2.agent_engine.remote_relay import RemoteRuntime
    from apps.v2.agent_engine.remote_relay.transport.target import parse_access_url

    target = parse_access_url(handle.access_url, username=handle.username, password=handle.password)
    target = dataclasses.replace(target, keepalive_count_max=40)
    return RemoteRuntime(target)


Bootstrap = Callable[[Any], Any]


class PreparedLease(ExperimentLease):
    """main's ``ExperimentLease`` whose ``acquire`` also readies the machine for the experiment agent.

    ``run_experiment_on_machine`` owns the acquire → serve code → RSA sequence and expects the
    machine it gets back from ``acquire`` to carry Docker, git and RSA's two images (main's come
    from a pre-baked image). The CLI line has no such image, so the bootstrap runs here, over the
    very runtime ``wait_ready`` just proved — no second channel, no Cloud Assistant timing to guess.
    A failed bootstrap releases the machine and raises ``LeaseError``, the same contract as a
    machine that never became ready (invariant one: no exit after provisioning leaks a machine).
    """

    def __init__(self, *, bootstrap: Bootstrap | None = None, on_bootstrap: Callable[[Any], None] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bootstrap = bootstrap
        self._on_bootstrap = on_bootstrap
        self.bootstrap_result: Any = None
        #: set by ``RunLease.adopt``: the machine is already held, ``acquire`` only reconnects
        self.adopted = False

    async def acquire(self, spec: dict[str, Any]) -> Any:
        if self.adopted:
            # a resumed phase hands the flow this lease; the flow calls acquire as it always does,
            # and gets a live runtime to the machine lease.json says is still ours
            if self.handle is None or self.state is not LeaseState.READY:
                raise LeaseError(f"adopted lease is {self.state.value} without a handle; nothing to reconnect to")
            runtime = await self._bring_up(self.handle)
            self._runtime = runtime
            return runtime
        runtime = await super().acquire(spec)
        if self._bootstrap is None:
            return runtime
        try:
            self.bootstrap_result = await self._bootstrap(runtime)
        except Exception as exc:
            await self.release(reason=f"bootstrap_failed:{type(exc).__name__}")
            self.state = LeaseState.FAILED
            raise LeaseError(f"machine ready but could not be prepared for the experiment agent (released): {exc}") from exc
        if self._on_bootstrap is not None:
            self._on_bootstrap(self.bootstrap_result)
        return runtime


# ---------------------------------------------------------------------------
# the run's lease
# ---------------------------------------------------------------------------


class RunLease:
    """Durable ``lease.json``: acquire / release / hard cap; the executor's access details."""

    def __init__(
        self,
        run_dir: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        events: Callable[..., Any] | None = None,
        client_factory: Callable[[], Any] | None = None,
        runtime_factory: Callable[[LeaseHandle], Any] | None = None,
        ready_timeout: float = READY_TIMEOUT_SECONDS,
        bring_up_backoff: float = BRING_UP_BACKOFF,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.events = events
        self.path = self.run_dir / "lease.json"
        self.secrets_path = self.run_dir / "secrets" / "remote-compute.json"
        self._client_factory = client_factory or (lambda: EcsClient(AliyunSettings.from_env()))
        self._client: Any = None
        self._runtime_factory = runtime_factory or _runtime_factory
        self._ready_timeout = ready_timeout
        self._bring_up_backoff = bring_up_backoff
        self._backend: AliyunLeaseBackend | None = None

    # -- record ---------------------------------------------------------------------

    @property
    def record(self) -> LeaseRecord | None:
        if not self.path.exists():
            return None
        return LeaseRecord(**json.loads(self.path.read_text(encoding="utf-8")))

    def _save(self, record: LeaseRecord, note: str | None = None) -> None:
        if note:
            record.events.append({"at": self.clock().isoformat(), "note": note})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dataclasses.asdict(record), indent=2) + "\n", encoding="utf-8")
        if self.events is not None:
            self.events("lease." + record.state, instance_id=record.instance_id, host=record.host, note=note or "")

    @property
    def active(self) -> bool:
        record = self.record
        return record is not None and record.state in {"running", "provisioning", "release_failed"}

    # -- the account, before any machine ---------------------------------------------------

    def client(self) -> Any:
        """The ECS client (built once); the compute phase asks it about stock, prices and the GPU image."""
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    @property
    def gpu_image_configured(self) -> bool:
        return bool(getattr(getattr(self.client(), "settings", None), "gpu_image_id", ""))

    def machine_probe(self, instance_type: str) -> dict[str, Any]:
        probe = getattr(self.client(), "machine_probe", None)
        return dict(probe(instance_type) or {}) if probe is not None else {}

    # -- backend ----------------------------------------------------------------------

    def backend(self) -> AliyunLeaseBackend:
        if self._backend is None:
            client = self.client()
            self.secrets_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self.secrets_path.exists():
                self.secrets_path.touch(mode=0o600)
                self.secrets_path.write_text('{"schema_version": "%s", "secrets": {}}\n' % SECRET_STORE_SCHEMA_VERSION)
            store = FileRemoteComputeSecretStore(self.secrets_path.resolve())
            self._backend = AliyunLeaseBackend(client, secrets_store=store, clock=self.clock, on_record=lambda r: self._save(r, "created" if not r.host else "address known"))
        return self._backend

    # -- lifecycle ---------------------------------------------------------------------

    async def acquire(self, *, run_id: str, instance_type: str, hard_cap_seconds: float) -> LeaseRecord:
        if self.active:
            raise LeaseError("this run already holds a machine (lease.json is running); release it first")
        backend = self.backend()
        hard_cap = (self.clock() + timedelta(seconds=hard_cap_seconds)).isoformat()
        lease = ExperimentLease(
            run_id=run_id, backend=backend, runtime_factory=self._runtime_factory,
            hard_cap_seconds=hard_cap_seconds, ready_timeout=self._ready_timeout, bring_up_backoff=self._bring_up_backoff,
        )
        spec = {"instance_type": instance_type, "hard_cap": hard_cap}
        try:
            runtime = await lease.acquire(spec)
            await runtime.close()
        except LeaseError as exc:
            record = backend.last
            if record is not None:
                record.state = "released" if lease.handle is not None else "failed"
                record.released_at = self.clock().isoformat() if record.state == "released" else None
                self._save(record, f"acquire failed: {exc}")
            raise
        record = backend.last
        assert record is not None
        from apps.v2.agent.paper2code.execution.remote_daemon import generate_keypair

        _private, public_key = await asyncio.to_thread(generate_keypair, self.secrets_path.parent)
        await asyncio.to_thread(backend.authorize_key, record.instance_id, public_key)
        record.state = "running"
        self._save(record, "ready, key authorized")
        return record

    # -- the experiment-agent path (PLAN-3 item 4a) -------------------------------------------
    #
    # ``run_experiment_on_machine`` wants a PLANNED lease it can ``acquire`` itself, and it releases
    # (or deliberately holds) the machine by main's release policy. The line's part is to hand it
    # one whose backend is the ECS client above, whose runtime is main's remote_relay, whose
    # ``acquire`` also bootstraps the machine, and whose every state change lands in ``lease.json``
    # — so ``status`` and the ``release`` backstop keep working from the file alone.

    @staticmethod
    def spec_for(instance_type: str, hard_cap_seconds: float, *, clock: Callable[[], datetime] | None = None, **extra: Any) -> dict[str, Any]:
        """The ``spec`` dict the flow passes to ``lease.acquire``: what the backend's ``provision`` reads plus the flow's own keys."""
        now = (clock or (lambda: datetime.now(UTC)))()
        spec = {"instance_type": instance_type, "hard_cap": (now + timedelta(seconds=hard_cap_seconds)).isoformat()}
        spec.update(extra)
        return spec

    def experiment_lease(self, *, run_id: str, hard_cap_seconds: float, bootstrap: Bootstrap | None | bool = True) -> PreparedLease:
        """A PLANNED lease for ``run_experiment_on_machine``; ``bootstrap=False`` skips the machine preparation (tests, pre-baked images)."""
        if self.active:
            raise LeaseError("this run already holds a machine (lease.json is running); release it first")
        if bootstrap is True:
            from apps.v2.agent.paper2code.execution.machine_bootstrap import bootstrap_machine

            bootstrap = bootstrap_machine
        elif bootstrap is False:
            bootstrap = None
        return PreparedLease(
            run_id=run_id, backend=self.backend(), runtime_factory=self._runtime_factory,
            hard_cap_seconds=hard_cap_seconds, ready_timeout=self._ready_timeout, bring_up_backoff=self._bring_up_backoff,
            event_sink=self._lease_event, bootstrap=bootstrap, on_bootstrap=self._record_bootstrap,
        )

    def adopt(self, *, run_id: str, hard_cap_seconds: float | None = None) -> PreparedLease:
        """Re-attach to the machine ``lease.json`` says this run still holds (after a review point held it).

        The password comes back from the run's secret store; the returned lease is READY with its handle set,
        never acquired again — ``release()`` works on it, ``run_experiment_on_machine`` must not (it acquires).
        """
        record = self.record
        if record is None or record.state not in {"running", "provisioning", "release_failed"}:
            raise LeaseError("no machine to adopt: lease.json does not show a held machine")
        self.backend()  # ensures the secret store exists
        store = FileRemoteComputeSecretStore(self.secrets_path.resolve())
        password = store.resolve(record.secret_ref, version=record.secret_version)
        remaining = hard_cap_seconds
        if remaining is None:
            remaining = max((datetime.fromisoformat(record.hard_cap) - self.clock()).total_seconds(), 1.0)
        lease = PreparedLease(
            run_id=run_id, backend=self.backend(), runtime_factory=self._runtime_factory,
            hard_cap_seconds=remaining, ready_timeout=self._ready_timeout, bring_up_backoff=self._bring_up_backoff,
            event_sink=self._lease_event, bootstrap=None,
        )
        lease.handle = LeaseHandle(
            usage_id=record.instance_id, resource_id=record.instance_id, access_url=record.access_url,
            username=record.username, password=password, spec={"instance_type": record.instance_type, "hard_cap": record.hard_cap},
        )
        lease.state = LeaseState.READY
        lease.adopted = True
        self._backend.last = record  # type: ignore[union-attr]
        return lease

    def _lease_event(self, event: LeaseEvent) -> None:
        record = self.backend().last or self.record
        if record is None:
            return
        state, detail = event.state, event.detail
        if state is LeaseState.READY:
            record.state = "running"
            self._save(record, "ready")
        elif state is LeaseState.RELEASED:
            record.state = "released"
            record.released_at = self.clock().isoformat()
            self._save(record, f"released: {detail}")
        elif state is LeaseState.FAILED:
            if record.state == "released":
                self._save(record, f"failed after release: {detail}")
            elif detail.startswith("release_failed"):
                record.state = "release_failed"
                self._save(record, detail[:200])
            elif record.state == "provisioning" and record.host:
                # the machine exists and no release succeeded: say so, never pretend it is gone
                record.state = "release_failed"
                self._save(record, f"failed with the machine possibly still there: {detail[:160]}")
            else:
                record.state = "failed"
                self._save(record, f"failed: {detail[:160]}")

    def _record_bootstrap(self, result: Any) -> None:
        record = self.backend().last or self.record
        if record is None:
            return
        record.bootstrap = result.record() if hasattr(result, "record") else dict(result or {})
        self._save(record, "machine prepared for the experiment agent")

    async def release(self, reason: str) -> LeaseRecord | None:
        """Delete the machine. A failed delete is recorded as ``release_failed`` — never as released."""
        record = self.record
        if record is None or record.state in {"released", "failed"}:
            return record
        backend = self.backend()
        try:
            await backend.release(run_id="", usage_id=record.instance_id, reason=reason)
        except Exception as exc:
            record.state = "release_failed"
            self._save(record, f"release failed ({reason}): {type(exc).__name__}: {str(exc)[:160]}")
            raise LeaseError(f"DeleteInstance {record.instance_id} failed: {exc}") from exc
        record.state = "released"
        record.released_at = self.clock().isoformat()
        self._save(record, f"released: {reason}")
        return record

    async def force_release(self, reason: str = "manual release") -> dict[str, Any]:
        """The backstop: ask the account whether the instance still exists and delete it if so,
        whatever ``lease.json`` claims (a release may have died after the API call went through,
        or been recorded before it did)."""
        record = self.record
        if record is None:
            return {"instance_id": None, "existed": False, "deleted": False, "state": None}
        client = self.backend().client
        exists = await asyncio.to_thread(client.instance_exists, record.instance_id)
        deleted = False
        if exists:
            await asyncio.to_thread(client.delete_instance, record.instance_id)
            deleted = True
        if record.state != "released":
            record.state = "released"
            record.released_at = self.clock().isoformat()
        self._save(record, f"force release ({reason}): {'deleted' if deleted else 'instance already gone'}")
        return {"instance_id": record.instance_id, "existed": exists, "deleted": deleted, "state": record.state}

    def seconds_to_cap(self) -> float | None:
        record = self.record
        if record is None or record.state not in {"running", "provisioning"}:
            return None
        return (datetime.fromisoformat(record.hard_cap) - self.clock()).total_seconds()

    def cap_reached(self) -> bool:
        remaining = self.seconds_to_cap()
        return remaining is not None and remaining <= 0

    def mark_bootstrap(self, required: bool) -> None:
        record = self.record
        if record is None:
            return
        record.bootstrap_required = required
        self._save(record, "bootstrap required (Docker installed on first contact)" if required else "docker present")


__all__ = [
    "CPU_TIERS",
    "GPU_CATALOG",
    "GPU_TIER_KEYS",
    "MINIMAL_BOOTSTRAP",
    "AliyunLeaseBackend",
    "AliyunSettings",
    "EcsClient",
    "LeaseError",
    "LeaseRecord",
    "PreparedLease",
    "RunLease",
    "instance_type_for",
    "is_gpu_type",
    "job_limits",
    "public_ip_from_instance",
]
