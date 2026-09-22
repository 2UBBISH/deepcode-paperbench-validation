"""Agent-owned Remote Compute effects with bounded SSH and provider routing."""

from __future__ import annotations

import logging
import time

import base64
import hashlib
import hmac
import json
import math
import posixpath
import shlex
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote

from apps.v2.reliability.circuit import CircuitPolicy, CircuitRegistry, DependencyUnavailable
from apps.v2.reliability.http_dependency import guarded_request
from apps.v2.reliability.control_response import read_control_response
from apps.v2.reliability.deadline_transport import DeadlineHTTPTransport
from apps.v2.reliability.ssh_channel import execute_command, connection_deadline

from .commands import RemoteComputeCommandError, RemoteComputeProvider
from .models import RemoteComputeAction, RemoteComputeCommand, RemoteEnvironmentDescriptor
from .provider_operations import (
    ProviderOperationRecord,
    ProviderResolution,
    ProviderResolutionState,
)
from .provider_operation_runtime import ProviderPreEffectRetryableError
from .transfers import DurableSftpTransferExecutor

try:  # pragma: no cover - deployment checks enforce this dependency
    import httpx
    import paramiko
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    paramiko = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_ALIYUN_CONNECT_ATTEMPTS = 3
_ALIYUN_CONNECT_BACKOFF_SECONDS = 1.0


SSH_CIRCUITS = CircuitRegistry(CircuitPolicy.from_env())


class AutoDLControl(Protocol):
    def power_on(self, instance_uuid: str, *, start_command: str | None = None) -> None: ...
    def power_off(self, instance_uuid: str) -> None: ...
    def release(self, instance_uuid: str) -> None: ...
    def get_status(self, instance_uuid: str) -> str: ...


class AliyunControl(Protocol):
    def start_instance(self, instance_id: str) -> None: ...
    def stop_instance(self, instance_id: str) -> None: ...
    def delete_instance(self, instance_id: str) -> None: ...
    def api(self, action: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AutoDLControlClient:
    api_key: str
    base_url: str = "https://api.autodl.com"
    timeout_seconds: float = 30.0

    def power_on(self, instance_uuid: str, *, start_command: str | None = None) -> None:
        body: dict[str, Any] = {"instance_uuid": instance_uuid, "payload": "gpu"}
        if start_command:
            body["start_command"] = start_command
        self._request("/api/v1/dev/instance/pro/power_on", body)

    def power_off(self, instance_uuid: str) -> None:
        self._request(
            "/api/v1/dev/instance/pro/power_off",
            {"instance_uuid": instance_uuid},
        )

    def release(self, instance_uuid: str) -> None:
        self._request(
            "/api/v1/dev/instance/pro/release",
            {"instance_uuid": instance_uuid},
        )

    def gpu_stock(self, region: str, filters: Mapping[str, Any]) -> list[dict[str, Any]]:
        payload = self._call(
            "POST",
            "/api/v1/dev/machine/region/gpu_stock",
            body={"region_sign": region, **dict(filters)},
        )
        rows = payload.get("data")
        return [dict(item) for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []

    def create_instance(self, body: Mapping[str, Any]) -> str:
        payload = self._call(
            "POST",
            "/api/v1/dev/instance/pro/create",
            body=body,
        )
        instance_uuid = str(payload.get("data") or "").strip()
        if not instance_uuid:
            raise RemoteComputeCommandError("REMOTE_PROVIDER_RESPONSE_INVALID")
        return instance_uuid

    def get_status(self, instance_uuid: str) -> str:
        payload = self._call(
            "GET",
            "/api/v1/dev/instance/pro/status",
            params={"instance_uuid": instance_uuid},
        )
        return str(payload.get("data") or "").strip().lower()

    def get_snapshot(self, instance_uuid: str) -> dict[str, Any]:
        payload = self._call(
            "GET",
            "/api/v1/dev/instance/pro/snapshot",
            params={"instance_uuid": instance_uuid},
        )
        data = payload.get("data")
        return dict(data) if isinstance(data, Mapping) else {}

    def list_instances(self, *, page: int = 1, page_size: int = 100) -> list[dict[str, Any]]:
        payload = self._call(
            "POST",
            "/api/v1/dev/instance/pro/list",
            body={"page_index": page, "page_size": page_size},
        )
        data = payload.get("data")
        rows = data.get("list") if isinstance(data, Mapping) else None
        return [dict(item) for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []

    def _request(self, path: str, body: Mapping[str, Any]) -> None:
        self._call("POST", path, body=body)

    def _call(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx is required for AutoDL controls")
        if not self.api_key:
            raise RemoteComputeCommandError("REMOTE_PROVIDER_NOT_CONFIGURED")
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, trust_env=False,
                transport=DeadlineHTTPTransport(timeout_seconds=self.timeout_seconds),
            ) as client:
                scope = hashlib.sha256(f"{self.api_key}\0{path}".encode()).hexdigest()
                response = guarded_request(
                    f"autodl:{self.base_url}:{scope}",
                    lambda: _read_provider_response(
                        client, method, f"{self.base_url.rstrip('/')}{path}",
                        json=None if body is None else dict(body),
                        params=None if params is None else dict(params),
                        headers={
                            "Authorization": self.api_key,
                            "Content-Type": "application/json",
                        },
                    ),
                )
        except DependencyUnavailable as exc:
            raise RemoteComputeCommandError("REMOTE_PROVIDER_CIRCUIT_OPEN", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise RemoteComputeCommandError(
                "REMOTE_PROVIDER_UNAVAILABLE", retryable=True
            ) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise RemoteComputeCommandError(
                "REMOTE_PROVIDER_UNAVAILABLE", retryable=True
            )
        if response.status_code >= 400:
            raise RemoteComputeCommandError("REMOTE_PROVIDER_REJECTED")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteComputeCommandError(
                "REMOTE_PROVIDER_UNAVAILABLE", retryable=True
            ) from exc
        if not isinstance(payload, Mapping) or str(payload.get("code") or "") != "Success":
            raise RemoteComputeCommandError("REMOTE_PROVIDER_REJECTED")
        return payload


class AliyunControlClient:
    def __init__(
        self,
        *,
        access_key_id: str,
        access_key_secret: str,
        region_id: str,
        timeout_seconds: float = 30.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not access_key_id or not access_key_secret or not region_id:
            raise ValueError("Aliyun control credentials and region are required")
        self._access_key_id = access_key_id
        self._access_key_secret = access_key_secret
        self._region_id = region_id
        self._timeout = timeout_seconds
        self._sleep = sleeper
        self._endpoint = f"https://ecs.{region_id}.aliyuncs.com/"

    def start_instance(self, instance_id: str) -> None:
        self._request("StartInstance", instance_id)

    def stop_instance(self, instance_id: str) -> None:
        self._request(
            "StopInstance",
            instance_id,
            extra={"StoppedMode": "StopCharging", "ForceStop": "true"},
        )

    def delete_instance(self, instance_id: str) -> None:
        self._request("DeleteInstance", instance_id, extra={"Force": "true"})

    def api(self, action: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        return self._api(action, {"RegionId": self._region_id, **dict(params or {})})

    def _request(
        self,
        action: str,
        instance_id: str,
        *,
        extra: Mapping[str, str] | None = None,
    ) -> None:
        self._api(
            action,
            {
                "RegionId": self._region_id,
                "InstanceId": instance_id,
                **dict(extra or {}),
            },
        )

    def _api(self, action: str, request_params: Mapping[str, Any]) -> Mapping[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx is required for Aliyun controls")
        params: dict[str, Any] = {
            "Action": action,
            "Version": "2014-05-26",
            "Format": "JSON",
            "AccessKeyId": self._access_key_id,
            "SignatureMethod": "HMAC-SHA1",
            "Timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "SignatureVersion": "1.0",
            "SignatureNonce": str(uuid.uuid4()),
            **{
                key: _aliyun_parameter(value)
                for key, value in request_params.items()
                if value is not None and value != ""
            },
        }
        canonical = "&".join(
            f"{_aliyun_quote(key)}={_aliyun_quote(value)}"
            for key, value in sorted(params.items())
        )
        string_to_sign = f"GET&%2F&{_aliyun_quote(canonical)}"
        digest = hmac.new(
            f"{self._access_key_secret}&".encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        params["Signature"] = base64.b64encode(digest).decode("ascii")
        scope = hashlib.sha256(f"{self._access_key_id}\0{action}".encode()).hexdigest()
        try:
            with httpx.Client(
                timeout=self._timeout, trust_env=False,
                transport=DeadlineHTTPTransport(timeout_seconds=self._timeout),
            ) as client:
                # A failure before the request leaves the socket (TCP/TLS
                # connect) has no provider-side effect, so it is safe to try
                # again — the ECS endpoint's TLS handshake is flaky enough
                # from some networks that one attempt lost real rents.
                for attempt in range(_ALIYUN_CONNECT_ATTEMPTS):
                    try:
                        response = guarded_request(
                            f"aliyun:{self._endpoint}:{scope}",
                            lambda: _read_provider_response(client, "GET", self._endpoint, params=params),
                        )
                        break
                    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                        if attempt + 1 >= _ALIYUN_CONNECT_ATTEMPTS:
                            raise
                        logger.warning(
                            "aliyun %s connect failed (attempt %s/%s): %s",
                            action, attempt + 1, _ALIYUN_CONNECT_ATTEMPTS, exc,
                        )
                        self._sleep(_ALIYUN_CONNECT_BACKOFF_SECONDS * (attempt + 1))
        except DependencyUnavailable as exc:
            raise RemoteComputeCommandError("REMOTE_PROVIDER_CIRCUIT_OPEN", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise RemoteComputeCommandError(
                "REMOTE_PROVIDER_UNAVAILABLE", retryable=True
            ) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise RemoteComputeCommandError(
                "REMOTE_PROVIDER_UNAVAILABLE", retryable=True
            )
        if response.status_code >= 400:
            # Aliyun explains rejections in the body (Code/Message); keep the
            # error contract stable but do not lose the diagnosis.
            logger.warning("aliyun %s rejected: status=%s body=%s", action, response.status_code, response.text[:512])
            raise RemoteComputeCommandError("REMOTE_PROVIDER_REJECTED")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteComputeCommandError(
                "REMOTE_PROVIDER_UNAVAILABLE", retryable=True
            ) from exc
        if not isinstance(payload, Mapping) or payload.get("Code"):
            logger.warning("aliyun %s rejected: body=%s", action, response.text[:512])
            raise RemoteComputeCommandError("REMOTE_PROVIDER_REJECTED")
        return payload


def _read_provider_response(client: Any, method: str, url: str, **kwargs: Any) -> Any:
    # Read inside guarded_request so stream failures count against the operation
    # circuit; keep admission failure distinct from an uncertain cloud effect.
    headers = {**kwargs.pop("headers", {}), "Accept-Encoding": "identity"}
    with client.stream(method, url, headers=headers, **kwargs) as response:
        def validate(document):
            try:
                payload = document.json()
            except ValueError as exc:
                raise httpx.RemoteProtocolError("Cloud control response is not JSON") from exc
            if not isinstance(payload, Mapping):
                raise httpx.RemoteProtocolError("Cloud control response is not an object")

        return read_control_response(
            response, maximum_bytes=1024 * 1024,
            accepted_statuses=frozenset(range(200, 400)),
            validate_success=validate,
        )


def _aliyun_quote(value: Any) -> str:
    return quote(str(value), safe="~").replace("+", "%20").replace("*", "%2A")


def _aliyun_parameter(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _migration_command_from_operation(
    operation: ProviderOperationRecord,
) -> RemoteComputeCommand:
    payload = operation.request

    def text(name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str):
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_CHECKPOINT_INVALID"
            )
        return value

    def port(name: str) -> int:
        value = payload.get(name)
        if value is None:
            return 22
        if isinstance(value, bool):
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_CHECKPOINT_INVALID"
            )
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_CHECKPOINT_INVALID"
            ) from exc
        if not 1 <= parsed <= 65_535:
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_CHECKPOINT_INVALID"
            )
        return parsed

    try:
        command_generation = int(payload["command_generation"])
        expected_resource_version = int(payload["expected_resource_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteComputeCommandError(
            "REMOTE_MIGRATION_CHECKPOINT_INVALID"
        ) from exc
    return RemoteComputeCommand(
        command_id=operation.operation_id,
        resource_id=operation.resource_id,
        resource_tid=operation.resource_tid,
        owner_uid=operation.owner_uid,
        action=RemoteComputeAction.MIGRATE,
        command_generation=command_generation,
        expected_resource_version=expected_resource_version,
        rid=operation.rid,
        target_resource_id=operation.target_resource_id,
        idempotency_key=operation.idempotency_key,
        request_fingerprint=operation.request_sha256,
        provider=operation.provider,
        external_resource_id=operation.external_resource_id,
        access_host=text("access_host"),
        access_port=port("access_port"),
        access_username=text("access_username"),
        remote_root=text("remote_root"),
        secret_ref=text("secret_ref"),
        secret_version=text("secret_version"),
        target_access_host=text("target_access_host"),
        target_access_port=port("target_access_port"),
        target_access_username=text("target_access_username"),
        target_remote_root=text("target_remote_root"),
        target_secret_ref=text("target_secret_ref"),
        target_secret_version=text("target_secret_version"),
    )


class StrictRemoteComputeProvider(RemoteComputeProvider):
    """Execute only the six frozen V2 commands against configured authorities."""

    def __init__(
        self,
        *,
        autodl_factory: Callable[[], AutoDLControl],
        aliyun_factory: Callable[[], AliyunControl],
        known_hosts: str,
        connect_timeout_seconds: float = 10.0,
        command_timeout_seconds: float = 300.0,
        migration_timeout_seconds: float = 900.0,
        transfer_executor: DurableSftpTransferExecutor | None = None,
    ) -> None:
        if paramiko is None:
            raise RuntimeError("paramiko is required for Remote Compute effects")
        if not known_hosts or connect_timeout_seconds <= 0:
            raise ValueError("known_hosts and a positive SSH timeout are required")
        if not math.isfinite(command_timeout_seconds) or not 1 <= command_timeout_seconds <= 900:
            raise ValueError("SSH command deadline must be between 1 and 900 seconds")
        if not math.isfinite(migration_timeout_seconds) or not 1 <= migration_timeout_seconds <= 3600:
            raise ValueError("SSH migration deadline must be between 1 and 3600 seconds")
        self._migration_timeout = migration_timeout_seconds
        self._command_timeout = command_timeout_seconds
        self._autodl_factory = autodl_factory
        self._aliyun_factory = aliyun_factory
        self._known_hosts = known_hosts
        self._timeout = connect_timeout_seconds
        self._transfer_executor = transfer_executor

    def apply(
        self,
        command: RemoteComputeCommand,
        *,
        secret: str,
        target_secret: str | None = None,
    ) -> Mapping[str, Any]:
        if command.action in {
            RemoteComputeAction.POWER_ON,
            RemoteComputeAction.POWER_OFF,
            RemoteComputeAction.RELEASE,
        }:
            self._lifecycle(command)
            return {"provider": command.provider, "external_resource_id": command.external_resource_id}
        if command.action is RemoteComputeAction.CLEANUP:
            removed = self._cleanup(command, secret=secret)
            return {"removed_count": removed}
        if command.action is RemoteComputeAction.MIGRATE:
            if target_secret is None:
                raise RemoteComputeCommandError("REMOTE_TARGET_SECRET_REQUIRED")
            return self._migrate(
                command,
                source_secret=secret,
                target_secret=target_secret,
            )
        if command.action is RemoteComputeAction.APPLY_QUOTA:
            return {"resource_policy": dict(command.resource_policy or {})}
        raise RemoteComputeCommandError("REMOTE_ACTION_UNSUPPORTED")

    def reconcile(
        self,
        operation: ProviderOperationRecord,
        *,
        secret: str,
        target_secret: str | None = None,
    ) -> ProviderResolution:
        try:
            action = RemoteComputeAction(operation.action)
        except ValueError:
            return ProviderResolution(
                ProviderResolutionState.FAILED,
                {"reason": "unsupported-action"},
                "REMOTE_ACTION_UNSUPPORTED",
            )
        if action is RemoteComputeAction.APPLY_QUOTA:
            policy = operation.request.get("resource_policy")
            return ProviderResolution(
                ProviderResolutionState.APPLIED,
                {"resource_policy": dict(policy) if isinstance(policy, Mapping) else {}},
            )
        if action is RemoteComputeAction.MIGRATE:
            if self._transfer_executor is None:
                return ProviderResolution(
                    ProviderResolutionState.FAILED,
                    {"reason": "durable-transfer-runtime-is-not-configured"},
                    "REMOTE_MIGRATION_DURABILITY_REQUIRED",
                )
            if target_secret is None:
                return ProviderResolution(
                    ProviderResolutionState.FAILED,
                    {"reason": "target-secret-is-not-available"},
                    "REMOTE_TARGET_SECRET_REQUIRED",
                )
            try:
                receipt = self._migrate(
                    _migration_command_from_operation(operation),
                    source_secret=secret,
                    target_secret=target_secret,
                )
            except RemoteComputeCommandError as exc:
                return ProviderResolution(
                    (
                        ProviderResolutionState.UNKNOWN
                        if exc.retryable
                        else ProviderResolutionState.FAILED
                    ),
                    {"error_code": exc.code},
                    exc.code,
                )
            except Exception as exc:
                return ProviderResolution(
                    ProviderResolutionState.UNKNOWN,
                    {"runtime_error": type(exc).__name__},
                    "REMOTE_MIGRATION_RECONCILIATION_UNAVAILABLE",
                )
            return ProviderResolution(
                ProviderResolutionState.APPLIED,
                dict(receipt),
            )
        if action not in {
            RemoteComputeAction.POWER_ON,
            RemoteComputeAction.POWER_OFF,
            RemoteComputeAction.RELEASE,
        }:
            return ProviderResolution(
                ProviderResolutionState.UNKNOWN,
                {"reason": "provider-effect-is-not-queryable"},
                "REMOTE_PROVIDER_RESULT_UNKNOWN",
            )
        external_id = operation.external_resource_id.strip()
        provider = operation.provider.strip().lower()
        try:
            if provider in {"autodl", "autodl_pro"}:
                actual = self._autodl_factory().get_status(external_id)
            elif provider in {"aliyun", "aliyun_ecs"}:
                payload = self._aliyun_factory().api(
                    "DescribeInstances",
                    {"InstanceIds": [external_id]},
                )
                instances = payload.get("Instances")
                rows = instances.get("Instance") if isinstance(instances, Mapping) else None
                if not isinstance(rows, list) or not rows:
                    actual = "not_found"
                else:
                    actual = str(rows[0].get("Status") or "").lower()
            else:
                return ProviderResolution(
                    ProviderResolutionState.FAILED,
                    {"provider": operation.provider},
                    "REMOTE_PROVIDER_UNSUPPORTED",
                )
        except Exception as exc:
            return ProviderResolution(
                ProviderResolutionState.UNKNOWN,
                {"query_error": type(exc).__name__},
                "REMOTE_PROVIDER_RECONCILIATION_UNAVAILABLE",
            )
        expected = {
            RemoteComputeAction.POWER_ON: {"running", "power_on", "started"},
            RemoteComputeAction.POWER_OFF: {"stopped", "shutdown", "power_off"},
            RemoteComputeAction.RELEASE: {"released", "deleted", "not_found"},
        }[action]
        state = (
            ProviderResolutionState.APPLIED
            if actual in expected
            else ProviderResolutionState.NOT_APPLIED
        )
        return ProviderResolution(
            state,
            {
                "actual_status": actual,
                "external_resource_id": external_id,
                "provider": operation.provider,
            },
        )

    def compensate(
        self,
        operation: ProviderOperationRecord,
        *,
        secret: str,
        target_secret: str | None = None,
    ) -> Mapping[str, Any]:
        del secret, target_secret
        try:
            action = RemoteComputeAction(operation.action)
        except ValueError as exc:
            raise RemoteComputeCommandError("REMOTE_ACTION_UNSUPPORTED") from exc
        if action is RemoteComputeAction.APPLY_QUOTA:
            return {"compensated": True, "mode": "NO_EXTERNAL_EFFECT"}
        inverse = {
            RemoteComputeAction.POWER_ON: RemoteComputeAction.POWER_OFF,
            RemoteComputeAction.POWER_OFF: RemoteComputeAction.POWER_ON,
        }.get(action)
        if inverse is None:
            raise RemoteComputeCommandError("REMOTE_COMPENSATION_UNSUPPORTED")
        external_id = operation.external_resource_id.strip()
        provider = operation.provider.strip().lower()
        if provider in {"autodl", "autodl_pro"}:
            client = self._autodl_factory()
            try:
                actual = client.get_status(external_id).strip().lower()
            except RemoteComputeCommandError as exc:
                if not exc.retryable:
                    raise
                raise ProviderPreEffectRetryableError(exc.code) from exc
            except Exception as exc:
                raise ProviderPreEffectRetryableError() from exc
            expected = (
                {"running", "power_on", "started"}
                if inverse is RemoteComputeAction.POWER_ON
                else {"stopped", "shutdown", "power_off"}
            )
            if actual in expected:
                return {
                    "compensated": True,
                    "external_resource_id": external_id,
                    "provider": operation.provider,
                    "actual_status": actual,
                    "replayed": True,
                }
            if inverse is RemoteComputeAction.POWER_ON:
                client.power_on(external_id)
            else:
                client.power_off(external_id)
        elif provider in {"aliyun", "aliyun_ecs"}:
            client = self._aliyun_factory()
            try:
                payload = client.api(
                    "DescribeInstances",
                    {"InstanceIds": [external_id]},
                )
            except RemoteComputeCommandError as exc:
                if not exc.retryable:
                    raise
                raise ProviderPreEffectRetryableError(exc.code) from exc
            except Exception as exc:
                raise ProviderPreEffectRetryableError() from exc
            instances = payload.get("Instances")
            rows = instances.get("Instance") if isinstance(instances, Mapping) else None
            actual = (
                "not_found"
                if not isinstance(rows, list) or not rows
                else str(rows[0].get("Status") or "").lower()
            )
            expected = (
                {"running", "starting"}
                if inverse is RemoteComputeAction.POWER_ON
                else {"stopped", "stopping"}
            )
            if actual in expected:
                return {
                    "compensated": True,
                    "external_resource_id": external_id,
                    "provider": operation.provider,
                    "actual_status": actual,
                    "replayed": True,
                }
            if inverse is RemoteComputeAction.POWER_ON:
                client.start_instance(external_id)
            else:
                client.stop_instance(external_id)
        else:
            raise RemoteComputeCommandError("REMOTE_PROVIDER_UNSUPPORTED")
        return {
            "compensated": True,
            "external_resource_id": external_id,
            "provider": operation.provider,
        }

    def scan_environment(
        self,
        descriptor: RemoteEnvironmentDescriptor,
        *,
        secret: str,
    ) -> Mapping[str, Any]:
        """Run a bounded, read-only inventory against one Product descriptor."""

        encoded = base64.b64encode(_ENVIRONMENT_SCAN_SCRIPT.encode("utf-8")).decode("ascii")
        command = (
            "python3 -c "
            + shlex.quote(
                "import base64;exec(base64.b64decode(" + repr(encoded) + ").decode('utf-8'))"
            )
            + " -- "
            + shlex.quote(descriptor.remote_root)
        )
        try:
            with self._ssh_connection(
                host=descriptor.host, port=descriptor.port,
                username=descriptor.username, secret=secret,
            ) as client:
                raw, error, exit_code = execute_command(
                    client, command, maximum=_MAX_SCAN_OUTPUT_BYTES, timeout_seconds=75.0,
                )
        except Exception as exc:
            raise RemoteComputeCommandError(
                "REMOTE_ENVIRONMENT_SCAN_UNAVAILABLE", retryable=True
            ) from exc
        if exit_code != 0:
            raise RemoteComputeCommandError(
                "REMOTE_ENVIRONMENT_SCAN_FAILED",
                retryable=bool(error),
            )
        if len(raw) > _MAX_SCAN_OUTPUT_BYTES:
            raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_LIMIT_EXCEEDED")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID") from exc
        return _validate_environment_scan(payload)

    def _lifecycle(self, command: RemoteComputeCommand) -> None:
        external_id = command.external_resource_id.strip()
        if not external_id:
            raise RemoteComputeCommandError("REMOTE_EXTERNAL_ID_MISSING")
        provider = command.provider.strip().lower()
        if provider in {"autodl", "autodl_pro"}:
            client = self._autodl_factory()
            actual = client.get_status(external_id).strip().lower()
            expected = {
                RemoteComputeAction.POWER_ON: {"running", "power_on", "started"},
                RemoteComputeAction.POWER_OFF: {"stopped", "shutdown", "power_off"},
                RemoteComputeAction.RELEASE: {"released", "deleted", "not_found"},
            }[command.action]
            if actual in expected:
                return
            if command.action is RemoteComputeAction.POWER_ON:
                client.power_on(external_id)
            elif command.action is RemoteComputeAction.POWER_OFF:
                client.power_off(external_id)
            else:
                client.release(external_id)
            return
        if provider in {"aliyun", "aliyun_ecs"}:
            client = self._aliyun_factory()
            payload = client.api(
                "DescribeInstances",
                {"InstanceIds": [external_id]},
            )
            instances = payload.get("Instances")
            rows = instances.get("Instance") if isinstance(instances, Mapping) else None
            actual = (
                "not_found"
                if not isinstance(rows, list) or not rows
                else str(rows[0].get("Status") or "").strip().lower()
            )
            expected = {
                RemoteComputeAction.POWER_ON: {"running", "starting"},
                RemoteComputeAction.POWER_OFF: {"stopped", "stopping"},
                RemoteComputeAction.RELEASE: {"not_found"},
            }[command.action]
            if actual in expected:
                return
            if command.action is RemoteComputeAction.POWER_ON:
                client.start_instance(external_id)
            elif command.action is RemoteComputeAction.POWER_OFF:
                client.stop_instance(external_id)
            else:
                client.delete_instance(external_id)
            return
        raise RemoteComputeCommandError("REMOTE_PROVIDER_UNSUPPORTED")

    def _cleanup(self, command: RemoteComputeCommand, *, secret: str) -> int:
        commands: list[str] = []
        for item in command.cleanup_items:
            manager = str(item.get("manager") or "")
            if manager == "pip":
                name = _package_name(item.get("name"))
                commands.append(f"python3 -m pip uninstall -y -- {shlex.quote(name)}")
            elif manager == "conda":
                name = _package_name(item.get("name"))
                commands.append(f"conda remove -y -- {shlex.quote(name)}")
            elif manager == "path":
                path = _confined_path(command.remote_root, item.get("path"))
                commands.append(f"rm -rf -- {shlex.quote(path)}")
            else:
                raise RemoteComputeCommandError("REMOTE_CLEANUP_ITEM_INVALID")
        if not commands:
            raise RemoteComputeCommandError("REMOTE_CLEANUP_ITEM_INVALID")
        self._exec_ssh(command, secret, " && ".join(commands))
        return len(commands)

    def _migrate(
        self,
        command: RemoteComputeCommand,
        *,
        source_secret: str,
        target_secret: str,
    ) -> Mapping[str, Any]:
        if not command.target_access_host or not command.target_remote_root:
            raise RemoteComputeCommandError("REMOTE_MIGRATION_TARGET_INVALID")
        if self._transfer_executor is None:
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_DURABILITY_REQUIRED"
            )
        target_root = _confined_root(command.target_remote_root)
        target_command = RemoteComputeCommand(
            **{
                field: getattr(command, field)
                for field in command.__dataclass_fields__
                if field not in {"access_host", "access_port", "access_username", "remote_root"}
            },
            access_host=command.target_access_host,
            access_port=command.target_access_port,
            access_username=command.target_access_username,
            remote_root=target_root,
        )
        with self._migration_admission(command, target_command):
            with self._connection(command, source_secret, count_operation_failures=False) as source_client, self._connection(target_command, target_secret, count_operation_failures=False) as target_client:
                with connection_deadline((source_client, target_client), timeout_seconds=self._migration_timeout):
                    with source_client.open_sftp() as source, target_client.open_sftp() as target:
                        source.get_channel().settimeout(self._timeout)
                        target.get_channel().settimeout(self._timeout)
                        return self._transfer_executor.migrate(
                            command,
                            source=source,
                            target=target,
                        )

    @contextmanager
    def _migration_admission(self, source, target):
        # Pair scope avoids assigning an ambiguous transfer failure to either
        # host, and keeps a broken migration route from being retried forever.
        key = "ssh-migration:" + hashlib.sha256(json.dumps([
            [item.access_host.lower(), item.access_port, item.access_username]
            for item in (source, target)
        ], separators=(",", ":")).encode()).hexdigest()
        try:
            permit = SSH_CIRCUITS.acquire(key)
        except DependencyUnavailable:
            raise RemoteComputeCommandError("REMOTE_MIGRATION_CIRCUIT_OPEN", retryable=True) from None
        outcome = "neutral"
        try:
            yield
            outcome = "success"
        except (FileNotFoundError, PermissionError):
            raise
        except (OSError, EOFError, paramiko.SSHException):
            outcome = "failure"
            raise
        finally:
            permit.finish(outcome)

    def _exec_ssh(self, command: RemoteComputeCommand, secret: str, shell: str) -> None:
        with self._connection(command, secret) as client:
            _output, _error, exit_code = execute_command(
                client, shell, maximum=_MAX_SCAN_OUTPUT_BYTES, timeout_seconds=self._command_timeout,
            )
            if exit_code != 0:
                raise RemoteComputeCommandError("REMOTE_SSH_EFFECT_FAILED")

    def _connection(self, command: RemoteComputeCommand, secret: str, *, count_operation_failures: bool = True) -> Any:
        return self._ssh_connection(
            host=command.access_host,
            port=command.access_port,
            username=command.access_username,
            secret=secret,
            count_operation_failures=count_operation_failures,
        )

    @contextmanager
    def _ssh_connection(self, *, host: str, port: int, username: str, secret: str, count_operation_failures: bool = True):
        key = "ssh:" + hashlib.sha256(json.dumps(
            [host.lower(), port, username], separators=(",", ":"),
        ).encode()).hexdigest()
        try:
            permit = SSH_CIRCUITS.acquire(key)
        except DependencyUnavailable:
            raise RemoteComputeCommandError("REMOTE_SSH_CIRCUIT_OPEN", retryable=True) from None
        client = None
        outcome = "neutral"
        try:
            client = self._connect_fields(host=host, port=port, username=username, secret=secret)
            yield client
            outcome = "success"
        except (FileNotFoundError, PermissionError):
            raise
        except (OSError, EOFError, paramiko.SSHException):
            # A two-host transfer cannot attribute an arbitrary body failure
            # to either endpoint; only its connection failures are counted.
            outcome = "failure" if client is None or count_operation_failures else "neutral"
            raise
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                permit.finish(outcome)

    def _connect_fields(
        self,
        *,
        host: str,
        port: int,
        username: str,
        secret: str,
    ) -> Any:
        client = paramiko.SSHClient()
        try:
            client.load_host_keys(self._known_hosts)
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            connect: dict[str, Any] = {
                "hostname": host,
                "port": port,
                "username": username,
                "timeout": self._timeout,
                "banner_timeout": self._timeout,
                "auth_timeout": self._timeout,
                "channel_timeout": self._timeout,
                "allow_agent": False,
                "look_for_keys": False,
            }
            if "BEGIN" in secret:
                connect["pkey"] = _private_key(secret)
            else:
                connect["password"] = secret
            client.connect(**connect)
            return client
        except Exception:
            client.close()
            raise


_MAX_SCAN_OUTPUT_BYTES = 1024 * 1024
_ENVIRONMENT_SCAN_SCRIPT = r"""
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

root = os.path.normpath(sys.argv[2])
if not root.startswith('/') or root == '/' or '\x00' in root:
    raise SystemExit(2)

protected = {'pip', 'setuptools', 'wheel', 'python', 'conda', 'openssl'}
paths = [
    ('workspace', root, '工作区', False),
    ('workspace_cache', os.path.join(root, '.cache'), '工作区缓存', True),
    ('local_site', '/root/.local', '用户 site-packages', False),
    ('hf_cache', '/root/.cache/huggingface', 'HuggingFace 缓存', True),
    ('pip_cache', '/root/.cache/pip', 'pip 缓存', True),
    ('conda_root', '/root/miniconda3', 'Conda', False),
    ('opt_conda', '/opt/conda', 'Conda', False),
]

def size(path):
    if not os.path.exists(path):
        return None
    try:
        run = subprocess.run(
            ['du', '-sb', '--', path], capture_output=True, text=True,
            timeout=8, check=False,
        )
        return int(run.stdout.split()[0]) if run.returncode == 0 else None
    except Exception:
        return None

path_items = []
for key, path, label, cleanable in paths:
    measured = size(path)
    if measured is not None:
        path_items.append({
            'id': 'path:' + key, 'kind': 'path', 'path': path,
            'label': label, 'size_bytes': measured, 'cleanable': cleanable,
        })

usage = shutil.disk_usage(root)
packages = []
for dist in metadata.distributions():
    name = str(dist.metadata.get('Name') or getattr(dist, 'name', '')).strip()
    if not name:
        continue
    packages.append({
        'id': 'pip:' + name, 'manager': 'pip', 'name': name,
        'version': str(dist.version or ''), 'size_bytes': None,
        'cleanable': name.lower() not in protected,
        'protected': name.lower() in protected,
    })
packages.sort(key=lambda item: item['name'].lower())

conda = shutil.which('conda')
if conda:
    try:
        run = subprocess.run(
            [conda, 'list', '--json'], capture_output=True, text=True,
            timeout=10, check=False,
        )
        rows = json.loads(run.stdout) if run.returncode == 0 else []
    except Exception:
        rows = []
    for item in rows[:300] if isinstance(rows, list) else []:
        name = str(item.get('name') or '').strip()
        if name:
            packages.append({
                'id': 'conda:' + name, 'manager': 'conda', 'name': name,
                'version': str(item.get('version') or ''), 'size_bytes': None,
                'cleanable': name.lower() not in protected,
                'protected': name.lower() in protected,
            })

try:
    run = subprocess.run(
        ['dpkg-query', '-W', '-f=${Package}\t${Version}\t${Installed-Size}\n'],
        capture_output=True, text=True, timeout=10, check=False,
    )
    rows = run.stdout.splitlines() if run.returncode == 0 else []
except Exception:
    rows = []
for line in rows[:120]:
    parts = line.split('\t')
    if len(parts) != 3 or not parts[0]:
        continue
    try:
        installed_size = max(0, int(parts[2])) * 1024
    except ValueError:
        installed_size = 0
    packages.append({
        'id': 'apt:' + parts[0], 'manager': 'apt', 'name': parts[0],
        'version': parts[1], 'size_bytes': installed_size,
        'cleanable': False, 'protected': True,
    })

print(json.dumps({
    'captured_at': datetime.now(timezone.utc).isoformat(),
    'python': sys.executable,
    'platform': sys.platform,
    'disk': [{
        'path': '整体占用', 'total_bytes': usage.total,
        'used_bytes': usage.used, 'free_bytes': usage.free,
    }],
    'paths': sorted(path_items, key=lambda item: -item['size_bytes'])[:20],
    'packages': packages[:720],
}, ensure_ascii=True, separators=(',', ':')))
"""


def _validate_environment_scan(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "captured_at", "python", "platform", "disk", "paths", "packages"
    }:
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID")
    if not isinstance(raw["captured_at"], str) or len(raw["captured_at"]) > 64:
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID")
    if not isinstance(raw["disk"], list) or len(raw["disk"]) > 8:
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID")
    if not isinstance(raw["paths"], list) or len(raw["paths"]) > 20:
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID")
    if not isinstance(raw["packages"], list) or len(raw["packages"]) > 720:
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID")
    if not all(_valid_disk_item(item) for item in raw["disk"]):
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID")
    if not all(_valid_path_item(item) for item in raw["paths"]):
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID")
    if not all(_valid_package_item(item) for item in raw["packages"]):
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_INVALID")
    encoded = json.dumps(raw, ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_SCAN_OUTPUT_BYTES:
        raise RemoteComputeCommandError("REMOTE_ENVIRONMENT_SCAN_LIMIT_EXCEEDED")
    return dict(raw)


def _bounded_text(value: Any, maximum: int = 4096) -> bool:
    return isinstance(value, str) and "\x00" not in value and len(value.encode("utf-8")) <= maximum


def _bounded_size(value: Any, *, nullable: bool = False) -> bool:
    return (nullable and value is None) or (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 0 <= value <= (1 << 63) - 1
    )


def _valid_disk_item(item: Any) -> bool:
    return (
        isinstance(item, Mapping)
        and set(item) == {"path", "total_bytes", "used_bytes", "free_bytes"}
        and _bounded_text(item["path"], 255)
        and all(_bounded_size(item[key]) for key in ("total_bytes", "used_bytes", "free_bytes"))
    )


def _valid_path_item(item: Any) -> bool:
    return (
        isinstance(item, Mapping)
        and set(item) == {"id", "kind", "path", "label", "size_bytes", "cleanable"}
        and all(_bounded_text(item[key], 4096) for key in ("id", "kind", "path", "label"))
        and _bounded_size(item["size_bytes"], nullable=True)
        and isinstance(item["cleanable"], bool)
    )


def _valid_package_item(item: Any) -> bool:
    return (
        isinstance(item, Mapping)
        and set(item) == {
            "id", "manager", "name", "version", "size_bytes", "cleanable", "protected"
        }
        and all(_bounded_text(item[key], 512) for key in ("id", "manager", "name", "version"))
        and _bounded_size(item["size_bytes"], nullable=True)
        and isinstance(item["cleanable"], bool)
        and isinstance(item["protected"], bool)
    )


def _confined_root(value: str) -> str:
    path = posixpath.normpath(str(value or ""))
    if not path.startswith("/") or path == "/" or "\x00" in path:
        raise RemoteComputeCommandError("REMOTE_PATH_OUTSIDE_ROOT")
    return path


def _confined_path(root: str, value: Any) -> str:
    base = PurePosixPath(_confined_root(root))
    candidate = PurePosixPath(posixpath.normpath(str(value or "")))
    if not candidate.is_absolute() or candidate == base or base not in candidate.parents:
        raise RemoteComputeCommandError("REMOTE_PATH_OUTSIDE_ROOT")
    return str(candidate)


def _package_name(value: Any) -> str:
    name = str(value or "").strip()
    protected = {"python", "pip", "setuptools", "wheel", "conda", "openssl"}
    if (
        not name
        or name.lower() in protected
        or len(name) > 200
        or not all(ch.isalnum() or ch in "._-" for ch in name)
    ):
        raise RemoteComputeCommandError("REMOTE_PACKAGE_INVALID")
    return name


def _private_key(value: str) -> Any:
    for key_type in (
        paramiko.Ed25519Key,
        paramiko.ECDSAKey,
        paramiko.RSAKey,
    ):
        try:
            return key_type.from_private_key(StringIO(value))
        except Exception:
            continue
    raise RemoteComputeCommandError("REMOTE_PRIVATE_KEY_INVALID")


def _copy_sftp_tree(source: Any, target: Any, source_root: str, target_root: str) -> None:
    max_files = 100_000
    max_bytes = 100 * 1024 * 1024 * 1024
    files = 0
    transferred = 0
    stack = [(source_root, target_root)]
    while stack:
        source_dir, target_dir = stack.pop()
        _ensure_remote_directory(target, target_dir)
        for entry in source.listdir_attr(source_dir):
            files += 1
            if files > max_files:
                raise RemoteComputeCommandError("REMOTE_MIGRATION_LIMIT_EXCEEDED")
            source_path = posixpath.join(source_dir, entry.filename)
            target_path = posixpath.join(target_dir, entry.filename)
            if stat.S_ISLNK(entry.st_mode):
                raise RemoteComputeCommandError("REMOTE_MIGRATION_SYMLINK_REJECTED")
            if stat.S_ISDIR(entry.st_mode):
                stack.append((source_path, target_path))
                continue
            if not stat.S_ISREG(entry.st_mode):
                continue
            transferred += int(entry.st_size)
            if transferred > max_bytes:
                raise RemoteComputeCommandError("REMOTE_MIGRATION_LIMIT_EXCEEDED")
            with source.open(source_path, "rb") as reader, target.open(target_path, "wb") as writer:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    writer.write(chunk)


def _ensure_remote_directory(sftp: Any, path: str) -> None:
    current = "/"
    for part in PurePosixPath(path).parts[1:]:
        current = posixpath.join(current, part)
        try:
            metadata = sftp.lstat(current)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RemoteComputeCommandError("REMOTE_MIGRATION_TARGET_INVALID")
        except OSError:
            sftp.mkdir(current, mode=0o700)


__all__ = ["AliyunControlClient", "AutoDLControlClient", "StrictRemoteComputeProvider"]
