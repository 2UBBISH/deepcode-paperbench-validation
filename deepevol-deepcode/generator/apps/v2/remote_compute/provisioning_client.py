"""Strict Product-to-Agent transport for Remote Compute provider effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from apps.common.api_envelope import error_code_of
import httpx

from apps.v2.reliability.control_response import read_control_response

from apps.common.v2_ids import format_typed_id

from .models import RemoteComputeResource
from .provisioning import ProvisionedRemoteCompute


class AgentRemoteComputeUnavailable(RuntimeError):
    def __init__(self, code: str = "REMOTE_PROVISIONING_UNAVAILABLE") -> None:
        super().__init__(code)
        self.code = code


class AgentRemoteComputeRejected(RuntimeError):
    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class AgentRemoteComputeProvisioningClient:
    def __init__(
        self,
        client: httpx.Client,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 240.0,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Agent Remote Compute base URL must be an HTTP(S) origin")
        if len(token.encode("utf-8")) < 32:
            raise ValueError("Agent Remote Compute token must contain at least 32 bytes")
        if timeout_seconds <= 0:
            raise ValueError("Agent Remote Compute timeout must be positive")
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout_seconds

    def catalog(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        requirements: Mapping[str, int],
    ) -> tuple[Mapping[str, Any], ...]:
        data = self._request(
            "/internal/v2/remote-compute/catalog",
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            body={"requirements": dict(requirements)},
            validate_data=self._catalog,
        )
        return self._catalog(data)

    @staticmethod
    def _catalog(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        options = data.get("options")
        if (
            not isinstance(options, list)
            or len(options) > 16
            or any(not isinstance(option, Mapping) for option in options)
        ):
            raise AgentRemoteComputeUnavailable("REMOTE_PROVIDER_RESPONSE_INVALID")
        return tuple(dict(option) for option in options)

    def provision(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        resource_id: UUID,
        target: str,
        spec: Mapping[str, Any],
    ) -> ProvisionedRemoteCompute:
        data = self._request(
            "/internal/v2/remote-compute/provision",
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            body={
                "resource_id": format_typed_id("rcres", resource_id),
                "target": target,
                "spec": dict(spec),
            },
            validate_data=self._provisioned,
        )
        return self._provisioned(data)

    def activate(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        resource: RemoteComputeResource,
    ) -> ProvisionedRemoteCompute:
        data = self._request(
            "/internal/v2/remote-compute/activate",
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            body={
                "resource_id": format_typed_id("rcres", resource.resource_id),
                "resource": _resource_wire(resource),
            },
            validate_data=self._provisioned,
        )
        return self._provisioned(data)

    def rollback(
        self,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        provisioned: ProvisionedRemoteCompute,
    ) -> None:
        data = self._request(
            "/internal/v2/remote-compute/rollback",
            resource_tid=resource_tid,
            owner_uid=owner_uid,
            body={"provisioned": provisioned.internal_dict()},
            validate_data=self._rollback_receipt,
        )
        self._rollback_receipt(data)

    @staticmethod
    def _rollback_receipt(data: Mapping[str, Any]) -> None:
        if data != {"rolled_back": True}:
            raise AgentRemoteComputeUnavailable("REMOTE_ROLLBACK_RECEIPT_INVALID")

    def _request(
        self,
        path: str,
        *,
        resource_tid: UUID,
        owner_uid: UUID,
        body: Mapping[str, Any],
        validate_data=None,
    ) -> Mapping[str, Any]:
        def validate(response: httpx.Response) -> None:
            try:
                document = response.json()
            except ValueError as exc:
                raise AgentRemoteComputeUnavailable("REMOTE_PROVIDER_RESPONSE_INVALID") from exc
            data = document.get("data") if isinstance(document, Mapping) else None
            if not isinstance(data, Mapping):
                raise AgentRemoteComputeUnavailable("REMOTE_PROVIDER_RESPONSE_INVALID")
            if validate_data is not None:
                validate_data(data)

        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}{path}",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept-Encoding": "identity",
                    "X-DeepEvol-Source-Service": "PRODUCT_API",
                    "X-DeepEvol-Resource-Tid": format_typed_id("tid", resource_tid),
                    "X-DeepEvol-Owner-Uid": format_typed_id("uid", owner_uid),
                    "Accept": "application/json",
                },
                json=dict(body),
                timeout=self.timeout,
                extensions={"total_timeout_seconds": self.timeout},
            ) as streamed:
                response = read_control_response(streamed, maximum_bytes=1024 * 1024, validate_success=validate)
        except httpx.HTTPError as exc:
            raise AgentRemoteComputeUnavailable() from exc
        try:
            document = response.json()
        except ValueError as exc:
            raise AgentRemoteComputeUnavailable("REMOTE_PROVIDER_RESPONSE_INVALID") from exc
        if not isinstance(document, Mapping):
            raise AgentRemoteComputeUnavailable("REMOTE_PROVIDER_RESPONSE_INVALID")
        if response.status_code != 200:
            code = error_code_of(document)
            error_code = code if isinstance(code, str) and code else "REMOTE_PROVISIONING_REJECTED"
            if response.status_code >= 500:
                raise AgentRemoteComputeUnavailable(error_code)
            raise AgentRemoteComputeRejected(error_code, response.status_code)
        data = document.get("data")
        if not isinstance(data, Mapping):
            raise AgentRemoteComputeUnavailable("REMOTE_PROVIDER_RESPONSE_INVALID")
        return data

    @staticmethod
    def _provisioned(data: Mapping[str, Any]) -> ProvisionedRemoteCompute:
        try:
            return ProvisionedRemoteCompute.from_dict(data.get("provisioned"))
        except ValueError as exc:
            raise AgentRemoteComputeUnavailable("REMOTE_PROVIDER_RESPONSE_INVALID") from exc


def _resource_wire(resource: RemoteComputeResource) -> dict[str, object]:
    return {
        "provider": resource.provider,
        "name": resource.name,
        "status": resource.status,
        "access_host": resource.access_host,
        "access_port": resource.access_port,
        "access_username": resource.access_username,
        "secret_ref": resource.secret_ref,
        "secret_version": resource.secret_version,
        "remote_root": resource.remote_root,
        "external_resource_id": resource.external_resource_id,
        "accelerator_type": resource.accelerator_type,
        "accelerator_count": resource.accelerator_count,
        "vram_gb": resource.vram_gb,
        "billing_mode": resource.billing_mode,
        "hourly_price_credits": resource.hourly_price_credits,
        "region": resource.region,
        "instance_type": resource.instance_type,
        "cpu_cores": resource.cpu_cores,
        "memory_gb": resource.memory_gb,
        "storage_gb": resource.storage_gb,
    }


__all__: Sequence[str] = (
    "AgentRemoteComputeProvisioningClient",
    "AgentRemoteComputeRejected",
    "AgentRemoteComputeUnavailable",
)
