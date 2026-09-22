"""Product-to-Agent transport for live Remote Compute inventory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from apps.v2.reliability.control_response import read_control_response

from apps.common.v2_ids import format_typed_id

from .models import RemoteComputeResource, RemoteEnvironmentDescriptor


class RemoteEnvironmentScanUnavailable(RuntimeError):
    pass


class AgentRemoteEnvironmentScanClient:
    def __init__(
        self,
        client: httpx.Client,
        endpoint_url: str,
        token: str,
        *,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not endpoint_url.startswith(("http://", "https://")):
            raise ValueError("Agent environment scan endpoint must be HTTP(S)")
        if len(token.encode("utf-8")) < 32:
            raise ValueError("Agent environment scan token must contain at least 32 bytes")
        if timeout_seconds <= 0:
            raise ValueError("Agent environment scan timeout must be positive")
        self.client = client
        self.endpoint_url = endpoint_url
        self.token = token
        self.timeout = timeout_seconds

    def scan(self, resource: RemoteComputeResource) -> Mapping[str, Any]:
        descriptor = RemoteEnvironmentDescriptor.from_resource(resource)
        receipt: dict[str, Any] = {}

        def validate(response: httpx.Response) -> None:
            nonlocal receipt
            try:
                payload = response.json()
            except ValueError as exc:
                raise RemoteEnvironmentScanUnavailable("Agent environment scan returned invalid JSON") from exc
            data = payload.get("data") if isinstance(payload, Mapping) else None
            environment = data.get("environment") if isinstance(data, Mapping) else None
            if not isinstance(environment, Mapping):
                raise RemoteEnvironmentScanUnavailable("Agent environment scan returned an invalid payload")
            receipt = dict(environment)

        try:
            with self.client.stream(
                "POST",
                self.endpoint_url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept-Encoding": "identity",
                    "X-DeepEvol-Source-Service": "PRODUCT_API",
                    "X-DeepEvol-Resource-Tid": format_typed_id("tid", resource.resource_tid),
                    "X-DeepEvol-Owner-Uid": format_typed_id("uid", resource.owner_uid),
                },
                json={"descriptor": descriptor.internal_dict()},
                timeout=self.timeout,
                extensions={"total_timeout_seconds": self.timeout},
            ) as streamed:
                response = read_control_response(streamed, maximum_bytes=1024 * 1024, validate_success=validate)
        except httpx.HTTPError as exc:
            raise RemoteEnvironmentScanUnavailable("Agent environment scan transport is unavailable") from exc
        if response.status_code != 200:
            raise RemoteEnvironmentScanUnavailable(f"Agent environment scan failed with status {response.status_code}")
        return receipt


__all__ = [
    "AgentRemoteEnvironmentScanClient",
    "RemoteEnvironmentScanUnavailable",
]
