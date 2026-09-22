"""Strict Product-to-Agent transport for durable Remote Compute Sagas."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse

import httpx

from apps.v2.reliability.control_response import read_control_response

from apps.common.api_envelope import error_code_of
from apps.common.v2_ids import InvalidTypedId, format_typed_id, parse_typed_id
from apps.v2.reliability import FailureClass

from .saga_models import RemoteComputeOperation
from .saga_worker import (
    RemoteComputeEffectDisposition,
    RemoteComputeEffectOutcome,
    RemoteComputeEffectPreparation,
    RemoteComputePreEffectFailure,
)


class AgentRemoteComputeSagaUnavailable(RuntimeError):
    def __init__(self, code: str = "REMOTE_COMPUTE_AGENT_UNAVAILABLE") -> None:
        super().__init__(code)
        self.code = code


class AgentRemoteComputeSagaRejected(RuntimeError):
    def __init__(
        self,
        code: str,
        status_code: int,
        *,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.retry_after = retry_after


class AgentRemoteComputeSagaEffectClient:
    """Call the Agent's prepare/apply/reconcile boundary with strict receipts."""

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

    def prepare(
        self,
        operation: RemoteComputeOperation,
    ) -> RemoteComputeEffectPreparation:
        if operation.rid is None or operation.resource_id is None:
            raise ValueError("Remote Compute Saga preparation requires run and resource")
        try:
            data = self._request(
                method="POST",
                path="/internal/v2/remote-compute/provider-operations/prepare",
                operation=operation,
                body={
                    "operation_id": format_typed_id("op", operation.operation_id),
                    "rid": format_typed_id("rid", operation.rid),
                    "resource_id": format_typed_id("rcres", operation.resource_id),
                    "operation_kind": operation.operation_kind.value,
                    "provider": operation.provider,
                    "provider_operation_id": operation.provider_operation_id,
                    "request": dict(operation.request),
                    "request_sha256": operation.request_sha256.hex(),
                },
                validate_data=lambda data: self._preparation(data, operation),
            )
        except AgentRemoteComputeSagaUnavailable as exc:
            # The prepare endpoint is contractually non-dispatching. Even if its
            # response is lost, repeating this stable operation cannot egress.
            raise RemoteComputePreEffectFailure(
                exc.code,
                failure_class=FailureClass.NETWORK_PRE_EFFECT,
            ) from exc
        except AgentRemoteComputeSagaRejected as exc:
            if exc.status_code != 429:
                raise
            raise RemoteComputePreEffectFailure(
                exc.code,
                failure_class=FailureClass.RATE_LIMITED_PRE_EFFECT,
                retry_after=exc.retry_after,
            ) from exc
        return self._preparation(data, operation)

    @staticmethod
    def _preparation(data: Mapping[str, Any], operation: RemoteComputeOperation) -> RemoteComputeEffectPreparation:
        preparation = data.get("preparation")
        replayed = data.get("replayed")
        if not isinstance(preparation, Mapping) or not isinstance(replayed, bool):
            raise RemoteComputePreEffectFailure(
                "REMOTE_COMPUTE_PREPARATION_RESPONSE_INVALID",
                failure_class=FailureClass.NETWORK_PRE_EFFECT,
            )
        try:
            if set(preparation) != {
                "operation_id",
                "provider_operation_id",
                "dispatch_ref",
            }:
                raise ValueError("unexpected preparation fields")
            receipt = RemoteComputeEffectPreparation(
                operation_id=parse_typed_id(preparation.get("operation_id"), expected_prefix="op"),
                provider_operation_id=_required_text(preparation.get("provider_operation_id"), maximum=256),
                dispatch_ref=_required_text(preparation.get("dispatch_ref"), maximum=512),
            )
        except (InvalidTypedId, TypeError, ValueError) as exc:
            raise RemoteComputePreEffectFailure(
                "REMOTE_COMPUTE_PREPARATION_RESPONSE_INVALID",
                failure_class=FailureClass.NETWORK_PRE_EFFECT,
            ) from exc
        if (
            receipt.operation_id != operation.operation_id
            or receipt.provider_operation_id != operation.provider_operation_id
            or receipt.dispatch_ref != f"/internal/v2/remote-compute/operations/{operation.provider_operation_id}"
        ):
            raise RemoteComputePreEffectFailure(
                "REMOTE_COMPUTE_PREPARATION_RESPONSE_INVALID",
                failure_class=FailureClass.NETWORK_PRE_EFFECT,
            )
        return receipt

    def apply(
        self,
        operation: RemoteComputeOperation,
        preparation: RemoteComputeEffectPreparation,
    ) -> RemoteComputeEffectOutcome:
        if (
            preparation.operation_id != operation.operation_id
            or preparation.provider_operation_id != operation.provider_operation_id
        ):
            raise ValueError("Remote Compute preparation does not match operation")
        data = self._request(
            method="POST",
            path=(
                f"/internal/v2/remote-compute/provider-operations/{format_typed_id('op', operation.operation_id)}/apply"
            ),
            operation=operation,
            body={
                "provider_operation_id": preparation.provider_operation_id,
                "dispatch_ref": preparation.dispatch_ref,
            },
            validate_data=_parse_outcome,
        )
        return _parse_outcome(data)

    def reconcile(
        self,
        operation: RemoteComputeOperation,
    ) -> RemoteComputeEffectOutcome:
        data = self._request(
            method="GET",
            path=(f"/internal/v2/remote-compute/provider-operations/{format_typed_id('op', operation.operation_id)}"),
            operation=operation,
            body=None,
            validate_data=_parse_outcome,
        )
        return _parse_outcome(data)

    def _request(
        self,
        *,
        method: str,
        path: str,
        operation: RemoteComputeOperation,
        body: Mapping[str, Any] | None,
        validate_data=None,
    ) -> Mapping[str, Any]:
        def validate(response: httpx.Response) -> None:
            try:
                document = response.json()
            except ValueError as exc:
                raise AgentRemoteComputeSagaUnavailable("REMOTE_COMPUTE_AGENT_RESPONSE_INVALID") from exc
            data = document.get("data") if isinstance(document, Mapping) else None
            if not isinstance(data, Mapping):
                raise AgentRemoteComputeSagaUnavailable("REMOTE_COMPUTE_AGENT_RESPONSE_INVALID")
            if validate_data is not None:
                validate_data(data)

        try:
            with self.client.stream(
                method,
                f"{self.base_url}{path}",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept-Encoding": "identity",
                    "X-DeepEvol-Source-Service": "PRODUCT_API",
                    "X-DeepEvol-Resource-Tid": format_typed_id("tid", operation.resource_tid),
                    "X-DeepEvol-Owner-Uid": format_typed_id("uid", operation.owner_uid),
                    "Accept": "application/json",
                },
                json=None if body is None else dict(body),
                timeout=self.timeout,
                extensions={"total_timeout_seconds": self.timeout},
            ) as streamed:
                response = read_control_response(streamed, maximum_bytes=1024 * 1024, validate_success=validate)
        except httpx.HTTPError as exc:
            raise AgentRemoteComputeSagaUnavailable() from exc
        try:
            document = response.json()
        except ValueError as exc:
            raise AgentRemoteComputeSagaUnavailable("REMOTE_COMPUTE_AGENT_RESPONSE_INVALID") from exc
        if not isinstance(document, Mapping):
            raise AgentRemoteComputeSagaUnavailable("REMOTE_COMPUTE_AGENT_RESPONSE_INVALID")
        if response.status_code != 200:
            raw_code = error_code_of(document)
            code = raw_code if isinstance(raw_code, str) and raw_code else "REMOTE_COMPUTE_AGENT_REJECTED"
            if response.status_code >= 500:
                raise AgentRemoteComputeSagaUnavailable(code)
            raise AgentRemoteComputeSagaRejected(
                code,
                response.status_code,
                retry_after=response.headers.get("Retry-After"),
            )
        data = document.get("data")
        if not isinstance(data, Mapping):
            raise AgentRemoteComputeSagaUnavailable("REMOTE_COMPUTE_AGENT_RESPONSE_INVALID")
        return data


def _parse_outcome(data: Mapping[str, Any]) -> RemoteComputeEffectOutcome:
    outcome = data.get("outcome")
    if not isinstance(outcome, Mapping) or set(outcome) != {
        "disposition",
        "provider_request_ref",
        "provider_status",
        "result",
        "error_code",
        "retry_after_seconds",
    }:
        raise AgentRemoteComputeSagaUnavailable("REMOTE_COMPUTE_AGENT_RESPONSE_INVALID")
    result = outcome.get("result")
    retry_after = outcome.get("retry_after_seconds")
    try:
        if result is not None and not isinstance(result, Mapping):
            raise ValueError("outcome result is not an object")
        if isinstance(retry_after, bool) or not isinstance(retry_after, int):
            raise ValueError("retry delay is not an integer")
        if not 0 <= retry_after <= 86_400:
            raise ValueError("retry delay is outside bounds")
        return RemoteComputeEffectOutcome(
            disposition=RemoteComputeEffectDisposition(outcome.get("disposition")),
            provider_request_ref=_optional_text(outcome.get("provider_request_ref"), maximum=512),
            provider_status=_optional_text(outcome.get("provider_status"), maximum=128),
            result=None if result is None else dict(result),
            error_code=_optional_text(outcome.get("error_code"), maximum=128),
            retry_after=timedelta(seconds=retry_after),
        )
    except (TypeError, ValueError) as exc:
        raise AgentRemoteComputeSagaUnavailable("REMOTE_COMPUTE_AGENT_RESPONSE_INVALID") from exc


def _required_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError("required text is invalid")
    return value


def _optional_text(value: object, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _required_text(value, maximum=maximum)


__all__ = [
    "AgentRemoteComputeSagaEffectClient",
    "AgentRemoteComputeSagaRejected",
    "AgentRemoteComputeSagaUnavailable",
]
