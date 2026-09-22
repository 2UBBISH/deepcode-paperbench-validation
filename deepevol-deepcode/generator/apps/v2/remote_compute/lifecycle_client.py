"""Strict Agent-to-Product transport for run-scoped Remote Compute lifecycle."""

from __future__ import annotations

from apps.common.api_envelope import error_code_of
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import httpx

from apps.v2.reliability.control_response import read_control_response

from apps.common.v2_ids import (
    InvalidTypedId,
    derive_uuid7,
    format_typed_id,
    parse_typed_id,
)
from apps.v2.agent.models import ExecutionRunContext, LeaseAuthority
from apps.v2.remote_compute.saga_models import (
    RemoteComputeEffectPhase,
    RemoteComputeOperationKind,
    RemoteComputeOperationStatus,
    stable_provider_operation_id,
)


class ProductRemoteComputeUnavailable(RuntimeError):
    def __init__(self, code: str = "REMOTE_COMPUTE_PRODUCT_UNAVAILABLE") -> None:
        super().__init__(code)
        self.code = code


class ProductRemoteComputeRejected(RuntimeError):
    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class ProductRemoteComputePending(ProductRemoteComputeUnavailable):
    def __init__(self, operation_id: UUID, retry_after: int) -> None:
        super().__init__("REMOTE_COMPUTE_SELECTION_PENDING")
        self.operation_id = operation_id
        self.retry_after = retry_after


class ProductRemoteComputeLifecycleClient:
    """Call Product using only the current Agent run and tenant authority."""

    def __init__(
        self,
        client: httpx.Client,
        endpoint_url: str,
        token: str,
        *,
        timeout_seconds: float = 240.0,
    ) -> None:
        parsed = urlparse(endpoint_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path.rstrip("/") != "/internal/v2/runs"
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Product Remote Compute URL must target /internal/v2/runs")
        if len(token.encode("utf-8")) < 32:
            raise ValueError("Product Remote Compute token must contain at least 32 bytes")
        if timeout_seconds <= 0:
            raise ValueError("Product Remote Compute timeout must be positive")
        self.client = client
        self.base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        self.token = token
        self.timeout = timeout_seconds

    def preflight(
        self,
        context: ExecutionRunContext,
        authority: LeaseAuthority,
        *,
        prompt: str,
        resource_policy: Mapping[str, int] | None = None,
        resource_requirements: Mapping[str, int | bool] | None = None,
    ) -> Mapping[str, Any]:
        body: dict[str, Any] = {
            "sid": format_typed_id("sid", context.sid),
            "prompt": prompt,
            "resource_policy": dict(resource_policy or {}),
        }
        if resource_requirements:
            body["resource_requirements"] = _requirements(resource_requirements)
        data = self._request(
            context,
            authority,
            method="POST",
            suffix="remote-compute/preflight",
            body=body,
        )
        preflight = data.get("preflight")
        if not isinstance(preflight, Mapping):
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
        _bounded_json(preflight, maximum_bytes=64 * 1024)
        _validate_preflight(preflight)
        return dict(preflight)

    def select(
        self,
        context: ExecutionRunContext,
        authority: LeaseAuthority,
        *,
        prompt: str,
        selected_target: str,
        resource_policy: Mapping[str, int] | None = None,
        resource_requirements: Mapping[str, int | bool] | None = None,
    ) -> Mapping[str, Any]:
        _validate_authority(context, authority)
        policy = dict(resource_policy or {})
        operation_id, idempotency_key = _selection_identity(
            context,
            prompt=prompt,
            selected_target=selected_target,
            resource_policy=policy,
        )
        rid = format_typed_id("rid", context.rid)
        body: dict[str, Any] = {
            "sid": format_typed_id("sid", context.sid),
            "prompt": prompt,
            "selected_target": selected_target,
            "operation_id": format_typed_id("op", operation_id),
            "idempotency_key": idempotency_key,
            "resource_policy": policy,
            "expires_at": None,
        }
        if resource_requirements:
            body["resource_requirements"] = _requirements(resource_requirements)
        try:
            response = self._bounded_request(
                "POST",
                f"{self.base_url}/{rid}/remote-compute/selection",
                headers=self._headers(context),
                json=body,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ProductRemoteComputeUnavailable() from exc
        data = _response_data(response, accepted_statuses={200, 202})
        if response.status_code == 202:
            _validate_pending_selection(
                data,
                context=context,
                expected_target=selected_target,
                expected_operation_id=operation_id,
            )
            raise ProductRemoteComputePending(
                operation_id,
                int(data["retry_after"]),
            )
        _bounded_json(data, maximum_bytes=64 * 1024)
        _validate_selection(
            data,
            expected_target=selected_target,
            expected_operation_id=(None if selected_target == "local_cpu" else operation_id),
        )
        return dict(data)

    def finish(
        self,
        context: ExecutionRunContext,
        authority: LeaseAuthority,
        *,
        ended_at: datetime,
        outcome: str,
    ) -> Mapping[str, Any] | None:
        if ended_at.tzinfo is None or ended_at.utcoffset() is None:
            raise ValueError("Remote Compute finish time must be timezone-aware")
        normalized = outcome.strip().upper()
        if normalized not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            raise ValueError("Remote Compute outcome must be terminal")
        data = self._request(
            context,
            authority,
            method="POST",
            suffix="remote-compute/usage/finish",
            body={"ended_at": ended_at.isoformat(), "outcome": normalized},
            missing_is_none=True,
        )
        if data is None:
            return None
        usage = data.get("usage")
        if not isinstance(usage, Mapping) or not isinstance(data.get("replayed"), bool):
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
        _validate_usage(usage, context=context)
        return dict(data)

    def release(
        self,
        context: ExecutionRunContext,
        authority: LeaseAuthority,
        *,
        reason: str = "",
    ) -> Mapping[str, Any]:
        """Ask Product to release the machine this run provisioned.

        Returns the release receipt (``released`` true/false with a reason) or
        raises :class:`ProductRemoteComputePending` while the durable RELEASE
        operation is still running; poll :meth:`get_operation_status` and call
        again — the operation identity is stable, so the retry replays.
        """

        _validate_authority(context, authority)
        rid = format_typed_id("rid", context.rid)
        try:
            response = self._bounded_request(
                "POST",
                f"{self.base_url}/{rid}/remote-compute/release",
                headers=self._headers(context),
                json={
                    "sid": format_typed_id("sid", context.sid),
                    "reason": str(reason or "")[:512],
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ProductRemoteComputeUnavailable() from exc
        data = _response_data(response, accepted_statuses={200, 202})
        _bounded_json(data, maximum_bytes=64 * 1024)
        if response.status_code == 202:
            operation_id = data.get("operation_id")
            retry_after = data.get("retry_after")
            if not isinstance(operation_id, str) or not isinstance(retry_after, int):
                raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
            try:
                parsed = parse_typed_id(operation_id, expected_prefix="op")
            except (InvalidTypedId, TypeError, ValueError) as exc:
                raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID") from exc
            raise ProductRemoteComputePending(parsed, max(1, retry_after))
        if not isinstance(data.get("released"), bool):
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
        return dict(data)

    def get_operation_status(
        self,
        context: ExecutionRunContext,
        authority: LeaseAuthority,
        *,
        operation_id: UUID,
    ) -> Mapping[str, Any]:
        """Poll one Product-owned durable operation under the active run lease."""

        _validate_authority(context, authority)
        operation_tid = format_typed_id("op", operation_id)
        origin = self.base_url.removesuffix("/internal/v2/runs")
        try:
            response = self._bounded_request(
                "GET",
                f"{origin}/internal/v2/remote-compute/operations/{operation_tid}",
                headers=self._headers(context),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ProductRemoteComputeUnavailable() from exc
        data = _response_data(response)
        operation = data.get("operation")
        if not isinstance(operation, Mapping):
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
        _bounded_json(operation, maximum_bytes=64 * 1024)
        _validate_operation_status(
            operation,
            expected_operation_id=operation_id,
            context=context,
        )
        return dict(operation)

    def _request(
        self,
        context: ExecutionRunContext,
        authority: LeaseAuthority,
        *,
        method: str,
        suffix: str,
        body: Mapping[str, Any] | None = None,
        missing_is_none: bool = False,
    ) -> Mapping[str, Any] | None:
        _validate_authority(context, authority)
        rid = format_typed_id("rid", context.rid)
        try:
            response = self._bounded_request(
                method,
                f"{self.base_url}/{rid}/{suffix}",
                headers=self._headers(context),
                json=None if body is None else dict(body),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ProductRemoteComputeUnavailable() from exc
        if response.status_code == 404 and missing_is_none:
            return None
        return _response_data(response)

    def _bounded_request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = {**kwargs.pop("headers", {}), "Accept-Encoding": "identity"}
        with self.client.stream(method, url, headers=headers, **kwargs) as response:
            return read_control_response(response, maximum_bytes=128 * 1024, accepted_statuses=frozenset({200, 202}))

    def _headers(self, context: ExecutionRunContext) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "X-DeepEvol-Source-Service": "AGENT_EXECUTION",
            "X-DeepEvol-Resource-Tid": format_typed_id("tid", context.resource_tid),
            "X-DeepEvol-Owner-Uid": format_typed_id("uid", context.owner_uid),
            "Accept": "application/json",
        }


def _response_data(
    response: httpx.Response,
    *,
    accepted_statuses: set[int] | frozenset[int] = frozenset({200}),
) -> Mapping[str, Any]:
    """Decode the Product envelope without weakening its error semantics."""

    try:
        document = response.json()
    except ValueError as exc:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID") from exc
    if not isinstance(document, Mapping):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    if response.status_code not in accepted_statuses:
        raw_code = error_code_of(document)
        code = raw_code if isinstance(raw_code, str) and raw_code else "REMOTE_COMPUTE_REJECTED"
        if response.status_code >= 500:
            raise ProductRemoteComputeUnavailable(code)
        raise ProductRemoteComputeRejected(code, response.status_code)
    data = document.get("data")
    if not isinstance(data, Mapping):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    return data


def _validate_authority(
    context: ExecutionRunContext,
    authority: LeaseAuthority,
) -> None:
    if (
        context.resource_tid,
        context.owner_uid,
        context.rid,
    ) != (
        authority.resource_tid,
        authority.owner_uid,
        authority.rid,
    ):
        raise ValueError("Remote Compute request is outside the worker lease")


def _validate_operation_status(
    operation: Mapping[str, Any],
    *,
    expected_operation_id: UUID,
    context: ExecutionRunContext,
) -> None:
    expected_fields = {
        "operation_id",
        "operation_kind",
        "provider",
        "provider_operation_id",
        "resource_id",
        "binding_id",
        "rid",
        "compensates_operation_id",
        "status",
        "effect_phase",
        "attempt_count",
        "max_attempts",
        "next_attempt_at",
        "provider_request_ref",
        "provider_status",
        "error_code",
        "created_at",
        "updated_at",
        "completed_at",
        "terminal",
    }
    if set(operation) != expected_fields:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    try:
        operation_id = parse_typed_id(operation.get("operation_id"), expected_prefix="op")
        rid = parse_typed_id(operation.get("rid"), expected_prefix="rid")
        kind = RemoteComputeOperationKind(operation.get("operation_kind"))
        status = RemoteComputeOperationStatus(operation.get("status"))
        phase = RemoteComputeEffectPhase(operation.get("effect_phase"))
        resource_id_raw = operation.get("resource_id")
        binding_id_raw = operation.get("binding_id")
        compensates_raw = operation.get("compensates_operation_id")
        if resource_id_raw is not None:
            parse_typed_id(resource_id_raw, expected_prefix="rcres")
        if binding_id_raw is not None:
            parse_typed_id(binding_id_raw, expected_prefix="rcbind")
        if compensates_raw is not None:
            parse_typed_id(compensates_raw, expected_prefix="op")
    except (InvalidTypedId, TypeError, ValueError) as exc:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID") from exc
    if operation_id != expected_operation_id or rid != context.rid:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_AUTHORITY_MISMATCH")
    if (kind is RemoteComputeOperationKind.COMPENSATE) != (compensates_raw is not None):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    provider = operation.get("provider")
    provider_operation_id = operation.get("provider_operation_id")
    if (
        not isinstance(provider, str)
        or not provider
        or len(provider.encode("utf-8")) > 128
        or provider_operation_id != stable_provider_operation_id(operation_id, kind)
    ):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    attempt_count = operation.get("attempt_count")
    max_attempts = operation.get("max_attempts")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 0 <= attempt_count <= max_attempts <= 100
        or max_attempts < 1
    ):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    for field in ("provider_request_ref", "provider_status", "error_code"):
        value = operation.get(field)
        if value is not None and (
            not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512 or "\x00" in value
        ):
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    timestamps: dict[str, datetime | None] = {}
    for field in ("next_attempt_at", "created_at", "updated_at", "completed_at"):
        value = operation.get(field)
        if value is None and field == "completed_at":
            timestamps[field] = None
            continue
        if not isinstance(value, str):
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
        timestamps[field] = parsed
    terminal = operation.get("terminal")
    if (
        not isinstance(terminal, bool)
        or terminal is not status.terminal
        or (timestamps["completed_at"] is not None) is not status.terminal
        or timestamps["updated_at"] < timestamps["created_at"]  # type: ignore[operator]
    ):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    allowed_phases = {
        RemoteComputeOperationStatus.PENDING: {RemoteComputeEffectPhase.INTENT_PERSISTED},
        # A worker holds the operation while it applies or reconciles; the
        # phase is whatever the previous attempt left behind.
        RemoteComputeOperationStatus.CLAIMED: {
            RemoteComputeEffectPhase.INTENT_PERSISTED,
            RemoteComputeEffectPhase.EFFECT_MAY_HAVE_OCCURRED,
            RemoteComputeEffectPhase.PROVIDER_ACCEPTED,
            RemoteComputeEffectPhase.COMPENSATION_REQUIRED,
        },
        RemoteComputeOperationStatus.PROVIDER_PENDING: {RemoteComputeEffectPhase.PROVIDER_ACCEPTED},
        RemoteComputeOperationStatus.PROVIDER_SUCCEEDED: {
            RemoteComputeEffectPhase.RECEIPT_PERSISTED,
            RemoteComputeEffectPhase.COMPENSATION_RECEIPT_PERSISTED,
        },
        RemoteComputeOperationStatus.COMPENSATION_PENDING: {RemoteComputeEffectPhase.COMPENSATION_REQUIRED},
        RemoteComputeOperationStatus.RECONCILIATION_REQUIRED: {RemoteComputeEffectPhase.EFFECT_MAY_HAVE_OCCURRED},
        RemoteComputeOperationStatus.SUCCEEDED: {
            RemoteComputeEffectPhase.RECEIPT_PERSISTED,
            RemoteComputeEffectPhase.COMPENSATION_RECEIPT_PERSISTED,
        },
        RemoteComputeOperationStatus.FAILED: {
            RemoteComputeEffectPhase.INTENT_PERSISTED,
            RemoteComputeEffectPhase.NO_EFFECT_RECEIPT_PERSISTED,
        },
        RemoteComputeOperationStatus.COMPENSATED: {RemoteComputeEffectPhase.COMPENSATION_RECEIPT_PERSISTED},
        RemoteComputeOperationStatus.CANCELLED: {RemoteComputeEffectPhase.INTENT_PERSISTED},
    }
    if phase not in allowed_phases.get(status, set()):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")


def _validate_preflight(preflight: Mapping[str, Any]) -> None:
    required = preflight.get("required")
    local = preflight.get("local_option")
    cloud = preflight.get("cloud_options")
    recommended = preflight.get("recommended_target")
    estimated = preflight.get("estimated_cost")
    if (
        not isinstance(required, Mapping)
        or not isinstance(local, Mapping)
        or local.get("target") != "local_cpu"
        or not isinstance(cloud, list)
        or len(cloud) > 12
        or any(not isinstance(item, Mapping) for item in cloud)
        or not isinstance(recommended, str)
        or not recommended
        or not isinstance(estimated, Mapping)
    ):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    targets = {"local_cpu"}
    for item in cloud:
        target = item.get("target")
        if not isinstance(target, str) or not target.startswith(("cloud_gpu:", "cloud_cpu:")) or target in targets:
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
        targets.add(target)
    if recommended not in targets:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")


def _validate_selection(
    data: Mapping[str, Any],
    *,
    expected_target: str,
    expected_operation_id: UUID | None = None,
) -> None:
    expected_fields = {"selected_target", "resource", "usage", "replayed"}
    if expected_operation_id is not None:
        expected_fields.add("operation_id")
    if set(data) != expected_fields:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    if data.get("selected_target") != expected_target or not isinstance(data.get("replayed"), bool):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    resource = data.get("resource")
    usage = data.get("usage")
    if expected_target == "local_cpu":
        if resource is not None or usage is not None:
            raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
        return
    try:
        operation_id = parse_typed_id(
            data.get("operation_id"),
            expected_prefix="op",
        )
    except InvalidTypedId as exc:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID") from exc
    if operation_id != expected_operation_id:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_AUTHORITY_MISMATCH")
    if not isinstance(resource, Mapping) or not isinstance(usage, Mapping):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    try:
        resource_id = parse_typed_id(resource.get("resource_id"), expected_prefix="rcres")
        usage_resource_id = parse_typed_id(usage.get("resource_id"), expected_prefix="rcres")
        parse_typed_id(usage.get("binding_id"), expected_prefix="rcbind")
    except InvalidTypedId as exc:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID") from exc
    if resource_id != usage_resource_id:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")


def _validate_pending_selection(
    data: Mapping[str, Any],
    *,
    context: ExecutionRunContext,
    expected_target: str,
    expected_operation_id: UUID,
) -> None:
    if set(data) != {
        "selected_target",
        "operation_id",
        "operation",
        "status_path",
        "retry_after",
    }:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    if data.get("selected_target") != expected_target:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    operation = data.get("operation")
    retry_after = data.get("retry_after")
    if (
        not isinstance(operation, Mapping)
        or isinstance(retry_after, bool)
        or not isinstance(retry_after, int)
        or not 1 <= retry_after <= 300
        or data.get("operation_id") != format_typed_id("op", expected_operation_id)
        or data.get("status_path")
        != ("/internal/v2/remote-compute/operations/" + format_typed_id("op", expected_operation_id))
    ):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")
    _validate_operation_status(
        operation,
        expected_operation_id=expected_operation_id,
        context=context,
    )


def _selection_identity(
    context: ExecutionRunContext,
    *,
    prompt: str,
    selected_target: str,
    resource_policy: Mapping[str, int],
) -> tuple[UUID, str]:
    try:
        canonical = json.dumps(
            {
                "contract": "remote-compute-selection@v2",
                "prompt": prompt,
                "resource_policy": dict(resource_policy),
                # The Product side keys replays on (tenant, idempotency_key), so
                # the run must be part of it: the same user re-running the same
                # prompt is a new rent, not a replay of the old operation.
                "rid": format_typed_id("rid", context.rid),
                "selected_target": selected_target,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Remote Compute selection is not canonical") from exc
    digest = hashlib.sha256(canonical).digest()
    operation_id = derive_uuid7(
        context.rid,
        b"deepevol:remote-compute-selection:v2\0" + digest,
    )
    return operation_id, "rcsel_" + digest.hex()


def _validate_usage(
    usage: Mapping[str, Any],
    *,
    context: ExecutionRunContext,
) -> None:
    try:
        resource_tid = parse_typed_id(usage.get("resource_tid"), expected_prefix="tid")
        owner_uid = parse_typed_id(usage.get("owner_uid"), expected_prefix="uid")
        rid = parse_typed_id(usage.get("rid"), expected_prefix="rid")
        parse_typed_id(usage.get("binding_id"), expected_prefix="rcbind")
        parse_typed_id(usage.get("resource_id"), expected_prefix="rcres")
    except InvalidTypedId as exc:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID") from exc
    if (resource_tid, owner_uid, rid) != (
        context.resource_tid,
        context.owner_uid,
        context.rid,
    ):
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_AUTHORITY_MISMATCH")


def _requirements(value: Mapping[str, int | bool]) -> dict[str, int | bool]:
    out: dict[str, int | bool] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 64:
            raise ValueError("remote compute requirement key is invalid")
        if isinstance(item, bool):
            out[key] = item
        elif isinstance(item, int) and item >= 0:
            out[key] = item
        else:
            raise ValueError("remote compute requirement value is invalid")
    if len(out) > 16:
        raise ValueError("remote compute requirements are oversized")
    return out


def _bounded_json(value: Mapping[str, Any], *, maximum_bytes: int) -> None:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID") from exc
    if len(encoded) > maximum_bytes:
        raise ProductRemoteComputeUnavailable("REMOTE_COMPUTE_RESPONSE_INVALID")


__all__ = [
    "ProductRemoteComputeLifecycleClient",
    "ProductRemoteComputePending",
    "ProductRemoteComputeRejected",
    "ProductRemoteComputeUnavailable",
]
