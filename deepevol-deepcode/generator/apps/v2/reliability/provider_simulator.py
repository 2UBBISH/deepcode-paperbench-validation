"""Queryable, deterministic Provider simulator for the V2 reliability profile.

The simulator is deliberately impossible to construct outside the exact test
environment.  It models the boundary that matters to recovery tests: an
effect can be refused before it happens, or succeed while its response is
lost afterwards.  Durable application code can then reconcile by stable
idempotency key or Provider request ID instead of blindly sending again.
"""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from .canonical import canonical_json_sha256


_MAX_BODY_BYTES = 1024 * 1024
_MAX_KEY_BYTES = 512
_MAX_TOKEN_BYTES = 512
_BEFORE_EFFECT = "before_effect"
_AFTER_EFFECT = "after_effect"
_HTTP_ERROR = "http_error"
_DISCONNECT = "disconnect"


class ProviderSimulatorConfigurationError(ValueError):
    """Raised when the test-only construction gates are incomplete."""


class _DisconnectAfterEffectResponse(Response):
    """Start an HTTP response and deliberately leave its body incomplete.

    This exercises the materially different boundary where the Provider has
    committed an effect but the caller cannot know the result from the socket.
    The simulator process remains alive so its control plane can reconcile the
    operation by the stable idempotency key.
    """

    media_type = "application/json"

    def __init__(self, *, request_id: str) -> None:
        partial = b'{"simulated":"response-started"'
        super().__init__(content=partial, status_code=200)
        self.headers["content-length"] = str(len(partial) + 4096)
        self.headers["x-request-id"] = request_id

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": self.body,
                "more_body": True,
            }
        )


@dataclass(frozen=True, slots=True)
class ProviderSimulatorSettings:
    environment: str
    enabled: bool
    api_key: str
    control_token: str
    model: str = "reliability-model-v1"

    def __post_init__(self) -> None:
        if self.environment != "test" or not self.enabled:
            raise ProviderSimulatorConfigurationError(
                "Provider simulator requires environment=test and explicit enablement"
            )
        _require_secret(self.api_key, "api_key", minimum=16)
        _require_secret(self.control_token, "control_token", minimum=32)
        _require_text(self.model, "model", maximum=256)
        if hmac.compare_digest(self.api_key, self.control_token):
            raise ProviderSimulatorConfigurationError(
                "Provider API and control identities must be distinct"
            )

    @classmethod
    def from_env(cls, source: dict[str, str] | None = None) -> "ProviderSimulatorSettings":
        values = os.environ if source is None else source
        return cls(
            environment=values.get("DEEPEVOL_ENVIRONMENT", "").strip().lower(),
            enabled=(
                values.get("DEEPEVOL_V2_RELIABILITY_PROVIDER_SIMULATOR_ENABLED", "")
                == "1"
            ),
            api_key=values.get("DEEPEVOL_V2_RELIABILITY_PROVIDER_API_KEY", ""),
            control_token=values.get(
                "DEEPEVOL_V2_RELIABILITY_PROVIDER_CONTROL_TOKEN",
                "",
            ),
            model=values.get(
                "DEEPEVOL_V2_RELIABILITY_PROVIDER_MODEL",
                "reliability-model-v1",
            ),
        )


@dataclass(slots=True)
class _Fault:
    phase: str
    remaining: int
    status_code: int
    retry_after_seconds: int | None
    transport: str

    def consume(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


@dataclass(frozen=True, slots=True)
class _Operation:
    request_id: str
    idempotency_key: str
    request_sha256: str
    response: dict[str, Any]
    usage: dict[str, int]
    created_at: str

    def public_document(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "effect_count": 1,
            "idempotency_key": self.idempotency_key,
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "response": self.response,
            "usage": self.usage,
        }


class ProviderSimulatorState:
    """Thread-safe in-memory authority for one disposable simulator process."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._operations_by_key: dict[str, _Operation] = {}
        self._operations_by_request: dict[str, _Operation] = {}
        self._requests: list[dict[str, Any]] = []
        self._faults: dict[str, _Fault] = {}

    def reset(self) -> None:
        with self._lock:
            self._operations_by_key.clear()
            self._operations_by_request.clear()
            self._requests.clear()
            self._faults.clear()

    def configure_fault(
        self,
        *,
        phase: str,
        count: int,
        status_code: int,
        retry_after_seconds: int | None,
        transport: str = _HTTP_ERROR,
    ) -> dict[str, Any]:
        if phase not in {_BEFORE_EFFECT, _AFTER_EFFECT}:
            raise ValueError("phase must be before_effect or after_effect")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 10_000:
            raise ValueError("count must be an integer from 0 through 10000")
        if (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or not 400 <= status_code <= 599
        ):
            raise ValueError("status_code must be an HTTP error status")
        if retry_after_seconds is not None and (
            isinstance(retry_after_seconds, bool)
            or not isinstance(retry_after_seconds, int)
            or not 0 <= retry_after_seconds <= 86_400
        ):
            raise ValueError("retry_after_seconds must be between 0 and 86400")
        if transport not in {_HTTP_ERROR, _DISCONNECT}:
            raise ValueError("transport must be http_error or disconnect")
        if transport == _DISCONNECT and phase != _AFTER_EFFECT:
            raise ValueError("disconnect transport is valid only after_effect")
        fault = _Fault(phase, count, status_code, retry_after_seconds, transport)
        with self._lock:
            self._faults[phase] = fault
            return _fault_document(fault)

    def apply(
        self,
        *,
        idempotency_key: str,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any], dict[str, str], bool]:
        request_sha256 = canonical_json_sha256(payload)
        with self._lock:
            existing = self._operations_by_key.get(idempotency_key)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    self._record_request(
                        idempotency_key=idempotency_key,
                        request_sha256=request_sha256,
                        request_id=existing.request_id,
                        outcome="IDEMPOTENCY_CONFLICT",
                        effect_applied=False,
                        status_code=409,
                    )
                    return 409, {
                        "error": {
                            "code": "IDEMPOTENCY_CONFLICT",
                            "message": "the idempotency key is bound to another request",
                        }
                    }, {}, False
                self._record_request(
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    request_id=existing.request_id,
                    outcome="REPLAYED_RECEIPT",
                    effect_applied=False,
                    status_code=200,
                )
                return 200, existing.response, {"x-request-id": existing.request_id}, False

            before = self._faults.get(_BEFORE_EFFECT)
            if before is not None and before.consume():
                self._record_request(
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    request_id=None,
                    outcome="FAILED_BEFORE_EFFECT",
                    effect_applied=False,
                    status_code=before.status_code,
                )
                status, document, headers = _fault_response(before, phase=_BEFORE_EFFECT)
                return status, document, headers, False

            request_id = f"simreq_{canonical_json_sha256({'key': idempotency_key})[:32]}"
            usage = _usage_for(payload)
            response = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": f"simulated:{request_sha256[:24]}",
                            "role": "assistant",
                        },
                    }
                ],
                "created": 0,
                "id": request_id,
                "model": str(payload.get("model") or "reliability-model-v1"),
                "object": "chat.completion",
                "usage": usage,
            }
            operation = _Operation(
                request_id=request_id,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response=response,
                usage=usage,
                created_at=datetime.now(UTC).isoformat(),
            )
            self._operations_by_key[idempotency_key] = operation
            self._operations_by_request[request_id] = operation

            after = self._faults.get(_AFTER_EFFECT)
            if after is not None and after.consume():
                disconnected = after.transport == _DISCONNECT
                self._record_request(
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    request_id=request_id,
                    outcome=(
                        "RESPONSE_DISCONNECTED_AFTER_EFFECT"
                        if disconnected
                        else "RESPONSE_LOST_AFTER_EFFECT"
                    ),
                    effect_applied=True,
                    status_code=0 if disconnected else after.status_code,
                )
                if disconnected:
                    return 200, response, {"x-request-id": request_id}, True
                status, document, headers = _fault_response(after, phase=_AFTER_EFFECT)
                return status, document, headers, False

            self._record_request(
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                request_id=request_id,
                outcome="EFFECT_APPLIED",
                effect_applied=True,
                status_code=200,
            )
            return 200, response, {"x-request-id": request_id}, False

    def operation_by_key(self, idempotency_key: str) -> dict[str, Any] | None:
        with self._lock:
            operation = self._operations_by_key.get(idempotency_key)
            return operation.public_document() if operation is not None else None

    def operation_by_request(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            operation = self._operations_by_request.get(request_id)
            return operation.public_document() if operation is not None else None

    def usage_by_request(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            operation = self._operations_by_request.get(request_id)
            if operation is None:
                return None
            return {"request_id": operation.request_id, "usage": operation.usage}

    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._requests]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "faults": {
                    phase: _fault_document(fault)
                    for phase, fault in sorted(self._faults.items())
                },
                "operation_count": len(self._operations_by_key),
                "request_count": len(self._requests),
            }

    def _record_request(
        self,
        *,
        idempotency_key: str,
        request_sha256: str,
        request_id: str | None,
        outcome: str,
        effect_applied: bool,
        status_code: int,
    ) -> None:
        self._requests.append(
            {
                "attempt_ordinal": len(self._requests) + 1,
                "effect_applied": effect_applied,
                "idempotency_key": idempotency_key,
                "outcome": outcome,
                "request_id": request_id,
                "request_sha256": request_sha256,
                "status_code": status_code,
            }
        )


def create_provider_simulator(
    settings: ProviderSimulatorSettings,
    *,
    state: ProviderSimulatorState | None = None,
) -> FastAPI:
    simulator = state or ProviderSimulatorState()
    app = FastAPI(
        title="DeepEvol V2 reliability Provider simulator",
        version="1.0.0-test-only",
    )
    app.state.provider_simulator = simulator

    def provider_authorized(authorization: str | None) -> bool:
        expected = f"Bearer {settings.api_key}"
        return authorization is not None and hmac.compare_digest(authorization, expected)

    def control_authorized(token: str | None) -> bool:
        return token is not None and hmac.compare_digest(token, settings.control_token)

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready() -> dict[str, str]:
        return {"mode": "test-only", "status": "ready"}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> JSONResponse:
        if not provider_authorized(authorization):
            return _unauthorized()
        return JSONResponse({"data": [{"id": settings.model, "object": "model"}], "object": "list"})

    @app.post("/v1/chat/completions")
    async def chat_completion(
        request: Request,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        if not provider_authorized(authorization):
            return _unauthorized()
        try:
            key = _idempotency_key(idempotency_key)
            payload = _strict_body(await request.body())
            if payload.get("stream") is True:
                raise ValueError("streaming is not implemented by this deterministic simulator")
            status, document, headers, disconnect = simulator.apply(
                idempotency_key=key,
                payload=payload,
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": {"code": "INVALID_REQUEST", "message": str(exc)}},
                status_code=422,
            )
        if disconnect:
            return _DisconnectAfterEffectResponse(request_id=headers["x-request-id"])
        return JSONResponse(document, status_code=status, headers=headers)

    @app.put("/__test__/faults")
    async def configure_fault(
        request: Request,
        x_test_control_token: str | None = Header(default=None),
    ) -> JSONResponse:
        if not control_authorized(x_test_control_token):
            return _unauthorized()
        try:
            payload = _strict_body(await request.body())
            document = simulator.configure_fault(
                phase=payload.get("phase"),
                count=payload.get("count"),
                status_code=payload.get("status_code", 503),
                retry_after_seconds=payload.get("retry_after_seconds"),
                transport=payload.get("transport", _HTTP_ERROR),
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": {"code": "INVALID_FAULT", "message": str(exc)}},
                status_code=422,
            )
        return JSONResponse(document)

    @app.post("/__test__/reset")
    def reset(x_test_control_token: str | None = Header(default=None)) -> JSONResponse:
        if not control_authorized(x_test_control_token):
            return _unauthorized()
        simulator.reset()
        return JSONResponse({"status": "reset"})

    @app.get("/__test__/operations/by-idempotency-key")
    def operation_by_key(
        idempotency_key: str,
        x_test_control_token: str | None = Header(default=None),
    ) -> JSONResponse:
        if not control_authorized(x_test_control_token):
            return _unauthorized()
        try:
            document = simulator.operation_by_key(_idempotency_key(idempotency_key))
        except ValueError as exc:
            return JSONResponse(
                {"error": {"code": "INVALID_REQUEST", "message": str(exc)}},
                status_code=422,
            )
        return _found(document)

    @app.get("/__test__/operations/{request_id}")
    def operation_by_request(
        request_id: str,
        x_test_control_token: str | None = Header(default=None),
    ) -> JSONResponse:
        if not control_authorized(x_test_control_token):
            return _unauthorized()
        return _found(simulator.operation_by_request(request_id))

    @app.get("/__test__/usage/{request_id}")
    def usage_by_request(
        request_id: str,
        x_test_control_token: str | None = Header(default=None),
    ) -> JSONResponse:
        if not control_authorized(x_test_control_token):
            return _unauthorized()
        return _found(simulator.usage_by_request(request_id))

    @app.get("/__test__/requests")
    def requests(x_test_control_token: str | None = Header(default=None)) -> JSONResponse:
        if not control_authorized(x_test_control_token):
            return _unauthorized()
        return JSONResponse({"requests": simulator.requests(), **simulator.snapshot()})

    return app


def create_provider_simulator_from_env() -> FastAPI:
    return create_provider_simulator(ProviderSimulatorSettings.from_env())


def _strict_body(body: bytes) -> dict[str, Any]:
    if not body or len(body) > _MAX_BODY_BYTES:
        raise ValueError("request body must be non-empty and at most 1 MiB")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for name, value in pairs:
            if name in document:
                raise ValueError("request JSON contains duplicate object keys")
            document[name] = value
        return document

    try:
        payload = json.loads(
            body,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _usage_for(payload: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = max(1, len(json.dumps(payload.get("messages", []), ensure_ascii=False)) // 4)
    completion_tokens = 8
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _idempotency_key(value: str | None) -> str:
    if (
        value is None
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_KEY_BYTES
    ):
        raise ValueError("Idempotency-Key must contain 1 to 512 bounded UTF-8 bytes")
    return value


def _require_secret(value: str, name: str, *, minimum: int) -> None:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) < minimum
        or len(value.encode("utf-8")) > _MAX_TOKEN_BYTES
    ):
        raise ProviderSimulatorConfigurationError(
            f"{name} must contain {minimum} to {_MAX_TOKEN_BYTES} UTF-8 bytes"
        )


def _require_text(value: str, name: str, *, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > maximum:
        raise ProviderSimulatorConfigurationError(
            f"{name} must be non-empty and at most {maximum} UTF-8 bytes"
        )


def _fault_document(fault: _Fault) -> dict[str, Any]:
    return {
        "phase": fault.phase,
        "remaining": fault.remaining,
        "retry_after_seconds": fault.retry_after_seconds,
        "status_code": fault.status_code,
        "transport": fault.transport,
    }


def _fault_response(fault: _Fault, *, phase: str) -> tuple[int, dict[str, Any], dict[str, str]]:
    headers = (
        {"Retry-After": str(fault.retry_after_seconds)}
        if fault.retry_after_seconds is not None
        else {}
    )
    return fault.status_code, {
        "error": {
            "code": "INJECTED_PROVIDER_FAULT",
            "message": f"test fault injected at {phase}",
            "phase": phase,
        }
    }, headers


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {"error": {"code": "UNAUTHORIZED", "message": "invalid simulator credential"}},
        status_code=401,
    )


def _found(document: dict[str, Any] | None) -> JSONResponse:
    if document is None:
        return JSONResponse(
            {"error": {"code": "NOT_FOUND", "message": "operation was not found"}},
            status_code=404,
        )
    return JSONResponse(document)


__all__ = [
    "ProviderSimulatorConfigurationError",
    "ProviderSimulatorSettings",
    "ProviderSimulatorState",
    "create_provider_simulator",
    "create_provider_simulator_from_env",
]
