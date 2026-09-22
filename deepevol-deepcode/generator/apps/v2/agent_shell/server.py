"""The shell itself: one loopback HTTP server per run attempt.

Two faces, one token:

* ``/v1/chat/completions`` -> the run's Gateway model (admission + billing);
* ``/placement``, ``/exec``, ``/jobs/*``, ``/fs/*`` -> the bound
  :class:`~.targets.ExecutionTarget`.

The server is synchronous HTTP (``ThreadingHTTPServer``, like the existing
loopback endpoint) fronting a private asyncio loop that owns the target: relay
runtimes are asyncio objects and must be driven from the loop that created
them.  ``bind_target`` / ``unbind_target`` switch placements atomically; an
Agent request that races a switch sees ``SHELL_PLACEMENT_UNAVAILABLE`` and
retries, never a half-closed session.

Nothing here imports vendored agent code.  The only Agent-facing values are
``base_url`` and ``token``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import httpx

from .chat import complete_chat
from .egress import EgressError, EgressGateway, KeyPoolPort, MAX_UPSTREAM_RESPONSE_BYTES
from .ledger import ShellUsageLedger
from .metering import MeteredGatewayBackend
from .protocol import (
    MAX_JSON_BODY_BYTES,
    MAX_UPLOAD_BYTES,
    SHELL_PROTOCOL_VERSION,
    ExecRequest,
    Placement,
    PlacementKind,
    ShellErrorCode,
    encode,
)
from .targets import ExecutionTarget

logger = logging.getLogger(__name__)

_STREAM_QUEUE_END = object()


class _RequestError(Exception):
    def __init__(self, status: HTTPStatus, code: ShellErrorCode, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class AgentShell:
    """Per-run middle layer.  Construct, ``bind_target`` when a placement exists,
    hand ``base_url``/``token`` to the Agent, ``close`` with the attempt."""

    def __init__(
        self,
        *,
        model_facade: Any | None = None,
        model_name: str = "",
        gateway_backend: Any | None = None,
        key_pools: KeyPoolPort | None = None,
        ledger: ShellUsageLedger | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        stream_chunk_timeout: float = 30.0,
        egress_timeout: float = 60.0,
        # Standalone (sidecar) mode: the parent drives ``/control/*`` with
        # this token and describes placements as JSON specs that
        # ``placement_factory`` turns into ExecutionTargets.
        control_token: str | None = None,
        placement_factory: Callable[[Mapping[str, Any]], ExecutionTarget] | None = None,
        # Host to announce in base_url when the shell listens on a non-loopback
        # address (Agents in other containers reach it through this name).
        advertise_host: str | None = None,
        # Model relay: an OpenAI-compatible endpoint (the worker's in-process
        # shell face) that answers ``/v1/chat/completions`` for this attempt.
        model_relay: tuple[str, str] | None = None,
    ) -> None:
        self._facade = model_facade
        self._model_name = str(model_name or "").strip()
        self.ledger = ledger or ShellUsageLedger()
        self._control_token = control_token
        self._placement_factory = placement_factory
        self._advertise_host = (advertise_host or "").strip() or None
        self._model_relay = model_relay
        self._relay_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0), trust_env=False)
        self._shutdown_requested = threading.Event()
        # Credentialed provider egress (research key pools live behind it).
        self._egress = EgressGateway(key_pools)
        self._egress_timeout = float(egress_timeout)
        self._egress_client = httpx.Client(timeout=httpx.Timeout(self._egress_timeout, connect=10.0), follow_redirects=False, trust_env=False)
        # In-process lines (LangChain models over the Gateway) get a metered
        # view of the run's backend so their usage lands in the same ledger.
        self._gateway_backend = (
            None if gateway_backend is None else MeteredGatewayBackend(gateway_backend, self.ledger)
        )
        self._token = secrets.token_urlsafe(32)
        self._target: ExecutionTarget | None = None
        self._target_lock = threading.Lock()
        self._generation = 0
        self._stream_chunk_timeout = float(stream_chunk_timeout)
        self._closed = False

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="agent-shell-loop", daemon=True
        )
        self._loop_thread.start()

        self._server = ThreadingHTTPServer((host, int(port)), self._handler_class())
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, name="agent-shell-http", daemon=True
        )
        self._server_thread.start()

    # ---------------------------------------------------------------- public
    @property
    def token(self) -> str:
        return self._token

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{self._advertise_host or host}:{port}"

    @property
    def local_url(self) -> str:
        """The bound address itself (for the parent on the same host)."""
        host, port = self._server.server_address[:2]
        return f"http://{'127.0.0.1' if host in {'0.0.0.0', ''} else host}:{port}"

    @property
    def openai_base_url(self) -> str:
        """What goes into ``OPENAI_BASE_URL`` for OpenAI-client callers."""

        return f"{self.base_url}/v1"

    @property
    def gateway_backend(self) -> Any:
        """The run's Gateway backend, metered.  Hand this to every line that
        builds a ``GatewayChatModel`` instead of the raw backend."""

        if self._gateway_backend is None:
            raise RuntimeError("this shell was created without a Gateway backend")
        return self._gateway_backend

    @property
    def model_name(self) -> str:
        return self._model_name

    def bind_model_relay(self, base_url: str, token: str, *, model_name: str = "") -> None:
        """Answer ``/v1`` by forwarding to another OpenAI-compatible endpoint
        (sidecar mode: the worker's in-process shell face)."""

        self._model_relay = (base_url.rstrip("/"), token)
        if model_name:
            self._model_name = str(model_name).strip()

    @property
    def shutdown_requested(self) -> threading.Event:
        return self._shutdown_requested

    def bind_model(self, facade: Any, *, model_name: str) -> None:
        """Attach (or replace) the OpenAI-shaped facade behind ``/v1``.  Lines
        that need raw OpenAI HTTP (SetupX / rsa) call this with a facade built
        over :attr:`gateway_backend`; the metered backend books the calls, the
        HTTP face then does not book them again."""

        self._facade = facade
        self._model_name = str(model_name or "").strip()

    @property
    def placement(self) -> Placement:
        with self._target_lock:
            target = self._target
        if target is None:
            return Placement(kind=PlacementKind.NONE, generation=self._generation, ready=False)
        placement = target.placement
        return Placement(kind=placement.kind, generation=self._generation, label=placement.label,
                         ready=placement.ready)

    def bind_target(self, target: ExecutionTarget, *, start: bool = True, timeout: float = 600.0) -> Placement:
        """Route execution to ``target`` from now on.  The previous target is
        closed.  ``start`` opens the target's session on the shell loop before
        the placement is advertised, so ``ready`` means "commands will run"."""

        if self._closed:
            raise RuntimeError("agent shell is closed")
        if start:
            self._run(target.start(), timeout=timeout)
        with self._target_lock:
            previous, self._target = self._target, target
            self._generation += 1
        if previous is not None:
            self._close_target(previous)
        return self.placement

    def unbind_target(self) -> None:
        with self._target_lock:
            previous, self._target = self._target, None
        if previous is not None:
            self._close_target(previous)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.unbind_target()
        finally:
            for client in (self._egress_client, self._relay_client):
                try:
                    client.close()
                except Exception:  # pragma: no cover
                    pass
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:  # pragma: no cover - best-effort teardown
                logger.debug("agent shell http close failed", exc_info=True)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            if not self._loop.is_running():
                self._loop.close()

    def __enter__(self) -> "AgentShell":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --------------------------------------------------------------- internals
    def _run(self, coroutine: Any, *, timeout: float | None = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=timeout)

    def _close_target(self, target: ExecutionTarget) -> None:
        try:
            self._run(target.close(), timeout=60)
        except Exception:
            logger.warning("agent shell: closing the previous placement failed", exc_info=True)

    def _current_target(self) -> tuple[ExecutionTarget, Placement]:
        with self._target_lock:
            target = self._target
        if target is None:
            raise _RequestError(HTTPStatus.SERVICE_UNAVAILABLE, ShellErrorCode.PLACEMENT_UNAVAILABLE,
                                "no execution placement is bound to this run")
        return target, self.placement

    # ------------------------------------------------------------ operations
    def _op_egress(self, provider: str, query: Mapping[str, list[str]], method: str,
                   headers: Mapping[str, str], body: bytes) -> tuple[int, dict[str, str], bytes]:
        upstream = (query.get("u") or [""])[0]
        if not upstream:
            raise _RequestError(HTTPStatus.BAD_REQUEST, ShellErrorCode.BAD_REQUEST, "egress needs ?u=<upstream url>")
        forward = {k: v for k, v in headers.items() if k.lower() not in {"x-shell-token", "x-shell-timeout"}}
        try:
            plan = self._egress.plan(provider, upstream=upstream, method=method, headers=forward, body=body)
        except EgressError as exc:
            self.ledger.record_egress(provider, failed=True)
            raise _RequestError(HTTPStatus(exc.status), ShellErrorCode.EGRESS_REFUSED, f"{exc.code}: {exc}") from exc
        timeout = self._egress_timeout
        raw_timeout = headers.get("X-Shell-Timeout") or headers.get("x-shell-timeout")
        if raw_timeout:
            try:
                timeout = max(1.0, min(float(raw_timeout), 900.0))
            except ValueError:
                pass
        started = time.monotonic()
        try:
            response = self._egress_client.request(
                plan.method, plan.upstream, headers=plan.headers, content=plan.body or None,
                timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            )
        except httpx.HTTPError as exc:
            self.ledger.record_egress(provider, failed=True, seconds=time.monotonic() - started)
            raise _RequestError(HTTPStatus.BAD_GATEWAY, ShellErrorCode.EGRESS_FAILED, f"{type(exc).__name__}: {exc}") from exc
        content = response.content
        if len(content) > MAX_UPSTREAM_RESPONSE_BYTES:
            self.ledger.record_egress(provider, failed=True, seconds=time.monotonic() - started)
            raise _RequestError(HTTPStatus.BAD_GATEWAY, ShellErrorCode.EGRESS_FAILED, "upstream response exceeds the egress cap")
        self._egress.settle(plan, response.status_code)
        self.ledger.record_egress(provider, failed=response.status_code >= 400, seconds=time.monotonic() - started,
                                 bytes_in=len(content), bytes_out=len(body), credentialed=plan.credential is not None)
        out_headers = {k: v for k, v in response.headers.items()
                       if k.lower() in {"content-type", "retry-after", "x-ratelimit-remaining", "x-ratelimit-limit", "etag", "cache-control"}}
        # The Agent must never learn which key served it.
        return response.status_code, out_headers, content

    def _op_chat(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._facade is None and self._model_relay is not None:
            return self._op_chat_relay(payload)
        if self._facade is None:
            raise _RequestError(HTTPStatus.SERVICE_UNAVAILABLE, ShellErrorCode.MODEL_UNAVAILABLE,
                                "this run has no model bound to its shell")
        # With a metered backend the facade is built over it and books its
        # own calls; only a facade over a raw backend is booked here.
        book = self._gateway_backend is None
        try:
            body, usage = complete_chat(self._facade, payload, default_model=self._model_name)
        except ValueError:
            raise
        except Exception as exc:
            if book:
                self.ledger.record_llm(None, failed=True)
            logger.warning("agent shell: gateway call failed: %s", type(exc).__name__, exc_info=True)
            raise _RequestError(HTTPStatus.BAD_GATEWAY, ShellErrorCode.MODEL_FAILED,
                                f"gateway invocation failed: {type(exc).__name__}") from exc
        if book:
            self.ledger.record_llm(usage)
        return body

    def _op_ready(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Bring the placement up (opening its session if it was bound lazily)
        and wait until it runs commands.  Auth failures on a relay target
        surface immediately as SHELL_TARGET_FAILED; the lease's bounded
        bring-up retry decides what to do with them."""

        target, _ = self._current_target()
        timeout = float(payload.get("timeout") or 300.0)
        if not timeout > 0:
            raise ValueError("timeout must be positive")
        self._run(target.wait_ready(timeout=timeout), timeout=None)
        return self.placement.to_dict()

    def _op_chat_relay(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        assert self._model_relay is not None
        base_url, token = self._model_relay
        if payload.get("stream"):
            raise ValueError("streaming is not supported by the shell")
        try:
            response = self._relay_client.post(
                f"{base_url}/v1/chat/completions", json=dict(payload), headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            self.ledger.record_llm(None, failed=True)
            raise _RequestError(HTTPStatus.BAD_GATEWAY, ShellErrorCode.MODEL_FAILED, f"model relay failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            self.ledger.record_llm(None, failed=True)
            try:
                message = str((response.json().get("error") or {}).get("message") or "")
            except ValueError:
                message = ""
            raise _RequestError(HTTPStatus(response.status_code) if response.status_code in HTTPStatus.__members__.values() else HTTPStatus.BAD_GATEWAY,
                                ShellErrorCode.MODEL_FAILED, message or f"model relay answered {response.status_code}")
        body = response.json()
        usage = body.get("usage") if isinstance(body, Mapping) else None
        self.ledger.record_llm({
            "input_tokens": int((usage or {}).get("prompt_tokens") or 0),
            "output_tokens": int((usage or {}).get("completion_tokens") or 0),
            "total_tokens": int((usage or {}).get("total_tokens") or 0),
        } if isinstance(usage, Mapping) else None)
        return body

    # --------------------------------------------------------------- control
    def _op_control(self, method: str, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if action == "placement" and method == "POST":
            if self._placement_factory is None:
                raise _RequestError(HTTPStatus.NOT_IMPLEMENTED, ShellErrorCode.PLACEMENT_UNAVAILABLE,
                                    "this shell cannot build placements from specs")
            spec = payload.get("spec")
            if not isinstance(spec, Mapping):
                raise ValueError("placement spec must be an object")
            target = self._placement_factory(spec)
            placement = self.bind_target(target, start=bool(payload.get("start", True)),
                                         timeout=float(payload.get("timeout") or 600.0))
            return placement.to_dict()
        if action == "placement" and method == "DELETE":
            self.unbind_target()
            return self.placement.to_dict()
        if action == "model" and method == "POST":
            base_url, token = payload.get("base_url"), payload.get("token")
            if not isinstance(base_url, str) or not isinstance(token, str) or not base_url or not token:
                raise ValueError("model relay needs base_url and token")
            self.bind_model_relay(base_url, token, model_name=str(payload.get("model_name") or ""))
            return {"ok": True, "model": self._model_name}
        if action == "usage" and method == "GET":
            return self.ledger.snapshot()
        if action == "shutdown" and method == "POST":
            self._shutdown_requested.set()
            return {"ok": True}
        if action == "health" and method == "GET":
            return {"ok": True, "placement": self.placement.to_dict(), "model": bool(self._facade or self._model_relay)}
        raise _RequestError(HTTPStatus.NOT_FOUND, ShellErrorCode.NOT_FOUND, f"no control action {method} {action}")

    def _op_exec(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = ExecRequest.from_dict(payload)
        target, placement = self._current_target()
        started = time.monotonic()
        failed = True
        extra: dict[str, Any] = {}
        if request.command_id and getattr(target, "accepts_command_id", False):
            extra["command_id"] = request.command_id
        try:
            result = self._run(
                target.exec(
                    request.command,
                    cwd=request.cwd,
                    env=request.env or None,
                    timeout=request.timeout,
                    durable=request.durable,
                    **extra,
                ),
                timeout=None,
            )
            failed = result.exit_status != 0
        finally:
            self.ledger.record_exec(placement, seconds=time.monotonic() - started, failed=failed)
        return encode(result)

    def _op_spawn(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = ExecRequest.from_dict(payload)
        job_id = payload.get("job_id")
        if job_id is not None and not isinstance(job_id, str):
            raise ValueError("job_id must be a string")
        target, placement = self._current_target()
        status = self._run(
            target.spawn(request.command, cwd=request.cwd, env=request.env or None, job_id=job_id)
        )
        self.ledger.record_spawn(placement)
        return encode(status)

    def _op_job_status(self, job_id: str) -> dict[str, Any]:
        target, _ = self._current_target()
        return encode(self._run(target.job_status(job_id)))

    def _op_job_wait(self, job_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        target, placement = self._current_target()
        timeout = payload.get("timeout")
        timeout = None if timeout is None else float(timeout)
        started = time.monotonic()
        result = self._run(target.wait(job_id, timeout=timeout), timeout=None)
        self.ledger.record_exec(placement, seconds=time.monotonic() - started, failed=result.exit_status != 0)
        return encode(result)

    def _op_job_kill(self, job_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        target, _ = self._current_target()
        sig = str(payload.get("sig") or "TERM")
        return {"killed": bool(self._run(target.kill(job_id, sig=sig)))}

    def _stream_job(self, job_id: str, query: Mapping[str, list[str]], write: Callable[[bytes], None]) -> None:
        """Pump ``target.stream`` events into chunked NDJSON on the HTTP thread."""

        target, _ = self._current_target()
        stdout_offset = int((query.get("stdout_offset") or ["0"])[0])
        stderr_offset = int((query.get("stderr_offset") or ["0"])[0])
        follow = (query.get("follow") or ["1"])[0] not in {"0", "false", "no"}
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for event in target.stream(
                    job_id, stdout_offset=stdout_offset, stderr_offset=stderr_offset, follow=follow
                ):
                    await queue.put(event)
            except Exception as exc:  # surfaced to the client as a terminal error line
                await queue.put(exc)
            finally:
                await queue.put(_STREAM_QUEUE_END)

        async def next_item() -> Any:
            return await queue.get()

        pump_future = asyncio.run_coroutine_threadsafe(pump(), self._loop)
        try:
            while True:
                item = self._run(next_item(), timeout=None)
                if item is _STREAM_QUEUE_END:
                    break
                if isinstance(item, Exception):
                    write(json.dumps({"error": {"code": str(ShellErrorCode.TARGET_FAILED),
                                                "message": f"{type(item).__name__}: {item}"}}).encode("utf-8") + b"\n")
                    break
                write(json.dumps(encode(item), ensure_ascii=False).encode("utf-8") + b"\n")
        finally:
            pump_future.cancel()

    def _op_fs(self, name: str, payload: Mapping[str, Any]) -> Any:
        target, placement = self._current_target()
        path = payload.get("path")
        if name in {"ls", "stat", "exists", "mkdir", "rm", "append", "read"} and not isinstance(path, str):
            raise ValueError("path must be a string")
        if name == "ls":
            return {"entries": encode(self._run(target.ls(path)))}
        if name == "glob":
            pattern = payload.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("pattern must be a non-empty string")
            return {"matches": list(self._run(target.glob(pattern)))}
        if name == "stat":
            return encode(self._run(target.stat(path)))
        if name == "exists":
            return {"exists": bool(self._run(target.exists(path)))}
        if name == "mkdir":
            self._run(target.mkdir(path, parents=bool(payload.get("parents", True))))
            return {"ok": True}
        if name == "rm":
            self._run(target.rm(path, recursive=bool(payload.get("recursive", False)),
                                missing_ok=bool(payload.get("missing_ok", True))))
            return {"ok": True}
        if name == "move":
            src, dst = payload.get("src"), payload.get("dst")
            if not isinstance(src, str) or not isinstance(dst, str):
                raise ValueError("src and dst must be strings")
            self._run(target.move(src, dst))
            return {"ok": True}
        if name == "append":
            content = payload.get("content")
            if not isinstance(content, str):
                raise ValueError("content must be a string")
            self._run(target.append_file(path, content, encoding=str(payload.get("encoding") or "utf-8")))
            self.ledger.record_transfer(placement, uploaded=len(content.encode("utf-8")))
            return {"ok": True}
        raise _RequestError(HTTPStatus.NOT_FOUND, ShellErrorCode.NOT_FOUND, f"unknown fs operation {name}")

    def _op_read_bytes(self, query: Mapping[str, list[str]]) -> bytes:
        target, placement = self._current_target()
        path = (query.get("path") or [""])[0]
        if not path:
            raise ValueError("path is required")
        raw_max = (query.get("max_bytes") or [""])[0]
        max_bytes = int(raw_max) if raw_max else None
        data = self._run(target.read_bytes(path, max_bytes=max_bytes), timeout=None)
        self.ledger.record_transfer(placement, downloaded=len(data))
        return bytes(data)

    def _op_write_bytes(self, query: Mapping[str, list[str]], data: bytes) -> dict[str, Any]:
        target, placement = self._current_target()
        path = (query.get("path") or [""])[0]
        if not path:
            raise ValueError("path is required")
        parents = (query.get("parents") or ["1"])[0] not in {"0", "false", "no"}
        self._run(target.write_bytes(path, data, parents=parents), timeout=None)
        self.ledger.record_transfer(placement, uploaded=len(data))
        return {"ok": True, "bytes": len(data)}

    # ---------------------------------------------------------------- handler
    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        shell = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "DeepEvolAgentShell/1"
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                logger.debug("agent shell %s", format % args)

            # -- replies ---------------------------------------------------
            def _json(self, status: HTTPStatus, body: Any) -> None:
                raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _bytes(self, data: bytes) -> None:
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _error(self, status: HTTPStatus, code: ShellErrorCode, message: str) -> None:
                self._json(status, {"error": {"code": str(code), "message": message}})

            # -- request plumbing ------------------------------------------
            def _authorized(self) -> bool:
                auth = self.headers.get("Authorization", "")
                if auth.startswith("Bearer ") and secrets.compare_digest(auth[7:].strip(), shell._token):
                    return True
                # Egress requests carry the upstream's own Authorization, so
                # the shell token travels in a dedicated header there.
                token = self.headers.get("X-Shell-Token", "")
                if token and secrets.compare_digest(token.strip(), shell._token):
                    return True
                # Keep-alive: the unread body would be parsed as the next
                # request line.  Drain small bodies, drop the connection for
                # anything larger.
                length = int(self.headers.get("Content-Length") or 0)
                if 0 < length <= MAX_JSON_BODY_BYTES:
                    self.rfile.read(length)
                elif length > MAX_JSON_BODY_BYTES:
                    self.close_connection = True
                self._error(HTTPStatus.UNAUTHORIZED, ShellErrorCode.UNAUTHORIZED, "invalid shell token")
                return False

            def _body(self, limit: int) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                if length < 0 or length > limit:
                    raise _RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, ShellErrorCode.PAYLOAD_TOO_LARGE,
                                        "body size out of bounds")
                return self.rfile.read(length) if length else b""

            def _json_body(self) -> Mapping[str, Any]:
                raw = self._body(MAX_JSON_BODY_BYTES)
                if not raw:
                    return {}
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("body must be JSON") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError("body must be an object")
                return payload

            def _control_authorized(self) -> bool:
                if shell._control_token is None:
                    return False
                auth = self.headers.get("Authorization", "")
                return auth.startswith("Bearer ") and secrets.compare_digest(auth[7:].strip(), shell._control_token)

            def _dispatch(self, method: str) -> None:
                parts = urlsplit(self.path)
                path = parts.path.rstrip("/") or "/"
                query = parse_qs(parts.query)
                if path.startswith("/control/"):
                    # The parent's face: its own token, never the Agent's.
                    if not self._control_authorized():
                        self._error(HTTPStatus.UNAUTHORIZED, ShellErrorCode.UNAUTHORIZED, "invalid control token")
                        return
                    try:
                        self._json(HTTPStatus.OK, shell._op_control(method, path[len("/control/"):], self._json_body() if method != "GET" else {}))
                    except _RequestError as exc:
                        self._error(exc.status, exc.code, str(exc))
                    except (ValueError, TypeError) as exc:
                        self._error(HTTPStatus.BAD_REQUEST, ShellErrorCode.BAD_REQUEST, str(exc))
                    except Exception as exc:
                        logger.warning("agent shell control %s %s failed", method, path, exc_info=True)
                        self._error(HTTPStatus.BAD_GATEWAY, ShellErrorCode.TARGET_FAILED, f"{type(exc).__name__}: {exc}")
                    return
                if not self._authorized():
                    return
                try:
                    self._route(method, path, query)
                except _RequestError as exc:
                    if exc.code is ShellErrorCode.PAYLOAD_TOO_LARGE:
                        self.close_connection = True
                    self._error(exc.status, exc.code, str(exc))
                except (ValueError, TypeError) as exc:
                    self._error(HTTPStatus.BAD_REQUEST, ShellErrorCode.BAD_REQUEST, str(exc))
                except NotImplementedError as exc:
                    self._error(HTTPStatus.NOT_IMPLEMENTED, ShellErrorCode.TARGET_FAILED, str(exc))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, ShellErrorCode.NOT_FOUND, f"{type(exc).__name__}: {exc}")
                except Exception as exc:
                    logger.warning("agent shell: %s %s failed: %s", method, path, type(exc).__name__, exc_info=True)
                    self._error(HTTPStatus.BAD_GATEWAY, ShellErrorCode.TARGET_FAILED,
                                f"{type(exc).__name__}: {exc}")

            def _route(self, method: str, path: str, query: Mapping[str, list[str]]) -> None:
                if path.startswith("/egress/"):
                    provider = path[len("/egress/"):].split("/", 1)[0]
                    if provider == "providers" and method == "GET":
                        self._json(HTTPStatus.OK, {"providers": shell._egress.providers()})
                        return
                    body = self._body(MAX_UPLOAD_BYTES)
                    status, headers, content = shell._op_egress(provider, query, method, dict(self.headers.items()), body)
                    self.send_response(int(status))
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                if method == "GET" and path == "/placement":
                    self._json(HTTPStatus.OK, {"protocol": SHELL_PROTOCOL_VERSION, **shell.placement.to_dict()})
                    return
                if method == "GET" and path == "/usage":
                    self._json(HTTPStatus.OK, shell.ledger.snapshot())
                    return
                if method == "GET" and path in {"/v1/models", "/models"}:
                    models = [shell._model_name] if shell._model_name else []
                    self._json(HTTPStatus.OK, {"object": "list", "data": [
                        {"id": name, "object": "model", "owned_by": "deepevol-gateway"} for name in models
                    ]})
                    return
                if method == "POST" and path in {"/v1/chat/completions", "/chat/completions"}:
                    self._json(HTTPStatus.OK, shell._op_chat(self._json_body()))
                    return
                if method == "POST" and path == "/placement/ready":
                    self._json(HTTPStatus.OK, shell._op_ready(self._json_body()))
                    return
                if method == "POST" and path == "/exec":
                    self._json(HTTPStatus.OK, shell._op_exec(self._json_body()))
                    return
                if method == "POST" and path == "/jobs":
                    self._json(HTTPStatus.OK, shell._op_spawn(self._json_body()))
                    return
                if path.startswith("/jobs/"):
                    rest = path[len("/jobs/"):].split("/")
                    job_id = rest[0]
                    action = rest[1] if len(rest) > 1 else ""
                    if method == "GET" and not action:
                        self._json(HTTPStatus.OK, shell._op_job_status(job_id))
                        return
                    if method == "GET" and action == "stream":
                        self._stream(job_id, query)
                        return
                    if method == "POST" and action == "wait":
                        self._json(HTTPStatus.OK, shell._op_job_wait(job_id, self._json_body()))
                        return
                    if method == "POST" and action == "kill":
                        self._json(HTTPStatus.OK, shell._op_job_kill(job_id, self._json_body()))
                        return
                if path.startswith("/fs/"):
                    name = path[len("/fs/"):]
                    if method == "GET" and name == "read":
                        self._bytes(shell._op_read_bytes(query))
                        return
                    if method == "POST" and name == "write":
                        self._json(HTTPStatus.OK, shell._op_write_bytes(query, self._body(MAX_UPLOAD_BYTES)))
                        return
                    if method == "POST":
                        self._json(HTTPStatus.OK, shell._op_fs(name, self._json_body()))
                        return
                raise _RequestError(HTTPStatus.NOT_FOUND, ShellErrorCode.NOT_FOUND, f"no route {method} {path}")

            def _stream(self, job_id: str, query: Mapping[str, list[str]]) -> None:
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def write(chunk: bytes) -> None:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                    self.wfile.flush()

                try:
                    shell._stream_job(job_id, query, write)
                except _RequestError as exc:
                    write(json.dumps({"error": {"code": str(exc.code), "message": str(exc)}}).encode("utf-8") + b"\n")
                finally:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()

            def do_GET(self) -> None:  # noqa: N802
                self._dispatch("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._dispatch("POST")

            def do_PUT(self) -> None:  # noqa: N802
                self._dispatch("PUT")

            def do_DELETE(self) -> None:  # noqa: N802
                self._dispatch("DELETE")

            def do_PATCH(self) -> None:  # noqa: N802
                self._dispatch("PATCH")

        return Handler


__all__ = ["AgentShell"]
