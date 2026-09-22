"""Hard process boundary for the optional synchronous Mem0 OSS SDK."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Event, Lock
from typing import Any

from apps.v2.memory.errors import MemoryErrorCode, MemorySystemError


_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_WORKER_MODULE = "apps.v2.reliability.mem0_process_worker"


class Mem0ProcessClient:
    """Mem0 client facade whose every SDK call can be killed at a deadline."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        timeout_seconds: float,
        command_prefix: Sequence[str] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("mem0 process timeout must be positive")
        self._config = dict(config)
        self._timeout_seconds = timeout_seconds
        self._command_prefix = tuple(command_prefix or (sys.executable, "-m", _WORKER_MODULE))
        self._closed = False
        self._state_lock = Lock()
        self._cancel_requested = Event()
        self._process: subprocess.Popen[Any] | None = None

    def probe(self) -> None:
        self._call("probe", {})

    def add(self, messages: Any, **kwargs: Any) -> Mapping[str, Any] | list[Any]:
        return self._call("add", {"messages": messages, "kwargs": kwargs})

    def get(self, memory_id: str) -> Mapping[str, Any] | None:
        return self._call("get", {"memory_id": memory_id})

    def get_all(self, **kwargs: Any) -> Mapping[str, Any] | list[Any]:
        return self._call("get_all", {"kwargs": kwargs})

    def search(self, query: str, **kwargs: Any) -> Mapping[str, Any] | list[Any]:
        return self._call("search", {"query": query, "kwargs": kwargs})

    def delete(self, memory_id: str) -> Any:
        return self._call("delete", {"memory_id": memory_id})

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
        self.cancel_current()

    def cancel_current(self) -> None:
        self._cancel_requested.set()
        with self._state_lock:
            process = self._process
        if process is not None:
            _stop_process_group(process)

    def _call(self, operation: str, arguments: Mapping[str, Any]) -> Any:
        with self._state_lock:
            if self._closed:
                raise MemorySystemError(
                    MemoryErrorCode.CIRCUIT_OPEN,
                    "mem0 process client is closed",
                    retryable=True,
                )
        self._cancel_requested.clear()
        request = _encode_json(
            {
                "version": 1,
                "operation": operation,
                "config": self._config,
                "arguments": dict(arguments),
            },
            maximum=_MAX_REQUEST_BYTES,
            description="mem0 worker request",
        )
        with tempfile.TemporaryDirectory(prefix="deepevol-mem0-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            response_path = root / "response.json"
            _write_private(request_path, request)
            process = subprocess.Popen(
                (*self._command_prefix, str(request_path), str(response_path)),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
            with self._state_lock:
                self._process = process
            if self._cancel_requested.is_set():
                _stop_process_group(process)
            try:
                process.wait(timeout=self._timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _stop_process_group(process)
                raise TimeoutError("mem0 SDK worker timed out") from exc
            finally:
                with self._state_lock:
                    if self._process is process:
                        self._process = None
            if process.returncode != 0:
                if self._cancel_requested.is_set():
                    raise TimeoutError("mem0 SDK worker was cancelled")
                raise RuntimeError("mem0 SDK worker exited unsuccessfully")
            response = _read_response(response_path)
        if response.get("ok") is True:
            return response.get("result")
        _raise_worker_error(response.get("error"))


def _encode_json(value: Any, *, maximum: int, description: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MemorySystemError(
            MemoryErrorCode.PROTOCOL_ERROR,
            f"{description} is not strict JSON",
            retryable=False,
            cause=exc,
        ) from exc
    if len(encoded) > maximum:
        raise MemorySystemError(
            MemoryErrorCode.PROTOCOL_ERROR,
            f"{description} exceeds its byte limit",
            retryable=False,
        )
    return encoded


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_response(path: Path) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError("mem0 SDK worker produced no response") from exc
    if size <= 0 or size > _MAX_RESPONSE_BYTES:
        raise RuntimeError("mem0 SDK worker response has an invalid size")
    with path.open("rb") as handle:
        payload = handle.read(_MAX_RESPONSE_BYTES + 1)
    if len(payload) != size or len(payload) > _MAX_RESPONSE_BYTES:
        raise RuntimeError("mem0 SDK worker response exceeded its byte limit")
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("mem0 SDK worker returned invalid JSON") from exc
    if not isinstance(decoded, Mapping) or decoded.get("version") != 1:
        raise RuntimeError("mem0 SDK worker returned an invalid envelope")
    if decoded.get("ok") not in {True, False}:
        raise RuntimeError("mem0 SDK worker omitted its outcome")
    return decoded


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> Any:
    raise ValueError(f"non-finite JSON value: {value}")


def _raise_worker_error(value: Any) -> None:
    kind = value if isinstance(value, str) else "internal"
    if kind == "configuration":
        raise MemorySystemError(
            MemoryErrorCode.CONFIGURATION,
            "mem0 SDK is unavailable in the worker environment",
            retryable=False,
        )
    if kind == "authorization":
        raise PermissionError("mem0 SDK rejected authorization")
    if kind == "invalid_argument":
        raise ValueError("mem0 SDK rejected the request")
    if kind == "timeout":
        raise TimeoutError("mem0 SDK operation timed out")
    if kind == "unavailable":
        raise ConnectionError("mem0 SDK dependency is unavailable")
    raise RuntimeError("mem0 SDK worker reported an internal failure")


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


__all__ = ["Mem0ProcessClient"]
