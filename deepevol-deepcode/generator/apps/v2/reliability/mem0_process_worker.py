"""Private reliability subprocess entrypoint for one Mem0 OSS SDK operation."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_OPERATIONS = {"probe", "add", "get", "get_all", "search", "delete"}


def _read_request(path: Path) -> Mapping[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > _MAX_REQUEST_BYTES:
        raise ValueError("invalid request size")
    with path.open("rb") as handle:
        payload = handle.read(_MAX_REQUEST_BYTES + 1)
    if len(payload) != size or len(payload) > _MAX_REQUEST_BYTES:
        raise ValueError("incomplete request")
    decoded = json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    if not isinstance(decoded, Mapping) or decoded.get("version") != 1:
        raise ValueError("invalid request envelope")
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


def _execute(request: Mapping[str, Any]) -> Any:
    operation = request.get("operation")
    config = request.get("config")
    arguments = request.get("arguments")
    if operation not in _OPERATIONS or not isinstance(config, Mapping) or not isinstance(arguments, Mapping):
        raise ValueError("invalid worker operation")

    from mem0 import Memory

    client = Memory.from_config(dict(config))
    try:
        if operation == "probe":
            return None
        if operation == "add":
            kwargs = arguments.get("kwargs")
            if not isinstance(kwargs, Mapping):
                raise ValueError("invalid add arguments")
            return client.add(arguments.get("messages"), **dict(kwargs))
        if operation == "get":
            return client.get(str(arguments.get("memory_id", "")))
        if operation == "get_all":
            kwargs = arguments.get("kwargs")
            if not isinstance(kwargs, Mapping):
                raise ValueError("invalid get_all arguments")
            return client.get_all(**dict(kwargs))
        if operation == "search":
            kwargs = arguments.get("kwargs")
            if not isinstance(kwargs, Mapping):
                raise ValueError("invalid search arguments")
            return client.search(str(arguments.get("query", "")), **dict(kwargs))
        if operation == "delete":
            return client.delete(str(arguments.get("memory_id", "")))
        raise ValueError("unsupported worker operation")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _error_kind(error: BaseException) -> str:
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "configuration"
    if isinstance(error, PermissionError):
        return "authorization"
    if isinstance(error, ValueError):
        return "invalid_argument"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (ConnectionError, OSError)):
        return "unavailable"
    return "internal"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_response(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_RESPONSE_BYTES:
        encoded = b'{"version":1,"ok":false,"error":"internal"}'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        return 2
    response_path = Path(arguments[1])
    try:
        request = _read_request(Path(arguments[0]))
        result = _execute(request)
        response = {"version": 1, "ok": True, "result": _json_safe(result)}
    except BaseException as error:
        response = {"version": 1, "ok": False, "error": _error_kind(error)}
    try:
        _write_response(response_path, response)
    except BaseException:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
