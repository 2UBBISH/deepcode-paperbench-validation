"""Small, strict canonical-JSON helpers for durable reliability identities.

The representation is intentionally narrower than ``json.dumps(default=...)``:
callers must make UUIDs, datetimes, enums, and bytes explicit before hashing.
That keeps a durable identity from changing when an object's incidental Python
representation changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


def _normalize_json(value: Any, *, path: str, active: set[int]) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        # JSON has one numeric zero. Normalizing negative zero avoids two hashes
        # for values that every downstream contract treats as equal.
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{path} contains a cycle")
        active.add(marker)
        try:
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} contains a non-string object key")
                normalized[key] = _normalize_json(
                    item,
                    path=f"{path}.{key}",
                    active=active,
                )
            return normalized
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{path} contains a cycle")
        active.add(marker)
        try:
            return [_normalize_json(item, path=f"{path}[{index}]", active=active) for index, item in enumerate(value)]
        finally:
            active.remove(marker)
    raise TypeError(f"{path} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the project's deterministic UTF-8 JSON representation.

    Object keys are sorted, insignificant whitespace is removed, non-finite
    numbers and implicit object encodings are rejected, and tuples normalize to
    JSON arrays. ``ensure_ascii`` makes the byte representation independent of
    an output stream's Unicode settings.
    """

    normalized = _normalize_json(value, path="$", active=set())
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha256_hex(payload: bytes | bytearray | memoryview) -> str:
    """Hash an explicit byte payload as lowercase SHA-256 hex."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("sha256_hex requires bytes-like input")
    return hashlib.sha256(bytes(payload)).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash the strict canonical JSON representation of ``value``."""

    return sha256_hex(canonical_json_bytes(value))


__all__ = ["canonical_json_bytes", "canonical_json_sha256", "sha256_hex"]
