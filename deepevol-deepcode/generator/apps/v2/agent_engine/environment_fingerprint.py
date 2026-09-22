"""Canonical, secret-free execution environment fingerprints.

The fingerprint is an execution compatibility boundary, not a deployment
inventory.  Callers must provide immutable component digests explicitly; the
module deliberately never reads the hostname, process environment, credentials,
or service endpoints.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from apps.v2.reliability import canonical_json_sha256


ENVIRONMENT_FINGERPRINT_SCHEMA = "agent-environment-fingerprint.v1"
_FLAG_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_FORBIDDEN_FLAG_FRAGMENTS = frozenset(
    {
        "credential",
        "endpoint",
        "hostname",
        "password",
        "secret",
        "token",
        "url",
    }
)


class EnvironmentFingerprintMismatch(RuntimeError):
    """A checkpoint/run belongs to a different execution environment."""


def _digest(value: str, name: str) -> str:
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _bounded_identifier(value: str, name: str, *, maximum: int) -> str:
    if not value or value != value.strip() or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must contain 1 to {maximum} bounded UTF-8 bytes")
    return value


def _semantic_flags(values: Mapping[str, bool | int]) -> tuple[tuple[str, bool | int], ...]:
    if not isinstance(values, Mapping) or len(values) > 128:
        raise ValueError("execution semantic flags must be a bounded mapping")
    normalized: list[tuple[str, bool | int]] = []
    for name, value in values.items():
        if not isinstance(name, str) or _FLAG_NAME.fullmatch(name) is None:
            raise ValueError("execution semantic flag names must be canonical")
        lowered = name.lower()
        if any(fragment in lowered for fragment in _FORBIDDEN_FLAG_FRAGMENTS):
            raise ValueError(f"execution semantic flag {name!r} may contain deployment or secret data")
        if isinstance(value, bool):
            normalized_value: bool | int = value
        elif isinstance(value, int) and -(2**63) <= value < 2**63:
            normalized_value = value
        else:
            raise ValueError("execution semantic flag values must be booleans or signed 64-bit integers")
        normalized.append((name, normalized_value))
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ExecutionEnvironmentSnapshot:
    """Immutable inputs that decide whether checkpoint recovery is safe."""

    worker_image_sha256: str
    dependency_lock_sha256: str
    workflow_build_id: str
    graph_schema_sha256: str
    checkpoint_schema_version: str
    tool_registry_sha256: str
    sandbox_image_sha256: str
    sandbox_policy_sha256: str
    execution_semantics: tuple[tuple[str, bool | int], ...] = ()
    schema_version: str = ENVIRONMENT_FINGERPRINT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "worker_image_sha256",
            "dependency_lock_sha256",
            "graph_schema_sha256",
            "tool_registry_sha256",
            "sandbox_image_sha256",
            "sandbox_policy_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _bounded_identifier(self.workflow_build_id, "workflow_build_id", maximum=512)
        _bounded_identifier(
            self.checkpoint_schema_version,
            "checkpoint_schema_version",
            maximum=160,
        )
        if self.schema_version != ENVIRONMENT_FINGERPRINT_SCHEMA:
            raise ValueError("environment fingerprint schema version is unsupported")
        if not isinstance(self.execution_semantics, tuple):
            raise TypeError("execution_semantics must use the canonical tuple representation")
        canonical_flags = _semantic_flags(dict(self.execution_semantics))
        if canonical_flags != self.execution_semantics or len(dict(self.execution_semantics)) != len(
            self.execution_semantics
        ):
            raise ValueError("execution_semantics must be unique and canonically ordered")

    @classmethod
    def build(
        cls,
        *,
        worker_image_sha256: str,
        dependency_lock_sha256: str,
        workflow_build_id: str,
        graph_schema_sha256: str,
        checkpoint_schema_version: str,
        tool_registry_sha256: str,
        sandbox_image_sha256: str,
        sandbox_policy_sha256: str,
        execution_semantics: Mapping[str, bool | int] | None = None,
    ) -> ExecutionEnvironmentSnapshot:
        return cls(
            worker_image_sha256=worker_image_sha256,
            dependency_lock_sha256=dependency_lock_sha256,
            workflow_build_id=workflow_build_id,
            graph_schema_sha256=graph_schema_sha256,
            checkpoint_schema_version=checkpoint_schema_version,
            tool_registry_sha256=tool_registry_sha256,
            sandbox_image_sha256=sandbox_image_sha256,
            sandbox_policy_sha256=sandbox_policy_sha256,
            execution_semantics=_semantic_flags(execution_semantics or {}),
        )

    @classmethod
    def from_materials(
        cls,
        *,
        worker_image_sha256: str,
        dependency_lock: bytes,
        workflow_build_id: str,
        graph_schema: Any,
        checkpoint_schema_version: str,
        tool_registry: Any,
        sandbox_image_sha256: str,
        sandbox_policy: Any,
        execution_semantics: Mapping[str, bool | int] | None = None,
    ) -> ExecutionEnvironmentSnapshot:
        if not isinstance(dependency_lock, bytes) or not dependency_lock:
            raise ValueError("dependency_lock must contain the immutable lockfile bytes")
        return cls.build(
            worker_image_sha256=worker_image_sha256,
            dependency_lock_sha256=sha256(dependency_lock).hexdigest(),
            workflow_build_id=workflow_build_id,
            graph_schema_sha256=canonical_json_sha256(graph_schema),
            checkpoint_schema_version=checkpoint_schema_version,
            tool_registry_sha256=canonical_json_sha256(tool_registry),
            sandbox_image_sha256=sandbox_image_sha256,
            sandbox_policy_sha256=canonical_json_sha256(sandbox_policy),
            execution_semantics=execution_semantics,
        )

    def as_document(self) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "execution_semantics": dict(self.execution_semantics),
            "graph_schema_sha256": self.graph_schema_sha256,
            "sandbox_image_sha256": self.sandbox_image_sha256,
            "sandbox_policy_sha256": self.sandbox_policy_sha256,
            "schema_version": self.schema_version,
            "tool_registry_sha256": self.tool_registry_sha256,
            "worker_image_sha256": self.worker_image_sha256,
            "workflow_build_id": self.workflow_build_id,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_json_sha256(self.as_document())

    def require_fingerprint(self, expected: str) -> None:
        expected_digest = _digest(expected, "expected environment fingerprint")
        if self.fingerprint != expected_digest:
            raise EnvironmentFingerprintMismatch(
                "execution environment is incompatible with the durable run/checkpoint"
            )


__all__ = [
    "ENVIRONMENT_FINGERPRINT_SCHEMA",
    "EnvironmentFingerprintMismatch",
    "ExecutionEnvironmentSnapshot",
]
