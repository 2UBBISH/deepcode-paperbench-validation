"""Immutable Product-authorized chat fallback contract; no implicit routes."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Mapping
import re


@dataclass(frozen=True, slots=True)
class AuthorizedChatRoute:
    provider: str
    model: str
    provider_account_ref: str
    configuration_sha256: str
    capabilities: tuple[str, ...]
    context_tokens: int
    max_output_tokens: int
    input_microcredits_per_million: int
    output_microcredits_per_million: int
    provider_configuration_sha256: str = ""

    def __post_init__(self):
        for name in ("provider", "model", "provider_account_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value.encode()) > 256:
                raise ValueError(f"invalid route {name}")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.provider):
            raise ValueError("invalid route provider")
        if not isinstance(self.configuration_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.configuration_sha256):
            raise ValueError("invalid route configuration digest")
        if self.provider_configuration_sha256 and (
            not isinstance(self.provider_configuration_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.provider_configuration_sha256)
        ):
            raise ValueError("invalid provider configuration digest")
        if not isinstance(self.provider_configuration_sha256, str):
            raise ValueError("invalid provider configuration digest")
        if not isinstance(self.capabilities, tuple) or not self.capabilities or len(self.capabilities) > 32:
            raise ValueError("route requires bounded capabilities")
        if any(not isinstance(item, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", item) for item in self.capabilities):
            raise ValueError("invalid route capability")
        if tuple(sorted(set(self.capabilities))) != self.capabilities or "chat" not in self.capabilities:
            raise ValueError("route capabilities must be canonical and include chat")
        for name in ("context_tokens", "max_output_tokens", "input_microcredits_per_million", "output_microcredits_per_million"):
            value = getattr(self, name)
            minimum = 1 if name.endswith("tokens") else 0
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 10**12:
                raise ValueError(f"invalid route {name}")
        if self.max_output_tokens > self.context_tokens:
            raise ValueError("output limit exceeds context")

    @property
    def resource_key(self) -> str:
        return f"{self.provider}:{self.model}"

    def as_wire(self) -> dict:
        value = asdict(self)
        value["capabilities"] = list(self.capabilities)
        return value

    @classmethod
    def from_wire(cls, value):
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("route has missing or unknown fields")
        if not isinstance(value["capabilities"], list):
            raise ValueError("route capabilities must be an array")
        return cls(**{**value, "capabilities": tuple(value["capabilities"])})


@dataclass(frozen=True, slots=True)
class ChatFallbackPolicy:
    routes: tuple[AuthorizedChatRoute, ...]
    deadline_seconds: int = 60

    def __post_init__(self):
        if not isinstance(self.routes, tuple) or not 2 <= len(self.routes) <= 4:
            raise ValueError("fallback policy requires primary and 1 to 3 backups")
        if any(not isinstance(route, AuthorizedChatRoute) for route in self.routes):
            raise ValueError("invalid fallback route")
        if isinstance(self.deadline_seconds, bool) or not isinstance(self.deadline_seconds, int) or not 1 <= self.deadline_seconds <= 120:
            raise ValueError("fallback deadline must be 1 to 120 seconds")
        primary = self.routes[0]
        if len({route.resource_key for route in self.routes}) != len(self.routes):
            raise ValueError("duplicate fallback resource")
        for backup in self.routes[1:]:
            if not set(primary.capabilities) <= set(backup.capabilities):
                raise ValueError("backup lacks primary capabilities")
            if backup.context_tokens < primary.context_tokens or backup.max_output_tokens < primary.max_output_tokens:
                raise ValueError("backup has smaller token limits")
            if (backup.input_microcredits_per_million > primary.input_microcredits_per_million or
                    backup.output_microcredits_per_million > primary.output_microcredits_per_million):
                raise ValueError("backup is more expensive")

    def as_wire(self) -> dict:
        return {"routes": [route.as_wire() for route in self.routes], "deadline_seconds": self.deadline_seconds}

    @classmethod
    def from_wire(cls, value):
        if not isinstance(value, Mapping) or set(value) != {"routes", "deadline_seconds"} or not isinstance(value["routes"], list):
            raise ValueError("invalid fallback policy")
        return cls(tuple(AuthorizedChatRoute.from_wire(route) for route in value["routes"]), value["deadline_seconds"])


def load_chat_fallback_policy(path) -> ChatFallbackPolicy | None:
    """Load explicit startup configuration; absent configuration disables fallback."""
    if path is None:
        return None
    import json
    import os
    import stat

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate fallback configuration field")
            result[key] = value
        return result

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 65536:
            raise ValueError("fallback configuration must be a regular file of at most 64 KiB")
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError("fallback configuration is oversized")
    policy = ChatFallbackPolicy.from_wire(json.loads(raw, object_pairs_hook=unique))
    if any(not route.provider_configuration_sha256 for route in policy.routes):
        raise ValueError("fallback configuration requires Gateway configuration fingerprints")
    if len({route.model for route in policy.routes}) != len(policy.routes):
        raise ValueError("fallback model names must uniquely identify routes")
    return policy
