"""Shared, persistence-free event replay window decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EventReplayPage:
    """One authoritative page from a retained transactional outbox stream."""

    events: tuple[Mapping[str, Any], ...]
    first_available_seq: int | None
    last_available_seq: int | None
    resync_required: bool

    def __post_init__(self) -> None:
        if (self.first_available_seq is None) != (self.last_available_seq is None):
            raise ValueError("event replay bounds must both be present or absent")
        if self.first_available_seq is not None:
            if self.first_available_seq < 1 or self.last_available_seq is None:
                raise ValueError("event replay bounds must be positive")
            if self.last_available_seq < self.first_available_seq:
                raise ValueError("event replay bounds are reversed")
        if self.resync_required and self.events:
            raise ValueError("a replay page requiring resync must not return partial events")


class EventReplayError(RuntimeError):
    """A replay stream cannot be safely exposed to the authenticated caller."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def replay_window_requires_resync(
    *,
    after_seq: int,
    first_available_seq: int | None,
    historical_last_seq: int | None,
) -> bool:
    """Return whether the requested successor has fallen out of retention."""

    if isinstance(after_seq, bool) or not isinstance(after_seq, int) or after_seq < 0:
        raise ValueError("after_seq must be a non-negative integer")
    if first_available_seq is not None:
        if first_available_seq < 1:
            raise ValueError("first_available_seq must be positive")
        return after_seq + 1 < first_available_seq
    if historical_last_seq is not None:
        if historical_last_seq < 1:
            raise ValueError("historical_last_seq must be positive")
        return after_seq < historical_last_seq
    return False


__all__ = [
    "EventReplayError",
    "EventReplayPage",
    "replay_window_requires_resync",
]
