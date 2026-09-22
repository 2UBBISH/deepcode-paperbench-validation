"""Explicitly test-only failure injection; inert under every default setting."""

from __future__ import annotations

import os
import re
import json
import sys
from collections.abc import Callable
from enum import Enum
from typing import Any


TEST_ENVIRONMENT_VARIABLE = "DEEPEVOL_ENVIRONMENT"
FAILPOINT_ENABLE_VARIABLE = "DEEPEVOL_V2_RELIABILITY_FAILPOINTS_ENABLED"
FAILPOINT_LIST_VARIABLE = "DEEPEVOL_V2_RELIABILITY_FAILPOINTS"
FAILPOINT_ACTION_VARIABLE = "DEEPEVOL_V2_RELIABILITY_FAILPOINT_ACTION"
FAILPOINT_COMPONENT_VARIABLE = "DEEPEVOL_V2_RELIABILITY_COMPONENT"
_EXCEPTION_ACTION = "exception"
_PROCESS_EXIT_ACTION = "process_exit"
_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class FailpointConfigurationError(ValueError):
    """The explicitly enabled test failpoint list is malformed."""


class FailpointTriggered(BaseException):
    """A deliberate process-boundary-like failure used only in tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"test failpoint triggered: {name}")


def _require_name(name: str) -> None:
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError("failpoint name must be a lowercase bounded key")


def failpoints_enabled() -> bool:
    """Return true only when both independent, exact test gates are present."""

    return (
        os.environ.get(TEST_ENVIRONMENT_VARIABLE, "").strip().lower() == "test"
        and os.environ.get(FAILPOINT_ENABLE_VARIABLE, "") == "1"
    )


def configured_failpoints() -> frozenset[str]:
    """Read exact failpoint names; wildcard activation is intentionally absent."""

    if not failpoints_enabled():
        return frozenset()
    raw = os.environ.get(FAILPOINT_LIST_VARIABLE, "")
    if not raw:
        return frozenset()
    names = tuple(item.strip() for item in raw.split(","))
    if any(not item or _NAME.fullmatch(item) is None for item in names):
        raise FailpointConfigurationError("test failpoint list contains an invalid name")
    return frozenset(names)


def is_failpoint_enabled(name: str) -> bool:
    _require_name(name)
    return name in configured_failpoints()


def failpoint_action() -> str:
    """Return the explicitly selected action for an enabled test harness.

    Production and development processes always resolve to ``exception`` even
    if an unrelated environment happens to contain the action variable.  A
    malformed action is rejected only after both test-only gates are active.
    """

    if not failpoints_enabled():
        return _EXCEPTION_ACTION
    action = os.environ.get(FAILPOINT_ACTION_VARIABLE, _EXCEPTION_ACTION)
    if action not in {_EXCEPTION_ACTION, _PROCESS_EXIT_ACTION}:
        raise FailpointConfigurationError(
            "test failpoint action must be exception or process_exit"
        )
    return action


def trigger_failpoint(name: str) -> None:
    """Trigger one exact test-only boundary using the configured action.

    ``process_exit`` deliberately uses :func:`os._exit`: exception handlers,
    ASGI middleware, and transaction cleanup must not turn a crash-boundary
    test into an ordinary error response.  A single JSON line is flushed first
    so the harness can retain the boundary and PID without logging requests,
    credentials, or other sensitive state.
    """

    if is_failpoint_enabled(name):
        if failpoint_action() == _PROCESS_EXIT_ACTION:
            component = os.environ.get(
                FAILPOINT_COMPONENT_VARIABLE,
                "unspecified-v2-process",
            )
            if _NAME.fullmatch(component) is None:
                component = "invalid-component-label"
            sys.stderr.write(
                json.dumps(
                    {
                        "action": _PROCESS_EXIT_ACTION,
                        "boundary": name,
                        "component": component,
                        "event": "reliability.failpoint.triggered",
                        "pid": os.getpid(),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            sys.stderr.flush()
            os._exit(137)
            raise RuntimeError("os._exit unexpectedly returned")
        raise FailpointTriggered(name)


def enum_failpoint_injector(prefix: str) -> Callable[[Any], None]:
    """Bind a domain crash-point enum to an exact harness failpoint name."""

    _require_name(prefix)

    def inject(point: Any) -> None:
        value = point.value if isinstance(point, Enum) else point
        if not isinstance(value, str) or not value:
            raise TypeError("crash point must be a non-empty string or Enum")
        name = f"{prefix}.{value.lower()}"
        _require_name(name)
        trigger_failpoint(name)

    return inject


__all__ = [
    "FAILPOINT_ACTION_VARIABLE",
    "FAILPOINT_COMPONENT_VARIABLE",
    "FAILPOINT_ENABLE_VARIABLE",
    "FAILPOINT_LIST_VARIABLE",
    "TEST_ENVIRONMENT_VARIABLE",
    "FailpointConfigurationError",
    "FailpointTriggered",
    "configured_failpoints",
    "enum_failpoint_injector",
    "failpoint_action",
    "failpoints_enabled",
    "is_failpoint_enabled",
    "trigger_failpoint",
]
