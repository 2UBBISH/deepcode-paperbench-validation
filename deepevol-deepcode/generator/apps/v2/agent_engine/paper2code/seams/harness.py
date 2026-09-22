"""Permission seam: who decides whether a tool call may run.

Only ``workflows/code_implementation_workflow.py`` uses this, in one place:

    security_cfg = getattr(get_runtime().config, "security", None)
    permission_engine = build_permission_engine(
        security_cfg, cwd=code_directory, default_mode=PermissionMode.FULL_AUTO
    )
    mode = permission_engine.mode
    approval_cb = None
    if mode is not PermissionMode.FULL_AUTO:
        approval_cb = TerminalApprover().as_async()
    spec = AgentRunSpec(..., permission_checker=permission_engine.evaluate,
                        approval_callback=approval_cb)

REAL
    ``PermissionMode``, ``PermissionDecision`` (enums, copied).

CONTRACT
    ``build_permission_engine`` → an object with ``.mode`` and
        ``.evaluate(tool_name, arguments) -> (decision, reason)``.
    ``TerminalApprover`` → ``as_async()`` returns
        ``async (tool_name, arguments, reason) -> bool``.

DeepCode's engine is ~800 lines of path-fencing and rule matching; the
kernel does not ship it. The simplest conforming implementation is
``FullAutoEngine`` below, which allows everything and is what a headless
run wants.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Protocol


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"
    FULL_AUTO = "full_auto"


class PermissionEngine(Protocol):
    """What ``build_permission_engine`` must return."""

    @property
    def mode(self) -> PermissionMode: ...

    def evaluate(
        self, tool_name: str, arguments: Mapping[str, object] | None = None
    ) -> tuple[PermissionDecision, str]: ...


class FullAutoEngine:
    """Conforming engine that allows every call. Fine for unattended runs."""

    mode = PermissionMode.FULL_AUTO

    def evaluate(
        self, tool_name: str, arguments: Mapping[str, object] | None = None
    ) -> tuple[PermissionDecision, str]:
        return PermissionDecision.ALLOW, ""


def build_permission_engine(
    security: Any,
    *,
    cwd: str | None = None,
    default_mode: PermissionMode = PermissionMode.FULL_AUTO,
) -> PermissionEngine:
    """CONTRACT. Build the engine from ``KernelConfig.security``.

    ``security`` is :class:`seams.config.SecurityConfig` or ``None``.
    ``cwd`` is the generated-code directory the engine may fence writes to.
    Return :class:`FullAutoEngine` if you do not need policy.
    """
    raise NotImplementedError(
        "build_permission_engine: return an object with .mode and "
        ".evaluate(name, args) -> (decision, reason); FullAutoEngine() is a "
        "valid answer"
    )


class TerminalApprover:
    """CONTRACT. Ask a human whether a gated tool call may proceed.

    Only constructed when ``engine.mode`` is not ``FULL_AUTO``. The runner
    calls the async wrapper with ``(tool_name, arguments, reason)`` and
    expects a ``bool``. DeepCode's version prompts on stdin in a worker
    thread and remembers per-tool "always" answers.
    """

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
        is_interactive: Callable[[], bool] | None = None,
    ) -> None:
        self._input = input_fn
        self._output = output_fn
        self._is_interactive = is_interactive
        self._always_allow: set[str] = set()

    def __call__(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        reason: str | None = None,
    ) -> bool:
        raise NotImplementedError(
            "TerminalApprover.__call__: prompt the user and return True to allow"
        )

    def as_async(self) -> Callable[[str, Mapping[str, Any] | None, str | None], Any]:
        async def _acall(
            tool_name: str,
            arguments: Mapping[str, Any] | None = None,
            reason: str | None = None,
        ) -> bool:
            return await asyncio.to_thread(self.__call__, tool_name, arguments, reason)

        return _acall


__all__ = [
    "FullAutoEngine",
    "PermissionDecision",
    "PermissionEngine",
    "PermissionMode",
    "TerminalApprover",
    "build_permission_engine",
]
