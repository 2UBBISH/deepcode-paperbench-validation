"""Agent seam: ``Agent.__aenter__`` / ``__aexit__`` / ``attach_llm`` and ``AugmentedLLM.generate``.

The engine instantiates ``seams.compat.Agent`` directly, so the line cannot
subclass its way in: :func:`bind` assigns the implementations below onto
the seam classes once, at install time (``config.install`` does it; tests
call it directly). Binding is idempotent.

``__aenter__`` builds the in-process registry for ``server_names`` from the
run's :class:`ToolContext` (carried by the installed runtime) and registers
every tool; a server name the line does not provide raises. ``generate``
maps ``RequestParams`` through the seam's own ``build_run_spec`` and runs
the spec on ``AgentRunner``, whose ``run`` the line binds in ``runner.py``.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from apps.v2.agent.paper2code.tools.registry import build_registry, current_tool_context
from apps.v2.agent_engine.paper2code.seams import compat
from apps.v2.agent_engine.paper2code.seams.agent_runtime import AgentRunner, AgentRunResult
from apps.v2.agent_engine.paper2code.seams.config import get_runtime


class PaperAugmentedLLM(compat.AugmentedLLM):
    async def generate(
        self,
        message: str,
        request_params: compat.RequestParams | None = None,
    ) -> AgentRunResult:
        spec = self.build_run_spec(message, request_params)
        return await AgentRunner(self.provider).run(spec)


async def _aenter(self: compat.Agent) -> compat.Agent:
    ctx = current_tool_context()
    registry = build_registry(self.server_names, ctx)
    for name in registry.tool_names:
        self.register_tool(registry.get(name))
    self._connected = True
    logger.debug(
        "Agent '{}': {} in-process tools from servers {}",
        self.name,
        len(registry),
        self.server_names,
    )
    return self


async def _aexit(self: compat.Agent, exc_type: Any, exc: Any, tb: Any) -> None:
    try:
        await self.tool_registry.aclose()
    except Exception as err:
        logger.warning("Agent '{}': error while clearing tools: {}", self.name, err)
    self._connected = False


async def _attach_llm(
    self: compat.Agent,
    llm_class: type[compat.AugmentedLLM] | None = None,
    *,
    phase: str = "default",
    provider_name: str | None = None,
    connection_id: str | None = None,
    model: str | None = None,
) -> compat.AugmentedLLM:
    runtime = get_runtime()
    requested = provider_name or getattr(llm_class, "PROVIDER_NAME", None)
    provider = runtime.provider_for(
        provider_name=requested,
        connection_id=connection_id,
        phase=phase,
        model=model,
    )
    resolved_name = requested or getattr(provider, "provider_name", None) or "paper2code"
    return PaperAugmentedLLM(agent=self, provider=provider, provider_name=resolved_name, phase=phase)


def bind() -> None:
    """Install the implementations onto the seam classes (idempotent)."""
    compat.Agent.__aenter__ = _aenter  # type: ignore[method-assign]
    compat.Agent.__aexit__ = _aexit  # type: ignore[method-assign]
    compat.Agent.attach_llm = _attach_llm  # type: ignore[method-assign]
    compat.AugmentedLLM.generate = PaperAugmentedLLM.generate  # type: ignore[method-assign]


__all__ = ["PaperAugmentedLLM", "bind"]
