"""Model seam: the provider the business layer talks to, and how it finds one.

Two kinds of things live here.

REAL (copied from DeepCode ``core/providers/base.py`` and
``core/llm_runtime.py``, trimmed to what the business layer touches)
    ``ToolCallRequest``, ``LLMResponse``, ``GenerationSettings``,
    ``LLMProfile``, ``get_workflow_provider``, ``attach_workflow_llm``.

CONTRACT (the integrator implements)
    ``LLMProvider.chat_with_retry`` and ``LLMProvider.get_default_model``.

Who calls the provider directly (everything else goes through
``seams.compat.Agent`` → ``AugmentedLLM``):

    tools/code_indexer.py                      ``provider.chat_with_retry(...)``
    workflows/agents/memory_agent_concise.py   ``client.chat_with_retry(...)``
    workflows/code_implementation_workflow.py  ``provider.get_default_model()``
                                               and hands the provider to
                                               ``AgentRunner(provider)``

Both direct callers use only ``messages``, ``model``, ``max_tokens``,
``temperature``, ``retry_mode`` and read back ``response.content`` and
``response.finish_reason`` (``"error"`` means the call failed and
``content`` carries the message).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from apps.v2.agent_engine.paper2code.seams.config import KernelRuntime, get_runtime

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apps.v2.agent_engine.paper2code.seams.compat import Agent, AugmentedLLM


# ---------------------------------------------------------------------------
# Wire types (REAL)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCallRequest:
    """One tool call the model asked for.

    ``id`` is echoed back in the ``tool`` message so the provider can pair
    request and result. ``arguments`` is already-parsed JSON.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None


@dataclass(slots=True)
class LLMResponse:
    """One completed provider response.

    ``finish_reason`` values the business layer distinguishes:

    - ``"stop"``     normal completion (default)
    - ``"tool_calls"`` the model wants tools run (``tool_calls`` non-empty)
    - ``"length"``   output was cut at ``max_tokens``
    - ``"error"``    the call failed; ``content`` holds the error text

    Anything else is passed through untouched.
    """

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    retry_after: float | None = None
    reasoning_content: str | None = None
    reasoning_summary: str | None = None
    provider_state: dict[str, Any] | None = None
    thinking_blocks: list[dict] | None = None
    error_status_code: int | None = None
    error_kind: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    error_retry_after_s: float | None = None
    error_should_retry: bool | None = None
    partial_output: bool = False

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def should_execute_tools(self) -> bool:
        return self.finish_reason != "error" and self.has_tool_calls


@dataclass(slots=True)
class GenerationSettings:
    """Defaults a provider applies when a call omits a parameter."""

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


# ---------------------------------------------------------------------------
# Provider (CONTRACT)
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """The one object every model call in the business layer goes through.

    Implement the two abstract methods. ``generation`` must be an instance
    of :class:`GenerationSettings`; ``get_workflow_provider`` logs its
    fields and ``chat_with_retry`` is expected to fall back to them.
    """

    generation: GenerationSettings

    def __init__(self, generation: GenerationSettings | None = None) -> None:
        self.generation = generation or GenerationSettings()

    @abstractmethod
    def get_default_model(self) -> str:
        """The model id this provider instance was built for."""

    @property
    def default_model(self) -> str:
        return self.get_default_model()

    @abstractmethod
    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """One chat completion with the provider's own retry policy.

        Contract the business layer relies on:

        - ``messages`` are OpenAI-style dicts: ``{"role", "content"}`` plus
          ``tool_calls`` on assistant messages and ``tool_call_id`` on
          ``role: "tool"`` messages.
        - ``tools`` are OpenAI function-tool schemas as produced by
          :meth:`seams.agent_runtime.Tool.to_schema`; ``None`` means no tools.
        - ``None`` for ``max_tokens`` / ``temperature`` / ``reasoning_effort``
          means "use ``self.generation``".
        - ``retry_mode`` is ``"standard"`` or ``"persistent"``; the latter is
          used by the indexer and should retry longer/harder.
        - Never raise for a model-side failure. Return
          ``LLMResponse(content=<error text>, finish_reason="error")`` so
          callers can decide; raising is reserved for programmer errors.
        - ``on_retry_wait(message)`` if given is awaited before each retry
          sleep so the caller can surface progress.
        """


# ---------------------------------------------------------------------------
# Resolution helpers (REAL, thin)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMProfile:
    """Resolved LLM selection for one workflow call (for logs and reports)."""

    provider_name: str
    phase: str
    model: str
    reasoning_effort: str | None
    max_tokens: int
    connection_id: str | None = None
    context_window: int | None = None


def get_workflow_provider(
    *,
    phase: str,
    provider_name: str | None = None,
    connection_id: str | None = None,
    model: str | None = None,
    execution_profile: Any | None = None,
    runtime: KernelRuntime | None = None,
) -> tuple[LLMProvider, LLMProfile]:
    """Resolve a provider for code that calls the model without an ``Agent``.

    Used by ``tools/code_indexer.py`` and
    ``workflows/code_implementation_workflow.py``. Delegates to
    :meth:`seams.config.KernelRuntime.provider_for`.
    """
    active_runtime = runtime or get_runtime()
    provider = active_runtime.provider_for(
        provider_name=provider_name,
        connection_id=connection_id,
        phase=phase,
        model=model,
        execution_profile=execution_profile,
    )
    resolved_provider = (
        provider_name
        or active_runtime.config.get_provider_name(model)
        or active_runtime.config.llm_provider
        or "auto"
    ).lower()
    profile = LLMProfile(
        provider_name=resolved_provider,
        phase=phase,
        model=provider.get_default_model(),
        reasoning_effort=provider.generation.reasoning_effort,
        max_tokens=provider.generation.max_tokens,
        connection_id=connection_id,
    )
    logger.info(
        "Resolved workflow LLM: phase={} provider={} model={} reasoning_effort={} max_tokens={}",
        profile.phase,
        profile.provider_name,
        profile.model,
        profile.reasoning_effort,
        profile.max_tokens,
    )
    return provider, profile


async def attach_workflow_llm(
    agent: "Agent",
    *,
    phase: str,
    provider_name: str | None = None,
    connection_id: str | None = None,
    model: str | None = None,
) -> "AugmentedLLM":
    """Attach an LLM to an agent with explicit workflow phase semantics.

    Every ``Agent`` user in the business layer goes through this instead of
    calling ``agent.attach_llm`` directly, so phase selection is logged in
    one place.
    """
    llm = await agent.attach_llm(
        phase=phase,
        provider_name=provider_name,
        connection_id=connection_id,
        model=model,
    )
    logger.info(
        "Attached workflow LLM: agent={} phase={} provider={} model={} reasoning_effort={}",
        agent.name,
        phase,
        llm.provider_name,
        llm.provider.get_default_model(),
        llm.provider.generation.reasoning_effort,
    )
    return llm


__all__ = [
    "GenerationSettings",
    "LLMProfile",
    "LLMProvider",
    "LLMResponse",
    "ToolCallRequest",
    "attach_workflow_llm",
    "get_workflow_provider",
]
