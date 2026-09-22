"""Agent seam: the ``Agent`` / ``AugmentedLLM`` pair every phase talks to.

The business layer was written against ``mcp_agent``'s API and DeepCode
kept that shape in ``core/compat/``. The usage pattern is always:

    agent = Agent(name="...", instruction=SYSTEM_PROMPT, server_names=[...])
    async with agent:                      # or await agent.__aenter__()
        llm = await attach_workflow_llm(agent, phase="planning")
        text = await llm.generate_str(message=prompt, request_params=params)
        raw  = await agent.call_tool("read_file", {"path": "..."})

REAL
    ``RequestParams`` (copied), the ``Agent`` container: ``name``,
    ``instruction``, ``server_names``, ``tool_registry``, ``register_tool``,
    ``list_tools``, ``call_tool``; ``AugmentedLLM.generate_str`` and the
    ``build_run_spec`` helper that maps ``RequestParams`` onto
    ``AgentRunSpec`` exactly as DeepCode did.

CONTRACT
    ``Agent.__aenter__`` / ``__aexit__``  — connect the MCP servers named
        in ``server_names`` and register their tools into
        ``self.tool_registry`` as ``mcp_<server>_<tool>``; tear them down.
        Server launch details come from
        ``get_runtime().config.mcp_servers`` (see ``seams/mcp_servers.json``).
    ``Agent.attach_llm``                   — resolve a provider for the phase
        and return an ``AugmentedLLM`` bound to this agent.
    ``AugmentedLLM.generate``              — run one turn through an
        ``AgentRunner`` (``build_run_spec`` does the parameter mapping).

Usage sites
    workflows/agent_orchestration_engine.py   planner, reference analyzer,
                                              github downloader, chat planner
    workflows/plan_review_runtime.py          plan revision
    workflows/agents/document_segmentation_agent.py
    workflows/agents/requirement_analysis_agent.py
    workflows/code_implementation_workflow.py ``_initialize_mcp_agent``
                                              (uses the registry + call_tool,
                                              runs its own AgentRunner)
"""

from __future__ import annotations

from typing import Any, Iterable, Type

from loguru import logger

from apps.v2.agent_engine.paper2code.seams.agent_runtime import AgentRunResult, AgentRunSpec, ToolRegistry
from apps.v2.agent_engine.paper2code.seams.config import KernelRuntime, get_runtime
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMProvider

# ---------------------------------------------------------------------------
# RequestParams (REAL, verbatim from core/compat/request_params.py)
# ---------------------------------------------------------------------------

_KNOWN_FIELDS = frozenset(
    {
        "max_tokens",
        "maxTokens",
        "temperature",
        "reasoning_effort",
        "model",
        "use_history",
        "max_iterations",
        "parallel_tool_calls",
        "tool_filter",
        "max_tool_result_chars",
        "context_window_tokens",
        "context_block_limit",
        "provider_retry_mode",
        "retry_wait_callback",
        "checkpoint_callback",
        "llm_timeout_s",
        "enforce_default_max_iterations",
        "metadata",
    }
)

_WARNED_UNKNOWN_FIELDS: set[str] = set()


class RequestParams:
    """Per-call LLM parameters passed to ``generate_str``.

    Fields the business layer actually sets: ``maxTokens``, ``temperature``,
    ``max_iterations``, ``llm_timeout_s``, ``enforce_default_max_iterations``,
    ``checkpoint_callback``. Everything else exists for API stability.

    ``tool_filter`` is ``{server_name: {tool_names}}``; ``None`` or ``{}``
    means no filtering. See :func:`apply_tool_filter`.
    """

    __slots__ = (
        "max_tokens",
        "maxTokens",
        "temperature",
        "reasoning_effort",
        "model",
        "use_history",
        "max_iterations",
        "parallel_tool_calls",
        "tool_filter",
        "max_tool_result_chars",
        "context_window_tokens",
        "context_block_limit",
        "provider_retry_mode",
        "retry_wait_callback",
        "checkpoint_callback",
        "llm_timeout_s",
        "enforce_default_max_iterations",
        "metadata",
    )

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        maxTokens: int | None = None,  # noqa: N803 - legacy spelling
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        model: str | None = None,
        use_history: bool = True,
        max_iterations: int = 1,
        parallel_tool_calls: bool = False,
        tool_filter: dict[str, set[str]] | None = None,
        max_tool_result_chars: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        provider_retry_mode: str = "standard",
        retry_wait_callback: Any | None = None,
        checkpoint_callback: Any | None = None,
        llm_timeout_s: float | None = None,
        enforce_default_max_iterations: bool = True,
        metadata: dict[str, Any] | None = None,
        **unknown: Any,
    ) -> None:
        self.max_tokens = max_tokens
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.model = model
        self.use_history = use_history
        self.max_iterations = max_iterations
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_filter = tool_filter
        self.max_tool_result_chars = max_tool_result_chars
        self.context_window_tokens = context_window_tokens
        self.context_block_limit = context_block_limit
        self.provider_retry_mode = provider_retry_mode
        self.retry_wait_callback = retry_wait_callback
        self.checkpoint_callback = checkpoint_callback
        self.llm_timeout_s = llm_timeout_s
        self.enforce_default_max_iterations = enforce_default_max_iterations
        self.metadata = dict(metadata) if metadata else None

        if unknown:
            new_names = [name for name in unknown if name not in _WARNED_UNKNOWN_FIELDS]
            if new_names:
                _WARNED_UNKNOWN_FIELDS.update(new_names)
                logger.warning(
                    "RequestParams: ignoring unknown kwargs {}. Add them to "
                    "seams.compat._KNOWN_FIELDS if you need their behaviour.",
                    new_names,
                )
            self.metadata = {**(self.metadata or {}), **unknown}

    def resolved_max_tokens(self) -> int | None:
        """First non-``None`` of ``max_tokens`` / ``maxTokens``."""
        if self.max_tokens is not None:
            return self.max_tokens
        return self.maxTokens

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        body = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__slots__)
        return f"RequestParams({body})"


# ---------------------------------------------------------------------------
# Tool filter (REAL, verbatim semantics from core/compat/agent.py)
# ---------------------------------------------------------------------------

_WARNED_MISSING_SERVERS: set[tuple[str, str]] = set()


def apply_tool_filter(
    registry: ToolRegistry,
    tool_filter: dict[str, set[str]] | None,
    *,
    agent_name: str = "<unknown>",
    agent_server_names: Iterable[str] | None = None,
) -> ToolRegistry:
    """Return ``registry`` or a filtered view of it honouring ``tool_filter``.

    * ``None`` / ``{}`` → unchanged.
    * ``{server: {names}}`` → keep only ``mcp_<server>_*`` tools for listed
      servers; within a server keep only the listed bare names, or all when
      the set is empty. Non-``mcp_`` tools are always kept.
    """
    if not tool_filter:
        return registry

    requested = set(agent_server_names or ())
    filtered = ToolRegistry()
    present_servers: set[str] = set()
    for name in registry.tool_names:
        if name.startswith("mcp_"):
            rest = name[len("mcp_") :]
            for server in tool_filter:
                # [paper2code C2] exposed tool names are sanitized ("-" -> "_")
                # by the line's registry; match filters against that form.
                prefix = f"{server}_".replace("-", "_")
                if rest.startswith(prefix):
                    present_servers.add(server)
                    bare = rest[len(prefix) :]
                    allowed = tool_filter[server]
                    if not allowed or bare in allowed:
                        filtered.register(registry.get(name))  # type: ignore[arg-type]
                    break
        else:
            filtered.register(registry.get(name))  # type: ignore[arg-type]

    for server in tool_filter:
        if server in present_servers:
            continue
        key = (agent_name, server)
        if server in requested:
            if key not in _WARNED_MISSING_SERVERS:
                _WARNED_MISSING_SERVERS.add(key)
                logger.warning(
                    "Agent '{}': tool_filter names server '{}' which the agent "
                    "requested but which is not connected; its tools are absent.",
                    agent_name,
                    server,
                )
        else:
            logger.debug(
                "Agent '{}': tool_filter names server '{}' which is not in "
                "the agent's server_names; ignoring (expected under ParallelLLM).",
                agent_name,
                server,
            )
    return filtered


# ---------------------------------------------------------------------------
# AugmentedLLM (generate_str REAL; generate CONTRACT)
# ---------------------------------------------------------------------------


class AugmentedLLM:
    """One agent + one provider, exposing ``generate_str``.

    ``AnthropicAugmentedLLM`` / ``OpenAIAugmentedLLM`` / ``GoogleAugmentedLLM``
    are marker subclasses legacy call sites may pass to ``attach_llm``.
    """

    PROVIDER_NAME: str | None = None
    DEFAULT_MAX_ITERATIONS: int = 8
    DEFAULT_MAX_TOOL_RESULT_CHARS: int = 60_000

    def __init__(
        self,
        agent: "Agent",
        provider: LLMProvider,
        provider_name: str,
        phase: str = "default",
    ) -> None:
        self.agent = agent
        self.provider = provider
        self.provider_name = provider_name
        self.phase = phase

    async def generate_str(
        self,
        message: str,
        request_params: RequestParams | None = None,
    ) -> str:
        """Run one ``user`` turn and return the final text.

        Raises ``RuntimeError`` when the run errored without producing text.
        """
        result = await self.generate(message=message, request_params=request_params)
        if result.error and not result.final_content:
            raise RuntimeError(f"AugmentedLLM error: {result.error}")
        return result.final_content or ""

    def build_run_spec(
        self,
        message: str,
        request_params: RequestParams | None = None,
    ) -> AgentRunSpec:
        """Map ``RequestParams`` onto an ``AgentRunSpec`` exactly as DeepCode did.

        System prompt = ``agent.instruction``; tools = ``agent.tool_registry``
        after ``apply_tool_filter``. ``max_iterations`` is raised to
        ``DEFAULT_MAX_ITERATIONS`` when tools are present unless the caller
        set ``enforce_default_max_iterations=False``.
        """
        params = request_params or RequestParams()
        tools = apply_tool_filter(
            self.agent.tool_registry,
            params.tool_filter,
            agent_name=self.agent.name,
            agent_server_names=self.agent.server_names,
        )

        messages: list[dict[str, Any]] = []
        if self.agent.instruction:
            messages.append({"role": "system", "content": self.agent.instruction})
        messages.append({"role": "user", "content": message})

        requested_iterations = max(int(params.max_iterations or 1), 1)
        if params.enforce_default_max_iterations:
            max_iterations = max(
                requested_iterations,
                self.DEFAULT_MAX_ITERATIONS if tools.tool_names else 1,
            )
        else:
            max_iterations = requested_iterations

        return AgentRunSpec(
            initial_messages=messages,
            tools=tools,
            model=params.model or self.provider.get_default_model(),
            max_iterations=max_iterations,
            max_tool_result_chars=(
                params.max_tool_result_chars or self.DEFAULT_MAX_TOOL_RESULT_CHARS
            ),
            temperature=params.temperature,
            max_tokens=params.resolved_max_tokens(),
            reasoning_effort=params.reasoning_effort,
            session_key=self.agent.name,
            context_window_tokens=params.context_window_tokens,
            context_block_limit=params.context_block_limit,
            provider_retry_mode=params.provider_retry_mode,
            retry_wait_callback=params.retry_wait_callback,
            checkpoint_callback=params.checkpoint_callback,
            llm_timeout_s=params.llm_timeout_s,
            concurrent_tools=bool(params.parallel_tool_calls),
        )

    async def generate(
        self,
        message: str,
        request_params: RequestParams | None = None,
    ) -> AgentRunResult:
        """CONTRACT. Run one turn and return the full result.

        Reference behaviour: ``spec = self.build_run_spec(message,
        request_params)`` then ``return await AgentRunner(self.provider).run(spec)``
        with the integrator's ``AgentRunner``.
        """
        raise NotImplementedError(
            "AugmentedLLM.generate: build a spec with build_run_spec() and run "
            "it through your AgentRunner"
        )


class AnthropicAugmentedLLM(AugmentedLLM):
    PROVIDER_NAME = "anthropic"


class OpenAIAugmentedLLM(AugmentedLLM):
    PROVIDER_NAME = "openai"


class GoogleAugmentedLLM(AugmentedLLM):
    PROVIDER_NAME = "google"


# ---------------------------------------------------------------------------
# Agent (container REAL; lifecycle + attach CONTRACT)
# ---------------------------------------------------------------------------


class Agent:
    """A named system prompt plus the MCP servers whose tools it may call.

    The container behaviour (registry, ``call_tool`` name resolution) is
    real. Connecting servers and resolving a provider are the integrator's.
    """

    def __init__(
        self,
        name: str,
        instruction: str = "",
        server_names: Iterable[str] | None = None,
        functions: Iterable[Any] | None = None,  # accepted for compat, unused
        connection_persistence: bool = True,
        human_input_callback: Any | None = None,
        request_params: RequestParams | None = None,
    ) -> None:
        self.name = name
        self.instruction = instruction
        self.server_names: list[str] = list(server_names or [])
        self.functions = list(functions or [])
        self.connection_persistence = connection_persistence
        self.human_input_callback = human_input_callback
        self.request_params = request_params

        self._runtime: KernelRuntime | None = None
        self._tool_registry: ToolRegistry = ToolRegistry()
        self._connected: bool = False

    # -- lifecycle (CONTRACT) ---------------------------------------------

    async def __aenter__(self) -> "Agent":
        """CONTRACT. Connect every server in ``server_names``.

        For each name look up ``get_runtime().config.mcp_servers[name]``,
        start or connect to it, and register each of its tools into
        ``self.tool_registry`` under ``mcp_<name>_<tool>``. DeepCode logged a
        warning and continued when a server was missing or failed; the
        kernel leaves that policy to you but ``docs/INTEGRATION.md``
        recommends failing loudly.
        """
        raise NotImplementedError(
            "Agent.__aenter__: connect the MCP servers in server_names and "
            "register their tools (docs/INTEGRATION.md § Agent)"
        )

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """CONTRACT. Tear down what ``__aenter__`` started.

        Must not raise on teardown noise; the business layer calls this from
        ``finally`` blocks and does not expect exceptions.
        """
        raise NotImplementedError("Agent.__aexit__: close the MCP servers")

    async def attach_llm(
        self,
        llm_class: Type[AugmentedLLM] | None = None,
        *,
        phase: str = "default",
        provider_name: str | None = None,
        connection_id: str | None = None,
        model: str | None = None,
    ) -> AugmentedLLM:
        """CONTRACT. Return an ``AugmentedLLM`` bound to this agent.

        Reference behaviour: resolve ``provider = get_runtime().provider_for(
        provider_name=provider_name or llm_class.PROVIDER_NAME, phase=phase,
        model=model)`` and return ``YourAugmentedLLM(agent=self,
        provider=provider, provider_name=<resolved name>, phase=phase)``.
        """
        raise NotImplementedError(
            "Agent.attach_llm: resolve a provider for the phase and return "
            "your AugmentedLLM subclass"
        )

    # -- container (REAL) ---------------------------------------------------

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    def register_tool(self, tool: Any) -> None:
        self._tool_registry.register(tool)

    async def list_tools(self) -> dict[str, Any]:
        return {"tools": list(self._tool_registry.get_definitions())}

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Invoke a registered tool by bare or wrapped name."""
        params = arguments or {}
        if not self._tool_registry.has(name):
            for candidate in self._tool_registry.tool_names:
                if candidate.endswith(f"_{name}") or candidate.endswith(name):
                    name = candidate
                    break
        return await self._tool_registry.execute(name, params)


__all__ = [
    "Agent",
    "AnthropicAugmentedLLM",
    "AugmentedLLM",
    "GoogleAugmentedLLM",
    "OpenAIAugmentedLLM",
    "RequestParams",
    "apply_tool_filter",
    "get_runtime",
]
