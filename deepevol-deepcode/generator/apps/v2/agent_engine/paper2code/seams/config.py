"""Configuration seam: what the business layer reads, and where it reads it from.

DeepCode loads a large layered ``deepcode_config.json`` into a Pydantic
model and exposes it through a process-wide runtime singleton. The business
layer only ever touches six groups of that model. :class:`KernelConfig`
mirrors exactly those six groups as plain dataclasses and nothing else.

The integrator builds one :class:`KernelConfig`, wraps it in a
:class:`KernelRuntime` subclass that knows how to construct model providers,
and installs it once with :func:`set_runtime`. Every ``get_runtime()`` call
inside the business layer then resolves to that object.

What is REAL here
    ``KernelConfig`` and its nested dataclasses, ``MCPServerConfig``,
    ``ResolvedAgentSettings``, ``get_runtime`` / ``set_runtime`` /
    ``use_runtime``.

What is a CONTRACT (raises ``NotImplementedError``)
    ``KernelRuntime.provider_for`` — turn a (provider_name, phase, model)
    request into an :class:`seams.llm_runtime.LLMProvider`.

Field-by-field origin (DeepCode ``core/config.py``):
    workspace.root / workspace.max_input_mb         → ``WorkspaceConfig``
    tools.default_search_server / tools.mcp_servers → ``ToolsConfig``
    security.*                                      → ``SecurityConfig`` (opaque
                                                       to us; handed to
                                                       ``build_permission_engine``)
    agents.defaults.{max_tokens, base_max_tokens,
                     retry_max_tokens, model, ...}  → ``AgentDefaults``
    agents.{planning, implementation}               → ``AgentPhase``
    document_segmentation.{enabled,
                           size_threshold_chars}    → ``DocumentSegmentationConfig``
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMProvider


# ---------------------------------------------------------------------------
# MCP server description (mutable: workflows.environment appends to ``args``)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MCPServerConfig:
    """One MCP server the business layer may ask an ``Agent`` to connect to.

    ``name`` is the key the business layer uses in ``server_names``
    (``"filesystem"``, ``"document-segmentation"``, ...). See
    ``seams/mcp_servers.json`` for the seven servers the Paper2Code path
    declares and their launch commands.
    """

    name: str
    type: str | None = None  # "stdio" | "sse" | "streamableHttp"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    enabled_tools: list[str] = field(default_factory=lambda: ["*"])
    tool_timeout: int = 300
    description: str | None = None


# ---------------------------------------------------------------------------
# The six config groups the business layer reads
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkspaceConfig:
    """``workflows.environment`` reads both fields."""

    root: str = "./deepcode_lab"
    max_input_mb: int = 100


@dataclass(slots=True)
class ToolsConfig:
    """``default_search_server`` names the MCP server used for reference
    lookups when no better one is configured (``agent_orchestration_engine``).
    ``mcp_servers`` is the table ``Agent`` implementations connect from."""

    default_search_server: str = "filesystem"
    mcp_servers: dict[str, MCPServerConfig] = field(default_factory=dict)


@dataclass(slots=True)
class SecurityConfig:
    """Passed opaquely to :func:`seams.harness.build_permission_engine`.

    The fields mirror DeepCode's ``SecurityConfig`` so an integrator porting
    DeepCode's permission engine has the same shape to read; an integrator
    with their own policy can ignore everything but ``permission_mode``.
    """

    access_preset: str | None = None  # "ask" | "read_only" | "full_access"
    permission_mode: str = "full_auto"  # see seams.harness.PermissionMode
    permissions: dict[str, Any] = field(default_factory=dict)
    sandbox: bool = True


@dataclass(slots=True)
class AgentDefaults:
    """Generation defaults shared by every phase.

    ``get_token_limits`` (utils/llm_utils.py) reads ``max_tokens``,
    ``base_max_tokens`` and ``retry_max_tokens``; ``get_default_models``
    reads ``model`` through :meth:`KernelConfig.resolve_phase`.
    """

    connection: str | None = None
    provider: str = "auto"
    model: str = ""
    max_tokens: int = 8192
    temperature: float = 0.1
    reasoning_effort: str | None = None
    base_max_tokens: int | None = None
    retry_max_tokens: int | None = None
    max_tokens_policy: str | None = None


@dataclass(slots=True)
class AgentPhase:
    """Per-phase overrides; ``None`` falls back to :class:`AgentDefaults`."""

    connection: str | None = None
    provider: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None


@dataclass(slots=True)
class AgentsConfig:
    defaults: AgentDefaults = field(default_factory=AgentDefaults)
    planning: AgentPhase = field(default_factory=AgentPhase)
    implementation: AgentPhase = field(default_factory=AgentPhase)


@dataclass(slots=True)
class DocumentSegmentationConfig:
    """``should_use_document_segmentation`` (utils/llm_utils.py)."""

    enabled: bool = True
    size_threshold_chars: int = 50_000


@dataclass(frozen=True, slots=True)
class ResolvedAgentSettings:
    """Phase + defaults merged into one immutable view."""

    connection: str | None
    provider: str
    model: str
    max_tokens: int
    temperature: float
    reasoning_effort: str | None
    base_max_tokens: int | None
    retry_max_tokens: int | None
    max_tokens_policy: str | None


@dataclass(slots=True)
class KernelConfig:
    """Everything the business layer reads from configuration. Nothing more."""

    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    document_segmentation: DocumentSegmentationConfig = field(
        default_factory=DocumentSegmentationConfig
    )

    # -- accessors the business layer calls -------------------------------

    @property
    def mcp_servers(self) -> dict[str, MCPServerConfig]:
        """Alias of ``tools.mcp_servers`` (DeepCode exposed both spellings)."""
        return self.tools.mcp_servers

    @property
    def llm_provider(self) -> str:
        """Forced provider name, or ``"auto"``."""
        return self.agents.defaults.provider or "auto"

    def resolve_phase(self, phase: str = "default") -> ResolvedAgentSettings:
        """Merge ``agents.defaults`` with the phase override (if any).

        Verbatim port of ``DeepCodeConfig.resolve_phase``: only
        ``"planning"`` and ``"implementation"`` have overrides; any other
        phase name resolves to the defaults.
        """
        defaults = self.agents.defaults
        override: AgentPhase | None
        if phase == "planning":
            override = self.agents.planning
        elif phase == "implementation":
            override = self.agents.implementation
        else:
            override = None

        def _pick(name: str) -> Any:
            if override is not None:
                value = getattr(override, name)
                if value is not None:
                    return value
            return getattr(defaults, name)

        return ResolvedAgentSettings(
            connection=_pick("connection"),
            provider=_pick("provider"),
            model=_pick("model"),
            max_tokens=_pick("max_tokens"),
            temperature=_pick("temperature"),
            reasoning_effort=_pick("reasoning_effort"),
            base_max_tokens=defaults.base_max_tokens,
            retry_max_tokens=defaults.retry_max_tokens,
            max_tokens_policy=defaults.max_tokens_policy,
        )

    def model_for_phase(self, phase: str = "default") -> str:
        chosen = (self.resolve_phase(phase).model or "").strip()
        if not chosen:
            raise ValueError(f"No model configured for phase '{phase}'")
        return chosen

    def get_provider_name(self, model: str | None = None) -> str | None:
        """DeepCode matched a provider by model-name heuristics here.

        The kernel has no provider registry, so this returns the forced
        provider name when one is set and ``None`` otherwise. An integrator
        that routes by model name should do so inside
        :meth:`KernelRuntime.provider_for` instead.
        """
        forced = self.llm_provider
        return None if forced == "auto" else forced


# Symbol kept for import stability: utils/llm_utils.py type-hints against it.
DeepCodeConfig = KernelConfig


# ---------------------------------------------------------------------------
# Runtime singleton
# ---------------------------------------------------------------------------


class RuntimeNotConfigured(RuntimeError):
    """``get_runtime()`` was called before the integrator installed one."""


class KernelRuntime:
    """Process-wide holder of the config plus the provider factory.

    Subclass it and implement :meth:`provider_for`. The business layer also
    reads ``runtime.config`` and ``runtime.logger``.

    ``context.config.mcp.servers`` mirrors a legacy ``mcp_agent`` namespace
    that ``workflows.environment`` mutates in place to append workspace
    roots to the ``filesystem`` server's ``args``. It is the same dict object
    as ``config.mcp_servers``.
    """

    def __init__(self, config: KernelConfig, *, logger: Any | None = None) -> None:
        self.config = config
        if logger is None:
            from loguru import logger as _loguru_logger

            logger = _loguru_logger
        self.logger = logger
        self._mcp_servers = config.mcp_servers
        self.context = _ContextNamespace(
            config=_ConfigNamespace(mcp=_MCPNamespace(servers=self._mcp_servers))
        )

    @property
    def mcp_servers(self) -> dict[str, MCPServerConfig]:
        return self._mcp_servers

    def provider_for(
        self,
        *,
        provider_name: str | None = None,
        connection_id: str | None = None,
        phase: str = "default",
        model: str | None = None,
        execution_profile: Any | None = None,
    ) -> "LLMProvider":
        """CONTRACT. Return a provider for the requested phase/model.

        Callers (``seams.llm_runtime.get_workflow_provider`` and the
        integrator's ``Agent.attach_llm``) pass whichever of the keyword
        arguments they know; treat all of them as hints. ``phase`` is one of
        ``"default"``, ``"planning"``, ``"implementation"`` and should be
        resolved through :meth:`KernelConfig.resolve_phase`.
        """
        raise NotImplementedError(
            "KernelRuntime.provider_for: the integrator must return an "
            "seams.llm_runtime.LLMProvider for the requested phase/model"
        )

    async def aclose(self) -> None:  # pragma: no cover - optional
        """Release provider resources. Optional; the default does nothing."""
        return None


@dataclass(slots=True)
class _MCPNamespace:
    servers: dict[str, MCPServerConfig]


@dataclass(slots=True)
class _ConfigNamespace:
    mcp: _MCPNamespace


@dataclass(slots=True)
class _ContextNamespace:
    config: _ConfigNamespace


_runtime_lock = threading.Lock()
_runtime: KernelRuntime | None = None


def get_runtime() -> KernelRuntime:
    """Return the installed runtime or raise :class:`RuntimeNotConfigured`.

    DeepCode lazily loaded ``deepcode_config.json`` here. The kernel has no
    file format, so an unconfigured process is an error rather than a
    silent default.
    """
    if _runtime is None:
        raise RuntimeNotConfigured(
            "No KernelRuntime installed. Call seams.config.set_runtime(...) "
            "before running any workflow."
        )
    return _runtime


def set_runtime(runtime: KernelRuntime | None) -> None:
    """Install (or clear, with ``None``) the process-wide runtime."""
    global _runtime
    with _runtime_lock:
        _runtime = runtime


@contextmanager
def use_runtime(runtime: KernelRuntime) -> Iterator[KernelRuntime]:
    """Temporarily install ``runtime`` for the duration of a ``with`` block."""
    previous = _runtime
    set_runtime(runtime)
    try:
        yield runtime
    finally:
        set_runtime(previous)


__all__ = [
    "AgentDefaults",
    "AgentPhase",
    "AgentsConfig",
    "DeepCodeConfig",
    "DocumentSegmentationConfig",
    "KernelConfig",
    "KernelRuntime",
    "MCPServerConfig",
    "ResolvedAgentSettings",
    "RuntimeNotConfigured",
    "SecurityConfig",
    "ToolsConfig",
    "WorkspaceConfig",
    "get_runtime",
    "set_runtime",
    "use_runtime",
]
