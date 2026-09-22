"""Loop seam: the tool-calling agent loop the business layer runs on.

Everything the loop *carries* is real code copied from DeepCode
(``core/agent_runtime/tools/{base,registry,alias}.py``,
``core/agent_runtime/hook.py``, the ``AgentRunSpec`` / ``AgentRunResult``
dataclasses from ``core/agent_runtime/runner.py``). The loop *itself*,
``AgentRunner.run``, is the contract. Its behavioral obligations are spelled
out in the method docstring and, at length, in ``docs/INTEGRATION.md``.

Who depends on what:

    workflows/code_implementation_workflow.py
        builds a ``ToolRegistry`` of ``AliasedTool``s, subclasses
        ``AgentHook``, fills an ``AgentRunSpec`` and calls
        ``AgentRunner(provider).run(spec)``.
    seams/compat.py  (``AugmentedLLM.generate``)
        the integrator's implementation is expected to do the same for every
        other agent in the business layer.
    workflows/agent_orchestration_engine.py
        reads ``AgentRunResult`` fields: ``final_content``, ``stop_reason``,
        ``tools_used``, ``usage``, ``error``.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMProvider, LLMResponse, ToolCallRequest

# ===========================================================================
# Tools (REAL)
# ===========================================================================

_ToolT = TypeVar("_ToolT", bound="Tool")

_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

_DESCRIPTION_MAX_CHARS = 2_000
_DESCRIPTION_MIN_CHARS = 20


def description_quality_issues(description: str) -> list[str]:
    issues: list[str] = []
    text = str(description or "")
    if not text.strip():
        issues.append("description is empty")
    elif len(text) < _DESCRIPTION_MIN_CHARS:
        issues.append(
            f"description is only {len(text)} chars; be more specific "
            f"(min {_DESCRIPTION_MIN_CHARS})"
        )
    if len(text) > _DESCRIPTION_MAX_CHARS:
        issues.append(
            f"description is {len(text)} chars (max {_DESCRIPTION_MAX_CHARS}); "
            "trim it — tool definitions count against the prompt budget"
        )
    return issues


def sanitize_description(
    description: str,
    *,
    name: str = "tool",
    max_chars: int = _DESCRIPTION_MAX_CHARS,
) -> str:
    text = str(description or "").strip()
    if len(text) <= max_chars:
        return text or f"{name} tool (no description provided)"
    cut = text[:max_chars]
    boundary = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("\n"))
    if boundary > _DESCRIPTION_MIN_CHARS:
        cut = cut[: boundary + 1]
    return cut + " …[truncated]"


class ToolResult(str):
    """Model-visible tool text with execution metadata attached.

    Still a ``str`` so every consumer that expects text keeps working.
    """

    is_error: bool
    metadata: dict[str, Any]

    def __new__(
        cls,
        content: str,
        *,
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        result = super().__new__(cls, content)
        result.is_error = is_error
        result.metadata = dict(metadata or {})
        return result

    def with_content(self, content: str) -> "ToolResult":
        return type(self)(content, is_error=self.is_error, metadata=self.metadata)


class Schema(ABC):
    """JSON Schema fragment helpers used for parameter validation."""

    @staticmethod
    def resolve_json_schema_type(t: Any) -> str | None:
        if isinstance(t, list):
            return next((x for x in t if x != "null"), None)
        return t  # type: ignore[return-value]

    @staticmethod
    def subpath(path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    @staticmethod
    def validate_json_schema_value(
        val: Any, schema: dict[str, Any], path: str = ""
    ) -> list[str]:
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get(
            "nullable", False
        )
        t = Schema.resolve_json_schema_type(raw_type)
        label = path or "parameter"

        if nullable and val is None:
            return []
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (
            not isinstance(val, _JSON_TYPE_MAP["number"]) or isinstance(val, bool)
        ):
            return [f"{label} should be number"]
        if (
            t in _JSON_TYPE_MAP
            and t not in ("integer", "number")
            and not isinstance(val, _JSON_TYPE_MAP[t])
        ):
            return [f"{label} should be {t}"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if t == "object":
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"missing required {Schema.subpath(path, k)}")
            for k, v in val.items():
                if k in props:
                    errors.extend(
                        Schema.validate_json_schema_value(
                            v, props[k], Schema.subpath(path, k)
                        )
                    )
        if t == "array":
            if "minItems" in schema and len(val) < schema["minItems"]:
                errors.append(f"{label} must have at least {schema['minItems']} items")
            if "maxItems" in schema and len(val) > schema["maxItems"]:
                errors.append(f"{label} must be at most {schema['maxItems']} items")
            if "items" in schema:
                prefix = f"{path}[{{}}]" if path else "[{}]"
                for i, item in enumerate(val):
                    errors.extend(
                        Schema.validate_json_schema_value(
                            item, schema["items"], prefix.format(i)
                        )
                    )
        return errors

    @staticmethod
    def fragment(value: Any) -> dict[str, Any]:
        to_js = getattr(value, "to_json_schema", None)
        if callable(to_js):
            return to_js()
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected schema object or dict, got {type(value).__name__}")

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]: ...

    def validate_value(self, value: Any, path: str = "") -> list[str]:
        return Schema.validate_json_schema_value(value, self.to_json_schema(), path)


class Tool(ABC):
    """One capability the model can call.

    Subclasses provide ``name``, ``description``, ``parameters`` (a JSON
    Schema ``object``) and ``async execute(**kwargs)``. The MCP tools the
    business layer asks for arrive through the integrator's ``Agent``
    implementation as instances of this class registered under
    ``mcp_<server>_<tool>`` names (see ``build_aliased_registry``).
    """

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    _BOOL_TRUE = frozenset(("true", "1", "yes"))
    _BOOL_FALSE = frozenset(("false", "0", "no"))

    @staticmethod
    def _resolve_type(t: Any) -> str | None:
        return Schema.resolve_json_schema_type(t)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...

    @property
    def read_only(self) -> bool:
        return False

    @property
    def timeout_s(self) -> float | None:
        """Wall-clock budget for one ``execute()``; the runner enforces it."""
        return None

    @property
    def concurrency_safe(self) -> bool:
        return self.read_only and not self.exclusive

    @property
    def exclusive(self) -> bool:
        return False

    def presentation_detail(self, arguments: dict[str, Any]) -> str | None:
        return None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any: ...

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        return {
            k: self._cast_value(v, props[k]) if k in props else v
            for k, v in obj.items()
        }

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        t = self._resolve_type(schema.get("type"))

        if t == "boolean" and isinstance(val, bool):
            return val
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if t in self._TYPE_MAP and t not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[t]
            if isinstance(val, expected):
                return val

        if isinstance(val, str) and t in ("integer", "number"):
            try:
                return int(val) if t == "integer" else float(val)
            except ValueError:
                return val

        if t == "string":
            return val if val is None else str(val)

        if t == "boolean" and isinstance(val, str):
            low = val.lower()
            if low in self._BOOL_TRUE:
                return True
            if low in self._BOOL_FALSE:
                return False
            return val

        if t == "array" and isinstance(val, list):
            items = schema.get("items")
            return [self._cast_value(x, items) for x in val] if items else val

        if t == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return Schema.validate_json_schema_value(
            params, {**schema, "type": "object"}, ""
        )

    def to_schema(self) -> dict[str, Any]:
        """OpenAI function-tool schema; what ``LLMProvider.chat_with_retry`` gets."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool_parameters(schema: dict[str, Any]) -> Callable[[type[_ToolT]], type[_ToolT]]:
    """Class decorator: attach a JSON Schema as the ``parameters`` property."""

    def decorator(cls: type[_ToolT]) -> type[_ToolT]:
        frozen = deepcopy(schema)

        @property
        def parameters(self: Any) -> dict[str, Any]:
            return deepcopy(frozen)

        cls._tool_parameters_schema = deepcopy(frozen)
        cls.parameters = parameters  # type: ignore[assignment]

        abstract = getattr(cls, "__abstractmethods__", None)
        if abstract is not None and "parameters" in abstract:
            cls.__abstractmethods__ = frozenset(abstract - {"parameters"})  # type: ignore[misc]

        return cls

    return decorator


class ToolRegistry:
    """Name → :class:`Tool` map with optional ownership of MCP server stacks.

    ``get_definitions`` orders built-in tools before ``mcp_*`` tools and
    caches the result until the registry changes. ``execute`` never raises
    for a tool failure: it returns an ``"Error ..."`` string with a hint the
    model can act on.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._owned_server_stacks: dict[str, AsyncExitStack] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        if self._cached_definitions is not None:
            return self._cached_definitions

        definitions = [tool.to_schema() for tool in self._tools.values()]
        builtins: list[dict[str, Any]] = []
        mcp_tools: list[dict[str, Any]] = []
        for schema in definitions:
            name = self._schema_name(schema)
            if name.startswith("mcp_"):
                mcp_tools.append(schema)
            else:
                builtins.append(schema)

        builtins.sort(key=self._schema_name)
        mcp_tools.sort(key=self._schema_name)
        self._cached_definitions = builtins + mcp_tools
        return self._cached_definitions

    def prepare_call(
        self,
        name: str,
        params: dict[str, Any],
    ) -> tuple[Tool | None, dict[str, Any], str | None]:
        if not isinstance(params, dict) and name in ("write_file", "read_file"):
            return (
                None,
                params,
                (
                    f"Error: Tool '{name}' parameters must be a JSON object, got {type(params).__name__}. "
                    'Use named parameters: tool_name(param1="value1", param2="value2")'
                ),
            )

        tool = self._tools.get(name)
        if not tool:
            return (
                None,
                params,
                (
                    f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"
                ),
            )

        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return (
                tool,
                cast_params,
                (f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)),
            )
        return tool, cast_params, None

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        _HINT = "\n\n[Analyze the error above and try a different approach.]"
        tool, params, error = self.prepare_call(name, params)
        if error:
            return error + _HINT

        try:
            assert tool is not None
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def read_only_tool_names(self) -> frozenset[str]:
        return frozenset(name for name, tool in self._tools.items() if tool.read_only)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def attach_server_stack(self, server_name: str, stack: AsyncExitStack) -> None:
        """Track a per-MCP-server ``AsyncExitStack`` so :meth:`aclose` drains it."""
        self._owned_server_stacks[server_name] = stack

    @staticmethod
    def _close_timeout_s() -> float:
        raw = os.environ.get("DEEPCODE_MCP_CLOSE_TIMEOUT_S", "8").strip()
        try:
            value = float(raw)
        except ValueError:
            return 8.0
        return max(value, 0.1)

    async def aclose(self) -> None:
        """Close every owned MCP server stack and forget all tools."""
        errors: list[BaseException] = []
        timeout_s = self._close_timeout_s()
        for name, stack in list(self._owned_server_stacks.items()):
            try:
                await asyncio.wait_for(stack.aclose(), timeout=timeout_s)
            except asyncio.TimeoutError:
                errors.append(
                    TimeoutError(
                        f"MCP server '{name}' close timed out after {timeout_s:g}s"
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - log and continue
                errors.append(exc)
            finally:
                self._owned_server_stacks.pop(name, None)
        try:
            await asyncio.wait_for(self._exit_stack.aclose(), timeout=timeout_s)
        except asyncio.TimeoutError:
            errors.append(
                TimeoutError(
                    f"ToolRegistry exit stack close timed out after {timeout_s:g}s"
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        self._tools.clear()
        self._cached_definitions = None
        for exc in errors:
            if _is_benign_cancel_teardown(exc):
                logger.debug("ToolRegistry.aclose: benign anyio teardown noise: {}", exc)
            else:
                logger.warning("ToolRegistry.aclose: error draining stack: {}", exc)


def _is_benign_cancel_teardown(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.CancelledError):
        return True
    msg = str(exc).lower()
    benign_markers = (
        "cancel scope in a different task",
        "isn't the current task",
        "cancelled via cancel scope",
    )
    return any(marker in msg for marker in benign_markers)


class AliasedTool(Tool):
    """A tool re-exposed under a different name; delegates everything else.

    MCP tools register as ``mcp_<server>_<tool>``; prompts address them by
    bare name (``write_file``, ``read_code_mem``). The implementation
    workflow builds a model-facing registry of these.
    """

    def __init__(self, inner: Tool, alias: str):
        self._inner = inner
        self._alias = alias

    @property
    def inner(self) -> Tool:
        return self._inner

    @property
    def name(self) -> str:
        return self._alias

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._inner.parameters

    @property
    def read_only(self) -> bool:
        return self._inner.read_only

    @property
    def exclusive(self) -> bool:
        return self._inner.exclusive

    @property
    def concurrency_safe(self) -> bool:
        return self._inner.concurrency_safe

    async def execute(self, **kwargs: Any) -> Any:
        return await self._inner.execute(**kwargs)


def find_tool_by_suffix(registry: ToolRegistry, bare_name: str) -> Tool | None:
    """Exact name first, then any ``*_<bare_name>`` match."""
    tool = registry.get(bare_name)
    if tool is not None:
        return tool
    suffix = f"_{bare_name}"
    for candidate in registry.tool_names:
        if candidate.endswith(suffix):
            return registry.get(candidate)
    return None


def build_aliased_registry(
    source: ToolRegistry,
    bare_names: list[str],
) -> tuple[ToolRegistry, list[str]]:
    """Registry exposing ``bare_names`` aliased from ``source``, plus the misses.

    The returned registry owns no server stacks; ``source`` keeps
    responsibility for MCP process cleanup.
    """
    aliased = ToolRegistry()
    missing: list[str] = []
    for bare_name in bare_names:
        tool = find_tool_by_suffix(source, bare_name)
        if tool is None:
            missing.append(bare_name)
            continue
        if tool.name == bare_name:
            aliased.register(tool)
        else:
            aliased.register(AliasedTool(tool, bare_name))
    return aliased, missing


# ===========================================================================
# Hooks (REAL)
# ===========================================================================


@dataclass(slots=True)
class AgentHookContext:
    """Mutable per-iteration state the runner exposes to hooks.

    ``messages`` is the live conversation list; a hook may append to it or
    replace its contents in place (``context.messages[:] = ...``). The
    implementation workflow's hook does both.
    """

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tool_events: list[dict[str, str]] = field(default_factory=list)
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    response_ordinal: int = 0


class AgentHook:
    """Lifecycle surface. Every method is a no-op by default.

    The implementation workflow overrides ``before_execute_tools``,
    ``after_iteration`` and ``finalize_content``. See
    ``docs/INTEGRATION.md`` for the order the runner must call these in.
    """

    def __init__(self, reraise: bool = False) -> None:
        self._reraise = reraise

    def wants_streaming(self) -> bool:
        return False

    async def before_iteration(self, context: AgentHookContext) -> None:
        pass

    async def before_model_request(self, context: AgentHookContext) -> None:
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        pass

    async def on_reasoning_stream(
        self, context: AgentHookContext, delta: str, channel: Any
    ) -> None:
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        pass

    async def on_model_response(self, context: AgentHookContext) -> None:
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        pass

    def finalize_content(
        self, context: AgentHookContext, content: str | None
    ) -> str | None:
        return content


class CompositeHook(AgentHook):
    """Fan-out hook that delegates to an ordered list of hooks."""

    __slots__ = ("_hooks",)

    def __init__(self, hooks: list[AgentHook]) -> None:
        super().__init__()
        self._hooks = list(hooks)

    def wants_streaming(self) -> bool:
        return any(h.wants_streaming() for h in self._hooks)

    async def _for_each_hook_safe(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> None:
        for h in self._hooks:
            if getattr(h, "_reraise", False):
                await getattr(h, method_name)(*args, **kwargs)
                continue
            try:
                await getattr(h, method_name)(*args, **kwargs)
            except Exception:
                logger.exception(
                    "AgentHook.{} error in {}", method_name, type(h).__name__
                )

    async def before_iteration(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("before_iteration", context)

    async def before_model_request(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("before_model_request", context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        await self._for_each_hook_safe("on_stream", context, delta)

    async def on_reasoning_stream(
        self, context: AgentHookContext, delta: str, channel: Any
    ) -> None:
        await self._for_each_hook_safe("on_reasoning_stream", context, delta, channel)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self._for_each_hook_safe("on_stream_end", context, resuming=resuming)

    async def on_model_response(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("on_model_response", context)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("before_execute_tools", context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("after_iteration", context)

    def finalize_content(
        self, context: AgentHookContext, content: str | None
    ) -> str | None:
        for h in self._hooks:
            content = h.finalize_content(context, content)
        return content


# ===========================================================================
# Run spec / result (REAL)
# ===========================================================================

_DEFAULT_ERROR_MESSAGE = "I encountered an error processing your request."
DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for one agent execution.

    DeepCode's spec carries many more knobs (compaction strategy, token
    meter, C3/C4 hooks, repeat-call reminders). Only the fields the business
    layer sets, plus the plain ones an integrator is likely to want, are
    kept here. Field semantics are in ``docs/INTEGRATION.md``.
    """

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int | None
    max_tool_result_chars: int
    transient_context_messages: tuple[dict[str, Any], ...] = ()
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    workspace: Path | None = None
    session_key: str | None = None
    context_window_tokens: int | None = None
    context_block_limit: int | None = None
    provider_retry_mode: str = "standard"
    progress_callback: Any | None = None
    retry_wait_callback: Any | None = None
    checkpoint_callback: Any | None = None
    injection_callback: Any | None = None
    llm_timeout_s: float | None = None
    should_stop_callback: Any | None = None
    tool_filter: Any | None = None
    permission_checker: Any | None = None
    approval_callback: Any | None = None

    def allowed_tool_names(self) -> frozenset[str] | None:
        if self.tool_filter is None:
            return None
        value = self.tool_filter()
        if value is None:
            return None
        return frozenset(str(name) for name in value)

    def tool_definitions(self) -> list[dict[str, Any]]:
        definitions = self.tools.get_definitions()
        allowed = self.allowed_tool_names()
        if allowed is None:
            return definitions
        return [
            schema
            for schema in definitions
            if ToolRegistry._schema_name(schema) in allowed
        ]


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of one agent execution.

    ``stop_reason`` values the business layer inspects:
    ``"completed"``, ``"max_iterations"``, ``"callback_stop"``,
    ``"tool_error"``, ``"error"``, ``"empty_final_response"``.
    """

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False


# ===========================================================================
# Runner (CONTRACT)
# ===========================================================================


class AgentRunner:
    """The tool-calling loop. ``run`` is the contract.

    Construct with the provider every model call goes through. The
    implementation workflow does ``AgentRunner(provider).run(spec)``; the
    integrator's ``AugmentedLLM.generate`` is expected to do the same.

    Obligations of ``run`` (the full text is in ``docs/INTEGRATION.md``):

    1. Start from ``spec.initial_messages``. Loop: check
       ``should_stop_callback`` → request model → if tool calls: run them,
       append results, call hooks, loop; else: try ``injection_callback``,
       and loop if it returned messages, otherwise finish.
    2. ``should_stop_callback()`` is awaited before every model request.
       A non-empty string means stop now with ``stop_reason="callback_stop"``
       and that string as ``final_content``.
    3. ``injection_callback()`` is awaited when the model produced a final
       answer (no tool calls). Returned ``{"role","content"}`` dicts are
       appended and the loop continues; empty means finish.
    4. Hook order per iteration: ``before_iteration`` → ``before_model_request``
       → (provider call) → ``on_model_response`` → if tools:
       ``before_execute_tools`` → (execute) → ``after_iteration``; if no
       tools: ``finalize_content`` → ``after_iteration``.
    5. ``permission_checker(name, args)`` returns ``(decision, reason)`` with
       decision in ``"allow" | "ask" | "deny"``; ``ask`` is resolved by
       ``approval_callback(name, args, reason) -> bool`` or denied when there
       is none. Denials become an ``"Error: permission denied: ..."`` tool
       result, never an exception.
    6. Tool results longer than ``max_tool_result_chars`` are truncated.
       Tool exceptions become ``"Error executing <name>: ..."`` results.
    7. ``max_iterations`` counts model requests. On exhaustion, finish with
       ``stop_reason="max_iterations"`` and ``max_iterations_message`` (or
       :data:`DEFAULT_MAX_ITERATIONS_MESSAGE`) as ``final_content``.
    8. A provider ``finish_reason == "error"`` ends the run with
       ``stop_reason="error"`` and ``error`` set; ``final_content`` falls
       back to ``spec.error_message``.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        raise NotImplementedError(
            "AgentRunner.run: the integrator must implement the tool-calling "
            "loop described in docs/INTEGRATION.md"
        )


__all__ = [
    "AgentHook",
    "AgentHookContext",
    "AgentRunResult",
    "AgentRunSpec",
    "AgentRunner",
    "AliasedTool",
    "CompositeHook",
    "DEFAULT_MAX_ITERATIONS_MESSAGE",
    "Schema",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "build_aliased_registry",
    "description_quality_issues",
    "find_tool_by_suffix",
    "sanitize_description",
    "tool_parameters",
]
