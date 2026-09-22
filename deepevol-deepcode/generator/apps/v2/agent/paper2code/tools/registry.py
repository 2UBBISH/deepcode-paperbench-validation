"""Server name → in-process tools, with the names the model sees sanitized.

The engine asks an ``Agent`` for servers by name (``"code-implementation"``,
``"fetch"``, ...). Upstream DeepCode launched each as an MCP subprocess and
exposed its tools as ``mcp_<server>_<tool>``. This line runs the same
functions in-process and keeps the exposed-name convention with one change
carried over from the validation repo: hyphens become underscores, because
some OpenAI-compatible endpoints silently drop tool calls whose declared
name contains ``-``. ``seams.compat.apply_tool_filter`` matches on the
sanitized form (VENDOR.md entry 2).

Unknown server names raise; upstream logged and continued, which let a
pipeline run to completion with no tools and produce garbage.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.v2.agent_engine.paper2code.seams.agent_runtime import (
    Tool,
    ToolRegistry,
    sanitize_description,
)
from apps.v2.agent_engine.paper2code.seams.config import get_runtime

EXPOSED_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

SERVER_NAMES: tuple[str, ...] = (
    "code-implementation",
    "command-executor",
    "document-segmentation",
    "code-reference-indexer",
    "github-downloader",
    "filesystem",
    "fetch",
)

MAX_SAME_URL_FETCHES = 2


class UnknownToolServer(ValueError):
    """``server_names`` named a server this line does not provide."""


def sanitize_name(name: str) -> str:
    return name.replace("-", "_")


def exposed_name(server: str, tool: str) -> str:
    name = sanitize_name(f"mcp_{server}_{tool}")
    if not EXPOSED_NAME_RE.match(name):
        raise ValueError(f"tool name {name!r} is not safe for tool-call decoding")
    return name


@dataclass(slots=True)
class ToolContext:
    """What the tools need from the run: paths, the denylist, the execution port.

    ``fetch_ledger`` counts ``(tool, url)`` servings for the repeat-fetch
    guard; it lives here so it is per run, not per process.
    """

    workspace: Path
    denylist: tuple[str, ...] = ()
    port: Any | None = None
    fetch_ledger: dict[tuple[str, str], int] = field(default_factory=dict)
    # where clones land whatever target_path the model passes (the task's code_base/);
    # None → <workspace>/code_base. The engine's git_clone would otherwise use the cwd.
    code_base: Path | None = None


def current_tool_context() -> ToolContext:
    """The context the installed runtime carries; raises if the runtime has none."""
    runtime = get_runtime()
    ctx = getattr(runtime, "tool_context", None)
    if ctx is None:
        raise RuntimeError(
            "the installed KernelRuntime has no tool_context; install "
            "apps.v2.agent.paper2code.config.PaperKernelRuntime or set the attribute"
        )
    return ctx


# ---------------------------------------------------------------------------
# denylist
# ---------------------------------------------------------------------------


def parse_denylist(text: str) -> tuple[str, ...]:
    """PaperBench ``blacklist.txt`` → patterns: one per line, ``#`` comments, blanks dropped."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return tuple(out)


def url_like_values(kwargs: dict[str, Any]) -> list[str]:
    """Every string argument that could carry a URL."""
    out: list[str] = []
    for value in kwargs.values():
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(v for v in value if isinstance(v, str))
    return out


def denylist_hit(denylist: Iterable[str], values: Iterable[str]) -> tuple[str, str] | None:
    """``(value, pattern)`` for the first argument containing a denied substring."""
    vals = list(values)
    for pattern in denylist:
        needle = pattern.strip().lower()
        if not needle:
            continue
        for value in vals:
            if needle in value.lower():
                return value, pattern
    return None


def denylist_refusal(value: str) -> str:
    return (
        f"BLOCKED: '{value}' is on this task's blacklist of prohibited "
        "resources (the paper's own implementation or a known reference "
        "implementation). Accessing it would be cheating. Do not retry this "
        "or any other URL under it; continue using only the paper text and "
        "unrelated sources."
    )


# ---------------------------------------------------------------------------
# schema helpers
# ---------------------------------------------------------------------------


def normalize_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """FastMCP/pydantic schema → the plain JSON Schema the engine's validator reads.

    Drops ``title`` keys and folds ``anyOf: [{type: X}, {type: "null"}]``
    into ``type: [X, "null"]`` so ``Tool.validate_params`` sees a type.
    """

    def _walk(node: Any) -> Any:
        if isinstance(node, list):
            return [_walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title":
                continue
            out[key] = _walk(value)
        any_of = out.get("anyOf")
        if isinstance(any_of, list) and "type" not in out:
            types = [
                branch.get("type")
                for branch in any_of
                if isinstance(branch, dict) and isinstance(branch.get("type"), str)
            ]
            if len(types) == len(any_of) and types:
                out.pop("anyOf")
                out["type"] = types[0] if len(types) == 1 else types
        return out

    result = _walk(deepcopy(schema or {}))
    result.setdefault("type", "object")
    result.setdefault("properties", {})
    return result


# ---------------------------------------------------------------------------
# tool classes
# ---------------------------------------------------------------------------


class FunctionTool(Tool):
    """A plain (async or sync) function exposed as an engine ``Tool``."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
        read_only: bool = False,
        timeout_s: float | None = None,
    ) -> None:
        if not EXPOSED_NAME_RE.match(name):
            raise ValueError(f"tool name {name!r} is not safe for tool-call decoding")
        self._name = name
        self._description = sanitize_description(description, name=name)
        self._parameters = normalize_schema(parameters)
        self._fn = fn
        self._read_only = read_only
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._parameters)

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def timeout_s(self) -> float | None:
        return self._timeout_s

    async def execute(self, **kwargs: Any) -> Any:
        result = self._fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result


class DenylistedTool(Tool):
    """Refuse any call whose string arguments mention a denied resource."""

    def __init__(self, inner: Tool, ctx: ToolContext) -> None:
        self._inner = inner
        self._ctx = ctx

    @property
    def inner(self) -> Tool:
        return self._inner

    @property
    def name(self) -> str:
        return self._inner.name

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
    def timeout_s(self) -> float | None:
        return self._inner.timeout_s

    async def execute(self, **kwargs: Any) -> Any:
        hit = denylist_hit(self._ctx.denylist, url_like_values(kwargs))
        if hit is not None:
            return denylist_refusal(hit[0])
        return await self._inner.execute(**kwargs)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def _server_builders() -> dict[str, Callable[[ToolContext], Sequence[Tool]]]:
    from apps.v2.agent.paper2code.tools import fetch, filesystem, kernel_servers

    return {
        "code-implementation": kernel_servers.code_implementation_tools,
        "command-executor": kernel_servers.command_executor_tools,
        "document-segmentation": kernel_servers.document_segmentation_tools,
        "code-reference-indexer": kernel_servers.code_reference_indexer_tools,
        "github-downloader": kernel_servers.github_downloader_tools,
        "filesystem": filesystem.tools,
        "fetch": fetch.tools,
    }


def build_registry(server_names: Iterable[str], ctx: ToolContext) -> ToolRegistry:
    """Registry holding every tool of every named server; unknown names raise."""
    builders = _server_builders()
    registry = ToolRegistry()
    for server in server_names:
        builder = builders.get(server)
        if builder is None:
            raise UnknownToolServer(
                f"tool server {server!r} is not provided by the paper2code line; "
                f"known: {', '.join(SERVER_NAMES)}"
            )
        for tool in builder(ctx):
            registry.register(tool)
    return registry


__all__ = [
    "EXPOSED_NAME_RE",
    "MAX_SAME_URL_FETCHES",
    "SERVER_NAMES",
    "DenylistedTool",
    "FunctionTool",
    "ToolContext",
    "UnknownToolServer",
    "build_registry",
    "current_tool_context",
    "denylist_hit",
    "denylist_refusal",
    "exposed_name",
    "normalize_schema",
    "parse_denylist",
    "sanitize_name",
    "url_like_values",
]
