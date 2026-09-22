"""The engine's five shipped servers, called in-process.

Four of them are ``FastMCP`` modules: the decorator leaves the original
function in place and registers it with the module's ``mcp._tool_manager``,
which also holds the pydantic-generated JSON Schema. We read that registry
and wrap each entry as a :class:`FunctionTool`. ``command_executor`` is a
low-level ``mcp.server.Server``; its two tools are dispatched through its
``handle_call_tool``.

Module-level state in those servers (``code_implementation_server.WORKSPACE_DIR``
set by the engine through ``set_workspace``; ``document_segmentation_server``'s
index cache) works unchanged: one run is one process.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.tools.registry import (
    DenylistedTool,
    FunctionTool,
    ToolContext,
    exposed_name,
)
from apps.v2.agent_engine.paper2code.seams.agent_runtime import Tool

_READ_ONLY = frozenset(
    {
        "read_file",
        "read_multiple_files",
        "read_code_mem",
        "search_code",
        "get_file_structure",
        "get_operation_history",
        "read_document_segments",
        "get_document_overview",
        "search_code_references",
        "get_indexes_overview",
        "parse_github_urls",
    }
)


def _fastmcp_tools(server: str, mcp: Any, *, exclude: frozenset[str] = frozenset()) -> list[Tool]:
    out: list[Tool] = []
    for entry in mcp._tool_manager.list_tools():
        if entry.name in exclude:
            continue
        out.append(
            FunctionTool(
                name=exposed_name(server, entry.name),
                description=entry.description or entry.name,
                parameters=entry.parameters,
                fn=entry.fn,
                read_only=entry.name in _READ_ONLY,
            )
        )
    return out


def code_implementation_tools(ctx: ToolContext) -> Sequence[Tool]:
    """The engine's tools as they are, ``execute_python`` / ``execute_bash`` included: the implement phase
    runs model code the way upstream does — a local subprocess inside the engine's write-fence sandbox
    (``support/sandbox``: seatbelt on macOS, bwrap on Linux) behind ``command_guard`` (PLAN-3 §2 8i,
    owner 2026-09-17: "与上游一致"). The port-routed pair in ``tools/execute.py`` is no longer registered;
    remote execution is step 10's alone."""
    from apps.v2.agent_engine.paper2code.tools import code_implementation_server as srv

    return _fastmcp_tools("code-implementation", srv.mcp)


def document_segmentation_tools(ctx: ToolContext) -> Sequence[Tool]:
    from apps.v2.agent_engine.paper2code.tools import document_segmentation_server as srv

    return _fastmcp_tools("document-segmentation", srv.mcp)


def code_reference_indexer_tools(ctx: ToolContext) -> Sequence[Tool]:
    from apps.v2.agent_engine.paper2code.tools import code_reference_indexer as srv

    return _fastmcp_tools("code-reference-indexer", srv.mcp)


class CloneIntoCodeBase(Tool):
    """``git_clone`` whose target is always ``<code_base>/<name>``.

    The engine's tool resolves an empty or relative ``target_path`` against
    the process cwd — on a real run the download agent passed ``""`` and
    two repositories landed in the repository root. The name is the last
    path component the model gave, or the one inferred from the URL.
    """

    def __init__(self, inner: Tool, ctx: ToolContext) -> None:
        self._inner = inner
        self._ctx = ctx

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def parameters(self) -> Any:
        return self._inner.parameters

    @property
    def timeout_s(self) -> float | None:
        return self._inner.timeout_s

    def target_for(self, repo_url: str, target_path: str | None) -> str:
        from apps.v2.agent_engine.paper2code.tools.git_command import GitHubURLExtractor

        base = self._ctx.code_base or (Path(self._ctx.workspace) / "code_base")
        name = Path(str(target_path or "")).name.strip()
        if not name or name in {".", ".."}:
            name = GitHubURLExtractor().infer_repo_name(repo_url)
            if name in {"", "repository"}:  # the engine's generic fallback for non-GitHub URLs
                name = Path(repo_url.rstrip("/")).name.removesuffix(".git") or "repository"
        return str(Path(base) / name)

    async def execute(self, **kwargs: Any) -> Any:
        if "repo_url" in kwargs:
            kwargs["target_path"] = self.target_for(str(kwargs["repo_url"]), kwargs.get("target_path"))
        return await self._inner.execute(**kwargs)


def github_downloader_tools(ctx: ToolContext) -> Sequence[Tool]:
    from apps.v2.agent_engine.paper2code.tools import git_command as srv

    tools: list[Tool] = []
    for tool in _fastmcp_tools("github-downloader", srv.mcp):
        if tool.name.endswith("_git_clone"):
            tool = CloneIntoCodeBase(tool, ctx)
        tools.append(DenylistedTool(tool, ctx))
    return tools


# -- command-executor (low-level Server) --------------------------------------


@functools.lru_cache(maxsize=1)
def _command_executor_definitions() -> tuple[Any, ...]:
    """The server's ``list_tools`` result, computed once on a private loop."""
    from apps.v2.agent_engine.paper2code.tools import command_executor as srv

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return tuple(pool.submit(asyncio.run, srv.handle_list_tools()).result())


async def _call_command_executor(tool_name: str, **kwargs: Any) -> str:
    from apps.v2.agent_engine.paper2code.tools import command_executor as srv

    blocks = await srv.handle_call_tool(tool_name, dict(kwargs))
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else str(block))
    return "\n".join(parts) or "(no output)"


def command_executor_tools(ctx: ToolContext) -> Sequence[Tool]:
    out: list[Tool] = []
    for definition in _command_executor_definitions():
        out.append(
            FunctionTool(
                name=exposed_name("command-executor", definition.name),
                description=definition.description or definition.name,
                parameters=dict(definition.inputSchema or {}),
                fn=functools.partial(_call_command_executor, definition.name),
            )
        )
    return out


__all__ = [
    "CloneIntoCodeBase",
    "code_implementation_tools",
    "code_reference_indexer_tools",
    "command_executor_tools",
    "document_segmentation_tools",
    "github_downloader_tools",
]
