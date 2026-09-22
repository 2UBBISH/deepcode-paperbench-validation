"""``filesystem`` server: the two tools the reference analyzer is allowed.

Upstream used ``@modelcontextprotocol/server-filesystem`` rooted at the
workspace. Here ``read_text_file`` and ``list_directory`` are rooted at the
run's workspace and refuse anything that resolves outside it. Output
formats follow the upstream server (``[DIR]`` / ``[FILE]`` listing lines).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from apps.v2.agent.paper2code.tools.registry import FunctionTool, ToolContext, exposed_name
from apps.v2.agent_engine.paper2code.seams.agent_runtime import Tool

_MAX_READ_CHARS = 400_000


def resolve_within(root: Path, path: str) -> Path:
    base = root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if resolved != base and not resolved.is_relative_to(base):
        raise PermissionError(f"Access denied - path outside allowed directories: {path}")
    return resolved


def tools(ctx: ToolContext) -> Sequence[Tool]:
    root = Path(ctx.workspace)

    async def read_text_file(path: str, head: int | None = None, tail: int | None = None) -> str:
        try:
            target = resolve_within(root, path)
            if not target.is_file():
                return f"Error: not a file: {path}"
            text = target.read_text(encoding="utf-8", errors="replace")
        except PermissionError as exc:
            return f"Error: {exc}"
        except OSError as exc:
            return f"Error reading {path}: {exc}"
        if head is not None:
            text = "\n".join(text.splitlines()[: max(int(head), 0)])
        elif tail is not None:
            text = "\n".join(text.splitlines()[-max(int(tail), 0) :] if int(tail) > 0 else [])
        if len(text) > _MAX_READ_CHARS:
            text = text[:_MAX_READ_CHARS] + f"\n\n[truncated at {_MAX_READ_CHARS} characters]"
        return text

    async def list_directory(path: str) -> str:
        try:
            target = resolve_within(root, path)
            if not target.is_dir():
                return f"Error: not a directory: {path}"
            entries = sorted(target.iterdir(), key=lambda p: p.name)
        except PermissionError as exc:
            return f"Error: {exc}"
        except OSError as exc:
            return f"Error listing {path}: {exc}"
        lines = [f"[DIR] {p.name}" if p.is_dir() else f"[FILE] {p.name}" for p in entries]
        return "\n".join(lines) or "(empty directory)"

    return [
        FunctionTool(
            name=exposed_name("filesystem", "read_text_file"),
            description=(
                "Read the complete contents of a file as text. Use head or tail to "
                "read only the first or last N lines. Only paths inside the run "
                "workspace are allowed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "head": {"type": "integer", "description": "first N lines only"},
                    "tail": {"type": "integer", "description": "last N lines only"},
                },
                "required": ["path"],
            },
            fn=read_text_file,
            read_only=True,
        ),
        FunctionTool(
            name=exposed_name("filesystem", "list_directory"),
            description=(
                "List files and directories at a path, one per line, prefixed "
                "[FILE] or [DIR]. Only paths inside the run workspace are allowed."
            ),
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            fn=list_directory,
            read_only=True,
        ),
    ]


__all__ = ["resolve_within", "tools"]
