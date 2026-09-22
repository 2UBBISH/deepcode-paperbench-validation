"""C3: the engine's own tools run in-process through the bound Agent seam."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from apps.v2.agent.paper2code import agent as agent_mod
from apps.v2.agent.paper2code.tools.registry import ToolContext
from apps.v2.agent_engine.paper2code.seams.compat import Agent
from apps.v2.agent_engine.paper2code.seams.config import KernelConfig, KernelRuntime, use_runtime


class _Runtime(KernelRuntime):
    def __init__(self, config: KernelConfig, ctx: ToolContext) -> None:
        super().__init__(config)
        self.tool_context = ctx


def test_write_file_then_read_code_mem_through_agent(tmp_path: Path) -> None:
    agent_mod.bind()
    ctx = ToolContext(workspace=tmp_path)
    with use_runtime(_Runtime(KernelConfig(), ctx)):
        asyncio.run(_scenario(tmp_path))


async def _scenario(tmp_path: Path) -> None:
    task_dir = tmp_path / "paper_t"
    code_dir = task_dir / "generate_code"
    agent = Agent(name="CodeImplementationAgent", instruction="x", server_names=["code-implementation", "command-executor"])
    async with agent:
        assert "mcp_code_implementation_write_file" in agent.tool_registry
        setup = json.loads(await agent.call_tool("set_workspace", {"workspace_path": str(code_dir)}))
        assert setup["status"] == "success"
        written = json.loads(
            await agent.call_tool("write_file", {"file_path": "pkg/core.py", "content": "def f():\n    return 1\n"})
        )
        assert written["status"] == "success"
        assert (code_dir / "pkg" / "core.py").read_text() == "def f():\n    return 1\n"

        (task_dir / "implement_code_summary.md").write_text(
            "=" * 80 + "\n## IMPLEMENTATION File pkg/core.py; ROUND 1\n" + "=" * 80
            + "\n**Core Purpose:** returns one\n**Public Interface:** f()\n"
        )
        mem = json.loads(await agent.call_tool("read_code_mem", {"file_paths": ["pkg/core.py", "pkg/none.py"]}))
        found = {r["file_path"]: r["status"] for r in mem["results"]}
        assert found["pkg/core.py"] == "summary_found"
        assert found["pkg/none.py"] != "summary_found"

        listing = await agent.call_tool("execute_commands", {"commands": "mkdir -p a/b\ntouch a/b/c.txt", "working_directory": str(code_dir)})
        assert (code_dir / "a" / "b" / "c.txt").exists(), listing
    assert len(agent.tool_registry) == 0


def test_agent_without_tool_context_fails_loudly(tmp_path: Path) -> None:
    agent_mod.bind()
    with use_runtime(KernelRuntime(KernelConfig())):
        agent = Agent(name="x", instruction="", server_names=["fetch"])
        with pytest.raises(RuntimeError, match="tool_context"):
            asyncio.run(agent.__aenter__())
