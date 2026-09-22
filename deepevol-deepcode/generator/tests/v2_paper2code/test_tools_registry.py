"""C3: the seven servers build in-process, names are decode-safe, denylist bites."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from apps.v2.agent.paper2code.tools import registry
from apps.v2.agent.paper2code.tools.registry import (
    EXPOSED_NAME_RE,
    SERVER_NAMES,
    ToolContext,
    UnknownToolServer,
    build_registry,
    normalize_schema,
    parse_denylist,
)
from apps.v2.agent_engine.paper2code.seams.agent_runtime import build_aliased_registry


def test_all_seven_servers_build_with_safe_names(tmp_path: Path) -> None:
    ctx = ToolContext(workspace=tmp_path)
    reg = build_registry(SERVER_NAMES, ctx)
    assert len(reg) >= 20
    for name in reg.tool_names:
        assert EXPOSED_NAME_RE.match(name), name
        assert "-" not in name
    assert "mcp_github_downloader_git_clone" in reg
    assert "mcp_command_executor_execute_commands" in reg
    assert "mcp_document_segmentation_analyze_and_segment_document" in reg
    assert "mcp_code_reference_indexer_search_code_references" in reg
    assert "mcp_filesystem_read_text_file" in reg
    assert "mcp_fetch_fetch" in reg
    for schema in reg.get_definitions():
        assert schema["function"]["parameters"]["type"] == "object"


def test_bare_names_resolve_through_aliasing(tmp_path: Path) -> None:
    reg = build_registry(["code-implementation", "code-reference-indexer"], ToolContext(workspace=tmp_path))
    aliased, missing = build_aliased_registry(
        reg, ["write_file", "read_code_mem", "execute_python", "execute_bash", "search_code_references"]
    )
    assert missing == []
    assert aliased.get("execute_python").inner.name == "mcp_code_implementation_execute_python"


def test_unknown_server_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownToolServer):
        build_registry(["brave-search"], ToolContext(workspace=tmp_path))


def test_denylist_refuses_clone_of_blacklisted_repo(tmp_path: Path) -> None:
    ctx = ToolContext(workspace=tmp_path, denylist=parse_denylist("# authors\ngithub.com/authors/official-impl\n\n"))
    reg = build_registry(["github-downloader"], ctx)
    out = asyncio.run(
        reg.execute(
            "mcp_github_downloader_git_clone",
            {"repo_url": "https://github.com/Authors/Official-Impl.git", "target_path": str(tmp_path / "x")},
        )
    )
    assert out.startswith("BLOCKED:")
    assert not (tmp_path / "x").exists()


def test_execute_tools_are_the_engine_s_local_sandboxed_ones(tmp_path: Path, monkeypatch) -> None:
    # S3 (PLAN-3 8i): the implement phase executes like upstream — locally, write-fenced — never through the port
    from apps.v2.agent_engine.paper2code.tools import code_implementation_server as srv

    monkeypatch.setattr(srv, "WORKSPACE_DIR", tmp_path)

    class ExplodingPort:
        async def run(self, job):
            raise AssertionError("the port must not see implement-phase execution")

    reg = build_registry(["code-implementation"], ToolContext(workspace=tmp_path, port=ExplodingPort()))
    assert reg.get("mcp_code_implementation_execute_bash")._fn is srv.execute_bash
    assert reg.get("mcp_code_implementation_execute_python")._fn is srv.execute_python
    out = json.loads(asyncio.run(reg.execute("mcp_code_implementation_execute_bash", {"command": "echo hi"})))
    assert out["status"] == "success"
    assert out["stdout"].strip() == "hi"
    assert "remote" not in str(out.get("sandbox", ""))
    out = asyncio.run(reg.execute("mcp_code_implementation_execute_bash", {"command": "rm -rf /"}))
    assert "prohibited" in out.lower() or "blocked" in out.lower()


def test_normalize_schema_folds_nullable_anyof() -> None:
    schema = normalize_schema(
        {
            "properties": {
                "target_path": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "T"},
                "repo_url": {"title": "Repo Url", "type": "string"},
            },
            "required": ["repo_url"],
            "title": "git_cloneArguments",
            "type": "object",
        }
    )
    assert schema["properties"]["target_path"]["type"] == ["string", "null"]
    assert "title" not in schema
    assert "title" not in schema["properties"]["repo_url"]


def test_current_tool_context_requires_runtime_attribute() -> None:
    from apps.v2.agent_engine.paper2code.seams.config import KernelConfig, KernelRuntime, use_runtime

    with use_runtime(KernelRuntime(KernelConfig())), pytest.raises(RuntimeError, match="tool_context"):
        registry.current_tool_context()


def test_git_clone_target_is_forced_under_code_base(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "upstream-repo"
    repo.mkdir()
    (repo / "f.py").write_text("x = 1\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x", "HOME": str(tmp_path)}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)

    code_base = tmp_path / "task" / "code_base"
    reg = build_registry(["github-downloader"], ToolContext(workspace=tmp_path, code_base=code_base))
    out = asyncio.run(reg.execute("mcp_github_downloader_git_clone", {"repo_url": str(repo), "target_path": ""}))
    assert "Successfully cloned" in out, out
    assert (code_base / "upstream-repo" / "f.py").exists()
    out = asyncio.run(reg.execute("mcp_github_downloader_git_clone", {"repo_url": str(repo), "target_path": "/etc/elsewhere/second-copy"}))
    assert (code_base / "second-copy" / "f.py").exists(), out
    assert not list(Path.cwd().glob("upstream-repo"))
