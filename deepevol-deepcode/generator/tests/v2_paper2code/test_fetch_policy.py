"""C3: fetch refuses private hosts and denied resources, and stops repeat fetches."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from apps.v2.agent.paper2code.tools import fetch as fetch_mod
from apps.v2.agent.paper2code.tools.registry import ToolContext, build_registry


def _reg(tmp_path: Path, denylist=()):
    ctx = ToolContext(workspace=tmp_path, denylist=tuple(denylist))
    return build_registry(["fetch"], ctx), ctx


def test_private_network_urls_are_refused(tmp_path: Path) -> None:
    reg, _ = _reg(tmp_path)
    for url in ("http://127.0.0.1/x", "http://10.0.0.5/", "ftp://example.org/a", "http://user:pw@example.org/"):
        out = asyncio.run(reg.execute("mcp_fetch_fetch", {"url": url}))
        assert out.startswith("Error"), (url, out)


def test_denylisted_url_is_blocked_before_any_request(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    async def _boom(url: str):
        calls.append(url)
        raise AssertionError("must not be called")

    monkeypatch.setattr(fetch_mod, "_download", _boom)
    reg, _ = _reg(tmp_path, denylist=["github.com/authors/repo"])
    out = asyncio.run(reg.execute("mcp_fetch_fetch", {"url": "https://GitHub.com/Authors/Repo/blob/main/x.py"}))
    assert out.startswith("BLOCKED:")
    assert calls == []


def test_repeat_fetch_ledger_and_paging(tmp_path: Path, monkeypatch) -> None:
    async def _fake(url: str):
        return "text/html", b"<html><body><h1>Title</h1><p>" + b"x" * 30 + b"</p></body></html>"

    monkeypatch.setattr(fetch_mod, "_download", _fake)
    reg, ctx = _reg(tmp_path)
    url = "https://example.org/page"
    first = asyncio.run(reg.execute("mcp_fetch_fetch", {"url": url, "max_length": 10}))
    assert first.startswith(f"Contents of {url}:")
    assert "start_index of 10" in first
    second = asyncio.run(reg.execute("mcp_fetch_fetch", {"url": url, "max_length": 10, "start_index": 10}))
    assert second.startswith("Contents of")
    third = asyncio.run(reg.execute("mcp_fetch_fetch", {"url": url}))
    assert third.startswith("ALREADY FETCHED")
    assert ctx.fetch_ledger[("fetch", url)] == 2


def test_failed_downloads_count_toward_the_ledger(tmp_path: Path, monkeypatch) -> None:
    async def _fail(url: str):
        raise OSError("connection refused")

    monkeypatch.setattr(fetch_mod, "_download", _fail)
    reg, _ctx = _reg(tmp_path)
    url = "https://example.org/dead"
    for _ in range(2):
        assert asyncio.run(reg.execute("mcp_fetch_fetch", {"url": url})).startswith("Error")
    assert asyncio.run(reg.execute("mcp_fetch_fetch", {"url": url})).startswith("ALREADY FETCHED")


def test_fake_ip_dns_falls_back_to_system_resolver(monkeypatch) -> None:
    class _Resolver:
        async def resolve(self, host, port=0, family=0):
            raise OSError("URL resolves to a non-public address: 198.18.0.61")

    monkeypatch.setattr(fetch_mod, "_SafeResolver", _Resolver)
    connector = asyncio.run(fetch_mod._connector_for("github.com"))
    assert connector._resolver.__class__.__name__ != "_Resolver"

    class _Private:
        async def resolve(self, host, port=0, family=0):
            raise OSError("URL resolves to a non-public address: 10.0.0.5")

    monkeypatch.setattr(fetch_mod, "_SafeResolver", _Private)
    with pytest.raises(OSError, match="non-public"):
        asyncio.run(fetch_mod._connector_for("intranet"))
