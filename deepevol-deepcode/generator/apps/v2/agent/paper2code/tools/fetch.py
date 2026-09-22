"""``fetch`` server: one tool, ``fetch(url, max_length, start_index, raw)``.

Same argument surface as ``mcp-server-fetch``. The transport reuses the
engine's safe-download policy (public HTTP(S) only, DNS-rebinding guard,
five validated redirects) and adds the two guards the validation repo
found necessary: the paper's denylist, and a per-run ledger that stops the
model re-fetching the same dead URL more than twice. HTML is converted to
Markdown with the engine's converter; bodies are capped at 100 KiB.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from urllib.parse import urlparse

import aiohttp
from loguru import logger

from apps.v2.agent.paper2code.tools.registry import (
    MAX_SAME_URL_FETCHES,
    FunctionTool,
    ToolContext,
    denylist_hit,
    denylist_refusal,
    exposed_name,
)
from apps.v2.agent_engine.paper2code.seams.agent_runtime import Tool
from apps.v2.agent_engine.paper2code.tools.document_conversion import _html_to_markdown
from apps.v2.agent_engine.paper2code.tools.pdf_downloader import (
    _request_with_safe_redirects,
    _SafeResolver,
    _validate_public_url,
)

BODY_CAP_BYTES = 100 * 1024
DEFAULT_MAX_LENGTH = 5000
REQUEST_TIMEOUT_S = 30.0
_USER_AGENT = "DeepEvol-paper2code/1.0 (+reference mining; respects robots via denylist)"


# RFC 2544 benchmark range; never a real destination. Proxy clients in "fake-IP"
# mode (Clash, Surge, ...) answer DNS from it and route the connection through
# the proxy, so the engine's rebinding guard would refuse every host on such a
# developer machine. Falling back to the system resolver there keeps the URL
# validation and the denylist; real private addresses are still refused.
FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


def _fake_ip_in(message: str) -> bool:
    for token in message.replace(":", " ").split():
        try:
            if ipaddress.ip_address(token.strip("[]")) in FAKE_IP_NETWORK:
                return True
        except ValueError:
            continue
    return False


async def _connector_for(host: str) -> aiohttp.TCPConnector:
    try:
        await _SafeResolver().resolve(host, 443)
    except OSError as exc:
        if _fake_ip_in(str(exc)):
            logger.warning("fetch: {} resolves into the fake-IP range (proxy DNS); using the system resolver", host)
            return aiohttp.TCPConnector()
        raise
    return aiohttp.TCPConnector(resolver=_SafeResolver())


async def _download(url: str) -> tuple[str, bytes]:
    """``(content_type, body)`` for a public URL; raises on policy or HTTP errors."""
    _validate_public_url(url)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    connector = await _connector_for(urlparse(url).hostname or "")
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers={"User-Agent": _USER_AGENT}
    ) as session:
        response = await _request_with_safe_redirects(session, "GET", url)
        async with response:
            if response.status >= 400:
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"HTTP {response.status}",
                )
            content_type = response.headers.get("Content-Type", "")
            body = await response.content.read(BODY_CAP_BYTES + 1)
            return content_type, body[:BODY_CAP_BYTES]


def _to_text(content_type: str, body: bytes, *, raw: bool) -> str:
    text = body.decode("utf-8", errors="replace")
    is_html = "text/html" in content_type.lower() or text.lstrip()[:200].lower().startswith(
        ("<!doctype html", "<html")
    )
    if is_html and not raw:
        return _html_to_markdown(text)
    return text


def tools(ctx: ToolContext) -> Sequence[Tool]:
    async def fetch(
        url: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        start_index: int = 0,
        raw: bool = False,
    ) -> str:
        hit = denylist_hit(ctx.denylist, [url])
        if hit is not None:
            return denylist_refusal(url)
        key = ("fetch", url)
        seen = ctx.fetch_ledger.get(key, 0)
        if seen >= MAX_SAME_URL_FETCHES:
            return (
                f"ALREADY FETCHED: '{url}' has been retrieved {seen} times in this "
                "session and returned no further information. Re-fetching it will "
                "not help — the page is unavailable or does not contain what you "
                "need. Move on to a different source, or proceed with the "
                "information you already have."
            )
        try:
            content_type, body = await _download(url)
        except Exception as exc:
            ctx.fetch_ledger[key] = seen + 1
            return f"Error: failed to fetch {url}: {exc}"
        ctx.fetch_ledger[key] = seen + 1
        text = _to_text(content_type, body, raw=raw)
        start = max(int(start_index), 0)
        limit = max(int(max_length), 1)
        chunk = text[start : start + limit]
        if not chunk:
            return f"<error>No more content available at start_index={start}.</error>"
        remaining = len(text) - (start + limit)
        if remaining > 0:
            chunk += (
                f"\n\n<error>Content truncated. Call the fetch tool with a start_index of "
                f"{start + limit} to get more content.</error>"
            )
        return f"Contents of {url}:\n{chunk}"

    return [
        FunctionTool(
            name=exposed_name("fetch", "fetch"),
            description=(
                "Fetch a URL from the internet and extract its contents as markdown. "
                "Use start_index to page through long pages. Resources on this "
                "task's blacklist are refused."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum number of characters to return",
                        "default": DEFAULT_MAX_LENGTH,
                    },
                    "start_index": {
                        "type": "integer",
                        "description": "Start output at this character index",
                        "default": 0,
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "Return raw content instead of markdown",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
            fn=fetch,
            read_only=True,
            timeout_s=REQUEST_TIMEOUT_S + 10,
        )
    ]


__all__ = ["BODY_CAP_BYTES", "DEFAULT_MAX_LENGTH", "tools"]
