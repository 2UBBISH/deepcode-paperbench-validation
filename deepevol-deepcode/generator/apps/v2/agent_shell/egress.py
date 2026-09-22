"""Credentialed egress: provider HTTP calls the Agent makes without holding keys.

The Agent sends a normal provider request — same method, path, query, headers
and body it always sent — but addressed to the shell
(``/egress/<provider>?u=<upstream url>``) and with **credential handles**
(``shell:<provider>`` / ``shell:<provider>#<field>``) wherever it would have
put the real key.  The shell:

1. checks the upstream host is one the provider rule allows,
2. asks the :class:`KeyPoolPort` for a credential (the pool's rotation,
   soft/hard daily caps and cooldowns run here, next to the key files),
3. substitutes the handles in headers, query and body, forwards the request,
4. classifies the answer (429 / 401 / 403) back into the pool,
5. books the call in the ledger.

Nothing provider-specific is hard-coded beyond the host allowlist: the
handle-substitution contract lets every provider's own placement (bearer,
``x-api-key`` header, ``api_key`` query parameter, JSON body field) go through
unchanged.  Requests without a handle are forwarded as-is (public endpoints,
page fetches) and still metered.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlsplit

HANDLE_PREFIX = "shell:"
_HANDLE = re.compile(r"shell:([a-z][a-z0-9_]*)(?:#([a-z][a-z0-9_]*))?")

# Bodies larger than this are forwarded without substitution (a handle inside
# a multi-megabyte upload is not a real use case; scanning it is).
MAX_SUBSTITUTED_BODY_BYTES = 2 * 1024 * 1024
MAX_UPSTREAM_RESPONSE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProviderRule:
    """Where a provider lives and which upstream hosts the shell will reach for it."""

    name: str
    hosts: tuple[str, ...]
    # Whether requests for this provider may carry no handle at all (public
    # endpoints of the same API).  Requests with a handle always need a key.
    allow_anonymous: bool = True

    def allows(self, host: str) -> bool:
        host = host.lower()
        return any(host == h or host.endswith("." + h) for h in self.hosts)


DEFAULT_PROVIDER_RULES: tuple[ProviderRule, ...] = (
    ProviderRule("deepxiv", ("deepxiv.org", "api.deepxiv.org", "deepxiv.cn")),
    ProviderRule("openalex", ("api.openalex.org",)),
    ProviderRule("semantic_scholar", ("api.semanticscholar.org",)),
    ProviderRule("serper", ("google.serper.dev",), allow_anonymous=False),
    ProviderRule("tavily", ("api.tavily.com",), allow_anonymous=False),
    ProviderRule("crossref", ("api.crossref.org",)),
    ProviderRule("arxiv", ("export.arxiv.org", "arxiv.org")),
    ProviderRule("orcid", ("pub.orcid.org",)),
)


@dataclass(frozen=True, slots=True)
class Credential:
    """One acquired key: its pool id and the handle → secret substitutions."""

    key_id: str
    substitutions: Mapping[str, str]


class KeyPoolPort(Protocol):
    """What the shell needs from a key pool; implemented next to the pool files."""

    def providers(self) -> Mapping[str, bool]:
        """provider name -> has at least one configured key."""
        ...

    def acquire(self, provider: str) -> Credential | None: ...

    def record(self, provider: str, key_id: str, *, rate_limited: bool, auth_error: bool) -> None: ...


class EgressError(Exception):
    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def find_handles(*texts: str) -> set[tuple[str, str]]:
    """All ``(provider, field)`` handles mentioned in the given strings."""

    found: set[tuple[str, str]] = set()
    for text in texts:
        for match in _HANDLE.finditer(text or ""):
            found.add((match.group(1), match.group(2) or ""))
    return found


def handle(provider: str, field_name: str = "") -> str:
    return f"{HANDLE_PREFIX}{provider}#{field_name}" if field_name else f"{HANDLE_PREFIX}{provider}"


def substitute(text: str, substitutions: Mapping[str, str]) -> str:
    # Longest handles first so ``shell:openalex#email`` wins over ``shell:openalex``.
    for key in sorted(substitutions, key=len, reverse=True):
        text = text.replace(key, substitutions[key])
    return text


def substitute_query(query: str, substitutions: Mapping[str, str]) -> str:
    """Query strings arrive percent-encoded (``#`` → ``%23``); match both
    spellings and percent-encode the secret so the URL stays valid."""

    for key in sorted(substitutions, key=len, reverse=True):
        secret = quote(substitutions[key], safe="")
        for form in sorted({quote(key, safe=""), quote(key, safe=":"), key.replace("#", "%23"), key}, key=len, reverse=True):
            query = query.replace(form, secret)
    return query


@dataclass
class EgressPlan:
    provider: ProviderRule
    upstream: str
    method: str
    headers: dict[str, str]
    query: str
    body: bytes
    credential: Credential | None = None
    handles: set[tuple[str, str]] = field(default_factory=set)


class EgressGateway:
    """Provider allowlist + credential substitution; transport-agnostic."""

    def __init__(self, pools: KeyPoolPort | None, *, rules: tuple[ProviderRule, ...] = DEFAULT_PROVIDER_RULES) -> None:
        self._pools = pools
        self._rules = {rule.name: rule for rule in rules}

    @property
    def rules(self) -> Mapping[str, ProviderRule]:
        return self._rules

    def providers(self) -> dict[str, dict[str, Any]]:
        configured = dict(self._pools.providers()) if self._pools is not None else {}
        return {
            name: {"hosts": list(rule.hosts), "configured": bool(configured.get(name, False))}
            for name, rule in self._rules.items()
        }

    def plan(self, provider_name: str, *, upstream: str, method: str, headers: Mapping[str, str], body: bytes) -> EgressPlan:
        rule = self._rules.get(provider_name)
        if rule is None:
            raise EgressError("SHELL_EGRESS_PROVIDER_UNKNOWN", f"unknown egress provider {provider_name!r}", status=404)
        parts = urlsplit(upstream)
        if parts.scheme != "https" or not parts.hostname:
            raise EgressError("SHELL_EGRESS_UPSTREAM_INVALID", "upstream must be an https URL")
        if not rule.allows(parts.hostname):
            raise EgressError("SHELL_EGRESS_HOST_REFUSED", f"{parts.hostname} is not a {rule.name} host", status=403)
        clean_headers = {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}
        body_text = body.decode("utf-8", errors="replace") if len(body) <= MAX_SUBSTITUTED_BODY_BYTES else ""
        handles = find_handles(unquote(parts.query), body_text, *clean_headers.values())
        foreign = {p for p, _ in handles if p != rule.name}
        if foreign:
            raise EgressError("SHELL_EGRESS_HANDLE_MISMATCH", f"handles for {sorted(foreign)} on a {rule.name} request", status=403)
        plan = EgressPlan(provider=rule, upstream=upstream, method=method.upper(), headers=clean_headers,
                          query=parts.query, body=body, handles=handles)
        if handles:
            if self._pools is None:
                raise EgressError("SHELL_EGRESS_POOL_UNAVAILABLE", "this shell has no key pools bound", status=503)
            credential = self._pools.acquire(rule.name)
            if credential is None:
                raise EgressError("SHELL_EGRESS_KEYS_EXHAUSTED", f"{rule.name} key pool is exhausted or cooling", status=503)
            plan.credential = credential
            missing = {handle(p, f) for p, f in handles} - set(credential.substitutions)
            if missing:
                raise EgressError("SHELL_EGRESS_HANDLE_UNKNOWN", f"no substitution for {sorted(missing)}", status=400)
            plan.headers = {k: substitute(v, credential.substitutions) for k, v in clean_headers.items()}
            plan.query = substitute_query(parts.query, credential.substitutions)
            if body_text:
                plan.body = substitute(body_text, credential.substitutions).encode("utf-8")
            plan.upstream = parts._replace(query=plan.query).geturl()
        elif not rule.allow_anonymous:
            raise EgressError("SHELL_EGRESS_HANDLE_REQUIRED", f"{rule.name} requests need a credential handle", status=400)
        return plan

    def settle(self, plan: EgressPlan, status_code: int) -> None:
        if plan.credential is None or self._pools is None:
            return
        rate_limited = status_code == 429
        auth_error = status_code in (401, 403)
        if rate_limited or auth_error:
            self._pools.record(plan.provider.name, plan.credential.key_id, rate_limited=rate_limited, auth_error=auth_error)


_HOP_BY_HOP = frozenset({
    "host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length", "authorization-shell",
})


__all__ = [
    "DEFAULT_PROVIDER_RULES",
    "Credential",
    "EgressError",
    "EgressGateway",
    "EgressPlan",
    "HANDLE_PREFIX",
    "KeyPoolPort",
    "ProviderRule",
    "find_handles",
    "handle",
    "substitute",
    "substitute_query",
]
