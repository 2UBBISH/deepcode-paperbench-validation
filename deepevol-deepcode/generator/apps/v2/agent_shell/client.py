"""Agent-side stubs: everything an Agent holds is a URL and a token.

:class:`ShellRuntime` speaks the relay ``Runtime`` protocol over the shell's
HTTP surface, so vendored code written against ``RemoteRuntime`` (rsa's
``RemoteDockerBackend``, the experiment flow's control connection) runs
unchanged while never learning where the machine is or how to log in.

:class:`ShellWorkspaceExecutor` is the synchronous counterpart for the chat
graph's ``execute`` tool.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import posixpath
import shlex
import tarfile
import tempfile
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import Any

import httpx

from .protocol import (
    ExecEvent,
    ExecResult,
    FileStat,
    JobStatus,
    Placement,
    ShellError,
    ShellErrorCode,
    decode_exec_event,
    decode_exec_result,
    decode_file_stat,
    decode_job_status,
)

# Directory transfers are staged through a tarball on the placement itself;
# the default excludes mirror the relay's (VCS metadata, caches, virtualenvs).
DEFAULT_SYNC_EXCLUDES: tuple[str, ...] = (".git", "__pycache__", ".venv", "node_modules", ".mypy_cache")

_LONG = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)


class _Http:
    """Thin error-mapping wrapper.  Sync ``httpx`` is used from a thread for
    the workspace executor; async for the runtime."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    def headers(self) -> dict[str, str]:
        return dict(self._headers)

    @staticmethod
    def raise_for(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        code, message = ShellErrorCode.TARGET_FAILED, response.text[:300]
        try:
            error = response.json().get("error") or {}
            code = ShellErrorCode(str(error.get("code") or code))
            message = str(error.get("message") or message)
        except (ValueError, TypeError):
            pass
        raise ShellError(code, message, status=response.status_code)


class ShellRuntime:
    """A relay ``Runtime`` whose machine is whatever the shell routes to."""

    def __init__(self, base_url: str, token: str, *, name: str = "agent-shell") -> None:
        self._http = _Http(base_url, token)
        self.name = name
        self._client: httpx.AsyncClient | None = None
        self._cwd = ""

    def __repr__(self) -> str:  # never the token
        return f"ShellRuntime(base_url={self._http.base_url!r}, cwd={self._cwd!r})"

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._http.base_url, headers=self._http.headers(), timeout=_LONG
            )

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> "ShellRuntime":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def wait_ready(self, *, timeout: float = 300.0) -> None:
        """Ask the shell to bring its placement up.  The relay's semantics
        carry through: an authentication failure raises at once, everything
        else is polled on the shell side until ``timeout``."""

        await self.start()
        await self._post_json("/placement/ready", {"timeout": float(timeout)})

    async def placement(self) -> Placement:
        return Placement.from_dict(await self._get_json("/placement"))

    # ------------------------------------------------------------ transport
    def _client_or_raise(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ShellRuntime is not started")
        return self._client

    async def _get_json(self, path: str, **params: Any) -> Any:
        response = await self._client_or_raise().get(path, params={k: v for k, v in params.items() if v is not None})
        self._http.raise_for(response)
        return response.json()

    async def _post_json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = await self._client_or_raise().post(path, json=payload or {})
        self._http.raise_for(response)
        return response.json()

    # ------------------------------------------------------------ execution
    @property
    def cwd(self) -> str:
        return self._cwd

    async def chdir(self, path: str) -> None:
        """Like a shell ``cd``: sticks across calls.  A relative path resolves
        against the placement's current directory, which is asked for once."""

        if posixpath.isabs(path):
            self._cwd = posixpath.normpath(path)
            return
        base = self._cwd
        if not base:
            probe = await self.exec("pwd", timeout=30)
            base = probe.stdout.strip().splitlines()[-1] if probe.exit_status == 0 and probe.stdout.strip() else ""
        self._cwd = posixpath.normpath(posixpath.join(base, path)) if base else path

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_event: Any = None,
        durable: bool | None = None,
    ) -> ExecResult:
        payload = {
            "command": command,
            "cwd": cwd if cwd is not None else (self._cwd or None),
            "env": dict(env or {}),
            "timeout": timeout,
            "durable": durable,
        }
        result = decode_exec_result(await self._post_json("/exec", payload))
        if on_event is not None:
            if result.stdout:
                on_event(ExecEvent(kind="stdout", data=result.stdout, offset=0))
            if result.stderr:
                on_event(ExecEvent(kind="stderr", data=result.stderr, offset=0))
        return result

    async def check_exec(self, command: str, **kwargs: Any) -> ExecResult:
        result = await self.exec(command, **kwargs)
        if result.exit_status != 0:
            raise RuntimeError(f"command failed ({result.exit_status}): {command}\n{result.output[-2000:]}")
        return result

    async def spawn(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        payload = {"command": command, "cwd": cwd if cwd is not None else (self._cwd or None),
                   "env": dict(env or {}), "job_id": job_id}
        return decode_job_status(await self._post_json("/jobs", payload))

    async def stream(
        self,
        job_id: str,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        follow: bool = True,
    ) -> AsyncIterator[ExecEvent]:
        client = self._client_or_raise()
        params = {"stdout_offset": stdout_offset, "stderr_offset": stderr_offset, "follow": "1" if follow else "0"}
        async with client.stream("GET", f"/jobs/{job_id}/stream", params=params) as response:
            if response.status_code >= 400:
                await response.aread()
                self._http.raise_for(response)
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                if "error" in raw:
                    error = raw["error"]
                    raise ShellError(error.get("code", ShellErrorCode.TARGET_FAILED), error.get("message", ""))
                yield decode_exec_event(raw)

    async def job_status(self, job_id: str) -> JobStatus:
        return decode_job_status(await self._get_json(f"/jobs/{job_id}"))

    async def wait(self, job_id: str, *, timeout: float | None = None) -> ExecResult:
        return decode_exec_result(await self._post_json(f"/jobs/{job_id}/wait", {"timeout": timeout}))

    async def kill(self, job_id: str, *, sig: str = "TERM") -> bool:
        return bool((await self._post_json(f"/jobs/{job_id}/kill", {"sig": sig})).get("killed"))

    # ----------------------------------------------------------- filesystem
    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        response = await self._client_or_raise().get(
            "/fs/read", params={"path": path, **({"max_bytes": max_bytes} if max_bytes is not None else {})}
        )
        self._http.raise_for(response)
        return response.content

    async def read_file(self, path: str, *, encoding: str = "utf-8", max_bytes: int | None = None) -> str:
        return (await self.read_bytes(path, max_bytes=max_bytes)).decode(encoding, errors="replace")

    async def write_bytes(self, path: str, content: bytes, *, parents: bool = True) -> None:
        response = await self._client_or_raise().post(
            "/fs/write", params={"path": path, "parents": "1" if parents else "0"}, content=content,
            headers={"Content-Type": "application/octet-stream"},
        )
        self._http.raise_for(response)

    async def write_file(self, path: str, content: str, *, encoding: str = "utf-8", parents: bool = True) -> None:
        await self.write_bytes(path, content.encode(encoding), parents=parents)

    async def append_file(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        await self._post_json("/fs/append", {"path": path, "content": content, "encoding": encoding})

    async def ls(self, path: str = ".") -> list[FileStat]:
        return [decode_file_stat(e) for e in (await self._post_json("/fs/ls", {"path": path}))["entries"]]

    async def glob(self, pattern: str) -> list[str]:
        return list((await self._post_json("/fs/glob", {"pattern": pattern}))["matches"])

    async def stat(self, path: str) -> FileStat:
        return decode_file_stat(await self._post_json("/fs/stat", {"path": path}))

    async def exists(self, path: str) -> bool:
        return bool((await self._post_json("/fs/exists", {"path": path}))["exists"])

    async def mkdir(self, path: str, *, parents: bool = True) -> None:
        await self._post_json("/fs/mkdir", {"path": path, "parents": parents})

    async def rm(self, path: str, *, recursive: bool = False, missing_ok: bool = True) -> None:
        await self._post_json("/fs/rm", {"path": path, "recursive": recursive, "missing_ok": missing_ok})

    async def move(self, src: str, dst: str) -> None:
        await self._post_json("/fs/move", {"src": src, "dst": dst})

    # ------------------------------------------------------------ transfers
    async def upload(self, local_path: str, remote_path: str, *, parents: bool = True) -> None:
        data = await asyncio.to_thread(Path(local_path).read_bytes)
        await self.write_bytes(remote_path, data, parents=parents)

    async def download(self, remote_path: str, local_path: str, *, parents: bool = True) -> None:
        data = await self.read_bytes(remote_path)
        target = Path(local_path)
        if parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def sync_up(self, local_dir: str, remote_dir: str, *, excludes: Iterable[str] = DEFAULT_SYNC_EXCLUDES) -> Any:
        """Stage a local tree as one tarball, unpack it on the placement."""

        excluded = set(excludes or ())
        buffer = io.BytesIO()
        root = Path(local_dir)

        def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            parts = Path(info.name).parts
            return None if any(part in excluded for part in parts) else info

        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(root, arcname=".", filter=_filter)
        staging = f"{remote_dir.rstrip('/')}/.deepevol-sync-up.tgz"
        await self.mkdir(remote_dir, parents=True)
        await self.write_bytes(staging, buffer.getvalue(), parents=True)
        await self.check_exec(
            f"tar -xzf {shlex.quote(staging)} -C {shlex.quote(remote_dir)} && rm -f {shlex.quote(staging)}"
        )
        return _SyncReport(bytes_transferred=buffer.tell(), direction="up")

    async def sync_down(self, remote_dir: str, local_dir: str, *, excludes: Iterable[str] = DEFAULT_SYNC_EXCLUDES) -> Any:
        """Pack the placement's tree into one tarball, unpack it locally."""

        exclude_flags = " ".join(f"--exclude={shlex.quote(name)}" for name in (excludes or ()))
        staging = f"/tmp/.deepevol-sync-down-{os.getpid()}-{int(time.time() * 1000)}.tgz"
        await self.check_exec(
            f"tar -czf {shlex.quote(staging)} {exclude_flags} -C {shlex.quote(remote_dir)} ."
        )
        try:
            data = await self.read_bytes(staging)
        finally:
            await self.exec(f"rm -f {shlex.quote(staging)}")
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)

        def _extract() -> None:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                _safe_extract(archive, target)

        await asyncio.to_thread(_extract)
        return _SyncReport(bytes_transferred=len(data), direction="down")


class _SyncReport:
    def __init__(self, *, bytes_transferred: int, direction: str) -> None:
        self.bytes_transferred = int(bytes_transferred)
        self.direction = direction

    @property
    def summary(self) -> str:
        return f"sync_{self.direction}: {self.bytes_transferred} bytes"


def _safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    resolved_root = target.resolve()
    for member in archive.getmembers():
        destination = (resolved_root / member.name).resolve()
        if destination != resolved_root and resolved_root not in destination.parents:
            raise ValueError(f"archive member escapes the target directory: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"archive links are not accepted: {member.name}")
    archive.extractall(resolved_root)


class ShellWorkspaceExecutor:
    """Synchronous ``execute`` for the chat graph's tool: one shell call."""

    def __init__(self, base_url: str, token: str) -> None:
        self._http = _Http(base_url, token)

    def __repr__(self) -> str:
        return f"ShellWorkspaceExecutor(base_url={self._http.base_url!r})"

    def placement(self) -> Placement:
        with httpx.Client(base_url=self._http.base_url, headers=self._http.headers(), timeout=10.0) as client:
            response = client.get("/placement")
            self._http.raise_for(response)
            return Placement.from_dict(response.json())

    def execute(self, command: str, *, timeout: float | None = None, env: dict[str, str] | None = None,
                cwd: str | None = None, command_id: str | None = None) -> ExecResult:
        with httpx.Client(base_url=self._http.base_url, headers=self._http.headers(), timeout=_LONG) as client:
            response = client.post(
                "/exec", json={"command": command, "timeout": timeout, "env": dict(env or {}), "cwd": cwd,
                               "command_id": command_id}
            )
            self._http.raise_for(response)
            return decode_exec_result(response.json())


def scratch_dir(prefix: str = "deepevol-shell-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


# ------------------------------------------------------------------ egress
#
# The Agent side of credentialed egress.  ``ShellEgress`` is what a process
# holds (URL + token + provider hosts); the transports below make any httpx
# client route provider traffic through the shell transparently, so the
# retrieval code keeps its own request shapes and only swaps the key for a
# handle (``shell:<provider>``).

_EGRESS_ENV_URL = "DEEPEVOL_AGENT_SHELL_URL"
_EGRESS_ENV_TOKEN = "DEEPEVOL_AGENT_SHELL_TOKEN"


class ShellEgress:
    def __init__(self, base_url: str, token: str, *, hosts: Mapping[str, tuple[str, ...]] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._hosts: dict[str, tuple[str, ...]] = dict(hosts) if hosts else {}

    def __repr__(self) -> str:  # never the token
        return f"ShellEgress(base_url={self.base_url!r}, providers={sorted(self._hosts)})"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ShellEgress | None":
        env = os.environ if environ is None else environ
        url = str(env.get(_EGRESS_ENV_URL) or "").strip()
        token = str(env.get(_EGRESS_ENV_TOKEN) or "").strip()
        if not url or not token:
            return None
        return cls(url, token)

    def env(self) -> dict[str, str]:
        """What to put in a child process's environment."""
        return {_EGRESS_ENV_URL: self.base_url, _EGRESS_ENV_TOKEN: self._token}

    def providers(self) -> dict[str, dict[str, Any]]:
        with httpx.Client(base_url=self.base_url, timeout=10.0, trust_env=False) as client:
            response = client.get("/egress/providers", headers={"X-Shell-Token": self._token})
            _Http.raise_for(response)
            providers = response.json()["providers"]
        self._hosts = {name: tuple(info.get("hosts") or ()) for name, info in providers.items()}
        return providers

    def configured(self, provider: str) -> bool:
        return bool(self.providers().get(provider, {}).get("configured"))

    def provider_for(self, host: str) -> str | None:
        if not self._hosts:
            try:
                self.providers()
            except Exception:
                return None
        host = (host or "").lower()
        for name, hosts in self._hosts.items():
            if any(host == h or host.endswith("." + h) for h in hosts):
                return name
        return None

    def rewrite(self, request: httpx.Request) -> httpx.Request | None:
        """The same request addressed to the shell, or None when the host is
        not a known provider (the request then goes out directly)."""
        provider = self.provider_for(request.url.host)
        if provider is None:
            return None
        upstream = str(request.url)
        headers = dict(request.headers)
        headers.pop("host", None)
        headers["X-Shell-Token"] = self._token
        timeout = request.extensions.get("timeout") if isinstance(request.extensions, Mapping) else None
        if isinstance(timeout, Mapping) and timeout.get("read"):
            headers["X-Shell-Timeout"] = str(timeout["read"])
        return httpx.Request(
            request.method,
            httpx.URL(f"{self.base_url}/egress/{provider}", params={"u": upstream}),
            headers=headers,
            content=request.content,
            extensions=dict(request.extensions),
        )


class ShellEgressAsyncTransport(httpx.AsyncBaseTransport):
    """Wrap any async transport; provider hosts are re-addressed to the shell."""

    def __init__(self, egress: ShellEgress, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._egress = egress
        self._inner = inner or httpx.AsyncHTTPTransport(retries=0, trust_env=False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        rewritten = self._egress.rewrite(request)
        return await self._inner.handle_async_request(rewritten or request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class ShellEgressTransport(httpx.BaseTransport):
    """Sync twin of :class:`ShellEgressAsyncTransport`."""

    def __init__(self, egress: ShellEgress, inner: httpx.BaseTransport | None = None) -> None:
        self._egress = egress
        self._inner = inner or httpx.HTTPTransport(retries=0, trust_env=False)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        rewritten = self._egress.rewrite(request)
        return self._inner.handle_request(rewritten or request)

    def close(self) -> None:
        self._inner.close()


__all__ = [
    "DEFAULT_SYNC_EXCLUDES",
    "ShellEgress",
    "ShellEgressAsyncTransport",
    "ShellEgressTransport",
    "ShellRuntime",
    "ShellWorkspaceExecutor",
]
