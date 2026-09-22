"""Execution placements the shell can route to.

An :class:`ExecutionTarget` is exactly the relay ``Runtime`` protocol plus a
credential-free :class:`Placement` description.  Two implementations:

* :class:`RelayTarget` — a rented machine (or, in tests, a local relay
  runtime).  The ``RemoteRuntime`` inside it is the only object in the process
  that knows the machine's password; it is constructed here, from the lease
  handle, and never handed to Agent code.
* :class:`LocalDockerTarget` — the run's registered per-run Docker sandbox
  (the chat graph's ``execute`` tool).  Commands go through the sandbox
  context; file operations act on the attempt workspace the sandbox mounts,
  confined to that root.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import shutil
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .protocol import (
    TIMEOUT_EXIT_STATUS,
    ExecEvent,
    ExecResult,
    FileStat,
    JobStatus,
    Placement,
    PlacementKind,
)


@runtime_checkable
class ExecutionTarget(Protocol):
    """The relay ``Runtime`` surface plus where it runs."""

    @property
    def placement(self) -> Placement: ...

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def wait_ready(self, *, timeout: float = 300.0) -> None: ...
    async def exec(self, command: str, *, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout: float | None = None, on_event: Any = None,
                   durable: bool | None = None) -> ExecResult: ...
    async def spawn(self, command: str, *, cwd: str | None = None, env: dict[str, str] | None = None,
                    job_id: str | None = None) -> JobStatus: ...
    def stream(self, job_id: str, *, stdout_offset: int = 0, stderr_offset: int = 0,
               follow: bool = True) -> AsyncIterator[ExecEvent]: ...
    async def job_status(self, job_id: str) -> JobStatus: ...
    async def wait(self, job_id: str, *, timeout: float | None = None) -> ExecResult: ...
    async def kill(self, job_id: str, *, sig: str = "TERM") -> bool: ...
    async def read_file(self, path: str, *, encoding: str = "utf-8", max_bytes: int | None = None) -> str: ...
    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes: ...
    async def write_file(self, path: str, content: str, *, encoding: str = "utf-8", parents: bool = True) -> None: ...
    async def write_bytes(self, path: str, content: bytes, *, parents: bool = True) -> None: ...
    async def append_file(self, path: str, content: str, *, encoding: str = "utf-8") -> None: ...
    async def ls(self, path: str = ".") -> list[FileStat]: ...
    async def glob(self, pattern: str) -> list[str]: ...
    async def stat(self, path: str) -> FileStat: ...
    async def exists(self, path: str) -> bool: ...
    async def mkdir(self, path: str, *, parents: bool = True) -> None: ...
    async def rm(self, path: str, *, recursive: bool = False, missing_ok: bool = True) -> None: ...
    async def move(self, src: str, dst: str) -> None: ...
    async def upload(self, local_path: str, remote_path: str, *, parents: bool = True) -> None: ...
    async def download(self, remote_path: str, local_path: str, *, parents: bool = True) -> None: ...


class RelayTarget:
    """A relay ``Runtime`` (remote machine or local) as a shell placement.

    ``runtime_factory`` is called lazily on :meth:`start` so that constructing
    the target has no side effects (the relay opens its SSH session in
    ``start``); the factory closes over whatever credential it needs and that
    closure never leaves the shell.
    """

    def __init__(
        self,
        runtime_factory: Callable[[], Any],
        *,
        placement: Placement,
    ) -> None:
        self._factory = runtime_factory
        self._runtime: Any = None
        self._placement = replace(placement, kind=PlacementKind.REMOTE_LEASE)

    @property
    def placement(self) -> Placement:
        return replace(self._placement, ready=self._runtime is not None)

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError("relay target is not started")
        return self._runtime

    async def start(self) -> None:
        if self._runtime is None:
            runtime = self._factory()
            await runtime.start()
            self._runtime = runtime

    async def close(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            await runtime.close()

    async def wait_ready(self, *, timeout: float = 300.0) -> None:
        await self.start()
        await self._runtime.wait_ready(timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        # Every other Runtime method is a straight pass-through.  ``__getattr__``
        # only fires for names not defined on this class, so the lifecycle
        # methods above keep their semantics.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.runtime, name)


class LocalDockerTarget:
    """The run's registered Docker sandbox as a shell placement.

    ``execute`` is the sandbox context's synchronous entry point (``docker exec``
    into the per-run container); it is run in a worker thread so the shell's
    event loop keeps serving.  Durable jobs are not part of the sandbox contract
    and are refused explicitly rather than emulated.
    """

    def __init__(
        self,
        *,
        workspace_root: str | os.PathLike[str],
        context_resolver: Callable[[], Any],
        label: str = "sandbox",
        max_output_bytes: int = 100_000,
        default_timeout: float = 300.0,
        max_timeout: float = 3_600.0,
    ) -> None:
        root = Path(workspace_root)
        if not root.is_absolute():
            raise ValueError("LocalDockerTarget requires an absolute workspace root")
        self._root = root.resolve(strict=False)
        self._resolve_context = context_resolver
        self._label = label
        self._max_output_bytes = int(max_output_bytes)
        self._default_timeout = float(default_timeout)
        self._max_timeout = float(max_timeout)
        self._cwd = "/workspace"

    # ----------------------------------------------------------- placement
    @property
    def placement(self) -> Placement:
        context = self._resolve_context()
        ready = bool(context is not None and getattr(context, "is_running", False))
        return Placement(kind=PlacementKind.LOCAL_DOCKER, generation=0, label=self._label, ready=ready)

    @property
    def cwd(self) -> str:
        return self._cwd

    # The sandbox emits per-command lifecycle events keyed by the caller's
    # operation id; the shell forwards ``command_id`` only to targets that
    # advertise this.
    accepts_command_id = True

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def wait_ready(self, *, timeout: float = 300.0) -> None:
        deadline = time.monotonic() + float(timeout)
        while not self.placement.ready:
            if time.monotonic() >= deadline:
                raise TimeoutError("the run's Docker sandbox did not become ready")
            await asyncio.sleep(0.5)

    # ----------------------------------------------------------- execution
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_event: Any = None,
        durable: bool | None = None,
        command_id: str | None = None,
    ) -> ExecResult:
        if durable:
            raise NotImplementedError("the Docker sandbox placement has no durable jobs")
        context = self._resolve_context()
        if context is None or not getattr(context, "is_running", False):
            raise RuntimeError("the run's Docker sandbox is not running")
        effective = self._default_timeout if timeout is None else float(timeout)
        if effective < 1 or effective > self._max_timeout:
            raise ValueError(f"command timeout must be between 1 and {int(self._max_timeout)} seconds")
        shell_command = command
        if cwd:
            shell_command = f"cd {_quote(cwd)} && {command}"
        started = time.monotonic()
        response = await asyncio.to_thread(
            context.execute,
            shell_command,
            timeout=int(effective),
            max_output_bytes=self._max_output_bytes,
            env=dict(env or {}),
            command_id=command_id,
        )
        exit_status = int(getattr(response, "exit_code", 1))
        output = str(getattr(response, "output", ""))
        truncated = bool(getattr(response, "truncated", False))
        result = ExecResult(
            exit_status=exit_status,
            stdout=output,
            stderr="",
            duration_ms=int((time.monotonic() - started) * 1000),
            command=command,
            cwd=cwd or self._cwd,
            timed_out=exit_status == TIMEOUT_EXIT_STATUS,
            stdout_truncated=truncated,
        )
        if on_event is not None and output:
            on_event(ExecEvent(kind="stdout", data=output, offset=0))
        return result

    async def spawn(self, command: str, *, cwd: str | None = None, env: dict[str, str] | None = None,
                    job_id: str | None = None) -> JobStatus:
        raise NotImplementedError("the Docker sandbox placement has no durable jobs")

    async def stream(self, job_id: str, *, stdout_offset: int = 0, stderr_offset: int = 0,
                     follow: bool = True) -> AsyncIterator[ExecEvent]:
        raise NotImplementedError("the Docker sandbox placement has no durable jobs")
        yield  # pragma: no cover - makes this an async generator for the protocol

    async def job_status(self, job_id: str) -> JobStatus:
        raise NotImplementedError("the Docker sandbox placement has no durable jobs")

    async def wait(self, job_id: str, *, timeout: float | None = None) -> ExecResult:
        raise NotImplementedError("the Docker sandbox placement has no durable jobs")

    async def kill(self, job_id: str, *, sig: str = "TERM") -> bool:
        raise NotImplementedError("the Docker sandbox placement has no durable jobs")

    # ---------------------------------------------------------- filesystem
    def _host(self, path: str, *, must_exist: bool = False) -> Path:
        """Map a sandbox path (``/workspace/x`` or relative) onto the host root.

        Confinement is by resolved prefix: symlinks that escape the workspace
        are refused, exactly like the strict workspace backend.
        """

        raw = str(path or ".")
        if raw.startswith("/workspace"):
            raw = raw[len("/workspace"):].lstrip("/") or "."
        elif raw.startswith("/"):
            raise ValueError("paths outside /workspace are not reachable in the sandbox placement")
        candidate = (self._root / raw).resolve(strict=False)
        if candidate != self._root and self._root not in candidate.parents:
            raise ValueError("path escapes the run workspace")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(path)
        return candidate

    def _display(self, host_path: Path) -> str:
        rel = host_path.relative_to(self._root).as_posix()
        return "/workspace" if rel == "." else f"/workspace/{rel}"

    def _stat(self, host_path: Path) -> FileStat:
        info = host_path.lstat()
        return FileStat(
            path=self._display(host_path),
            name=host_path.name or "workspace",
            is_dir=host_path.is_dir(),
            is_symlink=host_path.is_symlink(),
            size=int(info.st_size),
            mtime=float(info.st_mtime),
            mode=int(info.st_mode),
        )

    async def read_file(self, path: str, *, encoding: str = "utf-8", max_bytes: int | None = None) -> str:
        return (await self.read_bytes(path, max_bytes=max_bytes)).decode(encoding, errors="replace")

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        host = self._host(path, must_exist=True)
        if not host.is_file():
            raise FileNotFoundError(path)
        data = await asyncio.to_thread(host.read_bytes)
        return data[:max_bytes] if max_bytes is not None else data

    async def write_file(self, path: str, content: str, *, encoding: str = "utf-8", parents: bool = True) -> None:
        await self.write_bytes(path, content.encode(encoding), parents=parents)

    async def write_bytes(self, path: str, content: bytes, *, parents: bool = True) -> None:
        host = self._host(path)
        if parents:
            host.parent.mkdir(parents=True, exist_ok=True)
        temporary = host.with_name(f".{host.name}.shell-tmp")
        await asyncio.to_thread(temporary.write_bytes, content)
        os.replace(temporary, host)

    async def append_file(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        host = self._host(path)
        host.parent.mkdir(parents=True, exist_ok=True)
        with host.open("a", encoding=encoding) as stream:
            stream.write(content)

    async def ls(self, path: str = ".") -> list[FileStat]:
        host = self._host(path, must_exist=True)
        if not host.is_dir():
            return [self._stat(host)]
        return [self._stat(child) for child in sorted(host.iterdir(), key=lambda p: p.name)]

    async def glob(self, pattern: str) -> list[str]:
        base = self._host(".")
        matches: list[str] = []
        for candidate in sorted(base.rglob("*")):
            rel = candidate.relative_to(base).as_posix()
            if fnmatch.fnmatch(rel, pattern.lstrip("/").removeprefix("workspace/")):
                matches.append(self._display(candidate))
        return matches

    async def stat(self, path: str) -> FileStat:
        return self._stat(self._host(path, must_exist=True))

    async def exists(self, path: str) -> bool:
        try:
            return self._host(path, must_exist=True).exists()
        except (FileNotFoundError, ValueError):
            return False

    async def mkdir(self, path: str, *, parents: bool = True) -> None:
        self._host(path).mkdir(parents=parents, exist_ok=True)

    async def rm(self, path: str, *, recursive: bool = False, missing_ok: bool = True) -> None:
        try:
            host = self._host(path, must_exist=True)
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        if host == self._root:
            raise ValueError("refusing to remove the workspace root")
        if host.is_dir() and not host.is_symlink():
            if not recursive:
                raise IsADirectoryError(path)
            await asyncio.to_thread(shutil.rmtree, host)
        else:
            host.unlink(missing_ok=missing_ok)

    async def move(self, src: str, dst: str) -> None:
        source = self._host(src, must_exist=True)
        destination = self._host(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.move, str(source), str(destination))

    async def upload(self, local_path: str, remote_path: str, *, parents: bool = True) -> None:
        await self.write_bytes(remote_path, Path(local_path).read_bytes(), parents=parents)

    async def download(self, remote_path: str, local_path: str, *, parents: bool = True) -> None:
        target = Path(local_path)
        if parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self.read_bytes(remote_path))


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


__all__ = ["ExecutionTarget", "LocalDockerTarget", "RelayTarget"]
