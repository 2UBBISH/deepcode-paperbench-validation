"""`LocalRuntime`：跟 `RemoteRuntime` 同签名的本地实现。

存在的意义有三个：没租到机器时的降级路径、单测里不连真机的基线、以及一个随时可对照的
"本地语义应该长什么样"的参照物。作业布局刻意和远端保持一致（同样的 stdout.log /
exit_code / pgid），所以流式续读的逻辑两边是同一套心智模型。
"""

from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from .execution.fast import TIMEOUT_HINT
from .execution.jobs import validate_job_id
from .execution.shellstate import (
    LEGACY_STATE_MARKER,
    ShellState,
    job_is_starting,
    new_state_marker,
    wrap_job_script,
)
from .execution.stream import CappedText, StreamDecoder
from .types import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    DURABLE_THRESHOLD_SECONDS,
    ExecEvent,
    ExecResult,
    FileStat,
    JobStatus,
    RelayError,
    RemoteCommandError,
    RemoteFileNotFoundError,
    TIMEOUT_EXIT_STATUS,
    normalize_termination_signal,
)


EventCallback = Callable[[ExecEvent], Any]

DEFAULT_LOCAL_JOBS_ROOT = "~/.remote-relay/local-jobs"


class LocalRuntime:
    """在本机跑命令、读写本机文件。"""

    def __init__(
        self,
        *,
        cwd: str = "",
        env: dict[str, str] | None = None,
        jobs_root: str = DEFAULT_LOCAL_JOBS_ROOT,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        durable_threshold: float = DURABLE_THRESHOLD_SECONDS,
        shell: str = "/bin/bash",
    ) -> None:
        # 本地不需要 conda PATH / 学术加速那套 bootstrap，也不该往用户环境里塞镜像源。
        self.state = ShellState(cwd=cwd or os.getcwd(), env=dict(env or {}), bootstrap="")
        self.jobs_root = Path(jobs_root).expanduser()
        self.max_output_bytes = max_output_bytes
        self.durable_threshold = float(durable_threshold)
        self.shell = shell
        self._processes: dict[str, Any] = {}
        """留着子进程句柄只为让事件循环收尸，避免僵尸进程。"""

    # ── 生命周期 ──────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return "local"

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def wait_ready(self, *, timeout: float = 300.0) -> None:
        return None

    async def __aenter__(self) -> "LocalRuntime":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    # ── 执行 ──────────────────────────────────────────────────────────────
    @property
    def cwd(self) -> str:
        return self.state.cwd

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_event: EventCallback | None = None,
        durable: bool | None = None,
    ) -> ExecResult:
        timeout_s = float(timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS)
        use_durable = (
            bool(durable)
            if durable is not None
            else timeout_s > self.durable_threshold
        )
        if use_durable:
            job = await self.spawn(command, cwd=cwd, env=env)
            result = await self._wait_job(
                job.job_id,
                timeout=timeout,
                on_event=on_event,
                terminate_on_timeout=True,
            )
            if cwd is None and not result.timed_out:
                self.state.observe(result.cwd)
            return result

        marker = new_state_marker()
        script = self.state.build(
            command,
            cwd=cwd,
            env=env,
            track_state=True,
            state_marker=marker,
        )
        started = time.monotonic()

        process = await asyncio.create_subprocess_exec(
            self.shell,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout_decoder = StreamDecoder(strip_state=True, state_marker=marker)
        stderr_decoder = StreamDecoder(strip_state=False)
        stdout_sink = CappedText(self.max_output_bytes)
        stderr_sink = CappedText(self.max_output_bytes)

        async def pump(reader: Any, decoder: StreamDecoder, sink: CappedText, kind: str) -> None:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                text = decoder.feed(chunk)
                if text:
                    sink.append(text)
                    await _emit(on_event, ExecEvent(kind=kind, data=text, offset=decoder.offset))  # type: ignore[arg-type]
            tail = decoder.finish()
            if tail:
                sink.append(tail)
                await _emit(on_event, ExecEvent(kind=kind, data=tail, offset=decoder.offset))  # type: ignore[arg-type]

        timed_out = False
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    pump(process.stdout, stdout_decoder, stdout_sink, "stdout"),
                    pump(process.stderr, stderr_decoder, stderr_sink, "stderr"),
                    process.wait(),
                ),
                timeout=timeout_s,
            )
            exit_status = int(process.returncode or 0)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_process_group(process.pid)
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            exit_status = TIMEOUT_EXIT_STATUS
            stderr_sink.append(f"\n[remote-relay] 命令超过 {timeout_s:.0f}s 墙钟超时，已终止。")

        final_cwd = (
            stdout_decoder.state_payload
            if stdout_decoder.state_payload.startswith("/")
            else (cwd if cwd is not None else self.state.cwd)
        )
        if cwd is None and not timed_out:
            self.state.observe(stdout_decoder.state_payload)
        await _emit(on_event, ExecEvent(kind="exit", exit_status=exit_status))
        return ExecResult(
            exit_status=exit_status,
            stdout=stdout_sink.value(),
            stderr=stderr_sink.value(),
            duration_ms=int((time.monotonic() - started) * 1000),
            command=command,
            cwd=final_cwd,
            timed_out=timed_out,
            stdout_truncated=stdout_sink.truncated,
            stderr_truncated=stderr_sink.truncated,
        )

    async def check_exec(self, command: str, **kwargs: Any) -> ExecResult:
        result = await self.exec(command, **kwargs)
        if not result.success:
            raise RemoteCommandError(result)
        return result

    async def chdir(self, path: str) -> str:
        import shlex

        result = await self.check_exec(f"cd -- {shlex.quote(path)}")
        return result.cwd

    # ── 耐久作业 ──────────────────────────────────────────────────────────
    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / validate_job_id(job_id)

    async def spawn(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        job_id = validate_job_id(job_id or f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}")
        marker = new_state_marker()
        directory = self._job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        body = self.state.build(command, cwd=cwd, env=env, track_state=False)
        (directory / "run.sh").write_text(
            wrap_job_script(body, state_marker=marker), encoding="utf-8"
        )
        (directory / "cmd").write_text(command, encoding="utf-8")
        (directory / "cwd").write_text(
            cwd if cwd is not None else self.state.cwd, encoding="utf-8"
        )
        (directory / "state_marker").write_bytes(marker)
        (directory / "started_at").write_text(str(int(time.time())), encoding="utf-8")
        (directory / "stdout.log").write_bytes(b"")
        (directory / "stderr.log").write_bytes(b"")
        exit_code = directory / "exit_code"
        if exit_code.exists():
            exit_code.unlink()

        stdout_handle = (directory / "stdout.log").open("wb")
        stderr_handle = (directory / "stderr.log").open("wb")
        try:
            process = await asyncio.create_subprocess_exec(
                self.shell,
                str(directory / "run.sh"),
                stdout=stdout_handle,
                stderr=stderr_handle,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "RELAY_JOB_DIR": str(directory)},
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        # 本地知道子进程 pid，直接写下去，省掉"作业自己写 pgid"的启动竞态。
        # start_new_session=True 让它成为组长，所以 pid == pgid。
        (directory / "pgid").write_text(f"{process.pid}\n", encoding="utf-8")
        self._processes[job_id] = process
        return JobStatus(
            job_id=job_id,
            running=True,
            command=command,
            cwd=cwd or self.state.cwd,
            started_at=time.time(),
            state_marker=marker.decode("ascii"),
        )

    async def job_status(self, job_id: str) -> JobStatus:
        directory = self._job_dir(job_id)
        if not directory.is_dir():
            raise RelayError(f"job not found: {job_id}")
        exit_status = _read_int(directory / "exit_code")
        pgid = _read_int(directory / "pgid")
        started_at = float(_read_int(directory / "started_at") or 0)
        running = False
        if pgid and exit_status is None:
            try:
                os.killpg(pgid, 0)
                running = True
            except (ProcessLookupError, PermissionError, OSError):
                running = False
        return JobStatus(
            job_id=job_id,
            running=running or job_is_starting(exit_status, pgid, started_at),
            exit_status=exit_status,
            pgid=pgid,
            command=_read_text(directory / "cmd"),
            cwd=_read_text(directory / "cwd"),
            started_at=started_at,
            stdout_size=_file_size(directory / "stdout.log"),
            stderr_size=_file_size(directory / "stderr.log"),
            state_marker=(
                _read_bytes(directory / "state_marker") or LEGACY_STATE_MARKER
            ).decode("ascii"),
        )

    async def stream(
        self,
        job_id: str,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        follow: bool = True,
    ) -> AsyncIterator[ExecEvent]:
        directory = self._job_dir(job_id)
        if not directory.is_dir():
            raise RelayError(f"job not found: {job_id}")
        initial_status = await self.job_status(job_id)
        marker = (
            initial_status.state_marker.encode("ascii")
            if initial_status.state_marker
            else LEGACY_STATE_MARKER
        )
        decoders = {
            "stdout": StreamDecoder(start_offset=stdout_offset, state_marker=marker),
            "stderr": StreamDecoder(start_offset=stderr_offset, strip_state=False),
        }
        paths = {"stdout": directory / "stdout.log", "stderr": directory / "stderr.log"}

        while True:
            moved = False
            for kind, decoder in decoders.items():
                # 按 consumed（读取游标）而不是 offset（续读游标）读：offset 不含被哨兵扣住的
                # 字节，用它当读取起点会把同几个字节反复读进来，循环永不收敛。
                while True:
                    chunk = _read_from(paths[kind], decoder.consumed)
                    if not chunk:
                        break
                    moved = True
                    text = decoder.feed(chunk)
                    if text:
                        yield ExecEvent(kind=kind, data=text, offset=decoder.offset)  # type: ignore[arg-type]
            if not follow:
                break
            status = await self.job_status(job_id)
            if status.finished and not moved:
                break
            if not status.running and status.exit_status is None and not moved:
                break
            if not moved:
                await asyncio.sleep(0.1)
        for kind, decoder in decoders.items():
            tail = decoder.finish()
            if tail:
                yield ExecEvent(kind=kind, data=tail, offset=decoder.offset)  # type: ignore[arg-type]
        status = await self.job_status(job_id)
        yield ExecEvent(kind="exit", exit_status=status.exit_status)

    async def wait(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        on_event: EventCallback | None = None,
    ) -> ExecResult:
        return await self._wait_job(job_id, timeout=timeout, on_event=on_event)

    async def _wait_job(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        on_event: EventCallback | None = None,
        terminate_on_timeout: bool = False,
    ) -> ExecResult:
        started = time.monotonic()
        stdout_sink = CappedText(self.max_output_bytes)
        stderr_sink = CappedText(self.max_output_bytes)
        exit_status: int | None = None
        stdout_offset = 0
        stderr_offset = 0

        async def accept(event: ExecEvent) -> None:
            nonlocal exit_status, stdout_offset, stderr_offset
            if event.kind == "exit":
                exit_status = event.exit_status
            elif event.kind == "stdout":
                stdout_offset = event.offset
                stdout_sink.append(event.data)
            else:
                stderr_offset = event.offset
                stderr_sink.append(event.data)
            await _emit(on_event, event)

        async def consume() -> None:
            async for event in self.stream(job_id):
                await accept(event)

        timed_out = False
        try:
            if timeout is None:
                await consume()
            else:
                await asyncio.wait_for(consume(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            if terminate_on_timeout:
                await self._terminate_job(
                    job_id, sig="TERM", exit_status=TIMEOUT_EXIT_STATUS
                )
                async for event in self.stream(
                    job_id,
                    stdout_offset=stdout_offset,
                    stderr_offset=stderr_offset,
                    follow=False,
                ):
                    await accept(event)
                exit_status = TIMEOUT_EXIT_STATUS
                stderr_sink.append(TIMEOUT_HINT.format(seconds=float(timeout or 0)))

        status = await self.job_status(job_id)
        return ExecResult(
            exit_status=(
                exit_status
                if exit_status is not None
                else (status.exit_status if status.exit_status is not None else -1)
            ),
            stdout=stdout_sink.value(),
            stderr=stderr_sink.value(),
            duration_ms=int((time.monotonic() - started) * 1000),
            command=status.command,
            cwd=status.cwd,
            job_id=job_id,
            timed_out=timed_out,
            stdout_truncated=stdout_sink.truncated,
            stderr_truncated=stderr_sink.truncated,
        )

    async def kill(self, job_id: str, *, sig: str = "TERM") -> bool:
        name, exit_status = normalize_termination_signal(sig)
        return await self._terminate_job(job_id, sig=name, exit_status=exit_status)

    async def _terminate_job(self, job_id: str, *, sig: str, exit_status: int) -> bool:
        status = await self.job_status(job_id)
        if not status.pgid:
            return False
        killed = _kill_process_group(status.pgid, sig)
        process = self._processes.get(job_id)
        if killed:
            if process is not None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    _kill_process_group(status.pgid, "KILL")
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        pass
            else:
                for _ in range(50):
                    if not _process_group_exists(status.pgid):
                        break
                    await asyncio.sleep(0.1)
                if _process_group_exists(status.pgid):
                    _kill_process_group(status.pgid, "KILL")
        exit_code = self._job_dir(job_id) / "exit_code"
        if killed:
            temp = exit_code.with_suffix(".tmp")
            temp.write_text(str(int(exit_status)), encoding="utf-8")
            temp.replace(exit_code)
        return killed

    async def list_jobs(self) -> list[str]:
        if not self.jobs_root.is_dir():
            return []
        return sorted(p.name for p in self.jobs_root.iterdir() if p.is_dir())

    # ── 文件系统 ──────────────────────────────────────────────────────────
    def _resolve(self, path: str) -> Path:
        raw = Path(str(path or ".")).expanduser()
        if raw.is_absolute():
            return raw
        return Path(self.state.cwd or os.getcwd()) / raw

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        target = self._resolve(path)
        if not target.is_file():
            raise RemoteFileNotFoundError(f"local path not found: {target}")
        data = await asyncio.to_thread(target.read_bytes)
        return data if max_bytes is None else data[:max_bytes]

    async def read_file(
        self, path: str, *, encoding: str = "utf-8", max_bytes: int | None = None
    ) -> str:
        raw = await self.read_bytes(path, max_bytes=max_bytes)
        return raw.decode(encoding, errors="replace")

    async def write_bytes(self, path: str, content: bytes, *, parents: bool = True) -> None:
        target = self._resolve(path)
        if parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, content)

    async def write_file(
        self, path: str, content: str, *, encoding: str = "utf-8", parents: bool = True
    ) -> None:
        await self.write_bytes(path, str(content).encode(encoding), parents=parents)

    async def append_file(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        def _append() -> None:
            with target.open("ab") as handle:
                handle.write(str(content).encode(encoding))

        await asyncio.to_thread(_append)

    async def ls(self, path: str = ".") -> list[FileStat]:
        target = self._resolve(path)
        if not target.is_dir():
            raise RemoteFileNotFoundError(f"local directory not found: {target}")
        entries = [_local_stat(child) for child in sorted(target.iterdir())]
        entries.sort(key=lambda item: (not item.is_dir, item.name))
        return entries

    async def glob(self, pattern: str) -> list[str]:
        raw = Path(str(pattern)).expanduser()
        if raw.is_absolute():
            root = Path(raw.anchor)
            relative = str(raw.relative_to(root))
        else:
            root = Path(self.state.cwd or os.getcwd())
            relative = str(raw)
        return sorted(str(item) for item in root.glob(relative))

    async def stat(self, path: str) -> FileStat:
        target = self._resolve(path)
        if not target.exists():
            raise RemoteFileNotFoundError(f"local path not found: {target}")
        return _local_stat(target)

    async def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    async def mkdir(self, path: str, *, parents: bool = True) -> None:
        self._resolve(path).mkdir(parents=parents, exist_ok=True)

    async def rm(self, path: str, *, recursive: bool = False, missing_ok: bool = True) -> None:
        target = self._resolve(path)
        if not target.exists():
            if missing_ok:
                return
            raise RemoteFileNotFoundError(f"local path not found: {target}")
        if target.is_dir():
            if recursive:
                shutil.rmtree(target, ignore_errors=missing_ok)
            else:
                target.rmdir()
        else:
            target.unlink()

    async def move(self, src: str, dst: str) -> None:
        source = self._resolve(src)
        destination = self._resolve(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    # ── 本地 ↔ "远端"（本地实现里就是一次拷贝） ──────────────────────────
    async def upload(
        self, local_path: str, remote_path: str, *, parents: bool = True, verify: bool = False
    ) -> None:
        await self._copy(Path(local_path).expanduser(), self._resolve(remote_path), parents)

    async def download(
        self, remote_path: str, local_path: str, *, parents: bool = True, verify: bool = False
    ) -> None:
        await self._copy(self._resolve(remote_path), Path(local_path).expanduser(), parents)

    async def _copy(self, source: Path, destination: Path, parents: bool) -> None:
        if not source.is_file():
            raise RemoteFileNotFoundError(f"file not found: {source}")
        if parents:
            destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, str(source), str(destination))


# ── 小工具 ────────────────────────────────────────────────────────────────
async def _emit(callback: EventCallback | None, event: ExecEvent) -> None:
    if callback is None:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result


def _kill_process_group(pid: int, sig: str = "TERM") -> bool:
    name, _ = normalize_termination_signal(sig)
    number = getattr(signal, f"SIG{name}")
    try:
        os.killpg(pid, number)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _read_from(path: Path, offset: int) -> bytes:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(65536)
    except OSError:
        return b""


def _local_stat(path: Path) -> FileStat:
    try:
        info = path.lstat()
    except OSError:
        return FileStat(path=str(path), name=path.name)
    return FileStat(
        path=str(path),
        name=path.name,
        is_dir=path.is_dir(),
        is_symlink=path.is_symlink(),
        size=info.st_size,
        mtime=info.st_mtime,
        mode=info.st_mode,
    )
