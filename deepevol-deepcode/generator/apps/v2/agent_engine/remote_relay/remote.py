"""`RemoteRuntime`：把一台远端机器装成"本地"。

Agent 拿到的就是这个对象。它的方法签名和 `LocalRuntime` 一模一样，所以上层代码根本
不需要知道自己跑在哪：

    runtime = RemoteRuntime("ssh -p 2222 root@1.2.3.4", password="…")
    await runtime.wait_ready()

    await runtime.exec("cd /root/exp")          # cwd 会粘住
    await runtime.exec("apt-get install -y tree")
    print(await runtime.read_file("train.py"))  # 相对路径按粘住的 cwd 解析

    job = await runtime.spawn("python train.py")        # 几小时的实验
    async for event in runtime.stream(job.job_id):      # 实时输出，断线自动续
        print(event.data, end="")

`exec()` 的超时超过 `DURABLE_THRESHOLD_SECONDS` 时会自动转成耐久作业——"挂在一条 SSH
channel 上跑半小时"本身就是个不可靠的赌注，与其让 Agent 记得区分，不如默认帮它选对。
"""

from __future__ import annotations

import asyncio
import shlex
import time
from typing import Any, AsyncIterator, Callable

from .diagnostics import Tracer
from .execution.fast import run_fast
from .execution.jobs import DEFAULT_JOBS_ROOT, JobManager
from .execution.shellstate import DEFAULT_REMOTE_ENV, ShellState, new_state_marker
from .fs.sftp import RemoteFileSystem
from .fs.transfer import DEFAULT_EXCLUDES, FileTransfer, SyncReport
from .transport.ssh import SSHTransport
from .transport.target import SSHTarget, parse_access_url
from .types import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    DURABLE_THRESHOLD_SECONDS,
    ExecEvent,
    ExecResult,
    FileStat,
    JobStatus,
    RelayAuthError,
    RelayConnectionError,
    RemoteCommandError,
)


EventCallback = Callable[[ExecEvent], Any]


class RemoteRuntime:
    """一台远端机器的完整门面：执行 + 作业 + 文件。"""

    def __init__(
        self,
        target: SSHTarget | str,
        *,
        username: str = "root",
        password: str | None = None,
        cwd: str = "",
        env: dict[str, str] | None = None,
        default_env: bool = True,
        jobs_root: str = DEFAULT_JOBS_ROOT,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        durable_threshold: float = DURABLE_THRESHOLD_SECONDS,
        transport: Any = None,
        tracer: Tracer | None = None,
    ) -> None:
        if isinstance(target, str):
            target = parse_access_url(target, username=username, password=password)
        self.target = target
        self.transport = transport if transport is not None else SSHTransport(target)
        base_env = dict(DEFAULT_REMOTE_ENV) if default_env else {}
        if env:
            base_env.update(env)
        self.state = ShellState(cwd=cwd, env=base_env)
        self.jobs = JobManager(
            self.transport, self.state, root=jobs_root, max_output_bytes=max_output_bytes
        )
        self.files = RemoteFileSystem(self.transport, self.state)
        self.transfer = FileTransfer(self.transport, self.files, exec_fn=self._raw_exec)
        self.max_output_bytes = max_output_bytes
        self.durable_threshold = float(durable_threshold)
        self.tracer = tracer if tracer is not None else Tracer()

    # ── 生命周期 ──────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self.target.display

    async def start(self) -> None:
        await self.transport.connect()

    async def close(self) -> None:
        await self.transport.close()

    async def __aenter__(self) -> "RemoteRuntime":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def wait_ready(self, *, timeout: float = 300.0, interval: float = 5.0) -> None:
        """轮询直到这台机器真的能跑命令。

        刚租来的机器要等 sshd 起来、cloud-init 跑完，端口通了不代表能用。认证失败会立刻
        抛出——那是凭据错了，再等一百年也不会好。
        """
        deadline = time.monotonic() + float(timeout)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                remaining = max(0.0, deadline - time.monotonic())
                await asyncio.wait_for(self.transport.connect(), timeout=remaining)
                remaining = max(0.0, deadline - time.monotonic())
                if remaining <= 0:
                    break
                result, _ = await run_fast(
                    self.transport,
                    "true",
                    timeout=min(20.0, remaining),
                    track_state=False,
                    state_marker=new_state_marker(),
                    remote_watchdog=True,
                )
                if result.success:
                    return
                last_error = RelayConnectionError(f"probe command failed: {result.output.strip()}")
            except RelayAuthError:
                raise
            except Exception as exc:
                last_error = exc
            remaining = max(0.0, deadline - time.monotonic())
            if remaining > 0:
                await asyncio.sleep(min(interval, remaining))
        raise RelayConnectionError(
            f"{self.name} not ready within {timeout:.0f}s: "
            f"{type(last_error).__name__ if last_error else 'unknown'}: {last_error}"
        )

    # ── 执行 ──────────────────────────────────────────────────────────────
    @property
    def cwd(self) -> str:
        return self.state.cwd

    async def _raw_exec(self, script: str, *, timeout: float = 60.0) -> ExecResult:
        """跑一条中转层自己的辅助脚本（不带 bootstrap、不动粘性状态）。"""
        marker = new_state_marker()
        result, _ = await run_fast(
            self.transport,
            script,
            timeout=timeout,
            track_state=False,
            state_marker=marker,
            remote_watchdog=True,
            max_output_bytes=1 << 20,
        )
        return result

    def _should_be_durable(self, durable: bool | None, timeout: float | None) -> bool:
        if durable is not None:
            return durable
        effective = DEFAULT_TIMEOUT_SECONDS if timeout is None else float(timeout)
        return effective > self.durable_threshold

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
        """跑一条命令并等它结束。on_event 非空时按 chunk 实时回推输出。"""
        started = time.monotonic()
        if self._should_be_durable(durable, timeout):
            job = await self.jobs.start(command, cwd=cwd, env=env)
            result = await self.jobs.wait(
                job.job_id,
                timeout=timeout,
                on_event=on_event,
                terminate_on_timeout=True,
            )
            if cwd is None and not result.timed_out:
                self.state.observe(result.cwd)
        else:
            marker = new_state_marker()
            script = self.state.build(
                command,
                cwd=cwd,
                env=env,
                track_state=True,
                state_marker=marker,
            )
            result, payload = await run_fast(
                self.transport,
                script,
                command=command,
                cwd=cwd or self.state.cwd,
                timeout=timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS,
                on_event=on_event,
                max_output_bytes=self.max_output_bytes,
                track_state=True,
                state_marker=marker,
                remote_watchdog=True,
            )
            final_cwd = payload if payload.startswith("/") else (cwd or self.state.cwd)
            if cwd is None and not result.timed_out:
                self.state.observe(payload)
            result = ExecResult(
                exit_status=result.exit_status,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=result.duration_ms,
                command=command,
                cwd=final_cwd,
                timed_out=result.timed_out,
                stdout_truncated=result.stdout_truncated,
                stderr_truncated=result.stderr_truncated,
            )
        self.tracer.record(
            "exec",
            target=self.name,
            command=command,
            exit_status=result.exit_status,
            stdout=result.stdout,
            stderr=result.stderr,
            ms=int((time.monotonic() - started) * 1000),
            job_id=result.job_id,
        )
        return result

    async def check_exec(self, command: str, **kwargs: Any) -> ExecResult:
        """跑命令，非 0 直接抛 `RemoteCommandError`。中转层自己的内部步骤用它。"""
        result = await self.exec(command, **kwargs)
        if not result.success:
            raise RemoteCommandError(result)
        return result

    async def chdir(self, path: str) -> str:
        """显式切目录并返回切换后的 cwd。等价于 `exec("cd <path>")`。"""
        result = await self.check_exec(f"cd -- {shlex.quote(str(path))}")
        return result.cwd

    # ── 耐久作业 ──────────────────────────────────────────────────────────
    async def spawn(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        status = await self.jobs.start(command, cwd=cwd, env=env, job_id=job_id)
        self.tracer.record("spawn", target=self.name, command=command, job_id=status.job_id)
        return status

    def stream(
        self,
        job_id: str,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        follow: bool = True,
    ) -> AsyncIterator[ExecEvent]:
        return self.jobs.stream(
            job_id, stdout_offset=stdout_offset, stderr_offset=stderr_offset, follow=follow
        )

    async def job_status(self, job_id: str) -> JobStatus:
        return await self.jobs.status(job_id)

    async def wait(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        on_event: EventCallback | None = None,
    ) -> ExecResult:
        return await self.jobs.wait(job_id, timeout=timeout, on_event=on_event)

    async def kill(self, job_id: str, *, sig: str = "TERM") -> bool:
        killed = await self.jobs.kill(job_id, sig=sig)
        self.tracer.record("kill", target=self.name, command=job_id, exit_status=0 if killed else 1)
        return killed

    async def list_jobs(self) -> list[str]:
        return await self.jobs.list_jobs()

    # ── 文件系统 ──────────────────────────────────────────────────────────
    async def read_file(
        self, path: str, *, encoding: str = "utf-8", max_bytes: int | None = None
    ) -> str:
        return await self.files.read_file(path, encoding=encoding, max_bytes=max_bytes)

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        return await self.files.read_bytes(path, max_bytes=max_bytes)

    async def write_file(
        self, path: str, content: str, *, encoding: str = "utf-8", parents: bool = True
    ) -> None:
        await self.files.write_file(path, content, encoding=encoding, parents=parents)

    async def write_bytes(self, path: str, content: bytes, *, parents: bool = True) -> None:
        await self.files.write_bytes(path, content, parents=parents)

    async def append_file(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        await self.files.append_file(path, content, encoding=encoding)

    async def ls(self, path: str = ".") -> list[FileStat]:
        return await self.files.ls(path)

    async def glob(self, pattern: str) -> list[str]:
        return await self.files.glob(pattern)

    async def stat(self, path: str) -> FileStat:
        return await self.files.stat(path)

    async def exists(self, path: str) -> bool:
        return await self.files.exists(path)

    async def mkdir(self, path: str, *, parents: bool = True) -> None:
        await self.files.mkdir(path, parents=parents)

    async def rm(self, path: str, *, recursive: bool = False, missing_ok: bool = True) -> None:
        await self.files.rm(path, recursive=recursive, missing_ok=missing_ok)

    async def move(self, src: str, dst: str) -> None:
        await self.files.move(src, dst)

    # ── 本地 ↔ 远端 ───────────────────────────────────────────────────────
    async def upload(
        self, local_path: str, remote_path: str, *, parents: bool = True, verify: bool = False
    ) -> None:
        await self.transfer.upload(local_path, remote_path, parents=parents, verify=verify)
        self.tracer.record("upload", target=self.name, command=f"{local_path} -> {remote_path}")

    async def download(
        self, remote_path: str, local_path: str, *, parents: bool = True, verify: bool = False
    ) -> None:
        await self.transfer.download(remote_path, local_path, parents=parents, verify=verify)
        self.tracer.record("download", target=self.name, command=f"{remote_path} -> {local_path}")

    async def sync_up(
        self, local_dir: str, remote_dir: str, *, excludes: Any = DEFAULT_EXCLUDES
    ) -> SyncReport:
        report = await self.transfer.sync_up(local_dir, remote_dir, excludes=excludes)
        self.tracer.record(
            "sync_up", target=self.name, command=f"{local_dir} -> {remote_dir}", stdout=report.summary
        )
        return report

    async def sync_down(
        self, remote_dir: str, local_dir: str, *, excludes: Any = DEFAULT_EXCLUDES
    ) -> SyncReport:
        report = await self.transfer.sync_down(remote_dir, local_dir, excludes=excludes)
        self.tracer.record(
            "sync_down", target=self.name, command=f"{remote_dir} -> {local_dir}", stdout=report.summary
        )
        return report
