"""耐久作业：长实验的正式跑法。

**这是整个中转层最关键的一块。** 训练脚本要跑几小时，期间 SSH 断线、Wi-Fi 抖动、Agent
自己重启，都是必然会发生的事。所以命令不挂在 SSH channel 上，而是在远端落成一个作业：

    ~/.remote-relay/jobs/<job_id>/
        run.sh        实际执行的脚本（带 bootstrap / cd / export）
        cmd           原始命令，给人看的
        pgid          进程组 id，kill 用
        stdout.log    全量输出，是唯一真源
        stderr.log
        exit_code     原子写入（写 .tmp 再 mv），出现即代表作业结束
        started_at

于是"实时输出"退化成一个很朴素的问题：**从日志文件的第 N 个字节开始读**。断了就换个
offset 再读一次，不丢也不重；本地 Agent 重启后凭 job_id 就能接着看。远端进程用 setsid
拉进独立会话，SSH 关闭时的 SIGHUP 打不到它。

跟随式流读只占一条 channel：一条命令里同时 tail 两个日志（stderr 那路重定向到 fd 2），
并在 exit_code 出现后自行退出——不需要额外的轮询通道。
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import posixpath
import re
import shlex
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from ..types import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ExecEvent,
    ExecResult,
    JobStatus,
    RelayConnectionError,
    RelayError,
    RelayTimeoutError,
    TIMEOUT_EXIT_STATUS,
    normalize_termination_signal,
)
from .fast import TIMEOUT_HINT, run_fast
from .shellstate import (
    LEGACY_STATE_MARKER,
    ShellState,
    job_is_starting,
    new_state_marker,
    wrap_job_script,
)
from .stream import CappedText, StreamDecoder


DEFAULT_JOBS_ROOT = "$HOME/.remote-relay/jobs"
# tail 的轮询间隔。默认 1s 对"实时"来说太钝了，实验日志会一秒一顿。
_TAIL_SLEEP = "0.1"
# 跟随通道退出后，再补一次非跟随读，兜住 tail 收尾时的竞态。
_CATCHUP_TIMEOUT = 30.0
_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def new_job_id() -> str:
    return f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def validate_job_id(job_id: str) -> str:
    value = str(job_id or "")
    if not _JOB_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "job_id must be 1-128 characters and contain only letters, digits, '.', '_', or '-'"
        )
    return value


@dataclass
class _StreamCursor:
    offset: int = 0


class JobManager:
    """一台远端机器上的作业集合。"""

    def __init__(
        self,
        transport: Any,
        state: ShellState,
        *,
        root: str = DEFAULT_JOBS_ROOT,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        self.transport = transport
        self.state = state
        self.root = root
        self.max_output_bytes = max_output_bytes

    # ── 内部：跑一条辅助命令（状态查询、kill 之类） ─────────────────────
    async def _run(self, script: str, *, timeout: float = 60.0) -> ExecResult:
        marker = new_state_marker()
        result, _ = await run_fast(
            self.transport,
            script,
            timeout=timeout,
            track_state=False,
            state_marker=marker,
            remote_watchdog=True,
            max_output_bytes=1024 * 1024,
        )
        if result.timed_out:
            raise RelayTimeoutError(f"relay helper command timed out after {timeout:.0f}s")
        return result

    def _job_dir_expr(self, job_id: str) -> str:
        """作业目录的 shell 表达式（root 里含 $HOME，得留给远端展开）。"""
        job_id = validate_job_id(job_id)
        root = str(self.root).rstrip("/")
        if root == "$HOME" or root == "~":
            return f'"$HOME"/{shlex.quote(job_id)}'
        if root.startswith("$HOME/"):
            relative = posixpath.join(root[len("$HOME/") :], job_id)
            return f'"$HOME"/{shlex.quote(relative)}'
        if root.startswith("~/"):
            relative = posixpath.join(root[2:], job_id)
            return f'"$HOME"/{shlex.quote(relative)}'
        return shlex.quote(posixpath.join(root, job_id))

    def _root_expr(self) -> str:
        root = str(self.root).rstrip("/")
        if root == "$HOME" or root == "~":
            return '"$HOME"'
        if root.startswith("$HOME/"):
            return f'"$HOME"/{shlex.quote(root[len("$HOME/") :])}'
        if root.startswith("~/"):
            return f'"$HOME"/{shlex.quote(root[2:])}'
        return shlex.quote(root)

    # ── 启动 ──────────────────────────────────────────────────────────────
    async def start(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        """把命令作为后台作业启动，一个往返就返回。"""
        job_id = validate_job_id(job_id or new_job_id())
        marker = new_state_marker()
        body = self.state.build(command, cwd=cwd, env=env, track_state=False)
        run_sh = wrap_job_script(body, state_marker=marker)
        encoded = base64.b64encode(run_sh.encode("utf-8")).decode("ascii")
        cmd_encoded = base64.b64encode(str(command).encode("utf-8")).decode("ascii")
        cwd_encoded = base64.b64encode(str(cwd or self.state.cwd).encode("utf-8")).decode("ascii")
        marker_encoded = base64.b64encode(marker).decode("ascii")
        d = self._job_dir_expr(job_id)
        launcher = "\n".join(
            [
                "set -e",
                f"d={d}",
                'mkdir -p "$d"',
                # 用带引号的 heredoc 传脚本，彻底绕开引号地狱：base64 里不会出现分隔符。
                'base64 -d > "$d/run.sh" <<\'__RELAY_RUNSH__\'',
                encoded,
                "__RELAY_RUNSH__",
                f"printf '%s' '{cmd_encoded}' | base64 -d > \"$d/cmd\"",
                f"printf '%s' '{cwd_encoded}' | base64 -d > \"$d/cwd\"",
                f"printf '%s' '{marker_encoded}' | base64 -d > \"$d/state_marker\"",
                'date +%s > "$d/started_at"',
                ': > "$d/stdout.log"',
                ': > "$d/stderr.log"',
                'rm -f "$d/exit_code"',
                'RELAY_JOB_DIR="$d" setsid nohup bash "$d/run.sh"'
                ' > "$d/stdout.log" 2> "$d/stderr.log" < /dev/null &',
                "disown 2>/dev/null || true",
                'echo "$d"',
            ]
        )
        result = await self._run(launcher, timeout=60.0)
        if not result.success:
            raise RelayError(
                f"cannot start job {job_id} on {self.transport.display}: {result.output.strip()}"
            )
        return JobStatus(
            job_id=job_id,
            running=True,
            command=command,
            cwd=cwd or self.state.cwd,
            started_at=time.time(),
            state_marker=marker.decode("ascii"),
        )

    # ── 状态 ──────────────────────────────────────────────────────────────
    async def status(self, job_id: str) -> JobStatus:
        d = self._job_dir_expr(job_id)
        script = "\n".join(
            [
                f"d={d}",
                'if [ ! -d "$d" ]; then echo "missing=1"; exit 0; fi',
                'rc=$(cat "$d/exit_code" 2>/dev/null || true)',
                'pgid=$(cat "$d/pgid" 2>/dev/null || true)',
                "running=0",
                'if [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null; then running=1; fi',
                'so=$(wc -c < "$d/stdout.log" 2>/dev/null || echo 0)',
                'se=$(wc -c < "$d/stderr.log" 2>/dev/null || echo 0)',
                'st=$(cat "$d/started_at" 2>/dev/null || echo 0)',
                'cmd=$(cat "$d/cmd" 2>/dev/null | head -c 4000 | base64 | tr -d "\\n")',
                'cwd=$(cat "$d/cwd" 2>/dev/null | base64 | tr -d "\\n")',
                'marker=$(cat "$d/state_marker" 2>/dev/null | base64 | tr -d "\\n")',
                'printf "missing=0\\nrc=%s\\npgid=%s\\nrunning=%s\\nstdout=%s\\nstderr=%s\\nstarted=%s\\ncmd=%s\\ncwd=%s\\nmarker=%s\\n"'
                ' "$rc" "$pgid" "$running" "$so" "$se" "$st" "$cmd" "$cwd" "$marker"',
            ]
        )
        result = await self._run(script, timeout=45.0)
        fields = _parse_fields(result.stdout)
        if fields.get("missing") != "0":
            raise RelayError(f"job not found: {job_id}")
        command = ""
        if fields.get("cmd"):
            try:
                command = base64.b64decode(fields["cmd"]).decode("utf-8", errors="replace")
            except Exception:
                command = ""
        cwd = _decode_field(fields.get("cwd"))
        marker = _decode_bytes_field(fields.get("marker")) or LEGACY_STATE_MARKER
        exit_status = _maybe_int(fields.get("rc"))
        pgid = _maybe_int(fields.get("pgid"))
        started_at = float(_maybe_int(fields.get("started")) or 0)
        running = fields.get("running") == "1"
        return JobStatus(
            job_id=job_id,
            # 刚拉起、pgid 还没落盘的窗口里，"查不到进程"不等于"作业没了"。
            running=running or job_is_starting(exit_status, pgid, started_at),
            exit_status=exit_status,
            pgid=pgid,
            command=command,
            cwd=cwd,
            started_at=started_at,
            stdout_size=int(_maybe_int(fields.get("stdout")) or 0),
            stderr_size=int(_maybe_int(fields.get("stderr")) or 0),
            state_marker=marker.decode("ascii"),
        )

    # ── 流式读取 ──────────────────────────────────────────────────────────
    def _follow_script(self, job_id: str, stdout_offset: int, stderr_offset: int) -> str:
        d = self._job_dir_expr(job_id)
        return "\n".join(
            [
                f"d={d}",
                # 两路 tail 共用一条 channel：stdout 走 fd 1，stderr 重定向到 fd 2。
                f'tail -s {_TAIL_SLEEP} -c +{stdout_offset + 1} -F "$d/stdout.log" 2>/dev/null &',
                "__t1=$!",
                # 重定向从左到右生效：先把日志复制到原始 fd 2，再静音 tail 自己的报错。
                # 若写成 ``2>/dev/null 1>&2``，日志也会跟着 fd 2 一起掉进 /dev/null。
                f'tail -s {_TAIL_SLEEP} -c +{stderr_offset + 1} -F "$d/stderr.log" 1>&2 2>/dev/null &',
                "__t2=$!",
                # exit_code 一出现就代表作业结束；再多给 tail 半秒把尾巴追平。
                'while [ ! -f "$d/exit_code" ]; do sleep 0.2; done',
                "sleep 0.5",
                "kill $__t1 $__t2 2>/dev/null || true",
                "wait $__t1 2>/dev/null || true",
                "wait $__t2 2>/dev/null || true",
            ]
        )

    def _catchup_script(self, job_id: str, stdout_offset: int, stderr_offset: int) -> str:
        d = self._job_dir_expr(job_id)
        return "\n".join(
            [
                f"d={d}",
                f'tail -c +{stdout_offset + 1} "$d/stdout.log" 2>/dev/null || true',
                f'tail -c +{stderr_offset + 1} "$d/stderr.log" 1>&2 2>/dev/null || true',
            ]
        )

    async def stream(
        self,
        job_id: str,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        follow: bool = True,
    ) -> AsyncIterator[ExecEvent]:
        """从给定字节偏移读作业输出。

        follow=True 时一直读到作业结束；中途断线会自动重连并从当前 offset 续读，
        所以调用方拿到的事件流是连续的、不重不漏。
        """
        out = _StreamCursor(offset=stdout_offset)
        err = _StreamCursor(offset=stderr_offset)
        snapshot: JobStatus

        while True:
            try:
                snapshot = await self.status(job_id)
                break
            except (RelayConnectionError, RelayTimeoutError):
                if not follow or getattr(self.transport, "closed", False):
                    raise
                await asyncio.sleep(1.0)
        marker = (
            snapshot.state_marker.encode("ascii")
            if snapshot.state_marker
            else LEGACY_STATE_MARKER
        )

        if follow and not snapshot.finished and snapshot.running:
            while True:
                try:
                    async for event in self._read_once(
                        self._follow_script(job_id, out.offset, err.offset),
                        out,
                        err,
                        state_marker=marker,
                    ):
                        yield event
                    snapshot = await self.status(job_id)
                except (RelayConnectionError, RelayTimeoutError):
                    # 断线本身不是错误，是这套设计预期内的事件：重连后接着从 offset 读。
                    if getattr(self.transport, "closed", False):
                        raise
                    await asyncio.sleep(1.0)
                    continue
                if snapshot.finished:
                    break
                if not snapshot.running:
                    # 既没在跑也没写下退出码——多半是机器重启把进程带走了。别死循环。
                    break
                await asyncio.sleep(0.5)

        # 收尾补读：tail 被 kill 的瞬间可能还有几个字节没吐出来。
        while True:
            try:
                async for event in self._read_once(
                    self._catchup_script(job_id, out.offset, err.offset),
                    out,
                    err,
                    state_marker=marker,
                    timeout=_CATCHUP_TIMEOUT,
                ):
                    yield event
                snapshot = await self.status(job_id)
                break
            except (RelayConnectionError, RelayTimeoutError):
                if not follow or getattr(self.transport, "closed", False):
                    raise
                await asyncio.sleep(1.0)
        yield ExecEvent(kind="exit", exit_status=snapshot.exit_status)

    async def _read_once(
        self,
        script: str,
        out: _StreamCursor,
        err: _StreamCursor,
        *,
        state_marker: bytes,
        timeout: float | None = None,
    ) -> AsyncIterator[ExecEvent]:
        """跑一次读取脚本，把两路输出解码成事件。游标就地推进。"""
        queue: asyncio.Queue[ExecEvent | None] = asyncio.Queue()
        failure: list[BaseException] = []

        # 状态帧只存在于 stdout。stderr 是用户数据，绝不能因为碰巧包含相似文本而被截断。
        decoders = {
            "stdout": (
                StreamDecoder(start_offset=out.offset, state_marker=state_marker),
                out,
            ),
            "stderr": (StreamDecoder(start_offset=err.offset, strip_state=False), err),
        }

        async def _drive() -> None:
            try:
                await self._raw_pump(script, decoders, queue, timeout=timeout)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # 交给消费端在排空后重新抛，别在半路丢事件
                failure.append(exc)
            finally:
                queue.put_nowait(None)

        task = asyncio.ensure_future(_drive())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if failure:
            raise failure[0]

    async def _raw_pump(
        self,
        script: str,
        decoders: dict[str, tuple[StreamDecoder, _StreamCursor]],
        queue: asyncio.Queue,
        *,
        timeout: float | None = None,
    ) -> None:
        proc = await self.transport.open_process(script)

        async def pump(kind: str, read: Callable[[int], Any]) -> None:
            decoder, cursor = decoders[kind]
            while True:
                chunk = await read(65536)
                if not chunk:
                    break
                text = decoder.feed(chunk)
                cursor.offset = decoder.offset
                if text:
                    queue.put_nowait(ExecEvent(kind=kind, data=text, offset=cursor.offset))  # type: ignore[arg-type]
            tail = decoder.finish()
            cursor.offset = decoder.offset
            if tail:
                queue.put_nowait(ExecEvent(kind=kind, data=tail, offset=cursor.offset))  # type: ignore[arg-type]

        try:
            pumps = asyncio.gather(
                pump("stdout", proc.read_stdout), pump("stderr", proc.read_stderr)
            )
            if timeout is None:
                await pumps
            else:
                try:
                    await asyncio.wait_for(pumps, timeout=timeout)
                except asyncio.TimeoutError as exc:
                    proc.terminate()
                    raise RelayTimeoutError("timed out while catching up job logs") from exc
        finally:
            await proc.close()

    # ── 等待 / 终止 ───────────────────────────────────────────────────────
    async def wait(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        on_event: Callable[[ExecEvent], Any] | None = None,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        terminate_on_timeout: bool = False,
    ) -> ExecResult:
        """等作业结束并收集完整输出。"""
        started = time.monotonic()
        stdout_sink = CappedText(self.max_output_bytes)
        stderr_sink = CappedText(self.max_output_bytes)
        exit_status: int | None = None
        current_stdout_offset = stdout_offset
        current_stderr_offset = stderr_offset

        async def _accept(event: ExecEvent) -> None:
            nonlocal exit_status, current_stdout_offset, current_stderr_offset
            if event.kind == "exit":
                exit_status = event.exit_status
            elif event.kind == "stdout":
                current_stdout_offset = event.offset
                stdout_sink.append(event.data)
            else:
                current_stderr_offset = event.offset
                stderr_sink.append(event.data)
            if on_event is not None:
                callback_result = on_event(event)
                if inspect.isawaitable(callback_result):
                    await callback_result

        async def _consume() -> None:
            async for event in self.stream(
                job_id, stdout_offset=stdout_offset, stderr_offset=stderr_offset, follow=True
            ):
                await _accept(event)

        timed_out = False
        try:
            if timeout is None:
                await _consume()
            else:
                await asyncio.wait_for(_consume(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            if terminate_on_timeout:
                await self._terminate(job_id, sig="TERM", exit_status=TIMEOUT_EXIT_STATUS)
                async for event in self.stream(
                    job_id,
                    stdout_offset=current_stdout_offset,
                    stderr_offset=current_stderr_offset,
                    follow=False,
                ):
                    await _accept(event)
                exit_status = TIMEOUT_EXIT_STATUS
                stderr_sink.append(TIMEOUT_HINT.format(seconds=float(timeout or 0)))

        # 作业已经跑完了，最后这一次状态查询只是收尾。机器在重负载下（pip 装
        # torch 时磁盘和 CPU 都满）一条 45s 的辅助命令也可能超时；真机上这一
        # 下就把整轮 setup 判成 BridgeError 太冤——和 stream() 里一样，有界重试。
        status = await self._status_with_retry(job_id)
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

    async def _status_with_retry(self, job_id: str, *, attempts: int = 4, backoff: float = 2.0) -> JobStatus:
        """`status()`，但对断线/辅助命令超时做有界重试（连接已关闭时立即抛）。"""
        last: Exception | None = None
        for attempt in range(max(attempts, 1)):
            try:
                return await self.status(job_id)
            except (RelayConnectionError, RelayTimeoutError) as exc:
                last = exc
                if getattr(self.transport, "closed", False) or attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(backoff * (attempt + 1))
        raise last if last else RelayError("status unavailable")

    async def kill(self, job_id: str, *, sig: str = "TERM") -> bool:
        """按进程组终止作业，连带它拉起的所有子进程。"""
        name, exit_status = normalize_termination_signal(sig)
        return await self._terminate(job_id, sig=name, exit_status=exit_status)

    async def _terminate(self, job_id: str, *, sig: str, exit_status: int) -> bool:
        d = self._job_dir_expr(job_id)
        script = "\n".join(
            [
                f"d={d}",
                'pgid=$(cat "$d/pgid" 2>/dev/null || true)',
                '__relay_i=0; while [ -z "$pgid" ] && [ ! -f "$d/exit_code" ] '
                '&& [ "$__relay_i" -lt 50 ]; do sleep 0.1; '
                'pgid=$(cat "$d/pgid" 2>/dev/null || true); __relay_i=$((__relay_i + 1)); done',
                'if [ -z "$pgid" ]; then echo "killed=0"; exit 0; fi',
                # 负号 = 整个进程组。训练脚本 fork 出来的 dataloader / 子进程一个都跑不掉。
                f'if kill -{sig} -- "-$pgid" 2>/dev/null; then __relay_killed=1; '
                'echo "killed=1"; else __relay_killed=0; echo "killed=0"; fi',
                # 给 TERM/INT 一点清理时间；仍不退出就 KILL，保证调用返回时进程组确实消失。
                'if [ "$__relay_killed" -eq 1 ]; then',
                '__relay_i=0; while kill -0 -- "-$pgid" 2>/dev/null && [ "$__relay_i" -lt 50 ]; do '
                'sleep 0.1; __relay_i=$((__relay_i + 1)); done',
                'if kill -0 -- "-$pgid" 2>/dev/null; then kill -KILL -- "-$pgid" 2>/dev/null || true; fi',
                '__relay_i=0; while kill -0 -- "-$pgid" 2>/dev/null && [ "$__relay_i" -lt 20 ]; do '
                'sleep 0.1; __relay_i=$((__relay_i + 1)); done',
                # trap 可能先写入 143；这里最后原子覆盖，timeout 才能稳定呈现为 124。
                f'echo {int(exit_status)} > "$d/exit_code.tmp"',
                'mv -f "$d/exit_code.tmp" "$d/exit_code"',
                "fi",
            ]
        )
        result = await self._run(script, timeout=45.0)
        return "killed=1" in result.stdout

    async def list_jobs(self) -> list[str]:
        script = f"ls -1 {self._root_expr()} 2>/dev/null || true"
        result = await self._run(script, timeout=45.0)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    async def cleanup(self, job_id: str) -> None:
        d = self._job_dir_expr(job_id)
        await self._run(f'd={d}\nrm -rf -- "$d"', timeout=45.0)


def _parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def _maybe_int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _decode_bytes_field(value: str | None) -> bytes:
    if not value:
        return b""
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return b""


def _decode_field(value: str | None) -> str:
    return _decode_bytes_field(value).decode("utf-8", errors="replace")
