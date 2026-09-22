"""快路径执行：短命令直接挂在一条 SSH channel 上跑完。

`ls` / `cat` / `pip -V` 这类命令走这里，只有一个往返。真正会长跑的东西请走 jobs.py 的
耐久作业——那边 SSH 断了也不丢。

这里有两个从现有 server-use 继承来的教训，改动时务必保留：

1. **stdout 和 stderr 必须并发排空。** 只读一路，另一路的缓冲写满后会把远端 SSH 流控窗口
   顶死，远端进程再也写不出去、也就永远退不出——表现为命令莫名其妙挂住
   （core.py:191 那段注释）。
2. **墙钟超时不能依赖"等进程退出"这个动作本身。** 得有独立的 deadline，到点就主动掐通道，
   否则一条跑飞的命令能把 Agent 卡到天荒地老（core.py:196）。
"""

from __future__ import annotations

import asyncio
import inspect
import shlex
import time
import uuid
from typing import Any, Callable

from ..types import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    ExecEvent,
    ExecResult,
    TIMEOUT_EXIT_STATUS,
)
from .stream import STATE_MARKER, CappedText, StreamDecoder


EventCallback = Callable[[ExecEvent], Any]

_REMOTE_TIMEOUT_PAYLOAD = "__REMOTE_RELAY_TIMEOUT__"

# 超时提示原样保留 core.py:209 的措辞：模型看到 124 时最容易犯的错是判定"环境坏了"然后
# 去乱改代码，实际上多半只是训练/编译本来就慢。这段话是防止它走上歧路的。
TIMEOUT_HINT = (
    "\n[remote-relay] 命令超过 {seconds:.0f}s 墙钟超时，已终止远端进程组。"
    "\n[remote-relay] 若是训练/编译/大包安装等重命令的正常耗时，请改用耐久作业"
    "（spawn/后台运行再轮询日志），或把 timeout 调大；不要据此判定环境或代码出错。"
)


def _wrap_remote_timeout(script: str, timeout: float, marker: bytes) -> str:
    """让远端自己执行墙钟超时，避免依赖 OpenSSH 不可靠的 signal request。

    OpenSSH 只给 channel 对应的 shell 发 TERM 时，其子进程可能继续持有 stdout/stderr，随后
    客户端会永久卡在 wait_closed()。GNU timeout 在远端建立并终止整个命令进程组；状态文件
    用来区分“用户命令自己返回 124”和“watchdog 真正触发超时”。
    """
    timeout_s = max(0.001, float(timeout))
    # 给 timeout 的 TERM/KILL 与 shell 收尾留一小段预算，保证客户端墙钟 deadline 先不触发。
    reserve = min(0.25, max(0.02, timeout_s * 0.02))
    remote_limit = max(0.001, timeout_s - reserve)
    kill_after = max(0.01, reserve / 2)
    status_path = f"/tmp/.remote-relay-fast-{uuid.uuid4().hex}.status"
    marker_text = marker.decode("ascii")

    runner = "\n".join(
        [
            '/bin/bash -c "$1"',
            "__relay_rc=$?",
            'printf "%s\\n" "$__relay_rc" > "$2"',
            'exit "$__relay_rc"',
        ]
    )
    return "\n".join(
        [
            "umask 077",
            f"__relay_status={shlex.quote(status_path)}",
            'rm -f -- "$__relay_status"',
            "timeout --signal=TERM "
            f"--kill-after={kill_after:.3f}s {remote_limit:.3f}s "
            f"/bin/bash -c {shlex.quote(runner)} relay-fast "
            f"{shlex.quote(script)} \"$__relay_status\"",
            "__relay_timeout_rc=$?",
            'if [ -f "$__relay_status" ]; then',
            '  __relay_rc=$(cat "$__relay_status" 2>/dev/null || echo 1)',
            '  rm -f -- "$__relay_status"',
            '  exit "$__relay_rc"',
            "fi",
            'rm -f -- "$__relay_status"',
            'if [ "$__relay_timeout_rc" -eq 124 ] || '
            '[ "$__relay_timeout_rc" -eq 137 ]; then',
            f"  printf '%s%s\\n' {shlex.quote(marker_text)} "
            f"{shlex.quote(_REMOTE_TIMEOUT_PAYLOAD)}",
            "  exit 124",
            "fi",
            'exit "$__relay_timeout_rc"',
        ]
    )


async def _emit(callback: EventCallback | None, event: ExecEvent) -> None:
    if callback is None:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result


async def _pump(
    read: Callable[[int], Any],
    kind: str,
    *,
    decoder: StreamDecoder,
    sink: CappedText,
    callback: EventCallback | None,
) -> None:
    """把一路输出读到 EOF，边读边解码、边推事件。"""
    while True:
        chunk = await read(65536)
        if not chunk:
            break
        text = decoder.feed(chunk)
        if text:
            sink.append(text)
            await _emit(callback, ExecEvent(kind=kind, data=text, offset=decoder.offset))  # type: ignore[arg-type]
    tail = decoder.finish()
    if tail:
        sink.append(tail)
        await _emit(callback, ExecEvent(kind=kind, data=tail, offset=decoder.offset))  # type: ignore[arg-type]


async def run_fast(
    transport: Any,
    script: str,
    *,
    command: str = "",
    cwd: str = "",
    timeout: float | None = None,
    on_event: EventCallback | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    track_state: bool = True,
    state_marker: bytes = STATE_MARKER,
    remote_watchdog: bool = False,
) -> tuple[ExecResult, str]:
    """跑一条命令并等它结束。返回 (结果, 哨兵回传的状态载荷)。"""
    timeout_s = float(timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS)
    started = time.monotonic()

    if remote_watchdog:
        script = _wrap_remote_timeout(script, timeout_s, state_marker)

    stdout_decoder = StreamDecoder(
        strip_state=track_state or remote_watchdog,
        state_marker=state_marker,
    )
    stderr_decoder = StreamDecoder(strip_state=False)
    stdout_sink = CappedText(max_output_bytes)
    stderr_sink = CappedText(max_output_bytes)

    proc = await transport.open_process(script)
    timed_out = False
    exit_status = 1
    try:
        pumps = asyncio.gather(
            _pump(
                proc.read_stdout,
                "stdout",
                decoder=stdout_decoder,
                sink=stdout_sink,
                callback=on_event,
            ),
            _pump(
                proc.read_stderr,
                "stderr",
                decoder=stderr_decoder,
                sink=stderr_sink,
                callback=on_event,
            ),
        )
        try:
            await asyncio.wait_for(pumps, timeout=timeout_s)
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise asyncio.TimeoutError
            exit_status = await asyncio.wait_for(
                proc.wait(), timeout=remaining
            )
        except asyncio.TimeoutError:
            timed_out = True
            proc.terminate()
            # 掐掉通道后两路会很快 EOF；给一点缓冲收尾，收不完就算了，已读到的照样返回。
            pumps.cancel()
            try:
                await asyncio.wait_for(pumps, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            exit_status = TIMEOUT_EXIT_STATUS
            stderr_sink.append(TIMEOUT_HINT.format(seconds=timeout_s))
    finally:
        await proc.close()

    state_payload = stdout_decoder.state_payload
    if remote_watchdog and state_payload == _REMOTE_TIMEOUT_PAYLOAD:
        if not timed_out:
            stderr_sink.append(TIMEOUT_HINT.format(seconds=timeout_s))
        timed_out = True
        exit_status = TIMEOUT_EXIT_STATUS
        state_payload = ""

    await _emit(on_event, ExecEvent(kind="exit", exit_status=exit_status))
    result = ExecResult(
        exit_status=exit_status,
        stdout=stdout_sink.value(),
        stderr=stderr_sink.value(),
        duration_ms=int((time.monotonic() - started) * 1000),
        command=command or script,
        cwd=cwd,
        timed_out=timed_out,
        stdout_truncated=stdout_sink.truncated,
        stderr_truncated=stderr_sink.truncated,
    )
    return result, state_payload
