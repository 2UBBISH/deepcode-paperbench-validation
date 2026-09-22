"""远端中转层的公共数据类型与错误家族。

这里的类型是 Agent 唯一会碰到的东西：`LocalRuntime` 和 `RemoteRuntime` 都返回同一组
结构，所以上层代码在"本地跑"和"远端跑"之间切换时不需要改一行。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


# 墙钟超时约定的退出码（与 GNU timeout 一致，也与现有 server-use 对齐）。
TIMEOUT_EXIT_STATUS = 124
# 命令被信号终止时的约定退出码（128 + SIGTERM）。
TERMINATED_EXIT_STATUS = 143
# 单条命令每路输出的默认上限：超出后只保留 head+tail，防止远端洪泛撑爆 Agent 上下文。
DEFAULT_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
# 单条命令的默认墙钟超时（5 分钟）。超过这个量级的命令会自动转成耐久作业，
# 因为"挂在一条 SSH channel 上跑半小时"本身就是个不可靠的赌注。
DEFAULT_TIMEOUT_SECONDS = 300.0
# exec() 自动转耐久作业的阈值：要的超时比这个长，就说明它可能长到会撞上断线。
DURABLE_THRESHOLD_SECONDS = 300.0

# kill() 只接受确实会终止进程的信号。除了避免把任意文本插进远端 shell，这也防止调用方用
# STOP/CONT 之类非终止信号，却让作业被错误标记成 finished。
TERMINATION_SIGNALS: dict[str, int] = {
    "HUP": 129,
    "INT": 130,
    "QUIT": 131,
    "KILL": 137,
    "TERM": TERMINATED_EXIT_STATUS,
}


def normalize_termination_signal(sig: str) -> tuple[str, int]:
    name = str(sig or "").strip().upper().removeprefix("SIG")
    if name not in TERMINATION_SIGNALS:
        allowed = ", ".join(TERMINATION_SIGNALS)
        raise ValueError(f"unsupported termination signal {sig!r}; expected one of: {allowed}")
    return name, TERMINATION_SIGNALS[name]


EventKind = Literal["stdout", "stderr", "exit"]


@dataclass(frozen=True)
class ExecEvent:
    """流式输出的一个增量事件。

    offset 是该路输出在**解码前的字节流**中的位置（data 结束处），而不是字符数——耐久作业
    的断线续读就是拿它当游标去 `tail -c +<offset>`，必须是字节。
    """

    kind: EventKind
    data: str = ""
    offset: int = 0
    exit_status: int | None = None
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ExecResult:
    """一条命令跑完的完整结果。"""

    exit_status: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    command: str = ""
    cwd: str = ""
    job_id: str | None = None
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def success(self) -> bool:
        return self.exit_status == 0

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated

    @property
    def output(self) -> str:
        """stdout + stderr 的合并视图，方便直接塞给模型看。"""
        if not self.stderr:
            return self.stdout
        if not self.stdout:
            return self.stderr
        return f"{self.stdout}\n{self.stderr}"


@dataclass(frozen=True)
class FileStat:
    path: str
    name: str
    is_dir: bool = False
    is_symlink: bool = False
    size: int = 0
    mtime: float = 0.0
    mode: int = 0


@dataclass(frozen=True)
class JobStatus:
    """耐久作业的状态快照。

    running=False 且 exit_status=None 表示作业目录还在、但进程既不在跑也没写下退出码——
    通常是机器重启把进程带走了，上层该按"结果不可知"处理，而不是当成成功。
    """

    job_id: str
    running: bool
    exit_status: int | None = None
    pgid: int | None = None
    command: str = ""
    cwd: str = ""
    started_at: float = 0.0
    stdout_size: int = 0
    stderr_size: int = 0
    # 每个作业独有的哨兵。它随作业落盘，Agent 重启后仍能安全剥离状态帧。
    state_marker: str = field(default="", repr=False)

    @property
    def finished(self) -> bool:
        return self.exit_status is not None


# ── 错误分层 ────────────────────────────────────────────────────────────────
# 分两族，因为上层的处置完全不同：
#   RelayConnectionError 家族 = 基础设施问题（网络/认证/机器没起来），该重试或换机器，
#     绝不能把它当成"实验失败"回给模型让它瞎改代码；
#   RemoteCommandError    = 远端命令退出码非 0，这是正常业务结果，原样回给模型。


class RelayError(Exception):
    """中转层的错误基类。"""


class RelayConnectionError(RelayError):
    """连不上 / 连接中断 / 重连耗尽——基础设施级，可重试。"""


class RelayAuthError(RelayConnectionError):
    """认证失败。重试没有意义，得换凭据。"""


class RelayTimeoutError(RelayError):
    """中转层自身的操作超时（如 SFTP 卡住）。命令的墙钟超时不走这里，走退出码 124。"""


class RemoteCommandError(RelayError):
    """远端命令返回非 0。只有 `check_exec()` 这类显式要求成功的入口才抛。"""

    def __init__(self, result: ExecResult) -> None:
        super().__init__(
            f"command exited with {result.exit_status}: {result.command}\n{result.output}"
        )
        self.result = result


class RemoteFileNotFoundError(RelayError):
    """远端路径不存在。"""


class RelayPathError(RelayError):
    """路径越界或非法（如逃逸出 workspace 根）。"""
