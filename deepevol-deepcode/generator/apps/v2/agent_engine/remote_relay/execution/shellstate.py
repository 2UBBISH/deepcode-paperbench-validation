"""粘性 shell 状态：让远端的 cwd/env 像本地 shell 一样跨调用保持。

每条命令都会被包一层，尾部打一个哨兵把 `pwd` 回传：

    …bootstrap…
    cd -- '/root/exp'
    export FOO='bar'
    { <用户命令>
    }
    __relay_rc=$?
    printf '\\n__RELAY_STATE__%s\\n' "$(pwd)"
    exit $__relay_rc

于是 Agent 先 `cd /root/exp` 再 `python train.py`，第二条命令确实跑在 /root/exp 下——
跟本地 shell 一个手感。哨兵由 `stream.SentinelSplitter` 在流式路径上就地剥掉。

bootstrap 和默认 env 沿用现有 server-use 的经验值（core.py:28/135/140）：conda 的 PATH、
AutoDL 的学术加速、pip/HF 国内镜像。不在 AutoDL 上时这些都是安全的空操作。
"""

from __future__ import annotations

import shlex
import time
import uuid
from dataclasses import dataclass, field


DEFAULT_BOOTSTRAP = "\n".join(
    [
        "export PATH=/root/miniconda3/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:$PATH",
        # AutoDL 的「学术资源加速」——存在就 source，能大幅加速 github/HF/pip 的对外下载。
        "[ -f /etc/network_turbo ] && . /etc/network_turbo >/dev/null 2>&1",
        # 把本机地址排除出代理，别误伤本地回环连接。
        'export no_proxy="127.0.0.1,localhost,::1,${no_proxy}" '
        'NO_PROXY="127.0.0.1,localhost,::1,${NO_PROXY}"',
    ]
)

DEFAULT_REMOTE_ENV: dict[str, str] = {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "PIP_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/",
    # 远端是非交互会话，apt 弹配置界面会直接把命令挂死。
    "DEBIAN_FRONTEND": "noninteractive",
}

LEGACY_STATE_MARKER = b"\n__RELAY_STATE__"


def new_state_marker() -> bytes:
    """生成一次运行专用的状态帧前缀，避免用户输出与固定哨兵意外碰撞。"""
    return b"\n__RELAY_STATE__" + uuid.uuid4().hex.encode("ascii") + b"__"


def state_print(marker: bytes) -> str:
    value = bytes(marker).decode("ascii")
    return f"printf '%s%s\\n' {shlex.quote(value)} \"$(pwd)\""

# 作业脚本的收尾必须走 trap EXIT，不能顺序写在命令后面：用户命令里一句 `exit 9` 会让 shell
# 当场退出，顺序写法就永远轮不到写 exit_code，作业在轮询方眼里会变成"人间蒸发"。
# TERM 也转成一次正常 exit，这样被 kill 的作业同样会留下退出码（143），流读方不会一直等。
JOB_STARTUP = 'echo $$ > "$RELAY_JOB_DIR/pgid"'
# pgid 还没写出来时，状态查询无从判断死活。给一个启动宽限期，超过就认定作业根本没起来。
JOB_STARTUP_GRACE_SECONDS = 30.0


def job_epilogue(marker: bytes) -> str:
    return "\n".join(
        [
            "__relay_finish() {",
            "  __relay_rc=$?",
            '  __relay_cwd="$(pwd)"',
            '  printf "%s\\n" "$__relay_cwd" > "$RELAY_JOB_DIR/cwd.tmp"',
            '  mv -f "$RELAY_JOB_DIR/cwd.tmp" "$RELAY_JOB_DIR/cwd"',
            f"  {state_print(marker)}",
            '  echo "$__relay_rc" > "$RELAY_JOB_DIR/exit_code.tmp"',
            '  mv -f "$RELAY_JOB_DIR/exit_code.tmp" "$RELAY_JOB_DIR/exit_code"',
            "}",
            "trap __relay_finish EXIT",
            "trap 'exit 143' TERM",
        ]
    )


def wrap_job_script(body: str, *, state_marker: bytes = LEGACY_STATE_MARKER) -> str:
    """把已经 build 好的命令体包成一个耐久作业脚本。本地与远端共用，保证两边语义一致。"""
    return "\n".join(["#!/bin/bash", JOB_STARTUP, job_epilogue(state_marker), body])


def job_is_starting(exit_status: int | None, pgid: int | None, started_at: float) -> bool:
    """作业刚拉起、还没写下 pgid 的窗口。这段时间内不能判定成"已消失"。"""
    if exit_status is not None or pgid:
        return False
    if not started_at:
        return True
    return (time.time() - started_at) < JOB_STARTUP_GRACE_SECONDS


@dataclass
class ShellState:
    """一台远端机器上的"当前 shell"。"""

    cwd: str = ""
    """空串表示用远端默认目录（通常是 $HOME）。"""
    env: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_REMOTE_ENV))
    bootstrap: str = DEFAULT_BOOTSTRAP

    def build(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        track_state: bool = True,
        state_marker: bytes = LEGACY_STATE_MARKER,
    ) -> str:
        """把用户命令包成可以直接丢给远端 shell 的完整脚本。

        cwd/env 是本次调用的一次性覆盖，不写回粘性状态；粘性状态只由哨兵回传更新。
        """
        command = str(command or "").strip()
        if not command:
            raise ValueError("command is required")

        lines: list[str] = []
        if self.bootstrap:
            lines.append(self.bootstrap)

        effective_cwd = cwd if cwd is not None else self.cwd
        if effective_cwd:
            quoted = shlex.quote(effective_cwd)
            lines.append(
                f"cd -- {quoted} || {{ echo '[remote-relay] cd failed: '{quoted} >&2; exit 1; }}"
            )

        merged: dict[str, str] = dict(self.env)
        if env:
            merged.update(env)
        for key, value in merged.items():
            if value is None:
                continue
            lines.append(f"export {key}={shlex.quote(str(value))}")

        # 用 { } 包住用户命令，保证多行/带 & 的命令也能整体拿到退出码。
        lines.append("{ " + command + "\n}")
        if track_state:
            lines.append("__relay_rc=$?")
            lines.append(state_print(state_marker))
            lines.append("exit $__relay_rc")
        return "\n".join(lines)

    def observe(self, payload: str) -> None:
        """吃下哨兵回传的载荷，更新粘性 cwd。"""
        candidate = str(payload or "").strip()
        if candidate.startswith("/"):
            self.cwd = candidate

    def set_env(self, **values: str) -> None:
        self.env.update({k: str(v) for k, v in values.items() if v is not None})
