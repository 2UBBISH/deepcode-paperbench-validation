"""JSONL 追踪：把每次远端操作（命令、退出码、耗时、字节数）落一行。

字段沿用现有 server-use 的 `.remote-ssh-debug.jsonl` 形状（core.py:892），这样已有的调试
面板不用改就能读。全程 best-effort——写日志失败绝不影响主流程。

    REMOTE_RELAY_TRACE=0        关闭
    REMOTE_RELAY_TRACE_LOG=...  指定落盘位置（默认 ~/.remote-relay/trace.jsonl）
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


_MAX_BYTES = 8 * 1024 * 1024
_FIELD_LIMIT = 6000


def _default_path() -> Path:
    override = os.environ.get("REMOTE_RELAY_TRACE_LOG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".remote-relay" / "trace.jsonl"


def _enabled_by_env() -> bool:
    value = os.environ.get("REMOTE_RELAY_TRACE")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "off", "no"}


class Tracer:
    def __init__(self, path: str | Path | None = None, *, enabled: bool | None = None) -> None:
        self.path = Path(path).expanduser() if path else _default_path()
        self.enabled = _enabled_by_env() if enabled is None else bool(enabled)

    def record(
        self,
        kind: str,
        *,
        target: str = "",
        command: str = "",
        exit_status: int | None = None,
        stdout: str = "",
        stderr: str = "",
        ms: int | None = None,
        **extra: object,
    ) -> None:
        if not self.enabled:
            return
        try:
            entry = {
                "ts": time.time(),
                "kind": kind,
                "target": target,
                "command": str(command)[:8000],
                "exit_status": exit_status,
                "stdout": (stdout or "")[-_FIELD_LIMIT:],
                "stderr": (stderr or "")[-_FIELD_LIMIT:],
                "ms": ms,
            }
            entry.update(extra)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            if self.path.stat().st_size > _MAX_BYTES:
                tail = self.path.read_text(encoding="utf-8", errors="ignore").splitlines()[-2000:]
                self.path.write_text("\n".join(tail) + "\n", encoding="utf-8")
        except Exception:
            pass  # 追踪永远不该把主流程带崩
