"""remote_relay：把一台 SSH 远端机器装成"本地"，供 Agent 无感使用。

    from remote_relay import RemoteRuntime

    runtime = RemoteRuntime("ssh -p 2222 root@1.2.3.4", password="…")
    await runtime.wait_ready()
    result = await runtime.exec("nvidia-smi")

`LocalRuntime` 与之同签名，切换只是构造时换一个实现。
"""

from .diagnostics import Tracer
from .execution.shellstate import DEFAULT_BOOTSTRAP, DEFAULT_REMOTE_ENV, ShellState
from .fs import DEFAULT_EXCLUDES, SyncReport
from .local import LocalRuntime
from .remote import RemoteRuntime
from .runtime import Runtime
from .transport import SSHTarget, SSHTransport, parse_access_url
from .types import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    DURABLE_THRESHOLD_SECONDS,
    TIMEOUT_EXIT_STATUS,
    ExecEvent,
    ExecResult,
    FileStat,
    JobStatus,
    RelayAuthError,
    RelayConnectionError,
    RelayError,
    RelayPathError,
    RelayTimeoutError,
    RemoteCommandError,
    RemoteFileNotFoundError,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_EXCLUDES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_REMOTE_ENV",
    "DEFAULT_TIMEOUT_SECONDS",
    "DURABLE_THRESHOLD_SECONDS",
    "ExecEvent",
    "ExecResult",
    "FileStat",
    "JobStatus",
    "LocalRuntime",
    "RelayAuthError",
    "RelayConnectionError",
    "RelayError",
    "RelayPathError",
    "RelayTimeoutError",
    "RemoteCommandError",
    "RemoteFileNotFoundError",
    "RemoteRuntime",
    "Runtime",
    "SSHTarget",
    "SSHTransport",
    "ShellState",
    "SyncReport",
    "TIMEOUT_EXIT_STATUS",
    "Tracer",
    "__version__",
    "parse_access_url",
]
