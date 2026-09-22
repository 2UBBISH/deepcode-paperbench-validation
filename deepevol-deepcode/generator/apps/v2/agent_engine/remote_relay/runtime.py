"""`Runtime`：本地与远端共用的同一组签名。

这是整个中转层的"无感"落点——Agent 的工具层只认这个协议，跑在本机还是跑在租来的
远端机器上，只是构造时换一个实现的事。任何新增能力都必须两边同时实现，否则就漏了。
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Protocol, runtime_checkable

from .types import ExecEvent, ExecResult, FileStat, JobStatus


EventCallback = Callable[[ExecEvent], None]


@runtime_checkable
class Runtime(Protocol):
    """一台"机器"的抽象：能跑命令、能读写文件、能带耐久作业。"""

    # ── 生命周期 ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        """建立底层连接（本地实现是空操作）。幂等。"""

    async def close(self) -> None:
        """释放连接。幂等。"""

    async def wait_ready(self, *, timeout: float = 300.0) -> None:
        """轮询直到这台机器真的能跑命令。

        专门伺候刚租来的机器还在 cloud-init、sshd 没起来的窗口。
        """

    # ── 执行 ──────────────────────────────────────────────────────────────
    @property
    def cwd(self) -> str:
        """当前工作目录。像本地 shell 一样，`cd` 会跨调用粘住。"""
        ...

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
        """跑一条命令并等它结束。

        on_event 非空时按 chunk 实时回推 `ExecEvent`——训练脚本几小时里吐的输出会边跑边到，
        而不是等命令结束才一次性返回。

        durable=True 强制走耐久作业（远端落日志文件，SSH 断了也不丢）；None 表示由实现按
        timeout 自行判断。
        """
        ...

    # ── 耐久作业：长实验的正式入口 ────────────────────────────────────────
    async def spawn(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        """把命令作为后台作业启动，立刻返回。进程独立于本次连接存活。"""
        ...

    def stream(
        self,
        job_id: str,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        follow: bool = True,
    ) -> AsyncIterator[ExecEvent]:
        """从给定字节偏移续读作业输出。

        断线重连、Agent 重启都只是"换个 offset 再调一次"——不丢也不重。
        follow=False 表示读到当前末尾就停，不等新输出。
        """
        ...

    async def job_status(self, job_id: str) -> JobStatus: ...

    async def wait(self, job_id: str, *, timeout: float | None = None) -> ExecResult:
        """等作业结束并收集完整输出。"""
        ...

    async def kill(self, job_id: str, *, sig: str = "TERM") -> bool:
        """按进程组杀掉作业（连带它拉起的所有子进程）。"""
        ...

    # ── 文件系统 ──────────────────────────────────────────────────────────
    async def read_file(
        self, path: str, *, encoding: str = "utf-8", max_bytes: int | None = None
    ) -> str: ...

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes: ...

    async def write_file(
        self, path: str, content: str, *, encoding: str = "utf-8", parents: bool = True
    ) -> None: ...

    async def write_bytes(self, path: str, content: bytes, *, parents: bool = True) -> None: ...

    async def append_file(self, path: str, content: str, *, encoding: str = "utf-8") -> None: ...

    async def ls(self, path: str = ".") -> list[FileStat]: ...

    async def glob(self, pattern: str) -> list[str]: ...

    async def stat(self, path: str) -> FileStat: ...

    async def exists(self, path: str) -> bool: ...

    async def mkdir(self, path: str, *, parents: bool = True) -> None: ...

    async def rm(self, path: str, *, recursive: bool = False, missing_ok: bool = True) -> None: ...

    async def move(self, src: str, dst: str) -> None: ...

    # ── 本地 ↔ 这台机器的文件搬运 ─────────────────────────────────────────
    async def upload(self, local_path: str, remote_path: str, *, parents: bool = True) -> None:
        """把本地文件送上去。LocalRuntime 上退化成一次拷贝。"""
        ...

    async def download(self, remote_path: str, local_path: str, *, parents: bool = True) -> None:
        """把这台机器上的文件取回本地。"""
        ...
