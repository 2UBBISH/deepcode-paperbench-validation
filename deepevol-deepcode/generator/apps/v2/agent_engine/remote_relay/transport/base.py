"""传输层抽象。

把"怎么开一条远端进程通道"和"怎么读写远端文件"收敛成两个窄接口，上面的 exec / fs
只依赖接口。单测因此可以塞一个 `FakeTransport` 进去，把断线、慢 chunk、半个 UTF-8
字符这些真机上难复现的场景全部编排出来，一次真连接都不用建。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RemoteProcess(Protocol):
    """一条远端进程通道。

    read_stdout / read_stderr 返回 b"" 表示该路 EOF。两路必须能并发读——只读一路而让另一路
    的缓冲写满，会把远端 SSH 流控窗口顶死、进程再也退不出去（现有 server-use 的
    core.py:191 就是为这个坑写的）。
    """

    async def read_stdout(self, n: int = 65536) -> bytes: ...

    async def read_stderr(self, n: int = 65536) -> bytes: ...

    async def wait(self) -> int:
        """等进程退出并返回退出码。"""
        ...

    def terminate(self) -> None:
        """请求终止（best-effort，不抛）。"""

    async def close(self) -> None:
        """关闭通道，释放 channel 名额。幂等。"""


@runtime_checkable
class Transport(Protocol):
    """一条到远端机器的连接（内部可有多路 channel 复用）。"""

    @property
    def connected(self) -> bool: ...

    @property
    def display(self) -> str:
        """给日志用的可读标识。"""
        ...

    async def connect(self) -> None:
        """建立连接。幂等；已连接时直接返回。"""

    async def close(self) -> None:
        """断开连接。幂等。"""

    async def open_process(self, command: str) -> RemoteProcess:
        """开一条进程通道跑 command。会占用一个 channel 名额，直到 close()。"""
        ...

    async def sftp(self) -> Any:
        """返回一个 SFTP 客户端（asyncssh.SFTPClient 兼容接口）。"""
        ...
