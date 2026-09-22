"""基于 asyncssh 的 SSH 传输：连接复用、keepalive、自动重连、channel 名额限流。

为什么不用 paramiko：paramiko 是同步的，流式要靠 `recv_ready()` 自己轮询，还得手写防死锁
（现有 server-use 的 core.py 就是这么干的）。asyncssh 原生 async、单连接多 channel 复用、
内建 keepalive、SFTP 也是异步的——正好是这一层需要的全部东西。
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import asyncssh

from ..types import (
    RelayAuthError,
    RelayConnectionError,
    TERMINATED_EXIT_STATUS,
)
from .target import SSHTarget


logger = logging.getLogger("remote_relay.ssh")

# 重连退避：首次 1s，每次翻倍，封顶 20s，带 ±20% 抖动（避免多台机器同时重连打同一个网关）。
_RECONNECT_ATTEMPTS = 5
_RECONNECT_BASE_SECONDS = 1.0
_RECONNECT_MAX_SECONDS = 20.0
_PROCESS_CLOSE_TIMEOUT_SECONDS = 1.0


def _backoff_delay(attempt: int) -> float:
    base = min(_RECONNECT_BASE_SECONDS * (2**attempt), _RECONNECT_MAX_SECONDS)
    return base * random.uniform(0.8, 1.2)


class SSHProcess:
    """asyncssh 进程通道的薄封装。

    唯一的额外职责是把 channel 名额（信号量）在 close() 时还回去——忘了还，跑到第 9 条命令
    就会静静地卡住，而 sshd 那边不会给任何错误提示。
    """

    def __init__(self, proc: Any, release: Any) -> None:
        self._proc = proc
        self._release = release
        self._closed = False

    async def read_stdout(self, n: int = 65536) -> bytes:
        return await self._read(self._proc.stdout, n)

    async def read_stderr(self, n: int = 65536) -> bytes:
        return await self._read(self._proc.stderr, n)

    async def _read(self, reader: Any, n: int) -> bytes:
        # 把 asyncssh 的断线异常统一翻译成 RelayConnectionError。耐久作业的续读循环靠这个
        # 类型来区分"网络断了，重连接着读"和"真出错了"——不翻译的话断线会被当成致命错误。
        try:
            return await reader.read(n)
        except asyncssh.ConnectionLost as exc:
            raise RelayConnectionError(f"connection lost while reading: {exc}") from exc
        except (asyncssh.DisconnectError, ConnectionError, EOFError) as exc:
            raise RelayConnectionError(f"connection dropped while reading: {exc}") from exc

    async def wait(self) -> int:
        # 用 wait_closed() 而不是 wait()：wait() 内部会去收集 stdout/stderr，与我们并发的
        # read_stdout/read_stderr 抢同一个 reader。这里只等通道关闭，输出由调用方自己排空。
        try:
            await self._proc.wait_closed()
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError) as exc:
            raise RelayConnectionError(f"connection lost while waiting for exit: {exc}") from exc
        status = self._proc.exit_status
        if status is None:
            # 被信号打死时 exit_status 为 None，asyncssh 把信号名放在 exit_signal 里。
            signal_info = getattr(self._proc, "exit_signal", None)
            return TERMINATED_EXIT_STATUS if signal_info else 1
        return int(status)

    def terminate(self) -> None:
        try:
            self._proc.terminate()
        except Exception:  # best-effort：通道可能已经没了
            pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.close()
            try:
                await asyncio.wait_for(
                    self._proc.wait_closed(), timeout=_PROCESS_CLOSE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                # OpenSSH 的 signal request 可能只杀外层 shell，子进程继续持有通道。强制
                # abort 后必须立即归还 semaphore；远端 watchdog 会负责杀完整进程组。
                channel = getattr(self._proc, "channel", None)
                if channel is not None:
                    channel.abort()
        except Exception:
            pass
        finally:
            self._release()


class SSHTransport:
    """一条到远端机器的连接。多路 channel 在其上复用。"""

    def __init__(self, target: SSHTarget) -> None:
        self.target = target
        self._conn: Any = None
        self._sftp: Any = None
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max(1, target.max_sessions))
        self._closed = False
        self.generation = 0
        """每次成功（重）连接 +1。上层据此判断"这中间断过"，从而决定要不要按 offset 续读。"""

    # ── 属性 ──────────────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        conn = self._conn
        return conn is not None and not conn.is_closed()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def display(self) -> str:
        return self.target.display

    # ── 连接管理 ──────────────────────────────────────────────────────────
    def _connect_options(self) -> dict[str, Any]:
        target = self.target
        options: dict[str, Any] = {
            "port": target.port,
            "username": target.username,
            "known_hosts": target.known_hosts,  # None = 不校验；租来的临时机器指纹每次都不同
            "connect_timeout": target.connect_timeout,
            "keepalive_interval": target.keepalive_interval,
            "keepalive_count_max": target.keepalive_count_max,
        }
        if target.password:
            options["password"] = target.password

        client_keys: list[Any] = []
        if target.private_key:
            client_keys.append(
                asyncssh.import_private_key(target.private_key, target.passphrase)
            )
        if target.private_key_path:
            client_keys.append(target.private_key_path)
        if client_keys:
            options["client_keys"] = client_keys
        elif target.password:
            # 显式给了密码就别再去试 ssh-agent / ~/.ssh 里的一堆 key：那些 key 会先被 sshd
            # 拒绝若干次，撞上 MaxAuthTries 后连密码都还没轮到就被踢掉。
            options["client_keys"] = []
        return options

    async def connect(self) -> None:
        """建立连接。幂等；已连上直接返回。失败按退避重试，耗尽抛 RelayConnectionError。"""
        if self._closed:
            raise RelayConnectionError(f"transport already closed: {self.display}")
        if self.connected:
            return
        async with self._lock:
            if self.connected:
                return
            await self._connect_locked()

    async def _connect_locked(self) -> None:
        last_error: Exception | None = None
        for attempt in range(_RECONNECT_ATTEMPTS):
            if attempt:
                await asyncio.sleep(_backoff_delay(attempt - 1))
            try:
                self._conn = await asyncssh.connect(self.target.host, **self._connect_options())
                self._sftp = None
                self.generation += 1
                logger.info(
                    "remote-relay connected to %s (generation=%d)", self.display, self.generation
                )
                return
            except asyncssh.PermissionDenied as exc:
                # 认证失败重试没有意义——换凭据才有用，别在这儿耗 5 轮退避。
                raise RelayAuthError(f"SSH authentication failed for {self.display}: {exc}") from exc
            except asyncssh.HostKeyNotVerifiable as exc:
                raise RelayAuthError(f"SSH host key not verifiable for {self.display}: {exc}") from exc
            except (OSError, asyncssh.Error, asyncio.TimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "remote-relay connect attempt %d/%d to %s failed: %s: %s",
                    attempt + 1,
                    _RECONNECT_ATTEMPTS,
                    self.display,
                    type(exc).__name__,
                    exc,
                )
        raise RelayConnectionError(
            f"cannot connect to {self.display} after {_RECONNECT_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    async def _reconnect(self) -> None:
        async with self._lock:
            conn = self._conn
            if conn is not None and not conn.is_closed():
                return  # 别人已经重连好了
            self._conn = None
            self._sftp = None
            await self._connect_locked()

    async def close(self) -> None:
        """永久关闭 transport。关闭后的对象不能再次连接。"""
        self._closed = True
        await self.disconnect()

    async def disconnect(self) -> None:
        """只掐掉当前连接，保留后续自动重连能力。

        真网络中断与测试里的强制断线都应走这里；close() 只用于 Runtime 生命周期结束。
        """
        async with self._lock:
            sftp, self._sftp = self._sftp, None
            conn, self._conn = self._conn, None
        if sftp is not None:
            try:
                sftp.exit()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
                await conn.wait_closed()
            except Exception:
                pass

    # ── 通道 ──────────────────────────────────────────────────────────────
    async def open_process(self, command: str) -> SSHProcess:
        """开一条进程通道。连接断了会自动重连一次再试。"""
        await self.connect()
        await self._sem.acquire()
        released = False

        def _release() -> None:
            nonlocal released
            if not released:
                released = True
                self._sem.release()

        try:
            for attempt in range(2):
                try:
                    proc = await self._conn.create_process(command, encoding=None)
                    return SSHProcess(proc, _release)
                except (asyncssh.ChannelOpenError, asyncssh.ConnectionLost, OSError) as exc:
                    if attempt == 0:
                        logger.warning(
                            "remote-relay channel open failed on %s (%s), reconnecting",
                            self.display,
                            type(exc).__name__,
                        )
                        await self._reconnect()
                        continue
                    raise RelayConnectionError(
                        f"cannot open channel on {self.display}: {type(exc).__name__}: {exc}"
                    ) from exc
        except BaseException:
            _release()
            raise
        raise RelayConnectionError(f"cannot open channel on {self.display}")  # pragma: no cover

    async def sftp(self) -> Any:
        """返回复用的 SFTP 客户端。连接断了会重建。"""
        await self.connect()
        if self._sftp is not None and self.connected:
            return self._sftp
        async with self._lock:
            if not self.connected:
                self._conn = None
                self._sftp = None
                await self._connect_locked()
            if self._sftp is None:
                self._sftp = await self._conn.start_sftp_client()
            return self._sftp

    async def reset_sftp(self) -> None:
        """SFTP 通道出错后强制重建（下次 sftp() 会新开一条）。"""
        async with self._lock:
            sftp, self._sftp = self._sftp, None
        if sftp is not None:
            try:
                sftp.exit()
            except Exception:
                pass
