"""远端文件系统：read/write/ls/glob/stat/mkdir/rm/move，全部走 SFTP。

远端是唯一真源——本地不留副本、不做镜像。所以 Agent 的"读文件"就是一次 SFTP 往返
（到阿里云约 20-80ms），换来的是不会有任何"两边不一致"的窗口。

相对路径按粘性 cwd 解析，`~` 按远端 home 展开，跟在本地 shell 里的手感一致。
"""

from __future__ import annotations

import posixpath
import stat as stat_module
from typing import Any

import asyncssh

from ..types import (
    RelayConnectionError,
    RelayError,
    RemoteFileNotFoundError,
)
from ..execution.shellstate import ShellState


class RemoteFileSystem:
    """一台远端机器上的文件系统视图。"""

    def __init__(self, transport: Any, state: ShellState) -> None:
        self.transport = transport
        self.state = state
        self._home: str | None = None

    # ── 路径 ──────────────────────────────────────────────────────────────
    async def home(self) -> str:
        if self._home is None:
            sftp = await self.transport.sftp()
            self._home = str(await sftp.realpath("."))
        return self._home

    async def resolve(self, path: str) -> str:
        """把 Agent 给的路径变成远端绝对路径。"""
        raw = str(path or "").strip()
        if not raw:
            raw = "."
        if raw == "~" or raw.startswith("~/"):
            return posixpath.normpath(posixpath.join(await self.home(), raw[2:] if len(raw) > 1 else ""))
        if raw.startswith("/"):
            return posixpath.normpath(raw)
        base = self.state.cwd or await self.home()
        return posixpath.normpath(posixpath.join(base, raw))

    # ── 内部 ──────────────────────────────────────────────────────────────
    async def _sftp(self) -> Any:
        return await self.transport.sftp()

    async def _guard(self, coro: Any, *, path: str = "") -> Any:
        try:
            return await coro
        except asyncssh.SFTPNoSuchFile as exc:
            raise RemoteFileNotFoundError(f"remote path not found: {path}") from exc
        except asyncssh.SFTPPermissionDenied as exc:
            raise RelayError(f"remote permission denied: {path}: {exc}") from exc
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError) as exc:
            # SFTP 通道跟着连接一起没了，标记重建，下次调用会新开一条。
            await self.transport.reset_sftp()
            raise RelayConnectionError(f"connection lost during SFTP op on {path}: {exc}") from exc
        except asyncssh.SFTPError as exc:
            raise RelayError(f"remote SFTP error on {path}: {exc}") from exc

    # ── 读 ────────────────────────────────────────────────────────────────
    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        target = await self.resolve(path)
        sftp = await self._sftp()
        handle = await self._guard(sftp.open(target, "rb"), path=target)
        try:
            if max_bytes is None:
                return await self._guard(handle.read(), path=target)
            return await self._guard(handle.read(max_bytes), path=target)
        finally:
            try:
                await handle.close()
            except Exception:
                pass

    async def read_file(
        self, path: str, *, encoding: str = "utf-8", max_bytes: int | None = None
    ) -> str:
        raw = await self.read_bytes(path, max_bytes=max_bytes)
        return raw.decode(encoding, errors="replace")

    # ── 写 ────────────────────────────────────────────────────────────────
    async def write_bytes(self, path: str, content: bytes, *, parents: bool = True) -> None:
        target = await self.resolve(path)
        if parents:
            await self.mkdir(posixpath.dirname(target) or "/", parents=True)
        sftp = await self._sftp()
        handle = await self._guard(sftp.open(target, "wb"), path=target)
        try:
            await self._guard(handle.write(content), path=target)
        finally:
            try:
                await handle.close()
            except Exception:
                pass

    async def write_file(
        self, path: str, content: str, *, encoding: str = "utf-8", parents: bool = True
    ) -> None:
        await self.write_bytes(path, str(content).encode(encoding), parents=parents)

    async def append_file(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        target = await self.resolve(path)
        sftp = await self._sftp()
        handle = await self._guard(sftp.open(target, "ab"), path=target)
        try:
            await self._guard(handle.write(str(content).encode(encoding)), path=target)
        finally:
            try:
                await handle.close()
            except Exception:
                pass

    # ── 目录 / 元信息 ─────────────────────────────────────────────────────
    async def ls(self, path: str = ".") -> list[Any]:
        from ..types import FileStat

        target = await self.resolve(path)
        sftp = await self._sftp()
        entries = await self._guard(sftp.readdir(target), path=target)
        out: list[FileStat] = []
        for entry in entries:
            name = entry.filename
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            if name in {".", ".."}:
                continue
            out.append(_to_file_stat(posixpath.join(target, name), name, entry.attrs))
        out.sort(key=lambda item: (not item.is_dir, item.name))
        return out

    async def glob(self, pattern: str) -> list[str]:
        target = await self.resolve(pattern)
        sftp = await self._sftp()
        try:
            matches = await sftp.glob(target)
        except asyncssh.SFTPNoSuchFile:
            return []  # asyncssh 在零命中时会抛，但"没匹配到"不是错误
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError) as exc:
            await self.transport.reset_sftp()
            raise RelayConnectionError(f"connection lost during glob: {exc}") from exc
        except asyncssh.SFTPError as exc:
            raise RelayError(f"remote glob failed for {target}: {exc}") from exc
        result = []
        for item in matches:
            result.append(item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item))
        return sorted(result)

    async def stat(self, path: str) -> Any:
        target = await self.resolve(path)
        sftp = await self._sftp()
        attrs = await self._guard(sftp.stat(target), path=target)
        return _to_file_stat(target, posixpath.basename(target), attrs)

    async def exists(self, path: str) -> bool:
        target = await self.resolve(path)
        sftp = await self._sftp()
        try:
            return bool(await sftp.exists(target))
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError) as exc:
            await self.transport.reset_sftp()
            raise RelayConnectionError(f"connection lost during exists({target}): {exc}") from exc
        except asyncssh.SFTPError:
            return False

    async def mkdir(self, path: str, *, parents: bool = True) -> None:
        target = await self.resolve(path)
        if target in {"/", ""}:
            return
        sftp = await self._sftp()
        if parents:
            try:
                await sftp.makedirs(target, exist_ok=True)
                return
            except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError) as exc:
                await self.transport.reset_sftp()
                raise RelayConnectionError(f"connection lost during mkdir({target}): {exc}") from exc
            except asyncssh.SFTPError as exc:
                raise RelayError(f"cannot create remote directory {target}: {exc}") from exc
        await self._guard(sftp.mkdir(target), path=target)

    async def rm(self, path: str, *, recursive: bool = False, missing_ok: bool = True) -> None:
        target = await self.resolve(path)
        sftp = await self._sftp()
        try:
            if recursive:
                await sftp.rmtree(target, ignore_errors=missing_ok)
                return
            if await sftp.isdir(target):
                await sftp.rmdir(target)
            else:
                await sftp.remove(target)
        except asyncssh.SFTPNoSuchFile:
            if missing_ok:
                return
            raise RemoteFileNotFoundError(f"remote path not found: {target}") from None
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError) as exc:
            await self.transport.reset_sftp()
            raise RelayConnectionError(f"connection lost during rm({target}): {exc}") from exc
        except asyncssh.SFTPError as exc:
            raise RelayError(f"cannot remove remote path {target}: {exc}") from exc

    async def move(self, src: str, dst: str) -> None:
        source = await self.resolve(src)
        target = await self.resolve(dst)
        sftp = await self._sftp()
        await self._guard(sftp.rename(source, target), path=source)


def _to_file_stat(path: str, name: str, attrs: Any) -> Any:
    from ..types import FileStat

    permissions = int(getattr(attrs, "permissions", 0) or 0)
    return FileStat(
        path=path,
        name=name,
        is_dir=stat_module.S_ISDIR(permissions) if permissions else False,
        is_symlink=stat_module.S_ISLNK(permissions) if permissions else False,
        size=int(getattr(attrs, "size", 0) or 0),
        mtime=float(getattr(attrs, "mtime", 0) or 0),
        mode=permissions,
    )
