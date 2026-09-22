"""本地 ↔ 远端的文件搬运：单文件上传下载、整目录同步、可选 sha256 校验。

远端是唯一真源，所以这一层不承担"保持两边一致"的职责——它只做显式的搬运：把数据集/代码
推上去、把训练产物取回来。目录同步按 size+mtime 跳过没变的文件，重跑一次实验不会把几十 GB
的数据集再传一遍。
"""

from __future__ import annotations

import asyncio
import hashlib
import posixpath
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import asyncssh

from ..types import RelayConnectionError, RelayError, RemoteFileNotFoundError
from .sftp import RemoteFileSystem


# 同步时默认跳过的目录：传上去既没用又极慢。
DEFAULT_EXCLUDES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
        ".DS_Store",
        ".idea",
        ".vscode",
    }
)


@dataclass
class SyncReport:
    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    bytes_transferred: int = 0

    @property
    def summary(self) -> str:
        return (
            f"{len(self.uploaded)} transferred / {len(self.skipped)} skipped, "
            f"{self.bytes_transferred} bytes"
        )


class FileTransfer:
    """搬运器。exec_fn 用来跑 sha256sum 这类远端小命令（校验时才用到）。"""

    def __init__(
        self,
        transport: Any,
        fs: RemoteFileSystem,
        *,
        exec_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.transport = transport
        self.fs = fs
        self._exec_fn = exec_fn

    # ── 单文件 ────────────────────────────────────────────────────────────
    async def upload(
        self,
        local_path: str,
        remote_path: str,
        *,
        parents: bool = True,
        verify: bool = False,
    ) -> None:
        source = Path(local_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"local file not found: {source}")
        target = await self.fs.resolve(remote_path)
        if parents:
            await self.fs.mkdir(posixpath.dirname(target) or "/", parents=True)
        sftp = await self.transport.sftp()
        try:
            await sftp.put(str(source), target)
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError) as exc:
            await self.transport.reset_sftp()
            raise RelayConnectionError(f"connection lost while uploading {source}: {exc}") from exc
        except asyncssh.SFTPError as exc:
            raise RelayError(f"cannot upload {source} -> {target}: {exc}") from exc
        if verify:
            await self._verify(source, target)

    async def download(
        self,
        remote_path: str,
        local_path: str,
        *,
        parents: bool = True,
        verify: bool = False,
    ) -> None:
        source = await self.fs.resolve(remote_path)
        destination = Path(local_path).expanduser()
        if parents:
            destination.parent.mkdir(parents=True, exist_ok=True)
        sftp = await self.transport.sftp()
        try:
            await sftp.get(source, str(destination))
        except asyncssh.SFTPNoSuchFile as exc:
            raise RemoteFileNotFoundError(f"remote file not found: {source}") from exc
        except (asyncssh.ConnectionLost, asyncssh.DisconnectError, ConnectionError) as exc:
            await self.transport.reset_sftp()
            raise RelayConnectionError(f"connection lost while downloading {source}: {exc}") from exc
        except asyncssh.SFTPError as exc:
            raise RelayError(f"cannot download {source} -> {destination}: {exc}") from exc
        if verify:
            await self._verify(destination, source)

    async def _verify(self, local: Path, remote: str) -> None:
        """两边算 sha256 对一下。大文件传输后想确认没坏时才开。"""
        if self._exec_fn is None:
            raise RelayError("checksum verification requires an exec function")
        local_digest = await asyncio.to_thread(_sha256_file, local)
        result = await self._exec_fn(f"sha256sum -- {shlex.quote(remote)} 2>/dev/null | cut -d' ' -f1")
        remote_digest = str(getattr(result, "stdout", "") or "").strip()
        if not remote_digest:
            raise RelayError(f"cannot compute remote checksum for {remote}")
        if remote_digest != local_digest:
            raise RelayError(
                f"checksum mismatch for {remote}: local={local_digest} remote={remote_digest}"
            )

    # ── 目录同步 ──────────────────────────────────────────────────────────
    async def sync_up(
        self,
        local_dir: str,
        remote_dir: str,
        *,
        excludes: Iterable[str] = DEFAULT_EXCLUDES,
        delete: bool = False,
    ) -> SyncReport:
        """把本地目录推到远端。按 size+mtime 跳过没变的文件。"""
        root = Path(local_dir).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"local directory not found: {root}")
        target_root = await self.fs.resolve(remote_dir)
        exclude_set = set(excludes)
        report = SyncReport()

        remote_index = await self._remote_index(target_root)
        if delete:
            await self.fs.rm(target_root, recursive=True, missing_ok=True)
            remote_index = {}
        await self.fs.mkdir(target_root, parents=True)

        for path in sorted(root.rglob("*")):
            if any(part in exclude_set for part in path.relative_to(root).parts):
                continue
            relative = path.relative_to(root).as_posix()
            remote_path = posixpath.join(target_root, relative)
            if path.is_dir():
                await self.fs.mkdir(remote_path, parents=True)
                continue
            if not path.is_file():
                continue
            local_stat = path.stat()
            existing = remote_index.get(relative)
            if (
                existing is not None
                and existing[0] == local_stat.st_size
                and existing[1] + 1 >= local_stat.st_mtime
            ):
                report.skipped.append(relative)
                continue
            await self.upload(str(path), remote_path, parents=True)
            report.uploaded.append(relative)
            report.bytes_transferred += local_stat.st_size
        return report

    async def sync_down(
        self,
        remote_dir: str,
        local_dir: str,
        *,
        excludes: Iterable[str] = DEFAULT_EXCLUDES,
    ) -> SyncReport:
        """把远端目录拉回本地（取产物用）。"""
        source_root = await self.fs.resolve(remote_dir)
        destination_root = Path(local_dir).expanduser()
        destination_root.mkdir(parents=True, exist_ok=True)
        exclude_set = set(excludes)
        report = SyncReport()

        for relative, (size, _mtime) in sorted((await self._remote_index(source_root)).items()):
            if any(part in exclude_set for part in relative.split("/")):
                continue
            destination = destination_root / relative
            if destination.is_file() and destination.stat().st_size == size:
                report.skipped.append(relative)
                continue
            await self.download(posixpath.join(source_root, relative), str(destination), parents=True)
            report.uploaded.append(relative)
            report.bytes_transferred += size
        return report

    async def _remote_index(self, root: str) -> dict[str, tuple[int, float]]:
        """远端目录下所有文件的 {相对路径: (size, mtime)}。目录不存在时返回空。"""
        index: dict[str, tuple[int, float]] = {}
        try:
            if not await self.fs.exists(root):
                return index
        except RelayError:
            return index

        async def walk(directory: str, prefix: str) -> None:
            for entry in await self.fs.ls(directory):
                relative = f"{prefix}{entry.name}"
                if entry.is_dir:
                    await walk(entry.path, f"{relative}/")
                else:
                    index[relative] = (entry.size, entry.mtime)

        await walk(root, "")
        return index


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
