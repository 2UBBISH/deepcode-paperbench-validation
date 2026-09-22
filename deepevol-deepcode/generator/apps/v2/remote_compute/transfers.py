"""Generation-fenced, resumable SFTP file migration for Remote Compute."""

from __future__ import annotations

import hashlib
import os
import posixpath
import socket
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid5

from apps.v2.reliability import HeartbeatRunner, Lease, canonical_json_sha256

from .commands import RemoteComputeCommandError
from .models import RemoteComputeAction, RemoteComputeCommand
from .provider_operations import (
    ProviderOperationInvalidTransition,
    ProviderOperationStaleLease,
    PsycopgProviderOperationRepository,
    TransferItemCreate,
    TransferItemLease,
    TransferItemRecord,
    TransferItemStatus,
)


class TransferCrashPoint(StrEnum):
    AFTER_DISCOVERY = "after_discovery"
    AFTER_PART_WRITE = "after_part_write"
    AFTER_PART_CHECKPOINT = "after_part_checkpoint"
    AFTER_VERIFY = "after_verify"
    AFTER_ATOMIC_RENAME = "after_atomic_rename"
    AFTER_TERMINAL_CHECKPOINT = "after_terminal_checkpoint"


@dataclass(frozen=True, slots=True)
class _DiscoveredFile:
    item_key: str
    source_path: str
    target_path: str
    size_bytes: int


class DurableSftpTransferExecutor:
    """Resume deterministic file parts and fence stale transfer generations."""

    def __init__(
        self,
        repository: PsycopgProviderOperationRepository,
        *,
        holder: str | None = None,
        part_size_bytes: int = 8 * 1024 * 1024,
        lease_ttl: timedelta = timedelta(seconds=120),
        heartbeat_interval: timedelta = timedelta(seconds=20),
        heartbeat_stop_timeout_seconds: float = 5.0,
        fault_injector: Callable[[TransferCrashPoint], None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not 1 <= part_size_bytes <= 64 * 1024 * 1024:
            raise ValueError("transfer part size must be between 1 byte and 64 MiB")
        if heartbeat_interval * 3 >= lease_ttl:
            raise ValueError("transfer heartbeat interval must be below one third of lease TTL")
        if heartbeat_stop_timeout_seconds <= 0:
            raise ValueError("transfer heartbeat stop timeout must be positive")
        self.repository = repository
        self.holder = holder or f"{socket.gethostname()}:{os.getpid()}"
        self.part_size_bytes = part_size_bytes
        self.lease_ttl = lease_ttl
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_stop_timeout_seconds = heartbeat_stop_timeout_seconds
        self.fault_injector = fault_injector
        self.clock = clock

    def migrate(
        self,
        command: RemoteComputeCommand,
        *,
        source: Any,
        target: Any,
    ) -> Mapping[str, Any]:
        if command.action is not RemoteComputeAction.MIGRATE:
            raise ValueError("durable SFTP transfer requires a MIGRATE command")
        source_root = _confined_root(command.remote_root)
        target_root = _confined_root(command.target_remote_root)
        discovered = _discover_files(source, source_root, target_root)
        records: dict[str, TransferItemRecord] = {}
        for item in discovered:
            record = self.repository.create_transfer_item(
                TransferItemCreate(
                    transfer_id=uuid5(command.command_id, item.item_key),
                    operation_id=command.command_id,
                    resource_tid=command.resource_tid,
                    owner_uid=command.owner_uid,
                    item_key=item.item_key,
                    source_ref=item.source_path,
                    target_ref=item.target_path,
                    size_bytes=item.size_bytes,
                    part_size_bytes=self.part_size_bytes,
                )
            )
            records[item.item_key] = record

        persisted = self.repository.list_transfer_items(
            resource_tid=command.resource_tid,
            owner_uid=command.owner_uid,
            operation_id=command.command_id,
        )
        if {item.item_key for item in persisted} != set(records):
            raise RemoteComputeCommandError("REMOTE_MIGRATION_SOURCE_CHANGED")
        self._inject(TransferCrashPoint.AFTER_DISCOVERY)

        for item in discovered:
            record = records[item.item_key]
            if record.status is TransferItemStatus.COMPLETED:
                _verify_completed_item(source, target, record)
                continue
            if record.status in {
                TransferItemStatus.FAILED,
                TransferItemStatus.MANUAL_REVIEW,
            }:
                raise RemoteComputeCommandError(
                    record.last_error_code or "REMOTE_MIGRATION_MANUAL_REVIEW"
                )
            lease = self.repository.claim_transfer_item(
                resource_tid=command.resource_tid,
                owner_uid=command.owner_uid,
                transfer_id=record.transfer_id,
                holder=self.holder,
                lease_ttl=self.lease_ttl,
            )
            if lease is None:
                current = self.repository.get_transfer_item(
                    resource_tid=command.resource_tid,
                    owner_uid=command.owner_uid,
                    transfer_id=record.transfer_id,
                )
                if current is not None and current.status is TransferItemStatus.COMPLETED:
                    _verify_completed_item(source, target, current)
                    continue
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_ITEM_IN_PROGRESS",
                    retryable=True,
                )
            self._resume_item(source, target, lease)

        completed = self.repository.list_transfer_items(
            resource_tid=command.resource_tid,
            owner_uid=command.owner_uid,
            operation_id=command.command_id,
        )
        if len(completed) != len(discovered) or any(
            item.status is not TransferItemStatus.COMPLETED for item in completed
        ):
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_INCOMPLETE",
                retryable=True,
            )
        manifest = [
            {
                "item_key": item.item_key,
                "sha256": item.final_sha256,
                "size_bytes": item.size_bytes,
            }
            for item in completed
        ]
        return {
            "transferred": True,
            "file_count": len(completed),
            "total_bytes": sum(item.size_bytes for item in completed),
            "manifest_sha256": canonical_json_sha256(manifest),
        }

    def _resume_item(
        self,
        source: Any,
        target: Any,
        lease: TransferItemLease,
    ) -> None:
        item = lease.item
        lease_box = [lease]
        acquired_at = item.heartbeat_at or self.clock()
        generic_lease = Lease(
            holder=lease.holder,
            generation=lease.generation,
            token_hash=lease.token_hash,
            acquired_at=acquired_at,
            expires_at=lease.expires_at,
            operation_id=item.transfer_id,
        )

        def renew(current: Lease) -> Lease:
            renewed = self.repository.heartbeat_transfer(
                lease_box[0],
                lease_ttl=self.lease_ttl,
            )
            lease_box[0] = renewed
            return current.renewed(expires_at=renewed.expires_at)

        heartbeat = HeartbeatRunner(
            generic_lease,
            renew,
            interval=self.heartbeat_interval,
            clock=self.clock,
            thread_name=f"remote-transfer-{item.transfer_id}",
        )
        heartbeat.start()
        heartbeat_stopped = False
        try:
            stage = item.status
            source_sha256 = item.source_sha256
            final_sha256 = item.final_sha256
            partial_path = _partial_path(item)
            if stage in {TransferItemStatus.DISCOVER, TransferItemStatus.COPY_PARTIAL}:
                _copy_remaining_parts(
                    source,
                    target,
                    item,
                    partial_path=partial_path,
                    checkpoint=lambda offset: self.repository.record_copy_progress(
                        lease_box[0], offset
                    ),
                    after_write=lambda: self._inject(
                        TransferCrashPoint.AFTER_PART_WRITE
                    ),
                    after_checkpoint=lambda: self._inject(
                        TransferCrashPoint.AFTER_PART_CHECKPOINT
                    ),
                )
                source_sha256 = _sha256_remote_file(
                    source,
                    item.source_ref,
                    expected_size=item.size_bytes,
                )
                self.repository.begin_verify(lease_box[0], source_sha256)
                stage = TransferItemStatus.VERIFY

            if stage is TransferItemStatus.VERIFY:
                if source_sha256 is None:
                    raise RemoteComputeCommandError(
                        "REMOTE_MIGRATION_CHECKPOINT_INVALID"
                    )
                final_sha256 = _sha256_remote_file(
                    target,
                    partial_path,
                    expected_size=item.size_bytes,
                )
                if final_sha256 != source_sha256:
                    self.repository.record_transfer_integrity_failure(
                        lease_box[0],
                        error_code="REMOTE_MIGRATION_HASH_MISMATCH",
                    )
                    raise RemoteComputeCommandError(
                        "REMOTE_MIGRATION_HASH_MISMATCH"
                    )
                self.repository.record_verified(lease_box[0], final_sha256)
                self._inject(TransferCrashPoint.AFTER_VERIFY)
                stage = TransferItemStatus.ATOMIC_RENAME

            if stage is TransferItemStatus.ATOMIC_RENAME:
                expected_sha256 = final_sha256 or source_sha256
                if expected_sha256 is None:
                    raise RemoteComputeCommandError(
                        "REMOTE_MIGRATION_CHECKPOINT_INVALID"
                    )
                self._atomic_promote(
                    target,
                    lease_box[0],
                    partial_path=partial_path,
                    expected_sha256=expected_sha256,
                )
                self._inject(TransferCrashPoint.AFTER_ATOMIC_RENAME)

            heartbeat_stopped = heartbeat.stop(
                timeout_seconds=self.heartbeat_stop_timeout_seconds
            )
            if not heartbeat_stopped or heartbeat.error is not None:
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_LEASE_LOST",
                    retryable=True,
                )
            self.repository.complete_atomic_rename(lease_box[0])
            self._inject(TransferCrashPoint.AFTER_TERMINAL_CHECKPOINT)
        except (ProviderOperationInvalidTransition, ProviderOperationStaleLease) as exc:
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_LEASE_LOST",
                retryable=True,
            ) from exc
        finally:
            if not heartbeat_stopped:
                heartbeat.stop(
                    timeout_seconds=self.heartbeat_stop_timeout_seconds
                )

    def _atomic_promote(
        self,
        target: Any,
        lease: TransferItemLease,
        *,
        partial_path: str,
        expected_sha256: str,
    ) -> None:
        item = lease.item
        if _remote_exists(target, item.target_ref):
            actual = _sha256_remote_file(
                target,
                item.target_ref,
                expected_size=item.size_bytes,
            )
            if actual != expected_sha256:
                self.repository.record_transfer_integrity_failure(
                    lease,
                    error_code="REMOTE_MIGRATION_TARGET_CONFLICT",
                )
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_TARGET_CONFLICT"
                ) from None
            if _remote_exists(target, partial_path):
                target.remove(partial_path)
            return
        try:
            target.rename(partial_path, item.target_ref)
        except OSError:
            if not _remote_exists(target, item.target_ref):
                raise
            actual = _sha256_remote_file(
                target,
                item.target_ref,
                expected_size=item.size_bytes,
            )
            if actual != expected_sha256:
                self.repository.record_transfer_integrity_failure(
                    lease,
                    error_code="REMOTE_MIGRATION_TARGET_CONFLICT",
                )
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_TARGET_CONFLICT"
                ) from None

    def _inject(self, point: TransferCrashPoint) -> None:
        if self.fault_injector is not None:
            self.fault_injector(point)


def _discover_files(
    source: Any,
    source_root: str,
    target_root: str,
) -> tuple[_DiscoveredFile, ...]:
    max_entries = 100_000
    max_bytes = 100 * 1024 * 1024 * 1024
    entries_seen = 0
    total_bytes = 0
    files: list[_DiscoveredFile] = []
    stack = [(source_root, "")]
    while stack:
        source_dir, relative_dir = stack.pop()
        entries = sorted(
            source.listdir_attr(source_dir),
            key=lambda entry: str(entry.filename),
        )
        for entry in entries:
            entries_seen += 1
            if entries_seen > max_entries:
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_LIMIT_EXCEEDED"
                )
            filename = _safe_filename(entry.filename)
            item_key = (
                filename
                if not relative_dir
                else posixpath.join(relative_dir, filename)
            )
            if len(item_key.encode("utf-8")) > 512:
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_PATH_TOO_LONG"
                )
            source_path = posixpath.join(source_dir, filename)
            mode = int(entry.st_mode)
            if stat.S_ISLNK(mode):
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_SYMLINK_REJECTED"
                )
            if stat.S_ISDIR(mode):
                stack.append((source_path, item_key))
                continue
            if not stat.S_ISREG(mode):
                continue
            size_bytes = int(entry.st_size)
            if size_bytes < 0:
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_SOURCE_INVALID"
                )
            total_bytes += size_bytes
            if total_bytes > max_bytes:
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_LIMIT_EXCEEDED"
                )
            files.append(
                _DiscoveredFile(
                    item_key=item_key,
                    source_path=source_path,
                    target_path=posixpath.join(target_root, item_key),
                    size_bytes=size_bytes,
                )
            )
    return tuple(sorted(files, key=lambda item: item.item_key))


def _copy_remaining_parts(
    source: Any,
    target: Any,
    item: TransferItemRecord,
    *,
    partial_path: str,
    checkpoint: Callable[[int], Any],
    after_write: Callable[[], None],
    after_checkpoint: Callable[[], None],
) -> None:
    metadata = source.lstat(item.source_ref)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or int(metadata.st_size) != item.size_bytes
    ):
        raise RemoteComputeCommandError("REMOTE_MIGRATION_SOURCE_CHANGED")
    _ensure_remote_directory(target, posixpath.dirname(item.target_ref))
    partial_exists = _remote_exists(target, partial_path)
    if item.offset_bytes > 0:
        if not partial_exists:
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_STAGING_LOST"
            )
        partial = target.lstat(partial_path)
        if (
            stat.S_ISLNK(partial.st_mode)
            or not stat.S_ISREG(partial.st_mode)
            or int(partial.st_size) < item.offset_bytes
        ):
            raise RemoteComputeCommandError(
                "REMOTE_MIGRATION_STAGING_INVALID"
            )

    mode = "r+b" if partial_exists else "wb"
    offset = item.offset_bytes
    with source.open(item.source_ref, "rb") as reader, target.open(
        partial_path, mode
    ) as writer:
        reader.seek(offset)
        writer.seek(offset)
        while offset < item.size_bytes:
            expected = min(item.part_size_bytes, item.size_bytes - offset)
            chunk = reader.read(expected)
            if not isinstance(chunk, bytes) or len(chunk) != expected:
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_SOURCE_CHANGED"
                )
            writer.write(chunk)
            flush = getattr(writer, "flush", None)
            if callable(flush):
                flush()
            after_write()
            offset += len(chunk)
            checkpoint(offset)
            after_checkpoint()
        truncate = getattr(writer, "truncate", None)
        if callable(truncate):
            truncate(item.size_bytes)


def _verify_completed_item(
    source: Any,
    target: Any,
    item: TransferItemRecord,
) -> None:
    if item.source_sha256 is None or item.final_sha256 != item.source_sha256:
        raise RemoteComputeCommandError("REMOTE_MIGRATION_CHECKPOINT_INVALID")
    source_sha256 = _sha256_remote_file(
        source,
        item.source_ref,
        expected_size=item.size_bytes,
    )
    target_sha256 = _sha256_remote_file(
        target,
        item.target_ref,
        expected_size=item.size_bytes,
    )
    if source_sha256 != item.source_sha256 or target_sha256 != item.final_sha256:
        raise RemoteComputeCommandError("REMOTE_MIGRATION_COMPLETED_HASH_DRIFT")


def _sha256_remote_file(
    sftp: Any,
    path: str,
    *,
    expected_size: int,
) -> str:
    digest = hashlib.sha256()
    observed = 0
    with sftp.open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_SOURCE_INVALID"
                )
            observed += len(chunk)
            if observed > expected_size:
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_SIZE_MISMATCH"
                )
            digest.update(chunk)
    if observed != expected_size:
        raise RemoteComputeCommandError("REMOTE_MIGRATION_SIZE_MISMATCH")
    return digest.hexdigest()


def _remote_exists(sftp: Any, path: str) -> bool:
    try:
        metadata = sftp.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise RemoteComputeCommandError("REMOTE_MIGRATION_SYMLINK_REJECTED")
    return True


def _partial_path(item: TransferItemRecord) -> str:
    return f"{item.target_ref}.deepevol-{item.transfer_id}.partial"


def _safe_filename(value: Any) -> str:
    filename = str(value)
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\x00" in filename
    ):
        raise RemoteComputeCommandError("REMOTE_MIGRATION_SOURCE_INVALID")
    return filename


def _confined_root(value: str) -> str:
    candidate = PurePosixPath(posixpath.normpath(str(value or "")))
    if not candidate.is_absolute() or str(candidate) == "/" or "\x00" in str(candidate):
        raise RemoteComputeCommandError("REMOTE_MIGRATION_TARGET_INVALID")
    return str(candidate)


def _ensure_remote_directory(sftp: Any, path: str) -> None:
    current = "/"
    for part in PurePosixPath(path).parts[1:]:
        current = posixpath.join(current, part)
        try:
            metadata = sftp.lstat(current)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(
                metadata.st_mode
            ):
                raise RemoteComputeCommandError(
                    "REMOTE_MIGRATION_TARGET_INVALID"
                )
        except OSError:
            sftp.mkdir(current, mode=0o700)


__all__ = ["DurableSftpTransferExecutor", "TransferCrashPoint"]
