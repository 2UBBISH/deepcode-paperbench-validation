"""Agent-side V2 remote-compute workspace access.

The Agent is the only component that resolves the deployment-owned secret and
opens an SSH/SFTP connection.  Product returns an opaque descriptor only;
this module deliberately has no SQLite fallback and never accepts a caller
supplied host, root, or credential.
"""

from __future__ import annotations

from apps.v2.reliability.circuit import CircuitPolicy, CircuitRegistry, DependencyUnavailable
from apps.v2.reliability.control_response import build_control_http_client, read_control_response
from apps.v2.reliability.ssh_channel import drain_channel, execute_command, SSHOutputLimitExceeded

import math
from threading import Lock, Timer
import hashlib
import json
import base64
import io
import os
import posixpath
import shlex
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from apps.common.v2_ids import format_typed_id

import httpx

from .models import RemoteWorkspaceDescriptor
from .secrets import FileRemoteComputeSecretResolver, RemoteComputeSecretError

try:  # pragma: no cover - import availability is covered by deployment checks
    import paramiko
except ImportError:  # pragma: no cover
    paramiko = None  # type: ignore[assignment]


WORKSPACE_SSH_CIRCUITS = CircuitRegistry(CircuitPolicy.from_env())

_MAX_PATH_BYTES = 4096
_MAX_TREE_ITEMS = 2000
_MAX_PREVIEW_BYTES = 1024 * 1024
_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_COMMAND_BYTES = 32 * 1024
_MAX_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_TRANSFER_BYTES = 2 * 1024 * 1024 * 1024
_SKIP_PARTS = frozenset({".git", "__pycache__", ".pytest_cache", "node_modules", ".cache", ".venv", "venv"})


class RemoteWorkspaceError(RuntimeError):
    """Stable error exposed by the workspace compatibility transport."""

    def __init__(self, code: str, *, status_code: int = 404) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class RemoteWorkspaceUnavailable(RemoteWorkspaceError):
    def __init__(self) -> None:
        super().__init__("REMOTE_WORKSPACE_UNAVAILABLE", status_code=503)


@dataclass(frozen=True, slots=True)
class RemoteWorkspaceDownload:
    """Lazy SFTP download stream (same shape as the Workspace compat download)."""

    name: str
    size_bytes: int
    content_type: str
    chunks: Iterator[bytes]


@dataclass(frozen=True, slots=True)
class RemoteComputeRunLocator:
    """The immutable run/lease scope used to resolve a Product descriptor."""

    resource_tid: UUID
    owner_uid: UUID
    sid: UUID
    rid: UUID
    lease_generation: int

    def __post_init__(self) -> None:
        if self.lease_generation < 1:
            raise ValueError("remote compute lease generation must be positive")


def remote_compute_locator(context: Any, authority: Any) -> RemoteComputeRunLocator:
    """Build a locator only from the worker's verified run and lease objects."""

    values = {name: getattr(context, name, None) for name in ("resource_tid", "owner_uid", "sid", "rid")}
    if any(not isinstance(value, UUID) for value in values.values()):
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_AUTHORITY_MALFORMED", status_code=422)
    generation = getattr(authority, "generation", None)
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_AUTHORITY_MALFORMED", status_code=422)
    if (
        values["resource_tid"] != getattr(authority, "resource_tid", None)
        or values["owner_uid"] != getattr(authority, "owner_uid", None)
        or values["rid"] != getattr(authority, "rid", None)
    ):
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_AUTHORITY_INVALID", status_code=403)
    return RemoteComputeRunLocator(**values, lease_generation=generation)


class ProductRemoteWorkspaceClient:
    """Fetch a run-bound descriptor from Product's authenticated V2 endpoint."""

    def __init__(
        self,
        endpoint_url: str,
        token: str,
        *,
        source_service: str = "AGENT_EXECUTION",
        timeout_seconds: float = 10.0,
    ) -> None:
        parsed = urlparse(endpoint_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("remote workspace endpoint must be an HTTP(S) URL")
        if len(token.encode("utf-8")) < 32:
            raise ValueError("remote workspace service token must contain at least 32 bytes")
        if not source_service or len(source_service) > 64:
            raise ValueError("remote workspace source service is invalid")
        if timeout_seconds <= 0:
            raise ValueError("remote workspace timeout must be positive")
        # Keep a configured path, while preventing a query/fragment from being
        # smuggled into the request target.
        self.base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        self.token = token
        self.source_service = source_service
        self.timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 5.0))
        self.timeout_seconds = timeout_seconds

    def resolve(self, locator: Any) -> RemoteWorkspaceDescriptor | None:
        """Resolve the descriptor for one Agent execution locator.

        A 404 is an expected "no compute binding" result.  Transport failures
        are surfaced as a stable 503 so Product can preserve its compatibility
        response semantics.
        """

        rid = _typed_id(locator.rid, "rid")
        sid = _typed_id(locator.sid, "sid")
        resource_tid = _typed_id(locator.resource_tid, "tid")
        owner_uid = _typed_id(locator.owner_uid, "uid")
        if self.base_url.endswith("/internal/v2/runs"):
            url = f"{self.base_url}/{rid}/remote-workspace"
        else:
            url = f"{self.base_url}/internal/v2/runs/{rid}/remote-workspace"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept-Encoding": "identity",
            "X-DeepEvol-Source-Service": self.source_service,
            "X-DeepEvol-Resource-Tid": resource_tid,
            "X-DeepEvol-Owner-Uid": owner_uid,
        }
        params = {
            "sid": sid,
            "rid": rid,
            "lease_generation": str(int(locator.lease_generation)),
        }
        descriptor: RemoteWorkspaceDescriptor | None = None

        def validate(response: httpx.Response) -> None:
            nonlocal descriptor
            try:
                payload = response.json()
            except ValueError as exc:
                raise RemoteWorkspaceUnavailable() from exc
            if not isinstance(payload, Mapping):
                raise RemoteWorkspaceUnavailable()
            data = payload.get("data", payload)
            if not isinstance(data, Mapping):
                raise RemoteWorkspaceUnavailable()
            try:
                descriptor = RemoteWorkspaceDescriptor.from_dict(data)
            except (KeyError, TypeError, ValueError) as exc:
                raise RemoteWorkspaceUnavailable() from exc

        try:
            with build_control_http_client("agent-remote-workspace", timeout_seconds=self.timeout_seconds) as client:
                with client.stream("GET", url, headers=headers, params=params) as streamed:
                    response = read_control_response(
                        streamed, maximum_bytes=64 * 1024, accepted_statuses=frozenset(range(200, 400)),
                        validate_success=validate
                    )
        except httpx.HTTPError as exc:
            raise RemoteWorkspaceUnavailable() from exc
        if response.status_code == 404:
            return None
        if response.status_code in {401, 403}:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_UNAUTHORIZED", status_code=503)
        if response.status_code >= 500:
            raise RemoteWorkspaceUnavailable()
        if response.status_code >= 400:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_INVALID", status_code=503)
        return descriptor


class ParamikoRemoteWorkspaceBrowser:
    """Bounded SFTP browser for a Product-authorized remote workspace."""

    def __init__(
        self,
        descriptor_client: ProductRemoteWorkspaceClient,
        secret_resolver: FileRemoteComputeSecretResolver,
        *,
        known_hosts: str | None = None,
        connect_timeout_seconds: float = 10.0,
        session_timeout_seconds: float = 300.0,
        max_tree_items: int = _MAX_TREE_ITEMS,
        max_preview_bytes: int = _MAX_PREVIEW_BYTES,
        local_workspace_root: str | os.PathLike[str] | None = None,
        max_transfer_bytes: int = _MAX_TRANSFER_BYTES,
    ) -> None:
        if paramiko is None:
            raise RuntimeError("paramiko is required for V2 remote compute")
        if connect_timeout_seconds <= 0:
            raise ValueError("remote SSH timeout must be positive")
        if not 1 <= max_tree_items <= 10_000:
            raise ValueError("remote tree item limit is invalid")
        if not 1 <= max_preview_bytes <= 16 * 1024 * 1024:
            raise ValueError("remote preview limit is invalid")
        if max_transfer_bytes < 1 or max_transfer_bytes > _MAX_TRANSFER_BYTES:
            raise ValueError("remote transfer limit is invalid")
        if not math.isfinite(session_timeout_seconds) or not 1 <= session_timeout_seconds <= 3600:
            raise ValueError("remote session deadline must be between 1 and 3600 seconds")
        self.session_timeout_seconds = session_timeout_seconds
        self.descriptor_client = descriptor_client
        self.secret_resolver = secret_resolver
        self.known_hosts = known_hosts
        self.connect_timeout_seconds = connect_timeout_seconds
        self.max_tree_items = max_tree_items
        self.max_preview_bytes = max_preview_bytes
        self.local_workspace_root = None if local_workspace_root is None else _real_local_root(local_workspace_root)
        self.max_transfer_bytes = max_transfer_bytes
        self.secret_resolver.validate()
        if self.known_hosts:
            known_hosts_path = Path(self.known_hosts)
            if not known_hosts_path.is_absolute() or known_hosts_path.is_symlink():
                raise ValueError("remote compute known-hosts file is unavailable")
            try:
                metadata = known_hosts_path.stat()
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077 or metadata.st_size > 4 * 1024 * 1024:
                    raise ValueError("remote compute known-hosts file is not private")
                keys = paramiko.HostKeys(filename=str(known_hosts_path))
            except Exception as exc:
                raise ValueError("remote compute known-hosts file is invalid") from exc
            if not keys:
                raise ValueError("remote compute known-hosts file has no host keys")

    def for_local_workspace_root(
        self,
        root: str | os.PathLike[str],
    ) -> "ParamikoRemoteWorkspaceBrowser":
        """Return the same transport with a run-scoped local workspace."""

        return ParamikoRemoteWorkspaceBrowser(
            self.descriptor_client,
            self.secret_resolver,
            known_hosts=self.known_hosts,
            connect_timeout_seconds=self.connect_timeout_seconds,
            session_timeout_seconds=self.session_timeout_seconds,
            max_tree_items=self.max_tree_items,
            max_preview_bytes=self.max_preview_bytes,
            local_workspace_root=root,
            max_transfer_bytes=self.max_transfer_bytes,
        )

    def resolve(self, locator: Any) -> RemoteWorkspaceDescriptor | None:
        return self.descriptor_client.resolve(locator)

    def ready(self, locator: Any) -> bool:
        descriptor = self.resolve(locator)
        if descriptor is None or not self._descriptor_is_current(descriptor):
            return False
        try:
            with self._connection(descriptor) as (_client, sftp):
                self._assert_no_symlink(sftp, descriptor.remote_root, ())
                self._stat_root(sftp, descriptor.remote_root)
            return True
        except RemoteWorkspaceError:
            return False

    def list_servers(self, locator: Any) -> dict[str, Any]:
        """Return the single Product-authorized server for this leased run.

        V2 intentionally has no ambient server inventory.  The model sees a
        compatibility-shaped list, but it can only select the descriptor that
        Product bound to the current ``rid`` and lease.
        """

        descriptor = self._required_descriptor(locator)
        return {
            "servers": [
                {
                    "connection_name": descriptor.connection_name,
                    "name": descriptor.connection_name,
                    "provider": descriptor.provider,
                    "status": descriptor.status,
                    "host": descriptor.host,
                    "port": descriptor.port,
                    "username": descriptor.username,
                    "workspace_scope": "current_run",
                    "resource_id": format_typed_id("rcres", descriptor.resource_id),
                }
            ]
        }

    def execute_command(
        self,
        locator: Any,
        *,
        connection_name: str,
        command: str,
        cwd: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        descriptor = self._required_descriptor(locator)
        self._assert_connection_name(descriptor, connection_name)
        if not isinstance(command, str) or not command.strip():
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_COMMAND_INVALID", status_code=422)
        if len(command.encode("utf-8")) > _MAX_COMMAND_BYTES or "\x00" in command:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_COMMAND_INVALID", status_code=422)
        if cwd is None or cwd == "":
            remote_cwd = descriptor.remote_root
        else:
            remote_cwd = _workspace_path(descriptor.remote_root, cwd)
        timeout = _bounded_timeout(timeout_ms)
        try:
            with self._connection(descriptor) as (client, _sftp):
                if isinstance(client, _WorkspaceSSHSession):
                    timeout = min(timeout, client.remaining_seconds())
                try:
                    stdout, stderr, exit_status = execute_command(
                        client.client if isinstance(client, _WorkspaceSSHSession) else client,
                        f"cd -- {shell_quote(remote_cwd)} && {command}",
                        maximum=_MAX_COMMAND_OUTPUT_BYTES,
                        timeout_seconds=timeout,
                    )
                except TimeoutError:
                    raise RemoteWorkspaceError("REMOTE_WORKSPACE_COMMAND_TIMEOUT", status_code=504) from None
                except SSHOutputLimitExceeded:
                    raise RemoteWorkspaceError("REMOTE_WORKSPACE_OUTPUT_TOO_LARGE", status_code=413) from None
        except RemoteWorkspaceError:
            raise
        except Exception as exc:
            raise _remote_failure(exc) from None
        return {
            "success": exit_status == 0,
            "connection_name": descriptor.connection_name,
            "exit_status": exit_status,
            "stdout": stdout.decode("utf-8", "replace"),
            "stderr": stderr.decode("utf-8", "replace"),
            "workspace_scope": "current_run",
        }

    def upload(
        self,
        locator: Any,
        *,
        connection_name: str,
        local_path: str,
        remote_path: str,
    ) -> dict[str, Any]:
        descriptor = self._required_descriptor(locator)
        self._assert_connection_name(descriptor, connection_name)
        local = self._confined_local_path(local_path, require_file=True)
        relative = _workspace_relative_parts(descriptor.remote_root, remote_path, require_child=True)
        target = _confined_path(descriptor.remote_root, relative)
        size = local.stat().st_size
        if size > self.max_transfer_bytes:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_TRANSFER_TOO_LARGE", status_code=413)
        try:
            with self._connection(descriptor) as (_client, sftp):
                sftp.put(str(local), target)
                metadata = sftp.stat(target)
        except RemoteWorkspaceError:
            raise
        except Exception as exc:
            raise _remote_failure(exc) from None
        return {
            "success": True,
            "connection_name": descriptor.connection_name,
            "local_path": str(local),
            "remote_path": target,
            "size_bytes": int(metadata.st_size or size),
            "workspace_scope": "current_run",
        }

    def download_to(
        self,
        locator: Any,
        *,
        connection_name: str,
        remote_path: str,
        local_path: str,
    ) -> dict[str, Any]:
        descriptor = self._required_descriptor(locator)
        self._assert_connection_name(descriptor, connection_name)
        relative = _workspace_relative_parts(descriptor.remote_root, remote_path, require_child=True)
        source = _confined_path(descriptor.remote_root, relative)
        local = self._confined_local_path(local_path, require_file=False)
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connection(descriptor) as (_client, sftp):
                self._assert_no_symlink(sftp, descriptor.remote_root, relative)
                metadata = self._stat_file(sftp, source)
                size = int(metadata.st_size or 0)
                if size > self.max_transfer_bytes:
                    raise RemoteWorkspaceError("REMOTE_WORKSPACE_TRANSFER_TOO_LARGE", status_code=413)
                temporary = local.with_name(f".{local.name}.part")
                try:
                    with temporary.open("wb") as handle:
                        sftp.getfo(source, handle)
                    os.replace(temporary, local)
                finally:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
        except RemoteWorkspaceError:
            raise
        except Exception as exc:
            raise _remote_failure(exc, file=True) from None
        return {
            "success": True,
            "connection_name": descriptor.connection_name,
            "remote_path": source,
            "local_path": str(local),
            "size_bytes": size,
            "workspace_scope": "current_run",
        }

    def tree(self, locator: Any, relative_path: str) -> dict[str, Any]:
        descriptor = self._required_descriptor(locator)
        relative = _relative_remote_parts(relative_path)
        remote_path = _confined_path(descriptor.remote_root, relative)
        try:
            with self._connection(descriptor) as (_client, sftp):
                self._assert_no_symlink(sftp, descriptor.remote_root, relative)
                self._stat_directory(sftp, remote_path)
                names = sorted(
                    sftp.listdir_attr(remote_path),
                    key=lambda item: (not stat.S_ISDIR(item.st_mode), item.filename.casefold()),
                )
                items: list[dict[str, Any]] = []
                for entry in names[: self.max_tree_items]:
                    name = str(entry.filename)
                    if not name or name in _SKIP_PARTS:
                        continue
                    mode = int(entry.st_mode)
                    if stat.S_ISLNK(mode):
                        continue
                    is_dir = stat.S_ISDIR(mode)
                    if not is_dir and not stat.S_ISREG(mode):
                        continue
                    child = _confined_path(descriptor.remote_root, (*relative, name))
                    items.append(
                        {
                            "name": name,
                            "path": _public_compute_path(locator, (*relative, name)),
                            "type": "dir" if is_dir else "file",
                            "size": None if is_dir else int(entry.st_size or 0),
                            "mtime": int(entry.st_mtime or 0),
                            "source": "compute",
                            "remote_path": child,
                        }
                    )
        except RemoteWorkspaceError:
            raise
        except Exception as exc:
            raise _remote_failure(exc) from None
        return {
            "subpath": _public_compute_path(locator, relative),
            "workspace_ready": True,
            "items": items,
            "source": "compute",
        }

    def file(self, locator: Any, relative_path: str) -> dict[str, Any]:
        descriptor = self._required_descriptor(locator)
        relative = _relative_remote_parts(relative_path, require_child=True)
        remote_path = _confined_path(descriptor.remote_root, relative)
        try:
            with self._connection(descriptor) as (_client, sftp):
                self._assert_no_symlink(sftp, descriptor.remote_root, relative)
                metadata = self._stat_file(sftp, remote_path)
                with sftp.open(remote_path, "rb") as handle:
                    raw = handle.read(self.max_preview_bytes + 1)
        except RemoteWorkspaceError:
            raise
        except Exception as exc:
            raise _remote_failure(exc, file=True) from None
        truncated = len(raw) > self.max_preview_bytes
        raw = raw[: self.max_preview_bytes]
        binary = b"\x00" in raw[:8192]
        return {
            "name": relative[-1],
            "path": _public_compute_path(locator, relative),
            "size": int(metadata.st_size or 0),
            "is_binary": binary,
            "truncated": truncated,
            "content": None if binary else raw.decode("utf-8", "replace"),
            "source": "compute",
            "remote_path": remote_path,
        }

    def download(self, locator: Any, relative_path: str) -> Any:
        """Return a compatibility ``WorkspaceDownload`` with a lazy SFTP stream."""

        descriptor = self._required_descriptor(locator)
        relative = _relative_remote_parts(relative_path, require_child=True)
        remote_path = _confined_path(descriptor.remote_root, relative)
        client = sftp = handle = None
        try:
            client, sftp = self._connect(descriptor)
            self._assert_no_symlink(sftp, descriptor.remote_root, relative)
            metadata = self._stat_file(sftp, remote_path)
            size = int(metadata.st_size)
            if not 0 <= size <= self.max_transfer_bytes:
                raise RemoteWorkspaceError("REMOTE_WORKSPACE_OUTPUT_TOO_LARGE", status_code=413)
            handle = sftp.open(remote_path, "rb")
        except Exception as exc:
            if isinstance(client, _WorkspaceSSHSession):
                client.record_error(exc)
            _close_quietly(handle, sftp, client)
            if _workspace_transport_failure(exc):
                raise RemoteWorkspaceUnavailable() from None
            if isinstance(exc, RemoteWorkspaceError):
                raise
            raise _remote_failure(exc, file=True) from None

        return RemoteWorkspaceDownload(
            name=relative[-1],
            size_bytes=size,
            content_type="application/octet-stream",
            chunks=_SftpDownload(handle, sftp, client, size),
        )

    def _required_descriptor(self, locator: Any) -> RemoteWorkspaceDescriptor:
        descriptor = self.resolve(locator)
        if descriptor is None:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_NOT_AVAILABLE", status_code=404)
        if not self._descriptor_is_current(descriptor):
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_NOT_AVAILABLE", status_code=404)
        return descriptor

    @contextmanager
    def _connection(self, descriptor: RemoteWorkspaceDescriptor):
        client = sftp = None
        try:
            client, sftp = self._connect(descriptor)
            yield client, sftp
            if isinstance(client, _WorkspaceSSHSession):
                client.check_deadline()
                client.outcome = "success"
        except Exception as exc:
            if isinstance(client, _WorkspaceSSHSession):
                client.record_error(exc)
            if _workspace_transport_failure(exc):
                raise RemoteWorkspaceUnavailable() from None
            raise
        finally:
            _close_quietly(sftp, client)

    @staticmethod
    def _assert_connection_name(descriptor: RemoteWorkspaceDescriptor, connection_name: str) -> None:
        if not isinstance(connection_name, str) or connection_name != descriptor.connection_name:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_CONNECTION_INVALID", status_code=403)

    def _confined_local_path(self, raw: str, *, require_file: bool) -> Path:
        if self.local_workspace_root is None:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_LOCAL_ROOT_UNAVAILABLE", status_code=503)
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_INVALID", status_code=422)
        candidate = Path(raw)
        if candidate.is_absolute():
            target = candidate.resolve(strict=False)
        else:
            target = (self.local_workspace_root / candidate).resolve(strict=False)
        root = self.local_workspace_root
        if target != root and root not in target.parents:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_FORBIDDEN", status_code=403)
        if target.exists() and target.is_symlink():
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_FORBIDDEN", status_code=403)
        if require_file and (not target.exists() or not target.is_file()):
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_FILE_NOT_FOUND", status_code=404)
        return target

    @staticmethod
    def _descriptor_is_current(descriptor: RemoteWorkspaceDescriptor) -> bool:
        if descriptor.status not in {"ACTIVE", "RUNNING"}:
            return False
        if descriptor.expires_at is None:
            return True
        if descriptor.expires_at.tzinfo is None or descriptor.expires_at.utcoffset() is None:
            return False
        return descriptor.expires_at > datetime.now(UTC)

    def _connect(self, descriptor: RemoteWorkspaceDescriptor):
        workflow_remaining = _workflow_remaining_seconds()
        session_timeout = self.session_timeout_seconds
        if workflow_remaining is not None:
            session_timeout = min(session_timeout, workflow_remaining)
        session_started = time.monotonic()
        try:
            secret = self.secret_resolver.resolve(
                descriptor.secret_ref,
                version=descriptor.secret_version,
            )
        except RemoteComputeSecretError as exc:
            raise RemoteWorkspaceUnavailable() from exc
        key = (
            "workspace-ssh:"
            + hashlib.sha256(
                json.dumps(
                    [descriptor.host.lower(), descriptor.port, descriptor.username],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        try:
            permit = WORKSPACE_SSH_CIRCUITS.acquire(key)
        except DependencyUnavailable:
            raise RemoteWorkspaceUnavailable() from None
        client = sftp = session = None
        try:
            client = paramiko.SSHClient()
            if self.known_hosts:
                client.load_host_keys(self.known_hosts)
            else:
                client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            connect_remaining = session_timeout - (time.monotonic() - session_started)
            if connect_remaining <= 0:
                raise TimeoutError("remote workspace workflow deadline exceeded")
            connect_timeout = min(self.connect_timeout_seconds, connect_remaining)
            kwargs: dict[str, Any] = {
                "hostname": descriptor.host,
                "port": descriptor.port,
                "username": descriptor.username,
                "timeout": connect_timeout,
                "banner_timeout": connect_timeout,
                "auth_timeout": connect_timeout,
                "channel_timeout": connect_timeout,
                "look_for_keys": False,
                "allow_agent": False,
            }
            if secret.lstrip().startswith("-----BEGIN"):
                kwargs["pkey"] = _private_key(secret)
            else:
                kwargs["password"] = secret
            client.connect(**kwargs)
            session_remaining = session_timeout - (time.monotonic() - session_started)
            if session_remaining <= 0:
                raise TimeoutError("remote workspace workflow deadline exceeded")
            session = _WorkspaceSSHSession(
                client,
                permit,
                timeout_seconds=session_remaining,
            )
            sftp = client.open_sftp()
            sftp.get_channel().settimeout(min(self.connect_timeout_seconds, session_remaining))
            session.check_deadline()
            return session, sftp
        except BaseException as exc:
            if session is not None:
                session.record_error(exc)
                session.close()
            _close_quietly(sftp, client)
            permit.finish("failure" if _workspace_transport_failure(exc) else "neutral")
            if not isinstance(exc, Exception):
                raise
            raise _remote_failure(exc) from None

    @staticmethod
    def _stat_root(sftp: Any, root: str) -> Any:
        return ParamikoRemoteWorkspaceBrowser._stat_directory(sftp, root)

    @staticmethod
    def _assert_no_symlink(sftp: Any, root: str, parts: tuple[str, ...]) -> None:
        """Reject symlink components before any SFTP operation follows them."""

        current = posixpath.normpath(root)
        # lstat() on the final root alone cannot detect a symlink in one of its
        # parent components because the SFTP server has already followed it.
        # Walk the absolute root components first, then the user-relative path.
        root_components = tuple(part for part in PurePosixPath(current).parts if part != "/")
        current = "/"
        for part in root_components:
            current = posixpath.join(current, part)
            try:
                metadata = sftp.lstat(current)
            except Exception as exc:
                raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_NOT_FOUND", status_code=404) from exc
            if stat.S_ISLNK(int(metadata.st_mode)):
                raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_FORBIDDEN", status_code=403)
        for index, part in enumerate(parts):
            current = posixpath.join(current, part)
            try:
                metadata = sftp.lstat(current)
            except Exception as exc:
                raise RemoteWorkspaceError(
                    "REMOTE_WORKSPACE_FILE_NOT_FOUND" if index == len(parts) - 1 else "REMOTE_WORKSPACE_PATH_NOT_FOUND",
                    status_code=404,
                ) from exc
            if stat.S_ISLNK(int(metadata.st_mode)):
                raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_FORBIDDEN", status_code=403)
            if index < len(parts) - 1 and not stat.S_ISDIR(int(metadata.st_mode)):
                raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_NOT_FOUND", status_code=404)

    @staticmethod
    def _stat_directory(sftp: Any, path: str) -> Any:
        try:
            metadata = sftp.stat(path)
        except Exception as exc:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_NOT_FOUND", status_code=404) from exc
        if stat.S_ISLNK(int(metadata.st_mode)) or not stat.S_ISDIR(int(metadata.st_mode)):
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_NOT_FOUND", status_code=404)
        return metadata

    @staticmethod
    def _stat_file(sftp: Any, path: str) -> Any:
        try:
            metadata = sftp.stat(path)
        except Exception as exc:
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_FILE_NOT_FOUND", status_code=404) from exc
        if stat.S_ISLNK(int(metadata.st_mode)) or not stat.S_ISREG(int(metadata.st_mode)):
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_FILE_NOT_FOUND", status_code=404)
        return metadata


class StrictV2RemoteComputeToolset:
    """Compatibility tools bound to exactly one Product-authorized run."""

    def __init__(
        self,
        browser: ParamikoRemoteWorkspaceBrowser,
        locator: RemoteComputeRunLocator,
        workspace_root: Path,
        *,
        debug_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(locator, RemoteComputeRunLocator):
            raise TypeError("strict V2 remote tools require a run locator")
        root = Path(workspace_root).expanduser()
        if not root.is_absolute() or root.is_symlink():
            raise ValueError("strict V2 remote tools require an absolute workspace root")
        self.browser = browser
        self.locator = locator
        self.workspace_root = root.resolve()
        self.debug_sink = debug_sink

    def _debug(self, entry: Mapping[str, Any]) -> None:
        if self.debug_sink is None:
            return
        try:
            self.debug_sink(entry)
        except Exception:
            # Debug telemetry must never alter the remote operation outcome.
            pass

    def tools(self) -> list[Any]:
        from langchain_core.tools import tool

        browser = self.browser
        locator = self.locator
        workspace_root = self.workspace_root
        emit_debug = self._debug

        @tool(
            "server-use__list_servers",
            description="List the single remote compute server authorized for this run.",
        )
        def list_servers() -> dict[str, Any]:
            return browser.list_servers(locator)

        @tool(
            "server-use__execute_command",
            description="Execute a shell command in the authorized run workspace.",
        )
        def execute_command(
            connection_name: str,
            command: str,
            cwd: str | None = None,
            timeout_ms: int | None = None,
        ) -> dict[str, Any]:
            started = time.monotonic()
            try:
                result = browser.execute_command(
                    locator,
                    connection_name=connection_name,
                    command=command,
                    cwd=cwd,
                    timeout_ms=timeout_ms,
                )
            except Exception as exc:
                emit_debug(
                    {
                        "ts": time.time(),
                        "kind": "exec",
                        "connection": connection_name,
                        "command": command,
                        "exit_status": None,
                        "stdout": "",
                        "stderr": str(getattr(exc, "code", type(exc).__name__)),
                        "ms": int((time.monotonic() - started) * 1_000),
                    }
                )
                raise
            emit_debug(
                {
                    "ts": time.time(),
                    "kind": "exec",
                    "connection": result.get("connection_name", connection_name),
                    "command": command,
                    "exit_status": result.get("exit_status"),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "ms": int((time.monotonic() - started) * 1_000),
                }
            )
            return result

        @tool(
            "server-use__upload",
            description="Upload a file from this run's local workspace to remote compute.",
        )
        def upload(
            connection_name: str,
            local_path: str,
            remote_path: str,
        ) -> dict[str, Any]:
            local = _run_local_path(workspace_root, local_path, require_file=True)
            result = browser.upload(
                locator,
                connection_name=connection_name,
                local_path=str(local),
                remote_path=remote_path,
            )
            emit_debug(
                {
                    "ts": time.time(),
                    "kind": "upload",
                    "connection": result.get("connection_name", connection_name),
                    "command": f"[SFTP upload] {local_path} -> {remote_path}",
                    "exit_status": 0,
                    "stdout": "",
                    "stderr": "",
                    "ms": None,
                }
            )
            return result

        @tool(
            "server-use__download",
            description="Download a remote file into this run's local workspace.",
        )
        def download(
            connection_name: str,
            remote_path: str,
            local_path: str,
        ) -> dict[str, Any]:
            local = _run_local_path(workspace_root, local_path, require_file=False)
            result = browser.download_to(
                locator,
                connection_name=connection_name,
                remote_path=remote_path,
                local_path=str(local),
            )
            emit_debug(
                {
                    "ts": time.time(),
                    "kind": "download",
                    "connection": result.get("connection_name", connection_name),
                    "command": f"[SFTP download] {remote_path} -> {local_path}",
                    "exit_status": 0,
                    "stdout": "",
                    "stderr": "",
                    "ms": None,
                }
            )
            return result

        return [list_servers, execute_command, upload, download]


def _typed_id(value: Any, prefix: str) -> str:
    raw = str(value)
    if not raw:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_AUTHORITY_MALFORMED", status_code=422)
    # UUID values are formatted by the shared ID helper at the call boundary;
    # avoid importing parsing machinery here so test doubles can use UUIDs.
    return f"{prefix}_{raw}" if "_" not in raw else raw


def _real_local_root(raw: str | os.PathLike[str]) -> Path:
    root = Path(raw).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("remote compute local workspace root must be absolute and real")
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _run_local_path(root: Path, raw: str, *, require_file: bool) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_INVALID", status_code=422)
    candidate = Path(raw)
    target = candidate.resolve(strict=False) if candidate.is_absolute() else (root / candidate).resolve(strict=False)
    if target != root and root not in target.parents:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_FORBIDDEN", status_code=403)
    if target.exists() and target.is_symlink():
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_FORBIDDEN", status_code=403)
    if require_file and (not target.exists() or not target.is_file()):
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_FILE_NOT_FOUND", status_code=404)
    return target


def _bounded_timeout(raw: int | None) -> float:
    if raw is None:
        return 300.0
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1_000 <= raw <= 900_000:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_TIMEOUT_INVALID", status_code=422)
    return raw / 1000.0


def _workflow_remaining_seconds() -> float | None:
    from apps.v2.agent_engine.tools.runtime_context import get_tool_runtime_context

    raw = get_tool_runtime_context().get("v2_workflow_remaining_seconds")
    if raw is None:
        return None
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(raw)
        or raw <= 0
    ):
        raise RemoteWorkspaceError(
            "REMOTE_WORKSPACE_DEADLINE_INVALID",
            status_code=503,
        )
    return float(raw)


def _drain_channel(
    channel: Any,
    *,
    maximum: int,
    timeout_seconds: float,
) -> tuple[bytes, bytes]:
    try:
        return drain_channel(channel, maximum=maximum, timeout_seconds=timeout_seconds)
    except TimeoutError:
        _close_quietly(channel)
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_COMMAND_TIMEOUT", status_code=504) from None
    except SSHOutputLimitExceeded:
        _close_quietly(channel)
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_OUTPUT_TOO_LARGE", status_code=413) from None


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def _relative_remote_parts(raw_path: str, *, require_child: bool = False) -> tuple[str, ...]:
    if not isinstance(raw_path, str):
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_INVALID", status_code=422)
    normalized = raw_path.strip().replace("\\", "/")
    if "\x00" in normalized or len(normalized.encode("utf-8")) > _MAX_PATH_BYTES:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_INVALID", status_code=422)
    if normalized.startswith("/"):
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_FORBIDDEN", status_code=403)
    parts = tuple(PurePosixPath(normalized).parts)
    if any(part in {"", ".", ".."} for part in parts):
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_INVALID", status_code=422)
    if require_child and not parts:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_FILE_NOT_FOUND", status_code=404)
    return parts


def _workspace_relative_parts(root: str, raw_path: str, *, require_child: bool = False) -> tuple[str, ...]:
    if not isinstance(raw_path, str) or "\x00" in raw_path:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_INVALID", status_code=422)
    normalized_root = posixpath.normpath(root)
    candidate = raw_path.strip().replace("\\", "/")
    if candidate.startswith("/"):
        normalized = posixpath.normpath(candidate)
        prefix = normalized_root.rstrip("/") + "/"
        if normalized != normalized_root and not normalized.startswith(prefix):
            raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_FORBIDDEN", status_code=403)
        candidate = normalized[len(normalized_root.rstrip("/")) :].lstrip("/")
    return _relative_remote_parts(candidate, require_child=require_child)


def _workspace_path(root: str, raw_path: str) -> str:
    parts = _workspace_relative_parts(root, raw_path)
    return _confined_path(root, parts)


def _confined_path(root: str, parts: tuple[str, ...]) -> str:
    if not isinstance(root, str) or not root.startswith("/") or "\x00" in root:
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_NOT_FOUND", status_code=404)
    normalized_root = posixpath.normpath(root)
    target = posixpath.normpath(posixpath.join(normalized_root, *parts))
    if target != normalized_root and not target.startswith(normalized_root.rstrip("/") + "/"):
        raise RemoteWorkspaceError("REMOTE_WORKSPACE_PATH_NOT_FOUND", status_code=404)
    return target


def _public_compute_path(locator: Any, parts: tuple[str, ...]) -> str:
    # A stable binding-specific token prevents the remote host/root from being
    # exposed as an authority in URLs.  The token is not a credential.
    binding = str(getattr(locator, "rid", ""))
    token = base64.urlsafe_b64encode(binding.encode("utf-8")).decode("ascii").rstrip("=")
    return "@compute" + (f"/{token}/{'/'.join(parts)}" if parts else f"/{token}")


def _private_key(value: str) -> Any:
    if paramiko is None:  # pragma: no cover
        raise RemoteWorkspaceUnavailable()
    stream = io.StringIO(value)
    for cls_name in ("Ed25519Key", "RSAKey", "ECDSAKey", "DSSKey"):
        cls = getattr(paramiko, cls_name, None)
        if cls is None:
            continue
        try:
            stream.seek(0)
            return cls.from_private_key(stream)
        except Exception:
            continue
    raise RemoteWorkspaceUnavailable()


def _remote_failure(exc: Exception, *, file: bool = False) -> RemoteWorkspaceError:
    # Do not include exception text: Paramiko may echo a username, host, or
    # provider-specific authentication detail into the exception message.
    if _workspace_transport_failure(exc):
        return RemoteWorkspaceUnavailable()
    return RemoteWorkspaceError(
        "REMOTE_WORKSPACE_FILE_NOT_FOUND" if file else "REMOTE_WORKSPACE_UNAVAILABLE",
        status_code=404 if file else 503,
    )


def _close_quietly(*objects: Any) -> None:
    for obj in objects:
        if obj is None:
            continue
        try:
            obj.close()
        except Exception:
            pass


__all__ = [
    "ParamikoRemoteWorkspaceBrowser",
    "ProductRemoteWorkspaceClient",
    "RemoteComputeRunLocator",
    "RemoteWorkspaceError",
    "RemoteWorkspaceUnavailable",
    "StrictV2RemoteComputeToolset",
    "remote_compute_locator",
]


class _SftpDownload(Iterator[bytes]):
    """Own open resources even before the first iteration."""

    def __init__(self, handle, sftp, client, size):
        self.handle, self.sftp, self.client = handle, sftp, client
        self.remaining = size
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        if isinstance(self.client, _WorkspaceSSHSession):
            try:
                self.client.check_deadline()
            except RemoteWorkspaceUnavailable:
                self.close()
                raise
        if self.remaining == 0:
            if isinstance(self.client, _WorkspaceSSHSession):
                self.client.outcome = "success"
            self.close()
            raise StopIteration
        try:
            block = self.handle.read(min(_STREAM_CHUNK_BYTES, self.remaining))
            if isinstance(self.client, _WorkspaceSSHSession):
                self.client.check_deadline()
            if not block or len(block) > self.remaining:
                raise RemoteWorkspaceError("REMOTE_WORKSPACE_FILE_CHANGED", status_code=409)
            self.remaining -= len(block)
            return block
        except Exception as exc:
            if isinstance(self.client, _WorkspaceSSHSession):
                self.client.record_error(exc)
            self.close()
            if isinstance(exc, RemoteWorkspaceError):
                raise
            raise _remote_failure(exc, file=True) from None

    def close(self):
        if not self.closed:
            self.closed = True
            _close_quietly(self.handle, self.sftp, self.client)


def _workspace_transport_failure(exc: BaseException) -> bool:
    # Domain wrappers preserve their cause/context. Path/permission failures
    # must not disable an otherwise reachable host.
    for _ in range(8):
        if isinstance(exc, (FileNotFoundError, PermissionError)):
            return False
        if isinstance(exc, (OSError, EOFError, paramiko.SSHException)):
            return True
        exc = exc.__cause__ or exc.__context__
        if exc is None:
            break
    return False


class _WorkspaceSSHSession:
    def __init__(self, client, permit, *, timeout_seconds=300.0):
        self.client = client
        self.permit = permit
        self.outcome = "neutral"
        self.closed = False
        self.expired = False
        self._lock = Lock()
        self._deadline = time.monotonic() + timeout_seconds
        self._timer = Timer(timeout_seconds, lambda: self._close(expired=True))
        self._timer.daemon = True
        self._timer.start()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def check_deadline(self):
        if time.monotonic() >= self._deadline:
            self._close(expired=True)
        if self.expired:
            raise RemoteWorkspaceUnavailable()

    def remaining_seconds(self):
        self.check_deadline()
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self._close(expired=True)
            raise RemoteWorkspaceUnavailable()
        return remaining

    def record_error(self, exc):
        if _workspace_transport_failure(exc):
            self.outcome = "failure"

    def close(self):
        self._close(expired=False)

    def _close(self, *, expired):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self.expired = expired
            outcome = "failure" if expired else self.outcome
            self._timer.cancel()
            # Settle the permit before tearing down the transport: closing a
            # stalled client can block, and the caller unblocked by that close
            # must already observe the circuit's in_flight/open state.
            self.permit.finish(outcome)
        self.client.close()
