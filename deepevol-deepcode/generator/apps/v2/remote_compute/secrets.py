"""Deployment-owned remote-compute secret resolution and rotation.

Only the Agent control plane may mutate this file.  Product and Agent workers
exchange opaque ``secret_ref`` values; plaintext credentials never cross the
service boundary or enter Product PostgreSQL.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from threading import RLock


SECRET_STORE_SCHEMA_VERSION = "remote-compute-secret-store@v1"


class RemoteComputeSecretError(RuntimeError):
    code = "REMOTE_COMPUTE_SECRET_UNAVAILABLE"


class FileRemoteComputeSecretResolver:
    def __init__(self, path: Path, *, max_bytes: int = 1_048_576) -> None:
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("remote compute secrets path must be an absolute real path")
        if max_bytes < 1 or max_bytes > 16 * 1024 * 1024:
            raise ValueError("remote compute secrets max_bytes is invalid")
        self.path = path
        self.max_bytes = max_bytes
        self._lock = RLock()
        self._cache: tuple[int, int, int, dict[str, str]] | None = None

    def resolve(self, secret_ref: str, *, version: str = "") -> str:
        ref = str(secret_ref or "").strip()
        if not ref or len(ref) > 512 or any(ch.isspace() for ch in ref):
            raise RemoteComputeSecretError("secret reference is invalid")
        with self._lock:
            values = self._load()
        value = values.get(ref)
        if not isinstance(value, str) or not value:
            raise RemoteComputeSecretError("remote compute secret is not configured")
        # Version is an audit/fencing hint.  A versioned map may use ref@version
        # while preserving the stable ref accepted by Product.
        if version:
            versioned = values.get(f"{ref}@{version}")
            if isinstance(versioned, str) and versioned:
                return versioned
        return value

    def validate(self) -> int:
        """Load the deployment store now so readiness fails before a run does."""

        with self._lock:
            return len(self._load())

    def _load(self) -> dict[str, str]:
        if self.path.is_symlink():
            raise RemoteComputeSecretError("remote compute secrets file must not be a symlink")
        try:
            metadata = self.path.stat()
        except OSError as exc:
            raise RemoteComputeSecretError("remote compute secrets file is unavailable") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > self.max_bytes:
            raise RemoteComputeSecretError("remote compute secrets file is invalid")
        if metadata.st_mode & 0o077:
            raise RemoteComputeSecretError("remote compute secrets file permissions are too broad")
        cached = self._cache
        cache_key = (metadata.st_mtime_ns, metadata.st_size, metadata.st_ino)
        if cached is not None and cached[:3] == cache_key:
            return cached[3]
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteComputeSecretError("remote compute secrets file is not valid JSON") from exc
        if not isinstance(document, dict) or document.get("schema_version") != SECRET_STORE_SCHEMA_VERSION:
            raise RemoteComputeSecretError("remote compute secrets document schema is unsupported")
        raw = document.get("secrets")
        if not isinstance(raw, dict):
            raise RemoteComputeSecretError("remote compute secrets document has no secrets map")
        values = {
            str(key): value
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, str) and value
        }
        self._cache = (*cache_key, values)
        return values


class FileRemoteComputeSecretStore(FileRemoteComputeSecretResolver):
    """Atomically persist provider-issued access credentials on Agent only."""

    def put(self, secret_ref: str, value: str, *, version: str) -> None:
        ref = _secret_ref(secret_ref)
        secret = str(value or "")
        revision = str(version or "").strip()
        if not secret or len(secret.encode("utf-8")) > self.max_bytes // 2:
            raise RemoteComputeSecretError("remote compute secret value is invalid")
        if not revision or len(revision.encode("utf-8")) > 256 or any(
            character.isspace() for character in revision
        ):
            raise RemoteComputeSecretError("remote compute secret version is invalid")
        with self._lock:
            values = self._load()
            next_values = {**values, ref: secret, f"{ref}@{revision}": secret}
            self._publish(next_values)

    def delete(self, secret_ref: str, *, version: str = "") -> bool:
        ref = _secret_ref(secret_ref)
        revision = str(version or "").strip()
        with self._lock:
            values = self._load()
            keys = {f"{ref}@{revision}"} if revision else {ref}
            next_values = {
                key: value
                for key, value in values.items()
                if key not in keys and not (not revision and key.startswith(f"{ref}@"))
            }
            if len(next_values) == len(values):
                return False
            self._publish(next_values)
            return True

    def _publish(self, values: dict[str, str]) -> None:
        parent = self.path.parent
        if parent.is_symlink():
            raise RemoteComputeSecretError("remote compute secrets directory must not be a symlink")
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            document = json.dumps(
                {
                    "schema_version": SECRET_STORE_SCHEMA_VERSION,
                    "secrets": dict(sorted(values.items())),
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ).encode("utf-8") + b"\n"
            if len(document) > self.max_bytes:
                raise RemoteComputeSecretError("remote compute secrets file exceeds its size limit")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                dir=parent,
            )
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    stream.write(document)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, self.path)
                directory_descriptor = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
        except RemoteComputeSecretError:
            raise
        except OSError as exc:
            raise RemoteComputeSecretError("remote compute secret update failed") from exc
        self._cache = None
        self._load()


def _secret_ref(value: str) -> str:
    ref = str(value or "").strip()
    if not ref or len(ref) > 512 or any(character.isspace() for character in ref):
        raise RemoteComputeSecretError("secret reference is invalid")
    return ref


__all__ = [
    "SECRET_STORE_SCHEMA_VERSION",
    "FileRemoteComputeSecretResolver",
    "FileRemoteComputeSecretStore",
    "RemoteComputeSecretError",
]
