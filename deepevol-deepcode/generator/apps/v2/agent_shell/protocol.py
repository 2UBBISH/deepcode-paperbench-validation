"""Wire contract between an Agent and its shell.

Plain JSON over loopback HTTP.  The execution surface mirrors the relay
``Runtime`` protocol one-to-one so the Agent-side stub (:class:`client.ShellRuntime`)
can stand in for a ``RemoteRuntime`` without the vendored rsa/SetupX code
noticing.  Every result type here round-trips through :func:`encode` /
:func:`decode_*` losslessly; nothing on the wire ever carries a credential.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any, Literal, Mapping

SHELL_PROTOCOL_VERSION = "agent-shell@v1"

# Wall-clock timeout exit status, as GNU timeout and the relay report it.
TIMEOUT_EXIT_STATUS = 124

# ------------------------------------------------------------ result types
#
# Field-for-field the relay's ``remote_relay.types`` records, redeclared here
# so the shell package has no import on the Agent engine: the shell is meant
# to run as its own sidecar eventually, and the Agent-side consumers of these
# values (rsa's backend, the experiment flow) are duck-typed on the fields.

EventKind = Literal["stdout", "stderr", "exit"]


@dataclass(frozen=True)
class ExecEvent:
    kind: EventKind
    data: str = ""
    offset: int = 0
    exit_status: int | None = None
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ExecResult:
    exit_status: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    command: str = ""
    cwd: str = ""
    job_id: str | None = None
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def success(self) -> bool:
        return self.exit_status == 0

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated

    @property
    def output(self) -> str:
        if not self.stderr:
            return self.stdout
        if not self.stdout:
            return self.stderr
        return f"{self.stdout}\n{self.stderr}"


@dataclass(frozen=True)
class FileStat:
    path: str
    name: str
    is_dir: bool = False
    is_symlink: bool = False
    size: int = 0
    mtime: float = 0.0
    mode: int = 0


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    running: bool
    exit_status: int | None = None
    pgid: int | None = None
    command: str = ""
    cwd: str = ""
    started_at: float = 0.0
    stdout_size: int = 0
    stderr_size: int = 0
    state_marker: str = field(default="", repr=False)

    @property
    def finished(self) -> bool:
        return self.exit_status is not None

# Bounded request bodies: a chat completion with a long context is the largest
# legitimate request; file uploads are streamed separately under their own cap.
MAX_JSON_BODY_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


class PlacementKind(StrEnum):
    """Where the run's commands execute.  The Agent never sees more than this."""

    NONE = "NONE"
    LOCAL_DOCKER = "LOCAL_DOCKER"
    REMOTE_LEASE = "REMOTE_LEASE"


@dataclass(frozen=True, slots=True)
class Placement:
    kind: PlacementKind = PlacementKind.NONE
    # Monotonic per shell; a lease upgrade (bigger machine) bumps it so an Agent
    # that cached nothing still sees "the machine changed under me" if it asks.
    generation: int = 0
    # Free-form, credential-free label for reports ("ecs.c7.2xlarge", "sandbox").
    label: str = ""
    ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"kind": str(self.kind), "generation": self.generation, "label": self.label, "ready": self.ready}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Placement":
        return cls(
            kind=PlacementKind(str(raw.get("kind") or PlacementKind.NONE)),
            generation=int(raw.get("generation") or 0),
            label=str(raw.get("label") or ""),
            ready=bool(raw.get("ready")),
        )


class ShellErrorCode(StrEnum):
    UNAUTHORIZED = "SHELL_UNAUTHORIZED"
    BAD_REQUEST = "SHELL_BAD_REQUEST"
    NOT_FOUND = "SHELL_NOT_FOUND"
    PLACEMENT_UNAVAILABLE = "SHELL_PLACEMENT_UNAVAILABLE"
    TARGET_FAILED = "SHELL_TARGET_FAILED"
    MODEL_UNAVAILABLE = "SHELL_MODEL_UNAVAILABLE"
    MODEL_FAILED = "SHELL_MODEL_FAILED"
    PAYLOAD_TOO_LARGE = "SHELL_PAYLOAD_TOO_LARGE"
    EGRESS_REFUSED = "SHELL_EGRESS_REFUSED"
    EGRESS_FAILED = "SHELL_EGRESS_FAILED"


class ShellError(RuntimeError):
    """Raised on the Agent side when the shell refuses or fails a request."""

    def __init__(self, code: ShellErrorCode | str, message: str = "", *, status: int = 500) -> None:
        self.code = ShellErrorCode(code) if not isinstance(code, ShellErrorCode) else code
        self.status = int(status)
        super().__init__(message or str(self.code))


# --------------------------------------------------------------- encoding

def encode(value: Any) -> Any:
    """Dataclass results -> JSON-safe dicts (lists/dicts recurse)."""

    if isinstance(value, Placement):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        # Shell-side or relay-side record: both serialise by field name.
        return asdict(value)
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    return value


def _only_fields(cls: type, raw: Mapping[str, Any]) -> dict[str, Any]:
    names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in raw.items() if k in names}


def decode_exec_result(raw: Mapping[str, Any]) -> ExecResult:
    return ExecResult(**_only_fields(ExecResult, raw))


def decode_job_status(raw: Mapping[str, Any]) -> JobStatus:
    return JobStatus(**_only_fields(JobStatus, raw))


def decode_file_stat(raw: Mapping[str, Any]) -> FileStat:
    return FileStat(**_only_fields(FileStat, raw))


def decode_exec_event(raw: Mapping[str, Any]) -> ExecEvent:
    return ExecEvent(**_only_fields(ExecEvent, raw))


@dataclass(frozen=True, slots=True)
class ExecRequest:
    command: str
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None
    durable: bool | None = None
    # Caller-side identity of the command (the tool operation id) so the
    # placement's lifecycle events line up with the durable tool record.
    command_id: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExecRequest":
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        env = raw.get("env") or {}
        if not isinstance(env, Mapping) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()
        ):
            raise ValueError("env must map strings to strings")
        timeout = raw.get("timeout")
        if timeout is not None:
            timeout = float(timeout)
            if not timeout > 0:
                raise ValueError("timeout must be positive")
        cwd = raw.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError("cwd must be a string")
        durable = raw.get("durable")
        if durable is not None and not isinstance(durable, bool):
            raise ValueError("durable must be a boolean")
        command_id = raw.get("command_id")
        if command_id is not None and (not isinstance(command_id, str) or len(command_id) > 128):
            raise ValueError("command_id must be a short string")
        return cls(command=command, cwd=cwd, env=dict(env), timeout=timeout, durable=durable,
                   command_id=command_id or None)


__all__ = [
    "ExecEvent",
    "ExecResult",
    "FileStat",
    "JobStatus",
    "TIMEOUT_EXIT_STATUS",
    "MAX_JSON_BODY_BYTES",
    "MAX_UPLOAD_BYTES",
    "SHELL_PROTOCOL_VERSION",
    "ExecRequest",
    "Placement",
    "PlacementKind",
    "ShellError",
    "ShellErrorCode",
    "decode_exec_event",
    "decode_exec_result",
    "decode_file_stat",
    "decode_job_status",
    "encode",
]
