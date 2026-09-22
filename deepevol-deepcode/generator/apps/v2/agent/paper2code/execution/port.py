"""The execution port: one remote job at a time, from a local workspace.

``tools/execute.py`` (the model's ``execute_python`` / ``execute_bash``) and
``verification_hook.py`` (the engine's mechanical verification) both submit
a :class:`Job` and read back a :class:`JobResult`. Executors live next to
this file (``job_executor.py``, ``leased_runtime.py``); this module holds
only the contract and the value types so the tools can be built before any
executor exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

TAIL_BYTES = 64 * 1024


@dataclass(slots=True)
class Job:
    """One command to run against a copy of ``workspace``.

    ``command`` runs under ``bash -lc`` with the synced workspace as the
    working directory. When ``script`` is set the executor materializes it
    outside the workspace and exports its path as ``$JOB_SCRIPT`` before
    running ``command`` (so ``execute_python`` never writes into the
    generated repository).
    """

    workspace: Path
    command: str
    timeout_s: float
    script: str | None = None
    label: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class JobResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    machine: str
    job_dir: Path | None = None
    timed_out: bool = False
    error: str | None = None
    synced_back: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None

    def to_record(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout[-TAIL_BYTES:],
            "stderr_tail": self.stderr[-TAIL_BYTES:],
            "duration_s": round(self.duration_s, 3),
            "machine": self.machine,
            "job_dir": str(self.job_dir) if self.job_dir else None,
            "timed_out": self.timed_out,
            "error": self.error,
            "synced_back": list(self.synced_back),
        }


class ExecutionPort(Protocol):
    async def run(self, job: Job) -> JobResult: ...

    async def close(self) -> None: ...


def tail(text: str, limit: int = TAIL_BYTES) -> str:
    """Keep the last ``limit`` bytes of ``text`` (the engine's verification keeps tails)."""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return data[-limit:].decode("utf-8", errors="replace")
