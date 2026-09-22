"""Route the engine's mechanical verification through the execution port.

The engine discovers test commands (pytest / unittest / npm test / cargo
test) and, by default, runs them as local subprocesses. This line never
runs generated code on the driver host, so the workflow's
``verification_runner`` (VENDOR.md entry 8) is pointed at a coroutine that
submits the same command as a :class:`Job` and maps the result back to the
engine's ``VerificationResult`` (same 300 s budget, same 64 KiB tails).
"""

from __future__ import annotations

import shlex
from pathlib import Path

from apps.v2.agent.paper2code.execution.port import TAIL_BYTES, ExecutionPort, Job, tail
from apps.v2.agent_engine.paper2code.support.verification import VerificationCommand, VerificationResult
from apps.v2.agent_engine.paper2code.workflows import code_implementation_workflow as _workflow


def make_verification_runner(port: ExecutionPort):
    async def run_verification_remote(
        root: Path,
        command: VerificationCommand,
        *,
        timeout_seconds: int = 300,
    ) -> VerificationResult:
        job = Job(
            workspace=Path(root),
            command=shlex.join(command.argv),
            timeout_s=float(timeout_seconds),
            label=f"verify:{command.id}",
        )
        result = await port.run(job)
        stdout = result.stdout
        stderr = result.stderr
        if result.error and not result.timed_out:
            stderr = (stderr + "\n" if stderr else "") + f"[execution port] {result.error}"
        return VerificationResult(
            command=command,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_ms=int(result.duration_s * 1000),
            stdout=tail(stdout),
            stderr=tail(stderr),
            output_truncated=len(stdout.encode("utf-8", errors="replace")) > TAIL_BYTES
            or len(stderr.encode("utf-8", errors="replace")) > TAIL_BYTES,
        )

    return run_verification_remote


def install_verification_runner(port: ExecutionPort) -> None:
    """Make every ``CodeImplementationWorkflow`` built from now on verify through ``port``."""
    _workflow.VERIFICATION_RUNNER = make_verification_runner(port)


def uninstall_verification_runner() -> None:
    _workflow.VERIFICATION_RUNNER = None


__all__ = ["install_verification_runner", "make_verification_runner", "uninstall_verification_runner"]
