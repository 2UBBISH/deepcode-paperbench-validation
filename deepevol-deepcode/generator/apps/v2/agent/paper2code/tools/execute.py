"""``execute_python`` / ``execute_bash`` routed to the run's execution port — **not registered since S3**.

Until PLAN-3 §2 8i the implement phase ran model code through these (the generated repository
synced to the run's remote machine, the command in a one-shot container). The owner's review
(2026-09-17) put the implement phase back on upstream's behaviour: the engine's own
``execute_python`` / ``execute_bash`` — a local subprocess in the engine's write-fence sandbox
(``support/sandbox``) behind ``command_guard`` — and the engine's local verification. This
module stays only until the execution port retires with the rest of it (PLAN-3 §0 "退役");
``tools/kernel_servers.code_implementation_tools`` no longer calls :func:`execution_tools`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.execution.port import Job, JobResult
from apps.v2.agent.paper2code.tools.registry import FunctionTool, ToolContext, exposed_name
from apps.v2.agent_engine.paper2code.seams.agent_runtime import Tool
from apps.v2.agent_engine.paper2code.support.command_guard import screen_command

DEFAULT_TIMEOUT_S = 30
MAX_TIMEOUT_S = 1800


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _workspace() -> Path | None:
    from apps.v2.agent_engine.paper2code.tools import code_implementation_server as srv

    return Path(srv.WORKSPACE_DIR) if srv.WORKSPACE_DIR is not None else None


def _result_payload(result: JobResult, *, kind: str, extra: dict[str, Any]) -> dict[str, Any]:
    ok = result.ok
    payload: dict[str, Any] = {
        "status": "success" if ok else "error",
        "return_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        **extra,
        "sandbox": f"remote-docker@{result.machine}",
        "duration_s": round(result.duration_s, 2),
    }
    if result.timed_out:
        payload["message"] = f"{kind} execution timeout ({extra.get('timeout')} seconds)"
    elif result.error:
        payload["message"] = f"{kind} execution failed: {result.error}"
    else:
        payload["message"] = f"{kind} execution {'successful' if ok else 'failed'}"
    return payload


def execution_tools(ctx: ToolContext) -> Sequence[Tool]:
    async def _run(job: Job, *, kind: str, extra: dict[str, Any]) -> str:
        if ctx.port is None:
            return _dump(
                {
                    "status": "error",
                    "message": (
                        f"{kind} execution is unavailable: this run has no execution "
                        "port (offline mode). Continue implementing; verification "
                        "runs separately."
                    ),
                    **extra,
                }
            )
        try:
            result = await ctx.port.run(job)
        except Exception as exc:
            return _dump({"status": "error", "message": f"{kind} execution failed: {exc}", **extra})
        return _dump(_result_payload(result, kind=kind, extra=extra))

    async def execute_python(code: str, timeout: int = DEFAULT_TIMEOUT_S) -> str:
        timeout = int(min(max(int(timeout), 1), MAX_TIMEOUT_S))
        workspace = _workspace()
        if workspace is None:
            return _dump({"status": "error", "message": "workspace is not set; call set_workspace first"})
        job = Job(
            workspace=workspace,
            command='python "$JOB_SCRIPT"',
            timeout_s=float(timeout),
            script=code,
            label="execute_python",
        )
        return await _run(job, kind="Python code", extra={"timeout": timeout})

    async def execute_bash(command: str, timeout: int = DEFAULT_TIMEOUT_S) -> str:
        timeout = int(min(max(int(timeout), 1), MAX_TIMEOUT_S))
        blocked = screen_command(command)
        if blocked is not None:
            return _dump(
                {
                    "status": "error",
                    "message": f"Dangerous command prohibited ({blocked}): {command}",
                }
            )
        workspace = _workspace()
        if workspace is None:
            return _dump({"status": "error", "message": "workspace is not set; call set_workspace first"})
        job = Job(workspace=workspace, command=command, timeout_s=float(timeout), label="execute_bash")
        return await _run(job, kind="Bash command", extra={"command": command, "timeout": timeout})

    return [
        FunctionTool(
            name=exposed_name("code-implementation", "execute_python"),
            description=(
                "Execute Python code in the run's isolated container (the "
                "generated repository is its working directory) and return a JSON "
                "result with status, return_code, stdout and stderr."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": DEFAULT_TIMEOUT_S},
                },
                "required": ["code"],
            },
            fn=execute_python,
            timeout_s=MAX_TIMEOUT_S + 120,
        ),
        FunctionTool(
            name=exposed_name("code-implementation", "execute_bash"),
            description=(
                "Execute a bash command in the run's isolated container (the "
                "generated repository is its working directory) and return a JSON "
                "result with status, return_code, stdout and stderr."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": DEFAULT_TIMEOUT_S},
                },
                "required": ["command"],
            },
            fn=execute_bash,
            timeout_s=MAX_TIMEOUT_S + 120,
        ),
    ]


__all__ = ["DEFAULT_TIMEOUT_S", "MAX_TIMEOUT_S", "execution_tools"]
