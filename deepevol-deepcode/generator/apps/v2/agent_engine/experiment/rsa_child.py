"""rsa / SetupX in their own process, talking to the run's Agent shell.

Why a child process and not a thread:

* the vendored rsa loop is synchronous and can wedge (an SSH read that never
  returns, a pip that never finishes); a thread cannot be killed, so the hard
  cap used to leave a zombie thread holding the worker's default executor
  (measured: 11 minutes waiting after the machine was already released).  A
  process is killed with the cap;
* the Agent code holds nothing but the shell's URL and token — that is the
  whole point of the shell — so it needs none of the worker's objects, and
  running it out of process is the first step of running it elsewhere.

Wire: the parent writes one JSON job to stdin; the child answers with one
JSON result on stdout (logs go to stderr).  The result is the *evidence* the
flow needs, computed here where the rich rsa objects live — the parent never
rehydrates rsa types.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULT_SCHEMA = "rsa-child-result@v1"


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one rsa run against the shell described by ``job``."""

    from apps.v2.agent_engine.rsa.agent import RSAAgent, UserInstruction
    from apps.v2.agent_engine.rsa.pipeline import PipelineConfig
    from apps.v2.agent_shell import ShellRuntime

    from .evidence import begin_recording, bind_round_recorder, collect_evidence, take_recorded_rounds
    from .oom import detect_oom
    from .run_flow import _gpu_aware_backend_class

    base_url = str(job["shell_base_url"])
    token = str(job["shell_token"])
    integrity: dict[str, Any] = {"checked": False, "reason": "守卫未执行", "expected": True}
    config = PipelineConfig(
        store=Path(job["store"]),
        work=Path(job["work"]),
        backend=str(job.get("backend") or "small"),
        execution_backend="remote",
        remote_target=f"shell {base_url}",
        remote_username="",
        remote_password=None,
        remote_private_key=None,
    )
    if job.get("max_rounds"):
        config.max_rounds = int(job["max_rounds"])
    config.deepevol_integrity = integrity  # type: ignore[attr-defined]
    config.deepevol_shell_endpoint = (base_url, token)  # type: ignore[attr-defined]

    def runtime_factory(**_ignored: Any) -> Any:
        return ShellRuntime(base_url, token, name="rsa-shell")

    config.remote_backend = _gpu_aware_backend_class(integrity)(
        target=config.remote_target,
        runtime_factory=runtime_factory,
        username="",
        password=None,
        private_key=None,
    )
    bind_round_recorder()
    begin_recording()
    try:
        outcome = RSAAgent(config).run(
            UserInstruction(
                repository=str(job["repo_url"]),
                instruction=str(job["instruction"]),
                revision=str(job.get("revision") or ""),
                session_id=str(job.get("session_id") or ""),
            )
        )
    finally:
        rounds = take_recorded_rounds()

    status = str(getattr(getattr(outcome, "status", None), "value", "") or getattr(outcome, "status", "") or "")
    pending = getattr(outcome, "pending", None)
    pipeline = getattr(outcome, "pipeline", None)
    oom = detect_oom(outcome)
    try:
        evidence = collect_evidence(outcome, setup_rounds=rounds)
    except Exception:  # diagnostic only; never fail the run for it
        logger.warning("rsa child: evidence collection failed", exc_info=True)
        evidence = {}
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "error": str(getattr(outcome, "error", "") or ""),
        "pending_message": str(getattr(pending, "message", "") or ""),
        "escalation_md": str(getattr(pipeline, "escalation_md", "") or ""),
        "oom": None if oom is None else dataclasses.asdict(oom),
        "evidence": evidence,
        "setup_rounds": rounds,
        "integrity": dict(integrity),
    }


def main() -> int:
    logging.basicConfig(level=os.environ.get("DEEPEVOL_RSA_CHILD_LOG_LEVEL", "INFO"), stream=sys.stderr)
    try:
        job = json.loads(sys.stdin.read())
        if not isinstance(job, dict):
            raise ValueError("rsa child job must be a JSON object")
        result = run_job(job)
    except Exception as exc:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "pending_message": "",
            "escalation_md": "",
            "oom": None,
            "evidence": {},
            "setup_rounds": [],
            "integrity": {"checked": False, "reason": "子进程异常", "expected": True},
            "crashed": True,
        }
    sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
