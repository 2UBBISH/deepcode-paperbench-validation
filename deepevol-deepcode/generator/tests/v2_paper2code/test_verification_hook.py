"""C5: the engine's verification runs through the execution port when installed."""

from __future__ import annotations

import asyncio
from pathlib import Path

from apps.v2.agent.paper2code import verification_hook
from apps.v2.agent.paper2code.execution.port import Job, JobResult
from apps.v2.agent_engine.paper2code.seams.config import AgentDefaults, AgentsConfig, KernelConfig, KernelRuntime, use_runtime
from apps.v2.agent_engine.paper2code.workflows import code_implementation_workflow as impl


class _Port:
    def __init__(self) -> None:
        self.jobs: list[Job] = []

    async def run(self, job: Job) -> JobResult:
        self.jobs.append(job)
        return JobResult(exit_code=1, stdout="F.\n1 failed", stderr="x" * 70_000, duration_s=1.5, machine="i-test")

    async def close(self) -> None:
        pass


def test_engine_verification_goes_through_the_port(tmp_path: Path) -> None:
    code_dir = tmp_path / "generate_code"
    code_dir.mkdir()
    (code_dir / "test_a.py").write_text("import unittest\n")
    port = _Port()
    verification_hook.install_verification_runner(port)
    try:
        config = KernelConfig(agents=AgentsConfig(defaults=AgentDefaults(model="m")))
        with use_runtime(KernelRuntime(config)):
            workflow = impl.CodeImplementationWorkflow(require_verification=True)
            results = asyncio.run(workflow._verify_generated_code(code_dir))
    finally:
        verification_hook.uninstall_verification_runner()
    assert len(port.jobs) == 1
    job = port.jobs[0]
    assert job.workspace == code_dir
    assert job.command == "python3 -m unittest discover -v"
    assert job.timeout_s == 300
    assert results[0]["command_id"] == "unittest"
    assert results[0]["passed"] is False
    assert results[0]["exit_code"] == 1
    assert results[0]["duration_ms"] == 1500
    assert results[0]["output_truncated"] is True
    assert len(results[0]["stderr"]) <= 64 * 1024


def test_default_runner_is_local_when_nothing_installed() -> None:
    verification_hook.uninstall_verification_runner()
    config = KernelConfig(agents=AgentsConfig(defaults=AgentDefaults(model="m")))
    with use_runtime(KernelRuntime(config)):
        workflow = impl.CodeImplementationWorkflow()
    assert workflow.verification_runner is impl.run_verification
