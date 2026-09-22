"""C6: each of the four gates has a passing and a failing case."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from apps.v2.agent.paper2code import gates
from apps.v2.agent.paper2code.config import RunConfig, RunPaths, build_kernel_config
from apps.v2.agent_engine.paper2code.seams.config import KernelRuntime
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMProvider, LLMResponse


class _Probe(LLMProvider):
    def __init__(self, response: LLMResponse) -> None:
        super().__init__()
        self.response = response

    def get_default_model(self) -> str:
        return "DeepSeek-V4-Flash-Vision-Exp"

    async def chat_with_retry(self, messages, **kwargs) -> LLMResponse:
        return self.response


def _run(tmp_path: Path, **kw) -> tuple[RunConfig, RunPaths]:
    run = RunConfig(run_id="r1", paper_dir=str(tmp_path / "paper"), paper_sha256="x", **kw)
    return run, RunPaths(tmp_path / "run").ensure()


def test_preflight_passes_and_fails(tmp_path: Path) -> None:
    run, paths = _run(tmp_path, denylist=("github.com/a/b",))
    runtime = KernelRuntime(build_kernel_config(run, paths))
    ok = _Probe(LLMResponse(content="OK", usage={"reasoning_tokens": 0}))
    result = asyncio.run(gates.preflight(run, runtime, ok, expected_denylist=("github.com/a/b",), git_check=False))
    assert result.passed, result.detail
    # S2: the run's other model slots and the context window are on record
    assert result.detail["context_window"] == 1_000_000
    assert result.detail["experiment_model"] == "DeepSeek-V4-Flash-Vision-Exp"
    assert result.detail["figures_model"] == "DeepSeek-V4-Flash-Vision-Exp"

    small_run, _ = _run(tmp_path / "s", context_window=32768)
    small = asyncio.run(gates.preflight(small_run, KernelRuntime(build_kernel_config(small_run, paths)), ok, expected_denylist=(), git_check=False))
    assert not small.passed
    assert "context_window" in small.detail["failures"][0]

    runtime.config.agents.planning.model = "other-model"
    result = asyncio.run(gates.preflight(run, runtime, ok, expected_denylist=("github.com/a/b",), git_check=False))
    assert not result.passed
    assert "overrides" in result.detail["failures"][0]

    runtime.config.agents.planning.model = None
    runtime.config.agents.defaults.max_tokens = 8192
    assert not asyncio.run(gates.preflight(run, runtime, ok, expected_denylist=("github.com/a/b",), git_check=False)).passed

    runtime = KernelRuntime(build_kernel_config(run, paths))
    thinking = _Probe(LLMResponse(content="OK", usage={"reasoning_tokens": 12}))
    assert not asyncio.run(gates.preflight(run, runtime, thinking, expected_denylist=("github.com/a/b",), git_check=False)).passed
    broken = _Probe(LLMResponse(content="HTTP 401", finish_reason="error"))
    assert not asyncio.run(gates.preflight(run, runtime, broken, expected_denylist=("github.com/a/b",), git_check=False)).passed

    empty_run, _ = _run(tmp_path / "e")
    assert not asyncio.run(gates.preflight(empty_run, KernelRuntime(build_kernel_config(empty_run, paths)), ok, expected_denylist=("github.com/a/b",), git_check=False)).passed
    assert asyncio.run(gates.preflight(empty_run, KernelRuntime(build_kernel_config(empty_run, paths)), ok, expected_denylist=(), git_check=False)).passed


def test_plan_source_gate(tmp_path: Path) -> None:
    assert not gates.plan_source(tmp_path).passed
    (tmp_path / "planning_result_meta.json").write_text(json.dumps({"status": "success", "source": "generated", "plan_chars": 4000}))
    assert gates.plan_source(tmp_path).passed
    (tmp_path / "planning_result_meta.json").write_text(json.dumps({"status": "success", "source": "coerced_from_freeform"}))
    result = gates.plan_source(tmp_path)
    assert not result.passed
    assert result.detail["source"] == "coerced_from_freeform"


def test_implementation_status_gate() -> None:
    assert gates.implementation_status({"status": "completed", "inner_status": "completed", "files_completed": 9}).passed
    unverified = {"status": "incomplete", "inner_status": "unverified", "abort_reason": "no_tests_discovered", "files_completed": 9, "total_files": 9, "unimplemented_files": []}
    result = gates.implementation_status(unverified)
    assert result.passed
    assert result.detail["verified"] is False
    assert not gates.implementation_status({**unverified, "unimplemented_files": ["a.py"], "files_completed": 8}).passed
    # S3: the engine's tests run on the driver host; a local failure is recorded, not fatal (step 10 verifies on the machine)
    local_fail = gates.implementation_status({**unverified, "inner_status": "test_failed", "abort_reason": "generated_tests_failed"})
    assert local_fail.passed
    assert local_fail.detail["verified"] is False
    assert local_fail.detail["local_tests_failed"] is True
    assert not gates.implementation_status({**unverified, "inner_status": "test_failed", "unimplemented_files": ["a.py"], "files_completed": 8}).passed


def test_implementation_status_records_empty_files_without_failing(tmp_path: Path) -> None:
    # T13 (pinn, 2026-09-18): every tree had a 0-byte opt_for_pinns/src/pdes.py next to a full src/pdes.py
    code = tmp_path / "generate_code"
    (code / "pkg" / "src").mkdir(parents=True)
    (code / "src").mkdir()
    (code / "pkg" / "__init__.py").write_text("")
    (code / "pkg" / "src" / "pdes.py").write_text("")
    (code / "src" / "pdes.py").write_text("def get_pde():\n    return 1\n")
    (code / "pkg" / "empty.yaml").write_text("")
    (code / "pkg" / "data.bin").write_bytes(b"")
    (code / "README.md").write_text("x")
    gate = gates.implementation_status({"status": "incomplete", "inner_status": "unverified", "files_completed": 4, "total_files": 4}, code_dir=code)
    assert gate.passed
    assert gate.detail["empty_files"] == [
        {"path": "pkg/empty.yaml", "twins": []},
        {"path": "pkg/src/pdes.py", "twins": ["src/pdes.py"]},
    ]  # __init__.py and a .bin are not source; the twin is named
    assert "empty_files" not in gates.implementation_status({"status": "completed", "inner_status": "completed", "files_completed": 9}).detail
    result = gates.implementation_status({"status": "incomplete", "inner_status": "max_iterations", "abort_reason": "cap", "unimplemented_files": ["a.py"]})
    assert not result.passed
    assert result.detail["abort_reason"] == "cap"


def test_ownership_gate(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run").ensure()
    paths.paper_md.write_bytes(b"paper body\n")
    task = tmp_path / "task"
    (task / "generate_code" / "pkg").mkdir(parents=True)
    (task / "paper.md").write_bytes(b"paper body\n")
    for i in range(5):
        (task / "generate_code" / "pkg" / f"m{i}.py").write_text("x")
    assert gates.ownership(task, paths).passed
    (task / "generate_code" / "pkg" / "m0.py").unlink()
    result = gates.ownership(task, paths)
    assert not result.passed
    assert result.detail["generated_files"] == 4
    (task / "paper.md").write_bytes(b"another paper\n")
    result = gates.ownership(task, paths, min_files=1)
    assert not result.passed
    assert "differs" in result.detail["failures"][0]


def test_run_config_round_trips_the_new_slots(tmp_path: Path) -> None:
    import pytest

    run, _ = _run(tmp_path, experiment_model="DeepSeek-V4-Flash", context_window=128_000)
    run.validate()
    data = run.to_dict()
    assert data["experiment_model"] == "DeepSeek-V4-Flash"
    assert data["context_window"] == 128_000
    assert RunConfig.from_dict(data).context_window == 128_000
    # a run.json from before S2 loads with the defaults
    legacy = {k: v for k, v in data.items() if k not in {"experiment_model", "context_window"}}
    loaded = RunConfig.from_dict(legacy)
    assert loaded.experiment_model == "DeepSeek-V4-Flash-Vision-Exp"
    assert loaded.context_window == 1_000_000
    with pytest.raises(ValueError, match="context_window"):
        _run(tmp_path / "z", context_window=0)[0].validate()


def test_implementation_status_gate_records_the_fidelity_audit_without_judging_it(monkeypatch) -> None:
    # ADR 0004: the audit is a record — its findings land in the gate's detail, the verdict ignores them
    ok = {"status": "completed", "inner_status": "unverified", "abort_reason": "verification_disabled", "files_completed": 3, "total_files": 3, "unimplemented_files": []}
    monkeypatch.setenv("DEEPCODE_PAPER_FIDELITY", "1")
    missing = gates.implementation_status(ok)
    assert missing.passed
    assert missing.detail["paper_fidelity"]["error"] == "paper fidelity audit missing"
    failed = gates.implementation_status({**ok, "paper_fidelity": {"passed": False, "violations": ["write of x: bound section 1 not read before it"]}})
    assert failed.passed
    assert failed.detail["paper_fidelity"]["violations"]
    monkeypatch.setenv("DEEPCODE_PAPER_FIDELITY", "0")
    assert "paper_fidelity" not in gates.implementation_status(ok).detail
