"""C7: the whole line offline — scripted model, fake execution port, a synthetic EMA-Detect paper."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from apps.v2.agent.paper2code import figures
from tests.v2_paper2code.fake_vision import answer_probe
from apps.v2.agent.paper2code.driver import Driver, DriverError
from apps.v2.agent.paper2code.execution.port import Job, JobResult
from apps.v2.agent.paper2code.phases import PHASES, figures_on_but_undescribed
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMProvider, LLMResponse, ToolCallRequest

PLANNED_FILES = ["main.py", "src/__init__.py", "src/ema.py", "src/detector.py", "src/data.py", "tests/__init__.py", "tests/test_ema.py"]

PAPER = """# EMA-Detect: A Minimal Streaming Anomaly Detector

## Abstract

We present EMA-Detect, a lightweight streaming anomaly detector for univariate time series.
It maintains an exponential moving average and an exponential moving variance and flags a
point as anomalous when its standardized deviation exceeds a threshold.

## Method

Given x_t, the EMA is m_t = a * x_t + (1 - a) * m_{t-1}; the variance follows the same recursion.
A point is anomalous when |x_t - m_{t-1}| / sqrt(v_{t-1}) > k.

## Experiments

We inject point anomalies into a synthetic series and report F1 = 0.90 at k = 3.

## References

[1] Some Author. A related detector. https://github.com/example/related-detector
"""


def _plan(marker: str = "") -> str:
    body = f"""```yaml
complete_reproduction_plan:
  paper_info:
    title: "EMA-Detect"
  file_structure: |
    ema_detect/
    ├── main.py            # entry point: python main.py runs the synthetic benchmark
    ├── src/
    │   ├── __init__.py
    │   ├── ema.py
    │   ├── detector.py
    │   └── data.py
    └── tests/
        ├── __init__.py
        └── test_ema.py
  implementation_components:
    - name: ema
      file: src/ema.py
      purpose: exponential moving average and variance recursions
    - name: detector
      file: src/detector.py
      purpose: threshold rule on the standardized deviation
    - name: data
      file: src/data.py
      purpose: synthetic series with injected point anomalies
    - name: main
      file: main.py
      purpose: command-line entry that runs the benchmark and prints F1
  validation_approach:
    - run tests/test_ema.py with pytest
    - report F1 on the synthetic benchmark at k = 3
  environment_setup:
    python: "3.11"
    packages: [numpy, pytest]
  implementation_strategy: |
    Implement ema.py first, then detector.py, then data.py, then the tests.{marker}
"""
    padding = "".join(f"    # note {i}: keep the recursion numerically stable and document every threshold choice.\n" for i in range(40))
    return body + padding + "```\n"


class ScriptedProvider(LLMProvider):
    """Answers each engine agent by what it is asked; tracks files it 'wrote' across memory resets."""

    def __init__(self, repo_path: Path) -> None:
        super().__init__()
        self.repo_path = repo_path
        self.written: list[str] = []
        self.cloned = False
        self.structured = False
        self.calls: list[str] = []
        self._ids = 0
        self.revisions = 0

    def get_default_model(self) -> str:
        return "DeepSeek-V4-Flash-Vision-Exp"

    def _call(self, name: str, arguments: dict[str, Any]) -> LLMResponse:
        self._ids += 1
        return LLMResponse(content=None, tool_calls=[ToolCallRequest(id=f"call_{self._ids}", name=name, arguments=arguments)], finish_reason="tool_calls", usage=self._usage())

    @staticmethod
    def _usage() -> dict[str, int]:
        return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "reasoning_tokens": 0}

    async def chat_with_retry(self, messages, tools=None, model=None, max_tokens=None, temperature=None, reasoning_effort=None, tool_choice=None, retry_mode="standard", on_retry_wait=None) -> LLMResponse:
        system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
        user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
        names = {t["function"]["name"] for t in (tools or [])}
        text = lambda s: LLMResponse(content=s, usage=self._usage())  # noqa: E731

        if isinstance(user, list):  # image parts: the vision probe and the figure pass (a VLM answers both)
            parts = user
            user = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
            assert any(p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/") for p in parts)
            if user.startswith(figures.VISION_PROBE_PROMPT):
                self.calls.append("vision_probe")
                assert model == "DeepSeek-V4-Flash-Vision-Exp"  # the vision model, not the pinned phase model
                return text(answer_probe(messages))
            if user.startswith("This image is a figure from a research paper"):
                self.calls.append("figure")
                return text("Line plot. x: threshold k (1..5); y: detection rate (0..1); one curve per window size, three seeds shaded.")
            raise AssertionError(f"unexpected image request: {user[:80]}")

        if user.startswith("Reply with the single word OK"):
            self.calls.append("probe")
            return text("OK")
        if user.startswith("You are turning the free-text sections of a code reproduction blueprint"):
            self.calls.append("environment_spec")
            assert "## environment_setup" in user
            assert "python: \"3.11\"" in user
            return text('```json\n{"language": {"name": "python", "version": "3.11"}, "python_packages": ["numpy", {"name": "pytest", "spec": null}], "gpu": {"required": false, "reason": "CPU-only synthetic series"}, "datasets": [], "external_tools": [], "run_commands": ["python main.py"], "notes": []}\n```')
        if "=== PAPER CONTENT START ===" in user or "SEGMENTED DOCUMENT CONTEXT" in user:
            self.calls.append("plan")
            return text(_plan())
        if "feedback" in user.lower() and ("plan" in user.lower()) and "file_structure" in user:
            self.calls.append("revise")
            self.revisions += 1
            return text(_plan(marker=f" Revision {self.revisions}: {user[-60:].strip()[:0]}address the feedback."))
        if system.startswith("You are an expert academic paper reference analyzer"):
            self.calls.append("references")
            return text(
                "selected_references:\n  - title: related detector\n    github_info:\n"
                "      repository_url: https://github.com/example/related-detector\n"
                "      local_mirror: " + str(self.repo_path) + "\n"
            )
        if "mcp_github_downloader_git_clone" in names:
            self.calls.append("download")
            if not self.cloned:
                self.cloned = True
                match = re.search(r'target_path="([^"]+)/<repo-name>"', system)
                assert match, system[:300]
                return self._call("mcp_github_downloader_git_clone", {"repo_url": str(self.repo_path), "target_path": f"{match.group(1)}/related-detector"})
            return text("related-detector: success")
        if "mcp_command_executor_execute_commands" in names:
            self.calls.append("structure")
            if not self.structured:
                self.structured = True
                match = re.search(r"Target Directory: (.+)/generate_code/", user)
                assert match, user[:200]
                commands = "mkdir -p src tests\n" + "\n".join(f"touch {f}" for f in PLANNED_FILES)
                return self._call("mcp_command_executor_execute_commands", {"commands": commands, "working_directory": f"{match.group(1)}/generate_code"})
            return text("structure created")
        if "write_file" in names:
            self.calls.append("implement")
            pending = [f for f in PLANNED_FILES if f not in self.written]
            if pending:
                target = pending[0]
                self.written.append(target)
                content = f'"""{target}: part of EMA-Detect."""\n\n\ndef marker():\n    return "{target}"\n'
                if target.endswith("test_ema.py"):
                    content = "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n"
                return self._call("write_file", {"file_path": target, "content": content})
            return text("All files implemented")
        if system.startswith("You are an expert code implementation summarizer"):
            self.calls.append("summary")
            return text("**Core Purpose:** part of EMA-Detect\n**Public Interface:** marker()\n**Dependencies:** none\n")
        self.calls.append("other")
        return text("OK")


class FakePort:
    def __init__(self) -> None:
        self.jobs: list[Job] = []

    async def run(self, job: Job) -> JobResult:
        self.jobs.append(job)
        return JobResult(exit_code=0, stdout="1 passed in 0.01s\n", stderr="", duration_s=0.5, machine="fake")

    async def close(self) -> None:
        pass


FIGURE_TAIL = "\n![](assets/asset_1.png)\n\nFigure 1. Detection rate versus the threshold k.\n"


def _paper_dir(tmp_path: Path) -> Path:
    root = tmp_path / "ema-detect"
    root.mkdir()
    (root / "paper.md").write_text(PAPER + FIGURE_TAIL, encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets" / "asset_1.png").write_bytes(figures.probe_png())
    (root / "addendum.md").write_text("Use k = 3 for every reported number.\n", encoding="utf-8")
    (root / "blacklist.txt").write_text("github.com/authors/ema-detect-official\n", encoding="utf-8")
    (root / "rubric.json").write_text("{}")
    return root


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "related-detector"
    repo.mkdir()
    (repo / "detector.py").write_text("def related():\n    return 1\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x", "HOME": str(tmp_path)}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    return repo


async def _drive(driver: Driver, provider: ScriptedProvider, port: FakePort, *, until: str, ask: bool | None = None) -> str:
    await driver.open(provider=provider, port=port, git_check=False, ask=ask)
    try:
        return await driver.run_until(until)
    finally:
        await driver.close()


def test_run_until_environment_run_all_green(tmp_path: Path) -> None:
    paper = _paper_dir(tmp_path)
    repo = _repo(tmp_path)
    run_dir = tmp_path / "run"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", skip=("index",), run_id="t1")
    with pytest.raises(DriverError):
        Driver.init(run_dir, paper_dir=str(paper))

    provider, port = ScriptedProvider(repo), FakePort()
    outcome = asyncio.run(_drive(Driver(run_dir), provider, port, until="environment_run"))
    assert outcome == "completed", json.loads((run_dir / "status.json").read_text())

    status = json.loads((run_dir / "status.json").read_text())
    for name in PHASES[:-1]:
        assert status["phases"][name]["status"] == "completed", (name, status["phases"][name])
    assert status["phases"]["optimize"]["status"] == "pending"
    for gate in ("preflight", "plan_source", "implementation_status", "ownership"):
        assert status["gates"][gate]["passed"], (gate, status["gates"][gate])
    assert sorted(p.name for p in (run_dir / "phases").glob("*_*.json") if ".attempts" not in p.name and ".request" not in p.name) == [
        f"{i:02d}_{n}.json" for i, n in enumerate(PHASES[:-1], start=1)
    ]

    task_dir = run_dir / "workspace" / "tasks" / "paper_t1"
    assert (task_dir / "paper.md").read_bytes() == (run_dir / "input" / "paper.md").read_bytes()
    # figures: the VLM probe passed, the one figure was described from its image and put back after its reference;
    # the benchmark bytes and their hash are untouched
    assert status["gates"]["preflight"]["detail"]["vision"]["supported"] is True
    intake = json.loads((run_dir / "phases" / "01_intake.json").read_text())["result"]
    assert intake["figures"]["status"] == "ok"
    assert intake["figures"]["source"] == "model"
    assert intake["figures"]["described"] == 1
    raw = (run_dir / "input" / "paper.raw.md").read_bytes()
    assert raw.startswith((PAPER + FIGURE_TAIL).encode())
    assert b"# Addendum" in raw
    assert figures.BLOCK_START.encode() not in raw
    assert json.loads((run_dir / "run.json").read_text())["paper_sha256"] == hashlib.sha256(raw).hexdigest()
    enriched = (task_dir / "paper.md").read_text()
    assert enriched.index("![](assets/asset_1.png)") < enriched.index(figures.BLOCK_START) < enriched.index("Figure 1. Detection rate")
    assert "> Line plot. x: threshold k" in enriched
    assert provider.calls.count("figure") == 1
    assert (task_dir / "plan_versions" / "initial_plan.v00.generated.txt").exists()
    assert (task_dir / "code_base" / "related-detector" / "detector.py").exists()
    generated = sorted(str(p.relative_to(task_dir / "generate_code")) for p in (task_dir / "generate_code").rglob("*.py"))
    assert generated == sorted(PLANNED_FILES)
    implement = json.loads((run_dir / "phases" / "08_implement.json").read_text())["result"]
    assert implement["generation_status"] == "completed"
    assert implement["files_completed"] == 7
    # no execution before step 10 (owner 2026-09-20): upstream's post-generation `python3 -m pytest` run is off unless
    # PAPER2CODE_IMPLEMENT_VERIFY=1, so the status is "unverified" and no verification record exists; the coding agent
    # has no execute tools either, and nothing goes through the port
    assert implement["inner_status"] == "unverified"
    assert not implement["verification"]
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e for e in events if e.get("kind") == "implement.execution"][-1]["verification_run"] is False
    assert [j.label for j in port.jobs if j.label.startswith(("verify:", "execute_"))] == []
    assert all(j.label.startswith("environment_run:") for j in port.jobs)
    index = json.loads((run_dir / "phases" / "07_index.json").read_text())["result"]
    assert index["status"] == "skipped"
    plan = json.loads((run_dir / "phases" / "03_plan.json").read_text())["result"]
    assert plan["environment_spec"]["status"] == "ok"
    assert plan["environment_spec"]["packages"] == 2
    assert plan["environment_spec"]["gpu_required"] is False
    spec_file = json.loads((run_dir / "environment_spec.json").read_text())
    assert spec_file["spec"]["language"] == {"name": "python", "version": "3.11"}
    assert [p["name"] for p in spec_file["spec"]["python_packages"]] == ["numpy", "pytest"]
    assert spec_file["spec"]["run_commands"] == ["python main.py"]
    assert "environment_spec" in provider.calls
    compute = json.loads((run_dir / "phases" / "09_compute.json").read_text())["result"]
    assert compute["status"] == "decided"
    assert compute["decision"]["mode"] == "auto"
    assert compute["decision"]["tier"] == "economy"
    assert compute["decision"]["instance_type"] == "ecs.c7.xlarge"  # the cheapest machine in the line's catalogue
    assert compute["port_configured"] is False  # FakePort rents nothing
    assert compute["estimate"]["needs_gpu"] is False
    assert (run_dir / "phases" / "09_compute.request.json").exists()
    request = json.loads((run_dir / "phases" / "09_compute.request.json").read_text())
    assert request["interaction_type"] == "compute_review"
    assert request["questions"][0]["type"] == "multiple_choice"
    assert [c["value"] for c in request["questions"][0]["choices"]] == ["economy", "standard", "cancel"]
    env_run = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert env_run["verification_passed"] is None  # no local verification run happened (nothing to copy through)
    assert env_run["compile_check"]["passed"] is True
    assert env_run["requirements_installed"] is None  # the fake port never builds an image
    assert env_run["entry_smoke"]["entry"] == "main.py"
    assert env_run["entry_smoke"]["entry_source"] == "plan"
    assert env_run["entry_smoke"]["status"] == "ok"
    assert [j.label for j in port.jobs][-2:] == ["environment_run:compileall", "environment_run:entry"]
    assert port.jobs[-1].command == "python main.py --help"
    assert any(json.loads(line)["kind"] == "entry_smoke" for line in (run_dir / "events.jsonl").read_text().splitlines())
    events = [json.loads(line)["kind"] for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert "phase.started" in events
    assert "phase.finished" in events
    assert "gate" in events
    assert "plan" in provider.calls
    assert "download" in provider.calls
    assert "implement" in provider.calls

    # submit: four gates + environment_run completed -> the generated repository lands in the pool with a manifest
    pool = tmp_path / "pool"
    record = Driver(run_dir).submit(paper="ema-detect", trial="trial1", dest_root=pool)
    target = pool / "ema-detect" / "trial1"
    assert sorted(str(p.relative_to(target)) for p in target.rglob("*.py")) == sorted(PLANNED_FILES)
    assert record["files"] == len(PLANNED_FILES)
    assert set(record["sha256"]) == set(PLANNED_FILES)
    assert record["caliber"]["model"] == "DeepSeek-V4-Flash-Vision-Exp"
    assert record["caliber"]["thinking"] == "disabled"
    assert json.loads((run_dir / "submission.json").read_text())["target"] == str(target.resolve())
    assert json.loads((run_dir / "status.json").read_text())["submitted_at"]
    with pytest.raises(DriverError, match="exists"):
        Driver(run_dir).submit(paper="ema-detect", trial="trial1", dest_root=pool)
    assert Driver(run_dir).submit(paper="ema-detect", trial="trial1", dest_root=pool, force=True)["files"] == len(PLANNED_FILES)

    # rerun the plan: the old plan is archived, a new one generated, later phases superseded
    driver = Driver(run_dir)
    (run_dir / "jobs" / "image.json").write_text(json.dumps({"requirements_sha": "x", "image": "base", "status": "failed"}))
    moved = driver.rerun("plan")
    assert any("initial_plan.superseded" in m for m in moved)
    assert any(m.endswith(".superseded." + m.rsplit(".", 1)[-1]) and "image.json" in m for m in moved)  # environment_run's image state moved aside
    assert any("environment_spec.json.superseded" in m for m in moved)
    assert not (run_dir / "jobs" / "image.json").exists()
    assert driver.phase_status("implement") == "pending"
    outcome = asyncio.run(_drive(driver, provider, FakePort(), until="plan"))
    assert outcome == "completed"
    versions = sorted(p.name for p in (task_dir / "plan_versions").iterdir())
    assert any(v.startswith("initial_plan.superseded") for v in versions)
    assert "initial_plan.v00.generated.txt" in versions


def test_submit_refuses_before_the_gates(tmp_path: Path) -> None:
    from apps.v2.agent.paper2code.submit import check_submittable

    paper = _paper_dir(tmp_path)
    run_dir = tmp_path / "run-early"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", run_id="t0")
    with pytest.raises(DriverError, match="gate preflight not passed"):
        Driver(run_dir).submit(paper="ema-detect", trial="t", dest_root=tmp_path / "pool")
    assert not (tmp_path / "pool").exists()
    from apps.v2.agent.paper2code.submit import tree_kind

    gates = {g: {"passed": True} for g in ("preflight", "plan_source", "implementation_status", "ownership")}
    failed = {"gates": gates, "phases": {"compute": {"status": "completed"}, "environment_run": {"status": "failed"}}}
    assert check_submittable(failed) == ["environment_run is failed and compute is completed: neither the repaired tree nor a stage-9 tree"]
    running = {"gates": gates, "phases": {"compute": {"status": "completed"}, "environment_run": {"status": "running"}}}
    assert check_submittable(running)
    # owner's rule (2026-09-18 evening): PaperBench comparisons stop at stage 9 and grade that tree
    stage9 = {"gates": gates, "phases": {"compute": {"status": "completed"}, "environment_run": {"status": "pending"}}}
    assert check_submittable(stage9) == []
    assert tree_kind(stage9, None) == "stage9"
    repaired = {"gates": gates, "phases": {"compute": {"status": "completed"}, "environment_run": {"status": "completed"}}}
    assert tree_kind(repaired, None) == "repaired"
    assert tree_kind(repaired, "pre_repair") == "pre_repair"
    assert check_submittable({"gates": gates, "phases": {"compute": {"status": "pending"}}})  # not past stage 9 yet


def test_ask_stops_at_plan_review_and_reads_decisions(tmp_path: Path) -> None:
    paper = _paper_dir(tmp_path)
    repo = _repo(tmp_path)
    run_dir = tmp_path / "run-ask"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", skip=("index",), run_id="t2", ask=True)
    provider = ScriptedProvider(repo)

    outcome = asyncio.run(_drive(Driver(run_dir), provider, FakePort(), until="implement"))
    assert outcome == "waiting"
    request = run_dir / "phases" / "04_plan_review.request.json"
    decision = run_dir / "phases" / "04_plan_review.decision.json"
    assert request.exists()
    payload = json.loads(request.read_text())
    assert payload["interaction_type"] == "plan_review"
    assert payload["data"]["modification_round"] == 0
    status = json.loads((run_dir / "status.json").read_text())
    assert status["phases"]["plan_review"]["status"] == "waiting"
    assert status["phases"]["references"]["status"] == "pending"

    decision.write_text(json.dumps({"action": "modify", "feedback": "add a CLI entry point"}))
    driver = Driver(run_dir)
    outcome = asyncio.run(_step(driver, provider, "plan_review"))
    assert outcome == "waiting"
    task_dir = run_dir / "workspace" / "tasks" / "paper_t2"
    assert (task_dir / "plan_versions" / "initial_plan.v01.ai.txt").exists()
    assert "Revision 1" in (task_dir / "initial_plan.txt").read_text()
    assert json.loads(request.read_text())["data"]["modification_round"] == 1
    assert not decision.exists()  # consumed

    decision.write_text(json.dumps({"action": "approve"}))
    driver = Driver(run_dir)
    outcome = asyncio.run(_step(driver, provider, "plan_review"))
    assert outcome == "completed"
    review = json.loads((run_dir / "phases" / "04_plan_review.json").read_text())["result"]
    assert review["status"] == "approved"
    assert review["rounds"] == 1
    history = [json.loads(line)["event"] for line in (task_dir / "plan_review_history.jsonl").read_text().splitlines()]
    assert history[0] == "review_started"
    assert history[-1] == "review_approved"

    outcome = asyncio.run(_drive(Driver(run_dir), provider, FakePort(), until="implement"))
    assert outcome == "completed"

    # the second review point: compute stops with a request in ask_user shape until a decision file exists
    class ConfigurablePort(FakePort):
        def __init__(self) -> None:
            super().__init__()
            self.configured: list[dict[str, Any]] = []
            self.started = False
            self.lease = type("L", (), {"record": None})()

        def configure(self, *, instance_type=None, hard_cap_seconds=None) -> None:
            self.configured.append({"instance_type": instance_type, "hard_cap_seconds": hard_cap_seconds})

    port = ConfigurablePort()
    outcome = asyncio.run(_drive(Driver(run_dir), provider, port, until="environment_run"))
    assert outcome == "waiting"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["phases"]["compute"]["status"] == "waiting"
    request = json.loads((run_dir / "phases" / "09_compute.request.json").read_text())
    assert request["interaction_type"] == "compute_review"
    assert request["default"] == "economy"
    (run_dir / "phases" / "09_compute.decision.json").write_text(json.dumps({"action": "approve", "answers": ["standard"], "run_hours": 1.5}))
    outcome = asyncio.run(_drive(Driver(run_dir), provider, port, until="environment_run"))
    assert outcome == "completed"
    compute = json.loads((run_dir / "phases" / "09_compute.json").read_text())["result"]
    assert compute["decision"] == {"mode": "ask", "action": "approve", "tier": "standard", "instance_type": "ecs.c7.2xlarge", "run_hours": 1.5, "gpu": False, "hourly_price_cny": None, "in_stock": None, "escalation_type": None}
    assert port.configured == [{"instance_type": "ecs.c7.2xlarge", "hard_cap_seconds": 5400.0}]
    assert not (run_dir / "phases" / "09_compute.decision.json").exists()  # consumed
    assert (run_dir / "phases" / "09_compute.decision.consumed.json").exists()


async def _step(driver: Driver, provider: ScriptedProvider, phase: str) -> str:
    await driver.open(provider=provider, port=FakePort(), git_check=False)
    try:
        return await driver.step(phase)
    finally:
        await driver.close()


def test_github_urls_in_reference_report() -> None:
    from apps.v2.agent.paper2code.phases import github_urls_in

    report = "selected_references:\n  - url: https://github.com/a/b.\n  - https://www.github.com/a/b\n  - https://github.com/c/d-e_f)\nnothing else"
    assert github_urls_in(report) == ["https://github.com/a/b", "https://www.github.com/a/b", "https://github.com/c/d-e_f"]
    assert github_urls_in("no repositories were found for this paper") == []
    shorthand = "### 1. Isaac Gym\n- **Repository**: NVIDIA-Omniverse/IsaacGym\n- Stars: 3k\n  - Repository: openai/baselines\n- Repository: (likely private)\n"
    assert github_urls_in(shorthand) == ["https://github.com/NVIDIA-Omniverse/IsaacGym", "https://github.com/openai/baselines"]


def test_degenerate_reference_reports_are_named() -> None:
    from apps.v2.agent.paper2code.phases import reference_report_is_degenerate

    assert reference_report_is_degenerate("") == "empty reference report"
    assert "DEEPCODE_REFERENCE_MAX_ITERATIONS" in reference_report_is_degenerate(
        "I reached the maximum number of tool call iterations (8) without completing the task."
    )
    assert reference_report_is_degenerate("selected_references: []  # the paper cites no code") is None


def test_references_phase_retries_once_with_a_doubled_budget(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from apps.v2.agent.paper2code import phases

    seen: list[str | None] = []

    async def fake_reference_agent(dir_info, logger):
        seen.append(os.environ.get(phases.REFERENCE_ITERATIONS_ENV))
        if len(seen) == 1:
            return "I reached the maximum number of tool call iterations (40) without completing the task."
        return "selected_references:\n- https://github.com/x/y  # the one repository"

    monkeypatch.setattr(phases.engine, "orchestrate_reference_intelligence_agent", fake_reference_agent)
    monkeypatch.setenv(phases.REFERENCE_ITERATIONS_ENV, "40")
    events: list[tuple[str, dict]] = []
    ctx = SimpleNamespace(load_dir_info=lambda: {"reference_path": str(tmp_path / "reference.txt")}, events=lambda kind, **f: events.append((kind, f)), logger=None)
    result = asyncio.run(phases.phase_references(ctx))
    assert seen == ["40", "80"]  # the second call ran under the doubled cap …
    assert os.environ[phases.REFERENCE_ITERATIONS_ENV] == "40"  # … and the environment is back afterwards
    assert result["iterations_retry"] == {"iterations": 80, "after": 40}
    assert result["github_urls"] == ["https://github.com/x/y"]
    assert events[0][0] == "references.retry"
    assert events[0][1]["iterations"] == 80

    async def always_out(dir_info, logger):
        return "I reached the maximum number of tool call iterations (80) without completing the task."

    monkeypatch.setattr(phases.engine, "orchestrate_reference_intelligence_agent", always_out)
    with pytest.raises(phases.PhaseError, match="maximum number of tool call iterations"):
        asyncio.run(phases.phase_references(ctx))


def test_index_repository_stats_flags_the_full_index_fallback(tmp_path: Path) -> None:
    from apps.v2.agent.paper2code.phases import index_repository_stats

    def index(name: str, before: int, after: int, enabled: bool = True) -> None:
        (tmp_path / f"{name}_index.json").write_text(json.dumps({
            "repo_name": name,
            "analysis_metadata": {"pre_filtering_enabled": enabled, "files_before_filtering": before, "files_after_filtering": after},
            "file_summaries": [{"file": f"f{i}.py"} for i in range(after)],
            "relationships": [{"a": 1}, {"b": 2}],
        }))

    index("small", 23, 19)
    index("big", 263, 263)
    index("off", 10, 10, enabled=False)
    (tmp_path / "broken_index.json").write_text("{not json")
    stats = {s["repository"]: s for s in index_repository_stats(tmp_path)}
    assert stats["small"]["prefilter_fallback_suspected"] is False
    assert stats["small"]["files_analyzed"] == 19
    assert stats["small"]["file_summaries"] == 19
    assert stats["big"]["prefilter_fallback_suspected"] is True
    assert stats["off"]["prefilter_fallback_suspected"] is False
    assert stats["broken"]["error"] == "unreadable index file"
    assert index_repository_stats(tmp_path / "missing") == []


def test_figures_off_keeps_the_benchmark_bytes(tmp_path: Path) -> None:
    paper = _paper_dir(tmp_path)
    run_dir = tmp_path / "run"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", skip=("index",), run_id="t2", figures="off")
    provider, port = ScriptedProvider(_repo(tmp_path)), FakePort()
    assert asyncio.run(_drive(Driver(run_dir), provider, port, until="intake")) == "completed"
    status = json.loads((run_dir / "status.json").read_text())
    assert "vision" not in status["gates"]["preflight"]["detail"]
    assert "vision_probe" not in provider.calls
    intake = json.loads((run_dir / "phases" / "01_intake.json").read_text())["result"]
    assert intake["figures"] == {"status": "off"}
    assert not (run_dir / "input" / "paper.raw.md").exists()
    assert figures.BLOCK_START.encode() not in (run_dir / "input" / "paper.md").read_bytes()


def test_figures_on_refuses_when_no_figure_was_described() -> None:
    ok = {"status": "ok", "figures_found": 2, "described": 1, "skipped": [{"ref": "a", "reason": "lfs_pointer"}], "failed": []}
    assert figures_on_but_undescribed("on", ok) is None
    none_found = {"status": "ok", "figures_found": 0, "described": 0, "skipped": [], "failed": []}
    assert figures_on_but_undescribed("on", none_found) is None  # a paper without figure references is not a failure
    pointers = {"status": "ok", "figures_found": 3, "described": 0, "skipped": [{"ref": "a", "reason": "lfs_pointer"}, {"ref": "b", "reason": "too_large:9"}], "failed": [{"ref": "c", "reason": "empty reply"}]}
    reason = figures_on_but_undescribed("on", pointers)
    assert reason is not None
    assert reason.startswith("3 figure references, 0 described (empty reply, lfs_pointer, too_large)")
    assert "hydrate the assets" in reason
    assert figures_on_but_undescribed("on", {"status": "failed", "error": "HTTPError: 500"}) == "figure pass failed: HTTPError: 500"
    assert figures_on_but_undescribed("on", {"status": "unsupported", "reason": "probe"}) == "figure pass unsupported: probe"
    for mode in ("auto", "off"):  # only the explicit caliber refuses
        assert figures_on_but_undescribed(mode, pointers) is None
        assert figures_on_but_undescribed(mode, {"status": "failed", "error": "x"}) is None


def test_figures_on_with_lfs_pointer_assets_fails_intake(tmp_path: Path) -> None:
    paper = _paper_dir(tmp_path)
    (paper / "assets" / "asset_1.png").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 48213\n")
    run_dir = tmp_path / "run"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", skip=("index",), run_id="t3", figures="on")
    provider, port = ScriptedProvider(_repo(tmp_path)), FakePort()
    assert asyncio.run(_drive(Driver(run_dir), provider, port, until="intake")) != "completed"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["phases"]["intake"]["status"] == "failed"
    assert "--figures on but no figure was described: 1 figure references, 0 described (lfs_pointer)" in status["phases"]["intake"]["error"]
    assert "figure" not in provider.calls  # nothing was sent to the vision model: the bytes were a pointer
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert any(e.get("kind") == "figures" and e.get("status") == "refused" for e in events)


def test_rerun_environment_run_resets_the_tree_to_the_first_commit_and_relocate_rewrites_paths(tmp_path: Path) -> None:
    from apps.v2.agent.paper2code.code_repo import CodeRepo
    from apps.v2.agent.paper2code.submit import code_directory

    paper = _paper_dir(tmp_path)
    run_dir = tmp_path / "run"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", skip=("index",), run_id="t4", figures="off")
    provider, port = ScriptedProvider(_repo(tmp_path)), FakePort()
    assert asyncio.run(_drive(Driver(run_dir), provider, port, until="environment_run")) == "completed"
    driver = Driver(run_dir)
    code = code_directory(driver.run, driver.paths)
    repo = CodeRepo.for_run(run_dir, code)
    first = repo.commit("environment_run round 0")  # the local mode records no history itself; stand in for step 10's first commit
    (code / "main.py").write_text("# edited by a repair round\n")
    (code / "extra.py").write_text("print('later')\n")
    repo.commit("repair 1")
    (code / "uncommitted.py").write_text("x = 1\n")
    driver.rerun("environment_run")
    assert "# edited" not in (code / "main.py").read_text()
    assert not (code / "extra.py").exists()
    assert not (code / "uncommitted.py").exists()  # committed for the record, then removed with the reset
    assert repo.first_commit() == first
    assert driver.phase_status("environment_run") == "pending"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    reset = [e for e in events if e.get("kind") == "tree.reset"]
    assert reset
    assert reset[-1]["to"] == first
    assert reset[-1]["removed"] == 2
    # --keep-tree leaves the edits alone
    (code / "again.py").write_text("y = 2\n")
    driver.rerun("environment_run", keep_tree=True)
    assert (code / "again.py").exists()

    # a copied run directory: the absolute paths inside point at the old place until relocate rewrites them
    moved = tmp_path / "moved-run"
    shutil.copytree(run_dir, moved)
    assert str(run_dir) in (moved / "dir_info.json").read_text()
    out = Driver(moved).relocate()
    assert out["old_root"] == str(run_dir)
    assert out["new_root"] == str(moved.resolve())
    assert "dir_info.json" in out["files"]
    assert any(f.startswith("phases/") for f in out["files"])
    assert str(run_dir) not in (moved / "dir_info.json").read_text()
    assert Driver(moved).relocate()["files"] == []  # idempotent


def test_planning_fanout_runs_the_two_analysis_agents_before_the_planner(tmp_path: Path, monkeypatch) -> None:
    """PLAN-3 item 7 (VENDOR 11): run.json.planning_fanout=true → DEEPCODE_PLANNING_FANOUT=1 → Concept + Algorithm
    analysis on the paper, their outputs under "# Worker outputs" in the planner's message. Off by default."""
    paper = _paper_dir(tmp_path)
    repo = _repo(tmp_path)

    class Recording(ScriptedProvider):
        def __init__(self, repo_path):
            super().__init__(repo_path)
            self.planning: list[tuple[str, bool]] = []

        async def chat_with_retry(self, messages, tools=None, **kw):
            system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
            user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
            if isinstance(user, str) and "=== PAPER CONTENT START ===" in user:
                self.planning.append((system[:60], "# Worker outputs" in user))
            return await super().chat_with_retry(messages, tools=tools, **kw)

    monkeypatch.delenv("DEEPCODE_PLANNING_FANOUT", raising=False)
    monkeypatch.delenv("DEEPCODE_PLANNER_CONTEXT_WINDOW", raising=False)
    run_dir = tmp_path / "run-fanout"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", skip=("index",), run_id="t7", planning_fanout=True, context_window=200_000)
    assert json.loads((run_dir / "run.json").read_text())["planning_fanout"] is True
    provider = Recording(repo)
    assert asyncio.run(_drive(Driver(run_dir), provider, FakePort(), until="plan")) == "completed"
    assert os.environ["DEEPCODE_PLANNING_FANOUT"] == "1"
    assert os.environ["DEEPCODE_PLANNER_CONTEXT_WINDOW"] == "200000"
    assert len(provider.planning) == 3
    assert [worker for _, worker in provider.planning] == [False, False, True]  # the two analyses see the paper; the planner sees the paper + their outputs
    systems = [system for system, _ in provider.planning]
    assert any("COMPLETE implementation details" in s for s in systems)  # PAPER_ALGORITHM_ANALYSIS_PROMPT
    assert any("COMPREHENSIVE analysis" in s for s in systems)  # PAPER_CONCEPT_ANALYSIS_PROMPT
    attempts = [json.loads(line) for line in (run_dir / "workspace" / "tasks" / "paper_t7" / "planning_attempts.jsonl").read_text().splitlines() if line.strip()]
    assert attempts[-1]["fanout"] == ["ConceptAnalysisAgent", "AlgorithmAnalysisAgent"]
    assert attempts[-1]["segment_budget_chars"] is None  # a small paper: traditional (unsegmented) planning

    # the default: one planner call, no worker outputs, the switch reads "0"
    run_dir2 = tmp_path / "run-single"
    Driver.init(run_dir2, paper_dir=str(paper), compute="local", skip=("index",), run_id="t8")
    provider2 = Recording(repo)
    assert asyncio.run(_drive(Driver(run_dir2), provider2, FakePort(), until="plan")) == "completed"
    assert os.environ["DEEPCODE_PLANNING_FANOUT"] == "0"
    assert provider2.planning == [(provider2.planning[0][0], False)]


def test_plan_phase_reads_source_pointers_back_from_the_blueprint(tmp_path: Path, monkeypatch) -> None:
    """ADR 0004: the manifest is the plan read back — files bound to the sections their Section 2 paragraphs point at.
    The offline plan's Section 2 is a list without pointers, so every file is glue and the phase records that."""
    monkeypatch.setenv("DEEPCODE_PAPER_FIDELITY", "1")
    monkeypatch.setenv("DEEPCODE_PLAN_REPLANS", "0")
    paper = _paper_dir(tmp_path)
    repo = _repo(tmp_path)
    run_dir = tmp_path / "run"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", skip=("index",), run_id="t5")
    provider, port = ScriptedProvider(repo), FakePort()
    outcome = asyncio.run(_drive(Driver(run_dir), provider, port, until="plan"))
    status = json.loads((run_dir / "status.json").read_text())
    assert outcome == "completed", status["phases"]["plan"]
    record = json.loads((run_dir / "phases" / "03_plan.json").read_text())["result"]
    assert record["source_manifest"]["files"] == len(PLANNED_FILES)
    assert record["source_manifest"]["pointers"] == 0
    assert record["source_manifest"]["glue"] == len(PLANNED_FILES)  # the manifest itself is frozen at implement


def test_plan_phase_replans_once_when_the_blueprint_has_no_source_pointer(tmp_path: Path, monkeypatch) -> None:
    """A plan with no `Source:` line anywhere ignored the addendum: planned once more with that said
    (PLANNER_FEEDBACK_ENV); the second plan is accepted whatever it says."""
    from apps.v2.agent.paper2code import phases
    from apps.v2.agent_engine.paper2code.workflows import paper_readback as pr

    monkeypatch.setenv("DEEPCODE_PAPER_FIDELITY", "1")
    paper = _paper_dir(tmp_path)
    run_dir = tmp_path / "run"
    Driver.init(run_dir, paper_dir=str(paper), compute="local", skip=("index",), run_id="t6")
    seen: list[str] = []

    async def fake_preprocess(dir_info, logger):
        dir_info["use_segmentation"] = False
        return {"status": "skipped"}

    async def fake_plan(dir_info, logger, _unused, strict_plan_validation=True):
        seen.append(os.environ.get(pr.PLANNER_FEEDBACK_ENV, ""))
        path = Path(dir_info["paper_dir"]) / "initial_plan.txt"
        text = _plan()
        if len(seen) == 2:  # the re-plan points at the paper
            text = text.replace("      purpose: exponential moving average and variance recursions", "      purpose: \"exponential moving average and variance recursions. Source: §Method\"")
        path.write_text(text, encoding="utf-8")
        dir_info["initial_plan_path"] = str(path)
        (Path(dir_info["paper_dir"]) / "planning_result_meta.json").write_text(json.dumps({"status": "success", "source": "generated", "plan_chars": len(text)}))

    monkeypatch.setattr(phases.engine, "orchestrate_document_preprocessing_agent", fake_preprocess)
    monkeypatch.setattr(phases.engine, "orchestrate_code_planning_agent", fake_plan)
    provider, port = ScriptedProvider(_repo(tmp_path)), FakePort()
    outcome = asyncio.run(_drive(Driver(run_dir), provider, port, until="plan"))
    assert outcome == "completed"
    assert len(seen) == 2
    assert seen[0] == ""
    assert "NO `Source:" in seen[1]
    assert os.environ.get(pr.PLANNER_FEEDBACK_ENV) is None
    record = json.loads((run_dir / "phases" / "03_plan.json").read_text())["result"]
    assert record["source_manifest"]["pointers"] == 1
    assert record["source_manifest"]["paper_files"] == 1
    assert record["source_manifest"]["replans"] == 1
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [e["reason"] for e in events if e.get("kind") == "plan.replan"] == ["no_source_pointers"]
