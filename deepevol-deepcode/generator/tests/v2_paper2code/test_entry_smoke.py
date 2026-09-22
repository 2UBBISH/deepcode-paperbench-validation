"""D0: entry resolution from the blueprint or by heuristic, and the record-only smoke."""

from __future__ import annotations

import asyncio
from pathlib import Path

from apps.v2.agent.paper2code.entry_smoke import (
    ENTRY_BARE_LABEL,
    ENTRY_LABEL,
    EntryPoint,
    entry_command,
    entry_named_in_plan,
    find_entry,
    run_entry_smoke,
)
from apps.v2.agent.paper2code.execution.port import Job, JobResult

PLAN = """```yaml
complete_reproduction_plan:
  file_structure: |
    sapg/
    ├── scripts/
    │   └── train_agent.py   # entry point for every experiment
    ├── main.py
    └── sapg/
        └── model.py
  implementation_strategy: |
    Start with the model, then the trainer. The entry script wires configs.
```
"""


def _touch(root: Path, *rel: str) -> None:
    for r in rel:
        path = root / r
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("print('hi')\n")


def test_plan_annotation_wins_over_root_main(tmp_path: Path) -> None:
    _touch(tmp_path, "scripts/train_agent.py", "main.py", "sapg/model.py")
    assert entry_named_in_plan(PLAN) == "train_agent.py"
    entry = find_entry(PLAN, tmp_path)
    assert entry == EntryPoint(Path("scripts/train_agent.py"), "plan")
    assert entry_command(entry.path, "--help") == "cd scripts && python train_agent.py --help"


def test_plan_path_with_project_prefix_and_strategy_sentence(tmp_path: Path) -> None:
    plan = "file_structure: |\n  proj/\n  └── run_all.py\nimplementation_strategy: |\n  The entry is proj/run_all.py; run `python run_all.py`.\n"
    _touch(tmp_path, "run_all.py")
    assert find_entry(plan, tmp_path) == EntryPoint(Path("run_all.py"), "plan")


def test_heuristic_order_at_root(tmp_path: Path) -> None:
    _touch(tmp_path, "train.py", "run_b.py", "run_a.py", "experiment_1.py")
    assert find_entry("", tmp_path) == EntryPoint(Path("run_a.py"), "heuristic")
    (tmp_path / "main.py").write_text("")
    assert find_entry("no entry here", tmp_path) == EntryPoint(Path("main.py"), "heuristic")


def test_heuristic_unique_child_project_and_none(tmp_path: Path) -> None:
    _touch(tmp_path, "sapg/train.py", "tests/test_x.py", "docs/notes.py")
    assert find_entry("", tmp_path) == EntryPoint(Path("sapg/train.py"), "heuristic")
    _touch(tmp_path, "other/main.py")
    assert find_entry("", tmp_path) is None  # two child projects: ambiguous
    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_entry("", empty) is None
    assert find_entry("", tmp_path / "missing") is None


def test_plan_entry_that_does_not_exist_falls_back(tmp_path: Path) -> None:
    _touch(tmp_path, "main.py")
    plan = "file_structure: |\n  ├── cli.py  # entry point\n"
    assert find_entry(plan, tmp_path) == EntryPoint(Path("main.py"), "heuristic")


class _Port:
    def __init__(self, results: list[JobResult]) -> None:
        self.results = list(results)
        self.jobs: list[Job] = []

    async def run(self, job: Job) -> JobResult:
        self.jobs.append(job)
        return self.results.pop(0)

    async def close(self) -> None:
        pass


def _result(exit_code: int | None, *, timed_out: bool = False) -> JobResult:
    return JobResult(exit_code=exit_code, stdout="out", stderr="err", duration_s=1.25, machine="fake", timed_out=timed_out)


def test_smoke_help_ok_stops_after_one_job(tmp_path: Path) -> None:
    port = _Port([_result(0)])
    record = asyncio.run(run_entry_smoke(port, tmp_path, EntryPoint(Path("main.py"), "heuristic")))
    assert record["status"] == "ok"
    assert [j.label for j in port.jobs] == [ENTRY_LABEL]
    assert record["attempts"][0]["command"] == "python main.py --help"
    assert port.jobs[0].timeout_s == 60.0


def test_smoke_help_fails_then_bare_run(tmp_path: Path) -> None:
    port = _Port([_result(2), _result(0)])
    record = asyncio.run(run_entry_smoke(port, tmp_path, EntryPoint(Path("sub/run.py"), "plan")))
    assert record["status"] == "ok"
    assert [j.label for j in port.jobs] == [ENTRY_LABEL, ENTRY_BARE_LABEL]
    assert port.jobs[1].command == "cd sub && python run.py"
    assert [a["exit_code"] for a in record["attempts"]] == [2, 0]


def test_smoke_timeout_and_failure_are_recorded_not_raised(tmp_path: Path) -> None:
    port = _Port([_result(1), _result(None, timed_out=True)])
    record = asyncio.run(run_entry_smoke(port, tmp_path, EntryPoint(Path("main.py"), "heuristic")))
    assert record["status"] == "timeout"
    port = _Port([_result(1), _result(1)])
    record = asyncio.run(run_entry_smoke(port, tmp_path, EntryPoint(Path("main.py"), "heuristic")))
    assert record["status"] == "failed"
    assert len(record["attempts"]) == 2


def test_smoke_without_entry_or_port() -> None:
    assert asyncio.run(run_entry_smoke(_Port([]), Path("."), None))["status"] == "no_entry"
    record = asyncio.run(run_entry_smoke(None, Path("."), EntryPoint(Path("main.py"), "plan")))
    assert record["status"] == "no_port"
    assert record["entry"] == "main.py"
