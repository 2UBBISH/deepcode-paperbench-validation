"""Static syntax check of the generated tree (owner 09-20): compile-only, repair rounds with edit_file, no container."""

from __future__ import annotations

import asyncio
from pathlib import Path

from apps.v2.agent.paper2code import syntax_check
from tests.v2_paper2code.test_repair import ScriptedProvider

BROKEN = 'import torch\n\n\ndef f(n, device):\n    return torch.rand(n, device="cpu", device=device)\n'


def _tree(tmp_path: Path) -> Path:
    code = tmp_path / "generate_code"
    (code / "pkg").mkdir(parents=True)
    (code / "pkg" / "__init__.py").write_text("")
    (code / "pkg" / "prior.py").write_text(BROKEN)
    (code / "main.py").write_text("print('ok')\n")
    (code / "__pycache__").mkdir()
    (code / "__pycache__" / "junk.py").write_text("def (:\n")  # skipped directory
    (code / "notes.txt").write_text("not python (\n")
    return code


def test_compile_tree_catches_codegen_errors_not_only_parse_errors(tmp_path: Path) -> None:
    code = _tree(tmp_path)
    errors = syntax_check.compile_tree(code)
    assert [e["path"] for e in errors] == ["pkg/prior.py"]  # the repeated keyword parses; only compile() rejects it
    assert errors[0]["line"] == 5
    assert "keyword argument repeated" in errors[0]["message"]
    assert "    5> " in errors[0]["excerpt"]
    (code / "pkg" / "prior.py").write_text(BROKEN.replace(', device="cpu"', ""))
    assert syntax_check.compile_tree(code) == []


def test_messages_name_every_file_and_the_check_only_path_records_without_a_provider(tmp_path: Path) -> None:
    code = _tree(tmp_path)
    errors = syntax_check.compile_tree(code)
    messages = syntax_check.build_messages(errors, round_no=1, denylist=("https://github.com/x/y",))
    assert messages[0]["content"] == syntax_check.SYSTEM_PROMPT
    assert "pkg/prior.py:5" in messages[1]["content"]
    assert "https://github.com/x/y" in messages[1]["content"]
    events: list[tuple[str, dict]] = []
    record = asyncio.run(syntax_check.check_and_repair(code, provider_factory=None, model="fake", events=lambda name, **kw: events.append((name, kw))))
    assert record["rounds"] == []
    assert [e["path"] for e in record["remaining"]] == ["pkg/prior.py"]
    assert events == [("implement.syntax", {"stage": "initial", "errors": 1, "files": ["pkg/prior.py"]})]


def test_one_scripted_round_fixes_the_file_and_the_recompile_confirms(tmp_path: Path, monkeypatch) -> None:
    code = _tree(tmp_path)
    monkeypatch.setenv(syntax_check.ENV_ROUNDS, "2")
    provider = ScriptedProvider(
        [
            ("read_text_file", {"path": "pkg/prior.py"}),
            ("edit_file", {"path": "pkg/prior.py", "old_string": 'torch.rand(n, device="cpu", device=device)', "new_string": "torch.rand(n, device=device)"}),
            ("finish", {"summary": "dropped the duplicate device kwarg"}),
        ]
    )
    events: list[tuple[str, dict]] = []
    record = asyncio.run(syntax_check.check_and_repair(code, provider_factory=lambda: provider, model="fake", events=lambda name, **kw: events.append((name, kw))))
    assert len(record["initial_errors"]) == 1
    assert record["remaining"] == []
    assert len(record["rounds"]) == 1  # the second round is not spent once the tree compiles
    assert record["rounds"][0]["outcome"]["edited"] == ["pkg/prior.py"]
    assert record["rounds"][0]["outcome"]["finished"] is True
    assert record["rounds"][0]["errors_after"] == 0
    assert (code / "pkg" / "prior.py").read_text() == BROKEN.replace('device="cpu", ', "")
    assert [e[1]["stage"] for e in events] == ["initial", "round1"]


def test_disabled_and_zero_rounds(monkeypatch) -> None:
    monkeypatch.setenv(syntax_check.ENV_ENABLED, "0")
    assert syntax_check.enabled() is False
    monkeypatch.setenv(syntax_check.ENV_ENABLED, "1")
    assert syntax_check.enabled() is True
    monkeypatch.setenv(syntax_check.ENV_ROUNDS, "0")
    assert syntax_check.rounds() == 0
    monkeypatch.setenv(syntax_check.ENV_ROUNDS, "x")
    assert syntax_check.rounds() == syntax_check.DEFAULT_ROUNDS


def test_fidelity_session_governs_the_repair_round(tmp_path: Path) -> None:
    """ADR 0004 inside the syntax round: a file the plan points at is edited only after read_paper in an earlier
    reply, the trace records the edit, and the audit is clean on the repaired bytes."""
    from apps.v2.agent_engine.paper2code.workflows import source_fidelity as sf
    from tests.v2_paper2code.test_source_fidelity import PAPER, PLAN

    task = tmp_path / "task"
    code = task / "generate_code"
    (code / "project" / "src").mkdir(parents=True)
    (task / "paper.md").write_text(PAPER, encoding="utf-8")
    (task / "initial_plan.txt").write_text(PLAN, encoding="utf-8")
    sf.freeze_manifest(task, PLAN)
    gen = sf.FidelitySession(task, code)
    gen.read(file_path="project/src/loss.py")
    gen.next_turn()
    (code / "project" / "src" / "loss.py").write_text('def total(on, off, lam=1.0):\n    return f(on, lam=lam, lam=lam)\n', encoding="utf-8")
    gen.written("project/src/loss.py", gen.authorize("project/src/loss.py"))
    for rel, body in (("project/src/train.py", "train = 1\n"), ("project/src/__init__.py", ""), ("project/README.md", "docs\n")):
        if rel.endswith("train.py"):
            gen.read(file_path=rel)
            gen.read(file_path=rel)
            gen.next_turn()
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(body, encoding="utf-8")
        gen.written(rel, gen.authorize(rel))
    assert [e["path"] for e in syntax_check.compile_tree(code)] == ["project/src/loss.py"]

    provider = ScriptedProvider(
        [
            ("read_text_file", {"path": "project/src/loss.py"}),
            ("edit_file", {"path": "project/src/loss.py", "old_string": "lam=lam, lam=lam", "new_string": "lam=lam"}),  # refused: no read yet
            ("read_paper", {"file_path": "project/src/loss.py"}),
            ("edit_file", {"path": "project/src/loss.py", "old_string": "lam=lam, lam=lam", "new_string": "lam=lam"}),  # a later reply: allowed
            ("finish", {"summary": "removed the duplicate keyword"}),
        ]
    )
    record = asyncio.run(syntax_check.check_and_repair(code, provider_factory=lambda: provider, model="fake", task_dir=task))
    assert record["remaining"] == []
    outcome = record["rounds"][0]["outcome"]
    assert outcome["edited"] == ["project/src/loss.py"]
    tool_replies = [m for m in provider.requests[-1] if m.get("role") == "tool"]
    assert "SOURCE_READ_REQUIRED" in tool_replies[1]["content"]
    report = sf.audit(task, code)
    assert report["passed"] is True, report["violations"]
    assert report["read_receipts"] == 4
