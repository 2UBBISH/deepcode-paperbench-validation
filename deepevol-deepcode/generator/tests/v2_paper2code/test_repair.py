"""PLAN-3 item 5: the repair agent — classification, the tools' fences, one scripted round."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from apps.v2.agent.paper2code import repair
from apps.v2.agent_engine.paper2code.seams.llm_runtime import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest


class ScriptedProvider(LLMProvider):
    """Replies with a fixed sequence of tool calls, then a final message."""

    def __init__(self, script: list[tuple[str, dict]]) -> None:
        super().__init__(GenerationSettings())
        self.script = list(script)
        self.requests: list[list[dict]] = []

    def get_default_model(self) -> str:
        return "fake"

    async def chat_with_retry(self, messages, tools=None, model=None, max_tokens=None, temperature=None, reasoning_effort=None, tool_choice=None, retry_mode="standard", on_retry_wait=None) -> LLMResponse:
        self.requests.append(list(messages))
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "reasoning_tokens": 0}
        if not self.script:
            return LLMResponse(content="done: fixed the import", finish_reason="stop", usage=usage)
        name, args = self.script.pop(0)
        args = {k: (v() if callable(v) else v) for k, v in args.items()}  # lazy values: read the file as it is *now*
        return LLMResponse(content=None, tool_calls=[ToolCallRequest(id=f"c{len(self.requests)}", name=name, arguments=args)], finish_reason="tool_calls", usage=usage)


def rewrite(code: Path, path: str, content: str) -> list[tuple[str, dict]]:
    """A scripted read + whole-content edit of an existing file (write_file creates files only since T2)."""
    return [
        ("read_text_file", {"path": path}),
        ("edit_file", {"path": path, "old_string": lambda: (code / path).read_text(), "new_string": content}),
    ]


def _code(tmp_path: Path) -> Path:
    code = tmp_path / "generate_code"
    (code / "pkg").mkdir(parents=True)
    (code / "main.py").write_text("from pkg.model import Model\nprint(Model().run())\n")
    (code / "pkg" / "__init__.py").write_text("")
    (code / "pkg" / "model.py").write_text("class Model:\n    def run(self):\n        return undefined_name\n")
    (code / "requirements.txt").write_text("numpy\n")
    return code


def test_classify_separates_environment_from_code(tmp_path: Path) -> None:
    code = _code(tmp_path)
    mods = repair.repo_modules(code)
    assert mods == {"main", "pkg", "__init__", "model"}
    assert repair.classify("ModuleNotFoundError: No module named 'torch'", repo_modules_=mods)[0] == repair.ENVIRONMENT
    assert repair.classify("ModuleNotFoundError: No module named 'pkg.missing'", repo_modules_=mods)[0] == repair.CODE
    assert repair.classify("ImportError: cannot import name 'Foo' from 'pkg.model'", repo_modules_=mods)[0] == repair.CODE
    assert repair.classify("ImportError: cannot import name 'x' from 'scipy.sparse'", repo_modules_=mods)[0] == repair.ENVIRONMENT
    assert repair.classify("/bin/sh: 1: python3.8: not found", repo_modules_=mods)[0] == repair.ENVIRONMENT
    assert repair.classify("FileNotFoundError: [Errno 2] No such file or directory: 'data/train.npz'", repo_modules_=mods)[0] == repair.ENVIRONMENT
    assert repair.classify("NameError: name 'undefined_name' is not defined", repo_modules_=mods)[0] == repair.CODE
    assert repair.classify("RuntimeError: Found no NVIDIA driver on your system", repo_modules_=mods) == (repair.GPU, "the code wants CUDA and this machine has no GPU")
    assert repair.classify("unimportable: experiments: ModuleNotFoundError: No module named 'experiments'; sapg: RuntimeError: CUDA unavailable", repo_modules_={"sapg", "experiments"})[0] == repair.GPU  # the machine first
    assert repair.gpu_needed("isaacgym is a GPU-only build; the environment cannot be completed on this machine")
    assert repair.gpu_needed("Torch reports CUDA unavailable and the model requires a GPU")
    assert not repair.gpu_needed("numpy<2 is missing from the container")


def test_repo_modules_sees_nested_trees_as_the_code_s_own(tmp_path: Path) -> None:
    # S1 ①: the entry may run from src/ or with a PYTHONPATH — a missing `pkg` there is the code's problem
    code = tmp_path / "generate_code"
    (code / "src" / "pkg" / "sub").mkdir(parents=True)
    (code / "src" / "pkg" / "__init__.py").write_text("")
    (code / "src" / "pkg" / "sub" / "core.py").write_text("x = 1\n")
    (code / "scripts" / "train.py").parent.mkdir()
    (code / "scripts" / "train.py").write_text("import pkg\n")
    (code / "__pycache__").mkdir()
    (code / "__pycache__" / "junk.py").write_text("")
    (code / ".venv" / "lib").mkdir(parents=True)
    (code / ".venv" / "lib" / "numpy.py").write_text("")
    mods = repair.repo_modules(code)
    assert mods == {"src", "pkg", "sub", "core", "scripts", "train", "__init__"}
    assert repair.classify("ModuleNotFoundError: No module named 'pkg.sub'", repo_modules_=mods)[0] == repair.CODE
    assert repair.classify("ModuleNotFoundError: No module named 'numpy'", repo_modules_=mods)[0] == repair.ENVIRONMENT
    assert repair.repo_modules(tmp_path / "missing") == set()


def test_messages_carry_evidence_files_and_denylist(tmp_path: Path) -> None:
    code = _code(tmp_path)
    verdict = {"verdict": "FAIL", "rung": "G2", "passed_expected": 1, "expected_n": 3, "exit_code": 1, "statuses": {"criteria_G2.py::test_run_completes": "failed", "criteria_G2.py::test_artifact_00": "passed"}, "failure_tails": {"criteria_G2.py::test_run_completes": "Traceback ...\nNameError: name 'undefined_name' is not defined"}}
    messages = repair.build_messages(goal="run it", criterion={"rungs": [{"rung": "G2", "command": "python main.py", "workdir": "/workspace/repo", "artifacts": ["/workspace/out/x.json"]}]}, verdict=verdict, classification=(repair.CODE, "no environment signature"), code_dir=code, denylist=("https://github.com/x/y",), round_no=2, previous_summaries=["round 1: renamed a variable"])
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "Repair round 2" in user
    assert "python main.py" in user
    assert "criteria_G2.py::test_run_completes" in user
    assert "NameError" in user
    assert "main.py, pkg/__init__.py, pkg/model.py" in user
    assert "https://github.com/x/y" in user
    assert "round 1: renamed a variable" in user


def test_tools_stay_inside_the_repository_and_refuse_the_denylist(tmp_path: Path) -> None:
    code = _code(tmp_path)
    probes: list[str] = []
    registry, state = repair.build_tools(code, probe=lambda cmd, t: probes.append(cmd) or "ok\n", denylist=("https://github.com/x/y",))
    assert sorted(t["function"]["name"] for t in registry.get_definitions()) == ["edit_file", "finish", "list_directory", "read_text_file", "run_in_container", "write_file"]
    listing = asyncio.run(registry.get("read_text_file").execute(path="pkg/model.py"))
    assert listing.startswith("     1\tclass Model")  # numbered, as edit_file's old_string must not be
    assert "Access denied" in asyncio.run(registry.get("read_text_file").execute(path="../outside.py"))
    assert "[FILE] main.py" in asyncio.run(registry.get("list_directory").execute(path="."))
    out = json.loads(asyncio.run(registry.get("edit_file").execute(path="pkg/model.py", old_string="return undefined_name", new_string="return 1")))
    assert out["status"] == "ok"
    assert (code / "pkg" / "model.py").read_text().endswith("return 1\n")
    assert "Access denied" in asyncio.run(registry.get("write_file").execute(path="/etc/passwd", content="x"))
    refused = asyncio.run(registry.get("run_in_container").execute(command="git clone https://github.com/x/y"))
    assert refused.startswith("BLOCKED:")
    assert probes == []
    assert asyncio.run(registry.get("run_in_container").execute(command="python main.py")) == "ok\n"
    assert probes == ["python main.py"]
    assert asyncio.run(registry.get("finish").execute(summary="?", attribution="hardware")).startswith("Error: attribution")
    assert state.finished is None
    asyncio.run(registry.get("finish").execute(summary="fixed"))
    assert state.finished == "fixed"
    assert state.attribution == repair.CODE
    assert state.written == ["pkg/model.py"]
    assert state.tool_calls == 9


def test_one_scripted_round_edits_probes_and_finishes(tmp_path: Path) -> None:
    code = _code(tmp_path)
    provider = ScriptedProvider(
        [
            ("read_text_file", {"path": "pkg/model.py"}),
            ("edit_file", {"path": "pkg/model.py", "old_string": "return undefined_name", "new_string": "return 42"}),
            ("run_in_container", {"command": "python main.py"}),
            ("finish", {"summary": "defined the missing name"}),
        ]
    )
    messages = repair.build_messages(goal="run it", criterion=None, verdict={"verdict": "FAIL", "log_tail": "NameError: name 'undefined_name' is not defined"}, classification=(repair.CODE, "x"), code_dir=code)
    outcome = asyncio.run(repair.run_repair_round(provider, model="fake", code_dir=code, messages=messages, probe=lambda cmd, t: "42\n"))
    assert outcome.summary == "defined the missing name"
    assert outcome.written == ["pkg/model.py"]
    assert outcome.edited == ["pkg/model.py"]
    assert outcome.created == []
    assert outcome.probes == 1
    assert outcome.probes_after_write == 1
    assert outcome.tool_calls == 4
    assert outcome.finished is True
    assert outcome.stop_reason == "callback_stop"
    assert outcome.attribution == repair.CODE
    assert (code / "pkg" / "model.py").read_text().endswith("return 42\n")
    record = outcome.record()
    assert record["usage"]["total_tokens"] > 0
    assert record["finished"] is True
    assert record["probes_after_write"] == 1


def test_finish_can_attribute_the_failure_to_the_environment(tmp_path: Path) -> None:
    # S1 ②: the agent's structured way out when the cause is not in the code
    code = _code(tmp_path)
    provider = ScriptedProvider([("read_text_file", {"path": "main.py"}), ("finish", {"summary": "libgl is missing", "attribution": "environment"})])
    messages = repair.build_messages(goal="run it", criterion=None, verdict={"verdict": "FAIL", "log_tail": "ImportError: libGL.so.1"}, classification=(repair.CODE, "x"), code_dir=code)
    outcome = asyncio.run(repair.run_repair_round(provider, model="fake", code_dir=code, messages=messages, probe=None))
    assert outcome.attribution == repair.ENVIRONMENT
    assert outcome.summary == "libgl is missing"
    assert outcome.written == []
    assert outcome.record()["attribution"] == "environment"
    assert "attribution" in repair.SYSTEM_PROMPT
    assert "requirements.txt (that triggers" not in repair.SYSTEM_PROMPT


def test_tool_call_budget_stops_the_round(tmp_path: Path) -> None:
    code = _code(tmp_path)
    provider = ScriptedProvider([("list_directory", {"path": "."})] * 10)
    outcome = asyncio.run(repair.run_repair_round(provider, model="fake", code_dir=code, messages=repair.build_messages(goal="g", criterion=None, verdict=None, classification=(repair.CODE, "x"), code_dir=code), probe=None, max_tool_calls=3))
    # two calls spent, the third slot is finish's: every further list is refused unspent until the runner's own
    # iteration cap (max_tool_calls + 5) ends the round
    assert outcome.tool_calls == 2
    assert outcome.stop_reason == "max_iterations"
    assert outcome.written == []
    assert outcome.finished is False  # S9: six rounds of six ended like this — the record now says so


def test_an_unproduced_output_artifact_is_not_a_missing_asset(tmp_path: Path) -> None:
    # sapg-2 GPU run, 2026-09-18: the run died on `ModuleNotFoundError: sapg.networks` (the code's own module) and the
    # artifact tests then reported FileNotFoundError on /workspace/out/…/checkpoint_final.pt — that is the consequence
    from apps.v2.agent.paper2code.repair_loop import failure_text

    verdict = {
        "verdict": "FAIL", "rung": "G2",
        "failure_tails": {
            "test_artifact_04_workspace_out_smoke_checkpoint_final_pt": "FileNotFoundError: [Errno 2] No such file or directory: '/workspace/repo/workspace/out/smoke/checkpoint_final.pt'",
            "test_run_completes": "Traceback ...\n    from .networks import ActorNetwork\nModuleNotFoundError: No module named 'sapg.networks'",
        },
        "log_tail": "6 failed",
    }
    text = failure_text(verdict)
    assert "sapg.networks" in text
    assert "checkpoint_final.pt" not in text  # the artifact tests' tails are not routing evidence
    assert repair.classify(text, repo_modules_={"sapg", "main"}) == (repair.CODE, "the repository's own module or name is missing")
    only_artifacts = "FileNotFoundError: No such file or directory: '/workspace/out/smoke/history.json'"
    assert repair.classify(only_artifacts, repo_modules_={"sapg"})[0] == repair.CODE
    assert repair.classify("FileNotFoundError: [Errno 2] No such file or directory: 'data/train.npz'", repo_modules_={"sapg"})[0] == repair.ENVIRONMENT


def test_probe_budget_refuses_further_probes_but_not_writes(tmp_path: Path) -> None:
    # sapg-2 GPU run 6: 28 and 34 probes out of 40 calls, the second round wrote nothing
    code = _code(tmp_path)
    registry, state = repair.build_tools(code, probe=lambda cmd, t: "ok\n", max_probes=2)
    assert asyncio.run(registry.get("run_in_container").execute(command="a")) == "ok\n"
    assert asyncio.run(registry.get("run_in_container").execute(command="b")) == "ok\n"
    refused = json.loads(asyncio.run(registry.get("run_in_container").execute(command="c")))
    assert refused["status"] == "error"
    assert "probe budget of 2" in refused["message"]
    assert len(state.probes) == 2
    assert state.tool_calls == 3  # the refused probe still costs a call
    out = json.loads(asyncio.run(registry.get("write_file").execute(path="smoke_note.py", content="print(1)\n")))
    assert out["status"] == "ok"
    assert "at most 15 probes" in repair.SYSTEM_PROMPT


# --- T2 (2026-09-18): edits instead of rewrites, new files only, syntax check, the finish nudge, traceback excerpts


def test_edit_file_needs_a_read_and_a_unique_match(tmp_path: Path) -> None:
    code = _code(tmp_path)
    (code / "pkg" / "model.py").write_text("class Model:\n    def run(self):\n        return x\n    def go(self):\n        return x\n")
    registry, state = repair.build_tools(code, probe=None)
    edit = registry.get("edit_file")
    assert asyncio.run(edit.execute(path="pkg/model.py", old_string="return x", new_string="return 1")).startswith("Error: read pkg/model.py")
    asyncio.run(registry.get("read_text_file").execute(path="pkg/model.py"))
    twice = asyncio.run(edit.execute(path="pkg/model.py", old_string="return x", new_string="return 1"))
    assert "occurs 2 times" in twice
    missing = asyncio.run(edit.execute(path="pkg/model.py", old_string="return y", new_string="return 1"))
    assert missing.startswith("Error: old_string not found")
    assert "closest line 3" in missing
    assert asyncio.run(edit.execute(path="pkg/model.py", old_string="", new_string="1")).startswith("Error: old_string is empty")
    assert asyncio.run(edit.execute(path="nope.py", old_string="a", new_string="b")).startswith("Error: not a file")
    one = json.loads(asyncio.run(edit.execute(path="pkg/model.py", old_string="def run(self):\n        return x", new_string="def run(self):\n        return 1")))
    assert one == {"status": "ok", "path": "pkg/model.py", "replacements": 1, "chars": len((code / "pkg" / "model.py").read_text())}
    both = json.loads(asyncio.run(edit.execute(path="pkg/model.py", old_string="return x", new_string="return 2", replace_all=True)))
    assert both["replacements"] == 1  # only `go` still had it
    assert (code / "pkg" / "model.py").read_text().count("return 2") == 1
    assert state.edited == ["pkg/model.py"]
    assert state.written == ["pkg/model.py"]


def test_writes_that_break_the_syntax_are_refused_and_leave_the_file_alone(tmp_path: Path) -> None:
    # sapg-s9-off round 3: algorithm.py overwritten with a 2k fragment, main.py with a 4k one; the next trial failed
    # on the agent's own NameError and round 4 was spent putting the files back
    code = _code(tmp_path)
    before = (code / "pkg" / "model.py").read_text()
    registry, state = repair.build_tools(code, probe=None)
    asyncio.run(registry.get("read_text_file").execute(path="pkg/model.py"))
    out = asyncio.run(registry.get("edit_file").execute(path="pkg/model.py", old_string="    def run(self):", new_string="    def run(self:"))
    assert out.startswith("Error: the edit would leave pkg/model.py unparsable (SyntaxError")
    assert (code / "pkg" / "model.py").read_text() == before
    assert state.written == []
    out = asyncio.run(registry.get("write_file").execute(path="pkg/new.py", content="def f(:\n"))
    assert out.startswith("Error: pkg/new.py would not parse")
    assert not (code / "pkg" / "new.py").exists()
    ok = json.loads(asyncio.run(registry.get("write_file").execute(path="pkg/new.py", content="def f():\n    return 1\n")))
    assert ok["status"] == "ok"
    assert state.created == ["pkg/new.py"]
    assert "pkg/new.py" in state.read  # a file the agent wrote counts as read for edit_file


def test_write_file_creates_only_and_only_source_shapes(tmp_path: Path) -> None:
    # sapg-s9-on rounds 3–4: main.py.flatpatch and _patch_import.txt went into the product
    code = _code(tmp_path)
    registry, state = repair.build_tools(code, probe=None)
    write = registry.get("write_file")
    assert "exists; change it with edit_file" in asyncio.run(write.execute(path="main.py", content="print(1)\n"))
    assert (code / "main.py").read_text().startswith("from pkg.model")
    assert "not a source file" in asyncio.run(write.execute(path="main.py.flatpatch", content="x"))
    assert "not a source file" in asyncio.run(write.execute(path="patch.bin", content="x"))
    assert not (code / "main.py.flatpatch").exists()
    ok = json.loads(asyncio.run(write.execute(path="configs/smoke.yaml", content="steps: 1\n")))
    assert ok["path"] == "configs/smoke.yaml"
    assert state.created == ["configs/smoke.yaml"]


def test_the_nudge_arrives_once_at_the_thirtieth_call(tmp_path: Path) -> None:
    code = _code(tmp_path)
    registry, state = repair.build_tools(code, probe=None, max_tool_calls=12, nudge_at=4)
    outs = [asyncio.run(registry.get("list_directory").execute(path=".")) for _ in range(6)]
    assert "[budget: 4 of 12 tool calls used" in outs[3]
    assert not any("[budget:" in o for o in outs[:3] + outs[4:])
    assert state.nudged is True
    assert repair.NUDGE_AT == 30


def test_the_last_tool_call_is_reserved_for_finish(tmp_path: Path) -> None:
    # T2 rerun: two rounds read the reminder at 30 and edited into the cap without a word; the trial judged a tree
    # the agent never declared done
    code = _code(tmp_path)
    registry, state = repair.build_tools(code, probe=lambda cmd, t: "ok\n", max_tool_calls=4, nudge_at=2)
    assert "[FILE] main.py" in asyncio.run(registry.get("list_directory").execute(path="."))
    assert "[budget: 2 of 4" in asyncio.run(registry.get("list_directory").execute(path="."))
    assert "[FILE] main.py" in asyncio.run(registry.get("list_directory").execute(path="."))  # call 3, still free
    refused = asyncio.run(registry.get("run_in_container").execute(command="python main.py"))
    assert refused.startswith("Error: this is the last of 4 tool calls — only finish is allowed now")
    assert state.probes == []
    assert state.tool_calls == 3  # a refused call is not spent: the slot stays finish's
    assert asyncio.run(registry.get("edit_file").execute(path="main.py", old_string="a", new_string="b")).startswith("Error: this is the last")
    assert asyncio.run(registry.get("finish").execute(summary="ran out; the edit to main.py is unverified")) == "recorded; stop calling tools and reply with the same summary"
    assert state.finished.startswith("ran out")
    assert state.tool_calls == 4


def test_a_scripted_round_that_never_finishes_is_forced_to(tmp_path: Path) -> None:
    code = _code(tmp_path)
    provider = ScriptedProvider([("list_directory", {"path": "."})] * 6 + [("finish", {"summary": "forced"})])
    outcome = asyncio.run(repair.run_repair_round(provider, model="fake", code_dir=code, messages=repair.build_messages(goal="g", criterion=None, verdict=None, classification=(repair.CODE, "x"), code_dir=code), probe=None, max_tool_calls=4))
    # calls 1–3 list, call 4 is refused (reserved), the scripted 5th and 6th lists are refused too, then finish lands
    assert outcome.finished is True
    assert outcome.summary == "forced"


def test_the_prompt_names_empty_files_and_their_non_empty_twins(tmp_path: Path) -> None:
    # T13: the pinn repair round found the empty file by itself and then mis-fixed it; now it is told up front
    code = _code(tmp_path)
    (code / "pkg" / "pdes.py").write_text("")
    (code / "pdes.py").write_text("def get_pde():\n    return 1\n")
    assert repair.empty_files(code) == [{"path": "pkg/pdes.py", "twins": ["pdes.py"]}]
    user = repair.build_messages(goal="g", criterion=None, verdict={"verdict": "FAIL", "log_tail": "ImportError: cannot import name 'get_pde'"}, classification=(repair.CODE, "x"), code_dir=code)[1]["content"]
    assert "EMPTY (0 bytes)" in user
    assert "pkg/pdes.py (same name, non-empty: pdes.py)" in user
    (code / "pkg" / "pdes.py").write_text("x = 1\n")
    assert repair.empty_files(code) == []
    assert "EMPTY (0 bytes)" not in repair.build_messages(goal="g", criterion=None, verdict=None, classification=(repair.CODE, "x"), code_dir=code)[1]["content"]


def test_the_prompt_carries_the_source_around_the_traceback_frames(tmp_path: Path) -> None:
    # S9: every round opened with 15–17 probes looking at what the traceback already named
    code = _code(tmp_path)
    (code / "pkg" / "model.py").write_text("".join(f"line{n} = {n}\n" for n in range(1, 61)) + "raise ValueError('boom')\n")
    verdict = {
        "verdict": "FAIL", "rung": "G2",
        "failure_tails": {
            "test_run_completes": 'Traceback (most recent call last):\n  File "/workspace/repo/main.py", line 2, in <module>\n    print(Model().run())\n  File "/workspace/repo/pkg/model.py", line 61, in run\n    raise ValueError\n  File "/usr/lib/python3.10/site.py", line 9, in x\nValueError: boom',
            "test_artifact_00": "FileNotFoundError: /workspace/out/x.json",
        },
    }
    excerpt = repair.traceback_excerpts(verdict["failure_tails"]["test_run_completes"], code)
    assert excerpt.startswith("--- pkg/model.py lines 41-61 (traceback line 61) ---\n    41\tline41 = 41")  # innermost frame first
    assert "--- main.py lines 1-2 (traceback line 2) ---" in excerpt
    assert "site.py" not in excerpt
    user = repair.build_messages(goal="g", criterion=None, verdict=verdict, classification=(repair.CODE, "x"), code_dir=code)[1]["content"]
    assert "Source around the traceback frames" in user
    assert "    61\traise ValueError('boom')" in user
    assert "old_string" in repair.SYSTEM_PROMPT
    assert "read_text_file` first" in repair.SYSTEM_PROMPT


def test_relative_output_paths_are_output_paths_too() -> None:
    # the compiler may write the artifact path relative to the repository (workspace/out/smoke/…)
    assert repair.is_output_path("/workspace/repo/workspace/out/smoke/checkpoint_final.pt")
    assert repair.is_output_path("/workspace/out/smoke/x.json")
    assert not repair.is_output_path("data/train.npz")
    text = "RuntimeError: mat1 and mat2 shapes cannot be multiplied\nFileNotFoundError: [Errno 2] No such file or directory: '/workspace/repo/workspace/out/smoke/checkpoint_final.pt'"
    assert repair.classify(text, repo_modules_={"sapg"}) == (repair.CODE, "no environment signature in the failure")


def test_artifact_tails_under_the_repository_do_not_route_to_the_environment() -> None:
    # sapg-s9-off / -on (2026-09-18): the compiler expected outputs under /workspace/repo/figures and
    # /workspace/repo/runs; their FileNotFound tails turned a ValueError and a bad CLI argument into "asset missing"
    from apps.v2.agent.paper2code.repair_loop import failure_text

    verdict = {
        "verdict": "FAIL", "rung": "G2",
        "failure_tails": {
            "test_run_completes": "ValueError: num_envs (1) must be divisible by num_blocks (6)",
            "test_artifact_00_figures_curves_npz": "FileNotFoundError: [Errno 2] No such file or directory: '/workspace/repo/figures/curves.npz'",
        },
        # the log tail is the whole pytest log, artifact tails included
        "log_tail": "ValueError: num_envs (1) must be divisible by num_blocks (6)\nFileNotFoundError: [Errno 2] No such file or directory: '/workspace/repo/figures/curves.npz'\n7 failed",
    }
    assert repair.classify(failure_text(verdict), repo_modules_={"sapg"}) == (repair.CODE, "no environment signature in the failure")
    only_log = {"verdict": "FAIL", "log_tail": "ModuleNotFoundError: No module named 'torch'"}
    assert repair.classify(failure_text(only_log), repo_modules_={"sapg"})[0] == repair.ENVIRONMENT  # no tails: the log stands in
