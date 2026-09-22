"""PLAN-3 item 5: the repair loop — routing, commit → container → re-frozen pin → judge, bounds. Offline:
a real frozen ladder (RSA's Freezer with a fake collector), a fake container backend, seams for the
adjudicator and SetupX, the scripted provider from test_repair."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from apps.v2.agent.paper2code import repair, repair_loop
from apps.v2.agent.paper2code.code_repo import CodeRepo
from apps.v2.agent_engine.rsa.criterion import Artifact, Criterion, Ladder, Rung
from apps.v2.agent_engine.rsa.freezer import Freezer
from tests.v2_paper2code.test_repair import ScriptedProvider, _code, rewrite


class FakeBackend:
    def __init__(self) -> None:
        self.container_id = ""
        self.commands: list[str] = []
        self.checkpoints: list[str] = []
        self.rollbacks = 0
        self.closed = False
        self.head = "old"

    def attach(self, container_id, workdir=None):
        self.container_id = container_id

    def run(self, cmd, timeout=300, workdir=None, env=None):
        self.commands.append(cmd)
        if "git checkout" in cmd:
            self.head = cmd.split("--detach ")[1].split(" ")[0]
            return SimpleNamespace(exit_code=0, stdout=self.head + "\n", stderr="", success=True, output=self.head + "\n")
        return SimpleNamespace(exit_code=0, stdout="ok\n", stderr="", success=True, output="ok\n")

    def create_checkpoint(self, tag):
        self.checkpoints.append(tag)
        return "ck"

    def rollback_to_checkpoint(self, n_frames=1):
        self.rollbacks += 1
        return True

    def collect_test_ids(self, path, timeout=120):
        text = Path(path).read_text()
        return [f"{Path(path).name}::{line.split('(')[0].split()[1]}" for line in text.splitlines() if line.startswith("def test_")]

    def close(self):
        self.closed = True


def _frozen(tmp_path: Path, repo_url: str, commit: str):
    ladder = Ladder(
        repo_url=repo_url, commit=commit, goal="run it",
        rungs={
            Rung.G0: Criterion(rung=Rung.G0, repo_url=repo_url, commit=commit, command="true", workdir="/workspace/repo", timeout=60),
            Rung.G2: Criterion(rung=Rung.G2, repo_url=repo_url, commit=commit, command="python main.py", workdir="/workspace/repo", timeout=600, artifacts=[Artifact(path="/workspace/out/x.json", check="exists")]),
        },
    )
    return Freezer(tmp_path / "store").freeze(ladder, overwrite=True, collector=FakeBackend().collect_test_ids)


def _outcome(frozen, *, container="rsa-1", log_tail="NameError: name 'undefined_name' is not defined"):
    verdict = SimpleNamespace(verdict="FAIL", rung="G2", expected_n=2, passed_expected=1, pass_rate=0.5, missing=["criteria_G2.py::test_run_completes"], statuses={"criteria_G2.py::test_run_completes": "failed"}, status_counts={}, exit_code=1, parsed_any=True, duration_s=1.0, log_tail=log_tail, failure_tails={"criteria_G2.py::test_run_completes": log_tail}, reason="", tampered=None, changed_files=[], alarms=[])
    rung = SimpleNamespace(rung="G2", terminal=SimpleNamespace(value="failed"), note="", rounds=[], verdict=verdict)
    pipeline = SimpleNamespace(container_id=container, terminal="failed", reached="G0", tokens={}, outcome=SimpleNamespace(per_rung=[rung], reached="G0"))
    return SimpleNamespace(status=SimpleNamespace(value="failed"), frozen=frozen, pipeline=pipeline, compile=None, error="", pipeline_attempts=[])


def _verdict(rung: str, passed: bool):
    return SimpleNamespace(verdict="PASS" if passed else "FAIL", rung=rung, expected_n=1, passed_expected=int(passed), pass_rate=float(passed), missing=[] if passed else ["t"], statuses={}, status_counts={}, exit_code=0 if passed else 1, parsed_any=True, duration_s=0.1, log_tail="" if passed else "AssertionError: still broken", failure_tails={}, reason="", tampered=None, changed_files=[], alarms=[])


def _loop(tmp_path: Path, code: Path, repo: CodeRepo, frozen, provider_scripts: list, *, rounds=3, judge=None, environment_round=None, token_cap=repair_loop.TOKEN_CAP):
    backend = FakeBackend()
    served: list[str] = []
    providers = list(provider_scripts)
    loop = repair_loop.RepairLoop(
        config=SimpleNamespace(workdir="/workspace/repo", backend="small", base_image="x", max_steps=10, disclose_ids=True),
        outcome=_outcome(frozen), repo=repo, code_dir=code, goal="run it", rounds=rounds, store=tmp_path / "store", out_dir=tmp_path / "adj",
        provider_factory=lambda: ScriptedProvider(providers.pop(0)), model="fake",
        serve_repo=lambda git_dir: served.append(str(git_dir)) or "git://127.0.0.1:9418/repo.git",
        backend_factory=lambda config: backend, denylist=("https://github.com/x/y",), judge=judge, environment_round=environment_round, token_cap=token_cap,
    )
    return loop, backend, served


def test_code_round_commits_moves_the_container_refreezes_and_passes(tmp_path: Path) -> None:
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://127.0.0.1:9418/repo.git", first)
    judged: list[tuple[str, str]] = []

    def judge(backend, fl, round_no):
        judged.append((round_no, fl.ladder.commit))
        return [_verdict("G0", True), _verdict("G2", True)]

    script = [*rewrite(code, "pkg/model.py", "class Model:\n    def run(self):\n        return 42\n"), ("run_in_container", {"command": "python main.py"}), ("finish", {"summary": "defined the name"})]
    loop, backend, served = _loop(tmp_path, code, repo, frozen, [script], judge=judge)
    passed, records = loop.run()
    assert passed
    assert loop.stop_reason == "criterion passed"
    assert len(records) == 1
    rec = records[0].record()
    assert rec["kind"] == repair.CODE
    assert rec["agent"]["written"] == ["pkg/model.py"]
    assert rec["agent"]["probes"] == 1
    assert rec["passed"] is True
    new_sha = repo.head()
    assert new_sha != first
    assert rec["commit"] == new_sha
    assert judged == [(1, new_sha)]  # the ladder was re-frozen with the new pin before judging
    assert loop.frozen.ladder.commit == new_sha
    assert all(c.commit == new_sha for c in loop.frozen.ladder.rungs.values())
    assert loop.frozen.ladder_id != frozen.ladder_id
    assert served == [str(repo.git_dir)]
    assert backend.container_id == "rsa-1"
    assert backend.checkpoints == ["repair-probe-1"]
    assert backend.rollbacks == 1
    assert any("git checkout --quiet --force --detach " + new_sha in c for c in backend.commands)
    assert any("base64 -d > pkg/model.py" in c for c in backend.commands)  # the probe saw the agent's file
    assert backend.closed


def test_environment_signature_routes_to_setupx_and_the_agent_s_attribution_routes_the_next_round(tmp_path: Path) -> None:
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://127.0.0.1:9418/repo.git", first)
    env_rounds: list[int] = []
    # round 2 fails identically to round 1 — no signature stop, because the agent has already re-routed round 3
    verdicts_by_round = {1: [_verdict("G0", False)], 2: [_verdict("G0", False)], 3: [_verdict("G0", True), _verdict("G2", True)]}

    def judge(backend, fl, round_no):
        return verdicts_by_round[round_no]

    def environment_round(backend, last, round_no):
        env_rounds.append(round_no)
        return 5

    # round 2's agent changes requirements.txt — which alone forces nothing now — and says "environment"
    script = [*rewrite(code, "requirements.txt", "numpy\ntorch\n"), ("finish", {"summary": "torch is not importable in this container", "attribution": "environment"})]
    loop, _backend, _served = _loop(tmp_path, code, repo, frozen, [script], judge=judge, environment_round=environment_round)
    loop.outcome = _outcome(frozen, log_tail="ModuleNotFoundError: No module named 'torch'")
    passed, records = loop.run()
    assert passed
    kinds = [(r.round_no, r.kind, r.reason) for r in records]
    assert kinds[0][1] == repair.ENVIRONMENT  # a third-party module missing → SetupX first
    assert kinds[0][2] == "third-party module missing: torch"
    assert kinds[1][1] == repair.CODE  # G0 still failing with an AssertionError → the agent
    assert kinds[2][1] == repair.ENVIRONMENT  # the agent's attribution, not the heuristic, routes round 3
    assert kinds[2][2].startswith("the repair agent attributed the failure to the environment: torch is not importable")
    assert records[1].agent["attribution"] == "environment"
    assert records[1].commit != first  # the requirements change was still committed and judged
    assert records[1].verdicts[-1]["verdict"] == "FAIL"
    assert records[1].signature == records[0].signature  # same failure twice, but the route changed, so no stop
    assert env_rounds == [1, 3]
    assert records[0].environment_actions == 5
    assert records[2].passed


def test_agent_attribution_without_a_change_defers_to_setupx_without_judging(tmp_path: Path) -> None:
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://127.0.0.1:9418/repo.git", first)
    judged: list[int] = []
    env_rounds: list[int] = []

    def judge(backend, fl, round_no):
        judged.append(round_no)
        return [_verdict("G0", True), _verdict("G2", True)]

    def environment_round(backend, last, round_no):
        env_rounds.append(round_no)
        return 2

    loop, _b, _s = _loop(tmp_path, code, repo, frozen, [[("finish", {"summary": "libGL.so.1 is missing", "attribution": "environment"})]], judge=judge, environment_round=environment_round)
    passed, records = loop.run()
    assert passed
    assert [r.kind for r in records] == [repair.CODE, repair.ENVIRONMENT]
    assert records[0].error == ""  # "changed nothing" is not an error when the agent handed the round on
    assert records[0].commit == first
    assert records[0].verdicts == []
    assert judged == [2]  # round 1 was not judged: same code, same environment
    assert env_rounds == [2]
    assert records[1].passed


def test_same_failure_signature_twice_stops_the_loop(tmp_path: Path) -> None:
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://127.0.0.1:9418/repo.git", first)

    def judge(backend, fl, round_no):
        # the same AssertionError every round, only the timing and an address differ
        v = _verdict("G0", False)
        v.log_tail = f"took {round_no}.{round_no}s at 0x7f{round_no}a\nAssertionError: still broken"
        v.failure_tails = {"t": v.log_tail}
        return [v]

    scripts = [[*rewrite(code, "main.py", f"print({n})\n"), ("finish", {"summary": f"try {n}"})] for n in range(3)]
    loop, _b, _s = _loop(tmp_path, code, repo, frozen, scripts, judge=judge)
    passed, records = loop.run()
    assert not passed
    assert len(records) == 2  # round 0 → NameError; round 1 → AssertionError (new); round 2 → the same → stop
    assert records[0].signature == records[1].signature
    assert loop.stop_reason == f"round 2 reproduced the previous failure (signature {records[1].signature})"
    assert repair_loop.failure_signature({"log_tail": "line 12: boom 0xdeadbeef 3.5s"}) == repair_loop.failure_signature({"log_tail": "line 99:  boom  0x1 0.2 s"})
    assert repair_loop.failure_signature({"log_tail": "boom", "rung": "G0"}) != repair_loop.failure_signature({"log_tail": "boom", "rung": "G2"})


def test_rounds_exhausted_and_no_change_and_token_cap(tmp_path: Path) -> None:
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://127.0.0.1:9418/repo.git", first)

    def judge(backend, fl, round_no):
        v = _verdict("G0", False)
        v.log_tail = f"AssertionError: broken in a new way #{round_no}"  # distinct signatures: no early stop
        v.failure_tails = {"t": v.log_tail}
        return [v]

    # the agent edits every round but the judge never passes → rounds exhausted
    scripts = [[*rewrite(code, "main.py", f"print({n})\n"), ("finish", {"summary": f"try {n}"})] for n in range(3)]
    loop, _b, _s = _loop(tmp_path, code, repo, frozen, scripts, judge=judge)
    passed, records = loop.run()
    assert not passed
    assert len(records) == 3
    assert loop.stop_reason == "3 repair rounds used"
    assert [r.commit != first for r in records] == [True, True, True]

    # an agent that changes nothing ends the loop
    code2 = _code(tmp_path / "two")
    repo2 = CodeRepo.for_run(tmp_path / "two", code2)
    c2 = repo2.commit("round 0")
    loop2, _b2, _s2 = _loop(tmp_path / "two", code2, repo2, _frozen(tmp_path / "two", "git://x/repo.git", c2), [[("finish", {"summary": "looks fine"})]], judge=judge)
    passed2, records2 = loop2.run()
    assert not passed2
    assert loop2.stop_reason == "the repair agent changed nothing"
    assert records2[0].error == "the repair agent changed nothing"

    # the phase token cap stops code rounds before they start
    code3 = _code(tmp_path / "three")
    repo3 = CodeRepo.for_run(tmp_path / "three", code3)
    c3 = repo3.commit("round 0")
    loop3, _b3, _s3 = _loop(tmp_path / "three", code3, repo3, _frozen(tmp_path / "three", "git://x/repo.git", c3), [[("finish", {"summary": "x"})]], judge=judge, token_cap=1)
    loop3.tokens_used = 5
    passed3, _records3 = loop3.run()
    assert not passed3
    assert "token cap" in loop3.stop_reason


def test_zero_rounds_and_missing_container_do_nothing(tmp_path: Path) -> None:
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://x/repo.git", first)
    loop, _backend, _s = _loop(tmp_path, code, repo, frozen, [], rounds=0)
    assert loop.run() == (False, [])
    assert loop.stop_reason == "repair_rounds is 0"
    loop2, _b, _s2 = _loop(tmp_path, code, repo, frozen, [])
    loop2.container_id = ""
    assert loop2.run() == (False, [])
    assert "no container" in loop2.stop_reason


def test_run_line_pipeline_drives_the_four_boxes_on_rsa_s_pieces(tmp_path: Path, monkeypatch) -> None:
    """S8: RSA's run_pipeline replaced by the controller — asset gate as upstream, then 搭建环境 → 试跑 → (归因) →
    修复代码 → 试跑, on the same container, with the re-pinned ladder; the PipelineOutcome shape RSAAgent expects."""
    import contextlib

    from apps.v2.agent.paper2code import environment_controller as ec
    from apps.v2.agent.paper2code.repair_loop import ControllerSeams, run_line_pipeline
    from apps.v2.agent_engine.rsa import setupx_interop

    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://127.0.0.1:9418/repo.git", first)
    backend = FakeBackend()

    @contextlib.contextmanager
    def no_setupx(*args, **kwargs):
        yield None

    monkeypatch.setattr(setupx_interop, "setupx_configured", no_setupx)
    env_calls: list[tuple[int, str]] = []
    judge_calls: list[tuple[int, str]] = []
    verdicts = [
        [_verdict("G0", True), _verdict("G2", False)],  # after env round 1
        [_verdict("G0", True), _verdict("G2", False)],  # after env round 2 (the kickback fixed nothing visible)
        [_verdict("G0", True), _verdict("G2", True)],  # after the repair
    ]
    verdicts[0][1].log_tail = "ModuleNotFoundError: No module named 'torch'"
    verdicts[0][1].failure_tails = {"t": verdicts[0][1].log_tail}
    verdicts[1][1].log_tail = "NameError: name 'undefined_name' is not defined"
    verdicts[1][1].failure_tails = {"t": verdicts[1][1].log_tail}

    def environment_round(be, last, round_no):
        env_calls.append((round_no, str(last.get("log_tail") or "")[:20]))
        be.attach("rsa-9", "/workspace/repo")
        return ec.SetupResult(container_id="rsa-9", actions=4 + round_no)

    def judge(be, fl, round_no):
        judge_calls.append((round_no, fl.ladder.commit))
        return verdicts.pop(0)

    scripts = [[*rewrite(code, "pkg/model.py", "class Model:\n    def run(self):\n        return 42\n"), ("finish", {"summary": "defined the name"})]]
    states: list = []
    seams = ControllerSeams(
        repo=repo, code_dir=code, goal="run it", repair_rounds=3, environment_rounds=3, store=tmp_path / "store", out_dir=tmp_path / "adj",
        provider_factory=lambda: ScriptedProvider(scripts.pop(0)), model="fake", serve_repo=lambda git_dir: "git://127.0.0.1:9418/repo.git",
        denylist=("https://github.com/x/y",), environment_round=environment_round, judge=judge, on_state=states.append,
    )
    cfg = SimpleNamespace(workdir="/workspace/repo", backend="small", base_image="x", max_steps=10, disclose_ids=True, execution_backend="remote", remote_backend=backend, remote_target="ssh")
    result = run_line_pipeline(frozen, cfg, out_dir=tmp_path / "pipe", seams=seams)

    assert result.terminal == "success"
    assert result.reached == "G2"
    assert result.container_id == "rsa-9"
    assert [r.rung for r in result.outcome.per_rung] == ["G0", "G2"]
    assert result.outcome.per_rung[1].verdict.verdict == "PASS"
    state = states[0]
    assert state.passed
    assert [r.kind for r in state.records] == [ec.ENVIRONMENT, ec.ENVIRONMENT, ec.REPAIR]
    assert state.records[1].reason == "third-party module missing: torch"
    assert state.records[2].reason == "no environment signature in the failure"
    assert env_calls == [(1, ""), (2, "ModuleNotFoundError:")]  # round 1 has no kickback; round 2 carries the verdict
    new_sha = repo.head()
    assert new_sha != first
    assert judge_calls == [(1, first), (2, first), (3, new_sha)]  # the trial after the repair judges the re-pinned ladder
    assert result.frozen.ladder.commit == new_sha
    assert state.records[2].commit == new_sha
    assert backend.checkpoints == ["repair-probe-3"]
    assert backend.rollbacks == 1
    assert any("git checkout --quiet --force --detach " + new_sha in c for c in backend.commands)
    written = json.loads((tmp_path / "pipe" / "result.json").read_text())
    assert written["controller"]["rounds"][2]["agent"]["written"] == ["pkg/model.py"]
    assert written["terminal"] == "success"


def test_failed_rollback_is_diagnosed_and_the_worker_reports_container_lost(tmp_path: Path, monkeypatch) -> None:
    import contextlib

    from apps.v2.agent.paper2code.repair_loop import ControllerSeams, ControllerWorkers
    from apps.v2.agent_engine.rsa import setupx_interop

    @contextlib.contextmanager
    def no_setupx(*args, **kwargs):
        yield None

    monkeypatch.setattr(setupx_interop, "setupx_configured", no_setupx)
    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://127.0.0.1:9418/repo.git", first)

    class LosingBackend(FakeBackend):
        def rollback_to_checkpoint(self, n_frames=1):
            self.rollbacks += 1
            self.container_id = ""  # RSA's behaviour when the checkpoint container fails to start
            return False

        def _exec_host(self, cmd, timeout):
            return SimpleNamespace(output="CONTAINER ID  IMAGE\n(none)\n", success=True)

    backend = LosingBackend()
    events: list = []
    served: list[str] = []
    seams = ControllerSeams(
        repo=repo, code_dir=code, goal="run it", repair_rounds=3, environment_rounds=3, store=tmp_path / "store", out_dir=tmp_path / "adj",
        provider_factory=lambda: ScriptedProvider([*rewrite(code, "main.py", "print(1)\n"), ("finish", {"summary": "x"})]),
        model="fake", serve_repo=lambda git_dir: served.append(str(git_dir)) or "git://x/repo.git", events=lambda kind, **f: events.append((kind, f)),
    )
    cfg = SimpleNamespace(workdir="/workspace/repo", backend="small", base_image="x", max_steps=10, disclose_ids=True)
    workers = ControllerWorkers(seams, config=cfg, frozen=frozen, backend=backend)
    state = SimpleNamespace(container_id="rsa-1", round_no=2, last_verdict={"verdict": "FAIL", "rung": "G2", "log_tail": "NameError: x"}, environment_rounds_used=1)
    result = workers.repair(state)
    assert result.container_lost is True
    assert result.changed is True
    assert result.commit == repo.head() != first
    assert result.error == ""
    assert served == [str(repo.git_dir)]  # re-served for the container that will be rebuilt
    assert not any("git checkout" in c for c in backend.commands)  # nothing to move: no container
    assert workers.frozen.ladder.commit == repo.head()  # still re-pinned
    kinds = [k for k, _ in events]
    assert "repair.rollback_failed" in kinds
    assert events[[k for k, _ in events].index("repair.rollback_failed")][1]["container_lost"] is True


def test_output_directory_assets_are_dropped_before_g1_and_output_dirs_are_created_before_a_trial(tmp_path: Path, monkeypatch) -> None:
    # sapg-2 GPU run 5 (2026-09-18): the compiler declared the goal's output directory (/workspace/out/smoke) as an
    # input asset and G1 blocked the run; the trial box makes those directories and the ladder loses such assets
    import contextlib

    from apps.v2.agent.paper2code.repair_loop import ControllerSeams, run_line_pipeline
    from apps.v2.agent_engine.rsa import setupx_interop
    from apps.v2.agent_engine.rsa.criterion import Asset

    code = _code(tmp_path)
    repo = CodeRepo.for_run(tmp_path, code)
    first = repo.commit("round 0")
    frozen = _frozen(tmp_path, "git://127.0.0.1:9418/repo.git", first)
    frozen.ladder.assets = [Asset(kind="path", name="output_dir", path="/workspace/out/smoke", why="the entry writes here")]
    backend = FakeBackend()

    @contextlib.contextmanager
    def no_setupx(*args, **kwargs):
        yield None

    monkeypatch.setattr(setupx_interop, "setupx_configured", no_setupx)
    events: list = []
    seams = ControllerSeams(
        repo=repo, code_dir=code, goal="run it", repair_rounds=1, environment_rounds=1, store=tmp_path / "store", out_dir=tmp_path / "adj",
        provider_factory=lambda: ScriptedProvider([]), model="fake", serve_repo=lambda git_dir: "git://x/repo.git",
        environment_round=lambda be, last, n: (be.attach("c-1", "/workspace/repo") or 1), judge=lambda be, fl, n: [_verdict("G0", True), _verdict("G2", True)],
        events=lambda kind, **f: events.append((kind, f)),
    )
    cfg = SimpleNamespace(workdir="/workspace/repo", backend="small", base_image="x", max_steps=10, disclose_ids=True, execution_backend="remote", remote_backend=backend, remote_target="ssh")
    result = run_line_pipeline(frozen, cfg, seams=seams)
    assert result.terminal == "success"  # not blocked: the output-dir asset was dropped, G1 never ran
    assert frozen.ladder.assets == []
    assert ("controller.assets_dropped", {"assets": ["output_dir"]}) in events

    # the trial worker created the artifacts' parent directories before judging
    assert any(c.startswith("mkdir -p /workspace/out") for c in backend.commands), backend.commands


def test_artifact_paths_are_normalised_before_the_freeze() -> None:
    # T3 (owner 2026-09-18): relative `workspace/out/…` becomes absolute; a relative twin of an absolute artifact is dropped
    from apps.v2.agent_engine.rsa import pipeline as rsa_pipeline
    from apps.v2.agent_engine.rsa.criterion import Artifact, Criterion, Ladder, Rung

    c = Criterion(
        rung=Rung.G2, repo_url="git://x/repo.git", commit="a" * 40,
        command="python sapg/main.py --figures-dir /workspace/out/smoke/figures --output-dir /workspace/out/smoke",
        artifacts=[Artifact(path="figures/summary.json"), Artifact(path="workspace/out/smoke/figures/summary.json"), Artifact(path="workspace/out/smoke/curves.npz"), Artifact(path="runs/x.json"), Artifact(path="/workspace/out/smoke/figures", check="exists")],
    )
    ladder = Ladder(repo_url="git://x/repo.git", commit="a" * 40, goal="g", rungs={Rung.G2: c})
    changed = repair_loop.normalise_ladder_artifacts(ladder)
    assert changed == [
        "workspace/out/smoke/figures/summary.json -> /workspace/out/smoke/figures/summary.json",
        "workspace/out/smoke/curves.npz -> /workspace/out/smoke/curves.npz",
        "dropped figures/summary.json (duplicate of /workspace/out/smoke/figures/summary.json)",
    ]
    assert [a.path for a in c.artifacts] == ["/workspace/out/smoke/figures/summary.json", "/workspace/out/smoke/curves.npz", "runs/x.json", "/workspace/out/smoke/figures"]
    assert c.artifacts[3].check == "exists"  # the other fields survive
    assert repair_loop.normalise_ladder_artifacts(ladder) == []  # idempotent
    # a command without an absolute output directory keeps its relative artifacts (the code writes there)
    d = Criterion(rung=Rung.G2, repo_url="git://x/repo.git", commit="a" * 40, command="python main.py", artifacts=[Artifact(path="runs/x.json"), Artifact(path="workspace/out/y.json")])
    assert repair_loop.normalise_ladder_artifacts(Ladder(repo_url="git://x/repo.git", commit="a" * 40, goal="g", rungs={Rung.G2: d})) == ["workspace/out/y.json -> /workspace/out/y.json"]
    assert [a.path for a in d.artifacts] == ["runs/x.json", "/workspace/out/y.json"]
    # installed for the run as RSA's Freezer, restored afterwards
    seen: list = []
    original = rsa_pipeline.Freezer
    restore = repair_loop.install_artifact_normaliser(lambda kind, **f: seen.append((kind, f)))
    try:
        assert rsa_pipeline.Freezer is not original
        assert issubclass(rsa_pipeline.Freezer, original)
    finally:
        restore()
    assert rsa_pipeline.Freezer is original
