"""PLAN-3 item 4d/4e: environment_run as one call to main's experiment agent, offline.

main's ``run_experiment_on_machine`` runs for real; the machine (fake ECS + fake relay runtime),
RSA (a scripted agent) and the two machine-side commands are doubles. What is under test is the
line's part: the goal, the runner's interaction handling (decision file / unattended / waiting),
the records (``environment.json``, state, request), the held machine across a review point, and
the resume through ``RunLease.adopt``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.v2.agent.paper2code import experiment_step
from apps.v2.agent.paper2code.driver import Driver
from apps.v2.agent.paper2code.execution.aliyun_lease import RunLease
from apps.v2.agent.paper2code.phases import ExperimentSeams
from apps.v2.agent_engine.experiment import run_flow
from apps.v2.agent_engine.rsa.agent import AgentOutcome, AgentStatus, InteractionRequest, InteractionResponse
from tests.v2_paper2code.fake_aliyun import FakeEcs, FakeRuntime
from tests.v2_paper2code.test_driver_offline import FakePort, ScriptedProvider, _paper_dir, _repo


class ScriptedRsa:
    """``RSAAgent`` as the runner sees it: ``run(request, interaction=...)`` → ``AgentOutcome``.

    ``script`` is a list of steps: ``("ask", kind)`` asks the handler and records the answer,
    ``("outcome", status)`` returns. A ``stop`` answer to a question ends the run as FAILED, the way RSA does.
    """

    instances: list = []  # noqa: RUF012 - tests read what ran

    def __init__(self, config, script):
        self.config, self.script = config, list(script)
        self.requests: list = []
        self.answers: list = []
        ScriptedRsa.instances.append(self)

    def run(self, request, *, interaction=None):
        self.requests.append(request)
        for step, value in self.script:
            if step == "ask":
                event = InteractionRequest(kind=value, message=f"question of kind {value}", data={"instruction": request.instruction})
                response = interaction(event) if interaction is not None else None
                if response is None:
                    return AgentOutcome(AgentStatus.NEEDS_USER, request, pending=event)
                self.answers.append((value, response))
                if response.action == "stop":
                    return AgentOutcome(AgentStatus.FAILED, request, pending=event)
            elif step == "outcome":
                status = AgentStatus(value)
                pipeline = SimpleNamespace(container_id="rsa-cafe0123" if status is AgentStatus.SUCCESS else "", terminal=value, reached="G2" if status is AgentStatus.SUCCESS else "", tokens={"total": 1234}, outcome=None)
                pending = InteractionRequest(kind="criterion_review", message="falsified", data={}) if status is AgentStatus.RECOMPILE else None
                return AgentOutcome(status, request, pipeline=pipeline, pending=pending)
        raise AssertionError("script exhausted")


class HostExec:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def __call__(self, config, command, *, timeout=600.0) -> str:
        self.commands.append(command)
        if "docker commit" in command:
            return "987654321\n"
        return ""


class FakeLoopback:
    started = 0
    closed = 0

    def start(self):
        FakeLoopback.started += 1
        return self

    def close(self):
        FakeLoopback.closed += 1

    def target(self):
        return SimpleNamespace(provider="fake", model_id="DeepSeek-V4-Flash", base_url="http://127.0.0.1:1/v1", api_key="t", redacted=lambda: {"provider": "fake"})


def _seams(run_dir: Path, ecs: FakeEcs, script, host: HostExec) -> ExperimentSeams:
    return ExperimentSeams(
        run_lease=RunLease(run_dir, client_factory=lambda: ecs, runtime_factory=FakeRuntime, ready_timeout=1.0, bring_up_backoff=0.0),
        loopback_factory=lambda ctx: FakeLoopback(),
        agent_factory=lambda config: ScriptedRsa(config, script),
        host_exec=host,
        bootstrap=False,
    )


@pytest.fixture(autouse=True)
def served_repos(monkeypatch) -> list[str]:
    """Serve-the-code needs a real runtime; here the bundle is made (local git) and the URL is invented."""
    served: list[str] = []

    async def fake_serve(runtime, repo_dir, *, port=9418):
        from apps.v2.agent_engine.experiment.git_daemon import local_bundle

        local_bundle(repo_dir)  # proves the code history is bundle-able
        served.append(str(repo_dir))
        return "git://127.0.0.1:9418/repo.git"

    monkeypatch.setattr(run_flow, "serve_repo_on_machine", fake_serve)
    monkeypatch.setattr(run_flow, "_setupx_env", lambda target: run_flow._noop_ctx())
    monkeypatch.setattr(run_flow, "HEARTBEAT_SECONDS", 3600.0)
    ScriptedRsa.instances.clear()
    FakeRuntime.ready = True
    return served


def _init(tmp_path: Path, *, ask: bool = False) -> tuple[Path, Path]:
    paper = _paper_dir(tmp_path)
    repo = _repo(tmp_path)
    run_dir = tmp_path / "run"
    Driver.init(run_dir, paper_dir=str(paper), compute="aliyun", skip=("index",), run_id="t4", ask=ask)
    return run_dir, repo


async def _run(run_dir: Path, repo: Path, seams: ExperimentSeams, *, until: str = "environment_run", ask: bool | None = None) -> str:
    driver = Driver(run_dir)
    await driver.open(provider=ScriptedProvider(repo), port=FakePort(), git_check=False, experiment=seams, ask=ask)
    try:
        return await driver.run_until(until)
    finally:
        await driver.close()


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_round_zero_success_records_environment_and_releases(tmp_path: Path, served_repos: list[str]) -> None:
    run_dir, repo = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    outcome = asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, [("outcome", "success")], host)))
    assert outcome == "completed", json.loads((run_dir / "status.json").read_text())

    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "passed"
    assert result["agent_status"] == "success"
    assert result["reached"] == "G2"
    assert result["round"] == 0
    assert result["image"] == "paper2code-env:t4-r0"
    assert result["released"] is True
    assert result["denylist_touched"] == []
    assert result["entry"] == "main.py"

    env = json.loads((run_dir / "environment.json").read_text())
    assert env["container_id"] == "rsa-cafe0123"
    assert env["image_size_bytes"] == 987654321
    assert env["commit"] == result["commit"]
    assert len(env["commit"]) == 40
    assert env["goal"].startswith("Goal: prove that this freshly generated research repository runs")
    assert "Entry point: `main.py`" in env["goal"]
    assert "github.com/authors/ema-detect-official" in env["goal"]  # the denylist, as a thing not to touch
    assert env["answered"] == []
    assert host.commands
    assert host.commands[0].startswith("docker commit rsa-cafe0123 paper2code-env:t4-r0")

    lease = json.loads((run_dir / "lease.json").read_text())
    assert lease["state"] == "released"
    assert lease["instance_type"] == "ecs.c7.xlarge"
    assert ecs.instances == {}
    assert served_repos == [str(run_dir / "code.git")]
    assert (run_dir / "rsa" / "report.md").read_text()
    assert (run_dir / "rsa" / "store").is_dir()
    rsa = ScriptedRsa.instances[0]
    assert rsa.requests[0].repository == "git://127.0.0.1:9418/repo.git"
    assert rsa.requests[0].instruction == env["goal"]
    assert rsa.config.remote_target.startswith("ssh -p 22 root@10.0.0.")
    assert rsa.config.remote_password
    assert FakeLoopback.started >= 1
    assert FakeLoopback.closed >= 1
    state = json.loads((run_dir / "phases" / "10_environment_run.state.json").read_text())
    assert state["status"] == "success"
    assert state["pending"] is None
    assert not (run_dir / "phases" / "10_environment_run.request.json").exists()
    kinds = [e["kind"] for e in _events(run_dir)]
    assert "experiment.start" in kinds
    assert "experiment.image_ready" in kinds
    assert "experiment.done" in kinds
    # the product directory is untouched by the history
    assert not (run_dir / "workspace" / "tasks" / "paper_t4" / "generate_code" / ".git").exists()


def test_unattended_questions_get_the_defaults_and_a_failed_criterion_still_completes_the_phase(tmp_path: Path) -> None:
    run_dir, repo = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    script = [("ask", "clarification"), ("ask", "asset"), ("outcome", "success")]
    outcome = asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, script, host)))
    assert outcome == "completed"
    rsa = ScriptedRsa.instances[0]
    assert [(k, r.action) for k, r in rsa.answers] == [("clarification", "answer"), ("asset", "stop")]
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "failed"  # RSA ended FAILED after the stop; recorded, not raised
    assert result["agent_status"] == "failed"
    assert result["released"] is True
    assert [a["source"] for a in result["answered"]] == ["unattended", "policy"]  # the asset card is policy, not a default
    assert host.commands == []  # nothing to commit
    assert ecs.instances == {}


def test_asset_card_fails_the_run_even_under_ask_and_releases_the_machine(tmp_path: Path) -> None:
    # PLAN-3 §0 "资产缺 / fallback" (S2): a missing asset means the environment cannot be built — no question, no held machine
    run_dir, repo = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    script = [("ask", "asset"), ("outcome", "success")]
    assert asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, script, host), until="compute")) == "completed"
    outcome = asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, script, host), ask=True))
    assert outcome == "completed"  # the phase ends (failed), the driver does not wait
    assert not (run_dir / "phases" / "10_environment_run.request.json").exists()
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "failed"
    assert result["released"] is True
    assert result["repair"]["rounds_used"] == 0
    assert result["repair"]["stop_reason"].startswith("assets missing")
    assert ecs.instances == {}
    env = json.loads((run_dir / "environment.json").read_text())
    assert env["answered"][0]["source"] == "policy"
    assert env["answered"][0]["action"] == "stop"
    assert "experiment.assets_missing" in [e["kind"] for e in _events(run_dir)]


def test_ask_holds_the_machine_writes_the_request_and_resumes_from_the_decision(tmp_path: Path) -> None:
    run_dir, repo = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    script = [("ask", "clarification"), ("outcome", "success")]
    assert asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, script, host), until="compute")) == "completed"  # the earlier review points auto-approve
    outcome = asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, script, host), ask=True))
    assert outcome == "waiting"

    request = json.loads((run_dir / "phases" / "10_environment_run.request.json").read_text())
    assert request["interaction_type"] == "experiment_clarification"
    assert request["kind"] == "clarification"
    assert request["questions"][0]["type"] == "text"
    assert request["questions"][0]["custom"] is True
    assert request["questions"][0]["text"] == "question of kind clarification"
    lease = json.loads((run_dir / "lease.json").read_text())
    assert lease["state"] == "running"  # main's policy: needs_user holds the machine
    assert list(ecs.instances) == ["i-fake1"]
    env = json.loads((run_dir / "environment.json").read_text())
    assert env["status"] == "needs_user"
    assert env["released"] is False
    state = json.loads((run_dir / "phases" / "10_environment_run.state.json").read_text())
    assert state["pending"] == "clarification"

    # nobody answered yet: running again waits again, rents nothing new
    assert asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, script, host), ask=True)) == "waiting"
    assert list(ecs.instances) == ["i-fake1"]

    (run_dir / "phases" / "10_environment_run.decision.json").write_text(json.dumps({"kind": "clarification", "action": "answer", "message": "use k = 3"}))
    outcome = asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, script, host), ask=True))
    assert outcome == "completed"
    second = ScriptedRsa.instances[-1]
    assert second.answers == [("clarification", InteractionResponse(action="answer", message="use k = 3"))]
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "passed"
    assert result["answered"] == [{"kind": "clarification", "source": "decision", "action": "answer", "message": "use k = 3"}]
    assert ecs.calls.count(("create", "i-fake1")) == 1  # the held machine was adopted, not re-rented
    assert ecs.instances == {}
    assert json.loads((run_dir / "lease.json").read_text())["state"] == "released"
    assert (run_dir / "phases" / "10_environment_run.decision.consumed.r0.json").exists()
    assert not (run_dir / "phases" / "10_environment_run.decision.json").exists()


def test_unattended_recompile_does_not_hold_a_machine(tmp_path: Path) -> None:
    run_dir, repo = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    outcome = asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, [("outcome", "recompile")], host)))
    assert outcome == "completed"
    assert len(ScriptedRsa.instances) == 2  # the one automatic recompile, then the unattended stop
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "needs_user"
    assert result["agent_status"] == "recompile"
    assert result["released"] is True  # the flow held it (main's policy); the phase let it go, nobody is watching
    assert ecs.instances == {}
    assert json.loads((run_dir / "lease.json").read_text())["state"] == "released"


def test_rejected_criterion_is_recompiled_once_with_the_reasons_before_anyone_is_asked(tmp_path: Path) -> None:
    run_dir, repo = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    scripts = [[("ask", "criterion_review"), ("outcome", "recompile")], [("outcome", "success")]]
    seams = ExperimentSeams(
        run_lease=RunLease(run_dir, client_factory=lambda: ecs, runtime_factory=FakeRuntime, ready_timeout=1.0, bring_up_backoff=0.0),
        loopback_factory=lambda ctx: FakeLoopback(),
        agent_factory=lambda config: ScriptedRsa(config, scripts.pop(0)),
        host_exec=host,
        bootstrap=False,
    )
    assert asyncio.run(_run(run_dir, repo, seams, until="compute")) == "completed"
    assert asyncio.run(_run(run_dir, repo, seams, ask=True)) == "completed"  # even under --ask: the automatic recompile comes first
    first, second = ScriptedRsa.instances
    assert first.answers[0][1].action == "retry"
    assert second.requests[0].instruction.startswith(first.requests[0].instruction)
    assert "Correction: the previous criterion was rejected" in second.requests[0].instruction
    assert "\n" not in second.requests[0].instruction
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "passed"
    assert result["answered"] == [{"kind": "criterion_review", "source": "auto-recompile", "action": "retry", "message": "recompile with the falsification reasons"}]
    assert not (run_dir / "phases" / "10_environment_run.request.json").exists()
    assert any(e["kind"] == "experiment.recompile" for e in _events(run_dir))
    assert ecs.instances == {}


def test_cancelled_compute_skips_the_machine(tmp_path: Path) -> None:
    run_dir, repo = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    seams = _seams(run_dir, ecs, [("outcome", "success")], host)
    assert asyncio.run(_run(run_dir, repo, seams, until="compute")) == "completed"
    compute_file = run_dir / "phases" / "09_compute.json"
    record = json.loads(compute_file.read_text())
    record["result"] = {"status": "cancelled", "decision": {"action": "cancel"}}
    compute_file.write_text(json.dumps(record))
    assert asyncio.run(_run(run_dir, repo, seams)) == "completed"
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "skipped"
    assert ecs.calls == [] or all(c[0] != "create" for c in ecs.calls)


# --- the pure parts ---------------------------------------------------------------------------------


def test_goal_is_the_basic_template_without_the_blueprint_validation() -> None:
    # PLAN-3 §0 "目标句" (S2): entry, GPU or not, outputs, denylist, unavailable tools, minimal scale — nothing else
    spec = {"external_tools": [{"name": "IsaacGym", "installable": True}, {"name": "PhysX", "installable": None}, {"name": "MuJoCo", "installable": False}]}
    goal = experiment_step.build_goal(entry="main.py", environment_spec=spec, denylist=("https://github.com/x/y",))
    assert "Entry point: `main.py`" in goal
    assert "There is no GPU" in goal
    assert "MuJoCo, PhysX" in goal
    assert "IsaacGym" not in goal.split("These external tools")[1].split(".")[0]
    assert "https://github.com/x/y" in goal
    assert "/workspace/out" in goal
    assert "Scale: minimal" in goal
    assert "validat" not in goal.lower()
    assert "fallback" not in goal.lower()
    assert "\n" not in goal  # RSA writes it as one `# goal:` comment line in the frozen criterion
    assert "command-line options" not in goal  # no flags known → nothing claimed


def test_goal_lists_the_entry_s_declared_options_read_statically(tmp_path: Path) -> None:
    # T2 (2026-09-18): RSA's compiler invented `--pbt_interval` / `--num-blocks` for sapg and the first repair round
    # of both S9 runs went to `unrecognized arguments`; the goal now carries the parser's real surface
    (tmp_path / "sapg").mkdir()
    (tmp_path / "sapg" / "main.py").write_text(
        "import argparse\n"
        "def _common(p):\n    p.add_argument('--seed', type=int)\n    p.add_argument('-d', '--device', default='cpu')\n"
        "def build():\n    parser = argparse.ArgumentParser(prog='sapg')\n    sub = parser.add_subparsers(dest='cmd')\n"
        "    t = sub.add_parser('train')\n    _common(t)\n    t.add_argument('--num-iterations', type=int)\n"
        "    e = sub.add_parser('eval', help='x')\n    e.add_argument('--checkpoint')\n    parser.add_argument('--seed')\n    return parser\n"
        "if __name__ == '__main__':\n    build().parse_args()\n"
    )
    flags = experiment_step.entry_flags(tmp_path, "sapg/main.py")
    assert flags == {"options": ["--seed", "-d", "--device", "--num-iterations", "--checkpoint"], "subcommands": ["train", "eval"]}
    goal = experiment_step.build_goal(entry="sapg/main.py", environment_spec={}, flags=flags)
    assert "sub-commands: train, eval; and these command-line options and no others: --seed, -d, --device, --num-iterations, --checkpoint." in goal
    assert "never invent an option" in goal
    assert "\n" not in goal
    # nothing to read → nothing claimed; a bad file or a path outside the tree reads as nothing
    assert experiment_step.entry_flags(tmp_path, None) == {"options": [], "subcommands": []}
    (tmp_path / "broken.py").write_text("def (:\n")
    assert experiment_step.entry_flags(tmp_path, "broken.py") == {"options": [], "subcommands": []}
    assert experiment_step.entry_flags(tmp_path, "../outside.py") == {"options": [], "subcommands": []}


def test_goal_names_the_declared_requirements_and_the_recompile_hint_addresses_a_bare_pass(tmp_path: Path) -> None:
    # T5 pinn (2026-09-18): torch pre-installed (T4) + G0 declaring only `opt_for_pinns, torch` → the bare image passed
    # G0 and RSA's falsifier rejected the criterion twice ("measuring nothing")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "requirements.txt").write_text("# deps\ntorch==2.0.0\nnumpy>=1.23,<2.0  # arrays\nPyYAML>=6.0\n-r extra.txt\nscipy\n")
    (tmp_path / "pkg" / "requirements-dev.txt").write_text("pytest\nnumpy\n")
    assert experiment_step.declared_requirements(tmp_path) == ["torch", "numpy", "pyyaml", "scipy", "pytest"]
    goal = experiment_step.build_goal(entry="main.py", environment_spec={}, requirements=["torch", "numpy", "scipy"])
    assert "Third-party packages the code declares (requirements.txt): torch, numpy, scipy." in goal
    assert "\n" not in goal
    assert "requirements.txt" not in experiment_step.build_goal(entry="main.py", environment_spec={})

    from types import SimpleNamespace

    point = SimpleNamespace(problems=["the criterion passes in the bare environment, so it is not measuring what a configured environment provides"])
    report = SimpleNamespace(must_recompile=True, points=[point], reasons=[])
    outcome = SimpleNamespace(compile=SimpleNamespace(falsification={"G0": report}))
    hint = experiment_step.falsification_hint(outcome)
    assert hint.startswith("Correction: the previous criterion was rejected by static falsification — G0: the criterion passes in the bare environment")
    assert "name the third-party packages the code declares in requirements.txt, not only the repository's own modules" in hint


def test_goal_shows_literal_choices_including_those_imported_from_another_module(tmp_path: Path) -> None:
    # T2 rerun 11:40: with the option list in the goal the compiler invented a *value* (`--task dummy` against
    # `choices=sorted(TASK_CONFIGS.keys())`, TASK_CONFIGS a dict literal in sapg/sapg/config.py)
    (tmp_path / "sapg" / "sapg").mkdir(parents=True)
    (tmp_path / "sapg" / "sapg" / "config.py").write_text("class A: pass\nTASK_CONFIGS = {\n    'regrasping': A,\n    'throw': A,\n}\n")
    (tmp_path / "sapg" / "main.py").write_text(
        "import argparse\nfrom sapg.config import TASK_CONFIGS\nALGORITHMS = ('sapg', 'ppo')\n"
        "p = argparse.ArgumentParser()\np.add_argument('--task', choices=sorted(TASK_CONFIGS.keys()))\n"
        "p.add_argument('--algorithm', choices=ALGORITHMS)\np.add_argument('--agg', choices=['a', 'b'])\n"
        "p.add_argument('--n', type=int, choices=range(3))\np.add_argument('--unknown', choices=make())\n"
    )
    flags = experiment_step.entry_flags(tmp_path, "sapg/main.py")
    assert flags["options"] == ["--task {regrasping,throw}", "--algorithm {sapg,ppo}", "--agg {a,b}", "--n", "--unknown"]
    goal = experiment_step.build_goal(entry="sapg/main.py", environment_spec={}, flags=flags)
    assert "--task {regrasping,throw}" in goal
    assert "pass one of those values; never invent an option or a value" in goal
    gpu = experiment_step.build_goal(entry=None, environment_spec=None, gpu_available=True)
    assert "A GPU is available" in gpu
    assert "no GPU" not in gpu
    assert "Entry point: the repository's main script" in gpu


@pytest.mark.parametrize(
    ("decision", "kind", "expected"),
    [
        (None, "clarification", {"action": "answer", "message": experiment_step.UNATTENDED["clarification"]["message"]}),
        (None, "asset", dict(experiment_step.ASSET_POLICY)),
        ({"action": "drop_assets", "assets": ["weights.pt"]}, "asset", dict(experiment_step.ASSET_POLICY)),  # a decision cannot drop assets
        ({"kind": "approval", "answers": ["approve"]}, "approval", {"action": "approve", "message": ""}),
        ({"kind": "asset", "action": "retry"}, "clarification", {"action": "answer", "message": experiment_step.UNATTENDED["clarification"]["message"]}),
    ],
)
def test_response_from_decision(decision, kind, expected) -> None:
    assert experiment_step.response_from_decision(decision, kind) == expected


def test_denylist_hits_normalise_urls() -> None:
    denylist = ("https://github.com/jayeshs999/sapg", "http://example.org/data/")
    ledger = 'git clone https://github.com/JAYESHS999/sapg.git /workspace/orig\ncurl -O https://example.org/data/set.tar'
    assert experiment_step.denylist_hits(denylist, ledger) == list(denylist)
    assert experiment_step.denylist_hits(denylist, "pip install torch", "git clone git://127.0.0.1:9418/repo.git") == []


def test_missing_modules_are_named_before_renting(monkeypatch) -> None:
    assert experiment_step.missing_modules() == []
    missing = experiment_step.missing_modules(("docker", "no_such_module_p2c"))
    assert len(missing) == 1
    assert missing[0].startswith("no_such_module_p2c (ModuleNotFoundError")


def test_setupx_addendum_is_appended_once_and_survives_format(tmp_path: Path) -> None:
    root = tmp_path / "setupx"
    (root / "src").mkdir(parents=True)
    engine = root / "src" / "llm_engine.py"
    engine.write_text('class LLMEngine:\n    SYSTEM_PROMPT_TEMPLATE = "You are {name}. Rules:"\n', encoding="utf-8")
    assert experiment_step.append_setupx_addendum(root, ("https://github.com/x/y",)) is True
    assert experiment_step.append_setupx_addendum(root, ("https://github.com/x/y",)) is True
    body = engine.read_text()
    assert body.count(experiment_step.SETUPX_ADDENDUM_MARK) == 1
    namespace: dict = {}
    exec(compile(body, str(engine), "exec"), namespace)
    template = namespace["LLMEngine"].SYSTEM_PROMPT_TEMPLATE
    rendered = template.format(name="agent")
    assert "MACHINE FACTS" in rendered
    assert "download.pytorch.org/whl/cpu" in rendered
    assert "https://github.com/x/y" in rendered
    assert "/workspace/out" in rendered
    # the facts change (the GPU stage after a CPU stage, same process): the file and the live class both follow
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("setupx_llm_engine_test", engine)
    module = importlib.util.module_from_spec(spec)
    sys.modules["setupx_llm_engine_test"] = module
    try:
        spec.loader.exec_module(module)
        assert "Accelerator: none" in module.LLMEngine.SYSTEM_PROMPT_TEMPLATE
        assert experiment_step.append_setupx_addendum(root, ("https://github.com/x/y",), gpu_available=True) is True
        assert "NVIDIA GPU" in module.LLMEngine.SYSTEM_PROMPT_TEMPLATE
        assert "Accelerator: none" not in module.LLMEngine.SYSTEM_PROMPT_TEMPLATE
        assert module.LLMEngine.SYSTEM_PROMPT_TEMPLATE.count("MACHINE FACTS") == 1
        body2 = engine.read_text()
        assert body2.count(experiment_step.SETUPX_ADDENDUM_MARK) == 1
        assert "NVIDIA GPU" in body2
        assert "Accelerator: none" not in body2
        assert experiment_step.append_setupx_addendum(root, ("https://github.com/x/y",), gpu_available=True) is True  # unchanged: no rewrite
        assert engine.read_text() == body2
    finally:
        sys.modules.pop("setupx_llm_engine_test", None)
    assert "must run" not in rendered  # facts, not instructions (SetupX is a black box, PLAN-3 §0)
    assert experiment_step.append_setupx_addendum(None) is False
    assert experiment_step.append_setupx_addendum(tmp_path / "nowhere") is False
    gpu = experiment_step.setupx_addendum(gpu_available=True)
    assert "NVIDIA GPU" in gpu
    assert "whl/cpu" not in gpu
    assert "Pre-installed in the container" not in gpu  # T4 parked (opt-in PAPER2CODE_TORCH_PREINSTALL=1)
    import os

    os.environ["PAPER2CODE_TORCH_PREINSTALL"] = "1"
    try:
        assert "Pre-installed in the container: torch 2.1.2 (CUDA cu121 build) and torchvision 0.16.2" in experiment_step.setupx_addendum(gpu_available=True)
    finally:
        os.environ.pop("PAPER2CODE_TORCH_PREINSTALL", None)


def test_controller_runs_inside_rsa_and_pre_repair_snapshot_submits_round_zero(tmp_path: Path, monkeypatch) -> None:
    """S8: the runner installs the controller as RSA's run_pipeline; the scripted RSA composes like the real one
    (compile → run_pipeline → status). 搭建环境 → trial fails on code → 修复代码 → trial passes, on the held machine."""
    import contextlib

    from apps.v2.agent.paper2code import environment_controller as ec
    from apps.v2.agent.paper2code.code_repo import CodeRepo
    from apps.v2.agent_engine.rsa import setupx_interop
    from apps.v2.agent_engine.rsa import agent as rsa_agent
    from tests.v2_paper2code.test_repair import ScriptedProvider as RepairProvider
    from tests.v2_paper2code.test_repair import rewrite
    from tests.v2_paper2code.test_repair_loop import FakeBackend, _frozen, _verdict

    @contextlib.contextmanager
    def no_setupx(*args, **kwargs):
        yield None

    monkeypatch.setattr(setupx_interop, "setupx_configured", no_setupx)
    run_dir, repo_src = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    code_dir = run_dir / "workspace" / "tasks" / "paper_t4" / "generate_code"
    judged: list[tuple[int, str]] = []
    env_rounds: list[int] = []
    fake_backend = FakeBackend()

    class ComposingRsa(ScriptedRsa):
        """Compiles a real frozen ladder, then calls whatever ``rsa.agent.run_pipeline`` is — the controller."""

        def run(self, request, *, interaction=None):
            self.requests.append(request)
            self.config.execution_backend = "remote"
            self.config.remote_backend = fake_backend
            self.config.store = str(run_dir / "rsa" / "store")
            frozen = _frozen(run_dir / "rsa", request.repository, CodeRepo.for_run(run_dir, code_dir).head())
            result = rsa_agent.run_pipeline(frozen, self.config, out_dir=run_dir / "rsa" / "pipe")
            status = AgentStatus.SUCCESS if result.terminal == "success" else AgentStatus.FAILED
            return AgentOutcome(status, request, pipeline=result, frozen=frozen)

    def environment_round(backend, last, round_no):
        env_rounds.append(round_no)
        backend.attach("rsa-held", "/workspace/repo")
        return ec.SetupResult(container_id="rsa-held", actions=3)

    replies = [[_verdict("G0", True), _verdict("G2", False)], [_verdict("G0", True), _verdict("G2", True)]]
    replies[0][1].log_tail = "NameError: name 'k' is not defined"
    replies[0][1].failure_tails = {"criteria_G2.py::test_run_completes": replies[0][1].log_tail}

    def judge(backend, frozen, round_no):
        judged.append((round_no, frozen.ladder.commit))
        return replies.pop(0)

    seams = ExperimentSeams(
        run_lease=RunLease(run_dir, client_factory=lambda: ecs, runtime_factory=FakeRuntime, ready_timeout=1.0, bring_up_backoff=0.0),
        loopback_factory=lambda ctx: FakeLoopback(),
        agent_factory=lambda config: ComposingRsa(config, []),
        host_exec=host,
        bootstrap=False,
        repair_overrides={
            "provider_factory": lambda: RepairProvider([*rewrite(code_dir, "main.py", "print('repaired')\n"), ("finish", {"summary": "defined k"})]),
            "serve_repo_factory": lambda config: (lambda git_dir: "git://127.0.0.1:9418/repo.git"),
            "judge": judge,
            "environment_round": environment_round,
        },
    )
    assert asyncio.run(_run(run_dir, repo_src, seams)) == "completed"
    assert rsa_agent.run_pipeline.__name__ == "run_pipeline"  # restored after the run
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "passed"
    assert result["agent_status"] == "success"
    assert result["repair"] == {"rounds_used": 2, "passed": True, "stop_reason": "criterion passed", "accepted_failing": False, "gpu_required": False, "configured": 3}
    env = json.loads((run_dir / "environment.json").read_text())
    assert [r["kind"] for r in env["repair"]["rounds"]] == ["environment", "repair"]
    assert env["repair"]["rounds"][0]["environment_actions"] == 3
    assert env["repair"]["rounds"][1]["reason"] == "no environment signature in the failure"
    assert env["repair"]["rounds"][1]["agent"]["written"] == ["main.py"]
    assert env["image"] == "paper2code-env:t4-r0"  # the repaired, passing container was committed
    assert env["container_id"] == "rsa-held"
    repo = CodeRepo.for_run(run_dir, code_dir)
    assert env_rounds == [1]
    assert judged == [(1, repo.first_commit()), (2, repo.head())]  # the trial after the repair judges the new pin
    assert repo.first_commit() != repo.head()
    assert (code_dir / "main.py").read_text() == "print('repaired')\n"
    assert json.loads((run_dir / "lease.json").read_text())["state"] == "released"
    kinds = [e["kind"] for e in _events(run_dir)]
    assert "controller.installed" in kinds
    assert "controller.round" in kinds
    assert "controller.judged" in kinds
    assert "repair.done" in kinds
    assert json.loads((run_dir / "rsa" / "pipe" / "result.json").read_text())["controller"]["passed"] is True

    # the two ends of the paired comparison come from this one run
    pool = tmp_path / "pool"
    before = Driver(run_dir).submit(paper="ema-detect", trial="pre", dest_root=pool, snapshot="pre_repair")
    after = Driver(run_dir).submit(paper="ema-detect", trial="post", dest_root=pool)
    assert before["snapshot"] == "pre_repair"
    assert before["snapshot_commit"] == repo.first_commit()
    assert (pool / "ema-detect" / "pre" / "main.py").read_text() != "print('repaired')\n"
    assert (pool / "ema-detect" / "post" / "main.py").read_text() == "print('repaired')\n"
    assert after["snapshot"] is None
    assert set(before["sha256"]) == set(after["sha256"])  # same file set, different content for main.py
    assert before["sha256"]["main.py"] != after["sha256"]["main.py"]


def test_controller_stop_is_the_repair_review_point_default_accept(tmp_path: Path, monkeypatch) -> None:
    """S8: a stop without a pass (here: the same failure twice) → repair_review, unattended default accept."""
    import contextlib

    from apps.v2.agent.paper2code import environment_controller as ec
    from apps.v2.agent.paper2code.code_repo import CodeRepo
    from apps.v2.agent_engine.rsa import setupx_interop
    from apps.v2.agent_engine.rsa import agent as rsa_agent
    from tests.v2_paper2code.test_repair import ScriptedProvider as RepairProvider
    from tests.v2_paper2code.test_repair import rewrite
    from tests.v2_paper2code.test_repair_loop import FakeBackend, _frozen, _verdict

    @contextlib.contextmanager
    def no_setupx(*args, **kwargs):
        yield None

    monkeypatch.setattr(setupx_interop, "setupx_configured", no_setupx)
    run_dir, repo_src = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    code_dir = run_dir / "workspace" / "tasks" / "paper_t4" / "generate_code"
    fake_backend = FakeBackend()

    class ComposingRsa(ScriptedRsa):
        def run(self, request, *, interaction=None):
            self.requests.append(request)
            self.config.execution_backend = "remote"
            self.config.remote_backend = fake_backend
            self.config.store = str(run_dir / "rsa" / "store")
            frozen = _frozen(run_dir / "rsa", request.repository, CodeRepo.for_run(run_dir, code_dir).head())
            result = rsa_agent.run_pipeline(frozen, self.config)
            return AgentOutcome(AgentStatus.SUCCESS if result.terminal == "success" else AgentStatus.FAILED, request, pipeline=result, frozen=frozen)

    def environment_round(backend, last, round_no):
        backend.attach("rsa-held", "/workspace/repo")
        return ec.SetupResult(container_id="rsa-held", actions=1)

    def judge(backend, frozen, round_no):
        v = _verdict("G2", False)
        v.log_tail = "AssertionError: still broken"
        v.failure_tails = {"t": v.log_tail}
        return [_verdict("G0", True), v]

    providers = [RepairProvider([*rewrite(code_dir, "main.py", f"print({n})\n"), ("finish", {"summary": f"try {n}"})]) for n in range(3)]
    seams = ExperimentSeams(
        run_lease=RunLease(run_dir, client_factory=lambda: ecs, runtime_factory=FakeRuntime, ready_timeout=1.0, bring_up_backoff=0.0),
        loopback_factory=lambda ctx: FakeLoopback(), agent_factory=lambda config: ComposingRsa(config, []), host_exec=host, bootstrap=False,
        repair_overrides={"provider_factory": lambda: providers.pop(0), "serve_repo_factory": lambda config: (lambda git_dir: "git://x/repo.git"), "judge": judge, "environment_round": environment_round},
    )
    assert asyncio.run(_run(run_dir, repo_src, seams)) == "completed"
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "accepted"  # failed, then accepted at the review point (unattended default)
    assert result["repair"]["passed"] is False
    assert result["repair"]["accepted_failing"] is True
    assert "reproduced unchanged" in result["repair"]["stop_reason"]
    env = json.loads((run_dir / "environment.json").read_text())
    assert [r["kind"] for r in env["repair"]["rounds"]] == ["environment", "repair"]
    assert env["answered"][-1]["kind"] == "repair_review"
    assert env["answered"][-1]["action"] == "accept"
    assert env["released"] is True
    assert ecs.instances == {}


def test_gpu_code_starts_on_the_cpu_tier_with_the_escalation_machine_recorded(tmp_path: Path) -> None:
    """PLAN-3 S6 + two-stage compute (owner 2026-09-18): needs_gpu + a GPU image → the GPU tiers are offered and
    priced, but step 10 starts on the CPU tier (CPU image, CPU wording) with the escalation machine recorded."""
    run_dir, repo = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    seams = _seams(run_dir, ecs, [("outcome", "success")], host)
    assert asyncio.run(_run(run_dir, repo, seams, until="implement")) == "completed"
    code_dir = run_dir / "workspace" / "tasks" / "paper_t4" / "generate_code"
    (code_dir / "src" / "ema.py").write_text("import torch\nmodel = torch.nn.Linear(4, 4)\ndevice = 'cuda' if torch.cuda.is_available() else 'cpu'\nmodel.to(device)\n")
    assert asyncio.run(_run(run_dir, repo, _seams(run_dir, ecs, [("outcome", "success")], host))) == "completed"
    compute = json.loads((run_dir / "phases" / "09_compute.json").read_text())["result"]
    assert compute["estimate"]["needs_gpu"] is True
    assert compute["gpu_available"] is True
    assert compute["gpu"] is False  # CPU first
    assert compute["decision"]["tier"] == "economy"
    assert compute["decision"]["instance_type"] == "ecs.c7.xlarge"
    assert compute["decision"]["escalation_type"] == "ecs.gn6i-c4g1.xlarge"
    assert ("probe", "ecs.gn6i-c8g1.2xlarge") in ecs.calls  # the GPU tiers are still priced for the review point
    lease = json.loads((run_dir / "lease.json").read_text())
    assert lease["instance_type"] == "ecs.c7.xlarge"
    assert lease["image_id"] == "m-fixture"
    env = json.loads((run_dir / "environment.json").read_text())
    assert "There is no GPU" in env["goal"]
    assert env["gpu_available"] is False
    assert ecs.instances == {}

    # picking a GPU tier at the review point instead is resolve_decision's job (test_compute: gpu-economy → gpu True, no escalation)

    # the same code without a GPU image: CPU fallback, CPU image, CPU wording
    (tmp_path / "nogpu").mkdir()
    run_dir2, repo2 = _init(tmp_path / "nogpu")
    ecs2 = FakeEcs(gpu_image_id="")
    assert asyncio.run(_run(run_dir2, repo2, _seams(run_dir2, ecs2, [("outcome", "success")], HostExec()), until="implement")) == "completed"
    (run_dir2 / "workspace" / "tasks" / "paper_t4" / "generate_code" / "src" / "ema.py").write_text("import torch\nmodel = torch.nn.Linear(4, 4)\nmodel.to('cuda')\n")
    assert asyncio.run(_run(run_dir2, repo2, _seams(run_dir2, ecs2, [("outcome", "success")], HostExec()))) == "completed"
    compute2 = json.loads((run_dir2 / "phases" / "09_compute.json").read_text())["result"]
    assert compute2["estimate"]["needs_gpu"] is True
    assert compute2["gpu_available"] is False
    assert compute2["gpu"] is False
    assert compute2["decision"]["instance_type"] == "ecs.c7.xlarge"
    assert json.loads((run_dir2 / "lease.json").read_text())["image_id"] == "m-fixture"
    assert "There is no GPU" in json.loads((run_dir2 / "environment.json").read_text())["goal"]


def test_a_cuda_failure_on_the_cpu_stage_escalates_to_the_gpu_machine(tmp_path: Path, monkeypatch) -> None:
    """Two-stage compute end to end: the CPU stage's trial fails on CUDA → the controller stops with GPU_REQUIRED
    (no review point) → the phase releases the CPU machine, sets the CPU stage aside, rents the escalation
    machine on the GPU image, runs the boxes again with GPU facts and passes."""
    import contextlib

    from apps.v2.agent.paper2code import environment_controller as ec
    from apps.v2.agent.paper2code.code_repo import CodeRepo
    from apps.v2.agent_engine.rsa import setupx_interop
    from apps.v2.agent_engine.rsa import agent as rsa_agent
    from tests.v2_paper2code.test_repair_loop import FakeBackend, _frozen, _verdict

    @contextlib.contextmanager
    def no_setupx(*args, **kwargs):
        yield None

    monkeypatch.setattr(setupx_interop, "setupx_configured", no_setupx)
    run_dir, repo_src = _init(tmp_path)
    ecs, host = FakeEcs(), HostExec()
    code_dir = run_dir / "workspace" / "tasks" / "paper_t4" / "generate_code"
    fake_backend = FakeBackend()
    goals: list[str] = []

    class ComposingRsa(ScriptedRsa):
        def run(self, request, *, interaction=None):
            self.requests.append(request)
            goals.append(request.instruction)
            self.config.execution_backend = "remote"
            self.config.remote_backend = fake_backend
            self.config.store = str(run_dir / "rsa" / "store")
            frozen = _frozen(run_dir / "rsa", request.repository, CodeRepo.for_run(run_dir, code_dir).head())
            result = rsa_agent.run_pipeline(frozen, self.config)
            return AgentOutcome(AgentStatus.SUCCESS if result.terminal == "success" else AgentStatus.FAILED, request, pipeline=result, frozen=frozen)

    env_rounds: list[int] = []

    def environment_round(backend, last, round_no):
        env_rounds.append(round_no)
        backend.attach("rsa-held", "/workspace/repo")
        return ec.SetupResult(container_id="rsa-held", actions=1)

    stage: list[str] = []

    def judge(backend, frozen, round_no):
        if not stage:  # the CPU stage: CUDA is missing
            stage.append("cpu")
            v = _verdict("G2", False)
            v.log_tail = "RuntimeError: Torch not compiled with CUDA enabled"
            v.failure_tails = {"criteria_G2.py::test_run_completes": v.log_tail}
            return [_verdict("G0", True), v]
        return [_verdict("G0", True), _verdict("G2", True)]

    seams = ExperimentSeams(
        run_lease=RunLease(run_dir, client_factory=lambda: ecs, runtime_factory=FakeRuntime, ready_timeout=1.0, bring_up_backoff=0.0),
        loopback_factory=lambda ctx: FakeLoopback(), agent_factory=lambda config: ComposingRsa(config, []), host_exec=host, bootstrap=False,
        repair_overrides={"provider_factory": lambda: None, "serve_repo_factory": lambda config: (lambda git_dir: "git://x/repo.git"), "judge": judge, "environment_round": environment_round},
    )
    assert asyncio.run(_run(run_dir, repo_src, seams)) == "completed"
    result = json.loads((run_dir / "phases" / "10_environment_run.json").read_text())["result"]
    assert result["status"] == "passed"
    assert result["instance_type"] == "ecs.gn6i-c4g1.xlarge"
    assert result["gpu_available"] is True
    assert result["escalation"]["from"] == "ecs.c7.xlarge"
    assert result["escalation"]["to"] == "ecs.gn6i-c4g1.xlarge"
    assert result["escalation"]["reason"].startswith("gpu required: the code wants CUDA")
    assert result["escalation"]["cpu_stage"]["repair"]["gpu_required"] is True
    assert result["escalation"]["cpu_stage"]["repair"]["accepted_failing"] is False  # no review point on the way
    assert env_rounds == [1, 1]  # the environment box ran once per stage
    assert len([c for c in ecs.calls if c[0] == "create"]) == 2  # two machines rented, in turn
    assert ecs.instances == {}  # both released
    assert "There is no GPU" in goals[0]
    assert "A GPU is available" in goals[1]
    escalation = json.loads((run_dir / "phases" / experiment_step.ESCALATION_FILE).read_text())
    assert escalation["to"] == "ecs.gn6i-c4g1.xlarge"
    assert all(".cpu_stage." in str(moved) for moved in escalation["cpu_stage"]["moved"])
    assert any(str(moved).split("/")[-1].startswith("rsa.cpu_stage.") for moved in escalation["cpu_stage"]["moved"])
    assert (run_dir / "environment.json").is_file()
    assert json.loads((run_dir / "environment.json").read_text())["gpu_available"] is True
    events = [e["kind"] for e in _events(run_dir)]
    assert "repair.gpu_required" in events
    assert "experiment.escalate" in events
    assert events.index("experiment.escalate") < len(events) - 1
    lease = json.loads((run_dir / "lease.json").read_text())
    assert lease["instance_type"] == "ecs.gn6i-c4g1.xlarge"
    assert lease["image_id"] == "m-fixture-gpu"


def test_denylist_audit_scans_commands_not_thoughts(tmp_path: Path) -> None:
    # sapg-2 GPU run 3 (2026-09-18): SetupX wrote "the off-limits URL is https://github.com/jayeshs999/sapg, so I must not fetch it"
    # in a thought, and the audit marked the run denylist_touched. Only commands count.
    logs = tmp_path / "setupx-logs" / "round0"
    logs.mkdir(parents=True)
    (logs / "a.log").write_text(
        '2026-09-18 02:21:22 | INFO | setup_agent.llm | {"thought": "the off-limits URL is https://github.com/jayeshs999/sapg, so I must not fetch it", '
        '"action_type": "SHELL_COMMAND", "content": {"command": "cd /workspace/repo && cat README.md"}}\n'
        '2026-09-18 02:21:30 | INFO | setup_agent.llm | {"thought": "install", "action_type": "SHELL_COMMAND", "content": {"command": "pip install torch"}}\n'
    )
    rounds = [{"round_no": 1, "actions": ["SHELL_COMMAND: cd /workspace/repo && ls → exit=0"]}]
    commands = experiment_step.ledger_commands(rounds, logs)
    assert commands == [
        "SHELL_COMMAND: cd /workspace/repo && ls → exit=0",
        'SHELL_COMMAND: {"command": "cd /workspace/repo && cat README.md"}',
        'SHELL_COMMAND: {"command": "pip install torch"}',
    ]
    denylist = ("https://github.com/jayeshs999/sapg",)
    assert experiment_step.denylist_hits(denylist, experiment_step.ledger_text(rounds, logs)) == []
    (logs / "b.log").write_text('{"thought": "get it", "action_type": "SHELL_COMMAND", "content": {"command": "git clone https://github.com/jayeshs999/sapg /tmp/orig"}}\n')
    assert experiment_step.denylist_hits(denylist, experiment_step.ledger_text(rounds, logs)) == list(denylist)


def test_with_retries_retries_transport_failures_with_a_fresh_attempt_each_time() -> None:
    import asyncio

    from apps.v2.agent.paper2code.experiment_step import with_retries

    calls: list[int] = []
    notes: list[tuple[int, str]] = []

    async def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("connection lost while uploading repo.bundle")
        return "git://127.0.0.1:9418/repo.git"

    url = asyncio.run(with_retries(flaky, attempts=3, delays=(0.0, 0.0), on_retry=lambda n, exc: notes.append((n, str(exc)[:15]))))
    assert url.startswith("git://")
    assert len(calls) == 3
    assert notes == [(1, "connection lost"), (2, "connection lost")]

    async def hopeless() -> str:
        raise OSError("still lost")

    with pytest.raises(OSError, match="still lost"):
        asyncio.run(with_retries(hopeless, attempts=2, delays=(0.0,)))
