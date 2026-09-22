"""PLAN-3 S8: the mechanical scheduler — pure transitions, budgets, and the full offline chain
environment problem → 搭建环境 → trial fails on code → 修复代码 → trial passes."""

from __future__ import annotations

from apps.v2.agent.paper2code import environment_controller as ec
from apps.v2.agent.paper2code import repair


def _fail(rung: str, text: str) -> dict:
    return {"verdict": "FAIL", "rung": rung, "exit_code": 1, "statuses": {"t": "failed"}, "failure_tails": {"t": text}, "log_tail": text}


def _pass(rung: str) -> dict:
    return {"verdict": "PASS", "rung": rung, "exit_code": 0, "statuses": {"t": "passed"}}


def test_transitions_are_mechanical() -> None:
    s = ec.ControllerState()
    assert s.route == ec.ENVIRONMENT
    ec.after_environment(s, ec.SetupResult(container_id="c1", actions=7))
    assert (s.route, s.container_id, s.environment_rounds_used) == (ec.TRIAL, "c1", 1)
    # an environment failure signature → back to the environment worker
    ec.after_trial(s, [_pass("G0"), _fail("G2", "ModuleNotFoundError: No module named 'torch'")], repo_modules={"sapg"})
    assert s.route == ec.ENVIRONMENT
    assert s.reason == "third-party module missing: torch"
    assert s.trials == 1
    ec.after_environment(s, ec.SetupResult(container_id="c1", actions=3))
    # a code failure → the repair worker
    ec.after_trial(s, [_pass("G0"), _fail("G2", "NameError: name 'undefined_name' is not defined")], repo_modules={"sapg"})
    assert s.route == ec.REPAIR
    assert s.reason == "no environment signature in the failure"
    # the agent changed the code → judge it
    ec.after_repair(s, ec.RepairResult(attribution=repair.CODE, changed=True, commit="abc"))
    assert (s.route, s.commit, s.repair_rounds_used) == (ec.TRIAL, "abc", 1)
    ec.after_trial(s, [_pass("G0"), _pass("G2")])
    assert (s.route, s.passed, s.stop_reason) == (ec.DONE, True, "criterion passed")


def test_same_failure_twice_stops_unless_the_agent_re_routed() -> None:
    s = ec.ControllerState(route=ec.TRIAL, container_id="c1")
    ec.after_trial(s, [_fail("G0", "AssertionError: still broken at 0x7f 2.5s")])
    assert s.route == ec.REPAIR
    ec.after_repair(s, ec.RepairResult(attribution=repair.CODE, changed=True, commit="a1"))
    ec.after_trial(s, [_fail("G0", "AssertionError: still  broken at 0x1f 3.1s")])
    assert s.route == ec.STOPPED
    assert "reproduced unchanged" in s.stop_reason

    # the agent said "environment" while changing the code: the hand-over wins over the signature stop
    s2 = ec.ControllerState(route=ec.TRIAL, container_id="c1")
    ec.after_trial(s2, [_fail("G0", "AssertionError: still broken")])
    ec.after_repair(s2, ec.RepairResult(attribution=repair.ENVIRONMENT, changed=True, commit="a1"))
    ec.after_trial(s2, [_fail("G0", "AssertionError: still broken")], route_hint=ec.ENVIRONMENT)
    assert s2.route == ec.ENVIRONMENT
    assert "attributed the failure to the environment" in s2.reason


def test_repair_without_a_change_hands_over_or_stops() -> None:
    s = ec.ControllerState(route=ec.REPAIR, container_id="c1")
    ec.after_repair(s, ec.RepairResult(attribution=repair.ENVIRONMENT, changed=False, summary="libGL missing"))
    assert s.route == ec.ENVIRONMENT
    assert "libGL missing" in s.reason
    s2 = ec.ControllerState(route=ec.REPAIR, container_id="c1")
    ec.after_repair(s2, ec.RepairResult(attribution=repair.CODE, changed=False))
    assert (s2.route, s2.stop_reason) == (ec.STOPPED, "the repair agent changed nothing")
    s3 = ec.ControllerState(route=ec.ENVIRONMENT)
    ec.after_environment(s3, ec.SetupResult(container_id="", completed=False, error="docker died"))
    assert s3.route == ec.STOPPED
    assert "left no container" in s3.stop_reason


def test_budgets_stop_before_dispatch() -> None:
    s = ec.ControllerState(route=ec.ENVIRONMENT, environment_rounds_used=3)
    ec.budget_check(s, environment_rounds=3, repair_rounds=3)
    assert (s.route, s.stop_reason) == (ec.STOPPED, "3 environment round(s) used")
    s = ec.ControllerState(route=ec.REPAIR, repair_rounds_used=2)
    ec.budget_check(s, environment_rounds=3, repair_rounds=2)
    assert s.stop_reason == "2 repair round(s) used"
    s = ec.ControllerState(route=ec.TRIAL, repair_rounds_used=9, environment_rounds_used=9)
    ec.budget_check(s, environment_rounds=1, repair_rounds=1)
    assert s.route == ec.TRIAL  # a trial is never budgeted


def test_full_chain_offline() -> None:
    """environment problem → 搭建环境 → trial fails on code → 修复代码 → trial passes; every box is a double."""
    calls: list[str] = []
    trial_replies = [
        [_pass("G0"), _fail("G2", "ModuleNotFoundError: No module named 'gym'")],  # after env round 1: environment
        [_pass("G0"), _fail("G2", "TypeError: cannot unpack non-iterable NoneType object")],  # after env round 2: code
        [_pass("G0"), _pass("G2")],  # after the repair
    ]
    containers = iter(["c-1", "c-1", "c-1"])

    def environment(state):
        calls.append(f"env:{state.round_no}:{state.reason[:24]}")
        return ec.SetupResult(container_id=next(containers), actions=5)

    def trial(state):
        calls.append(f"trial:{state.round_no}:{state.container_id}:{state.commit or '-'}")
        return trial_replies.pop(0)

    def repair_worker(state):
        calls.append(f"repair:{state.round_no}:{state.last_verdict['rung']}")
        return ec.RepairResult(attribution=repair.CODE, changed=True, commit="deadbeef", summary="returned a tuple", record={"written": ["sapg/x.py"]})

    events: list[tuple] = []
    controller = ec.EnvironmentController(
        ec.Workers(environment=environment, trial=trial, repair=repair_worker), environment_rounds=3, repair_rounds=3,
        repo_modules=lambda: {"sapg"}, events=lambda kind, **f: events.append((kind, f)),
    )
    state = controller.run()
    assert state.passed
    assert state.route == ec.DONE
    assert calls == [
        "env:1:round 0: the environment",
        "trial:1:c-1:-",
        "env:2:third-party module missi",
        "trial:2:c-1:-",
        "repair:3:G2",
        "trial:3:c-1:deadbeef",
    ]
    assert [r.kind for r in state.records] == [ec.ENVIRONMENT, ec.ENVIRONMENT, ec.REPAIR]
    assert state.records[2].agent == {"written": ["sapg/x.py"]}
    assert state.records[2].passed is True
    assert (state.environment_rounds_used, state.repair_rounds_used, state.trials) == (2, 1, 3)
    assert [e[0] for e in events][:3] == ["controller.round", "controller.trial", "controller.judged"]
    assert events[-1] == ("controller.stop", {"passed": True, "reason": "criterion passed", "rounds": 3, "gpu_required": False})
    assert state.to_dict()["rounds"][0]["environment_actions"] == 5


def test_worker_exceptions_become_stops_not_crashes() -> None:
    def boom(state):
        raise RuntimeError("ssh dropped")

    controller = ec.EnvironmentController(ec.Workers(environment=boom, trial=boom, repair=boom), environment_rounds=2, repair_rounds=2)
    state = controller.run()
    assert state.route == ec.STOPPED
    assert "left no container" in state.stop_reason
    assert state.records[0].error == "RuntimeError: ssh dropped"

    def env_ok(state):
        return ec.SetupResult(container_id="c")

    controller = ec.EnvironmentController(ec.Workers(environment=env_ok, trial=boom, repair=boom), environment_rounds=2, repair_rounds=2)
    state = controller.run()
    assert state.stop_reason == "the trial worker failed: RuntimeError: ssh dropped"
    assert state.records[0].error == state.stop_reason


def test_a_lost_probe_container_rebuilds_the_environment_at_the_new_commit() -> None:
    # sapg-2 GPU run 3: RSA's rollback_to_checkpoint returned False (its `docker run` of the checkpoint failed) and
    # left no container; the worker reports container_lost and the controller sends the next round to 搭建环境
    s = ec.ControllerState(route=ec.REPAIR, container_id="c1")
    ec.after_repair(s, ec.RepairResult(attribution=repair.CODE, changed=True, commit="n1", container_lost=True))
    assert (s.route, s.container_id, s.commit) == (ec.ENVIRONMENT, "", "n1")
    assert "rollback failed" in s.reason
    assert "at the new commit" in s.reason
    s2 = ec.ControllerState(route=ec.REPAIR, container_id="c1")
    ec.after_repair(s2, ec.RepairResult(attribution=repair.CODE, changed=False, container_lost=True))
    assert s2.route == ec.ENVIRONMENT
    assert "at the new commit" not in s2.reason


def test_repair_result_carries_the_replaced_container_and_grade_error_rebuilds() -> None:
    # sapg-s9-off (2026-09-18): RSA's rollback replaced the container after the repair round; the controller still held
    # the old id, SetupX was pointed at a removed container, the trial then returned GRADE_ERROR
    s = ec.ControllerState(route=ec.REPAIR, container_id="old")
    ec.after_repair(s, ec.RepairResult(attribution=repair.CODE, changed=True, commit="n1", container_id="new"))
    assert (s.route, s.container_id) == (ec.TRIAL, "new")
    ec.after_trial(s, [{"verdict": "GRADE_ERROR", "rung": "G0", "reason": "working-tree reset failed: /workspace/repo is not a git working tree"}])
    assert s.route == ec.ENVIRONMENT
    assert s.container_id == ""
    assert "could not be applied" in s.reason


def test_a_gpu_class_failure_stops_for_the_escalation_on_a_cpu_machine_and_is_code_on_a_gpu_one() -> None:
    # two-stage compute (owner 2026-09-18): CPU first; the run's own evidence rents the GPU
    cpu = ec.ControllerState(commit="c0", container_id="k", route=ec.TRIAL)
    ec.after_trial(cpu, [_fail("G2", "RuntimeError: Torch not compiled with CUDA enabled")])
    assert cpu.route == ec.STOPPED
    assert cpu.gpu_required is True
    assert cpu.stop_reason == f"{ec.GPU_REQUIRED}: the code wants CUDA and this machine has no GPU"
    assert cpu.to_dict()["gpu_required"] is True
    gpu = ec.ControllerState(commit="c0", container_id="k", route=ec.TRIAL, gpu_available=True)
    ec.after_trial(gpu, [_fail("G2", "RuntimeError: Torch not compiled with CUDA enabled")])
    assert gpu.route == ec.REPAIR
    assert gpu.gpu_required is False
    assert gpu.reason == "the code's CUDA use fails on a machine that has a GPU"
    # SetupX's own word
    s = ec.ControllerState(commit="c0")
    ec.after_environment(s, ec.SetupResult(container_id="k", actions=12, gpu_required=True))
    assert s.route == ec.STOPPED
    assert s.stop_reason.startswith(ec.GPU_REQUIRED)
    s_gpu = ec.ControllerState(commit="c0", gpu_available=True)
    ec.after_environment(s_gpu, ec.SetupResult(container_id="k", actions=12, gpu_required=True))
    assert s_gpu.route == ec.TRIAL
    # the repair agent's word, change or no change: the commit comes along
    r = ec.ControllerState(commit="c0", container_id="k", route=ec.REPAIR)
    ec.after_repair(r, ec.RepairResult(attribution=repair.ENVIRONMENT, changed=True, commit="c1", summary="the model requires a GPU; isaacgym is a GPU-only build"))
    assert r.route == ec.STOPPED
    assert r.commit == "c1"
    assert r.stop_reason.startswith(f"{ec.GPU_REQUIRED}: the repair agent says so")
    r2 = ec.ControllerState(commit="c0", container_id="k", route=ec.REPAIR)
    ec.after_repair(r2, ec.RepairResult(attribution=repair.ENVIRONMENT, changed=False, summary="libGL.so.1 is missing"))
    assert r2.route == ec.ENVIRONMENT
