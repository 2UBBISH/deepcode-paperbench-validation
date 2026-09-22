"""Step 10's scheduler, the owner's four-box shape (PLAN-3 §0 "第 10 步的形状", §2.3 S8).

::

    input repository ──▶ controller ──▶ 搭建环境 (SetupX, black box)   ──▶ SetupResult
                            │  ▲       远程初步执行 (adjudicate ≤ G2)    ──▶ list[Verdict]
                            │  └────── 修复代码 (the line's repair agent) ──▶ RepairOutcome
                            └──▶ done / stopped

The controller is **mechanical**: a state dict and pure transition functions, no model call of its own.
Round 0 is no longer RSA's ``run_pipeline`` doing "SetupX rounds ↔ adjudicate" by itself (its Router
kicks every failure back to SetupX, which then spends its steps on code bugs it may not fix); the
controller calls the environment worker and the trial worker as two separate steps from the first
round on, and after every failed trial decides mechanically where the next round goes:

* the failure text carries an environment signature (``repair.classify``) → 搭建环境;
* otherwise → 修复代码; the repair agent may hand the round back with
  ``attribution="environment"`` (PLAN-3 S1 ②), which routes the next round to 搭建环境;
* the same failure signature twice in a row (S1 ③) → stopped, into the runner's ``repair_review``;
* the failure wants a GPU this machine does not have (``repair.GPU`` from the failure text, SetupX's
  FINISH saying "GPU-only", or the agent's environment attribution saying so) → stopped with the
  ``GPU_REQUIRED`` reason: the phase re-rents the GPU tier and starts the environment box again there
  (two-stage compute, owner 2026-09-18 afternoon: step 10 starts on the cheapest CPU machine, the
  GPU is rented only on evidence; on a GPU machine the same signal is a code failure like any other);
* passed → done.

Budgets: ``environment_rounds`` (SetupX rounds, the run's ``SETUPX_MAX_ROUNDS``) and ``repair_rounds``
(``run.json.repair_rounds``); each trial follows a change and is not budgeted separately.

The workers' interfaces are fixed (:class:`Workers`): the real ones live in ``experiment_step`` /
``repair_loop`` (SetupX's ``run_round`` behind the flow's seams, RSA's ``Adjudicator``, the repair
agent with commit → move container → re-freeze); tests pass doubles. This module imports nothing
from RSA.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from apps.v2.agent.paper2code import repair
from apps.v2.agent.paper2code.repair_loop import failure_signature, failure_text

ENVIRONMENT = "environment"
TRIAL = "trial"
REPAIR = "repair"
DONE = "done"
STOPPED = "stopped"
ROUTES = (ENVIRONMENT, TRIAL, REPAIR, DONE, STOPPED)
#: ``stop_reason`` prefix of the compute escalation (the phase reads it; ``repair_review`` does not fire on it)
GPU_REQUIRED = "gpu required"


@dataclass(slots=True)
class SetupResult:
    """What 搭建环境 returns: the container it left behind and what it did (the black box's own summary)."""

    container_id: str
    actions: int = 0
    completed: bool = True
    error: str = ""
    requested_help: bool = False
    #: SetupX's own FINISH said the build or run needs a GPU (``repair.gpu_needed`` on its text)
    gpu_required: bool = False


@dataclass(slots=True)
class RepairResult:
    """What 修复代码 returns: the agent's outcome plus what the runner did with it (commit, re-pin)."""

    attribution: str
    changed: bool
    commit: str = ""
    summary: str = ""
    record: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    #: the probe container could not be restored after the agent's probes; the environment must be rebuilt
    container_lost: bool = False
    #: the container after the round (RSA's rollback replaces it); empty = unchanged / unknown
    container_id: str = ""


@dataclass(slots=True)
class Workers:
    """The three boxes. Each is a callable the controller invokes with the current state."""

    environment: Callable[["ControllerState"], SetupResult]
    trial: Callable[["ControllerState"], list[dict[str, Any]]]  # verdict dicts in the evidence's shape, rung order, stops at the first failure
    repair: Callable[["ControllerState"], RepairResult]


@dataclass(slots=True)
class RoundRecord:
    round_no: int
    kind: str
    reason: str
    seconds: float = 0.0
    commit: str = ""
    container_id: str = ""
    environment_actions: int = 0
    agent: dict[str, Any] | None = None
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    signature: str = ""
    error: str = ""

    def record(self) -> dict[str, Any]:
        return {
            "round": self.round_no, "kind": self.kind, "reason": self.reason, "seconds": round(self.seconds, 1),
            "commit": self.commit, "container_id": self.container_id, "environment_actions": self.environment_actions,
            "agent": self.agent, "verdicts": list(self.verdicts), "passed": self.passed, "signature": self.signature, "error": self.error,
        }


@dataclass(slots=True)
class ControllerState:
    route: str = ENVIRONMENT
    reason: str = "round 0: the environment is not built yet"
    round_no: int = 0
    container_id: str = ""
    commit: str = ""
    last_verdict: dict[str, Any] = field(default_factory=dict)
    last_signature: str = ""
    environment_rounds_used: int = 0
    repair_rounds_used: int = 0
    trials: int = 0
    passed: bool = False
    stop_reason: str = ""
    records: list[RoundRecord] = field(default_factory=list)
    #: the machine has a GPU: a GPU-class failure is then the code's, not the compute tier's
    gpu_available: bool = False
    gpu_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route, "reason": self.reason, "round": self.round_no, "container_id": self.container_id, "commit": self.commit,
            "environment_rounds_used": self.environment_rounds_used, "repair_rounds_used": self.repair_rounds_used, "trials": self.trials,
            "passed": self.passed, "stop_reason": self.stop_reason, "last_signature": self.last_signature,
            "gpu_available": self.gpu_available, "gpu_required": self.gpu_required,
            "rounds": [r.record() for r in self.records],
        }


def _stop_for_gpu(state: ControllerState, why: str) -> ControllerState:
    state.gpu_required = True
    state.route, state.stop_reason = STOPPED, f"{GPU_REQUIRED}: {why}"
    state.reason = state.stop_reason
    return state


# ---------------------------------------------------------------------------
# pure transitions
# ---------------------------------------------------------------------------


def after_environment(state: ControllerState, result: SetupResult) -> ControllerState:
    """搭建环境 returned: the trial follows, on the container it left (a crashed setup loop still gets judged
    when it left a container — the verdict is the evidence; without one there is nothing to judge)."""
    state.environment_rounds_used += 1
    if result.container_id:
        state.container_id = result.container_id
    if result.gpu_required and not state.gpu_available:
        return _stop_for_gpu(state, "the environment worker says the build or run needs a GPU")
    if not state.container_id:
        state.route, state.stop_reason = STOPPED, f"the environment worker left no container ({result.error or 'no error text'})"
        state.reason = state.stop_reason
        return state
    state.route = TRIAL
    state.reason = "the environment worker returned; judge the frozen criterion on its container"
    return state


def after_trial(state: ControllerState, verdicts: list[dict[str, Any]], *, repo_modules: set[str] = frozenset(), route_hint: str | None = None) -> ControllerState:
    """远程初步执行 returned: pass → done; the same failure again → stopped; else the mechanical route.

    ``route_hint`` is the repair agent's ``attribution="environment"`` from the round before this trial: the
    agent said the failure it saw was the environment's, so if the trial still fails the next round is
    搭建环境 regardless of the heuristic — and the repeated-signature stop does not fire for that hand-over.
    """
    state.trials += 1
    passed = bool(verdicts) and all(str(v.get("verdict", "")) == "PASS" for v in verdicts)
    if passed:
        state.passed, state.route, state.stop_reason = True, DONE, "criterion passed"
        state.reason = state.stop_reason
        return state
    last = dict(verdicts[-1]) if verdicts else dict(state.last_verdict)
    if str(last.get("verdict", "")) == "GRADE_ERROR":
        # the ruler could not be applied (a broken or vanished container, not the code): rebuild the environment
        state.last_verdict = last
        state.container_id = ""
        state.route = ENVIRONMENT
        state.reason = f"the criterion could not be applied ({str(last.get('reason') or 'GRADE_ERROR')[:120]}); rebuild the environment"
        return state
    signature = failure_signature(last)
    if route_hint is None and state.last_signature and signature == state.last_signature:
        state.route = STOPPED
        state.stop_reason = f"the previous failure reproduced unchanged (signature {signature})"
        state.reason = state.stop_reason
        state.last_verdict = last
        return state
    state.last_verdict, state.last_signature = last, signature
    if route_hint == ENVIRONMENT:
        state.route, state.reason = ENVIRONMENT, "the repair agent attributed the failure to the environment"
        return state
    kind, why = repair.classify(failure_text(last), repo_modules_=repo_modules)
    if kind == repair.GPU:
        if not state.gpu_available:
            return _stop_for_gpu(state, why)
        kind, why = repair.CODE, "the code's CUDA use fails on a machine that has a GPU"
    state.route = ENVIRONMENT if kind == repair.ENVIRONMENT else REPAIR
    state.reason = why
    return state


def after_repair(state: ControllerState, result: RepairResult) -> ControllerState:
    """修复代码 returned: a change is judged next; no change and "environment" hands over; no change and
    "code" means the agent had nothing — stop."""
    state.repair_rounds_used += 1
    if result.commit:
        state.commit = result.commit
    if result.container_id:
        state.container_id = result.container_id
    if result.error:
        state.route, state.stop_reason = STOPPED, f"the repair worker failed: {result.error}"
        state.reason = state.stop_reason
        return state
    if result.container_lost:
        state.container_id = ""
        state.route = ENVIRONMENT
        state.reason = "the probe container was lost after the repair agent's probes (rollback failed); rebuild the environment" + (" at the new commit" if result.changed else "")
        return state
    if result.attribution == repair.ENVIRONMENT and not state.gpu_available and repair.gpu_needed(result.summary):
        # its change (if any) is committed and comes along to the GPU machine
        return _stop_for_gpu(state, f"the repair agent says so: {result.summary[:160]}")
    if result.changed:
        state.route = TRIAL
        state.reason = "the repair agent changed the code; judge it" + (" (it also says the environment is at fault)" if result.attribution == repair.ENVIRONMENT else "")
        return state
    if result.attribution == repair.ENVIRONMENT:
        state.route, state.reason = ENVIRONMENT, f"the repair agent attributed the failure to the environment: {result.summary[:160]}"
        return state
    state.route, state.stop_reason = STOPPED, "the repair agent changed nothing"
    state.reason = state.stop_reason
    return state


def budget_check(state: ControllerState, *, environment_rounds: int, repair_rounds: int) -> ControllerState:
    """Before dispatching: a route whose budget is spent stops the loop (the runner's review point follows)."""
    if state.route == ENVIRONMENT and state.environment_rounds_used >= environment_rounds:
        state.route, state.stop_reason = STOPPED, f"{environment_rounds} environment round(s) used"
        state.reason = state.stop_reason
    elif state.route == REPAIR and state.repair_rounds_used >= repair_rounds:
        state.route, state.stop_reason = STOPPED, f"{repair_rounds} repair round(s) used"
        state.reason = state.stop_reason
    return state


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


class EnvironmentController:
    """Drives the state through the workers until done / stopped. ``events`` gets one call per transition."""

    def __init__(
        self,
        workers: Workers,
        *,
        environment_rounds: int,
        repair_rounds: int,
        repo_modules: Callable[[], set[str]] | None = None,
        events: Callable[..., Any] | None = None,
        state: ControllerState | None = None,
    ) -> None:
        self.workers = workers
        self.environment_rounds, self.repair_rounds = int(environment_rounds), int(repair_rounds)
        self._repo_modules = repo_modules or (lambda: set())
        self.events = events
        self.state = state or ControllerState()

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.events is not None:
            try:
                self.events(kind, **fields)
            except Exception:
                pass

    def run(self) -> ControllerState:
        state = self.state
        route_hint: str | None = None
        while True:
            budget_check(state, environment_rounds=self.environment_rounds, repair_rounds=self.repair_rounds)
            if state.route in (DONE, STOPPED):
                self._emit("controller.stop", passed=state.passed, reason=state.stop_reason, rounds=len(state.records), gpu_required=state.gpu_required)
                return state
            started = time.monotonic()
            if state.route == ENVIRONMENT:
                state.round_no += 1
                record = RoundRecord(round_no=state.round_no, kind=ENVIRONMENT, reason=state.reason, commit=state.commit)
                state.records.append(record)
                self._emit("controller.round", round=state.round_no, route=ENVIRONMENT, reason=state.reason)
                try:
                    result = self.workers.environment(state)
                except Exception as exc:
                    result = SetupResult(container_id=state.container_id, completed=False, error=f"{type(exc).__name__}: {exc}")
                record.environment_actions, record.error = result.actions, result.error
                after_environment(state, result)
                record.container_id = state.container_id
                route_hint = None
            elif state.route == TRIAL:
                record = state.records[-1] if state.records else None
                self._emit("controller.trial", round=state.round_no, container=state.container_id)
                try:
                    verdicts = self.workers.trial(state)
                except Exception as exc:
                    state.route, state.stop_reason = STOPPED, f"the trial worker failed: {type(exc).__name__}: {exc}"
                    state.reason = state.stop_reason
                    if record is not None:
                        record.error = state.stop_reason
                    continue
                after_trial(state, verdicts, repo_modules=self._repo_modules(), route_hint=route_hint)
                route_hint = None
                if record is not None:
                    record.verdicts = [dict(v) for v in verdicts]
                    record.passed = state.passed
                    record.signature = state.last_signature
                    record.seconds += time.monotonic() - started
                self._emit("controller.judged", round=state.round_no, passed=state.passed, next=state.route, reason=state.reason)
                continue
            else:  # REPAIR
                state.round_no += 1
                record = RoundRecord(round_no=state.round_no, kind=REPAIR, reason=state.reason, commit=state.commit, container_id=state.container_id)
                state.records.append(record)
                self._emit("controller.round", round=state.round_no, route=REPAIR, reason=state.reason)
                try:
                    result = self.workers.repair(state)
                except Exception as exc:
                    result = RepairResult(attribution=repair.CODE, changed=False, error=f"{type(exc).__name__}: {exc}")
                record.agent, record.error = result.record or None, result.error
                after_repair(state, result)
                record.commit = state.commit
                route_hint = ENVIRONMENT if (result.attribution == repair.ENVIRONMENT and result.changed) else None
            record.seconds = time.monotonic() - started


__all__ = [
    "DONE",
    "ENVIRONMENT",
    "GPU_REQUIRED",
    "REPAIR",
    "ROUTES",
    "STOPPED",
    "TRIAL",
    "ControllerState",
    "EnvironmentController",
    "RepairResult",
    "RoundRecord",
    "SetupResult",
    "Workers",
    "after_environment",
    "after_repair",
    "after_trial",
    "budget_check",
]
