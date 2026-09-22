"""The state machine around the setup loop. Three terminal states, and only three.

Section 5.7. Success, escalation-on-stall, and budget-exhausted are kept apart
because collapsing them destroys the thing escalation is for. Four of 33 measured
runs ended on the wall clock; folding those into "the agent gave up and asked for
help" makes the escalation channel noisier exactly where it needs to be quiet.

The Router owns every decision the agent used to make about itself:

* **when to stop trying** -- the failing set failing to shrink for K rounds, not
  the agent's own sense that it is stuck. An agent may *request* escalation, and
  the request adds one to the stall counter; it does not open the door.
* **whether a failure is even the agent's problem** -- two cases are diverted
  before the stall gate sees them, because charging them to the agent would make
  it work on something it cannot fix:
    - the run completed cleanly and an artefact is still absent, which means a
      stage was never reached: a shrink-configuration fault, back to the Compiler.
    - the criterion fails at tier 1 and passes at tier 2, which means the failure
      was manufactured by shrinking (BatchNorm on batch=1, an empty class in a 1%
      subset, a scheduler dividing by zero at epochs=1).

A note on the stall signal. SetupX has a `[NO PROGRESS]` detector
(`agent.py:412-488`) and it is tempting to reuse. It compares a digest of
`(exit_code, stdout, stderr)`, which includes pytest's `=== 1 failed in 3.21s ===`
timing line and pip's progress output, so it cannot fire on the pytest and pip
re-runs where stalling actually happens -- measured trigger rates 0.5% and 0.4%.
A Router relying on it would never escalate and would always burn to the budget.
The primary signal here is the Adjudicator's failing set; `[NO PROGRESS]` is at
most a hint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Protocol

from .adjudicator import FAIL, GRADE_ERROR, PASS, Verdict
from .criterion import Criterion, NEEDS_USER_APPROVAL, Rung
from .escalation import EscalationCard, build_card
from .freezer import FrozenCriterion, FrozenLadder
from .kickback import ActionLedger, grading_contract, kickback


class Terminal(str, Enum):
    SUCCESS = "success"
    ESCALATED = "escalated"          # stalled; a human is asked
    BUDGET = "budget_exhausted"      # stopped by its own frozen budget
    RECOMPILE = "recompile"          # the criterion or the shrink config is wrong
    BLOCKED = "blocked"              # an asset is missing; never entered the loop
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class RoundActions:
    """What one pass of the setup loop did."""
    actions: list[str] = field(default_factory=list)
    stopped_voluntarily: bool = False
    requested_help: bool = False
    error: str = ""


@dataclass
class RoundRecord:
    round_no: int
    rung: str
    verdict: str
    missing_n: int
    passed_expected: int
    expected_n: int
    tier: int = 1
    loop_error: str = ""             # the setup loop itself crashed
    diverted: str = ""               # "shrink-artefact" | "stage-skipped" | ""
    counted_toward_stall: bool = True
    elapsed_s: float = 0.0
    actions_n: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Budget:
    wall_seconds: float = 3600.0
    tokens: int = 2_000_000
    started: float = field(default_factory=time.monotonic)
    tokens_used: Callable[[], int] = lambda: 0

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def exhausted(self) -> str:
        if self.elapsed() >= self.wall_seconds:
            return (f"wall clock: {self.elapsed():.0f}s of {self.wall_seconds:.0f}s")
        used = self.tokens_used()
        if self.tokens and used >= self.tokens:
            return f"tokens: {used} of {self.tokens}"
        return ""


@dataclass
class RouterConfig:
    max_rounds: int = 8
    # K=3 is the design document's own guess, and it is recorded as one. It is the
    # number of consecutive rounds whose failing set does not shrink before the
    # system stops and asks.
    stall_k: int = 3
    two_tier: bool = True
    disclose_ids: bool = True
    # Rebuilding loses real progress, so it is the last thing tried rather than a
    # routine reaction to a failed round (design section 8.1).
    rebuild_before_escalating: bool = True


class SetupLoop(Protocol):
    def __call__(self, *, round_no: int, contract: str, kickback: str,
                 rebuild: bool = False) -> RoundActions: ...


@dataclass
class RungOutcome:
    rung: str
    terminal: Terminal
    verdict: Verdict | None = None
    rounds: list[RoundRecord] = field(default_factory=list)
    card: EscalationCard | None = None
    note: str = ""


@dataclass
class RunOutcome:
    terminal: Terminal
    reached: str = ""
    per_rung: list[RungOutcome] = field(default_factory=list)
    card: EscalationCard | None = None

    def to_dict(self) -> dict:
        return {
            "terminal": self.terminal.value,
            "reached": self.reached,
            "per_rung": [
                {"rung": r.rung, "terminal": r.terminal.value, "note": r.note,
                 "rounds": [x.to_dict() for x in r.rounds],
                 "verdict": r.verdict.to_dict() if r.verdict else None}
                for r in self.per_rung
            ],
            "escalation": self.card.to_dict() if self.card else None,
        }


class Router:
    def __init__(self, cfg: RouterConfig, *, setup_loop: SetupLoop,
                 adjudicate: Callable[[FrozenCriterion, int], Verdict],
                 budget: Budget,
                 approve_rung: Callable[[Rung], bool] | None = None):
        self.cfg = cfg
        self.setup_loop = setup_loop
        self.adjudicate = adjudicate
        self.budget = budget
        # G3/G4' cost hours or days. Absent an explicit approver, they are not
        # attempted -- silence is not consent for somebody else's GPU time.
        self.approve_rung = approve_rung or (lambda r: False)

    # -- top level ---------------------------------------------------------

    def run(self, ladder: FrozenLadder) -> RunOutcome:
        out = RunOutcome(terminal=Terminal.SUCCESS)
        for c in ladder.ladder.ordered():
            rung = c.rung
            if rung in NEEDS_USER_APPROVAL and not self.approve_rung(rung):
                out.per_rung.append(RungOutcome(
                    rung=rung.value, terminal=Terminal.NEEDS_APPROVAL,
                    note=f"{rung.value} costs hours or days and was not approved"))
                out.terminal = Terminal.SUCCESS if out.reached else Terminal.NEEDS_APPROVAL
                return out

            ro = self._run_rung(ladder.get(rung))
            out.per_rung.append(ro)
            if ro.terminal is not Terminal.SUCCESS:
                # The ladder is monotone: a rung that did not pass forbids the next.
                out.terminal = ro.terminal
                out.card = ro.card
                return out
            out.reached = rung.value
        return out

    # -- one rung ----------------------------------------------------------

    def _run_rung(self, frozen: FrozenCriterion) -> RungOutcome:
        c = frozen.criterion
        ro = RungOutcome(rung=c.rung.value, terminal=Terminal.ESCALATED)
        ledger = ActionLedger()
        contract = grading_contract(c, frozen.expected,
                                    disclose_ids=self.cfg.disclose_ids)

        stall = 0
        prev_missing: set[str] | None = None
        prev_n: int | None = None
        message = ""
        # Two flags, not one: `rebuild_next` is consumed by the next round, while
        # `rebuild_used` latches for the whole rung. Collapsing them into one makes
        # the "one rebuild before escalating" fire on every stall instead of once,
        # and the rung then never escalates at all.
        rebuild_next = False
        rebuild_used = False

        for round_no in range(1, self.cfg.max_rounds + 1):
            if (why := self.budget.exhausted()):
                ro.terminal = Terminal.BUDGET
                ro.card = build_card("budget", c, ro.verdict,
                                     history=ledger.tried_this_round,
                                     ineffective=ledger.ineffective,
                                     extra_facts=[f"stopped by its own frozen budget ({why})"])
                return ro

            t0 = time.monotonic()
            actions = self.setup_loop(round_no=round_no, contract=contract,
                                      kickback=message, rebuild=rebuild_next)
            rebuild_next = False

            v = self.adjudicate(frozen, 1)
            ro.verdict = v
            rec = RoundRecord(round_no=round_no, rung=c.rung.value, verdict=v.verdict,
                              missing_n=len(v.missing), passed_expected=v.passed_expected,
                              expected_n=v.expected_n, elapsed_s=round(time.monotonic() - t0, 1),
                              actions_n=len(actions.actions), loop_error=actions.error)

            if v.verdict == PASS:
                rec.counted_toward_stall = False
                ro.rounds.append(rec)
                ro.terminal = Terminal.SUCCESS
                return ro

            if v.verdict == GRADE_ERROR:
                # The ruler could not be applied. Charging that to the agent would
                # send it to fix an environment that was never measured.
                rec.diverted = "grade-error"
                rec.counted_toward_stall = False
                ro.rounds.append(rec)
                ro.terminal = Terminal.RECOMPILE
                # A crashed setup loop is the likelier cause than a wrong ruler,
                # and it has to lead the report: swallowing it turns a broken
                # harness into what reads like a broken criterion.
                ro.note = ((f"the setup loop crashed: {actions.error}. " if actions.error else "")
                           + v.reason)
                ro.card = build_card(
                    "grade-error", c, v,
                    extra_facts=(["the criterion could not be applied"]
                                 + ([f"the setup loop raised: {actions.error}"]
                                    if actions.error else [])))
                return ro

            # -- diversions, before the stall gate sees anything -------------

            if self._stage_was_skipped(v, frozen):
                rec.diverted = "stage-skipped"
                rec.counted_toward_stall = False
                ro.rounds.append(rec)
                ro.terminal = Terminal.RECOMPILE
                ro.note = (
                    "the run completed cleanly yet an artefact is absent, so a stage "
                    "was never reached: the shrink configuration skipped it "
                    f"(missing: {', '.join(v.missing[:5])})")
                ro.card = build_card("shrink-config", c, v, extra_facts=[ro.note])
                return ro

            if self.cfg.two_tier and c.shrink_tier2 is not None:
                v2 = self.adjudicate(frozen, 2)
                if v2.verdict == PASS:
                    rec.diverted = "shrink-artefact"
                    rec.counted_toward_stall = False
                    rec.tier = 2
                    ro.rounds.append(rec)
                    ro.terminal = Terminal.RECOMPILE
                    ro.note = (
                        "the criterion fails at the small shrink tier and passes at "
                        "the larger one, so the failure was manufactured by shrinking "
                        "(batch size, an empty class in the subset, a scheduler at "
                        "one epoch) rather than by the environment")
                    ro.card = build_card("shrink-artefact", c, v,
                                         extra_facts=[ro.note])
                    return ro

            # -- real failure ------------------------------------------------

            missing = set(v.missing)
            shrank = prev_n is None or len(missing) < prev_n
            stall = 0 if shrank else stall + 1
            if actions.requested_help:
                # The agent may ask. Asking moves the counter; it does not decide.
                stall += 1
            ledger.note_round(actions.actions, gap_shrank=shrank)
            ro.rounds.append(rec)

            tails = v.failure_tails
            message = kickback(v, ledger, round_no=round_no,
                               rounds_total=self.cfg.max_rounds, tails=tails,
                               previous_missing=prev_missing)
            prev_missing, prev_n = missing, len(missing)

            if stall >= self.cfg.stall_k:
                if self.cfg.rebuild_before_escalating and not rebuild_used:
                    # One rebuild before giving up: rounds of repair accumulate
                    # wrong pins and conflicting installs, and a clean container
                    # sometimes clears what no further command can.
                    rebuild_next = rebuild_used = True
                    stall = 0
                    ro.note = "rebuilt the container once before escalating"
                    continue
                ro.terminal = Terminal.ESCALATED
                ro.card = build_card(
                    "stalled", c, v, tails=tails,
                    history=ledger.tried_this_round, ineffective=ledger.ineffective,
                    guesses=self._guesses(v),
                    extra_facts=[f"the failing set did not shrink for "
                                 f"{self.cfg.stall_k} consecutive rounds"])
                return ro

        ro.terminal = Terminal.ESCALATED
        ro.card = build_card(
            "round-limit", c, ro.verdict,
            tails=ro.verdict.failure_tails if ro.verdict else {},
            history=ledger.tried_this_round, ineffective=ledger.ineffective,
            guesses=self._guesses(ro.verdict) if ro.verdict else [],
            extra_facts=[f"reached the {self.cfg.max_rounds}-round limit"])
        return ro

    # -- discriminators -----------------------------------------------------

    @staticmethod
    def _stage_was_skipped(v: Verdict, frozen: FrozenCriterion) -> bool:
        """Did the run finish cleanly and still not produce an artefact?

        Section 4.5 routes a missing artefact back to the Compiler as a shrink
        error. Taken literally that misfires on research repositories, where the
        commonest cause of a missing figure is that the plotting stage *ran* and
        crashed on an absent optional dependency -- which is an environment fault
        and exactly what the agent should be fixing.

        The frozen criterion already contains the discriminator. `test_run_completes`
        asserts the command exited 0. If it passed, the command ran to completion
        without error and the artefact is still absent, which can only mean the
        stage was never entered. If it failed, something raised, and that is the
        agent's problem.
        """
        run_test = next((t for t in frozen.expected if t.endswith("::test_run_completes")),
                        None)
        if run_test is None or run_test in v.missing:
            return False
        artifact_tests = {t for t in frozen.expected if "::test_artifact_" in t}
        missing = set(v.missing)
        return bool(missing) and missing <= artifact_tests

    @staticmethod
    def _guesses(v: Verdict) -> list[str]:
        """Cheap, deterministic hypotheses -- and labelled as hypotheses.

        No model is asked for these. The card renders them under a heading that
        says they are not established, because the measured problem is that
        confident wrong explanations are expensive for a user to check.
        """
        out: list[str] = []
        text = v.log_tail + "\n".join(v.failure_tails.values())
        if "ModuleNotFoundError" in text or "ImportError" in text:
            out.append("an import fails; the named module is probably an optional "
                       "dependency that a plain requirements install misses")
        if "CUDA" in text or "nvidia" in text.lower():
            out.append("something touches CUDA; the driver or toolkit version may "
                       "not match what the pinned framework expects")
        if "No such file or directory" in text:
            out.append("a path is absent; this is often a dataset or weight file "
                       "that was never downloaded rather than a packaging fault")
        if "Permission denied" in text:
            out.append("a permission error, which usually means a write outside the "
                       "workspace rather than a missing package")
        if v.status_counts.get("SKIPPED"):
            out.append("some assertions were skipped; skips never count as passes here")
        return out
