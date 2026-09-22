"""What the agent is told between rounds.

Two distinct messages, delivered through two different channels, and mixing them
up costs a round:

* **The grading contract** -- what the environment will be judged on. Constant for
  the whole run, disclosed once up front.
* **The kickback** -- what specifically failed last round, what was already tried,
  and what has already been ruled out. Rewritten every round.

Section 5.7 forbids "failed, keep going". The prohibition has a measured price:
across the same 33 repositories, a message carrying only the grading command
rescued 6-7, while one carrying the specific failing test ids rescued 10, and the
convergence was faster as well -- `dask` hit a 4-hour wall clock with the command
alone and finished in 1.6 hours with the ids. Twenty-six of 34 failures were
missing 20 tests or fewer, usually one optional dependency; knowing *which* is
the whole difference between a fixable report and a shrug.

The wording of the contract is taken from `harness/verifier_hint.py:88-126`
verbatim rather than paraphrased, because that is the text that was measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .adjudicator import Verdict
from .criterion import Criterion

# pandas has 212,017 expected tests. A cap is not politeness, it is the
# difference between a usable prompt and a buried one.
MAX_LISTED = 40
MAX_TAILS = 8


@dataclass
class ActionLedger:
    """What has been tried, and what has been ruled out. Survives across rounds.

    This is the only state carried between rounds. Everything else is deliberately
    discarded: a fresh agent object per round is how the conversation gets reset,
    and a multi-round ReAct transcript is what produced a measured 2.61M-token
    single run.
    """

    ineffective: list[str] = field(default_factory=list)
    tried_this_round: list[str] = field(default_factory=list)

    def note_round(self, actions: list[str], gap_shrank: bool) -> None:
        self.tried_this_round = actions[-40:]
        if not gap_shrank:
            # Only actions from a round that moved nothing are called ineffective.
            # A round that shrank the gap contains, somewhere in it, something that
            # worked, and we cannot tell which -- so nothing from it is condemned.
            for a in actions:
                if a not in self.ineffective:
                    self.ineffective.append(a)
        del self.ineffective[:-60]


def grading_contract(c: Criterion, expected: list[str], *, disclose_ids: bool = True) -> str:
    """The frozen ruler, disclosed to the agent.

    Not an answer leak. The agent is being asked to build an environment in which
    these assertions hold; concealing them buys nothing but wasted rounds, and the
    reset before adjudication is what makes disclosure survivable -- editing the
    repository to satisfy a listed assertion stops working at the moment it is
    graded.
    """
    parts = [
        "[GRADING CONTRACT - external, authoritative]",
        f"This environment is graded by running a frozen criterion from {c.workdir}.",
        "The criterion runs, in full:",
        f"    {c.command}",
        "",
        "and then asserts on what that run produced: files that must exist and load,",
        "numbers that must be present and sane.",
        "",
        "Verification counts ONLY if that exact command runs and those assertions pass.",
        "Do NOT substitute a weaker check: a version probe, a bare import, or a "
        "different invocation does not qualify.",
        "If the command cannot run at all, the environment is NOT ready, however "
        "complete the installation looks.",
    ]
    if disclose_ids and expected:
        shown = expected[:MAX_LISTED]
        parts += [
            "",
            f"The criterion contains {len(expected)} assertions"
            + (f" ({len(shown)} shown)" if len(shown) < len(expected) else "")
            + ", ALL of which must pass:",
            *(f"    {t}" for t in shown),
        ]
    parts += [
        "",
        "Before grading, the working tree is restored to its pinned commit and the "
        "declared output files are deleted. Editing the repository, its tests or its "
        "configuration therefore cannot affect the verdict -- only the environment can.",
    ]
    return "\n".join(parts)


def kickback(v: Verdict, ledger: ActionLedger, *, round_no: int, rounds_total: int,
             tails: dict[str, str] | None = None,
             previous_missing: set[str] | None = None) -> str:
    """The between-rounds report. Never "failed, continue"."""
    tails = tails or {}
    missing = v.missing
    parts = [
        f"[ROUND {round_no} of at most {rounds_total} - EXTERNAL VERDICT: {v.verdict}]",
        f"{v.passed_expected} of {v.expected_n} frozen assertions passed. "
        f"{len(missing)} did not.",
    ]

    if previous_missing is not None:
        fixed = sorted(previous_missing - set(missing))
        broke = sorted(set(missing) - previous_missing)
        if fixed:
            parts += ["", f"Fixed since last round ({len(fixed)}): "
                          + ", ".join(_short(t) for t in fixed[:12])]
        if broke:
            parts += [f"NEWLY broken since last round ({len(broke)}): "
                      + ", ".join(_short(t) for t in broke[:12])
                      + "  <- something done last round caused this"]
        if not fixed and not broke and missing:
            parts += ["", "The failing set is UNCHANGED from last round. Repeating the "
                          "same approach will not move it; try a different hypothesis."]

    if missing:
        shown = missing[:MAX_LISTED]
        parts += ["", f"Assertions still failing ({len(shown)} of {len(missing)} shown):"]
        parts += [f"    {t}" for t in shown]

    shown_tails = 0
    for t in missing:
        name = t.split("::")[-1]
        tail = tails.get(name)
        if not tail:
            continue
        parts += ["", f"--- {name} ---", tail.strip()[-900:]]
        shown_tails += 1
        if shown_tails >= MAX_TAILS:
            break

    if not missing and v.verdict != "PASS":
        # A GRADE_ERROR: the ruler could not be applied. Say so plainly rather than
        # letting the agent infer that its environment is at fault.
        parts += ["", f"The criterion could not be applied: {v.reason}",
                  "This is a problem with the harness or the checkout, not "
                  "necessarily with your environment."]

    if ledger.tried_this_round:
        parts += ["", "What you did last round:"]
        parts += [f"    {a}" for a in ledger.tried_this_round[:20]]

    if ledger.ineffective:
        parts += ["", "Already tried in earlier rounds and confirmed NOT to help "
                      "(do not repeat these):"]
        parts += [f"    {a}" for a in ledger.ineffective[-25:]]

    parts += [
        "",
        "You are configuring the ENVIRONMENT. The repository is restored to its "
        "pinned commit before grading, so editing its files, tests or configuration "
        "changes nothing.",
    ]
    return "\n".join(parts)


def _short(test_id: str) -> str:
    return test_id.split("::")[-1]
