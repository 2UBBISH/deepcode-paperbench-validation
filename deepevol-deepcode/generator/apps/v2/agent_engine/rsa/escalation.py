"""What the system hands back when it stops: evidence and options, never a conclusion.

Section 5.8. The reason this is a hard rule rather than a style preference is that
the agent's wrong explanations are *persuasive* -- they arrive with specific
technical detail and the right vocabulary. Three measured examples, all confident,
all wrong, and all three later overturned when the same repositories were retried:

    typeguard    "mypy's output format differs across versions"
    pdfplumber   "Ghostscript is missing; it is a system tool, not a Python
                  package, so it is outside a setup agent's remit"
    pipdeptree   "the virtualenv version is legal; the tests have not kept up"

To evaluate any of those a user needs a working reference environment -- exactly
the thing they do not have and asked for. So a conclusion here is worse than
nothing: it is a wrong answer that is expensive to check. Facts are labelled
facts, guesses are labelled guesses, and the decision stays with the user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .adjudicator import Verdict
from .criterion import Criterion


@dataclass
class Option:
    key: str
    label: str
    detail: str


@dataclass
class EscalationCard:
    reason: str                       # why we stopped: "stalled" | "budget" | "assets" ...
    rung: str
    goal: str
    command: str

    facts: list[str] = field(default_factory=list)
    failing: list[str] = field(default_factory=list)
    error_tails: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    ineffective: list[str] = field(default_factory=list)

    guesses: list[str] = field(default_factory=list)   # always rendered as guesses
    options: list[Option] = field(default_factory=list)
    alarms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        out = [f"# Stopped: {self.reason}", "",
               f"Goal: {self.goal}", f"Rung: {self.rung}", f"Command: {self.command}", ""]

        out += ["## What is true (measured, not inferred)", ""]
        out += [f"- {f}" for f in self.facts] or ["- (nothing recorded)"]

        if self.failing:
            out += ["", f"## Assertions that did not pass ({len(self.failing)})", ""]
            out += [f"- `{t}`" for t in self.failing[:40]]
            if len(self.failing) > 40:
                out += [f"- ... and {len(self.failing) - 40} more"]

        for name, tail in list(self.error_tails.items())[:6]:
            out += ["", f"### {name}", "", "```", tail.strip()[-1200:], "```"]

        if self.alarms:
            out += ["", "## Warnings about the criterion itself", ""]
            out += [f"- {a}" for a in self.alarms]

        if self.ineffective:
            out += ["", "## Tried, and did not help", ""]
            out += [f"- {a}" for a in self.ineffective[-20:]]

        out += ["", "## Guesses (NOT established -- the agent could be wrong here)", ""]
        out += [f"- (guess) {g}" for g in self.guesses] or ["- (none offered)"]

        out += ["", "## Your call", ""]
        for o in self.options:
            out += [f"- **{o.key}** - {o.label}", f"  {o.detail}"]
        return "\n".join(out)


def build_card(reason: str, c: Criterion, v: Verdict | None, *,
               tails: dict[str, str] | None = None,
               history: list[str] | None = None,
               ineffective: list[str] | None = None,
               guesses: list[str] | None = None,
               extra_facts: list[str] | None = None) -> EscalationCard:
    facts = list(extra_facts or [])
    if v is not None:
        facts += [
            f"external verdict: {v.verdict} ({v.passed_expected}/{v.expected_n} "
            f"frozen assertions passed)",
            f"the working tree was restored to {c.commit[:12]} before grading"
            + (" (reset succeeded)" if v.reset_ok else " (RESET FAILED)"),
        ]
        if v.tampered:
            facts.append(
                f"the agent had modified {v.changed_n} tracked file(s) before the "
                f"reset: {', '.join(v.changed_files[:6])}")
        if v.cleared_artifacts:
            facts.append("declared output files were deleted before the run, so a "
                         "passing artefact check means this run produced it")

    card = EscalationCard(
        reason=reason,
        rung=c.rung.value,
        goal=c.goal,
        command=c.command,
        facts=facts,
        failing=list(v.missing) if v else [],
        error_tails=dict(tails or {}),
        history=list(history or []),
        ineffective=list(ineffective or []),
        guesses=list(guesses or []),
        alarms=list(v.alarms) if v else [],
    )
    card.options = _options(reason, bool(card.alarms))
    return card


def _options(reason: str, has_alarms: bool) -> list[Option]:
    opts = [
        Option("criterion-wrong",
               "The criterion is wrong; amend it and retry",
               "Nothing the agent installs can satisfy an assertion that does not "
               "describe this experiment. Amending requires your approval and is "
               "recorded in the freeze history."),
        Option("repo-or-assets",
               "The repository or its assets genuinely cannot run here",
               "Unmaintained pins, a yanked dependency, a dataset behind a licence, "
               "a Python or CUDA version that is no longer obtainable. This is the "
               "boundary the product does not cross."),
        Option("continue",
               "Keep trying, with more budget",
               "Repeated attempts do help: pass@1 59.5% vs pass@3 70.3% on "
               "BuildBench. Say how much more wall clock, tokens or GPU time to spend."),
    ]
    if has_alarms:
        opts.insert(0, Option(
            "check-the-ruler",
            "Check the criterion's test ids before anything else",
            "The numbers carry the fingerprint of ids that do not line up rather "
            "than of a broken environment -- the same shape that judged 98 of "
            "gunicorn's 245 genuinely passing tests as missing."))
    if reason == "budget":
        opts = [o for o in opts if o.key != "continue"] + [
            Option("more-budget", "Raise the budget and resume",
                   "The run was stopped by its own frozen budget, not by a failure. "
                   "The gap at the moment it stopped is above.")]
    return opts
