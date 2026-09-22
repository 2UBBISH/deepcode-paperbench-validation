"""Three-point falsification: does this criterion measure anything?

Compiling is a one-shot act by a model, and once the result is frozen a mistake
in it is invisible -- it presents as a wide FAIL, not as an error. We made four
such mistakes freezing criteria by hand (a dangling `-k`, a `../` prefix, an
absolute path in a parametrize id, a clone of upstream HEAD instead of the pin),
and all four silently polluted their data. So the criterion is attacked before it
is trusted.

    (1) bare container          the criterion MUST fail        -> else recompile
    (2) top-level deps only     the criterion SHOULD fail      -> else disclose
    (3) static, no execution    referenced paths must line up  -> else recompile

G0-G2 run all three. G3/G4' run only (3): a criterion that takes hours cannot be
run twice more just to check itself.

**Point (2) is where criterion strength actually becomes measurable, and this
implementation turns it into a number.** Point (1) cannot do it: a bare container
fails a worthless criterion and an excellent one alike -- `def test_import():
import foo` does fail with nothing installed. What separates them is exactly the
point-(2) environment: top-level requirements present, datasets, weights and
optional extras absent. A weak criterion is *satisfied* there; a criterion that
reaches the end of the script is not. So point (2) reports the fraction of
assertions that survive, and a high fraction is disclosed as weakness. That is
the 24-point gap between "imports and starts" (97.7%) and "the expected set
passes" (73.4%), available before the run instead of after it.
"""

from __future__ import annotations

import re
import shlex
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterator

from .adjudicator import Adjudicator, GRADE_ERROR, PASS
from .bridge import Bridge
from .criterion import Criterion, LOOP_RUNGS, Rung
from .freezer import FrozenCriterion

# A criterion that still mostly passes with nothing installed is not measuring the
# environment. The threshold is a disclosure trigger, not a gate: what counts as
# weak depends on the goal, and the design document is explicit that point (2)
# informs the user rather than blocking automatically.
WEAK_SURVIVAL = 0.5

FALSIFIED = "FALSIFIED"   # the criterion failed, as it must have
SURVIVED = "SURVIVED"     # the criterion passed where it should not have
ERROR = "ERROR"


@dataclass
class PointResult:
    point: int
    name: str
    verdict: str
    passed_expected: int = 0
    expected_n: int = 0
    detail: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def survival(self) -> float:
        """Fraction of the criterion that still passes in a deliberately broken env."""
        return round(self.passed_expected / max(self.expected_n, 1), 4)


@dataclass
class FalsificationReport:
    rung: str
    points: list[PointResult] = field(default_factory=list)
    accepted: bool = False
    must_recompile: bool = False
    disclosures: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        head = "accepted" if self.accepted else (
            "REJECTED -> recompile" if self.must_recompile else "blocked")
        bits = [f"{p.name}: {p.verdict}"
                + (f" ({p.passed_expected}/{p.expected_n} still pass)" if p.point != 3 else "")
                for p in self.points]
        return f"[{self.rung}] {head}; " + "; ".join(bits)


# --------------------------------------------------------------------------
# Point 3 -- static, and therefore always affordable
# --------------------------------------------------------------------------

_FLAGLIKE = re.compile(r"^-")
_PATHLIKE = re.compile(r"[/\\]|\.(py|ya?ml|json|toml|cfg|ini|sh|txt|csv|md)$")


# Where the command writes, rather than reads. A redirect target and the value of
# an `--output-file` flag are things the run is supposed to *create*, so demanding
# that they already exist at the frozen commit rejects a perfectly good criterion:
# measured on proselint, `command references 'proselint_config.json', which does
# not exist at commit dbed789c` -- for the file the command's whole purpose is to
# produce.
_REDIRECTS = frozenset({">", ">>", "1>", "2>", "&>", "2>&1", "|", "tee"})
_OUTPUT_FLAGS = frozenset({
    "-o", "--out", "--outdir", "--out-dir", "--output", "--output-dir",
    "--outputdir", "--output-directory", "--output-file", "--output-path",
    "--dest", "--destination", "--target-dir", "--outputfile",
})


def referenced_paths(c: Criterion) -> list[str]:
    """Repository paths the frozen command reads.

    Extracted from the command rather than asked of the model: the point of a
    static check is that it does not depend on the same reading that produced the
    thing being checked. Outputs are excluded -- a path the command creates is not
    evidence of anything at the frozen commit.
    """
    out: list[str] = []
    try:
        tokens = shlex.split(c.command)
    except ValueError:
        tokens = c.command.split()
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in _REDIRECTS or any(tok.startswith(r) for r in (">", ">>")):
            skip_next = tok in _REDIRECTS       # `>out.json` carries its own value
            continue
        if _FLAGLIKE.match(tok):
            name, sep, val = tok.partition("=")
            if name.lower() in _OUTPUT_FLAGS:
                skip_next = not sep
                continue
            # `--config=configs/x.yaml`
            if sep and _PATHLIKE.search(val):
                out.append(val)
            continue
        if _PATHLIKE.search(tok) and not tok.startswith(("http://", "https://")):
            out.append(tok)
    # Drop the interpreter itself: `python`, `/usr/bin/python3`, `.venv/bin/python`.
    out = [p for p in out if not re.search(r"(^|/)(python[\d.]*|uv|poetry|bash|sh)$", p)]
    # An absolute path is not a repository path, so asking whether the frozen
    # commit contains it is a category error rather than a check. Measured on
    # robotframework, where the compiled command was
    #
    #     python -m robot --outputdir /output --log /output/log.html \
    #                     --report /output/report.html atest/...
    #
    # and point 3 reported `/output/log.html` as "does not exist at commit
    # 6f22043" -- a file the command itself creates, in a directory that is not
    # in the repository at all. `_OUTPUT_FLAGS` covers `--outputdir`, but no list
    # of flag names is ever complete, and this rule needs none: whatever lives
    # outside the working tree cannot be an input the commit was supposed to
    # carry.
    return [p for p in out if not p.startswith("/")]


def falsify_static(c: Criterion, tracked: set[str]) -> PointResult:
    """Point 3. `tracked` is the file list of the frozen commit.

    Two independent ways a criterion can be statically wrong, and they fail in
    opposite directions:

    * A **script or config it names is absent** at the frozen commit. The criterion
      is then constantly false and no amount of configuration will satisfy it.
    * An **artefact it asserts on already exists** at the frozen commit. The
      criterion is then partly constantly true: `exists("README.md")` passes
      before anything has run. This is the failure the design document does not
      name and that point (1) can miss, because a criterion can be vacuous in one
      assertion while genuinely failing in another.
    """
    r = PointResult(point=3, name="static-paths", verdict=FALSIFIED)
    missing = [p for p in referenced_paths(c) if p.lstrip("./") not in tracked]
    if missing:
        r.verdict = ERROR
        r.problems += [
            f"command references {p!r}, which does not exist at commit {c.commit[:12]}"
            for p in missing
        ]

    for a in c.artifacts:
        if a.path.lstrip("./") in tracked:
            r.verdict = ERROR
            r.problems.append(
                f"artefact {a.path!r} is already tracked at the frozen commit, so "
                f"`{a.check}` on it passes before the run: the assertion is vacuous"
            )

    if c.workdir and not c.workdir.startswith("/"):
        r.problems.append(f"workdir {c.workdir!r} is not absolute")
        r.verdict = ERROR

    r.detail = ("all referenced paths exist and no artefact pre-exists"
                if r.verdict == FALSIFIED else f"{len(r.problems)} problem(s)")
    return r


def tracked_at(bridge: Bridge, workdir: str, commit: str) -> set[str]:
    """Every path the commit contains -- **directories included**.

    `git ls-tree -r --name-only` lists files only, so a command whose argument is
    a directory was reported as referencing something that does not exist at the
    frozen commit, and the whole ladder was rejected. Measured on cookiecutter:
    `command references 'tests/fake-repo-pre', which does not exist at commit
    c88fbe92` -- for a template directory that is right there. Directory
    arguments are ordinary (`--input-dir`, a template root, a data directory), so
    the parent prefixes belong in the set.
    """
    cmd = f"git ls-tree -r --name-only {commit}" if commit else "git ls-files"
    res = bridge.run(cmd, timeout=180, workdir=workdir)
    if not res.ok:
        raise RuntimeError(f"cannot list files at {commit or 'HEAD'}: {res.output[:300]}")
    files = {l.strip() for l in res.output.splitlines() if l.strip()}
    dirs: set[str] = set()
    for f in files:
        parts = f.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return files | dirs


# --------------------------------------------------------------------------
# Points 1 and 2 -- executed
# --------------------------------------------------------------------------

# A provisioner hands out a Bridge onto a deliberately-incomplete environment and
# tears it down afterwards. Abstracted so the falsifier can be exercised without
# Docker, and so the real one can reuse SetupX's own container creation -- if the
# bare environment is not built the same way the agent's is, point (1) is
# answering a question about a different container.
Provisioner = Callable[[], "Iterator[tuple[Bridge, dict[str, str]]]"]


def falsify_executed(point: int, name: str, frozen: FrozenCriterion,
                     provision: Provisioner, *, expect: str) -> PointResult:
    r = PointResult(point=point, name=name, verdict=ERROR,
                    expected_n=len(frozen.expected))
    try:
        with contextmanager(provision)() as (bridge, agent_env):
            adj = Adjudicator(bridge, workdir=frozen.criterion.workdir)
            # No reset: this environment was provisioned by us, not by an agent,
            # so there is nothing to undo -- and making falsification depend on the
            # clone being deep enough would fail it for an unrelated reason.
            v = adj.adjudicate(frozen, agent_env=agent_env, reset=False)
    except Exception as e:  # provisioning failed; that is not evidence either way
        r.detail = f"could not provision the {name} environment: {type(e).__name__}: {e}"
        return r

    if v.verdict == GRADE_ERROR:
        r.detail = f"the criterion could not be applied: {v.reason}"
        return r

    r.passed_expected = v.passed_expected
    r.expected_n = v.expected_n
    r.verdict = SURVIVED if v.verdict == PASS else FALSIFIED
    r.detail = (f"{v.passed_expected}/{v.expected_n} assertions still pass "
                f"in the {name} environment")
    if r.verdict == SURVIVED:
        r.problems.append(
            f"the criterion passes in the {name} environment, so it is not "
            f"measuring what {expect}")
    elif point == 1 and r.passed_expected:
        # Not fatal on its own -- a "the output directory is writable" assertion
        # legitimately passes with nothing installed -- but worth surfacing,
        # because the usual cause is an assertion that cannot distinguish states.
        r.problems.append(
            f"{r.passed_expected} of {r.expected_n} assertions pass with nothing "
            "installed at all; check that each of them can actually fail.")
    elif point == 2 and r.survival > WEAK_SURVIVAL:
        r.problems.append(
            f"weak: {r.survival:.0%} of the criterion is already satisfied by "
            "installing top-level requirements, with no datasets, weights or "
            "optional extras. A criterion that stops at the training loop misses "
            "the tail of the script, which is where the fragile dependencies are.")
    return r


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def falsify(frozen: FrozenCriterion, *, tracked: set[str],
            bare: Provisioner | None = None,
            toplevel: Provisioner | None = None) -> FalsificationReport:
    c = frozen.criterion
    rep = FalsificationReport(rung=c.rung.value)

    p3 = falsify_static(c, tracked)
    rep.points.append(p3)
    if p3.verdict == ERROR:
        rep.must_recompile = True
        rep.reasons += p3.problems

    if c.rung not in LOOP_RUNGS:
        # An hours-long criterion is not run twice more to check itself. Static
        # falsification is what remains affordable, and it is applied in full.
        rep.accepted = not rep.must_recompile
        if rep.accepted:
            rep.disclosures.append(
                f"{c.rung.value} was checked statically only; points 1 and 2 are not "
                "run at this rung because the criterion costs hours to execute.")
        return rep

    if bare is not None:
        p1 = falsify_executed(1, "bare", frozen, bare,
                              expect="a configured environment provides")
        rep.points.append(p1)
        if p1.verdict == SURVIVED:
            # Constantly true. Recompiling is the only remedy; disclosing it would
            # invite approving a criterion that cannot fail.
            rep.must_recompile = True
            rep.reasons += p1.problems
        elif p1.verdict == ERROR:
            rep.reasons.append(f"point 1 inconclusive: {p1.detail}")
        else:
            rep.disclosures += p1.problems

    if toplevel is not None and not rep.must_recompile:
        p2 = falsify_executed(2, "top-level-deps-only", frozen, toplevel,
                              expect="the full set of dependencies and assets provides")
        rep.points.append(p2)
        # Never an automatic block. On research repositories point 2 fires often,
        # and that is the good case: it is catching the plotting and reporting tail
        # -- the optional extras -- which means the criterion reaches the end of the
        # script rather than stopping at the training loop.
        rep.disclosures += p2.problems
        if p2.verdict == SURVIVED:
            rep.disclosures.append(
                "point 2: the criterion passes with only top-level requirements "
                "installed and no datasets or weights. Confirm this is intended -- "
                "it usually means the criterion does not reach the stages that need them.")

    rep.accepted = not rep.must_recompile
    return rep
