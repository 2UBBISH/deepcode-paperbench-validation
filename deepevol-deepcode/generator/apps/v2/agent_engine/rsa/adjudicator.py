"""The only thing in this system allowed to say whether the environment is ready.

Section 5.6. Nothing here calls an LLM and nothing here weighs whether a failure
"counts". It resets the working tree, runs the frozen criterion, and compares two
sets. That is the entire mandate, and the narrowness is the point: on the
SWE-smith 128, when the agent asserted "this failure is not the environment's
fault" it was wrong 59% of the time (deepseek 17/29, gemma 22/34) -- and that was
not a lapse, it was the behaviour the verifier's own prompt asked for.

The order of operations is load-bearing and is not a matter of taste:

    1. capture tamper telemetry        <- BEFORE the reset, or it reads clean forever
    2. reset the working tree          <- cheating stops working here
    3. check the ruler's own integrity
    4. run the frozen criterion
    5. compare sets, and only then decide

Step 1 before step 2 is the part that is easy to get backwards. `harness/l3_grade.py`
runs `check_tampered` *after* grading (:212-213), which was correct while nothing
reset the tree; move the reset in front of it without moving the telemetry and
every run reports a pristine repository from then on, silently. Twelve tampered
runs were measured on the 128, three of which had been graded PASS.

Why reset rather than detect: an LLM asked to spot cheating misses it. Once the
tree is restored from the frozen commit, there is nothing left to detect.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .bridge import Bridge, BridgeError, DockerBridge
from .criterion import Criterion
from .freezer import FrozenCriterion, sha256_file
from .render import snapshot_paths
from .testlog import PASSING, counts, failure_tails, norm, parse_log


# A verdict is one of these and nothing else. In particular there is no
# "passed with acceptable failures": that category is where the 59% lived.
PASS = "PASS"
FAIL = "FAIL"
GRADE_ERROR = "GRADE_ERROR"      # the ruler could not be applied; NOT a failure of the environment


@dataclass
class Verdict:
    verdict: str
    rung: str = ""
    reset_ok: bool = False
    authoritative: bool = True

    expected_n: int = 0
    passed_expected: int = 0
    pass_rate: float = 0.0
    missing: list[str] = field(default_factory=list)        # FULL list, never sampled
    extra_passed: list[str] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)  # FULL {test_id: status}
    status_counts: dict[str, int] = field(default_factory=dict)

    exit_code: int = 0
    parsed_any: bool = False
    duration_s: float = 0.0
    log_tail: str = ""
    # Parsed from the COMPLETE log at judgement time. Re-deriving them later from
    # `log_tail` loses them exactly when they matter most: the tail is capped, and
    # a run with many failures pushes the earliest -- often the causative --
    # traceback out of it. These tails are the whole reason a kickback works;
    # measured, a message carrying them rescued 10 repositories where one carrying
    # only the command rescued 6-7.
    failure_tails: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    # Telemetry. Recorded, never consulted when deciding.
    head_before: str = ""
    head_after: str = ""
    tampered: bool | None = None
    changed_files: list[str] = field(default_factory=list)
    changed_n: int = 0
    cleared_artifacts: list[str] = field(default_factory=list)

    # Section 11: a criterion can be wrong in a way three-point falsification
    # cannot catch, and it presents as a large-area FAIL rather than an error.
    alarms: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == PASS

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.verdict == GRADE_ERROR:
            return f"GRADE_ERROR: {self.reason}"
        return (f"{self.verdict} {self.passed_expected}/{self.expected_n} expected"
                + (f", {len(self.missing)} missing" if self.missing else "")
                + (f", {len(self.extra_passed)} unexpected passes" if self.extra_passed else ""))


class Adjudicator:
    def __init__(self, bridge: Bridge, *, workdir: str | None = None,
                 accept: frozenset[str] = PASSING):
        self.bridge = bridge
        self.workdir = workdir or bridge.workdir
        self.accept = accept

    # -- the five steps ---------------------------------------------------

    def adjudicate(self, frozen: FrozenCriterion, *, agent_env: dict[str, str] | None = None,
                   out_dir: Path | None = None, reset: bool = True,
                   tier: int = 1) -> Verdict:
        """Judge the environment against one frozen rung.

        `reset=False` produces a *preview*: the same comparison without restoring
        the working tree. The setup loop uses it for fast in-loop feedback, and it
        is marked `authoritative=False` so no report can mistake it for a verdict.
        The delta between a passing preview and a failing authoritative run is
        itself the tamper signal.
        """
        t0 = time.time()
        c = frozen.criterion.with_shrink(tier)
        v = Verdict(verdict=GRADE_ERROR, rung=c.rung.value, authoritative=reset)

        if not frozen.expected:
            v.reason = ("criterion has an empty expected set; every run satisfies it. "
                        "Refusing to issue a verdict.")
            return self._finish(v, t0, out_dir)

        # 1. Telemetry BEFORE the reset.
        try:
            self._capture_tamper(v)
        except BridgeError as e:
            v.reason = f"cannot reach the environment: {e}"
            return self._finish(v, t0, out_dir)

        # 2. Reset.
        if reset:
            ok, why = self._reset(c)
            v.reset_ok = ok
            if not ok:
                # Not a FAIL. An un-reset adjudication is an adjudication whose
                # anti-cheat guarantee does not hold, and reporting it as a pass or
                # a fail either way would be reporting something we did not measure.
                v.reason = f"working-tree reset failed: {why}"
                return self._finish(v, t0, out_dir)
            v.head_after = self._git("rev-parse HEAD").strip()

        # 2b. Clear the declared artefacts, so "it exists" means "this run made it".
        cleared, why = self._clear_artifacts(c)
        v.cleared_artifacts = cleared
        if why:
            v.reason = why
            return self._finish(v, t0, out_dir)

        # 3. The ruler's own integrity.
        try:
            frozen.check_integrity()
        except Exception as e:
            v.reason = str(e)
            return self._finish(v, t0, out_dir)

        # 4. Run it.
        try:
            code, log = self._run_criteria(frozen, c, agent_env or {})
        except BridgeError as e:
            v.reason = f"criteria run failed: {e}"
            return self._finish(v, t0, out_dir)
        v.exit_code = code
        v.log_tail = log[-4000:]
        v.failure_tails = failure_tails(log)

        # 5. Compare.
        statuses = parse_log(log, "pytest_rA")
        v.statuses = statuses
        v.status_counts = counts(statuses)
        v.parsed_any = bool(statuses)
        passed = {norm(t) for t, s in statuses.items() if s in self.accept}
        expected = {norm(t): t for t in frozen.expected}

        v.expected_n = len(expected)
        v.missing = sorted(raw for k, raw in expected.items() if k not in passed)
        v.extra_passed = sorted(passed - set(expected))
        v.passed_expected = v.expected_n - len(v.missing)
        v.pass_rate = round(v.passed_expected / max(v.expected_n, 1), 4)
        v.verdict = PASS if not v.missing else FAIL
        v.alarms = self._alarms(v)
        return self._finish(v, t0, out_dir)

    # -- steps in detail ---------------------------------------------------

    def _capture_tamper(self, v: Verdict) -> None:
        """What the agent changed under the frozen commit, recorded before undoing it.

        Tracked files only. Installation legitimately scatters `.egg-info/`,
        `__pycache__/` and `.pytest_cache/` through the tree, and counting those
        would flag every run that ever succeeded.
        """
        v.head_before = self._git("rev-parse HEAD").strip()
        r = self.bridge.run("git status --porcelain --untracked-files=no",
                            timeout=120, workdir=self.workdir)
        if not r.ok:
            v.tampered = None
            v.reason = f"git status failed: {r.output[:200]}"
            return
        changed = [l[3:].strip() for l in r.output.splitlines() if l.strip()]
        v.tampered = bool(changed)
        v.changed_files = changed[:50]
        v.changed_n = len(changed)

    def _reset(self, c: Criterion) -> tuple[bool, str]:
        """Restore tracked files to the frozen commit; leave untracked ones alone.

        The design document says `git checkout -- .`, which restores modified
        tracked files relative to whatever HEAD currently is. That is not quite
        enough here: an agent may commit, or check out a different revision, and
        then `git checkout -- .` faithfully restores the *wrong* baseline. Naming
        the frozen commit explicitly subsumes it and is the same operation when
        HEAD has not moved.

        Untracked files survive on purpose. `.egg-info/`, an in-place compiled
        extension, a built wheel -- deleting those would destroy the environment
        the agent legitimately built and turn every adjudication into a fresh
        install.
        """
        if not self.bridge.run("git rev-parse --is-inside-work-tree",
                               timeout=60, workdir=self.workdir).ok:
            return False, f"{self.workdir} is not a git working tree"

        if c.commit:
            r = self.bridge.run(f"git checkout --force {c.commit}",
                                timeout=300, workdir=self.workdir)
            if not r.ok:
                # A shallow clone will not contain the pinned commit. That is a
                # setup error in how the run was launched, not a failing
                # environment, so it must surface as GRADE_ERROR.
                return False, (f"git checkout --force {c.commit[:12]} failed "
                               f"(shallow clone?): {r.output[-400:]}")
        else:
            r = self.bridge.run("git checkout -- .", timeout=300, workdir=self.workdir)
            if not r.ok:
                return False, f"git checkout -- . failed: {r.output[-400:]}"

        left = self.bridge.run("git status --porcelain --untracked-files=no",
                               timeout=120, workdir=self.workdir)
        if left.ok and left.output.strip():
            return False, ("tracked files still modified after reset: "
                           + " ".join(left.output.split()[:20]))
        return True, ""

    def _clear_artifacts(self, c: Criterion) -> tuple[list[str], str]:
        """Delete the declared outputs before running, so their presence means something.

        Without this, the artefact manifest silently stops working after the first
        adjudication: `figs/curve.png` left behind by round 1 makes the plotting
        stage look like it ran in round 2, even though it never did. The whole
        point of the manifest (section 4.5) is to detect a stage that was skipped,
        and a stale file is indistinguishable from a fresh one.

        The reset does not cover this. Outputs are untracked by construction, and
        untracked files are deliberately preserved -- that is what keeps the
        agent's `.egg-info/` and compiled extensions alive. So outputs have to be
        removed by name, and only outputs: a declared artefact that is *tracked* at
        the frozen commit is part of the repository, and deleting it would break
        the very run we are about to judge.
        """
        paths = [a.path for a in c.artifacts]
        if not paths:
            return [], ""
        tracked: set[str] = set()
        r = self.bridge.run("git ls-files -z", timeout=120, workdir=self.workdir)
        if r.ok:
            tracked = {p for p in r.output.split("\0") if p}
        removable = [p for p in paths if p.lstrip("./") not in tracked]

        # Removing the declared *files* is not enough to restore the pre-run
        # state. Many commands refuse to write into a directory they created
        # earlier -- cookiecutter answers `Error: "out/fake-project" directory
        # already exists` -- so once anything in the loop has run the command
        # once (an in-loop VERIFY does exactly that), no later adjudication can
        # ever pass. Measured: eight consecutive rounds, same message, on a
        # criterion that is provably sound in a fresh container.
        #
        # So the untracked ancestor directories of each declared output go too.
        # Only untracked ones: a directory that exists at the frozen commit is
        # part of the repository, and deleting it would break the run being judged.
        for p in list(removable):
            parts = p.lstrip("./").split("/")
            for i in range(1, len(parts)):
                anc = "/".join(parts[:i])
                if anc and anc not in tracked and anc not in removable:
                    removable.append(anc)
        if not removable:
            return [], ""
        quoted = " ".join(f"'{p}'" for p in removable)
        rm = self.bridge.run(f"rm -rf -- {quoted}", timeout=300, workdir=self.workdir)
        if not rm.ok:
            return [], f"could not clear declared artefacts: {rm.output[-300:]}"
        return removable, ""

    def _run_criteria(self, frozen: FrozenCriterion, c: Criterion,
                      agent_env: dict[str, str]) -> tuple[int, str]:
        """Run the frozen pytest on the host, pointed at the environment under test.

        Invoked from the criteria directory with a bare basename so that test ids
        stay `criteria_G2.py::test_x` wherever the store lives -- ids are the
        expected set, and an id that shifts with a path is an expected set that
        cannot be compared.

        `-rA` is required (it produces the summary lines `pytest_rA` parses) and
        `--tb=short` is deliberate: the Router hands raw error tails back to the
        agent, and the measured difference between a kickback with specifics and
        one without was 10 repositories rescued versus 6.
        """
        env = dict(os.environ)
        env.update(self.bridge.prelude_env(
            command=self._full_command(c),
            timeout=c.timeout,
            repeats=c.repeats,
            env=agent_env,
            workdir=c.workdir,
            snapshot=snapshot_paths(c),
        ))
        remote_criteria_dir = ""
        if hasattr(self.bridge, "run_criteria"):
            budget = c.timeout * max(1, c.repeats) + 900
            return self.bridge.run_criteria(  # type: ignore[attr-defined]
                frozen.path, command=self._full_command(c), timeout=c.timeout,
                repeats=c.repeats, env=agent_env, workdir=c.workdir,
                snapshot=snapshot_paths(c),
            )
        if hasattr(self.bridge, "put_file"):
            remote_criteria_dir = f"/tmp/rsa-criteria-{os.getpid()}-{id(frozen):x}"
            remote_path = f"{remote_criteria_dir}/{frozen.basename}"
            self.bridge.put_file(frozen.path, remote_path)  # type: ignore[attr-defined]
            criteria_cwd = remote_criteria_dir
            criteria_arg = frozen.basename
        else:
            criteria_cwd = str(frozen.path.parent)
            criteria_arg = frozen.basename
        argv = ["python3" if hasattr(self.bridge, "put_file") else sys.executable,
                "-m", "pytest", "-rA", "--tb=short", "-p", "no:cacheprovider",
                criteria_arg]
        # The host-side budget must outlast the runs the criteria file makes; the
        # per-run bound lives in the prelude, which is where a timeout can still be
        # reported as a failing test rather than as a dead grader.
        budget = c.timeout * max(1, c.repeats) + 900
        try:
            if hasattr(self.bridge, "run") and hasattr(self.bridge, "put_file"):
                result = self.bridge.run(" ".join(shlex.quote(x) for x in argv),
                                         timeout=budget, workdir=criteria_cwd,
                                         env=env)
                return result.exit_code, result.output
            p = subprocess.run(argv, cwd=frozen.path.parent, capture_output=True,
                               text=True, timeout=budget, env=env)
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "")
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", "replace")
            # Keep whatever was produced. `harness/measure_reference.py:100`
            # discards it and returns a bare "TIMEOUT"; for G3/G4' criteria,
            # timeouts are routine and the partial log is the only evidence there is.
            return 124, (partial or "") + f"\n[rsa] criteria run exceeded {budget}s"
        except OSError as e:
            raise BridgeError(f"cannot run pytest: {e}") from e
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    @staticmethod
    def _full_command(c: Criterion) -> str:
        """The frozen command plus the shrink flags, if any."""
        extra = c.shrink.as_cli(c.command) if c.shrink else []
        return " ".join([c.command, *extra]) if extra else c.command

    def _git(self, args: str) -> str:
        r = self.bridge.run(f"git {args}", timeout=120, workdir=self.workdir)
        return r.output if r.ok else ""

    # -- section 11: the criterion might itself be wrong --------------------

    def _alarms(self, v: Verdict) -> list[str]:
        """Patterns that mean "suspect the ruler", not "suspect the environment".

        `gunicorn` is the worked example and the reason this exists: its test ids
        embedded an absolute path, so 98 of 245 genuinely passing tests were judged
        missing while 100 unrecognised ones passed. All three falsification points
        pass on such a criterion -- the bare container really does fail, the
        top-level-only install really does fail, and every referenced path really
        does exist. The only signature is numerical, so it has to be checked here.
        """
        out: list[str] = []
        n_missing, n_extra = len(v.missing), len(v.extra_passed)

        if not v.parsed_any:
            out.append(
                "NO_PARSE: not one line of the output parsed as a test result. "
                "The criteria file probably failed to start (collection error, "
                "missing pytest) -- this is not evidence that the environment is bad."
            )
            return out

        if n_missing and n_extra and abs(n_missing - n_extra) <= max(2, 0.15 * n_missing):
            out.append(
                f"ID_MISMATCH_SUSPECTED: {n_missing} expected tests missing while "
                f"{n_extra} unexpected ones passed. Near-equal counts are the "
                "fingerprint of test ids that do not line up (absolute paths baked "
                "into parametrize ids, or a shifted `../` prefix), not of a broken "
                "environment. Compare the two lists before believing this verdict."
            )
        if v.expected_n and v.passed_expected == 0 and n_extra >= v.expected_n:
            out.append(
                f"TOTAL_MISS_WITH_EXTRAS: 0/{v.expected_n} expected passed while "
                f"{n_extra} unexpected ones did. Usually the wrong checkout "
                "(upstream HEAD instead of the pinned commit) rather than a "
                "failed setup."
            )
        if v.tampered and v.verdict == PASS:
            out.append(
                f"PASSED_AFTER_TAMPER: the tree carried {v.changed_n} modified "
                "tracked file(s) before the reset. The verdict is computed on the "
                "reset tree and stands, but the agent was editing the repository."
            )
        return out

    # -- persistence --------------------------------------------------------

    def _finish(self, v: Verdict, t0: float, out_dir: Path | None) -> Verdict:
        v.duration_s = round(time.time() - t0, 2)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"adjudication_{v.rung or 'X'}{'' if v.authoritative else '_preview'}.json"
            # The full statuses map and the full missing list are persisted on
            # purpose. The Router's stall gate compares failing-criteria sets
            # round over round, and `harness/l3_grade.py` truncates `missing` to 20
            # -- verified across all 701 records on disk -- which makes exactly
            # that comparison impossible after the fact.
            (out_dir / name).write_text(
                json.dumps(v.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return v


def adjudicate(bridge: Bridge, frozen: FrozenCriterion, **kw) -> Verdict:
    """Convenience wrapper mirroring `harness/l3_grade.grade_run`."""
    return Adjudicator(bridge, workdir=frozen.criterion.workdir).adjudicate(frozen, **kw)
