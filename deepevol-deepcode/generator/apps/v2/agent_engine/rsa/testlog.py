"""Parse a test runner's output into {test_id: status}, and make ids comparable.

Vendored, not imported, from ``harness/measure_reference.py:parse_log`` and
``harness/l3_grade.py:_norm``. Two reasons for the copy:

1. ``harness/`` is not under version control, and this repository is. A ruler
   whose parsing rules can change underneath a frozen criterion is not frozen --
   the criterion's ``expected`` set was computed by *some* version of this
   parser, and if that version is not pinned alongside it the set means nothing.
2. This package is meant to stand alone (the harness lives in a sibling tree
   that may not be present).

The parser shapes are kept byte-faithful to the harness so that the numbers in
``docs/experiments/06-l3-ruler-smith128.md`` remain comparable to anything this
system measures. Where behaviour is deliberately different from the harness it
is called out in a comment.
"""

from __future__ import annotations

import re

STATUSES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")

# Only these count as a pass. SKIPPED and XFAIL explicitly do not: a missing
# optional dependency turns a test into a SKIP, the runner still exits 0, and the
# whole suite reports green while the environment is incomplete. That failure
# mode was measured -- 11 of deepseek's 34 failures exited 0, and `pydantic`
# exited 0 while 662 expected tests never ran.
#
# XPASS is where this deliberately parts company with the harness. Both
# `l3_grade.py:151` and `measure_reference.py:171` count it, because they grade a
# repository's own pre-existing suite, where an unexpectedly-passing xfail is a
# genuine pass somebody else marked pessimistically. Generated criteria are not
# in that position: `rsa.linter` bans `xfail` outright, so an XPASS in one can
# only mean the criteria file is not the one that was frozen.
PASSING_STRICT = frozenset({"PASSED"})
PASSING_LENIENT = frozenset({"PASSED", "XPASS"})
PASSING = PASSING_STRICT

PARSERS = {
    # `-rA` short summary:            PASSED tests/test_x.py::test_y
    "pytest_rA": re.compile(rf"^({'|'.join(STATUSES)})\s+(\S+)"),
    # `--verbose` progress lines:     tests/test_x.py::test_y PASSED
    "pytest_verbose": re.compile(rf"^(\S+)\s+({'|'.join(STATUSES)})"),
    # xdist:                          [gw3] PASSED tests/test_x.py::test_y
    "pytest_xdist": re.compile(rf"^\[gw\d+\]\s+({'|'.join(STATUSES)})\s+(\S+)"),
    # unittest:                       test_foo (tests.TestBar) ... ok
    "unittest_dots": re.compile(r"^(.+?)\s\.\.\.\s(ok|skipped.*|FAIL|ERROR)$"),
}

UNITTEST_STATUS = {"ok": "PASSED", "FAIL": "FAILED", "ERROR": "ERROR"}

# The Adjudicator always invokes the criteria file with `-rA`, because it owns
# that invocation (unlike the harness, which had to accept whatever command a
# benchmark froze). Everything else is retained for reading logs produced
# elsewhere -- e.g. a repository's own suite selected as a cheap G2.
DEFAULT_PARSER = "pytest_rA"


def parse_log(log: str, kind: str = DEFAULT_PARSER) -> dict[str, str]:
    """Map test id -> status. Unknown lines are ignored, never guessed at."""
    out: dict[str, str] = {}

    if kind == "mypy_lastword":
        # Status word anywhere on the line, id is the last token. Loose by
        # construction; kept faithful to SWE-smith rather than tightened.
        for line in log.splitlines():
            for status in ("PASSED", "FAILED"):
                if status in line and line.split():
                    out[line.split()[-1]] = status
                    break
        return out

    try:
        pat = PARSERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown log parser {kind!r}; known: {sorted(PARSERS) + ['mypy_lastword']}"
        ) from None

    swapped = kind in ("pytest_verbose", "unittest_dots")  # these put the id first
    for line in log.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        status, tid = (m.group(2), m.group(1)) if swapped else (m.group(1), m.group(2))
        if kind == "unittest_dots":
            status = UNITTEST_STATUS.get(
                status, "SKIPPED" if status.startswith("skipped") else status
            )
            out[tid] = status
            continue
        # A pytest id always contains "::" or a path separator; this rejects lines
        # like "PASSED in 3.2s" and progress output that starts with a status word.
        if "::" in tid or "/" in tid:
            out[tid] = status
    return out


def passed_set(log: str, kind: str = DEFAULT_PARSER,
               accept: frozenset[str] = PASSING) -> set[str]:
    """The normalised set of test ids that actually passed."""
    return {norm(t) for t, s in parse_log(log, kind).items() if s in accept}


# Roots a checkout can live under, longest first so that a nested match cannot
# shadow a longer one. `/workspace/repo` is where SetupX clones
# (environment_manager.py); `/testbed` is the SWE-smith image layout; `/criteria`
# is where this package mounts the frozen criterion (adjudicator.py).
_ROOTS = ("/workspace/repo", "/criteria", "/testbed")


def norm(tid: str) -> str:
    """Make ids comparable across containers whose checkout lives elsewhere.

    Two distinct path artefacts, both of which look like a broken environment and
    neither of which is one (both were measured on the SWE-smith 128):

    `../` depth. pytest emits ids relative to its rootdir, so a suite outside the
    package root carries a `../` prefix whose length depends on the working
    directory. `mido`: the reference image runs from /testbed and reports
    `../dev/tests/test_syx.py::test_read`, the agent container runs from
    /workspace/repo and reports `../../dev/tests/test_syx.py::test_read`. Same
    test, zero set overlap -- 0/122 expected while 122 unrelated-looking tests
    passed.

    Absolute paths inside parametrize ids. A suite that parametrizes over files
    it globbed bakes the checkout root into the id. `gunicorn`: reference reports
    `test_http_parser[/testbed/tests/requests/invalid/001.http]`, agent reports
    `test_http_parser[/workspace/repo/tests/...]` -- 98 of 245 expected tests
    "missing" while 100 unexpected ones passed. Affected 5 of 128 repositories.
    """
    while tid.startswith("../"):
        tid = tid[3:]
    for root in _ROOTS:
        tid = tid.replace(root + "/", "<root>/")
    return tid


# pytest prints one banner per failing test in its FAILURES / ERRORS sections:
#   _______________________ test_artifact_02_figs_curve_png ______________________
# The banner is the only reliable delimiter -- tracebacks contain everything else.
_FAIL_BANNER = re.compile(r"^_{3,}\s+(\S.*?)\s+_{3,}$", re.M)
_SECTION_END = re.compile(r"^=+ (short test summary|warnings summary|.* passed|.* failed)",
                          re.M)


def failure_tails(log: str, limit: int = 1200) -> dict[str, str]:
    """Per-test error tails, keyed by test function name.

    The Router hands these back to the agent, and the difference is measured: on
    the same 33 repositories, a kickback carrying only the command rescued 6-7,
    while one carrying the specific failing test ids rescued 10. A single tail per
    failing criterion is what makes "a missing optional dependency" actionable
    instead of "something went wrong".
    """
    out: dict[str, str] = {}
    matches = list(_FAIL_BANNER.finditer(log))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(log)
        body = log[start:end]
        if (stop := _SECTION_END.search(body)):
            body = body[:stop.start()]
        name = m.group(1).strip()
        # A parametrised banner reads `test_x[param]`; keep it, it is the id.
        body = body.strip()
        if body:
            out[name] = body[-limit:]
    return out


def counts(statuses: dict[str, str]) -> dict[str, int]:
    c: dict[str, int] = {}
    for s in statuses.values():
        c[s] = c.get(s, 0) + 1
    return c
