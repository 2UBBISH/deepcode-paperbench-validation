"""Gate for free-code criteria.

Most goals fit `{command, artifacts, metrics, sanity}` and are rendered from the
schema, where every construct is a predicate some real environment fails. A goal
that genuinely does not fit may be written as free pytest -- but then the one
protection the schema provided for free is gone, and it has to be reimposed here.

The thing being guarded against is not incompetence, it is the path of least
resistance. Asked to "write a pytest that checks the environment", the cheapest
correct-looking answer is `def test_import(): import foo`. It fails in a bare
container, so it survives point (1) of falsification, and it proves almost
nothing: measured on the SWE-smith 128, "imports and starts" saturated at 97.7%
while "the expected set all passes" sat at 73.4%. The 24-point gap is entirely
criteria of that shape.

So: no assertion that cannot fail, no swallowing of errors, no skipping, and at
least one assertion that actually looks at an artefact or a number.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# Names from the prelude that constitute a real observation of the run. A file
# whose assertions never reach one of these is not checking the environment.
PROBE_NAMES = frozenset({
    "exists", "size", "sha256", "read_text", "read_json", "loads_ok",
    "metric_from_json", "metric_from_stdout", "series", "series_from_stdout",
    "distinct_values", "target", "target_runs", "sh",
})

# Skipping is banned outright, in every spelling. A skipped test is not a passing
# test -- the Adjudicator already refuses to count SKIP -- but a criteria file
# that skips itself produces a smaller expected set at freeze time, which is the
# same weakening arriving one step earlier.
SKIP_CALLS = frozenset({"skip", "importorskip", "xfail", "exit", "fail"})
SKIP_MARKS = frozenset({"skip", "skipif", "xfail"})

ALLOWED_IMPORTS = frozenset({
    # The prelude is spliced in, so the file needs nothing else. These are the
    # standard-library names a criterion might reasonably reach for.
    "json", "os", "re", "math", "pathlib", "subprocess", "hashlib", "shlex",
    "csv", "itertools", "collections", "statistics", "datetime", "time",
    "tempfile", "string", "textwrap", "__future__", "dataclasses", "typing",
    "pytest",
})


@dataclass
class Finding:
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: [{self.rule}] {self.message}"


class LintError(ValueError):
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__(
            "free-code criterion rejected:\n  " + "\n  ".join(str(f) for f in findings)
        )


def lint(source: str) -> list[Finding]:
    """Return every rule violation. Empty means the criterion may be frozen."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [Finding(e.lineno or 0, "syntax", f"does not parse: {e.msg}")]

    f: list[Finding] = []
    tests: list[ast.FunctionDef] = []
    probes_used = False

    for node in ast.walk(tree):
        # -- assertions that cannot fail ---------------------------------
        if isinstance(node, ast.Assert):
            t = node.test
            if isinstance(t, ast.Constant) and t.value:
                f.append(Finding(node.lineno, "vacuous-assert",
                                 f"`assert {t.value!r}` can never fail"))
            elif isinstance(t, (ast.Tuple, ast.List)) and t.elts:
                f.append(Finding(node.lineno, "vacuous-assert",
                                 "asserting a non-empty tuple is always true "
                                 "(a stray comma turns any assertion into this)"))
            elif isinstance(t, (ast.Lambda, ast.FunctionDef)):
                f.append(Finding(node.lineno, "vacuous-assert",
                                 "asserting a function object is always true"))

        # -- swallowed failures ------------------------------------------
        if isinstance(node, ast.ExceptHandler):
            # Independent rules, so `except: pass` reports both. It violates both.
            if node.type is None:
                f.append(Finding(node.lineno, "bare-except",
                                 "a bare `except:` turns a broken environment into a pass"))
            if all(isinstance(s, (ast.Pass, ast.Continue)) for s in node.body):
                f.append(Finding(node.lineno, "swallowed-except",
                                 "`except ...: pass` hides exactly what is being tested"))

        # -- skipping -----------------------------------------------------
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else \
                   getattr(node.func, "id", "")
            root = _root_name(node.func)
            if name in SKIP_CALLS and root in ("pytest", "unittest", ""):
                if name in ("skip", "importorskip", "xfail"):
                    f.append(Finding(node.lineno, "skip",
                                     f"`{name}` removes the test from the expected set; "
                                     "a criterion that can excuse itself is not a criterion"))
            if name in PROBE_NAMES:
                probes_used = True

        # -- skip markers --------------------------------------------------
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                tests.append(node)
            for dec in node.decorator_list:
                mark = _mark_name(dec)
                if mark in SKIP_MARKS:
                    f.append(Finding(node.lineno, "skip-marker",
                                     f"@pytest.mark.{mark} is not allowed on a criterion"))

        # -- imports --------------------------------------------------------
        if isinstance(node, ast.Import):
            for a in node.names:
                mod = a.name.split(".")[0]
                if mod not in ALLOWED_IMPORTS:
                    f.append(Finding(node.lineno, "import",
                                     f"criteria may not import {mod!r}: a criterion that "
                                     "pulls in its own dependency no longer measures the "
                                     "environment under test"))
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mod = node.module.split(".")[0]
            if mod not in ALLOWED_IMPORTS:
                f.append(Finding(node.lineno, "import",
                                 f"criteria may not import {mod!r}"))

    if not tests:
        f.append(Finding(0, "no-tests", "no `test_*` function: nothing would be collected"))

    if not probes_used:
        f.append(Finding(
            0, "no-observation",
            "no assertion reaches an artefact or a number. At least one test must call "
            f"one of {sorted(PROBE_NAMES)[:6]}... -- an exit-code check alone is the "
            "criterion shape that measured 97.7% while the real pass rate was 73.4%"))

    empty = [t for t in tests if _is_trivial(t)]
    for t in empty:
        f.append(Finding(t.lineno, "empty-test",
                         f"`{t.name}` contains no assertion"))

    return sorted(f, key=lambda x: (x.line, x.rule))


def check(source: str) -> None:
    """Raise `LintError` if the criterion may not be frozen."""
    findings = lint(source)
    if findings:
        raise LintError(findings)


def _root_name(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return getattr(node, "id", "")


def _mark_name(dec: ast.AST) -> str:
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Attribute) and _root_name(dec) == "pytest":
        return dec.attr
    return ""


def _is_trivial(fn: ast.FunctionDef) -> bool:
    for n in ast.walk(fn):
        if isinstance(n, ast.Assert):
            return False
        # `pytest.raises`, `unittest`-style self.assertX, and an explicit
        # `raise` all count as real checks.
        if isinstance(n, ast.Raise):
            return False
        if isinstance(n, ast.Call):
            name = n.func.attr if isinstance(n.func, ast.Attribute) else \
                   getattr(n.func, "id", "")
            if name.startswith("assert") or name == "raises":
                return False
    return True
