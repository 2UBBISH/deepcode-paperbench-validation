"""Render a `Criterion` into the pytest file that will be frozen and run.

Schema first, free code only as a fallback. The measured failure mode is not that
the model cannot write a criterion, it is that it writes a *weak* one: asked to
"write a pytest that checks the environment", the easiest thing to produce is
``def test_import(): import foo``, which fails in a bare container and therefore
survives point (1) of falsification while proving almost nothing. On the SWE-smith
128 the gap between "imports and starts" (97.7%) and "the expected set all passes"
(73.4%) was 24 points, and all of it is that kind of criterion.

Free code has unboundedly many ways to be weak (``assert True``, a bare
``except``, asserting only on the exit code). A schema has none: every construct
here is a predicate that some real environment fails. So the compiler is pushed
to express goals as ``{command, artifacts, metrics, sanity}``, and anything that
genuinely will not fit goes through `rsa.linter` before it may be frozen.

Test ids are the criterion's `expected` set, so their spelling is load-bearing:
they are generated from a deterministic slug and the Adjudicator always invokes
pytest from the criteria directory with a bare filename, which keeps ids of the
form ``criteria_G2.py::test_metric_acc`` regardless of where the store lives.
"""

from __future__ import annotations

import re
from pathlib import Path

from .criterion import Artifact, Comparison, Criterion, Metric, MetricSource, Sanity

_PRELUDE_PATH = Path(__file__).resolve().parent / "prelude.py"


def prelude_source() -> str:
    return _PRELUDE_PATH.read_text(encoding="utf-8")


def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return (s[:limit] or "x").rstrip("_")


def snapshot_paths(c: Criterion) -> list[str]:
    """Result files worth copying aside after each repeat (see prelude._snapshot)."""
    paths: set[str] = set()
    for m in c.metrics:
        if m.source.kind == "json":
            paths.add(m.source.path)
    for s in c.sanity:
        if s.source and s.source.kind == "json":
            paths.add(s.source.path)
    return sorted(paths)


# --------------------------------------------------------------------------
# Fragment renderers
# --------------------------------------------------------------------------

def _value_expr(src: MetricSource, run_index: str = "0") -> str:
    if src.kind == "stdout":
        return f"metric_from_stdout({src.pattern!r}, run_index={run_index})"
    return f"metric_from_json({src.path!r}, {list(src.jsonpath)!r}, run_index={run_index})"


def _series_expr(src: MetricSource, run_index: str = "0") -> str:
    if src.kind == "stdout":
        return f"series_from_stdout({src.pattern!r}, run_index={run_index})"
    return f"series({src.path!r}, {list(src.jsonpath)!r}, run_index={run_index})"


def _comparison_expr(cmp: Comparison, var: str = "v") -> tuple[str, str]:
    """(python expression, human-readable requirement)."""
    if cmp.kind == "cmp":
        return f"{var} {cmp.op} {cmp.value!r}", f"{cmp.op} {cmp.value!r}"
    if cmp.kind == "within":
        return (f"abs({var} - {cmp.value!r}) <= {cmp.tol!r}",
                f"within {cmp.tol!r} of {cmp.value!r}")
    return (f"{cmp.lo!r} <= {var} <= {cmp.hi!r}", f"in [{cmp.lo!r}, {cmp.hi!r}]")


def _render_run_test(c: Criterion) -> tuple[str, str]:
    name = "test_run_completes"
    body = f'''
def {name}():
    """The frozen command runs to completion.

    Necessary, never sufficient: a run can exit 0 having skipped every stage that
    would have failed. The artefact and metric tests below are what make this
    more than an exit-code check.
    """
    runs = target_runs()
    for i, r in enumerate(runs):
        assert r["exit_code"] == 0, (
            f"run {{i}} exited {{r['exit_code']}} (timeout=124); last 3000 chars:\\n"
            + r["output"][-3000:]
        )
'''
    return name, body


def _render_artifact(a: Artifact, i: int) -> tuple[str, str]:
    name = f"test_artifact_{i:02d}_{_slug(a.path)}"
    stage = f"  # stage: {a.stage}" if a.stage else ""
    # The path lands inside generated f-string *messages*, so a brace in it is a
    # format slot in the file we emit. An unresolved `{}` from static analysis
    # produced `f"{}: size={n}"`, which does not parse -- and pytest could then
    # not collect the criteria file at all. `resolve_sinks` now drops such paths;
    # this makes the renderer safe regardless of where a path came from.
    p = a.path.replace("{", "{{").replace("}", "}}")
    head = f'''
def {name}():
    """{a.check} :: {p}{stage}

    A missing artefact means the stage that writes it never ran. Section 4.5:
    that is a shrink configuration error, routed back to the Compiler, not a
    setup failure routed back to the agent.
    """
    target_runs()
'''
    if a.check == "exists":
        body = f'    assert exists({a.path!r}), "{p} was never created"\n'
    elif a.check == "nonempty":
        body = (
            f'    n = size({a.path!r})\n'
            f'    assert n > 0, f"{p}: size={{n}} (-1 means absent; 0 means the '
            f'stage opened the file and wrote nothing)"\n'
        )
    elif a.check == "hash":
        body = (
            f'    got = sha256({a.path!r})\n'
            f'    assert got == {a.sha256!r}, f"{p}: sha256 {{got or "<absent>"}}"\n'
        )
    elif a.check == "schema":
        body = (
            f'    obj = read_json({a.path!r})\n'
            f'    missing = [k for k in {list(a.schema_keys)!r} if k not in obj]\n'
            f'    assert not missing, f"{p}: missing keys {{missing}}"\n'
            f'    empty = [k for k in {list(a.schema_keys)!r} '
            f'if obj[k] in (None, "", [], {{}})]\n'
            f'    assert not empty, f"{p}: keys present but empty {{empty}}"\n'
        )
    else:  # loadable
        body = (
            f'    ok, log = loads_ok({a.path!r}, {a.loader!r})\n'
            f'    assert ok, f"{p} exists but {a.loader} cannot deserialise it:\\n{{log}}"\n'
        )
    return name, head + body


def _render_metric(m: Metric) -> tuple[str, str] | None:
    if m.comparison is None:
        # Nothing to assert here. Rendering a test anyway would produce exactly
        # the vacuous assertion the linter exists to forbid.
        return None
    name = f"test_metric_{_slug(m.name)}"
    expr, human = _comparison_expr(m.comparison)
    body = f'''
def {name}():
    """{m.name} {human}  (from {m.source.render()})"""
    target_runs()
    v = {_value_expr(m.source)}
    assert {expr}, f"{m.name}={{v!r}}, required {human}"
'''
    return name, body


def _render_sanity(s: Sanity, i: int, by_name: dict[str, Metric],
                   repeats: int) -> tuple[str, str]:
    src = s.source or (by_name[s.metric].source if s.metric in by_name else None)
    if src is None:
        raise ValueError(f"sanity check {s.kind!r} has neither a source nor a known metric")
    label = s.metric or src.render()
    name = f"test_sanity_{i:02d}_{s.kind}_{_slug(label)}"

    if s.kind == "no_nan_inf":
        body = f'''
def {name}():
    """{label} is finite -- catches a run that diverged and still finished."""
    target_runs()
    v = {_value_expr(src)}
    assert is_finite(v), f"{label}={{v!r}} is NaN or infinite: training diverged"
'''
    elif s.kind == "above_trivial_baseline":
        base = s.baseline if s.baseline is not None else 1.0 / s.num_classes
        why = (f"an explicit baseline" if s.baseline is not None
               else f"chance on {s.num_classes} classes")
        body = f'''
def {name}():
    """{label} beats the trivial baseline ({base!r}, {why}).

    Passing this does not mean the result is good. Failing it means the model did
    not learn at all, which no amount of environment configuration will fix and
    which a "did it finish" criterion would have reported as success.
    """
    target_runs()
    v = {_value_expr(src)}
    assert v > {base!r}, f"{label}={{v!r}} does not beat the trivial baseline {base!r}"
'''
    elif s.kind == "improves":
        mrc = s.min_rel_change
        if s.direction == "down":
            cmp_expr = (f"s[-1] < s[0]" if mrc == 0
                        else f"s[-1] <= s[0] - abs(s[0]) * {mrc!r}")
            want = f"decrease{'' if mrc == 0 else f' by at least {mrc:.1%} of its first value'}"
        else:
            cmp_expr = (f"s[-1] > s[0]" if mrc == 0
                        else f"s[-1] >= s[0] + abs(s[0]) * {mrc!r}")
            want = f"increase{'' if mrc == 0 else f' by at least {mrc:.1%} of its first value'}"
        body = f'''
def {name}():
    """{label} must {want} across the run.

    Catches lr=0, a detached graph, a frozen model: the pipeline completes, every
    artefact appears, and nothing was learned.
    """
    target_runs()
    s = {_series_expr(src)}
    assert len(s) >= 2, f"{label}: need at least two points to see a trend, got {{len(s)}}"
    assert {cmp_expr}, f"{label} went {{s[0]!r}} -> {{s[-1]!r}}; required: {want}"
'''
    elif s.kind == "not_constant":
        if src.kind != "json":
            raise ValueError("not_constant needs a json source pointing at a prediction array")
        body = f'''
def {name}():
    """Predictions take more than one value.

    A model that answers the majority class for every input can score well on an
    imbalanced split; this is the check that says so.
    """
    target_runs()
    n = distinct_values({src.path!r}, {list(src.jsonpath)!r})
    assert n > 1, f"all predictions are the same value ({{n}} distinct)"
'''
    else:  # reproducible
        body = f'''
def {name}():
    """The same seed gives the same number twice (tolerance {s.tol!r}).

    Without it, "the result looks sane" can be luck. Each repeat's result files
    are snapshotted separately, so this compares two runs rather than one file
    against itself.
    """
    target_runs()
    vals = [{_value_expr(src, run_index="i")} for i in range({repeats})]
    assert len(vals) >= 2, "reproducibility needs repeats >= 2"
    spread = max(vals) - min(vals)
    assert spread <= {s.tol!r}, f"{label} varied by {{spread!r}} across repeats: {{vals!r}}"
'''
    return name, body


# --------------------------------------------------------------------------
# Whole-file rendering
# --------------------------------------------------------------------------

def render(c: Criterion) -> tuple[str, list[str]]:
    """Return (pytest source, ordered test function names).

    The names are returned rather than re-derived later so that `expected` and the
    file can never disagree about what was generated. The Freezer still confirms
    them against a real `pytest --collect-only`, because a name this function
    believes it emitted and a name pytest actually collects are different claims.
    """
    header = [
        "# -*- coding: utf-8 -*-",
        "# GENERATED CRITERIA FILE -- FROZEN. Do not edit.",
        f"# rung:    {c.rung.value}",
        f"# repo:    {c.repo_url}",
        f"# commit:  {c.commit}",
        f"# goal:    {c.goal}",
        f"# command: {c.command}",
        "#",
        "# Runs on the host, observes the environment under test through the bridge",
        "# in the prelude below. Nothing here is installed into that environment.",
        "",
    ]

    if c.free_pytest:
        # Already linted by rsa.linter at compile time; frozen verbatim so the
        # hash covers exactly what a human reviewed.
        body = c.free_pytest
        source = "\n".join(header) + prelude_source() + "\n\n# ---- free-code criterion ----\n\n" + body
        names = sorted(set(re.findall(r"^def (test_\w+)\s*\(", body, re.M)))
        return source, names

    fragments: list[str] = []
    names: list[str] = []

    for renderer in (_render_run_test(c),):
        n, frag = renderer
        names.append(n)
        fragments.append(frag)

    for i, a in enumerate(c.artifacts):
        n, frag = _render_artifact(a, i)
        names.append(n)
        fragments.append(frag)

    by_name = {m.name: m for m in c.metrics}
    for m in c.metrics:
        rendered = _render_metric(m)
        if rendered is None:
            continue
        n, frag = rendered
        names.append(n)
        fragments.append(frag)

    for i, s in enumerate(c.sanity):
        n, frag = _render_sanity(s, i, by_name, c.repeats)
        names.append(n)
        fragments.append(frag)

    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(
            f"criterion renders colliding test ids {sorted(dupes)}; ids are the "
            "expected set, so a collision would silently shrink it"
        )

    source = "\n".join(header) + prelude_source() + "\n\n# ---- generated assertions ----\n" + "".join(fragments)
    return source, names


def criteria_filename(c: Criterion) -> str:
    """Stable basename. pytest ids embed it, so it must not depend on the store path."""
    return f"criteria_{c.rung.name}.py"
