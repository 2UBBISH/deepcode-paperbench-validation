"""The criterion model: what "it runs" means, written down before anything is configured.

Section 4.2 of ``docs/DESIGN-research-setup-agent.md``. One ``Criterion`` per rung
of the ladder, each independently frozen and hashed; a ``Ladder`` holds the family
plus the things shared across rungs (the repository, the goal, the asset list).

Two properties everything else depends on:

* **Total order on rungs.** The gates are monotone -- G2 not passing forbids G3 --
  so the rung has to be comparable, not just a label.
* **Canonical serialisation.** ``canonical_json`` is what gets hashed. Dict order,
  whitespace and unicode escaping must not perturb the hash, or "the criterion did
  not change" becomes unprovable and the freeze is decorative.

The dataclasses carry no behaviour beyond validation and (de)serialisation.
Rendering them into a pytest file is ``rsa.compiler.render``; judging a run
against them is ``rsa.adjudicator``. Keeping those apart is what lets the
Adjudicator stay deterministic while the Compiler is an LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Rung(str, Enum):
    """The ladder. Order is significance, not declaration convenience."""

    G0 = "G0"       # static: import graph intact, entry script parses          (seconds)
    G1 = "G1"       # assets present: data, weights, credentials, GPU, CUDA     (seconds)
    G2 = "G2"       # minimal-scale run: whole pipeline, artefact manifest full (minutes)
    G3 = "G3"       # short experiment: shrunk config, sanity assertions        (hours)
    G4P = "G4'"     # full-scale completion: still "finished + sane", not paper numbers (days)

    @property
    def index(self) -> int:
        return _RUNG_ORDER.index(self)

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Rung):
            return NotImplemented
        return self.index < other.index

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Rung):
            return NotImplemented
        return self.index <= other.index


_RUNG_ORDER = [Rung.G0, Rung.G1, Rung.G2, Rung.G3, Rung.G4P]

# The configuration loop -- kickback, the stall gate, three-point falsification --
# runs only here. G3/G4' are one-shot final verifications: hours-to-days each, so
# re-running them in a repair loop is not affordable and failing them does not
# automatically send the agent back to configuring.
LOOP_RUNGS = frozenset({Rung.G0, Rung.G1, Rung.G2})


def next_rung(r: Rung) -> Rung | None:
    i = r.index + 1
    return _RUNG_ORDER[i] if i < len(_RUNG_ORDER) else None


# G3 and G4' cost hours or days of somebody's GPU. They do not start because a
# planner decided the time was right.
NEEDS_USER_APPROVAL = frozenset({Rung.G3, Rung.G4P})


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------

# A metric's value is read from somewhere in the run's output. Two shapes cover
# what research scripts actually do; both are parsed here rather than at render
# time so a malformed source is rejected at compile time, not discovered as a
# mass FAIL hours later.
#
#   results.json:.test_acc          a JSON file and a dotted path into it
#   results.json:.folds[0].acc      indices allowed
#   stdout:/final acc: ([0-9.]+)/   a regex over the run's captured output,
#                                   group 1 is the value (last match wins)
_SOURCE_STDOUT = re.compile(r"^stdout:/(?P<pat>.+)/$", re.DOTALL)
_JSONPATH_TOKEN = re.compile(r"\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)|\[(?P<idx>\d+)\]")


@dataclass(frozen=True)
class MetricSource:
    kind: str                       # "json" | "stdout"
    path: str = ""                  # json: file path relative to workdir
    jsonpath: tuple[Any, ...] = ()  # json: ("folds", 0, "acc")
    pattern: str = ""               # stdout: regex with one capturing group

    @staticmethod
    def parse(spec: str) -> "MetricSource":
        m = _SOURCE_STDOUT.match(spec.strip())
        if m:
            pat = m.group("pat")
            try:
                compiled = re.compile(pat)
            except re.error as e:
                raise ValueError(f"metric source {spec!r}: bad regex ({e})") from None
            if compiled.groups != 1:
                raise ValueError(
                    f"metric source {spec!r}: need exactly one capturing group, got {compiled.groups}"
                )
            return MetricSource(kind="stdout", pattern=pat)

        path, sep, jp = spec.partition(":")
        if not sep or not jp.startswith("."):
            raise ValueError(
                f"metric source {spec!r}: expected '<file>:.<json.path>' or 'stdout:/<regex>/'"
            )
        tokens: list[Any] = []
        pos = 0
        while pos < len(jp):
            tm = _JSONPATH_TOKEN.match(jp, pos)
            if not tm:
                raise ValueError(f"metric source {spec!r}: cannot parse json path at {jp[pos:]!r}")
            tokens.append(tm.group("key") if tm.group("key") else int(tm.group("idx")))
            pos = tm.end()
        return MetricSource(kind="json", path=path.strip(), jsonpath=tuple(tokens))

    def render(self) -> str:
        if self.kind == "stdout":
            return f"stdout:/{self.pattern}/"
        return self.path + ":" + "".join(
            f".{t}" if isinstance(t, str) else f"[{t}]" for t in self.jsonpath
        )


# The comparison grammar. Deliberately tiny and closed: an assertion the compiler
# cannot express here is a signal that the goal needs a free-code criterion (which
# then has to survive the linter), not a reason to widen the grammar until
# anything parses.
_CMP = re.compile(r"^(?P<op>>=|<=|>|<|==|!=)\s*(?P<val>-?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)$")
_WITHIN = re.compile(
    r"^within\s+(?P<tol>[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+of\s+"
    r"(?P<ref>-?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)$"
)
_IN_RANGE = re.compile(
    r"^in\s*\[\s*(?P<lo>-?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"(?P<hi>-?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\]$"
)


@dataclass(frozen=True)
class Comparison:
    kind: str            # "cmp" | "within" | "in_range"
    op: str = ""
    value: float = 0.0
    tol: float = 0.0
    lo: float = 0.0
    hi: float = 0.0

    @staticmethod
    def parse(expr: str) -> "Comparison":
        e = " ".join(expr.strip().split())
        if (m := _CMP.match(e)):
            return Comparison(kind="cmp", op=m.group("op"), value=float(m.group("val")))
        if (m := _WITHIN.match(e)):
            return Comparison(kind="within", tol=float(m.group("tol")), value=float(m.group("ref")))
        if (m := _IN_RANGE.match(e)):
            lo, hi = float(m.group("lo")), float(m.group("hi"))
            if lo > hi:
                raise ValueError(f"assertion {expr!r}: empty range, lo > hi")
            return Comparison(kind="in_range", lo=lo, hi=hi)
        raise ValueError(
            f"assertion {expr!r} not understood; allowed: '>= 0.1', '< 2', '== 3', "
            "'within 0.5 of 92.3', 'in [0.85, 0.95]'"
        )

    def render(self) -> str:
        if self.kind == "cmp":
            return f"{self.op} {_num(self.value)}"
        if self.kind == "within":
            return f"within {_num(self.tol)} of {_num(self.value)}"
        return f"in [{_num(self.lo)}, {_num(self.hi)}]"

    def holds(self, x: float) -> bool:
        """Evaluate. Used by the renderer's embedded runtime and by tests."""
        if self.kind == "cmp":
            return {
                ">=": x >= self.value, "<=": x <= self.value,
                ">": x > self.value, "<": x < self.value,
                "==": x == self.value, "!=": x != self.value,
            }[self.op]
        if self.kind == "within":
            return abs(x - self.value) <= self.tol
        return self.lo <= x <= self.hi


def _num(v: float) -> str:
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)


# Every artefact check is a deterministic predicate over a path. `exists` alone is
# the one that lets a script "succeed" by touching an empty file and returning 0,
# so it is never the default the compiler is nudged toward -- see compiler/render.
ARTIFACT_CHECKS = ("exists", "nonempty", "hash", "schema", "loadable")

# Loaders for `loadable`. A checkpoint that exists but cannot be deserialised is
# the single most common way a run looks finished and is not.
LOADERS = ("json", "torch", "numpy", "pickle", "csv", "image", "yaml")


@dataclass(frozen=True)
class Artifact:
    path: str
    check: str = "nonempty"
    sha256: str = ""            # check == "hash"
    loader: str = ""            # check == "loadable"
    schema_keys: tuple[str, ...] = ()   # check == "schema": keys that must be present and non-empty
    # Which stage of the pipeline this artefact proves ran. Free text from static
    # extraction ("plot_results", "summarize"). Not judged -- it is what makes a
    # missing artefact legible as "the shrink config skipped a stage" (section 4.5).
    stage: str = ""

    def __post_init__(self) -> None:
        if self.check not in ARTIFACT_CHECKS:
            raise ValueError(f"artifact {self.path!r}: unknown check {self.check!r}")
        if self.check == "hash" and not self.sha256:
            raise ValueError(f"artifact {self.path!r}: check='hash' needs sha256")
        if self.check == "loadable":
            if self.loader not in LOADERS:
                raise ValueError(
                    f"artifact {self.path!r}: check='loadable' needs loader in {LOADERS}"
                )
        if self.check == "schema" and not self.schema_keys:
            raise ValueError(f"artifact {self.path!r}: check='schema' needs schema_keys")


# Spellings for "extract this number, but do not assert a bound on it". The model
# reaches for these once it is told not to invent thresholds it cannot justify,
# and it is right to: the assertion that belongs on an unknown-scale metric is a
# finiteness or direction check from the sanity set, not a guessed inequality.
FINITE_ALIASES = frozenset({
    "is_finite", "finite", "isfinite", "not nan", "not_nan", "no_nan",
    "no nan", "any", "none", "", "-",
})


@dataclass(frozen=True)
class Metric:
    name: str
    source: MetricSource
    # None means the value is extracted and named, but carries no bound of its
    # own. It still renders nothing on its own; the sanity checks that reference
    # it are what assert.
    comparison: Comparison | None = None

    @staticmethod
    def build(name: str, source: str, assertion: str) -> "Metric":
        a = " ".join((assertion or "").strip().lower().split())
        cmp_ = None if a in FINITE_ALIASES else Comparison.parse(assertion)
        return Metric(name=name, source=MetricSource.parse(source), comparison=cmp_)


# Section 4.4. "Sane" has to be a predicate, not an impression, or the LLM verdict
# walks back in through this door. Each kind is a closed, deterministic check.
SANITY_KINDS = (
    "no_nan_inf",             # metric-valued outputs are finite
    "above_trivial_baseline", # acc > 1/num_classes, or R^2 > 0
    "improves",               # loss down / metric up from first step to last
    "not_constant",           # predictions take more than one distinct value
    "reproducible",           # same seed, same key numbers on a second run
)


@dataclass(frozen=True)
class Sanity:
    kind: str
    # Interpretation is per-kind; validated below so a malformed check cannot
    # silently render into a test that always passes.
    metric: str = ""            # name of a Metric declared on the same criterion
    source: MetricSource | None = None
    num_classes: int = 0        # above_trivial_baseline, classification
    baseline: float | None = None   # above_trivial_baseline, explicit override
    direction: str = ""         # improves: "down" (loss) | "up" (score)
    min_rel_change: float = 0.0 # improves: required relative movement, e.g. 0.01
    tol: float = 0.0            # reproducible: allowed absolute drift

    def __post_init__(self) -> None:
        if self.kind not in SANITY_KINDS:
            raise ValueError(f"unknown sanity kind {self.kind!r}; allowed: {SANITY_KINDS}")
        if self.kind == "above_trivial_baseline" and not self.num_classes and self.baseline is None:
            raise ValueError("above_trivial_baseline needs num_classes or an explicit baseline")
        if self.kind == "improves" and self.direction not in ("up", "down"):
            raise ValueError("improves needs direction='up' or 'down'")
        if self.kind in ("no_nan_inf", "above_trivial_baseline", "improves", "not_constant") \
                and not self.metric and self.source is None:
            raise ValueError(f"sanity {self.kind!r} needs a metric name or a source")


# --------------------------------------------------------------------------
# Shrinking, budget, assets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Shrink:
    """Scale knobs only. Never stage knobs -- section 4.5's iron rule.

    ``flatten`` is the half that is easy to forget and fatal to omit: lowering
    ``max_steps`` does not reach a stage guarded by ``if step % eval_every == 0``.
    The gates have to be pressed flat as well, or the shrunk run systematically
    skips the tail of the script -- which is exactly where the fragile optional
    dependencies live (matplotlib backends, openpyxl, wandb, onnx).
    """

    knobs: dict[str, Any] = field(default_factory=dict)     # max_steps, batch_size, subset, ...
    flatten: dict[str, Any] = field(default_factory=dict)   # eval_every=1, save_every=1, ...

    def as_cli(self, command: str = "") -> list[str]:
        """Render the knobs the way the frozen command spells its own flags.

        argparse accepts ``--max-steps 2`` for a ``max_steps`` option, so that
        was the only spelling.  nanoGPT's ``configurator.py`` (and every other
        hand-rolled ``--key=value`` parser) rejects both the dash and the
        space: the adjudication failed on every round with ``Overriding``
        errors while the environment was fine, and the setup agent burned its
        budget proving the criterion's command shape was at fault.  When the
        frozen command already carries long flags, copy their spelling:
        ``=`` if they use it, underscores if they keep them.
        """
        equals, underscores = _flag_style(command)
        out: list[str] = []
        for k, v in sorted({**self.knobs, **self.flatten}.items()):
            name = k if underscores else k.replace("_", "-")
            if equals:
                out.append(f"--{name}={v}")
            else:
                out += [f"--{name}", str(v)]
        return out


_LONG_FLAG = re.compile(r"(?<!\S)--([A-Za-z][\w-]*)(=)?")


def _flag_style(command: str) -> tuple[bool, bool]:
    """(uses ``--k=v``, keeps underscores) as observed on the frozen command.

    Defaults to argparse's conventional ``--dash-case value`` when the command
    carries no long flags at all.
    """
    flags = _LONG_FLAG.findall(command or "")
    if not flags:
        return False, False
    equals = any(eq for _, eq in flags)
    dashed = any("-" in name for name, _ in flags)
    underscored = any("_" in name for name, _ in flags)
    if dashed and not underscored:
        return equals, False
    if underscored:
        return equals, True
    # Single-word flags only.  ``--key=value`` parsers are hand-rolled over
    # module globals (underscores by construction); ``--key value`` is
    # argparse, whose dests were derived from dash-case flags.
    return equals, equals


@dataclass(frozen=True)
class Budget:
    """Part of the criterion, not a runtime preference.

    A budget agreed after the fact is not a budget; it is a post-hoc excuse. The
    third terminal state (budget exhausted) is only meaningful because the number
    was frozen with everything else -- 4 of 33 measured repositories ended on the
    wall clock, and folding those into "escalated" is what destroys the
    signal-to-noise of escalation.
    """

    wall_seconds: int = 3600
    tokens: int = 2_000_000
    gpu_hours: float = 0.0


@dataclass(frozen=True)
class Asset:
    """One row of the G1 checklist. Every row is a deterministic check."""

    kind: str          # "path" | "file_hash" | "env_var" | "gpu" | "cuda" | "command"
    name: str
    path: str = ""
    sha256: str = ""
    env_var: str = ""
    min_vram_gb: float = 0.0
    gpu_name_contains: str = ""
    cuda_min: str = ""
    cuda_max: str = ""
    command: str = ""
    # Which pipeline stage needs it. Shrinking the scale does not shrink the asset
    # requirements: the tail of the script still wants the wandb key and the font.
    stage: str = ""
    why: str = ""

    def __post_init__(self) -> None:
        allowed = ("path", "file_hash", "env_var", "gpu", "cuda", "command")
        if self.kind not in allowed:
            raise ValueError(f"asset {self.name!r}: unknown kind {self.kind!r}; allowed {allowed}")


# --------------------------------------------------------------------------
# Criterion and Ladder
# --------------------------------------------------------------------------

@dataclass
class Criterion:
    rung: Rung
    repo_url: str
    commit: str

    command: str = ""
    workdir: str = "/workspace/repo"
    env: dict[str, str] = field(default_factory=dict)
    seed: int = 1234
    timeout: int = 1800

    budget: Budget = field(default_factory=Budget)
    shrink: Shrink | None = None
    shrink_tier2: Shrink | None = None      # section 4.5's discriminator

    artifacts: list[Artifact] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    sanity: list[Sanity] = field(default_factory=list)
    repeats: int = 1

    # A free-code criterion, used when the goal does not fit the schema above. It
    # must survive rsa.compiler.linter before it can be frozen.
    free_pytest: str = ""

    # Filled by the Freezer. Empty until then, and the Adjudicator refuses to run
    # a criterion whose expectations were never measured -- an empty `expected`
    # set is satisfied by every possible run.
    test_file_sha256: str = ""
    expected: list[str] = field(default_factory=list)

    # Provenance, for the escalation card and for post-hoc audit.
    goal: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.rung, str):
            self.rung = Rung(self.rung)
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if any(s.kind == "reproducible" for s in self.sanity) and self.repeats < 2:
            raise ValueError("a 'reproducible' sanity check needs repeats >= 2")
        if self.free_pytest and (self.artifacts or self.metrics or self.sanity):
            raise ValueError(
                "a criterion is either schema-rendered or free code, not both: "
                "mixing them makes the linter's guarantee unenforceable"
            )
        names = [m.name for m in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate metric names: {names}")
        known = set(names)
        for s in self.sanity:
            if s.metric and s.metric not in known:
                raise ValueError(f"sanity check references unknown metric {s.metric!r}")

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rung"] = self.rung.value
        d["metrics"] = [
            {"name": m.name, "source": m.source.render(),
             "assert": m.comparison.render() if m.comparison else ""}
            for m in self.metrics
        ]
        d["sanity"] = [_sanity_to_dict(s) for s in self.sanity]
        for key in ("shrink", "shrink_tier2"):
            if d[key] is None:
                d.pop(key)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Criterion":
        d = dict(d)
        d["rung"] = Rung(d["rung"])
        d["budget"] = Budget(**d["budget"]) if isinstance(d.get("budget"), dict) else Budget()
        for key in ("shrink", "shrink_tier2"):
            v = d.get(key)
            d[key] = Shrink(**v) if isinstance(v, dict) else None
        d["artifacts"] = [
            Artifact(**{**a, "schema_keys": tuple(a.get("schema_keys", ()))})
            for a in d.get("artifacts", [])
        ]
        d["metrics"] = [
            Metric.build(m["name"], m["source"], m["assert"]) for m in d.get("metrics", [])
        ]
        d["sanity"] = [_sanity_from_dict(s) for s in d.get("sanity", [])]
        return Criterion(**d)

    def canonical_json(self) -> str:
        """Byte-stable serialisation. This, and only this, is what gets hashed."""
        return canonical_json(self.to_dict())

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    # -- convenience -------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        return bool(self.test_file_sha256 and self.expected)

    def with_shrink(self, tier: int) -> "Criterion":
        """Tier-2 view for the shrink-artefact discriminator (section 4.5).

        Only the shrink knobs move. The command, the assertions and `expected` are
        the same objects, because the point of the second tier is to ask whether
        the *same* criterion passes at a slightly larger scale -- a tier that also
        relaxed the assertions would answer a different question.
        """
        if tier == 1 or self.shrink_tier2 is None:
            return self
        import copy
        c = copy.deepcopy(self)
        c.shrink = self.shrink_tier2
        return c


def _sanity_to_dict(s: Sanity) -> dict:
    d = asdict(s)
    d["source"] = s.source.render() if s.source else ""
    return {k: v for k, v in d.items() if v not in ("", 0, 0.0, None) or k == "kind"}


def _sanity_from_dict(d: dict) -> Sanity:
    d = dict(d)
    src = d.pop("source", "")
    return Sanity(source=MetricSource.parse(src) if src else None, **d)


@dataclass
class Ladder:
    """The criterion family for one (repository, goal) pair."""

    repo_url: str
    commit: str
    goal: str
    rungs: dict[Rung, Criterion] = field(default_factory=dict)
    assets: list[Asset] = field(default_factory=list)
    # The rung the user has approved the system to attempt. G0-G2 are automatic;
    # anything above is opt-in and stays here until the user says otherwise.
    approved_through: Rung = Rung.G2
    # One clarification round is allowed at compile time (section 5.1). Recorded
    # so a later reader can tell an assumed goal from a confirmed one.
    clarification: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.approved_through, str):
            self.approved_through = Rung(self.approved_through)
        self.rungs = {Rung(k): v for k, v in self.rungs.items()}

    def ordered(self) -> list[Criterion]:
        return [self.rungs[r] for r in _RUNG_ORDER if r in self.rungs]

    def attemptable(self) -> list[Criterion]:
        return [c for c in self.ordered() if c.rung <= self.approved_through]

    def to_dict(self) -> dict:
        return {
            "repo_url": self.repo_url,
            "commit": self.commit,
            "goal": self.goal,
            "approved_through": self.approved_through.value,
            "clarification": self.clarification,
            "assets": [asdict(a) for a in self.assets],
            "rungs": {r.value: c.to_dict() for r, c in sorted(self.rungs.items(),
                                                              key=lambda kv: kv[0].index)},
        }

    @staticmethod
    def from_dict(d: dict) -> "Ladder":
        return Ladder(
            repo_url=d["repo_url"],
            commit=d["commit"],
            goal=d.get("goal", ""),
            approved_through=Rung(d.get("approved_through", "G2")),
            clarification=d.get("clarification", {}),
            assets=[Asset(**a) for a in d.get("assets", [])],
            rungs={Rung(k): Criterion.from_dict(v) for k, v in d.get("rungs", {}).items()},
        )

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def canonical_json(obj: Any) -> str:
    """Stable across dict insertion order, whitespace and non-ASCII content."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
