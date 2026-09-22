"""Goal + repository -> a criterion ladder.

The one component where a model decides anything, and its authority is bounded on
both sides. Before it: static extraction has already produced the artefact
manifest, the gating knobs and the scale knobs, and those are **authoritative** --
the model may enrich an entry (say which loader proves a checkpoint deserialises)
but may not remove one. After it: the Falsifier attacks whatever comes out, and
the Freezer refuses anything the linter rejects.

That split follows the project's own rule -- anything that can be a deterministic
check must not be left to a model. It also fixes a specific failure: a model
writing the artefact manifest from the same reading of the script that produced
the shrink configuration will omit the same stage twice, and the manifest exists
precisely to catch a stage the shrink skipped.

The model's real job is the part that genuinely requires reading English: which
command corresponds to the user's goal, what the numbers in the result file mean,
and what "working" would look like for this particular experiment.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from ..criterion import (
    Artifact, Asset, Budget, Criterion, Ladder, Metric, MetricSource, Rung, Sanity, Shrink,
)
from ..static_facts import (DEFAULT_SAFE_LOADERS, RepoFacts, Sink, analyse,
                            printed_formats_for, resolve_sinks)
from .llm import LLM, LLMError

SYSTEM = """\
You compile a user's natural-language environment/setup request into a
machine-checkable pytest criterion.

You are NOT configuring the environment and NOT running the user's command. You
are writing down, in advance, what would count as proof that the requested
command or selected tests pass after setup. The criterion will be frozen and
used by an external deterministic adjudicator while SetupX configures the
environment.

Hard rules:

1. The criterion must FAIL in a container where nothing is installed, and must
   FAIL when only the top-level requirements are installed with no datasets,
   weights or optional extras. A criterion satisfied by an import alone is
   worthless. For an explicit pytest request, preserve the user's test paths and
   options exactly; a successful pytest exit is evidence that those tests pass.
2. Shrink the SCALE, never the PIPELINE. You may reduce steps, epochs, batch size
   and dataset fraction. You may not skip stages. Every stage of the script must
   still be reached at least once -- especially the last ones (writing results,
   exporting, plotting), because that is where optional dependencies fail.
3. Because of rule 2, every `*_every` / `*_freq` / `*_interval` knob must be set
   to 1 in `shrink.flatten`, or the stages they gate will never run.
4. For an explicit pytest request, pytest exit code 0 is the aggregate assertion:
   it means the selected tests passed, while "no tests collected" is non-zero.
   Do not invent output artefacts or numeric metrics for a test suite. For other
   commands, assert on files and values a broken environment gets wrong rather
   than only on an exit code.
5. Only claim a metric exists if you can point at where it is written. If the
   script prints it, use a stdout regex; if it writes a JSON file, use a path.
   Inventing a results file that the script never writes makes the criterion
   constantly false.
6. NEVER invent a numeric threshold you cannot justify. If you do not know the
   scale a metric lives on, do not guess one: assert only that it is finite
   (sanity `no_nan_inf`), that it beats chance when you know the number of
   classes, or that it moves in the right direction (sanity `improves`). A
   guessed threshold like "loss < 1" makes the criterion permanently
   unsatisfiable, and a permanently unsatisfiable criterion is indistinguishable
   from a broken environment -- the setup agent will burn its whole budget trying
   to install its way out of your arithmetic.
7. Do not copy the example paths out of the schema below. Every artefact path
   must come from the manifest above, resolved against the command YOU chose.

Reply with ONE JSON object and nothing else."""

SCHEMA_DOC = """\
{
  "needs_clarification": "",        // non-empty ONLY if the goal is too vague to
                                    // pick a command; you get exactly one round
  "command":      "python train.py --config configs/cifar10_small.yaml",
  "entrypoint":   "train.py",
  "timeout":      900,              // seconds for ONE shrunk run
  "shrink":       {"knobs": {"max_steps": 2, "batch_size": 1},
                   "flatten": {"eval_every": 1, "save_every": 1}},
  "shrink_tier2": {"knobs": {"max_steps": 8, "batch_size": 4},
                   "flatten": {"eval_every": 1, "save_every": 1}},
  "artifacts": [
    {"path": "results.json", "check": "schema",
     "schema_keys": ["test_acc"], "stage": "summarise"},
    {"path": "out/model.pt",  "check": "loadable", "loader": "torch", "stage": "save"}
  ],                                // check: exists|nonempty|hash|schema|loadable
                                    // loader: json|torch|numpy|pickle|csv|image|yaml
  "metrics": [
    {"name": "acc", "source": "results.json:.test_acc", "assert": ">= 0.10"},
    {"name": "loss", "source": "stdout:/loss=([0-9.]+)/", "assert": "< 10"}
  ],                                // assert: ">= x" | "< x" | "within t of r"
                                    //         | "in [lo, hi]"
  "sanity": [
    {"kind": "no_nan_inf", "metric": "acc"},
    {"kind": "above_trivial_baseline", "metric": "acc", "num_classes": 10},
    {"kind": "improves", "source": "results.json:.loss_curve",
     "direction": "down", "min_rel_change": 0.05},
    {"kind": "not_constant", "source": "results.json:.predictions"},
    {"kind": "reproducible", "metric": "acc", "tol": 0.0}
  ],
  "assets": [
    {"kind": "path", "name": "cifar10", "path": "data/cifar-10-batches-py",
     "stage": "load", "why": "the training set is not vendored"},
    {"kind": "env_var", "name": "wandb", "env_var": "WANDB_API_KEY",
     "stage": "log", "why": "the logging stage authenticates at import"},
    {"kind": "gpu", "name": "gpu", "min_vram_gb": 12, "stage": "train", "why": "..."}
  ],                                // kind: path|file_hash|env_var|gpu|cuda|command
  "notes": "anything the user should know about how this was chosen"
}"""


@dataclass
class CompileResult:
    ladder: Ladder | None
    facts: RepoFacts
    raw: dict = field(default_factory=dict)
    question: str = ""
    notes: str = ""
    disclosures: list[str] = field(default_factory=list)

    @property
    def needs_user(self) -> bool:
        return bool(self.question)


def needs_clarification(r: CompileResult) -> bool:
    return r.needs_user


# --------------------------------------------------------------------------

def extract_pytest_command(instruction: str) -> str:
    """Extract an explicit pytest command from the user's instruction.

    This is deliberately conservative: it only returns a command when the user
    wrote ``pytest`` (usually in backticks).  It prevents the compiler from
    changing the requested test selection while leaving vague requests such as
    "make the test suite work" to the normal clarification path.
    """
    text = str(instruction or "")
    candidates = re.findall(r"`([^`]*\bpytest\b[^`]*)`", text, flags=re.I)
    if not candidates:
        candidates = re.findall(r"(?m)^\s*((?:python(?:3(?:\.\d+)?)?\s+-m\s+)?pytest\b[^\n]*)",
                                text, flags=re.I)
    for candidate in candidates:
        command = candidate.strip().rstrip("。.,;，；")
        # Do not treat prose containing the word pytest as a shell command.
        if re.match(r"^(?:(?:python(?:3(?:\.\d+)?)?|uv\s+run)\s+-m\s+)?pytest\b",
                    command, flags=re.I):
            return command
    return ""


def infer_pytest_command(instruction: str, repo_path: str | Path) -> str:
    """Choose a safe default pytest selection for an unqualified test request.

    The default is only used when the user clearly asks for tests but does not
    name a command.  A named test file is kept as the selection; otherwise the
    repository's pytest discovery is used.  We do not inspect or execute tests
    here, so this remains a planning step and the frozen criterion still proves
    the result later.
    """
    explicit = extract_pytest_command(instruction)
    if explicit:
        return explicit
    text = str(instruction or "").lower()
    asks_for_tests = any(term in text for term in (
        "pytest", "test suite", "tests", "测试", "测试集", "测试套件",
    ))
    if not asks_for_tests:
        return ""
    mentioned = re.findall(
        r"(?<![\w./-])([\w./-]*(?:test_[\w.-]+|[\w.-]+_test)\.py)(?![\w./-])",
        instruction,
        flags=re.I,
    )
    paths: list[str] = []
    for p in mentioned:
        p = p.lstrip("./")
        if p not in paths and (Path(repo_path) / p).is_file():
            paths.append(p)
    return "python -m pytest " + " ".join(paths) + " -q" if paths else "python -m pytest -q"


def is_pytest_command(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for i, token in enumerate(tokens):
        name = token.rsplit("/", 1)[-1]
        if name == "pytest":
            return True
        if token == "-m" and i + 1 < len(tokens) and tokens[i + 1] == "pytest":
            return True
    return False


def _facts_prompt(facts: RepoFacts, goal: str, limit: int = 60,
                  pytest_command: str = "") -> str:
    eps = "\n".join(
        f"  {e.file}  (main-guard={e.has_main_guard})  options: "
        + (", ".join(f"{k}={v}" for k, v in sorted(e.options.items())[:25]) or "none")
        for e in facts.entrypoints[:8]
    )
    # Only the sinks a plausible command could own. The unfiltered manifest is
    # where the model copied python-control's twenty-five `control/tests/*.png`
    # into its own `artifacts` list -- it was reading them off this prompt.
    own = [s for s in facts.artifact_manifest() if not _is_foreign(s.file, set())]
    sinks = "\n".join(
        f"  {s.path:<40} {s.kind:<11} via {s.func:<14} in {s.stage}()  [{s.file}:{s.line}]"
        for s in own[:limit]
    )
    dyn = [s for s in facts.sinks if not s.literal][:12]
    dynamic = "\n".join(f"  {s.path}  (built at runtime, in {s.stage}())" for s in dyn)
    return f"""\
## Goal (verbatim from the user)

{goal}

## Extracted from the repository (static analysis -- these are facts, not guesses)

User command extracted from the instruction:
  {pytest_command or '(none; ask for clarification if the test selection is not clear)'}

When the extracted command is non-empty, preserve its pytest target paths and
options.  The setup agent must make that exact test selection pass; do not
replace it with the whole repository suite or with a different script.

Entry points, best first:
{eps or '  none found'}

Commands quoted in the README:
{chr(10).join('  ' + c for c in facts.readme_commands[:15]) or '  none found'}

Files the code writes (THE ARTEFACT MANIFEST -- every one of these must appear in
your `artifacts` list, because a missing one is how we detect a stage that never
ran; you decide the appropriate `check` and `loader` for each):
{sinks or '  none found'}

Write targets whose path is built at runtime (you may reference them if you can
work out the concrete path from a config; otherwise ignore them):
{dynamic or '  none'}

Gating knobs found in guards like `if step % X == 0` -- ALL of these must be set
to 1 in shrink.flatten or the stages they guard will not run:
  {', '.join(facts.gating_knobs) or 'none found'}

HARDCODED gates -- guards with a literal interval and NO command-line flag. You
cannot flatten these, so your step count must EXCEED the largest of them or the
stages they guard (often the checkpoint save and the evaluation) never run at all:
{chr(10).join(f"  every {g['interval']} {g['counter']}s, at {g['file']}:{g['line']} in {g['stage']}()" for g in facts.hardcoded_gates[:12]) or '  none found'}
  => minimum step count to reach every stage: {facts.min_steps_for_all_stages() or 'no constraint'}

Scale knobs available on the command line:
  {', '.join(facts.scale_knobs) or 'none found'}

What the script actually PRINTS (literal text, with `{{...}}` where a runtime
value is substituted, and the file that prints it in brackets). If you read a
metric from stdout, your regex must match one of these EXACTLY as written, and
it must be printed by the script your command runs -- do not invent a format,
and do not borrow one from another file. `loss=([0-9.]+)` will not match a line
that reads `loss 3.2345`:
{chr(10).join('  ' + f + '  [' + ', '.join(facts.printed_format_files.get(f, ())[:3]) + ']' for f in facts.printed_formats[:25]) or '  none found'}

Environment variables the code reads:
  {', '.join(facts.env_vars) or 'none'}

Paths that look like inputs (datasets, weights) rather than outputs:
{chr(10).join('  ' + p for p in facts.data_paths[:25]) or '  none'}

Imports that are optional (inside try/except or inside a function) -- these are
the ones a partial install misses, and they usually live in the reporting tail:
  {', '.join(facts.optional_imports[:30]) or 'none'}

Top-level third-party imports:
  {', '.join(facts.third_party_imports[:30]) or 'none'}

Config files:
{chr(10).join('  ' + c for c in facts.config_files[:25]) or '  none'}

Signs the code wants a GPU:
{chr(10).join('  ' + h for h in facts.gpu_hints[:10]) or '  none'}

## Reply format

{SCHEMA_DOC}
"""


class Compiler:
    def __init__(self, llm: LLM, *, workdir: str = "/workspace/repo"):
        self.llm = llm
        self.workdir = workdir

    def compile(self, repo_path: str | Path, *, repo_url: str, commit: str, goal: str,
                clarification: str = "", timeout_default: int = 900) -> CompileResult:
        facts = analyse(repo_path)
        pytest_command = infer_pytest_command(goal, repo_path)
        user = _facts_prompt(facts, goal, pytest_command=pytest_command)
        if clarification:
            user += (f"\n## The user answered your clarifying question\n\n{clarification}\n"
                     "\nDo not ask again; compile the criterion now.")

        raw = self.llm.json(SYSTEM, user)

        question = str(raw.get("needs_clarification") or "").strip()
        if question and not clarification:
            return CompileResult(ladder=None, facts=facts, raw=raw, question=question)

        g2, dropped = self._build_g2(raw, facts, repo_url=repo_url, commit=commit,
                                     goal=goal, timeout_default=timeout_default,
                                     command_hint=pytest_command)

        # One repair round, and only for assertions we had to DROP. Rejecting a
        # regex that matches nothing is right, but silently keeping the remainder
        # leaves a criterion weakened to artefact checks alone -- which is the
        # weak-criterion failure this whole design exists to prevent. The model
        # has the real format strings in front of it, so it can usually fix this;
        # compiling is a one-off, and the round costs a few thousand tokens.
        if dropped:
            retry = (user + "\n\n## Your previous reply had to be partly discarded\n\n"
                     + "\n".join(f"- {d}" for d in dropped)
                     + "\n\nEvery regex you write MUST match one of the printed format "
                       "strings listed above, character for character apart from the "
                       "substituted values. Note that a value like 2.4140031337738037 is "
                       "NOT matched by \\d+ -- use [0-9.]+ or [-0-9.eE+]+ for anything "
                       "that could be a float.\n"
                       "Reply with the FULL JSON object again, corrected.")
            try:
                raw2 = self.llm.json(SYSTEM, retry)
                g2b, dropped2 = self._build_g2(raw2, facts, repo_url=repo_url,
                                               commit=commit, goal=goal,
                                               timeout_default=timeout_default,
                                               command_hint=pytest_command)
                if len(dropped2) < len(dropped):
                    raw, g2, dropped = raw2, g2b, dropped2
            except (LLMError, ValueError):
                pass        # keep the first attempt; it is already valid, just weaker
        # G0 only when it has something that can fail. Its two assertions are
        # "the entry script parses" -- which py_compile answers from syntax alone
        # and no environment can break -- and "the declared imports resolve". With
        # the import set correctly scoped to the command, some repositories have
        # no third-party imports at all (robotframework is pure stdlib), and a
        # rung that cannot fail is not a rung: falsification point (1) would
        # rightly reject the whole ladder for it. Dropping it leaves G2, which is
        # what the user actually asked about.
        rungs = {Rung.G2: g2}
        if not is_pytest_command(g2.command) and _g0_modules(facts, g2.command):
            rungs[Rung.G0] = build_g0(facts, repo_url=repo_url, commit=commit,
                                      goal=goal, workdir=self.workdir,
                                      command=g2.command)
        ladder = Ladder(
            repo_url=repo_url, commit=commit, goal=goal,
            rungs=rungs,
            assets=[] if is_pytest_command(g2.command) else _assets(raw, facts, g2.command),
            clarification={"question": question, "answer": clarification} if clarification else {},
        )
        disclosures = guessed_thresholds(g2) + [
            f"an assertion was discarded and NOT replaced: {d}" for d in dropped]
        if not g2.metrics and not g2.sanity:
            disclosures.append(
                "This criterion asserts only that the command completes and that its "
                "output files exist. Nothing checks that the numbers are sane, so it "
                "cannot tell 'it ran' from 'it ran and learned nothing'. Consider "
                "supplying an expected range yourself, or recompiling.")
        return CompileResult(ladder=ladder, facts=facts, raw=raw,
                             notes=str(raw.get("notes") or ""),
                             disclosures=disclosures)

    # -- the model's reply, bounded by the static facts --------------------

    def _build_g2(self, raw: dict, facts: RepoFacts, *, repo_url: str, commit: str,
                  goal: str, timeout_default: int,
                  command_hint: str = "") -> tuple[Criterion, list[str]]:
        command = str(raw.get("command") or "").strip()
        if not command:
            raise LLMError("the reply names no command; nothing could be run")
        # An explicit pytest invocation is part of the user's contract.  The LLM
        # may choose the criterion assertions and shrink flags, but it must not
        # silently replace ``pytest tests/test_x.py`` with another command.
        if command_hint:
            command = command_hint
        command, repairs = repair_command(command, facts)

        if is_pytest_command(command):
            timeout = int(raw.get("timeout") or timeout_default)
            return Criterion(
                rung=Rung.G2,
                repo_url=repo_url,
                commit=commit,
                goal=goal,
                command=command,
                workdir=self.workdir,
                timeout=timeout,
                budget=Budget(wall_seconds=timeout * 6),
                notes=" | ".join([str(raw.get("notes") or ""), *repairs,
                                  "pytest selection extracted from the user instruction"]
                                 ).strip(" |"),
            ), []

        artifacts = _merge_artifacts(raw.get("artifacts") or [], facts, command,
                                     goal=goal)
        # Only what THIS command's scripts print counts: a repository-wide list
        # let `loss: ([0-9.]+)` through on nanoGPT because bench.py prints
        # `loss: {lossf}` while the frozen train.py prints `loss {lossf}`.
        printed = printed_formats_for(facts, command)
        metrics, dropped = [], []
        for m in raw.get("metrics") or []:
            if not isinstance(m, dict) or not {"name", "source"} <= set(m):
                continue
            try:
                built = Metric.build(m["name"], m["source"], m.get("assert", ""))
                if built.source.kind == "stdout" and not stdout_regex_matches_reality(
                        built.source.pattern, printed):
                    dropped.append(
                        f"dropped metric {m.get('name')!r}: its regex "
                        f"/{built.source.pattern}/ matches nothing the script prints")
                    continue
                metrics.append(built)
            except ValueError as e:
                # A single unparseable metric is not worth losing the whole
                # compile over -- the artefact and sanity assertions still stand,
                # and the drop is reported rather than swallowed.
                dropped.append(f"dropped metric {m.get('name')!r}: {e}")
        known = {m.name for m in metrics}
        sanity = [s for s in (_sanity(d, known) for d in raw.get("sanity") or []) if s]
        # A sanity check reading stdout needs a regex that matches reality just as
        # much as a metric does, and it is the one that carries the assertion once
        # metrics are declared without bounds.
        kept = []
        for chk in sanity:
            if (chk.source and chk.source.kind == "stdout"
                    and not stdout_regex_matches_reality(chk.source.pattern,
                                                         printed)):
                dropped.append(
                    f"dropped sanity {chk.kind!r}: its regex "
                    f"/{chk.source.pattern}/ matches nothing the script prints")
                continue
            kept.append(chk)
        sanity = kept
        # A metric declared without a bound asked for a finiteness check in all
        # but name; give it one, or the value is extracted and never examined.
        covered = {s.metric for s in sanity if s.kind == "no_nan_inf"}
        for m in metrics:
            if m.comparison is None and m.name not in covered:
                sanity.append(Sanity(kind="no_nan_inf", metric=m.name))
        repeats = 2 if any(s.kind == "reproducible" for s in sanity) else 1

        # A trend needs two observations. When the values being compared are
        # printed inside a hardcoded gate, reaching that gate once is not enough:
        # makemore prints its train/test loss under `step % 500 == 0`, so 501
        # steps yield exactly one point and `improves` fails for arithmetic
        # reasons rather than for anything about the environment.
        needs_trend = any(s.kind == "improves" for s in sanity)
        shrink = _shrink(raw.get("shrink"), facts, trend=needs_trend)
        return Criterion(
            rung=Rung.G2,
            repo_url=repo_url,
            commit=commit,
            goal=goal,
            command=command,
            workdir=self.workdir,
            timeout=int(raw.get("timeout") or timeout_default),
            repeats=repeats,
            budget=Budget(wall_seconds=int(raw.get("timeout") or timeout_default) * 6),
            shrink=shrink,
            shrink_tier2=_shrink(raw.get("shrink_tier2"), facts, tier2=True,
                                 trend=needs_trend),
            artifacts=artifacts,
            metrics=metrics,
            sanity=sanity,
            notes=" | ".join([str(raw.get("notes") or "")] + repairs + dropped).strip(" |"),
        ), dropped


# Placeholder names that stand for a counter rather than a measurement. Anything
# else is assumed to be a real-valued number, because that is what a metric is.
_INTEGERISH = re.compile(
    r"^(step|steps|epoch|epochs|iter|iters|iteration|i|j|n|idx|index|count|"
    r"size|num|n_\w+|num_\w+|\w*_count|\w*_size|batch\w*)$", re.I)
_PLACEHOLDER_ANY = re.compile(r"\{([^{}]*)\}")


def _synthesise(fmt: str) -> str:
    r"""One plausible line of real output, built from a real format string.

    Types come from the placeholder's own name: `{step}` is a counter, `{loss}`
    is a measurement. Getting this right is the whole point -- a regex written as
    `train loss: \d+ test loss: (...)` passes if the sample happens to be an
    integer and fails against the `3.2456` a training script actually prints.
    """
    def one(m: re.Match) -> str:
        name = m.group(1).strip()
        if name:
            return "500" if _INTEGERISH.match(name) else "3.2456"
        # Anonymous slot: fall back to the literal word in front of it.
        # `step {}` is a counter however the f-string was written.
        before = fmt[:m.start()].rstrip(" :=|,[(")
        word = re.split(r"[^A-Za-z_]", before)[-1] if before else ""
        return "500" if word and _INTEGERISH.match(word) else "3.2456"
    return _PLACEHOLDER_ANY.sub(one, fmt)


def stdout_regex_matches_reality(pattern: str, printed_formats: list[str]) -> bool:
    r"""Does this regex match any line the script can actually print?

    We know the real format strings (static extraction), so a synthetic line can
    be built from each and the regex tried against it. That turns "the model
    invented a regex" -- a permanently-false criterion that falsification cannot
    see -- into a compile-time check.

    Observed twice on the first real repository: `loss=([0-9.]+)` for a script
    printing `loss 3.2456`, and then `train loss: \d+ test loss: ([0-9.]+)`,
    where `\d+` cannot cross the decimal point of the value in front of it.
    """
    try:
        rx = re.compile(pattern)
    except re.error:
        return False
    if not printed_formats:
        return True                    # nothing to check against; do not block
    return any(rx.search(_synthesise(f)) for f in printed_formats)


def guessed_thresholds(c: Criterion) -> list[str]:
    """Numeric thresholds the model chose, surfaced for a human to sanity-check.

    Falsification structurally cannot catch these. A threshold that is merely too
    strict fails in a bare container (point 1 satisfied), fails with only
    top-level requirements (point 2 satisfied), and references paths that all
    exist (point 3 satisfied) -- yet it can never be met, and the Router will
    spend its whole budget before escalating. The only thing that could detect it
    is a run in an already-correct environment, which is precisely what we do not
    have.

    Observed on the first real repository: `loss < 1` for a character-level
    language model whose loss starts near 3.3 and was being asked to converge in
    two steps.

    So it is disclosed instead. The compile report prints these, and the user
    reads them before `rsa run` makes the criterion binding.
    """
    out: list[str] = []
    for m in c.metrics:
        cmp_ = m.comparison
        if cmp_ is None:
            continue          # no bound was claimed, so nothing was guessed
        if cmp_.kind == "cmp" and cmp_.op in ("<", "<=", ">", ">="):
            out.append(
                f"metric {m.name!r} asserts {cmp_.render()!r}. This threshold was "
                "chosen by the model, not measured. If it is wrong the criterion "
                "can never pass, and that is indistinguishable from a broken "
                "environment. Check it against what you expect this experiment to "
                "produce, or weaken it to a finiteness/direction check.")
        elif cmp_.kind in ("within", "in_range"):
            out.append(
                f"metric {m.name!r} asserts {cmp_.render()!r}, a range the model "
                "chose. Confirm it, or replace it with one you know.")
    return out


# Flags that belong to the interpreter, not to the script being run. `-m` is the
# one that matters: `python -m pkg ...` is the commonest invocation there is, and
# checking it against the repository's argparse tables rejected three of ten real
# tasks outright ("the command uses '-m', which lark/tools/nearley.py does not
# accept"). Everything after the module or script name is the script's.
_INTERPRETER_FLAGS = frozenset({
    "-m", "-c", "-u", "-O", "-OO", "-B", "-E", "-I", "-s", "-S", "-b", "-bb",
    "-d", "-q", "-v", "-W", "-X", "-P", "-R",
})
_PY = re.compile(r"(^|/)(python[\d.]*|uv|poetry|pipenv|hatch|nox|tox)$")


def _entry_flags(command: str, facts: RepoFacts) -> tuple[dict[str, str], str]:
    """The flag table of the script this command actually runs, and its name.

    Scoped, not unioned. The union of every argparse table in the repository is
    wrong in both directions and was measured doing both: it rejected lark's
    `lark.tools.standalone` command using `lark/tools/nearley.py`'s table, and it
    accepted `python -m cantools ...` only because an unrelated subcommand
    (`subparsers/monitor.py:568`) happens to declare `-m/--frame-id-mask` -- which
    also gave it licence to rewrite that flag into something else entirely.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {}, ""

    target = ""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _PY.search(tok) or tok in ("run", "exec"):
            i += 1
            continue
        if tok == "-m" and i + 1 < len(tokens):
            target = tokens[i + 1]
            break
        if tok == "-c":
            return {}, ""                     # inline source: no script table
        if tok.startswith("-"):
            i += 1
            continue
        target = tok
        break

    if not target:
        return {}, ""

    if target.endswith(".py"):
        want = [ep for ep in facts.entrypoints if ep.file == target.lstrip("./")]
    else:
        # `-m pkg.mod` -> files under that package, or a file of that name.
        parts = target.split(".")
        want = [ep for ep in facts.entrypoints
                if any(seg and (f"/{seg}/" in f"/{ep.file}"
                                or ep.file.endswith(f"/{seg}.py")
                                or ep.file == f"{seg}.py")
                       for seg in parts)]
    flags: dict[str, str] = {}
    for ep in want:
        flags.update(ep.flags)
    return flags, target


def repair_command(command: str, facts: RepoFacts) -> tuple[str, list[str]]:
    """Repair a flag that unambiguously misspells one the entry script declares.

    Observed on the first real repository: the model produced `-batch_size 1` for
    a script declaring `--batch-size`. argparse rejects it, the run never starts,
    and the Router would hand that to the agent as an environment problem it
    cannot fix.

    **This no longer raises.** The previous version failed the whole compile on
    any flag it did not recognise, and on ten real repositories it was wrong far
    more often than right: `-m` is an interpreter flag, and the table it checked
    against was the union of every argparse parser in the tree. An unknown flag
    is now reported as a note -- the criterion still gets frozen, the Router still
    sees it fail, and the escalation card still prints the command, which is what
    lets a user say "that is not what I asked you to run".
    """
    known, target = _entry_flags(command, facts)
    if not known:
        return command, []                    # nothing trustworthy to check against

    def norm(f: str) -> str:
        return f.lstrip("-").replace("_", "-").lower()

    canonical: dict[str, str] = {}
    for flag in known:
        canonical.setdefault(norm(flag), flag if flag.startswith("--") else flag)
    for flag in known:                        # prefer the long spelling
        if flag.startswith("--"):
            canonical[norm(flag)] = flag

    try:
        tokens = shlex.split(command)
    except ValueError:
        return command, []

    out, notes = [], []
    seen_target = False
    for tok in tokens:
        if tok == target or tok.lstrip("./") == target:
            seen_target = True
        if (not tok.startswith("-") or tok == "-" or _is_negative_number(tok)
                or not seen_target or tok in _INTERPRETER_FLAGS):
            out.append(tok)
            continue
        name, sep, inline = tok.partition("=")
        if name in known:
            out.append(tok)
            continue
        fixed = canonical.get(norm(name))
        if fixed:
            out.append(fixed + sep + inline)
            notes.append(f"repaired flag {name!r} -> {fixed!r}")
            continue
        out.append(tok)
        notes.append(
            f"{name!r} is not declared by {target}; if the run fails on an "
            f"unrecognised argument this is why. Declared: "
            f"{', '.join(sorted(known)[:15])}")
    return " ".join(shlex.quote(t) if " " in t else t for t in out), notes


def _is_negative_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


# Directories whose write targets belong to somebody else's run. A sink found in
# `tests/` is a fixture the test suite creates, not an output of the user's
# command, and asserting on it makes the criterion permanently false. Measured:
# cookiecutter's manifest was three paths from `tests/`, cantools' contained
# `out.pdf` from `tests/test_plot.py:1444`, and umap's contained eleven files
# from `examples/galaxy10sdss.py` -- for a command that runs none of them.
_FOREIGN_DIRS = ("tests/", "test/", "testing/", "docs/", "doc/", "examples/",
                 "example/", "benchmarks/", "benchmark/", "bench/", "perf_tests/",
                 "tools/", "scripts/", "ci/")


def _is_foreign(sink_file: str, command_files: set[str]) -> bool:
    """Is this sink defined in a file the chosen command does not run?

    Exact file match, not directory match: `python examples/sinusoid.py` legitimately
    writes `sinusoid.png` from `examples/sinusoid.py`, but `examples/tanh.py` in the
    same directory is a different program and its `tanh.png` is nothing to do with
    this run.
    """
    f = sink_file.replace("\\", "/").lstrip("./")
    if f in command_files:
        return False
    return any(f.startswith(d) or f"/{d}" in f"/{f}" for d in _FOREIGN_DIRS)


def _command_files(command: str) -> set[str]:
    """Repository files the command names outright."""
    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()
    return {t.lstrip("./") for t in toks if t.endswith(".py")}


def _merge_artifacts(proposed: list, facts: RepoFacts, command: str,
                     goal: str = "") -> list[Artifact]:
    """Static manifest is authoritative; the model may only enrich an entry.

    Every literal write target found in the source becomes an assertion whether or
    not the model mentioned it. That is the whole mechanism by which a skipped
    stage is detected, and leaving it to the model would let the same oversight
    that produced a bad shrink configuration also hide its consequences.

    **Scoped to the command.** Static analysis reads the whole tree, so the
    unfiltered manifest asserts on files that only the test suite or a different
    example ever writes -- which is not a stage this run skipped, it is a stage
    this run was never supposed to have.
    """
    by_path: dict[str, dict] = {}
    for p in proposed:
        if isinstance(p, dict) and p.get("path"):
            by_path[str(p["path"]).lstrip("./")] = p

    out: list[Artifact] = []
    seen: set[str] = set()
    cmd_files = _command_files(command)
    # Resolved against the command the model just chose, so a path built from an
    # `--output-dir` flag becomes concrete instead of being dropped.
    for s in resolve_sinks(facts, command):
        key = s.path.lstrip("./")
        if key in seen or _is_foreign(s.file, cmd_files):
            continue
        seen.add(key)
        out.append(_artifact_from(by_path.get(key, {}), default_path=s.path, sink=s))

    # Anything the model added that static analysis could not see -- a path spelled
    # out in a config file, for instance -- is kept only if it sits under a
    # directory this command actually writes to **and** the user actually named
    # it. Recognising `--output-dir out` (which the argparse-only reader missed)
    # rescued cookiecutter's real artefact and, in the same move, let through
    # trafilatura's `out/some_file.json` -- a filename the model invented, under a
    # root that is now legitimately writable. The goal text is the only authority
    # on what the user expects to come out, so an unbacked path has to appear in it.
    #
    # Observed on the first real repository: asked for a criterion whose command
    # used `-o names`, the model also listed `out/model.pt` -- the example path
    # from this module's own schema documentation. Nothing ever creates it, so the
    # assertion is constantly false, and no falsification point catches it (a bare
    # container fails the criterion anyway, and the path genuinely does not exist
    # at the frozen commit, which is what point 3 requires).
    allowed = _output_roots(facts, command)
    goal_text = " ".join((goal or "").split()).lower()
    for key, p in by_path.items():
        if key in seen or "{" in key:
            continue
        root = key.split("/", 1)[0]
        if allowed and root not in allowed:
            continue
        # `stdout` is not a file. The model proposed exactly that for umap, and
        # the goal text mentions the word, so a goal check alone lets it through.
        if "." not in key and "/" not in key:
            continue
        named = (key.lower() in goal_text
                 or key.rsplit("/", 1)[-1].lower() in goal_text)
        if goal_text and not named:
            continue
        seen.add(key)
        out.append(_artifact_from(p, default_path=str(p["path"]), sink=None))
    return out


# Output-directory flags, read straight off the command line. `cli_values` can
# only resolve these when the entry point uses argparse, and plenty do not:
# cookiecutter uses click, so `--output-dir out` was invisible and the model's
# one correct artefact (`out/fake-project/README.rst`) was filtered out for
# naming a root nothing had declared -- while three paths from `tests/` stayed.
_OUTPUT_FLAGS = frozenset({
    "-o", "--out", "--outdir", "--out-dir", "--output", "--output-dir",
    "--outputdir", "--output-directory", "--output-file", "--output-path",
    "--dir", "--destination", "--dest", "--target-dir",
})


def _output_roots(facts: RepoFacts, command: str) -> set[str]:
    """Top-level directories this command is known to write into."""
    from ..static_facts import cli_values
    cmd_files = _command_files(command)
    roots = {s.path.split("/", 1)[0] for s in resolve_sinks(facts, command)
             if not _is_foreign(s.file, cmd_files)}
    for dest, value in cli_values(facts, command).items():
        if any(t in dest for t in ("dir", "out", "path", "output")) and value:
            roots.add(str(value).strip("/").split("/", 1)[0])

    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()
    for i, tok in enumerate(toks):
        name, sep, inline = tok.partition("=")
        if name.lower() not in _OUTPUT_FLAGS:
            continue
        value = inline if sep else (toks[i + 1] if i + 1 < len(toks) else "")
        if value and not value.startswith("-"):
            roots.add(value.strip("/").split("/", 1)[0])
    return {r for r in roots if r}


def _artifact_from(p: dict, *, default_path: str, sink: Sink | None) -> Artifact:
    check = str(p.get("check") or "").strip()
    loader = str(p.get("loader") or "").strip()
    keys = tuple(p.get("schema_keys") or ())
    stage = str(p.get("stage") or (sink.stage if sink else "")) or ""

    if check == "loadable" and not loader:
        loader = (sink.loader if sink else "")
        if loader not in DEFAULT_SAFE_LOADERS:
            # Asked to prove it deserialises without saying how, and nothing about
            # the writing call says either. Guessing here produces the worst kind
            # of failure: a real artefact reported as corrupt.
            check, loader = "nonempty", ""
    if check == "schema" and not keys:
        check = "nonempty"          # a schema check with no keys asserts nothing
    if check == "hash":
        check = "nonempty"          # the model cannot know an output's hash in advance
    if check not in ("exists", "nonempty", "schema", "loadable"):
        # Default to the strongest check the sink *reliably* supports. `exists` is
        # never chosen by default: a stage that opens a file and writes nothing
        # passes it. `loadable` is chosen only when the writing call determined the
        # format -- `Path(x).write_text(...)` does not, so those get `nonempty`.
        loader = (sink.loader if sink else "")
        check, loader = (("loadable", loader) if loader in DEFAULT_SAFE_LOADERS
                         else ("nonempty", ""))
    return Artifact(path=default_path, check=check, loader=loader,
                    schema_keys=keys, stage=stage)


def _knob_flag(dest: str, known: dict[str, str]) -> str:
    """The flag spelling for a dest, or "" if the script has no such option."""
    for flag, d in known.items():
        if d == dest:
            return flag
    guess = "--" + dest.replace("_", "-")
    return guess if guess in known else ""


def _shrink(d: object, facts: RepoFacts, *, tier2: bool = False,
            trend: bool = False) -> Shrink | None:
    knobs = dict((d or {}).get("knobs") or {}) if isinstance(d, dict) else {}
    flatten = dict((d or {}).get("flatten") or {}) if isinstance(d, dict) else {}

    # Every shrink knob ends up on the command line via Shrink.as_cli(), so a knob
    # the script does not declare is a flag argparse will reject -- and then EVERY
    # round fails, identically, for a reason no amount of installing can fix.
    #
    # Measured: the model proposed `eval_every` and `save_every` for makemore,
    # which declares neither. The three rounds that followed cost 88k tokens and
    # all five assertions failed every time on
    #   `makemore.py: error: unrecognized arguments: --eval-every 1 --save-every 1`
    # The Router correctly saw a gap that never shrank and escalated, which is the
    # right behaviour for the wrong reason: the criterion was unrunnable from the
    # start. The command was already validated this way; the shrink was not.
    known: dict[str, str] = {}
    for ep in facts.entrypoints:
        known.update(ep.flags)
    if known:
        knobs = {k: v for k, v in knobs.items() if _knob_flag(k, known)}
        flatten = {k: v for k, v in flatten.items() if _knob_flag(k, known)}

    # Rule 3, enforced rather than requested: every gating knob static analysis
    # found is pressed flat -- but only when it is reachable from the command
    # line at all. makemore's gates are hardcoded literals with no flag behind
    # them, and the answer there is the step floor below, not an invented flag.
    for k in facts.gating_knobs:
        if not known or _knob_flag(k, known):
            flatten.setdefault(k, 1)
    # A hardcoded gate cannot be flattened, so the step count has to clear it.
    # Left to itself the model shrinks to 2 steps and the checkpoint stage --
    # sitting inside `if step % 500 == 0` -- is never entered, which the artefact
    # manifest then correctly reports as a stage that did not run. Raising the
    # count here fixes the cause instead of arguing with the symptom.
    floor = facts.min_steps_for_all_stages()
    if trend and facts.hardcoded_gates:
        # Two passes through the widest gate, so a trend has two points to compare.
        floor = max(g["interval"] for g in facts.hardcoded_gates) * 2 + 1
    if floor > 1:
        for key in ("max_steps", "max_step", "num_steps", "steps", "train_steps",
                    "iters", "iterations"):
            if key in knobs and int(knobs[key]) < floor:
                knobs[key] = floor
    if not knobs and not flatten:
        return None
    if tier2 and not knobs:
        return None
    return Shrink(knobs=knobs, flatten=flatten)


def _sanity(d: object, known: set[str]) -> Sanity | None:
    if not isinstance(d, dict) or not d.get("kind"):
        return None
    src = d.get("source")
    kw = {k: v for k, v in d.items()
          if k in ("kind", "metric", "num_classes", "baseline", "direction",
                   "min_rel_change", "tol")}
    if kw.get("metric") and kw["metric"] not in known:
        # A sanity check pointing at a metric that was not declared cannot be
        # rendered. Dropping it is right; failing the whole compile is not.
        return None
    try:
        return Sanity(source=MetricSource.parse(src) if src else None, **kw)
    except (ValueError, TypeError):
        return None


def _circular(asset: dict, command: str) -> bool:
    """Is this "asset" the goal itself?

    Measured on mopidy: the model declared
    `{"kind": "command", "name": "mopidy_config_run", "why": "The goal is to
    execute the specific command and check its output"}`. The Asset Gate then
    checked, *before any configuring*, whether `mopidy config` runs -- got
    `exited 127: command not found` -- and blocked the run. That is circular:
    the thing to be achieved cannot also be a precondition for starting.
    """
    if str(asset.get("kind")) != "command":
        return True if asset.get("kind") == "command" and not asset.get("command") else False
    probe = " ".join(str(asset.get("command") or "").split())
    if not probe:
        return True
    goal = " ".join(command.split())
    return probe in goal or goal in probe or probe.split()[0] in goal.split()


def _assets(raw: dict, facts: RepoFacts, command: str = "") -> list[Asset]:
    out: list[Asset] = []
    for a in raw.get("assets") or []:
        if not isinstance(a, dict) or not a.get("kind"):
            continue
        if command and _circular(a, command):
            continue
        kw = {k: v for k, v in a.items()
              if k in ("kind", "name", "path", "sha256", "env_var", "min_vram_gb",
                       "gpu_name_contains", "cuda_min", "cuda_max", "command",
                       "stage", "why")}
        kw.setdefault("name", kw.get("path") or kw.get("env_var") or kw["kind"])
        try:
            out.append(Asset(**kw))
        except (ValueError, TypeError):
            continue

    # Every environment variable the code reads is a candidate credential, and
    # shrinking the scale does not shrink the asset requirements.
    named = {a.env_var for a in out if a.env_var}
    for v in facts.env_vars:
        if v not in named and any(t in v.upper() for t in
                                  ("KEY", "TOKEN", "SECRET", "PASSWORD", "API")):
            out.append(Asset(kind="env_var", name=v.lower(), env_var=v,
                             why="read by the code; looks like a credential"))
    return out


# --------------------------------------------------------------------------
# G0 is derived, not generated
# --------------------------------------------------------------------------

def _g0_modules(facts: RepoFacts, command: str = "") -> set[str]:
    """Third-party imports G0 may legitimately require, and no others.

    Two exclusions, both measured:

    * **The repository's own modules.** autograd's examples do `import data`,
      `import black_box_svi`, `import rnn` -- sibling files, not packages. G0
      required all seven, and no installation on earth satisfies that: the rung
      was permanently unsatisfiable, so nothing above it could ever be reached.
    * **Imports only ever seen in tests, docs or examples.** cookiecutter's G0
      demanded `pytest` and `freezegun`; cantools' the same shape. That is the
      maintainer's development environment, not what the user asked to run --
      and G0 gates every rung above it.
    """
    local = {m.lower() for m in facts.local_modules}
    # A file the goal's own command runs is not somebody else's: autograd's goal
    # *is* `python examples/sinusoid.py`, so numpy imported there is exactly what
    # the user needs G0 to check.
    cmd_files = _command_files(command)
    # What the command actually runs. Excluding tests/docs/examples was not
    # enough: proselint's G0 demanded `SublimeLinter`, `gmail`, `worker`,
    # `fastapi`, `apscheduler` -- imports of `app.py` and `clock.py`, a web demo
    # sitting at the repository root that `python -m proselint` never touches.
    # Three of those are not installable packages at all, so the rung was
    # unsatisfiable and, being the first rung, blocked everything above it.
    # Measured: G0 failed in the *gold* environment on 8 of 12 repositories.
    owned = _owned_paths(facts, command)

    out: set[str] = set()
    for mod in facts.third_party_imports:
        if mod.lower() in local:
            continue
        # Only the *unconditional* import sites count, and only those outside
        # tests/docs/examples. python-control guards every `import slycot` with
        # try/except except in `examples/slycot-import-test.py`; requiring a
        # Fortran extension on that evidence would block every rung above G0.
        sites = facts.toplevel_files.get(mod) or facts.import_files.get(mod) or []
        if not sites:
            out.add(mod)
            continue
        if owned:
            if any(_is_owned(f, owned) for f in sites):
                out.add(mod)
            continue
        if all(_is_foreign(f, cmd_files) for f in sites):
            continue
        out.add(mod)
    return out


def _owned_paths(facts: RepoFacts, command: str) -> set[str]:
    """Path prefixes and exact files the command runs, or an empty set.

    `python -m proselint ...` owns `proselint/`; `python examples/sinusoid.py`
    owns that one file. Empty when the command names neither (a console script
    whose package cannot be located, or `python -c`), and the caller then falls
    back to the weaker directory-based rule.
    """
    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()

    owned: set[str] = set()
    root = Path(facts.root) if facts.root else None
    names: list[str] = []
    for i, tok in enumerate(toks):
        if tok.endswith(".py"):
            owned.add(tok.lstrip("./"))
        elif tok == "-m" and i + 1 < len(toks):
            names.append(toks[i + 1].split(".")[0])
        elif tok == "-c":
            return set()

    if not owned and not names:
        # A console script: `trafilatura ...`, `mopidy config`. Its name is
        # almost always the package name.
        first = next((t for t in toks if not t.startswith("-")), "")
        if first and not _PY.search(first) and "/" not in first:
            names.append(first.replace("-", "_"))

    for name in names:
        for cand in (name, f"src/{name}"):
            if root is None or (root / cand).is_dir():
                owned.add(cand + "/")
    return owned


def _is_owned(path: str, owned: set[str]) -> bool:
    p = path.replace("\\", "/").lstrip("./")
    return any(p == o or p.startswith(o) for o in owned)


def build_g0(facts: RepoFacts, *, repo_url: str, commit: str, goal: str,
             workdir: str = "/workspace/repo", command: str = "") -> Criterion:
    """Static rung: the entry point parses and the declared imports resolve.

    Written from the static facts rather than by the model, because there is
    nothing here to decide. It is the cheapest rung and the one whose failure is
    most legible -- an unresolvable import names the missing package outright.
    """
    entry = facts.entrypoints[0].file if facts.entrypoints else ""
    mods = sorted(_g0_modules(facts, command))[:40]
    lines = [
        "def test_entrypoint_parses():",
        '    """The script the goal refers to is syntactically valid under this Python."""',
        f"    code, out = sh('python3 -m py_compile {entry}')" if entry else
        "    code, out = (0, '')",
        "    assert code == 0, out[-2000:]",
        "",
        "def test_declared_imports_resolve():",
        '    """Every top-level third-party import can actually be imported."""',
        f"    mods = {mods!r}",
        "    failed = []",
        "    for m in mods:",
        "        code, out = sh('python3 -c ' + repr('import ' + m))",
        "        if code != 0:",
        "            failed.append(m + ': ' + out.strip().splitlines()[-1][:160] "
        "if out.strip() else m)",
        "    assert not failed, 'unimportable: ' + '; '.join(failed)",
    ]
    return Criterion(
        rung=Rung.G0,
        repo_url=repo_url,
        commit=commit,
        goal=goal,
        command="true",          # G0 observes the environment; it runs no experiment
        workdir=workdir,
        timeout=300,
        free_pytest="\n".join(lines) + "\n",
        notes="derived from static analysis; no model involvement",
    )
