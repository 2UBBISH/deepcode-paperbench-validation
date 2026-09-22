"""Read the repository without running it, and without asking a model.

Feeds the Compiler. Everything here is `ast` and file reads, so it is repeatable
and cheap, and -- more importantly -- it is not the part that can be talked into
a convenient answer. The model's job is to turn a goal into a command; the
artefact manifest, the gating knobs and the asset list are extracted, not
generated.

The two extractions section 4.5 makes load-bearing:

* **Artefact sinks.** Every place the code writes something to disk. A shrunk run
  that produces all of them reached every stage; one that skips a stage is caught
  by name. This is what turns "did the whole pipeline run" into a deterministic
  question, and it is why the manifest cannot be left to the model -- a model
  writing the manifest from the same reading that produced the shrink config will
  make the same omission twice.

* **Gating knobs.** Lowering `max_steps` does not reach a stage guarded by
  `if step % eval_every == 0`. Those guards have to be found and pressed flat, or
  the shrunk run systematically skips the tail of the script -- which is exactly
  where the fragile optional dependencies are (plotting backends, spreadsheet
  writers, experiment trackers, exporters).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Files that are never the experiment.
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox",
             "build", "dist", ".eggs", "site-packages", ".mypy_cache", ".pytest_cache"}

# Calls that put bytes on disk. `(module_or_obj, attr) -> (argument index holding
# the path, artefact kind)`. Index -1 means "look at the keyword `path`/`filename`".
SINK_CALLS: dict[str, tuple[int, str]] = {
    "torch.save": (1, "checkpoint"),
    "torch.jit.save": (1, "checkpoint"),
    "np.save": (0, "array"),
    "np.savez": (0, "array"),
    "np.savez_compressed": (0, "array"),
    "numpy.save": (0, "array"),
    "numpy.savetxt": (0, "array"),
    "json.dump": (1, "result"),
    "pickle.dump": (1, "result"),
    "joblib.dump": (1, "checkpoint"),
    "savefig": (0, "figure"),
    "imwrite": (0, "figure"),
    "to_csv": (0, "table"),
    "to_json": (0, "table"),
    "to_excel": (0, "table"),
    "to_parquet": (0, "table"),
    "to_markdown": (0, "table"),
    # These say nothing about the format -- `Path(x).write_text(...)` writes JSON,
    # CSV or a PNG blob with equal enthusiasm. Inferring a loader from the file
    # extension instead was tried and is wrong for the same reason: a `.ckpt`
    # written by `write_text` is not a torch checkpoint. Kind "file" therefore
    # carries no loader, and such artefacts get a `nonempty` check.
    "write_text": (-2, "file"),
    "write_bytes": (-2, "file"),
    "open": (0, "file"),            # only a sink when the mode says so
    "FileHandler": (0, "log"),
    "SummaryWriter": (0, "log"),
    "save_pretrained": (0, "checkpoint"),
    "onnx.export": (2, "export"),
}

# Loaders the criteria file can use to prove an artefact deserialises. Only the
# calls that *determine* the serialisation format get one.
KIND_TO_LOADER = {
    "checkpoint": "torch", "array": "numpy", "result": "json", "table": "csv",
    "figure": "image", "log": "", "export": "", "file": "",
}

# Loaders safe to apply by default. `image` and `yaml` are excluded on purpose:
# they need Pillow or PyYAML *in the environment under test*, so asserting with
# them silently adds a dependency the user's goal may not include. They are used
# only when the criterion asks for them explicitly.
DEFAULT_SAFE_LOADERS = frozenset({"json", "torch", "numpy", "pickle", "csv"})

_EVERY_RE = re.compile(r"(_every|_freq|_frequency|_interval|_steps_per)$")
_PLACEHOLDER_ANY = re.compile(r"\{[^{}]*\}")

# Left-hand operands that mark `X % Y == 0` as a stage gate rather than a
# divisibility constraint. `makemore` is the worked example: `assert n_embd %
# n_head == 0` would otherwise register n_head as something to flatten to 1.
_COUNTERS = frozenset({
    "step", "steps", "i", "j", "n", "it", "iter", "iteration", "iters", "epoch",
    "epochs", "global_step", "batch_idx", "batch_i", "idx", "counter", "t",
})
_SCALE_RE = re.compile(
    r"^(epochs?|num_epochs|max_steps?|num_steps|train_steps|iters?|iterations|"
    r"batch_size|train_batch_size|per_device_train_batch_size|subset|subsample|"
    r"num_samples|n_samples|limit|max_samples|max_train_samples|max_eval_samples|"
    r"num_workers|seq_len|max_length)$"
)


@dataclass
class Sink:
    path: str              # literal path, or a pattern with {} where it was dynamic
    literal: bool
    kind: str
    func: str              # call that produced it, e.g. "torch.save"
    stage: str             # enclosing function name -- the stage this proves ran
    file: str
    line: int

    @property
    def loader(self) -> str:
        return KIND_TO_LOADER.get(self.kind, "")


@dataclass
class EntryPoint:
    file: str
    has_main_guard: bool
    options: dict[str, str] = field(default_factory=dict)   # dest -> default (repr)
    # Every spelling of every flag -> its dest ("-o" and "--work-dir" both ->
    # "work_dir"). Needed to resolve `{work_dir}/model.pt` against the command
    # the Compiler chose; without the short forms, `-o out` is unreadable.
    flags: dict[str, str] = field(default_factory=dict)
    score: int = 0


@dataclass
class RepoFacts:
    root: str
    entrypoints: list[EntryPoint] = field(default_factory=list)
    sinks: list[Sink] = field(default_factory=list)
    gating_knobs: list[str] = field(default_factory=list)
    # Stages gated by a hardcoded interval (`if step % 500 == 0`) rather than by
    # a flag. Nothing on the command line can flatten these, so the shrink has to
    # run PAST the largest of them or the stage is simply never reached.
    hardcoded_gates: list[dict] = field(default_factory=list)
    scale_knobs: list[str] = field(default_factory=list)
    # Format strings the script actually prints. A metric read from stdout needs a
    # regex that matches REAL output; a model asked to invent one writes
    # `loss=([0-9.]+)` for a script that prints `loss 3.2345`, and the criterion is
    # then permanently unsatisfiable in a way falsification cannot see -- the bare
    # container fails it for unrelated reasons, so point (1) is satisfied.
    printed_formats: list[str] = field(default_factory=list)
    # format -> repository files that print it. `printed_formats` is repo-wide,
    # and on nanoGPT that let `loss: ([0-9.]+)` pass the reality check because
    # bench.py prints `loss: {lossf}` while the frozen command runs train.py,
    # which prints `loss {lossf}`. The Compiler scopes the check to the files
    # the command actually executes (see `printed_formats_for`).
    printed_format_files: dict[str, list[str]] = field(default_factory=dict)
    # file -> repository-local modules it imports (transitive closure is built
    # on demand by `printed_formats_for`).
    local_import_edges: dict[str, list[str]] = field(default_factory=dict)
    env_vars: list[str] = field(default_factory=list)
    data_paths: list[str] = field(default_factory=list)
    third_party_imports: list[str] = field(default_factory=list)
    optional_imports: list[str] = field(default_factory=list)
    # module -> the repository files that import it, and the set of names the
    # repository itself provides. Both exist for one reason: to stop G0 asserting
    # that a sibling script or a test-only package is installable.
    import_files: dict[str, list[str]] = field(default_factory=dict)
    toplevel_files: dict[str, list[str]] = field(default_factory=dict)
    local_modules: list[str] = field(default_factory=list)
    gpu_hints: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    readme_commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def min_steps_for_all_stages(self) -> int:
        """Smallest step count that reaches every hardcoded gate at least once.

        Shrinking below this silently skips stages -- which is precisely the
        failure section 4.5 forbids, arriving through a route no flag can fix.
        """
        return max((g["interval"] for g in self.hardcoded_gates), default=0) + 1

    def artifact_manifest(self) -> list[Sink]:
        """Sinks with a literal path -- the ones that can become assertions.

        Dynamic paths are kept in `sinks` for the Compiler to reason about but are
        not asserted on: a manifest entry whose path is a guess fails for the wrong
        reason and sends the Router chasing an environment problem that is not there.
        """
        seen: set[str] = set()
        out = []
        for s in self.sinks:
            if s.literal and s.path not in seen:
                seen.add(s.path)
                out.append(s)
        return out


# --------------------------------------------------------------------------

def _dotted(node: ast.AST) -> str:
    """`a.b.c` for Attribute/Name chains; the bare attribute for anything else."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _str_of(node: ast.AST | None, assigns: dict | None = None) -> tuple[str, bool]:
    """(rendered path, is_literal). Unresolvable parts render as `{name}`.

    `assigns` carries string-valued local assignments seen earlier in the same
    file, which is what makes the dominant sink shape resolvable:

        out_path = os.path.join(args.work_dir, "model.pt")
        torch.save(model.state_dict(), out_path)

    Without it the sink reads `{out_path}` -- a placeholder naming a local
    variable, which no command line can fill in.
    """
    if node is None:
        return "", False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, True
    if isinstance(node, ast.JoinedStr):          # f-string
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            else:
                out.append(_str_of(v.value if isinstance(v, ast.FormattedValue) else v,
                                   assigns)[0])
        return "".join(out), False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l, ll = _str_of(node.left, assigns)
        r, rl = _str_of(node.right, assigns)
        return l + r, ll and rl
    if isinstance(node, ast.Call):
        # os.path.join("out", "x.json") / Path("out") / open(...)
        name = _dotted(node.func)
        if name.endswith("join"):
            parts = [_str_of(a, assigns) for a in node.args]
            return "/".join(p for p, _ in parts), all(ok for _, ok in parts)
        if name in ("Path", "open"):
            return _str_of(node.args[0], assigns) if node.args else ("", False)
    # A named placeholder, not a bare `{}`. Research scripts almost always derive
    # their outputs from an `--output-dir` style argument:
    #     torch.save(model.state_dict(), os.path.join(args.work_dir, "model.pt"))
    # Rendering that as `{}/model.pt` throws away the one piece of information
    # that makes it resolvable. `{work_dir}/model.pt` can be filled in from the
    # command the Compiler chose, which turns the commonest sink shape in the
    # whole corpus from unusable into an assertion.
    if isinstance(node, (ast.Name, ast.Attribute)):
        leaf = node.attr if isinstance(node, ast.Attribute) else node.id
        if assigns and isinstance(node, ast.Name) and leaf in assigns:
            return assigns[leaf]
        return "{" + leaf + "}", False
    return "{}", False


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, facts: RepoFacts):
        self.rel = rel
        self.facts = facts
        self.stack: list[str] = []
        self.in_try = 0
        self.in_assert = 0
        # name -> (rendered, is_literal) for string-valued assignments seen so far.
        self.assigns: dict[str, tuple[str, bool]] = {}

    # -- context ------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Try(self, node: ast.Try):
        self.in_try += 1
        for n in node.body:
            self.visit(n)
        self.in_try -= 1
        for n in node.handlers + node.orelse + node.finalbody:
            self.visit(n)

    # -- imports -------------------------------------------------------

    def visit_Import(self, node: ast.Import):
        for a in node.names:
            self._note_import(a.name.split(".")[0])
            self._note_local_import(a.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and node.level == 0:
            self._note_import(node.module.split(".")[0])
            self._note_local_import(node.module)
        elif node.level > 0:
            # `from . import model` / `from .model import GPT`: relative to the
            # importing file's package, which is always repository-local.
            for a in node.names:
                self._note_local_import((node.module or a.name))
        self.generic_visit(node)

    def _note_local_import(self, mod: str):
        root = mod.split(".", 1)[0]
        if not root:
            return
        edges = self.facts.local_import_edges.setdefault(self.rel, [])
        if root not in edges:
            edges.append(root)

    def _note_import(self, mod: str):
        if mod in _STDLIB:
            return
        # Where each import was seen. G0 asserts that third-party imports resolve,
        # and without this it asserted on names that only appear in `tests/` --
        # `pytest` and `freezegun` for cookiecutter -- turning "configure what the
        # user asked for" into "reproduce the maintainer's dev environment".
        self.facts.import_files.setdefault(mod, [])
        if self.rel not in self.facts.import_files[mod]:
            self.facts.import_files[mod].append(self.rel)
        # An import inside a `try` or inside a function body is an optional
        # dependency in all but name. Those are the ones a partial install
        # misses, and the ones whose failure surfaces at the END of a run.
        if self.in_try or self.stack:
            if mod not in self.facts.optional_imports:
                self.facts.optional_imports.append(mod)
        else:
            if mod not in self.facts.third_party_imports:
                self.facts.third_party_imports.append(mod)
            # Where it is imported *unconditionally*. A module can appear in both
            # lists -- python-control guards `import slycot` everywhere except in
            # `examples/slycot-import-test.py` -- and only the unconditional sites
            # say anything about what the code needs to start.
            self.facts.toplevel_files.setdefault(mod, [])
            if self.rel not in self.facts.toplevel_files[mod]:
                self.facts.toplevel_files[mod].append(self.rel)

    # -- calls ---------------------------------------------------------

    def visit_Call(self, node: ast.Call):
        name = _dotted(node.func)
        short = name.rsplit(".", 1)[-1]

        for key in (name, short):
            if key in SINK_CALLS:
                idx, kind = SINK_CALLS[key]
                self._record_sink(node, key, idx, kind)
                break

        if short == "print" or (short in ("info", "debug", "warning", "error")
                                and _root_of(node.func) in ("logging", "logger", "log",
                                                            "self")):
            for a in node.args[:2]:
                if isinstance(a, (ast.JoinedStr, ast.Constant)):
                    text, _ = self._s(a)
                    text = " ".join(text.split())
                    if (text and len(text) > 3 and len(self.facts.printed_formats) < 40
                            and text not in self.facts.printed_formats):
                        self.facts.printed_formats.append(text)
                    if text and len(text) > 3:
                        owners = self.facts.printed_format_files.setdefault(text, [])
                        if self.rel not in owners:
                            owners.append(self.rel)

        if name in ("os.environ.get", "os.getenv"):
            val, lit = self._s(node.args[0] if node.args else None)
            if lit and val not in self.facts.env_vars:
                self.facts.env_vars.append(val)

        if short in ("cuda", "to", "device") or name.startswith("torch.cuda"):
            hint = f"{self.rel}:{node.lineno} {name}"
            if len(self.facts.gpu_hints) < 40 and _looks_gpu(node, name):
                self.facts.gpu_hints.append(hint)

        if short == "add_argument":
            self._record_argument(node)

        self.generic_visit(node)

    def _record_sink(self, node: ast.Call, key: str, idx: int, kind: str):
        if idx == -2:                       # Path(x).write_text(content)
            target = node.func.value if isinstance(node.func, ast.Attribute) else None
            path, lit = self._s(target)
        elif idx < len(node.args):
            path, lit = self._s(node.args[idx])
        else:
            kw = {k.arg: k.value for k in node.keywords}
            path, lit = self._s(kw.get("path") or kw.get("filename") or kw.get("f")
                                or kw.get("log_dir"))

        if not path:
            return
        # `open(p, 'w')` is a sink; `open(p)` is a read. Only the mode says which.
        if key == "open":
            mode = ""
            if len(node.args) > 1:
                mode, _ = self._s(node.args[1])
            if "w" not in mode and "a" not in mode and "x" not in mode:
                return
        self.facts.sinks.append(Sink(
            path=path, literal=lit, kind=kind, func=key,
            stage=self.stack[-1] if self.stack else "<module>",
            file=self.rel, line=node.lineno,
        ))

    def _record_argument(self, node: ast.Call):
        flag, lit = self._s(node.args[0] if node.args else None)
        if not lit:
            return
        kw = {k.arg: k.value for k in node.keywords}
        dest, _ = self._s(kw.get("dest"))
        # argparse derives the dest from the FIRST long option, not from the
        # first positional argument -- `add_argument("-o", "--work-dir")` is
        # `work_dir`, not `o`.
        spellings = [f for f, ok in (self._s(a) for a in node.args) if ok]
        longest = next((f for f in spellings if f.startswith("--")), flag)
        name = dest or longest.lstrip("-").replace("-", "_")
        if not name:
            return
        for ep in self.facts.entrypoints:
            if ep.file == self.rel:
                for f in spellings:
                    ep.flags[f] = name
                break
        default = ""
        if "default" in kw:
            try:
                default = repr(ast.literal_eval(kw["default"]))
            except (ValueError, TypeError, SyntaxError):
                default = "<expr>"
        for ep in self.facts.entrypoints:
            if ep.file == self.rel:
                ep.options[name] = default
                break
        if _EVERY_RE.search(name) and name not in self.facts.gating_knobs:
            self.facts.gating_knobs.append(name)
        elif _SCALE_RE.match(name) and name not in self.facts.scale_knobs:
            self.facts.scale_knobs.append(name)

    # -- gating guards --------------------------------------------------

    def _s(self, node):
        return _str_of(node, self.assigns)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            rendered, lit = _str_of(node.value, self.assigns)
            # Only keep an assignment that carries real text. A binding like
            #     train_loss = evaluate(model, ...)
            # renders as a bare "{}", and recording it is actively harmful: the
            # name is then LOST at every use site, so
            #     f"step {step} train loss: {train_loss}"
            # becomes "step {} train loss: {}" instead of keeping the names. The
            # synthesiser types placeholders by name, so erasing the names makes
            # it substitute a float for a step counter and wrongly reject
            # `step \d+ train loss: ([0-9.]+)` -- a regex that is perfectly correct.
            literal_part = _PLACEHOLDER_ANY.sub("", rendered).strip()
            if literal_part and ("/" in rendered or "." in rendered or "{" in rendered):
                self.assigns[node.targets[0].id] = (rendered, lit)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        # `assert n_embd % n_head == 0` is a shape constraint, not a stage gate.
        # Flattening `n_head` to 1 would silently rewrite the model architecture.
        self.in_assert += 1
        self.generic_visit(node)
        self.in_assert -= 1

    def visit_Compare(self, node: ast.Compare):
        # `step % eval_every == 0`: the modulus operand names the knob that has to
        # be flattened, whether or not it was ever an argparse option.
        if (not self.in_assert
                and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mod)
                and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
            knob = _dotted(node.left.right) or getattr(node.left.right, "id", "")
            knob = knob.rsplit(".", 1)[-1]
            counter = (_dotted(node.left.left) or "").rsplit(".", 1)[-1].lower()
            # Two independent ways to recognise a real gate. Requiring one of them
            # is what keeps divisibility checks and hash-bucket arithmetic out.
            looks_gated = bool(_EVERY_RE.search(knob)) or counter in _COUNTERS
            if knob and not knob.isdigit() and looks_gated \
                    and knob not in self.facts.gating_knobs:
                self.facts.gating_knobs.append(knob)
            # `if step % 500 == 0` -- a literal interval, unreachable by any flag.
            # makemore is the worked example: its checkpoint is written inside
            # such a guard, so a two-step shrink produces no checkpoint at all and
            # the artefact manifest correctly reports a stage that never ran.
            rhs = node.left.right
            if (counter in _COUNTERS and isinstance(rhs, ast.Constant)
                    and isinstance(rhs.value, int) and rhs.value > 1):
                row = {"interval": rhs.value, "counter": counter,
                       "file": self.rel, "line": node.lineno,
                       "stage": self.stack[-1] if self.stack else "<module>"}
                if row not in self.facts.hardcoded_gates:
                    self.facts.hardcoded_gates.append(row)
        self.generic_visit(node)


def _root_of(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return getattr(node, "id", "")


def _looks_gpu(node: ast.Call, name: str) -> bool:
    if name.startswith("torch.cuda"):
        return True
    for a in list(node.args) + [k.value for k in node.keywords]:
        s, lit = _str_of(a)
        if lit and "cuda" in s.lower():
            return True
    return False


# The interpreter's own list, plus the hand-written one below as a floor. A short
# hand list was "enough" while these names only decorated a prompt; once G0
# started *asserting* that every third-party import resolves, every stdlib module
# missing from the list became a package the agent was told to install --
# measured: `errno`, `bisect`, `calendar`, `builtins`, `operator`, `tkinter`,
# `smtplib`, `fileinput` and a dozen more across ten repositories.
_STDLIB = set(getattr(__import__("sys"), "stdlib_module_names", ())) | {
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "enum", "functools", "glob", "gzip",
    "hashlib", "io", "importlib", "inspect", "itertools", "json", "logging", "math",
    "multiprocessing", "os", "pathlib", "pickle", "random", "re", "shutil", "signal",
    "socket", "string", "subprocess", "sys", "tempfile", "textwrap", "time", "types",
    "typing", "unittest", "urllib", "uuid", "warnings", "weakref", "zipfile", "__future__",
}

_README_CMD = re.compile(
    r"^\s*(?:\$\s*)?((?:python3?|uv run|poetry run|bash|sh|accelerate launch|"
    r"torchrun|make)\s+[^\n`]{3,200})$", re.M)


def _py_files(root: Path, limit: int = 400) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def cli_values(facts: RepoFacts, command: str) -> dict[str, str]:
    """What each argparse dest is set to, given a concrete command.

    Flags first, argparse defaults second. The defaults matter as much as the
    flags: a script whose `--work-dir` defaults to `out` writes to `out` whether
    or not the command mentions it.
    """
    flags: dict[str, str] = {}
    defaults: dict[str, str] = {}
    for ep in facts.entrypoints:
        flags.update(ep.flags)
        defaults.update(ep.options)

    values: dict[str, str] = {}
    for dest, raw in defaults.items():
        try:
            v = ast.literal_eval(raw) if raw else None
        except (ValueError, SyntaxError):
            v = None
        if isinstance(v, (str, int, float)):
            values[dest] = str(v)

    try:
        tokens = __import__("shlex").split(command)
    except ValueError:
        tokens = command.split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            name, sep, inline = tok.partition("=")
            dest = flags.get(name)
            if dest:
                if sep:
                    values[dest] = inline
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    values[dest] = tokens[i + 1]
                    i += 1
        i += 1
    return values


def resolve_sinks(facts: RepoFacts, command: str) -> list[Sink]:
    """Every sink whose path is now concrete, literal ones included.

    This is what makes the artefact manifest work on real research code. The
    dominant shape in the corpus is an output path derived from an
    `--output-dir` style flag, so a manifest limited to string literals is empty
    for most repositories -- and an empty manifest silently disables the one
    mechanism that detects a stage the shrink configuration skipped.
    """
    values = cli_values(facts, command)
    out: list[Sink] = []
    seen: set[str] = set()
    for s in facts.sinks:
        path = s.path
        if not s.literal:
            path = _PLACEHOLDER.sub(
                lambda m: values.get(m.group(1), m.group(0)), path)
            if _PLACEHOLDER.search(path):
                continue                     # still unknown; not assertable
        # `_PLACEHOLDER` only matches a NAMED slot, so an anonymous `{}` -- what
        # `_str_of` emits when it cannot read the expression at all -- slipped
        # through as if it were a literal path. It then reached the criteria file
        # as `f"{}: size={n}"`, which does not parse, and pytest could not collect
        # the file at all: two of ten real repositories died that way with no
        # criterion ever produced. Any surviving brace means the path is unknown.
        if "{" in path or "}" in path:
            continue
        path = path.strip("/") or path
        if path in seen:
            continue
        seen.add(path)
        out.append(Sink(path=path, literal=True, kind=s.kind, func=s.func,
                        stage=s.stage, file=s.file, line=s.line))
    return out


def analyse(root: str | Path) -> RepoFacts:
    root = Path(root).resolve()
    facts = RepoFacts(root=str(root))

    # Names the repository itself provides. A script importing its neighbour
    # (`from data import load_mnist`, `import black_box_svi`) is not naming a
    # package anybody can install: measured on autograd, G0 demanded seven such
    # names and was therefore permanently unsatisfiable.
    for p in _py_files(root, limit=2000):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        stem = p.stem
        if stem != "__init__" and stem not in facts.local_modules:
            facts.local_modules.append(stem)
        if p.name == "__init__.py":
            pkg = p.parent.name
            if pkg and pkg not in facts.local_modules:
                facts.local_modules.append(pkg)

    for p in _py_files(root):
        rel = str(p.relative_to(root))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=rel)
        except SyntaxError:
            # A research repo may legitimately carry a file for another Python
            # version. Skipping it is right; failing the whole analysis is not.
            continue

        src = p.read_text(encoding="utf-8", errors="replace")
        has_main = '__main__' in src
        if has_main or "add_argument" in src:
            facts.entrypoints.append(EntryPoint(file=rel, has_main_guard=has_main))
        _Visitor(rel, facts).visit(tree)

    # Rank entry points: an argparse'd script with a __main__ guard at the top of
    # the tree is a likelier experiment than a helper buried three levels down.
    for ep in facts.entrypoints:
        ep.score = (
            (40 if ep.has_main_guard else 0)
            + min(len(ep.options), 20) * 2
            - str(ep.file).count("/") * 5
            + (15 if re.search(r"(train|main|run|experiment|eval)", ep.file) else 0)
            - (30 if re.match(r"(tests?|docs?|examples?)/", ep.file) else 0)
        )
    facts.entrypoints.sort(key=lambda e: -e.score)
    facts.entrypoints = facts.entrypoints[:15]

    for name in ("requirements.txt", "requirements-dev.txt", "pyproject.toml",
                 "setup.py", "setup.cfg", "environment.yml", "Pipfile"):
        f = root / name
        if f.exists():
            facts.requirements.append(name)

    for pat in ("*.yaml", "*.yml", "*.json", "*.toml", "*.ini", "*.cfg"):
        for f in sorted(root.rglob(pat)):
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            rel = str(f.relative_to(root))
            if rel.startswith(("configs/", "config/", "conf/")) or "config" in rel.lower():
                facts.config_files.append(rel)
            if len(facts.config_files) >= 60:
                break

    for name in ("README.md", "README.rst", "README.txt", "readme.md"):
        f = root / name
        if f.exists():
            text = f.read_text(encoding="utf-8", errors="replace")
            facts.readme_commands = [
                m.group(1).strip() for m in _README_CMD.finditer(text)
            ][:40]
            break

    # Literal paths that look like inputs rather than outputs.
    sink_paths = {s.path for s in facts.sinks}
    for p in _py_files(root, limit=200):
        for m in re.finditer(r"['\"]([\w./-]*(?:data|dataset|checkpoints?|weights|"
                             r"pretrained|ckpt)[\w./-]*)['\"]", p.read_text(
                                 encoding="utf-8", errors="replace")):
            v = m.group(1)
            if v not in sink_paths and v not in facts.data_paths and "/" in v:
                facts.data_paths.append(v)
            if len(facts.data_paths) >= 40:
                break

    return facts


_COMMAND_SCRIPT = re.compile(r"(?<![\w/.-])([\w./-]+\.py)(?![\w/.-])")
_COMMAND_MODULE = re.compile(r"(?:^|\s)-m\s+([\w.]+)")


def command_files(facts: RepoFacts, command: str) -> list[str]:
    """Repository files the frozen command executes: named scripts, `-m` modules
    and their repository-local import closure. Empty when nothing in the command
    resolves to a file the analysis saw (a bare `make test`, say)."""
    known = set(facts.printed_format_files and
                {f for files in facts.printed_format_files.values() for f in files}) | set(
        facts.local_import_edges)
    for files in facts.import_files.values():
        known.update(files)
    seeds: list[str] = []
    for m in _COMMAND_SCRIPT.finditer(command or ""):
        rel = m.group(1).lstrip("./")
        if rel in known or (Path(facts.root) / rel).is_file():
            if rel not in seeds:
                seeds.append(rel)
    for m in _COMMAND_MODULE.finditer(command or ""):
        base = m.group(1).replace(".", "/")
        for rel in (f"{base}.py", f"{base}/__main__.py", f"{base}/__init__.py"):
            if rel in known or (Path(facts.root) / rel).is_file():
                if rel not in seeds:
                    seeds.append(rel)
                break
    closure: list[str] = []
    todo = list(seeds)
    while todo:
        rel = todo.pop(0)
        if rel in closure:
            continue
        closure.append(rel)
        parent = str(Path(rel).parent)
        for mod in facts.local_import_edges.get(rel, []):
            if mod not in facts.local_modules:
                continue
            for cand in (f"{mod}.py", f"{mod}/__init__.py",
                         f"{parent}/{mod}.py" if parent != "." else f"{mod}.py",
                         f"{parent}/{mod}/__init__.py" if parent != "." else f"{mod}/__init__.py"):
                if cand in known or (Path(facts.root) / cand).is_file():
                    if cand not in closure and cand not in todo:
                        todo.append(cand)
                    break
    return closure


def printed_formats_for(facts: RepoFacts, command: str) -> list[str]:
    """The print formats a stdout regex must match for THIS command.

    Repository-wide formats let an invented regex slip through whenever some
    other script happens to print the shape the model guessed. When the command
    names files the analysis saw, only their (transitively imported) formats
    count; otherwise the repository-wide list is the best available evidence.
    """
    files = set(command_files(facts, command))
    if not files:
        return list(facts.printed_formats)
    scoped = [fmt for fmt in facts.printed_formats
              if files.intersection(facts.printed_format_files.get(fmt, ()))]
    return scoped or list(facts.printed_formats)
