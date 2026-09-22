"""Repair rounds (PLAN-3 item 5): the line's own ReAct loop over a criterion the experiment agent could not satisfy.

The experiment agent proves whether the generated code runs; it never changes the code. When the
frozen criterion fails for a reason that lives in the code, this module's agent edits
``generate_code/`` — reading the evidence (the failing tests' tails, the command output), the
files, and the blueprint — and the runner then commits, re-serves and re-judges the *same*
criterion in the *same* container (``adr/0002``, PLAN-3 §0 "修复轮"). Failures that live in the
environment (a third-party module missing, a command not found, an asset absent) are not this
agent's to fix: :func:`classify` sends those back to SetupX for an environment round.

The agent runs on the line's own runner (``runner.PaperAgentRunner``) with six tools: read (numbered
lines), list, ``edit_file`` (one exact, unique ``old_string`` → ``new_string`` replacement, the shape
every current coding agent uses) and ``write_file`` (new files only) inside the code directory, a probe
that executes one command in the held container (checkpointed before the agent starts and rolled back
before judging, so nothing the probe does survives into the judged environment), and ``finish``.
Budget (PLAN-3 §0 "修复 agent 预算"): at most ``MAX_TOOL_CALLS`` tool calls per round, of which at
most ``MAX_PROBES`` probes; one reply at most 32768 tokens; the phase-level token cap is enforced by
the runner across rounds. At ``NUDGE_AT`` calls the tool result carries a reminder to finish.

Why the tools look like this (T2, 2026-09-18, from the six sapg S9 repair rounds): with a whole-file
``write_file`` as the only way to change code, the agent rewrote ``main.py`` two to four times a round
(20–30k characters each), truncated ``algorithm.py`` to a 2k fragment mid-round, and built its own patch
mechanism (``patch.py``, ``main.py.flatpatch``, ``_patch_import.txt``) that went into the product; not
one round called ``finish`` — all six ended on the call budget with whatever state the tree was in;
and every round opened with 15–17 probes before reading a file. Hence: edits instead of rewrites,
``write_file`` for new files only, an extension allowlist, a syntax check on every write, a reminder at
30 calls, the traceback frames' source in the prompt so the first probes are not spent looking.

The agent never sees the rubric (it is not in the workspace), never touches the blueprint or the
criterion, and cannot install anything into the judged environment. It edits code only: when it
concludes the failure lives in the environment it says so through ``finish(attribution=
"environment")`` and the loop routes the next round to SetupX (PLAN-3 §2.3 S1). A changed
``requirements.txt`` on its own no longer forces an environment round.
"""

from __future__ import annotations

import difflib
import json
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.tools.filesystem import resolve_within
from apps.v2.agent.paper2code.tools.registry import FunctionTool, denylist_hit, denylist_refusal, url_like_values
from apps.v2.agent_engine.paper2code.seams.agent_runtime import AgentRunSpec, ToolRegistry

MAX_TOOL_CALLS = 40
#: probes per round (of the tool calls above). sapg-2 GPU run 6 (2026-09-18): both repair rounds spent their 40 calls on
#: 28 and 34 probes and the second wrote nothing — the budget must leave room for reading, writing and finishing
MAX_PROBES = 15
#: tool call at which the tool result reminds the agent to finish (S9: no round of six called ``finish``)
NUDGE_AT = 30
MAX_TOOL_RESULT_CHARS = 12_000
MAX_WRITE_CHARS = 400_000
PROBE_TIMEOUT_S = 300
EVIDENCE_TAIL_CHARS = 6_000
FILE_HINT_LIMIT = 40
#: files the agent may create or edit; the S9 junk (``.flatpatch``, ``patch.py`` as a tool) had no reason to exist
#: once edits are a tool, so this only fences the obvious non-source shapes
SOURCE_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".txt", ".md", ".sh")
#: source lines shown around each traceback frame that points into the repository
FRAME_CONTEXT_LINES = 20
FRAME_EXCERPT_CHARS = 8_000
#: where the judged container mounts the repository (``repair_loop.CONTAINER_REPO``); tracebacks name files under it
CONTAINER_REPO_PREFIX = "/workspace/repo/"

ENVIRONMENT = "environment"
CODE = "code"
#: a third routing kind (never an agent attribution): the failure says the machine has no GPU and the code wants one —
#: on a CPU machine the controller stops for the compute escalation instead of asking the agent to make the code
#: run on CPU (owner, 2026-09-18 afternoon: changing the method's device is not a repair)
GPU = "gpu"
ATTRIBUTIONS = (CODE, ENVIRONMENT)
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}

_MODULE_NOT_FOUND = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_\.]+)'")
_IMPORT_ERROR = re.compile(r"ImportError: cannot import name '[^']+' from '([A-Za-z0-9_\.]+)'")
_COMMAND_NOT_FOUND = re.compile(r"(?:command not found|No such file or directory: '?(?:python|pip|bash|sh)[^']*'?|/bin/sh: \d+: [^:]+: not found)")
_ASSET_MISSING = re.compile(r"(?:FileNotFoundError|No such file or directory).*\.(?:pt|pth|ckpt|npz|npy|h5|hdf5|pkl|parquet|csv|tar|zip|bin|safetensors)", re.I)
_CUDA = re.compile(r"(?:CUDA (?:error|unavailable|not available)|Torch not compiled with CUDA|Found no NVIDIA driver|libcuda\.so)", re.I)
#: what SetupX's finish text or the repair agent's summary says when the machine is the wrong kind: a GPU-only build
#: (isaacgym, a CUDA extension) or a run that cannot go without CUDA
_GPU_NEEDED = re.compile(
    r"(?:GPU[- ]only|requires? (?:a |an )?(?:NVIDIA |physical )?GPU|needs? (?:a |an )?(?:NVIDIA )?GPU|without (?:a )?GPU"
    r"|CUDA(?: is)? (?:required|unavailable|not available)|no (?:NVIDIA )?GPU (?:is )?(?:present|available|on this machine)|cannot run on CPU)",
    re.I,
)


def gpu_needed(text: str) -> bool:
    """Whether a worker's own words (SetupX's FINISH, the agent's ``finish`` summary) say the failure needs a GPU."""
    return bool(_GPU_NEEDED.search(text or ""))


def repo_modules(code_dir: Path) -> set[str]:
    """Importable names the repository itself could provide: every directory name on the path to a ``.py``
    file and every module stem, at any depth.

    Generated trees nest (``src/pkg/…``, ``pkg/sub/…``) and the entry command may run from any of those
    directories or with a ``PYTHONPATH``; a missing ``pkg`` in such a tree is the code's problem, not a
    third-party package (PLAN-3 §2.3 S1 ①). Over-inclusion only costs a code round where an environment
    round was due, and the agent's ``finish(attribution="environment")`` corrects that.
    """
    names: set[str] = set()
    root = Path(code_dir)
    if not root.is_dir():
        return names
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        names.update(rel.parts[:-1])
        names.add(rel.stem)
    return names


#: files that are legitimately empty
_EMPTY_OK = {"__init__.py", ".gitkeep", "py.typed"}


def empty_files(code_dir: Path) -> list[dict[str, Any]]:
    """Zero-byte source files in the tree, each with the non-empty files of the same name elsewhere (the engine
    writes the content to one path and leaves the planned twin empty — pinn, 2026-09-18: all three trees,
    baseline included, had a 0-byte ``opt_for_pinns/src/pdes.py`` next to a 3.8 KB ``src/pdes.py``; the trial
    failed on the import and the repair round mis-fixed it). T13."""
    root = Path(code_dir)
    if not root.is_dir():
        return []
    by_name: dict[str, list[Path]] = {}
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        files.append(rel)
        by_name.setdefault(path.name, []).append(rel)
    out: list[dict[str, Any]] = []
    for rel in sorted(files):
        if rel.name in _EMPTY_OK or rel.suffix not in {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".sh", ".txt", ".md"}:
            continue
        if (root / rel).stat().st_size != 0:
            continue
        twins = [str(t) for t in by_name.get(rel.name, []) if t != rel and (root / t).stat().st_size > 0]
        out.append({"path": str(rel), "twins": twins})
    return out


#: where the criterion's own output artifacts live: a ``FileNotFoundError`` on one of these is the run not having
#: produced it (the consequence of the real failure), never a missing input asset
OUTPUT_DIRS = ("workspace/out",)  # matches /workspace/out/… and the compiler's relative workspace/out/… alike


def is_output_path(path: str) -> bool:
    return any(out in str(path or "") for out in OUTPUT_DIRS)


def _asset_missing(text: str) -> bool:
    for match in _ASSET_MISSING.finditer(text):
        if not is_output_path(match.group(0)):
            return True
    return False


def classify(text: str, *, repo_modules_: set[str] = frozenset()) -> tuple[str, str]:
    """``(ENVIRONMENT | CODE | GPU, reason)`` for one failure text (PLAN-3 §0 "每轮失败怎么归因": a heuristic on the
    adjudication output). Pass the run's own failure first (``failure_text`` orders it so): an artifact test's
    "file not found" under the output directory is the run not having got there, not an asset problem
    (sapg-2, 2026-09-18: ``ModuleNotFoundError: sapg.networks`` was routed to SetupX because the artifact tail
    named ``/workspace/out/smoke/checkpoint_final.pt``)."""
    text = text or ""
    for match in _MODULE_NOT_FOUND.finditer(text):
        top = match.group(1).split(".")[0]
        if top not in repo_modules_:
            return ENVIRONMENT, f"third-party module missing: {match.group(1)}"
    for match in _IMPORT_ERROR.finditer(text):
        top = match.group(1).split(".")[0]
        if top not in repo_modules_:
            return ENVIRONMENT, f"third-party import failed: {match.group(1)}"
    if _COMMAND_NOT_FOUND.search(text):
        return ENVIRONMENT, "command not found"
    # before the code's own errors: a run that wants CUDA on a CPU machine is the machine's problem whatever else
    # is broken — repairing the rest here would be repaired again on the GPU machine (T3b, 2026-09-18)
    if _CUDA.search(text):
        return GPU, "the code wants CUDA and this machine has no GPU"
    if _MODULE_NOT_FOUND.search(text) or _IMPORT_ERROR.search(text):
        return CODE, "the repository's own module or name is missing"
    if _asset_missing(text):
        return ENVIRONMENT, "an asset (data / weights) is missing"
    return CODE, "no environment signature in the failure"


# ---------------------------------------------------------------------------
# what the agent is told
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are repairing a freshly generated research code repository so that a frozen, mechanical
criterion passes. You cannot change the criterion, the environment, or the blueprint — only the files under the
repository root. The environment was configured by a separate agent and is fixed; anything you pip-install in a
probe is discarded before judging. You change code only. If the failure is genuinely the environment's — a
third-party package or system tool missing, a data or weight file absent, a CUDA-only build on a CPU machine —
do not try to work around it with installs; state it through `finish` with `attribution: "environment"` and a
summary naming exactly what is missing, and the environment agent takes the next round. You may still add the
dependency to requirements.txt so the environment agent sees it, but the file alone changes nothing.

Work in this order. (1) Read: the failure output and the source excerpts below name the files and lines — open
them with `read_text_file` first; do not probe before you have read the failing code. (2) Change: `edit_file`
replaces one exact, unique `old_string` with `new_string` (read the file first; widen the string if it is not
unique); `write_file` creates a new file only. Make the smallest correct change; never reproduce a whole file.
(3) Verify: `run_in_container` runs one command inside the judged container from the repository root (it sees
your files as they are now); say what a probe is meant to verify before running it. Budget: 40 tool calls, of
which at most 15 probes — keep at least 3 probes for after your edit; at tool call 30 you will be reminded to
finish, and the 40th call can only be `finish`. (4) `finish`: call it when the criterion should pass
(`attribution: "code"`, a short summary of what you changed and why), or when you cannot finish in budget — say
so in the summary; a round that ends without `finish` is judged on whatever state the tree is in. Do not add tests; do not change the command-line interface
the criterion invokes unless the failure is in that interface. Keep the paper's method intact — the goal is a
running implementation, not a stub. Never fetch or read code from the URLs listed as off limits."""


_FRAME = re.compile(r'File "([^"]+)", line (\d+)')


def traceback_excerpts(text: str, code_dir: Path, *, context: int = FRAME_CONTEXT_LINES, limit: int = FRAME_EXCERPT_CHARS) -> str:
    """Numbered source around every traceback frame that points into the repository (paths under
    ``CONTAINER_REPO_PREFIX`` or relative to ``code_dir``), innermost frames first, each file:line once. S9's
    agent spent its first 15 probes looking at what the traceback already named; this puts it in the prompt."""
    root = Path(code_dir)
    seen: set[tuple[str, int]] = set()
    parts: list[str] = []
    total = 0
    for raw, line_s in reversed(_FRAME.findall(text or "")):
        rel = raw[len(CONTAINER_REPO_PREFIX):] if raw.startswith(CONTAINER_REPO_PREFIX) else raw
        if rel.startswith("/") or any(part in SKIP_DIRS or part.startswith(".") for part in Path(rel).parts[:-1]):
            continue
        try:
            target = resolve_within(root, rel)
        except PermissionError:
            continue
        if not target.is_file():
            continue
        line_no = int(line_s)
        key = (rel, line_no)
        if key in seen:
            continue
        seen.add(key)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        lo, hi = max(1, line_no - context), min(len(lines), line_no + context)
        body = "\n".join(f"{n:>6}\t{lines[n - 1]}" for n in range(lo, hi + 1))
        part = f"--- {rel} lines {lo}-{hi} (traceback line {line_no}) ---\n{body}"
        if total + len(part) > limit:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def build_messages(
    *,
    goal: str,
    criterion: dict[str, Any] | None,
    verdict: dict[str, Any] | None,
    classification: tuple[str, str],
    code_dir: Path,
    blueprint_excerpt: str = "",
    round_no: int = 1,
    denylist: tuple[str, ...] | list[str] = (),
    previous_summaries: list[str] | None = None,
) -> list[dict[str, Any]]:
    verdict = verdict or {}
    files = sorted(str(p.relative_to(code_dir)) for p in Path(code_dir).rglob("*.py"))[:FILE_HINT_LIMIT]
    failing = [t for t, status in (verdict.get("statuses") or {}).items() if str(status).lower() not in {"passed", "pass"}]
    tails = verdict.get("failure_tails") or {}
    # the run's own failure first; the artifact tests only say what it did not produce
    ordered = sorted(tails.items(), key=lambda item: "artifact" in str(item[0]))
    tail_text = "\n\n".join(f"--- {name} ---\n{str(text)[-EVIDENCE_TAIL_CHARS:]}" for name, text in ordered[:6])
    parts = [
        f"Repair round {round_no}.",
        f"Goal the criterion was compiled from: {goal}",
    ]
    if criterion:
        rungs = criterion.get("rungs") or []
        for rung in rungs:
            parts.append(f"Criterion rung {rung.get('rung')}: command `{rung.get('command')}` in `{rung.get('workdir')}`; expected artifacts: {', '.join(rung.get('artifacts') or []) or '-'}")
    parts.append(f"Last verdict: {verdict.get('verdict', '?')} on rung {verdict.get('rung', '?')} — {verdict.get('passed_expected', 0)}/{verdict.get('expected_n', 0)} expected tests passed; exit code {verdict.get('exit_code')}.")
    if failing:
        parts.append("Failing tests: " + ", ".join(failing[:20]))
    parts.append(f"Line's classification of the failure: {classification[0]} ({classification[1]}).")
    if tail_text:
        parts.append("Failure output:\n" + tail_text)
    elif verdict.get("log_tail"):
        parts.append("Command output tail:\n" + str(verdict["log_tail"])[-EVIDENCE_TAIL_CHARS:])
    excerpts = traceback_excerpts(tail_text or str(verdict.get("log_tail") or ""), code_dir)
    if excerpts:
        parts.append("Source around the traceback frames in the repository (line numbers as read_text_file shows them):\n" + excerpts)
    if previous_summaries:
        parts.append("Earlier repair rounds this run: " + " | ".join(previous_summaries[-3:]))
    if blueprint_excerpt:
        parts.append("Blueprint excerpt (what the code is meant to do):\n" + blueprint_excerpt[:4000])
    empties = empty_files(code_dir)
    if empties:
        parts.append(
            "Files in the repository that are EMPTY (0 bytes) — an import of one of them fails; fill the file, or move the same-named non-empty file's content in: "
            + "; ".join(e["path"] + (f" (same name, non-empty: {', '.join(e['twins'])})" if e["twins"] else "") for e in empties[:10])
        )
    parts.append("Python files in the repository: " + ", ".join(files) + (" …" if len(files) >= FILE_HINT_LIMIT else ""))
    if denylist:
        parts.append("Off limits (never fetch or read): " + ", ".join(denylist))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RepairState:
    written: list[str] = field(default_factory=list)  # created + edited, in first-touch order
    created: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)
    read: set[str] = field(default_factory=set)
    probes: list[dict[str, Any]] = field(default_factory=list)
    finished: str | None = None
    attribution: str = CODE
    tool_calls: int = 0
    nudged: bool = False


def _syntax_error(path: Path, content: str) -> str | None:
    """``compile()`` verdict for a ``.py`` file; None when it compiles (or is not Python). ``compile``, not
    ``ast.parse``: a repeated keyword argument (fre-t14 ``prior.py:503``) parses and only fails code generation."""
    if path.suffix != ".py":
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(content, path.name, "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"{type(exc).__name__} at line {exc.lineno}: {exc.msg}"
    except (ValueError, RecursionError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _closest_line(text: str, needle: str) -> str:
    """The file line most like the first line of ``needle`` — the hint an ``edit_file`` miss returns."""
    first = (needle.strip().splitlines() or [""])[0].strip()
    if not first:
        return ""
    lines = text.splitlines()
    best = difflib.get_close_matches(first, [ln.strip() for ln in lines if ln.strip()], n=1, cutoff=0.5)
    if not best:
        return ""
    for n, ln in enumerate(lines, 1):
        if ln.strip() == best[0]:
            return f"closest line {n}: {ln.strip()[:160]}"
    return ""


def build_tools(
    code_dir: Path,
    *,
    probe: Callable[[str, float], str] | None,
    denylist: tuple[str, ...] | list[str] = (),
    state: RepairState | None = None,
    max_probes: int = MAX_PROBES,
    max_tool_calls: int = MAX_TOOL_CALLS,
    nudge_at: int = NUDGE_AT,
    fidelity: Any | None = None,
) -> tuple[ToolRegistry, RepairState]:
    """``fidelity`` (a ``source_fidelity.FidelitySession``, ADR 0004) makes the write tools honour the reading
    obligations: a file the plan points at can only be changed after its bound sections were read with ``read_paper``
    in an earlier model turn (the caller's hook bumps the turn per model response), and every change is recorded in
    the trace. Without it the tools are the plain T2 set."""
    root = Path(code_dir)
    state = state or RepairState()
    denylist = tuple(denylist)

    def _count() -> str:
        """One tool call spent; at ``nudge_at`` the result carries the reminder to finish (once)."""
        state.tool_calls += 1
        if state.tool_calls == nudge_at and not state.nudged:
            state.nudged = True
            return f"\n\n[budget: {state.tool_calls} of {max_tool_calls} tool calls used — finish now: verify what you changed with one probe if you have not, then call finish with a summary]"
        return ""

    def _reserved() -> str | None:
        """The last call of the budget is ``finish``'s (T2 rerun 12:20: both 40-call rounds read the reminder at 30
        and kept editing into the cap; the tree was judged without a word from the agent)."""
        if state.tool_calls >= max_tool_calls - 1 and state.finished is None:
            # not counted: the slot stays finish's (the runner's own iteration cap bounds repeated refusals)
            return f"Error: this is the last of {max_tool_calls} tool calls — only finish is allowed now; summarise what you changed and what is unverified"
        return None

    def _rel(target: Path) -> str:
        return str(target.relative_to(root.resolve()))

    def _touch(target: Path, *, created: bool) -> str:
        rel = _rel(target)
        if rel not in state.written:
            state.written.append(rel)
        bucket = state.created if created else state.edited
        if rel not in bucket:
            bucket.append(rel)
        return rel

    async def read_text_file(path: str, head: int | None = None, tail: int | None = None) -> str:
        reserved = _reserved()
        if reserved:
            return reserved
        nudge = _count()
        try:
            target = resolve_within(root, path)
            if not target.is_file():
                return f"Error: not a file: {path}" + nudge
            text = target.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, OSError) as exc:
            return f"Error: {exc}" + nudge
        state.read.add(_rel(target))
        lines = list(enumerate(text.splitlines(), 1))
        if head is not None:
            lines = lines[: max(int(head), 0)]
        elif tail is not None:
            lines = lines[-max(int(tail), 0) :] if int(tail) > 0 else []
        text = "\n".join(f"{n:>6}\t{line}" for n, line in lines)
        return (text if len(text) <= MAX_TOOL_RESULT_CHARS else text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]") + nudge

    async def list_directory(path: str = ".") -> str:
        reserved = _reserved()
        if reserved:
            return reserved
        nudge = _count()
        try:
            target = resolve_within(root, path)
            if not target.is_dir():
                return f"Error: not a directory: {path}" + nudge
            entries = sorted(target.iterdir(), key=lambda p: p.name)
        except (PermissionError, OSError) as exc:
            return f"Error: {exc}" + nudge
        return ("\n".join(f"[DIR] {p.name}" if p.is_dir() else f"[FILE] {p.name}" for p in entries if p.name not in {"__pycache__", ".git"}) or "(empty directory)") + nudge

    def _check_content(target: Path, content: str) -> str | None:
        if len(content) > MAX_WRITE_CHARS:
            return f"Error: content too large ({len(content)} chars)"
        hit = denylist_hit(denylist, [content] if url_like_values({"content": content}) else [])
        if hit is not None:
            return denylist_refusal(hit[1])
        return None

    async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        reserved = _reserved()
        if reserved:
            return reserved
        nudge = _count()
        try:
            target = resolve_within(root, path)
        except PermissionError as exc:
            return f"Error: {exc}" + nudge
        if not target.is_file():
            return f"Error: not a file: {path} (write_file creates new files)" + nudge
        rel = _rel(target)
        if rel not in state.read:
            return f"Error: read {rel} with read_text_file before editing it" + nudge
        if not old_string:
            return "Error: old_string is empty" + nudge
        if old_string == new_string:
            return "Error: old_string and new_string are the same" + nudge
        refused = _check_content(target, new_string)
        if refused:
            return refused + nudge
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error: {exc}" + nudge
        n = text.count(old_string)
        if n == 0:
            hint = _closest_line(text, old_string)
            return f"Error: old_string not found in {rel}" + (f" ({hint})" if hint else "") + "; re-read the file and copy the text exactly" + nudge
        if n > 1 and not replace_all:
            return f"Error: old_string occurs {n} times in {rel}; include more surrounding lines to make it unique, or pass replace_all=true" + nudge
        updated = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        err = _syntax_error(target, updated)
        if err:
            return f"Error: the edit would leave {rel} unparsable ({err}); nothing was changed" + nudge
        receipts, refused = _authorize(rel)
        if refused:
            return refused + nudge
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"Error: {exc}" + nudge
        _touch(target, created=False)
        _written(rel, receipts)
        return json.dumps({"status": "ok", "path": rel, "replacements": n if replace_all else 1, "chars": len(updated)}) + nudge

    async def write_file(path: str, content: str) -> str:
        reserved = _reserved()
        if reserved:
            return reserved
        nudge = _count()
        try:
            target = resolve_within(root, path)
        except PermissionError as exc:
            return f"Error: {exc}" + nudge
        if target.exists():
            return f"Error: {path} exists; change it with edit_file (write_file creates new files only)" + nudge
        if target.suffix.lower() not in SOURCE_SUFFIXES:
            return f"Error: {path} is not a source file (allowed: {', '.join(SOURCE_SUFFIXES)})" + nudge
        refused = _check_content(target, content)
        if refused:
            return refused + nudge
        err = _syntax_error(target, content)
        if err:
            return f"Error: {path} would not parse ({err}); nothing was written" + nudge
        receipts, refused = _authorize(_rel(target)) if fidelity is not None else ([], None)
        if refused:
            return refused + nudge
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"Error: {exc}" + nudge
        rel = _touch(target, created=True)
        state.read.add(rel)
        _written(rel, receipts)
        return json.dumps({"status": "ok", "path": rel, "chars": len(content)}) + nudge

    def _authorize(rel: str) -> tuple[list[int], str | None]:
        """Receipts for a write under the fidelity session, or the refusal text (read-before-write, planned paths)."""
        if fidelity is None:
            return [], None
        try:
            return fidelity.authorize(rel), None
        except Exception as exc:  # FidelityError: SOURCE_READ_REQUIRED / SOURCE_FILE_UNPLANNED / …
            return [], f"Error: {exc}"

    def _written(rel: str, receipts: list[int]) -> None:
        if fidelity is not None:
            fidelity.written(rel, receipts)

    async def read_paper(file_path: str = "", section: str = "", part: int = 1) -> str:
        reserved = _reserved()
        if reserved:
            return reserved
        nudge = _count()
        if fidelity is None:
            return "Error: no paper is attached to this round" + nudge
        try:
            text = fidelity.read(file_path=file_path, section=section, part=int(part or 1))
        except Exception as exc:  # FidelityError
            return f"Error: {exc}" + nudge
        return (text if len(text) <= MAX_TOOL_RESULT_CHARS else text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]") + nudge

    async def run_in_container(command: str, timeout: int = PROBE_TIMEOUT_S) -> str:
        reserved = _reserved()
        if reserved:
            return reserved
        nudge = _count()
        if probe is None:
            return json.dumps({"status": "error", "message": "no container is attached; reason from the files and the evidence"}) + nudge
        if len(state.probes) >= max_probes:
            return json.dumps({"status": "error", "message": f"probe budget of {max_probes} for this round is used; make your change with edit_file and call finish"}) + nudge
        hit = denylist_hit(denylist, url_like_values({"command": command}))
        if hit is not None:
            return denylist_refusal(hit[1]) + nudge
        timeout_s = float(min(max(int(timeout), 1), PROBE_TIMEOUT_S))
        try:
            output = probe(command, timeout_s)
        except Exception as exc:
            output = f"probe failed: {type(exc).__name__}: {exc}"
        state.probes.append({"command": command[:300], "output_chars": len(output or ""), "after_write": bool(state.written)})
        output = output or ""
        return (output if len(output) <= MAX_TOOL_RESULT_CHARS else output[:MAX_TOOL_RESULT_CHARS // 2] + "\n[...]\n" + output[-MAX_TOOL_RESULT_CHARS // 2 :]) + nudge

    async def finish(summary: str, attribution: str = CODE) -> str:
        _count()
        attribution = str(attribution or CODE).strip().lower()
        if attribution not in ATTRIBUTIONS:
            return f"Error: attribution must be one of {list(ATTRIBUTIONS)}, got {attribution!r}"
        state.finished = (summary or "").strip()[:2000]
        state.attribution = attribution
        return "recorded; stop calling tools and reply with the same summary"

    registry = ToolRegistry()
    for tool in (
        FunctionTool(name="read_text_file", description="Read a file under the repository root with line numbers (optionally only the first `head` or last `tail` lines). Read a file before editing it.", parameters={"type": "object", "properties": {"path": {"type": "string"}, "head": {"type": "integer"}, "tail": {"type": "integer"}}, "required": ["path"]}, fn=read_text_file, read_only=True),
        FunctionTool(name="list_directory", description="List a directory under the repository root.", parameters={"type": "object", "properties": {"path": {"type": "string", "default": "."}}}, fn=list_directory, read_only=True),
        FunctionTool(name="edit_file", description="Replace one exact occurrence of `old_string` with `new_string` in an existing file (copy old_string verbatim from read_text_file, without the line numbers; it must be unique unless replace_all is true). A Python file that would not parse after the edit is left unchanged.", parameters={"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean", "default": False}}, "required": ["path", "old_string", "new_string"]}, fn=edit_file),
        FunctionTool(name="write_file", description="Create a new source file under the repository root (creates parents). Refuses paths that already exist — use edit_file for those — and non-source extensions.", parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, fn=write_file),
        FunctionTool(name="run_in_container", description="Run one shell command inside the judged container from the repository root (your files are synced first); anything you install here is discarded before judging. Read the failing code before probing; say what the probe verifies.", parameters={"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": PROBE_TIMEOUT_S}}, "required": ["command"]}, fn=run_in_container, timeout_s=float(PROBE_TIMEOUT_S + 30)),
        FunctionTool(name="finish", description="Declare the round done. attribution='code' (default): you changed the code and the criterion should pass now. attribution='environment': the failure is the environment's (a missing package, tool, asset, or a GPU-only build); say exactly what is missing in the summary and the environment agent takes the next round.", parameters={"type": "object", "properties": {"summary": {"type": "string"}, "attribution": {"type": "string", "enum": list(ATTRIBUTIONS), "default": CODE}}, "required": ["summary"]}, fn=finish),
    ):
        registry.register(tool)
    if fidelity is not None:
        registry.register(FunctionTool(name="read_paper", description="Read one page of a paper section (ADR 0004): file_path = the file you are about to change → its next unread page of its bound sections (the plan's `Source:` line), with what is still unread; or section = any section; part = page. A bound file can only be edited in a LATER model turn than its reads, after every page of every bound section; SOURCE_READ_REQUIRED on an edit lists the pages still unread.", parameters={"type": "object", "properties": {"file_path": {"type": "string"}, "section": {"type": "string"}, "part": {"type": "integer", "default": 1}}}, fn=read_paper, read_only=True))
    return registry, state


# ---------------------------------------------------------------------------
# one round
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RepairOutcome:
    summary: str
    written: list[str]
    probes: int
    tool_calls: int
    stop_reason: str
    usage: dict[str, int]
    error: str | None = None
    attribution: str = CODE
    finished: bool = False  # the agent called ``finish`` (a round that ends on the budget is judged as it stands)
    created: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)
    probes_after_write: int = 0  # probes run after the first change — the verification the prompt asks for

    def record(self) -> dict[str, Any]:
        return {
            "summary": self.summary, "written": list(self.written), "created": list(self.created), "edited": list(self.edited),
            "probes": self.probes, "probes_after_write": self.probes_after_write, "tool_calls": self.tool_calls,
            "finished": self.finished, "stop_reason": self.stop_reason, "usage": dict(self.usage), "error": self.error,
            "attribution": self.attribution,
        }


async def run_repair_round(
    provider: Any,
    *,
    model: str,
    code_dir: Path,
    messages: list[dict[str, Any]],
    probe: Callable[[str, float], str] | None,
    denylist: tuple[str, ...] | list[str] = (),
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_tokens: int = 32768,
    runner: Any | None = None,
    fidelity: Any | None = None,
    hook: Any | None = None,
) -> RepairOutcome:
    """One pass of the repair agent; files are changed in place, nothing is committed here. ``fidelity`` /
    ``hook``: see :func:`build_tools` — the hook is what bumps the session's turn per model response."""
    from apps.v2.agent.paper2code.runner import PaperAgentRunner

    tools, state = build_tools(Path(code_dir), probe=probe, denylist=denylist, max_tool_calls=max_tool_calls, fidelity=fidelity)

    async def should_stop() -> str | None:
        if state.finished is not None:
            return "finish called"
        if state.tool_calls >= max_tool_calls:
            return f"tool-call budget of {max_tool_calls} reached"
        return None

    spec = AgentRunSpec(
        initial_messages=messages,
        tools=tools,
        model=model,
        max_iterations=max_tool_calls + 5,
        max_tool_result_chars=MAX_TOOL_RESULT_CHARS,
        max_tokens=max_tokens,
        workspace=Path(code_dir),
        should_stop_callback=should_stop,
        hook=hook,
    )
    agent_runner = runner or PaperAgentRunner(provider)
    result = await agent_runner.run(spec)
    summary = state.finished or (result.final_content or "").strip()[:2000]
    return RepairOutcome(
        summary=summary,
        written=list(state.written),
        probes=len(state.probes),
        tool_calls=state.tool_calls,
        stop_reason=result.stop_reason,
        usage=dict(result.usage or {}),
        error=result.error,
        attribution=state.attribution,
        finished=state.finished is not None,
        created=list(state.created),
        edited=list(state.edited),
        probes_after_write=sum(1 for pr in state.probes if pr.get("after_write")),
    )


__all__ = [
    "ATTRIBUTIONS",
    "CODE",
    "ENVIRONMENT",
    "GPU",
    "MAX_PROBES",
    "MAX_TOOL_CALLS",
    "NUDGE_AT",
    "OUTPUT_DIRS",
    "SOURCE_SUFFIXES",
    "SYSTEM_PROMPT",
    "RepairOutcome",
    "RepairState",
    "build_messages",
    "build_tools",
    "classify",
    "empty_files",
    "gpu_needed",
    "is_output_path",
    "repo_modules",
    "run_repair_round",
    "traceback_excerpts",
]
