"""Entry smoke for ``environment_run``: find the generated repository's entry script and run it once.

Record only (PLAN-2 D0). The phase never fails on this result; it exists so
every run says whether the repository *starts*, which neither ``compileall``
nor a discovered test command answers.

Entry resolution, in order:

1. The blueprint. A line inside ``file_structure`` or ``implementation_strategy``
   that names a ``.py`` file and says "entry" (``main.py  # entry point``,
   ``python run.py is the entry``). The file must exist under the code
   directory (a leading project-name segment from the tree is tolerated).
2. The code directory root, then the unique child project directory:
   ``main.py``, ``run*.py``, ``train.py``, ``experiment*.py``, in that order.
3. Nothing: ``entry: None`` and ``status: "no_entry"``.

The smoke is ``python <entry> --help`` (60 s); when that exits non-zero the
script runs bare (60 s). A timeout counts as not started.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.execution.port import Job

ENTRY_HELP_TIMEOUT_S = 60.0
ENTRY_RUN_TIMEOUT_S = 60.0
ENTRY_LABEL = "environment_run:entry"
ENTRY_BARE_LABEL = "environment_run:entry_bare"
ENTRY_PATTERNS: tuple[str, ...] = ("main.py", "run*.py", "train.py", "experiment*.py")
SKIP_DIRS = frozenset({"tests", "test", "__pycache__", ".git"})
BLUEPRINT_KEYS = ("file_structure", "implementation_strategy")

_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?P<key>file_structure|implementation_strategy)\s*:")
_ENTRY_WORD = re.compile(r"entry|入口", re.IGNORECASE)
_PY_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.py\b")
_TREE_CHARS = re.compile(r"[│├└─`*]")


@dataclass(frozen=True, slots=True)
class EntryPoint:
    path: Path  # relative to the code directory
    source: str  # "plan" | "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {"entry": self.path.as_posix(), "entry_source": self.source}


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def blueprint_blocks(plan_text: str) -> list[str]:
    """The lines under ``file_structure:`` and ``implementation_strategy:`` (block scalars or lists)."""
    lines = (plan_text or "").splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        match = _KEY_LINE.match(lines[i])
        if not match:
            i += 1
            continue
        indent = len(match.group("indent"))
        body = [lines[i][match.end():]]
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            body.append(line)
            i += 1
        blocks.append("\n".join(body))
    return blocks


def entry_named_in_plan(plan_text: str) -> str | None:
    """The first ``.py`` path on a blueprint line that says "entry", or ``None``."""
    blocks = blueprint_blocks(plan_text) or [plan_text or ""]
    for block in blocks:
        for raw in block.splitlines():
            if not _ENTRY_WORD.search(raw):
                continue
            line = _TREE_CHARS.sub(" ", raw)
            token = _PY_TOKEN.search(line)
            if token:
                return token.group(0).strip("./") if token.group(0).startswith("./") else token.group(0)
    return None


def _resolve_in(code_dir: Path, candidate: str) -> Path | None:
    rel = Path(candidate)
    if (code_dir / rel).is_file():
        return rel
    if len(rel.parts) > 1 and (code_dir / Path(*rel.parts[1:])).is_file():
        return Path(*rel.parts[1:])
    hits = sorted(
        p.relative_to(code_dir)
        for p in code_dir.rglob(rel.name)
        if p.is_file() and not any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(code_dir).parts[:-1])
    )
    return hits[0] if hits else None


def _candidates_in(directory: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in ENTRY_PATTERNS:
        found.extend(sorted(p for p in directory.glob(pattern) if p.is_file()))
    return found


def entry_by_heuristic(code_dir: Path) -> Path | None:
    """``main.py`` / ``run*.py`` / ``train.py`` / ``experiment*.py`` at the root, else in the unique child project."""
    root = _candidates_in(code_dir)
    if root:
        return root[0].relative_to(code_dir)
    children = sorted(p for p in code_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith("."))
    with_entries = [(child, _candidates_in(child)) for child in children]
    with_entries = [(child, found) for child, found in with_entries if found]
    if len(with_entries) == 1:
        return with_entries[0][1][0].relative_to(code_dir)
    return None


def find_entry(plan_text: str, code_dir: Path) -> EntryPoint | None:
    code_dir = Path(code_dir)
    if not code_dir.is_dir():
        return None
    named = entry_named_in_plan(plan_text)
    if named:
        resolved = _resolve_in(code_dir, named)
        if resolved is not None:
            return EntryPoint(resolved, "plan")
    guessed = entry_by_heuristic(code_dir)
    if guessed is not None:
        return EntryPoint(guessed, "heuristic")
    return None


# ---------------------------------------------------------------------------
# the smoke itself
# ---------------------------------------------------------------------------


def entry_command(entry: Path, args: str = "") -> str:
    """``python <name> <args>`` from the entry's own directory (scripts assume their cwd)."""
    invocation = f"python {shlex.quote(entry.name)}" + (f" {args}" if args else "")
    if entry.parent != Path("."):
        return f"cd {shlex.quote(entry.parent.as_posix())} && {invocation}"
    return invocation


async def run_entry_smoke(port: Any | None, code_dir: Path, entry: EntryPoint | None) -> dict[str, Any]:
    """Run the entry through ``port``; ``status`` is ok | failed | timeout | no_entry | no_port."""
    if entry is None:
        return {"entry": None, "entry_source": None, "status": "no_entry", "attempts": []}
    record: dict[str, Any] = {**entry.to_dict(), "status": "no_port", "attempts": []}
    if port is None:
        return record
    plan = ((ENTRY_LABEL, "--help", ENTRY_HELP_TIMEOUT_S), (ENTRY_BARE_LABEL, "", ENTRY_RUN_TIMEOUT_S))
    for label, args, timeout_s in plan:
        command = entry_command(entry.path, args)
        result = await port.run(Job(workspace=Path(code_dir), command=command, timeout_s=timeout_s, label=label))
        record["attempts"].append(
            {
                "label": label,
                "command": command,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_s": round(result.duration_s, 2),
                "machine": result.machine,
                "error": result.error,
                "synced_back": list(result.synced_back),
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
        )
        if result.ok:
            record["status"] = "ok"
            return record
    last = record["attempts"][-1]
    record["status"] = "timeout" if last["timed_out"] else "failed"
    return record


__all__ = [
    "ENTRY_BARE_LABEL",
    "ENTRY_LABEL",
    "ENTRY_PATTERNS",
    "EntryPoint",
    "entry_by_heuristic",
    "entry_command",
    "entry_named_in_plan",
    "find_entry",
    "run_entry_smoke",
]
