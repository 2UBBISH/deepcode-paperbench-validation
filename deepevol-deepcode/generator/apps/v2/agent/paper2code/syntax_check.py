"""Static syntax check of the generated tree after implement (owner 09-20): ``compile()`` every ``.py`` under
``generate_code/`` — parsing only, nothing is imported or run — and let the repair agent fix what does not parse.

Why: the coding agent writes each file once and never runs anything (its tools are write_file, search_code_references
and read_paper), so a duplicated keyword argument or an unbalanced bracket survives into the stage-9 tree and the judge
docks every leaf that file carries (fre-t14: ``prior.py:503`` ``torch.rand(..., device="cpu", device=device)``). The desktop
arms may run ``py_compile`` themselves; this is the line's equivalent — a static check, not execution (the batch rule
is "quick checks yes, experiments no").

``DEEPCODE_SYNTAX_CHECK`` (default on) gates it; ``DEEPCODE_SYNTAX_ROUNDS`` (default 2) bounds the repair rounds. Each
round is one :func:`repair.run_repair_round` with no container attached (``probe=None``): the agent reads the failing
files, fixes them with ``edit_file`` (which itself refuses an edit that would not parse) and calls ``finish``; the
harness re-compiles after every round. Remaining errors are recorded, not fatal — the tree is judged as it is.

Under paper fidelity (ADR 0004, ``task_dir`` given) the round runs inside a ``FidelitySession``: a file the plan
points at can only be edited after ``read_paper`` exposed its bound sections in an earlier model turn (``_TurnHook``
bumps the turn per model response) and every edit is recorded in the trace — so the audit that follows in
``phase_implement`` sees the repaired bytes as recorded writes.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code import repair
from apps.v2.agent_engine.paper2code.seams.agent_runtime import AgentHook, AgentHookContext

ENV_ENABLED = "DEEPCODE_SYNTAX_CHECK"
ENV_ROUNDS = "DEEPCODE_SYNTAX_ROUNDS"
DEFAULT_ROUNDS = 2
MAX_TOOL_CALLS = 20
CONTEXT_LINES = 6
ERROR_LIMIT = 12

SYSTEM_PROMPT = """You are fixing syntax errors in a freshly generated research code repository. The files were written
without ever being run; the harness compiled every Python file (parse only, nothing executed) and lists the ones that do
not parse, with the error and the surrounding lines. Fix each listed file with the smallest correct change: open it with
`read_text_file`, then `edit_file` one exact, unique `old_string` → `new_string` (the tool refuses an edit that would still
not parse). Do not rewrite files, do not change behaviour beyond what the error requires, do not add tests. There is no
container: `run_in_container` does nothing here — the harness re-compiles after you call `finish`. Budget: 20 tool calls.
When every listed file parses (or you cannot fix one), call `finish` with `attribution: "code"` and a one-line summary."""

FIDELITY_RULE = """

Source fidelity is on: a file listed below with `[sections: …]` is bound to those paper sections. Before editing it, call
`read_paper(file_path=<that file>)` repeatedly until it reports nothing unread (every page of every bound section) and wait
for the text; the edit must come in a later reply than the reads (an edit in the same reply is refused with
SOURCE_READ_REQUIRED). Files marked glue need no read."""


def enabled() -> bool:
    return os.environ.get(ENV_ENABLED, "1").strip().lower() not in {"0", "false", "no", "off"}


def rounds() -> int:
    try:
        return max(0, int(os.environ.get(ENV_ROUNDS, str(DEFAULT_ROUNDS))))
    except ValueError:
        return DEFAULT_ROUNDS


def compile_tree(code_dir: Path) -> list[dict[str, Any]]:
    """Every ``.py`` under ``code_dir`` that does not parse: path (relative), line, column, message, excerpt.
    Empty files are the T13 check's business (they parse); byte-level decoding failures count as errors."""
    root = Path(code_dir)
    errors: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in repair.SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        rel = str(path.relative_to(root))
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append({"path": rel, "line": 0, "col": 0, "message": f"cannot read: {exc}", "excerpt": ""})
            continue
        try:
            # compile(), not ast.parse(): a repeated keyword argument, `return` outside a function or `await` outside
            # async are code-generation errors that parse fine (fre-t14's prior.py:503 parses). Nothing is executed.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # invalid escape sequences etc. are warnings, not errors
                compile(source, rel, "exec", dont_inherit=True)
        except SyntaxError as exc:  # covers IndentationError / TabError
            line = int(exc.lineno or 0)
            errors.append({
                "path": rel, "line": line, "col": int(exc.offset or 0), "message": f"{type(exc).__name__}: {exc.msg}",
                "excerpt": _excerpt(source, line),
            })
        except (ValueError, RecursionError) as exc:  # null bytes, absurd nesting
            errors.append({"path": rel, "line": 0, "col": 0, "message": f"{type(exc).__name__}: {exc}", "excerpt": ""})
    return errors


def _excerpt(source: str, line: int, context: int = CONTEXT_LINES) -> str:
    lines = source.splitlines()
    if not line or not lines:
        return ""
    lo, hi = max(1, line - context), min(len(lines), line + context)
    return "\n".join(f"{n:5d}{'>' if n == line else ' '} {lines[n - 1]}" for n in range(lo, hi + 1))


def build_messages(
    errors: list[dict[str, Any]], *, round_no: int, denylist: tuple[str, ...] | list[str] = (), manifest: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    parts = [f"Syntax round {round_no}: {len(errors)} file(s) do not parse."]
    for err in errors[:ERROR_LIMIT]:
        binding = ""
        if manifest is not None:
            sections = ((manifest.get("files") or {}).get(err["path"]) or {}).get("sections") or []
            binding = f"\n[sections: {', '.join(sections)}]" if sections else "\n[glue file: no paper read needed]"
        parts.append(f"--- {err['path']}:{err['line']}:{err['col']} — {err['message']}{binding}\n{err['excerpt']}".rstrip())
    if len(errors) > ERROR_LIMIT:
        parts.append(f"… and {len(errors) - ERROR_LIMIT} more: " + ", ".join(e["path"] for e in errors[ERROR_LIMIT:]))
    if denylist:
        parts.append("Off limits (never fetch or read): " + ", ".join(denylist))
    system = SYSTEM_PROMPT + (FIDELITY_RULE if manifest is not None else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


class _TurnHook(AgentHook):
    """One model response = one fidelity turn (a read and the edit it authorizes must be in different replies)."""

    def __init__(self, session: Any) -> None:
        super().__init__()
        self._session = session

    async def on_model_response(self, context: AgentHookContext) -> None:
        self._session.next_turn()


async def check_and_repair(
    code_dir: Path,
    *,
    provider_factory: Callable[[], Any] | None,
    model: str,
    max_rounds: int | None = None,
    denylist: tuple[str, ...] | list[str] = (),
    events: Callable[..., Any] | None = None,
    runner: Any | None = None,
    task_dir: Path | None = None,
) -> dict[str, Any]:
    """Compile; while errors remain and rounds are left, run one repair round and compile again.
    ``provider_factory`` ``None`` = check only (offline tests; ``--skip`` style dry runs). ``task_dir`` = the paper task
    directory when paper fidelity is on (its ``source_manifest.json`` / ``source_trace.json`` scope the session)."""
    max_rounds = rounds() if max_rounds is None else max_rounds
    initial = compile_tree(code_dir)
    record: dict[str, Any] = {"initial_errors": initial, "rounds": [], "remaining": initial}
    if events:
        events("implement.syntax", stage="initial", errors=len(initial), files=[e["path"] for e in initial[:ERROR_LIMIT]])
    if not initial or provider_factory is None or max_rounds <= 0:
        return record
    session = hook = manifest = None
    if task_dir is not None:
        from apps.v2.agent_engine.paper2code.workflows.source_fidelity import FidelityError, FidelitySession

        try:
            session = FidelitySession(Path(task_dir), Path(code_dir))
        except FidelityError as exc:
            record["fidelity_error"] = str(exc)
            if events:
                events("implement.syntax", stage="fidelity_unavailable", error=str(exc))
            return record
        hook, manifest = _TurnHook(session), session.manifest
    remaining = initial
    provider = provider_factory()
    for round_no in range(1, max_rounds + 1):
        outcome = await repair.run_repair_round(
            provider, model=model, code_dir=Path(code_dir), probe=None, denylist=denylist,
            messages=build_messages(remaining, round_no=round_no, denylist=denylist, manifest=manifest), max_tool_calls=MAX_TOOL_CALLS,
            runner=runner, fidelity=session, hook=hook,
        )
        remaining = compile_tree(code_dir)
        record["rounds"].append({"round": round_no, "outcome": outcome.record(), "errors_after": len(remaining), "files_after": [e["path"] for e in remaining[:ERROR_LIMIT]]})
        if events:
            events("implement.syntax", stage=f"round{round_no}", errors=len(remaining), edited=list(outcome.edited), finished=outcome.finished)
        if not remaining:
            break
    record["remaining"] = remaining
    return record


__all__ = ["ENV_ENABLED", "ENV_ROUNDS", "SYSTEM_PROMPT", "build_messages", "check_and_repair", "compile_tree", "enabled", "rounds"]
