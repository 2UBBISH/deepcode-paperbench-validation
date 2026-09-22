"""Step 10 as one call to main's experiment agent, with the line's runner injected (PLAN-3 item 4, ``adr/0002``).

``run_experiment_on_machine`` (main) owns the sequence rent → serve the code → run RSA → decide
about the machine; the line contributes four things it cannot know:

* the **goal** RSA compiles its criterion from (:func:`build_goal`): a basic template — the
  entry command, whether there is a GPU, the output directory, the URLs it may not fetch, the
  external tools that are not available, minimal scale. It only measures "the environment is up
  and the entry runs at minimal scale"; the blueprint's ``validation_approach`` is *not* in it
  (owner, 2026-09-17: the real run criteria come from a separate module, PLAN-3 §0 "目标句");
* the **controller** (S8, ``environment_controller`` + ``repair_loop.run_line_pipeline``): RSA's
  ``run_pipeline`` — asset gate then its Router — is replaced for the duration of the run by the
  line's mechanical scheduler: 搭建环境 (SetupX, black box) and 远程初步执行 (adjudicate ≤ G2) as two
  separate calls from round 0 on, 修复代码 (the repair agent) when the failure is the code's, the
  next box decided by the failure signature or the agent's attribution; RSA's compile / falsify /
  asset gate and its interaction cards are untouched;
* the **runner** (:class:`LineRsaRunner`): RSA driven with an interaction handler that answers
  from a decision file (``--ask``), or by the unattended defaults (PLAN-3 §0 "无人值守时 RSA
  三张卡"), or — under ``--ask`` with no decision yet — hands the pending question back so the
  flow holds the machine and the phase waits; on success it commits the configured container
  to an image and keeps the SET_ENV values and SetupX's own logs (the command ledger) beside
  the run;
* the **request/decision files** in DeepEvol's ``ask_user`` question shape, one pair for the
  whole phase (``10_environment_run.request.json`` / ``.decision.json``), matched by the kind of
  question RSA asked;
* the **denylist audit** over everything the agent did on the machine: a run whose ledger
  names a denylisted URL is marked ``denylist_touched`` and is void for any comparison.

Nothing here changes RSA or SetupX; the runner reuses the flow's own seams (durable backend,
round recorder, integrity guard) exactly as ``_default_rsa_runner`` does.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shlex
import shutil
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

REQUEST_FILE = "10_environment_run.request.json"
DECISION_FILE = "10_environment_run.decision.json"
STATE_FILE = "10_environment_run.state.json"
#: two-stage compute: written when the CPU stage stopped for a GPU; its presence makes the phase run on the GPU tier
ESCALATION_FILE = "10_environment_run.escalation.json"
ENVIRONMENT_FILE = "environment.json"
RSA_DIRNAME = "rsa"
SETUPX_LOGS_DIRNAME = "setupx-logs"
ENV_IMAGE_PREFIX = "paper2code-env"

T = TypeVar("T")

KINDS = ("clarification", "asset", "approval", "criterion_review", "escalation", "repair_review")
#: SetupX budgets for round 0 (temporary seam adjustment, PLAN-3 §0 "临时调整"): RSA's defaults are 200 steps ×
#: 8 rounds, sized for arbitrary repositories. On sapg-2 (2026-09-17) SetupX had the environment right after
#: ~30 steps and then spent 70+ more diagnosing *code* bugs it is forbidden to fix — steps the repair loop is
#: for. Smaller budgets hand the failure over sooner; ``PAPER2CODE_SETUPX_MAX_STEPS`` / ``_MAX_ROUNDS`` override.
SETUPX_MAX_STEPS = int(os.environ.get("PAPER2CODE_SETUPX_MAX_STEPS", "60"))
SETUPX_MAX_ROUNDS = int(os.environ.get("PAPER2CODE_SETUPX_MAX_ROUNDS", "3"))
#: G1's asset card is never a question (PLAN-3 §0 "资产缺 / fallback", owner 2026-09-17): a declared input or
#: machine asset that is not there means the environment cannot be built — the run records it and fails,
#: with the machine released; no ``drop_assets``, no ``retry``, no holding the machine for an answer
ASSET_POLICY: dict[str, str] = {"action": "stop", "message": "missing assets: the environment cannot be built for this code; recorded and failed (PLAN-3 §0 资产缺)"}
#: what the run does when nobody is asked (PLAN-3 §0): never "approve" anything that spends hours or
#: weakens the criterion; ``clarification`` proceeds on the goal as written
UNATTENDED: dict[str, dict[str, str]] = {
    "clarification": {"action": "answer", "message": "Proceed on the goal exactly as written; there is nothing to add."},
    "asset": dict(ASSET_POLICY),
    "approval": {"action": "stop", "message": "unattended run: rungs beyond G2 are not started (PLAN-3 item 8)"},
    "criterion_review": {"action": "stop", "message": "unattended run: a rejected criterion is recorded, not rewritten"},
    "escalation": {"action": "stop", "message": "unattended run: no further budget is granted"},
}


# ---------------------------------------------------------------------------
# before renting anything: can this process even run RSA?
# ---------------------------------------------------------------------------

#: modules RSA / SetupX import at module top; missing ones only surface after a machine is rented
#: (real machine 2026-09-17: ``ModuleNotFoundError: No module named 'docker'`` two minutes into the lease —
#: the docker SDK sits in main's ``agent-runtime`` extra, which the line's venv did not carry)
REQUIRED_MODULES = ("docker", "dotenv", "httpx", "asyncssh", "apps.v2.agent_engine.rsa.agent", "apps.v2.agent_engine.experiment.run_flow")


def missing_modules(names: tuple[str, ...] = REQUIRED_MODULES) -> list[str]:
    import importlib

    missing: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # ImportError or anything the import raises
            missing.append(f"{name} ({type(exc).__name__}: {str(exc)[:80]})")
    return missing


# ---------------------------------------------------------------------------
# the goal RSA compiles from
# ---------------------------------------------------------------------------


def one_line(text: str) -> str:
    """RSA renders the goal into the frozen criterion as ``# goal: <text>`` — one comment line
    (``rsa/render.py:270``). A goal with a newline puts its second line outside the comment and the
    criterion file no longer parses (real machine 2026-09-17: ``SyntaxError: unterminated string literal``
    on the denylist sentence). So the goal is always one line."""
    return re.sub(r"\s+", " ", text or "").strip()


#: how many options the goal lists before it truncates (the S9 entries declared 40–60)
ENTRY_FLAG_LIMIT = 80


def _literal_values(tree: ast.Module, name: str) -> list[str] | None:
    """Constant strings of a module-level ``name = {…}`` / ``[…]`` / ``(…)`` literal (dict → its keys)."""
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        value = node.value
        items = value.keys if isinstance(value, ast.Dict) else value.elts if isinstance(value, (ast.List, ast.Tuple)) else None
        if items is None:
            return None
        return [i.value for i in items if isinstance(i, ast.Constant) and isinstance(i.value, str)]
    return None


def _imported_from(tree: ast.Module, name: str) -> str | None:
    """``from a.b import name`` → ``"a.b"``; ``from .b import name`` → ``".b"``."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(a.asname == name or (a.asname is None and a.name == name) for a in node.names):
            return "." * (node.level or 0) + (node.module or "")
    return None


def _choices(expr: ast.AST, tree: ast.Module, entry_path: Path, root: Path) -> list[str] | None:
    """The literal ``choices=`` of an ``add_argument``: a list/tuple of strings, or a name (also through ``sorted``,
    ``list``, ``tuple``, ``.keys()``) bound to a module-level literal here or in the module it is imported from."""
    node = expr
    for _ in range(3):  # sorted(list(X.keys())) → X
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"sorted", "list", "tuple"} and len(node.args) == 1:
            node = node.args[0]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "keys" and not node.args:
            node = node.func.value
        else:
            break
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        return values or None
    if not isinstance(node, ast.Name):
        return None
    values = _literal_values(tree, node.id)
    if values is not None:
        return values or None
    module = _imported_from(tree, node.id)
    if module is None:
        return None
    # resolve the module the way the entry's own directory (a script) and the repository root (a package) would
    parts = module.lstrip(".").split(".") if module.lstrip(".") else []
    bases = [entry_path.parent] + ([entry_path.parent.parents[len(module) - len(module.lstrip(".")) - 1]] if module.startswith("..") else []) + [root]
    for base in bases:
        for candidate in (base.joinpath(*parts).with_suffix(".py") if parts else None, base.joinpath(*parts, "__init__.py") if parts else None):
            if candidate is None or not candidate.is_file():
                continue
            try:
                if not candidate.resolve().is_relative_to(root):
                    continue
                other = ast.parse(candidate.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError, ValueError):
                continue
            values = _literal_values(other, node.id)
            if values is not None:
                return values or None
    return None


def entry_flags(code_dir: Path, entry: str | None) -> dict[str, list[str]]:
    """The command-line surface the entry file declares, read statically: every ``add_argument("--x", …)`` option
    string — with its ``choices`` when they are literal (``{a,b,c}`` as argparse's usage prints them; a name bound
    to a module-level dict/list literal here or in the module it is imported from counts) — and every
    ``add_parser("name")`` sub-command in that one file (``ast``; nothing is executed). Options defined in another
    module are not seen — the goal then lists what it can. Used by :func:`build_goal` (T2, 2026-09-18): RSA's
    compiler wrote the S9 commands from the goal alone and invented options (``--pbt_interval``, ``--num-blocks``
    on a parser without them), so the first repair round of both runs went to ``unrecognized arguments``; with
    the option list in the goal it then invented a *value* (``--task dummy`` against ``choices=sorted(TASK_CONFIGS)``,
    T2 rerun 11:40) — hence the choices."""
    empty: dict[str, list[str]] = {"options": [], "subcommands": []}
    if not entry:
        return empty
    try:
        root = Path(code_dir).resolve()
        target = (root / entry).resolve()
        if not target.is_relative_to(root) or not target.is_file() or target.suffix != ".py":
            return empty
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return empty
    options: list[str] = []
    seen: set[str] = set()
    subcommands: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        strings = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if node.func.attr == "add_argument":
            names = [v for v in strings if v.startswith("-") and v not in seen]
            if not names:
                continue
            seen.update(names)
            choices_kw = next((k.value for k in node.keywords if k.arg == "choices"), None)
            choices = _choices(choices_kw, tree, target, root) if choices_kw is not None else None
            suffix = " {" + ",".join(choices) + "}" if choices else ""
            options.extend([*names[:-1], names[-1] + suffix])
        elif node.func.attr == "add_parser" and strings and strings[0] not in subcommands:
            subcommands.append(strings[0])
    return {"options": options, "subcommands": subcommands}


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def declared_requirements(code_dir: Path) -> list[str]:
    """Package names from every ``requirements*.txt`` under the generated tree (first match wins, order kept):
    what the goal tells the compiler the code depends on, so that G0's import check names packages the
    environment has to provide. T5 (pinn, 2026-09-18): with torch pre-installed on the machine (T4) the compiler
    declared ``['opt_for_pinns', 'torch']`` for G0, both resolved in the bare image, and RSA's falsifier rejected
    the criterion twice as "measuring nothing"."""
    names: list[str] = []
    root = Path(code_dir)
    if not root.is_dir():
        return names
    for path in sorted(root.rglob("requirements*.txt"), key=lambda q: (len(q.relative_to(root).parts), str(q))):  # shallowest first
        if any(part in SKIP_DIRS_REQ or part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            m = _REQ_NAME.match(line)
            if m:
                name = m.group(1).lower()
                if name not in names:
                    names.append(name)
    return names


SKIP_DIRS_REQ = {"__pycache__", ".git", ".venv", "venv", "node_modules"}


def build_goal(
    *,
    entry: str | None,
    environment_spec: dict[str, Any] | None,
    denylist: tuple[str, ...] | list[str] = (),
    scale: str = "minimal",
    gpu_available: bool = False,
    flags: dict[str, list[str]] | None = None,
    requirements: list[str] | None = None,
) -> str:
    """The goal handed to RSA's compiler — the basic template (PLAN-3 §0 "目标句", owner 2026-09-17): the entry,
    its declared options (:func:`entry_flags`), GPU or not, the output directory, the denylist, unavailable
    external tools, minimal scale. Nothing from the blueprint's ``validation_approach`` and no fallback
    instructions: the code's own default path decides whether it runs. One line, see :func:`one_line`."""
    spec = environment_spec or {}
    flags = flags or {}
    options = [str(o) for o in (flags.get("options") or [])]
    subcommands = [str(c) for c in (flags.get("subcommands") or [])]
    unavailable = [
        t.get("name") for t in (spec.get("external_tools") or []) if isinstance(t, dict) and t.get("name") and t.get("installable") is not True
    ]
    gpu_line = "A GPU is available and may be used." if gpu_available else "There is no GPU on this machine; everything runs on CPU."
    lines = [
        "Goal: prove that this freshly generated research repository runs end to end at minimal scale "
        "(a handful of steps or iterations, the smallest configuration the code accepts) and leaves its "
        "output artifacts (results, logs, checkpoints or figures) on disk.",
        "",
        f"Entry point: `{entry}`." if entry else "Entry point: the repository's main script (see its README).",
        gpu_line,
    ]
    if requirements:
        lines.append("Third-party packages the code declares (requirements.txt): " + ", ".join(requirements[:40]) + ".")
    if options:
        listed = ", ".join(options[:ENTRY_FLAG_LIMIT]) + (", …" if len(options) > ENTRY_FLAG_LIMIT else "")
        lines.append(
            (f"The entry declares these sub-commands: {', '.join(subcommands)}; and " if subcommands else "The entry declares ")
            + f"these command-line options and no others: {listed}. Use only options from this list, spelled exactly "
            "as listed (hyphens and underscores are not interchangeable); where an option shows its choices in braces, "
            "pass one of those values; never invent an option or a value."
        )
    if unavailable:
        lines.append(
            "These external tools are not available on this machine and must not be installed: "
            + ", ".join(sorted({str(t) for t in unavailable}))
            + "."
        )
    if denylist:
        lines.append("Never fetch, clone or read from these URLs (they are the paper's own code and are off limits): " + ", ".join(denylist) + ".")
    lines.append(f"Scale: {scale}. Do not run full experiments; a criterion that takes hours is not wanted here.")
    # RSA's static falsification treats every relative path-like token of the command as an input that
    # must exist at the frozen commit, exempting only a fixed list of output flags (``--output-dir``…) and
    # absolute paths (``rsa/falsifier.py``). Generated code spells its flags freely (``--output_dir`` on
    # sapg-2 → "does not exist at commit" → the criterion was rejected), so outputs go outside the tree.
    lines.append(
        "Write all outputs to an absolute directory outside the repository, e.g. /workspace/out/smoke "
        "(pass it to whatever output flag the code has), and name the expected artifacts by those absolute paths; "
        "never pass a relative output directory that does not exist in the repository. "
        "The output directory is created for the run and is not an input asset: do not declare it, or anything under it, as an asset. "
        "Expect only the artifacts the entry command itself writes (its log, history, checkpoint or summary); "
        "do not require outputs of other scripts in the repository."
    )
    return one_line(" ".join(line for line in lines if line))


# ---------------------------------------------------------------------------
# request / decision files (DeepEvol ask_user question shape)
# ---------------------------------------------------------------------------


def build_request(event: Any, *, run_id: str, round_no: int) -> dict[str, Any]:
    kind = str(getattr(event, "kind", "") or "")
    message = str(getattr(event, "message", "") or "")
    data = getattr(event, "data", None) or {}
    choices_by_kind = {
        "clarification": ([], True),
        "asset": ([("drop_assets", "Drop the missing assets from the criterion", "the run continues without them; the criterion gets weaker"),
                   ("retry", "Retry the asset check", "after you put the assets in place yourself"),
                   ("stop", "Stop", "record BLOCKED with the evidence and release the machine")], False),
        "approval": ([("approve", "Approve G3/G4'", "hours or days of machine time"), ("stop", "Stop at G2", "the run completes at the minimal-scale rung")], False),
        "criterion_review": ([("stop", "Stop", "record the rejected criterion")], True),
        "escalation": ([("continue", "Continue with more budget", "one more round of environment configuration"),
                        ("approve", "Continue and approve G3/G4'", ""), ("stop", "Stop", "record the escalation card and release the machine")], False),
        "repair_review": ([("accept", "Accept as is", "the last judged state is recorded; the run goes on"),
                           ("continue", "Continue repairing", "another set of repair rounds (the run restarts step 10 from round 0)"),
                           ("abort", "Abort", "release the machine; environment_run fails")], False),
    }
    choices, custom = choices_by_kind.get(kind, ([("stop", "Stop", "")], True))
    question: dict[str, Any] = {
        "header": {"clarification": "回答编译器的问题", "asset": "资产缺失", "approval": "批准更长的验证", "criterion_review": "判据被证伪拒绝", "escalation": "配环境卡住了", "repair_review": "修复轮用尽"}.get(kind, kind),
        "text": message,
        "type": "text" if not choices else "multiple_choice",
        "choices": [{"value": v, "label": label, "description": desc} for v, label, desc in choices],
        "custom": custom,
        "required": True,
    }
    return {
        "version": 1,
        "interaction_type": f"experiment_{kind}",
        "run_id": run_id,
        "round": round_no,
        "kind": kind,
        "questions": [question],
        "data": _jsonable(data),
        "how_to_answer": {
            "file": DECISION_FILE,
            "shape": {"action": "one of the choice values (or 'answer' for a text question)", "message": "free text: the answer, or the revised goal for criterion_review", "assets": "asset names for drop_assets (optional)"},
        },
    }


def response_from_decision(decision: dict[str, Any] | None, kind: str) -> dict[str, Any]:
    """The RSA ``InteractionResponse`` fields for ``kind``: from the decision file when it answers that kind, else the
    unattended default. ``asset`` is policy, never a decision (PLAN-3 §0 "资产缺 / fallback")."""
    if kind == "asset":
        return dict(ASSET_POLICY)
    if decision and (decision.get("kind") in (None, "", kind)):
        answers = decision.get("answers")
        action = decision.get("action") or (answers[0] if isinstance(answers, list) and answers else None)
        message = str(decision.get("message") or "")
        if action:
            out = {"action": str(action), "message": message}
            if decision.get("assets"):
                out["assets"] = tuple(str(a) for a in decision["assets"])
            if decision.get("grant"):
                out["grant"] = dict(decision["grant"])
            return out
    return dict(UNATTENDED.get(kind, {"action": "stop", "message": "unattended run"}))


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# what the configuring agent is told about this machine (temporary seam adjustment)
# ---------------------------------------------------------------------------

SETUPX_ADDENDUM_MARK = "# --- Paper2Code line: machine notes for the setup agent ---"


def setupx_addendum(denylist: tuple[str, ...] | list[str] = (), *, gpu_available: bool = False) -> str:
    """Appended to SetupX's system prompt (a per-process copy, main's own mechanism in
    ``run_flow._append_contract_to_checkout``). SetupX is a black box to this line (PLAN-3 §0 "搭建环境 =
    黑盒"): it decides for itself what to install and how. The setup agent never sees the goal, only the
    grading contract, so this is a **list of facts about this machine and this run** — not instructions:

    * the accelerator (none, or CUDA) and, for CPU, where the CPU torch wheels are (the default PyPI
      ``torch`` is the CUDA build; on sapg-2, 2026-09-17, a 555 MB wheel crawled in at 1–2 MB/s);
    * the exec channel's per-command limit (300 s);
    * where the graded command writes;
    * the run's denylist — RSA has no channel for it (PLAN-3 §0 "容器里的黑名单").
    """
    lines = ["", "[MACHINE FACTS from the Paper2Code line]"]
    if gpu_available:
        lines.append(
            "- Accelerator: an NVIDIA GPU with the CUDA driver and the container toolkit; CUDA wheels of torch work here "
            "(the default PyPI `torch` wheel is the CUDA build and pulls ~3 GB of nvidia-* wheels; https://download.pytorch.org/whl/cu128 has the same)."
        )
    else:
        lines.append(
            "- Accelerator: none. CUDA builds cannot run; the default PyPI `torch` wheel is the CUDA build (hundreds of MB); "
            "CPU wheels are at https://download.pytorch.org/whl/cpu."
        )
    from apps.v2.agent.paper2code.execution.machine_bootstrap import TORCH_PREINSTALLED

    if os.environ.get("PAPER2CODE_TORCH_PREINSTALL") == "1":
        torch_v, tv_v = TORCH_PREINSTALLED
        lines.append(
            f"- Pre-installed in the container: torch {torch_v} ({'CUDA cu121' if gpu_available else 'CPU'} build) and torchvision {tv_v}; "
            "a requirement they satisfy is already met — installing them again costs the machine minutes to an hour of downloads."
        )
    lines += [
        # measured 2026-09-18 from cn-hongkong: mirrors.aliyun.com 0.25 MB/s over HTTP/1.1 (pip), tuna / pytorch 12 MB/s
        "- Package index: pip is configured to https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple (~12 MB/s from this machine); "
        "mirrors.aliyun.com serves pip at ~0.25 MB/s here and is not worth switching to.",
        "- Exec channel: one command is cut off after 300 seconds; a longer command only finishes when run in the background with its output in a log file.",
        "- Outputs: the graded command writes under /workspace/out (absolute), not into the repository.",
    ]
    if denylist:
        lines.append("- Off limits for this run (never fetched, cloned, pip-installed from, or read): " + ", ".join(denylist) + ".")
    return "\n".join(lines) + "\n"


ADDENDUM_HEAD = "[MACHINE FACTS from the Paper2Code line]"


def append_setupx_addendum(root: str | os.PathLike[str] | None, denylist: tuple[str, ...] | list[str] = (), *, gpu_available: bool = False) -> bool:
    """Patch ``<root>/src/llm_engine.py`` (marker) the way main patches its contract in; False when not done.

    The SetupX checkout is per *process* (``run_flow._setupx_env``) and its ``llm_engine`` module stays in
    ``sys.modules``, so the facts are replaced — in the file and on the live class — whenever they differ from
    what is there: the two-stage compute's GPU stage runs in the same process as its CPU stage (T3b real run
    2026-09-18 14:30: SetupX on the T4 read "Accelerator: none" and installed the CPU wheels)."""
    if not root:
        return False
    target = Path(root) / "src" / "llm_engine.py"
    try:
        body = target.read_text(encoding="utf-8")
    except OSError:
        return False
    addendum = setupx_addendum(denylist, gpu_available=gpu_available).replace("{", "{{").replace("}", "}}")  # the template is .format()ed
    patch = (
        f"\n\n{SETUPX_ADDENDUM_MARK}\n"
        "try:\n"
        f"    LLMEngine.SYSTEM_PROMPT_TEMPLATE = LLMEngine.SYSTEM_PROMPT_TEMPLATE + {addendum!r}\n"
        "except NameError:\n"
        "    pass\n"
    )
    if SETUPX_ADDENDUM_MARK in body:
        base = body[: body.index(SETUPX_ADDENDUM_MARK)].rstrip("\n")
        if body.rstrip("\n") == (base + patch).rstrip("\n"):
            return True
    else:
        base = body
    try:
        target.write_text(base + patch, encoding="utf-8")
    except OSError:
        return False
    _refresh_live_setupx_prompt(target, addendum)
    return True


def _refresh_live_setupx_prompt(target: Path, addendum: str) -> None:
    """The already-imported ``llm_engine`` keeps the old template on its class: swap the facts there too."""
    import sys

    for module in list(sys.modules.values()):
        try:
            if Path(getattr(module, "__file__", "") or "").resolve() != target.resolve():
                continue
        except (OSError, TypeError):
            continue
        engine = getattr(module, "LLMEngine", None)
        template = getattr(engine, "SYSTEM_PROMPT_TEMPLATE", None)
        if not isinstance(template, str):
            continue
        head = template.split(ADDENDUM_HEAD)[0].rstrip("\n")
        engine.SYSTEM_PROMPT_TEMPLATE = head + addendum


# ---------------------------------------------------------------------------
# the runner: RSA round 0 with the line's interaction handling and records
# ---------------------------------------------------------------------------


class Waiting(Exception):
    """Raised inside RSA's thread when a question needs a person: carries the pending event out."""

    def __init__(self, event: Any) -> None:
        super().__init__(str(getattr(event, "kind", "question")))
        self.event = event


@dataclass(slots=True)
class MachineRecord:
    """What the runner established on the machine after RSA finished."""

    container_id: str = ""
    image: str = ""
    image_size_bytes: int | None = None
    set_env: dict[str, str] = field(default_factory=dict)
    setupx_logs: str = ""
    error: str = ""


def default_rsa_agent_factory(config: Any) -> Any:
    """Real RSA over the flow's durable backend — the same construction as ``run_flow._default_rsa_runner``."""
    from apps.v2.agent_engine.experiment.run_flow import _durable_runtime_factory, _gpu_aware_backend_class
    from apps.v2.agent_engine.rsa.agent import RSAAgent

    config.remote_backend = _gpu_aware_backend_class(getattr(config, "deepevol_integrity", None))(
        target=config.remote_target,
        runtime_factory=_durable_runtime_factory,
        username=config.remote_username,
        password=config.remote_password,
        private_key=config.remote_private_key,
    )
    return RSAAgent(config)


def default_host_exec(config: Any, command: str, *, timeout: float = 600.0) -> str:
    """Run one shell command on the machine (not in a container) through a fresh relay runtime."""
    from apps.v2.agent_engine.experiment.git_daemon import _exec
    from apps.v2.agent_engine.experiment.run_flow import _durable_runtime_factory

    async def go() -> str:
        runtime = _durable_runtime_factory(
            target=config.remote_target, username=config.remote_username, password=config.remote_password, private_key=config.remote_private_key
        )
        await runtime.start()
        try:
            return await _exec(runtime, command, timeout=timeout)
        finally:
            await runtime.close()

    return asyncio.run(go())


def default_backend_factory(config: Any) -> Any:
    """A fresh durable backend to the machine (the same construction as the agent factory, minus the agent)."""
    from apps.v2.agent_engine.experiment.run_flow import _durable_runtime_factory, _gpu_aware_backend_class

    return _gpu_aware_backend_class(getattr(config, "deepevol_integrity", None))(
        target=config.remote_target,
        runtime_factory=_durable_runtime_factory,
        username=config.remote_username,
        password=config.remote_password,
        private_key=config.remote_private_key,
    )


#: serving the code history to the machine (bundle upload over SSH) is retried on transport failures: on 2026-09-18 the
#: pinn figures-on run died after its repair round with ``GitDaemonError: connection lost while uploading repo.bundle``
#: (this Mac's network) — one lost packet must not void a round; each attempt gets a fresh runtime
SERVE_REPO_ATTEMPTS = 3
SERVE_REPO_DELAYS_S = (5.0, 20.0)


async def with_retries(attempt: Callable[[], Awaitable[T]], *, attempts: int = SERVE_REPO_ATTEMPTS, delays: Sequence[float] = SERVE_REPO_DELAYS_S,
                       on_retry: Callable[[int, BaseException], None] | None = None) -> T:
    """Run ``attempt`` up to ``attempts`` times; sleep ``delays[i]`` between tries; the last error is raised."""
    last: BaseException | None = None
    for i in range(max(1, attempts)):
        try:
            return await attempt()
        except Exception as exc:  # transport errors are not typed consistently across the relay / daemon layers
            last = exc
            if i + 1 >= attempts:
                break
            if on_retry:
                on_retry(i + 1, exc)
            await asyncio.sleep(delays[min(i, len(delays) - 1)] if delays else 0)
    assert last is not None
    raise last


def default_serve_repo(config: Any) -> Callable[[Path], str]:
    """``serve_repo(git_dir) -> url``: re-serve the code history to the machine's git daemon over a fresh runtime,
    retried (``SERVE_REPO_ATTEMPTS``) when the upload or the daemon setup fails on the wire."""

    def serve(git_dir: Path) -> str:
        from apps.v2.agent_engine.experiment.git_daemon import serve_repo_on_machine
        from apps.v2.agent_engine.experiment.run_flow import _durable_runtime_factory

        async def once() -> str:
            runtime = _durable_runtime_factory(
                target=config.remote_target, username=config.remote_username, password=config.remote_password, private_key=config.remote_private_key
            )
            await runtime.start()
            try:
                return await serve_repo_on_machine(runtime, git_dir)
            finally:
                await runtime.close()

        def note(n: int, exc: BaseException) -> None:
            logger.warning("serving the code history to the machine failed (attempt {}/{}): {}: {}; retrying", n, SERVE_REPO_ATTEMPTS, type(exc).__name__, str(exc)[:200])

        return asyncio.run(with_retries(once, on_retry=note))

    return serve


@dataclass(slots=True)
class RepairSeams:
    """What the repair loop needs from the run: rounds, the code, the model, and (tests) doubles for the machine side."""

    rounds: int
    repo: Any  # CodeRepo
    code_dir: Path
    provider_factory: Callable[[], Any]
    model: str
    blueprint_excerpt: str = ""
    backend_factory: Callable[[Any], Any] | None = None
    serve_repo_factory: Callable[[Any], Callable[[Path], str]] | None = None
    judge: Callable[..., list[Any]] | None = None
    environment_round: Callable[..., int] | None = None
    max_tool_calls: int = 40
    token_cap: int = 3_000_000


def setupx_last_env() -> dict[str, str]:
    try:
        from apps.v2.agent_engine.rsa import setup_loop

        return {str(k): str(v) for k, v in dict(getattr(setup_loop, "_LAST_ENV", {}) or {}).items()}
    except Exception:  # pragma: no cover - vendored module missing
        return {}


class LineRsaRunner:
    """``rsa_runner`` for ``run_experiment_on_machine``: one RSA run, answered and recorded the line's way."""

    def __init__(
        self,
        *,
        run_id: str,
        round_no: int,
        ask: bool,
        decision: dict[str, Any] | None,
        rsa_dir: Path,
        events: Callable[..., Any] | None = None,
        agent_factory: Callable[[Any], Any] | None = None,
        host_exec: Callable[..., str] | None = None,
        commit_image: bool = True,
        auto_recompile: int = 1,
        denylist: tuple[str, ...] | list[str] = (),
        repair: RepairSeams | None = None,
        gpu_available: bool = False,
    ) -> None:
        self.run_id, self.round_no, self.ask = run_id, round_no, ask
        self.denylist = tuple(denylist)
        self.gpu_available = bool(gpu_available)
        #: the run ended on G1's asset card; ``assets_missing`` names what the report could name
        self.asset_blocked = False
        self.assets_missing: list[str] = []
        self.addendum_applied = False
        self.repair = repair
        self.repair_records: list[dict[str, Any]] = []
        self.repair_stop_reason = ""
        self.repair_passed: bool | None = None
        self.accepted_failing = False
        self.gpu_required = False
        #: S8: the controller's final state (``environment_controller.ControllerState``) once the pipeline ran
        self.controller_state: Any = None
        #: temporary (PLAN-3 §0 "临时调整"): a criterion the falsifier rejects is recompiled once with the
        #: rejection reasons appended to the goal before anyone is asked — the falsifier's fixed flag list
        #: and the compiler's first guess disagree often enough (sapg-2: an ``--output_dir`` value) that
        #: stopping on the first rejection would end most unattended runs at compile time
        self.auto_recompile_left = max(int(auto_recompile), 0)
        self.decision = decision
        self.rsa_dir = Path(rsa_dir)
        self.events = events
        self._agent_factory = agent_factory or default_rsa_agent_factory
        self._host_exec = host_exec or default_host_exec
        self.commit_image = commit_image
        self.pending: Any = None
        self.outcome: Any = None
        self.answered: list[dict[str, Any]] = []
        self.machine = MachineRecord()
        self.setup_rounds: list[dict[str, Any]] = []

    # -- interaction ------------------------------------------------------------------------

    def _interaction(self, event: Any) -> Any:
        from apps.v2.agent_engine.rsa.agent import InteractionResponse

        kind = str(getattr(event, "kind", "") or "")
        if kind == "asset":
            # never a question (PLAN-3 §0 "资产缺 / fallback"): not under --ask, not from a decision file
            data = getattr(event, "data", None) or {}
            self.asset_blocked = True
            self.assets_missing = [str(r.get("name") or r.get("path") or "?") for r in (data.get("results") or []) if isinstance(r, dict) and r.get("status") not in (None, "ok")]
            self.answered.append({"kind": kind, "source": "policy", **ASSET_POLICY, "assets": list(self.assets_missing)})
            self._emit("experiment.assets_missing", assets=self.assets_missing)
            self._emit("experiment.answered", question=kind, action=ASSET_POLICY["action"], source="policy")
            return InteractionResponse(**ASSET_POLICY)
        if kind == "criterion_review" and self.auto_recompile_left > 0:
            # RSA ignores the answer and returns RECOMPILE; the runner recompiles (see __call__)
            self.answered.append({"kind": kind, "source": "auto-recompile", "action": "retry", "message": "recompile with the falsification reasons"})
            self._emit("experiment.answered", question=kind, action="retry", source="auto-recompile")
            return InteractionResponse(action="retry", message="recompile")
        answered_kinds = {a["kind"] for a in self.answered}
        decision = self.decision if (self.decision and kind not in answered_kinds) else None
        if decision is not None and decision.get("kind") in (None, "", kind):
            fields = response_from_decision(decision, kind)
            self.answered.append({"kind": kind, "source": "decision", **{k: v for k, v in fields.items() if k != "grant"}})
            self._emit("experiment.answered", question=kind, action=fields["action"], source="decision")
            return InteractionResponse(**fields)
        if self.ask:
            raise Waiting(event)
        fields = response_from_decision(None, kind)
        self.answered.append({"kind": kind, "source": "unattended", **fields})
        self._emit("experiment.answered", question=kind, action=fields["action"], source="unattended")
        return InteractionResponse(**fields)

    def _emit(self, event_kind: str, **fields: Any) -> None:
        if self.events is not None:
            try:
                self.events(event_kind, **fields)
            except Exception as exc:  # events never break the run
                logger.debug("event {} failed: {}", event_kind, exc)

    # -- the run -----------------------------------------------------------------------------

    def __call__(self, config: Any, repo_url: str, instruction: str, revision: str, session_id: str) -> Any:
        from apps.v2.agent_engine.experiment.evidence import begin_recording, bind_round_recorder, take_recorded_rounds
        from apps.v2.agent_engine.rsa.agent import AgentOutcome, AgentStatus, UserInstruction

        request = UserInstruction(repository=repo_url, instruction=instruction, revision=revision, session_id=session_id)
        # before SetupX is first imported in this process: the flow set RSA_SETUPX_ROOT just before calling us
        self.addendum_applied = append_setupx_addendum(os.environ.get("RSA_SETUPX_ROOT"), self.denylist, gpu_available=self.gpu_available)
        for name, value in (("max_steps", SETUPX_MAX_STEPS), ("max_rounds", SETUPX_MAX_ROUNDS)):
            if hasattr(config, name):
                setattr(config, name, min(int(getattr(config, name) or value), value))
        bind_round_recorder()
        begin_recording()
        from apps.v2.agent.paper2code.repair_loop import install_artifact_normaliser

        restore_freezer = install_artifact_normaliser(self.events)  # T3: artifact paths as the goal asked, before the freeze
        restore_pipeline = self._install_controller(config, instruction)
        try:
            while True:
                agent = self._agent_factory(config)  # a fresh agent per attempt: RSA closes its backend on the way out
                try:
                    outcome = agent.run(request, interaction=self._interaction)
                except Waiting as waiting:
                    self.pending = waiting.event
                    outcome = AgentOutcome(AgentStatus.NEEDS_USER, request, pending=waiting.event)
                    break
                if getattr(outcome, "status", None) is AgentStatus.RECOMPILE and self.auto_recompile_left > 0:
                    self.auto_recompile_left -= 1
                    hint = falsification_hint(outcome)
                    request = UserInstruction(repository=repo_url, instruction=one_line(f"{request.instruction} {hint}"), revision=revision, session_id=session_id)
                    self._emit("experiment.recompile", hint=hint[:400])
                    continue
                break
        finally:
            restore_pipeline()
            restore_freezer()
            self.setup_rounds = take_recorded_rounds()
        try:
            outcome.deepevol_setup_rounds = self.setup_rounds  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - frozen outcome types
            pass
        self.outcome = outcome
        status = str(getattr(getattr(outcome, "status", None), "value", "") or getattr(outcome, "status", "") or "")
        if self.asset_blocked:
            # the environment cannot be built: no repair round can help; the flow releases the machine on the terminal status
            self.repair_stop_reason = "assets missing: " + (", ".join(self.assets_missing) or "(see the asset report)")
        elif self.controller_state is not None and self.pending is None:
            try:
                outcome = self._after_controller(outcome)
            except Waiting as waiting:
                self.pending = waiting.event
                outcome = AgentOutcome(AgentStatus.NEEDS_USER, request, pending=waiting.event, frozen=getattr(outcome, "frozen", None), pipeline=getattr(outcome, "pipeline", None))
            self.outcome = outcome
            status = str(getattr(getattr(outcome, "status", None), "value", "") or getattr(outcome, "status", "") or "")
        self.machine.container_id = str(getattr(getattr(outcome, "pipeline", None), "container_id", "") or "")
        self.machine.set_env = setupx_last_env()
        self._keep_setupx_logs()
        if status == "success" and self.machine.container_id and self.commit_image:
            self._commit_container(config)
        return outcome

    def _install_controller(self, config: Any, goal: str) -> Callable[[], None]:
        """S8: point ``rsa.agent.run_pipeline`` at the line's controller for this run; returns the restore."""
        if self.repair is None:
            return lambda: None
        import functools

        from apps.v2.agent.paper2code.repair_loop import ControllerSeams, run_line_pipeline
        from apps.v2.agent_engine.rsa import agent as rsa_agent

        seams = self.repair

        def on_state(state: Any) -> None:
            self.controller_state = state

        controller_seams = ControllerSeams(
            repo=seams.repo, code_dir=Path(seams.code_dir), goal=goal, repair_rounds=int(seams.rounds),
            environment_rounds=int(getattr(config, "max_rounds", None) or SETUPX_MAX_ROUNDS),
            store=Path(config.store), out_dir=self.rsa_dir / "adjudications", provider_factory=seams.provider_factory, model=seams.model,
            serve_repo=(seams.serve_repo_factory or default_serve_repo)(config), denylist=self.denylist, blueprint_excerpt=seams.blueprint_excerpt,
            events=self.events, max_tool_calls=seams.max_tool_calls, token_cap=seams.token_cap,
            environment_round=seams.environment_round, judge=seams.judge, on_state=on_state, gpu_available=self.gpu_available,
        )
        original = rsa_agent.run_pipeline
        rsa_agent.run_pipeline = functools.partial(run_line_pipeline, seams=controller_seams)
        self._emit("controller.installed", repair_rounds=controller_seams.repair_rounds, environment_rounds=controller_seams.environment_rounds)

        def restore() -> None:
            rsa_agent.run_pipeline = original

        return restore

    def _after_controller(self, outcome: Any) -> Any:
        """Records from the controller's run; a stop that is not a pass is the ``repair_review`` point (default accept)."""
        from apps.v2.agent_engine.rsa.agent import InteractionRequest

        state = self.controller_state
        self.repair_records = [r.record() for r in state.records]
        self.repair_stop_reason = state.stop_reason
        self.repair_passed = bool(state.passed)
        self.gpu_required = bool(getattr(state, "gpu_required", False))
        self._emit("repair.done", passed=state.passed, rounds_used=len(state.records), reason=state.stop_reason,
                   environment_rounds=state.environment_rounds_used, repair_rounds=state.repair_rounds_used, trials=state.trials)
        pipeline = getattr(outcome, "pipeline", None)
        repinned = getattr(pipeline, "frozen", None)
        if repinned is not None:
            try:
                outcome.frozen = repinned  # the ladder as re-frozen after the last repair
            except Exception as exc:
                logger.debug("could not attach the re-pinned ladder: {}", exc)
        if state.passed or not state.records:
            return outcome
        if self.gpu_required:
            # the compute escalation, not a review: the phase re-rents the GPU tier and runs the boxes again there
            self._emit("repair.gpu_required", reason=state.stop_reason)
            return outcome
        status = str(getattr(getattr(outcome, "status", None), "value", "") or getattr(outcome, "status", "") or "")
        if status not in {"failed", "blocked"}:
            return outcome
        event = InteractionRequest(
            kind="repair_review",
            message=f"the controller stopped without the criterion passing ({state.stop_reason}); {state.environment_rounds_used} environment round(s), {state.repair_rounds_used} repair round(s), {state.trials} trial(s).",
            data={"rounds": self.repair_records, "stop_reason": state.stop_reason, "controller": state.to_dict()},
        )
        answer = self._interaction_or_default(event)
        self.accepted_failing = answer.get("action") != "abort"
        return outcome

    def _interaction_or_default(self, event: Any) -> dict[str, Any]:
        """Like ``_interaction`` but returns plain fields; ``repair_review`` defaults to ``accept`` (PLAN-3 §0)."""
        kind = str(getattr(event, "kind", "") or "")
        answered_kinds = {a["kind"] for a in self.answered}
        decision = self.decision if (self.decision and kind not in answered_kinds) else None
        if decision is not None and decision.get("kind") in (None, "", kind) and (decision.get("action") or decision.get("answers")):
            fields = response_from_decision(decision, kind)
            self.answered.append({"kind": kind, "source": "decision", **{k: v for k, v in fields.items() if k != "grant"}})
            self._emit("experiment.answered", question=kind, action=fields["action"], source="decision")
            return fields
        if self.ask:
            raise Waiting(event)
        fields = {"action": "accept", "message": "unattended run: the last judged state is recorded and the run goes on (PLAN-3 §0 默认 accept)"}
        self.answered.append({"kind": kind, "source": "unattended", **fields})
        self._emit("experiment.answered", question=kind, action="accept", source="unattended")
        return fields

    def _commit_container(self, config: Any) -> None:
        tag = f"{ENV_IMAGE_PREFIX}:{self.run_id[:12].lower()}-r{self.round_no}"
        cid = self.machine.container_id
        try:
            output = self._host_exec(
                config,
                f"docker commit {shlex.quote(cid)} {shlex.quote(tag)} >/dev/null && docker image inspect --format '{{{{.Size}}}}' {shlex.quote(tag)}",
                timeout=900.0,
            )
        except Exception as exc:
            self.machine.error = f"docker commit failed: {exc}"
            self._emit("experiment.image_failed", container=cid, error=str(exc)[:300])
            return
        self.machine.image = tag
        last = (output or "").strip().splitlines()[-1].strip() if (output or "").strip() else ""
        self.machine.image_size_bytes = int(last) if last.isdigit() else None
        self._emit("experiment.image_ready", image=tag, size_bytes=self.machine.image_size_bytes)

    def _keep_setupx_logs(self) -> None:
        root = os.environ.get("RSA_SETUPX_ROOT", "")
        source = Path(root) / "log" if root else None
        if source is None or not source.is_dir():
            return
        target = self.rsa_dir / SETUPX_LOGS_DIRNAME / f"round{self.round_no}"
        try:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            self.machine.setupx_logs = str(target)
        except OSError as exc:
            logger.warning("could not keep SetupX logs: {}", exc)


def falsification_hint(outcome: Any) -> str:
    """What the falsifier objected to, as one sentence the compiler can act on."""
    compile_outcome = getattr(outcome, "compile", None)
    reports = getattr(compile_outcome, "falsification", None) or {}
    problems: list[str] = []
    for rung, report in (reports.items() if isinstance(reports, dict) else []):
        if not getattr(report, "must_recompile", False):
            continue
        for point in getattr(report, "points", None) or []:
            for problem in getattr(point, "problems", None) or []:
                problems.append(f"{rung}: {problem}")
        for reason in getattr(report, "reasons", None) or []:
            problems.append(f"{rung}: {reason}")
    if not problems:
        problems.append("the criterion was rejected by falsification")
    hint = (
        "Correction: the previous criterion was rejected by static falsification — "
        + "; ".join(dict.fromkeys(problems))
        + ". Reference only files that exist in the repository; put every output under an absolute directory outside it "
        "(e.g. /workspace/out/smoke) and assert artifacts by those absolute paths."
    )
    if any("bare environment" in p for p in problems):
        hint += (
            " A rung that passes in the bare environment must depend on what the configured environment provides: "
            "make its import check name the third-party packages the code declares in requirements.txt, "
            "not only the repository's own modules."
        )
    return hint


# ---------------------------------------------------------------------------
# denylist audit over the ledger
# ---------------------------------------------------------------------------


def _url_needles(url: str) -> list[str]:
    raw = url.strip().lower()
    raw = re.sub(r"^[a-z]+://", "", raw)
    raw = re.sub(r"^www\.", "", raw)
    raw = raw.removesuffix("/").removesuffix(".git")
    needles = [raw]
    if raw.startswith("github.com/"):
        needles.append(raw.removeprefix("github.com/"))
    return [n for n in needles if len(n) >= 8]


def denylist_hits(denylist: tuple[str, ...] | list[str], *texts: str) -> list[str]:
    """Denylisted URLs that appear in any of ``texts`` (the rounds ledger, SetupX logs, probe output)."""
    blob = "\n".join(t for t in texts if t).lower()
    hits: list[str] = []
    for url in denylist:
        if any(needle in blob for needle in _url_needles(url)):
            hits.append(url)
    return hits


_LOG_ACTION = re.compile(r'"action_type":\s*"(?P<kind>[A-Z_]+)".*?"content":\s*(?P<content>\{.*?\}|"[^"]*")', re.S)


def ledger_commands(setup_rounds: list[dict[str, Any]], logs_dir: str | Path | None) -> list[str]:
    """Every command the agent ran on the machine: the rounds ledger (``SHELL_COMMAND: … → exit=…`` lines) and the
    ``content`` of every action in SetupX's logs. Thoughts are not commands — the machine facts name the
    off-limits URLs, and an agent that writes "I must not fetch it" has not fetched it (sapg-2, 2026-09-18)."""
    commands: list[str] = []
    for round_ in setup_rounds or []:
        for action in (round_.get("actions") if isinstance(round_, dict) else None) or []:
            commands.append(str(action))
    if logs_dir and Path(logs_dir).is_dir():
        for path in sorted(Path(logs_dir).rglob("*")):
            if not path.is_file() or path.stat().st_size >= 20_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in _LOG_ACTION.finditer(text):
                commands.append(f"{match.group('kind')}: {match.group('content')}")
    return commands


def ledger_text(setup_rounds: list[dict[str, Any]], logs_dir: str | Path | None) -> str:
    """What the denylist audit scans: commands only (see :func:`ledger_commands`)."""
    return "\n".join(ledger_commands(setup_rounds, logs_dir))


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def environment_record(*, runner: LineRsaRunner, result: Any, commit: str, goal: str, denylist: tuple[str, ...]) -> dict[str, Any]:
    data = dict(getattr(result, "data", None) or {})
    evidence = dict(data.get("evidence") or {})
    # the goal itself names the denylisted URLs (as things not to touch), so it is never scanned
    hits = denylist_hits(denylist, ledger_text(runner.setup_rounds, runner.machine.setupx_logs))
    machine = runner.machine
    return {
        "version": 1,
        "status": str(getattr(result, "status", "") or ""),
        "agent_status": str(data.get("agent_status") or evidence.get("agent_status") or ""),
        "round": runner.round_no,
        "commit": commit,
        "reached": evidence.get("reached") or evidence.get("terminal") or "",
        "criterion": evidence.get("criterion"),
        "container_id": machine.container_id,
        "image": machine.image or None,
        "image_size_bytes": machine.image_size_bytes,
        "set_env": machine.set_env,
        "setup_rounds": runner.setup_rounds,
        "setupx_logs": machine.setupx_logs or None,
        "answered": runner.answered,
        "assets_missing": list(runner.assets_missing),
        "repair": {"rounds": runner.repair_records, "passed": runner.repair_passed, "stop_reason": runner.repair_stop_reason, "accepted_failing": runner.accepted_failing, "gpu_required": bool(getattr(runner, "gpu_required", False))},
        "gpu_available": bool(getattr(runner, "gpu_available", False)),
        "denylist_touched": hits,
        "released": bool(getattr(result, "released", False)),
        "error": machine.error or evidence.get("error") or "",
        "goal": goal,
        "at": time.time(),
    }


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


__all__ = [
    "ASSET_POLICY",
    "DECISION_FILE",
    "ENVIRONMENT_FILE",
    "ENV_IMAGE_PREFIX",
    "ESCALATION_FILE",
    "KINDS",
    "REQUEST_FILE",
    "RSA_DIRNAME",
    "SETUPX_ADDENDUM_MARK",
    "SETUPX_MAX_ROUNDS",
    "SETUPX_MAX_STEPS",
    "STATE_FILE",
    "UNATTENDED",
    "LineRsaRunner",
    "MachineRecord",
    "RepairSeams",
    "Waiting",
    "append_setupx_addendum",
    "build_goal",
    "build_request",
    "declared_requirements",
    "default_backend_factory",
    "default_host_exec",
    "default_rsa_agent_factory",
    "default_serve_repo",
    "denylist_hits",
    "entry_flags",
    "environment_record",
    "falsification_hint",
    "ledger_commands",
    "ledger_text",
    "missing_modules",
    "one_line",
    "read_json",
    "response_from_decision",
    "setupx_addendum",
    "with_retries",
    "write_json",
]
