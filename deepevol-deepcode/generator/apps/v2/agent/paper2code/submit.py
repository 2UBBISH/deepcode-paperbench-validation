"""``submit``: put a finished run's generated repository into the PaperBench submission pool.

The pool is what the vendored judge grades (``vendor/paperbench-judge/judge.sh grade`` reads
``~/pb_submissions/<paper>/<trial>/``). A run may be submitted only when all four gates passed and
the run is past the compute phase — ``environment_run`` completed (the repaired tree, the product), or
not started at all (the **stage-9 tree**: owner's rule of 2026-09-18 evening, every PaperBench
comparison on V4-Flash stops at stage 9 and is graded there; step 10 is not part of those runs). A run
whose step 10 is in flight or failed is refused, so nothing half-built or of doubtful ownership is ever
graded. The copy excludes ``__pycache__``, ``.git`` and byte-code; ``<run>/submission.json`` records
the target, a sha256 manifest, the caliber and which tree it is (``tree``: ``repaired`` / ``stage9`` /
the snapshot name), and ``status.json`` gets ``submitted_at``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.config import RunConfig, RunPaths
from apps.v2.agent_engine.paper2code.workflows.environment import TASKS_DIRNAME

REQUIRED_GATES: tuple[str, ...] = ("preflight", "plan_source", "implementation_status", "ownership")
DEFAULT_DEST_ROOT = "~/pb_submissions"
EXCLUDED_NAMES = frozenset({"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")


class SubmitRefused(RuntimeError):
    """The run is not in a state that may be graded."""


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def code_directory(run: RunConfig, paths: RunPaths) -> Path:
    """The generated repository: the implement phase's ``code_directory`` or the task's ``generate_code``."""
    record = paths.phases_dir / "08_implement.json"
    if record.is_file():
        result = (json.loads(record.read_text(encoding="utf-8")).get("result") or {})
        if result.get("code_directory"):
            return Path(result["code_directory"])
    return paths.workspace / TASKS_DIRNAME / f"paper_{run.run_id}" / "generate_code"


def check_submittable(status: dict[str, Any]) -> list[str]:
    """Reasons the run may not be submitted (empty when it may)."""
    reasons: list[str] = []
    gates = status.get("gates") or {}
    for name in REQUIRED_GATES:
        if not (gates.get(name) or {}).get("passed"):
            reasons.append(f"gate {name} not passed")
    phases = status.get("phases") or {}
    compute = (phases.get("compute") or {}).get("status")
    phase = (phases.get("environment_run") or {}).get("status")
    if phase == "completed":
        return reasons
    if phase in (None, "pending", "skipped") and compute == "completed":
        return reasons  # the stage-9 tree (owner's rule, 2026-09-18): graded without step 10
    reasons.append(f"environment_run is {phase or 'pending'} and compute is {compute or 'pending'}: neither the repaired tree nor a stage-9 tree")
    return reasons


def tree_kind(status: dict[str, Any], snapshot: str | None) -> str:
    """What is being submitted: the named snapshot, the repaired tree, or the stage-9 tree."""
    if snapshot:
        return snapshot
    phase = ((status.get("phases") or {}).get("environment_run") or {}).get("status")
    return "repaired" if phase == "completed" else "stage9"


def _copy_tree(src: Path, dst: Path) -> list[Path]:
    def ignore(directory: str, names: list[str]) -> set[str]:
        return {n for n in names if n in EXCLUDED_NAMES or n.endswith(EXCLUDED_SUFFIXES)}

    shutil.copytree(src, dst, ignore=ignore)
    return sorted(p for p in dst.rglob("*") if p.is_file())


def submit(
    run: RunConfig,
    paths: RunPaths,
    status: dict[str, Any],
    *,
    paper: str,
    trial: str,
    dest_root: str | Path = DEFAULT_DEST_ROOT,
    force: bool = False,
    snapshot: str | None = None,
) -> dict[str, Any]:
    """Copy the generated repository into ``<dest_root>/<paper>/<trial>/`` and write the record.

    Raises :class:`SubmitRefused` when a gate is missing, ``environment_run`` is not completed, the
    code directory is empty, or the target exists (unless ``force``). Mutates ``status`` (the caller
    persists it). ``snapshot`` names a state of the code history (``pre_repair`` = the first commit,
    what the experiment agent judged before any repair round; or a commit) to submit instead of the
    working tree — the two ends of a paired comparison come from one run this way.
    """
    reasons = check_submittable(status)
    if reasons:
        raise SubmitRefused("; ".join(reasons))
    if not paper or "/" in paper or not trial or "/" in trial:
        raise SubmitRefused("paper and trial must be single path segments")
    src = code_directory(run, paths)
    if not src.is_dir() or not any(p.is_file() for p in src.rglob("*")):
        raise SubmitRefused(f"no generated repository at {src}")
    target = Path(dest_root).expanduser().resolve() / paper / trial
    if target.exists():
        if not force:
            raise SubmitRefused(f"{target} exists; pass --force to replace it")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_commit: str | None = None
    if snapshot:
        from apps.v2.agent.paper2code.code_repo import CodeRepo

        repo = CodeRepo.for_run(paths.root, src)
        snapshot_commit = repo.resolve_snapshot(snapshot)
        if snapshot_commit is None:
            raise SubmitRefused(f"snapshot {snapshot!r} is not in the code history ({repo.git_dir})")
        files = repo.export(snapshot_commit, target, excluded_names=EXCLUDED_NAMES, excluded_suffixes=EXCLUDED_SUFFIXES)
    else:
        files = _copy_tree(src, target)
    manifest = {p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    now = time.time()
    record = {
        "run_id": run.run_id,
        "paper": paper,
        "trial": trial,
        "source": str(src),
        "snapshot": snapshot,
        "snapshot_commit": snapshot_commit,
        "tree": tree_kind(status, snapshot),
        "target": str(target),
        "files": len(manifest),
        "sha256": manifest,
        "caliber": {
            "model": run.model,
            "thinking": getattr(run, "thinking", "disabled"),
            "provider_base_url": run.provider_base_url,
            "engine_commit": run.engine_commit,
            "compute": f"{run.compute}/{run.compute_tier}",
        },
        "gates": {name: (status.get("gates") or {}).get(name, {}).get("at") for name in REQUIRED_GATES},
        "submitted_at": _iso(now),
    }
    (paths.root / "submission.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status["submitted_at"] = record["submitted_at"]
    status["submission"] = {"paper": paper, "trial": trial, "target": str(target), "files": len(manifest)}
    return record


__all__ = ["DEFAULT_DEST_ROOT", "REQUIRED_GATES", "SubmitRefused", "check_submittable", "code_directory", "submit"]
