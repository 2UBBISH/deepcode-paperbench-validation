"""The four gates ``run_trial.sh`` applied around a DeepCode run, as code.

``preflight`` before the first model call; ``plan_source`` after planning
(the engine can wrap a failed planning into a coerced minimal plan and call
it success — such a run is void); ``implementation_status`` and
``ownership`` after implementation (all planned files written; the task
directory's paper is this run's paper; at least five files produced). A
failed gate fails its phase; the driver never enters the next one.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.config import KERNEL_MAX_TOKENS, RunConfig, RunPaths
from apps.v2.agent.paper2code.intake import sha256_file
from apps.v2.agent_engine.paper2code.seams.config import KernelRuntime
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMProvider

MIN_GENERATED_FILES = 5
PROBE_PROMPT = "Reply with the single word OK."


@dataclass(slots=True)
class GateResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail, "warnings": list(self.warnings)}


class GateFailed(RuntimeError):
    def __init__(self, result: GateResult) -> None:
        super().__init__(f"gate {result.name} failed: {json.dumps(result.detail, ensure_ascii=False, default=str)[:400]}")
        self.result = result


def _git_insteadof_lines() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "config", "--global", "--get-regexp", "insteadof"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


async def preflight(
    run: RunConfig,
    runtime: KernelRuntime,
    provider: LLMProvider,
    *,
    expected_denylist: tuple[str, ...] = (),
    probe: bool = True,
    git_check: bool = True,
    vision: bool = False,
) -> GateResult:
    """Model pinned for every phase, 32k output budget, denylist loaded, git fence, provider probe.

    ``expected_denylist`` is what the paper's ``blacklist.txt`` says now; every entry must be in the
    run's frozen denylist (an empty blacklist file is fine — EMA-Detect ships one).
    """
    detail: dict[str, Any] = {}
    warnings: list[str] = []
    failures: list[str] = []

    cfg = runtime.config
    overrides = {
        phase: getattr(cfg.agents, phase).model
        for phase in ("planning", "implementation")
        if getattr(cfg.agents, phase).model
    }
    detail["model"] = cfg.agents.defaults.model
    detail["phase_model_overrides"] = overrides
    if overrides:
        failures.append(f"phase model overrides present: {overrides}")
    if cfg.agents.defaults.model != run.model:
        failures.append(f"kernel model {cfg.agents.defaults.model!r} != run model {run.model!r}")

    detail["max_tokens"] = cfg.agents.defaults.max_tokens
    if cfg.agents.defaults.max_tokens < KERNEL_MAX_TOKENS:
        failures.append(f"max_tokens {cfg.agents.defaults.max_tokens} < {KERNEL_MAX_TOKENS}")
    # the run's other model slots and the context window, recorded so a run directory says what it ran on
    detail["experiment_model"] = getattr(run, "experiment_model", None)
    detail["figures_model"] = getattr(run, "figures_model", None)
    detail["context_window"] = int(getattr(run, "context_window", 0) or 0)
    if detail["context_window"] <= cfg.agents.defaults.max_tokens:
        failures.append(f"context_window {detail['context_window']} leaves no room beside max_tokens {cfg.agents.defaults.max_tokens}")

    detail["denylist"] = list(run.denylist)
    missing_entries = sorted(set(expected_denylist) - set(run.denylist))
    if missing_entries:
        failures.append(f"blacklist.txt entries missing from the run's denylist: {missing_entries}")

    if git_check and run.denylist:
        lines = _git_insteadof_lines()
        joined = "\n".join(lines).lower()
        unfenced = [entry for entry in run.denylist if entry.lower().rstrip("/").removesuffix(".git") not in joined]
        detail["git_insteadof_fenced"] = not unfenced
        if unfenced:
            warnings.append(
                "git insteadOf fence missing for: " + ", ".join(unfenced)
                + " (the in-process tools enforce the denylist; the git fence is the second line)"
            )

    if probe:
        response = await provider.chat_with_retry(
            [{"role": "user", "content": PROBE_PROMPT}], max_tokens=16, temperature=0.0, retry_mode="standard"
        )
        detail["probe"] = {
            "finish_reason": response.finish_reason,
            "reasoning_tokens": response.usage.get("reasoning_tokens"),
            "content": (response.content or "")[:40],
        }
        if response.finish_reason == "error":
            failures.append(f"provider probe failed: {response.content}")
        elif getattr(run, "thinking", "disabled") == "disabled" and response.usage.get("reasoning_tokens", 0) != 0:
            failures.append("provider probe returned reasoning tokens; thinking is not off")
        elif getattr(run, "thinking", "disabled") == "enabled" and not response.usage.get("reasoning_tokens", 0):
            warnings.append("thinking is enabled for this run but the probe returned no reasoning tokens (the model may skip thinking on a trivial prompt)")

    if probe and vision:
        # whether the model takes image parts (intake's figure pass, ``figures.py``); informational only
        from apps.v2.agent.paper2code.figures import vision_probe

        detail["vision"] = await vision_probe(provider, model=run.figures_model)
        if not detail["vision"]["supported"]:
            warnings.append(f"vision model {run.figures_model!r} does not take images ({detail['vision'].get('error')}); figures will not be described")

    detail["failures"] = failures
    return GateResult("preflight", not failures, detail, warnings)


def plan_source(task_dir: Path) -> GateResult:
    meta_path = Path(task_dir) / "planning_result_meta.json"
    if not meta_path.is_file():
        return GateResult("plan_source", False, {"reason": "planning_result_meta.json missing", "path": str(meta_path)})
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return GateResult("plan_source", False, {"reason": f"unreadable meta: {exc}"})
    source = str(meta.get("source") or "unknown")
    status = str(meta.get("status") or "unknown")
    detail = {"source": source, "status": status, "plan_chars": meta.get("plan_chars"), "mode": meta.get("mode")}
    return GateResult("plan_source", source == "generated" and status == "success", detail)


def implementation_status(result: dict[str, Any], code_dir: Path | None = None) -> GateResult:
    """Every planned file written — the generation loop ran to completion.

    ``completed`` is the engine's verdict when the discovered tests pass. ``unverified``
    (``no_tests_discovered``) and ``test_failed`` (``generated_tests_failed``) are the same loop
    outcome, all files written, with the engine's local verification not confirming it: since S3
    (PLAN-3 §2 8i) those tests run on the driver host, where the generated code's dependencies are
    usually absent, so a failed local run says little — step 10 on the machine is the verification
    that counts. Both pass the gate with ``verified`` false; every early stop of the loop fails it.
    ``code_dir`` adds ``empty_files`` (T13): zero-byte source files, recorded not failed — the repair loop
    gets them as evidence (``repair.empty_files``).
    """
    inner = result.get("inner_status") or result.get("status")
    unimplemented = list(result.get("unimplemented_files") or [])
    files_completed = int(result.get("files_completed") or 0)
    total_files = int(result.get("total_files") or 0)
    all_written = not unimplemented and files_completed >= total_files and files_completed > 0
    passed = inner == "completed" or (inner in {"unverified", "test_failed"} and all_written)
    detail = {
        "status": result.get("status"),
        "inner_status": inner,
        "abort_reason": result.get("abort_reason"),
        "files_completed": files_completed,
        "total_files": total_files,
        "unimplemented_files": unimplemented[:20],
        "verified": inner == "completed",
        "local_tests_failed": inner == "test_failed",
    }
    fidelity = result.get("paper_fidelity")
    if os.environ.get("DEEPCODE_PAPER_FIDELITY", "").strip().lower() in {"1", "true", "yes", "on"}:
        # ADR 0004: the audit is a record — the read-before-write refusal at write time is the mechanism, and the
        # audit only replays it; its findings go into the gate's detail, never into its verdict
        detail["paper_fidelity"] = fidelity or {"passed": False, "error": "paper fidelity audit missing"}
    if code_dir is not None:
        from apps.v2.agent.paper2code.repair import empty_files

        detail["empty_files"] = empty_files(Path(code_dir))[:20]
    return GateResult("implementation_status", passed, detail)


def ownership(task_dir: Path, paths: RunPaths, *, min_files: int = MIN_GENERATED_FILES) -> GateResult:
    task_dir = Path(task_dir)
    detail: dict[str, Any] = {}
    failures: list[str] = []
    task_paper = task_dir / "paper.md"
    if not task_paper.is_file():
        failures.append("task directory has no paper.md")
        detail["task_paper_sha256"] = None
    else:
        detail["task_paper_sha256"] = sha256_file(task_paper)
        detail["input_paper_sha256"] = sha256_file(paths.paper_md) if paths.paper_md.is_file() else None
        if detail["task_paper_sha256"] != detail["input_paper_sha256"]:
            failures.append("task directory paper.md differs from input/paper.md")
    code_dir = task_dir / "generate_code"
    count = sum(1 for p in code_dir.rglob("*") if p.is_file()) if code_dir.is_dir() else 0
    detail["generated_files"] = count
    detail["code_directory"] = str(code_dir)
    if count < min_files:
        failures.append(f"generate_code has {count} files (< {min_files})")
    detail["failures"] = failures
    return GateResult("ownership", not failures, detail)


__all__ = [
    "MIN_GENERATED_FILES",
    "GateFailed",
    "GateResult",
    "implementation_status",
    "ownership",
    "plan_source",
    "preflight",
]
