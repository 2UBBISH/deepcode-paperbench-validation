"""The eleven phases and the engine call each one makes (PLAN.md C7 table).

``dir_info`` — the engine's legacy per-task dict, which the preprocessing
step mutates (segmentation flags) — is persisted in ``<run>/dir_info.json``
after every phase and rebuilt from that file, never from memory, so any
phase can be rerun in a fresh process.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from apps.v2.agent.paper2code import config as cfg
from apps.v2.agent.paper2code import gates
from apps.v2.agent.paper2code import compute as compute_step
from apps.v2.agent.paper2code import experiment_step
from apps.v2.agent.paper2code import syntax_check
from apps.v2.agent.paper2code.code_repo import GIT_DIR_NAME, CodeRepo
from apps.v2.agent.paper2code.config import EventLog, RunConfig, RunPaths
from apps.v2.agent.paper2code.entry_smoke import find_entry, run_entry_smoke
from apps.v2.agent.paper2code.environment_spec import blueprint_sections, extract_environment_spec, summary as spec_summary, write_spec
from apps.v2.agent.paper2code.intake import load_bundle, prepare_input
from apps.v2.agent.paper2code.plan_review import review_step
from apps.v2.agent.paper2code.verification_hook import uninstall_verification_runner
from apps.v2.agent_engine.paper2code.workflows import agent_orchestration_engine as engine
from apps.v2.agent_engine.paper2code.workflows.environment import TASKS_DIRNAME, prepare_workflow_environment

PHASES: tuple[str, ...] = (
    "intake",
    "criteria",
    "plan",
    "plan_review",
    "references",
    "acquire",
    "index",
    "implement",
    "compute",
    "environment_run",
    "optimize",
)
PHASE_INDEX = {name: i + 1 for i, name in enumerate(PHASES)}
MODEL_PHASES = frozenset({"plan", "plan_review", "references", "acquire", "index", "implement"})


class PhaseWaiting(Exception):
    """The phase needs an operator decision; the driver records ``waiting`` and exits."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(result.get("request") or "waiting")
        self.result = result


class PhaseError(RuntimeError):
    pass


@dataclass(slots=True)
class ExperimentSeams:
    """What environment_run needs from outside the phase to call main's experiment agent.

    ``run_lease`` is the line's ``RunLease`` (the machine); ``flow`` is
    ``run_experiment_on_machine`` unless a test injects a double; ``loopback_factory`` builds the
    chat endpoint RSA talks to; ``agent_factory`` / ``host_exec`` are the runner's two machine
    seams (real RSA over the durable backend, one shell command on the machine). ``None`` for any
    of the last three means the real thing.
    """

    run_lease: Any
    flow: Callable[..., Any] | None = None
    loopback_factory: Callable[[Any], Any] | None = None
    agent_factory: Callable[[Any], Any] | None = None
    host_exec: Callable[..., str] | None = None
    commit_image: bool = True
    #: the machine bootstrap on acquire: True = the real one, False = none, or a callable (tests)
    bootstrap: Any = True
    #: overrides for the repair loop's seams (``experiment_step.RepairSeams`` fields; tests)
    repair_overrides: dict[str, Any] | None = None


@dataclass(slots=True)
class PhaseContext:
    run: RunConfig
    paths: RunPaths
    events: EventLog
    logger: Any
    port: Any | None
    ask: bool
    provider: Any | None = None
    #: the preflight vision probe's verdict (None = not probed): intake's figure pass runs only on True
    vision: bool | None = None
    #: set when step 10 goes to main's experiment agent (``--compute aliyun``); ``None`` keeps the
    #: record-only path (compileall + entry smoke on the port) that the local docker mode has
    experiment: ExperimentSeams | None = None

    @property
    def dir_info_path(self) -> Path:
        return self.paths.root / "dir_info.json"

    @property
    def task_dir(self) -> Path:
        return self.paths.workspace / TASKS_DIRNAME / f"paper_{self.run.run_id}"

    def load_dir_info(self) -> dict[str, Any]:
        if not self.dir_info_path.is_file():
            raise PhaseError("dir_info.json is missing; run the intake phase first")
        return json.loads(self.dir_info_path.read_text(encoding="utf-8"))

    def save_dir_info(self, dir_info: dict[str, Any]) -> None:
        self.dir_info_path.write_text(json.dumps(dir_info, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    def phase_file(self, name: str) -> Path:
        return self.paths.phases_dir / f"{PHASE_INDEX[name]:02d}_{name}.json"

    def phase_result(self, name: str) -> dict[str, Any] | None:
        path = self.phase_file(name)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return record.get("result")


def _gate(ctx: PhaseContext, result: gates.GateResult) -> dict[str, Any]:
    ctx.events("gate", name=result.name, passed=result.passed, warnings=result.warnings)
    if not result.passed:
        raise gates.GateFailed(result)
    return result.to_dict()


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------


async def phase_intake(ctx: PhaseContext) -> dict[str, Any]:
    bundle = load_bundle(ctx.run.paper_dir)
    record = prepare_input(bundle, ctx.paths)
    if record["paper_sha256"] != ctx.run.paper_sha256:
        raise PhaseError(
            f"input/paper.md now hashes to {record['paper_sha256'][:12]} but run.json froze {ctx.run.paper_sha256[:12]}; "
            "the paper directory changed since init"
        )
    figures_record = await _describe_figures(ctx, record["paper_sha256"])
    reason = figures_on_but_undescribed(ctx.run.figures, figures_record)
    if reason:
        # a figures-on run that described nothing is a figures-off run under the wrong label (the PaperBench assets are
        # LFS pointers until hydrated — skipped as lfs_pointer, 2026-09-19); the comparison must not be misread
        ctx.events("figures", status="refused", reason=reason, level="warning")
        raise PhaseError(f"--figures on but no figure was described: {reason}")
    task_dir = ctx.task_dir
    resume = task_dir.is_dir()
    raw_input = str(task_dir / "paper.md") if resume else str(ctx.paths.paper_md)
    if resume:
        shutil.copyfile(ctx.paths.paper_md, task_dir / "paper.md")
    wf = await prepare_workflow_environment(
        raw_input=raw_input,
        enable_indexing=True,
        task_kind="paper2code",
        task_id=ctx.run.run_id,
        logger=ctx.logger,
        workspace_root=ctx.paths.workspace,
    )
    if wf.skip_research_analysis:
        engine._record_acquired_artifacts(wf)
    else:
        await engine.acquire_input_artifact(wf, ctx.logger)
    dir_info = await engine.synthesize_workspace_infrastructure_agent(wf, ctx.logger)
    ctx.save_dir_info(dir_info)
    return {
        "task_dir": str(wf.task_dir),
        "task_id": wf.task_id,
        "resumed": bool(wf.skip_research_analysis),
        "paper_md": str(wf.paper_md_path),
        "paper_sha256": record["paper_sha256"],
        "addendum_included": record["addendum_included"],
        "denylist": record["denylist"],
        "figures": figures_record,
        "dir_info": dir_info,
    }


def figures_on_but_undescribed(mode: str, record: dict[str, Any]) -> str | None:
    """Why an explicit ``--figures on`` run must not go on with this figure-pass record (None when it may).

    ``on`` is the caliber "the paper's figures are part of the input"; a pass that described none of the paper's
    figures leaves the run indistinguishable from ``--figures off`` while labelled otherwise. ``auto`` and ``off``
    never refuse; a paper with no figure references is not a failure.
    """
    if mode != "on":
        return None
    status = record.get("status")
    if status != "ok":
        return f"figure pass {status}: {record.get('reason') or record.get('error') or '-'}"
    found = int(record.get("figures_found") or 0)
    if found and not int(record.get("described") or 0):
        reasons = sorted({str(e.get("reason", "")).split(":")[0] for e in (record.get("skipped") or []) + (record.get("failed") or [])})
        return f"{found} figure references, 0 described ({', '.join(reasons) or 'no detail'}); hydrate the assets (LFS pointers → image bytes) or run with --figures off"
    return None


async def _describe_figures(ctx: PhaseContext, raw_sha256: str) -> dict[str, Any]:
    """Intake's figure pass (``figures.py``): the images described and put back after their references.

    ``input/paper.md`` becomes the enriched text the engine reads; the benchmark bytes stay in
    ``input/paper.raw.md`` and ``run.json.paper_sha256`` is still their hash. Never raises: a failure is recorded.
    """
    from apps.v2.agent.paper2code import figures

    mode = ctx.run.figures
    if mode == "off":
        return {"status": "off"}
    if ctx.provider is None:
        return {"status": "skipped", "reason": "no provider on the phase context"}
    if not ctx.vision and mode == "auto":
        return {"status": "unsupported", "reason": "the preflight vision probe did not confirm image input", "vision": ctx.vision}
    raw = ctx.paths.paper_md.read_bytes()
    cached = figures.load_cache(ctx.paths.input_dir, raw_sha256)
    try:
        if cached is not None:
            record = {k: v for k, v in cached.items() if k not in {"raw_sha256", "enriched_sha256"}}
            source = "cache"
        else:
            record = await figures.describe_figures(
                ctx.provider, paper_dir=Path(ctx.run.paper_dir), text=raw.decode("utf-8", errors="replace"), model=ctx.run.figures_model
            )
            source = "model"
        summary = figures.apply(ctx.paths.input_dir, raw, record)
    except Exception as exc:  # the benchmark text is still in place; the run goes on without descriptions
        ctx.events("figures", status="failed", error=str(exc)[:300], level="warning")
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
    ctx.events("figures", status="ok", source=source, found=summary["figures_found"], described=summary["described"], skipped=len(summary["skipped"]), failed=len(summary["failed"]))
    return {"status": "ok", "source": source, "raw_input": str(ctx.paths.input_dir / figures.RAW_INPUT_NAME), "cache": str(figures.cache_path(ctx.paths.input_dir)), **summary}


async def phase_criteria(ctx: PhaseContext) -> dict[str, Any]:
    rubric = Path(ctx.run.paper_dir) / "rubric.json"
    return {"rubric_passthrough": rubric.is_file(), "rubric_path": str(rubric), "note": "rubric passthrough: not read by this line"}


PLAN_REPLANS_ENV = "DEEPCODE_PLAN_REPLANS"  # re-plans allowed when the plan has no Source pointer at all (default 1)


async def phase_plan(ctx: PhaseContext) -> dict[str, Any]:
    dir_info = ctx.load_dir_info()
    segmentation = await engine.orchestrate_document_preprocessing_agent(dir_info, ctx.logger)
    ctx.save_dir_info(dir_info)
    fidelity_on = os.environ.get("DEEPCODE_PAPER_FIDELITY", "").strip().lower() in {"1", "true", "yes", "on"}
    replans, feedback_env = 0, "DEEPCODE_PLANNER_FEEDBACK"
    os.environ.pop(feedback_env, None)
    try:
        while True:
            await engine.orchestrate_code_planning_agent(dir_info, ctx.logger, None, strict_plan_validation=True)
            ctx.save_dir_info(dir_info)
            plan_path = Path(dir_info["initial_plan_path"])
            if not plan_path.is_file():
                raise PhaseError("planning did not produce initial_plan.txt")
            if not fidelity_on:
                break
            # ADR 0004: the manifest is the plan read back (Source: §x.y per file); a plan with no Source line at all
            # ignored the addendum and is planned once more with that said. Nothing else about the binding can fail.
            reason, feedback = _plan_feedback(dir_info, plan_path)
            if reason is None or replans >= int(os.environ.get(PLAN_REPLANS_ENV, "1")):
                break
            replans += 1
            ctx.events("plan.replan", reason=reason, attempt=replans, detail=feedback[:300])
            os.environ[feedback_env] = feedback
    finally:
        os.environ.pop(feedback_env, None)
    gate = _gate(ctx, gates.plan_source(Path(dir_info["paper_dir"])))
    environment_spec = await _environment_spec(ctx, plan_path)
    plan_text = plan_path.read_text(encoding="utf-8")
    # VENDOR 13 counts (recorded, never judged): how much of the paper's mathematics the blueprint carries verbatim
    from apps.v2.agent_engine.paper2code.workflows.paper_readback import plan_fidelity_stats

    fidelity = plan_fidelity_stats(plan_text, ctx.paths.paper_md.read_text(encoding="utf-8", errors="replace") if ctx.paths.paper_md.is_file() else None)
    ctx.events("plan.fidelity", **fidelity)
    source_manifest: dict[str, Any] | None = None
    if os.environ.get("DEEPCODE_PAPER_FIDELITY", "").strip().lower() in {"1", "true", "yes", "on"}:
        # ADR 0003: implement will freeze this manifest and refuse a plan without structured source_items / source_files;
        # fail here (a `rerun --phase plan` away) instead of after references + index
        from apps.v2.agent_engine.paper2code.workflows.source_fidelity import FidelityError, compile_manifest

        paper_path = Path(dir_info["paper_dir"]) / "paper.md"
        try:
            manifest = compile_manifest(plan_text, paper_path.read_bytes())
        except (FidelityError, OSError) as exc:
            ctx.events("plan.source_manifest", ok=False, error=str(exc)[:300])
            raise PhaseError(f"the blueprint could not be read back for source pointers: {exc}") from exc
        source_manifest = {
            "files": len(manifest["files"]), "paper_files": sum(bool(e["sections"]) for e in manifest["files"].values()),
            "glue": len(manifest["glue"]), "pointers": manifest["pointers"], "orphan_pointers": manifest["orphan_pointers"],
            "unmatched": [u["pointer"] for u in manifest["unmatched"]], "replans": replans,
        }
        ctx.events("plan.source_manifest", ok=True, **source_manifest)
    return {
        "segmentation_status": segmentation.get("status"),
        "use_segmentation": dir_info.get("use_segmentation"),
        "segments_ready": dir_info.get("segments_ready"),
        "plan_path": str(plan_path),
        "plan_chars": len(plan_text),
        "plan_fidelity": fidelity,
        "source_manifest": source_manifest,
        "environment_spec": environment_spec,
        "gates": [gate],
    }


async def _environment_spec(ctx: PhaseContext, plan_path: Path) -> dict[str, Any]:
    """Read the environment spec out of the blueprint (one call); a failure is recorded, never raised."""
    if ctx.provider is None:
        return {"status": "skipped", "reason": "no provider on the phase context"}
    record = await extract_environment_spec(ctx.provider, plan_path.read_text(encoding="utf-8"))
    record["plan_path"] = str(plan_path)
    write_spec(ctx.paths.environment_spec_json, record)
    if record["status"] == "ok":
        brief = spec_summary(record["spec"])
        ctx.events("environment_spec", status="ok", **brief)
        return {"status": "ok", "path": str(ctx.paths.environment_spec_json), **brief}
    ctx.events("environment_spec", status="failed", error=record.get("error"), level="warning")
    return {"status": "failed", "path": str(ctx.paths.environment_spec_json), "error": record.get("error")}


def _plan_feedback(dir_info: dict[str, Any], plan_path: Path) -> tuple[str | None, str]:
    """``(reason, planner feedback)`` when the plan must be done again — it has no ``Source:`` pointer anywhere, so
    every file would be glue and the coding agent would never read the paper; ``(None, "")`` otherwise. A pointer
    that names no heading or a paragraph that names no file is recorded, not a reason to re-plan."""
    from apps.v2.agent_engine.paper2code.workflows.source_fidelity import FidelityError, compile_manifest

    try:
        manifest = compile_manifest(plan_path.read_text(encoding="utf-8"), (Path(dir_info["paper_dir"]) / "paper.md").read_bytes())
    except (FidelityError, OSError):
        return None, ""  # the compile after the loop reports a plan that cannot be read at all
    if manifest["pointers"] > 0:
        return None, ""
    return "no_source_pointers", (
        "Your previous plan had NO `Source: §…` line in Section 2. The coding agent reads only the paper sections you "
        "point at, so without them it implements every formula from your summary. In every Section 2 paragraph that "
        "implements something the paper specifies, name the file path(s) and add `Source: §<section>` for the section(s) "
        "to read. Keep everything else of the plan."
    )


async def phase_plan_review(ctx: PhaseContext) -> dict[str, Any]:
    dir_info = ctx.load_dir_info()
    result = await review_step(
        task_dir=Path(dir_info["paper_dir"]),
        plan_path=Path(dir_info["initial_plan_path"]),
        phases_dir=ctx.paths.phases_dir,
        ask=ctx.ask,
        logger=ctx.logger,
    )
    if result.get("status") == "waiting":
        raise PhaseWaiting(result)
    return result


DEGENERATE_REPORT_MARKERS = (
    "I reached the maximum number of tool call iterations",
    "I encountered an error processing your request",
)


def reference_report_is_degenerate(text: str) -> str | None:
    """The reason a reference report is not a report at all, or ``None``.

    The engine writes whatever the analyzer's last message was, including the
    runner's own max-iterations or error text; downstream that reads as "no
    references" and the run sails on without repositories. Name it instead.
    """
    body = (text or "").strip()
    if not body:
        return "empty reference report"
    for marker in DEGENERATE_REPORT_MARKERS:
        if body.startswith(marker):
            return f"reference analysis ended with the runner's message: {body[:80]!r} (raise DEEPCODE_REFERENCE_MAX_ITERATIONS and rerun --phase references)"
    return None


REFERENCE_ITERATIONS_ENV = "DEEPCODE_REFERENCE_MAX_ITERATIONS"
REFERENCE_ITERATIONS_RETRY_FACTOR = 2  # one automatic retry with the budget doubled (40 → 80: what pinn needed, 2026-09-18)


async def phase_references(ctx: PhaseContext) -> dict[str, Any]:
    """The reference report; when the analyzer runs out of tool-call iterations (long papers: pinn at 124k chars
    exhausted the default 40) the phase runs it once more with the budget doubled instead of failing — the engine
    reads the cap from the environment at call time, so the default (shared with the baseline repository) stays."""
    dir_info = ctx.load_dir_info()
    budget = int(os.environ.get(REFERENCE_ITERATIONS_ENV, cfg.ENV_DEFAULTS.get(REFERENCE_ITERATIONS_ENV, "40")))
    text = await engine.orchestrate_reference_intelligence_agent(dir_info, ctx.logger)
    reason = reference_report_is_degenerate(text)
    retried = None
    if reason is not None and "maximum number of tool call iterations" in (text or ""):
        raised = budget * REFERENCE_ITERATIONS_RETRY_FACTOR
        ctx.events("references.retry", reason=reason[:200], iterations=raised)
        previous = os.environ.get(REFERENCE_ITERATIONS_ENV)
        os.environ[REFERENCE_ITERATIONS_ENV] = str(raised)
        try:
            text = await engine.orchestrate_reference_intelligence_agent(dir_info, ctx.logger)
        finally:
            if previous is None:
                os.environ.pop(REFERENCE_ITERATIONS_ENV, None)
            else:
                os.environ[REFERENCE_ITERATIONS_ENV] = previous
        retried = {"iterations": raised, "after": budget}
        reason = reference_report_is_degenerate(text)
    if reason is not None:
        raise PhaseError(reason)
    return {
        "reference_path": dir_info["reference_path"],
        "reference_chars": len(text or ""),
        "github_urls": github_urls_in(text or ""),
        "iterations_retry": retried,
    }


GITHUB_URL_RE = re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.IGNORECASE)
# "Repository: owner/repo" — the shorthand some models write instead of a URL.
GITHUB_SHORTHAND_RE = re.compile(r"(?im)^\s*[-*]?\s*(?:\*\*)?repository(?:\*\*)?\s*:\s*`?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)`?\s*$")


def github_urls_in(text: str) -> list[str]:
    """Distinct GitHub repositories a reference report names (URLs or ``Repository: owner/repo``), as URLs."""
    seen: list[str] = []
    for match in GITHUB_URL_RE.findall(text or ""):
        url = match.rstrip(".,;)")
        if url not in seen:
            seen.append(url)
    for owner_repo in GITHUB_SHORTHAND_RE.findall(text or ""):
        url = f"https://github.com/{owner_repo.rstrip('.')}"
        if url not in seen:
            seen.append(url)
    return seen


async def phase_acquire(ctx: PhaseContext) -> dict[str, Any]:
    dir_info = ctx.load_dir_info()
    reference_path = Path(dir_info["reference_path"])
    if not reference_path.is_file():
        raise PhaseError("reference.txt is missing; run the references phase first")
    report = reference_path.read_text(encoding="utf-8")
    urls = github_urls_in(report)
    if not urls:
        # A paper whose reference report names no repository has nothing to clone; the engine's
        # fail-fast (VENDOR.md 4) is for an agent that narrated instead of cloning, not for this.
        note = "Repository acquisition skipped: the reference report names no GitHub repository."
        Path(dir_info["download_path"]).write_text(note + "\n", encoding="utf-8")
        ctx.events("acquire.skipped", reason="no_github_urls_in_reference_report")
        return {"code_base": str(Path(dir_info["paper_dir"]) / "code_base"), "repositories": [], "skipped": "no_github_urls_in_reference_report"}
    await engine.automate_repository_acquisition_agent(report, dir_info, ctx.logger)
    code_base = Path(dir_info["paper_dir"]) / "code_base"
    repos = sorted(p.name for p in code_base.iterdir() if p.is_dir() and not p.name.startswith(".")) if code_base.is_dir() else []
    return {"code_base": str(code_base), "repositories": repos}


async def phase_index(ctx: PhaseContext) -> dict[str, Any]:
    dir_info = ctx.load_dir_info()
    if "index" in ctx.run.skip:
        report = {"status": "skipped", "reason": "skipped_by_run_config", "message": "Codebase indexing skipped (--skip index)"}
        Path(dir_info["index_report_path"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    result = await engine.orchestrate_codebase_intelligence_agent(dir_info, ctx.logger)
    indexes = Path(dir_info["paper_dir"]) / "indexes"
    files = sorted(p.name for p in indexes.glob("*_index.json")) if indexes.is_dir() else []
    repositories = index_repository_stats(indexes)
    for repo in repositories:
        if repo["prefilter_fallback_suspected"]:
            ctx.events("index.prefilter_fallback", repository=repo["repository"], files=repo["files_found"], level="warning")
    return {"status": result.get("status"), "message": result.get("message"), "index_files": files, "repositories": repositories}


def index_repository_stats(indexes_dir: Path) -> list[dict[str, Any]]:
    """Per-repository indexing scope from the engine's ``<repo>_index.json`` metadata (read-only).

    The engine pre-filters each repository with one model call and, when that call's JSON cannot
    be parsed (large repositories overrun the output budget), silently analyses every file. The
    metadata does not say which happened, so ``prefilter_fallback_suspected`` is inferred: every
    file analysed although pre-filtering was on. A repository whose files were all genuinely
    selected looks the same; the flag is a pointer to the log, not a verdict.
    """
    stats: list[dict[str, Any]] = []
    if not indexes_dir.is_dir():
        return stats
    for path in sorted(indexes_dir.glob("*_index.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats.append({"repository": path.name[: -len("_index.json")], "error": "unreadable index file"})
            continue
        meta = data.get("analysis_metadata") or {}
        found = meta.get("files_before_filtering")
        analysed = meta.get("files_after_filtering")
        enabled = bool(meta.get("pre_filtering_enabled"))
        stats.append(
            {
                "repository": data.get("repo_name") or path.name[: -len("_index.json")],
                "files_found": found,
                "files_analyzed": analysed,
                "file_summaries": len(data.get("file_summaries") or []),
                "relationships": len(data.get("relationships") or []),
                "prefilter_enabled": enabled,
                "prefilter_fallback_suspected": bool(enabled and found and analysed == found),
            }
        )
    return stats


#: No code execution before step 10 (owner 2026-09-20, the Code-Dev comparison rule shared with the desktop arms): the
#: coding agent has no execute tools in indexed mode (`write_file`, `search_code_references`, `read_paper`; the structure
#: agent's mkdir/touch is not execution), and the engine's post-generation test run (`python3 -m pytest`, upstream's
#: verification) is switched off too — it never fed back into the tree, but the rule is "nothing runs".
#: PAPER2CODE_IMPLEMENT_VERIFY=1 restores upstream's verification run.
IMPLEMENT_VERIFY_ENV = "PAPER2CODE_IMPLEMENT_VERIFY"


async def phase_implement(ctx: PhaseContext) -> dict[str, Any]:
    """The engine's implementation loop as upstream runs it (PLAN-3 §2 8i / §2.3 S3), minus execution: the
    coding agent writes and reads (paper, references), and upstream's post-generation test run is off unless
    ``PAPER2CODE_IMPLEMENT_VERIFY=1``. ``uninstall_verification_runner`` is a guard against anything left installed."""
    dir_info = ctx.load_dir_info()
    paper_dir = Path(dir_info["paper_dir"])
    plan_text = Path(dir_info["initial_plan_path"]).read_text(encoding="utf-8")
    fidelity_enabled = os.environ.get("DEEPCODE_PAPER_FIDELITY", "").strip().lower() in {"1", "true", "yes", "on"}
    if fidelity_enabled:
        from apps.v2.agent_engine.paper2code.workflows.source_fidelity import FidelityError, freeze_manifest

        try:
            freeze_manifest(paper_dir, plan_text)
        except FidelityError as exc:
            raise PhaseError(f"paper fidelity manifest rejected the blueprint: {exc}") from exc
    uninstall_verification_runner()
    verify = os.environ.get(IMPLEMENT_VERIFY_ENV, "0") == "1"
    ctx.events("implement.execution", verification_run=verify, agent_execute_tools=False)
    result = await engine.synthesize_code_implementation_agent(
        dir_info, ctx.logger, None, enable_indexing=True, require_verification=verify
    )
    code_dir = Path(result.get("code_directory") or (paper_dir / "generate_code"))
    if syntax_check.enabled() and code_dir.is_dir():
        # owner 09-20: the desktop arms may run py_compile; the line's equivalent is a parse-only check of the tree
        # plus repair rounds with edit_file and no container (syntax_check.py). Remaining errors are recorded, not fatal.
        def syntax_provider_factory() -> Any:
            from apps.v2.agent.paper2code.provider import ParateraProvider

            return ParateraProvider.from_run(ctx.run, ctx.paths, events=ctx.events, log_dir=ctx.paths.llm_dir / "syntax")

        result = {**result, "syntax": await syntax_check.check_and_repair(
            code_dir, provider_factory=syntax_provider_factory if os.environ.get(ctx.run.provider_key_env) else None,
            model=ctx.run.model, denylist=ctx.run.denylist, events=ctx.events, task_dir=paper_dir if fidelity_enabled else None,
        )}
    if fidelity_enabled:
        # ADR 0004: the audit replays the receipts and is recorded (event + phase record); the gate reads it as detail only
        from apps.v2.agent_engine.paper2code.workflows.source_fidelity import FidelityError, audit

        try:
            result = {**result, "paper_fidelity": audit(paper_dir, Path(result.get("code_directory") or (paper_dir / "generate_code")))}
        except FidelityError as exc:
            result = {**result, "paper_fidelity": {"passed": False, "error": str(exc)}}
        fid = result["paper_fidelity"]
        ctx.events("implement.paper_fidelity", passed=bool(fid.get("passed")), receipts=fid.get("read_receipts"), paper_files=fid.get("paper_files"),
                   violations=len(fid.get("violations") or []), error=fid.get("error"))
    if not verify and result.get("inner_status") == "completed" and not result.get("verification"):
        # upstream reports a finished generation as "completed" when it did not verify; the line's gate reads that
        # word as "verified" — say what happened instead
        result = {**result, "inner_status": "unverified", "abort_reason": result.get("abort_reason") or "verification_disabled"}
    summary = {
        key: result.get(key)
        for key in (
            "status", "inner_status", "generation_status", "abort_reason", "files_completed",
            "total_files", "unimplemented_files", "code_directory", "verification", "message", "paper_reads", "paper_fidelity",
            "syntax",
        )
    }
    gate_status = gates.implementation_status(result, code_dir=Path(result.get("code_directory") or (Path(dir_info["paper_dir"]) / "generate_code")))
    ctx.events("gate", name=gate_status.name, passed=gate_status.passed)
    if gate_status.detail.get("empty_files"):
        ctx.events("implement.empty_files", files=[e["path"] for e in gate_status.detail["empty_files"]])
    ctx.events("implement.paper_reads", count=int(result.get("paper_reads") or 0), files=int(result.get("files_completed") or 0))
    gate_owner = gates.ownership(Path(dir_info["paper_dir"]), ctx.paths)
    ctx.events("gate", name=gate_owner.name, passed=gate_owner.passed)
    summary["gates"] = [gate_status.to_dict(), gate_owner.to_dict()]
    for gate in (gate_status, gate_owner):
        if not gate.passed:
            failed = gates.GateFailed(gate)
            failed.summary = summary  # type: ignore[attr-defined]
            raise failed
    return summary


async def phase_compute(ctx: PhaseContext) -> dict[str, Any]:
    """Compute spec → tier plan → review point → the machine the leased port will rent."""
    lease_path = ctx.paths.lease_json
    record = json.loads(lease_path.read_text(encoding="utf-8")) if lease_path.is_file() else {}
    # only a machine that is still ours short-circuits the decision; a released or failed lease is
    # history (sapg-2 sat on a released lease through four attempts and never re-decided)
    if record.get("state") in {"running", "provisioning"}:
        return {
            "status": "already_leased",
            "instance_id": record.get("instance_id"),
            "instance_type": record.get("instance_type"),
            "state": record.get("state"),
            "created_at": record.get("created_at"),
            "released_at": record.get("released_at"),
            "note": "a machine was already rented for this run; the compute decision is not re-taken",
        }
    implement = ctx.phase_result("implement") or {}
    code_dir = Path(implement.get("code_directory") or (ctx.task_dir / "generate_code"))
    spec_record = compute_step.read_json(ctx.paths.environment_spec_json) or {}
    environment_spec = spec_record.get("spec") if spec_record.get("status") == "ok" else None
    # S6: GPU tiers are rentable when the account has a GPU image; stock and prices come from the account too
    run_lease = getattr(ctx.experiment, "run_lease", None) if ctx.experiment is not None else None
    gpu_available = bool(getattr(run_lease, "gpu_image_configured", False)) if run_lease is not None else False
    probe = run_lease.machine_probe if (run_lease is not None and hasattr(run_lease, "machine_probe")) else None
    estimate = compute_step.estimate(code_dir, environment_spec, probe=probe, gpu_available=gpu_available) if code_dir.is_dir() else None
    if estimate is None:
        return {"status": "no_code", "note": f"no generated repository at {code_dir}; nothing to size"}
    request_path = ctx.paths.phases_dir / compute_step.REQUEST_FILE
    decision_path = ctx.paths.phases_dir / compute_step.DECISION_FILE
    default_key = estimate["default"]
    # a run initialised with a GPU tier prefers the GPU plan's tier of that name when one is offered
    wanted = str(getattr(ctx.run, "compute_tier", "") or "")
    if wanted.startswith(compute_step.GPU_KEY_PREFIX) and any(t["key"] == wanted and t["instance_type"] for t in estimate.get("gpu_machines") or []) and gpu_available:
        default_key = wanted
    request = compute_step.build_request(estimate, run_hours=ctx.run.run_hours, default_key=default_key)
    compute_step.write_json(request_path, request)
    decision_raw = compute_step.read_json(decision_path)
    if ctx.ask and decision_raw is None:
        ctx.events("compute.waiting", request=str(request_path))
        raise PhaseWaiting({"status": "waiting", "mode": "ask", "request": str(request_path), "decision_file": str(decision_path), "estimate": _estimate_brief(estimate)})
    try:
        decision = compute_step.resolve_decision(decision_raw if ctx.ask else None, estimate, default_key=default_key)
    except ValueError as exc:
        raise PhaseError(f"compute decision rejected: {exc}") from exc
    if decision_raw is not None:
        decision_path.rename(decision_path.with_suffix(".consumed.json"))
    if decision["action"] == compute_step.CANCEL:
        ctx.events("compute.cancelled")
        return {"status": "cancelled", "decision": decision, "estimate": _estimate_brief(estimate), "note": "no machine will be rented; environment_run records only"}
    port = ctx.port
    configured = False
    if port is not None and hasattr(port, "configure"):
        port.configure(instance_type=decision["instance_type"], hard_cap_seconds=(decision["run_hours"] * 3600.0) if decision["run_hours"] else None)
        configured = True
    ctx.events("compute.decided", mode=decision["mode"], tier=decision["tier"], instance_type=decision["instance_type"], gpu=decision.get("gpu", False), run_hours=decision["run_hours"] or ctx.run.run_hours)
    return {
        "status": "decided",
        "decision": decision,
        "gpu": bool(decision.get("gpu", False)),
        "gpu_available": gpu_available,
        "port_configured": configured,
        "run_hours": decision["run_hours"] or ctx.run.run_hours,
        "estimate": _estimate_brief(estimate),
        "request": str(request_path),
    }


def _estimate_brief(estimate: dict[str, Any]) -> dict[str, Any]:
    spec = estimate["spec"]
    return {
        "workload": spec.get("workload"),
        "needs_gpu": estimate["needs_gpu"],
        "gpu_available": estimate.get("gpu_available"),
        "default": estimate.get("default"),
        "confidence": spec.get("confidence"),
        "host_ram_gib": spec.get("host_ram_gib"),
        "storage_gib": spec.get("storage_gib"),
        "vram_gib": (spec.get("vram") or {}).get("high_gib"),
        "gpu_tiers": [t["label"] for t in (estimate["gpu_tiers"] or {}).get("tiers", [])],
        "cpu_tiers": {t["key"]: t["instance_type"] for t in estimate["cpu_tiers"]},
        "uses_gpu_evidence": (estimate["facts"].get("uses_gpu") or {}).get("evidence"),
        "notes": estimate["notes"],
    }


COMPILE_CHECK_COMMAND = "python -m compileall -q ."
COMPILE_CHECK_TIMEOUT_S = 300.0


async def _compile_check(ctx: PhaseContext, code_dir: Path) -> dict[str, Any] | None:
    """One model-free job on the run's machine: byte-compile the generated repository.

    The engine's verification only runs when it discovers a test command, so a repository without
    tests would never touch the execution port; this job guarantees every run exercises the lease,
    the image (requirements install) and the sync, and reports syntax errors mechanically.
    """
    if ctx.port is None or not code_dir.is_dir():
        return None
    from apps.v2.agent.paper2code.execution.port import Job

    result = await ctx.port.run(
        Job(workspace=code_dir, command=COMPILE_CHECK_COMMAND, timeout_s=COMPILE_CHECK_TIMEOUT_S, label="environment_run:compileall")
    )
    return {
        "command": COMPILE_CHECK_COMMAND,
        "passed": result.ok,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "machine": result.machine,
        "duration_s": round(result.duration_s, 2),
        "error": result.error,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def requirements_state(jobs_dir: Path) -> tuple[bool | None, dict[str, Any] | None]:
    """``(installed, image record)`` from ``jobs/image.json``; ``(None, None)`` when no image was ever attempted."""
    path = jobs_dir / "image.json"
    if not path.is_file():
        return None, None
    state = json.loads(path.read_text(encoding="utf-8"))
    installed = {"ok": True, "failed": False}.get(str(state.get("status")))
    return installed, {key: state.get(key) for key in ("status", "image", "requirements_sha", "reason")}


def _job_records(jobs_dir: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if not jobs_dir.is_dir():
        return jobs
    for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        record = job_dir / "result.json"
        meta = job_dir / "job.json"
        if not record.is_file():
            continue
        data = json.loads(record.read_text(encoding="utf-8"))
        label = json.loads(meta.read_text(encoding="utf-8")).get("label") if meta.is_file() else None
        jobs.append({"job": job_dir.name, "label": label, "exit_code": data.get("exit_code"), "timed_out": data.get("timed_out"), "machine": data.get("machine"), "duration_s": data.get("duration_s")})
    return jobs


def _plan_text(ctx: PhaseContext) -> str:
    if not ctx.dir_info_path.is_file():
        return ""
    plan_path = Path(ctx.load_dir_info().get("initial_plan_path") or "")
    return plan_path.read_text(encoding="utf-8") if plan_path.is_file() else ""


async def phase_environment_run(ctx: PhaseContext) -> dict[str, Any]:
    """Step 10: one call to main's experiment agent, the line's runner inside (adr/0002).

    Round 0 only here (PLAN-3 item 4); the repair loop (item 5) extends the runner. Without
    experiment seams (local docker mode, most offline tests) the old record-only path runs.
    """
    if ctx.experiment is None:
        return await _record_only_environment_run(ctx)
    implement = ctx.phase_result("implement") or {}
    code_dir = Path(implement.get("code_directory") or (ctx.task_dir / "generate_code"))
    if not code_dir.is_dir():
        raise PhaseError(f"no generated repository at {code_dir}; implement must complete first")
    compute = ctx.phase_result("compute") or {}
    if compute.get("status") == "cancelled":
        return {"status": "skipped", "note": "the compute review point cancelled the machine; nothing runs", "compute": compute.get("status")}
    decision = compute.get("decision") or {}
    instance_type = decision.get("instance_type") or (compute.get("instance_type"))
    if not instance_type:
        raise PhaseError("compute recorded no instance type; the compute phase must decide a tier first")
    # two-stage compute: after a GPU-required stop on the CPU tier the phase runs again on the escalation machine
    escalation_path = ctx.paths.phases_dir / experiment_step.ESCALATION_FILE
    escalation = experiment_step.read_json(escalation_path)
    if escalation and escalation.get("to"):
        instance_type = str(escalation["to"])
    run_hours = float(decision.get("run_hours") or compute.get("run_hours") or ctx.run.run_hours)
    hard_cap_seconds = run_hours * 3600.0
    seams = ctx.experiment
    run_lease = seams.run_lease

    state_path = ctx.paths.phases_dir / experiment_step.STATE_FILE
    request_path = ctx.paths.phases_dir / experiment_step.REQUEST_FILE
    decision_path = ctx.paths.phases_dir / experiment_step.DECISION_FILE
    state = experiment_step.read_json(state_path) or {"round": 0, "attempts": 0, "pending": None}
    round_no = int(state.get("round") or 0)
    decision_raw = experiment_step.read_json(decision_path)
    resuming = bool(getattr(run_lease, "active", False))
    if resuming and decision_raw is None and ctx.ask:
        # the machine is held for a question nobody has answered yet
        raise PhaseWaiting({"status": "waiting", "mode": "ask", "request": str(request_path), "decision_file": str(decision_path), "pending": state.get("pending")})

    missing = experiment_step.missing_modules()
    if missing:
        raise PhaseError("this process cannot run the experiment agent, nothing is rented: missing " + "; ".join(missing) + " (install main's agent-runtime extra into the venv)")
    repo = CodeRepo.for_run(ctx.paths.root, code_dir)
    commit = repo.commit(f"environment_run round {round_no}")
    env_spec = experiment_step.read_json(ctx.paths.environment_spec_json) or {}
    entry_point = find_entry(_plan_text(ctx), code_dir)
    entry = str(entry_point.path) if entry_point is not None else None
    gpu_available = bool(decision.get("gpu") or compute.get("gpu") or False) or bool(escalation and escalation.get("to"))
    flags = experiment_step.entry_flags(code_dir, entry)
    requirements = experiment_step.declared_requirements(code_dir)
    goal = experiment_step.build_goal(entry=entry, environment_spec=env_spec.get("spec") if "spec" in env_spec else env_spec, denylist=ctx.run.denylist, gpu_available=gpu_available, flags=flags, requirements=requirements)
    rsa_dir = ctx.paths.root / experiment_step.RSA_DIRNAME
    (rsa_dir / "store").mkdir(parents=True, exist_ok=True)
    (rsa_dir / "work").mkdir(parents=True, exist_ok=True)

    lease = (
        run_lease.adopt(run_id=ctx.run.run_id)
        if resuming
        else run_lease.experiment_lease(run_id=ctx.run.run_id, hard_cap_seconds=hard_cap_seconds, bootstrap=seams.bootstrap)
    )
    from apps.v2.agent.paper2code.execution.aliyun_lease import RunLease

    spec = RunLease.spec_for(instance_type, hard_cap_seconds, hourly_price_cny=float(decision.get("hourly_price_cny") or 0.0), wait_seconds=300)
    repair_rounds = int(getattr(ctx.run, "repair_rounds", 0) or 0)
    repair_seams = None
    if repair_rounds > 0:
        sections = blueprint_sections(_plan_text(ctx))
        excerpt = "\n\n".join(f"{key}:\n{sections[key][:2000]}" for key in ("implementation_components", "validation_approach") if sections.get(key))

        def repair_provider_factory() -> Any:
            from apps.v2.agent.paper2code.provider import ParateraProvider

            return ParateraProvider.from_run(ctx.run, ctx.paths, events=ctx.events, log_dir=ctx.paths.llm_dir / "repair")

        repair_kwargs: dict[str, Any] = {
            "rounds": repair_rounds, "repo": repo, "code_dir": code_dir, "provider_factory": repair_provider_factory,
            "model": ctx.run.model, "blueprint_excerpt": excerpt,
        }
        repair_kwargs.update(seams.repair_overrides or {})
        repair_seams = experiment_step.RepairSeams(**repair_kwargs)
    runner = experiment_step.LineRsaRunner(
        run_id=ctx.run.run_id, round_no=round_no, ask=ctx.ask, decision=decision_raw, rsa_dir=rsa_dir, events=ctx.events,
        agent_factory=seams.agent_factory, host_exec=seams.host_exec, commit_image=seams.commit_image, denylist=ctx.run.denylist,
        repair=repair_seams, gpu_available=gpu_available,
    )
    if seams.flow is None:
        from apps.v2.agent_engine.experiment.run_flow import run_experiment_on_machine as flow
    else:
        flow = seams.flow
    if seams.loopback_factory is None:
        from apps.v2.agent.paper2code.llm_loopback import LoopbackServer

        loopback = LoopbackServer.from_run(ctx.run, ctx.paths, events=ctx.events)
    else:
        loopback = seams.loopback_factory(ctx)
    state.update({"round": round_no, "attempts": int(state.get("attempts") or 0) + 1, "commit": commit, "resumed": resuming, "started_at": time.time()})
    experiment_step.write_json(state_path, state)
    ctx.events("experiment.start", round=round_no, commit=commit, instance_type=instance_type, resumed=resuming, entry=entry)

    def on_event(stage: str, detail: str = "") -> None:
        ctx.events("experiment.flow", stage=stage, detail=str(detail)[:400])

    loopback.start()
    try:
        result = await flow(
            lease=lease, spec=spec, local_repo_dir=repo.git_dir, instruction=goal, llm_target=loopback.target(),
            rsa_runner=runner, store=rsa_dir / "store", work=rsa_dir / "work", session_id=ctx.run.run_id,
            hard_cap_seconds=hard_cap_seconds, on_event=on_event,
        )
    finally:
        loopback.close()
    if decision_raw is not None:
        decision_path.rename(decision_path.with_suffix(f".consumed.r{round_no}.json"))
    (ctx.paths.root / "rsa" / "report.md").write_text(str(getattr(result, "markdown", "") or ""), encoding="utf-8")
    record = experiment_step.environment_record(runner=runner, result=result, commit=commit, goal=goal, denylist=ctx.run.denylist)
    experiment_step.write_json(ctx.paths.root / experiment_step.ENVIRONMENT_FILE, record)
    status = str(getattr(result, "status", "") or "")
    agent_status = record["agent_status"]
    if record["denylist_touched"]:
        ctx.events("experiment.denylist_touched", urls=record["denylist_touched"])
    ctx.events("experiment.done", status=status, agent_status=agent_status, released=record["released"], image=record["image"], reached=record["reached"])

    held = not record["released"] and status in {"needs_user", "oom"}
    if status == "needs_user" and runner.pending is not None and ctx.ask:
        request = experiment_step.build_request(runner.pending, run_id=ctx.run.run_id, round_no=round_no)
        experiment_step.write_json(request_path, request)
        state.update({"pending": request["kind"], "waiting_since": time.time()})
        experiment_step.write_json(state_path, state)
        ctx.events("experiment.waiting", question=request["kind"], request=str(request_path))
        raise PhaseWaiting({"status": "waiting", "mode": "ask", "request": str(request_path), "decision_file": str(decision_path), "kind": request["kind"], "machine_held": True})
    if held:
        # unattended and nobody to ask (recompile / an RSA needs_user the defaults could not answer): do not hold a machine
        try:
            released = await lease.release(reason=f"unattended:{agent_status or status}")
        except Exception as exc:  # the backstop command exists for exactly this
            released = False
            ctx.events("experiment.release_failed", error=str(exc)[:300])
        record["released"] = bool(released)
        experiment_step.write_json(ctx.paths.root / experiment_step.ENVIRONMENT_FILE, record)
    state.update({"pending": None, "finished_at": time.time(), "status": status, "agent_status": agent_status})
    experiment_step.write_json(state_path, state)
    repair_summary = record.get("repair") or {}
    phase_status = "passed" if status == "success" else ("failed" if status in {"failed", "cancelled"} else status)
    if phase_status == "failed" and repair_summary.get("accepted_failing"):
        phase_status = "accepted"
    outcome = {
        "status": phase_status,
        "agent_status": agent_status,
        "reached": record["reached"],
        "round": round_no,
        "commit": repo.head() or commit,
        "repair": {"rounds_used": len(repair_summary.get("rounds") or []), "passed": repair_summary.get("passed"), "stop_reason": repair_summary.get("stop_reason"), "accepted_failing": repair_summary.get("accepted_failing", False), "gpu_required": bool(repair_summary.get("gpu_required", False)), "configured": repair_rounds},
        "image": record["image"],
        "container_id": record["container_id"],
        "released": record["released"],
        "denylist_touched": record["denylist_touched"],
        "assets_missing": record.get("assets_missing") or [],
        "environment": str(ctx.paths.root / experiment_step.ENVIRONMENT_FILE),
        "report": str(ctx.paths.root / "rsa" / "report.md"),
        "goal_chars": len(goal),
        "entry": entry,
        "entry_source": entry_point.source if entry_point is not None else None,
        "answered": runner.answered,
        "note": "round 0 of the experiment agent; repair rounds are PLAN-3 item 5",
        "instance_type": instance_type,
        "gpu_available": gpu_available,
        "escalation": escalation,
    }
    if status == "cancelled":
        raise PhaseError("the experiment run was cancelled")
    if repair_summary.get("gpu_required") and not gpu_available:
        escalation_type = decision.get("escalation_type")
        if not escalation_type or escalation:
            outcome["status"] = "failed"
            outcome["note"] = "the run needs a GPU and no escalation machine is available (no GPU image, or already escalated)"
            return outcome
        return await _escalate_to_gpu(ctx, outcome, record, escalation_path, instance_type, str(escalation_type), lease)
    return outcome


async def _escalate_to_gpu(ctx: PhaseContext, cpu_outcome: dict[str, Any], record: dict[str, Any], escalation_path: Path, from_type: str, to_type: str, lease: Any) -> dict[str, Any]:
    """Two-stage compute (owner 2026-09-18): the CPU stage stopped because the run needs a GPU. Release the CPU
    machine, set the CPU stage's RSA records aside, write the escalation, and run the phase again on the GPU
    machine — the code is at the last repair commit, the goal is recompiled with GPU facts, the boxes start over."""
    if not record.get("released"):
        try:
            await lease.release(reason="escalation:gpu_required")
        except Exception as exc:
            ctx.events("experiment.release_failed", error=str(exc)[:300])
    stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    moved: list[str] = []
    for path in (ctx.paths.root / experiment_step.RSA_DIRNAME, ctx.paths.root / experiment_step.ENVIRONMENT_FILE, ctx.paths.phases_dir / experiment_step.STATE_FILE):
        if path.exists():
            target = path.with_name(f"{path.name}.cpu_stage.{stamp}")
            path.rename(target)
            moved.append(str(target))
    reason = str((record.get("repair") or {}).get("stop_reason") or "")
    escalation = {
        "from": from_type, "to": to_type, "reason": reason, "at": time.time(),
        "cpu_stage": {"commit": cpu_outcome.get("commit"), "repair": cpu_outcome.get("repair"), "reached": cpu_outcome.get("reached"), "moved": moved},
    }
    experiment_step.write_json(escalation_path, escalation)
    ctx.events("experiment.escalate", from_type=from_type, to_type=to_type, reason=reason[:300])
    outcome = await phase_environment_run(ctx)
    outcome["escalation"] = escalation
    return outcome


async def _record_only_environment_run(ctx: PhaseContext) -> dict[str, Any]:
    """The pre-item-4 path: requirements image, compileall and the entry smoke on the port, all record-only."""
    implement = ctx.phase_result("implement") or {}
    code_dir = Path(implement.get("code_directory") or (ctx.task_dir / "generate_code"))
    compile_check = await _compile_check(ctx, code_dir)
    entry = find_entry(_plan_text(ctx), code_dir)
    entry_smoke = await run_entry_smoke(ctx.port, code_dir, entry)
    ctx.events("entry_smoke", entry=entry_smoke["entry"], source=entry_smoke["entry_source"], status=entry_smoke["status"])
    requirements_installed, requirements_image = requirements_state(ctx.paths.jobs_dir)
    verification = implement.get("verification") or []
    return {
        "verification": verification,
        "verification_passed": all(v.get("passed") for v in verification) if verification else None,
        "compile_check": compile_check,
        "requirements_installed": requirements_installed,
        "requirements_image": requirements_image,
        "entry_smoke": entry_smoke,
        "jobs": _job_records(ctx.paths.jobs_dir),
        "note": "environment set-up, repair rounds and the full experiment run are deferred (PLAN.md §8); compileall and the entry smoke are record-only",
    }


async def phase_optimize(ctx: PhaseContext) -> dict[str, Any]:
    return {"stub": True, "note": "optimisation against the evaluation standard is deferred (PLAN.md §8)"}


PHASE_FUNCTIONS = {
    "intake": phase_intake,
    "criteria": phase_criteria,
    "plan": phase_plan,
    "plan_review": phase_plan_review,
    "references": phase_references,
    "acquire": phase_acquire,
    "index": phase_index,
    "implement": phase_implement,
    "compute": phase_compute,
    "environment_run": phase_environment_run,
    "optimize": phase_optimize,
}


# ---------------------------------------------------------------------------
# rerun support: what to move aside so the engine regenerates
# ---------------------------------------------------------------------------


def archive_phase_artifacts(ctx: PhaseContext, name: str, stamp: str) -> list[str]:
    """Move the artifacts a phase produced out of the engine's way; returns what moved."""
    if not ctx.dir_info_path.is_file():
        return []
    dir_info = ctx.load_dir_info()
    task_dir = Path(dir_info["paper_dir"])
    moved: list[str] = []

    def aside(path: Path) -> None:
        if path.exists():
            target = path.with_name(f"{path.name}.superseded.{stamp}")
            path.rename(target)
            moved.append(str(target))

    if name == "plan":
        plan = Path(dir_info["initial_plan_path"])
        if plan.is_file():
            versions = task_dir / "plan_versions"
            versions.mkdir(exist_ok=True)
            shutil.copyfile(plan, versions / f"initial_plan.superseded.{stamp}.txt")
            moved.append(str(versions / f"initial_plan.superseded.{stamp}.txt"))
            plan.unlink()
        aside(task_dir / "planning_result_meta.json")
        aside(task_dir / "planning_attempts.jsonl")
        aside(ctx.paths.environment_spec_json)
        aside(task_dir / "plan_review_history.jsonl")
        for extra in (ctx.paths.phases_dir / "04_plan_review.state.json", ctx.paths.phases_dir / "04_plan_review.request.json"):
            aside(extra)
    elif name == "plan_review":
        for extra in (ctx.paths.phases_dir / "04_plan_review.state.json", ctx.paths.phases_dir / "04_plan_review.request.json"):
            aside(extra)
    elif name == "references":
        aside(Path(dir_info["reference_path"]))
    elif name == "acquire":
        aside(task_dir / "code_base")
        aside(Path(dir_info["download_path"]))
    elif name == "index":
        aside(task_dir / "indexes")
        aside(Path(dir_info["index_report_path"]))
    elif name == "implement":
        aside(task_dir / "generate_code")
        aside(task_dir / "implement_code_summary.md")
        aside(Path(dir_info["implementation_report_path"]))
        # the code history (code_repo.CodeRepo) describes the superseded generate_code/; a fresh
        # implementation starts its own
        aside(ctx.paths.root / GIT_DIR_NAME)
    elif name == "compute":
        for extra in (ctx.paths.phases_dir / compute_step.REQUEST_FILE, ctx.paths.phases_dir / compute_step.DECISION_FILE):
            aside(extra)
    elif name == "environment_run":
        # superseding the phase re-establishes the environment: a requirements image recorded as failed
        # would otherwise make every later job silently fall back to the base image
        aside(ctx.paths.jobs_dir / "image.json")
        for extra in (experiment_step.STATE_FILE, experiment_step.REQUEST_FILE, experiment_step.DECISION_FILE, experiment_step.ESCALATION_FILE):
            aside(ctx.paths.phases_dir / extra)
        aside(ctx.paths.root / experiment_step.ENVIRONMENT_FILE)
        aside(ctx.paths.root / experiment_step.RSA_DIRNAME)
    return moved


__all__ = [
    "MODEL_PHASES",
    "PHASES",
    "PHASE_FUNCTIONS",
    "PHASE_INDEX",
    "ExperimentSeams",
    "PhaseContext",
    "PhaseError",
    "PhaseWaiting",
    "archive_phase_artifacts",
    "github_urls_in",
    "index_repository_stats",
    "reference_report_is_degenerate",
    "requirements_state",
]
