"""The file-backed run driver: init / step / run / rerun / status / release.

``status.json`` is the run's authority (PLAN.md §2): the current phase, each
phase's state (``pending`` ``running`` ``completed`` ``failed`` ``waiting``
``superseded``), the gate results. Every attempt writes
``phases/<nn>_<name>.json`` (start/end, inputs, the engine's return value,
gates, error) and appends to ``phases/<nn>_<name>.attempts.jsonl``.
``events.jsonl`` carries ``phase.*``, ``gate``, ``llm.call``, ``job.run``,
``lease.*``. The execution port is closed in ``finally`` so a leased
machine is released even when a phase fails.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from apps.v2.agent.paper2code import config as cfg
from apps.v2.agent.paper2code import gates
from apps.v2.agent.paper2code.config import EventLog, RunConfig, RunPaths
from apps.v2.agent.paper2code.intake import load_bundle, prepare_input, read_denylist
from apps.v2.agent.paper2code.phases import (
    MODEL_PHASES,
    PHASE_FUNCTIONS,
    PHASE_INDEX,
    PHASES,
    PhaseContext,
    PhaseWaiting,
    archive_phase_artifacts,
)
from apps.v2.agent_engine.paper2code.workflows.plan_review_runtime import PlanReviewCancelled

STATUS_VERSION = 1


class DriverError(RuntimeError):
    pass


def _now() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + "Z"


def new_run_id(prefix: str = "") -> str:
    stamp = time.strftime("%m%d%H%M", time.gmtime())
    return f"{prefix}{stamp}{uuid.uuid4().hex[:4]}"


class Driver:
    def __init__(self, run_dir: Path) -> None:
        self.paths = RunPaths(Path(run_dir).expanduser().resolve())
        if not self.paths.run_json.is_file():
            raise DriverError(f"{self.paths.run_json} does not exist; run `init` first")
        self.run = RunConfig.load(self.paths.run_json)
        self.status = self._load_status()
        self.events = EventLog(self.paths.events_jsonl)
        self._ctx: PhaseContext | None = None
        self._runtime: Any | None = None
        self._port: Any | None = None
        self._provider: Any | None = None

    # -- init ----------------------------------------------------------------------

    @classmethod
    def init(
        cls,
        run_dir: Path,
        *,
        paper_dir: str,
        model: str = cfg.DEFAULT_MODEL,
        provider_base_url: str = cfg.DEFAULT_PROVIDER_BASE_URL,
        provider_key_env: str = cfg.DEFAULT_PROVIDER_KEY_ENV,
        provider_stream: bool = False,
        thinking: str = "disabled",
        compute: str = "aliyun",
        compute_tier: str = "enough",
        run_hours: float = 2.0,
        repair_rounds: int = 3,
        figures: str = "auto",
        figures_model: str = cfg.DEFAULT_FIGURES_MODEL,
        experiment_model: str = cfg.DEFAULT_EXPERIMENT_MODEL,
        context_window: int = cfg.DEFAULT_CONTEXT_WINDOW,
        planning_fanout: bool = False,
        skip: tuple[str, ...] = (),
        ask: bool = False,
        run_id: str | None = None,
        engine_commit: str = "",
    ) -> "Driver":
        paths = RunPaths(Path(run_dir).expanduser().resolve())
        if paths.run_json.exists():
            raise DriverError(f"{paths.root} already holds a run; choose another --run-dir")
        paths.ensure()
        bundle = load_bundle(paper_dir)
        record = prepare_input(bundle, paths)
        run = RunConfig(
            run_id=run_id or new_run_id(),
            paper_dir=str(bundle.paper_dir),
            paper_sha256=record["paper_sha256"],
            model=model,
            provider_base_url=provider_base_url,
            provider_key_env=provider_key_env,
            provider_stream=provider_stream,
            thinking=thinking,
            compute=compute,
            compute_tier=compute_tier,
            run_hours=run_hours,
            repair_rounds=int(repair_rounds),
            figures=figures,
            figures_model=figures_model,
            experiment_model=experiment_model,
            context_window=int(context_window),
            planning_fanout=bool(planning_fanout),
            denylist=tuple(record["denylist"]),
            skip=tuple(skip),
            ask=ask,
            engine_commit=engine_commit,
            extra={"addendum_included": record["addendum_included"], "rubric_present": record["rubric_present"]},
        )
        run.validate()
        run.save(paths.run_json)
        status = {
            "version": STATUS_VERSION,
            "run_id": run.run_id,
            "created_at": _iso(run.created_at),
            "updated_at": _iso(_now()),
            "current_phase": None,
            "phases": {name: {"status": "pending", "attempts": 0} for name in PHASES},
            "gates": {},
        }
        paths.status_json.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        driver = cls(paths.root)
        driver.events("run.init", run_id=run.run_id, paper_dir=run.paper_dir, model=run.model, compute=run.compute)
        return driver

    # -- status --------------------------------------------------------------------

    def _load_status(self) -> dict[str, Any]:
        if self.paths.status_json.is_file():
            return json.loads(self.paths.status_json.read_text(encoding="utf-8"))
        return {"version": STATUS_VERSION, "run_id": self.run.run_id, "phases": {n: {"status": "pending", "attempts": 0} for n in PHASES}, "gates": {}}

    def _save_status(self) -> None:
        self.status["updated_at"] = _iso(_now())
        self.paths.status_json.write_text(json.dumps(self.status, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    def phase_status(self, name: str) -> str:
        return self.status["phases"][name]["status"]

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run.run_id,
            "run_dir": str(self.paths.root),
            "paper_dir": self.run.paper_dir,
            "model": self.run.model,
            "experiment_model": self.run.experiment_model,
            "figures_model": self.run.figures_model,
            "context_window": self.run.context_window,
            "planning_fanout": self.run.planning_fanout,
            "compute": f"{self.run.compute}/{self.run.compute_tier}",
            "current_phase": self.status.get("current_phase"),
            "phases": {name: self.status["phases"][name] for name in PHASES},
            "gates": self.status.get("gates", {}),
            "lease": json.loads(self.paths.lease_json.read_text()) if self.paths.lease_json.is_file() else None,
        }

    # -- lifecycle -------------------------------------------------------------------

    async def open(
        self, *, ask: bool | None = None, provider: Any | None = None, port: Any | None = None, probe: bool = True, git_check: bool = True, experiment: Any | None = None
    ) -> PhaseContext:
        """Install the seams, build provider and port, run preflight once. ``experiment`` overrides step 10's seams (tests)."""
        if self._ctx is not None:
            return self._ctx
        ask = self.run.ask if ask is None else ask
        self._port = port if port is not None else self._make_port()
        self._runtime = cfg.install(self.run, self.paths, provider=provider, port=self._port, events=self.events)
        self._provider = self._runtime.provider
        self._ctx = PhaseContext(
            run=self.run, paths=self.paths, events=self.events, logger=logger, port=self._port, ask=ask, provider=self._provider,
            experiment=experiment if experiment is not None else self._make_experiment_seams(),
        )
        if not self.status["gates"].get("preflight", {}).get("passed"):
            bundle = load_bundle(self.run.paper_dir)
            result = await gates.preflight(
                self.run, self._runtime, self._provider,
                expected_denylist=read_denylist(bundle), probe=probe, git_check=git_check, vision=self.run.figures != "off",
            )
            self.status["gates"]["preflight"] = {**result.to_dict(), "at": _iso(_now())}
            self._save_status()
            self.events("gate", name="preflight", passed=result.passed, warnings=result.warnings)
            for warning in result.warnings:
                logger.warning("preflight: {}", warning)
            if not result.passed:
                raise gates.GateFailed(result)
        vision = ((self.status["gates"].get("preflight") or {}).get("detail") or {}).get("vision")
        self._ctx.vision = bool(vision.get("supported")) if isinstance(vision, dict) else None
        return self._ctx

    def _make_port(self) -> Any:
        if self.run.compute == "aliyun":
            from apps.v2.agent.paper2code.execution.aliyun_lease import instance_type_for
            from apps.v2.agent.paper2code.execution.leased_runtime import LeasedExecutionPort

            return LeasedExecutionPort(
                run_id=self.run.run_id, run_dir=self.paths.root, jobs_dir=self.paths.jobs_dir,
                instance_type=instance_type_for(self.run.compute_tier), hard_cap_seconds=self.run.run_hours * 3600,
                events=self.events,
            )
        from apps.v2.agent.paper2code.execution.job_executor import LocalDockerHost, RemoteDockerExecutor

        return RemoteDockerExecutor(host=LocalDockerHost(), run_id=self.run.run_id, jobs_dir=self.paths.jobs_dir, events=self.events)

    def _make_experiment_seams(self) -> Any:
        """Step 10 goes to main's experiment agent when a machine can be rented; local docker mode keeps the record-only path."""
        if self.run.compute != "aliyun":
            return None
        from apps.v2.agent.paper2code.execution.aliyun_lease import RunLease
        from apps.v2.agent.paper2code.phases import ExperimentSeams

        return ExperimentSeams(run_lease=RunLease(self.paths.root, events=self.events))

    async def close(self) -> None:
        port, self._port = self._port, None
        runtime, self._runtime = self._runtime, None
        self._ctx = None
        try:
            if port is not None:
                await port.close()
        finally:
            if runtime is not None:
                try:
                    await runtime.aclose()
                except Exception as exc:
                    logger.debug("runtime close: {}", exc)
            from apps.v2.agent_engine.paper2code.seams.config import set_runtime

            set_runtime(None)

    # -- phases --------------------------------------------------------------------

    def _check_prerequisites(self, name: str) -> None:
        for earlier in PHASES[: PHASE_INDEX[name] - 1]:
            state = self.phase_status(earlier)
            if state != "completed":
                raise DriverError(f"phase {name!r} needs {earlier!r} completed first (it is {state})")

    async def step(self, name: str) -> str:
        """Run one phase; returns its resulting status."""
        if name not in PHASE_INDEX:
            raise DriverError(f"unknown phase {name!r}; phases: {', '.join(PHASES)}")
        self._check_prerequisites(name)
        ctx = await self.open()
        entry = self.status["phases"][name]
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["status"] = "running"
        entry["started_at"] = _iso(_now())
        entry.pop("error", None)
        self.status["current_phase"] = name
        self._save_status()
        self.events("phase.started", phase=name, attempt=entry["attempts"])
        started = _now()
        record: dict[str, Any] = {"phase": name, "index": PHASE_INDEX[name], "attempt": entry["attempts"], "started_at": _iso(started),
                                  "input_sha256": self.run.paper_sha256, "run_id": self.run.run_id}
        outcome = "failed"
        try:
            result = await PHASE_FUNCTIONS[name](ctx)
            record["result"] = result
            outcome = "completed"
        except PhaseWaiting as waiting:
            record["result"] = waiting.result
            outcome = "waiting"
        except PlanReviewCancelled as exc:
            record["error"] = f"cancelled: {exc}"
            outcome = "failed"
        except gates.GateFailed as exc:
            record["error"] = str(exc)
            record["gate"] = exc.result.to_dict()
            summary = getattr(exc, "summary", None)
            if summary is not None:
                record["result"] = summary
            outcome = "failed"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-4000:]
            outcome = "failed"
        record["finished_at"] = _iso(_now())
        record["duration_s"] = round(_now() - started, 2)
        record["status"] = outcome
        self._write_phase_record(name, record)
        entry["status"] = outcome
        entry["finished_at"] = record["finished_at"]
        if outcome == "failed":
            entry["error"] = record.get("error")
        if "gate" in record:
            self.status["gates"][record["gate"]["name"]] = {**record["gate"], "at": record["finished_at"]}
        for gate in (record.get("result") or {}).get("gates", []) if isinstance(record.get("result"), dict) else []:
            self.status["gates"][gate["name"]] = {**gate, "at": record["finished_at"]}
        self._save_status()
        self.events(f"phase.{'finished' if outcome == 'completed' else outcome}", phase=name, attempt=entry["attempts"], duration_s=record["duration_s"], error=record.get("error"))
        if outcome == "failed":
            logger.error("phase {} failed: {}", name, record.get("error"))
        elif outcome == "waiting":
            logger.info("phase {} is waiting: {}", name, record["result"].get("request"))
        else:
            logger.info("phase {} completed in {}s", name, record["duration_s"])
        return outcome

    def _write_phase_record(self, name: str, record: dict[str, Any]) -> None:
        path = self.paths.phases_dir / f"{PHASE_INDEX[name]:02d}_{name}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        with (self.paths.phases_dir / f"{PHASE_INDEX[name]:02d}_{name}.attempts.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: v for k, v in record.items() if k != "result"}, ensure_ascii=False, default=str) + "\n")

    async def run_until(self, until: str = "environment_run") -> str:
        """Run phases in order up to ``until``; stops at the first failed or waiting phase."""
        if until not in PHASE_INDEX:
            raise DriverError(f"unknown phase {until!r}")
        last = "completed"
        for name in PHASES[: PHASE_INDEX[until]]:
            if self.phase_status(name) == "completed":
                continue
            last = await self.step(name)
            if last != "completed":
                break
        return last

    def rerun(self, name: str, *, keep_tree: bool = False) -> list[str]:
        """Mark ``name`` and every later phase superseded and move their artifacts aside.

        Re-running ``environment_run`` also puts ``generate_code/`` back to the first commit of ``code.git`` — the
        tree as the experiment agent first judged it, i.e. the stage-9 tree — unless ``keep_tree``; the repair
        rounds' edits stay in the history. So a run that went through step 10 once can be used to validate step 10
        again from the same starting point, and a stage-9 run from another session needs nothing at all (its
        ``environment_run`` is pending: ``run --until environment_run``)."""
        if name not in PHASE_INDEX:
            raise DriverError(f"unknown phase {name!r}")
        if name == "environment_run" and not keep_tree:
            self._reset_tree_to_first_commit()
        stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        ctx = PhaseContext(run=self.run, paths=self.paths, events=self.events, logger=logger, port=None, ask=self.run.ask)
        moved: list[str] = []
        for later in PHASES[PHASE_INDEX[name] - 1 :]:
            entry = self.status["phases"][later]
            if entry["status"] != "pending":
                entry["status"] = "superseded"
                entry["superseded_at"] = _iso(_now())
            moved.extend(archive_phase_artifacts(ctx, later, stamp))
        # a superseded phase is runnable again: treat it as pending for prerequisites
        for later in PHASES[PHASE_INDEX[name] - 1 :]:
            if self.status["phases"][later]["status"] == "superseded":
                self.status["phases"][later]["status"] = "pending"
                self.status["phases"][later]["superseded_at"] = _iso(_now())
        self._save_status()
        self.events("phase.rerun", phase=name, moved=moved)
        return moved

    def _reset_tree_to_first_commit(self) -> None:
        from apps.v2.agent.paper2code.code_repo import CodeRepo
        from apps.v2.agent.paper2code.submit import code_directory

        repo = CodeRepo.for_run(self.paths.root, code_directory(self.run, self.paths))
        first = repo.first_commit()
        if first is None:
            return  # step 10 never ran: the tree is the stage-9 tree already
        if repo.dirty():
            repo.commit("tree before rerun (uncommitted edits)")
        counts = repo.restore(first)
        self.events("tree.reset", to=first, **counts)
        logger.info("generate_code reset to the first commit {} ({} files restored, {} later files removed)", first[:12], counts["restored"], counts["removed"])

    def relocate(self) -> dict[str, Any]:
        """Rewrite the absolute run-directory path recorded inside a run that was copied or moved.

        The phases, ``dir_info.json``, ``status.json``, ``environment.json``, ``environment_spec.json`` and
        ``submission.json`` carry the directory's absolute path (the engine's task directory lives under
        ``workspace/``); after a copy they point at the old place. The old root is read off ``dir_info.json``
        (everything before ``/workspace/``). Returns what was rewritten."""
        new_root = str(self.paths.root)
        dir_info = json.loads((self.paths.root / "dir_info.json").read_text(encoding="utf-8")) if (self.paths.root / "dir_info.json").is_file() else {}
        sample = str(dir_info.get("paper_dir") or dir_info.get("workspace_dir") or "")
        marker = f"/{self.paths.workspace.name}/"
        if marker not in sample:
            raise DriverError("dir_info.json names no workspace path; nothing to relocate (has the intake phase run?)")
        old_root = sample.split(marker, 1)[0]
        if old_root == new_root:
            return {"old_root": old_root, "new_root": new_root, "files": []}
        files: list[str] = []
        candidates = [self.paths.root / n for n in ("dir_info.json", "status.json", "run.json", "environment.json", "environment_spec.json", "submission.json")]
        candidates += sorted(self.paths.phases_dir.glob("*.json")) + sorted(self.paths.phases_dir.glob("*.jsonl"))
        candidates += sorted(self.paths.workspace.glob("tasks/*/dir_info.json"))
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if old_root not in text:
                continue
            path.write_text(text.replace(old_root, new_root), encoding="utf-8")
            files.append(str(path.relative_to(self.paths.root)))
        self.status = json.loads(self.paths.status_json.read_text(encoding="utf-8"))
        self.events("run.relocated", old_root=old_root, new_root=new_root, files=files)
        return {"old_root": old_root, "new_root": new_root, "files": files}

    def submit(self, *, paper: str, trial: str, dest_root: str | Path = "~/pb_submissions", force: bool = False, snapshot: str | None = None) -> dict[str, Any]:
        """Copy the generated repository into the PaperBench submission pool (four gates + environment_run required).

        ``snapshot="pre_repair"`` submits the code as the experiment agent first judged it (round 0 in
        ``code.git``) — the comparison caliber — instead of the repaired working tree (PLAN-3 item 6)."""
        from apps.v2.agent.paper2code.submit import SubmitRefused, submit

        try:
            record = submit(self.run, self.paths, self.status, paper=paper, trial=trial, dest_root=dest_root, force=force, snapshot=snapshot)
        except SubmitRefused as exc:
            raise DriverError(f"submit refused: {exc}") from exc
        self._save_status()
        self.events("submitted", paper=paper, trial=trial, target=record["target"], files=record["files"])
        return record

    async def release(self, reason: str = "manual release") -> dict[str, Any] | None:
        """Force-release the leased machine (the backstop when a process died mid-run)."""
        from apps.v2.agent.paper2code.execution.aliyun_lease import RunLease

        lease = RunLease(self.paths.root, events=self.events)
        outcome = await lease.force_release(reason)
        return {**outcome, "lease": json.loads(self.paths.lease_json.read_text()) if self.paths.lease_json.is_file() else None}


def load_env_files(paths: list[str]) -> list[str]:
    """``KEY=VALUE`` lines into ``os.environ`` (existing values win); never logs values."""
    loaded: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_file():
            raise DriverError(f"env file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


__all__ = ["MODEL_PHASES", "Driver", "DriverError", "load_env_files", "new_run_id"]
