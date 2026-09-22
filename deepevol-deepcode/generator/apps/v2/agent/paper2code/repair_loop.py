"""The repair loop inside the runner (PLAN-3 item 5): same container, same frozen criterion content, new code.

After RSA's round 0 ends without the criterion passing, and while the machine is still held (the
loop runs inside ``LineRsaRunner`` before the flow decides about the machine), each round is:

1. **classify** the last failure (``repair.classify``): an environment signature sends the round to
   SetupX (``setup_loop.run_round`` on the same container, the same entry ``run_flow`` hooks),
   anything else to the line's repair agent; when the agent ends a round with
   ``finish(attribution="environment")`` the next round is SetupX's regardless of the heuristic
   (PLAN-3 §2.3 S1 ②) — the heuristic only decides where a round *starts*;
2. **change** — the agent edits ``generate_code/`` (probes run in the container between a
   checkpoint and a rollback, so nothing a probe installs survives), the change is committed,
   the bundle re-served, the container's clone moved to the new commit, and the ladder
   **re-frozen with the new commit pin** (same criterion content — the adjudicator resets the tree
   to the pinned commit, so the pin must follow the code, ``adr/0002``);
3. **judge** every rung up to G2 with RSA's ``Adjudicator``; the first rung that does not pass is
   the next round's evidence. A failure whose :func:`failure_signature` equals the previous
   judged failure's — the round changed nothing observable — stops the loop (S1 ③); the runner
   turns every early stop into the ``repair_review`` point.

Everything RSA-side is called through the two entry points ``run_flow`` already uses; no vendored
source changes. Bounded by ``rounds`` (``repair_rounds`` in run.json), the per-round tool-call
budget and the phase token cap.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from apps.v2.agent.paper2code import repair
from apps.v2.agent.paper2code.code_repo import CodeRepo

CONTAINER_REPO = "/workspace/repo"
TOKEN_CAP = 3_000_000
JUDGE_UP_TO = "G2"


@dataclass(slots=True)
class RoundRecord:
    round_no: int
    kind: str
    reason: str
    commit: str = ""
    agent: dict[str, Any] | None = None
    environment_actions: int = 0
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    seconds: float = 0.0
    error: str = ""
    signature: str = ""

    def record(self) -> dict[str, Any]:
        return {
            "round": self.round_no,
            "kind": self.kind,
            "reason": self.reason,
            "commit": self.commit,
            "agent": self.agent,
            "environment_actions": self.environment_actions,
            "verdicts": list(self.verdicts),
            "passed": self.passed,
            "seconds": round(self.seconds, 1),
            "error": self.error,
            "signature": self.signature,
        }


def verdict_dict(verdict: Any) -> dict[str, Any]:
    from apps.v2.agent_engine.experiment.evidence import _verdict

    return _verdict(verdict) or {}


def failure_text(verdict: dict[str, Any] | None) -> str:
    """What the routing looks at: the run's own failure tails (``test_run_completes`` and every non-artifact test),
    the log tail and the reason. The artifact tests' tails are left out on purpose — they only say what the run
    did not produce (a ``FileNotFoundError`` on the expected checkpoint), and the compiler puts those paths
    wherever it likes (sapg-s9, 2026-09-18: ``/workspace/repo/figures/curves.npz``), so they read as missing
    assets and sent a plain ``ValueError`` to SetupX twice."""
    verdict = verdict or {}
    tails = verdict.get("failure_tails") or {}
    run_tails = [str(v) for k, v in tails.items() if "artifact" not in str(k)]
    # the log tail is the whole pytest log — artifact tails included — so it stands in only when no run tail exists
    parts = [*(run_tails or [verdict.get("log_tail") or ""]), verdict.get("reason") or ""]
    return "\n".join(str(p) for p in parts if p)


_HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_DURATION = re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|ms|sec|seconds?)\b")
_LINE_NUMBER = re.compile(r"\bline \d+\b")
_WHITESPACE = re.compile(r"\s+")


def failure_signature(verdict: dict[str, Any] | None) -> str:
    """A short hash of *what* failed: rung, verdict, exit code, the failing test ids and the failure text
    with addresses, durations and line numbers normalised away. Two consecutive rounds with the same
    signature mean the round changed nothing the criterion can see."""
    verdict = verdict or {}
    failing = sorted(str(t) for t, status in (verdict.get("statuses") or {}).items() if str(status).lower() not in {"passed", "pass"})
    text = failure_text(verdict)
    text = _HEX_ADDRESS.sub("0x", text)
    text = _DURATION.sub("#s", text)
    text = _LINE_NUMBER.sub("line #", text)
    text = _WHITESPACE.sub(" ", text).strip()
    payload = "|".join([str(verdict.get("rung") or ""), str(verdict.get("verdict") or ""), str(verdict.get("exit_code") or ""), ",".join(failing), text])
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def last_verdict_of(outcome: Any) -> dict[str, Any]:
    """The last judged rung of an RSA outcome, in the evidence's dict shape."""
    from apps.v2.agent_engine.experiment.evidence import collect_evidence

    rungs = collect_evidence(outcome).get("rungs") or []
    for rung in reversed(rungs):
        if rung.get("verdict"):
            return dict(rung["verdict"])
    return {}


class RepairLoop:
    """One instance per runner call; ``run`` returns (passed, records, frozen)."""

    def __init__(
        self,
        *,
        config: Any,
        outcome: Any,
        repo: CodeRepo,
        code_dir: Path,
        goal: str,
        rounds: int,
        store: Path,
        out_dir: Path,
        provider_factory: Callable[[], Any],
        model: str,
        serve_repo: Callable[[Path], str],
        backend_factory: Callable[[Any], Any],
        denylist: tuple[str, ...] = (),
        blueprint_excerpt: str = "",
        events: Callable[..., Any] | None = None,
        max_tool_calls: int = repair.MAX_TOOL_CALLS,
        token_cap: int = TOKEN_CAP,
        tokens_used: int = 0,
        judge: Callable[[Any, Any, int], list[Any]] | None = None,
        environment_round: Callable[[Any, dict[str, Any], int], int] | None = None,
    ) -> None:
        self.config, self.outcome, self.repo, self.code_dir = config, outcome, repo, Path(code_dir)
        self.goal, self.rounds, self.store, self.out_dir = goal, int(rounds), Path(store), Path(out_dir)
        self.provider_factory, self.model = provider_factory, model
        self.serve_repo, self.backend_factory = serve_repo, backend_factory
        self.denylist, self.blueprint_excerpt, self.events = tuple(denylist), blueprint_excerpt, events
        self.max_tool_calls, self.token_cap, self.tokens_used = max_tool_calls, token_cap, tokens_used
        self.records: list[RoundRecord] = []
        self.frozen: Any = getattr(outcome, "frozen", None)
        self.container_id: str = str(getattr(getattr(outcome, "pipeline", None), "container_id", "") or "")
        self.summaries: list[str] = []
        self.stop_reason = ""
        #: set by ``_rollback_probes`` when a failed rollback left the backend without a container
        self.container_lost = False
        #: seams for tests: the real ones drive RSA's Adjudicator and SetupX's run_round
        self._judge_seam = judge
        self._environment_seam = environment_round

    def _emit(self, event_kind: str, **fields: Any) -> None:
        if self.events is not None:
            try:
                self.events(event_kind, **fields)
            except Exception as exc:
                logger.debug("event {} failed: {}", event_kind, exc)

    # -- driving ----------------------------------------------------------------------------

    def run(self) -> tuple[bool, list[RoundRecord]]:
        if self.rounds <= 0:
            self.stop_reason = "repair_rounds is 0"
            return False, []
        if self.frozen is None or not self.container_id:
            self.stop_reason = "no frozen criterion or no container to repair against"
            return False, []
        backend = self.backend_factory(self.config)
        try:
            backend.attach(self.container_id, CONTAINER_REPO)
        except Exception as exc:
            self.stop_reason = f"could not attach to container {self.container_id}: {exc}"
            return False, []
        try:
            return self._loop(backend)
        finally:
            try:
                backend.close()
            except Exception:
                pass

    def _loop(self, backend: Any) -> tuple[bool, list[RoundRecord]]:
        last = last_verdict_of(self.outcome)
        last_signature = failure_signature(last)
        #: the agent's ``finish(attribution="environment")`` decides the next round's route
        next_route: tuple[str, str] | None = None
        for round_no in range(1, self.rounds + 1):
            started = time.monotonic()
            text = failure_text(last)
            kind, reason = repair.classify(text, repo_modules_=repair.repo_modules(self.code_dir))
            if next_route is not None:
                kind, reason = next_route
                next_route = None
            record = RoundRecord(round_no=round_no, kind=kind, reason=reason, commit=self.repo.head() or "")
            self.records.append(record)
            self._emit("repair.round", round=round_no, route=kind, reason=reason)
            try:
                if kind == repair.ENVIRONMENT:
                    record.environment_actions = (self._environment_seam or self._environment_round)(backend, last, round_no)
                else:
                    if self.tokens_used >= self.token_cap:
                        record.error = f"phase token cap {self.token_cap} reached"
                        self.stop_reason = record.error
                        record.seconds = time.monotonic() - started
                        break
                    agent_outcome = self._code_round(backend, last, round_no)
                    record.agent = agent_outcome.record()
                    self.tokens_used += int(agent_outcome.usage.get("total_tokens", 0))
                    if agent_outcome.summary:
                        self.summaries.append(f"round {round_no}: {agent_outcome.summary[:200]}")
                    if agent_outcome.attribution == repair.ENVIRONMENT:
                        next_route = (repair.ENVIRONMENT, f"the repair agent attributed the failure to the environment: {agent_outcome.summary[:160]}")
                    new_sha = self.repo.commit(f"repair round {round_no}: {agent_outcome.summary[:60] or 'agent changes'}")
                    if new_sha == record.commit:
                        if next_route is None:
                            record.error = "the repair agent changed nothing"
                            self.stop_reason = record.error
                            record.seconds = time.monotonic() - started
                            break
                        # same code, same environment: nothing to judge; the next round is SetupX's
                        record.signature = last_signature
                        record.seconds = time.monotonic() - started
                        self._emit("repair.deferred", round=round_no, reason=next_route[1])
                        continue
                    record.commit = new_sha
                    self._move_container_to(backend, new_sha)
                    self.frozen = self._refreeze(backend, new_sha)
                verdicts = (self._judge_seam or self._judge)(backend, self.frozen, round_no)
            except Exception as exc:
                logger.exception("repair round {} failed", round_no)
                record.error = f"{type(exc).__name__}: {exc}"
                record.seconds = time.monotonic() - started
                self.stop_reason = record.error
                break
            record.verdicts = [verdict_dict(v) for v in verdicts]
            record.passed = bool(verdicts) and all(str(getattr(v, "verdict", "")) == "PASS" for v in verdicts)
            record.seconds = time.monotonic() - started
            self._emit("repair.judged", round=round_no, passed=record.passed, verdicts=[(d.get("rung"), d.get("verdict")) for d in record.verdicts])
            if record.passed:
                self.stop_reason = "criterion passed"
                return True, self.records
            last = record.verdicts[-1] if record.verdicts else last
            record.signature = failure_signature(last)
            if next_route is None and record.signature == last_signature:
                # the round changed nothing the criterion can see; another round of the same kind would too
                self.stop_reason = f"round {round_no} reproduced the previous failure (signature {record.signature})"
                break
            last_signature = record.signature
        if not self.stop_reason:
            self.stop_reason = f"{self.rounds} repair rounds used"
        return False, self.records

    # -- one environment round (SetupX on the same container) ------------------------------

    def _environment_round(self, backend: Any, last: dict[str, Any], round_no: int) -> int:
        from apps.v2.agent_engine.rsa import setup_loop as sl
        from apps.v2.agent_engine.rsa.adjudicator import Verdict
        from apps.v2.agent_engine.rsa.kickback import ActionLedger, grading_contract, kickback
        from apps.v2.agent_engine.rsa.setupx_interop import setupx_configured

        fc = self._first_failing_rung(last)
        verdict = Verdict(
            verdict=str(last.get("verdict") or "FAIL"), rung=str(last.get("rung") or fc.rung.value),
            expected_n=int(last.get("expected_n") or 0), passed_expected=int(last.get("passed_expected") or 0),
            missing=list(last.get("missing") or []), statuses=dict(last.get("statuses") or {}),
            exit_code=int(last.get("exit_code") or 0), log_tail=str(last.get("log_tail") or ""),
            failure_tails=dict(last.get("failure_tails") or {}), reason=str(last.get("reason") or ""),
        )
        contract = grading_contract(fc.criterion, fc.expected, disclose_ids=getattr(self.config, "disclose_ids", True))
        message = kickback(verdict, ActionLedger(), round_no=round_no, rounds_total=self.rounds, tails=verdict.failure_tails)
        with setupx_configured(self.config.backend, base_image=self.config.base_image, remote_backend=backend):
            sl.bind(sl.LoopContext(frozen=fc, workdir=self.config.workdir))
            actions, cid = sl.run_round(
                self.frozen.ladder.repo_url, revision=self.frozen.ladder.commit, max_steps=int(self.config.max_steps),
                contract=contract, kickback_text=message, container_id=self.container_id or None,
            )
        if cid:
            self.container_id = cid
            backend.attach(cid, CONTAINER_REPO)
        return len(getattr(actions, "actions", None) or [])

    # -- one code round (the line's agent) --------------------------------------------------

    def _code_round(self, backend: Any, last: dict[str, Any], round_no: int) -> repair.RepairOutcome:
        classification = repair.classify(failure_text(last), repo_modules_=repair.repo_modules(self.code_dir))
        criterion = self._criterion_dict()
        messages = repair.build_messages(
            goal=self.goal, criterion=criterion, verdict=last, classification=classification, code_dir=self.code_dir,
            blueprint_excerpt=self.blueprint_excerpt, round_no=round_no, denylist=self.denylist, previous_summaries=self.summaries,
        )
        backend.create_checkpoint(f"repair-probe-{round_no}")

        def probe(command: str, timeout_s: float) -> str:
            self._push_worktree(backend)
            result = backend.run(command, timeout=int(timeout_s), workdir=CONTAINER_REPO)
            return f"exit {getattr(result, 'exit_code', '?')}\n{getattr(result, 'output', '')}"

        async def go() -> repair.RepairOutcome:
            provider = self.provider_factory()
            try:
                return await repair.run_repair_round(
                    provider, model=self.model, code_dir=self.code_dir, messages=messages, probe=probe,
                    denylist=self.denylist, max_tool_calls=self.max_tool_calls,
                )
            finally:
                if hasattr(provider, "aclose"):
                    await provider.aclose()

        try:
            return asyncio.run(go())
        finally:
            self._rollback_probes(backend, round_no)

    def _rollback_probes(self, backend: Any, round_no: int) -> None:
        """Undo whatever the probes did. RSA's ``rollback_to_checkpoint`` swallows a failed ``docker run`` (it returns
        False and leaves the backend without a container — sapg-2 GPU run 3, 2026-09-18, surfaced three steps later
        as "remote Docker container is not initialized"); here the failure is logged with the daemon's view and
        ``container_lost`` tells the worker to rebuild the environment at the new commit instead of stopping."""
        self.container_lost = False
        try:
            ok = backend.rollback_to_checkpoint(1)
        except Exception as exc:
            logger.warning("rollback after the repair agent's probes raised: {}", exc)
            ok = False
        if ok is False:
            self.container_lost = not getattr(backend, "container_id", "")
            diag = ""
            try:
                probe = backend._exec_host("docker ps -a --format '{{.ID}} {{.Image}} {{.Status}}' | head -5; docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | head -8; df -h / | tail -1", 60)
                diag = str(getattr(probe, "output", "") or "")[-800:]
            except Exception as exc:
                diag = f"(diagnostic failed: {exc})"
            logger.warning("rollback after the repair agent's probes failed (round {}); container_lost={}; daemon: {}", round_no, self.container_lost, diag)
            self._emit("repair.rollback_failed", round=round_no, container_lost=self.container_lost, daemon=diag[-400:])

    def _push_worktree(self, backend: Any) -> None:
        """Copy the agent's current files into the container so a probe sees them (small text files, base64 over exec)."""
        for path in sorted(self.code_dir.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) > 200_000:
                continue
            rel = path.relative_to(self.code_dir).as_posix()
            payload = base64.b64encode(data).decode("ascii")
            backend.run(
                f"mkdir -p {shlex.quote(str(Path(rel).parent))} && printf %s {shlex.quote(payload)} | base64 -d > {shlex.quote(rel)}",
                timeout=60, workdir=CONTAINER_REPO,
            )

    # -- code → container → criterion ----------------------------------------------------

    def _move_container_to(self, backend: Any, sha: str) -> None:
        url = self.serve_repo(self.repo.git_dir)
        result = backend.run(
            f"git remote set-url origin {shlex.quote(url)} && git fetch --quiet origin && git checkout --quiet --force --detach {sha} && git rev-parse HEAD",
            timeout=300, workdir=CONTAINER_REPO,
        )
        if not getattr(result, "success", False) or sha not in getattr(result, "output", ""):
            raise RuntimeError(f"container could not move to {sha[:12]}: {getattr(result, 'output', '')[-400:]}")

    def _refreeze(self, backend: Any, sha: str) -> Any:
        from apps.v2.agent_engine.rsa.freezer import Freezer

        ladder = self.frozen.ladder
        ladder.commit = sha
        for criterion in ladder.rungs.values():
            criterion.commit = sha
        return Freezer(self.store).freeze(ladder, overwrite=True, collector=backend.collect_test_ids)

    def _judge(self, backend: Any, frozen: Any, round_no: int) -> list[Any]:
        from apps.v2.agent_engine.rsa.adjudicator import Adjudicator
        from apps.v2.agent_engine.rsa.criterion import Rung
        from apps.v2.agent_engine.rsa.pipeline import _env_of

        adjudicator = Adjudicator(backend, workdir=self.config.workdir)
        verdicts: list[Any] = []
        for rung, fc in sorted(frozen.rungs.items(), key=lambda item: item[0].index):
            if rung > Rung(JUDGE_UP_TO):
                break
            verdict = adjudicator.adjudicate(fc, agent_env=_env_of(self.container_id), out_dir=self.out_dir / f"round{round_no}" / rung.value, tier=1)
            verdicts.append(verdict)
            if str(getattr(verdict, "verdict", "")) != "PASS":
                break
        return verdicts

    def _ensure_output_dirs(self, backend: Any, frozen: Any) -> None:
        """The output directory the goal names is created for the run (the trial's courtesy, not the code's job to
        find it missing): ``mkdir -p`` the parent of every expected artifact under an output directory."""
        parents: set[str] = set()
        for fc in frozen.rungs.values():
            for artifact in getattr(fc.criterion, "artifacts", None) or []:
                path = str(getattr(artifact, "path", "") or "")
                if repair.is_output_path(path) and "/" in path:
                    parents.add(path.rsplit("/", 1)[0])
        if not parents:
            return
        try:
            backend.run("mkdir -p " + " ".join(shlex.quote(p) for p in sorted(parents)), timeout=60, workdir=CONTAINER_REPO)
        except Exception as exc:
            logger.debug("mkdir of the output directories failed: {}", exc)

    def _first_failing_rung(self, last: dict[str, Any]) -> Any:
        wanted = str(last.get("rung") or "")
        for rung, fc in self.frozen.rungs.items():
            if rung.value == wanted:
                return fc
        return self.frozen.rungs[min(self.frozen.rungs, key=lambda r: r.index)] if self.frozen.rungs else None

    def _criterion_dict(self) -> dict[str, Any]:
        rungs = []
        for rung, fc in sorted(self.frozen.rungs.items(), key=lambda item: item[0].index):
            c = fc.criterion
            rungs.append({"rung": rung.value, "command": c.command, "workdir": c.workdir, "artifacts": [a.path for a in c.artifacts], "expected": list(fc.expected)})
        return {"rungs": rungs}




# ---------------------------------------------------------------------------
# S8: the four-box controller on RSA's pieces — a drop-in for ``rsa.agent.run_pipeline``
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ControllerSeams:
    """What the controller's real workers need beyond RSA's ``PipelineConfig`` (the runner fills this from its seams)."""

    repo: CodeRepo
    code_dir: Path
    goal: str
    repair_rounds: int
    environment_rounds: int
    store: Path
    out_dir: Path
    provider_factory: Callable[[], Any]
    model: str
    serve_repo: Callable[[Path], str]
    denylist: tuple[str, ...] = ()
    blueprint_excerpt: str = ""
    events: Callable[..., Any] | None = None
    max_tool_calls: int = repair.MAX_TOOL_CALLS
    token_cap: int = TOKEN_CAP
    #: tests: doubles for the RSA-side workers
    environment_round: Callable[..., Any] | None = None
    judge: Callable[..., list[Any]] | None = None
    #: the runner reads the final state from here
    on_state: Callable[[Any], None] | None = None
    #: the machine has a GPU (two-stage compute): GPU-class failures are then code failures, never an escalation
    gpu_available: bool = False


class ControllerWorkers(RepairLoop):
    """The real 搭建环境 / 远程初步执行 / 修复代码 boxes, on the machinery ``RepairLoop`` already had.

    Built inside ``run_line_pipeline`` where RSA's backend, SetupX configuration and the frozen ladder
    exist; ``RepairLoop.run`` (the post-hoc path) is not used here — the controller drives the boxes.
    """

    def __init__(self, seams: ControllerSeams, *, config: Any, frozen: Any, backend: Any) -> None:
        from types import SimpleNamespace

        super().__init__(
            config=config, outcome=SimpleNamespace(frozen=frozen, pipeline=SimpleNamespace(container_id="")), repo=seams.repo,
            code_dir=seams.code_dir, goal=seams.goal, rounds=seams.repair_rounds, store=seams.store, out_dir=seams.out_dir,
            provider_factory=seams.provider_factory, model=seams.model, serve_repo=seams.serve_repo,
            backend_factory=lambda _config: backend, denylist=seams.denylist, blueprint_excerpt=seams.blueprint_excerpt,
            events=seams.events, max_tool_calls=seams.max_tool_calls, token_cap=seams.token_cap,
            judge=seams.judge, environment_round=seams.environment_round,
        )
        self.seams = seams
        self.backend = backend
        self.last_verdict_objects: list[Any] = []
        self.environment_rounds = int(seams.environment_rounds)

    # -- 搭建环境 ---------------------------------------------------------------------

    def environment(self, state: Any) -> Any:
        from apps.v2.agent.paper2code.environment_controller import SetupResult

        self.container_id = state.container_id
        round_no = state.environment_rounds_used + 1
        if self._environment_seam is not None:
            result = self._environment_seam(self.backend, dict(state.last_verdict), round_no)
            if isinstance(result, SetupResult):
                return result
            return SetupResult(container_id=self.container_id or state.container_id or "c-seam", actions=int(result or 0))
        return self._environment_round_from_state(state, round_no)

    def _environment_round_from_state(self, state: Any, round_no: int) -> Any:
        from apps.v2.agent.paper2code.environment_controller import SetupResult
        from apps.v2.agent_engine.rsa import setup_loop as sl
        from apps.v2.agent_engine.rsa.adjudicator import Verdict
        from apps.v2.agent_engine.rsa.kickback import ActionLedger, grading_contract, kickback

        last = dict(state.last_verdict)
        fc = self._first_failing_rung(last) if last else self.frozen.rungs[min(self.frozen.rungs, key=lambda r: r.index)]
        contract = grading_contract(fc.criterion, fc.expected, disclose_ids=getattr(self.config, "disclose_ids", True))
        message = ""
        if last:
            verdict = Verdict(
                verdict=str(last.get("verdict") or "FAIL"), rung=str(last.get("rung") or fc.rung.value),
                expected_n=int(last.get("expected_n") or 0), passed_expected=int(last.get("passed_expected") or 0),
                missing=list(last.get("missing") or []), statuses=dict(last.get("statuses") or {}),
                exit_code=int(last.get("exit_code") or 0), log_tail=str(last.get("log_tail") or ""),
                failure_tails=dict(last.get("failure_tails") or {}), reason=str(last.get("reason") or ""),
            )
            message = kickback(verdict, ActionLedger(), round_no=round_no, rounds_total=self.environment_rounds, tails=verdict.failure_tails)
        sl.bind(sl.LoopContext(frozen=fc, workdir=self.config.workdir))
        actions, cid = sl.run_round(
            self.frozen.ladder.repo_url, revision=self.frozen.ladder.commit, max_steps=int(self.config.max_steps),
            contract=contract, kickback_text=message, container_id=self.container_id or None,
        )
        if cid:
            self.container_id = cid
            self.backend.attach(cid, CONTAINER_REPO)
        # SetupX's own last word: a FINISH that says the build or run needs a GPU is the escalation signal
        finish_text = " ".join(str(a) for a in (getattr(actions, "actions", None) or []) if str(a).startswith("FINISH"))
        return SetupResult(
            container_id=self.container_id, actions=len(getattr(actions, "actions", None) or []),
            completed=not getattr(actions, "error", ""), error=str(getattr(actions, "error", "") or ""),
            requested_help=bool(getattr(actions, "requested_help", False)),
            gpu_required=repair.gpu_needed(finish_text),
        )

    # -- 远程初步执行 -----------------------------------------------------------------

    def trial(self, state: Any) -> list[dict[str, Any]]:
        self.container_id = state.container_id
        self._ensure_output_dirs(self.backend, self.frozen)
        verdicts = (self._judge_seam or self._judge)(self.backend, self.frozen, state.round_no)
        self.last_verdict_objects = list(verdicts)
        return [verdict_dict(v) for v in verdicts]

    # -- 修复代码 ---------------------------------------------------------------------

    def repair(self, state: Any) -> Any:
        from apps.v2.agent.paper2code.environment_controller import RepairResult

        self.container_id = state.container_id
        if self.tokens_used >= self.token_cap:
            return RepairResult(attribution=repair.CODE, changed=False, error=f"phase token cap {self.token_cap} reached")
        before = self.repo.head() or ""
        outcome = self._code_round(self.backend, dict(state.last_verdict), state.round_no)
        self.tokens_used += int(outcome.usage.get("total_tokens", 0))
        if outcome.summary:
            self.summaries.append(f"round {state.round_no}: {outcome.summary[:200]}")
        new_sha = self.repo.commit(f"repair round {state.round_no}: {outcome.summary[:60] or 'agent changes'}")
        changed = new_sha != before
        if changed:
            if self.container_lost:
                # no container to move: re-serve the history so the rebuilt container clones the new commit
                self.serve_repo(self.repo.git_dir)
            else:
                self._move_container_to(self.backend, new_sha)
            self.frozen = self._refreeze(self.backend, new_sha)
        # RSA's rollback replaces the container (docker rm + run from the checkpoint): the id the controller holds
        # is stale after every repair round (sapg-s9-off, 2026-09-18: SetupX was then pointed at the removed one)
        self.container_id = str(getattr(self.backend, "container_id", "") or "") if not self.container_lost else ""
        return RepairResult(
            attribution=outcome.attribution, changed=changed, commit=new_sha if changed else "", summary=outcome.summary,
            record=outcome.record(), container_lost=self.container_lost, container_id=self.container_id,
        )


def normalise_ladder_artifacts(ladder: Any, *, output_dirs: tuple[str, ...] = repair.OUTPUT_DIRS) -> list[str]:
    """T3 (owner 2026-09-18): the criterion's artifact paths as the goal asked for them, before the ladder is frozen.

    RSA's compiler writes the run's outputs relative to the repository (``workspace/out/smoke/curves.npz``,
    resolved under ``/workspace/repo/`` by the rendered criterion) even when the command passes an absolute
    ``--output-dir /workspace/out/smoke`` — on the T2 rerun the code wrote where it was told, the artifact tests
    looked under the repository, and the third repair round "fixed" it by mirroring the outputs into the
    repository. Two rules, line side only (RSA untouched): (a) a relative artifact under an output directory
    becomes the absolute path; (b) when the command names an absolute output directory, a relative artifact whose
    basename also appears under that directory is the compiler hedging — dropped. Returns what changed."""
    import dataclasses

    changed: list[str] = []
    rungs = getattr(ladder, "rungs", None) or {}
    criteria = list(rungs.values()) if isinstance(rungs, dict) else list(rungs)
    for criterion in criteria:  # Criterion is a plain dataclass (mutable), Artifact is frozen
        artifacts = list(getattr(criterion, "artifacts", None) or [])
        if not artifacts:
            continue
        command = str(getattr(criterion, "command", "") or "")
        pattern = r"(?<!\S)((?:/" + "|/".join(re.escape(o) for o in output_dirs) + r")(?:/[^\s'\"]*)?)"
        absolute_dirs = {m.rstrip("/") for m in re.findall(pattern, command)}
        made_absolute = []
        for artifact in artifacts:
            path = str(artifact.path or "")
            if not path.startswith("/") and any(path == o or path.startswith(o + "/") for o in output_dirs):
                artifact = dataclasses.replace(artifact, path="/" + path)
                changed.append(f"{path} -> {artifact.path}")
            made_absolute.append(artifact)
        absolute = {a.path for a in made_absolute if str(a.path).startswith("/")}
        kept = []
        for artifact in made_absolute:
            path = str(artifact.path)
            if not path.startswith("/") and absolute_dirs:
                name = path.rsplit("/", 1)[-1]
                twin = next((ab for ab in absolute if ab.rsplit("/", 1)[-1] == name and any(ab.startswith(d + "/") for d in absolute_dirs)), None)
                if twin is not None:
                    changed.append(f"dropped {path} (duplicate of {twin})")
                    continue
            kept.append(artifact)
        criterion.artifacts = kept
    return changed


def install_artifact_normaliser(events: Callable[..., Any] | None = None) -> Callable[[], None]:
    """Replace ``rsa.pipeline.Freezer`` for the run with one that normalises the ladder first; returns the restore
    (the same shape as the controller's installation, S8)."""
    from apps.v2.agent_engine.rsa import pipeline as rsa_pipeline

    original = rsa_pipeline.Freezer

    class LineFreezer(original):  # type: ignore[misc,valid-type]
        def freeze(self, ladder: Any, **kwargs: Any) -> Any:
            changed = normalise_ladder_artifacts(ladder)
            if changed:
                logger.info("normalised {} artifact path(s) before the freeze: {}", len(changed), changed)
                if events is not None:
                    try:
                        events("controller.artifacts_normalised", changes=changed)
                    except Exception:
                        pass
            return super().freeze(ladder, **kwargs)

    rsa_pipeline.Freezer = LineFreezer

    def restore() -> None:
        rsa_pipeline.Freezer = original

    return restore


def run_line_pipeline(frozen: Any, cfg: Any, *, out_dir: Path | None = None, seams: ControllerSeams) -> Any:
    """The line's ``run_pipeline``: RSA's asset gate as it is, then the four-box controller instead of RSA's
    Router. Same signature and return shape (``PipelineOutcome``) so ``RSAAgent.run`` needs no change —
    the runner sets ``rsa.agent.run_pipeline`` to a partial of this for the duration of the run (S8)."""
    from apps.v2.agent.paper2code.environment_controller import DONE, ControllerState, EnvironmentController, Workers
    from apps.v2.agent_engine.rsa.asset_gate import AssetGate, AssetReport
    from apps.v2.agent_engine.rsa.criterion import Rung
    from apps.v2.agent_engine.rsa.escalation import build_card
    from apps.v2.agent_engine.rsa.meter import TokenMeter
    from apps.v2.agent_engine.rsa.pipeline import PipelineOutcome, _container_backend, _setupx_container
    from apps.v2.agent_engine.rsa.router import RungOutcome, RunOutcome, Terminal
    from apps.v2.agent_engine.rsa.setupx_interop import setupx_configured

    t0 = time.monotonic()
    meter = TokenMeter()
    ladder = frozen.ladder
    remote = _container_backend(cfg)
    with setupx_configured(cfg.backend, base_image=cfg.base_image, remote_backend=remote):
        # an "asset" under the run's output directory is the compiler declaring the run's own output as an input
        # (sapg-2, 2026-09-18: `output_dir` = /workspace/out/smoke, "must exist and be writable"); the trial creates
        # those directories, so such entries are dropped before G1 rather than blocking the run
        dropped = [a for a in ladder.assets if repair.is_output_path(str(getattr(a, "path", "") or ""))]
        if dropped:
            ladder.assets = [a for a in ladder.assets if a not in dropped]
            logger.info("dropped {} output-directory asset(s) from the ladder before G1: {}", len(dropped), [getattr(a, "name", "?") for a in dropped])
            if seams.events is not None:
                try:
                    seams.events("controller.assets_dropped", assets=[getattr(a, "name", "?") for a in dropped])
                except Exception:
                    pass
        assets = AssetReport()
        if ladder.assets:
            with _setupx_container(ladder.repo_url, ladder.commit, cfg) as (bridge, _env):
                assets = AssetGate(bridge, workdir=cfg.workdir).check(ladder.assets)
            if assets.blocked:
                c = ladder.ordered()[0]
                card = build_card("assets", c, None, extra_facts=assets.facts())
                return PipelineOutcome(terminal=Terminal.BLOCKED.value, assets=assets, elapsed_s=round(time.monotonic() - t0, 1), escalation_md=card.render(), tokens=meter.to_dict())
        from apps.v2.agent_engine.rsa import setup_loop as sl

        sl._METER = meter
        workers = ControllerWorkers(seams, config=cfg, frozen=frozen, backend=remote)
        controller = EnvironmentController(
            Workers(environment=workers.environment, trial=workers.trial, repair=workers.repair),
            environment_rounds=seams.environment_rounds, repair_rounds=seams.repair_rounds,
            repo_modules=lambda: repair.repo_modules(seams.code_dir), events=seams.events,
            state=ControllerState(commit=ladder.commit, gpu_available=bool(seams.gpu_available)),
        )
        try:
            state = controller.run()
        finally:
            if remote is not None and hasattr(remote, "cleanup_snapshots"):
                try:
                    remote.cleanup_snapshots()
                except Exception as exc:
                    logger.debug("cleanup_snapshots: {}", exc)
    if seams.on_state is not None:
        seams.on_state(state)

    per_rung: list[Any] = []
    reached = ""
    for verdict in workers.last_verdict_objects:
        passed = str(getattr(verdict, "verdict", "")) == "PASS"
        per_rung.append(RungOutcome(rung=str(getattr(verdict, "rung", "")), terminal=Terminal.SUCCESS if passed else Terminal.ESCALATED, verdict=verdict, note=state.stop_reason))
        if passed:
            reached = str(getattr(verdict, "rung", ""))
    if state.route == DONE and not reached:
        reached = max((r.value for r in frozen.rungs if r <= Rung(JUDGE_UP_TO)), default="")
    run = RunOutcome(terminal=Terminal.SUCCESS if state.passed else Terminal.ESCALATED, reached=reached, per_rung=per_rung)
    result = PipelineOutcome(
        terminal=run.terminal.value, reached=reached, assets=assets, outcome=run, tokens=meter.to_dict(),
        container_id=state.container_id, elapsed_s=round(time.monotonic() - t0, 1),
    )
    result.frozen = workers.frozen  # type: ignore[attr-defined]  # the re-pinned ladder after repairs
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "terminal": result.terminal, "reached": reached, "elapsed_s": result.elapsed_s, "container_id": state.container_id,
            "tokens": result.tokens, "assets": assets.to_dict() if assets else None, "controller": state.to_dict(),
        }
        (out_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


__all__ = [
    "CONTAINER_REPO",
    "JUDGE_UP_TO",
    "TOKEN_CAP",
    "ControllerSeams",
    "ControllerWorkers",
    "RepairLoop",
    "RoundRecord",
    "failure_signature",
    "failure_text",
    "install_artifact_normaliser",
    "last_verdict_of",
    "normalise_ladder_artifacts",
    "run_line_pipeline",
    "verdict_dict",
]
