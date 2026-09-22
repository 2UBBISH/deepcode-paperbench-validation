"""End to end: goal -> criteria -> falsify -> freeze -> assets -> configure -> verdict.

The order is the design document's, and each step's authority is exactly what it
was given there. Nothing here decides anything: it wires the components together
and writes down what happened.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adjudicator import Adjudicator, Verdict
from .asset_gate import AssetGate, AssetReport
from .bridge import DockerBridge
from .criterion import Criterion, Ladder, Rung
from .escalation import build_card
from .falsifier import FalsificationReport, falsify, tracked_at
from .freezer import Freezer, FrozenCriterion, FrozenLadder
from .meter import TokenMeter, attach as attach_meter
from .router import Budget, RoundActions, Router, RouterConfig, RunOutcome, Terminal
from .setupx_interop import DEFAULT_BASE_IMAGE, setupx_configured


@dataclass
class PipelineConfig:
    store: Path
    work: Path
    backend: str = "small"
    base_image: str = DEFAULT_BASE_IMAGE
    workdir: str = "/workspace/repo"
    max_steps: int = 200
    max_rounds: int = 8
    stall_k: int = 3
    wall_seconds: int = 7200
    token_budget: int = 2_000_000
    approve: set[Rung] = field(default_factory=set)
    disclose_ids: bool = True
    execute_falsification: bool = True
    # Remote execution is the product default. ``local`` remains an explicit
    # development/test escape hatch and is never selected implicitly.
    execution_backend: str = "remote"
    remote_target: str = ""
    remote_username: str = "root"
    remote_password: str | None = None
    remote_private_key: str | None = None
    remote_backend: Any | None = None


def _container_backend(cfg: PipelineConfig):
    """Return the one backend shared by compile, setup and adjudication."""
    if cfg.execution_backend == "local":
        return None
    if cfg.remote_backend is not None:
        return cfg.remote_backend
    from .remote_backend import RemoteDockerBackend
    if not cfg.remote_target:
        raise RuntimeError(
            "RSA requires a remote_target; configure the SSH access URL before "
            "starting setup (use execution_backend='local' only for tests)"
        )
    cfg.remote_backend = RemoteDockerBackend(
        cfg.remote_target,
        username=cfg.remote_username,
        password=cfg.remote_password,
        private_key=cfg.remote_private_key,
        base_image=cfg.base_image,
        workdir=cfg.workdir,
    )
    return cfg.remote_backend


# --------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------

def clone(repo_url: str, dest: Path, commit: str = "") -> str:
    """Full clone, then pin. Returns the resolved commit.

    Not `--depth 1`. A shallow clone cannot be reset to a named commit later, and
    the Adjudicator's whole anti-cheat guarantee is `git checkout --force <sha>`.
    It also breaks `setuptools_scm` / `versioneer`, which several research repos
    use to derive a version at install time.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "--quiet", repo_url, str(dest)],
                   check=True, capture_output=True, text=True, timeout=1800)
    if commit:
        subprocess.run(["git", "checkout", "--quiet", "--force", commit],
                       cwd=dest, check=True, capture_output=True, text=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest,
                          capture_output=True, text=True, check=True)
    return head.stdout.strip()


@dataclass
class CompileOutcome:
    frozen: FrozenLadder | None = None
    falsification: dict[str, FalsificationReport] = field(default_factory=dict)
    question: str = ""
    notes: str = ""
    disclosures: list[str] = field(default_factory=list)
    tokens: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.frozen is not None and not self.question


def compile_ladder(repo_url: str, goal: str, cfg: PipelineConfig, *,
                   commit: str = "", clarification: str = "",
                   local_path: Path | None = None) -> CompileOutcome:
    from .compiler import Compiler
    from .compiler.llm import LLM

    remote = _container_backend(cfg)
    checkout = local_path or (cfg.work / "checkout")
    resolved = commit
    tracked: set[str] | None = None
    if remote is not None:
        if local_path is not None:
            raise RuntimeError(
                "local_path is not accepted by the remote product path; provide a "
                "repository URL so cloning happens on the remote server"
            )
        resolved, tracked = remote.checkout_to_local(repo_url, commit, checkout)
    elif local_path is None:
        resolved = clone(repo_url, checkout, commit)
    elif not resolved:
        resolved = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                                  capture_output=True, text=True).stdout.strip()

    with setupx_configured(cfg.backend, base_image=cfg.base_image,
                           remote_backend=remote) as session:
        llm = LLM.from_env()
        result = Compiler(llm, workdir=cfg.workdir).compile(
            checkout, repo_url=repo_url, commit=resolved, goal=goal,
            clarification=clarification)
        tokens = {"model": session.model, "backend": session.backend,
                  **llm.usage.__dict__, "total": llm.usage.total,
                  "cny": llm.usage.cny()}

    if result.needs_user:
        return CompileOutcome(question=result.question, tokens=tokens)

    ladder = result.ladder
    ladder.approved_through = max(cfg.approve, default=Rung.G2)
    freezer = Freezer(cfg.store)
    frozen = freezer.freeze(
        ladder, overwrite=True,
        collector=remote.collect_test_ids if remote is not None else None,
    )

    reports: dict[str, FalsificationReport] = {}
    if tracked is None:
        from .bridge import LocalBridge
        tracked = tracked_at(LocalBridge(checkout), str(checkout), resolved)
    for rung, fc in frozen.rungs.items():
        bare = tl = None
        if cfg.execute_falsification:
            bare = _bare_provisioner(repo_url, resolved, cfg)
            tl = _toplevel_provisioner(repo_url, resolved, cfg)
        reports[rung.value] = falsify(fc, tracked=tracked, bare=bare, toplevel=tl)

    return CompileOutcome(frozen=frozen, falsification=reports,
                          notes=result.notes, disclosures=result.disclosures,
                          tokens=tokens)


def _bare_provisioner(repo_url: str, commit: str, cfg: PipelineConfig):
    """Point (1): the base image with the repository cloned and nothing installed.

    Built with SetupX's own `create_container`, so it is the same container shape
    the agent will get. If the bare environment were assembled differently, point
    (1) would be answering a question about a container that never exists.
    """
    def _p():
        with _setupx_container(repo_url, commit, cfg) as (bridge, env):
            yield bridge, {}
    return _p


def _toplevel_provisioner(repo_url: str, commit: str, cfg: PipelineConfig):
    def _p():
        with _setupx_container(repo_url, commit, cfg) as (bridge, env):
            # Top-level requirements only. No datasets, no weights, no extras --
            # that is precisely the environment that separates a criterion which
            # reaches the end of the script from one that stops at the imports.
            bridge.run(
                "python3 -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple "
                "-r requirements.txt || true", timeout=1800, workdir=cfg.workdir)
            yield bridge, {}
    return _p


@contextmanager
def _setupx_container(repo_url: str, commit: str, cfg: PipelineConfig):
    remote = _container_backend(cfg)
    with setupx_configured(cfg.backend, base_image=cfg.base_image,
                           remote_backend=remote):
        from .setupx_interop import import_setupx
        _, em_mod, _, _ = import_setupx()
        env = em_mod.EnvironmentManager()
        cid = env.create_container(repo_url, commit)
        try:
            bridge = getattr(env, "backend", None)
            if bridge is None:
                bridge = DockerBridge(cid, workdir=cfg.workdir)
            yield bridge, env
        finally:
            try:
                # Temporary provisioning/gold environments own their checkpoint
                # images. Destroying only the container leaves those snapshots
                # behind and makes every task consume another image layer.
                env.cleanup()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

@dataclass
class PipelineOutcome:
    terminal: str
    reached: str = ""
    assets: AssetReport | None = None
    outcome: RunOutcome | None = None
    tokens: dict = field(default_factory=dict)
    container_id: str = ""
    elapsed_s: float = 0.0
    escalation_md: str = ""

    def to_dict(self) -> dict:
        return {
            "terminal": self.terminal,
            "reached": self.reached,
            "elapsed_s": self.elapsed_s,
            "container_id": self.container_id,
            "tokens": self.tokens,
            "assets": self.assets.to_dict() if self.assets else None,
            "run": self.outcome.to_dict() if self.outcome else None,
        }


def run_pipeline(frozen: FrozenLadder, cfg: PipelineConfig, *,
                 out_dir: Path | None = None) -> PipelineOutcome:
    from . import setup_loop as sl

    t0 = time.monotonic()
    meter = TokenMeter()
    ladder = frozen.ladder
    repo_url, commit = ladder.repo_url, ladder.commit
    state = {"cid": "", "round": 0}

    remote = _container_backend(cfg)
    with setupx_configured(cfg.backend, base_image=cfg.base_image,
                           remote_backend=remote):
        # -- G1 first. A missing dataset must not cost an hour of installing. --
        assets = AssetReport()
        if ladder.assets:
            with _setupx_container(repo_url, commit, cfg) as (bridge, env):
                assets = AssetGate(bridge, workdir=cfg.workdir).check(ladder.assets)
            if assets.blocked:
                c = ladder.ordered()[0]
                card = build_card("assets", c, None, extra_facts=assets.facts())
                return PipelineOutcome(
                    terminal=Terminal.BLOCKED.value, assets=assets,
                    elapsed_s=round(time.monotonic() - t0, 1),
                    escalation_md=card.render(),
                    tokens=meter.to_dict())

        first = frozen.rungs[ladder.ordered()[0].rung]
        sl.bind(sl.LoopContext(frozen=first, workdir=cfg.workdir))

        def loop(*, round_no, contract, kickback, rebuild=False):
            state["round"] = round_no
            if rebuild and state["cid"]:
                # Section 8.1's escalation of last resort. Dropping the container
                # loses real progress, which is why it happens once, immediately
                # before escalating, rather than after every failed round.
                state["cid"] = ""
            actions, cid = sl.run_round(
                repo_url, revision=commit, max_steps=cfg.max_steps,
                contract=contract, kickback_text=kickback,
                container_id=state["cid"] or None,
                is_final_round=False)
            state["cid"] = cid
            return actions

        def adjudicate(fc: FrozenCriterion, tier: int) -> Verdict:
            sl.bind(sl.LoopContext(frozen=fc, workdir=cfg.workdir))
            if not state["cid"]:
                return Verdict(verdict="GRADE_ERROR", rung=fc.rung.value,
                               reason="the setup loop produced no container")
            bridge = remote if remote is not None else DockerBridge(state["cid"], workdir=cfg.workdir)
            adj = Adjudicator(bridge, workdir=cfg.workdir)
            return adj.adjudicate(
                fc, agent_env=_env_of(state["cid"]), tier=tier,
                out_dir=(out_dir / f"round{state['round']}") if out_dir else None)

        router = Router(
            RouterConfig(max_rounds=cfg.max_rounds, stall_k=cfg.stall_k,
                         disclose_ids=cfg.disclose_ids),
            setup_loop=loop, adjudicate=adjudicate,
            budget=Budget(wall_seconds=cfg.wall_seconds, tokens=cfg.token_budget,
                          tokens_used=lambda: meter.total),
            approve_rung=lambda r: r in cfg.approve,
        )
        # Meter whatever client the first round creates.
        sl._METER = meter
        try:
            outcome = router.run(frozen)
        finally:
            # A completed router can no longer roll back. Keep the configured
            # container for the caller, but reclaim remote checkpoint images so
            # long-lived remote benchmark runs do not leak one image per round.
            if remote is not None:
                remote.cleanup_snapshots()

    res = PipelineOutcome(
        terminal=outcome.terminal.value, reached=outcome.reached,
        assets=assets, outcome=outcome, tokens=meter.to_dict(),
        container_id=state["cid"], elapsed_s=round(time.monotonic() - t0, 1),
        escalation_md=outcome.card.render() if outcome.card else "")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(
            json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if res.escalation_md:
            (out_dir / "escalation.md").write_text(res.escalation_md, encoding="utf-8")
    return res


def _env_of(container_id: str) -> dict[str, str]:
    """SET_ENV values the agent established, read back from the live manager."""
    from . import setup_loop as sl
    return dict(getattr(sl, "_LAST_ENV", {}) or {})
