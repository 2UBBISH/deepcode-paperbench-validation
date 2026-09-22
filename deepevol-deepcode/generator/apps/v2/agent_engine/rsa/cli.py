"""Command line: user-facing setup plus lower-level inspect/run commands.

Compiling and running are separate commands on purpose. Between them a human
reads the falsification report and decides whether the ruler is the right one --
which is the whole point of freezing it before any configuring starts. Running
`rsa run` is the moment the criterion becomes binding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import InteractionRequest, InteractionResponse, run_user_instruction
from .criterion import Rung
from .freezer import Freezer
from .pipeline import PipelineConfig, compile_ladder, run_pipeline

DEFAULT_STORE = Path.home() / ".rsa" / "store"
DEFAULT_WORK = Path.home() / ".rsa" / "work"


def _cfg(a: argparse.Namespace) -> PipelineConfig:
    approve = {Rung(r) for r in (a.approve or [])}
    cfg = PipelineConfig(
        store=Path(a.store), work=Path(a.work), backend=a.backend,
        max_steps=a.max_steps, max_rounds=a.max_rounds, stall_k=a.stall_k,
        wall_seconds=a.wall_seconds, token_budget=a.token_budget,
        approve=approve, disclose_ids=not a.hide_ids,
        execute_falsification=not a.no_exec_falsify,
        execution_backend=("local" if getattr(a, "local", "") else "remote"),
        remote_target=getattr(a, "remote", ""),
        remote_username=getattr(a, "remote_user", "root"),
        remote_password=os.environ.get("RSA_REMOTE_PASSWORD") or None,
        remote_private_key=getattr(a, "remote_key", "") or None,
    )
    a._rsa_cfg = cfg
    return cfg


def cmd_compile(a) -> int:
    cfg = _cfg(a)
    out = compile_ladder(a.repo, a.goal, cfg, commit=a.commit,
                         clarification=a.answer or "",
                         local_path=Path(a.local) if a.local else None)
    if out.question:
        print("The goal needs one clarification before a criterion can be written:\n")
        print(f"  {out.question}\n")
        print("Re-run with --answer '<your answer>'.")
        return 2

    fl = out.frozen
    print(f"ladder {fl.ladder_id}  ->  {fl.root}")
    print(f"repo   {fl.ladder.repo_url} @ {fl.ladder.commit[:12]}")
    print(f"goal   {fl.ladder.goal}")
    if out.notes:
        print(f"notes  {out.notes}")
    print(f"tokens {out.tokens.get('total', '?')} "
          f"({out.tokens.get('cny', 0)} CNY, {out.tokens.get('model', '?')})")
    print()
    for rung, fc in sorted(fl.rungs.items(), key=lambda kv: kv[0].index):
        c = fc.criterion
        weak = not c.metrics and not c.sanity and rung.value != "G0"
        print(f"[{rung.value}] {len(fc.expected)} assertions  ->  {fc.basename}"
              + ("   << NO NUMERIC CHECKS" if weak else ""))
        print(f"      command: {c.command}")
        if c.shrink:
            print(f"      shrink:  {c.shrink.knobs}  flatten: {c.shrink.flatten}")
        for a_ in c.artifacts:
            print(f"      artefact {a_.check:<9} {a_.path}"
                  + (f"  (stage: {a_.stage})" if a_.stage else ""))
        for m in c.metrics:
            bound = m.comparison.render() if m.comparison else "(no bound; sanity only)"
            print(f"      metric   {m.name} {bound}  <- {m.source.render()}")
        for s in c.sanity:
            print(f"      sanity   {s.kind} {s.metric or (s.source.render() if s.source else '')}")

        rep = out.falsification.get(rung.value)
        if rep:
            print(f"      falsification: {rep.summary()}")
            for r in rep.reasons:
                print(f"        REJECTED: {r}")
            for d in rep.disclosures:
                print(f"        note: {d}")
        print()

    if fl.ladder.assets:
        print("assets to be checked before any configuring (G1):")
        for x in fl.ladder.assets:
            print(f"      {x.kind:<10} {x.name}"
                  + (f"  <- {x.why}" if x.why else ""))
        print()

    if out.disclosures:
        print("READ THESE BEFORE RUNNING -- values the model chose rather than measured:")
        for d in out.disclosures:
            print(f"  ! {d}")
        print()

    blocked = [r for r in out.falsification.values() if r.must_recompile]
    if blocked:
        print("This ladder was REJECTED by falsification and must be recompiled.")
        return 3
    print(f"Frozen. Run it with:  rsa run {fl.ladder_id}")
    return 0


class ConsoleInteraction:
    """Small terminal adapter for the same interaction protocol a UI uses."""

    def request(self, event: InteractionRequest) -> InteractionResponse:
        print(f"\n[{event.kind}]\n{event.message}")
        if event.kind == "clarification":
            return InteractionResponse(action="answer", message=input("answer> ").strip())
        if event.kind == "asset":
            choice = input("type 'retry', 'drop' or 'stop'> ").strip().lower()
            if choice == "drop":
                return InteractionResponse(action="drop_assets")
            if choice == "retry":
                return InteractionResponse(action="retry")
            return InteractionResponse(action="stop")
        if event.kind == "approval":
            choice = input("type 'approve' or 'stop'> ").strip().lower()
            return InteractionResponse(action="approve" if choice == "approve" else "stop")
        choice = input("type 'continue', 'approve' or 'stop'> ").strip().lower()
        if choice == "continue":
            return InteractionResponse(action="continue", grant={"extra_rounds": 2,
                                                                    "wall_multiplier": 1.5})
        if choice == "approve":
            return InteractionResponse(action="approve")
        return InteractionResponse(action="stop")


def cmd_setup(a) -> int:
    """Accept one user instruction and own the full setup conversation."""
    cfg = _cfg(a)
    out = run_user_instruction(
        a.repo,
        a.instruction,
        cfg,
        interaction=ConsoleInteraction(),
        revision=a.commit,
        local_path=Path(a.local) if a.local else None,
    )
    print(f"\n=== {out.status.value.upper()} ===")
    if out.pipeline:
        print(f"reached: {out.pipeline.reached or '(none)'}")
        if out.pipeline.container_id:
            print(f"container: {out.pipeline.container_id}")
    if out.pending and out.status.value in ("needs_user", "recompile"):
        print(out.pending.message)
    if out.error:
        print(f"error: {out.error}", file=sys.stderr)
    return 0 if out.ok else 1


def cmd_run(a) -> int:
    cfg = _cfg(a)
    fl = Freezer(cfg.store).load(a.ladder_id)
    fl.check_integrity()
    out_dir = Path(a.out) if a.out else (cfg.work / "runs" / a.ladder_id)
    res = run_pipeline(fl, cfg, out_dir=out_dir)

    print(f"\n=== {res.terminal.upper()} ===")
    print(f"reached: {res.reached or '(none)'}   elapsed: {res.elapsed_s}s   "
          f"tokens: {res.tokens.get('total_tokens', 0)} "
          f"({res.tokens.get('cny', 0)} CNY)")
    if res.assets and res.assets.results:
        print(f"assets: {res.assets.summary()}")
    if res.outcome:
        for ro in res.outcome.per_rung:
            print(f"  [{ro.rung}] {ro.terminal.value}"
                  + (f" - {ro.note}" if ro.note else ""))
            for r in ro.rounds:
                mark = f" ({r.diverted})" if r.diverted else ""
                print(f"      round {r.round_no}: {r.verdict} "
                      f"{r.passed_expected}/{r.expected_n}{mark}  "
                      f"{r.actions_n} actions  {r.elapsed_s}s")
                if r.loop_error:
                    print(f"        SETUP LOOP CRASHED: {r.loop_error}")
    if res.escalation_md:
        print("\n" + res.escalation_md)
    print(f"\nrecord: {out_dir}/result.json")
    return 0 if res.terminal == "success" else 1


def cmd_show(a) -> int:
    fl = Freezer(Path(a.store)).load(a.ladder_id)
    print(json.dumps(fl.ladder.to_dict(), ensure_ascii=False, indent=2))
    print("\n--- freeze history ---")
    for h in Freezer(Path(a.store)).history(a.ladder_id):
        print(json.dumps(h, ensure_ascii=False))
    return 0


def cmd_adjudicate(a) -> int:
    """Judge a container that already exists. The ruler, used on its own."""
    from .adjudicator import Adjudicator
    from .remote_backend import RemoteDockerBackend
    fl = Freezer(Path(a.store)).load(a.ladder_id)
    fc = fl.get(Rung(a.rung))
    bridge = RemoteDockerBackend(
        a.remote,
        username=a.remote_user,
        password=os.environ.get("RSA_REMOTE_PASSWORD") or None,
        private_key=a.remote_key or None,
        workdir=fc.criterion.workdir,
    )
    a._rsa_backend = bridge
    bridge.attach(a.container, fc.criterion.workdir)
    v = Adjudicator(bridge, workdir=fc.criterion.workdir).adjudicate(
        fc, reset=not a.no_reset, out_dir=Path(a.out) if a.out else None)
    print(v.summary())
    for al in v.alarms:
        print(f"  ALARM: {al}")
    for t in v.missing[:40]:
        print(f"  missing: {t}")
    return 0 if v.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rsa",
        description="Accept a user's repository instruction, configure its environment, "
                    "and prove the requested pytest selection or command passes.")
    p.add_argument("--store", default=str(DEFAULT_STORE), help="frozen criteria store")
    p.add_argument("--work", default=str(DEFAULT_WORK), help="scratch directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--remote", default=os.environ.get("RSA_REMOTE_TARGET", ""),
                        help="SSH URL/command for the Docker host (or RSA_REMOTE_TARGET)")
        sp.add_argument("--remote-user", default=os.environ.get("RSA_REMOTE_USER", "root"))
        sp.add_argument("--remote-key", default=os.environ.get("RSA_REMOTE_KEY", ""),
                        help="SSH private-key path; password comes from RSA_REMOTE_PASSWORD")
        sp.add_argument("--backend", default="small", choices=["small", "large"],
                        help="small = gemma (free); large = deepseek (billed per token)")
        sp.add_argument("--max-steps", type=int, default=200)
        sp.add_argument("--max-rounds", type=int, default=8)
        sp.add_argument("--stall-k", type=int, default=3,
                        help="rounds without the failing set shrinking before escalating")
        sp.add_argument("--wall-seconds", type=int, default=7200)
        sp.add_argument("--token-budget", type=int, default=2_000_000)
        sp.add_argument("--approve", action="append", choices=["G3", "G4'"],
                        help="approve an expensive rung (hours to days); repeatable")
        sp.add_argument("--hide-ids", action="store_true",
                        help="do not disclose the criterion's assertion ids to the agent")
        sp.add_argument("--no-exec-falsify", action="store_true",
                        help="skip falsification points 1 and 2 (no Docker)")

    c = sub.add_parser("compile", help="write, falsify and freeze the criterion ladder")
    c.add_argument("repo", help="repository URL")
    c.add_argument("--goal", required=True, help="what you want to run, in your own words")
    c.add_argument("--commit", default="", help="pin to this commit (default: HEAD)")
    c.add_argument("--local", default="", help="use an existing checkout instead of cloning")
    c.add_argument("--answer", default="", help="answer to the compiler's clarifying question")
    common(c)
    c.set_defaults(func=cmd_compile)

    u = sub.add_parser(
        "setup",
        help="accept a user's instruction, generate pytest criteria, configure and verify",
    )
    u.add_argument("repo", help="repository URL")
    u.add_argument("--instruction", required=True,
                   help="the user's natural-language setup/run instruction")
    u.add_argument("--commit", default="", help="pin to this commit (default: HEAD)")
    u.add_argument("--local", default="", help="use an existing checkout for development")
    common(u)
    u.set_defaults(func=cmd_setup)

    r = sub.add_parser("run", help="configure the environment and adjudicate")
    r.add_argument("ladder_id")
    r.add_argument("--out", default="", help="where to write the run record")
    common(r)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("show", help="print a frozen ladder and its amendment history")
    s.add_argument("ladder_id")
    s.set_defaults(func=cmd_show)

    j = sub.add_parser("adjudicate", help="judge an existing container against a rung")
    j.add_argument("ladder_id")
    j.add_argument("--container", required=True)
    j.add_argument("--rung", default="G2")
    j.add_argument("--no-reset", action="store_true",
                   help="preview only: do not restore the working tree (not authoritative)")
    j.add_argument("--out", default="")
    j.add_argument("--remote", default=os.environ.get("RSA_REMOTE_TARGET", ""),
                   help="SSH URL/command for the Docker host (or RSA_REMOTE_TARGET)")
    j.add_argument("--remote-user", default=os.environ.get("RSA_REMOTE_USER", "root"))
    j.add_argument("--remote-key", default=os.environ.get("RSA_REMOTE_KEY", ""))
    j.set_defaults(func=cmd_adjudicate)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    finally:
        cfg = getattr(a, "_rsa_cfg", None)
        backend = getattr(cfg, "remote_backend", None) if cfg is not None else None
        standalone = getattr(a, "_rsa_backend", None)
        for candidate in (backend, standalone):
            if candidate is not None and hasattr(candidate, "close"):
                try:
                    candidate.close()
                except Exception:
                    pass
        if cfg is not None:
            cfg.remote_backend = None


if __name__ == "__main__":
    sys.exit(main())
