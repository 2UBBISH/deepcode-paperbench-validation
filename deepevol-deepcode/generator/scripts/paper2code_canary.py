#!/usr/bin/env python3
"""Drive one Paper2Code run from a run directory (PLAN.md C7).

  init    --run-dir runs/x --paper-dir <PaperBench paper dir> [--model M] [--compute aliyun|local]
          [--compute-tier enough|comfortable] [--run-hours H] [--skip index] [--ask]
  run     --run-dir runs/x [--until environment_run] [--ask]   phases in order; stops on failed or waiting
  step    --run-dir runs/x --phase <name> [--ask]              one phase (earlier phases must be completed)
  rerun   --run-dir runs/x --phase <name>                      supersede the phase and everything after it, then run it
  status  --run-dir runs/x                                     per-phase status, gates, lease
  release --run-dir runs/x                                     delete the rented machine (the backstop)
  submit  --run-dir runs/x --paper <id> --trial <name> [--dest-root ~/pb_submissions] [--force]
  relocate --run-dir runs/x        # after copying/moving a run directory: rewrite the absolute path recorded inside it
  rerun   --run-dir runs/x --phase environment_run [--keep-tree] --env-file …   # repeat step 10 from the stage-9 tree
                                                               copy generate_code/ into the judge's pool; only when all
                                                               four gates passed and environment_run completed

Phases: intake criteria plan plan_review references acquire index implement compute environment_run optimize.
The provider key is read from the variable named by --provider-key-env (default PARATERA_API_KEY), which
--env-file can supply; with --compute aliyun the ALIYUN_* / DEEPEVOL_API_ALIYUN_* account comes from a
second --env-file. Nothing from those files is ever printed. Logs go to <run-dir>/canary.log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from apps.v2.agent.paper2code import config as cfg  # noqa: E402
from apps.v2.agent.paper2code.driver import Driver, DriverError, load_env_files  # noqa: E402
from apps.v2.agent.paper2code.gates import GateFailed  # noqa: E402
from apps.v2.agent.paper2code.phases import PHASES  # noqa: E402


def _engine_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _configure_logging(run_dir: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("PAPER2CODE_LOG_LEVEL", "INFO"), format="{time:HH:mm:ss} | {level: <7} | {message}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "canary.log", level="DEBUG", rotation="50 MB", retention=5, enqueue=True,
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{line} | {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("init", "run", "step", "rerun", "status", "release", "submit", "relocate"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--paper-dir", help="init: the paper directory as PaperBench hands it over (paper.md required)")
    parser.add_argument("--model", default=cfg.DEFAULT_MODEL)
    parser.add_argument("--provider-base-url", default=cfg.DEFAULT_PROVIDER_BASE_URL)
    parser.add_argument("--provider-key-env", default=cfg.DEFAULT_PROVIDER_KEY_ENV)
    parser.add_argument("--provider-stream", action="store_true", help="use SSE streaming for model calls")
    parser.add_argument("--thinking", choices=cfg.THINKING_MODES, default="disabled", help="init: thinking:{type:…} on every call of every slot; enabled = the 09-19 deepseek-flash caliber (DeepSeek official, api.deepseek.com)")
    parser.add_argument("--compute", choices=cfg.COMPUTE_KINDS, default="aliyun", help="init: aliyun rents one ECS instance per run; local is for tests only")
    parser.add_argument("--compute-tier", default="enough", help=f"init: the tier preferred at the compute review point — {cfg.COMPUTE_TIERS} (CPU), {cfg.GPU_COMPUTE_TIERS} (GPU, needs ALIYUN_GPU_IMAGE_ID) or an ecs.* type; when the code needs a GPU the default is the smallest GPU tier anyway")
    parser.add_argument("--run-hours", type=float, default=2.0, help="init: hard cap on the rented machine (CPU ≈ 1–3 CNY/h; GPU T4 ≈ 8 CNY/h, A10 ≈ 17 CNY/h in cn-hongkong, live prices at the review point)")
    parser.add_argument("--figures", choices=cfg.FIGURE_MODES, default="auto", help="init: describe the paper's figures from the images and insert them after their references (auto = only when the vision model passes the probe)")
    parser.add_argument("--figures-model", default=cfg.DEFAULT_FIGURES_MODEL, help=f"init: the vision model for the figure pass (default {cfg.DEFAULT_FIGURES_MODEL}; the phase model stays pinned)")
    parser.add_argument("--experiment-model", default=cfg.DEFAULT_EXPERIMENT_MODEL, help=f"init: the model RSA / SetupX call through the loopback in step 10 (default {cfg.DEFAULT_EXPERIMENT_MODEL})")
    parser.add_argument("--context-window", type=int, default=cfg.DEFAULT_CONTEXT_WINDOW, help=f"init: the phase model's context window in tokens; the planner's segment budget derives from it (default {cfg.DEFAULT_CONTEXT_WINDOW})")
    parser.add_argument("--planning-fanout", action="store_true", help="init: run the legacy Concept + Algorithm analysis agents before the planner (PLAN-3 item 7; off = upstream)")
    parser.add_argument("--repair-rounds", type=int, default=3, help="init: repair rounds after the experiment agent's round 0 (0 for a comparison run)")
    parser.add_argument("--env-file", action="append", default=[], help="KEY=VALUE file loaded into the environment (repeatable)")
    parser.add_argument("--skip", action="append", default=[], choices=("index",), help="init: phases to skip")
    parser.add_argument("--ask", action="store_true", help="stop at plan_review and read the decision from a file")
    parser.add_argument("--until", default="environment_run", choices=PHASES, help="run: last phase to execute")
    parser.add_argument("--phase", choices=PHASES, help="step / rerun: the phase")
    parser.add_argument("--no-probe", action="store_true", help="skip the preflight provider probe (offline tests)")
    parser.add_argument("--run-id", help="init: fixed run id (default: generated)")
    parser.add_argument("--paper", help="submit: PaperBench paper id (the pool directory name)")
    parser.add_argument("--trial", help="submit: submission name under the paper")
    parser.add_argument("--dest-root", default="~/pb_submissions", help="submit: the judge's submission pool")
    parser.add_argument("--force", action="store_true", help="submit: replace an existing <paper>/<trial>")
    parser.add_argument("--snapshot", help="submit: a state from the code history instead of the working tree — pre_repair (round 0, the comparison caliber) or a commit")
    parser.add_argument("--keep-tree", action="store_true", help="rerun --phase environment_run: keep generate_code as it is (default: reset it to the first commit of code.git, the stage-9 tree)")
    return parser


async def _run_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.command == "init":
        if not args.paper_dir:
            raise DriverError("init needs --paper-dir")
        driver = Driver.init(
            run_dir, paper_dir=args.paper_dir, model=args.model, provider_base_url=args.provider_base_url,
            provider_key_env=args.provider_key_env, provider_stream=args.provider_stream, thinking=args.thinking, compute=args.compute,
            compute_tier=args.compute_tier, run_hours=args.run_hours, repair_rounds=args.repair_rounds, figures=args.figures, figures_model=args.figures_model,
            experiment_model=args.experiment_model, context_window=args.context_window, planning_fanout=args.planning_fanout, skip=tuple(args.skip), ask=args.ask,
            run_id=args.run_id, engine_commit=_engine_commit(),
        )
        print(json.dumps(driver.describe(), indent=2, ensure_ascii=False))
        return 0

    driver = Driver(run_dir)
    if args.command == "status":
        print(json.dumps(driver.describe(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "release":
        record = await driver.release()
        print(json.dumps(record, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.command == "relocate":
        print(json.dumps(driver.relocate(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "submit":
        if not args.paper or not args.trial:
            raise DriverError("submit needs --paper and --trial")
        record = driver.submit(paper=args.paper, trial=args.trial, dest_root=args.dest_root, force=args.force, snapshot=args.snapshot)
        print(json.dumps({k: v for k, v in record.items() if k != "sha256"}, indent=2, ensure_ascii=False))
        return 0

    ask = args.ask or driver.run.ask
    if args.command == "rerun":
        if not args.phase:
            raise DriverError("rerun needs --phase")
        moved = driver.rerun(args.phase, keep_tree=args.keep_tree)
        for path in moved:
            logger.info("moved aside: {}", path)
    try:
        await driver.open(ask=ask, probe=not args.no_probe)
        if args.command == "run":
            outcome = await driver.run_until(args.until)
        else:
            if not args.phase:
                raise DriverError(f"{args.command} needs --phase")
            outcome = await driver.step(args.phase)
    finally:
        await driver.close()
    print(json.dumps({"outcome": outcome, **driver.describe()}, indent=2, ensure_ascii=False, default=str))
    return 0 if outcome == "completed" else (3 if outcome == "waiting" else 2)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = load_env_files(args.env_file)
    _configure_logging(Path(args.run_dir).expanduser().resolve())
    if loaded:
        logger.info("loaded {} variables from --env-file", len(loaded))
    try:
        return asyncio.run(_run_command(args))
    except DriverError as exc:
        logger.error("{}", exc)
        return 2
    except GateFailed as exc:
        logger.error("{}", exc)
        print(json.dumps({"outcome": "failed", "gate": exc.result.to_dict()}, indent=2, ensure_ascii=False, default=str))
        return 2


if __name__ == "__main__":
    sys.exit(main())
