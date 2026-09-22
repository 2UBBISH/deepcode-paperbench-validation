#!/usr/bin/env python3
"""Enumerate (and optionally launch) the full experiment matrix of the paper.

The paper reports 5 manipulation tasks x {SAPG, PPO, DexPBT, PQL} x 5 seeds plus
the Sec. 6.3 ablations, each run collecting ~2e10 transitions (48-60 h on a
single GPU).  Such runs are executed *outside* this environment; this script
writes the exact commands so that they can be launched on a cluster.

Example::

    python scripts/run_experiments.py --dry-run --out experiments/commands.sh
    python scripts/run_experiments.py --execute --workers 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = {
    "regrasping": {
        "sapg": "sapg/configs/sapg_allegrokuka_regrasping.yaml",
        "ppo": "sapg/configs/ppo_allegrokuka.yaml",
        "pbt": "sapg/configs/pbt_allegrokuka.yaml",
        "pql": "sapg/configs/pql_allegrokuka.yaml",
    },
    "throw": {
        "sapg": "sapg/configs/sapg_allegrokuka_throw.yaml",
        "ppo": "sapg/configs/ppo_allegrokuka.yaml",
        "pbt": "sapg/configs/pbt_allegrokuka.yaml",
        "pql": "sapg/configs/pql_allegrokuka.yaml",
    },
    "reorientation": {
        "sapg": "sapg/configs/sapg_allegrokuka_reorientation.yaml",
        "ppo": "sapg/configs/ppo_allegrokuka.yaml",
        "pbt": "sapg/configs/pbt_allegrokuka.yaml",
        "pql": "sapg/configs/pql_allegrokuka.yaml",
    },
    "shadow_hand": {
        "sapg": "sapg/configs/sapg_shadow_hand.yaml",
        "ppo": "sapg/configs/ppo_inhand.yaml",
        "pbt": "sapg/configs/pbt_inhand.yaml",
        "pql": "sapg/configs/pql_inhand.yaml",
    },
    "allegro_hand": {
        "sapg": "sapg/configs/sapg_allegro_hand.yaml",
        "ppo": "sapg/configs/ppo_inhand.yaml",
        "pbt": "sapg/configs/pbt_inhand.yaml",
        "pql": "sapg/configs/pql_inhand.yaml",
    },
}

ABLATIONS = {
    "sapg_no_offpolicy": "sapg/configs/ablations/sapg_no_offpolicy.yaml",
    "sapg_symmetric": "sapg/configs/ablations/sapg_symmetric.yaml",
    "sapg_high_offpolicy": "sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml",
    "sapg_entropy0.003": "sapg/configs/ablations/sapg_entropy_0003.yaml",
    "sapg_entropy0.005": "sapg/configs/ablations/sapg_entropy_0005.yaml",
}

# The two easy tasks use a different architecture and schedule (Tables 3-4), so
# the ablations come in a matching flavour for them.
ABLATIONS_INHAND = {
    "sapg_no_offpolicy": "sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml",
    "sapg_symmetric": "sapg/configs/ablations_inhand/sapg_symmetric.yaml",
    "sapg_high_offpolicy": "sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml",
    "sapg_entropy0.003": "sapg/configs/ablations_inhand/sapg_entropy_0003.yaml",
    "sapg_entropy0.005": "sapg/configs/ablations_inhand/sapg_entropy_0005.yaml",
}
IN_HAND_TASKS = ("shadow_hand", "allegro_hand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper's experiment matrix")
    parser.add_argument("--tasks", nargs="*", default=list(CONFIGS))
    parser.add_argument("--methods", nargs="*", default=["sapg", "ppo", "pbt", "pql"])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--with-ablations", action="store_true")
    parser.add_argument("--logroot", default="runs")
    parser.add_argument("--out", default="experiments/commands.sh")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def build_commands(args) -> list:
    commands = []
    for task in args.tasks:
        for method in args.methods:
            if method not in CONFIGS[task]:
                continue
            for seed in args.seeds:
                logdir = os.path.join(args.logroot, f"{method}_{task}_seed{seed}")
                commands.append(
                    "python scripts/train.py --config {cfg} --logdir {logdir} --seed {seed}".format(
                        cfg=CONFIGS[task][method], logdir=logdir, seed=seed
                    )
                )
        if args.with_ablations:
            table = ABLATIONS_INHAND if task in IN_HAND_TASKS else ABLATIONS
            for label, cfg in table.items():
                for seed in args.seeds:
                    logdir = os.path.join(args.logroot, f"{label}_{task}_seed{seed}")
                    commands.append(
                        "python scripts/train.py --config {cfg} --logdir {logdir} "
                        "--seed {seed} --set env.name={task}".format(
                            cfg=cfg, logdir=logdir, seed=seed, task=task
                        )
                    )
    return commands


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        handle.write("\n".join(commands) + "\n")
    os.chmod(args.out, 0o755)
    print(f"[run_experiments] {len(commands)} runs written to {args.out}")

    if args.execute:
        from concurrent.futures import ThreadPoolExecutor

        def run(command: str) -> int:
            return subprocess.call(command, shell=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run, commands))


if __name__ == "__main__":
    main()
