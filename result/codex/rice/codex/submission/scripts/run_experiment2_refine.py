#!/usr/bin/env python3
"""Experiments II + III: Table 1 (dense) and Figure 2 (sparse MuJoCo)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.experiments.common import ckpt_path  # noqa: E402
from rice.experiments.refining import run_experiment_ii_and_iii  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--mask-ours", default=None)
    parser.add_argument("--mask-statemask", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    mask_paths = {}
    ours = args.mask_ours or ckpt_path("{}_ours_seed0.pt".format(args.env), kind="masks")
    statemask = args.mask_statemask or ckpt_path(
        "{}_statemask_seed0.pt".format(args.env), kind="masks"
    )
    mask_paths["ours"] = ours
    mask_paths["statemask"] = statemask
    result = run_experiment_ii_and_iii(
        args.env,
        args.agent or ckpt_path("{}_seed0.pt".format(args.env), kind="agents"),
        mask_paths,
        total_steps=args.steps,
        seeds=args.seeds,
        device=args.device,
        verbose=not args.quiet,
    )
    print("\n=== {} ===".format(result.env))
    print("no refine: {:.2f} ({:.2f})".format(result.no_refine["mean"], result.no_refine["std"]))
    print("fix explanation; vary refine method:")
    for method, values in result.vary_refine.items():
        print("  {:>12}: {:8.2f} ({:.2f})".format(method, values["mean"], values["std"]))
    print("fix refine; vary explanation:")
    for method, values in result.vary_explanation.items():
        print("  {:>12}: {:8.2f} ({:.2f})".format(method, values["mean"], values["std"]))


if __name__ == "__main__":
    main()
