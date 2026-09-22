#!/usr/bin/env python3
"""Experiment III (standalone): RICE refining with different explanations."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from rice.envs.registry import ENV_SPECS  # noqa: E402
from rice.experiments.common import ckpt_path, load_agent, save_json  # noqa: E402
from rice.experiments.refining import (  # noqa: E402
    default_refine_config,
    run_rice_with_explanation,
)
from rice.explanation.mask_io import load_mask_net  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--mask-ours", default=None)
    parser.add_argument("--mask-statemask", default=None)
    parser.add_argument(
        "--explanations", nargs="+", default=["random", "statemask", "ours"]
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    spec = ENV_SPECS[args.env]
    agent_path = args.agent or ckpt_path(
        "{}_seed0.pt".format(args.env), kind="agents"
    )
    policy = load_agent(args.env, agent_path, device=args.device)
    masks = {
        "ours": load_mask_net(
            args.mask_ours
            or ckpt_path("{}_ours_seed0.pt".format(args.env), kind="masks"),
            hidden=spec.mask_hidden,
        ),
        "statemask": load_mask_net(
            args.mask_statemask
            or ckpt_path("{}_statemask_seed0.pt".format(args.env), kind="masks"),
            hidden=spec.mask_hidden,
        ),
    }
    results = {}
    for explanation in args.explanations:
        values = []
        for seed in args.seeds:
            config = default_refine_config(
                args.env, seed=seed, total_steps=args.steps
            )
            out = run_rice_with_explanation(
                args.env,
                policy,
                masks.get(explanation, masks["ours"]),
                explanation=explanation,
                config=config,
                device=args.device,
            )
            values.append(float(out.final_eval))
        results[explanation] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        print(
            "{:>10}: {:.2f} ({:.2f})".format(
                explanation, results[explanation]["mean"], results[explanation]["std"]
            )
        )
    save_json(
        {"env": args.env, "results": results},
        ckpt_path("{}_explanations.json".format(args.env), kind="results"),
    )


if __name__ == "__main__":
    main()
