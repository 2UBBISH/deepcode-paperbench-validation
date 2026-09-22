#!/usr/bin/env python3
"""Run a single (application, refining method, seed) job.

This is the unit of work of the paper's grid: it is handy to launch one job per
GPU (or per CPU worker) on a cluster and to merge the resulting JSON files with
``scripts/make_tables.py``.

    python scripts/run_one.py --env Hopper --method rice      --seed 1 --steps 300000
    python scripts/run_one.py --env Hopper --method ppo       --seed 1 --steps 300000
    python scripts/run_one.py --env Hopper --method jsrl      --seed 1 --steps 300000
    python scripts/run_one.py --env Hopper --method statemask_r --seed 1 --steps 300000
    python scripts/run_one.py --env Hopper --method rice --explanation random --seed 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import ENV_SPECS  # noqa: E402
from rice.experiments.common import (  # noqa: E402
    ckpt_path,
    ensure_dir,
    evaluate_agent,
    load_agent,
    save_json,
)
from rice.experiments.refining import (  # noqa: E402
    default_refine_config,
    run_rice_with_explanation,
    run_single_refine,
)
from rice.explanation.mask_io import load_mask_net  # noqa: E402


METHODS = ("rice", "ppo", "jsrl", "statemask_r")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--explanation", default="ours",
                        choices=["ours", "statemask", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--rnd-lambda", type=float, default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    spec = ENV_SPECS[args.env]
    agent_path = args.agent or ckpt_path(
        "{}_seed{}.pt".format(args.env, args.seed), kind="agents"
    )
    policy = load_agent(args.env, agent_path, device=args.device)
    mask_path = args.mask or ckpt_path(
        "{}_ours_seed{}.pt".format(args.env, args.seed), kind="masks"
    )
    mask_net = (
        load_mask_net(mask_path, hidden=spec.mask_hidden)
        if os.path.exists(mask_path)
        else None
    )
    statemask_path = ckpt_path(
        "{}_statemask_seed{}.pt".format(args.env, args.seed), kind="masks"
    )
    statemask_net = (
        load_mask_net(statemask_path, hidden=spec.mask_hidden)
        if os.path.exists(statemask_path)
        else mask_net
    )

    config = default_refine_config(
        args.env, seed=args.seed, total_steps=args.steps, p=args.p, rnd_lambda=args.rnd_lambda
    )
    if args.method == "rice" and args.explanation != "ours":
        result = run_rice_with_explanation(
            args.env,
            policy,
            statemask_net if args.explanation == "statemask" else mask_net,
            explanation=args.explanation,
            config=config,
            device=args.device,
        )
    else:
        result = run_single_refine(
            args.env,
            policy,
            mask_net if args.explanation == "ours" else statemask_net,
            method="ours" if args.method == "rice" else args.method,
            explanation=args.explanation,
            config=config,
            device=args.device,
        )

    no_refine = evaluate_agent(args.env, policy, n_episodes=10, seed=args.seed + 1000)
    payload = {
        "env": args.env,
        "method": args.method,
        "explanation": args.explanation,
        "seed": args.seed,
        "steps": result.total_steps,
        "wall_time": result.wall_time,
        "final_return": result.final_eval,
        "no_refine": no_refine,
        "eval_history": result.eval_history,
        "config": {
            "p": config.p,
            "rnd_lambda": config.rnd_lambda,
            "total_steps": config.total_steps,
        },
    }
    out_dir = ensure_dir(args.out_dir or ckpt_path("run_one", kind="results"))
    path = os.path.join(
        out_dir,
        "{}_{}_{}_seed{}.json".format(
            args.env, args.method, args.explanation, args.seed
        ),
    )
    save_json(payload, path)
    print(
        "{} / {} / {} seed {}: {:.2f} (no refine: {:.2f}) in {:.1f}s -> {}".format(
            args.env,
            args.method,
            args.explanation,
            args.seed,
            result.final_eval,
            no_refine["mean_return"],
            result.wall_time,
            path,
        )
    )


if __name__ == "__main__":
    main()
