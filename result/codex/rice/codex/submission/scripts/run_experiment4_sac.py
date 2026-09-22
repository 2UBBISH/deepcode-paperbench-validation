#!/usr/bin/env python3
"""Experiment IV: refining a SAC agent after GAIL imitation (Figure 3)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.experiments.sac_refine import (  # noqa: E402
    SACExperimentConfig,
    run_experiment_iv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Hopper")
    parser.add_argument("--sac-steps", type=int, default=200_000)
    parser.add_argument("--gail-steps", type=int, default=200_000)
    parser.add_argument("--refine-steps", type=int, default=200_000)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    payload = run_experiment_iv(
        SACExperimentConfig(
            env_name=args.env,
            sac_steps=args.sac_steps,
            gail_steps=args.gail_steps,
            refine_steps=args.refine_steps,
            seeds=args.seeds,
        ),
        mask_path=args.mask,
    )
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()
