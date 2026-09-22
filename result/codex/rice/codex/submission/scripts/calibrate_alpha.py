#!/usr/bin/env python3
"""Pick the mask-network bonus ``alpha`` of one application.

    python scripts/calibrate_alpha.py --env SelfishMining \
        --agent checkpoints/agents/SelfishMining_seed0.pt

The candidate whose mask rate is closest to 0.5 is written to
``results/<env>_alpha_calibration.json`` and should be used as the application's
``EnvSpec.alpha`` (see ``docs/EXPERIMENTS.md`` for why the two corner regimes
make the explanation uninformative).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.experiments.calibration import DEFAULT_CANDIDATES, calibrate_alpha  # noqa: E402
from rice.experiments.common import ckpt_path, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    agent = args.agent or ckpt_path(
        "{}_seed{}.pt".format(args.env, args.seed), kind="agents"
    )
    result = calibrate_alpha(
        args.env,
        agent,
        candidates=DEFAULT_CANDIDATES,
        samples=args.samples,
        seed=args.seed,
        device=args.device,
    )
    save_json(
        result,
        ckpt_path("{}_alpha_calibration.json".format(args.env), kind="results"),
    )
    print("best alpha for {}: {}".format(args.env, result["best_alpha"]))


if __name__ == "__main__":
    main()
