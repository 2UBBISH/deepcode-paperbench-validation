"""Experiment II: Efficiency of mask training (Table 4).

Trains the RICE mask network and a StateMask-style baseline on identical fixed
sample counts and records wall-clock seconds.  Expected outcome: RICE is faster
on average across all domains.

Usage:
    python -m scripts.run_exp_ii_efficiency --domain hopper --target_path results/targets/mujoco/hopper/dense/seed_0/model.zip
    python -m scripts.run_exp_ii_efficiency --domain halfcheetah --sample_budgets 10000 50000 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on path when running as script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rice.evaluation.efficiency import (
    compare_efficiency,
    efficiency_from_domain,
    log_efficiency_table,
)
from rice.utils.config import DEFAULT_SAMPLE_BUDGETS, get_domain_config, table_4
from rice.utils.logger import make_logger


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RICE Experiment II: mask-training efficiency benchmark",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain name (e.g. hopper, walker2d, reacher, halfcheetah).",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        help="Path to a saved target-agent checkpoint. If omitted, a default path is inferred.",
    )
    parser.add_argument(
        "--sample_budgets",
        type=int,
        nargs="+",
        default=None,
        help="Fixed sample counts to benchmark. Defaults to Table 4 budgets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device for mask training.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/exp_ii_efficiency",
        help="Directory where logs and result JSON are written.",
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0 = silent, 1 = info, 2 = debug).",
    )
    parser.add_argument(
        "--env_kwargs",
        type=str,
        default=None,
        help="JSON string of extra environment kwargs (e.g. '{\"sparse\": true}').",
    )
    return parser.parse_args(argv)


def _parse_env_kwargs(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    return json.loads(raw)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    experiment_name = f"exp_ii_{args.domain}_seed_{args.seed}"
    save_dir = Path(args.save_dir) / args.domain / f"seed_{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = make_logger(
        log_dir=str(save_dir),
        experiment_name=experiment_name,
        use_tensorboard=args.use_tensorboard,
        use_csv=True,
        verbose=args.verbose,
    )

    logger.log_hyperparams({
        "experiment": "II_efficiency",
        "domain": args.domain,
        "target_path": args.target_path,
        "sample_budgets": args.sample_budgets,
        "seed": args.seed,
        "device": args.device,
    })

    env_kwargs = _parse_env_kwargs(args.env_kwargs)

    # Resolve default sample budgets from Table 4 if not provided.
    sample_budgets = args.sample_budgets
    if sample_budgets is None:
        sample_budgets = list(table_4().get(args.domain, DEFAULT_SAMPLE_BUDGETS).values())
        if not sample_budgets:
            sample_budgets = list(DEFAULT_SAMPLE_BUDGETS.values())

    logger.log("Starting efficiency benchmark", level=1)

    results = efficiency_from_domain(
        domain=args.domain,
        target_path=args.target_path,
        sample_budgets=sample_budgets,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        logger=logger,
        config=get_domain_config(args.domain, **env_kwargs),
        **env_kwargs,
    )

    table = log_efficiency_table(results, logger=logger)
    if args.verbose:
        print(table)

    # Persist raw results as JSON for downstream table generation.
    results_path = save_dir / "efficiency_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    logger.log("Saved results to", str(results_path), level=1)
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
