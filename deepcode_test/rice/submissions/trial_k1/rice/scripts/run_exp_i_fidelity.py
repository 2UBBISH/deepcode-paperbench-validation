"""Experiment I driver: explanation fidelity comparison (Figure 5 / Table 2).

Trains or loads a RICE mask network for a fixed sample budget, then compares
fidelity scores of RICE, StateMask, random masking, Integrated Gradients, and
AIRS by masking the top-k critical steps and measuring the resulting return.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rice.agents.mask_network import load_mask_network
from rice.agents.target_agent import TargetAgent
from rice.evaluation.fidelity import (
    FIDELITY_BUDGETS,
    compare_fidelity,
    fidelity_from_domain,
    log_fidelity_table,
)
from rice.utils.config import get_domain_config, table_4
from rice.utils.logger import make_logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RICE Experiment I: explanation fidelity comparison",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="hopper",
        help="Domain name (e.g., hopper, walker2d, reacher, halfcheetah, "
             "selfish_mining, cage, metadrive, malware).",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        required=True,
        help="Path to a saved pre-trained target agent checkpoint.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        default=None,
        help="Path to a saved RICE mask network. If omitted, the script "
             "expects --train-mask to be set.",
    )
    parser.add_argument(
        "--train-mask",
        action="store_true",
        help="Train a new mask network instead of loading one.",
    )
    parser.add_argument(
        "--sample-budget",
        type=int,
        default=None,
        help="Fixed environment-step budget for mask training. Defaults to "
             "the value in paper Table 4 for the domain.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Masking budgets k to evaluate (default: from FIDELITY_BUDGETS).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=("RICE", "StateMask", "Random", "IntegratedGradients", "AIRS"),
        help="Explanation methods to compare.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per budget/method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for evaluation rollouts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for mask/policy inference.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/exp_i_fidelity",
        help="Directory where logs and results are written.",
    )
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV logging.",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default=None,
        help='JSON string of extra kwargs passed to the environment factory.',
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse-reward MuJoCo variant (only for MuJoCo domains).",
    )
    return parser.parse_args(argv)


def _parse_env_kwargs(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    return json.loads(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    save_dir = Path(args.save_dir) / args.domain / f"seed_{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = make_logger(
        log_dir=str(save_dir),
        experiment_name=f"exp_i_{args.domain}",
        use_tensorboard=args.use_tensorboard,
        use_csv=not args.no_csv,
        verbose=1,
    )
    logger.log_hyperparams(vars(args))

    # Resolve sample budget for mask training if requested.
    sample_budget = args.sample_budget
    if sample_budget is None and args.train_mask:
        table4 = table_4()
        sample_budget = table4.get(args.domain.lower())
        if sample_budget is None:
            raise ValueError(
                f"No default sample budget for domain {args.domain!r}. "
                "Provide --sample-budget or add the domain to table_4()."
            )

    env_kwargs = _parse_env_kwargs(args.env_kwargs)
    if args.sparse and args.domain.lower() in {
        "hopper", "walker2d", "halfcheetah", "reacher"
    }:
        env_kwargs["sparse"] = True

    # Load target agent.
    target_agent = TargetAgent.load(args.target_path)
    logger.log_text("target_path", args.target_path)

    # Load or train mask network.
    mask_net = None
    if "RICE" in args.methods or args.train_mask:
        if args.train_mask:
            from rice.training.train_mask import train_mask

            mask_result = train_mask(
                domain=args.domain,
                target_agent=target_agent,
                save_dir=str(save_dir / "mask"),
                seed=args.seed,
                device=args.device,
                total_timesteps=sample_budget,
                **env_kwargs,
            )
            mask_net = mask_result["mask_net"]
            logger.log_text("mask_train_status", "trained")
            logger.log_text("mask_save_dir", str(save_dir / "mask"))
        else:
            if args.mask_path is None:
                raise ValueError(
                    "--mask-path is required when not using --train-mask."
                )
            # Infer observation/action spaces from the target agent env if
            # available; otherwise rely on the saved mask checkpoint.
            mask_net = load_mask_network(args.mask_path)
            logger.log_text("mask_path", args.mask_path)
            logger.log_text("mask_train_status", "loaded")

    budgets = args.budgets if args.budgets is not None else list(FIDELITY_BUDGETS)

    # Run fidelity comparison.
    results = fidelity_from_domain(
        agent=target_agent,
        domain=args.domain,
        mask_net=mask_net,
        methods=tuple(args.methods),
        budgets=tuple(budgets),
        n_episodes=args.n_episodes,
        seed=args.seed,
        config=None,
        logger=logger,
        **env_kwargs,
    )

    table = log_fidelity_table(results, logger=logger)
    print(table)

    # Persist raw results.
    results_path = save_dir / "fidelity_results.json"
    serializable: Dict[str, Any] = {}
    for method, budget_dict in results.items():
        serializable[method] = {}
        for budget, metrics in budget_dict.items():
            serializable[method][str(budget)] = {
                k: float(v) if hasattr(v, "__float__") else v
                for k, v in metrics.items()
            }
    with results_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    logger.log_text("results_path", str(results_path))

    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
