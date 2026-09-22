"""PPO batch-size saturation sweep (Figure 2).

Trains vanilla PPO across increasing batch sizes (num_envs) on a given task and
records the asymptotic performance. The goal is to reproduce the saturation trend
shown in Figure 2 of the SAPG paper: PPO performance saturates after a certain
batch size, while SAPG (dashed red line) achieves higher asymptotic performance.

Usage:
    python -m experiments.run_ppo_baseline --env_name allegro_kuka --task regrasping
    python -m experiments.run_ppo_baseline --env_name shadow_hand --batch_sizes 512,2048,4096,8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Allow running as a standalone script or module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from main import load_config, train  # noqa: E402

TASK_CONFIGS: Dict[str, str] = {
    "allegro_kuka": os.path.join(_REPO_ROOT, "configs", "allegro_kuka.yaml"),
    "shadow_hand": os.path.join(_REPO_ROOT, "configs", "shadow_hand.yaml"),
    "allegro_hand": os.path.join(_REPO_ROOT, "configs", "allegro_hand.yaml"),
}

# Default batch sizes (num_envs) to sweep over for Figure 2.
DEFAULT_BATCH_SIZES: List[int] = [512, 1024, 2048, 4096, 8192]


def resolve_config_path(args: argparse.Namespace) -> str:
    """Return explicit --config path or the task-registry default."""
    if args.config:
        return args.config
    if args.env_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown env_name '{args.env_name}'. Choose from {list(TASK_CONFIGS.keys())} "
            "or pass --config explicitly."
        )
    return TASK_CONFIGS[args.env_name]


def build_ppo_overrides(args: argparse.Namespace, batch_size: int) -> Dict[str, Any]:
    """Translate CLI args + current batch size into a config override dict.

    PPO uses a single shared policy across all environments, so num_workers is
    forced to 1 (single-block PPO reduces to standard PPO).
    """
    overrides: Dict[str, Any] = {
        "num_envs": batch_size,
        "num_workers": 1,
        "aggregation_mode": "none",
    }
    if args.task is not None:
        overrides["task"] = args.task
    if args.horizon is not None:
        overrides["horizon"] = args.horizon
    if args.max_iterations is not None:
        overrides["max_iterations"] = args.max_iterations
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    return overrides


def summarize_history(history: Dict[str, List[float]]) -> Dict[str, Any]:
    """Compute final/best reward and successes from a training history."""
    mean_reward = history.get("mean_reward", [])
    mean_successes = history.get("mean_successes", [])
    return {
        "final_mean_reward": float(mean_reward[-1]) if mean_reward else None,
        "best_mean_reward": float(max(mean_reward)) if mean_reward else None,
        "final_mean_successes": float(mean_successes[-1]) if mean_successes else None,
        "best_mean_successes": float(max(mean_successes)) if mean_successes else None,
        "num_iterations": len(mean_reward),
    }


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the PPO batch-size sweep and persist a JSON summary."""
    config_path = resolve_config_path(args)
    batch_sizes = args.batch_sizes or DEFAULT_BATCH_SIZES

    results: Dict[str, Any] = {
        "env_name": args.env_name,
        "task": args.task,
        "algo": "ppo",
        "batch_sizes": batch_sizes,
        "runs": {},
    }

    for batch_size in batch_sizes:
        print(f"\n{'='*70}\nPPO baseline | batch_size (num_envs) = {batch_size}\n{'='*70}")
        overrides = build_ppo_overrides(args, batch_size)
        cfg = load_config(config_path, overrides)
        history = train(cfg, algo="ppo")
        summary = summarize_history(history)
        results["runs"][str(batch_size)] = summary
        print(f"batch_size={batch_size}: {summary}")

    out_dir = args.output_dir or os.path.join(_REPO_ROOT, "runs", "ppo_baseline")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ppo_baseline_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved PPO baseline sweep summary to {out_path}")
    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO batch-size saturation sweep (Figure 2)."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (overrides task registry default).")
    parser.add_argument("--env_name", type=str, default="allegro_kuka",
                        choices=list(TASK_CONFIGS.keys()),
                        help="Environment name (used to pick default config).")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (e.g. regrasping/throw/reorientation).")
    parser.add_argument("--batch_sizes", type=str, default=None,
                        help="Comma-separated list of num_envs to sweep over.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.batch_sizes is not None:
        args.batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    run_sweep(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
