"""SAPG training runs.

This script runs the core SAPG (Split and Aggregate Policy Gradients) algorithm
on the manipulation tasks described in the paper:

    * AllegroKuka: Regrasping / Throw / Reorientation
    * ShadowHand:  In-hand cube reorientation (24-DoF)
    * AllegroHand: In-hand cube reorientation (16-DoF)

It is a thin wrapper around :func:`main.train` that:
    1. Loads the appropriate task config (with ``defaults:`` inheritance).
    2. Applies SAPG-specific overrides (aggregation mode, number of blocks, ...).
    3. Runs training and writes the resulting history to JSON.

Example
-------
    python -m experiments.run_sapg --config configs/allegro_kuka.yaml \
        --task regrasping --num_envs 4096 --num_workers 8

The metric reported per task follows the paper:
    * hard tasks (Regrasping / Throw / Reorientation): successes per episode
    * easy tasks (ShadowHand / AllegroHand reorientation): net episode reward
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Make the repository root importable when run as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main import load_config, train  # noqa: E402


# ---------------------------------------------------------------------------
# Task registry: maps a task name to its default config file.
# ---------------------------------------------------------------------------
TASK_CONFIGS: Dict[str, str] = {
    "allegro_kuka": "configs/allegro_kuka.yaml",
    "shadow_hand": "configs/shadow_hand.yaml",
    "allegro_hand": "configs/allegro_hand.yaml",
}

# Tasks whose primary metric is "successes per episode" (hard tasks).
SUCCESS_METRIC_TASKS = {"regrasping", "throw", "reorientation"}


def build_sapg_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI arguments into a config override dict for SAPG."""
    overrides: Dict[str, Any] = {}

    if args.env_name is not None:
        overrides["env_name"] = args.env_name
    if args.task is not None:
        overrides["task"] = args.task
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
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
    if args.aggregation_mode is not None:
        overrides["aggregation_mode"] = args.aggregation_mode
    if args.leader_id is not None:
        overrides["leader_id"] = args.leader_id
    if args.rotate_leader:
        overrides["rotate_leader"] = True
    if args.use_entropy_exploration:
        overrides["use_entropy_exploration"] = True
    if args.clip_epsilon is not None:
        overrides["clip_epsilon"] = args.clip_epsilon
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    # SAPG always uses the leader-based aggregation by default; the ablation
    # script overrides this to "symmetric" / "none".
    overrides.setdefault("aggregation_mode", "leader")
    return overrides


def resolve_config_path(args: argparse.Namespace) -> str:
    """Resolve which config file to load."""
    if args.config is not None:
        return args.config
    env_name = args.env_name or "allegro_kuka"
    rel = TASK_CONFIGS.get(env_name, TASK_CONFIGS["allegro_kuka"])
    return os.path.join(_ROOT, rel)


def summarize_history(history: Dict[str, List[float]], task: str) -> Dict[str, Any]:
    """Compute summary statistics from a training history."""
    summary: Dict[str, Any] = {"task": task, "iterations": len(history.get("iteration", []))}

    rewards = history.get("mean_reward", [])
    successes = history.get("mean_successes", [])

    if rewards:
        summary["final_mean_reward"] = float(rewards[-1])
        summary["best_mean_reward"] = float(max(rewards))
    if successes:
        summary["final_mean_successes"] = float(successes[-1])
        summary["best_mean_successes"] = float(max(successes))

    # Primary metric per paper: successes for hard tasks, reward otherwise.
    if task in SUCCESS_METRIC_TASKS and successes:
        summary["primary_metric"] = "successes_per_episode"
        summary["primary_value"] = summary.get("best_mean_successes", 0.0)
    else:
        summary["primary_metric"] = "net_episode_reward"
        summary["primary_value"] = summary.get("best_mean_reward", 0.0)

    return summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a single SAPG training job."""
    config_path = resolve_config_path(args)
    overrides = build_sapg_overrides(args)

    cfg = load_config(config_path, overrides=overrides)
    task = cfg.get("task", args.task or "regrasping")

    print("=" * 70)
    print("SAPG training run")
    print("=" * 70)
    print(f"  config        : {config_path}")
    print(f"  env_name      : {cfg.get('env_name')}")
    print(f"  task          : {task}")
    print(f"  num_envs      : {cfg.get('num_envs')}")
    print(f"  num_workers   : {cfg.get('num_workers')}")
    print(f"  horizon       : {cfg.get('horizon')}")
    print(f"  aggregation   : {cfg.get('aggregation_mode')}")
    print(f"  max_iterations: {cfg.get('max_iterations')}")
    print(f"  device        : {cfg.get('device')}")
    print("=" * 70)

    history = train(cfg, algo="sapg")

    summary = summarize_history(history, task)
    print("-" * 70)
    print("SAPG run summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("-" * 70)

    # Persist the summary alongside the run history.
    output_dir = cfg.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "sapg_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}")

    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAPG training.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (overrides --env_name).")
    parser.add_argument("--env_name", type=str, default=None,
                        choices=list(TASK_CONFIGS.keys()),
                        help="Environment name; selects the default config.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (regrasping|throw|reorientation).")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Total number of parallel environments.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of blocks / follower policies.")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Rollout horizon per update.")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Number of training iterations.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--aggregation_mode", type=str, default=None,
                        choices=["leader", "symmetric", "none"],
                        help="Aggregation strategy (default: leader).")
    parser.add_argument("--leader_id", type=int, default=None,
                        help="Index of the designated leader worker.")
    parser.add_argument("--rotate_leader", action="store_true",
                        help="Rotate the leader across updates.")
    parser.add_argument("--use_entropy_exploration", action="store_true",
                        help="Enable per-block learnable sigma (entropy exploration).")
    parser.add_argument("--clip_epsilon", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
