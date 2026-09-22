"""Ablation experiments for SAPG (Figure 6).

This script reproduces the ablation study from the SAPG paper (Figure 6), which
compares the leader-based aggregation of SAPG against alternative aggregation
strategies:

    * ``leader``     -- SAPG as proposed: one designated leader worker is updated
                        using the off-policy data collected by *all* followers
                        (importance-weighted / clipped surrogate).  This is the
                        blue curve in Figure 6.
    * ``symmetric``  -- No leader.  Every worker is updated with the off-policy
                        data from *all other* workers symmetrically.
    * ``none``       -- No aggregation at all: each worker is trained purely
                        on-policy (equivalent to running independent PPO policies
                        per block).

Additionally, the script supports a *block-count* ablation (varying
``num_workers`` while keeping ``num_envs`` fixed) which corresponds to the
"number of blocks" sweep discussed in the paper.

The expected qualitative result is that the leader-based SAPG aggregation
outperforms symmetric aggregation, which in turn outperforms no aggregation.

Usage
-----
    python -m experiments.run_ablations --env_name allegro_kuka --task regrasping
    python -m experiments.run_ablations --modes leader,symmetric,none \
        --num_workers_list 1,2,4,8 --max_iterations 500

The script writes ``ablation_summary.json`` into ``output_dir`` containing the
per-configuration training history and summary statistics, which can then be
consumed by ``eval/plot.py`` to reproduce Figure 6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Make the repository root importable so that ``main`` can be imported whether
# this file is executed as a module (``python -m experiments.run_ablations``)
# or as a standalone script (``python experiments/run_ablations.py``).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from main import load_config, train  # noqa: E402


# ---------------------------------------------------------------------------
# Task registry (mirrors run_sapg.py / run_ppo_baseline.py)
# ---------------------------------------------------------------------------
TASK_CONFIGS: Dict[str, str] = {
    "allegro_kuka": os.path.join(_REPO_ROOT, "configs", "allegro_kuka.yaml"),
    "shadow_hand": os.path.join(_REPO_ROOT, "configs", "shadow_hand.yaml"),
    "allegro_hand": os.path.join(_REPO_ROOT, "configs", "allegro_hand.yaml"),
}

# Aggregation modes reproduced in Figure 6.
DEFAULT_MODES: List[str] = ["leader", "symmetric", "none"]

# Block-count sweep (number of follower workers) for the block ablation.
DEFAULT_NUM_WORKERS: List[int] = [1, 2, 4, 8]

# Tasks whose primary metric is "successes per episode" rather than reward.
SUCCESS_METRIC_TASKS = {"regrasping", "throw", "reorientation"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_config_path(args: argparse.Namespace) -> str:
    """Return the explicit ``--config`` path or the task-registry default."""
    if getattr(args, "config", None):
        return args.config
    env_name = (args.env_name or "allegro_kuka").lower()
    if env_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown env_name '{env_name}'. "
            f"Available: {sorted(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[env_name]


def build_ablation_overrides(
    args: argparse.Namespace,
    mode: str,
    num_workers: int,
) -> Dict[str, Any]:
    """Build the config override dict for a single ablation configuration.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    mode:
        Aggregation mode: ``"leader"``, ``"symmetric"`` or ``"none"``.
    num_workers:
        Number of follower blocks (workers) for this configuration.
    """
    overrides: Dict[str, Any] = {
        "aggregation_mode": mode,
        "num_workers": int(num_workers),
    }

    # Optional CLI overrides (only applied when explicitly provided).
    if getattr(args, "task", None) is not None:
        overrides["task"] = args.task
    if getattr(args, "num_envs", None) is not None:
        overrides["num_envs"] = int(args.num_envs)
    if getattr(args, "horizon", None) is not None:
        overrides["horizon"] = int(args.horizon)
    if getattr(args, "max_iterations", None) is not None:
        overrides["max_iterations"] = int(args.max_iterations)
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = int(args.seed)
    if getattr(args, "device", None) is not None:
        overrides["device"] = args.device

    # Each ablation configuration gets its own output directory so that
    # checkpoints and histories do not clobber one another.
    base_out = getattr(args, "output_dir", None) or "runs/ablations"
    overrides["output_dir"] = os.path.join(
        base_out, f"{mode}_w{num_workers}"
    )
    return overrides


def summarize_history(history: Dict[str, List[float]], task: str) -> Dict[str, Any]:
    """Compute summary statistics from a training history dict.

    Returns final/best reward and successes, plus the task's primary metric.
    """
    def _last(key: str) -> Optional[float]:
        vals = history.get(key) or []
        return float(vals[-1]) if len(vals) else None

    def _best(key: str) -> Optional[float]:
        vals = history.get(key) or []
        return float(max(vals)) if len(vals) else None

    summary: Dict[str, Any] = {
        "final_mean_reward": _last("mean_reward"),
        "best_mean_reward": _best("mean_reward"),
        "final_mean_successes": _last("mean_successes"),
        "best_mean_successes": _best("mean_successes"),
        "num_iterations": len(history.get("iteration", []) or []),
    }

    if task in SUCCESS_METRIC_TASKS:
        summary["primary_metric"] = "mean_successes"
        summary["primary_value"] = summary["best_mean_successes"]
    else:
        summary["primary_metric"] = "mean_reward"
        summary["primary_value"] = summary["best_mean_reward"]

    return summary


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------
def run_mode_ablation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the aggregation-mode ablation (leader vs symmetric vs none)."""
    config_path = resolve_config_path(args)
    modes = [m.strip() for m in (args.modes or ",".join(DEFAULT_MODES)).split(",") if m.strip()]

    results: Dict[str, Any] = {}
    for mode in modes:
        # For the mode ablation we keep the number of blocks fixed (default 8,
        # or whatever the user requested via --num_workers).
        num_workers = int(args.num_workers or 8)
        print("=" * 78)
        print(f"[ablation] aggregation_mode={mode}  num_workers={num_workers}")
        print("=" * 78)

        overrides = build_ablation_overrides(args, mode, num_workers)
        cfg = load_config(config_path, overrides)
        history = train(cfg, algo="sapg")

        key = f"mode={mode}"
        results[key] = {
            "mode": mode,
            "num_workers": num_workers,
            "history": history,
            "summary": summarize_history(history, cfg.get("task", "regrasping")),
        }

    return results


def run_block_ablation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the block-count ablation (varying ``num_workers``).

    The aggregation mode is fixed to ``leader`` (SAPG) unless overridden.
    """
    config_path = resolve_config_path(args)
    mode = (args.modes or "leader").split(",")[0].strip() or "leader"
    workers_list = args.num_workers_list or DEFAULT_NUM_WORKERS

    results: Dict[str, Any] = {}
    for num_workers in workers_list:
        num_workers = int(num_workers)
        print("=" * 78)
        print(f"[ablation] blocks={num_workers}  aggregation_mode={mode}")
        print("=" * 78)

        overrides = build_ablation_overrides(args, mode, num_workers)
        cfg = load_config(config_path, overrides)
        history = train(cfg, algo="sapg")

        key = f"blocks={num_workers}"
        results[key] = {
            "mode": mode,
            "num_workers": num_workers,
            "history": history,
            "summary": summarize_history(history, cfg.get("task", "regrasping")),
        }

    return results


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the requested ablation(s) and persist a JSON summary."""
    config_path = resolve_config_path(args)
    base_cfg = load_config(config_path, None)
    task = getattr(args, "task", None) or base_cfg.get("task", "regrasping")
    env_name = getattr(args, "env_name", None) or base_cfg.get("env_name", "allegro_kuka")

    print("#" * 78)
    print(f"# SAPG ablation study (Figure 6)")
    print(f"#   env_name     : {env_name}")
    print(f"#   task         : {task}")
    print(f"#   config       : {config_path}")
    print(f"#   ablation     : {args.ablation}")
    print("#" * 78)

    if args.ablation == "blocks":
        results = run_block_ablation(args)
    else:
        results = run_mode_ablation(args)

    output_dir = getattr(args, "output_dir", None) or "runs/ablations"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "ablation_summary.json")

    payload = {
        "env_name": env_name,
        "task": task,
        "ablation": args.ablation,
        "config_path": config_path,
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[ablation] wrote summary to {summary_path}")
    for key, res in results.items():
        s = res["summary"]
        print(
            f"  {key:<20s} primary={s['primary_metric']}="
            f"{s['primary_value']}  final_reward={s['final_mean_reward']}"
        )

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAPG ablation experiments (Figure 6)."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config (overrides --env_name).")
    parser.add_argument("--env_name", type=str, default="allegro_kuka",
                        choices=sorted(TASK_CONFIGS.keys()),
                        help="Environment name.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (regrasping|throw|reorientation).")
    parser.add_argument("--ablation", type=str, default="modes",
                        choices=["modes", "blocks"],
                        help="Which ablation to run: aggregation modes or block count.")
    parser.add_argument("--modes", type=str, default=None,
                        help="Comma-separated aggregation modes "
                             "(leader,symmetric,none). Default: all three.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of blocks for the mode ablation.")
    parser.add_argument("--num_workers_list", type=str, default=None,
                        help="Comma-separated block counts for the block ablation.")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Total number of parallel environments.")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Rollout horizon per iteration.")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Number of training iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (e.g. cuda or cpu).")
    parser.add_argument("--output_dir", type=str, default="runs/ablations",
                        help="Directory to write results into.")
    args = parser.parse_args(argv)

    # Parse comma-separated integer list for the block ablation.
    if args.num_workers_list:
        args.num_workers_list = [
            int(x) for x in str(args.num_workers_list).split(",") if str(x).strip()
        ]
    else:
        args.num_workers_list = None

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
