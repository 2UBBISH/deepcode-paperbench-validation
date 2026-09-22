"""Batch-size scaling experiment (Figure 2 of the SAPG paper).

Trains PPO (baseline) and SAPG across increasing batch sizes on a chosen
environment/task and records the asymptotic performance.  The expected result
is that PPO performance saturates after a certain batch size while SAPG keeps
improving and reaches a higher asymptotic performance.

The script reuses :class:`main.SAPGTrainer` for both algorithms:

* ``PPO``  -> a single follower block (``num_blocks = 1``) with leader
  aggregation disabled (``sapg.mode = "symmetric"`` with a single worker is
  equivalent to plain PPO on the whole batch).
* ``SAPG`` -> ``num_blocks > 1`` followers whose off-policy transitions are
  aggregated into a leader policy.

Usage
-----
    python -m experiments.run_batchsize_sweep --env allegrokuka --task regrasping
    python -m experiments.run_batchsize_sweep --env shadow_hand --batch-sizes 1024 4096 16384
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Make the project root importable when executed directly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from main import SAPGTrainer, load_config  # noqa: E402
from utils.logger import MetricLogger  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZES: Sequence[int] = (1024, 4096, 16384, 65536)
DEFAULT_ALGORITHMS: Sequence[str] = ("ppo", "sapg")
DEFAULT_NUM_BLOCKS = 8
DEFAULT_ITERATIONS = 2000
DEFAULT_EVAL_EPISODES = 32

# Metric used to compare algorithms per environment family.
METRIC_BY_ENV = {
    "allegrokuka": "eval/successes",
    "shadow_hand": "eval/reward",
    "allegro_hand": "eval/reward",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (in place) and return it."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_sweep_config(
    base_cfg: Dict[str, Any],
    *,
    algorithm: str,
    batch_size: int,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    seed: Optional[int] = None,
    iterations: Optional[int] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a deep-copied config specialized for one (algorithm, batch size).

    ``batch_size`` is the total number of parallel environments.  For PPO the
    whole batch is a single block; for SAPG the batch is split into
    ``num_blocks`` equally sized follower blocks.
    """
    cfg = copy.deepcopy(base_cfg)
    algorithm = algorithm.lower()

    if algorithm not in ("ppo", "sapg"):
        raise ValueError(f"Unknown algorithm '{algorithm}' (expected 'ppo' or 'sapg')")

    # Total number of parallel environments == batch size.
    _deep_update(cfg, {"env": {"num_envs": int(batch_size)}})

    if algorithm == "ppo":
        # Plain PPO: one block, no leader aggregation.
        _deep_update(
            cfg,
            {
                "env": {"num_blocks": 1},
                "sapg": {"mode": "symmetric", "phi_dim": 0},
            },
        )
    else:
        # SAPG: split the batch into follower blocks + leader aggregation.
        blocks = max(1, min(int(num_blocks), int(batch_size)))
        _deep_update(
            cfg,
            {
                "env": {"num_blocks": blocks},
                "sapg": {"mode": "leader"},
            },
        )

    if seed is not None:
        _deep_update(cfg, {"run": {"seed": int(seed)}})
    if iterations is not None:
        _deep_update(cfg, {"run": {"total_iterations": int(iterations)}})
    if log_dir is not None:
        _deep_update(cfg, {"run": {"log_dir": log_dir}})

    cfg.setdefault("run", {})
    cfg["run"]["run_name"] = f"{algorithm}_bs{batch_size}"
    return cfg


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(
    base_cfg: Dict[str, Any],
    *,
    algorithm: str,
    batch_size: int,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    seed: int = 0,
    iterations: Optional[int] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    eval_every: int = 0,
    log_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train one (algorithm, batch size) configuration and return its history."""
    cfg = build_sweep_config(
        base_cfg,
        algorithm=algorithm,
        batch_size=batch_size,
        num_blocks=num_blocks,
        seed=seed,
        iterations=iterations,
        log_dir=log_dir,
    )

    env_name = str(cfg.get("env", {}).get("name", "allegrokuka"))
    metric_key = METRIC_BY_ENV.get(env_name, "eval/successes")

    run_log_dir = None
    if log_dir:
        run_log_dir = os.path.join(log_dir, f"{algorithm}_bs{batch_size}_seed{seed}")

    logger = MetricLogger(
        log_dir=run_log_dir,
        use_tensorboard=False,
        print_every=1,
        prefix="",
        verbose=verbose,
    )

    if verbose:
        print(
            f"[batchsize-sweep] algorithm={algorithm} batch_size={batch_size} "
            f"num_blocks={cfg['env'].get('num_blocks')} env={env_name}"
        )

    trainer = SAPGTrainer(cfg, logger=logger)
    history: List[Dict[str, float]] = []
    start = time.time()
    try:
        total_iters = int(cfg.get("run", {}).get("total_iterations", DEFAULT_ITERATIONS))
        for it in range(total_iters):
            trainer.train_iteration()
            if eval_every and (it + 1) % eval_every == 0:
                metrics = trainer.evaluate(num_episodes=eval_episodes, deterministic=True)
                record = {"iteration": it + 1}
                record.update({k: float(v) for k, v in metrics.items()})
                history.append(record)
                if verbose:
                    print(
                        f"  iter {it + 1}/{total_iters} "
                        f"{metric_key}={record.get(metric_key, float('nan')):.4f}"
                    )
        final_eval = trainer.evaluate(num_episodes=eval_episodes, deterministic=True)
    finally:
        try:
            trainer.close()
        except Exception:
            pass
        try:
            logger.close()
        except Exception:
            pass

    elapsed = time.time() - start
    final_metric = float(final_eval.get(metric_key, float("nan")))

    if verbose:
        print(
            f"[batchsize-sweep] done algorithm={algorithm} batch_size={batch_size} "
            f"{metric_key}={final_metric:.4f} ({elapsed:.1f}s)"
        )

    return {
        "algorithm": algorithm,
        "batch_size": int(batch_size),
        "num_blocks": int(cfg["env"].get("num_blocks", 1)),
        "metric_key": metric_key,
        "final_eval": {k: float(v) for k, v in final_eval.items()},
        "final_metric": final_metric,
        "history": history,
        "elapsed": elapsed,
        "config": cfg,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_batchsize_sweep(
    base_cfg: Dict[str, Any],
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    seed: int = 0,
    iterations: Optional[int] = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    eval_every: int = 0,
    log_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full batch-size sweep for every algorithm."""
    results: List[Dict[str, Any]] = []
    for algorithm in algorithms:
        for batch_size in batch_sizes:
            result = run_single(
                base_cfg,
                algorithm=algorithm,
                batch_size=int(batch_size),
                num_blocks=num_blocks,
                seed=seed,
                iterations=iterations,
                eval_episodes=eval_episodes,
                eval_every=eval_every,
                log_dir=log_dir,
                verbose=verbose,
            )
            results.append(result)

    summary: Dict[str, Dict[str, float]] = {}
    for result in results:
        summary.setdefault(result["algorithm"], {})[str(result["batch_size"])] = result[
            "final_metric"
        ]

    return {
        "results": results,
        "summary": summary,
        "batch_sizes": [int(b) for b in batch_sizes],
        "algorithms": list(algorithms),
    }


def summarize(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Return ``{algorithm: {batch_size: final_metric}}`` from sweep output."""
    if "summary" in results:
        return results["summary"]
    summary: Dict[str, Dict[str, float]] = {}
    for result in results.get("results", []):
        summary.setdefault(result["algorithm"], {})[str(result["batch_size"])] = result[
            "final_metric"
        ]
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAPG batch-size scaling experiment (Figure 2)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a base YAML config (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="allegrokuka",
        choices=["allegrokuka", "shadow_hand", "allegro_hand"],
        help="Environment family to run the sweep on.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (e.g. regrasping/throw/reorientation for AllegroKuka).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Total number of parallel environments to sweep over.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=list(DEFAULT_ALGORITHMS),
        help="Algorithms to compare (ppo, sapg).",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=DEFAULT_NUM_BLOCKS,
        help="Number of follower blocks used by SAPG.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override total training iterations (defaults to config value).",
    )
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate every N iterations (0 disables intermediate evaluation).",
    )
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the sweep summary as JSON.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    base_cfg = load_config(args.config, task=args.task)
    _deep_update(base_cfg, {"env": {"name": args.env}})
    if args.task is not None:
        _deep_update(base_cfg, {"env": {"task": args.task}})

    results = run_batchsize_sweep(
        base_cfg,
        batch_sizes=args.batch_sizes,
        algorithms=args.algorithms,
        num_blocks=args.num_blocks,
        seed=args.seed,
        iterations=args.iterations,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        log_dir=args.log_dir,
        verbose=verbose,
    )

    summary = summarize(results)
    if verbose:
        print("\n=== Batch-size sweep summary (final metric) ===")
        for algorithm, per_bs in summary.items():
            print(f"  {algorithm}:")
            for bs in sorted(per_bs, key=lambda x: int(x)):
                print(f"    batch_size={bs:>8}  metric={per_bs[bs]:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(
                {
                    "summary": summary,
                    "batch_sizes": results["batch_sizes"],
                    "algorithms": results["algorithms"],
                    "results": [
                        {
                            "algorithm": r["algorithm"],
                            "batch_size": r["batch_size"],
                            "num_blocks": r["num_blocks"],
                            "metric_key": r["metric_key"],
                            "final_metric": r["final_metric"],
                            "final_eval": r["final_eval"],
                            "history": r["history"],
                        }
                        for r in results["results"]
                    ],
                },
                fh,
                indent=2,
            )
        if verbose:
            print(f"\nWrote sweep results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
