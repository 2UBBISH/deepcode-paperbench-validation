"""Ablation experiments for SAPG (Figure 6).

Compares the default leader-based SAPG aggregation against ablations:

* ``leader``      -- SAPG with a designated leader trained on the union of all
                     followers' off-policy transitions (the paper's method).
* ``symmetric``   -- symmetric aggregation: no designated leader; every worker
                     is updated on the union of *all* workers' transitions.
* ``entropy``     -- leader-based SAPG but with per-block learnable sigma
                     vectors (entropy-based exploration variant).

The script trains each variant on a chosen environment/task and records the
evaluation metric (successes per episode for AllegroKuka, net episode reward
for the hand tasks) so the resulting curves can be compared.

Usage
-----
    python -m experiments.run_ablation --env allegrokuka --task regrasping \
        --variants leader symmetric entropy --iterations 200

The script is intentionally lightweight: it reuses :class:`SAPGTrainer` from
``main.py`` and only overrides the configuration keys that distinguish the
variants.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Make the project root importable when the script is executed directly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main import SAPGTrainer, load_config  # noqa: E402
from utils.logger import MetricLogger  # noqa: E402


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS: Dict[str, Dict[str, Any]] = {
    # Default SAPG: leader trained on the union of all follower transitions.
    "leader": {
        "sapg": {"mode": "leader", "use_importance_weights": True},
        "network": {"per_block_sigma": False},
    },
    # Ablation: symmetric aggregation (no designated leader).
    "symmetric": {
        "sapg": {"mode": "symmetric", "use_importance_weights": True},
        "network": {"per_block_sigma": False},
    },
    # Ablation: entropy-based exploration (per-block learnable sigma).
    "entropy": {
        "sapg": {"mode": "leader", "use_importance_weights": True},
        "network": {"per_block_sigma": True},
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (in place) and return it."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_variant_config(
    base_cfg: Dict[str, Any],
    variant: str,
    *,
    seed: Optional[int] = None,
    iterations: Optional[int] = None,
    num_blocks: Optional[int] = None,
    num_envs: Optional[int] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a deep-copied config specialised for ``variant``."""
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {sorted(VARIANTS)}"
        )

    cfg = copy.deepcopy(base_cfg)
    _deep_update(cfg, copy.deepcopy(VARIANTS[variant]))

    run = cfg.setdefault("run", {})
    run["run_name"] = f"{run.get('run_name', 'sapg')}_{variant}"
    if seed is not None:
        run["seed"] = int(seed)
    if iterations is not None:
        run["total_iterations"] = int(iterations)
    if log_dir is not None:
        run["log_dir"] = log_dir

    env = cfg.setdefault("env", {})
    if num_blocks is not None:
        env["num_blocks"] = int(num_blocks)
    if num_envs is not None:
        env["num_envs"] = int(num_envs)

    return cfg


def run_variant(
    base_cfg: Dict[str, Any],
    variant: str,
    *,
    seed: int = 0,
    iterations: Optional[int] = None,
    num_blocks: Optional[int] = None,
    num_envs: Optional[int] = None,
    eval_episodes: int = 32,
    eval_every: int = 0,
    log_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a single ablation variant and return its evaluation history."""
    cfg = build_variant_config(
        base_cfg,
        variant,
        seed=seed,
        iterations=iterations,
        num_blocks=num_blocks,
        num_envs=num_envs,
        log_dir=log_dir,
    )

    logger = MetricLogger(
        log_dir=cfg.get("run", {}).get("log_dir"),
        use_tensorboard=False,
        print_every=1,
        prefix=f"{variant}/",
        verbose=verbose,
    )

    trainer = SAPGTrainer(cfg, logger=logger)
    total_iterations = int(cfg.get("run", {}).get("total_iterations", 100))

    history: List[Dict[str, float]] = []
    start = time.time()
    try:
        for it in range(total_iterations):
            stats = trainer.train_iteration()
            record: Dict[str, float] = {"iteration": float(it)}
            for key, value in (stats or {}).items():
                if isinstance(value, (int, float)):
                    record[key] = float(value)

            if eval_every and (it + 1) % eval_every == 0:
                metrics = trainer.evaluate(num_episodes=eval_episodes)
                for key, value in metrics.items():
                    record[f"eval/{key}"] = float(value)

            history.append(record)
            if verbose and (it + 1) % max(1, total_iterations // 10) == 0:
                print(
                    f"[{variant}] iter {it + 1}/{total_iterations} "
                    f"({time.time() - start:.1f}s)"
                )
    finally:
        try:
            trainer.close()
        except Exception:
            pass
        try:
            logger.close()
        except Exception:
            pass

    return {
        "variant": variant,
        "config": cfg,
        "history": history,
        "final_eval": trainer.evaluate(num_episodes=eval_episodes)
        if hasattr(trainer, "evaluate")
        else {},
    }


def summarize(results: List[Dict[str, Any]], metric_key: str = "eval/successes") -> Dict[str, float]:
    """Extract the final evaluation metric for each variant."""
    summary: Dict[str, float] = {}
    for res in results:
        variant = res["variant"]
        final_eval = res.get("final_eval") or {}
        value = final_eval.get(metric_key)
        if value is None:
            # Fall back to the last recorded history entry.
            history = res.get("history") or []
            for record in reversed(history):
                if metric_key in record:
                    value = record[metric_key]
                    break
        summary[variant] = float(value) if value is not None else float("nan")
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAPG ablation experiments (Figure 6)")
    parser.add_argument("--env", type=str, default="allegrokuka",
                        choices=["allegrokuka", "shadow_hand", "allegro_hand"])
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (e.g. regrasping/throw/reorientation).")
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["leader", "symmetric", "entropy"])
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON path for the summary.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    base_cfg = load_config(
        os.path.join(_ROOT, "configs", "default.yaml"),
        task=args.env,
    )
    if args.task:
        base_cfg.setdefault("env", {})["task"] = args.task

    results: List[Dict[str, Any]] = []
    for variant in args.variants:
        print(f"=== Running variant: {variant} ===")
        res = run_variant(
            base_cfg,
            variant,
            seed=args.seed,
            iterations=args.iterations,
            num_blocks=args.num_blocks,
            num_envs=args.num_envs,
            eval_episodes=args.eval_episodes,
            eval_every=args.eval_every,
            log_dir=args.log_dir,
            verbose=not args.quiet,
        )
        results.append(res)

    metric_key = "eval/successes" if args.env == "allegrokuka" else "eval/reward"
    summary = summarize(results, metric_key=metric_key)

    print("\n=== Ablation summary ===")
    for variant, value in summary.items():
        print(f"  {variant:>12s}: {value:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(
                {
                    "env": args.env,
                    "task": args.task,
                    "metric": metric_key,
                    "summary": summary,
                    "histories": {r["variant"]: r["history"] for r in results},
                },
                fh,
                indent=2,
            )
        print(f"Wrote results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
