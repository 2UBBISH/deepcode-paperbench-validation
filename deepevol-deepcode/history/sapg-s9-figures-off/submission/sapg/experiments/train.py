"""Training orchestration for SAPG and baselines (Sec 5, Table 1).

This module runs the full experimental suite described in the SAPG paper:

  * 5 tasks: regrasping, throw, reorientation (AllegroKuka, hard),
             shadowhand, allegrohand (easy)
  * 4 algorithms: sapg, ppo, pql, dexpbt
  * 5 seeds per (task, algorithm) pair
  * ~2e10 transitions per run (configurable for smoke tests)

Each run writes a JSON history file (one record per outer iteration) plus a
final checkpoint.  ``run_suite`` aggregates histories across seeds and writes
mean / standard-error curves suitable for reproducing Fig. 2, Fig. 5, Fig. 6
and Table 1.

Usage
-----
    # single run
    python -m experiments.train --task regrasping --algorithm sapg --seed 0

    # full suite (5 tasks x 4 algorithms x 5 seeds)
    python -m experiments.train --suite --output-dir runs

    # quick smoke test on CPU with the dummy env
    python -m experiments.train --suite --dry-run --num-iterations 3 \
        --seeds 0 1 --tasks regrasping
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Make the repository root importable when run as a script.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sapg.config import (  # noqa: E402
    BASELINE_CONFIGS,
    EXPECTED_RESULTS,
    TASK_CONFIGS,
    SAPGConfig,
    get_config,
)

# Task metadata mirrors main.py so this module is self-contained.
TASK_METADATA: Dict[str, Dict[str, Any]] = {
    "regrasping": {"family": "allegrokuka", "obs_dim": 64, "action_dim": 23},
    "throw": {"family": "allegrokuka", "obs_dim": 64, "action_dim": 23},
    "reorientation": {"family": "allegrokuka", "obs_dim": 68, "action_dim": 23},
    "shadowhand": {"family": "shadowhand", "obs_dim": 96, "action_dim": 24},
    "allegrohand": {"family": "allegrohand", "obs_dim": 61, "action_dim": 16},
}

ALGORITHMS: Tuple[str, ...] = ("sapg", "ppo", "pql", "dexpbt")
TASKS: Tuple[str, ...] = tuple(TASK_METADATA.keys())

# Metric reported per task family (Table 1).
TASK_METRIC: Dict[str, str] = {
    "regrasping": "successes",
    "throw": "successes",
    "reorientation": "successes",
    "shadowhand": "episode_reward",
    "allegrohand": "episode_reward",
}


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    """A single (task, algorithm, seed) training run."""

    task: str
    algorithm: str
    seed: int
    num_iterations: Optional[int] = None
    total_transitions: Optional[float] = None
    dry_run: bool = False
    output_dir: str = "runs"
    device: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        return f"{self.task}_{self.algorithm}_seed{self.seed}"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.run_name)


# ---------------------------------------------------------------------------
# Environment / trainer construction
# ---------------------------------------------------------------------------
def build_env(config: SAPGConfig, dry_run: bool = False):
    """Build the vectorized environment, falling back to the dummy env."""
    from envs.isaacgym_wrapper import HAS_ISAACGYM, DummyVectorEnv, make_env

    if dry_run or not HAS_ISAACGYM:
        meta = TASK_METADATA.get(config.task, {})
        return DummyVectorEnv(
            task=config.task,
            num_envs=config.num_envs,
            num_blocks=config.num_blocks,
            obs_dim=meta.get("obs_dim", 64),
            action_dim=meta.get("action_dim", 23),
            device=config.device,
            seed=config.seed,
            horizon=config.horizon,
        )
    return make_env(config, dry_run=False)


def build_trainer(algorithm: str, config: SAPGConfig, env, obs_dim: int, action_dim: int):
    """Instantiate the trainer for ``algorithm``."""
    if algorithm == "sapg":
        from sapg.algorithm import build_trainer as _build

        return _build(config, env, obs_dim, action_dim)
    if algorithm == "ppo":
        from baselines.ppo import build_ppo_trainer as _build

        return _build(config, env, obs_dim, action_dim)
    if algorithm == "pql":
        from baselines.pql import build_pql_trainer as _build

        return _build(config, env, obs_dim, action_dim)
    if algorithm == "dexpbt":
        from baselines.dexpbt import build_dexpbt_trainer as _build

        return _build(config, env, obs_dim, action_dim)
    raise ValueError(f"Unknown algorithm: {algorithm!r}")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(spec: RunSpec) -> Dict[str, Any]:
    """Execute one training run and persist its history + checkpoint."""
    os.makedirs(spec.run_dir, exist_ok=True)

    overrides: Dict[str, Any] = dict(spec.overrides)
    overrides["seed"] = spec.seed
    if spec.num_iterations is not None:
        overrides["num_iterations"] = spec.num_iterations
    if spec.total_transitions is not None:
        overrides["total_transitions"] = spec.total_transitions
    if spec.device is not None:
        overrides["device"] = spec.device

    config = get_config(spec.task, spec.algorithm, **overrides)

    meta = TASK_METADATA[spec.task]
    obs_dim = getattr(config, "obs_dim", None) or meta["obs_dim"]
    action_dim = getattr(config, "action_dim", None) or meta["action_dim"]

    env = build_env(config, dry_run=spec.dry_run)
    trainer = build_trainer(spec.algorithm, config, env, obs_dim, action_dim)

    print(
        f"[train] {spec.run_name}: task={spec.task} algo={spec.algorithm} "
        f"seed={spec.seed} envs={config.num_envs} blocks={config.num_blocks} "
        f"iters={config.num_iterations} device={config.device}"
    )

    t0 = time.time()
    history = trainer.train(config.num_iterations)
    wall = time.time() - t0

    records = [h.to_dict() if hasattr(h, "to_dict") else dict(h) for h in history]

    history_path = os.path.join(spec.run_dir, "history.json")
    with open(history_path, "w") as fh:
        json.dump(
            {
                "task": spec.task,
                "algorithm": spec.algorithm,
                "seed": spec.seed,
                "config": config.to_dict(),
                "wall_time": wall,
                "history": records,
            },
            fh,
            indent=2,
        )

    try:
        import torch

        torch.save(trainer.state_dict(), os.path.join(spec.run_dir, "final.pt"))
    except Exception as exc:  # pragma: no cover - checkpointing is best effort
        print(f"[train] warning: could not save checkpoint: {exc}")

    try:
        env.close()
    except Exception:
        pass

    print(f"[train] {spec.run_name} finished in {wall:.1f}s -> {history_path}")
    return {
        "task": spec.task,
        "algorithm": spec.algorithm,
        "seed": spec.seed,
        "history": records,
        "wall_time": wall,
        "run_dir": spec.run_dir,
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------
def _metric_key(task: str) -> str:
    return TASK_METRIC.get(task, "episode_reward")


def _extract_curve(records: Sequence[Dict[str, Any]], metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (transitions, metric) arrays from a run's history records."""
    xs: List[float] = []
    ys: List[float] = []
    for rec in records:
        x = rec.get("total_transitions")
        if x is None:
            x = rec.get("iteration", len(xs))
        y = rec.get(metric)
        if y is None:
            y = rec.get("mean_episode_reward", rec.get("mean_reward", 0.0))
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def aggregate_runs(
    runs: Sequence[Dict[str, Any]],
    num_points: int = 100,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate per-seed curves onto a common x-grid (mean + std error).

    Returns a mapping ``task -> {algorithm -> {x, mean, stderr, n}}``.
    """
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for run in runs:
        grouped.setdefault(run["task"], {}).setdefault(run["algorithm"], []).append(run)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for task, by_algo in grouped.items():
        metric = _metric_key(task)
        out[task] = {}
        for algo, algo_runs in by_algo.items():
            curves = [_extract_curve(r["history"], metric) for r in algo_runs]
            curves = [(x, y) for x, y in curves if x.size > 0]
            if not curves:
                continue
            x_max = min(float(x[-1]) for x, _ in curves)
            x_min = max(float(x[0]) for x, _ in curves)
            if x_max <= x_min:
                x_grid = np.asarray([x_min], dtype=np.float64)
            else:
                x_grid = np.linspace(x_min, x_max, num_points)

            interp = np.stack(
                [np.interp(x_grid, x, y) for x, y in curves], axis=0
            )  # [n_seeds, num_points]
            mean = interp.mean(axis=0)
            n = interp.shape[0]
            stderr = interp.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            out[task][algo] = {
                "x": x_grid,
                "mean": mean,
                "stderr": stderr,
                "n": np.asarray([n]),
            }
    return out


def final_metrics(runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute final (last-iteration) mean +/- stderr per task/algorithm."""
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for run in runs:
        metric = _metric_key(run["task"])
        _, ys = _extract_curve(run["history"], metric)
        if ys.size == 0:
            continue
        grouped.setdefault(run["task"], {}).setdefault(run["algorithm"], []).append(float(ys[-1]))

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task, by_algo in grouped.items():
        out[task] = {}
        for algo, vals in by_algo.items():
            arr = np.asarray(vals, dtype=np.float64)
            n = arr.size
            out[task][algo] = {
                "mean": float(arr.mean()),
                "stderr": float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "n": int(n),
            }
    return out


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------
def run_suite(
    tasks: Sequence[str],
    algorithms: Sequence[str],
    seeds: Sequence[int],
    output_dir: str = "runs",
    num_iterations: Optional[int] = None,
    total_transitions: Optional[float] = None,
    dry_run: bool = False,
    device: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full (task x algorithm x seed) grid and aggregate results."""
    overrides = overrides or {}
    all_runs: List[Dict[str, Any]] = []

    for task in tasks:
        for algorithm in algorithms:
            for seed in seeds:
                spec = RunSpec(
                    task=task,
                    algorithm=algorithm,
                    seed=seed,
                    num_iterations=num_iterations,
                    total_transitions=total_transitions,
                    dry_run=dry_run,
                    output_dir=output_dir,
                    device=device,
                    overrides=overrides,
                )
                try:
                    all_runs.append(run_single(spec))
                except Exception as exc:  # keep the suite going
                    print(f"[train] ERROR {spec.run_name}: {exc}")

    os.makedirs(output_dir, exist_ok=True)
    agg = aggregate_runs(all_runs)
    finals = final_metrics(all_runs)

    summary = {
        "tasks": list(tasks),
        "algorithms": list(algorithms),
        "seeds": list(seeds),
        "num_runs": len(all_runs),
        "final_metrics": finals,
        "expected_results": EXPECTED_RESULTS,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Persist aggregated curves as npz for plotting.
    npz_payload: Dict[str, np.ndarray] = {}
    for task, by_algo in agg.items():
        for algo, curve in by_algo.items():
            npz_payload[f"{task}__{algo}__x"] = curve["x"]
            npz_payload[f"{task}__{algo}__mean"] = curve["mean"]
            npz_payload[f"{task}__{algo}__stderr"] = curve["stderr"]
    np.savez(os.path.join(output_dir, "curves.npz"), **npz_payload)

    print(f"[train] suite complete: {len(all_runs)} runs -> {output_dir}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SAPG / baselines (Table 1).")
    p.add_argument("--task", type=str, default=None, choices=list(TASKS))
    p.add_argument("--algorithm", type=str, default=None, choices=list(ALGORITHMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds for --suite mode (default: 0..4).")
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Tasks for --suite mode (default: all).")
    p.add_argument("--algorithms", type=str, nargs="+", default=None,
                   help="Algorithms for --suite mode (default: all).")
    p.add_argument("--suite", action="store_true",
                   help="Run the full task x algorithm x seed grid.")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--num-iterations", type=int, default=None)
    p.add_argument("--total-transitions", type=float, default=None,
                   help="Target transitions per run (paper: 2e10).")
    p.add_argument("--dry-run", action="store_true",
                   help="Use the CPU dummy env (no IsaacGym).")
    p.add_argument("--device", type=str, default=None)
    # Common config overrides.
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--aggregation", type=str, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    return p


def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for name in ("num_envs", "num_blocks", "entropy_coef", "aggregation", "learning_rate"):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    overrides = _collect_overrides(args)

    if args.suite:
        tasks = args.tasks or list(TASKS)
        algorithms = args.algorithms or list(ALGORITHMS)
        seeds = args.seeds if args.seeds is not None else list(range(5))
        run_suite(
            tasks=tasks,
            algorithms=algorithms,
            seeds=seeds,
            output_dir=args.output_dir,
            num_iterations=args.num_iterations,
            total_transitions=args.total_transitions,
            dry_run=args.dry_run,
            device=args.device,
            overrides=overrides,
        )
        return 0

    if args.task is None or args.algorithm is None:
        print("error: --task and --algorithm are required unless --suite is given")
        return 2

    spec = RunSpec(
        task=args.task,
        algorithm=args.algorithm,
        seed=args.seed,
        num_iterations=args.num_iterations,
        total_transitions=args.total_transitions,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        device=args.device,
        overrides=overrides,
    )
    run_single(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
