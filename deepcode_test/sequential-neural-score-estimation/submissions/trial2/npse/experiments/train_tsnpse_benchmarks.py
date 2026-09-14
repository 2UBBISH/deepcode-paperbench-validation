"""Train Truncated Sequential Neural Posterior Score Estimation (TSNPSE) on benchmarks.

This script reproduces the sequential TSNPSE benchmark experiments from the paper
(Figure 3 and associated C2ST comparisons). For each benchmark and total simulation
budget, it:

  1. obtains a test observation,
  2. runs TSNPSE with ``rounds`` sequential HPR-truncated proposal rounds,
  3. draws approximate posterior samples by solving the probability-flow ODE,
  4. compares the samples against reference posterior samples using C2ST and MMD,
  5. writes results to a JSON file for downstream evaluation/figure scripts.

Usage:
    python -m npse.experiments.train_tsnpse_benchmarks \
        --benchmarks slcp lotka_volterra \
        --budgets 10000 100000 \
        --rounds 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from npse.benchmarks import get_benchmark
from npse.src.metrics import c2st_score, mmd
from npse.src.tsnpse import train_tsnpse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "tsnpse_benchmarks"

ALL_BENCHMARKS = [
    "gaussian_linear",
    "gaussian_mixture",
    "two_moons",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "slcp",
    "sir",
    "lotka_volterra",
]

DEFAULT_BUDGETS = [1000, 10000, 100000]


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file, returning an empty dict if unavailable."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def select_device(device: Optional[str]) -> torch.device:
    """Resolve ``'auto'``/``None`` device specifications to a torch device."""
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_sde_config(
    benchmark_name: str,
    benchmark_config: Dict[str, Any],
    default_config: Dict[str, Any],
    cli_sde_type: Optional[str],
) -> Tuple[str, Optional[float], Optional[float]]:
    """Determine ``(sde_type, sigma_min, sigma_max)`` for a benchmark.

    Precedence: CLI > benchmark-specific config > global default config.
    """
    if cli_sde_type is not None:
        sde_type = cli_sde_type.lower()
    else:
        sde_type = str(
            benchmark_config.get("sde_type")
            or default_config.get("sde", {}).get("type", "ve")
        ).lower()

    ve_defaults = default_config.get("sde", {}).get("ve", {})
    sigma_min = benchmark_config.get("sigma_min")
    if sigma_min is None:
        sigma_min = ve_defaults.get("sigma_min", 0.05)

    sigma_max = benchmark_config.get("sigma_max")
    if sigma_max is None:
        sigma_max = ve_defaults.get("sigma_max")

    return sde_type, sigma_min, sigma_max


def parse_budgets(
    cli_budgets: Optional[List[int]],
    benchmark_config: Dict[str, Any],
    default_config: Dict[str, Any],
) -> List[int]:
    """Resolve simulation budgets from CLI, benchmark config, or defaults."""
    if cli_budgets:
        return [int(b) for b in cli_budgets]
    budgets = benchmark_config.get("budgets")
    if budgets is None:
        budgets = default_config.get("budgets", DEFAULT_BUDGETS)
    return [int(b) for b in budgets]


def get_observation(benchmark: Any, observation_index: int = 0) -> torch.Tensor:
    """Return a 1-D test observation for a benchmark.

    Benchmark implementations may return stored/reference observations when
    available; otherwise this samples from the prior predictive distribution.
    """
    n = max(1, int(observation_index) + 1)
    try:
        obs = benchmark.sample_observation(n)
    except TypeError:
        obs = benchmark.sample_observation()

    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu()
    else:
        obs = torch.as_tensor(obs, dtype=torch.float32)

    if obs.ndim == 0:
        obs = obs.reshape(1)
    elif obs.ndim == 1:
        pass
    else:
        idx = min(int(observation_index), obs.shape[0] - 1)
        obs = obs[idx]

    if obs.numel() != benchmark.x_dim:
        obs = obs.reshape(-1)
    return obs.reshape(-1)


def reference_samples(
    benchmark: Any, x_obs: torch.Tensor, n_samples: int
) -> Optional[np.ndarray]:
    """Return reference posterior samples as a NumPy array, or ``None``."""
    try:
        ref = benchmark.reference_posterior_samples(x_obs, n_samples)
    except Exception:
        return None
    if ref is None:
        return None
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    else:
        ref = np.asarray(ref)
    return ref.astype(np.float64)


def run_one(
    benchmark_name: str,
    budget: int,
    benchmark_config: Dict[str, Any],
    default_config: Dict[str, Any],
    sde_type: Optional[str],
    rounds: int,
    device: torch.device,
    seed: int,
    n_posterior: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Train and evaluate TSNPSE for one benchmark/budget combination."""
    result: Dict[str, Any] = {
        "benchmark": benchmark_name,
        "budget": int(budget),
        "rounds": int(rounds),
        "status": "running",
    }
    t_start = time.time()
    try:
        benchmark = get_benchmark(benchmark_name, device=device)
        x_obs = get_observation(
            benchmark, int(benchmark_config.get("observation_index", 0))
        )
        x_obs = x_obs.to(device=device, dtype=benchmark.dtype)

        sde_type_resolved, sigma_min, sigma_max = resolve_sde_config(
            benchmark_name, benchmark_config, default_config, sde_type
        )
        result["sde_type"] = sde_type_resolved

        trainer = train_tsnpse(
            benchmark=benchmark,
            x_obs=x_obs,
            total_budget=int(budget),
            rounds=int(rounds),
            sde_type=sde_type_resolved,
            seed=int(seed),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            verbose=verbose,
        )

        samples = trainer.sample_posterior(int(n_posterior))
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        else:
            samples = np.asarray(samples)
        samples = np.asarray(samples, dtype=np.float64)

        ref = reference_samples(benchmark, x_obs, int(n_posterior))
        if ref is None:
            result["status"] = "no_reference"
        else:
            result["c2st"] = float(c2st_score(samples, ref))
            result["mmd"] = float(mmd(samples, ref))
            result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - record and continue over benchmarks
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["elapsed_seconds"] = float(time.time() - t_start)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TSNPSE on SBI benchmarks and evaluate against references."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        choices=ALL_BENCHMARKS,
        help="Benchmark names to run (default: all).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=None,
        help="Total simulation budgets (default: config or 1000/10000/100000).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of sequential TSNPSE rounds (default: config or 10).",
    )
    parser.add_argument(
        "--sde-type",
        type=str,
        default=None,
        choices=["ve", "vp"],
        help="Override the SDE type.",
    )
    parser.add_argument(
        "--n-posterior",
        type=int,
        default=2000,
        help="Number of approximate posterior samples per evaluation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('auto', 'cpu', or a CUDA index).",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing default.yaml and benchmarks.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for results JSON output.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-run logging."
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    default_config = load_yaml(config_dir / "default.yaml")
    benchmarks_config = load_yaml(config_dir / "benchmarks.yaml")

    device = select_device(args.device)
    seed = int(args.seed) if args.seed is not None else int(
        default_config.get("seed", 0)
    )

    benchmarks = args.benchmarks or ALL_BENCHMARKS
    rounds_global = args.rounds
    results: List[Dict[str, Any]] = []

    for name in benchmarks:
        benchmark_config = benchmarks_config.get(name, {})
        budgets = parse_budgets(args.budgets, benchmark_config, default_config)
        rounds = (
            rounds_global
            if rounds_global is not None
            else int(
                benchmark_config.get("tsnpse", {}).get(
                    "rounds",
                    default_config.get("tsnpse", {}).get("rounds", 10),
                )
            )
        )

        for budget in budgets:
            if not args.quiet:
                print(
                    f"[TSNPSE] benchmark={name} budget={budget} rounds={rounds} "
                    f"device={device}",
                    flush=True,
                )
            result = run_one(
                benchmark_name=name,
                budget=budget,
                benchmark_config=benchmark_config,
                default_config=default_config,
                sde_type=args.sde_type,
                rounds=rounds,
                device=device,
                seed=seed,
                n_posterior=args.n_posterior,
                verbose=not args.quiet,
            )
            results.append(result)
            if not args.quiet:
                status = result.get("status", "unknown")
                if status == "ok":
                    print(
                        f"  -> C2ST={result.get('c2st'):.4f} "
                        f"MMD={result.get('mmd'):.6g} "
                        f"({result.get('elapsed_seconds'):.1f}s)",
                        flush=True,
                    )
                else:
                    print(
                        f"  -> status={status} "
                        f"({result.get('elapsed_seconds'):.1f}s)",
                        flush=True,
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tsnpse_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} result(s) to {output_path}")

    if any(r.get("status") == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
