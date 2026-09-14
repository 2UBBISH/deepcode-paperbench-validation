"""Reproduce Figure 6: SNPSE-A/B/C variant ablation.

This script trains the sequential neural posterior score estimation (SNPSE)
variants A, B and (optionally) C on the benchmarks specified in
``config/benchmarks.yaml`` and evaluates the resulting approximate posteriors
against reference posterior samples with C2ST and MMD.

Typical usage::

    python -m npse.experiments.run_snpse_variants \
        --benchmarks slcp gaussian_linear_uniform \
        --variants a b \
        --total-budget 10000 \
        --rounds 10

Results are written to a JSON file consumed by
``npse.experiments.evaluate_benchmarks`` and ``npse.experiments.make_figures``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from npse.benchmarks import get_benchmark
from npse.src.metrics import c2st_score, mmd
from npse.src.snpse import train_snpse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "snpse_variants"

ALL_BENCHMARKS: List[str] = [
    "gaussian_linear",
    "gaussian_mixture",
    "two_moons",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "slcp",
    "sir",
    "lotka_volterra",
]

ALL_VARIANTS: List[str] = ["a", "b", "c"]

DEFAULT_BENCHMARKS: List[str] = ["slcp", "gaussian_linear_uniform"]
DEFAULT_VARIANTS: List[str] = ["a", "b"]
DEFAULT_TOTAL_BUDGET: int = 10000
DEFAULT_ROUNDS: int = 10


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config file, returning an empty dict if unavailable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def select_device(device: Optional[str]) -> torch.device:
    """Resolve ``'auto'``/``None`` device specifications to a torch device."""
    if device is None or str(device).lower() in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_sde_config(
    benchmark_name: str,
    benchmark_config: Dict[str, Any],
    default_config: Dict[str, Any],
    cli_sde_type: Optional[str],
) -> Tuple[str, Optional[float], Optional[float]]:
    """Determine ``(sde_type, sigma_min, sigma_max)`` for a benchmark."""
    sde_defaults = default_config.get("sde", {}) if default_config else {}
    ve_defaults = sde_defaults.get("ve", {}) if isinstance(sde_defaults, dict) else {}

    sde_type = cli_sde_type or benchmark_config.get("sde_type") or sde_defaults.get("type") or "ve"
    sigma_min = benchmark_config.get("sigma_min", ve_defaults.get("sigma_min"))
    sigma_max = benchmark_config.get("sigma_max", ve_defaults.get("sigma_max"))
    return sde_type, sigma_min, sigma_max


def get_observation(benchmark: Any, observation_index: int = 0) -> torch.Tensor:
    """Return a flattened 1-D test observation for a benchmark."""
    x_obs = benchmark.sample_observation(observation_index + 1)
    if x_obs is None or (isinstance(x_obs, np.ndarray) and x_obs.size == 0):
        x_obs = benchmark.sample_observation(1)
    if x_obs is None:
        raise RuntimeError(f"Benchmark {getattr(benchmark, 'name', '?')} returned no observation")

    if isinstance(x_obs, np.ndarray):
        x_obs = torch.from_numpy(np.asarray(x_obs))
    else:
        x_obs = torch.as_tensor(x_obs)

    if x_obs.ndim == 2 and x_obs.shape[0] > 1:
        # sample_observation may return a batch of observations; select the
        # requested one if possible.
        idx = min(observation_index, x_obs.shape[0] - 1)
        x_obs = x_obs[idx]
    x_obs = x_obs.reshape(-1).to(benchmark.dtype)
    return x_obs


def reference_samples(
    benchmark: Any, x_obs: torch.Tensor, n_samples: int
) -> Optional[np.ndarray]:
    """Return reference posterior samples as a float64 NumPy array."""
    try:
        ref = benchmark.reference_posterior_samples(x_obs, n_samples)
    except Exception:
        return None
    if ref is None:
        return None
    if isinstance(ref, np.ndarray):
        arr = ref
    else:
        arr = ref.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def parse_budgets(
    cli_budget: Optional[int],
    variant_config: Dict[str, Any],
    default_config: Dict[str, Any],
) -> List[int]:
    """Resolve total simulation budgets from CLI, config, or defaults."""
    if cli_budget is not None:
        return [cli_budget]
    value = variant_config.get("total_budget")
    if value is None:
        value = variant_config.get("budgets")
    if value is None and isinstance(default_config, dict):
        value = default_config.get("budgets")
    if value is None:
        value = [DEFAULT_TOTAL_BUDGET]
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [DEFAULT_TOTAL_BUDGET]


def run_one(
    benchmark_name: str,
    variant: str,
    budget: int,
    benchmark_config: Dict[str, Any],
    variant_config: Dict[str, Any],
    default_config: Dict[str, Any],
    sde_type: Optional[str],
    rounds: int,
    device: torch.device,
    seed: int,
    n_posterior: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Train one SNPSE variant on one benchmark and evaluate it."""
    result: Dict[str, Any] = {
        "method": f"snpse_{variant}",
        "variant": variant,
        "benchmark": benchmark_name,
        "budget": budget,
        "rounds": rounds,
        "sde_type": sde_type,
        "seed": seed,
    }

    t_start = time.time()
    benchmark = get_benchmark(benchmark_name, device=device)

    try:
        x_obs = get_observation(benchmark, int(benchmark_config.get("observation_index", 0)))
    except Exception as exc:  # pragma: no cover - defensive reporting
        result["status"] = "error"
        result["error"] = f"observation error: {exc}"
        result["wall_time"] = time.time() - t_start
        return result

    # TSNPSE/SNPSE operate on a single observation.
    try:
        trainer = train_snpse(
            benchmark,
            x_obs=x_obs,
            total_budget=int(budget),
            rounds=int(rounds),
            variant=variant,
            sde_type=sde_type,
            seed=seed,
            sigma_min=benchmark_config.get("sigma_min"),
            sigma_max=benchmark_config.get("sigma_max"),
            verbose=verbose,
        )

        samples = trainer.sample_posterior(int(n_posterior), x_obs)
        if isinstance(samples, np.ndarray):
            samples_np = np.asarray(samples, dtype=np.float64)
        else:
            samples_np = samples.detach().cpu().numpy().astype(np.float64)

        if samples_np.ndim == 1:
            samples_np = samples_np.reshape(1, -1)

        ref = reference_samples(benchmark, x_obs, max(1000, int(n_posterior)))
        if ref is None:
            result["status"] = "no_reference"
        else:
            # Balance to the smaller sample count.
            n = min(samples_np.shape[0], ref.shape[0])
            if n < 10:
                result["status"] = "insufficient_samples"
            else:
                samples_np = samples_np[:n]
                ref = ref[:n]
                result["c2st"] = float(c2st_score(samples_np, ref))
                result["mmd"] = float(mmd(samples_np, ref))
                result["status"] = "ok"
        result["wall_time"] = time.time() - t_start
        return result
    except Exception as exc:  # pragma: no cover - defensive reporting
        result["status"] = "error"
        result["error"] = str(exc)
        result["wall_time"] = time.time() - t_start
        return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SNPSE-A/B/C variant ablation experiments (Figure 6)."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmark names to run (default: from config/benchmarks.yaml).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=ALL_VARIANTS,
        help="SNPSE variants to run: a, b, c (default: a b).",
    )
    parser.add_argument("--total-budget", type=int, default=None, help="Total simulation budget.")
    parser.add_argument("--rounds", type=int, default=None, help="Number of sequential rounds.")
    parser.add_argument("--sde-type", type=str, default=None, help="SDE type override (ve/vp).")
    parser.add_argument("--config-dir", type=str, default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default=None, help="Device: auto/cuda/cpu.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-posterior", type=int, default=1000, help="Posterior sample count.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    config_dir = Path(args.config_dir)
    default_config = load_yaml(config_dir / "default.yaml")
    benchmark_configs = load_yaml(config_dir / "benchmarks.yaml")
    variant_config = benchmark_configs.get("snpse_variants", {}) if benchmark_configs else {}

    device = select_device(args.device)

    benchmarks: List[str] = args.benchmarks or variant_config.get("benchmarks") or DEFAULT_BENCHMARKS
    variants: List[str] = args.variants or variant_config.get("variants") or DEFAULT_VARIANTS
    budgets: List[int] = parse_budgets(args.total_budget, variant_config, default_config)
    rounds: int = int(
        args.rounds or variant_config.get("rounds") or default_config.get("tsnpse", {}).get("rounds", DEFAULT_ROUNDS)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for benchmark_name in benchmarks:
        benchmark_config = benchmark_configs.get(benchmark_name, {}) if benchmark_configs else {}
        sde_type, _, _ = resolve_sde_config(
            benchmark_name, benchmark_config, default_config, args.sde_type
        )
        for variant in variants:
            for budget in budgets:
                if args.verbose:
                    print(
                        f"[snpse-{variant}] benchmark={benchmark_name} "
                        f"budget={budget} rounds={rounds} sde={sde_type}",
                        flush=True,
                    )
                result = run_one(
                    benchmark_name=benchmark_name,
                    variant=variant,
                    budget=budget,
                    benchmark_config=benchmark_config,
                    variant_config=variant_config,
                    default_config=default_config,
                    sde_type=sde_type,
                    rounds=rounds,
                    device=device,
                    seed=args.seed,
                    n_posterior=args.n_posterior,
                    verbose=args.verbose,
                )
                all_results.append(result)
                if args.verbose:
                    status = result.get("status")
                    c2st = result.get("c2st")
                    print(f"  -> status={status} c2st={c2st}", flush=True)

    output_path = output_dir / "snpse_variants_results.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, sort_keys=True)

    print(f"Wrote {len(all_results)} result(s) to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
