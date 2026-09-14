"""Non-sequential NPSE benchmark training and evaluation.

Reproduces the non-sequential Neural Posterior Score Estimation results
(Figure 2 / Appendix F of the paper). For each benchmark and simulation
budget, this script:

  1. obtains a test observation,
  2. trains an NPSE posterior score network with denoising posterior score
     matching,
  3. draws approximate posterior samples by solving the probability-flow ODE,
  4. compares the approximate samples to reference posterior samples using
     C2ST (primary metric) and MMD (secondary metric).

Results are written to a JSON file and printed as a table, so they can be
consumed by ``evaluate_benchmarks.py`` and ``make_figures.py``.

Examples
--------
    # Run all benchmarks at all budgets from the YAML config:
    python -m npse.experiments.train_npse_benchmarks

    # Run two benchmarks at two budgets:
    python -m npse.experiments.train_npse_benchmarks \
        --benchmarks slcp,lotka_volterra --budgets 1000,10000

    # Use the VP SDE and a different seed:
    python -m npse.experiments.train_npse_benchmarks \
        --sde-type vp --seed 1234
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npse.benchmarks import get_benchmark  # noqa: E402
from npse.src.metrics import c2st_score, mmd  # noqa: E402
from npse.src.npse import train_npse  # noqa: E402

DEFAULT_CONFIG_DIR = ROOT / "config"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "npse_benchmarks"
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
    """Load a YAML file, returning an empty dict if it is unavailable."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def select_device(device: Optional[str]) -> torch.device:
    """Resolve ``'auto'``/``None`` device specifications to a torch device."""
    if device is None or str(device).lower() in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_sde_config(
    benchmark_name: str,
    benchmark_config: Dict[str, Any],
    default_config: Dict[str, Any],
    cli_sde_type: Optional[str],
) -> Tuple[str, Optional[float], Optional[float]]:
    """Return ``(sde_type, sigma_min, sigma_max)`` for a benchmark."""
    sde_defaults = default_config.get("sde", {}) or {}
    ve_defaults = sde_defaults.get("ve", {}) or {}
    vp_defaults = sde_defaults.get("vp", {}) or {}

    sde_type = cli_sde_type or benchmark_config.get(
        "sde_type", sde_defaults.get("type", "ve")
    )
    sigma_min = benchmark_config.get("sigma_min", ve_defaults.get("sigma_min"))
    sigma_max = benchmark_config.get("sigma_max", ve_defaults.get("sigma_max"))

    if sde_type.lower() == "vp":
        # VP does not use sigma_min/sigma_max; leave them unset.
        sigma_min = benchmark_config.get("sigma_min", None) if sigma_min is not None else None
        sigma_max = benchmark_config.get("sigma_max", None) if sigma_max is not None else None

    return str(sde_type).lower(), sigma_min, sigma_max


def get_observation(
    benchmark: Any, observation_index: int = 0
) -> torch.Tensor:
    """Return a single 1-D test observation for the benchmark."""
    x_obs = benchmark.sample_observation(n=max(1, observation_index + 1))
    if isinstance(x_obs, (list, tuple)):
        x_obs = x_obs[0]
    x_obs = torch.as_tensor(x_obs, dtype=benchmark.dtype, device=benchmark.device)
    if x_obs.dim() == 2:
        idx = min(observation_index, x_obs.shape[0] - 1)
        x_obs = x_obs[idx]
    return x_obs.view(-1)


def reference_samples(
    benchmark: Any, x_obs: torch.Tensor, n_samples: int
) -> Optional[np.ndarray]:
    """Return reference posterior samples as a NumPy array, or ``None``."""
    try:
        ref = benchmark.reference_posterior_samples(x_obs, n_samples)
        if ref is None:
            return None
        ref = torch.as_tensor(ref, dtype=torch.float64).detach().cpu().numpy()
        return np.asarray(ref, dtype=np.float64)
    except Exception:
        return None


def run_one(
    benchmark_name: str,
    budget: int,
    benchmark_config: Dict[str, Any],
    default_config: Dict[str, Any],
    sde_type: Optional[str],
    device: torch.device,
    seed: int,
    n_posterior: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Train and evaluate NPSE for a single benchmark/budget combination."""
    start_time = time.time()
    record: Dict[str, Any] = {
        "benchmark": benchmark_name,
        "budget": int(budget),
        "seed": int(seed),
        "sde_type": None,
        "c2st": None,
        "mmd": None,
        "status": "ok",
        "elapsed_seconds": None,
        "error": None,
    }

    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        benchmark = get_benchmark(benchmark_name, device=device, dtype=torch.float32)
        obs_index = int(benchmark_config.get("observation_index", 0))
        x_obs = get_observation(benchmark, obs_index)

        sde_type, sigma_min, sigma_max = resolve_sde_config(
            benchmark_name, benchmark_config, default_config, sde_type
        )
        record["sde_type"] = sde_type

        if verbose:
            print(
                f"\n=== NPSE: {benchmark_name} | budget={budget} | "
                f"sde={sde_type} ===",
                flush=True,
            )

        trainer = train_npse(
            benchmark=benchmark,
            budget=int(budget),
            sde_type=sde_type,
            seed=seed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            verbose=verbose,
        )

        approx = trainer.sample_posterior(n_samples=n_posterior, x_obs=x_obs)
        approx_np = (
            torch.as_tensor(approx, dtype=torch.float64).detach().cpu().numpy()
        )
        approx_np = np.asarray(approx_np, dtype=np.float64)

        ref_np = reference_samples(benchmark, x_obs, n_posterior)
        if ref_np is None:
            record["status"] = "no_reference"
            record["error"] = "reference_posterior_samples returned None"
        else:
            if ref_np.ndim == 3 and ref_np.shape[0] == 1:
                ref_np = ref_np[0]
            if approx_np.ndim == 3 and approx_np.shape[0] == 1:
                approx_np = approx_np[0]

            # Match sample counts when possible.
            min_n = min(approx_np.shape[0], ref_np.shape[0])
            approx_np = approx_np[:min_n]
            ref_np = ref_np[:min_n]

            if verbose:
                print(
                    f"  approx shape={approx_np.shape}, ref shape={ref_np.shape}",
                    flush=True,
                )

            record["c2st"] = float(c2st_score(approx_np, ref_np))
            record["mmd"] = float(mmd(approx_np, ref_np))

            if verbose:
                print(
                    f"  C2ST={record['c2st']:.4f}, MMD={record['mmd']:.6f}",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 - report and continue other runs
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"  ERROR: {record['error']}", flush=True)

    record["elapsed_seconds"] = float(time.time() - start_time)
    return record


def parse_budgets(
    cli_budgets: Optional[List[int]],
    benchmark_config: Dict[str, Any],
    default_config: Dict[str, Any],
) -> List[int]:
    if cli_budgets:
        return [int(b) for b in cli_budgets]
    cfg_budgets = benchmark_config.get("budgets")
    if cfg_budgets:
        return [int(b) for b in cfg_budgets]
    return [int(b) for b in default_config.get("budgets", DEFAULT_BUDGETS)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate non-sequential NPSE on SBI benchmarks."
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark names (default: all benchmarks).",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default=None,
        help="Comma-separated simulation budgets (default: from config).",
    )
    parser.add_argument(
        "--sde-type",
        type=str,
        default=None,
        choices=["ve", "vp"],
        help="Override the SDE type for all benchmarks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory in which to save results JSON.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing default.yaml and benchmarks.yaml.",
    )
    parser.add_argument("--n-posterior", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    verbose = not args.quiet

    config_dir = Path(args.config_dir)
    default_config = load_yaml(config_dir / "default.yaml")
    benchmarks_config = load_yaml(config_dir / "benchmarks.yaml")

    if args.benchmarks:
        benchmark_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    else:
        benchmark_names = list(benchmarks_config.keys() - {
            "tsnpse", "snpse", "nlse", "snpse_variants", "pyloric"
        }) or ALL_BENCHMARKS
        # Preserve a stable, expected ordering.
        benchmark_names = [b for b in ALL_BENCHMARKS if b in benchmark_names]

    cli_budgets = None
    if args.budgets:
        cli_budgets = [int(b.strip()) for b in args.budgets.split(",") if b.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    print(f"Device: {device}", flush=True)

    for benchmark_name in benchmark_names:
        benchmark_config = benchmarks_config.get(benchmark_name, {}) or {}
        budgets = parse_budgets(cli_budgets, benchmark_config, default_config)
        for budget in budgets:
            record = run_one(
                benchmark_name=benchmark_name,
                budget=budget,
                benchmark_config=benchmark_config,
                default_config=default_config,
                sde_type=args.sde_type,
                device=device,
                seed=args.seed,
                n_posterior=args.n_posterior,
                verbose=verbose,
            )
            all_records.append(record)

    results_path = output_dir / "npse_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print("\n================ NPSE RESULTS ================", flush=True)
    header = f"{'benchmark':<22}{'budget':>9}{'sde':>5}{'c2st':>9}{'mmd':>11}{'status':>14}"
    print(header, flush=True)
    for r in all_records:
        c2st = f"{r['c2st']:.4f}" if r["c2st"] is not None else "---"
        mmd = f"{r['mmd']:.6f}" if r["mmd"] is not None else "---"
        print(
            f"{r['benchmark']:<22}{r['budget']:>9}{r['sde_type'] or '':>5}"
            f"{c2st:>9}{mmd:>11}{r['status']:>14}",
            flush=True,
        )
    print(f"\nResults written to {results_path}", flush=True)


if __name__ == "__main__":
    main()
