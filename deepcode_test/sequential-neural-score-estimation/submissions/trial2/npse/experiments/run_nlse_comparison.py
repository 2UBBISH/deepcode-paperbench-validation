"""Reproduce Figure 5: NPSE vs NLSE comparison.

This experiment compares non-sequential Neural Posterior Score Estimation
(NPSE) against Neural Likelihood Score Estimation (NLSE) on the benchmarks
where an exact perturbed-prior score is available under the VE SDE:

- ``gaussian_linear``        (Gaussian prior)
- ``gaussian_linear_uniform`` (uniform prior)
- ``gaussian_mixture``        (uniform prior)
- ``two_moons``               (uniform prior)

Both methods use the *same* calibrated diffusion SDE, the same observation,
and the same simulation budget.  Approximate posterior samples are produced by
solving the probability-flow ODE and are compared to reference posterior
samples with C2ST and MMD.  Results are written as JSON for downstream figure
generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from npse.benchmarks import get_benchmark
from npse.src.metrics import c2st_score, mmd
from npse.src.nlse import train_nlse
from npse.src.npse import build_sde, train_npse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "nlse_comparison"

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
# The four benchmarks used in the paper's NPSE-vs-NLSE comparison (Figure 5).
DEFAULT_BENCHMARKS = [
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "two_moons",
]
DEFAULT_BUDGETS = [10000]
DEFAULT_N_POSTERIOR = 2000


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config file, returning an empty dict if unavailable."""
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    """Resolve ``(sde_type, sigma_min, sigma_max)`` from CLI/config/defaults."""
    if cli_sde_type:
        sde_type = cli_sde_type.lower()
    else:
        sde_type = str(
            benchmark_config.get("sde_type")
            or default_config.get("sde", {}).get("type")
            or "ve"
        ).lower()

    ve_cfg = benchmark_config.get("ve", {})
    default_ve = default_config.get("sde", {}).get("ve", {})
    sigma_min = ve_cfg.get("sigma_min", default_ve.get("sigma_min", None))
    sigma_max = ve_cfg.get("sigma_max", default_ve.get("sigma_max", None))
    return sde_type, sigma_min, sigma_max


def parse_budgets(
    cli_budgets: Optional[List[int]],
    benchmark_config: Dict[str, Any],
    default_config: Dict[str, Any],
) -> List[int]:
    """Resolve simulation budgets from CLI, benchmark config, or defaults."""
    if cli_budgets:
        return [int(b) for b in cli_budgets]
    if "budgets" in benchmark_config:
        return [int(b) for b in benchmark_config["budgets"]]
    if "budgets" in default_config:
        return [int(b) for b in default_config["budgets"]]
    return list(DEFAULT_BUDGETS)


# ---------------------------------------------------------------------------
# Observation / reference helpers
# ---------------------------------------------------------------------------
def get_observation(benchmark: Any, observation_index: int = 0) -> torch.Tensor:
    """Return a flattened 1-D test observation for a benchmark."""
    try:
        obs = benchmark.sample_observation(observation_index + 1)
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        obs = torch.as_tensor(obs, device=benchmark.device, dtype=benchmark.dtype)
        if obs.ndim > 1:
            obs = obs[min(int(observation_index), obs.shape[0] - 1)]
        return obs.reshape(-1)
    except Exception:
        try:
            theta = benchmark.prior_sample(1)
            x = benchmark.simulator(theta)
            if isinstance(x, (list, tuple)):
                x = x[0]
            return torch.as_tensor(x, device=benchmark.device, dtype=benchmark.dtype).reshape(-1)
        except Exception:
            raise


def reference_samples(
    benchmark: Any, x_obs: torch.Tensor, n_samples: int
) -> Optional[np.ndarray]:
    """Return reference posterior samples as a float64 NumPy array, or None."""
    try:
        ref = benchmark.reference_posterior_samples(x_obs, n_samples)
        if ref is None:
            return None
        arr = np.asarray(ref, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Analytic prior scores for NLSE
# ---------------------------------------------------------------------------
def make_analytic_prior_score(benchmark: Any, sde: Any) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build an exact perturbed-prior score for benchmarks with analytic priors.

    For the four NLSE comparison benchmarks the prior is either independent
    Gaussian or independent uniform, so the VE/VP perturbed prior score can be
    computed in closed form using :mod:`npse.src.prior`.
    """
    try:
        from npse.src.prior import GaussianMixturePriorScore, UniformPriorScore
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(
            "Analytic prior-score classes are unavailable from npse.src.prior"
        ) from exc

    name = getattr(benchmark, "name", "")
    device = getattr(benchmark, "device", torch.device("cpu"))
    dtype = getattr(benchmark, "dtype", torch.float32)

    if name == "gaussian_linear":
        dim = int(getattr(benchmark, "theta_dim", 10))
        prior_var = float(getattr(benchmark, "prior_var", 0.1))
        mean = torch.zeros(dim, device=device, dtype=dtype)
        cov = torch.eye(dim, device=device, dtype=dtype) * prior_var
        return GaussianMixturePriorScore(sde, [1.0], [mean], [cov])

    if name in ("gaussian_linear_uniform", "two_moons", "gaussian_mixture"):
        low = float(getattr(benchmark, "low", -10.0))
        high = float(getattr(benchmark, "high", 10.0))
        return UniformPriorScore(sde, low, high)

    # Implicit-prior fallback: train a prior score network.  This is not used
    # by the default Figure-5 benchmarks but keeps the script robust.
    from npse.src.prior import train_prior_score_network

    prior_net = train_prior_score_network(
        benchmark.prior_sample,
        int(getattr(benchmark, "theta_dim", 1)),
        sde,
        device=device,
        dtype=dtype,
        verbose=False,
    )
    return prior_net


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
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
    """Train NPSE and NLSE on one benchmark and evaluate both posteriors."""
    t0 = time.time()
    result: Dict[str, Any] = {
        "benchmark": benchmark_name,
        "budget": budget,
        "sde_type": sde_type,
        "seed": seed,
        "status": "error",
    }

    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark = get_benchmark(benchmark_name, device=device)
    x_obs = get_observation(benchmark, int(benchmark_config.get("observation_index", 0)))
    ref = reference_samples(benchmark, x_obs, n_posterior)

    resolved_sde_type, sigma_min, sigma_max = resolve_sde_config(
        benchmark_name, benchmark_config, default_config, sde_type
    )
    result["sde_type"] = resolved_sde_type

    # Calibrate a single SDE from prior samples so that NPSE and NLSE are
    # evaluated under exactly the same diffusion process.
    prior_calibration = benchmark.prior_sample(2000).to(device)
    sde = build_sde(
        resolved_sde_type,
        prior_calibration,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    try:
        prior_score_fn = make_analytic_prior_score(benchmark, sde)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"prior_score: {exc}"
        return result

    result["reference_count"] = int(ref.shape[0]) if ref is not None else 0
    result["observation"] = x_obs.detach().cpu().numpy().tolist()

    # --- NLSE --------------------------------------------------------------
    try:
        if verbose:
            print(f"[NLSE] {benchmark_name} budget={budget}", flush=True)
        nlse_trainer = train_nlse(
            benchmark,
            prior_score_fn,
            budget,
            sde,
            seed=seed,
            verbose=verbose,
        )
        nlse_samples = nlse_trainer.sample_posterior(n_posterior, x_obs)
        nlse_np = np.asarray(nlse_samples.detach().cpu().numpy(), dtype=np.float64)
        if ref is not None:
            result["nlse_c2st"] = float(c2st_score(nlse_np, ref))
            result["nlse_mmd"] = float(mmd(nlse_np, ref))
        result["nlse_status"] = "ok"
    except Exception as exc:
        result["nlse_status"] = "error"
        result["nlse_error"] = str(exc)

    # --- NPSE --------------------------------------------------------------
    try:
        if verbose:
            print(f"[NPSE] {benchmark_name} budget={budget}", flush=True)
        npse_trainer = train_npse(
            benchmark,
            budget,
            sde=sde,
            seed=seed,
            verbose=verbose,
        )
        npse_samples = npse_trainer.sample_posterior(n_posterior, x_obs)
        npse_np = np.asarray(npse_samples.detach().cpu().numpy(), dtype=np.float64)
        if ref is not None:
            result["npse_c2st"] = float(c2st_score(npse_np, ref))
            result["npse_mmd"] = float(mmd(npse_np, ref))
        result["npse_status"] = "ok"
    except Exception as exc:
        result["npse_status"] = "error"
        result["npse_error"] = str(exc)

    if ref is None and result.get("nlse_status") != "error" and result.get("npse_status") != "error":
        result["status"] = "no_reference"
    elif result.get("nlse_status") == "error" and result.get("npse_status") == "error":
        result["status"] = "error"
    else:
        result["status"] = "ok"

    result["wall_time"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NPSE vs NLSE comparison (Figure 5 reproduction)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmark names to run (default: four analytic-prior benchmarks)",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=None,
        help="Simulation budgets (default: 10000)",
    )
    parser.add_argument("--sde-type", default=None, choices=["ve", "vp"], help="Override SDE type")
    parser.add_argument("--device", default=None, help="Device ('auto', 'cpu', 'cuda', ...)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-posterior", type=int, default=DEFAULT_N_POSTERIOR,
                        help="Number of posterior samples to draw")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR), help="Config directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose training output")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_config = load_yaml(config_dir / "default.yaml")
    benchmarks_config = load_yaml(config_dir / "benchmarks.yaml")
    nlse_config = benchmarks_config.get("nlse", {}) if isinstance(benchmarks_config, dict) else {}

    device = select_device(args.device)

    benchmark_names = args.benchmarks or nlse_config.get("benchmarks") or list(DEFAULT_BENCHMARKS)
    if args.benchmarks is None and args.budgets is None:
        budgets = nlse_config.get("budgets") or list(DEFAULT_BUDGETS)
    else:
        budgets = parse_budgets(args.budgets, nlse_config, default_config)

    results: List[Dict[str, Any]] = []
    for benchmark_name in benchmark_names:
        benchmark_config = (
            benchmarks_config.get(benchmark_name, {})
            if isinstance(benchmarks_config, dict)
            else {}
        )
        for budget in budgets:
            print(
                f"Running NLSE-vs-NPSE: benchmark={benchmark_name} "
                f"budget={budget} sde={args.sde_type or 'auto'}",
                flush=True,
            )
            row = run_one(
                benchmark_name,
                int(budget),
                benchmark_config,
                default_config,
                args.sde_type,
                device,
                int(args.seed),
                int(args.n_posterior),
                bool(args.verbose),
            )
            results.append(row)
            print(f"  -> {row.get('status')} "
                  f"(NPSE C2ST={row.get('npse_c2st')}, NLSE C2ST={row.get('nlse_c2st')})",
                  flush=True)

    out_path = output_dir / "nlse_comparison_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, sort_keys=True)
    print(f"Wrote results to {out_path}")

    n_errors = sum(1 for r in results if r.get("status") == "error")
    return 1 if n_errors == len(results) and results else 0


if __name__ == "__main__":
    sys.exit(main())
