#!/usr/bin/env python
"""Entry-point that dispatches every BaM reproduction experiment.

The script is the single command-line front-end for the reproduction package.
It loads the YAML configurations in ``bam_repro/configs`` (when PyYAML and the
files are available), translates them into the keyword arguments accepted by the
per-experiment drivers in ``bam_repro.experiments``, runs the requested
experiments, and writes a combined summary.

Experiments
-----------
gaussian       Sec. 5.1 / Fig. 5.1 (+ E.3): Gaussian targets, D = 4, 16, 64, 256.
               BaM uses the constant schedule ``lambda_t = B D``; ADVI / Score /
               Fisher / GSM use batch size 2 with grid-searched Adam rates
               (ADVI 0.01, Fisher 0.01, Score [0.01, 0.005, 0.001, 0.001] for
               D = 4, 16, 64, 256).  Init ``mu_0 ~ Uniform[0, 0.1]``,
               ``Sigma_0 = I``.  10 runs.
non_gaussian   Sec. 5.1 / Fig. 5.2 (+ E.4): sinh-arcsinh target, D = 10, six
               settings (skew s in {0.2, 1.0, 1.8} with tau = 1; tails tau in
               {0.1, 0.9, 1.7} with s = 0).  BaM uses the decaying schedule
               ``lambda_t = B D / (t + 1)``; baselines use B = 5 with ADVI 0.02,
               Fisher 0.05 and per-setting Score rates.  10 runs.
posteriordb    Sec. 5.2 / Fig. 5.3 (+ E.6): ark (D=7), gp-pois-regr (D=13),
               eight-schools-centered (D=10); relative mean / SD error against
               HMC reference moments, ``lambda_t = B D / (t + 1)``, B in {8, 32},
               5 runs.
vae            Sec. 5.3 / Fig. 5.4: CIFAR-10 decoder target p(z'|x'), pilot run
               of T = 100 iterations for learning-rate selection, then T = 1000;
               B in {10, 100, 300}; reconstruction MSE.

The cost axis for all experiments is the number of gradient evaluations
(wall-clock timings are recorded but explicitly out of scope for the paper's
comparisons, cf. Appendix E.2).

Usage
-----
::

    python -m bam_repro.scripts.run_all --experiments all
    python -m bam_repro.scripts.run_all --experiments gaussian non_gaussian
    python -m bam_repro.scripts.run_all --experiments all --quick
    python -m bam_repro.scripts.run_all --config bam_repro/configs/gaussian.yaml
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # PyYAML is optional; built-in defaults are used when unavailable.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


# --------------------------------------------------------------------------- #
# Import shims: support both ``import bam_repro...`` and direct execution.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)               # .../bam_repro
_REPO_ROOT = os.path.dirname(_PKG_ROOT)          # .../  (parent of bam_repro)

for _p in (_REPO_ROOT, _PKG_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from bam_repro.experiments import available_experiments as _available_experiments
except Exception:  # pragma: no cover - extremely defensive
    def _available_experiments() -> List[str]:  # type: ignore
        return []


ALL_EXPERIMENTS: Tuple[str, ...] = ("gaussian", "non_gaussian", "posteriordb", "vae")

#: YAML configuration file for each experiment (inside ``bam_repro/configs``).
CONFIG_FILES: Dict[str, str] = {
    "gaussian": "gaussian.yaml",
    "non_gaussian": "non_gaussian.yaml",
    "posteriordb": "posteriordb.yaml",
    "vae": "vae.yaml",
}

#: BaM schedules examined in Appendix E.3 (D = 16) / E.4 (D = 10).
SCHEDULE_NAMES: Tuple[str, ...] = ("BD", "BD/(t+1)", "BD/sqrt(t+1)", "B/(t+1)")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def config_dir() -> str:
    """Directory holding the YAML configuration files."""
    return os.path.join(_PKG_ROOT, "configs")


def load_config(experiment: str, path: Optional[str] = None) -> Dict[str, Any]:
    """Load the YAML configuration for ``experiment`` (empty dict if missing)."""
    if path is None:
        path = os.path.join(config_dir(), CONFIG_FILES.get(experiment, ""))
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        print(f"[run_all] PyYAML unavailable; using built-in defaults for {experiment!r}")
        return {}
    try:
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover
        print(f"[run_all] failed to parse {path}: {exc}; using built-in defaults")
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _filter_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keyword arguments ``func`` does not declare."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _learning_rate_map(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``learning_rates`` mapping expected by the experiment drivers."""
    out: Dict[str, Any] = {}
    for method in ("advi", "score", "fisher", "gsm"):
        block = cfg.get(method)
        if isinstance(block, dict) and "learning_rate" in block:
            out[method] = block.get("learning_rate")
    return out


def _resolve_learning_rates(
    cfg: Dict[str, Any], experiment: str
) -> Optional[Dict[str, Any]]:
    """Resolve per-method learning rates, expanding per-dim/per-setting dicts.

    The paper grid-searches Adam rates for the gradient-based baselines; the
    YAML stores either a scalar (all settings) or a dict keyed by dimension
    (Gaussian) / setting name (non-Gaussian) / model name (posteriordb).
    The drivers accept both forms, so dicts are passed through unchanged and
    only obvious nulls are dropped.
    """
    rates = _learning_rate_map(cfg)
    return rates or None


def _bam_batch_sizes(cfg: Dict[str, Any], default: Sequence[int]) -> Tuple[int, ...]:
    bam_cfg = cfg.get("bam") if isinstance(cfg.get("bam"), dict) else {}
    sizes = bam_cfg.get("batch_sizes", default)
    if isinstance(sizes, (int, float)):
        sizes = (int(sizes),)
    return tuple(int(s) for s in sizes)


def _dispatch(
    experiment: str,
    quick: bool,
    outdir: Optional[str],
    config: Optional[str],
    verbose: bool,
) -> Tuple[Optional[Any], float]:
    """Run one experiment; return ``(result, wallclock_seconds)``."""
    t0 = time.time()
    cfg = load_config(experiment, config)
    common: Dict[str, Any] = {"verbose": bool(verbose)}
    if outdir is not None:
        common["outdir"] = outdir
    if cfg.get("seed") is not None:
        common["seed"] = int(cfg["seed"])

    if experiment == "gaussian":
        from bam_repro.experiments import exp_gaussian as mod

        if quick:
            result = mod.run_gaussian(quick=True, **_filter_kwargs(mod.run_gaussian, common))
        else:
            kwargs: Dict[str, Any] = dict(common)
            if cfg:
                kwargs.update(
                    dims=tuple(cfg.get("dims", mod.PAPER_DIMS)),
                    n_runs=int(cfg.get("n_runs", mod.PAPER_GAUSSIAN_N_RUNS)),
                    methods=tuple(cfg.get("methods", mod.PAPER_GAUSSIAN_METHODS)),
                    bam_batch_sizes=_bam_batch_sizes(
                        cfg, mod.PAPER_GAUSSIAN_BATCH_SIZES["bam"]
                    ),
                    baseline_batch_size=int(cfg.get("baseline_batch_size", 2)),
                    schedule=str(
                        (cfg.get("bam") or {}).get("schedule", "BD")
                    ),
                    learning_rates=_resolve_learning_rates(cfg, experiment),
                    mu_scale=float(
                        (cfg.get("target") or {}).get("init_mean_scale", 0.1)
                    ),
                    history_points=int(cfg.get("history_points", 120)),
                    save=bool(cfg.get("save", True)),
                    figures=bool(cfg.get("figures", True)),
                )
                if isinstance(cfg.get("grad_budget"), dict):
                    kwargs["grad_budget"] = {
                        int(k): int(v) for k, v in cfg["grad_budget"].items()
                    }
            result = mod.run_gaussian_experiment(
                **_filter_kwargs(mod.run_gaussian_experiment, kwargs)
            )

    elif experiment == "non_gaussian":
        from bam_repro.experiments import exp_non_gaussian as mod

        if quick:
            result = mod.run_non_gaussian(
                quick=True, **_filter_kwargs(mod.run_non_gaussian, common)
            )
        else:
            kwargs = dict(common)
            if cfg:
                kwargs.update(
                    n_runs=int(cfg.get("n_runs", mod.PAPER_NON_GAUSSIAN_N_RUNS)),
                    methods=tuple(cfg.get("methods", mod.PAPER_NON_GAUSSIAN_METHODS)),
                    bam_batch_sizes=_bam_batch_sizes(
                        cfg, mod.PAPER_NON_GAUSSIAN_BATCH_SIZES["bam"]
                    ),
                    baseline_batch_size=int(cfg.get("baseline_batch_size", 5)),
                    schedule=str(
                        (cfg.get("bam") or {}).get(
                            "schedule", mod.PAPER_NON_GAUSSIAN_SCHEDULE
                        )
                    ),
                    learning_rates=_resolve_learning_rates(cfg, experiment),
                    mu_scale=float(
                        (cfg.get("target") or {}).get("init_mean_scale", 0.1)
                    ),
                    dim=int(cfg.get("dim", mod.PAPER_NON_GAUSSIAN_DIM)),
                    history_points=int(cfg.get("history_points", 120)),
                    kl_samples=int(
                        cfg.get("kl_samples", mod.PAPER_NON_GAUSSIAN_KL_SAMPLES)
                    ),
                    save=bool(cfg.get("save", True)),
                    figures=bool(cfg.get("figures", True)),
                )
                if isinstance(cfg.get("grad_budget"), dict):
                    kwargs["grad_budget"] = dict(cfg["grad_budget"])
            result = mod.run_non_gaussian_experiment(
                **_filter_kwargs(mod.run_non_gaussian_experiment, kwargs)
            )

    elif experiment == "posteriordb":
        from bam_repro.experiments import exp_posteriordb as mod

        if quick:
            result = mod.run_posteriordb(
                quick=True, **_filter_kwargs(mod.run_posteriordb, common)
            )
        else:
            kwargs = dict(common)
            if cfg:
                settings = cfg.get("settings") or cfg.get("models") or None
                kwargs.update(
                    n_runs=int(cfg.get("n_runs", mod.PAPER_POSTERIORDB_N_RUNS)),
                    methods=tuple(cfg.get("methods", mod.PAPER_POSTERIORDB_METHODS)),
                    batch_sizes=_bam_batch_sizes(
                        cfg, mod.PAPER_POSTERIORDB_BATCH_SIZES
                    ),
                    schedule=str(
                        (cfg.get("bam") or {}).get(
                            "schedule", mod.PAPER_POSTERIORDB_SCHEDULE
                        )
                    ),
                    learning_rates=_resolve_learning_rates(cfg, experiment),
                    mu_scale=float(
                        (cfg.get("target") or {}).get("init_mean_scale", 0.1)
                    ),
                    history_points=int(cfg.get("history_points", 120)),
                    save=bool(cfg.get("save", True)),
                    figures=bool(cfg.get("figures", True)),
                )
                if settings is not None:
                    kwargs["settings"] = tuple(settings)
                if isinstance(cfg.get("grad_budget"), dict):
                    kwargs["grad_budget"] = dict(cfg["grad_budget"])
                target_cfg = cfg.get("target") or {}
                if target_cfg.get("reference_samples"):
                    kwargs["reference_samples"] = target_cfg["reference_samples"]
            result = mod.run_posteriordb_experiment(
                **_filter_kwargs(mod.run_posteriordb_experiment, kwargs)
            )

    elif experiment == "vae":
        from bam_repro.experiments import exp_vae as mod

        if quick:
            result = mod.run_vae(quick=True, **_filter_kwargs(mod.run_vae, common))
        else:
            kwargs = dict(common)
            vae_cfg = cfg.get("vae") if isinstance(cfg.get("vae"), dict) else {}
            if cfg:
                kwargs.update(
                    n_runs=int(cfg.get("n_runs", mod.PAPER_VAE_N_RUNS)),
                    methods=tuple(cfg.get("methods", mod.PAPER_VAE_METHODS)),
                    batch_sizes=tuple(cfg.get("batch_sizes", mod.PAPER_VAE_BATCH_SIZES)),
                    T=int(cfg.get("T", mod.PAPER_VAE_T)),
                    pilot_T=int(cfg.get("pilot_T", mod.PAPER_VAE_PILOT_T)),
                    pilot=bool(cfg.get("pilot", True)),
                    pilot_seeds=tuple(cfg.get("pilot_seeds", (0,))),
                    mu_scale=float(cfg.get("init_mean_scale", 0.1)),
                    latent_dim=int(vae_cfg.get("latent_dim", 256)),
                    history_points=int(cfg.get("history_points", 120)),
                    image_index=int(cfg.get("image_index", 0)),
                    learning_rates=_resolve_learning_rates(cfg, experiment),
                    lambdas=(cfg.get("bam") or {}).get("lambda"),
                    save=bool(cfg.get("save", True)),
                    figures=bool(cfg.get("figures", True)),
                )
            result = mod.run_vae_experiment(
                **_filter_kwargs(mod.run_vae_experiment, kwargs)
            )

    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(f"unknown experiment {experiment!r}")

    return result, time.time() - t0


def _config_arg(experiment: str, config: Optional[str]) -> Optional[str]:
    """Resolve ``--config`` (a directory or a single file) for ``experiment``."""
    if config is None:
        return None
    if os.path.isdir(config):
        return os.path.join(config, CONFIG_FILES.get(experiment, ""))
    if os.path.exists(config):
        cfg = load_config(experiment, config)
        name = str(cfg.get("experiment", "")).strip()
        if name and name != experiment:
            return None
        return config
    return None


def schedule_sweep(
    dim: int = 10, batch_size: int = 5, n_iter: int = 5
) -> Dict[str, Any]:
    """Evaluate the documented BaM learning-rate schedules.

    Mirrors the ablation described in Appendix E.3 / E.4: ``lambda_t = B D``,
    ``B D / (t + 1)``, ``B D / sqrt(t + 1)`` and ``B / (t + 1)``.  The sweeps
    themselves are out of scope, but this exercises the schedule factory so the
    schedule strings used by the configs are known to resolve.
    """
    out: Dict[str, Any] = {"dim": dim, "batch_size": batch_size, "schedules": {}}
    try:
        from bam_repro.bam.learning_rate import make_schedule, schedule_values
    except Exception as exc:  # pragma: no cover
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    for name in SCHEDULE_NAMES:
        try:
            sched = make_schedule(name, batch_size=batch_size, dim=dim)
            vals = schedule_values(sched, n_iter, batch_size=batch_size, dim=dim)
            out["schedules"][name] = [float(v) for v in vals]
        except Exception as exc:  # pragma: no cover
            out["schedules"][name] = f"{type(exc).__name__}: {exc}"
    return out


def run_all(
    experiments: Sequence[str] = ALL_EXPERIMENTS,
    quick: bool = False,
    outdir: Optional[str] = None,
    config: Optional[str] = None,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the requested experiments sequentially and return a summary dict.

    Parameters
    ----------
    experiments:
        Subset of ``("gaussian", "non_gaussian", "posteriordb", "vae")``.
    quick:
        Run each experiment's cheap smoke test instead of the full sweep.
    outdir:
        Directory for results/figures (per-experiment defaults apply when None).
    config:
        Optional YAML config file, or a directory containing the config files.
    continue_on_error:
        Keep going with the remaining experiments when one fails.
    """
    names = tuple(experiments)
    unknown = [n for n in names if n not in ALL_EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown experiments: {unknown}")

    summary: Dict[str, Any] = {
        "quick": bool(quick),
        "outdir": outdir,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {},
    }

    for name in names:
        print("=" * 78)
        print(f"[run_all] {name}" + (" (quick)" if quick else ""))
        print("=" * 78)
        try:
            result, elapsed = _dispatch(
                name, quick, outdir, _config_arg(name, config), verbose
            )
        except Exception as exc:  # a missing optional dep must not kill the run
            traceback.print_exc()
            summary["experiments"][name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if not continue_on_error:
                raise
            continue

        entry: Dict[str, Any] = {"status": "ok", "wallclock_seconds": elapsed}
        if result is not None:
            try:
                if hasattr(result, "table"):
                    print(result.table())
                if hasattr(result, "to_dict"):
                    entry["summary"] = result.to_dict(include_runs=False)
                elif isinstance(result, dict):
                    entry["summary"] = result
            except Exception as exc:  # pragma: no cover
                entry["summary_error"] = f"{type(exc).__name__}: {exc}"
        summary["experiments"][name] = entry

    summary["schedule_sweep"] = schedule_sweep()
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, "run_all_summary.json")
            with open(path, "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
            print(f"[run_all] wrote {path}")
        except Exception as exc:  # pragma: no cover
            print(f"[run_all] could not write summary: {exc}")

    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_all",
        description=(
            "Reproduce the experiments of 'Batch and Match: Black-Box "
            "Variational Inference with a Score-Based Divergence'."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments", "-e", nargs="+", default=["all"],
        choices=list(ALL_EXPERIMENTS) + ["all"],
        help="experiments to run",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="YAML config file, or directory of configs (default: bam_repro/configs)",
    )
    parser.add_argument(
        "--outdir", "-o", default=None,
        help="directory for results and figures",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="run cheap smoke-test versions of the experiments",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="abort at the first failing experiment",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    experiments: Iterable[str] = args.experiments
    if "all" in experiments:
        experiments = ALL_EXPERIMENTS

    available = _available_experiments()
    if available:
        print(f"[run_all] importable experiments: {sorted(available)}")

    summary = run_all(
        experiments=tuple(experiments),
        quick=bool(args.quick),
        outdir=args.outdir,
        config=args.config,
        continue_on_error=not args.stop_on_error,
    )
    failed = [
        name
        for name, entry in summary["experiments"].items()
        if entry.get("status") != "ok"
    ]
    if failed:
        print(f"[run_all] experiments with errors: {failed}")
        return 1
    print("[run_all] done")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
