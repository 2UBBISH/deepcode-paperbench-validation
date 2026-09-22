"""Benchmark experiments for SNPSE / NPSE / TSNPSE (Section 5.2 of the paper).

Runs the non-sequential (NPSE) and sequential (TSNPSE) posterior score estimators on
the eight ``sbibm`` benchmark tasks (Lueckmann et al., 2021; Appendix E.1) with
simulation budgets of 1000, 10000 and 100000, using both the VE SDE and the VP SDE
for the forward noising process (Appendix E.3.1).  Posterior quality is reported with
the classification-based two-sample test (C2ST; 0.5 = perfect, 1.0 = worst).

Optionally, the paper's baselines (NPE and SNPE-C via ``sbibm``, TSNPE via the
mackelab reference implementation) are run with the same budgets.

Usage
-----
::

    python -m snpse.experiments.run_benchmarks --tasks slcp,two_moons \
        --methods npse,tsnpse --sdes ve,vp --budgets 1000,10000 \
        --output results/benchmarks.json

The script is intentionally defensive: every method/task combination is executed
inside a ``try``/``except`` block so a single failing configuration does not abort a
long benchmark sweep, and partial results are flushed to disk after each run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------------------
# Robust module resolution (supports both ``snpse.snpse`` and flat layouts).
# --------------------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)  # .../snpse
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _import_first(*candidates: str):
    """Import the first importable module from ``candidates``."""
    last_err: Optional[BaseException] = None
    for name in candidates:
        try:
            return __import__(name, fromlist=["*"])
        except Exception as exc:  # pragma: no cover - environment dependent
            last_err = exc
    raise ImportError(f"Could not import any of {candidates!r}: {last_err}")


_MODS: Dict[str, Any] = {}


def _module(key: str):
    """Lazily resolve a project module by logical name."""
    if key in _MODS:
        return _MODS[key]
    mapping = {
        "npse": (".npse", "snpse.npse", "snpse.snpse.npse", "npse"),
        "tsnpse": (".tsnpse", "snpse.tsnpse", "snpse.snpse.tsnpse", "tsnpse"),
        "sampler": (".sampler", "snpse.sampler", "snpse.snpse.sampler", "sampler"),
        "utils": (".utils", "snpse.utils", "snpse.snpse.utils", "utils"),
        "benchmarks": (".benchmarks", "snpse.tasks.benchmarks", "tasks.benchmarks", "benchmarks"),
        "c2st": (".c2st", "snpse.tasks.c2st", "tasks.c2st", "c2st"),
        "baselines": (".baselines", "snpse.experiments.baselines",
                      "experiments.baselines", "baselines"),
        "trainer": (".trainer", "snpse.trainer", "snpse.snpse.trainer", "trainer"),
    }
    candidates = mapping[key]
    mod = None
    if candidates[0].startswith("."):
        try:
            mod = _import_first(candidates[1], candidates[2], candidates[3])
        except ImportError:
            mod = None
    if mod is None:
        mod = _import_first(*[c for c in candidates if not c.startswith(".")])
    _MODS[key] = mod
    return mod


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEFAULT_TASKS: Tuple[str, ...] = (
    "gaussian_linear",
    "gaussian_mixture",
    "two_moons",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "slcp",
    "sir",
    "lotka_volterra",
)

DEFAULT_BUDGETS: Tuple[int, ...] = (1_000, 10_000, 100_000)
DEFAULT_METHODS: Tuple[str, ...] = ("npse", "tsnpse")
DEFAULT_SDES: Tuple[str, ...] = ("ve", "vp")

#: Number of posterior samples drawn for the C2ST evaluation.
DEFAULT_NUM_SAMPLES = 10_000

#: Sequential rounds (TSNPSE).  ``num_rounds`` should divide the budget.
DEFAULT_NUM_ROUNDS = 10


@dataclass
class BenchmarkRunConfig:
    """Configuration for a single (task, method, budget, sde) benchmark cell."""

    task: str = "slcp"
    method: str = "npse"
    budget: int = 1_000
    sde: str = "ve"
    num_samples: int = DEFAULT_NUM_SAMPLES
    num_rounds: int = DEFAULT_NUM_ROUNDS
    observation_index: int = 1
    seed: int = 0
    max_iters: int = 3_000
    hidden_dim: int = 256
    n_layers: int = 3
    device: Optional[str] = None
    standardise: bool = True
    c2st_folds: int = 10
    reference_samples: Optional[int] = None
    num_reference_samples: Optional[int] = None  # alias
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved_batch_size(self) -> int:
        """Paper batch size for the simulation budget (Section E.3.2)."""
        try:
            trainer = _module("trainer")
            return int(trainer.select_batch_size(self.budget))
        except Exception:
            table = {1_000: 50, 10_000: 200, 100_000: 500}
            key = min(table, key=lambda b: abs(b - self.budget))
            return table[key]

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out.pop("extra", None)
        return out


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _device(arg: Optional[str]) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: Optional[int]) -> Optional[torch.Generator]:
    try:
        utils = _module("utils")
        return utils.set_seed(seed)
    except Exception:
        if seed is None:
            return None
        torch.manual_seed(seed)
        return torch.Generator().manual_seed(seed)


def _select_sigma_min(dim: int) -> float:
    """Appendix E.3.1: sigma_min = 0.01 for 2-D tasks, else 0.05."""
    return 0.01 if int(dim) == 2 else 0.05


def _task_prior(task: Any) -> Any:
    """Return the prior object of a ``BenchmarkTask``-like object."""
    for name in ("prior", "get_prior"):
        attr = getattr(task, name, None)
        if attr is None:
            continue
        return attr() if callable(attr) and name == "get_prior" else attr
    return None


def _prior_sampler(task: Any):
    sample_fn = getattr(task, "sample_prior", None)
    if callable(sample_fn):
        return sample_fn
    prior = _task_prior(task)
    if prior is not None and hasattr(prior, "sample"):
        return prior.sample
    return None


def _prior_log_prob(task: Any):
    fn = getattr(task, "prior_log_prob", None)
    if callable(fn):
        return fn
    prior = _task_prior(task)
    if prior is not None and hasattr(prior, "log_prob"):
        return prior.log_prob
    return None


def _simulator(task: Any):
    sim = getattr(task, "simulate", None)
    if callable(sim):
        return sim
    return None


def _observation(task: Any) -> torch.Tensor:
    for name in ("x_obs", "observation"):
        value = getattr(task, name, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if isinstance(value, torch.Tensor):
            return value.clone()
        return torch.as_tensor(value, dtype=torch.float32)
    raise AttributeError("Task exposes neither 'x_obs' nor 'observation'.")


def _load_dataset(task_name: str, budget: int, seed: int, task: Any) -> Any:
    """Load/simulate ``budget`` prior-predictive pairs for a task."""
    benchmarks = _module("benchmarks")
    try:
        return benchmarks.load_dataset(
            task if not isinstance(task, str) else task_name,
            budget,
            seed=seed,
            verbose=False,
        )
    except TypeError:
        return benchmarks.load_dataset(task_name, budget, seed=seed)


def _split_dataset(dataset: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, dict):
        theta = dataset.get("theta")
        x = dataset.get("x")
    elif isinstance(dataset, (tuple, list)) and len(dataset) >= 2:
        theta, x = dataset[0], dataset[1]
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported dataset container: {type(dataset)!r}")
    return torch.as_tensor(theta, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32)


def _reference_posterior_samples(
    task: Any,
    n: int,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
):
    """Reference posterior samples from this repo's tasks or ``sbibm``."""
    c2st_mod = _module("c2st")
    try:
        return c2st_mod.reference_posterior(task, num_samples=n, seed=seed)
    except Exception:
        pass
    sampler_fn = getattr(task, "reference_posterior_samples", None)
    if callable(sampler_fn):
        try:
            return sampler_fn(n, generator=generator)
        except TypeError:
            return sampler_fn(n)
    raise AttributeError("No reference posterior available for this task.")


def _evaluate_c2st(
    task: Any,
    samples: torch.Tensor,
    config: BenchmarkRunConfig,
    reference_samples=None,
) -> Dict[str, Any]:
    """C2ST of approximate posterior samples against the reference posterior."""
    c2st_mod = _module("c2st")
    num_ref = config.reference_samples or config.num_reference_samples
    if num_ref is None:
        num_ref = int(samples.shape[0])
    if reference_samples is None:
        reference_samples = _reference_posterior_samples(
            task, num_ref, seed=config.seed
        )
    try:
        result = c2st_mod.task_c2st(
            task,
            samples,
            num_reference_samples=num_ref,
            seed=config.seed,
            n_folds=config.c2st_folds,
            return_result=True,
            reference_samples=reference_samples,
        )
    except TypeError:
        result = c2st_mod.task_c2st(
            task,
            samples,
            num_reference_samples=num_ref,
            seed=config.seed,
            n_folds=config.c2st_folds,
            return_result=True,
        )
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {"c2st": float(result)}


# --------------------------------------------------------------------------------------
# Method runners
# --------------------------------------------------------------------------------------


def run_npse_cell(
    task: Any,
    task_name: str,
    dataset: Any,
    config: BenchmarkRunConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Run the amortised NPSE method for one benchmark cell."""
    npse = _module("npse")
    theta, x = _split_dataset(dataset)
    x_obs = _observation(task)

    device = _device(config.device)
    theta_dim = int(theta.shape[-1])
    x_dim = int(x.shape[-1])

    cfg_kwargs: Dict[str, Any] = dict(
        sde=config.sde,
        sigma_min=_select_sigma_min(theta_dim),
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        max_iters=config.max_iters,
        budget=int(config.budget),
        batch_size=config.resolved_batch_size(),
        standardise=config.standardise,
        seed=config.seed,
        device=str(device),
        verbose=config.verbose,
    )
    cfg_kwargs.update(config.extra.get("npse", {}))

    model = npse.NPSE(theta_dim, x_dim, config=npse.NPSEConfig(**cfg_kwargs), device=device)
    model.fit(theta, x, generator=generator)
    theta_post = model.sample(x_obs, config.num_samples, generator=generator)
    return {
        "theta": theta_post.detach().cpu(),
        "model": model,
        "info": {"budget": int(config.budget), "sde": config.sde},
    }


def run_tsnpse_cell(
    task: Any,
    task_name: str,
    dataset: Any,
    config: BenchmarkRunConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Run the sequential TSNPSE method for one benchmark cell.

    The budget is split evenly over ``config.num_rounds`` rounds (Section 3.1).
    """
    tsnpse = _module("tsnpse")
    theta, x = _split_dataset(dataset)
    x_obs = _observation(task)

    device = _device(config.device)
    theta_dim = int(theta.shape[-1])
    x_dim = int(x.shape[-1])

    # Round-1 data from the (cached) prior-predictive dataset; remaining rounds are
    # simulated on the fly from the truncated proposal inside ``run_tsnpse``.
    num_rounds = max(1, int(config.num_rounds))
    initial_budget = max(1, int(config.budget) // num_rounds)
    per_round = initial_budget

    cfg_kwargs: Dict[str, Any] = dict(
        sde=config.sde,
        sigma_min=_select_sigma_min(theta_dim),
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        max_iters=config.max_iters,
        budget=int(config.budget),
        batch_size=config.resolved_batch_size(),
        standardise=config.standardise,
        seed=config.seed,
        device=str(device),
        verbose=config.verbose,
        num_rounds=num_rounds,
        initial_budget=initial_budget,
        simulations_per_round=per_round,
        num_samples=int(config.num_samples),
    )
    cfg_kwargs.update(config.extra.get("tsnpse", {}))

    prior = _task_prior(task)
    simulator = _simulator(task)
    prior_sample_fn = _prior_sampler(task)
    prior_log_prob_fn = _prior_log_prob(task)

    result = tsnpse.run_tsnpse(
        prior=prior,
        simulator=simulator,
        x_obs=x_obs,
        theta_dim=theta_dim,
        x_dim=x_dim,
        config=tsnpse.TSNPSEConfig(**cfg_kwargs),
        num_samples=config.num_samples,
        seed=config.seed,
        device=device,
        prior_sample_fn=prior_sample_fn,
        prior_log_prob_fn=prior_log_prob_fn,
        initial_theta=theta[:initial_budget],
        initial_x=x[:initial_budget],
    )
    theta_post = result["theta"]
    if isinstance(theta_post, torch.Tensor):
        theta_post = theta_post.detach().cpu()
    return {
        "theta": theta_post,
        "model": result.get("network"),
        "info": {
            "budget": int(config.budget),
            "sde": config.sde,
            "num_rounds": num_rounds,
            "initial_budget": initial_budget,
            "simulations_per_round": per_round,
        },
        "dataset": result.get("dataset"),
    }


def run_baseline_cell(
    task: Any,
    task_name: str,
    config: BenchmarkRunConfig,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Run one of the paper's baselines (NPE / SNPE-C / TSNPE / FMPE)."""
    baselines = _module("baselines")
    return baselines.run_baseline(
        config.method,
        task,
        config=baselines.BaselineConfig(
            method=config.method,
            budget=int(config.budget),
            num_rounds=int(config.num_rounds),
            num_samples=int(config.num_samples),
            seed=config.seed,
            backend=config.extra.get("backend", "auto"),
            verbose=config.verbose,
        ),
        device=config.device,
        generator=generator,
    )


METHOD_RUNNERS = {
    "npse": run_npse_cell,
    "tsnpse": run_tsnpse_cell,
    "npe": run_baseline_cell,
    "snpe_c": run_baseline_cell,
    "snpe-c": run_baseline_cell,
    "tsnpe": run_baseline_cell,
    "fmpe": run_baseline_cell,
}


# --------------------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------------------


def run_single(
    config: BenchmarkRunConfig,
    dataset: Any = None,
    reference_samples=None,
) -> Dict[str, Any]:
    """Run one (task, method, budget, sde) cell and return its result record."""
    key = f"{config.method}/{config.task}/{config.sde}/{config.budget}"
    record: Dict[str, Any] = {
        "key": key,
        "task": config.task,
        "method": config.method,
        "budget": int(config.budget),
        "sde": config.sde,
        "seed": int(config.seed),
        "status": "pending",
        "c2st": None,
        "config": config.as_dict(),
    }

    t0 = time.time()
    generator = _set_seed(config.seed)
    try:
        benchmarks = _module("benchmarks")
        task = benchmarks.get_task(config.task, observation_index=config.observation_index)

        if dataset is None:
            dataset = _load_dataset(config.task, int(config.budget), config.seed, task)

        runner = METHOD_RUNNERS.get(config.method)
        if runner is None:
            raise ValueError(
                f"Unknown method {config.method!r}; expected one of "
                f"{sorted(set(METHOD_RUNNERS))}"
            )

        if runner is run_baseline_cell:
            out = runner(task, config.task, config, generator=generator)
        else:
            out = runner(task, config.task, dataset, config, generator=generator)

        samples = out.get("theta")
        if samples is None:  # pragma: no cover - defensive
            raise RuntimeError("Method did not return posterior samples.")

        metrics = _evaluate_c2st(task, samples, config, reference_samples=reference_samples)
        record.update(
            {
                "status": "ok",
                "c2st": float(metrics.get("c2st", float("nan"))),
                "c2st_std": float(metrics.get("c2st_std", float("nan"))),
                "c2st_folds": metrics.get("scores"),
                "backend": metrics.get("backend"),
                "n_samples": int(samples.shape[0]),
                "runtime": time.time() - t0,
                "info": out.get("info", {}),
            }
        )
        try:
            del out["model"]
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001 - keep the sweep alive
        record.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
                "runtime": time.time() - t0,
            }
        )
        if config.verbose:
            print(record["traceback"], flush=True)
    return record


def run_sweep(
    tasks: Sequence[str] = DEFAULT_TASKS,
    methods: Sequence[str] = DEFAULT_METHODS,
    sdes: Sequence[str] = DEFAULT_SDES,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seed: int = 0,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    max_iters: int = 3_000,
    device: Optional[str] = None,
    output: Optional[str] = None,
    standardise: bool = True,
    c2st_folds: int = 10,
    observation_index: int = 1,
    extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full benchmark sweep, returning a results dictionary."""
    extra = extra or {}
    results: List[Dict[str, Any]] = []

    combinations: List[BenchmarkRunConfig] = []
    for method in methods:
        for task in tasks:
            method_sdes = [None] if method in ("npe", "snpe_c", "snpe-c") else list(sdes)
            for sde in method_sdes:
                for budget in budgets:
                    combinations.append(
                        BenchmarkRunConfig(
                            task=task,
                            method=method,
                            budget=int(budget),
                            sde=sde or "ve",
                            num_samples=int(num_samples),
                            num_rounds=int(num_rounds),
                            observation_index=int(observation_index),
                            seed=int(seed),
                            max_iters=int(max_iters),
                            device=device,
                            standardise=standardise,
                            c2st_folds=int(c2st_folds),
                            verbose=verbose,
                            extra=extra,
                        )
                    )

    total = len(combinations)
    if verbose:
        print(f"[run_benchmarks] {total} configurations", flush=True)

    cache: Dict[Tuple[str, int, int], Any] = {}
    references: Dict[Tuple[str, int], Any] = {}

    for i, cfg in enumerate(combinations, start=1):
        if verbose:
            print(
                f"[{i}/{total}] {cfg.method} | {cfg.task} | sde={cfg.sde} | "
                f"budget={cfg.budget}",
                flush=True,
            )
        key = (cfg.task, int(cfg.budget), int(cfg.seed))
        if key not in cache:
            try:
                benchmarks = _module("benchmarks")
                task_obj = benchmarks.get_task(cfg.task, observation_index=cfg.observation_index)
                cache[key] = _load_dataset(cfg.task, int(cfg.budget), cfg.seed, task_obj)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  dataset error: {type(exc).__name__}: {exc}", flush=True)
                cache[key] = None

        ref_key = (cfg.task, cfg.seed)
        if ref_key not in references:
            try:
                benchmarks = _module("benchmarks")
                task_obj = benchmarks.get_task(cfg.task, observation_index=cfg.observation_index)
                references[ref_key] = _reference_posterior_samples(
                    task_obj, int(num_samples), seed=cfg.seed
                )
            except Exception:
                references[ref_key] = None

        record = run_single(
            cfg,
            dataset=cache.get(key),
            reference_samples=references.get(ref_key),
        )
        results.append(record)
        if verbose:
            if record["status"] == "ok":
                print(f"  -> C2ST = {record['c2st']:.4f}", flush=True)
            else:
                print(f"  -> FAILED ({record.get('error')})", flush=True)

        if output:
            try:
                save_results(results, output, config={"seed": seed, "num_samples": num_samples})
            except Exception as exc:  # noqa: BLE001 - non-fatal
                if verbose:
                    print(f"  (could not write {output}: {exc})", flush=True)

    summary = {"results": results, "config": {"seed": seed, "num_samples": num_samples}}
    if output:
        save_results(results, output, config=summary["config"])
    if verbose:
        print(format_table(results), flush=True)
    return summary


# --------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------


def save_results(results: Sequence[Dict[str, Any]], path: str, config: Optional[Dict] = None) -> str:
    """Write results to JSON (creating parent directories)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"config": config or {}, "results": list(results)}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def load_results(path: str) -> List[Dict[str, Any]]:
    """Read a results file written by :func:`save_results`."""
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload.get("results", [])
    return payload


def aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate mean C2ST per (method, task, sde) over budgets."""
    buckets: Dict[str, List[float]] = {}
    for rec in results:
        if rec.get("status") != "ok" or rec.get("c2st") is None:
            continue
        key = f"{rec['method']}|{rec['task']}|{rec.get('sde')}|{rec['budget']}"
        buckets.setdefault(key, []).append(float(rec["c2st"]))
    out: Dict[str, Dict[str, float]] = {}
    for key, values in buckets.items():
        out[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    return out


def format_table(results: Sequence[Dict[str, Any]]) -> str:
    """Render a compact fixed-width table of the sweep results."""
    header = f"{'task':<24}{'method':<10}{'sde':<5}{'budget':>8}{'c2st':>10}{'status':>10}"
    lines = [header, "-" * len(header)]
    for rec in sorted(
        results,
        key=lambda r: (str(r.get("task")), str(r.get("method")), str(r.get("sde")), r.get("budget", 0)),
    ):
        c2st = rec.get("c2st")
        c2st_str = f"{c2st:.4f}" if isinstance(c2st, (int, float)) else "-"
        lines.append(
            f"{str(rec.get('task')):<24}{str(rec.get('method')):<10}"
            f"{str(rec.get('sde')):<5}{int(rec.get('budget', 0)):>8}"
            f"{c2st_str:>10}{str(rec.get('status')):>10}"
        )
    return "\n".join(lines)


def plot_results(results: Sequence[Dict[str, Any]], path: str) -> Optional[str]:
    """Reproduce the style of Figures 2/3 (C2ST vs. budget) if matplotlib is present."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None

    tasks = sorted({r["task"] for r in results if r.get("status") == "ok"})
    if not tasks:
        return None
    ncols = min(4, len(tasks))
    nrows = int(math.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for ax, task in zip(axes.ravel(), tasks):
        for method in sorted({r["method"] for r in results if r["task"] == task}):
            for sde in sorted({str(r.get("sde")) for r in results if r["task"] == task and r["method"] == method}):
                pts = sorted(
                    (
                        (int(r["budget"]), float(r["c2st"]))
                        for r in results
                        if r["task"] == task
                        and r["method"] == method
                        and str(r.get("sde")) == sde
                        and r.get("status") == "ok"
                        and isinstance(r.get("c2st"), (int, float))
                    ),
                    key=lambda p: p[0],
                )
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker="o", label=f"{method}-{sde}")
        ax.set_xscale("log")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
        ax.set_title(task)
        ax.set_xlabel("simulations")
        ax.set_ylabel("C2ST")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_list(value: str, cast=str) -> List[Any]:
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SNPSE benchmark experiments (paper Section 5.2)."
    )
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--sdes", type=str, default=",".join(DEFAULT_SDES))
    parser.add_argument("--budgets", type=str, default=",".join(str(b) for b in DEFAULT_BUDGETS))
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max-iters", type=int, default=3_000)
    parser.add_argument("--c2st-folds", type=int, default=10)
    parser.add_argument("--observation-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/benchmarks.json")
    parser.add_argument("--figure", type=str, default=None)
    parser.add_argument("--no-standardise", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_sweep(
        tasks=_parse_list(args.tasks),
        methods=_parse_list(args.methods),
        sdes=_parse_list(args.sdes),
        budgets=_parse_list(args.budgets, int),
        seed=args.seed,
        num_samples=args.num_samples,
        num_rounds=args.num_rounds,
        max_iters=args.max_iters,
        device=args.device,
        output=args.output,
        standardise=not args.no_standardise,
        c2st_folds=args.c2st_folds,
        observation_index=args.observation_index,
        verbose=not args.quiet,
    )
    if args.figure:
        plot_results(summary["results"], args.figure)
    n_ok = sum(1 for r in summary["results"] if r.get("status") == "ok")
    print(f"[run_benchmarks] {n_ok}/{len(summary['results'])} runs succeeded")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
