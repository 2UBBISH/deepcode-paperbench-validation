#!/usr/bin/env python3
"""Appendix / Section 5.4 ablations for Sequential Neural Posterior Score Estimation.

This driver reproduces the paper's ablation studies:

* **Alternative sequential variants** (Sections 3.2, C.2--C.4): SNPSE-A
  (post-hoc SIR correction, eq. 12--14), SNPSE-B (importance-weighted DSM loss,
  eq. 15/99) and SNPSE-C (score-space correction, eq. 19/103/123) compared with
  TSNPSE (Algorithm 1) on SLCP and Gaussian Linear Uniform.  The paper reports
  TSNPSE clearly better than all of them, with SNPSE-C failing badly
  (C2ST close to 1).
* **NPSE vs NLSE** (Appendix A.1/B, Figure 5): Neural Likelihood Score
  Estimation should match NPSE when the perturbed prior score is analytic, and
  be worse when the prior score has to be learned with Algorithm 2.
* **VE vs VP SDE sweep** (Appendix E.3.1): the recommended dimensionality split
  (VE for low-dimensional tasks, VP for higher-dimensional ones) is reproduced by
  sweeping both forward SDEs over the eight sbibm benchmark tasks.

All posterior approximations are scored with the paper's metric, C2ST
(0.5 = perfect, 1.0 = worst), computed with sbibm's default settings.

Every cell is wrapped in ``try/except`` so a single failing configuration never
aborts a sweep; results are flushed to JSON after every run.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Module resolution (works both as ``python -m snpse.experiments.run_ablations``
# and as ``python experiments/run_ablations.py``)
# ---------------------------------------------------------------------------

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)  # .../snpse


def _ensure_path() -> None:
    for candidate in (_project_root(), os.path.dirname(_project_root())):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)


_MODULES: Dict[str, Any] = {}
_MISSING: set = set()


def _import_first(names: Sequence[str]) -> Optional[Any]:
    _ensure_path()
    for name in names:
        if name in _MODULES:
            return _MODULES[name]
        if name in _MISSING:
            continue
        try:
            module = __import__(name, fromlist=["*"])
        except Exception:
            _MISSING.add(name)
            continue
        _MODULES[name] = module
        return module
    return None


def _module(name: str) -> Optional[Any]:
    """Resolve a project module across the nested/flat layouts."""
    if name in _MODULES:
        return _MODULES[name]
    if name.startswith("snpse."):
        leaf = name.split(".")[-1]
    else:
        leaf = name
    return _import_first(
        [
            name,
            f"snpse.{name}",
            f"snpse.snpse.{leaf}",
            f"snpse.experiments.{leaf}",
            leaf,
            name.replace(".", "_"),
        ]
    )


def _call_filtered(fn: Callable, *args, **kwargs) -> Any:
    """Call ``fn`` dropping keyword arguments it does not accept."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **accepted)


def _attribute(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            break
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


# ---------------------------------------------------------------------------
# Task / dataset helpers
# ---------------------------------------------------------------------------

DEFAULT_VARIANT_TASKS = ("slcp", "gaussian_linear_uniform")
DEFAULT_SDE_TASKS = (
    "gaussian_linear",
    "gaussian_mixture",
    "two_moons",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "slcp",
    "sir",
    "lotka_volterra",
)
ABLATION_VARIANTS = ("snpse_a", "snpse_b", "snpse_c", "tsnpse", "npse")
NLSE_METHODS = ("nlse", "npse")
SDES = ("ve", "vp")

_DATASETS: Dict[str, Any] = {}
_REFERENCES: Dict[str, Any] = {}


def get_task(task_name: str, observation_index: int = 1):
    """Resolve a benchmark task object (sbibm-backed when available)."""
    benchmarks = _module("tasks.benchmarks")
    if benchmarks is None:
        raise ImportError(
            "could not import snpse.tasks.benchmarks — run from the project root"
        )
    return benchmarks.get_task(task_name, observation_index=observation_index)


def load_dataset(task, task_name: str, budget: int, seed: int = 0):
    """Prior-predictive dataset for ``task`` with caching."""
    key = f"{task_name}|{budget}|{seed}"
    if key in _DATASETS:
        return _DATASETS[key]
    benchmarks = _module("tasks.benchmarks")
    if benchmarks is not None and hasattr(benchmarks, "load_dataset"):
        dataset = benchmarks.load_dataset(
            task, int(budget), seed=int(seed), verbose=False
        )
    else:  # pragma: no cover - benchmarks always provides this
        theta = task.sample_prior(int(budget))
        x = task.simulate(theta)
        dataset = {"theta": theta, "x": x}
    _DATASETS[key] = dataset
    return dataset


def split_dataset(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(theta, x)`` from whatever container ``load_dataset`` returned."""
    if isinstance(dataset, dict):
        theta = _attribute(dataset, "theta", "parameters", "thetas")
        x = _attribute(dataset, "x", "data", "observations")
        if theta is not None and x is not None:
            return torch.as_tensor(theta), torch.as_tensor(x)
    if isinstance(dataset, (tuple, list)) and len(dataset) >= 2:
        return torch.as_tensor(dataset[0]), torch.as_tensor(dataset[1])
    theta = _attribute(dataset, "theta", default=None)
    x = _attribute(dataset, "x", default=None)
    if theta is None or x is None:
        raise ValueError("could not extract (theta, x) from dataset")
    return torch.as_tensor(theta), torch.as_tensor(x)


def observation(task) -> torch.Tensor:
    x_obs = _attribute(task, "x_obs", "observation", "obs", default=None)
    if callable(x_obs):
        x_obs = x_obs()
    if x_obs is None and hasattr(task, "observation"):
        x_obs = task.observation()
    if x_obs is None:
        raise ValueError("task provides no observation")
    return torch.as_tensor(x_obs)


def prior_sampler(task) -> Callable:
    prior = _attribute(task, "prior", default=None)
    for candidate in (getattr(prior, "sample", None), getattr(task, "sample_prior", None)):
        if callable(candidate):
            return candidate
    raise ValueError("task provides no prior sampler")


def prior_log_prob(task) -> Optional[Callable]:
    prior = _attribute(task, "prior", default=None)
    candidate = getattr(prior, "log_prob", None)
    if callable(candidate):
        return candidate
    candidate = getattr(task, "prior_log_prob", None)
    if callable(candidate):
        return candidate
    return None


def simulator_fn(task) -> Callable:
    sim = _attribute(task, "simulator", "simulate", default=None)
    if callable(sim):
        return sim
    if hasattr(task, "simulate"):
        return task.simulate
    raise ValueError("task provides no simulator")


def task_dim_theta(task) -> int:
    value = _attribute(task, "dim_parameters", "dim_theta", default=None)
    if value is None:
        params = _attribute(task, "parameters", default=None)
        value = len(params) if params is not None else None
    if value is None:
        raise ValueError("could not determine parameter dimension")
    return int(value)


def task_dim_x(task) -> int:
    value = _attribute(task, "dim_data", "dim_x", default=None)
    if value is None:
        x_obs = observation(task)
        value = int(torch.as_tensor(x_obs).numel())
    return int(value)


def reference_samples(task, task_name: str, num_samples: int, seed: int = 0):
    key = f"{task_name}|{num_samples}|{seed}"
    if key in _REFERENCES:
        return _REFERENCES[key]
    c2st_mod = _module("tasks.c2st")
    samples = None
    if c2st_mod is not None and hasattr(c2st_mod, "reference_posterior"):
        try:
            samples = _call_filtered(
                c2st_mod.reference_posterior,
                task,
                num_samples=int(num_samples),
                seed=int(seed),
            )
        except Exception:
            samples = None
    if samples is None and hasattr(task, "reference_posterior_samples"):
        samples = task.reference_posterior_samples(int(num_samples))
    if samples is not None:
        _REFERENCES[key] = samples
    return samples


def evaluate_cell(task, task_name: str, theta, config, seed: int = 0) -> Dict[str, Any]:
    """C2ST against the task's reference posterior (sbibm defaults)."""
    c2st_mod = _module("tasks.c2st")
    if c2st_mod is None:
        raise ImportError("could not import snpse.tasks.c2st")
    num_reference = _attribute(config, "num_reference_samples", default=None)
    if num_reference is None:
        reference = reference_samples(
            task, task_name, int(torch.as_tensor(theta).shape[0]), seed=seed
        )
    else:
        reference = reference_samples(
            task, task_name, int(num_reference), seed=seed
        )
    result = _call_filtered(
        c2st_mod.task_c2st,
        task,
        theta,
        num_reference_samples=num_reference,
        seed=int(seed),
        n_folds=int(_attribute(config, "c2st_folds", default=10) or 10),
        return_result=True,
    )
    out: Dict[str, Any] = {}
    if isinstance(result, dict):
        out.update({k: v for k, v in result.items() if isinstance(v, (int, float, str))})
    elif hasattr(result, "to_dict"):
        out.update(
            {k: v for k, v in result.to_dict().items() if isinstance(v, (int, float, str))}
        )
    else:
        out["c2st"] = float(result)
    if reference is not None and "c2st" not in out:
        out["c2st"] = float(
            _call_filtered(c2st_mod.c2st, theta, reference, seed=int(seed))
        )
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationRunConfig:
    """Configuration for a single ablation cell (one method x task x budget x SDE)."""

    section: str = "variants"
    task: str = "slcp"
    method: str = "snpse_a"
    budget: int = 10000
    sde: str = "ve"
    num_rounds: int = 10
    num_samples: int = 10000
    observation_index: int = 1
    seed: int = 0
    max_iters: int = 3000
    hidden_dim: int = 256
    n_layers: int = 3
    batch_size: Optional[int] = None
    eps: float = 5e-4
    n_hpr_samples: int = 20000
    standardise: bool = True
    c2st_folds: int = 10
    num_reference_samples: Optional[int] = None
    device: Optional[str] = None
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved_batch_size(self) -> int:
        if self.batch_size is not None:
            return int(self.batch_size)
        trainer = _module("trainer")
        if trainer is not None and hasattr(trainer, "select_batch_size"):
            try:
                return int(trainer.select_batch_size(int(self.budget)))
            except Exception:
                pass
        if self.budget <= 1000:
            return 50
        if self.budget <= 10000:
            return 200
        return 500

    def as_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data.pop("extra", None)
        data["batch_size_resolved"] = self.resolved_batch_size()
        return data


# ---------------------------------------------------------------------------
# Single-cell runners
# ---------------------------------------------------------------------------

def _make_generator(seed: int):
    utils = _module("utils")
    if utils is not None and hasattr(utils, "set_seed"):
        try:
            gen = utils.set_seed(int(seed))
            if gen is not None:
                return gen
        except Exception:
            pass
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _build_prior_kwargs(task):
    kwargs: Dict[str, Any] = {}
    sampler = prior_sampler(task)
    log_prob = prior_log_prob(task)
    prior = _attribute(task, "prior", default=None)
    if prior is not None:
        kwargs["prior"] = prior
    kwargs["prior_sample_fn"] = sampler
    if log_prob is not None:
        kwargs["prior_log_prob_fn"] = log_prob
    return kwargs


def _run_npse_cell(task, task_name, dataset, config: AblationRunConfig, generator=None) -> Dict:
    npse_mod = _module("npse")
    if npse_mod is None:
        raise ImportError("could not import snpse.npse")
    theta, x = split_dataset(dataset)
    theta_dim, x_dim = task_dim_theta(task), task_dim_x(task)
    npse_cfg = _call_filtered(
        npse_mod.NPSEConfig,
        sde=config.sde,
        budget=int(config.budget),
        batch_size=config.batch_size,
        max_iters=int(config.max_iters),
        hidden_dim=int(config.hidden_dim),
        n_layers=int(config.n_layers),
        standardise=bool(config.standardise),
        seed=int(config.seed),
    )
    if hasattr(npse_mod, "NPSE"):
        model = npse_mod.NPSE(theta_dim, x_dim, config=npse_cfg)
        model.fit(theta, x, generator=generator)
        samples = model.sample(
            observation(task), int(config.num_samples), generator=generator
        )
    else:  # pragma: no cover - fallback to functional API
        result = _call_filtered(
            npse_mod.run_npse,
            theta,
            x,
            observation(task),
            num_samples=int(config.num_samples),
            config=npse_cfg,
            seed=int(config.seed),
        )
        model, samples = result.get("model"), result.get("theta")
    if isinstance(samples, tuple):
        samples = samples[0]
    return {"theta": samples, "model": model, "info": {"method": "npse"}}


def _run_tsnpse_cell(task, task_name, dataset, config: AblationRunConfig, generator=None) -> Dict:
    tsnpse_mod = _module("tsnpse")
    if tsnpse_mod is None:
        raise ImportError("could not import snpse.tsnpse")
    theta_dim, x_dim = task_dim_theta(task), task_dim_x(task)
    ts_cfg = _call_filtered(
        tsnpse_mod.TSNPSEConfig,
        sde=config.sde,
        budget=int(config.budget),
        num_rounds=int(config.num_rounds),
        eps=float(config.eps),
        n_hpr_samples=int(config.n_hpr_samples),
        batch_size=config.batch_size,
        max_iters=int(config.max_iters),
        hidden_dim=int(config.hidden_dim),
        n_layers=int(config.n_layers),
        standardise=bool(config.standardise),
        seed=int(config.seed),
    )
    kwargs = _build_prior_kwargs(task)
    result = _call_filtered(
        tsnpse_mod.run_tsnpse,
        kwargs.get("prior"),
        simulator_fn(task),
        observation(task),
        theta_dim,
        x_dim,
        config=ts_cfg,
        num_samples=int(config.num_samples),
        seed=int(config.seed),
        generator=generator,
        device=config.device,
        prior_sample_fn=kwargs.get("prior_sample_fn"),
        prior_log_prob_fn=kwargs.get("prior_log_prob_fn"),
    )
    return {
        "theta": result.get("theta"),
        "model": result.get("network"),
        "info": {"method": "tsnpse", "history": result.get("history")},
    }


def _run_variant_cell(
    task, task_name, dataset, config: AblationRunConfig, generator=None, variant: Optional[str] = None
) -> Dict:
    variant = (variant or config.method).lower()
    variants_mod = _module("snpse_variants")
    if variants_mod is None:
        raise ImportError("could not import snpse.snpse_variants")
    runner_name = {
        "snpse_a": "run_snpse_a",
        "snpse-a": "run_snpse_a",
        "a": "run_snpse_a",
        "snpse_b": "run_snpse_b",
        "snpse-b": "run_snpse_b",
        "b": "run_snpse_b",
        "snpse_c": "run_snpse_c",
        "snpse-c": "run_snpse_c",
        "c": "run_snpse_c",
    }.get(variant)
    if runner_name is None or not hasattr(variants_mod, runner_name):
        raise ValueError(f"unknown SNPSE variant: {variant}")
    runner = getattr(variants_mod, runner_name)
    theta_dim, x_dim = task_dim_theta(task), task_dim_x(task)
    kwargs = _build_prior_kwargs(task)
    variant_cfg = None
    if hasattr(variants_mod, "SNPSEConfig"):
        variant_cfg = _call_filtered(
            variants_mod.SNPSEConfig,
            sde=config.sde,
            budget=int(config.budget),
            num_rounds=int(config.num_rounds),
            batch_size=config.batch_size,
            max_iters=int(config.max_iters),
            hidden_dim=int(config.hidden_dim),
            n_layers=int(config.n_layers),
            standardise=bool(config.standardise),
            seed=int(config.seed),
        )
    extra = dict(config.extra)
    if variant == "snpse_b":
        extra.setdefault("normalise_weights", True)
    if variant == "snpse_c":
        extra.setdefault("use_analytic_prior_score", True)
    result = _call_filtered(
        runner,
        kwargs.get("prior"),
        simulator_fn(task),
        observation(task),
        theta_dim,
        x_dim,
        config=variant_cfg,
        num_samples=int(config.num_samples),
        seed=int(config.seed),
        generator=generator,
        device=config.device,
        sample_with=config.extra.get("sample_with", "base"),
        **extra,
    )
    return {
        "theta": result.get("theta"),
        "model": result.get("network"),
        "info": {"method": variant, "history": result.get("history")},
    }


def _run_nlse_cell(task, task_name, dataset, config: AblationRunConfig, generator=None) -> Dict:
    nlse_mod = _module("nlse")
    if nlse_mod is None:
        raise ImportError("could not import snpse.nlse")
    theta, x = split_dataset(dataset)
    theta_dim, x_dim = task_dim_theta(task), task_dim_x(task)
    kwargs = _build_prior_kwargs(task)
    nlse_cfg = None
    if hasattr(nlse_mod, "NLSEConfig"):
        nlse_cfg = _call_filtered(
            nlse_mod.NLSEConfig,
            sde=config.sde,
            budget=int(config.budget),
            batch_size=config.batch_size,
            max_iters=int(config.max_iters),
            hidden_dim=int(config.hidden_dim),
            n_layers=int(config.n_layers),
            standardise=bool(config.standardise),
            seed=int(config.seed),
        )
    result = _call_filtered(
        nlse_mod.run_nlse,
        theta=theta,
        x=x,
        x_obs=observation(task),
        theta_dim=theta_dim,
        x_dim=x_dim,
        config=nlse_cfg,
        num_samples=int(config.num_samples),
        seed=int(config.seed),
        generator=generator,
        device=config.device,
        simulator=simulator_fn(task),
        prior=kwargs.get("prior"),
        prior_spec=None,
        prior_sample_fn=kwargs.get("prior_sample_fn"),
        prior_log_prob_fn=kwargs.get("prior_log_prob_fn"),
    )
    samples = result.get("theta") if isinstance(result, dict) else result
    if isinstance(samples, tuple):
        samples = samples[0]
    return {"theta": samples, "model": result.get("model") if isinstance(result, dict) else None,
            "info": {"method": "nlse"}}


_CELL_RUNNERS: Dict[str, Callable] = {
    "npse": _run_npse_cell,
    "tsnpse": _run_tsnpse_cell,
    "snpse_a": _run_variant_cell,
    "snpse-a": _run_variant_cell,
    "snpse_b": _run_variant_cell,
    "snpse-b": _run_variant_cell,
    "snpse_c": _run_variant_cell,
    "snpse-c": _run_variant_cell,
    "nlse": _run_nlse_cell,
}


# ---------------------------------------------------------------------------
# Staged runs
# ---------------------------------------------------------------------------

def run_single(
    config: AblationRunConfig,
    dataset=None,
    flush: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run one ablation cell and score it with C2ST."""
    name = (config.method or "").lower()
    runner = _CELL_RUNNERS.get(name)
    record: Dict[str, Any] = {
        "section": config.section,
        "task": config.task,
        "method": config.method,
        "budget": int(config.budget),
        "sde": config.sde,
        "seed": int(config.seed),
        "status": "error",
    }
    if runner is None:
        record["error"] = f"unknown method {config.method!r}"
        if flush is not None:
            flush(record)
        return record

    start = time.time()
    try:
        generator = _make_generator(int(config.seed))
        task = get_task(config.task, observation_index=config.observation_index)
        if dataset is None:
            dataset = load_dataset(task, config.task, config.budget, seed=config.seed)
        if name in ("snpse_a", "snpse-a", "snpse_b", "snpse-b", "snpse_c", "snpse-c"):
            out = runner(task, config.task, dataset, config, generator=generator, variant=name)
        else:
            out = runner(task, config.task, dataset, config, generator=generator)
        theta = out.get("theta")
        if theta is None:
            raise RuntimeError("runner returned no posterior samples")
        theta = torch.as_tensor(theta)
        metrics = evaluate_cell(task, config.task, theta, config, seed=config.seed)
        record.update(metrics)
        record["status"] = "ok"
        record["n_samples"] = int(theta.shape[0])
        record["dim"] = int(theta.shape[-1])
        record["info"] = out.get("info", {})
        if flush is not None:
            record["model"] = None
    except Exception as exc:  # keep sweeping
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=6)
    finally:
        record["runtime"] = time.time() - start
    if flush is not None:
        flush(record)
    return record


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _maybe_flush(records: List[Dict[str, Any]], path: Optional[str], config: Optional[Dict] = None) -> None:
    if path:
        try:
            save_results(records, path, config=config)
        except Exception:
            pass


def run_variant_section(
    tasks: Sequence[str] = DEFAULT_VARIANT_TASKS,
    variants: Sequence[str] = ABLATION_VARIANTS,
    budgets: Sequence[int] = (10000,),
    sdes: Sequence[str] = ("ve",),
    num_rounds: int = 10,
    num_samples: int = 10000,
    seed: int = 0,
    max_iters: int = 3000,
    device: Optional[str] = None,
    c2st_folds: int = 10,
    observation_index: int = 1,
    extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    output: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """SNPSE-A/B/C vs TSNPSE (and NPSE) on the low-dimensional tasks."""
    records: List[Dict[str, Any]] = []
    for task_name in tasks:
        for budget in budgets:
            for sde in sdes:
                dataset = None
                try:
                    task = get_task(task_name, observation_index=observation_index)
                    dataset = load_dataset(task, task_name, int(budget), seed=seed)
                except Exception:
                    dataset = None
                for variant in variants:
                    cfg = AblationRunConfig(
                        section="variants",
                        task=task_name,
                        method=variant,
                        budget=int(budget),
                        sde=sde,
                        num_rounds=int(num_rounds),
                        num_samples=int(num_samples),
                        observation_index=int(observation_index),
                        seed=int(seed),
                        max_iters=int(max_iters),
                        device=device,
                        c2st_folds=int(c2st_folds),
                        verbose=bool(verbose),
                        extra=dict(extra or {}),
                    )
                    _progress(
                        verbose,
                        f"[variants] task={task_name} budget={budget} sde={sde} "
                        f"method={variant}",
                    )
                    record = run_single(cfg, dataset=dataset)
                    records.append(record)
                    _maybe_flush(records, output, config={"section": "variants"})
    return records


def run_nlse_section(
    tasks: Sequence[str] = ("gaussian_linear", "slcp", "two_moons"),
    methods: Sequence[str] = NLSE_METHODS,
    budget: int = 10000,
    sdes: Sequence[str] = ("ve",),
    num_samples: int = 10000,
    seed: int = 0,
    max_iters: int = 3000,
    device: Optional[str] = None,
    c2st_folds: int = 10,
    observation_index: int = 1,
    extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    output: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """NPSE vs NLSE (Figure 5): analytic vs learned perturbed prior score."""
    records: List[Dict[str, Any]] = []
    for task_name in tasks:
        dataset = None
        try:
            task = get_task(task_name, observation_index=observation_index)
            dataset = load_dataset(task, task_name, int(budget), seed=seed)
        except Exception:
            dataset = None
        for sde in sdes:
            for method in methods:
                cfg = AblationRunConfig(
                    section="nlse",
                    task=task_name,
                    method=method,
                    budget=int(budget),
                    sde=sde,
                    num_rounds=10,
                    num_samples=int(num_samples),
                    observation_index=int(observation_index),
                    seed=int(seed),
                    max_iters=int(max_iters),
                    device=device,
                    c2st_folds=int(c2st_folds),
                    verbose=bool(verbose),
                    extra=dict(extra or {}),
                )
                _progress(
                    verbose,
                    f"[nlse] task={task_name} budget={budget} sde={sde} method={method}",
                )
                record = run_single(cfg, dataset=dataset)
                records.append(record)
                _maybe_flush(records, output, config={"section": "nlse"})
    return records


def run_sde_section(
    tasks: Sequence[str] = DEFAULT_SDE_TASKS,
    methods: Sequence[str] = ("npse",),
    budgets: Sequence[int] = (10000,),
    sdes: Sequence[str] = SDES,
    num_rounds: int = 10,
    num_samples: int = 10000,
    seed: int = 0,
    max_iters: int = 3000,
    device: Optional[str] = None,
    c2st_folds: int = 10,
    observation_index: int = 1,
    extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    output: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """VE vs VP SDE sweep across the eight benchmark tasks (both SDE families)."""
    records: List[Dict[str, Any]] = []
    for task_name in tasks:
        dataset = None
        try:
            task = get_task(task_name, observation_index=observation_index)
        except Exception:
            task = None
        for budget in budgets:
            if task is not None:
                try:
                    dataset = load_dataset(task, task_name, int(budget), seed=seed)
                except Exception:
                    dataset = None
            for method in methods:
                for sde in sdes:
                    cfg = AblationRunConfig(
                        section="sde",
                        task=task_name,
                        method=method,
                        budget=int(budget),
                        sde=sde,
                        num_rounds=int(num_rounds),
                        num_samples=int(num_samples),
                        observation_index=int(observation_index),
                        seed=int(seed),
                        max_iters=int(max_iters),
                        device=device,
                        c2st_folds=int(c2st_folds),
                        verbose=bool(verbose),
                        extra=dict(extra or {}),
                    )
                    _progress(
                        verbose,
                        f"[sde] task={task_name} budget={budget} method={method} sde={sde}",
                    )
                    record = run_single(cfg, dataset=dataset)
                    records.append(record)
                    _maybe_flush(records, output, config={"section": "sde"})
    return records


def run_ablations(
    sections: Sequence[str] = ("variants", "nlse", "sde"),
    variant_tasks: Sequence[str] = DEFAULT_VARIANT_TASKS,
    variant_methods: Sequence[str] = ABLATION_VARIANTS,
    variant_budgets: Sequence[int] = (10000,),
    nlse_tasks: Sequence[str] = ("gaussian_linear", "slcp", "two_moons"),
    nlse_budget: int = 10000,
    sde_tasks: Sequence[str] = DEFAULT_SDE_TASKS,
    sde_methods: Sequence[str] = ("npse",),
    sde_budgets: Sequence[int] = (10000,),
    sdes: Sequence[str] = SDES,
    num_rounds: int = 10,
    num_samples: int = 10000,
    seed: int = 0,
    max_iters: int = 3000,
    device: Optional[str] = None,
    c2st_folds: int = 10,
    observation_index: int = 1,
    extra: Optional[Dict[str, Any]] = None,
    output: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the selected ablation sections and return ``{"results", "config"}``."""
    sections = [s.lower() for s in sections]
    results: List[Dict[str, Any]] = []
    if "variants" in sections:
        results += run_variant_section(
            tasks=variant_tasks,
            variants=variant_methods,
            budgets=variant_budgets,
            sdes=sdes,
            num_rounds=num_rounds,
            num_samples=num_samples,
            seed=seed,
            max_iters=max_iters,
            device=device,
            c2st_folds=c2st_folds,
            observation_index=observation_index,
            extra=extra,
            verbose=verbose,
            output=output,
        )
    if "nlse" in sections:
        results += run_nlse_section(
            tasks=nlse_tasks,
            methods=NLSE_METHODS,
            budget=nlse_budget,
            sdes=sdes,
            num_samples=num_samples,
            seed=seed,
            max_iters=max_iters,
            device=device,
            c2st_folds=c2st_folds,
            observation_index=observation_index,
            extra=extra,
            verbose=verbose,
            output=output,
        )
    if "sde" in sections:
        results += run_sde_section(
            tasks=sde_tasks,
            methods=sde_methods,
            budgets=sde_budgets,
            sdes=sdes,
            num_rounds=num_rounds,
            num_samples=num_samples,
            seed=seed,
            max_iters=max_iters,
            device=device,
            c2st_folds=c2st_folds,
            observation_index=observation_index,
            extra=extra,
            verbose=verbose,
            output=output,
        )
    config = {
        "sections": list(sections),
        "variant_tasks": list(variant_tasks),
        "variant_methods": list(variant_methods),
        "variant_budgets": list(variant_budgets),
        "nlse_tasks": list(nlse_tasks),
        "nlse_budget": int(nlse_budget),
        "sde_tasks": list(sde_tasks),
        "sde_methods": list(sde_methods),
        "sde_budgets": list(sde_budgets),
        "sdes": list(sdes),
        "num_rounds": int(num_rounds),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "max_iters": int(max_iters),
    }
    if output:
        save_results(results, output, config=config)
    return {"results": results, "config": config}


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> Optional[float]:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate C2ST by ``section|task|method|sde|budget``."""
    buckets: Dict[str, List[float]] = {}
    for record in results:
        if record.get("status") != "ok" or record.get("c2st") is None:
            continue
        key = "|".join(
            str(record.get(k, "")) for k in ("section", "task", "method", "sde", "budget")
        )
        buckets.setdefault(key, []).append(float(record["c2st"]))
    return {
        key: {"mean": _mean(values), "min": min(values), "max": max(values), "n": len(values)}
        for key, values in sorted(buckets.items())
    }


def summarise_sde_preference(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-task VE-vs-VP averages and the paper's dimensionality recommendation."""
    per_task: Dict[str, Dict[str, List[float]]] = {}
    for record in results:
        if record.get("section") != "sde" or record.get("status") != "ok":
            continue
        if record.get("c2st") is None:
            continue
        per_task.setdefault(str(record["task"]), {}).setdefault(str(record["sde"]), []).append(
            float(record["c2st"])
        )
    summary: Dict[str, Any] = {"per_task": {}, "low_dim": {}, "high_dim": {}}
    buckets: Dict[str, Dict[str, List[float]]] = {"low_dim": {}, "high_dim": {}}
    for task_name, by_sde in sorted(per_task.items()):
        row = {sde: _mean(values) for sde, values in by_sde.items()}
        dims = [r.get("dim") for r in results if r.get("task") == task_name and r.get("dim")]
        dim = int(dims[0]) if dims else None
        row["dim"] = dim
        if dim is not None:
            bucket = "low_dim" if dim <= 10 else "high_dim"
            for sde, value in row.items():
                if sde in SDES and value is not None:
                    buckets[bucket].setdefault(sde, []).append(float(value))
        summary["per_task"][task_name] = row
    for bucket, by_sde in buckets.items():
        summary[bucket] = {sde: _mean(values) for sde, values in by_sde.items()}
    return summary


def check_expectations(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare the observed C2STs against the paper's stated expectations."""
    agg = aggregate(results)
    checks: Dict[str, Any] = {}

    def get(section: str, task: str, method: str, sde: Optional[str] = None):
        for key, stats in agg.items():
            parts = key.split("|")
            if len(parts) != 5:
                continue
            sec, tsk, mth, sde_key, _budget = parts
            if sec == section and tsk == task and mth == method and (
                sde is None or sde_key == sde
            ):
                return stats["mean"]
        return None

    variant_rows = []
    for task in sorted({r["task"] for r in results if r.get("section") == "variants"}):
        best_alt = [
            get("variants", task, m)
            for m in ("snpse_a", "snpse_b", "snpse_c")
            if get("variants", task, m) is not None
        ]
        tsn = get("variants", task, "tsnpse")
        snpse_c = get("variants", task, "snpse_c")
        row = {
            "task": task,
            "tsnpse": tsn,
            "snpse_a": get("variants", task, "snpse_a"),
            "snpse_b": get("variants", task, "snpse_b"),
            "snpse_c": snpse_c,
            "npse": get("variants", task, "npse"),
        }
        if tsn is not None and best_alt:
            row["tsnpse_better_than_best_alt"] = bool(tsn <= min(best_alt))
        if snpse_c is not None:
            row["snpse_c_fails"] = bool(snpse_c >= 0.9)
        variant_rows.append(row)
    checks["variants"] = variant_rows

    nlse_rows = []
    for task in sorted({r["task"] for r in results if r.get("section") == "nlse"}):
        nlse, npse = get("nlse", task, "nlse"), get("nlse", task, "npse")
        row = {"task": task, "nlse": nlse, "npse": npse}
        if nlse is not None and npse is not None:
            row["delta_nlse_minus_npse"] = nlse - npse
            row["nlse_comparable"] = bool(abs(nlse - npse) <= 0.05)
        nlse_rows.append(row)
    checks["nlse"] = nlse_rows

    sde_summary = summarise_sde_preference(results)
    if sde_summary["low_dim"] or sde_summary["high_dim"]:
        low, high = sde_summary["low_dim"], sde_summary["high_dim"]
        sde_summary["ve_preferred_low_dim"] = bool(
            low.get("ve") is not None
            and low.get("vp") is not None
            and low["ve"] <= low["vp"]
        )
        sde_summary["vp_preferred_high_dim"] = bool(
            high.get("ve") is not None
            and high.get("vp") is not None
            and high["vp"] <= high["ve"]
        )
    checks["sde"] = sde_summary
    return checks


def format_table(results: Sequence[Dict[str, Any]]) -> str:
    """Compact text table of aggregated C2ST results."""
    rows = aggregate(results)
    if not rows:
        return "(no successful runs)"
    header = f"{'section':<9} {'task':<24} {'method':<9} {'sde':<3} {'budget':>7} {'C2ST':>7}"
    lines = [header, "-" * len(header)]
    for key, stats in rows.items():
        section, task, method, sde, budget = key.split("|")
        value = stats["mean"]
        lines.append(
            f"{section:<9} {task:<24} {method:<9} {sde:<3} {budget:>7} "
            f"{value:>7.4f}" if value is not None else key
        )
    return "\n".join(lines)


def save_results(results: Sequence[Dict[str, Any]], path: str, config: Optional[Dict[str, Any]] = None) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    serialisable = [
        {k: v for k, v in record.items() if k not in ("model",)}
        for record in results
    ]
    with open(path, "w") as handle:
        json.dump({"config": config or {}, "results": serialisable}, handle, indent=2, default=str)
    return path


def load_results(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload.get("results", []), payload.get("config", {})
    return payload, {}


def plot_results(results: Sequence[Dict[str, Any]], path: str) -> Optional[str]:
    """Three-panel ablation figure (variants, NLSE, VE-vs-VP by dimension)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    agg = aggregate(results)
    sections = [s for s in ("variants", "nlse", "sde") if any(k.startswith(s) for k in agg)]
    if not sections:
        return None

    fig, axes = plt.subplots(1, len(sections), figsize=(5.0 * len(sections), 4.0))
    if len(sections) == 1:
        axes = [axes]

    for axis, section in zip(axes, sections):
        subset = {k: v for k, v in agg.items() if k.startswith(section + "|")}
        if section == "sde":
            summary = summarise_sde_preference(results)
            tasks = sorted(summary["per_task"])
            width = 0.35
            for offset, sde in enumerate(SDES):
                values = [summary["per_task"][t].get(sde) for t in tasks]
                xs = [i + (offset - 0.5) * width for i in range(len(tasks))]
                axis.bar(xs, [v if v is not None else 0.0 for v in values], width, label=sde)
            axis.set_xticks(range(len(tasks)))
            axis.set_xticklabels(tasks, rotation=60, ha="right", fontsize=7)
            axis.set_ylabel("C2ST")
            axis.set_title("VE vs VP SDE")
            axis.legend()
            continue
        labels = sorted({k.split("|")[2] for k in subset})
        grouped: Dict[str, List[float]] = {label: [] for label in labels}
        tasks = sorted({k.split("|")[1] for k in subset})
        for task in tasks:
            for label in labels:
                values = [
                    stats["mean"]
                    for key, stats in subset.items()
                    if key.split("|")[1] == task and key.split("|")[2] == label
                ]
                grouped[label].append(_mean(values))
        width = 0.8 / max(len(labels), 1)
        for offset, label in enumerate(labels):
            xs = [i + offset * width for i in range(len(tasks))]
            axis.bar(
                xs,
                [v if v is not None else 0.0 for v in grouped[label]],
                width,
                label=label,
            )
        axis.set_xticks([i + 0.4 - width / 2 for i in range(len(tasks))])
        axis.set_xticklabels(tasks, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("C2ST")
        axis.set_title(section)
        axis.legend(fontsize=7)
    for axis in axes:
        axis.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SNPSE ablation experiments (Section 3.2 / Appendix A, C, E)."
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["variants", "nlse", "sde"],
        choices=["variants", "nlse", "sde", "all"],
        help="which ablation sections to run",
    )
    parser.add_argument("--variant-tasks", nargs="+", default=list(DEFAULT_VARIANT_TASKS))
    parser.add_argument("--variants", nargs="+", default=list(ABLATION_VARIANTS))
    parser.add_argument("--variant-budgets", nargs="+", type=int, default=[10000])
    parser.add_argument("--nlse-tasks", nargs="+", default=["gaussian_linear", "slcp", "two_moons"])
    parser.add_argument("--nlse-budget", type=int, default=10000)
    parser.add_argument("--sde-tasks", nargs="+", default=list(DEFAULT_SDE_TASKS))
    parser.add_argument("--sde-methods", nargs="+", default=["npse"])
    parser.add_argument("--sde-budgets", nargs="+", type=int, default=[10000])
    parser.add_argument("--sdes", nargs="+", default=list(SDES))
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--c2st-folds", type=int, default=10)
    parser.add_argument("--observation-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--figure", type=str, default=None, help="figure output path")
    parser.add_argument("--quick", action="store_true", help="short smoke-test configuration")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sections = args.sections
    if "all" in sections:
        sections = ["variants", "nlse", "sde"]

    variant_budgets = list(args.variant_budgets)
    sde_budgets = list(args.sde_budgets)
    sde_tasks = list(args.sde_tasks)
    variants = list(args.variants)
    nlse_tasks = list(args.nlse_tasks)
    num_samples = int(args.num_samples)
    num_rounds = int(args.num_rounds)
    max_iters = int(args.max_iters)

    if args.quick:
        variant_budgets = [1000]
        sde_budgets = [1000]
        variants = [v for v in variants if v in ("tsnpse", "snpse_c", "npse")]
        sde_tasks = sde_tasks[:2]
        nlse_tasks = nlse_tasks[:1]
        num_samples = min(num_samples, 2000)
        num_rounds = 2
        max_iters = min(max_iters, 500)

    output = args.output
    if output is None and not args.quick:
        output = os.path.join(os.getcwd(), "results", "ablations.json")
    elif output is None:
        output = os.path.join(os.getcwd(), "results", "ablations_quick.json")

    payload = run_ablations(
        sections=sections,
        variant_tasks=list(args.variant_tasks),
        variant_methods=variants,
        variant_budgets=variant_budgets,
        nlse_tasks=nlse_tasks,
        nlse_budget=int(args.nlse_budget),
        sde_tasks=sde_tasks,
        sde_methods=list(args.sde_methods),
        sde_budgets=sde_budgets,
        sdes=list(args.sdes),
        num_rounds=num_rounds,
        num_samples=num_samples,
        seed=int(args.seed),
        max_iters=max_iters,
        device=args.device,
        c2st_folds=int(args.c2st_folds),
        observation_index=int(args.observation_index),
        output=output,
        verbose=not args.quiet,
    )
    results = payload["results"]
    print(format_table(results))
    checks = check_expectations(results)
    print("\nExpectation checks:")
    print(json.dumps(checks, indent=2, default=str))
    if args.figure:
        path = plot_results(results, args.figure)
        if path:
            print(f"figure -> {path}")
    if output:
        print(f"results -> {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
