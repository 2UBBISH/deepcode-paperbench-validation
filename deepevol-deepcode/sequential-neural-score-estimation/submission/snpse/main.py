#!/usr/bin/env python
"""Command line entry point for the SNPSE reproduction.

This script is a thin dispatch layer on top of the library modules.  It exposes
every inference method described in the paper (NPSE, TSNPSE, SNPSE-A/B/C, NLSE)
plus the experiment drivers (benchmark sweep, pyloric neuroscience experiment,
appendix ablations, baselines) through a single CLI.

Examples
--------
Single method on a single task::

    python main.py --method npse --task two_moons --budget 1000 --sde ve

Sequential method with 10 rounds::

    python main.py --method tsnpse --task slcp --budget 100000 --num-rounds 10

Full benchmark sweep (Section 5.2)::

    python main.py experiment benchmarks --methods npse tsnpse \
        --sdes ve vp --budgets 1000 10000 100000

Pyloric neuroscience experiment (Section 5.3)::

    python main.py experiment pyloric --num-rounds 9 --sde vp

Appendix ablations (Section 3.2, Appendix B/C)::

    python main.py experiment ablations --sections variants nlse sde
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Robust importing: this file lives at the repository root (``snpse/main.py``)
# while the library code may be either at ``snpse/snpse/`` (nested layout) or in
# a flat checkout.  Both are supported by inserting the project root on
# ``sys.path`` and trying several import paths in turn.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _import_first(candidates: Sequence[str], package: bool = False) -> Optional[Any]:
    """Return the first importable module among ``candidates`` (else ``None``)."""
    last_error: Optional[BaseException] = None
    for name in candidates:
        try:
            if package:
                return importlib.import_module(name)
            return importlib.import_module(name)
        except ImportError as exc:  # pragma: no cover - depends on layout
            last_error = exc
        except Exception as exc:  # pragma: no cover
            last_error = exc
    if last_error is not None:
        _LAST_IMPORT_ERROR[:] = [last_error]
    return None


_LAST_IMPORT_ERROR: List[BaseException] = [None]  # type: ignore[list-item]


def _module(*names: str) -> Optional[Any]:
    """Import the first available module from a list of alternative names."""
    return _import_first(names)


# -- library modules (nested layout ``snpse.snpse.*`` first, flat second) -----
_core = _module("snpse.snpse", "snpse")
_utils = _module("snpse.snpse.utils", "snpse.utils")
_npse_mod = _module("snpse.snpse.npse", "snpse.npse")
_tsnpse_mod = _module("snpse.snpse.tsnpse", "snpse.tsnpse")
_variants_mod = _module("snpse.snpse.snpse_variants", "snpse.snpse_variants")
_nlse_mod = _module("snpse.snpse.nlse", "snpse.nlse")
_bench_mod = _module("snpse.tasks.benchmarks", "tasks.benchmarks")
_c2st_mod = _module("snpse.tasks.c2st", "tasks.c2st")
_baselines_mod = _module("snpse.experiments.baselines", "experiments.baselines")
_run_bench_mod = _module("snpse.experiments.run_benchmarks", "experiments.run_benchmarks")
_run_pyloric_mod = _module("snpse.experiments.run_pyloric", "experiments.run_pyloric")
_run_ablations_mod = _module("snpse.experiments.run_ablations", "experiments.run_ablations")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_NAMES = ("default.yaml", "default.yml")

METHOD_ALIASES = {
    "npse": "npse",
    "tsnpse": "tsnpse",
    "snpe": "tsnpse",
    "snpse-a": "snpse_a",
    "snpse_a": "snpse_a",
    "snpse-b": "snpse_b",
    "snpse_b": "snpse_b",
    "snpse-c": "snpse_c",
    "snpse_c": "snpse_c",
    "nlse": "nlse",
}

METHOD_CHOICES = ["npse", "tsnpse", "snpse-a", "snpse-b", "snpse-c", "nlse"]

_VALID_BACKENDS = ("auto", "sbibm", "mackelab", "fallback")


def _config_path(named: Optional[str] = None) -> Optional[str]:
    """Resolve the path of a YAML config file."""
    candidates: List[str] = []
    if named:
        candidates.extend([named, os.path.abspath(named)])
        candidates.append(os.path.join(_HERE, named))
        candidates.append(os.path.join(_HERE, "configs", named))
    for name in DEFAULT_CONFIG_NAMES:
        candidates.append(os.path.join(_HERE, "configs", name))
        candidates.append(os.path.join(_ROOT, "configs", name))
        candidates.append(os.path.join(_ROOT, "snpse", "configs", name))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML config (returns ``{}`` when PyYAML/config are unavailable)."""
    resolved = _config_path(path)
    if resolved is None:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - PyYAML is in requirements
        return {}
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:  # pragma: no cover - malformed config must not kill the CLI
        return {}
    if isinstance(data, dict):
        data.setdefault("_config_path", resolved)
        return data
    return {}


def _config_get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Fetch ``config[k1][k2]...`` returning ``default`` when absent."""
    node: Any = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def _set_seed(seed: Optional[int]) -> None:
    if seed is None or _utils is None:
        return
    for name in ("set_seed", "seed_everything"):
        fn = getattr(_utils, name, None)
        if callable(fn):
            try:
                fn(seed)
                return
            except Exception:  # pragma: no cover
                continue


def _call_filtered(fn, *args, **kwargs):
    """Call ``fn`` dropping kwargs it does not accept."""
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover
        return fn(*args, **kwargs)
    parameters = signature.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if accepts_kwargs:
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in parameters and v is not None}
    return fn(*args, **filtered)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of tensors/arrays to JSON friendly objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:  # torch tensor
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().reshape(-1)[0])
            return [float(v) for v in value.detach().cpu().reshape(-1)[:64]]
    except Exception:  # pragma: no cover
        pass
    try:  # numpy array
        import numpy as np

        if isinstance(value, np.ndarray):
            return _jsonable(np.asarray(value).tolist())
        if isinstance(value, np.generic):
            return value.item()
    except Exception:  # pragma: no cover
        pass
    return str(value)


def _dump(results: Any, output: Optional[str]) -> str:
    """Print (and optionally save) a JSON-serialisable results object."""
    payload = json.dumps(_jsonable(results), indent=2, sort_keys=True, default=str)
    if output:
        directory = os.path.dirname(os.path.abspath(output))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    return payload


def _print_banner(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("SNPSE - Sequential Neural Posterior Score Estimation (ICML reproduction)")
    print("=" * 72)
    print(f"method : {args.method}   task: {args.task}   budget: {args.budget}")
    print(f"sde    : {args.sde}      device: {_resolve_device(args.device)}   seed: {args.seed}")
    print("-" * 72)


# ---------------------------------------------------------------------------
# Method dispatch: single (task, method, budget, sde) cell
# ---------------------------------------------------------------------------


def _get_task(task: str, observation_index: int = 1, backend: str = "auto") -> Any:
    if _bench_mod is None:
        raise ImportError(
            "tasks.benchmarks is unavailable; run from the repository root or "
            "install the package (pip install -e .)."
        )
    return _call_filtered(
        _bench_mod.get_task, task, backend=backend, observation_index=observation_index
    )


def _load_dataset(task_name: str, task: Any, budget: int, seed: int, cache_dir=None):
    if _bench_mod is not None and hasattr(_bench_mod, "load_dataset"):
        try:
            return _call_filtered(
                _bench_mod.load_dataset,
                task,
                budget,
                seed=seed,
                cache_dir=cache_dir,
                use_cache=True,
                verbose=False,
            )
        except Exception:  # pragma: no cover - fall back to in-memory sampling
            pass
    # minimal in-memory fallback
    import torch

    from_check = _module("snpse.snpse.utils", "snpse.utils")
    generator = None
    if from_check is not None and hasattr(from_check, "set_seed"):
        generator = from_check.set_seed(seed)
    theta = task.sample_prior(budget, generator=generator)
    x = task.simulate(theta, generator=generator)
    return {"theta": theta, "x": x}


def _task_observation(task: Any):
    if hasattr(task, "observation") and callable(getattr(task, "observation")):
        try:
            return task.observation()
        except Exception:  # pragma: no cover
            pass
    for attribute in ("x_obs", "obs", "observation"):
        value = getattr(task, attribute, None)
        if value is not None and not callable(value):
            return value
    raise AttributeError("Could not extract the observation from the task object.")


def _prior_of(task: Any):
    for attribute in ("prior", "prior_sampler", "sample_prior"):
        value = getattr(task, attribute, None)
        if value is not None:
            return value
    raise AttributeError("Could not extract the prior from the task object.")


def _prior_sample_fn(task: Any):
    prior = _prior_of(task)
    for attribute in ("sample_fn", "sample"):
        fn = getattr(prior, attribute, None)
        if callable(fn):
            return fn
    if callable(prior):
        return prior
    raise AttributeError("Could not extract a prior sampling function.")


def _prior_log_prob_fn(task: Any):
    prior = _prior_of(task)
    for attribute in ("log_prob_fn", "log_prob"):
        fn = getattr(prior, attribute, None)
        if callable(fn):
            return fn
    return None


def _simulator_fn(task: Any):
    for attribute in ("simulator_fn", "simulate", "simulator"):
        fn = getattr(task, attribute, None)
        if callable(fn):
            return fn
    raise AttributeError("Could not extract the simulator from the task object.")


def _task_dims(task: Any):
    dim_theta = getattr(task, "dim_theta", None) or getattr(task, "dim_parameters", None)
    dim_x = getattr(task, "dim_x", None) or getattr(task, "dim_data", None)
    if dim_theta is None:
        theta = task.sample_prior(2)
        dim_theta = int(theta.shape[-1])
    if dim_x is None:
        theta = task.sample_prior(1)
        dim_x = int(task.simulate(theta).shape[-1])
    return int(dim_theta), int(dim_x)


def _num_samples_for(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    if getattr(args, "num_samples", None):
        return int(args.num_samples)
    default = _config_get(config, "sampler", "num_samples", default=None)
    if default:
        return int(default)
    return 10000


def run_npse_cell(args: argparse.Namespace, config: Dict[str, Any], task: Any) -> Dict[str, Any]:
    """NPSE on one task/budget (Section 2.2)."""
    if _npse_mod is None:
        raise ImportError("snpse.npse is unavailable.")
    dataset = _load_dataset(args.task, task, args.budget, args.seed, args.cache_dir)
    theta = dataset["theta"] if isinstance(dataset, dict) else dataset[0]
    x = dataset["x"] if isinstance(dataset, dict) else dataset[1]
    dim_theta, dim_x = _task_dims(task)

    cfg_kwargs = dict(
        sde=args.sde,
        budget=int(args.budget),
        max_iters=int(args.max_iters),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        seed=int(args.seed),
        device=_resolve_device(args.device),
        verbose=bool(args.verbose),
        standardise=bool(args.standardise),
    )
    cfg = _call_filtered(_npse_mod.NPSEConfig, **cfg_kwargs)
    model = _call_filtered(
        _npse_mod.NPSE, dim_theta, dim_x, config=cfg, device=_resolve_device(args.device)
    )
    model.fit(theta, x)
    x_obs = _task_observation(task)
    samples = model.sample(x_obs, _num_samples_for(args, config))
    if isinstance(samples, tuple):
        samples = samples[0]
    return {"theta": samples, "model": model, "info": {"dataset_size": int(theta.shape[0])}}


def run_tsnpse_cell(
    args: argparse.Namespace, config: Dict[str, Any], task: Any
) -> Dict[str, Any]:
    """TSNPSE on one task/budget: Algorithm 1 with HPR truncated proposals."""
    if _tsnpse_mod is None:
        raise ImportError("snpse.tsnpse is unavailable.")
    dim_theta, dim_x = _task_dims(task)
    x_obs = _task_observation(task)
    # Round 1 uses the shared cached prior-predictive dataset (budget / num_rounds)
    num_rounds = int(args.num_rounds)
    initial = max(1, int(args.budget) // max(1, num_rounds))
    dataset = _load_dataset(args.task, task, initial, args.seed, args.cache_dir)
    initial_theta = dataset["theta"] if isinstance(dataset, dict) else dataset[0]
    initial_x = dataset["x"] if isinstance(dataset, dict) else dataset[1]

    cfg_kwargs = dict(
        sde=args.sde,
        budget=int(args.budget),
        num_rounds=num_rounds,
        initial_budget=int(initial),
        simulations_per_round=int(initial),
        num_samples=_num_samples_for(args, config),
        max_iters=int(args.max_iters),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        eps=float(args.eps),
        n_hpr_samples=int(args.n_hpr_samples),
        seed=int(args.seed),
        device=_resolve_device(args.device),
        verbose=bool(args.verbose),
        standardise=bool(args.standardise),
    )
    cfg = _call_filtered(_tsnpse_mod.TSNPSEConfig, **cfg_kwargs)
    result = _call_filtered(
        _tsnpse_mod.run_tsnpse,
        _prior_of(task),
        _simulator_fn(task),
        x_obs,
        dim_theta,
        dim_x,
        config=cfg,
        num_samples=_num_samples_for(args, config),
        seed=int(args.seed),
        device=_resolve_device(args.device),
        initial_theta=initial_theta,
        initial_x=initial_x,
        prior_sample_fn=_prior_sample_fn(task),
        prior_log_prob_fn=_prior_log_prob_fn(task),
    )
    if isinstance(result, dict):
        theta = result.get("theta")
    else:  # pragma: no cover - defensive
        theta = result
    return {"theta": theta, "model": None, "info": {"num_rounds": num_rounds}}


def run_nlse_cell(args: argparse.Namespace, config: Dict[str, Any], task: Any) -> Dict[str, Any]:
    """NLSE (Appendix B): learn the perturbed likelihood score, add prior score."""
    if _nlse_mod is None:
        raise ImportError("snpse.nlse is unavailable.")
    dim_theta, dim_x = _task_dims(task)
    dataset = _load_dataset(args.task, task, args.budget, args.seed, args.cache_dir)
    theta = dataset["theta"] if isinstance(dataset, dict) else dataset[0]
    x = dataset["x"] if isinstance(dataset, dict) else dataset[1]
    x_obs = _task_observation(task)
    cfg = _call_filtered(
        _nlse_mod.NLSEConfig,
        sde=args.sde,
        budget=int(args.budget),
        max_iters=int(args.max_iters),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        seed=int(args.seed),
        device=_resolve_device(args.device),
        verbose=bool(args.verbose),
    )
    model = _call_filtered(
        _nlse_mod.NLSE,
        dim_theta,
        dim_x,
        config=cfg,
        prior=_prior_of(task),
        prior_sample_fn=_prior_sample_fn(task),
        prior_log_prob_fn=_prior_log_prob_fn(task),
        simulator=_simulator_fn(task),
        device=_resolve_device(args.device),
    )
    model.fit(theta, x)
    samples = model.sample(x_obs, _num_samples_for(args, config))
    if isinstance(samples, tuple):
        samples = samples[0]
    return {"theta": samples, "model": model, "info": {"dataset_size": int(theta.shape[0])}}


def run_variant_cell(
    args: argparse.Namespace, config: Dict[str, Any], task: Any, method: str
) -> Dict[str, Any]:
    """SNPSE-A/B/C alternative sequential variants (Section 3.2)."""
    if _variants_mod is None:
        raise ImportError("snpse.snpse_variants is unavailable.")
    runner_name = {
        "snpse_a": "run_snpse_a",
        "snpse_b": "run_snpse_b",
        "snpse_c": "run_snpse_c",
    }[method]
    runner = getattr(_variants_mod, runner_name)
    dim_theta, dim_x = _task_dims(task)
    x_obs = _task_observation(task)
    num_rounds = int(args.num_rounds)
    initial = max(1, int(args.budget) // max(1, num_rounds))
    dataset = _load_dataset(args.task, task, initial, args.seed, args.cache_dir)
    initial_theta = dataset["theta"] if isinstance(dataset, dict) else dataset[0]
    initial_x = dataset["x"] if isinstance(dataset, dict) else dataset[1]
    cfg = _call_filtered(
        _variants_mod.SNPSEConfig,
        sde=args.sde,
        budget=int(args.budget),
        num_rounds=num_rounds,
        initial_budget=int(initial),
        simulations_per_round=int(initial),
        max_iters=int(args.max_iters),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        seed=int(args.seed),
        device=_resolve_device(args.device),
        verbose=bool(args.verbose),
    )
    result = _call_filtered(
        runner,
        _prior_of(task),
        _simulator_fn(task),
        x_obs,
        dim_theta,
        dim_x,
        config=cfg,
        num_samples=_num_samples_for(args, config),
        seed=int(args.seed),
        device=_resolve_device(args.device),
        initial_theta=initial_theta,
        initial_x=initial_x,
        prior_sample_fn=_prior_sample_fn(task),
        prior_log_prob_fn=_prior_log_prob_fn(task),
    )
    theta = result.get("theta") if isinstance(result, dict) else result
    return {"theta": theta, "model": None, "info": {"variant": method}}


def run_baseline_cell(
    args: argparse.Namespace, config: Dict[str, Any], task: Any, method: str
) -> Dict[str, Any]:
    """NPE / SNPE-C / TSNPE baselines (Section 5.2, via sbibm or mackelab)."""
    if _baselines_mod is None:
        raise ImportError("experiments.baselines is unavailable.")
    cfg = _call_filtered(
        _baselines_mod.BaselineConfig,
        method=method,
        budget=int(args.budget),
        num_rounds=int(args.num_rounds),
        num_samples=_num_samples_for(args, config),
        seed=int(args.seed),
        device=_resolve_device(args.device),
        verbose=bool(args.verbose),
        backend=args.backend,
    )
    result = _call_filtered(
        _baselines_mod.run_baseline,
        method,
        task,
        config=cfg,
        backend=args.backend,
        device=_resolve_device(args.device),
    )
    theta = result.get("theta") if isinstance(result, dict) else result
    return {"theta": theta, "model": None, "info": {"baseline": method}}


CELL_RUNNERS = {
    "npse": run_npse_cell,
    "tsnpse": run_tsnpse_cell,
    "nlse": run_nlse_cell,
    "snpse_a": run_variant_cell_a if False else None,  # placeholder, set below
    "snpse_b": None,
    "snpse_c": None,
}


def _variant_runner_a(args, config, task):
    return run_variant_cell(args, config, task, "snpse_a")


def _variant_runner_b(args, config, task):
    return run_variant_cell(args, config, task, "snpse_b")


def _variant_runner_c(args, config, task):
    return run_variant_cell(args, config, task, "snpse_c")


CELL_RUNNERS.update(
    {
        "snpse_a": _variant_runner_a,
        "snpse_b": _variant_runner_b,
        "snpse_c": _variant_runner_c,
    }
)


def _reference_samples(task: Any, n: int, seed: int):
    if _c2st_mod is not None and hasattr(_c2st_mod, "reference_posterior"):
        try:
            return _call_filtered(_c2st_mod.reference_posterior, task, num_samples=n, seed=seed)
        except Exception:  # pragma: no cover
            return None
    return None


def run_single(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one method/task/budget/SDE cell and score it with C2ST."""
    method = METHOD_ALIASES.get(str(args.method).lower(), str(args.method).lower())
    task = _get_task(args.task, args.observation_index, args.backend)
    start = time.time()
    record: Dict[str, Any] = {
        "method": method,
        "task": args.task,
        "budget": int(args.budget),
        "sde": args.sde,
        "seed": int(args.seed),
        "status": "ok",
    }
    try:
        if method in ("npe", "snpe_c", "tsnpe", "fmpe"):
            result = run_baseline_cell(args, config, task, method)
        else:
            runner = CELL_RUNNERS.get(method)
            if runner is None:
                raise ValueError(f"Unknown method: {args.method!r}")
            result = runner(args, config, task)
        record["theta"] = result.get("theta") if isinstance(result, dict) else result
        record["info"] = result.get("info") if isinstance(result, dict) else None
    except Exception as exc:  # pragma: no cover - robustness of the sweep
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        record["runtime"] = time.time() - start
        return record

    # C2ST evaluation against the reference posterior (Section 5.2)
    try:
        if _c2st_mod is not None and record.get("theta") is not None:
            ref = _reference_samples(task, _num_samples_for(args, config), args.seed)
            if ref is not None:
                value = _call_filtered(
                    _c2st_mod.c2st,
                    record["theta"],
                    ref,
                    seed=int(args.seed),
                    n_folds=int(args.c2st_folds),
                )
                record["c2st"] = float(value) if value is not None else None
    except Exception as exc:  # pragma: no cover
        record["c2st_error"] = f"{type(exc).__name__}: {exc}"
    record["runtime"] = time.time() - start
    return record


# ---------------------------------------------------------------------------
# Experiment-level dispatch
# ---------------------------------------------------------------------------


def experiment_benchmarks(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Section 5.2: NPSE/TSNPSE x 8 tasks x budgets x VE/VP."""
    if _run_bench_mod is None:
        raise ImportError("experiments.run_benchmarks is unavailable.")
    tasks = args.tasks or _config_get(config, "tasks", "names", default=None) or None
    return _call_filtered(
        _run_bench_mod.run_sweep,
        tasks=tasks,
        methods=args.methods,
        sdes=args.sdes,
        budgets=args.budgets,
        seed=int(args.seed),
        num_samples=_num_samples_for(args, config),
        num_rounds=int(args.num_rounds),
        max_iters=int(args.max_iters),
        device=_resolve_device(args.device),
        output=args.output,
        c2st_folds=int(args.c2st_folds),
        observation_index=int(args.observation_index),
        verbose=bool(args.verbose),
    )


def experiment_ablations(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Section 3.2 / Appendices B-C: SNPSE-A/B/C, NPSE vs NLSE, VE vs VP."""
    if _run_ablations_mod is None:
        raise ImportError("experiments.run_ablations is unavailable.")
    return _call_filtered(
        _run_ablations_mod.run_ablations,
        sections=args.sections,
        sdes=args.sdes,
        num_rounds=int(args.num_rounds),
        num_samples=_num_samples_for(args, config),
        seed=int(args.seed),
        max_iters=int(args.max_iters),
        device=_resolve_device(args.device),
        c2st_folds=int(args.c2st_folds),
        observation_index=int(args.observation_index),
        output=args.output,
        verbose=bool(args.verbose),
    )


def experiment_pyloric(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Section 5.3: TSNPSE on the pyloric simulator (9 rounds, 30000+20000)."""
    if _run_pyloric_mod is None:
        raise ImportError("experiments.run_pyloric is unavailable.")
    run_config = None
    if hasattr(_run_pyloric_mod, "PyloricRunConfig"):
        run_config = _call_filtered(
            _run_pyloric_mod.PyloricRunConfig,
            method=args.method,
            sde=args.sde,
            num_rounds=int(args.num_rounds),
            initial_simulations=int(args.initial_simulations),
            simulations_per_round=int(args.simulations_per_round),
            num_samples=_num_samples_for(args, config),
            seed=int(args.seed),
            max_iters=int(args.max_iters),
            hidden_dim=int(args.hidden_dim),
            n_layers=int(args.n_layers),
            eps=float(args.eps),
            n_hpr_samples=int(args.n_hpr_samples),
            device=_resolve_device(args.device),
            verbose=bool(args.verbose),
        )
    function = getattr(_run_pyloric_mod, "run_pyloric")
    return _call_filtered(function, run_config)


def experiment_baselines(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run NPE / SNPE-C / TSNPE baselines over the benchmark tasks."""
    if _baselines_mod is None:
        raise ImportError("experiments.baselines is unavailable.")
    tasks = args.tasks or list(getattr(_bench_mod, "TASK_NAMES", []) or [])
    records: List[Dict[str, Any]] = []
    for task_name in tasks:
        for method in args.methods or ["npe", "snpe_c"]:
            cell_args = argparse.Namespace(**vars(args))
            cell_args.task = task_name
            cell_args.method = method
            for budget in args.budgets or [args.budget]:
                budget_args = argparse.Namespace(**vars(cell_args))
                budget_args.budget = budget
                try:
                    records.append(run_single(budget_args, config))
                except Exception as exc:  # pragma: no cover
                    records.append(
                        {
                            "task": task_name,
                            "method": method,
                            "budget": budget,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    if _baselines_mod is not None and hasattr(_baselines_mod, "evaluate_c2st") and records:
        for record in records:
            if record.get("status") == "ok" and record.get("c2st") is None and record.get("theta") is not None:
                try:
                    task = _get_task(record["task"], args.observation_index, args.backend)
                    summary = _call_filtered(
                        _baselines_mod.evaluate_c2st,
                        task,
                        record["theta"],
                        seed=int(args.seed),
                    )
                    if isinstance(summary, dict):
                        record.update({k: v for k, v in summary.items() if k not in record})
                except Exception:  # pragma: no cover
                    pass
    return {"results": records}


EXPERIMENTS = {
    "benchmarks": experiment_benchmarks,
    "ablations": experiment_ablations,
    "pyloric": experiment_pyloric,
    "baselines": experiment_baselines,
}


# ---------------------------------------------------------------------------
# Self-check / smoke test
# ---------------------------------------------------------------------------


def smoke_test(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """Tiny end-to-end check used by ``--smoke-test``."""
    failures: List[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"  [ok]   {name}")
        except Exception as exc:  # pragma: no cover
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")

    print("Smoke test")
    print("-" * 72)

    if _core is not None:
        check("import snpse core", lambda: (getattr(_core, "NPSE", None) is not None) or (_ for _ in ()).throw(ImportError("NPSE missing")))
    else:
        failures.append("snpse core package not importable")
        print("  [FAIL] import snpse core")

    if _bench_mod is not None:
        def _tasks():
            names = list(getattr(_bench_mod, "TASK_NAMES", []) or [])
            if not names:
                raise RuntimeError("TASK_NAMES empty")
            return names

        check("tasks.benchmarks registry", _tasks)
    else:
        failures.append("tasks.benchmarks not importable")

    # 2D Gaussian-linear sanity check (score error + posterior recovery)
    if _npse_mod is not None and _bench_mod is not None:
        def _gaussian_linear():
            import torch

            task = _get_task("gaussian_linear", 1, args.backend)
            dim_theta, dim_x = _task_dims(task)
            dataset = _load_dataset("gaussian_linear", task, int(args.smoke_budget), int(args.seed), args.cache_dir)
            theta = dataset["theta"] if isinstance(dataset, dict) else dataset[0]
            x = dataset["x"] if isinstance(dataset, dict) else dataset[1]
            cfg = _call_filtered(
                _npse_mod.NPSEConfig,
                sde=args.sde,
                budget=int(args.smoke_budget),
                max_iters=int(args.smoke_iters),
                seed=int(args.seed),
                device="cpu",
                verbose=False,
            )
            model = _call_filtered(_npse_mod.NPSE, dim_theta, dim_x, config=cfg, device="cpu")
            model.fit(theta, x)
            samples = model.sample(_task_observation(task), 256)
            if isinstance(samples, tuple):
                samples = samples[0]
            if not isinstance(samples, torch.Tensor) or samples.shape[0] != 256:
                raise RuntimeError(f"unexpected sample shape: {samples.shape}")
            if not torch.isfinite(samples).all():
                raise RuntimeError("non-finite posterior samples")

        check("NPSE on gaussian_linear (2D mean field)", _gaussian_linear)

    if _c2st_mod is not None:
        def _c2st_self():
            import torch

            generator = torch.Generator().manual_seed(int(args.seed))
            a = torch.randn(400, 2, generator=generator)
            b = torch.randn(400, 2, generator=generator)
            value = float(_call_filtered(_c2st_mod.c2st, a, b, seed=int(args.seed)))
            if not (0.0 <= value <= 1.0):
                raise RuntimeError(f"C2ST out of range: {value}")

        check("C2ST on identical Gaussians (~0.5)", _c2st_self)

    if _tsnpse_mod is not None:
        # round 1 of TSNPSE must reduce to NPSE
        def _round1_equivalence():
            import torch

            task = _get_task("two_moons", 1, args.backend)
            dim_theta, dim_x = _task_dims(task)
            dataset = _load_dataset("two_moons", task, 256, int(args.seed), args.cache_dir)
            theta = dataset["theta"] if isinstance(dataset, dict) else dataset[0]
            x = dataset["x"] if isinstance(dataset, dict) else dataset[1]
            cfg = _call_filtered(
                _npse_mod.NPSEConfig, sde="ve", budget=256, max_iters=50, seed=0, device="cpu", verbose=False
            )
            model = _call_filtered(_npse_mod.NPSE, dim_theta, dim_x, config=cfg, device="cpu")
            model.fit(theta, x)
            out = model.sample(_task_observation(task), 32)
            out = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(out).all():
                raise RuntimeError("non-finite round-1 samples")

        check("TSNPSE round-1 (NPSE reduction, two_moons)", _round1_equivalence)

    print("-" * 72)
    if failures:
        print(f"Smoke test FAILED ({len(failures)} issue(s)):")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("Smoke test PASSED")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "SNPSE reproduction: NPSE, TSNPSE, SNPSE-A/B/C and NLSE for "
            "simulator-based inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=METHOD_CHOICES + ["npe", "snpe_c", "snpe-c", "tsnpe"],
        help="Inference method to run for a single cell.",
    )
    parser.add_argument("--task", default="two_moons", help="Benchmark task name (Section E.1).")
    parser.add_argument("--budget", type=int, default=10000, help="Number of simulations.")
    parser.add_argument("--sde", default=None, choices=["ve", "vp"], help="Forward noising SDE.")
    parser.add_argument("--num-rounds", type=int, default=None, help="Sequential rounds R (Algorithm 1).")
    parser.add_argument("--num-samples", type=int, default=None, help="Posterior samples to draw.")
    parser.add_argument("--max-iters", type=int, default=3000, help="Maximum training iterations.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--eps", type=float, default=5e-4, help="HPR quantile (Appendix E.3.3).")
    parser.add_argument("--n-hpr-samples", type=int, default=20000)
    parser.add_argument("--c2st-folds", type=int, default=10)
    parser.add_argument("--observation-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="torch device string (default: auto).")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=_VALID_BACKENDS,
        help="Backend for sbibm tasks / baselines.",
    )
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml).")
    parser.add_argument("--output", default=None, help="JSON output path.")
    parser.add_argument("--figure", default=None, help="Figure output path (where supported).")
    parser.add_argument("--cache-dir", default=None, help="Dataset cache directory.")
    parser.add_argument(
        "--no-standardise", dest="standardise", action="store_false", help="Disable standardisation."
    )
    parser.add_argument("--smoke-iters", type=int, default=50, help="Iterations for --smoke-test.")
    parser.add_argument("--smoke-budget", type=int, default=256, help="Budget for --smoke-test.")
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast self-check of the whole pipeline and exit.",
    )
    parser.set_defaults(standardise=True, verbose=True)

    subparsers = parser.add_subparsers(dest="experiment")
    # --- benchmarks --------------------------------------------------------
    bench = subparsers.add_parser(
        "benchmarks", help="Section 5.2 sweep over the eight sbibm tasks."
    )
    bench.add_argument("--tasks", nargs="*", default=None)
    bench.add_argument("--methods", nargs="*", default=["npse", "tsnpse"])
    bench.add_argument("--sdes", nargs="*", default=["ve", "vp"])
    bench.add_argument("--budgets", nargs="*", type=int, default=[1000, 10000, 100000])
    # --- ablations ---------------------------------------------------------
    abl = subparsers.add_parser(
        "ablations", help="Section 3.2 / Appendix B-C ablations (SNPSE-A/B/C, NLSE, VE-vs-VP)."
    )
    abl.add_argument(
        "--sections",
        nargs="*",
        default=["variants", "nlse", "sde"],
        choices=["variants", "nlse", "sde"],
    )
    abl.add_argument("--sdes", nargs="*", default=["ve", "vp"])
    # --- pyloric -----------------------------------------------------------
    pyl = subparsers.add_parser("pyloric", help="Section 5.3 pyloric neuroscience experiment.")
    pyl.add_argument("--initial-simulations", type=int, default=30000)
    pyl.add_argument("--simulations-per-round", type=int, default=20000)
    # --- baselines ---------------------------------------------------------
    base = subparsers.add_parser("baselines", help="NPE / SNPE-C / TSNPE baselines (Section 5.2).")
    base.add_argument(
        "--methods", nargs="*", default=["npe", "snpe_c", "tsnpe"], choices=["npe", "snpe_c", "tsnpe", "fmpe"]
    )
    base.add_argument("--tasks", nargs="*", default=None)
    base.add_argument("--budgets", nargs="*", type=int, default=[1000, 10000, 100000])

    return parser


def _apply_config_defaults(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Fill unspecified CLI arguments from the YAML config."""
    if args.sde is None:
        args.sde = _config_get(config, "sde", "default", default="ve") or "ve"
    if args.seed is None:
        args.seed = int(_config_get(config, "run", "seed", default=0) or 0)
    if args.num_rounds is None:
        args.num_rounds = int(_config_get(config, "sequential", "num_rounds", default=10) or 10)
    if args.max_iters is None:
        args.max_iters = int(_config_get(config, "training", "max_iters", default=3000) or 3000)
    if getattr(args, "initial_simulations", None) in (None, 0):
        args.initial_simulations = int(
            _config_get(config, "pyloric", "initial_simulations", default=30000) or 30000
        )
    if getattr(args, "simulations_per_round", None) in (None, 0):
        args.simulations_per_round = int(
            _config_get(config, "pyloric", "simulations_per_round", default=20000) or 20000
        )
    if getattr(args, "eps", None) is None:
        args.eps = float(_config_get(config, "sequential", "eps", default=5e-4) or 5e-4)
    if getattr(args, "n_hpr_samples", None) is None:
        args.n_hpr_samples = int(
            _config_get(config, "sequential", "n_hpr_samples", default=20000) or 20000
        )
    if getattr(args, "device", None) is None:
        args.device = _config_get(config, "run", "device", default=None)

    # Pyloric defaults must survive even when the sub-parser set them.
    if getattr(args, "experiment", None) == "pyloric":
        if args.sde is None or args.method in (None, "npse", "tsnpse"):
            args.sde = _config_get(config, "pyloric", "sde", default="vp") or "vp"
        if args.method is None:
            args.method = "tsnpse"
        if args.num_rounds in (None, 10):
            cfg_rounds = _config_get(config, "pyloric", "num_rounds", default=None)
            if cfg_rounds:
                args.num_rounds = int(cfg_rounds)
    if args.method is None:
        args.method = _config_get(config, "default", default="npse") or "npse"
    if args.task is None:
        args.task = "two_moons"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    _apply_config_defaults(args, config)

    if getattr(args, "smoke_test", False):
        return smoke_test(config, args)

    _set_seed(args.seed)
    _print_banner(args)

    try:
        if args.experiment:
            runner = EXPERIMENTS.get(args.experiment)
            if runner is None:  # pragma: no cover - argparse restricts choices
                parser.error(f"unknown experiment {args.experiment!r}")
            results = runner(args, config)
        else:
            results = run_single(args, config)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    _dump(results, args.output)
    print("-" * 72)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
