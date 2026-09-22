"""Neuroscience experiment: pyloric network inference with sequential SNPSE.

Implements the Section 5.3 experiment of *Sequential Neural Posterior Score
Estimation*:  TSNPSE on the pyloric stomatogastric-ganglion simulator
(31 parameters -> 18 summary statistics) using

    * the VP SDE (paper recommends VP for high-dimensional problems),
    * 9 sequential rounds with ``30000`` initial simulations and
      ``20000`` additional simulations per round,
    * the "invalid summary statistic" replacement of Deistler et al. (2022a)
      (each non-finite statistic is set two standard deviations below its
      prior-predictive mean, Appendix E.2).

Reported diagnostics (Section 5.3 / Figures 4, 7, 8):

    * fraction of valid summary statistics per round (expected to climb to
      roughly 81% in the final round),
    * posterior predictive summary statistics compared with the observation,
    * posterior marginals (saved for plotting alongside Deistler 2022a /
      Gloeckler 2022),
    * simulation-based calibrated coverage (SBCC): the empirical coverage
      should match the nominal confidence level, in particular at high
      confidence levels.

This module is a *driver*: all algorithmic pieces live in
``snpse/snpse/*`` and ``snpse/tasks/pyloric.py``.  It is written so that it
degrades gracefully when the mackelab simulator or matplotlib are missing.

Usage::

    python experiments/run_pyloric.py --num-rounds 9 \\
        --initial-simulations 30000 --simulations-per-round 20000 \\
        --num-samples 10000 --output results/pyloric.json

    # quick smoke test
    python experiments/run_pyloric.py --num-rounds 2 \\
        --initial-simulations 300 --simulations-per-round 200 --quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Import plumbing: allow both `python -m snpse.experiments.run_pyloric` and
# `python experiments/run_pyloric.py` from the repository root.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repository root (contains the `snpse` package)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _import_first(candidates: Sequence[Tuple[str, str]]) -> Any:
    """Import the first ``(module, attribute)`` pair that resolves."""
    last_error: Optional[BaseException] = None
    for module_name, attribute in candidates:
        try:
            module = __import__(module_name, fromlist=[attribute])
        except Exception as exc:  # pragma: no cover - import fallbacks
            last_error = exc
            continue
        if hasattr(module, attribute):
            return getattr(module, attribute)
    if last_error is not None:
        raise ImportError(f"could not import any of {[c[0] for c in candidates]}: {last_error}")
    raise ImportError(f"could not import {[c[1] for c in candidates]}")


def _import_module(candidates: Sequence[str]) -> Any:
    last_error: Optional[BaseException] = None
    for name in candidates:
        try:
            return __import__(name, fromlist=["*"])
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise ImportError(f"could not import any of {list(candidates)}: {last_error}")


# --- pyloric task -----------------------------------------------------------
try:  # pragma: no cover - depends on layout
    from snpse.tasks.pyloric import (  # type: ignore
        PYLORIC_INITIAL_SIMULATIONS,
        PYLORIC_N_PARAMETERS,
        PYLORIC_N_SUMMARY_STATS,
        PYLORIC_NUM_ROUNDS,
        PYLORIC_SDE,
        PYLORIC_SIMULATIONS_PER_ROUND,
        InvalidStatisticReplacement,
        PyloricSimulator,
        PyloricTask,
        get_pyloric_task,
        load_pyloric_dataset,
        pyloric_tsnpse_config,
        round_budgets,
        sbcc_coverage,
        valid_fraction_history,
    )
except Exception:  # pragma: no cover
    _pyl = _import_module(["snpse.tasks.pyloric", "tasks.pyloric", "pyloric"])
    PYLORIC_INITIAL_SIMULATIONS = getattr(_pyl, "PYLORIC_INITIAL_SIMULATIONS", 30000)
    PYLORIC_SIMULATIONS_PER_ROUND = getattr(_pyl, "PYLORIC_SIMULATIONS_PER_ROUND", 20000)
    PYLORIC_NUM_ROUNDS = getattr(_pyl, "PYLORIC_NUM_ROUNDS", 9)
    PYLORIC_N_PARAMETERS = getattr(_pyl, "PYLORIC_N_PARAMETERS", 31)
    PYLORIC_N_SUMMARY_STATS = getattr(_pyl, "PYLORIC_N_SUMMARY_STATS", 18)
    PYLORIC_SDE = getattr(_pyl, "PYLORIC_SDE", "vp")
    get_pyloric_task = getattr(_pyl, "get_pyloric_task")
    load_pyloric_dataset = getattr(_pyl, "load_pyloric_dataset")
    pyloric_tsnpse_config = getattr(_pyl, "pyloric_tsnpse_config")
    round_budgets = getattr(_pyl, "round_budgets")
    valid_fraction_history = getattr(_pyl, "valid_fraction_history", None)
    sbcc_coverage = getattr(_pyl, "sbcc_coverage", None)
    InvalidStatisticReplacement = getattr(_pyl, "InvalidStatisticReplacement", None)
    PyloricTask = getattr(_pyl, "PyloricTask", None)
    PyloricSimulator = getattr(_pyl, "PyloricSimulator", None)

# --- TSNPSE driver ----------------------------------------------------------
try:  # pragma: no cover
    from snpse.snpse.tsnpse import TSNPSEConfig, run_tsnpse  # type: ignore
except Exception:  # pragma: no cover
    try:
        from snpse.tsnpse import TSNPSEConfig, run_tsnpse  # type: ignore
    except Exception:
        _tsnpse = _import_module(["snpse.snpse.tsnpse", "snpse.tsnpse", "tsnpse"])
        TSNPSEConfig = getattr(_tsnpse, "TSNPSEConfig")
        run_tsnpse = getattr(_tsnpse, "run_tsnpse")

# --- NPSE (optional round-1 baseline) --------------------------------------
try:  # pragma: no cover
    from snpse.snpse.npse import NPSE, NPSEConfig, run_npse  # type: ignore
except Exception:  # pragma: no cover
    try:
        from snpse.npse import NPSE, NPSEConfig, run_npse  # type: ignore
    except Exception:  # pragma: no cover
        NPSE = NPSEConfig = run_npse = None  # type: ignore

# --- misc helpers -----------------------------------------------------------
try:  # pragma: no cover
    from snpse.snpse.utils import ProgressBar, get_device, set_seed, to_tensor  # type: ignore
except Exception:  # pragma: no cover
    _utils = _import_module(["snpse.snpse.utils", "snpse.utils", "utils"])

    def set_seed(seed, deterministic=False):  # type: ignore
        if seed is None:
            return None
        import random

        random.seed(seed)
        torch.manual_seed(seed)
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen

    def get_device(device=None):  # type: ignore
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(value, dtype=None, device=None):  # type: ignore
        if isinstance(value, torch.Tensor):
            t = value
        else:
            t = torch.as_tensor(value, dtype=torch.float32)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t

    class ProgressBar:  # type: ignore
        def __init__(self, total=None, desc="", enabled=True):
            self.total, self.desc, self.enabled = total, desc, enabled
            self.n = 0

        def update(self, n=1):
            self.n += n
            if self.enabled:
                print(f"{self.desc} [{self.n}/{self.total}]", flush=True)

        def set_postfix(self, **kwargs):
            if self.enabled and kwargs:
                print(f"{self.desc} {kwargs}", flush=True)

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False


__all__ = [
    "PyloricRunConfig",
    "run_pyloric",
    "posterior_predictive",
    "compute_coverage",
    "summarise_marginals",
    "save_results",
    "load_results",
    "plot_results",
    "main",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PyloricRunConfig:
    """Configuration for one pyloric TSNPSE run (Section 5.3)."""

    method: str = "tsnpse"
    sde: str = "vp"  # paper uses the VP SDE for this high-dimensional task
    num_rounds: int = PYLORIC_NUM_ROUNDS
    initial_simulations: int = PYLORIC_INITIAL_SIMULATIONS
    simulations_per_round: int = PYLORIC_SIMULATIONS_PER_ROUND
    num_samples: int = 10000
    seed: int = 0
    max_iters: int = 3000
    hidden_dim: int = 256
    n_layers: int = 3
    batch_size: Optional[int] = None
    eps: float = 5e-4
    n_hpr_samples: int = 20000
    device: Optional[str] = None
    observation_path: Optional[str] = None
    observation: Optional[str] = None  # "default" | "file"
    posterior_predictive_samples: int = 500
    coverage: bool = True
    coverage_posterior_draws: int = 200
    coverage_reference_pool: int = 1000
    coverage_levels: Tuple[float, ...] = tuple(
        float(x) for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    )
    track_history: bool = True
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def budgets(self) -> List[int]:
        """Per-round simulation budgets ``[B0, B1, ..., B_{R-1}]``."""
        try:
            return list(round_budgets(self.initial_simulations, self.simulations_per_round, self.num_rounds))
        except Exception:
            return [self.initial_simulations] + [self.simulations_per_round] * (self.num_rounds - 1)

    def total_simulations(self) -> int:
        return int(sum(self.budgets()))

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["budgets"] = self.budgets()
        out["total_simulations"] = self.total_simulations()
        out["coverage_levels"] = list(self.coverage_levels)
        return out


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _call_filtered(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Call ``fn`` keeping only the keyword arguments it accepts.

    Keeps the driver robust against the (slightly different) signatures of the
    TSNPSE config/driver across versions of the code base.
    """
    try:
        import inspect

        sig = inspect.signature(fn)
        accepted = set(sig.parameters)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(*args, **kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return fn(*args, **kwargs)


def _to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype)
    try:
        return to_tensor(value, dtype=dtype)  # type: ignore[misc]
    except Exception:
        return torch.as_tensor(value, dtype=dtype)


def _extract_round_budget_chunks(record: Sequence[Dict[str, int]], budgets: Sequence[int]) -> List[List[Dict[str, int]]]:
    """Group simulator-call records into per-round chunks.

    The initial round uses a pre-generated (cached) data set, so the simulator
    is called once per subsequent round with exactly ``budgets[r]`` accepted
    parameter draws.  If the recorded totals match the expected ones we chunk
    accordingly; otherwise a single chunk holding everything is returned.
    """
    expected = list(budgets[1:])
    chunks: List[List[Dict[str, int]]] = []
    idx = 0
    for want in expected:
        got, chunk = 0, []
        while idx < len(record) and got < want:
            item = record[idx]
            chunk.append(item)
            got += int(item.get("n", 0))
            idx += 1
        chunks.append(chunk)
        if idx >= len(record) and len(chunks) < len(expected):
            chunks.extend([[] for _ in range(len(expected) - len(chunks))])
            break
    if len(chunks) != len(expected) or any(len(c) == 0 for c in chunks):
        # unexpected call pattern: fall back to treating everything as one block
        return [list(record)]
    return chunks


def _call_with_kwargs(fn: Callable[..., Any], **kwargs) -> Any:
    return _call_filtered(fn, **kwargs)


# ---------------------------------------------------------------------------
# Simulator wrapper that records the fraction of valid summary statistics
# ---------------------------------------------------------------------------
def make_counting_simulator(task: Any, record: List[Dict[str, int]]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap the pyloric simulator to count valid/invalid summary statistics.

    ``PyloricSimulator.simulate`` returns ``NaN`` rows for parameter values
    outside the valid regime; the task-level ``simulate`` then applies the
    invalid-statistic replacement (mean - 2 * std of the prior predictive).
    We call the raw simulator once, count validity, and finally hand the
    replacement-applied statistics to the training loop.
    """
    simulator = getattr(task, "simulator", None)
    raw_simulate = getattr(simulator, "simulate", None)
    replacement = getattr(task, "replacement", None)
    apply_replacement = getattr(replacement, "apply", None)
    task_simulate = getattr(task, "simulate", None)

    def fn(theta, *args, **kwargs):  # noqa: ANN001 - mirrors simulator API
        theta_t = _to_tensor(theta)
        x = None
        if raw_simulate is not None:
            try:
                x = _to_tensor(raw_simulate(theta_t))
            except Exception:
                x = None
        if x is None:
            if task_simulate is None:
                raise RuntimeError("no pyloric simulator available on the task object")
            x = _to_tensor(task_simulate(theta_t))
            finite = torch.isfinite(x)
            valid = finite.all(dim=-1)
            record.append(
                {
                    "n": int(theta_t.shape[0]),
                    "n_valid": int(valid.sum().item()),
                    "n_invalid": int((~valid).sum().item()),
                    "n_invalid_stats": int((~finite).sum().item()),
                }
            )
            return x

        finite = torch.isfinite(x)
        valid = finite.all(dim=-1)
        record.append(
            {
                "n": int(theta_t.shape[0]),
                "n_valid": int(valid.sum().item()),
                "n_invalid": int((~valid).sum().item()),
                "n_invalid_stats": int((~finite).sum().item()),
            }
        )
        if apply_replacement is not None:
            try:
                x = _to_tensor(apply_replacement(x))
            except Exception:  # pragma: no cover
                pass
        return x

    return fn


def _valid_fraction_from_records(chunks: Sequence[Sequence[Dict[str, int]]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for chunk in chunks:
        n = sum(int(c.get("n", 0)) for c in chunk)
        n_valid = sum(int(c.get("n_valid", 0)) for c in chunk)
        out.append(float(n_valid) / float(n) if n > 0 else None)
    return out


# ---------------------------------------------------------------------------
# Posterior predictive / marginals / coverage diagnostics
# ---------------------------------------------------------------------------
def posterior_predictive(
    task: Any,
    theta: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Simulate summary statistics from posterior draws and compare to ``x_obs``."""
    theta = _to_tensor(theta)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    if max_samples is not None and theta.shape[0] > max_samples:
        idx = torch.randperm(theta.shape[0], generator=generator)[:max_samples]
        theta = theta[idx]

    simulator = getattr(task, "simulator", None)
    simulate = getattr(simulator, "simulate", None)
    if simulate is None:
        simulate = getattr(task, "simulate")

    with torch.no_grad():
        x = _to_tensor(simulate(theta))

    replacement = getattr(task, "replacement", None)
    apply_replacement = getattr(replacement, "apply", None)
    if apply_replacement is not None:
        try:
            x = _to_tensor(apply_replacement(x))
        except Exception:  # pragma: no cover
            pass

    x_obs = observation(task)
    finite = torch.isfinite(x)
    out: Dict[str, torch.Tensor] = {
        "theta": theta,
        "x": x,
        "x_obs": x_obs.reshape(-1),
        "valid": finite.all(dim=-1),
    }
    try:
        mean = torch.nanmean(x, dim=0)
        std = torch.nanstd(x, dim=0)
        out["x_mean"] = mean
        out["x_std"] = std
        obs = x_obs.reshape(-1)
        diff = (mean - obs) / torch.clamp(std, min=1e-12)
        out["z_score"] = diff
        out["z_abs_mean"] = torch.nanmean(torch.abs(diff)).reshape(1)
        out["rmse"] = torch.sqrt(torch.nanmean((mean - obs) ** 2)).reshape(1)
    except Exception:  # pragma: no cover
        pass
    return out


def compute_coverage(
    task: Any,
    posterior_samples: torch.Tensor,
    levels: Sequence[float],
    num_posterior_draws: int = 200,
    reference_pool: int = 1000,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Simulation-based calibrated coverage (SBCC, Figure 8).

    For each of ``num_posterior_draws`` parameter draws ``theta*`` from the
    posterior, summarise ``x* ~ p(x | theta*)`` and rank the distance
    ``d(x*, x_obs)`` against the distances ``d(x*, x_l)`` where ``x_l`` are
    summary statistics simulated from a shared pool of posterior draws
    (an amortised-posterior approximation of exact SBCC, which would re-sample
    the posterior for every ``theta*``).  Empirical coverage should track the
    nominal confidence level, especially for high levels.
    """
    theta = _to_tensor(posterior_samples)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    n_pool = min(int(reference_pool), int(theta.shape[0]))
    n_draws = min(int(num_posterior_draws), int(theta.shape[0]))
    if n_pool < 10 or n_draws < 10:
        return {"available": False, "reason": "too few posterior samples"}

    perm = torch.randperm(theta.shape[0], generator=generator)
    pool_theta = theta[perm[:n_pool]]
    draw_theta = theta[perm[:n_draws]]

    x_obs = observation(task).reshape(-1)
    pool = posterior_predictive(task, pool_theta, generator=generator)
    x_pool = pool["x"]
    sigma = pool.get("x_std", torch.ones_like(x_obs))

    draws = posterior_predictive(task, draw_theta, generator=generator)
    x_draw = draws["x"]

    # normalise statistics by the posterior-predictive scale of the reference pool
    scale = torch.clamp(sigma, min=1e-12)

    with torch.no_grad():
        d_obs = torch.sqrt(torch.nanmean(((x_draw - x_obs.reshape(1, -1)) / scale) ** 2, dim=1))
        d_ref = torch.sqrt(
            torch.nanmean(((x_draw.unsqueeze(1) - x_pool.unsqueeze(0)) / scale.reshape(1, 1, -1)) ** 2, dim=2)
        )
        # "central" interval: reference distances inside the level's quantile
        ranks = (d_ref < d_obs.unsqueeze(1)).float().mean(dim=1)  # in [0, 1]
        central = (2 * torch.minimum(ranks, 1.0 - ranks)).clamp(0.0, 1.0)
        central = torch.nan_to_num(central, nan=0.0)

        empirical, nominal = [], []
        for level in levels:
            level = float(level)
            empirical.append(float((central <= level).float().mean().item()))
            nominal.append(level)

    coverage_error = [abs(e - n) for e, n in zip(empirical, nominal)]
    return {
        "available": True,
        "nominal": nominal,
        "empirical": empirical,
        "error": coverage_error,
        "mean_error": float(sum(coverage_error) / max(len(coverage_error), 1)),
        "high_level_error": float(
            sum(
                abs(e - n)
                for e, n in zip(empirical, nominal)
                if n >= 0.8
            )
            / max(sum(1 for n in nominal if n >= 0.8), 1)
        ),
        "n_posterior_draws": int(n_draws),
        "n_reference_pool": int(n_pool),
        "distance": "normalised mean squared distance",
    }


def sbcc_via_task(
    task: Any,
    posterior_samples: torch.Tensor,
    levels: Sequence[float],
    num_posterior_draws: int,
    reference_pool: int,
    generator: Optional[torch.Generator],
) -> Dict[str, Any]:
    """Try the task-provided ``sbcc_coverage`` helper, else use the local one."""
    if sbcc_coverage is not None:
        theta = _to_tensor(posterior_samples)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)

        def sample_posterior(n, generator=None):  # noqa: ANN001
            n = int(n)
            idx = torch.randint(theta.shape[0], (n,), generator=generator)
            return theta[idx]

        try:
            return _call_filtered(
                sbcc_coverage,
                run_simulator=getattr(task, "simulate", None),
                observation=observation(task),
                num_sims=int(num_posterior_draws),
                num_posterior_samples=int(reference_pool),
                sample_posterior=sample_posterior,
                generator=generator,
                distance=None,
                levels=list(levels),
            )
        except Exception as exc:  # pragma: no cover - fall through
            if os.environ.get("SNPSE_PYLORIC_DEBUG"):
                traceback.print_exc()
            _ = exc
    return compute_coverage(
        task,
        posterior_samples,
        levels=levels,
        num_posterior_draws=num_posterior_draws,
        reference_pool=reference_pool,
        generator=generator,
    )


def summarise_marginals(theta: torch.Tensor, prior: Any = None) -> Dict[str, Any]:
    """Mean/std/quantiles of the posterior marginals (Figure 7)."""
    theta = _to_tensor(theta)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    with torch.no_grad():
        mean = theta.mean(dim=0)
        std = theta.std(dim=0)
        q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
        quantiles = torch.quantile(theta, q, dim=0)
    out: Dict[str, Any] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "quantiles": {str(float(qq)): quantiles[i].tolist() for i, qq in enumerate(q)},
        "n_samples": int(theta.shape[0]),
    }
    bounds = None
    if prior is not None:
        try:
            bounds = prior.bounds()
        except Exception:
            bounds = None
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        try:
            low = _to_tensor(bounds[0]).reshape(-1)
            high = _to_tensor(bounds[1]).reshape(-1)
            width = torch.clamp(high - low, min=1e-12)
            out["prior_coverage"] = ((mean >= low) & (mean <= high)).float().mean().item()
            out["posterior_prior_width_ratio"] = (std / width).tolist()
        except Exception:  # pragma: no cover
            pass
    return out


def observation(task: Any) -> torch.Tensor:
    """Observed summary statistics as a 1-D tensor."""
    for attr in ("x_obs", "observation", "obs"):
        value = getattr(task, attr, None)
        if value is not None and not callable(value):
            return _to_tensor(value).reshape(-1)
    method = getattr(task, "observation", None)
    if callable(method):
        return _to_tensor(method()).reshape(-1)
    method = getattr(task, "default_observation", None)
    if callable(method):
        return _to_tensor(method()).reshape(-1)
    raise RuntimeError("could not obtain the pyloric observation x_obs")


def _load_observation(task: Any, config: PyloricRunConfig) -> Tuple[torch.Tensor, str]:
    """Observation from file (if given), else the task default."""
    path = config.observation_path or config.observation
    if path and path not in ("default", "task"):
        candidates = [path]
        if not os.path.isabs(path):
            candidates.append(os.path.join(_ROOT, "tasks", "data", path))
            candidates.append(os.path.join(_HERE, path))
        for candidate in candidates:
            if os.path.isfile(candidate):
                if candidate.endswith(".pt"):
                    x_obs = torch.load(candidate, map_location="cpu")
                    return _to_tensor(x_obs).reshape(-1), candidate
                import numpy as np  # noqa: PLC0415

                return _to_tensor(torch.as_tensor(np.load(candidate))).reshape(-1), candidate
        if config.verbose:
            print(f"[pyloric] observation file '{path}' not found; using the task default", flush=True)
    return observation(task), "task_default"


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run_pyloric(config: Optional[PyloricRunConfig] = None, **overrides: Any) -> Dict[str, Any]:
    """Run the pyloric TSNPSE experiment and return a results dictionary."""
    if config is None:
        config = PyloricRunConfig(**overrides)
    elif overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    device = get_device(config.device)
    generator = set_seed(config.seed) if config.seed is not None else None
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(int(config.seed or 0))

    t_start = time.time()
    result: Dict[str, Any] = {
        "config": config.as_dict(),
        "device": str(device),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- task ---------------------------------------------------------------
    task = get_pyloric_task() if callable(get_pyloric_task) else PyloricTask()
    if config.verbose:
        print(
            f"[pyloric] task ready: d_theta={getattr(task, 'dim_theta', PYLORIC_N_PARAMETERS)}, "
            f"d_x={getattr(task, 'dim_x', PYLORIC_N_SUMMARY_STATS)}, "
            f"backend={getattr(task, 'backend', getattr(getattr(task, 'simulator', None), 'backend', 'unknown'))}",
            flush=True,
        )

    # fit the invalid-statistic replacement on the prior predictive if needed
    ensure = getattr(task, "ensure_replacement", None)
    if callable(ensure):
        try:
            _call_filtered(ensure, n=max(1000, min(10000, config.initial_simulations // 3)), generator=generator)
        except Exception:  # pragma: no cover
            if config.verbose:
                traceback.print_exc()

    x_obs, obs_source = _load_observation(task, config)
    result["observation_source"] = obs_source
    result["x_obs"] = x_obs.tolist()

    # --- simulation budgets and initial (cached) data set --------------------
    budgets = config.budgets()
    result["budgets"] = budgets
    result["total_simulations"] = int(sum(budgets))

    initial_theta = initial_x = None
    dataset_info: Dict[str, Any] = {}
    try:
        dataset = _call_filtered(
            load_pyloric_dataset,
            num_simulations=int(budgets[0]),
            task=task,
            seed=int(config.seed or 0),
            generator=generator,
            return_valid=True,
            verbose=config.verbose,
        )
        if isinstance(dataset, dict):
            initial_theta = dataset.get("theta")
            initial_x = dataset.get("x")
            dataset_info = {
                k: (v.tolist() if isinstance(v, torch.Tensor) and v.numel() < 5000 else None)
                for k, v in dataset.items()
                if k != "theta" and k != "x"
            }
            valid = dataset.get("valid")
            if valid is not None:
                try:
                    dataset_info["initial_valid_fraction"] = float(_to_tensor(valid).float().mean().item())
                except Exception:
                    pass
    except Exception:  # pragma: no cover - the driver generates its own data
        if config.verbose:
            traceback.print_exc()
    result["dataset_info"] = dataset_info

    # --- TSNPSE configuration ----------------------------------------------
    ts_config = None
    if config.verbose:
        print(
            f"[pyloric] TSNPSE: {config.num_rounds} rounds, budgets={budgets}, "
            f"sde={config.sde}, num_samples={config.num_samples}",
            flush=True,
        )
    try:
        ts_config = _call_filtered(
            pyloric_tsnpse_config,
            num_rounds=int(config.num_rounds),
            initial_budget=int(budgets[0]),
            simulations_per_round=int(budgets[1] if len(budgets) > 1 else 0),
            batch_size=config.batch_size,
            sde=config.sde,
            num_samples=int(config.num_samples),
            max_iters=int(config.max_iters),
            hidden_dim=int(config.hidden_dim),
            n_layers=int(config.n_layers),
            eps=float(config.eps),
            n_hpr_samples=int(config.n_hpr_samples),
            seed=int(config.seed or 0),
            device=str(device),
            verbose=config.verbose,
            **config.extra,
        )
    except Exception:  # pragma: no cover - build a config by hand
        if config.verbose:
            traceback.print_exc()
        ts_config = TSNPSEConfig(
            sde=config.sde,
            num_rounds=int(config.num_rounds),
            initial_budget=int(budgets[0]),
            simulations_per_round=int(budgets[1] if len(budgets) > 1 else 0),
            num_samples=int(config.num_samples),
            eps=float(config.eps),
            n_hpr_samples=int(config.n_hpr_samples),
            seed=int(config.seed or 0),
        )
    result["config"]["tsnpse"] = ts_config.as_dict() if hasattr(ts_config, "as_dict") else str(ts_config)

    # --- run ----------------------------------------------------------------
    record: List[Dict[str, int]] = []
    simulator = make_counting_simulator(task, record)
    prior = getattr(task, "prior", None)

    kwargs: Dict[str, Any] = {
        "prior": prior,
        "simulator": simulator,
        "x_obs": x_obs,
        "theta_dim": int(getattr(task, "dim_theta", PYLORIC_N_PARAMETERS)),
        "x_dim": int(getattr(task, "dim_x", PYLORIC_N_SUMMARY_STATS)),
        "config": ts_config,
        "num_samples": int(config.num_samples),
        "with_log_prob": True,
        "seed": int(config.seed or 0),
        "device": str(device),
        "generator": generator,
        "verbose": config.verbose,
        "initial_theta": initial_theta,
        "initial_x": initial_x,
    }
    run_output = _call_filtered(run_tsnpse, **kwargs)
    if not isinstance(run_output, dict):  # pragma: no cover - defensive
        run_output = {"theta": _to_tensor(run_output)}

    posterior = _to_tensor(run_output.get("theta"))
    if posterior is None or posterior.numel() == 0:
        raise RuntimeError("TSNPSE returned no posterior samples")
    result["posterior_samples"] = int(posterior.shape[0])
    result["posterior_mean"] = posterior.mean(dim=0).tolist()

    # --- valid-statistic fraction per round (Figure 4c) ---------------------
    chunks = _extract_round_budget_chunks(record, budgets)
    per_round_valid = _valid_fraction_from_records(chunks)
    history: Optional[List[float]] = None
    if len(per_round_valid) == max(len(budgets) - 1, 1):
        history = [None] + per_round_valid  # round 1 uses the cached data set
        if dataset_info.get("initial_valid_fraction") is not None:
            history[0] = float(dataset_info["initial_valid_fraction"])
    else:
        history = per_round_valid
    result["valid_fraction_per_round"] = history
    result["simulator_calls"] = {
        "n_calls": len(record),
        "n_simulations": int(sum(int(r.get("n", 0)) for r in record)),
        "n_valid": int(sum(int(r.get("n_valid", 0)) for r in record)),
        "n_invalid": int(sum(int(r.get("n_invalid", 0)) for r in record)),
        "n_invalid_stats": int(sum(int(r.get("n_invalid_stats", 0)) for r in record)),
    }
    final_valid = None
    for value in reversed(history or []):
        if value is not None:
            final_valid = float(value)
            break
    if final_valid is None:
        tot = result["simulator_calls"]["n_simulations"]
        final_valid = (
            float(result["simulator_calls"]["n_valid"]) / float(tot) if tot else None
        )
    result["final_valid_fraction"] = final_valid
    if callable(valid_fraction_history) and history:
        try:
            result["valid_fraction_history"] = _call_filtered(
                valid_fraction_history,
                valid_per_round=[0.0 if v is None else v for v in history],
                budgets=budgets,
            )
        except Exception:  # pragma: no cover
            pass
    if config.verbose and final_valid is not None:
        print(
            f"[pyloric] valid summary statistics in the final round: {100.0 * final_valid:.1f}% "
            f"(paper reports ~81%)",
            flush=True,
        )

    # --- posterior predictive (Figure 7) ------------------------------------
    pred = posterior_predictive(
        task,
        posterior,
        generator=generator,
        max_samples=int(config.posterior_predictive_samples),
    )
    result["posterior_predictive"] = {
        "n": int(pred["x"].shape[0]),
        "valid_fraction": float(pred["valid"].float().mean().item()),
        "x_mean": pred.get("x_mean").tolist() if "x_mean" in pred else None,
        "x_obs": pred["x_obs"].tolist(),
        "z_abs_mean": float(pred["z_abs_mean"].item()) if "z_abs_mean" in pred else None,
        "rmse": float(pred["rmse"].item()) if "rmse" in pred else None,
    }
    result["marginals"] = summarise_marginals(posterior, prior=prior)

    # --- SBCC coverage (Figure 8) -------------------------------------------
    if config.coverage:
        try:
            coverage = sbcc_via_task(
                task,
                posterior,
                levels=config.coverage_levels,
                num_posterior_draws=int(config.coverage_posterior_draws),
                reference_pool=int(config.coverage_reference_pool),
                generator=generator,
            )
            result["coverage"] = coverage
            if config.verbose and coverage.get("available"):
                print(
                    "[pyloric] SBCC mean |empirical - nominal| = "
                    f"{coverage.get('mean_error', float('nan')):.3f} "
                    f"(high-level: {coverage.get('high_level_error', float('nan')):.3f})",
                    flush=True,
                )
        except Exception:  # pragma: no cover
            traceback.print_exc()
            result["coverage"] = {"available": False, "reason": "coverage computation failed"}

    # --- training history ---------------------------------------------------
    if config.track_history:
        run_history = run_output.get("history")
        if run_history is not None:
            if isinstance(run_history, (list, tuple)):
                result["round_history"] = [
                    (h.as_dict() if hasattr(h, "as_dict") else h) for h in run_history
                ]
            else:
                result["round_history"] = (
                    run_history.as_dict() if hasattr(run_history, "as_dict") else str(run_history)
                )
        regions = run_output.get("regions")
        if regions is not None:
            try:
                result["n_hpr_regions"] = int(len(regions))
            except Exception:  # pragma: no cover
                pass

    # --- persist arrays -----------------------------------------------------
    arrays_path = os.path.join(
        os.path.dirname(os.path.abspath(config.extra.get("arrays_path", "results/pyloric.npz"))),
        os.path.basename(config.extra.get("arrays_path", "pyloric.npz")),
    )
    try:
        arrays_path = config.extra.get("arrays_path", arrays_path)
        if arrays_path:
            os.makedirs(os.path.dirname(os.path.abspath(arrays_path)) or ".", exist_ok=True)
            torch.save(
                {
                    "theta": posterior.detach().cpu(),
                    "x_obs": x_obs.detach().cpu(),
                    "posterior_predictive_x": pred["x"].detach().cpu(),
                    "posterior_predictive_theta": pred["theta"].detach().cpu(),
                    "log_prob": _to_tensor(run_output["log_prob"]).detach().cpu()
                    if run_output.get("log_prob") is not None
                    else None,
                },
                arrays_path,
            )
            result["arrays_path"] = arrays_path
    except Exception:  # pragma: no cover
        if config.verbose:
            traceback.print_exc()

    result["runtime"] = float(time.time() - t_start)
    result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


# ---------------------------------------------------------------------------
# Persistence / plotting
# ---------------------------------------------------------------------------
def save_results(results: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=_json_default)
    return path


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return list(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def plot_results(results: Dict[str, Any], path: str) -> Optional[str]:
    """Figure 4c-style valid-fraction curve plus coverage / marginal panels."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:  # pragma: no cover
        return None

    config = results.get("config", {})
    budgets = results.get("budgets") or []
    valid = results.get("valid_fraction_per_round") or []

    n_panels = 3 if results.get("coverage", {}).get("available") else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    # valid fraction per round
    ax = axes[0]
    xs = list(range(1, len(valid) + 1))
    ys = [100.0 * v if v is not None else None for v in valid]
    ax.plot(xs, ys, marker="o", color="tab:blue")
    ax.axhline(81.0, ls="--", color="tab:red", label="paper (~81%)")
    ax.set_xlabel("round")
    ax.set_ylabel("valid summary statistics [%]")
    ax.set_title("Pyloric: valid statistics per round")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # posterior predictive vs observation
    ax = axes[1]
    obs = results.get("x_obs") or []
    pred = results.get("posterior_predictive", {})
    mean = pred.get("x_mean") or []
    if obs and mean and len(obs) == len(mean):
        idx = list(range(len(obs)))
        ax.plot(idx, obs, marker="o", ls="none", color="black", label="observation")
        ax.plot(idx, mean, marker="x", ls="none", color="tab:orange", label="posterior predictive mean")
        ax.set_xlabel("summary statistic")
        ax.set_ylabel("value")
        ax.set_title("Posterior predictive")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "posterior predictive unavailable", ha="center", va="center")

    # coverage
    if n_panels == 3:
        ax = axes[2]
        cov = results.get("coverage", {})
        nominal, empirical = cov.get("nominal", []), cov.get("empirical", [])
        ax.plot([0, 1], [0, 1], ls="--", color="grey", label="ideal")
        if nominal and empirical:
            ax.plot(nominal, empirical, marker="o", color="tab:green")
        ax.set_xlabel("nominal confidence level")
        ax.set_ylabel("empirical coverage")
        ax.set_title("SBCC coverage")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"SNPSE / TSNPSE - pyloric ({config.get('method', 'tsnpse')}, "
        f"{config.get('sde', 'vp').upper()} SDE, {config.get('num_rounds', '?')} rounds, "
        f"{results.get('total_simulations', '?')} simulations)"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pyloric (neuroscience) experiment for SNPSE/TSNPSE (Section 5.3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", default="tsnpse", choices=["tsnpse", "npse"])
    parser.add_argument("--sde", default=PYLORIC_SDE, choices=["ve", "vp"])
    parser.add_argument("--num-rounds", type=int, default=PYLORIC_NUM_ROUNDS)
    parser.add_argument("--initial-simulations", type=int, default=PYLORIC_INITIAL_SIMULATIONS)
    parser.add_argument("--simulations-per-round", type=int, default=PYLORIC_SIMULATIONS_PER_ROUND)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eps", type=float, default=5e-4)
    parser.add_argument("--n-hpr-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--observation", default=None, help="path to x_obs (.pt/.npy) or 'default'")
    parser.add_argument("--posterior-predictive-samples", type=int, default=500)
    parser.add_argument("--coverage", dest="coverage", action="store_true", default=True)
    parser.add_argument("--no-coverage", dest="coverage", action="store_false")
    parser.add_argument("--coverage-draws", type=int, default=200)
    parser.add_argument("--coverage-pool", type=int, default=1000)
    parser.add_argument("--output", default="results/pyloric.json")
    parser.add_argument("--figure", default="results/pyloric.png")
    parser.add_argument("--arrays", default="results/pyloric_arrays.pt")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    num_rounds = args.num_rounds
    initial = args.initial_simulations
    per_round = args.simulations_per_round
    num_samples = args.num_samples
    max_iters = args.max_iters
    if args.quick:
        num_rounds = min(num_rounds, 2)
        initial = min(initial, 300)
        per_round = min(per_round, 200)
        num_samples = min(num_samples, 500)
        max_iters = min(max_iters, 200)

    config = PyloricRunConfig(
        method=args.method,
        sde=args.sde,
        num_rounds=num_rounds,
        initial_simulations=initial,
        simulations_per_round=per_round,
        num_samples=num_samples,
        seed=args.seed,
        max_iters=max_iters,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        eps=args.eps,
        n_hpr_samples=args.n_hpr_samples,
        device=args.device,
        observation_path=args.observation,
        posterior_predictive_samples=args.posterior_predictive_samples,
        coverage=args.coverage,
        coverage_posterior_draws=args.coverage_draws,
        coverage_reference_pool=args.coverage_pool,
        verbose=not args.quiet,
        extra={"arrays_path": args.arrays},
    )

    try:
        results = run_pyloric(config)
    except Exception:  # pragma: no cover - report and fail loudly
        traceback.print_exc()
        return 1

    save_results(results, args.output)
    figure = plot_results(results, args.figure) if args.figure else None

    summary = {
        "method": results["config"]["method"],
        "sde": results["config"]["sde"],
        "rounds": results["config"]["num_rounds"],
        "total_simulations": results.get("total_simulations"),
        "final_valid_fraction": results.get("final_valid_fraction"),
        "posterior_samples": results.get("posterior_samples"),
        "posterior_predictive_z_abs_mean": results.get("posterior_predictive", {}).get("z_abs_mean"),
        "coverage_mean_error": results.get("coverage", {}).get("mean_error"),
        "valid_fraction_per_round": results.get("valid_fraction_per_round"),
        "runtime": results.get("runtime"),
        "output": args.output,
        "figure": figure,
    }
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
