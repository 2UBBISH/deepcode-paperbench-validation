"""Shared helpers for the benchmark experiments (Section 5.2, Appendix E.1)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


# --------------------------------------------------------------------------- #
# Benchmark task registry (Appendix E.1)
# --------------------------------------------------------------------------- #


@dataclass
class TaskSpec:
    name: str  # sbibm task name
    display: str
    sigma_min: float  # Appendix E.3.1
    dim_parameters: int
    dim_data: int


# sigma_min = 0.01 for the two-dimensional experiments (SIR and Two Moons) and
# 0.05 for all other experiments (Appendix E.3.1).
TASK_SPECS: Dict[str, TaskSpec] = {
    "gaussian_linear": TaskSpec("gaussian_linear", "Gaussian Linear", 0.05, 10, 10),
    "gaussian_mixture": TaskSpec("gaussian_mixture", "Gaussian Mixture", 0.05, 2, 2),
    "two_moons": TaskSpec("two_moons", "Two Moons", 0.01, 2, 2),
    "gaussian_linear_uniform": TaskSpec(
        "gaussian_linear_uniform", "Gaussian Linear Uniform", 0.05, 10, 10
    ),
    "bernoulli_glm": TaskSpec("bernoulli_glm", "Bernoulli GLM", 0.05, 10, 10),
    "slcp": TaskSpec("slcp", "SLCP", 0.05, 5, 8),
    "sir": TaskSpec("sir", "SIR", 0.01, 2, 10),
    "lotka_volterra": TaskSpec("lotka_volterra", "Lotka Volterra", 0.05, 4, 20),
}


def get_task(name: str):
    """Load an ``sbibm`` task (the addendum requires using the sbibm library)."""
    from sbibm.tasks import get_task as _get_task

    return _get_task(name)


def get_observation(task, num_observation: int = 1) -> torch.Tensor:
    return task.get_observation(num_observation).reshape(1, -1)


def get_reference_posterior_samples(task, num_observation: int = 1, num_samples: Optional[int] = None):
    n = num_samples or task.num_reference_posterior_samples
    return task.get_reference_posterior_samples(num_observation)[:n]


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #


BUDGETS: List[int] = [1000, 10000, 100000]


def batch_size_for(budget: int, sequential: bool) -> int:
    """Batch sizes from Appendix E.3.2.

    For budgets of 1000 or 10000 the batch size is 50 for non-sequential and
    200 for sequential experiments; for a budget of 100000 it is 500 for both.
    """
    if budget <= 10000:
        return 200 if sequential else 50
    return 500


def num_rounds_for(method: str) -> int:
    """Number of rounds R (10 for all sequential experiments by default)."""
    return 1 if method in ("npse", "npe") else 10


# --------------------------------------------------------------------------- #
# Method construction
# --------------------------------------------------------------------------- #


def build_snpse_method(
    method: str,
    task,
    dim_parameters: int,
    dim_data: int,
    budget: int,
    sigma_min: float,
    seed: int = 0,
    device: str = "cpu",
    smoke: bool = False,
    max_iters: Optional[int] = None,
    lr: Optional[float] = None,
    t_scale: Optional[float] = None,
    rounds: Optional[int] = None,
):
    """Instantiate a (TS)NPSE estimator for a given method string.

    Method strings:

    ``npse_ve`` / ``npse_vp``       -- non-sequential NPSE
    ``tsnpse_ve`` / ``tsnpse_vp``   -- Truncated Sequential NPSE
    ``snpse_a`` / ``snpse_b`` / ``snpse_c`` -- alternative sequential methods
    """
    from snpse import NPSE, TSNPSE, TSNPSEConfig, TrainingConfig, VariantConfig
    from snpse.snpse_variants import SNPSEA, SNPSEB, SNPSEC

    base = method.split("_")[0]
    parts = method.split("_")
    sde = "ve" if parts[-1] == "ve" else ("vp" if parts[-1] == "vp" else "ve")
    sequential = base in ("tsnpse",)
    variant = base == "snpse"

    tc = TrainingConfig(seed=seed, batch_size=batch_size_for(budget, sequential or variant))
    if max_iters is not None:
        tc.max_iters = max_iters
        # the paper's early-stopping patience is 1000 non-improving steps
        tc.patience = min(tc.patience, max(1, max_iters))
    if lr is not None:
        tc.lr = lr
    if t_scale is not None:
        tc.t_scale = t_scale
    if smoke:
        tc.max_iters = 60
        tc.patience = 20

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    if method.startswith("npse"):
        return (
            NPSE(
                dim_parameters,
                dim_data,
                prior,
                sde=sde,
                sigma_min=sigma_min,
                training_config=tc,
                device=device,
            ),
            "npse",
        )
    if method.startswith("tsnpse"):
        cfg = TSNPSEConfig(
            num_rounds=2 if smoke else (rounds or 10),
            num_simulations=budget,
            hpr_num_samples=2000 if smoke else 20000,
        )
        return (
            TSNPSE(
                dim_parameters,
                dim_data,
                prior,
                simulator,
                sde=sde,
                sigma_min=sigma_min,
                config=cfg,
                training_config=tc,
                device=device,
                seed=seed,
            ),
            "tsnpse",
        )
    if base == "snpse":
        cfg = VariantConfig(
            num_rounds=2 if smoke else (rounds or 10),
            num_simulations=budget,
        )
        cls = {"snpse_a": SNPSEA, "snpse_b": SNPSEB, "snpse_c": SNPSEC}[method]
        return (
            cls(
                dim_parameters,
                dim_data,
                prior,
                simulator,
                sde=sde,
                sigma_min=sigma_min,
                config=cfg,
                training_config=tc,
                device=device,
                seed=seed,
            ),
            method,
        )
    raise ValueError(f"unknown method {method}")


def run_npse(estimator, task, budget: int, verbose: bool = True):
    """Non-sequential NPSE: simulate `budget` samples from the prior."""
    theta = task.get_prior_dist().sample((budget,))
    x = task.get_simulator()(theta)
    estimator.fit(theta, x, verbose=verbose)
    return estimator


def run_sequential(estimator, x_obs: torch.Tensor, verbose: bool = True, smoke: bool = False):
    """Run a sequential estimator (TSNPSE / SNPSE-A/B/C)."""
    if smoke and hasattr(estimator, "config"):
        estimator.config.num_rounds = 2
        estimator.config.hpr_num_samples = getattr(estimator.config, "hpr_num_samples", 0) or 2000
    estimator.run(x_obs, verbose=verbose)
    return estimator, estimator.diagnostics


def draw_posterior_samples(estimator, method: str, x_obs: torch.Tensor, num_samples: int, seed: int = 0):
    """Draw ``num_samples`` posterior samples from a fitted estimator.

    SNPSE-A applies its post-hoc SIR correction at sampling time, so it is
    handled separately from the other methods.
    """
    if method.startswith("npse"):
        return estimator.sample(x_obs, num_samples, seed=seed)
    if method.startswith("tsnpse"):
        return estimator.npse.sample(x_obs, num_samples, seed=seed)
    if method == "snpse_a":
        return estimator.sample(x_obs, num_samples)
    return estimator.npse.sample(x_obs, num_samples, seed=seed)


# --------------------------------------------------------------------------- #
# Result bookkeeping
# --------------------------------------------------------------------------- #


def append_result(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def load_results(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
