"""Smoke test for the pyloric pipeline (Section 5.3) without NEURON.

The real simulator (`mackelab/pyloric`) requires NEURON, which is not available
in this environment.  This test injects a stand-in `pyloric` module exposing the
same three functions used by the adapter -- ``create_prior``, ``simulate`` and
``summary_stats`` -- so that the whole Section 5.3 code path can be exercised:
prior-predictive estimation, replacement of invalid summary statistics, the
non-uniform simulation schedule (30000 initial + 20000 per round) and the
per-round validity bookkeeping used for Figure 4c.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakePrior:
    def __init__(self, dim: int = 4) -> None:
        self.columns = [f"param_{i}" for i in range(dim)]
        self.numerical_prior = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(dim), 1.0), 1
        )

    def sample(self, shape):
        n = shape[0] if isinstance(shape, (tuple, list)) else shape
        return pd.DataFrame(np.random.randn(n, len(self.columns)), columns=self.columns)


def _install_fake_pyloric(dim_summary: int = 18) -> None:
    module = types.ModuleType("pyloric")

    def create_prior():
        return _FakePrior()

    def simulate(theta: pd.DataFrame):
        return theta.to_numpy() @ np.random.randn(len(theta.columns), 3)

    def summary_stats(traces, stats_customization=None):
        x = np.asarray(traces) + 0.1 * np.random.randn(*np.asarray(traces).shape)
        x = np.concatenate([x, x, x, x, x, x], axis=1)[:, :dim_summary]
        # ~20% of the simulations produce invalid (NaN) summary statistics
        x[::5, 0] = np.nan
        return pd.DataFrame(x)

    module.create_prior = create_prior
    module.simulate = simulate
    module.summary_stats = summary_stats
    sys.modules["pyloric"] = module


def test_pyloric_adapter_replaces_invalid_statistics():
    _install_fake_pyloric()
    from baselines.pyloric_adapter import PyloricProblem

    problem = PyloricProblem(dim_summary_statistics=18)
    replacement = problem.fit_prior_predictive(num_samples=50, seed=0)
    assert replacement.shape == (18,)
    stats, valid = problem.simulate_batch(problem.numerical_prior.sample((20,)))
    assert valid.shape == stats.shape
    assert not valid.all()  # the fake simulator produces NaNs
    fixed = problem.simulator(problem.numerical_prior.sample((20,)))
    assert torch.isfinite(fixed).all()
    assert problem.observation.shape == (1, 18)


def test_run_pyloric_end_to_end(tmp_path):
    _install_fake_pyloric()
    from experiments.run_pyloric import main

    out_dir = tmp_path / "pyloric"
    rc = main(
        [
            "--smoke",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    with open(out_dir / "pyloric_results.json") as fh:
        results = json.load(fh)
    assert len(results["fraction_valid_by_round"]) == 2
    assert all(0.0 <= v <= 1.0 for v in results["fraction_valid_by_round"])
    assert results["diagnostics"][0]["num_simulations_total"] == 20
    assert results["diagnostics"][1]["num_simulations_total"] == 40
    assert np.load(out_dir / "posterior_samples.npy").shape == (50, 4)
