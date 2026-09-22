"""Pyloric network simulator used for the neuroscience experiment (Section 5.3).

The addendum of the reproduction task states that the neuroscience problem
should be implemented using ``https://github.com/mackelab/tsnpe_neurips``.
That repository in turn uses the ``pyloric`` Python package
(``https://github.com/mackelab/pyloric``), which provides

* ``create_prior()``  -- the uniform prior over the 31 parameters of the
  pyloric network of the stomatogastric ganglion of *Cancer borealis*
  (Prinz et al., 2003, 2004);
* ``simulate(theta)`` -- the NEURON-based simulator producing 3 voltage traces;
* ``summary_stats(traces)`` -- the 18 summary statistics used as observations.

Observed data are the experimental recordings of Haddad & Marder (2021), as
distributed with the TSNPE repository.

Invalid summary statistics (which occur for more than 99% of prior samples) are
replaced by a value two standard deviations below the prior predictive of the
corresponding summary statistic, following Deistler et al. (2022a) and
Appendix E.2 of the paper.

Install the simulators with::

    bash scripts/setup_third_party.sh
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


#: Experimental observations (Haddad & Marder, 2021), taken from
#: ``l5pc/l5pc/model/utils.py:return_gt`` in the TSNPE repository.
PYLORIC_OBSERVATION_20D = np.array(
    [
        0.026145,
        0.004226,
        0.000143,
        3.137968,
        0.089259,
        0.002910,
        0.006827,
        0.007104,
        0.000990,
        0.973538,
        1.021945,
        287.198731,
        0.008752,
        0.000609,
        0.303472,
        0.008407,
        0.000994,
        0.983955,
        210.485284,
        0.000333,
    ],
    dtype=np.float32,
)


def available() -> bool:
    """Whether the ``pyloric`` simulator package can be imported."""
    try:
        import pyloric  # noqa: F401

        return True
    except Exception:
        return False


class PyloricProblem:
    """Prior, simulator and observation for the pyloric network experiment."""

    def __init__(self, dim_summary_statistics: int = 18) -> None:
        from pyloric import create_prior, simulate, summary_stats

        self._simulate = simulate
        self._summary_stats = summary_stats
        self.pandas_prior = create_prior()
        self.columns = list(self.pandas_prior.sample((1,)).columns)
        self.numerical_prior = self.pandas_prior.numerical_prior
        self.dim_parameters = len(self.columns)
        self.dim_summary_statistics = dim_summary_statistics
        self._replacement_values: Optional[np.ndarray] = None
        self._prior_predictive: Optional[np.ndarray] = None

    # -------------------------------------------------------------- simulator
    def simulate_batch(self, theta: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """Run the simulator; returns (summary statistics, validity mask)."""
        import pandas as pd

        theta_np = theta.detach().cpu().numpy()
        theta_df = pd.DataFrame(theta_np, columns=self.columns)
        traces = self._simulate(theta_df)
        stats = self._summary_stats(traces, stats_customization={"plateau_durations": True})
        stats_np = stats.to_numpy()[:, : self.dim_summary_statistics]
        valid = ~np.isnan(stats_np)
        return torch.as_tensor(stats_np, dtype=torch.float32), valid

    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulator with invalid summary statistics replaced (Appendix E.2)."""
        stats, valid = self.simulate_batch(theta)
        if not valid.all():
            if self._replacement_values is None:
                self.fit_prior_predictive()
            stats = torch.as_tensor(
                np.where(valid, stats.numpy(), self._replacement_values), dtype=torch.float32
            )
        return stats

    # ------------------------------------------------------- prior predictive
    def fit_prior_predictive(self, num_samples: int = 2000, seed: int = 0) -> np.ndarray:
        """Estimate the prior predictive summary statistics.

        Used both to define the replacement value for invalid summary
        statistics (``min - 2 * std``) and -- implicitly -- the ``sigma_max`` /
        standardisation of the score network, which are computed from the first
        round of simulations.
        """
        torch.manual_seed(seed)
        theta = self.numerical_prior.sample((num_samples,))
        stats, _ = self.simulate_batch(theta)
        stats_np = stats.numpy()
        self._prior_predictive = stats_np
        x_min = np.nanmin(stats_np, axis=0)
        x_std = np.nanstd(stats_np, axis=0)
        self._replacement_values = x_min - 2.0 * x_std
        return self._replacement_values

    @property
    def observation(self) -> torch.Tensor:
        return torch.as_tensor(
            PYLORIC_OBSERVATION_20D[: self.dim_summary_statistics], dtype=torch.float32
        ).reshape(1, -1)
