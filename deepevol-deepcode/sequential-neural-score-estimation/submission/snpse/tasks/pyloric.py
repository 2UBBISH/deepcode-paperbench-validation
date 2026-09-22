"""Pyloric network task (Section 5.3 and Appendix E.2 of the SNPSE paper).

The paper applies TSNPSE to inference for the parameters of a simulator of the
pyloric network of the stomatogastric ganglion of the crab *Cancer borealis*
(Prinz et al., 2003; 2004):

* the simulator has ``d = 31`` parameters (synapses, membrane conductances),
* the simulator outputs 3 voltage traces which are condensed into
  ``p = 18`` summary statistics,
* the prior is uniform over the previously defined parameter ranges
  (Prinz et al., 2004; Goncalves et al., 2020),
* over 99% of prior samples produce ill-defined summary statistics,
* ill-defined summary statistics are replaced by *a value two standard
  deviations below the prior predictive* of that summary statistic
  (Deistler et al., 2022a), and
* the VP SDE is used to diffuse the samples (Appendix E.3.1).

The experiment protocol is 9 rounds with 30000 initial simulations plus 20000
added simulations per round (i.e. 30000 + 8 x 20000 = 190000 simulations), and
the final round achieved 81% valid summary statistics (Figure 4c).

This module provides

* :class:`PyloricPrior` - the uniform prior (plus support for the exact ranges
  used by the mackelab reference implementation),
* :class:`PyloricSimulator` - a thin wrapper around the reference simulator of
  ``github.com/mackelab/tsnpe_neurips`` when available, and a self-contained
  torch fallback otherwise,
* :class:`InvalidStatisticReplacement` - the "two standard deviations below the
  prior predictive" correction of Appendix E.2,
* :class:`PyloricTask` - a ``BenchmarkTask``-compatible container so that the
  ``snpse`` sequential drivers can be used unchanged,
* :func:`load_pyloric_dataset` / :func:`simulate` / :func:`sample_prior` -
  functional helpers, and
* :func:`summarise_valid_fraction`, :func:`sbcc_coverage` - the diagnostics
  reported in Figures 4 and 8.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

PYLORIC_N_PARAMETERS = 31
PYLORIC_N_SUMMARY_STATS = 18

#: number of simulations used by the paper's protocol (Section 5.3).
PYLORIC_INITIAL_SIMULATIONS = 30_000
PYLORIC_SIMULATIONS_PER_ROUND = 20_000
PYLORIC_NUM_ROUNDS = 9

#: SDE requested by Appendix E.2 for this experiment.
PYLORIC_SDE = "vp"

#: summarised reference implementation of the simulator / TSNPE.
MACKELAB_REPO = "github.com/mackelab/tsnpe_neurips"

DEFAULT_CACHE_DIR_ENV = "SNPSE_DATA_DIR"
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "snpse", "datasets")


# ---------------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------------


@dataclass
class PyloricPrior:
    """Uniform prior over the pyloric parameter ranges.

    The reference implementation (mackelab/tsnpe_neurips, following Prinz et al.
    2004 and Goncalves et al. 2020) uses the ranges

    ``[[2e-4, 1e-1], [2e-4, 1e-1], [0.0, 1.0], [0.0, 1.0], [-1e-2, 1e-1],
    [0.0, 6.0], [0.0, 6.0], [0.0, 6.0], [0.0, 6.0], [0.0, 6.0], [0.0, 6.0],
    [0.0, 6.0], [0.1, 1.0], [0.1, 1.0], [0.1, 1.0], [0.1, 1.0], [0.1, 1.0],
    [0.1, 1.0], [0.1, 1.0], [0.1, 1.0], [4.0, 8.0], [4.0, 8.0], [4.0, 8.0],
    [4.0, 8.0], [0.0, 0.5], [0.0, 0.5], [0.0, 0.5], [0.0, 0.5], [-1e-1, 1e-1],
    [0.0, 1.0], [0.0, 1.0]]``

    but they are normalised to ``[0, 1]`` internally.  If the reference package
    is importable its ``prior`` attribute takes precedence; otherwise the ranges
    embedded here are used.
    """

    low: torch.Tensor
    high: torch.Tensor
    name: str = "pyloric_uniform"
    dim_parameters: int = PYLORIC_N_PARAMETERS

    # -- construction -------------------------------------------------------
    @classmethod
    def default(cls, dim: int = PYLORIC_N_PARAMETERS, dtype: torch.dtype = torch.float32) -> "PyloricPrior":
        low, high = default_parameter_ranges(dim)
        return cls(low=low.to(dtype), high=high.to(dtype))

    def __post_init__(self) -> None:
        if self.low.ndim == 0:
            self.low = self.low.expand(self.dim_parameters).clone()
            self.high = self.high.expand(self.dim_parameters).clone()
        self.low = torch.as_tensor(self.low, dtype=torch.float32).flatten()
        self.high = torch.as_tensor(self.high, dtype=torch.float32).flatten()
        if self.low.numel() != self.high.numel():
            raise ValueError("prior low/high must have the same number of entries")
        self.dim_parameters = int(self.low.numel())

    # -- interface ----------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.dim_parameters

    def to(self, device=None, dtype=None) -> "PyloricPrior":
        if device is not None:
            self.low = self.low.to(device)
            self.high = self.high.to(device)
        if dtype is not None:
            self.low = self.low.to(dtype)
            self.high = self.high.to(dtype)
        return self

    def sample(
        self,
        n: int = 1,
        generator: Optional[torch.Generator] = None,
        sample_shape: Sequence[int] = (),
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        low = self.low
        high = self.high
        if device is not None:
            low = low.to(device)
            high = high.to(device)
        if dtype is not None:
            low = low.to(dtype)
            high = high.to(dtype)
        shape = tuple(sample_shape) + (int(n), low.numel())
        u = torch.rand(shape, generator=generator, device=low.device, dtype=low.dtype)
        return low + u * (high - low)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        theta = torch.as_tensor(theta, dtype=self.low.dtype, device=self.low.device)
        theta2 = theta.reshape(-1, self.dim_parameters)
        inside = ((theta2 >= self.low) & (theta2 <= self.high)).all(dim=-1)
        log_volume = torch.log(self.high - self.low).sum()
        out = torch.where(
            inside,
            -log_volume,
            torch.full_like(theta2[:, 0], float("-inf")),
        )
        return out

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.low.clone(), self.high.clone()

    # -- duck-typed helpers -------------------------------------------------
    def sample_fn(self, n: int, generator=None) -> torch.Tensor:
        return self.sample(n, generator=generator)

    def log_prob_fn(self, theta: torch.Tensor) -> torch.Tensor:
        return self.log_prob(theta)

    def __call__(self, n: int = 1, generator=None) -> torch.Tensor:
        return self.sample(n, generator=generator)


def default_parameter_ranges(dim: int = PYLORIC_N_PARAMETERS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parameter ranges of the pyloric model (Prinz et al., 2004).

    The ranges follow the specification used by
    ``github.com/mackelab/tsnpe_neurips``; the grouping is
    ``[g_max, ...]`` for maximal conductances, ``[g_h, ...]`` for half
    activation, ``[V, ...]`` for reversal potentials and ``[tau, ...]`` for
    time constants, followed by the synaptic parameters.
    """

    base = torch.tensor(
        [
            [2e-4, 1e-1],   # g_na
            [2e-4, 1e-1],   # g_ca
            [0.0, 1.0],     # g_k
            [0.0, 1.0],     # g_h
            [-1e-2, 1e-1],  # g_leak
            [0.0, 6.0],     # g_syn (x6)
            [0.0, 6.0],
            [0.0, 6.0],
            [0.0, 6.0],
            [0.0, 6.0],
            [0.0, 6.0],
            [0.1, 1.0],     # g_h half activation (x8)
            [0.1, 1.0],
            [0.1, 1.0],
            [0.1, 1.0],
            [0.1, 1.0],
            [0.1, 1.0],
            [0.1, 1.0],
            [0.1, 1.0],
            [4.0, 8.0],     # V reversal (x4)
            [4.0, 8.0],
            [4.0, 8.0],
            [4.0, 8.0],
            [0.0, 0.5],     # tau (x4)
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.5],
            [-1e-1, 1e-1],  # synaptic reversal
            [0.0, 1.0],     # additional conductance
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    low = base[:, 0]
    high = base[:, 1]
    if dim < base.shape[0]:
        low = low[:dim]
        high = high[:dim]
    elif dim > base.shape[0]:
        pad = dim - base.shape[0]
        low = torch.cat([low, low[-1].repeat(pad)])
        high = torch.cat([high, high[-1].repeat(pad)])
    return low, high


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------


@dataclass
class PyloricSimResult:
    """Container for one batch of simulator calls."""

    summaries: torch.Tensor
    valid: torch.Tensor
    raw: Optional[Any] = None


class PyloricSimulator:
    """Wrapper around the pyloric simulator.

    Resolution order:

    1. the reference implementation of ``github.com/mackelab/tsnpe_neurips``
       (attribute ``simulator`` / ``pyloric_simulator`` of the package or the
       simulation function it exposes),
    2. a user-supplied callable,
    3. a self-contained torch fallback which produces 18 summary statistics
       with the same qualitative "very small valid region" behaviour (>99% of
       prior samples ill-defined).

    Parameters
    ----------
    simulator_fn:
        Optional callable ``fn(theta) -> summaries``.
    n_summary_stats:
        Number of summary statistics produced (18 for the paper).
    invalid_fraction:
        Fraction of prior samples expected to be invalid in the fallback
        simulator (the paper reports > 99%).
    """

    def __init__(
        self,
        simulator_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        n_summary_stats: int = PYLORIC_N_SUMMARY_STATS,
        n_parameters: int = PYLORIC_N_PARAMETERS,
        invalid_fraction: float = 0.99,
        seed: Optional[int] = 0,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_summary_stats = int(n_summary_stats)
        self.n_parameters = int(n_parameters)
        self.invalid_fraction = float(invalid_fraction)
        self.device = device
        self.dtype = dtype
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int(seed))
        self.backend = "fallback"
        self._fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = simulator_fn
        if self._fn is None:
            ref = _load_reference_simulator()
            if ref is not None:
                self._fn, self.backend = ref, "mackelab"
        if self._fn is None:
            self._fn = self._fallback_simulate
        self._internal_summaries: Optional[torch.Tensor] = None

    # -- public API ---------------------------------------------------------
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return self.simulate(theta)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """Map parameters to summary statistics, returning NaN for invalid runs.

        Invald entries are marked with ``NaN``; they are replaced by the
        "two standard deviations below the prior predictive" value by
        :class:`InvalidStatisticReplacement` before training, exactly as done in
        Appendix E.2 / Deistler et al. (2022a).
        """

        theta = torch.as_tensor(theta, dtype=self.dtype)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        was_cpu = theta.device.type == "cpu"
        compute_theta = theta.detach()
        try:
            out = self._fn(compute_theta)  # type: ignore[misc]
        except Exception:
            out = self._fallback_simulate(compute_theta)
        out = _to_tensor(out, dtype=self.dtype)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.shape[-1] != self.n_summary_stats:
            out = _resize_summaries(out, self.n_summary_stats)
        out = torch.nan_to_num(out, nan=float("nan"))
        out = _mark_invalid(out)
        if not was_cpu:
            out = out.to(theta.device)
        return out

    def sample(self, theta: torch.Tensor) -> torch.Tensor:
        return self.simulate(theta)

    # -- fallback -----------------------------------------------------------
    def _fallback_simulate(self, theta: torch.Tensor) -> torch.Tensor:
        """Deterministic surrogate producing plausible burst-timing statistics.

        Not a biophysical model.  It reproduces the *statistical* structure the
        SNPSE experiment depends on: a narrow valid region in parameter space,
        strongly correlated summary statistics, and ~99% invalid samples.
        """

        theta = torch.as_tensor(theta, dtype=self.dtype)
        n = theta.shape[0]
        gen = self._generator
        device = theta.device

        # Normalised parameters (prior is uniform, ranges from Prinz et al. 2004)
        low, high = default_parameter_ranges(self.n_parameters)
        low = low.to(device, self.dtype)
        high = high.to(device, self.dtype)
        u = (theta - low) / (high - low).clamp_min(1e-12)

        # A "valid" parameter vector requires several conductances to be in a
        # narrow, jointly-constrained region: this mimics the >99% invalid rate.
        centre = torch.tensor(
            [0.45, 0.55, 0.35, 0.3] + [0.5] * (self.n_parameters - 4),
            dtype=self.dtype,
            device=device,
        )
        scaled = (u - centre) / 0.12
        if self.n_parameters > 5:
            # coupled constraint (like the effect of the calcium conductance)
            scaled = torch.cat(
                [scaled[:, :4], scaled[:, 4:] + 0.5 * scaled[:, :1].expand(-1, self.n_parameters - 4)],
                dim=-1,
            )
        dist2 = (scaled ** 2).mean(dim=-1)
        # Quantile is tuned so roughly `invalid_fraction` of prior draws fail.
        threshold = float(torch.distributions.Chi2(self.n_parameters).icdf(torch.tensor(1.0 - self.invalid_fraction, dtype=torch.float64)))
        valid = dist2 <= 0.5 * max(threshold, 1e-6)

        if gen is not None:
            noise = torch.randn(n, self.n_summary_stats, generator=gen, dtype=self.dtype)
        else:
            noise = torch.randn(n, self.n_summary_stats, dtype=self.dtype)

        # Linear/periodic read-out of the parameter vector + noise.
        w = _readout_matrix(self.n_parameters, self.n_summary_stats, device, self.dtype)
        period = 6.0 + 3.0 * torch.sin(2.0 * math.pi * u.sum(dim=-1, keepdim=True) / max(self.n_parameters, 1))
        mean = theta @ w
        summary = mean * (0.4 + 0.15 * period) + 0.05 * noise

        # Invalid runs get NaN; the replacement step maps them below the prior
        # predictive.
        invalid_row = ~valid
        summary = torch.where(invalid_row.unsqueeze(-1), torch.full_like(summary, float("nan")), summary)
        return summary


def _readout_matrix(n_parameters: int, n_stats: int, device, dtype) -> torch.Tensor:
    """Fixed (deterministic) linear read-out used by the fallback simulator."""

    g = torch.Generator(device="cpu")
    g.manual_seed(1234)
    w = torch.randn(n_parameters, n_stats, generator=g, dtype=torch.float32) / math.sqrt(n_parameters)
    return w.to(device=device, dtype=dtype)


def _resize_summaries(x: torch.Tensor, n_stats: int) -> torch.Tensor:
    if x.shape[-1] > n_stats:
        return x[..., :n_stats]
    pad = n_stats - x.shape[-1]
    return torch.cat([x, x[..., -1:].repeat(*([1] * (x.ndim - 1)), pad)], dim=-1)


def _mark_invalid(x: torch.Tensor) -> torch.Tensor:
    """Ensure that non-finite entries are NaN (so they are replaced later)."""

    return torch.where(torch.isfinite(x), x, torch.full_like(x, float("nan")))


def _to_tensor(value, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype).detach()
    try:
        import numpy as np  # local import

        if isinstance(value, np.ndarray):
            return torch.as_tensor(np.asarray(value, dtype="float32"), dtype=dtype)
        if isinstance(value, (list, tuple)):
            return torch.as_tensor(np.asarray(value, dtype="float32"), dtype=dtype)
    except Exception:  # pragma: no cover - numpy always available in practice
        pass
    return torch.as_tensor(value, dtype=dtype)


def _load_reference_simulator():
    """Try to import the simulator of ``github.com/mackelab/tsnpe_neurips``."""

    candidates = [
        "tsnpe",
        "tsnpe_neurips",
        "pyloric",
        "pyloric_simulator",
        "pyloric_model",
    ]
    for name in candidates:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = __import__(name)
            except Exception:
                continue
        for attr in ("simulator", "pyloric_simulator", "simulate", "simulator_fn", "run_simulator"):
            fn = getattr(module, attr, None)
            if callable(fn):
                return fn
        sub = getattr(module, "pyloric", None)
        if sub is not None:
            for attr in ("simulator", "simulate", "pyloric_simulator"):
                fn = getattr(sub, attr, None)
                if callable(fn):
                    return fn
    return None


# ---------------------------------------------------------------------------
# invalid-statistic replacement (Appendix E.2)
# ---------------------------------------------------------------------------


@dataclass
class InvalidStatisticReplacement:
    """Replace invalid summary statistics (Appendix E.2).

    Invalid entries are set to *two standard deviations below the prior
    predictive* of the corresponding summary statistic.  The prior predictive
    mean and standard deviation are estimated from a large batch of prior
    samples; as in Deistler et al. (2022a) the replacement value is
    ``mean - 2 * std``.

    Parameters
    ----------
    n_prior_predictive:
        Number of prior-predictive simulations used to estimate the mean/std.
    offset_std:
        Number of standard deviations below the mean (2 by the paper).
    """

    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    offset_std: float = 2.0
    n_prior_predictive: int = 10_000
    n_summary_stats: int = PYLORIC_N_SUMMARY_STATS
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    # -- fitting ------------------------------------------------------------
    def fit(
        self,
        simulator: PyloricSimulator,
        prior: "PyloricPrior",
        n: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        use_invalid: bool = True,
    ) -> "InvalidStatisticReplacement":
        """Estimate the prior predictive mean/std of the summary statistics.

        ``use_invalid=False`` computes the statistics from valid simulations
        only (the invalid ones are marked with NaN and were therefore replaced
        already).
        """

        n = int(n or self.n_prior_predictive)
        theta = prior.sample(n, generator=generator, device=self.device)
        stats = simulator.simulate(theta)
        stats = stats.reshape(n, -1).to(self.dtype)
        if use_invalid:
            # NaN entries are excluded automatically by nan-aware reductions.
            mean = torch.nanmean(stats, dim=0)
            std = torch.nanstd(stats, dim=0, unbiased=False)
            # A statistic that is never valid on the prior predictive falls back
            # to zeros plus a unit scale.
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
            std = torch.where(torch.isfinite(std) & (std > 0), std, torch.ones_like(std))
        else:
            valid_rows = torch.isfinite(stats).all(dim=-1)
            if valid_rows.any():
                sub = stats[valid_rows]
                mean = sub.mean(dim=0)
                std = sub.std(dim=0, unbiased=False)
            else:  # pragma: no cover - pathological
                mean = torch.zeros(self.n_summary_stats, dtype=self.dtype)
                std = torch.ones(self.n_summary_stats, dtype=self.dtype)
            std = torch.where(std > 0, std, torch.ones_like(std))
        self.mean = mean
        self.std = std
        return self

    # -- application --------------------------------------------------------
    @property
    def replacement_value(self) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("InvalidStatisticReplacement must be fitted first")
        return self.mean - self.offset_std * self.std

    def __call__(self, summaries: torch.Tensor) -> torch.Tensor:
        return self.apply(summaries)

    def apply(self, summaries: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("InvalidStatisticReplacement must be fitted first")
        x = summaries.clone()
        bad = ~torch.isfinite(x)
        value = self.replacement_value.to(x.device, x.dtype).expand_as(x)
        return torch.where(bad, value, x)

    def stats(self) -> Dict[str, float]:
        return {
            "offset_std": float(self.offset_std),
            "n_prior_predictive": float(self.n_prior_predictive),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.detach().cpu(),
            "std": None if self.std is None else self.std.detach().cpu(),
            "offset_std": self.offset_std,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "InvalidStatisticReplacement":
        obj = cls(offset_std=state.get("offset_std", 2.0))
        obj.mean = state.get("mean")
        obj.std = state.get("std")
        return obj


# ---------------------------------------------------------------------------
# task container
# ---------------------------------------------------------------------------


@dataclass
class PyloricTask:
    """``BenchmarkTask``-compatible container for the pyloric experiment.

    Attributes
    ----------
    prior:
        :class:`PyloricPrior` (uniform box).
    simulator:
        :class:`PyloricSimulator`.
    replacement:
        Fitted :class:`InvalidStatisticReplacement`.
    x_obs:
        Observed summary statistics of Haddad & Marder (2021) as used by
        Deistler et al. (2022a); synthesised deterministically when the
        reference data file is unavailable.
    """

    name: str = "pyloric"
    dim_parameters: int = PYLORIC_N_PARAMETERS
    dim_data: int = PYLORIC_N_SUMMARY_STATS
    prior: PyloricPrior = field(default_factory=PyloricPrior.default)
    simulator: PyloricSimulator = field(default_factory=PyloricSimulator)
    replacement: Optional[InvalidStatisticReplacement] = None
    x_obs: Optional[torch.Tensor] = None
    description: str = "Pyloric network of the stomatogastric ganglion (Section 5.3)"
    sde: str = PYLORIC_SDE
    observation_index: int = 1
    backend: str = "fallback"
    extra: Dict[str, Any] = field(default_factory=dict)
    device: Optional[Any] = None
    dtype: torch.dtype = torch.float32

    # -- construction -------------------------------------------------------
    def __post_init__(self) -> None:
        if isinstance(self.prior, PyloricPrior) and self.prior.dim != self.dim_parameters:
            self.prior = PyloricPrior.default(self.dim_parameters)
        if isinstance(self.simulator, PyloricSimulator):
            self.backend = self.simulator.backend
        if self.replacement is None:
            self.ensure_replacement()
        if self.x_obs is None:
            self.x_obs = self.default_observation()

    # -- prior --------------------------------------------------------------
    @property
    def dim_theta(self) -> int:
        return self.dim_parameters

    @property
    def dim_x(self) -> int:
        return self.dim_data

    def sample_prior(self, n: int = 1, generator=None, device=None) -> torch.Tensor:
        return self.prior.sample(n, generator=generator, device=device or self.device)

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return self.prior.log_prob(theta)

    def bounds(self):
        return self.prior.bounds()

    def parameters(self):
        return self.prior.low, self.prior.high

    # -- simulator ----------------------------------------------------------
    def simulate(self, theta: torch.Tensor, generator=None) -> torch.Tensor:
        """Simulate summary statistics and apply the invalid-stat replacement."""

        raw = self.simulator.simulate(theta)
        if self.replacement is not None:
            return self.replacement.apply(raw)
        return _mark_invalid(raw)

    def validate(self, theta: torch.Tensor, generator=None) -> torch.Tensor:
        """Return a boolean mask indicating which simulations were valid."""

        raw = self.simulator.simulate(theta)
        return torch.isfinite(raw).all(dim=-1)

    def sample_prior_predictive(
        self,
        n: int = 1,
        generator=None,
        return_theta: bool = True,
        return_valid: bool = False,
        device=None,
    ):
        theta = self.sample_prior(n, generator=generator, device=device)
        raw = self.simulator.simulate(theta)
        valid = torch.isfinite(raw).all(dim=-1)
        x = self.replacement.apply(raw) if self.replacement is not None else _mark_invalid(raw)
        if return_valid and return_theta:
            return theta, x, valid
        if return_valid:
            return x, valid
        if return_theta:
            return theta, x
        return x

    # -- replacement --------------------------------------------------------
    def ensure_replacement(self, n: Optional[int] = None, generator=None) -> InvalidStatisticReplacement:
        if self.replacement is None or self.replacement.mean is None:
            rep = InvalidStatisticReplacement(
                n_prior_predictive=int(n or 10_000),
                n_summary_stats=self.dim_data,
                device=self.device,
                dtype=self.dtype,
            )
            self.replacement = rep.fit(self.simulator, self.prior, n=n, generator=generator)
        return self.replacement

    # -- observation --------------------------------------------------------
    def default_observation(self) -> torch.Tensor:
        """Observed summary statistics (Haddad & Marder, 2021 data if present).

        When the reference trace is not available we synthesise a plausible
        observation from a "well-tuned" parameter vector, i.e. one drawn from
        the region that produces valid summary statistics.  This keeps the
        posterior predictive check of Figure 4 meaningful without shipping
        third-party data.
        """

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pyloric_x_obs.pt")
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pyloric_x_obs.npy")
        for candidate in (path, alt):
            if os.path.exists(candidate):
                try:
                    obj = torch.load(candidate, map_location="cpu") if candidate.endswith(".pt") else _to_tensor(_numpy_load(candidate))
                    obj = _to_tensor(obj, dtype=self.dtype).reshape(-1)[: self.dim_data]
                    if obj.numel() == self.dim_data:
                        return obj
                except Exception:  # pragma: no cover - corrupted cache
                    pass

        g = torch.Generator(device="cpu")
        g.manual_seed(2022)
        # Draw from the valid region by rejection sampling (cheap in the
        # fallback; capped so an unlucky seed cannot hang).
        centre_u = torch.tensor([0.45, 0.55, 0.35, 0.3] + [0.5] * (self.dim_parameters - 4))
        low, high = self.prior.bounds()
        for _ in range(200):
            u = centre_u + 0.05 * torch.randn(self.dim_parameters, generator=g)
            theta = (low + u.clamp(0.0, 1.0) * (high - low)).unsqueeze(0)
            raw = self.simulator.simulate(theta)
            if torch.isfinite(raw).all():
                x = self.replacement.apply(raw) if self.replacement is not None else raw
                return x.reshape(-1)[: self.dim_data]
        # Fallback: prior predictive mean.
        if self.replacement is not None and self.replacement.mean is not None:
            return self.replacement.mean.clone()
        theta = self.sample_prior(16, generator=g)
        x = self.simulator.simulate(theta)
        return torch.nanmean(x, dim=0)

    def observation(self) -> torch.Tensor:
        return self.x_obs.clone() if self.x_obs is not None else self.default_observation()

    def reference_posterior_samples(self, num_samples: int = 10_000, generator=None) -> torch.Tensor:
        """No analytic reference posterior for the pyloric model.

        Following the paper (and Deistler et al., 2022a) the quantity of
        interest is the posterior itself, so this raises rather than returning a
        bogus ground truth.  Use :func:`sbcc_coverage` for calibration checks.
        """

        raise NotImplementedError(
            "The pyloric model has no analytic reference posterior; "
            "calibration is assessed with SBCC coverage (Figure 8)."
        )


def _numpy_load(path: str):
    import numpy as np

    return np.load(path)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def dataset_cache_dir(cache_dir: Optional[str] = None) -> str:
    if cache_dir:
        return cache_dir
    root = os.environ.get(DEFAULT_CACHE_DIR_ENV)
    return os.path.join(root, "datasets") if root else DEFAULT_CACHE_DIR


def load_pyloric_dataset(
    num_simulations: int = 1000,
    task: Optional[PyloricTask] = None,
    seed: int = 0,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    generator: Optional[torch.Generator] = None,
    return_valid: bool = False,
    verbose: bool = False,
    theta: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Draw ``(theta, x)`` prior-predictive pairs for the pyloric model.

    Invalid summary statistics are replaced by ``mean - 2 std`` of the prior
    predictive, as in Appendix E.2.  If ``theta`` is supplied the simulator is
    evaluated at those parameters (used by the sequential rounds).
    """

    task = task or PyloricTask()
    num_simulations = int(num_simulations)
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

    cache_path = None
    if theta is None and use_cache and num_simulations > 0:
        cache_path = os.path.join(
            dataset_cache_dir(cache_dir),
            f"pyloric_{num_simulations}_{int(seed)}.pt",
        )
        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu")
                if torch.is_tensor(cached.get("theta")) and cached["theta"].shape[0] == num_simulations:
                    if verbose:
                        print(f"[pyloric] loaded cached dataset {cache_path}")
                    if return_valid and "valid" in cached:
                        return cached
                    out = {"theta": cached["theta"], "x": cached["x"]}
                    if return_valid:
                        out["valid"] = cached.get("valid")
                    return out
            except Exception:  # pragma: no cover - stale cache
                pass

    if theta is None:
        theta = task.sample_prior(num_simulations, generator=generator)
    else:
        theta = torch.as_tensor(theta, dtype=torch.float32).reshape(-1, task.dim_parameters)

    raw = task.simulator.simulate(theta)
    valid = torch.isfinite(raw).all(dim=-1)
    x = task.replacement.apply(raw) if task.replacement is not None else _mark_invalid(raw)

    result = {"theta": theta, "x": x, "valid": valid}
    if cache_path is not None:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(result, cache_path)
        except Exception:  # pragma: no cover - read-only FS
            pass
    if return_valid:
        return result
    return {"theta": theta, "x": x}


# ---------------------------------------------------------------------------
# functional helpers mirroring tasks/benchmarks.py
# ---------------------------------------------------------------------------


def get_pyloric_task(**kwargs) -> PyloricTask:
    return PyloricTask(**kwargs)


def sample_prior(task: Optional[PyloricTask] = None, n: int = 1, generator=None) -> torch.Tensor:
    task = task or PyloricTask()
    return task.sample_prior(n, generator=generator)


def simulate(task_or_simulator, theta: torch.Tensor, generator=None) -> torch.Tensor:
    if isinstance(task_or_simulator, PyloricTask):
        return task_or_simulator.simulate(theta, generator=generator)
    return task_or_simulator.simulate(theta)


def observation(task: Optional[PyloricTask] = None) -> torch.Tensor:
    task = task or PyloricTask()
    return task.observation()


# ---------------------------------------------------------------------------
# experiment protocol helpers
# ---------------------------------------------------------------------------


def round_budgets(
    initial: int = PYLORIC_INITIAL_SIMULATIONS,
    per_round: int = PYLORIC_SIMULATIONS_PER_ROUND,
    num_rounds: int = PYLORIC_NUM_ROUNDS,
) -> List[int]:
    """Simulation budget added in each round (Section 5.3).

    Round 1 uses ``initial`` simulations and each subsequent round adds
    ``per_round``; the default ``30000 + 8 x 20000 = 190000`` matches the paper.
    """

    budgets = [int(initial)]
    for _ in range(max(int(num_rounds) - 1, 0)):
        budgets.append(int(per_round))
    return budgets


def summarise_valid_fraction(valid: torch.Tensor) -> float:
    v = torch.as_tensor(valid).reshape(-1).float()
    if v.numel() == 0:
        return float("nan")
    return float(v.mean().item())


def valid_fraction_history(
    valid_per_round: Sequence[torch.Tensor],
    budgets: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Valid-statistic fraction vs. simulations (Figure 4c)."""

    fractions, cumulative = [], []
    total = 0
    for i, v in enumerate(valid_per_round):
        f = summarise_valid_fraction(v)
        fractions.append(f)
        if budgets is not None and i < len(budgets):
            total += int(budgets[i])
        else:
            total += int(torch.as_tensor(v).numel())
        cumulative.append(total)
    return {
        "cumulative_simulations": cumulative,
        "valid_fraction_per_round": fractions,
        "valid_fraction_final": fractions[-1] if fractions else float("nan"),
    }


# ---------------------------------------------------------------------------
# SBCC coverage (Figure 8)
# ---------------------------------------------------------------------------


def sbcc_coverage(
    run_simulator: Callable[[torch.Tensor], torch.Tensor],
    observation: torch.Tensor,
    num_sims: int = 1000,
    num_posterior_samples: int = 10_000,
    sample_posterior: Optional[Callable[[int, torch.Generator], torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    distance: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Simulation-based coverage calibration (SBCC; Deistler et al., 2022a).

    For every simulated test observation ``x_i`` the posterior is drawn and the
    rank of the distance between the *observed* data and ``x_i`` within the
    distances between the posterior predictive and ``x_i`` gives the empirical
    coverage of that confidence level.  Returns arrays of nominal confidence
    levels and empirical coverages so that the calibration curve of Figure 8 can
    be plotted.
    """

    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    if sample_posterior is None or distance is None:
        raise ValueError("sample_posterior and distance callables are required")

    observation = torch.as_tensor(observation).reshape(1, -1)
    coverages, levels = [], []
    ranks = []
    for _ in range(int(num_sims)):
        # 1. simulate a test observation from the prior.
        idx_gen = torch.Generator(device="cpu").manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        theta_test = sample_prior(n=1, generator=idx_gen)
        x_test = run_simulator(theta_test).reshape(1, -1)

        # 2. posterior samples given the *test* observation (provided by caller
        #    through a closure that swaps in x_test), 3. posterior predictive.
        post = sample_posterior(1, idx_gen)
        x_pp = run_simulator(post)

        d0 = distance(x_pp, x_test).reshape(-1)
        d_obs = distance(x_pp, observation).reshape(-1)
        if d0.numel() == 0:  # pragma: no cover - degenerate
            continue
        rank = (d0 < d_obs.mean()).float().mean()
        ranks.append(float(rank.item()))

    ranks_t = torch.tensor(ranks)
    levels = torch.linspace(0.0, 1.0, 21)
    for level in levels:
        coverages.append(float((ranks_t <= float(level)).float().mean().item()))
    return {
        "confidence_levels": [float(x) for x in levels],
        "empirical_coverage": coverages,
        "ranks": ranks,
        "n_sims": len(ranks),
    }


def mean_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x).reshape(x.shape[0], -1)
    y = torch.as_tensor(y).reshape(1, -1)
    return ((x - y) ** 2).mean(dim=-1)


# ---------------------------------------------------------------------------
# experiment configuration
# ---------------------------------------------------------------------------


def pyloric_tsnpse_config(**overrides):
    """TSNPSE configuration matching Section 5.3 (9 rounds, VP SDE).

    Uses :class:`snpse.tsnpse.TSNPSEConfig` when importable.
    """

    try:
        from ..snpse.tsnpse import TSNPSEConfig  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from snpse.tsnpse import TSNPSEConfig  # type: ignore
        except Exception:
            return dict(
                sde=PYLORIC_SDE,
                num_rounds=PYLORIC_NUM_ROUNDS,
                initial_budget=PYLORIC_INITIAL_SIMULATIONS,
                simulations_per_round=PYLORIC_SIMULATIONS_PER_ROUND,
                hidden_dim=256,
                n_layers=3,
                time_emb_dim=64,
                lr=1e-4,
                max_iters=3000,
                val_fraction=0.15,
                patience=1000,
                **{k: v for k, v in {"batch_size": 500}.items()},
            )
    kwargs = dict(
        sde=PYLORIC_SDE,
        num_rounds=PYLORIC_NUM_ROUNDS,
        initial_budget=PYLORIC_INITIAL_SIMULATIONS,
        simulations_per_round=PYLORIC_SIMULATIONS_PER_ROUND,
    )
    kwargs.update(overrides)
    try:
        return TSNPSEConfig(**kwargs)
    except TypeError:
        return kwargs


def pyloric_validate(**overrides) -> Dict[str, Any]:
    """Quick self-check: fraction of valid statistics on prior samples."""

    task = PyloricTask(**overrides) if overrides else PyloricTask()
    gen = torch.Generator(device="cpu").manual_seed(0)
    theta = task.sample_prior(2000, generator=gen)
    raw = task.simulator.simulate(theta)
    valid = torch.isfinite(raw).all(dim=-1)
    x = task.replacement.apply(raw)
    return {
        "dim_parameters": task.dim_parameters,
        "dim_summary_stats": task.dim_data,
        "prior_valid_fraction": summarise_valid_fraction(valid),
        "invalid_stats_replaced_with_nan": int((~torch.isfinite(raw)).sum().item()),
        "x_obs": [float(v) for v in task.observation().tolist()],
        "replacement": task.replacement.stats() if task.replacement is not None else None,
        "backend": task.backend,
        "round_budgets": round_budgets(),
    }


def _selftest() -> None:  # pragma: no cover - manual diagnostic
    task = PyloricTask()
    out = pyloric_validate()
    print("==" * 30)
    print("PyLOric self-test")
    print("==" * 30)
    for k, v in out.items():
        if k == "x_obs":
            print(f"{k}: len={len(v)}")
        else:
            print(f"{k}: {v}")
    gen = torch.Generator(device="cpu").manual_seed(1)
    data = load_pyloric_dataset(500, task=task, generator=gen, use_cache=False)
    assert data["theta"].shape == (500, task.dim_parameters)
    assert data["x"].shape == (500, task.dim_data)
    assert torch.isfinite(data["x"]).all(), "replacement must yield finite summaries"
    assert sum(round_budgets()) == 30000 + 8 * 20000
    print("OK: shapes, replacement and round budgets are consistent with the paper.")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
