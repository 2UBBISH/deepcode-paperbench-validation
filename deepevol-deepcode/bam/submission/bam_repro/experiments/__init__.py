"""Experiment drivers for the *Batch and Match* (BaM) reproduction.

This package collects the four experiment families of the paper:

======================================  =================================================
Section / Figure                        Module
======================================  =================================================
Sec 5.1 / Fig 5.1 (+ E.3)               :mod:`.exp_gaussian`
Sec 5.1 / Fig 5.2 (+ E.4)               :mod:`.exp_non_gaussian`
Sec 5.2 / Fig 5.3 (+ E.6)               :mod:`.exp_posteriordb`
Sec 5.3 / Fig 5.4                       :mod:`.exp_vae`
======================================  =================================================

Every driver follows the same protocol so that results are directly comparable:

* the x axis is the number of **gradient evaluations** (wallclock timing is out of scope),
* the y axis is either a KL divergence (synthetic targets) or a relative
  mean / SD error against HMC reference moments (posteriorDB) or a
  reconstruction MSE (VAE),
* the protocol reports the mean over ``n_runs`` independent seeds
  (10 for the synthetic targets, 5 for posteriorDB) together with the
  standard error of the mean.

Each module exposes

* a ``run(...)``-style function performing a single (seed) replicate,
* a ``run_experiment(...)``-style function performing the full sweep
  (all dimensions / targets / batch sizes / seeds) and returning a nested
  ``dict`` of curves, and
* a ``main()`` CLI entry point, used by :mod:`bam_repro.scripts.run_all`.

The imports below are guarded so that a partially installed environment
(e.g. missing CIFAR-10 or BridgeStan) can still run the other experiments.
"""

from __future__ import annotations

import warnings
from typing import List

__version__ = "0.1.0"

__all__: List[str] = []

# ---------------------------------------------------------------------------
# Gaussian synthetic experiments (Sec 5.1, Fig 5.1 / E.3)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .exp_gaussian import (  # noqa: F401
        PAPER_DIMS,
        PAPER_GAUSSIAN_BATCH_SIZES,
        PAPER_GAUSSIAN_N_RUNS,
        PAPER_GAUSSIAN_SETTINGS,
        run_experiment as run_gaussian_experiment,
        run_replicate as run_gaussian_replicate,
        run_sweep as run_gaussian,
        GaussianExperimentResult,
    )

    __all__ += [
        "PAPER_DIMS",
        "PAPER_GAUSSIAN_BATCH_SIZES",
        "PAPER_GAUSSIAN_N_RUNS",
        "PAPER_GAUSSIAN_SETTINGS",
        "run_gaussian_experiment",
        "run_gaussian_replicate",
        "run_gaussian",
        "GaussianExperimentResult",
    ]
except Exception as exc:  # pragma: no cover
    warnings.warn(f"exp_gaussian unavailable: {exc!r}", RuntimeWarning)

# ---------------------------------------------------------------------------
# Non-Gaussian (sinh-arcsinh) experiments (Sec 5.1, Fig 5.2 / E.4)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .exp_non_gaussian import (  # noqa: F401
        PAPER_SKEW_SETTINGS,
        PAPER_TAIL_SETTINGS,
        PAPER_NON_GAUSSIAN_BATCH_SIZES,
        PAPER_NON_GAUSSIAN_N_RUNS,
        run_experiment as run_non_gaussian_experiment,
        run_replicate as run_non_gaussian_replicate,
        run_sweep as run_non_gaussian,
        NonGaussianExperimentResult,
    )

    __all__ += [
        "PAPER_SKEW_SETTINGS",
        "PAPER_TAIL_SETTINGS",
        "PAPER_NON_GAUSSIAN_BATCH_SIZES",
        "PAPER_NON_GAUSSIAN_N_RUNS",
        "run_non_gaussian_experiment",
        "run_non_gaussian_replicate",
        "run_non_gaussian",
        "NonGaussianExperimentResult",
    ]
except Exception as exc:  # pragma: no cover
    warnings.warn(f"exp_non_gaussian unavailable: {exc!r}", RuntimeWarning)

# ---------------------------------------------------------------------------
# posteriorDB experiments (Sec 5.2, Fig 5.3 / E.6)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .exp_posteriordb import (  # noqa: F401
        PAPER_POSTERIORDB_BATCH_SIZES,
        PAPER_POSTERIORDB_N_RUNS,
        run_experiment as run_posteriordb_experiment,
        run_replicate as run_posteriordb_replicate,
        run_sweep as run_posteriordb,
        PosteriorDBExperimentResult,
    )

    __all__ += [
        "PAPER_POSTERIORDB_BATCH_SIZES",
        "PAPER_POSTERIORDB_N_RUNS",
        "run_posteriordb_experiment",
        "run_posteriordb_replicate",
        "run_posteriordb",
        "PosteriorDBExperimentResult",
    ]
except Exception as exc:  # pragma: no cover
    warnings.warn(f"exp_posteriordb unavailable: {exc!r}", RuntimeWarning)

# ---------------------------------------------------------------------------
# VAE experiments (Sec 5.3, Fig 5.4)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .exp_vae import (  # noqa: F401
        PAPER_VAE_BATCH_SIZES,
        PAPER_VAE_T,
        PAPER_VAE_PILOT_T,
        run_experiment as run_vae_experiment,
        run_replicate as run_vae_replicate,
        run_sweep as run_vae,
        VAEExperimentResult,
    )

    __all__ += [
        "PAPER_VAE_BATCH_SIZES",
        "PAPER_VAE_T",
        "PAPER_VAE_PILOT_T",
        "run_vae_experiment",
        "run_vae_replicate",
        "run_vae",
        "VAEExperimentResult",
    ]
except Exception as exc:  # pragma: no cover
    warnings.warn(f"exp_vae unavailable: {exc!r}", RuntimeWarning)


def available_experiments() -> List[str]:
    """Return the names of the experiment modules that imported successfully."""
    names = []
    if "run_gaussian_experiment" in globals():
        names.append("gaussian")
    if "run_non_gaussian_experiment" in globals():
        names.append("non_gaussian")
    if "run_posteriordb_experiment" in globals():
        names.append("posteriordb")
    if "run_vae_experiment" in globals():
        names.append("vae")
    return names
