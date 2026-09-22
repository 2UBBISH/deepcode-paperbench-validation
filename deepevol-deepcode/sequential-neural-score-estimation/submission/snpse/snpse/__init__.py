"""SNPSE: Sequential Neural Posterior Score Estimation.

This package implements the methods described in the paper
"Sequential Neural Posterior Score Estimation" (SNPSE):

* ``NPSE``  -- amortised/round-1 Neural Posterior Score Estimation (Section 2.2)
* ``TSNPSE`` -- sequential variant using HPR-truncated prior proposals
  (Section 3.1, Algorithm 1)
* ``SNPSE-A``/``SNPSE-B``/``SNPSE-C`` -- alternative sequential variants with
  importance-weight corrections (Section 3.2, Appendices C.2-C.4)
* ``NLSE`` -- Neural Likelihood Score Estimation (Appendix B)

The package is intentionally importable with only ``torch`` installed; the
external reference implementations (``sbibm``, mackelab ``tsnpe``) are treated
as optional backends.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Forward noising SDEs
# --------------------------------------------------------------------------- #
from .sdes import (  # noqa: F401
    SDE,
    VESDE,
    VPSDE,
    T_FINAL,
    get_sde,
    sigma_max_technique1,
)

# --------------------------------------------------------------------------- #
# Score network
# --------------------------------------------------------------------------- #
from .score_network import (  # noqa: F401
    MLP,
    ScoreNetwork,
    EnergyNetwork,
    EnergyScoreNetwork,
    get_score_network,
    sinusoidal_embedding,
    embedding_dim,
    count_parameters,
)

# --------------------------------------------------------------------------- #
# Denoising score matching losses
# --------------------------------------------------------------------------- #
from .losses import (  # noqa: F401
    dsm_loss,
    npse_loss,
    tsnpse_loss,
    snpse_a_loss,
    snpse_b_loss,
    snpse_c_loss,
    nlse_loss,
    prior_score_loss,
    importance_weight,
)

# --------------------------------------------------------------------------- #
# Sampling / log densities
# --------------------------------------------------------------------------- #
from .sampler import (  # noqa: F401
    SamplerConfig,
    ProbabilityFlowSampler,
    sample_posterior,
    estimate_log_prob,
    reverse_sde_sample,
)

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
from .trainer import (  # noqa: F401
    TrainConfig,
    TrainHistory,
    Trainer,
    train_network,
    select_batch_size,
)

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
from .utils import (  # noqa: F401
    Standardiser,
    fit_standardiser,
    set_seed,
    get_device,
)

# --------------------------------------------------------------------------- #
# Main estimators
# --------------------------------------------------------------------------- #
from .npse import NPSE, NPSEConfig, run_npse  # noqa: F401

# The sequential machinery is imported lazily-tolerant: a failure here should
# not make the round-1 estimator unusable.
try:  # pragma: no cover - defensive
    from .hpr import (  # noqa: F401
        HPR_eps,
        HPRRegion,
        TruncatedPrior,
        compute_hpr_region,
        build_truncated_prior,
    )
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - defensive
    from .tsnpse import TSNPSE, TSNPSEConfig, run_tsnpse  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - defensive
    from .snpse_variants import (  # noqa: F401
        SNPSEConfig,
        SNPSEA,
        SNPSEB,
        SNPSEC,
        SNPSE_A,
        SNPSE_B,
        SNPSE_C,
        run_snpse_a,
        run_snpse_b,
        run_snpse_c,
    )
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - defensive
    from .nlse import (  # noqa: F401
        NLSE,
        NLSEConfig,
        run_nlse,
        PriorSpec,
        build_prior_score,
        train_prior_score_network,
    )
except Exception:  # pragma: no cover
    pass


__version__ = "0.1.0"

__all__ = [
    # sdes
    "SDE",
    "VESDE",
    "VPSDE",
    "T_FINAL",
    "get_sde",
    "sigma_max_technique1",
    # networks
    "MLP",
    "ScoreNetwork",
    "EnergyNetwork",
    "EnergyScoreNetwork",
    "get_score_network",
    "sinusoidal_embedding",
    "embedding_dim",
    "count_parameters",
    # losses
    "dsm_loss",
    "npse_loss",
    "tsnpse_loss",
    "snpse_a_loss",
    "snpse_b_loss",
    "snpse_c_loss",
    "nlse_loss",
    "prior_score_loss",
    "importance_weight",
    # sampler
    "SamplerConfig",
    "ProbabilityFlowSampler",
    "sample_posterior",
    "estimate_log_prob",
    "reverse_sde_sample",
    # trainer
    "TrainConfig",
    "TrainHistory",
    "Trainer",
    "train_network",
    "select_batch_size",
    # utils
    "Standardiser",
    "fit_standardiser",
    "set_seed",
    "get_device",
    # estimators
    "NPSE",
    "NPSEConfig",
    "run_npse",
    "TSNPSE",
    "TSNPSEConfig",
    "run_tsnpse",
    "SNPSEConfig",
    "SNPSEA",
    "SNPSEB",
    "SNPSEC",
    "SNPSE_A",
    "SNPSE_B",
    "SNPSE_C",
    "run_snpse_a",
    "run_snpse_b",
    "run_snpse_c",
    "NLSE",
    "NLSEConfig",
    "run_nlse",
    "__version__",
]
