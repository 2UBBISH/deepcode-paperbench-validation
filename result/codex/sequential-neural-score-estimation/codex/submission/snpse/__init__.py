"""Sequential Neural Posterior Score Estimation (SNPSE).

Reference implementation accompanying the reproduction of
"Sequential Neural Score Estimation: Likelihood-Free Inference with
Conditional Score Based Diffusion Models" (Sharrock, Simons, Liu & Beaumont,
ICML 2024).

Public API
----------
``NPSE``       -- non-sequential neural posterior score estimation (Section 2.2)
``TSNPSE``     -- truncated sequential NPSE, Algorithm 1 (Section 3.1)
``SNPSEA`` / ``SNPSEB`` / ``SNPSEC`` -- alternative sequential approaches
                  (Section 3.2, Appendix C)
``VESDE`` / ``VPSDE`` -- forward noising processes (Appendix E.3.1)
``ScoreNetwork`` -- score network architecture (Appendix E.3.2)
"""

from .diffusion import DiffusionConfig, DiffusionPosterior
from .metrics import c2st
from .networks import EnergyNetwork, ScoreNetwork
from .normalization import StandardizedDistribution, Standardizer
from .npse import NPSE, TrainingConfig
from .odeint import odeint, odeint_fixed
from .proposals import (
    MixtureProposal,
    PriorProposal,
    TruncatedProposal,
    TruncationBoundary,
    estimate_truncation_boundary,
)
from .sde import VESDE, VPSDE, compute_sigma_max, get_sde
from .snpse_variants import SNPSEA, SNPSEB, SNPSEC, VariantConfig
from .training import dsm_loss, train_score_network
from .tsnpse import TSNPSE, TSNPSEConfig

__all__ = [
    "NPSE",
    "TrainingConfig",
    "TSNPSE",
    "TSNPSEConfig",
    "SNPSEA",
    "SNPSEB",
    "SNPSEC",
    "VariantConfig",
    "DiffusionConfig",
    "DiffusionPosterior",
    "ScoreNetwork",
    "EnergyNetwork",
    "Standardizer",
    "StandardizedDistribution",
    "VESDE",
    "VPSDE",
    "get_sde",
    "compute_sigma_max",
    "dsm_loss",
    "train_score_network",
    "odeint",
    "odeint_fixed",
    "TruncationBoundary",
    "TruncatedProposal",
    "PriorProposal",
    "MixtureProposal",
    "estimate_truncation_boundary",
    "c2st",
]
