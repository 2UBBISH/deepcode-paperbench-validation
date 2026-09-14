"""Core source package for Neural Posterior Score Estimation (NPSE).

This package provides the building blocks for reproducing the paper
"Neural Posterior Score Estimation for Simulation-Based Inference":

- SDE machinery (VE/VP forward/reverse processes)
- Score-network architectures
- Denoising posterior score-matching losses
- Probability-flow ODE sampling and density evaluation
- Non-sequential (NPSE) and truncated sequential (TSNPSE) trainers
- Sequential variants (SNPSE-A/B/C) and the NLSE baseline
- Analytic/learned perturbed prior scores and HPR truncation
- Evaluation metrics (C2ST, MMD, SBC, posterior predictive checks)
"""

from .sde import (
    SDE,
    VESDE,
    VPSDE,
    ensure_tensor,
    estimate_sigma_max,
    get_sde,
    isotropic_gaussian_logp,
    reverse_sde_step,
)
from .networks import (
    MLP,
    EmbeddingMLP,
    SinusoidalTimeEmbedding,
    Standardizer,
    ScoreNetwork,
    PriorScoreNetwork,
    compute_standardization_stats,
)
from .losses import (
    Weighting,
    get_weighting,
    nlse_dsm_loss,
    npse_dsm_loss,
    npse_dsm_squared_error,
    prior_dsm_loss,
    sample_times,
    weighted_npse_dsm_loss,
)
from .sampling import (
    ProbabilityFlowSampler,
    euler_probability_flow_sample,
    sample_probability_flow,
)
from .density import (
    ProbabilityFlowDensity,
    hutchinson_divergence,
    log_probability_flow,
)
from .npse import (
    NPSETrainer,
    build_sde,
    default_npse_config,
    train_npse,
)
from .tsnpse import (
    TSNPSETrainer,
    default_tsnpse_config,
    train_tsnpse,
)
from .snpse import (
    SNPSETrainer,
    default_snpse_config,
    train_snpse,
    train_snpse_a,
    train_snpse_b,
    train_snpse_c,
)
from .nlse import (
    NLSETrainer,
    default_nlse_config,
    train_nlse,
)
from .prior import (
    AnalyticPriorScore,
    GaussianMixturePriorScore,
    UniformPriorScore,
    gaussian_mixture_perturbed_log_prob,
    gaussian_mixture_perturbed_score,
    make_prior_score_fn,
    normal_cdf,
    train_prior_score_network,
    uniform_perturbed_log_prob,
    uniform_perturbed_score,
)
from .hpr import (
    HPR_EPSILON_DEFAULT,
    HPR_N_SAMPLES_DEFAULT,
    HPRRegion,
    HPRTruncation,
    PriorSupportRegion,
    TruncatedProposalSampler,
    estimate_hpr_threshold,
    sample_truncated_proposal,
)
from .metrics import (
    c2st_score,
    c2st_nn_score,
    expected_calibration_error,
    median_heuristic_bandwidth,
    mmd,
    posterior_predictive_check,
    rbf_mmd,
    simulation_based_calibration,
)

__all__ = [
    # SDE
    "SDE",
    "VESDE",
    "VPSDE",
    "ensure_tensor",
    "estimate_sigma_max",
    "get_sde",
    "isotropic_gaussian_logp",
    "reverse_sde_step",
    # Networks
    "MLP",
    "EmbeddingMLP",
    "SinusoidalTimeEmbedding",
    "Standardizer",
    "ScoreNetwork",
    "PriorScoreNetwork",
    "compute_standardization_stats",
    # Losses
    "Weighting",
    "get_weighting",
    "nlse_dsm_loss",
    "npse_dsm_loss",
    "npse_dsm_squared_error",
    "prior_dsm_loss",
    "sample_times",
    "weighted_npse_dsm_loss",
    # Sampling
    "ProbabilityFlowSampler",
    "euler_probability_flow_sample",
    "sample_probability_flow",
    # Density
    "ProbabilityFlowDensity",
    "hutchinson_divergence",
    "log_probability_flow",
    # NPSE
    "NPSETrainer",
    "build_sde",
    "default_npse_config",
    "train_npse",
    # TSNPSE
    "TSNPSETrainer",
    "default_tsnpse_config",
    "train_tsnpse",
    # SNPSE
    "SNPSETrainer",
    "default_snpse_config",
    "train_snpse",
    "train_snpse_a",
    "train_snpse_b",
    "train_snpse_c",
    # NLSE
    "NLSETrainer",
    "default_nlse_config",
    "train_nlse",
    # Prior scores
    "AnalyticPriorScore",
    "GaussianMixturePriorScore",
    "UniformPriorScore",
    "gaussian_mixture_perturbed_log_prob",
    "gaussian_mixture_perturbed_score",
    "make_prior_score_fn",
    "normal_cdf",
    "train_prior_score_network",
    "uniform_perturbed_log_prob",
    "uniform_perturbed_score",
    # HPR
    "HPR_EPSILON_DEFAULT",
    "HPR_N_SAMPLES_DEFAULT",
    "HPRRegion",
    "HPRTruncation",
    "PriorSupportRegion",
    "TruncatedProposalSampler",
    "estimate_hpr_threshold",
    "sample_truncated_proposal",
    # Metrics
    "c2st_score",
    "c2st_nn_score",
    "expected_calibration_error",
    "median_heuristic_bandwidth",
    "mmd",
    "posterior_predictive_check",
    "rbf_mmd",
    "simulation_based_calibration",
]
