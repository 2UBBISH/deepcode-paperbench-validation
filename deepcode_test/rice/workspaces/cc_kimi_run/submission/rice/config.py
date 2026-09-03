"""Hyperparameter and environment configuration for RICE reproduction."""
from typing import Any, Dict

# Hyperparameters reported in Table 3 of the paper.
HYPERPARAMETERS: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "Walker2d-v3": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "Reacher-v2": {"p": 0.50, "lambda": 0.001, "alpha": 0.0001},
    "HalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    # Sparse MuJoCo variants.
    "SparseHopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "SparseHalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    # Real-world applications.
    "SelfishMining-v0": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "CageChallenge2-v0": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "MetaDrive-v0": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
}

# Default training budgets used in the paper.
TRAINING_BUDGETS: Dict[str, Dict[str, int]] = {
    "Hopper-v3": {"target": 1_000_000, "mask": 300_000, "refine": 500_000},
    "Walker2d-v3": {"target": 1_000_000, "mask": 300_000, "refine": 500_000},
    "Reacher-v2": {"target": 1_000_000, "mask": 300_000, "refine": 500_000},
    "HalfCheetah-v3": {"target": 1_000_000, "mask": 300_000, "refine": 500_000},
    "SparseHopper-v3": {"target": 1_000_000, "mask": 300_000, "refine": 500_000},
    "SparseHalfCheetah-v3": {"target": 1_000_000, "mask": 300_000, "refine": 500_000},
}

# Observation normalization is used for Walker2d and HalfCheetah in the paper.
NORMALIZE_OBS_ENVS = {"Walker2d-v3", "HalfCheetah-v3"}


def get_hparams(env_id: str) -> Dict[str, float]:
    """Return RICE hyperparameters for an environment."""
    return HYPERPARAMETERS.get(
        env_id,
        {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    )


def get_budgets(env_id: str) -> Dict[str, int]:
    """Return default training budgets for an environment."""
    return TRAINING_BUDGETS.get(
        env_id,
        {"target": 1_000_000, "mask": 300_000, "refine": 500_000},
    )
