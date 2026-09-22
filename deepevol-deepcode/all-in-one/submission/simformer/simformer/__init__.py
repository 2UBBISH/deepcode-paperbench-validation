"""Simformer: Simulation-based inference with probabilistic diffusion models.

This package implements the Simformer model from the paper
"Simformer: Simulation-based inference with probabilistic diffusion models".

The public API is exposed lazily so that importing the package does not
require every optional dependency (torch, scipy, sklearn, ...) to be present.
The main building blocks are

* :class:`~simformer.tokenizer.Tokenizer` -- SBI tokenizer for ``(theta, x)``.
* :class:`~simformer.transformer.SimformerScoreNetwork` -- transformer score net.
* :mod:`~simformer.diffusion` -- VESDE/VPSDE and reverse-time samplers.
* :mod:`~simformer.condition_masks` -- per-sample condition masks ``M_C``.
* :mod:`~simformer.attention_masks` -- structure-aware attention masks ``M_E``.
* :mod:`~simformer.graph_inversion` -- Webb (2018) graph inversion for
  conditioning-directed masks.
* :mod:`~simformer.training` -- masked denoising score matching.
* :mod:`~simformer.sampling` -- arbitrary-conditional reverse-SDE sampling.
* :mod:`~simformer.guidance` -- Algorithm 1 general guidance / constraints.
* :mod:`~simformer.embedding_nets` -- optional embedding nets for high-dim data.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__version__ = "0.1.0"

__all__ = [
    # defaults / meta
    "DEFAULT_TOKEN_DIM",
    "DEFAULT_N_LAYERS",
    "DEFAULT_N_HEADS",
    "DEFAULT_ATTENTION_SIZE",
    "DEFAULT_WIDENING_FACTOR",
    "DEFAULT_TIME_EMBED_DIM",
    # tokenizer
    "Tokenizer",
    "TokenSpec",
    "FunctionValuedSpec",
    "build_benchmark_spec",
    # transformer / score nets
    "TransformerScoreNetwork",
    "SimformerScoreNetwork",
    "ScoreNetwork",
    "ScoreNet",
    "Simformer",
    "TransformerConfig",
    "build_transformer",
    "build_score_network",
    "n_layers_for_task",
    # diffusion / SDE
    "SDE",
    "VESDE",
    "VPSDE",
    "VarianceExplodingSDE",
    "VariancePreservingSDE",
    "SDEConfig",
    "get_sde",
    "sde_from_config",
    "sample_reverse_sde",
    "reverse_sde_step",
    "probability_flow_ode",
    "log_likelihood_from_ode",
    "time_grid",
    # condition masks
    "ConditionMaskSampler",
    "ConditionMaskConfig",
    "sample_condition_masks",
    "apply_condition",
    "partially_noise",
    "posterior_condition_mask",
    "likelihood_condition_mask",
    "joint_mask",
    # attention masks
    "build_attention_mask",
    "mask_variants",
    "TASK_MASK_REGISTRY",
    # graph inversion
    "graph_inversion",
    "graph_inversion_batched",
    "adapt_attention_mask",
    "make_condition_aware_mask",
    # training
    "TrainingConfig",
    "SimformerTrainer",
    "train_simformer",
    "masked_denoising_score_matching_loss",
    # sampling
    "ConditionalSampler",
    "SamplingConfig",
    "SamplingResult",
    "sample_conditional",
    "sample_posterior",
    "sample_likelihood",
    "sample_arbitrary_conditionals",
    # guidance
    "GuidedSampler",
    "GuidanceConfig",
    "GuidanceResult",
    "general_guidance",
    "interval_constraint",
    # embedding nets
    "EmbeddingTokenizer",
    "EmbeddingSpec",
    "build_embedding_score_network",
    "build_gravitational_waves_model",
    # utils
    "GaussianFourierFeatures",
    "random_fourier_features",
    "available",
]

# Mapping from public symbol -> module path (relative to ``simformer`` package).
_SUBMODULES: Dict[str, str] = {}
for _name in (
    "Tokenizer",
    "TokenSpec",
    "FunctionValuedSpec",
    "build_benchmark_spec",
):
    _SUBMODULES[_name] = "tokenizer"

for _name in (
    "TransformerScoreNetwork",
    "SimformerScoreNetwork",
    "ScoreNetwork",
    "ScoreNet",
    "Simformer",
    "TransformerConfig",
    "build_transformer",
    "build_score_network",
    "n_layers_for_task",
):
    _SUBMODULES[_name] = "transformer"

for _name in (
    "SDE",
    "VESDE",
    "VPSDE",
    "VarianceExplodingSDE",
    "VariancePreservingSDE",
    "SDEConfig",
    "get_sde",
    "sde_from_config",
    "sample_reverse_sde",
    "reverse_sde_step",
    "probability_flow_ode",
    "log_likelihood_from_ode",
    "time_grid",
):
    _SUBMODULES[_name] = "diffusion"

for _name in (
    "ConditionMaskSampler",
    "ConditionMaskConfig",
    "sample_condition_masks",
    "apply_condition",
    "partially_noise",
    "posterior_condition_mask",
    "likelihood_condition_mask",
    "joint_mask",
):
    _SUBMODULES[_name] = "condition_masks"

for _name in (
    "build_attention_mask",
    "mask_variants",
    "TASK_MASK_REGISTRY",
):
    _SUBMODULES[_name] = "attention_masks"

for _name in (
    "graph_inversion",
    "graph_inversion_batched",
    "adapt_attention_mask",
    "make_condition_aware_mask",
):
    _SUBMODULES[_name] = "graph_inversion"

for _name in (
    "TrainingConfig",
    "SimformerTrainer",
    "train_simformer",
    "masked_denoising_score_matching_loss",
):
    _SUBMODULES[_name] = "training"

for _name in (
    "ConditionalSampler",
    "SamplingConfig",
    "SamplingResult",
    "sample_conditional",
    "sample_posterior",
    "sample_likelihood",
    "sample_arbitrary_conditionals",
    "make_score_fn",
    "random_conditional_targets",
):
    _SUBMODULES[_name] = "sampling"

for _name in (
    "GuidedSampler",
    "GuidanceConfig",
    "GuidanceResult",
    "general_guidance",
    "interval_constraint",
):
    _SUBMODULES[_name] = "guidance"

for _name in (
    "EmbeddingTokenizer",
    "EmbeddingSpec",
    "build_embedding_score_network",
    "build_gravitational_waves_model",
):
    _SUBMODULES[_name] = "embedding_nets"

for _name in (
    "GaussianFourierFeatures",
    "random_fourier_features",
):
    _SUBMODULES[_name] = "utils"

_MODULE_CACHE: Dict[str, Any] = {}


def _load_submodule(name: str) -> Any:
    """Import and cache a submodule of :mod:`simformer`."""
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ImportError:
        module = importlib.import_module(name)
    _MODULE_CACHE[name] = module
    return module


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = _load_submodule(_SUBMODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "available":
        return available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals()) + __all__))


def available() -> Dict[str, List[str]]:
    """Return a summary of the implemented submodules and their key classes."""
    return {
        "modules": [
            "tokenizer",
            "transformer",
            "diffusion",
            "condition_masks",
            "attention_masks",
            "graph_inversion",
            "training",
            "sampling",
            "guidance",
            "embedding_nets",
            "utils",
        ],
        "tasks": [
            "gaussian_linear",
            "gaussian_mixture",
            "two_moons",
            "slcp",
            "tree",
            "hmm",
            "lotka_volterra",
            "sird",
            "hodgkin_huxley",
            "gravitational_waves",
        ],
        "baselines": ["npe", "nle", "nre", "npse"],
        "eval": ["c2st", "coverage", "nll"],
    }
