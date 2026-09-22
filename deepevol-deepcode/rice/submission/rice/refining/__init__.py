"""RICE Stage-2 refining package (Algorithm 2).

This package implements the *refining* stage of RICE (Cheng et al., ICML 2024,
PMLR 235).  Given a frozen, pre-trained (locally-optimal / bottlenecked) policy
``pi`` and a Stage-1 mask-network explanation, the refinement engine optimises a
trainable copy ``pi_theta`` (initialised from ``pi``) with PPO on a **mixed
initial state distribution**

    mu(s) = beta * d_rho^pihat(s) + (1 - beta) * rho(s),

realised by Algorithm 2 as: with probability ``p`` (= ``beta``) roll the frozen
policy for ``K`` steps, identify the mask-argmax *critical* state and reset the
environment there; otherwise sample ``s_0 ~ rho`` from the default initial state
distribution.  The task reward is augmented with a normalised Random Network
Distillation (RND) intrinsic bonus

    R_t + lambda * || f(s_{t+1}) - fhat(s_{t+1}) ||^2 .

Sub-modules
-----------
``rnd``
    Frozen random target network ``f`` + trainable predictor ``fhat`` and the
    normalised intrinsic reward ``lambda * ||f - fhat||^2`` (Burda et al. 2018
    style running-std normalisation, since the paper leaves the form
    unspecified).
``mixed_init``
    Sampler for ``mu(s)`` (``RAND_NUM ~ U(0,1)``; reset to the critical state
    when ``RAND_NUM < p``, else ``s_0 ~ rho``).
``ppo_refine``
    The Algorithm-2 refinement loop: PPO (clipped surrogate, GAE, SB3 defaults
    where the paper is silent: gamma=0.99, clip=0.2, lr=3e-4, n_epochs=10,
    GAE lambda=0.95) over the mixed-init / RND-augmented dataset.

The package is intentionally tolerant of missing optional dependencies: each
sub-module is imported inside a ``try/except`` so ``import rice.refining`` never
fails on a minimal installation (no torch / no SB3 / no MuJoCo).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger("rice.refining")

__all__: List[str] = []

# ---------------------------------------------------------------------------
# RND intrinsic-reward module
# ---------------------------------------------------------------------------
_HAS_RND = False
try:  # pragma: no cover - availability depends on optional torch install
    from .rnd import (  # noqa: F401
        DEFAULT_LAMBDA,
        RND_DEFAULT_HIDDEN,
        RNDBonus,
        RNDConfig,
        RNDModule,
        RNDNetwork,
        build_rnd,
        build_rnd_network,
        make_rnd,
        normalize_bonus,
        rnd_bonus,
    )

    _HAS_RND = True
    __all__ += [
        "DEFAULT_LAMBDA",
        "RND_DEFAULT_HIDDEN",
        "RNDBonus",
        "RNDConfig",
        "RNDModule",
        "RNDNetwork",
        "build_rnd",
        "build_rnd_network",
        "make_rnd",
        "normalize_bonus",
        "rnd_bonus",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("rice.refining.rnd unavailable: %s", _exc)

# ---------------------------------------------------------------------------
# Mixed initial state distribution sampler mu(s)
# ---------------------------------------------------------------------------
_HAS_MIXED_INIT = False
try:  # pragma: no cover
    from .mixed_init import (  # noqa: F401
        DEFAULT_BETA,
        DEFAULT_P,
        InitSample,
        MixedInitConfig,
        MixedInitialStateSampler,
        build_mixed_init,
        build_mixed_init_sampler,
        describe_mixed_init,
        make_mixed_init_sampler,
        mixed_init_probability,
        mixture_weights,
        sample_initial_state,
        should_use_critical,
    )

    _HAS_MIXED_INIT = True
    __all__ += [
        "DEFAULT_BETA",
        "DEFAULT_P",
        "InitSample",
        "MixedInitConfig",
        "MixedInitialStateSampler",
        "build_mixed_init",
        "build_mixed_init_sampler",
        "describe_mixed_init",
        "make_mixed_init_sampler",
        "mixed_init_probability",
        "mixture_weights",
        "sample_initial_state",
        "should_use_critical",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("rice.refining.mixed_init unavailable: %s", _exc)

# ---------------------------------------------------------------------------
# Algorithm 2 refinement engine
# ---------------------------------------------------------------------------
_HAS_PPO_REFINE = False
try:  # pragma: no cover
    from .ppo_refine import (  # noqa: F401
        DEFAULT_BATCH_SIZE,
        DEFAULT_CLIP_RANGE,
        DEFAULT_GAE_LAMBDA,
        DEFAULT_GAMMA,
        DEFAULT_HORIZON,
        DEFAULT_N_EPOCHS,
        DEFAULT_PPO_LR,
        DEFAULT_TOTAL_TIMESTEPS,
        PPORefiner,
        RefinePPOConfig,
        RefineRollout,
        build_refiner,
        compute_gae,
        ensure_trainable_policy,
        evaluate_refined_policy,
        make_refiner,
        policy_action,
        prepare_action_for_env,
        refine,
        refine_policy,
        train_ppo_refine,
        unpack_reset,
        unpack_step,
    )

    _HAS_PPO_REFINE = True
    __all__ += [
        "DEFAULT_BATCH_SIZE",
        "DEFAULT_CLIP_RANGE",
        "DEFAULT_GAE_LAMBDA",
        "DEFAULT_GAMMA",
        "DEFAULT_HORIZON",
        "DEFAULT_N_EPOCHS",
        "DEFAULT_PPO_LR",
        "DEFAULT_TOTAL_TIMESTEPS",
        "PPORefiner",
        "RefinePPOConfig",
        "RefineRollout",
        "build_refiner",
        "compute_gae",
        "ensure_trainable_policy",
        "evaluate_refined_policy",
        "make_refiner",
        "policy_action",
        "prepare_action_for_env",
        "refine",
        "refine_policy",
        "train_ppo_refine",
        "unpack_reset",
        "unpack_step",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("rice.refining.ppo_refine unavailable: %s", _exc)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def available(include_unavailable: bool = False) -> Dict[str, bool]:
    """Report which refining sub-modules imported successfully.

    Parameters
    ----------
    include_unavailable:
        When ``False`` (default) only successfully imported sub-modules are
        listed; when ``True`` all known sub-modules appear with a boolean
        availability flag.
    """
    flags = {
        "rnd": _HAS_RND,
        "mixed_init": _HAS_MIXED_INIT,
        "ppo_refine": _HAS_PPO_REFINE,
    }
    if include_unavailable:
        return flags
    return {name: ok for name, ok in flags.items() if ok}


def require(submodule: str) -> Any:
    """Import and return a refining sub-module, raising a helpful error."""
    import importlib

    name = str(submodule).strip().lower().replace("-", "_")
    if name not in {"rnd", "mixed_init", "ppo_refine"}:
        raise KeyError(f"unknown rice.refining sub-module: {submodule!r}")
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"rice.refining.{name} is unavailable ({exc}). Install the optional "
            "dependencies (torch, stable-baselines3, gym/mujoco) to use the "
            "Stage-2 refinement pipeline."
        ) from exc


def describe() -> str:
    """One-line human-readable summary of the Stage-2 refining package."""
    flags = available(include_unavailable=True)
    parts = [f"{k}={'ok' if v else 'missing'}" for k, v in flags.items()]
    return "rice.refining [Stage-2 Algorithm 2: " + ", ".join(parts) + "]"


__all__ += ["available", "require", "describe"]
