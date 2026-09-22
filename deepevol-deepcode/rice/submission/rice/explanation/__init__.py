"""RICE Stage-1 explanation package (ICML 2024, PMLR 235).

This package implements the *explanation* half of RICE: a binary mask network
``~pi_theta(a_t^m | s_t)`` (a simplified StateMask) that is trained with
**vanilla PPO plus a blinding bonus** ``R'_t = R(s_t, a_t) + alpha * a_t^m``
(Algorithm 1).  The mask output ``a_t^m in {0, 1}`` composes with the frozen
target policy's action through the masked-action operator

    executed action  =  a_t           if a_t^m = 0   ("keep")
                        a_random      if a_t^m = 1   ("blind")

and the *step-level state importance* of ``s_t`` is defined as the probability
that the mask outputs ``0`` (i.e. "keep"),

    I(s_t) = P(a_t^m = 0 | s_t).

That scalar drives (a) critical-state selection for the mixed initial state
distribution ``mu(s) = beta * d_rho^{pi_hat}(s) + (1 - beta) * rho(s)`` used by
Stage 2, and (b) the sliding-window fidelity metric of Experiment I

    fidelity = log(d / d_max) - log(l / L),   l = L x K,  K in {10, 20, 30, 40}%.

Sub-modules
-----------
``mask_network``   mask net module, masked-action operator, reward augmentation.
``mask_trainer``   Algorithm 1 trainer (vanilla PPO + blinding bonus) + timers.
``importance``     state-importance scoring ``P(mask = 0 | s)`` and ranking.
``critical_state`` length-K rollouts and argmax-importance critical states.
``fidelity``       sliding-window fidelity evaluator (Experiment I).

The package is deliberately import-tolerant: each sub-module is imported inside
an independent ``try/except`` so ``import rice.explanation`` still succeeds when
optional heavy dependencies (torch, MuJoCo, SB3) are unavailable.  Use
:func:`available` / :func:`require` to probe or fetch a sub-module explicitly.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional

__all__: List[str] = []

_LOGGER = logging.getLogger("rice.explanation")

_HAS_MASK_NETWORK = False
_HAS_MASK_TRAINER = False
_HAS_IMPORTANCE = False
_HAS_CRITICAL_STATE = False
_HAS_FIDELITY = False


# ---------------------------------------------------------------------------
# mask_network -- mask net + masked-action operator (Stage 1 primitives)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - availability depends on torch
    from .mask_network import (  # noqa: F401
        BLIND_INDEX,
        KEEP_INDEX,
        MASK_BLIND,
        MASK_KEEP,
        NUM_MASK_ACTIONS,
        MaskCritic,
        MaskedActionOperator,
        MaskNetwork,
        apply_mask,
        augmented_reward,
        blind_mask,
        blinding_bonus,
        build_mask_critic,
        build_mask_network,
        describe_mask_network,
        flatten_observation,
        keep_probability_from_logits,
        load_mask_network,
        mask_entropy,
        mask_from_logits,
        masked_action,
        masked_action_batch,
        save_mask_network,
        state_importance,
    )

    __all__ += [
        "BLIND_INDEX",
        "KEEP_INDEX",
        "MASK_BLIND",
        "MASK_KEEP",
        "NUM_MASK_ACTIONS",
        "MaskCritic",
        "MaskedActionOperator",
        "MaskNetwork",
        "apply_mask",
        "augmented_reward",
        "blind_mask",
        "blinding_bonus",
        "build_mask_critic",
        "build_mask_network",
        "describe_mask_network",
        "flatten_observation",
        "keep_probability_from_logits",
        "load_mask_network",
        "mask_entropy",
        "mask_from_logits",
        "masked_action",
        "masked_action_batch",
        "save_mask_network",
        "state_importance",
    ]
    _HAS_MASK_NETWORK = True
except Exception as exc:  # pragma: no cover
    _LOGGER.debug("rice.explanation.mask_network unavailable: %s", exc)


# ---------------------------------------------------------------------------
# mask_trainer -- Algorithm 1 (vanilla PPO + blinding bonus)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - availability depends on torch
    from .mask_trainer import (  # noqa: F401
        DEFAULT_ALPHA,
        MASK_ACTION_DIM,
        MaskEnv,
        MaskPPOConfig,
        MaskTrainer,
        RolloutBatch,
        compute_gae,
        make_mask_env,
        train_mask_network,
    )

    __all__ += [
        "DEFAULT_ALPHA",
        "MASK_ACTION_DIM",
        "MaskEnv",
        "MaskPPOConfig",
        "MaskTrainer",
        "RolloutBatch",
        "compute_gae",
        "make_mask_env",
        "train_mask_network",
    ]
    _HAS_MASK_TRAINER = True
except Exception as exc:  # pragma: no cover
    _LOGGER.debug("rice.explanation.mask_trainer unavailable: %s", exc)


# ---------------------------------------------------------------------------
# importance -- step-level state-importance scoring
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .importance import (  # noqa: F401
        DEFAULT_BATCH_SIZE,
        IMPORTANCE_MODES,
        ImportanceScorer,
        TrajectoryImportance,
        aggregate_importance,
        argmax_importance,
        attach_scores_to_wrapper,
        extract_observations,
        make_scorer,
        min_max_normalize,
        most_important_index,
        normalize_scores,
        rank_states,
        score_batch,
        score_observations,
        score_trajectory,
        score_trajectory_summary,
        softmax_normalize,
        summarize_importance,
        top_k_indices,
        trajectory_importance,
    )

    __all__ += [
        "DEFAULT_BATCH_SIZE",
        "IMPORTANCE_MODES",
        "ImportanceScorer",
        "TrajectoryImportance",
        "aggregate_importance",
        "argmax_importance",
        "attach_scores_to_wrapper",
        "extract_observations",
        "make_scorer",
        "min_max_normalize",
        "most_important_index",
        "normalize_scores",
        "rank_states",
        "score_batch",
        "score_observations",
        "score_trajectory",
        "score_trajectory_summary",
        "softmax_normalize",
        "summarize_importance",
        "top_k_indices",
        "trajectory_importance",
    ]
    _HAS_IMPORTANCE = True
except Exception as exc:  # pragma: no cover
    _LOGGER.debug("rice.explanation.importance unavailable: %s", exc)


# ---------------------------------------------------------------------------
# critical_state -- length-K rollout + argmax-importance critical state
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .critical_state import (  # noqa: F401
        DEFAULT_K,
        CriticalState,
        CriticalStateSelector,
        TrajectoryRollout,
        attach_critical_state,
        critical_state_from_rollout,
        default_k,
        describe_critical_state,
        identify_critical_state,
        most_critical_index,
        policy_action,
        rank_trajectory,
        roll_trajectory,
        select_critical_states,
        top_k_critical_states,
    )

    __all__ += [
        "DEFAULT_K",
        "CriticalState",
        "CriticalStateSelector",
        "TrajectoryRollout",
        "attach_critical_state",
        "critical_state_from_rollout",
        "default_k",
        "describe_critical_state",
        "identify_critical_state",
        "most_critical_index",
        "policy_action",
        "rank_trajectory",
        "roll_trajectory",
        "select_critical_states",
        "top_k_critical_states",
    ]
    _HAS_CRITICAL_STATE = True
except Exception as exc:  # pragma: no cover
    _LOGGER.debug("rice.explanation.critical_state unavailable: %s", exc)


# ---------------------------------------------------------------------------
# fidelity -- sliding-window fidelity evaluator (Experiment I)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .fidelity import (  # noqa: F401
        DEFAULT_HORIZON,
        DEFAULT_K_VALUES,
        DEFAULT_N_TRAJECTORIES,
        DEFAULT_SEEDS,
        FidelityConfig,
        FidelityEvaluator,
        FidelityResult,
        TrajectoryFidelityDetail,
        evaluate_fidelity,
        evaluate_fidelity_multi_K,
        evaluate_methods,
        fidelity_from_rewards,
        fidelity_score_from_change,
        format_fidelity_table,
        random_importance_scores,
        training_time_reduction,
    )

    __all__ += [
        "DEFAULT_HORIZON",
        "DEFAULT_K_VALUES",
        "DEFAULT_N_TRAJECTORIES",
        "DEFAULT_SEEDS",
        "FidelityConfig",
        "FidelityEvaluator",
        "FidelityResult",
        "TrajectoryFidelityDetail",
        "evaluate_fidelity",
        "evaluate_fidelity_multi_K",
        "evaluate_methods",
        "fidelity_from_rewards",
        "fidelity_score_from_change",
        "format_fidelity_table",
        "random_importance_scores",
        "training_time_reduction",
    ]
    _HAS_FIDELITY = True
except Exception as exc:  # pragma: no cover
    _LOGGER.debug("rice.explanation.fidelity unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------
_SUBMODULES = ("mask_network", "mask_trainer", "importance", "critical_state", "fidelity")

_AVAILABILITY = {
    "mask_network": _HAS_MASK_NETWORK,
    "mask_trainer": _HAS_MASK_TRAINER,
    "importance": _HAS_IMPORTANCE,
    "critical_state": _HAS_CRITICAL_STATE,
    "fidelity": _HAS_FIDELITY,
}


def available(include_unavailable: bool = False) -> Dict[str, bool]:
    """Return availability flags for the explanation sub-modules.

    Parameters
    ----------
    include_unavailable:
        When ``False`` (default) only successfully imported sub-modules are
        reported; when ``True`` every known sub-module is listed with its flag.
    """
    if include_unavailable:
        return dict(_AVAILABILITY)
    return {name: flag for name, flag in _AVAILABILITY.items() if flag}


def require(submodule: str) -> Any:
    """Import and return one explanation sub-module by name.

    Raises
    ------
    KeyError
        If ``submodule`` is not one of the known Stage-1 modules.
    RuntimeError
        If the module exists but could not be imported (missing dependency).
    """
    name = str(submodule).strip().lower().replace("-", "_")
    if name not in _SUBMODULES:
        raise KeyError(
            f"unknown explanation sub-module {submodule!r}; expected one of {_SUBMODULES}"
        )
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"could not import rice.explanation.{name}: {exc}. "
            "Install the optional dependencies (torch, gym/mujoco, stable-baselines3)."
        ) from exc


def describe() -> str:
    """One-line human-readable summary of the Stage-1 modules."""
    ok = [name for name, flag in _AVAILABILITY.items() if flag]
    missing = [name for name, flag in _AVAILABILITY.items() if not flag]
    return (
        "rice.explanation (Stage 1: mask net + vanilla PPO with blinding bonus) "
        f"available={ok} missing={missing}"
    )


__all__ += ["available", "require", "describe"]
