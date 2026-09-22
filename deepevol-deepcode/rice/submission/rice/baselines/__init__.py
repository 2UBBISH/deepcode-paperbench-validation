"""Baseline methods for RICE (Cheng et al., ICML 2024, PMLR 235).

This package collects every comparison method used in the paper's experiments:

Explanation baselines (Section 4.1, "Baseline Explanation Methods")
------------------------------------------------------------------
* ``Random``                  -- pick a uniformly random visited state as the
  critical state (:mod:`rice.baselines.random_explanation`).
* ``StateMask``               -- the released StateMask method
  (github.com/nuwuxian/RL-state_mask), i.e. primal-dual mask training
  ``min |eta(pi) - eta(pi_bar)|`` via a Lagrangian multiplier alpha
  (:mod:`rice.baselines.statemask_r`).
* ``Integrated Gradients``    -- optional attribution-based explanation
  (:mod:`rice.baselines.integrated_gradients`).
* ``AIRS``                    -- optional adversarial/attribution-based
  explanation (:mod:`rice.baselines.airs`).

Refining baselines (Section 4.1, "Baseline Refining Methods")
-------------------------------------------------------------
* ``PPO fine-tuning``         -- continue PPO with a lowered learning rate
  (:mod:`rice.baselines.ppo_finetune`).
* ``StateMask-R``             -- reset to the critical state and fine-tune
  (always reset: ``p = beta = 1``) (:mod:`rice.baselines.statemask_r`).
* ``JSRL``                    -- Jump-Start RL curriculum with the pre-trained
  policy as the guide (github.com/steventango/jumpstart-rl)
  (:mod:`rice.baselines.jsrl`).
* ``SAC fine-tuning``         -- Experiment IV on a pre-trained SAC agent
  (:mod:`rice.baselines.sac_finetune`).
* ``GAIL``                    -- learn an approximate policy from an expert
  (used in Experiment IV) (:mod:`rice.baselines.gail`).
* ``SIL``                     -- self-imitation learning (optional, Table 5)
  (:mod:`rice.baselines.sil`).

All refining baselines use the *same* explanation as RICE whenever an
explanation is required, so that the comparison isolates the refining
mechanism (Section 4.2, Experiment II and III).

The module is intentionally defensive: every sub-module is imported inside a
``try/except`` so that the package stays importable (and the registry stays
introspectable) even when an optional dependency or a not-yet-implemented
baseline is missing.  ``available_baselines()`` reports what was imported.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

_LOGGER = logging.getLogger("rice.baselines")

__all__: List[str] = []

# --------------------------------------------------------------------------- #
# Explanation baselines
# --------------------------------------------------------------------------- #
_HAS_RANDOM = False
try:  # pragma: no cover - defensive import
    from .random_explanation import (  # noqa: F401
        RandomExplanation,
        RandomExplanationConfig,
        RandomImportanceScorer,
        RandomMaskNetwork,
        RandomStateSelector,
        build_random_explanation,
        identify_random_state,
        make_random_explanation,
        random_critical_state,
        random_explanation_for,
        random_explanation_scores,
        random_importance_scores,
        select_random_states,
    )

    _HAS_RANDOM = True
    __all__ += [
        "RandomExplanation",
        "RandomExplanationConfig",
        "RandomImportanceScorer",
        "RandomMaskNetwork",
        "RandomStateSelector",
        "build_random_explanation",
        "identify_random_state",
        "make_random_explanation",
        "random_critical_state",
        "random_explanation_for",
        "random_explanation_scores",
        "random_importance_scores",
        "select_random_states",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("random_explanation baseline unavailable: %s", _exc)

_HAS_STATEMASK = False
try:  # pragma: no cover - defensive import
    from .statemask_r import (  # noqa: F401
        DEFAULT_MASK_SAMPLES,
        StateMaskConfig,
        StateMaskExplainer,
        StateMaskR,
        StateMaskRefiner,
        StateMaskTrainer,
        describe_statemask_r,
        make_statemask_explainer,
        make_statemask_r,
        refine_from_critical_state,
        samples_for,
        statemask_r,
        statemask_r_baseline,
        train_statemask_network,
    )

    _HAS_STATEMASK = True
    __all__ += [
        "DEFAULT_MASK_SAMPLES",
        "StateMaskConfig",
        "StateMaskExplainer",
        "StateMaskR",
        "StateMaskRefiner",
        "StateMaskTrainer",
        "describe_statemask_r",
        "make_statemask_explainer",
        "make_statemask_r",
        "refine_from_critical_state",
        "samples_for",
        "statemask_r",
        "statemask_r_baseline",
        "train_statemask_network",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("statemask_r baseline unavailable: %s", _exc)

_HAS_IG = False
try:  # pragma: no cover - optional / defensive import
    from .integrated_gradients import (  # noqa: F401
        IntegratedGradients,
        IntegratedGradientsConfig,
        make_integrated_gradients,
    )

    _HAS_IG = True
    __all__ += [
        "IntegratedGradients",
        "IntegratedGradientsConfig",
        "make_integrated_gradients",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("integrated_gradients baseline unavailable: %s", _exc)

_HAS_AIRS = False
try:  # pragma: no cover - optional / defensive import
    from .airs import AIRS, AIRSConfig, make_airs  # noqa: F401

    _HAS_AIRS = True
    __all__ += ["AIRS", "AIRSConfig", "make_airs"]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("airs baseline unavailable: %s", _exc)

# --------------------------------------------------------------------------- #
# Refining baselines
# --------------------------------------------------------------------------- #
_HAS_PPO_FINETUNE = False
try:  # pragma: no cover - defensive import
    from .ppo_finetune import (  # noqa: F401
        DEFAULT_FINETUNE_LR,
        PPOFinetuneBaseline,
        PPOFinetuneConfig,
        PPOFinetuner,
        describe_ppo_finetune,
        finetune,
        make_finetuner,
        ppo_finetune_policy,
        train_ppo_finetune,
    )

    _HAS_PPO_FINETUNE = True
    __all__ += [
        "DEFAULT_FINETUNE_LR",
        "PPOFinetuneBaseline",
        "PPOFinetuneConfig",
        "PPOFinetuner",
        "describe_ppo_finetune",
        "finetune",
        "make_finetuner",
        "ppo_finetune_policy",
        "train_ppo_finetune",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("ppo_finetune baseline unavailable: %s", _exc)

_HAS_JSRL = False
try:  # pragma: no cover - defensive import
    from .jsrl import (  # noqa: F401
        JSRLConfig,
        JSRLRefiner,
        describe_jsrl,
        jsrl,
        jsrl_baseline,
        make_jsrl,
    )

    _HAS_JSRL = True
    __all__ += [
        "JSRLConfig",
        "JSRLRefiner",
        "describe_jsrl",
        "jsrl",
        "jsrl_baseline",
        "make_jsrl",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("jsrl baseline unavailable: %s", _exc)

_HAS_SAC_FINETUNE = False
try:  # pragma: no cover - defensive import
    from .sac_finetune import (  # noqa: F401
        SACFinetuneConfig,
        SACFinetuner,
        describe_sac_finetune,
        make_sac_finetuner,
        sac_finetune_policy,
    )

    _HAS_SAC_FINETUNE = True
    __all__ += [
        "SACFinetuneConfig",
        "SACFinetuner",
        "describe_sac_finetune",
        "make_sac_finetuner",
        "sac_finetune_policy",
    ]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("sac_finetune baseline unavailable: %s", _exc)

_HAS_GAIL = False
try:  # pragma: no cover - optional / defensive import
    from .gail import GAILConfig, GAILRefiner, gail, make_gail  # noqa: F401

    _HAS_GAIL = True
    __all__ += ["GAILConfig", "GAILRefiner", "gail", "make_gail"]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("gail baseline unavailable: %s", _exc)

_HAS_SIL = False
try:  # pragma: no cover - optional / defensive import
    from .sil import SILConfig, SILRefiner, make_sil, sil  # noqa: F401

    _HAS_SIL = True
    __all__ += ["SILConfig", "SILRefiner", "make_sil", "sil"]
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("sil baseline unavailable: %s", _exc)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _registry() -> Dict[str, Dict[str, Any]]:
    """Return a mapping ``name -> {"available": bool, "factory": callable|None}``.

    The factory is a best-effort callable used by the ``main.py`` CLI to build a
    baseline instance without importing each module by hand.
    """
    def _pick(module_name: str, *candidates: Any) -> Optional[Callable]:
        for cand in candidates:
            if callable(cand):
                return cand
        return None

    random_factory = _pick("random_explanation", globals().get("make_random_explanation"))
    statemask_factory = _pick("statemask_r", globals().get("make_statemask_r"))
    ig_factory = _pick("integrated_gradients", globals().get("make_integrated_gradients"))
    airs_factory = _pick("airs", globals().get("make_airs"))
    ppo_ft_factory = _pick("ppo_finetune", globals().get("make_finetuner"))
    jsrl_factory = _pick("jsrl", globals().get("make_jsrl"))
    sac_factory = _pick("sac_finetune", globals().get("make_sac_finetuner"))
    gail_factory = _pick("gail", globals().get("make_gail"))
    sil_factory = _pick("sil", globals().get("make_sil"))

    return {
        # explanation baselines
        "random": {"available": _HAS_RANDOM, "kind": "explanation", "factory": random_factory},
        "statemask": {"available": _HAS_STATEMASK, "kind": "explanation", "factory": statemask_factory},
        "integrated_gradients": {"available": _HAS_IG, "kind": "explanation", "factory": ig_factory},
        "airs": {"available": _HAS_AIRS, "kind": "explanation", "factory": airs_factory},
        # refining baselines
        "ppo_finetune": {"available": _HAS_PPO_FINETUNE, "kind": "refining", "factory": ppo_ft_factory},
        "statemask_r": {"available": _HAS_STATEMASK, "kind": "refining", "factory": statemask_factory},
        "jsrl": {"available": _HAS_JSRL, "kind": "refining", "factory": jsrl_factory},
        "sac_finetune": {"available": _HAS_SAC_FINETUNE, "kind": "refining", "factory": sac_factory},
        "gail": {"available": _HAS_GAIL, "kind": "refining", "factory": gail_factory},
        "sil": {"available": _HAS_SIL, "kind": "refining", "factory": sil_factory},
    }


def available_baselines(include_unavailable: bool = False) -> Dict[str, bool]:
    """Report which baseline modules were imported successfully.

    Args:
        include_unavailable: when ``True`` also list baselines whose module is
            missing (mapped to ``False``); otherwise only the imported ones.

    Returns:
        Mapping of baseline name to availability flag.
    """
    reg = _registry()
    if include_unavailable:
        return {name: info["available"] for name, info in reg.items()}
    return {name: info["available"] for name, info in reg.items() if info["available"]}


def get_baseline_factory(name: str) -> Optional[Callable]:
    """Return the factory callable for a baseline name (``None`` if missing)."""
    key = str(name).strip().lower().replace("-", "_")
    return _registry().get(key, {}).get("factory")


def make_baseline(name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a baseline by name.

    Raises:
        KeyError: if the baseline is unknown.
        RuntimeError: if the baseline module failed to import.
    """
    key = str(name).strip().lower().replace("-", "_")
    reg = _registry()
    if key not in reg:
        raise KeyError(f"unknown baseline '{name}'; available: {sorted(reg)}")
    if not reg[key]["available"]:
        raise RuntimeError(f"baseline '{name}' is not available (import failed)")
    factory = reg[key]["factory"]
    if factory is None:
        raise RuntimeError(f"baseline '{name}' exposes no factory")
    return factory(*args, **kwargs)


def describe_baselines() -> str:
    """One-line summary of the imported baselines, for logging."""
    available = available_baselines()
    missing = [n for n, ok in available_baselines(include_unavailable=True).items() if not ok]
    parts = [f"{n}" for n in sorted(available)]
    text = "baselines available: " + (", ".join(parts) if parts else "none")
    if missing:
        text += " | missing: " + ", ".join(sorted(missing))
    return text


__all__ += [
    "available_baselines",
    "get_baseline_factory",
    "make_baseline",
    "describe_baselines",
]
