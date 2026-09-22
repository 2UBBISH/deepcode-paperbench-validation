"""Baseline refining methods for RICE (Section 4.1 of the paper).

The paper compares RICE against three (plus SIL / SAC-GAIL secondary) refining
baselines:

* **PPO fine-tuning** (Schulman et al., 2017): "lowering the learning rate and
  continuing training with the PPO algorithm" (Section 4.1).
* **StateMask-R** (Cheng et al., 2023): "resetting to the critical state and
  continuing training from the critical state" (Section 4.1); the released
  implementation is https://github.com/nuwuxian/RL-state_mask (Section C.1).
* **JSRL** (Uchendu et al., 2023): a guided policy ``pi_g`` sets up a curriculum
  to train an exploration policy ``pi_e``; "Through initializing
  ``pi_e = pi_g``, we can transform JSRL to be a refining method" (Section 4.1).
  Implementation from https://github.com/steventango/jumpstart-rl (Section C.1).
* **SIL** (Oh et al., 2018): self-imitation learning -- used in the secondary
  comparison of Table 5.
* **SAC + GAIL** (Experiment IV): pre-train SAC, learn an approximated policy
  with GAIL, then refine with RICE.

All baselines share the same PPO update (:mod:`rice.algorithms.ppo`) and, where
relevant, the same Go-Explore-style state restoration
(:mod:`rice.algorithms.env_reset`) so that differences are attributable to the
refining strategy alone.

This package initializer uses PEP 562 lazy attribute resolution so that
``import rice.baselines`` never drags in torch/SB3 unless a baseline is actually
requested.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__: List[str] = [
    # ppo fine-tuning
    "PPOFineTuneConfig",
    "PPOFineTuner",
    "ppo_finetune",
    # statemask-r
    "StateMaskRConfig",
    "StateMaskRRefiner",
    "statemask_r_refine",
    # jsrl
    "JSRLConfig",
    "JSRLRefiner",
    "jsrl_refine",
    # self-imitation learning
    "SILConfig",
    "SILRefiner",
    "self_imitation_refine",
    # sac + gail
    "SACGAILConfig",
    "SACGAILPipeline",
    "sac_gail_refine",
    # registry helpers
    "BASELINE_REGISTRY",
    "BASELINE_NAMES",
    "available_baselines",
    "make_baseline",
    "run_baseline",
]

#: name -> (module, description).  ``REFINING_METHODS`` in
#: :mod:`rice.evaluation.refining_eval` uses these names.
BASELINE_REGISTRY: Dict[str, Tuple[str, str]] = {
    "ppo": ("ppo_finetune", "PPO fine-tuning (lowered learning rate)"),
    "ppo_finetune": ("ppo_finetune", "PPO fine-tuning (lowered learning rate)"),
    "statemask_r": ("statemask_r", "StateMask-R: reset to critical state + fine-tune"),
    "statemask-r": ("statemask_r", "StateMask-R: reset to critical state + fine-tune"),
    "jsrl": ("jsrl", "Jump-Start RL with pi_e <- pi_g"),
    "sil": ("self_imitation", "Self-Imitation Learning (Oh et al., 2018)"),
    "self_imitation": ("self_imitation", "Self-Imitation Learning (Oh et al., 2018)"),
    "sac_gail": ("sac_gail", "SAC pre-train + GAIL imitation, then refine (Exp. IV)"),
    "sac-gail": ("sac_gail", "SAC pre-train + GAIL imitation, then refine (Exp. IV)"),
}

#: canonical (deduplicated) baseline names
BASELINE_NAMES: Tuple[str, ...] = ("ppo_finetune", "statemask_r", "jsrl", "sil", "sac_gail")

#: lazily resolved symbol -> module (without the package prefix)
_LAZY: Dict[str, str] = {}


def _register_lazy(module: str, names: List[str]) -> None:
    for name in names:
        _LAZY[name] = module


_register_lazy(
    "ppo_finetune",
    ["PPOFineTuneConfig", "PPOFineTuner", "ppo_finetune"],
)
_register_lazy(
    "statemask_r",
    ["StateMaskRConfig", "StateMaskRRefiner", "statemask_r_refine"],
)
_register_lazy(
    "jsrl",
    ["JSRLConfig", "JSRLRefiner", "jsrl_refine"],
)
_register_lazy(
    "self_imitation",
    ["SILConfig", "SILRefiner", "self_imitation_refine"],
)
_register_lazy(
    "sac_gail",
    ["SACGAILConfig", "SACGAILPipeline", "sac_gail_refine"],
)


def _import_relative(module: str) -> Any:
    """Import ``rice.baselines.<module>`` tolerating several sys.path layouts."""
    import importlib

    candidates = [
        f"{__name__}.{module}",
        f"rice.baselines.{module}",
        f"rice.rice.baselines.{module}",
        module,
    ]
    last_error: Exception = ImportError(f"cannot import {module}")
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except ImportError as exc:  # pragma: no cover - defensive
            last_error = exc
    raise last_error


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod = _import_relative(_LAZY[name])
        value = getattr(mod, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


def available_baselines() -> List[str]:
    """Canonical baseline names understood by :func:`make_baseline`."""
    return list(BASELINE_NAMES)


def resolve_baseline_name(name: str) -> str:
    """Normalise a friendly baseline name to its canonical form."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in BASELINE_REGISTRY:
        raise KeyError(
            f"unknown baseline {name!r}; known baselines: {sorted(BASELINE_REGISTRY)}"
        )
    module = BASELINE_REGISTRY[key][0]
    # map module -> canonical name
    for canonical, (mod, _desc) in BASELINE_REGISTRY.items():
        if mod == module and canonical in BASELINE_NAMES:
            return canonical
    return module


def make_baseline(name: str = "ppo_finetune", **kwargs: Any) -> Any:
    """Build a baseline refiner instance (uninitialised w.r.t. env/policy).

    Parameters
    ----------
    name:
        One of :data:`BASELINE_NAMES` (aliases accepted).
    **kwargs:
        Forwarded to the baseline class constructor.

    Returns
    -------
    The baseline object.  Use its ``refine``/``run`` method with an env and a
    warm-start policy, or call :func:`run_baseline` for a one-shot run.
    """
    canonical = resolve_baseline_name(name)
    if canonical == "ppo_finetune":
        return getattr(_import_relative("ppo_finetune"), "PPOFineTuner")(**kwargs)
    if canonical == "statemask_r":
        return getattr(_import_relative("statemask_r"), "StateMaskRRefiner")(**kwargs)
    if canonical == "jsrl":
        return getattr(_import_relative("jsrl"), "JSRLRefiner")(**kwargs)
    if canonical == "sil":
        return getattr(_import_relative("self_imitation"), "SILRefiner")(**kwargs)
    if canonical == "sac_gail":
        return getattr(_import_relative("sac_gail"), "SACGAILPipeline")(**kwargs)
    raise KeyError(f"unknown baseline {name!r}")


def run_baseline(
    name: str = "ppo_finetune",
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    **kwargs: Any,
) -> Any:
    """One-shot run of a baseline refining method.

    Dispatches to the module-level functional entry points
    (``ppo_finetune`` / ``statemask_r_refine`` / ``jsrl_refine`` /
    ``self_imitation_refine`` / ``sac_gail_refine``).

    Returns a :class:`rice.algorithms.refine.RefineResult` (or a
    ``RefiningResult``-compatible object).
    """
    canonical = resolve_baseline_name(name)
    if canonical == "ppo_finetune":
        fn = getattr(_import_relative("ppo_finetune"), "ppo_finetune")
    elif canonical == "statemask_r":
        fn = getattr(_import_relative("statemask_r"), "statemask_r_refine")
    elif canonical == "jsrl":
        fn = getattr(_import_relative("jsrl"), "jsrl_refine")
    elif canonical == "sil":
        fn = getattr(_import_relative("self_imitation"), "self_imitation_refine")
    elif canonical == "sac_gail":
        fn = getattr(_import_relative("sac_gail"), "sac_gail_refine")
    else:  # pragma: no cover - guarded by resolve_baseline_name
        raise KeyError(f"unknown baseline {name!r}")
    return fn(env=env, policy=policy, mask_network=mask_network, **kwargs)
