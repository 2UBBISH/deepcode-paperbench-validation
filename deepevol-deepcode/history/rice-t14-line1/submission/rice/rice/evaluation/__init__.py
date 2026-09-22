"""RICE evaluation sub-package.

Aggregates the three evaluation pipelines of the paper:

* :mod:`rice.evaluation.fidelity_score` -- Experiment I: fidelity of an
  explanation method, ``Fidelity Score = log(d / d_max) - log(l / L)``
  (Source: Sec. 4.1 "Experiment Setup" -> Evaluation Metrics).
* :mod:`rice.evaluation.refining_eval` -- Experiments II/III/IV: final reward
  after refining (Table 1) and refining curves for the sparse MuJoCo games
  (Figure 2) / SAC-refining in Hopper (Figure 3).
* :mod:`rice.evaluation.hyperparam_sweep` -- Experiment V: sweeps over
  ``p`` in {0, 0.25, 0.5, 0.75, 1}, ``lambda`` in {0, 0.1, 0.01, 0.001} and
  ``alpha`` in {0.01, 0.001, 0.0001} (Source: Sec. 4.2 "Experiment Design" ->
  Experiment V).

Import strategy
---------------
The heavy modules (``torch``/``gym``) are imported *lazily*: ``import
rice.evaluation`` only needs the standard library and ``numpy`` (through the
light helper functions).  Attribute access (``rice.evaluation.X``) resolves the
symbols on demand through :pep:`562` module ``__getattr__``.  ``from
rice.evaluation import X`` also works thanks to this hook.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = [
    # fidelity (Experiment I)
    "FidelityEvaluator",
    "FidelityConfig",
    "FidelityResult",
    "EvalEpisode",
    "fidelity_score",
    "evaluate_explanation",
    "evaluate_methods",
    "sliding_window_average",
    "best_window",
    "make_importance_fn",
    "mask_actions",
    # refining (Experiments II/III/IV)
    "RefiningEvaluator",
    "RefiningConfig",
    "RefiningResult",
    "RefiningCurve",
    "evaluate_refining",
    "compare_refining_methods",
    "REFINING_METHODS",
    "EXPLANATION_METHODS",
    # hyperparameter sweep (Experiment V)
    "HyperparamSweep",
    "SweepConfig",
    "SweepResult",
    "DEFAULT_P_VALUES",
    "DEFAULT_LAMBDA_VALUES",
    "DEFAULT_ALPHA_VALUES",
    "sweep_p",
    "sweep_lambda",
    "sweep_alpha",
    "run_sweep",
    # metadata
    "TABLE1_REFERENCE",  # reference trend values from Table 1 (for sanity checks)
    "TABLE4_REFERENCE",  # reference mask-training sample budgets / timings
]


# --------------------------------------------------------------------------- #
# Reference numbers extracted from the paper (trend-checking only).
# --------------------------------------------------------------------------- #
#: Table 1 ("Agent Refining Performance"): task -> (no_refine, ppo_ft, jsrl,
#: statemask_r, ours, random, statemask).  Values are paper means; the addendum
#: asks for trends, not exact numbers.
TABLE1_REFERENCE: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {
        "no_refine": 3559.44, "ppo": 3638.75, "jsrl": 3635.08,
        "statemask_r": 3652.06, "ours": 3663.91,
        "random": 3648.98, "statemask": 3661.86,
    },
    "Walker2d-v3": {
        "no_refine": 3768.79, "ppo": 3965.63, "jsrl": 3963.57,
        "statemask_r": 3966.96, "ours": 3982.79,
        "random": 3969.64, "statemask": 3982.67,
    },
    "Reacher-v2": {
        "no_refine": -5.79, "ppo": -3.04, "jsrl": -3.23,
        "statemask_r": -3.45, "ours": -2.66,
        "random": -3.11, "statemask": -2.69,
    },
    "HalfCheetah-v3": {
        "no_refine": 2024.09, "ppo": 2133.31, "jsrl": 2128.04,
        "statemask_r": 2085.28, "ours": 2138.89,
        "random": 2132.01, "statemask": 2136.23,
    },
    "SelfishMining": {
        "no_refine": 14.36, "ppo": 14.93, "jsrl": 14.88,
        "statemask_r": 14.53, "ours": 16.56,
        "random": 15.09, "statemask": 16.49,
    },
    "CageChallenge2": {
        "no_refine": -23.64, "ppo": -23.58, "jsrl": -22.97,
        "statemask_r": -26.98, "ours": -20.02,
        "random": -25.94, "statemask": -20.07,
    },
    "Macro-v1": {
        "no_refine": 10.30, "ppo": 13.37, "jsrl": 11.26,
        "statemask_r": 7.62, "ours": 17.03,
        "random": 11.72, "statemask": 16.28,
    },
    # Malware mutation is explicitly OUT OF SCOPE for reproduction; kept only
    # so that lookups by the (excluded) table row do not KeyError.
    "MalwareMutation": {
        "no_refine": 42.20, "ppo": 49.33, "jsrl": 43.10,
        "statemask_r": 50.13, "ours": 57.53,
        "random": 48.60, "statemask": 57.16,
    },
}

#: Table 4 (Appendix C.3): mask-network training cost.  ``samples`` are the
#: fixed sample budgets quoted in the reproduction plan; ``time_ours`` /
#: ``time_statemask`` are the paper wall-clock seconds (paper reports an
#: average 16.8% drop in training time).
TABLE4_REFERENCE: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"samples": 3.0e5, "time_ours": 12426.0, "time_statemask": 15393.0},
    "Walker2d-v3": {"samples": 3.0e5, "time_ours": None, "time_statemask": None},
    "Reacher-v2": {"samples": 3.0e5, "time_ours": None, "time_statemask": None},
    "HalfCheetah-v3": {"samples": 3.0e5, "time_ours": None, "time_statemask": None},
    "SelfishMining": {"samples": 1.5e6, "time_ours": None, "time_statemask": None},
    "CageChallenge2": {"samples": 1.0e7, "time_ours": 65400.0, "time_statemask": 79382.0},
    "Macro-v1": {"samples": 2443260.0, "time_ours": None, "time_statemask": None},
}

#: Paper claim (Sec. 4.3): average drop in mask-network training time.
EFFICIENCY_DROP_REFERENCE: float = 0.168


# --------------------------------------------------------------------------- #
# Lazy attribute resolution
# --------------------------------------------------------------------------- #
_LAZY: Dict[str, Tuple[str, str]] = {}


def _register_lazy(module: str, names: List[str]) -> None:
    for _name in names:
        _LAZY[_name] = (module, _name)


_register_lazy(
    "rice.evaluation.fidelity_score",
    [
        "FidelityEvaluator", "FidelityConfig", "FidelityResult", "EvalEpisode",
        "fidelity_score", "evaluate_explanation", "evaluate_methods",
        "sliding_window_average", "best_window", "make_importance_fn",
        "mask_actions", "random_window_index", "estimated_d_max",
    ],
)
_register_lazy(
    "rice.evaluation.refining_eval",
    [
        "RefiningEvaluator", "RefiningConfig", "RefiningResult", "RefiningCurve",
        "evaluate_refining", "compare_refining_methods", "REFINING_METHODS",
        "EXPLANATION_METHODS", "summarize_table",
    ],
)
_register_lazy(
    "rice.evaluation.hyperparam_sweep",
    [
        "HyperparamSweep", "SweepConfig", "SweepResult", "DEFAULT_P_VALUES",
        "DEFAULT_LAMBDA_VALUES", "DEFAULT_ALPHA_VALUES", "sweep_p",
        "sweep_lambda", "sweep_alpha", "run_sweep",
    ],
)


def _import_relative(module: str) -> Any:
    """Import ``module`` (``rice.evaluation.X``) with a relative fallback."""
    import importlib

    candidates = [module, module.replace("rice.evaluation.", "rice.evaluation.", 1)]
    tail = module.rsplit(".", 1)[-1]
    candidates.append(f"{__name__}.{tail}")
    for cand in candidates:
        try:
            return importlib.import_module(cand)
        except ImportError:
            continue
    # last resort: plain tail import (works when the cwd is `rice/rice`)
    return importlib.import_module(tail)


def __getattr__(name: str) -> Any:
    """Lazily resolve the public evaluation symbols (PEP 562)."""
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = _import_relative(module_name)
        value = getattr(module, attr)
        globals()[name] = value  # cache
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + list(__all__)))
