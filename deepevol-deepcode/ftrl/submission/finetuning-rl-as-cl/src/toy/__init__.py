"""Toy examples (Appendix A) for "Fine-tuning Reinforcement Learning Models is
Secretly a Forgetting Mitigation Problem" (Wolczyk et al., 2024).

This sub-package holds the two analytical sanity checks used in the paper to
isolate *why* fine-tuning destroys pre-trained capabilities:

* :mod:`src.toy.two_state_mdp` -- the two-state MDP with a closed-form value
  function ``v_0(theta)`` (Appendix A.1).  Two counter-example families are
  reproduced:

  - *state coverage gap* -- the pre-trained optimum at ``theta = 0.11``
    (``v_0 = 2.22``) is not the global optimum ``v_0(1) = 10`` because the
    fine-tuning distribution does not visit the states that ``pi_*`` relied on;
  - *imperfect cloning gap* -- the pre-trained optimum at ``theta = 0.08``
    (``v_0 = 9.93``) sits in a local basin, so gradient ascent on the new
    objective moves away from it.

* :mod:`src.toy.apple_retrieval` -- the ``AppleRetrieval`` 1-D gridworld with a
  two-parameter linear sigmoid policy ``pi_{w,b}(o) = sigmoid(w * o + b)``
  trained with REINFORCE (Appendix A.2).  A Phase-2 solution is pre-trained and
  then fine-tuned on the full task while sweeping ``M`` (the distance to the
  apple) and ``c`` (the observation scale) to expose the forgetting mechanism.

Everything here is dependency light: NumPy is optional and PyTorch is not used
at all, so these experiments run on any CPU-only machine and act as the
cheapest end-to-end validation of the forgetting claims.

The module performs no eager imports -- sibling modules are resolved lazily via
PEP 562 ``__getattr__`` -- so that importing the package never fails when
optional plotting/array dependencies are absent.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    # sub-modules
    "two_state_mdp",
    "apple_retrieval",
    # helpers
    "load",
    "require",
    "available_modules",
    "missing_modules",
    "run_toy_examples",
    # two-state MDP (Appendix A.1)
    "TwoStateMDP",
    "Scenario",
    "LocalExtremum",
    "FineTuneResult",
    "SCENARIOS",
    "F_COVERAGE",
    "F_CLONING",
    "coverage_f",
    "coverage_f_grad",
    "cloning_f",
    "cloning_f_grad",
    "f_from_name",
    "f_grad_from_name",
    "two_state_value",
    "state1_value",
    "value_gradient",
    "numerical_gradient",
    "make_scenario",
    "scenario_from_config",
    "fine_tune",
    "local_extrema",
    "value_curve",
    "run_scenario",
    "plot_scenario",
    # AppleRetrieval (Appendix A.2)
    "AppleRetrievalEnv",
    "LinearSigmoidPolicy",
    "AppleConfig",
    "PHASE_1",
    "PHASE_2",
    "LEFT",
    "RIGHT",
    "sigmoid",
    "discounted_returns",
    "reinforce_gradient",
    "reinforce_update",
    "collect_episode",
    "pretrain_phase2",
    "fine_tune_full_task",
    "evaluate_policy",
    "policy_metrics",
    "run_apple_retrieval",
    "config_from_config",
    "summarize",
    "aggregate_results",
    "sweep_over_M",
    "sweep_over_c",
    "plot_sweep",
    "plot_finetuning_trace",
    "APPLE_RETRIEVAL_METRICS",
]


# ---------------------------------------------------------------------------
# Lazy module resolution
# ---------------------------------------------------------------------------

#: Short name -> dotted module path (relative imports preferred).
_MODULE_PATHS: Dict[str, str] = {
    "two_state_mdp": "src.toy.two_state_mdp",
    "apple_retrieval": "src.toy.apple_retrieval",
}

#: Re-exported symbol -> (short module name, attribute name).
_REEXPORTS: Dict[str, Tuple[str, str]] = {
    # ------------------------------------------------------------------
    # Appendix A.1 -- two-state MDP
    # ------------------------------------------------------------------
    "TwoStateMDP": ("two_state_mdp", "TwoStateMDP"),
    "Scenario": ("two_state_mdp", "Scenario"),
    "LocalExtremum": ("two_state_mdp", "LocalExtremum"),
    "FineTuneResult": ("two_state_mdp", "FineTuneResult"),
    "SCENARIOS": ("two_state_mdp", "SCENARIOS"),
    "F_COVERAGE": ("two_state_mdp", "F_COVERAGE"),
    "F_CLONING": ("two_state_mdp", "F_CLONING"),
    "coverage_f": ("two_state_mdp", "coverage_f"),
    "coverage_f_grad": ("two_state_mdp", "coverage_f_grad"),
    "cloning_f": ("two_state_mdp", "cloning_f"),
    "cloning_f_grad": ("two_state_mdp", "cloning_f_grad"),
    "f_from_name": ("two_state_mdp", "f_from_name"),
    "f_grad_from_name": ("two_state_mdp", "f_grad_from_name"),
    "two_state_value": ("two_state_mdp", "two_state_value"),
    "state1_value": ("two_state_mdp", "state1_value"),
    "value_gradient": ("two_state_mdp", "value_gradient"),
    "numerical_gradient": ("two_state_mdp", "numerical_gradient"),
    "make_scenario": ("two_state_mdp", "make_scenario"),
    "scenario_from_config": ("two_state_mdp", "scenario_from_config"),
    "fine_tune": ("two_state_mdp", "fine_tune"),
    "local_extrema": ("two_state_mdp", "local_extrema"),
    "value_curve": ("two_state_mdp", "value_curve"),
    "run_scenario": ("two_state_mdp", "run_scenario"),
    "plot_scenario": ("two_state_mdp", "plot_scenario"),
    # ------------------------------------------------------------------
    # Appendix A.2 -- AppleRetrieval
    # ------------------------------------------------------------------
    "AppleRetrievalEnv": ("apple_retrieval", "AppleRetrievalEnv"),
    "LinearSigmoidPolicy": ("apple_retrieval", "LinearSigmoidPolicy"),
    "AppleConfig": ("apple_retrieval", "AppleConfig"),
    "PHASE_1": ("apple_retrieval", "PHASE_1"),
    "PHASE_2": ("apple_retrieval", "PHASE_2"),
    "LEFT": ("apple_retrieval", "LEFT"),
    "RIGHT": ("apple_retrieval", "RIGHT"),
    "sigmoid": ("apple_retrieval", "sigmoid"),
    "discounted_returns": ("apple_retrieval", "discounted_returns"),
    "reinforce_gradient": ("apple_retrieval", "reinforce_gradient"),
    "reinforce_update": ("apple_retrieval", "reinforce_update"),
    "collect_episode": ("apple_retrieval", "collect_episode"),
    "pretrain_phase2": ("apple_retrieval", "pretrain_phase2"),
    "fine_tune_full_task": ("apple_retrieval", "fine_tune_full_task"),
    "evaluate_policy": ("apple_retrieval", "evaluate_policy"),
    "policy_metrics": ("apple_retrieval", "policy_metrics"),
    "run_apple_retrieval": ("apple_retrieval", "run_apple_retrieval"),
    "config_from_config": ("apple_retrieval", "config_from_config"),
    "summarize": ("apple_retrieval", "summarize"),
    "aggregate_results": ("apple_retrieval", "aggregate_results"),
    "sweep_over_M": ("apple_retrieval", "sweep_over_M"),
    "sweep_over_c": ("apple_retrieval", "sweep_over_c"),
    "plot_sweep": ("apple_retrieval", "plot_sweep"),
    "plot_finetuning_trace": ("apple_retrieval", "plot_finetuning_trace"),
    "APPLE_RETRIEVAL_METRICS": ("apple_retrieval", "APPLE_RETRIEVAL_METRICS"),
}

#: Memoised successfully imported modules.
_CACHE: Dict[str, ModuleType] = {}

#: Names whose import previously failed (avoid retrying / noisy tracebacks).
_FAILED: Dict[str, str] = {}


def _resolve(name: str) -> List[str]:
    """Return candidate dotted paths for ``name`` (short or dotted)."""
    if "." in name:
        return [name]
    candidates = []
    mapped = _MODULE_PATHS.get(name)
    if mapped:
        candidates.append(mapped)
    # Bare package-relative and top-level fallbacks.
    candidates.append(f"src.toy.{name}")
    candidates.append(f"toy.{name}")
    candidates.append(name)
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def load(name: str, required: bool = False) -> Optional[ModuleType]:
    """Import one toy sub-module lazily.

    Parameters
    ----------
    name:
        Short name (``"two_state_mdp"``, ``"apple_retrieval"``) or a dotted
        module path.
    required:
        When ``True`` the underlying :class:`ImportError` is re-raised; when
        ``False`` (default) ``None`` is returned on failure.

    Returns
    -------
    module or None
        The imported module, or ``None`` when unavailable and ``required`` is
        ``False``.
    """
    if name in _CACHE:
        return _CACHE[name]
    if name in _FAILED:
        if required:
            raise ImportError(_FAILED[name])
        return None

    last_error: Optional[BaseException] = None
    for path in _resolve(name):
        try:
            module = importlib.import_module(path)
        except Exception as exc:  # pragma: no cover - depends on environment
            last_error = exc
            continue
        _CACHE[name] = module
        _FAILED.pop(name, None)
        return module

    message = f"could not import toy module {name!r}: {last_error}"
    _FAILED[name] = message
    if required:
        raise ImportError(message) from last_error
    return None


def require(name: str) -> ModuleType:
    """Like :func:`load` but always raises :class:`ImportError` on failure."""
    module = load(name, required=True)
    if module is None:  # pragma: no cover - defensive
        raise ImportError(f"could not import toy module {name!r}")
    return module


def available_modules() -> List[str]:
    """Sorted short names of toy modules that currently import successfully."""
    return sorted(name for name in _MODULE_PATHS if load(name) is not None)


def missing_modules() -> List[str]:
    """Sorted short names of toy modules whose import failed."""
    return sorted(name for name in _MODULE_PATHS if load(name) is None)


def _module_of(symbol: str) -> ModuleType:
    short = _REEXPORTS[symbol][0]
    return require(short)


def run_toy_examples(
    output_dir: Optional[str] = None,
    *,
    mdp_scenarios: Optional[List[str]] = None,
    sweep: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run both Appendix-A sanity checks and return their summaries.

    Parameters
    ----------
    output_dir:
        Optional directory for JSON/plot artefacts.  When ``None`` nothing is
        written to disk.
    mdp_scenarios:
        Scenario names to run for the two-state MDP (defaults to both
        ``"coverage_gap"`` and ``"imperfect_cloning"``).
    sweep:
        Also run the (more expensive) ``AppleRetrieval`` sweeps over ``M`` and
        ``c``.
    verbose:
        Print progress lines.

    Returns
    -------
    dict
        ``{"two_state_mdp": {scenario: summary}, "apple_retrieval": summary,
        "sweeps": {...}}`` with any unavailable pieces marked
        ``"unavailable"``.
    """
    import os

    results: Dict[str, Any] = {}

    mdp = load("two_state_mdp")
    if mdp is None:
        results["two_state_mdp"] = "unavailable"
    else:
        if mdp_scenarios is None:
            mdp_scenarios = ["coverage_gap", "imperfect_cloning"]
        mdp_results: Dict[str, Any] = {}
        for scenario_name in mdp_scenarios:
            if verbose:
                print(f"[toy] two-state MDP scenario: {scenario_name}")
            try:
                mdp_results[scenario_name] = mdp.run_scenario(scenario_name)
            except Exception as exc:  # pragma: no cover - defensive
                mdp_results[scenario_name] = {"error": repr(exc)}
        results["two_state_mdp"] = mdp_results

    apple = load("apple_retrieval")
    if apple is None:
        results["apple_retrieval"] = "unavailable"
    else:
        if verbose:
            print("[toy] AppleRetrieval single run (M=30, c=1.0)")
        try:
            results["apple_retrieval"] = apple.run_apple_retrieval(M=30, c=1.0)
        except Exception as exc:  # pragma: no cover - defensive
            results["apple_retrieval"] = {"error": repr(exc)}

        if sweep:
            sweeps: Dict[str, Any] = {}
            for kind, fn in (("sweep_M", apple.sweep_over_M), ("sweep_c", apple.sweep_over_c)):
                if verbose:
                    print(f"[toy] AppleRetrieval {kind}")
                try:
                    sweeps[kind] = fn()
                except Exception as exc:  # pragma: no cover - defensive
                    sweeps[kind] = {"error": repr(exc)}
            results["sweeps"] = sweeps

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            import json

            json_path = os.path.join(output_dir, "toy_summary.json")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(_jsonable(results), handle, indent=2, default=str)
            results["summary_path"] = json_path
        except Exception as exc:  # pragma: no cover - defensive
            results.setdefault("warnings", []).append(f"could not write summary: {exc!r}")

    return results


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of nested containers into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return _jsonable(value.as_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    return repr(value)


# ---------------------------------------------------------------------------
# PEP 562 lazy attribute access
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly
    """Resolve sub-modules and re-exported symbols on first access."""
    if name in _MODULE_PATHS:
        module = load(name)
        if module is not None:
            globals()[name] = module
            return module
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name in _REEXPORTS:
        module = _module_of(name)
        attr = getattr(module, _REEXPORTS[name][1])
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - IDE convenience
    return sorted(set(globals()) | set(__all__))
