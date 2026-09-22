"""Explanation methods for RICE (step-level explanations of DRL agents).

This sub-package holds every black-box *explanation* provider used by RICE:

* :mod:`rice.explanation.random_explanation` -- the "Random" baseline of §4.1
  ("Random identifies critical steps by randomly selecting a visited state as
  the critical state.").
* :mod:`rice.explanation.statemask_adapter` -- the StateMask baseline of
  Cheng et al. (2023), i.e. the explanation our re-designed mask network is
  compared against in Experiment I / Experiment III.
* :mod:`rice.explanation.integrated_gradients` -- Integrated Gradients
  (Sundararajan et al. 2017), used in the §C.3 "Impact of Other Explanation
  Methods" comparison.
* :mod:`rice.explanation.airs_adapter` -- AIRS (Yu et al. 2023), the second
  additional explanation method of the same §C.3 comparison (Table 6).

Every provider implements the same duck-typed surface so that the refining loop
(:mod:`rice.algorithms.refine`), the mask-network machinery and the fidelity
evaluator (:mod:`rice.evaluation.fidelity_score`) can treat them uniformly::

    importance(states) -> np.ndarray          # per-state importance scores
    score(states)      -> np.ndarray          # alias of importance
    mask_prob_zero(states) -> np.ndarray      # P(a^m = 0 | s); == importance
    select_index(states) -> int               # argmax importance (critical step)
    best_index(states)   -> int               # alias of select_index
    critical_state(states, return_index=False)
    reset(seed=None)                          # re-seed (per-episode hook)
    state_dict() / load_state_dict(state)     # reproducible 3-seed runs

Importing this package is cheap: the providers are resolved lazily through the
PEP 562 ``__getattr__`` hook, so ``import rice.explanation`` never pulls in
torch / gym.  The registry below (``EXPLANATION_REGISTRY``/``make_explanation``)
is the single dispatch point used by the scripts and the evaluation layer.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    # ---- registry / dispatch -------------------------------------------------
    "EXPLANATION_REGISTRY",
    "EXPLANATION_NAMES",
    "EXPLANATION_ALIASES",
    "available_explanations",
    "resolve_explanation_name",
    "get_explanation_spec",
    "make_explanation",
    "build_explanation",
    "register_explanation",
    "explanation_importance",
    "select_critical_state_from_explanation",
    # ---- Random (baseline, §4.1) --------------------------------------------
    "RandomExplanation",
    "RandomExplanationConfig",
    "make_random_explanation",
    "build_random_explanation",
    # ---- StateMask (baseline, §4.1 / Cheng et al. 2023) ----------------------
    "StateMaskExplanation",
    "StateMaskExplanationConfig",
    "StateMaskAdapter",
    "make_statemask_explanation",
    "build_statemask",
    "train_statemask",
    # ---- Integrated Gradients (Sundararajan et al. 2017; §C.3) ---------------
    "IntegratedGradientsExplanation",
    "IntegratedGradientsConfig",
    "make_integrated_gradients",
    "build_integrated_gradients",
    # ---- AIRS (Yu et al. 2023; §C.3) ----------------------------------------
    "AIRSExplanation",
    "AIRSConfig",
    "make_airs_explanation",
    "build_airs",
    # ---- reference trends ----------------------------------------------------
    "TABLE6_REFERENCE",
    "TABLE6_ORDERING",
]


# ---------------------------------------------------------------------------
# Registry metadata: friendly name -> (module, factory attribute, description)
# ---------------------------------------------------------------------------
# "ours" is the re-designed mask network of §3.3; it lives in
# ``rice.algorithms.mask_network`` (Algorithm 1) but is exposed here as an
# explanation provider so that Experiment III can vary the explanation while
# keeping the refining method fixed.
EXPLANATION_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "ours": (
        "statemask_adapter",
        "make_statemask_explanation",
        "RICE re-designed mask network (Algorithm 1; importance = P(a^m = 0 | s))",
    ),
    "statemask": (
        "statemask_adapter",
        "make_statemask_explanation",
        "StateMask (Cheng et al. 2023) baseline explanation",
    ),
    "random": (
        "random_explanation",
        "make_random_explanation",
        "Random baseline: uniformly random visited state as the critical state",
    ),
    "integrated_gradients": (
        "integrated_gradients",
        "make_integrated_gradients",
        "Integrated Gradients (Sundararajan et al. 2017)",
    ),
    "airs": (
        "airs_adapter",
        "make_airs_explanation",
        "AIRS (Yu et al. 2023)",
    ),
}

#: Aliases accepted by :func:`resolve_explanation_name` (normalised to the
#: canonical keys of :data:`EXPLANATION_REGISTRY`).
EXPLANATION_ALIASES: Dict[str, str] = {
    "our": "ours",
    "rnd_mask": "ours",
    "mask": "ours",
    "mask_network": "ours",
    "rice": "ours",
    "state_mask": "statemask",
    "state-mask": "statemask",
    "statemask_adapter": "statemask",
    "rand": "random",
    "random_explanation": "random",
    "random-explanation": "random",
    "ig": "integrated_gradients",
    "integrated_gradient": "integrated_gradients",
    "integrated-gradients": "integrated_gradients",
    "grad": "integrated_gradients",
    "airs_adapter": "airs",
    "airs_explanation": "airs",
}

#: Canonical explanation names, in the order used in Table 6's comparison.
EXPLANATION_NAMES: Tuple[str, ...] = ("ours", "statemask", "random", "integrated_gradients", "airs")

# ---------------------------------------------------------------------------
# Lazily resolved symbols: name -> (module, attribute)
# ---------------------------------------------------------------------------
_LAZY: Dict[str, Tuple[str, str]] = {}


def _register_lazy(module: str, names: List[str]) -> None:
    """Populate :data:`_LAZY` with ``name -> (module, name)`` entries."""
    for name in names:
        _LAZY[name] = (module, name)


_register_lazy(
    "random_explanation",
    [
        "RandomExplanation",
        "RandomExplanationConfig",
        "make_random_explanation",
        "build_random_explanation",
    ],
)
_register_lazy(
    "statemask_adapter",
    [
        "StateMaskExplanation",
        "StateMaskExplanationConfig",
        "StateMaskAdapter",
        "StateMaskExplainer",
        "make_statemask_explanation",
        "build_statemask",
        "train_statemask",
    ],
)
_register_lazy(
    "integrated_gradients",
    [
        "IntegratedGradientsExplanation",
        "IntegratedGradientsConfig",
        "IntegratedGradients",
        "make_integrated_gradients",
        "build_integrated_gradients",
    ],
)
_register_lazy(
    "airs_adapter",
    [
        "AIRSExplanation",
        "AIRSConfig",
        "AIRSAdapter",
        "make_airs_explanation",
        "build_airs",
    ],
)


# ---------------------------------------------------------------------------
# Lazy import plumbing (tolerant to several sys.path layouts, see
# ``rice/evaluation/__init__.py`` for the same pattern).
# ---------------------------------------------------------------------------
def _import_relative(module: str) -> Any:
    """Import ``rice.explanation.<module>`` with fallbacks for repo layouts."""
    import importlib

    tail = module.split(".")[-1]
    candidates = (
        f"{__name__}.{module}",
        f"rice.explanation.{module}",
        f"rice.rice.explanation.{module}",
        f"explanation.{module}",
        tail,
    )
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except ImportError as exc:  # pragma: no cover - layout dependent
            last_error = exc
    raise ImportError(
        f"Unable to import explanation submodule {module!r} from any of {candidates}"
    ) from last_error


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = _import_relative(module_name)
        try:
            value = getattr(module, attr)
        except AttributeError:  # tolerate naming drift in the provider modules
            alias = _ATTRIBUTE_ALIASES.get(attr, ())
            for candidate in alias:
                if hasattr(module, candidate):
                    value = getattr(module, candidate)
                    break
            else:
                raise
        globals()[name] = value  # cache, so the hook only runs once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Tolerate small naming differences between the provider modules.
_ATTRIBUTE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "IntegratedGradientsExplanation": ("IntegratedGradients", "IntegratedGradientsExplainer"),
    "IntegratedGradientsConfig": ("IGConfig",),
    "make_integrated_gradients": ("build_integrated_gradients", "make_explanation"),
    "build_integrated_gradients": ("make_integrated_gradients",),
    "AIRSExplanation": ("AIRSAdapter", "AIRExplainer"),
    "AIRSConfig": ("AIRSExplanationConfig",),
    "make_airs_explanation": ("build_airs", "make_explanation"),
    "build_airs": ("make_airs_explanation",),
    "StateMaskExplanation": ("StateMaskAdapter", "StateMaskExplainer"),
    "StateMaskExplanationConfig": ("StateMaskConfig",),
    "RandomExplanationConfig": ("RandomConfig",),
    "RandomExplanation": ("RandomExplainer",),
}


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


# ---------------------------------------------------------------------------
# Registry helpers / dispatch
# ---------------------------------------------------------------------------
def resolve_explanation_name(name: str) -> str:
    """Map a canonical name or friendly alias to a registry key.

    Matching is case/whitespace/underscore insensitive so that CLI arguments
    such as ``--explanation Integrated-Gradients`` work.
    """
    if not isinstance(name, str):
        raise KeyError(f"Explanation name must be a string, got {type(name)!r}")
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in EXPLANATION_REGISTRY:
        return normalized
    if normalized in EXPLANATION_ALIASES:
        return EXPLANATION_ALIASES[normalized]
    raise KeyError(
        f"Unknown explanation {name!r}. Known names: {sorted(EXPLANATION_REGISTRY)}"
    )


def available_explanations() -> List[str]:
    """Return the canonical explanation names understood by :func:`make_explanation`."""
    return list(EXPLANATION_REGISTRY)


def get_explanation_spec(name: str = "ours") -> Dict[str, Any]:
    """Return ``{name, module, factory, description}`` registry metadata."""
    key = resolve_explanation_name(name)
    module, factory, description = EXPLANATION_REGISTRY[key]
    return {
        "name": key,
        "module": module,
        "factory": factory,
        "description": description,
    }


def register_explanation(
    name: str,
    module: str,
    factory: str,
    description: str = "",
    aliases: Optional[List[str]] = None,
) -> None:
    """Register a new explanation provider (used by tests / user extensions)."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    EXPLANATION_REGISTRY[key] = (module, factory, description)
    EXPLANATION_NAMES = tuple(EXPLANATION_REGISTRY)  # noqa: F841 (documented side effect)
    for alias in aliases or []:
        EXPLANATION_ALIASES[alias.strip().lower().replace(" ", "_").replace("-", "_")] = key


def make_explanation(name: str = "ours", **kwargs: Any) -> Any:
    """Build an explanation provider from its registry name.

    Parameters
    ----------
    name:
        One of :func:`available_explanations` (aliases accepted).
    **kwargs:
        Forwarded verbatim to the provider factory.  ``ours``/``statemask``
        accept ``env``, ``policy``, ``mask_network``, ``config``, ``rng``;
        ``random`` only needs ``seed``/``config``; ``integrated_gradients`` and
        ``airs`` need the target policy (black-box: only through its action
        distribution or ``predict`` interface).
    """
    key = resolve_explanation_name(name)

    # "ours" is the re-designed mask network of §3.3.  If the caller already
    # trained a mask network (or a checkpoint) it is forwarded to the adapter,
    # which then only wraps it for importance scoring.
    if key in ("ours", "statemask"):
        adapter = _import_relative("statemask_adapter")
        factory = getattr(adapter, "make_statemask_explanation", None) or getattr(
            adapter, "build_statemask", None
        )
        if factory is None:  # pragma: no cover - provider contract violation
            raise AttributeError("statemask_adapter does not expose a factory")
        return factory(**kwargs)

    module_name, factory_name, _ = EXPLANATION_REGISTRY[key]
    module = _import_relative(module_name)
    factory: Optional[Callable[..., Any]] = getattr(module, factory_name, None)
    if factory is None:
        for candidate in _ATTRIBUTE_ALIASES.get(factory_name, ()):
            factory = getattr(module, candidate, None)
            if factory is not None:
                break
    if factory is None:  # pragma: no cover - provider contract violation
        raise AttributeError(f"{module_name} does not expose {factory_name!r}")
    return factory(**kwargs)


#: Convenience alias mirroring ``build_statemask``/``build_airs`` naming.
build_explanation = make_explanation


def explanation_importance(explanation: Any, states: Any, **kwargs: Any) -> Any:
    """Call ``importance``/``score``/``mask_prob_zero`` on any provider.

    Uniform access point so the mask-network path and the third-party
    explanation adapters (IG / AIRS) behave identically inside the fidelity
    evaluator and the refining loop.
    """
    import numpy as _np

    if explanation is None:
        n = len(states) if hasattr(states, "__len__") else 1
        return _np.zeros(n, dtype=_np.float64)
    if isinstance(explanation, _np.ndarray):
        return _np.asarray(explanation, dtype=_np.float64)
    for attr in ("importance", "score", "mask_prob_zero", "importances"):
        fn = getattr(explanation, attr, None)
        if callable(fn):
            try:
                return _np.asarray(fn(states, **kwargs), dtype=_np.float64).reshape(-1)
            except TypeError:
                return _np.asarray(fn(states), dtype=_np.float64).reshape(-1)
    if callable(explanation):
        return _np.asarray(explanation(states), dtype=_np.float64).reshape(-1)
    return _np.asarray(explanation, dtype=_np.float64).reshape(-1)


def select_critical_state_from_explanation(
    explanation: Any,
    states: Any,
    return_index: bool = False,
    **kwargs: Any,
) -> Any:
    """Return the ``argmax``-importance (critical) state of a trajectory.

    Mirrors :func:`rice.algorithms.critical_state.select_critical_state` but
    works with any explanation provider; ties are broken towards the earliest
    time step (``np.argmax`` semantics), matching the paper's "most critical
    state in tau" wording of Algorithm 2.
    """
    import numpy as _np

    for attr in ("select_index", "best_index", "critical_index"):
        fn = getattr(explanation, attr, None)
        if callable(fn):
            try:
                index = int(fn(states, **kwargs))
                break
            except TypeError:
                index = int(fn(states))
                break
    else:
        scores = _np.asarray(explanation_importance(explanation, states), dtype=_np.float64)
        index = int(_np.argmax(scores)) if scores.size else 0

    try:
        state = states[index]
    except Exception:  # pragma: no cover - exotic containers
        state = states
    if return_index:
        return state, index
    return state


# ---------------------------------------------------------------------------
# Reference trends (Table 6, §C.3).  Per the benchmark addendum we reproduce
# trends, not exact numbers: using the explanation generated by our mask
# network, refining achieves the best outcome across all four applications,
# and Integrated Gradients / AIRS still beat the random baseline.
# ---------------------------------------------------------------------------
TABLE6_ORDERING: Tuple[str, ...] = ("ours", "airs", "integrated_gradients", "random")

TABLE6_REFERENCE: Dict[str, Any] = {
    "ordering": TABLE6_ORDERING,
    "tasks": ("Hopper-v3", "Walker2d-v3", "Reacher-v2", "HalfCheetah-v3"),
    "claim": (
        "RICE(Ours) > AIRS > Integrated Gradients > Random; the framework works "
        "with different explanation choices (Table 6, §C.3)."
    ),
    "exact_values_reported_in_paper": False,
    "note": (
        "Exact Table 6 numbers are not used as success criteria "
        "(addendum: reproduce trends, not exact numbers)."
    ),
}


def train_mask_network(
    env: Any,
    policy: Any,
    config: Any = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Train the RICE mask network (Algorithm 1) and return an explanation.

    Thin facade over :func:`rice.explanation.statemask_adapter.train_statemask`
    so that callers do not need to know whether the adapter module exposes the
    trainer directly.  ``alpha`` (Table 3: 1e-4) and the per-environment mask
    sample budget (Table 4) are honoured by the adapter/config.
    """
    adapter = _import_relative("statemask_adapter")
    factory = getattr(adapter, "train_statemask", None)
    if factory is None:  # pragma: no cover - adapter contract violation
        raise AttributeError("statemask_adapter does not expose train_statemask")
    return factory(env=env, target_policy=policy, config=config, seed=seed, **kwargs)
