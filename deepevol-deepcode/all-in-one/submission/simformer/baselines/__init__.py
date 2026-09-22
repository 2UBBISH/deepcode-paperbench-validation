"""Baseline methods for the Simformer reproduction.

This package collects the simulation-based-inference baselines that the paper
compares against (Sec. 4.1, Appendix A2.1/A3.1, Addendum "Training"):

* :mod:`simformer.baselines.npe_nle_nre`
    NPE (amortized posterior, ``sbi`` neural spline flow or a self-contained
    conditional MAF fallback), NLE (neural likelihood + MCMC) and NRE
    (neural ratio classifier + MCMC).
* :mod:`simformer.baselines.npse`
    NPSE (neural posterior score estimation with a conditional MLP score
    network and reverse-SDE sampling) plus the *posterior-only Simformer*
    variant that isolates the benefit of the full condition-mask mixture.

Design notes
------------
Importing this package never imports ``torch``/``sbi``/``sklearn`` eagerly:
every symbol is resolved lazily through a PEP 562 module ``__getattr__`` (the
same pattern used by :mod:`simformer` and :mod:`simformer.tasks`).  Symbols
that exist under the same name in both submodules (``build_baseline``,
``available_methods``, ``BASELINE_BUILDERS``, ...) are *not* re-exported
directly; instead this module provides unified dispatchers that route to the
correct implementation based on the (normalized) method name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    # unified dispatch API
    "BASELINE_METHODS",
    "METHOD_ALIASES",
    "canonical_method",
    "available_methods",
    "available_baselines",
    "build_baseline",
    "train_baseline",
    "evaluate_baseline",
    "evaluate_baseline_c2st",
    "build_all_baselines",
    "sbi_available",
    "summarize_baseline_results",
    # npe / nle / nre
    "BaselineConfig",
    "BaselinePosterior",
    "NPEBaseline",
    "NLEBaseline",
    "NREBaseline",
    "ConditionalMAF",
    "Standardizer",
    "train_npe",
    "train_nle",
    "train_nre",
    "build_npe",
    "build_nle",
    "build_nre",
    # npse (+ posterior-only simformer)
    "NPSEConfig",
    "ConditionalScoreMLP",
    "NPSESampler",
    "TimeFourierEmbedding",
    "NPSEBaseline",
    "SimformerPosteriorOnlyBaseline",
    "train_npse",
    "train_npse_simformer",
    "build_npse",
    "build_npse_simformer",
    "evaluate_npse_c2st",
]

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

#: Canonical method names implemented across the baseline submodules.
BASELINE_METHODS: Tuple[str, ...] = ("npe", "nle", "nre", "npse", "npse_simformer")

#: Accepted alternative spellings for every canonical method.
METHOD_ALIASES: Dict[str, str] = {
    # posterior estimation
    "npe": "npe",
    "snpe": "npe",
    "npe_a": "npe",
    "npe-a": "npe",
    "neural_posterior_estimation": "npe",
    "neural-posterior-estimation": "npe",
    "amortized": "npe",
    # likelihood estimation
    "nle": "nle",
    "snle": "nle",
    "neural_likelihood_estimation": "nle",
    "neural-likelihood-estimation": "nle",
    # ratio estimation
    "nre": "nre",
    "snre": "nre",
    "neural_ratio_estimation": "nre",
    "neural-ratio-estimation": "nre",
    # posterior score estimation
    "npse": "npse",
    "nps": "npse",
    "neural_posterior_score_estimation": "npse",
    "neural-posterior-score-estimation": "npse",
    # posterior-only Simformer
    "npse_simformer": "npse_simformer",
    "npse-simformer": "npse_simformer",
    "simformer_posterior": "npse_simformer",
    "simformer-posterior": "npse_simformer",
    "simformer_posterior_only": "npse_simformer",
    "posterior_only": "npse_simformer",
    "posterior-only": "npse_simformer",
}

#: Public symbol -> owning submodule (relative path).
_SUBMODULES: Dict[str, str] = {
    # ------------------------------------------------------------------ npe/nle/nre
    "BaselineConfig": "npe_nle_nre",
    "BaselinePosterior": "npe_nle_nre",
    "NPEBaseline": "npe_nle_nre",
    "NLEBaseline": "npe_nle_nre",
    "NREBaseline": "npe_nle_nre",
    "ConditionalMAF": "npe_nle_nre",
    "Standardizer": "npe_nle_nre",
    "train_npe": "npe_nle_nre",
    "train_nle": "npe_nle_nre",
    "train_nre": "npe_nle_nre",
    "build_npe": "npe_nle_nre",
    "build_nle": "npe_nle_nre",
    "build_nre": "npe_nle_nre",
    # ------------------------------------------------------------------ npse
    "NPSEConfig": "npse",
    "ConditionalScoreMLP": "npse",
    "NPSESampler": "npse",
    "TimeFourierEmbedding": "npse",
    "NPSEBaseline": "npse",
    "SimformerPosteriorOnlyBaseline": "npse",
    "train_npse": "npse",
    "train_npse_simformer": "npse",
    "build_npse": "npse",
    "build_npse_simformer": "npse",
}

_MODULE_CACHE: Dict[str, Any] = {}


def _load_module(name: str) -> Any:
    """Import (and cache) a sibling baseline submodule."""
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    try:
        from importlib import import_module

        module = import_module(f"{__name__}.{name}")
    except Exception:  # pragma: no cover - flat layout fallback
        from importlib import import_module

        module = import_module(name)
    _MODULE_CACHE[name] = module
    return module


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    owner = _SUBMODULES.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_load_module(owner), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(list(globals()) + __all__))


# ---------------------------------------------------------------------------
# Unified helpers
# ---------------------------------------------------------------------------


def canonical_method(method: Any) -> str:
    """Normalize a method name/alias to a canonical baseline name."""
    if method is None:
        raise ValueError("baseline method name must not be None")
    if not isinstance(method, str):
        # Allow passing a class/instance: derive from its name.
        method = getattr(method, "method", None) or type(method).__name__
    key = str(method).strip().lower().replace(" ", "_")
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    key_alt = key.replace("-", "_")
    if key_alt in METHOD_ALIASES:
        return METHOD_ALIASES[key_alt]
    key_alt2 = key.replace("_", "-")
    if key_alt2 in METHOD_ALIASES:
        return METHOD_ALIASES[key_alt2]
    raise ValueError(
        f"unknown baseline method {method!r}; available: {sorted(BASELINE_METHODS)}"
    )


def _module_methods(module_name: str) -> Tuple[str, ...]:
    module = _load_module(module_name)
    methods = getattr(module, "IMPLEMENTED_METHODS", None)
    if methods:
        return tuple(str(m) for m in methods)
    registry = getattr(module, "BASELINE_BUILDERS", None)
    if isinstance(registry, dict):
        return tuple(sorted(str(k) for k in registry))
    return ()


def available_methods() -> Tuple[str, ...]:
    """Return the canonical method names available in the installed modules."""
    found: List[str] = []
    for module_name in ("npe_nle_nre", "npse"):
        try:
            found.extend(_module_methods(module_name))
        except Exception:
            continue
    ordered = [m for m in BASELINE_METHODS if m in set(found)]
    # keep any extra methods the submodules expose, in a stable order
    ordered.extend(sorted(m for m in found if m not in set(ordered)))
    return tuple(ordered) if ordered else BASELINE_METHODS


#: Alias kept for symmetry with the submodules.
available_baselines = available_methods


def sbi_available() -> bool:  # noqa: D401 - short re-export
    """Whether the external ``sbi`` package could be imported."""
    try:
        return bool(_load_module("npe_nle_nre").sbi_available())
    except Exception:
        return False


def _resolve_config(method: str, config: Any) -> Any:
    if config is not None:
        return config
    if method in ("npse", "npse_simformer"):
        return _load_module("npse").NPSEConfig()
    return _load_module("npe_nle_nre").BaselineConfig(method=method)


def build_baseline(method: str, task: Any = None, config: Any = None, **kwargs: Any) -> Any:
    """Build an *untrained* baseline for ``method`` (canonicalized).

    Routes to the per-module builder; when the module does not expose a
    dedicated builder for the canonical method, the corresponding class is
    constructed directly.
    """
    canonical = canonical_method(method)
    if canonical in ("npse", "npse_simformer"):
        module = _load_module("npse")
        builder = getattr(module, "build_baseline", None)
        if callable(builder):
            try:
                return builder(canonical, task, config=config, **kwargs)
            except TypeError:
                pass
        cls = (
            module.NPSEBaseline
            if canonical == "npse"
            else module.SimformerPosteriorOnlyBaseline
        )
        cfg = _resolve_config(canonical, config)
        return cls(task, cfg, **kwargs) if cfg is not None else cls(task, **kwargs)

    module = _load_module("npe_nle_nre")
    builder = getattr(module, "build_baseline", None)
    if callable(builder):
        cfg = _resolve_config(canonical, config)
        for attempt in (
            lambda: builder(canonical, task, config=cfg, **kwargs),
            lambda: builder(canonical, task, **kwargs),
        ):
            try:
                return attempt()
            except TypeError:
                continue
    name = {"npe": "build_npe", "nle": "build_nle", "nre": "build_nre"}[canonical]
    fn = getattr(module, name)
    return fn(task, config=_resolve_config(canonical, config), **kwargs)


def train_baseline(
    method: str,
    task: Any,
    n_simulations: int = 10000,
    config: Any = None,
    seed: int = 0,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Train one baseline and return a fitted posterior object.

    The returned object follows the shared interface of
    :class:`~simformer.baselines.npe_nle_nre.BaselinePosterior`:
    ``.sample(n_samples, x_obs)`` / ``.posterior_samples(x_obs, n_samples)``.
    """
    canonical = canonical_method(method)
    if canonical in ("npse", "npse_simformer"):
        module = _load_module("npse")
        fn = getattr(
            module,
            "train_npse" if canonical == "npse" else "train_npse_simformer",
        )
        cfg = _resolve_config(canonical, config)
        for attempt in (
            lambda: fn(
                task,
                n_simulations=n_simulations,
                config=cfg,
                seed=seed,
                verbose=verbose,
                **kwargs,
            ),
            lambda: fn(task, n_simulations, cfg, seed=seed, verbose=verbose, **kwargs),
            lambda: fn(task, n_simulations=n_simulations, seed=seed, **kwargs),
        ):
            try:
                return attempt()
            except TypeError:
                continue
        raise RuntimeError(f"could not call npse trainer for method {canonical!r}")

    module = _load_module("npe_nle_nre")
    fn = getattr(module, "train_baseline", None)
    cfg = _resolve_config(canonical, config)
    if callable(fn):
        for attempt in (
            lambda: fn(
                canonical,
                task,
                n_simulations=n_simulations,
                config=cfg,
                seed=seed,
                verbose=verbose,
                **kwargs,
            ),
            lambda: fn(canonical, task, n_simulations, cfg, seed=seed, **kwargs),
        ):
            try:
                return attempt()
            except TypeError:
                continue
    name = {"npe": "train_npe", "nle": "train_nle", "nre": "train_nre"}[canonical]
    fn = getattr(module, name)
    for attempt in (
        lambda: fn(
            task,
            n_simulations=n_simulations,
            config=cfg,
            seed=seed,
            verbose=verbose,
            **kwargs,
        ),
        lambda: fn(task, n_simulations, cfg, seed=seed, verbose=verbose, **kwargs),
    ):
        try:
            return attempt()
        except TypeError:
            continue
    raise RuntimeError(f"could not call baseline trainer for method {canonical!r}")


def evaluate_baseline_c2st(
    baseline: Any,
    task: Any = None,
    *,
    n_targets: int = 10,
    n_samples: int = 1000,
    n_reference: int = 1000,
    n_steps: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
    return_result: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """C2ST evaluation of a baseline's posterior against reference samples.

    Dispatches to the evaluator of the module that produced ``baseline``
    (detected via ``isinstance``), tolerating signature differences between the
    two implementations.
    """
    try:
        npse_module = _load_module("npse")
        npse_classes = (
            npse_module.NPSEBaseline,
            npse_module.SimformerPosteriorOnlyBaseline,
        )
        if isinstance(baseline, npse_classes):
            fn = getattr(npse_module, "evaluate_npse_c2st")
            evaluator = lambda: fn(  # noqa: E731
                baseline,
                task,
                n_targets=n_targets,
                n_samples=n_samples,
                n_reference=n_reference,
                n_steps=n_steps,
                seed=seed,
                verbose=verbose,
                **kwargs,
            )
        else:
            raise LookupError
    except (LookupError, AttributeError, ImportError):
        module = _load_module("npe_nle_nre")
        fn = getattr(module, "evaluate_baseline_c2st")
        evaluator = lambda: fn(  # noqa: E731
            baseline,
            task,
            n_targets=n_targets,
            n_samples=n_samples,
            n_reference=n_reference,
            seed=seed,
            verbose=verbose,
            return_result=return_result,
            **kwargs,
        )

    try:
        return evaluator()
    except TypeError:
        # Minimal-signature fallback.
        if baseline is not None and task is not None:
            for module_name in ("npse", "npe_nle_nre"):
                module = _load_module(module_name)
                fn = getattr(module, "evaluate_baseline_c2st", None) or getattr(
                    module, "evaluate_npse_c2st", None
                )
                if callable(fn):
                    try:
                        return fn(baseline, task)
                    except TypeError:
                        continue
        raise


#: Alias matching the naming used in the submodules.
evaluate_baseline = evaluate_baseline_c2st


def build_all_baselines(
    task: Any,
    methods: Optional[Tuple[str, ...]] = None,
    *,
    n_simulations: int = 10000,
    seed: int = 0,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train every requested baseline and return ``{method: fitted_baseline}``.

    Failures are captured per method (as ``{"error": ...}``) so a single broken
    optional dependency does not abort the whole benchmark sweep.
    """
    requested = tuple(methods) if methods else available_methods()
    results: Dict[str, Any] = {}
    for name in requested:
        try:
            canonical = canonical_method(name)
        except ValueError:
            results[str(name)] = {"error": f"unknown method {name!r}"}
            continue
        try:
            results[canonical] = train_baseline(
                canonical,
                task,
                n_simulations=n_simulations,
                seed=seed,
                verbose=verbose,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - defensive
            results[canonical] = {"error": repr(exc)}
    return results


def summarize_baseline_results(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Flatten ``{method: c2st_summary}`` into a printable benchmark table."""
    table: Dict[str, Dict[str, float]] = {}
    for name, value in (results or {}).items():
        if isinstance(value, dict):
            flat: Dict[str, float] = {}
            for key, item in value.items():
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    flat[str(key)] = float(item)
            table[str(name)] = flat
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            table[str(name)] = {"c2st": float(value)}
    return table
