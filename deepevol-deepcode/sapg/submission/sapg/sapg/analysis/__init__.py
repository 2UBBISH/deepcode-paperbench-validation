"""Diversity analysis for SAPG (Section 6.4, Figures 7 and 8).

This package implements the two state-space diversity metrics used in the paper
to argue that SAPG explores a larger portion of the state space than vanilla
PPO:

* :mod:`sapg.analysis.diversity_pca` -- PCA reconstruction error of visited
  states as a function of the number of retained principal components ``k``
  (Figure 7).  A *slower* decrease of the reconstruction error with ``k``
  indicates a policy that covers more state-space directions.
* :mod:`sapg.analysis.diversity_mlp` -- training reconstruction error of a small
  two-hidden-layer ReLU MLP of varying width trained to reconstruct visited
  states (Figure 8).  A *higher* reconstruction error for the same width
  indicates more diverse (harder to compress) state distributions.

Both metrics follow the protocol described in Section 6.4 of the paper and its
addendum:

* PCA: reconstruction error of the states using the top-``k`` principal
  components, plotted versus ``k``.
* MLP: two-layer network of equal width (the x-axis of Figure 8), ReLU
  activation, Adam with PyTorch defaults, L2 reconstruction loss, trained on
  400k state-transitions per method.

Importing this package stays cheap: the two modules are resolved lazily via
:pep:`562` module level ``__getattr__`` so that neither ``numpy``, ``torch``,
``scikit-learn`` nor ``matplotlib`` are imported until a symbol is actually
requested.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__all__: List[str] = [
    # ---------------------------------------------------------------- PCA ---
    "PCAConfig",
    "PCAResult",
    "PCADiversity",
    "pca_reconstruction_error",
    "pca_reconstruction_curve",
    "pca_components",
    "explained_variance_ratio",
    "reconstruction_error_per_dimension",
    "compare_pca_diversity",
    "curve_decrease_rate",
    "saturation_point",
    "plot_pca_diversity",
    # ---------------------------------------------------------------- MLP ---
    "MLPConfig",
    "MLPResult",
    "MLPReconstructor",
    "MLPDiversity",
    "train_reconstruction_mlp",
    "mlp_reconstruction_error",
    "mlp_reconstruction_curve",
    "compare_mlp_diversity",
    "diversity_ranking",
    "error_increase_ratio",
    "curve_ordering",
    "plot_mlp_diversity",
    # ------------------------------------------------------------- shared ---
    "states_from_buffer",
    "collect_states",
    "METHOD_SAPG",
    "METHOD_PPO",
    "METHOD_RANDOM",
    "DEFAULT_K_VALUES",
    "DEFAULT_WIDTHS",
    "DEFAULT_MAX_SAMPLES",
    # --------------------------------------------------------------- meta ---
    "available_metrics",
    "make_metric",
]

# ---------------------------------------------------------------------------
# Lazy export registry -- maps a public name to the module that defines it.
# ---------------------------------------------------------------------------
_LAZY_EXPORTS: Dict[str, str] = {
    # PCA (Figure 7)
    "PCAConfig": "diversity_pca",
    "PCAResult": "diversity_pca",
    "PCADiversity": "diversity_pca",
    "pca_reconstruction_error": "diversity_pca",
    "pca_reconstruction_curve": "diversity_pca",
    "pca_components": "diversity_pca",
    "explained_variance_ratio": "diversity_pca",
    "reconstruction_error_per_dimension": "diversity_pca",
    "compare_pca_diversity": "diversity_pca",
    "curve_decrease_rate": "diversity_pca",
    "saturation_point": "diversity_pca",
    "plot_pca_diversity": "diversity_pca",
    "DEFAULT_K_VALUES": "diversity_pca",
    # MLP (Figure 8)
    "MLPConfig": "diversity_mlp",
    "MLPResult": "diversity_mlp",
    "MLPReconstructor": "diversity_mlp",
    "MLPDiversity": "diversity_mlp",
    "train_reconstruction_mlp": "diversity_mlp",
    "mlp_reconstruction_error": "diversity_mlp",
    "mlp_reconstruction_curve": "diversity_mlp",
    "compare_mlp_diversity": "diversity_mlp",
    "diversity_ranking": "diversity_mlp",
    "error_increase_ratio": "diversity_mlp",
    "curve_ordering": "diversity_mlp",
    "plot_mlp_diversity": "diversity_mlp",
    "DEFAULT_WIDTHS": "diversity_mlp",
    "DEFAULT_MAX_SAMPLES": "diversity_mlp",
    # Shared state-extraction helpers live in the PCA module and are reused
    # (semantically) by the MLP module.  Resolve to the PCA implementation.
    "states_from_buffer": "diversity_pca",
    "collect_states": "diversity_pca",
    # Method tags mirror the three curves of Figures 7 and 8.
    "METHOD_SAPG": "diversity_pca",
    "METHOD_PPO": "diversity_pca",
    "METHOD_RANDOM": "diversity_pca",
}


def _resolve(name: str) -> Any:
    """Import and return the symbol ``name`` from its owning submodule."""
    module_name = _LAZY_EXPORTS[name]
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except Exception as exc:  # pragma: no cover - defensive, optional deps
        raise AttributeError(
            f"Could not import 'sapg.analysis.{module_name}' while resolving "
            f"'{name}': {exc}"
        ) from exc

    try:
        value = getattr(module, name)
    except AttributeError:
        # Fall back to the sibling module (both expose near-identical helpers
        # such as ``states_from_buffer`` / ``collect_states``).
        other = "diversity_mlp" if module_name == "diversity_pca" else "diversity_pca"
        try:
            alt = importlib.import_module(f".{other}", __name__)
            value = getattr(alt, name)
        except Exception as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"module 'sapg.analysis' has no attribute {name!r}"
            ) from exc

    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatch
    """PEP 562 lazy attribute resolution (keeps ``import sapg.analysis`` cheap)."""
    if name in _LAZY_EXPORTS:
        return _resolve(name)
    raise AttributeError(f"module 'sapg.analysis' has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - introspection helper
    return sorted(set(__all__))


# ---------------------------------------------------------------------------
# Small convenience helpers
# ---------------------------------------------------------------------------
def available_metrics() -> List[str]:
    """Return the names of the implemented diversity metrics."""
    return ["pca", "mlp"]


def make_metric(name: str, states: Any = None, config: Any = None, **kwargs: Any) -> Any:
    """Factory building a diversity metric object by name.

    Parameters
    ----------
    name:
        ``"pca"`` (Figure 7) or ``"mlp"`` (Figure 8); case-insensitive and
        tolerant of dashes/spaces.
    states:
        Optional batch of visited states used to initialise the metric.
    config:
        Optional metric-specific configuration object or ``dict``.
    **kwargs:
        Forwarded to the metric constructor.

    Returns
    -------
    ``PCADiversity`` or ``MLPDiversity`` instance, or ``None`` when ``states``
    is not supplied (so callers may build the metric first and feed it states
    later, e.g. via ``collect_states``).
    """
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in ("pca", "pca_diversity", "principal_component_analysis"):
        cls = _resolve("PCADiversity")
    elif key in ("mlp", "mlp_diversity", "reconstruction_mlp"):
        cls = _resolve("MLPDiversity")
    else:
        raise ValueError(
            f"Unknown diversity metric {name!r}; expected one of "
            f"{available_metrics()}"
        )
    if states is None:
        return cls
    return cls(states, config=config, **kwargs)


# ---------------------------------------------------------------------------
# Convenience: run both Figure-7 and Figure-8 analyses on per-method batches.
# ---------------------------------------------------------------------------
def compare_diversity(
    state_batches: Dict[str, Any],
    k_values: Optional[Tuple[int, ...]] = None,
    widths: Optional[Tuple[int, ...]] = None,
    pca_config: Any = None,
    mlp_config: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute both diversity metrics for a mapping ``{method: states}``.

    Returns a dict ``{"pca": {...}, "mlp": {...}}`` suitable for the analysis
    script and for reproducing Figures 7 and 8 side by side.
    """
    pca_results = _resolve("compare_pca_diversity")(state_batches, k_values=k_values, config=pca_config)
    mlp_results = _resolve("compare_mlp_diversity")(state_batches, widths=widths, config=mlp_config)
    return {"pca": pca_results, "mlp": mlp_results}
