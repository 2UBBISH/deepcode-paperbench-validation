"""Analysis utilities for the forgetting-mitigation reproduction.

This sub-package gathers every *post-hoc* measurement used to support the
paper's claims (Wołczyk et al., 2024, "Fine-tuning Reinforcement Learning
Models is Secretly a Forgetting Mitigation Problem"):

* :mod:`src.analysis.cka` -- Centered Kernel Alignment between pre-trained
  (``pi_*``) and fine-tuned activations per layer (Appendix B.3, Figures 26/27).
* :mod:`src.analysis.forward_transfer` -- the forward-transfer metric
  ``FT = (AUC - AUC_b) / (1 - AUC_b)`` over prefix-task lengths
  (Appendix F, Table 6).
* :mod:`src.analysis.loglikelihood` -- expert-action log-likelihood
  ``E[log pi_theta(a*|s)]`` of the fine-tuned policy on ``push-wall``
  every 50k steps (Section 5, Figure 8).
* :mod:`src.analysis.pca_viz` -- frozen-basis PCA of the state space coloured
  by log-likelihood (Figure 8).
* :mod:`src.analysis.density_plots` -- NetHack level-visitation density plots.
* :mod:`src.analysis.return_distribution` -- per-method return distributions.
* :mod:`src.analysis.plotting` -- shared matplotlib helpers/aggregation.

All heavy third-party dependencies (NumPy, PyTorch, matplotlib) are imported
*lazily* by the individual modules, and each module is optional here, so that
importing :mod:`src.analysis` never fails on a machine that lacks one of them.
Use :func:`available_modules` / :func:`require` to discover what is importable.
"""

from __future__ import annotations

import importlib
from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ckanalysis_available",
    "available_modules",
    "missing_modules",
    "require",
    "load",
    "cka",
    "forward_transfer",
    "loglikelihood",
    "pca_viz",
    "density_plots",
    "return_distribution",
    "plotting",
]

#: Name -> importable dotted path of every analysis module shipped with this
#: reproduction.  Order follows the dependency chain (shared helpers first).
_MODULE_PATHS: Dict[str, str] = {
    "logging_utils": "src.common.logging_utils",
    "forward_transfer": "src.analysis.forward_transfer",
    "cka": "src.analysis.cka",
    "loglikelihood": "src.analysis.loglikelihood",
    "pca_viz": "src.analysis.pca_viz",
    "density_plots": "src.analysis.density_plots",
    "return_distribution": "src.analysis.return_distribution",
    "plotting": "src.analysis.plotting",
}

#: Names of the analysis modules that make up the paper's Figure 3/5/7/8,
#: Table 4/5/6 deliverables (used by tests and the ``analyze`` CLI).
PAPER_FIGURES: Dict[str, Tuple[str, ...]] = {
    "figure_3": ("return_distribution", "forward_transfer"),
    "figure_5": ("density_plots",),
    "figure_7": ("plotting", "forward_transfer"),
    "figure_8": ("loglikelihood", "pca_viz"),
    "figure_26": ("cka",),
    "figure_27": ("cka",),
    "table_6": ("forward_transfer",),
}

# Cache of successfully imported modules so `load` is cheap on repeat calls.
_CACHE: Dict[str, ModuleType] = {}


def _resolve(name: str) -> str:
    """Map a short module name (or a dotted path) to its importable path."""
    if name in _MODULE_PATHS:
        return _MODULE_PATHS[name]
    if "." in name:
        return name
    return "src.analysis." + name


def load(name: str, required: bool = False) -> Optional[ModuleType]:
    """Import and return one analysis module.

    Parameters
    ----------
    name:
        Short name (``"cka"``) or a fully qualified dotted path
        (``"src.analysis.cka"``).
    required:
        If ``True`` a failed import raises; otherwise ``None`` is returned so
        that optional setups keep working.

    Returns
    -------
    The imported module, or ``None`` when the import failed and
    ``required=False``.
    """
    path = _resolve(name)
    if path in _CACHE:
        return _CACHE[path]
    try:
        module = import_module(path)
    except Exception:  # pragma: no cover - depends on optional deps
        if required:
            raise
        return None
    _CACHE[path] = module
    return module


def require(name: str) -> ModuleType:
    """Like :func:`load` but always raises on a failed import."""
    return load(name, required=True)  # type: ignore[return-value]


def available_modules() -> List[str]:
    """Names of every analysis module that can be imported right now."""
    return sorted(name for name in _MODULE_PATHS if load(name) is not None)


def missing_modules() -> List[str]:
    """Names of the analysis modules whose import failed (missing deps)."""
    return sorted(name for name in _MODULE_PATHS if load(name) is None)


def ckanalysis_available() -> bool:
    """``True`` when the NumPy-backed CKA machinery is importable."""
    return load("cka") is not None


# ---------------------------------------------------------------------------
# Convenience re-exports of the most frequently used entry points.
#
# These are intentionally *lazy*: the attributes are resolved on first access
# via module ``__getattr__`` (PEP 562) so that installing only, say, torch and
# not matplotlib still lets users call the forward-transfer metric.
# ---------------------------------------------------------------------------
_REEXPORTS: Dict[str, Tuple[str, str]] = {
    # forward transfer (Table 6)
    "auc": ("forward_transfer", "auc"),
    "forward_transfer": ("forward_transfer", "forward_transfer"),
    "safe_forward_transfer": ("forward_transfer", "safe_forward_transfer"),
    "forward_transfer_curve": ("forward_transfer", "forward_transfer_curve"),
    "ForwardTransferTracker": ("forward_transfer", "ForwardTransferTracker"),
    "compute_table": ("forward_transfer", "compute_table"),
    "prefix_tasks": ("forward_transfer", "prefix_tasks"),
    "PRETRAINED_TASKS": ("forward_transfer", "PRETRAINED_TASKS"),
    "PREFIX_TASK_POOL": ("forward_transfer", "PREFIX_TASK_POOL"),
    # CKA (Figures 26/27)
    "cka": ("cka", "cka"),
    "cka_from_kernels": ("cka", "cka_from_kernels"),
    "batch_cka": ("cka", "batch_cka"),
    "hsic": ("cka", "hsic"),
    "linear_kernel": ("cka", "linear_kernel"),
    "rbf_kernel": ("cka", "rbf_kernel"),
    "layer_drift": ("cka", "layer_drift"),
    "recovery_score": ("cka", "recovery_score"),
    "CKATracker": ("cka", "CKATracker"),
    "activation_snapshots": ("cka", "activation_snapshots"),
    "plot_cka_heatmap": ("cka", "plot_cka_heatmap"),
    # expert log-likelihood (Figure 8)
    "expert_log_prob": ("loglikelihood", "expert_log_prob"),
    "expert_log_likelihood": ("loglikelihood", "expert_log_likelihood"),
    "loglikelihood_trace": ("loglikelihood", "loglikelihood_trace"),
    "collect_expert_transitions": ("loglikelihood", "collect_expert_transitions"),
    "ExpertDataset": ("loglikelihood", "ExpertDataset"),
    "LogLikelihoodTracker": ("loglikelihood", "LogLikelihoodTracker"),
    "plot_loglikelihood": ("loglikelihood", "plot_loglikelihood"),
    # PCA visualisation (Figure 8)
    "PCAVisualizer": ("pca_viz", "PCAVisualizer"),
    "PCAResult": ("pca_viz", "PCAResult"),
    "ProjectionTracker": ("pca_viz", "ProjectionTracker"),
    "pca_fit": ("pca_viz", "pca_fit"),
    "pca_transform": ("pca_viz", "pca_transform"),
    "project_states": ("pca_viz", "project_states"),
    "projection_grid": ("pca_viz", "projection_grid"),
    "plot_projection": ("pca_viz", "plot_projection"),
    "plot_projection_grid": ("pca_viz", "plot_projection_grid"),
    # density plots (Figure 5, NetHack level visitation)
    "level_density": ("density_plots", "level_density"),
    "visit_density": ("density_plots", "visit_density"),
    "plot_level_density": ("density_plots", "plot_level_density"),
    # return distributions (Figure 3a / 3b / 3c, Figure 6)
    "return_distribution": ("return_distribution", "return_distribution"),
    "bootstrap_ci": ("return_distribution", "bootstrap_ci"),
    "summarize_returns": ("return_distribution", "summarize_returns"),
    "plot_return_distribution": ("return_distribution", "plot_return_distribution"),
    # shared plotting
    "mean_ci": ("plotting", "mean_ci"),
    "plot_curve": ("plotting", "plot_curve"),
    "plot_multiple_curves": ("plotting", "plot_multiple_curves"),
    "save_figure": ("plotting", "save_figure"),
    "z_for": ("plotting", "z_for"),
}

__all__ += sorted(_REEXPORTS)


def __getattr__(name: str) -> Any:  # PEP 562 module-level attribute hook
    """Lazily resolve the curated re-export list and analysis modules."""
    if name in _REEXPORTS:
        module_name, attr = _REEXPORTS[name]
        module = load(module_name, required=True)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent lookups
        return value
    if name in _MODULE_PATHS:
        module = load(name, required=True)
        globals()[name] = module
        return module
    raise AttributeError(
        "module {!r} has no attribute {!r}".format(__name__, name)
    )


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + __all__))


def describe() -> Dict[str, Any]:
    """Human-readable summary of the analysis sub-package state."""
    available = available_modules()
    return {
        "package": __name__,
        "modules": list(_MODULE_PATHS),
        "available": available,
        "missing": missing_modules(),
        "figures": {key: list(value) for key, value in PAPER_FIGURES.items()},
        "share": len(available) / float(len(_MODULE_PATHS) or 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Tiny CLI: report which analysis modules are importable."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Report the status of the analysis sub-package."
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help="Module names that must be importable (raises otherwise).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    args = parser.parse_args(argv)

    if args.require:
        for name in args.require:
            require(name)

    info = describe()
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print("analysis package: {}".format(info["package"]))
        print("available : {}".format(", ".join(info["available"]) or "(none)"))
        print("missing   : {}".format(", ".join(info["missing"]) or "(none)"))
        for figure, modules in sorted(info["figures"].items()):
            print("  {} -> {}".format(figure, ", ".join(modules)))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
