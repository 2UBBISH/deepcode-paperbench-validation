"""Analysis package for the SMM visual-reprogramming reproduction (ICML 2024).

This package hosts the figure/report-level analysis utilities of the project:

* :mod:`smm_vr.analysis.plot_patch_size`
  Consumes the ``patch_size_curves.json`` artifact produced by
  ``smm_vr.experiments.run_ablations`` and renders the Figure-4 style accuracy as a
  function of patch size (Sec. 5 "Impact of Patch Size"), plus a text trend report.

* :mod:`smm_vr.analysis.tsne_features`
  Feature-space visualisation analysis (Sec. 5 "Feature Space Visualization
  Results"): t-SNE embeddings of the output-layer features (before label mapping)
  using 5000 randomly selected training samples per dataset, comparing SMM against
  the shared-mask baselines.

Reference scope notes (paper + addendum):

* Figures 1, 2 and 6 and the mask/shared-pattern visualisation subsection are
  explicitly **out of scope** for this reproduction, so no plotting code is
  provided for them.
* The t-SNE analysis is **qualitative**; it uses 5000 randomly selected training
  samples per dataset and must not be treated as a quantitative accuracy metric.

Guarded-import façade
---------------------
Every submodule is imported inside a ``try/except ImportError`` block so that
``import smm_vr.analysis`` never fails while the reproduction is being built up
incrementally (this mirrors the convention used by ``smm_vr.data``,
``smm_vr.models``, ``smm_vr.modules``, ``smm_vr.engine``, ``smm_vr.methods`` and
``smm_vr.experiments``).  Availability of each submodule is surfaced through the
``_PLOT_PATCH_SIZE_AVAILABLE`` / ``_TSNE_FEATURES_AVAILABLE`` flags.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

__all__: List[str] = []


def _extend(names: List[str]) -> None:
    """Append ``names`` to ``__all__`` without introducing duplicates."""
    for name in names:
        if name not in __all__:
            __all__.append(name)


# --------------------------------------------------------------------------------------
# plot_patch_size (Figure 4 / Table of the patch-size sweep)
# --------------------------------------------------------------------------------------
_PLOT_PATCH_SIZE_AVAILABLE = False

try:  # pragma: no cover - import guard, exercised by the package import itself
    from .plot_patch_size import (  # noqa: F401
        BACKBONE as PATCH_STUDY_BACKBONE,
        DEFAULT_CURVES_FILE,
        DEFAULT_FIGURE_NAME,
        DEFAULT_L,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PATCH_SIZE,
        DISPLAY_NAMES as PATCH_SIZE_DISPLAY_NAMES,
        MAX_POOLING_LAYERS_5,
        PATCH_SIZES,
        PATCH_STUDY_DATASETS,
        PATCH_STUDY_L,
        average_curve,
        best_l,
        build_arg_parser as build_plot_patch_size_arg_parser,
        check_patch_size_trend,
        curves_to_table,
        format_patch_size_table,
        format_trend_report,
        load_curves,
        main as plot_patch_size_main,
        normalise_curves,
        per_dataset_trend,
        plot_figure4,
        plot_patch_size_curves,
        plot_patch_size_from_file,
        resolve_curves_path,
    )

    _PLOT_PATCH_SIZE_AVAILABLE = True
    _extend(
        [
            "PATCH_STUDY_BACKBONE",
            "DEFAULT_CURVES_FILE",
            "DEFAULT_FIGURE_NAME",
            "DEFAULT_OUTPUT_DIR",
            "DEFAULT_L",
            "DEFAULT_PATCH_SIZE",
            "PATCH_SIZE_DISPLAY_NAMES",
            "MAX_POOLING_LAYERS_5",
            "PATCH_SIZES",
            "PATCH_STUDY_DATASETS",
            "PATCH_STUDY_L",
            "average_curve",
            "best_l",
            "build_plot_patch_size_arg_parser",
            "check_patch_size_trend",
            "curves_to_table",
            "format_patch_size_table",
            "format_trend_report",
            "load_curves",
            "plot_patch_size_main",
            "normalise_curves",
            "per_dataset_trend",
            "plot_figure4",
            "plot_patch_size_curves",
            "plot_patch_size_from_file",
            "resolve_curves_path",
        ]
    )
except ImportError:  # pragma: no cover - plotting/matplotlib may be unavailable
    pass


# --------------------------------------------------------------------------------------
# tsne_features (feature-space visualisation, 5000 samples per dataset)
# --------------------------------------------------------------------------------------
_TSNE_FEATURES_AVAILABLE = False

try:  # pragma: no cover - import guard
    from .tsne_features import (  # noqa: F401
        DEFAULT_PERPLEXITY,
        DEFAULT_SEED,
        SMM_METHOD,
        TSNE_SAMPLES,
        baseline_methods,
        build_arg_parser as build_tsne_arg_parser,
        compute_tsne,
        extract_feature_bank,
        main as tsne_main,
        plot_tsne_grid,
        run_tsne_experiment,
        tsne_embedding,
    )

    _TSNE_FEATURES_AVAILABLE = True
    _extend(
        [
            "DEFAULT_PERPLEXITY",
            "DEFAULT_SEED",
            "SMM_METHOD",
            "TSNE_SAMPLES",
            "baseline_methods",
            "build_tsne_arg_parser",
            "compute_tsne",
            "extract_feature_bank",
            "plot_tsne_grid",
            "run_tsne_experiment",
            "tsne_embedding",
            "tsne_main",
        ]
    )
except ImportError:  # pragma: no cover - scikit-learn may be unavailable
    pass


# --------------------------------------------------------------------------------------
# Fallback defaults (kept usable even when the submodules cannot be imported)
# --------------------------------------------------------------------------------------
if "PATCH_STUDY_L" not in __all__:  # pragma: no cover - only on partial builds
    PATCH_STUDY_L = (0, 1, 2, 3, 4)
    PATCH_SIZES = (1, 2, 4, 8, 16)
    DEFAULT_PATCH_SIZE = 8
    DEFAULT_L = 3
    PATCH_STUDY_DATASETS = ("cifar10", "svhn", "flowers102", "eurosat")
    _extend(
        [
            "PATCH_STUDY_L",
            "PATCH_SIZES",
            "DEFAULT_PATCH_SIZE",
            "DEFAULT_L",
            "PATCH_STUDY_DATASETS",
        ]
    )

try:  # pragma: no cover - fallback for the t-SNE sample count
    TSNE_SAMPLES  # type: ignore[used-before-def]
except NameError:  # pragma: no cover - only on partial builds
    TSNE_SAMPLES = 5000
    DEFAULT_PERPLEXITY = 30.0
    DEFAULT_SEED = 0
    SMM_METHOD = "ours"
    _extend(["TSNE_SAMPLES", "DEFAULT_PERPLEXITY", "DEFAULT_SEED", "SMM_METHOD"])


def available_analyses() -> Dict[str, bool]:
    """Return which analysis modules were imported successfully.

    Returns
    -------
    dict
        ``{"plot_patch_size": bool, "tsne_features": bool}``.
    """
    return {
        "plot_patch_size": _PLOT_PATCH_SIZE_AVAILABLE,
        "tsne_features": _TSNE_FEATURES_AVAILABLE,
    }


def list_analyses() -> List[str]:
    """Return the canonical names of the analysis modules."""
    return ["plot_patch_size", "tsne_features"]


def default_output_dir() -> str:
    """Return the default analysis output directory (``outputs/figures``)."""
    return os.environ.get("SMM_FIGURES_DIR", globals().get("DEFAULT_OUTPUT_DIR", "outputs/figures"))


def describe_analyses() -> List[Dict[str, str]]:
    """Return one metadata row per analysis module (for ``--list`` style output)."""
    return [
        {
            "name": "plot_patch_size",
            "module": "smm_vr.analysis.plot_patch_size",
            "entry": "plot_patch_size_from_file",
            "description": "Figure 4: accuracy vs patch size (l in {0,1,2,3,4}) "
            "from patch_size_curves.json, plus trend verification.",
        },
        {
            "name": "tsne_features",
            "module": "smm_vr.analysis.tsne_features",
            "entry": "run_tsne_experiment",
            "description": "Feature-space t-SNE visualisation of the output-layer "
            "features (before label mapping) using 5000 training samples per dataset.",
        },
    ]


def run_analysis(name: str, *args, **kwargs):
    """Dispatch to one analysis entry point by name.

    Parameters
    ----------
    name:
        ``"plot_patch_size"`` (aliases: ``"figure4"``, ``"patch_size"``) or
        ``"tsne"`` / ``"tsne_features"`` / ``"feature_space"``.
    *args, **kwargs:
        Forwarded to the resolved entry point.

    Raises
    ------
    ValueError
        If ``name`` is unknown or the requested analysis module is unavailable.
    """
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "figure4": "plot_patch_size",
        "patch_size": "plot_patch_size",
        "plot_patchsize": "plot_patch_size",
        "tsne": "tsne_features",
        "feature_space": "tsne_features",
        "features": "tsne_features",
    }
    key = aliases.get(key, key)

    if key == "plot_patch_size":
        if not _PLOT_PATCH_SIZE_AVAILABLE:  # pragma: no cover
            raise ValueError("analysis module 'plot_patch_size' is not available")
        return plot_patch_size_from_file(*args, **kwargs)
    if key == "tsne_features":
        if not _TSNE_FEATURES_AVAILABLE:  # pragma: no cover
            raise ValueError("analysis module 'tsne_features' is not available")
        return run_tsne_experiment(*args, **kwargs)
    raise ValueError(
        f"unknown analysis '{name}'; expected one of {sorted(local_analyses)}".format(
            sorted_analyses=sorted(list(set(aliases.values()) | set(list_analyses())))
        )
    )


# Keep a module-level alias so the message above stays valid even if we later rename.
local_analyses = set(list_analyses())
