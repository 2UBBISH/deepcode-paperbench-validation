"""Reproduction entry-point scripts for *LCA-on-the-Line*.

This package groups the top-level CLI drivers that reproduce the paper's
tables and figures:

``run_correlation.py``      -> Table 2, Figures 1 & 5 (LCA-on-the-Line)
``run_ood_prediction.py``   -> Table 3 (OOD performance prediction + baselines)
``run_latent_hierarchy.py`` -> Table 4 (K-means latent hierarchies)
``run_soft_label_probe.py`` -> Tables 5/6/9/10, Figure 8 (taxonomy soft labels)
``run_prompt_eval.py``      -> Table 14 (taxonomy-aware prompt engineering)

Every script is import-safe (it exposes ``build_arg_parser`` and ``main`` and
only touches torch/torchvision/open_clip/CLIP/``datasets`` lazily inside its
functions).  To keep ``import scripts`` cheap and avoid importing any of those
heavy stacks at package-import time, this initializer resolves the script
modules and their public symbols lazily via PEP 562 ``__getattr__``.

Examples
--------
>>> from scripts import run_correlation          # doctest: +SKIP
>>> run_correlation.main(["--from-cache-only"])  # doctest: +SKIP
>>> from scripts.run_prompt_eval import main     # doctest: +SKIP
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__version__ = "0.1.0"

#: Script modules (without the ``.py`` extension) that live in this package.
_SUBMODULES: Tuple[str, ...] = (
    "run_correlation",
    "run_ood_prediction",
    "run_latent_hierarchy",
    "run_soft_label_probe",
    "run_prompt_eval",
)

#: Public symbols re-exported from each script module.
_EXPORTS: Dict[str, str] = {
    # run_correlation.py -- Table 2 / Figures 1 & 5
    "build_score_table": "run_correlation",
    "compute_correlation_table": "run_correlation",
    "format_correlation_table": "run_correlation",
    "check_against_table2": "run_correlation",
    "check_baseline_collapse": "run_correlation",
    "check_table8": "run_correlation",
    "make_figures": "run_correlation",
    # run_ood_prediction.py -- Table 3
    "load_config": "run_ood_prediction",
    "parse_ood_roots": "run_ood_prediction",
    "load_records_from_json": "run_ood_prediction",
    "records_from_cache": "run_ood_prediction",
    "discover_cached_models": "run_ood_prediction",
    "build_or_load_records": "run_ood_prediction",
    "run_table3": "run_ood_prediction",
    "format_mae_table": "run_ood_prediction",
    "check_against_table3": "run_ood_prediction",
    "check_lca_beats_baselines": "run_ood_prediction",
    "success_summary": "run_ood_prediction",
    # run_latent_hierarchy.py -- Table 4
    "resolve_cache_dir": "run_latent_hierarchy",
    "resolve_results_dir": "run_latent_hierarchy",
    "find_cache_file": "run_latent_hierarchy",
    "load_npz": "run_latent_hierarchy",
    "build_accuracy_rows": "run_latent_hierarchy",
    "build_latent_hierarchies": "run_latent_hierarchy",
    "hierarchy_matrix": "run_latent_hierarchy",
    "id_lca_for_hierarchy": "run_latent_hierarchy",
    "correlation_for_hierarchy": "run_latent_hierarchy",
    "aggregate_statistics": "run_latent_hierarchy",
    "check_against_table4": "run_latent_hierarchy",
    "check_robustness": "run_latent_hierarchy",
    "synthetic_setup": "run_latent_hierarchy",
    # run_soft_label_probe.py -- Tables 5/6/9/10, Figure 8
    "build_wordnet_matrix": "run_soft_label_probe",
    "build_latent_matrix": "run_soft_label_probe",
    "run_backbone": "run_soft_label_probe",
    "run_table5": "run_soft_label_probe",
    "run_table6": "run_soft_label_probe",
    "run_table10": "run_soft_label_probe",
    "table10_correlations": "run_soft_label_probe",
    "sweep_interpolation": "run_soft_label_probe",
    "select_operating_points": "run_soft_label_probe",
    "make_probe_config": "run_soft_label_probe",
    "make_figure8": "run_soft_label_probe",
    "check_against_table5": "run_soft_label_probe",
    "check_against_table6": "run_soft_label_probe",
    "check_against_table10": "run_soft_label_probe",
    "check_soft_loss_improves_ood": "run_soft_label_probe",
    "check_latent_beats_baseline": "run_soft_label_probe",
    # run_prompt_eval.py -- Table 14
    "run_prompt_evaluation": "run_prompt_eval",
    "evaluate_prompts": "run_prompt_eval",
    "check_against_table14": "run_prompt_eval",
    "format_table14": "run_prompt_eval",
    "build_hierarchy": "run_prompt_eval",
    "load_class_names": "run_prompt_eval",
    "build_dataset": "run_prompt_eval",
    "build_loader": "run_prompt_eval",
    "build_prompt_encoder": "run_prompt_eval",
    "collect_image_features": "run_prompt_eval",
    "load_features_json": "run_prompt_eval",
    "synthetic_features": "run_prompt_eval",
}

#: Symbols that exist in more than one script; resolved in ``_SUBMODULES`` order.
_COMMON_EXPORTS: Tuple[str, ...] = (
    "main",
    "build_arg_parser",
    "save_json",
)

#: Paper-facing aliases (hyphenated / Table-number based names) -> canonical name.
_ALIASES: Dict[str, str] = {
    "RunCorrelation": "run_correlation",
    "RunOodPrediction": "run_ood_prediction",
    "RunLatentHierarchy": "run_latent_hierarchy",
    "RunSoftLabelProbe": "run_soft_label_probe",
    "RunPromptEval": "run_prompt_eval",
    "Table2": "compute_correlation_table",
    "Table3": "run_table3",
    "Table4": "build_latent_hierarchies",
    "Table5": "run_table5",
    "Table6": "run_table6",
    "Table10": "run_table10",
    "Table14": "format_table14",
    "Figure1": "make_figures",
    "Figure8": "make_figure8",
}

__all__ = ["__version__"] + list(_SUBMODULES) + sorted(_EXPORTS) + sorted(_ALIASES)


def _import_submodule(name: str) -> Any:
    """Import a script module, tolerating several ``sys.path`` layouts.

    Tries, in order: ``scripts.<name>`` (this package), the same package via its
    fully qualified ``__name__``, and finally the bare module name (flat layout
    where the script directory itself is on ``sys.path``).
    """
    candidates = list(
        dict.fromkeys(
            [
                f"{__name__}.{name}",
                f"scripts.{name}",
                name,
            ]
        )
    )
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except ImportError as exc:  # pragma: no cover - defensive
            last_error = exc
            continue
    if last_error is not None:  # pragma: no cover - defensive
        raise last_error
    raise ImportError(name)  # pragma: no cover - defensive


def _resolve(name: str) -> Any:
    """Resolve ``name`` to a script module, an exported symbol, or an alias."""
    if name in _SUBMODULES:
        return _import_submodule(name)

    target = _ALIASES.get(name, name)
    if target in _SUBMODULES:
        return _import_submodule(target)

    provider = _EXPORTS.get(target)
    if provider is not None:
        module = _import_submodule(provider)
        try:
            return getattr(module, target)
        except AttributeError:  # pragma: no cover - interface drift fallback
            pass

    if target in _COMMON_EXPORTS:
        for submodule in _SUBMODULES:
            module = _import_submodule(submodule)
            if hasattr(module, target):
                return getattr(module, target)

    # Last resort: probe every script module for the symbol.
    for submodule in _SUBMODULES:
        module = _import_submodule(submodule)
        if hasattr(module, target):
            return getattr(module, target)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute hook (memoized into module globals on access)."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    value = _resolve(name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """Advertise submodules, exports and aliases for introspection."""
    return sorted(
        set(globals())
        | set(_SUBMODULES)
        | set(_EXPORTS)
        | set(_COMMON_EXPORTS)
        | set(_ALIASES)
    )


def __getattr_many__() -> Dict[str, Any]:
    """Eagerly resolve every advertised name (test/introspection helper).

    Failures are returned in place of the resolved value instead of raising, so
    the caller can inspect partial results.
    """
    results: Dict[str, Any] = {}
    for name in sorted(set(_SUBMODULES) | set(_EXPORTS) | set(_COMMON_EXPORTS) | set(_ALIASES)):
        try:
            results[name] = _resolve(name)
        except Exception as exc:  # noqa: BLE001 - introspection helper
            results[name] = exc
    return results


def run_all(verbose: bool = True) -> Tuple[int, int, List[str]]:
    """Run the Phase A sanity suite from the test package.

    Convenience re-export so ``python -c "import scripts; scripts.run_all()"``
    works without knowing the ``tests`` layout.  Returns
    ``(passed, failed, failures)``.
    """
    import os
    import sys

    if verbose:
        print("Delegating to tests.test_lca_sanity.run_all() ...")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (repo_root, os.path.join(repo_root, "src")):
        if path not in sys.path:
            sys.path.insert(0, path)

    tests = importlib.import_module("tests.test_lca_sanity")
    return tests.run_all(verbose=verbose)  # type: ignore[no-any-return]


def main(argv: Optional[List[str]] = None) -> int:
    """Print available reproduction drivers (there is no single entry point).

    Each paper artifact has its own script; this helper simply lists them and
    returns 0.  Provided so that ``python -m scripts`` gives useful guidance.
    """
    lines = [
        "LCA-on-the-Line reproduction drivers (run one script per artifact):",
        "  python scripts/run_correlation.py       # Table 2, Figures 1 & 5",
        "  python scripts/run_ood_prediction.py    # Table 3",
        "  python scripts/run_latent_hierarchy.py  # Table 4",
        "  python scripts/run_soft_label_probe.py  # Tables 5/6/9/10, Figure 8",
        "  python scripts/run_prompt_eval.py       # Table 14",
        "",
        "Offline smoke tests: add --allow-synthetic / --synthetic and use --from-cache-only",
        "where available. See README.md for full instructions.",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
