"""Class-hierarchy package: WordNet tree, information content, LCA distances and matrices.

Public surface (lazily resolved to avoid heavy/circular imports):

* :mod:`src.hierarchy.wordnet`      -- ``WordNetHierarchy``, ``build_wordnet_hierarchy``
* :mod:`src.hierarchy.info_content` -- ``HierarchyScorer``, ``InformationScore``, ``DepthScore``
* :mod:`src.hierarchy.lca`          -- ``lca_distance``, ``pairwise_lca_matrix``, ``reverse_lca_matrix``
* :mod:`src.hierarchy.lca_matrix`   -- ``process_lca_matrix``, ``build_lca_matrix``
* :mod:`src.hierarchy.latent_kmeans`-- ``build_latent_hierarchy``, ``LatentHierarchy``

Notes
-----
The sub-modules have non-trivial import chains (``latent_kmeans`` -> ``lca_matrix`` ->
``lca`` -> ``info_content`` -> ``wordnet``).  Everything in this ``__init__`` is therefore
resolved on first attribute access (PEP 562) so that ``from src.hierarchy import lca``
never triggers an import cycle, and so that merely importing the package stays cheap.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

__version__ = "0.1.0"

#: mapping public name -> module that provides it (relative to this package)
_EXPORTS: Dict[str, str] = {
    # --- wordnet -----------------------------------------------------------------
    "WordNetHierarchy": "wordnet",
    "build_wordnet_hierarchy": "wordnet",
    "parse_hierarchy_csv": "wordnet",
    "build_parents_from_wnids": "wordnet",
    "build_synthetic_parents": "wordnet",
    "build_two_pair_hierarchy": "wordnet",
    "get_imagenet_wnids": "wordnet",
    "is_synset": "wordnet",
    "IMAGENET_NUM_CLASSES": "wordnet",
    # --- info_content ------------------------------------------------------------
    "NodeScorer": "info_content",
    "DepthScore": "info_content",
    "InformationScore": "info_content",
    "HierarchyScorer": "info_content",
    "compute_information_content": "info_content",
    "compute_depths": "info_content",
    "class_information_content": "info_content",
    "class_depth_score": "info_content",
    "lca_distance_information": "info_content",
    "lca_distance_depth": "info_content",
    # --- lca ---------------------------------------------------------------------
    "LcaDistance": "lca",
    "lca_distance": "lca",
    "pairwise_lca_distance": "lca",
    "pairwise_lca_matrix": "lca",
    "lca_distance_matrix_from_scores": "lca",
    "reverse_lca_matrix": "lca",
    "sanity_check_matrix": "lca",
    "DEFAULT_DISTANCE_MODE": "lca",
    # --- lca_matrix --------------------------------------------------------------
    "LcaMatrixProcessor": "lca_matrix",
    "process_lca_matrix": "lca_matrix",
    "min_max_scale": "lca_matrix",
    "power_scale": "lca_matrix",
    "invert_matrix": "lca_matrix",
    "lca_matrix_from_hierarchy": "lca_matrix",
    "build_lca_matrix": "lca_matrix",
    "to_torch_tensor": "lca_matrix",
    "from_torch_tensor": "lca_matrix",
    "matrix_diagonal_is_zero": "lca_matrix",
    "matrix_is_symmetric": "lca_matrix",
    "assert_lca_matrix_properties": "lca_matrix",
    "DEFAULT_LATENT_BASE_LEVEL": "lca_matrix",
    # --- latent_kmeans -----------------------------------------------------------
    "LatentHierarchy": "latent_kmeans",
    "ClassFeatureAccumulator": "latent_kmeans",
    "build_latent_hierarchy": "latent_kmeans",
    "latent_hierarchy_from_features": "latent_kmeans",
    "latent_hierarchies_from_sources": "latent_kmeans",
    "extract_latent_hierarchy_from_model": "latent_kmeans",
    "latent_lca_matrix": "latent_kmeans",
    "latent_distance_matrix": "latent_kmeans",
    "class_mean_features": "latent_kmeans",
    "extract_class_features": "latent_kmeans",
    "kmeans_levels": "latent_kmeans",
    "fit_kmeans": "latent_kmeans",
    "numpy_kmeans": "latent_kmeans",
    "l2_normalize": "latent_kmeans",
    "level_cluster_count": "latent_kmeans",
    "synthetic_latent_hierarchy": "latent_kmeans",
    "DEFAULT_MAX_LEVEL": "latent_kmeans",
    "DEFAULT_BASE_LEVEL": "latent_kmeans",
}

_SUBMODULES: Tuple[str, ...] = (
    "wordnet",
    "info_content",
    "lca",
    "lca_matrix",
    "latent_kmeans",
)

__all__ = ["__version__"] + list(_SUBMODULES) + sorted(_EXPORTS)


def _import_submodule(name: str) -> Any:
    """Import a sibling sub-module, tolerating flat ``sys.path`` layouts."""
    candidates = [
        f"{__name__}.{name}",
        f"src.hierarchy.{name}",
        f"hierarchy.{name}",
        name,
    ]
    last_exc: Optional[Exception] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - layout dependent
            last_exc = exc
    raise ImportError(f"could not import hierarchy sub-module {name!r}") from last_exc


def __getattr__(name: str) -> Any:
    # PEP 562 lazy attribute access.
    if name in _SUBMODULES:
        module = _import_submodule(name)
        globals()[name] = module
        return module
    if name in _EXPORTS:
        module = _import_submodule(_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + list(_SUBMODULES) + list(_EXPORTS)))


from typing import Optional  # noqa: E402  (kept last to avoid unused-import warnings)
