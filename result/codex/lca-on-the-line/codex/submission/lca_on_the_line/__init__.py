"""LCA-on-the-Line: benchmarking out-of-distribution generalization with class taxonomies.

This package reproduces the core contributions of

    "LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with
     Class Taxonomies" (Shi et al., ICML 2024)

The public API mirrors the main body of the paper:

* :mod:`lca_on_the_line.hierarchy` -- WordNet class hierarchy + LCA computation
* :mod:`lca_on_the_line.lca`       -- LCA / ELCA distances (Section 2, Appendix D)
* :mod:`lca_on_the_line.metrics`   -- R^2 / PEA / KEN / SPE / MAE (Appendix D.1)
* :mod:`lca_on_the_line.models`    -- the 75 pretrained VMs / VLMs (Appendix A)
* :mod:`lca_on_the_line.data`      -- ImageNet + 5 OOD datasets
* :mod:`lca_on_the_line.evaluate`  -- the main benchmark loop (Section 4.1 / 4.2)
* :mod:`lca_on_the_line.baselines` -- AC / Aline-S / Aline-D baselines (Section 4.2)
* :mod:`lca_on_the_line.latent`    -- K-means latent hierarchies (Section 4.3.1)
* :mod:`lca_on_the_line.soft_labels` -- LCA soft labels + linear probing (Section 4.3.2)
* :mod:`lca_on_the_line.prompt_engineering` -- taxonomy prompts (Section 4.3.3)
"""

from .hierarchy import WordNetHierarchy, load_imagenet_class_index, load_wordnet_hierarchy
from .lca import (
    d_lca_information,
    d_lca_path,
    dataset_lca,
    dataset_lca_from_matrix,
    dataset_elca,
)

__all__ = [
    "WordNetHierarchy",
    "load_imagenet_class_index",
    "load_wordnet_hierarchy",
    "d_lca_information",
    "d_lca_path",
    "dataset_lca",
    "dataset_lca_from_matrix",
    "dataset_elca",
]

__version__ = "1.0.0"
