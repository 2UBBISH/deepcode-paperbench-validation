#!/usr/bin/env python3
"""Figure 7: visualise pairwise LCA distance matrices.

    python scripts/plot_lca_matrices.py \
        --class-features results/class_features --out figure7.png

The WordNet matrix is generated on the fly; latent matrices are built from the
per-class features cached by ``experiments_latent`` (``<model>.npy``).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.analysis import figure7  # noqa: E402
from lca_on_the_line.hierarchy import load_wordnet_hierarchy  # noqa: E402
from lca_on_the_line.latent import latent_lca_distance_matrix  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-features", default=None,
                        help="directory of cached per-class features (<model>.npy)")
    parser.add_argument("--models", nargs="*", default=["resnet50", "CLIP_RN50"])
    parser.add_argument("--out", default="figure7.png")
    args = parser.parse_args(argv)

    matrices = {
        "WordNet": load_wordnet_hierarchy().lca_distance_matrix("information")
    }
    if args.class_features:
        for name in args.models:
            path = os.path.join(
                args.class_features, "%s.npy" % name.replace("/", "_")
            )
            if not os.path.exists(path):
                print("[figure7] no cached features for %s" % name)
                continue
            matrices[name] = latent_lca_distance_matrix(np.load(path))

    figure7(matrices, args.out)
    print("wrote", args.out, "with matrices:", list(matrices))


if __name__ == "__main__":
    main()
