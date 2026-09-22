"""Grid searches for the learning rates of the gradient-based baselines.

Both the main text (Sections 5.1-5.3) and the addendum state that the learning
rates of the gradient-based methods (ADVI, Score, Fisher) were chosen by a grid
search; the values selected in the paper are recorded in
:data:`PAPER_SELECTED` and used as defaults, and :data:`PAPER_GRIDS` gives the
candidate sets over which the search is performed.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Candidate grids used for ADAM's learning rate in the paper
PAPER_GRIDS: Dict[str, Sequence[float]] = {
    # Appendix E.3 (Gaussian targets)
    "gaussian_default": (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
    # Appendix E.4 (sinh-arcsinh targets)
    "shash": (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
    # Appendix E.6 (deep generative model)
    "vae_advi": (0.001, 0.01, 0.02, 0.05),
}

# Learning rates selected by the grid searches reported in the paper
PAPER_SELECTED: Dict[str, Dict[str, object]] = {
    # Appendix E.3: "For ADVI and Fisher, the selected learning rate was 0.01.
    # For Score, a different learning rate was selected for each dimension
    # D = 4, 16, 64, 256: [0.01, 0.005, 0.001, 0.001]"
    "gaussian": {
        "ADVI": 0.01,
        "Fisher": 0.01,
        "Score": {4: 0.01, 16: 0.005, 64: 0.001, 256: 0.001},
    },
    # Appendix E.4: "The final selected learning rates were 0.02 for ADVI and
    # 0.05 for Fisher. For Score [...] 0.01, 0.001, 0.001 for the skewed targets
    # and 0.001, 0.01, 0.01 for the varying-tail targets."
    "shash_skew": {"ADVI": 0.02, "Fisher": 0.05, "Score": {0.2: 0.01, 1.0: 0.001, 1.8: 0.001}},
    "shash_tail": {"ADVI": 0.02, "Fisher": 0.05, "Score": {0.1: 0.001, 0.9: 0.01, 1.7: 0.01}},
    # Appendix E.6: "For ADVI, we consistently find the best learning rate to be
    # l = 0.02 (after searching l = 0.001, 0.01, 0.02, 0.05)."
    "vae_advi": {"ADVI": 0.02},
    # Appendix E.6: BaM learning rates for each batch size in the deep
    # generative model experiment.
    "vae_bam": {10: 0.1, 100: 50.0, 300: 7500.0},
}


def grid_search(candidates: Iterable[float], run_fn: Callable[[float], dict],
                score_fn: Callable[[dict], float]) -> Tuple[float, List[dict]]:
    """Run ``run_fn`` for every candidate and return the best one (lowest score).

    ``score_fn`` should return a *lower-is-better* summary of a run, e.g. the
    final forward KL divergence or the final relative mean error.
    """
    table = []
    for lr in candidates:
        out = run_fn(lr)
        table.append({"learning_rate": float(lr), "score": float(score_fn(out))})
    best = min(table, key=lambda r: r["score"])
    return float(best["learning_rate"]), table
