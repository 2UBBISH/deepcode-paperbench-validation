"""Coreset selection baselines compared in Section 5.2.

The paper compares with (i) Uniform, (ii) EL2N, (iii) GraNd, (iv) Influential,
(v) Moderate, (vi) CCS and (vii) Probabilistic.  As stated in the paper, the
baselines are *reproduced from their code repositories*; this package
re-implements each of them from its published description.

Every baseline exposes ``select(...) -> LongTensor`` returning the indices of
the selected examples.
"""

from .uniform import uniform_select
from .el2n import el2n_select, compute_el2n_scores
from .grand import grand_select, compute_grand_scores
from .moderate import moderate_select, compute_moderate_scores
from .influential import influential_select, compute_influence_scores
from .ccs import ccs_select
from .probabilistic import (ProbabilisticCoreset, ProbabilisticConfig,
                            probabilistic_select)

BASELINES = {
    "Uniform": uniform_select,
    "EL2N": el2n_select,
    "GraNd": grand_select,
    "Influential": influential_select,
    "Moderate": moderate_select,
    "CCS": ccs_select,
    "Probabilistic": probabilistic_select,
}

__all__ = [
    "uniform_select", "el2n_select", "compute_el2n_scores", "grand_select",
    "compute_grand_scores", "moderate_select", "compute_moderate_scores",
    "influential_select", "compute_influence_scores", "ccs_select",
    "ProbabilisticCoreset", "ProbabilisticConfig", "probabilistic_select",
    "BASELINES",
]
