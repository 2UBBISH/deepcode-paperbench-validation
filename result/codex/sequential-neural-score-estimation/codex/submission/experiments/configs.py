"""Named experiment configurations.

``paper``
    The hyperparameters stated in Section 5.1 and Appendix E.3.2 of the paper
    (Adam with lr ``1e-4``, 3000 training iterations, batch size 50/200/500,
    15% validation split, early stopping after 1000 non-improving steps, 10
    rounds for sequential methods).

``extended``
    The same configuration with a longer training budget.  On CPU we found that
    the score network is still improving after the paper's 3000 iterations
    (the validation loss is monotonically decreasing), and that this matters
    most at small simulation budgets; this configuration is therefore provided
    for reproducing the paper's C2ST trends on slower hardware.  It does *not*
    change any methodological component.

``smoke``
    Tiny settings used by the test suite, so that the full pipeline can be
    exercised in under a minute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    max_iters: Optional[int] = None
    lr: Optional[float] = None
    t_scale: Optional[float] = None
    patience: Optional[int] = None
    smoke: bool = False


CONFIGS = {
    # Section 5.1 / Appendix E.3.2, verbatim
    "paper": RunConfig(max_iters=3000, lr=1e-4, t_scale=1000.0, patience=1000),
    # longer training budget (see module docstring)
    "extended": RunConfig(max_iters=15000, lr=1e-4, t_scale=1000.0, patience=1000),
    # literal reading of the time embedding in Appendix E.3.2 (sin(t / 10000^...))
    "paper_literal_embedding": RunConfig(max_iters=3000, lr=1e-4, t_scale=1.0, patience=1000),
    # quick smoke test
    "smoke": RunConfig(max_iters=60, lr=1e-4, t_scale=1000.0, patience=20, smoke=True),
}


def get_config(name: str) -> RunConfig:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name}; available: {sorted(CONFIGS)}")
    return CONFIGS[name]
