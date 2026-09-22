"""Configurations of the paper (Appendix A2.1).

The defaults reproduce the model configuration that was used for all benchmark
experiments; the tasks of Sec. 4.2-4.4 (Lotka-Volterra, SIRD, Hodgkin-Huxley) use
8 instead of 6 transformer layers.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Dict

from .model import SimformerConfig

#: token dimension, number of layers/heads, attention size, widening factor and
#: the dimension of the time embedding (Appendix A2.1)
PAPER_MODEL_KWARGS: Dict = dict(
    d_model=50,
    n_heads=4,
    attention_size=10,
    widening_factor=3.0,
    n_layers=6,
    time_embed_dim=128,
)

#: SDE constants of Appendix A2.1
PAPER_SDE_KWARGS: Dict = dict(
    sigma_max=15.0,
    sigma_min=1e-4,
    beta_min=0.01,
    beta_max=10.0,
)

#: training constants of Appendix A2.1
PAPER_TRAINING_KWARGS: Dict = dict(
    batch_size=1000,
    lr=1e-4,
    val_fraction=0.1,
    patience=20,
)

#: number of simulations per task for the main results
PAPER_N_SIMULATIONS = (1000, 10000, 100000)

#: number of reverse SDE steps
PAPER_NUM_STEPS = 500


def paper_config(task: str, mask_mode: str = "dense",
                 sde: str = "vesde", **overrides) -> SimformerConfig:
    """Return the Simformer configuration of the paper for a task."""
    kwargs = dict(PAPER_MODEL_KWARGS)
    if task in ("lotka_volterra", "sird", "hodgkin_huxley"):
        kwargs["n_layers"] = 8
    kwargs["mask_mode"] = mask_mode
    kwargs["sde"] = sde
    kwargs["num_steps"] = PAPER_NUM_STEPS
    kwargs.update(overrides)
    return SimformerConfig(**kwargs)


def as_dict(config: SimformerConfig) -> dict:
    return asdict(config)
