"""Simformer: all-in-one simulation-based inference.

Reference implementation of

    Gloeckler, Deistler, Weilbach, Wood, Macke:
    "All-in-one simulation-based inference", ICML 2024.

The package is organised along the sections of the paper::

    simformer.masks       -- attention masks M_E and graph inversion (Sec. 3.2)
    simformer.tokenizer   -- the tokenizer for SBI (Sec. 3.1)
    simformer.transformer -- transformer score network (Fig. 2)
    simformer.sde         -- VESDE / VPSDE (Appendix A2.1)
    simformer.model       -- training, sampling and guidance (Sec. 3.3, 3.4)
    simformer.tasks       -- the simulators of the experiments (Sec. 4)
    simformer.reference   -- MCMC reference samples of arbitrary conditionals
    simformer.baselines   -- NPE / NLE / NRE baselines (sbi library)
    simformer.metrics     -- C2ST and expected coverage
"""

from .masks import (graph_inversion, inversion_attention_mask, moralize,
                    undirected)
from .metrics import c2st_accuracy, expected_coverage
from .model import (CallableConstraint, Constraint, Interval, IntervalLowerBound,
                    IntervalUpperBound, LinearConstraint, Simformer,
                    SimformerConfig)
from .problem import Problem
from .sde import VESDE, VPSDE, SDE, get_sde
from .tasks import BENCHMARK_TASKS, TASKS, Task, get_task
from .tokenizer import IdentifierEmbedding, Tokenizer, TokenizerConfig
from .transformer import TransformerConfig, TransformerScoreNet

__all__ = [
    "Simformer", "SimformerConfig", "Problem", "Task", "TASKS",
    "BENCHMARK_TASKS", "get_task", "SDE", "VESDE", "VPSDE", "get_sde",
    "Tokenizer", "TokenizerConfig", "IdentifierEmbedding",
    "TransformerScoreNet", "TransformerConfig", "Constraint",
    "Interval", "IntervalUpperBound", "IntervalLowerBound", "LinearConstraint",
    "CallableConstraint", "graph_inversion", "inversion_attention_mask",
    "moralize", "undirected", "c2st_accuracy", "expected_coverage",
]

__version__ = "0.1.0"
