"""Source package for the *What Will My Model Forget?* reproduction.

This package implements the complete pipeline for forecasting which upstream
pretraining examples will be forgotten when an instruction-tuned seq2seq LM is
refined to fix a single error, together with a replay-based refinement
algorithm that reduces catastrophic forgetting.

Sub-packages
------------
``src.data``
    P3 / MMLU loaders, SQuAD-2.0-style exact-match grading, and dataset builders
    for ``D_PT``, ``D_PT_hat``, ``D_R`` (60/40 split, ID/OOD split).
``src.modeling``
    Base PTLM wrapper ``f_0``, the model-refinement engine (Head / LoRA /
    Full-FT), the encoding function ``h``, and the persistent cache layer that
    makes forecasting cheap.
``src.forgetting``
    Ground-truth forgetting labels ``z_ij`` plus the frequency prior ``b_j``.
``src.forecasters``
    Threshold baseline, trainable logit-change-transfer forecaster,
    representation-based forecaster, and all training losses.
``src.replay``
    Sequential replay-based refinement with pluggable replay-selection
    strategies.
``src.eval``
    Metrics (F1, precision, recall, Edit Success, EM Drop Ratio) and the
    evaluation harness that renders Tables 1-4, Table 7 and Figure 3.

Every heavy dependency (``torch``, ``transformers``, ``peft``, ``datasets``,
``promptsource``) is imported lazily inside functions so that lightweight
utilities (metrics, configuration parsing, self-tests) remain importable in
environments without a GPU stack.
"""

from __future__ import annotations

__version__ = "0.1.0"

#: The upstream pools / result tables the paper reports on.
__all__ = [
    "data",
    "modeling",
    "forgetting",
    "forecasters",
    "replay",
    "eval",
    "__version__",
]
