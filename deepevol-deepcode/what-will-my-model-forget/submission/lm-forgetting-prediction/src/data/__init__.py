"""Data pipeline for the "What Will My Model Forget?" reproduction.

This sub-package implements Phase 1 of the reproduction plan: loading the P3 and
MMLU task pools, grading model predictions with the SQuAD-2.0 exact-match rule,
and assembling the paper's dataset artifacts.

Modules
-------
- ``em_eval``           : SQuAD-2.0 style Exact Match (normalization + max-over-references).
                          Used for ``D_PT_hat`` filtering, ``D_R`` collection, the
                          base-EM sanity numbers (Table 7) and the Edit Success Rate.
- ``p3_loader``         : P3 (Public Pool of Prompts) train/test loaders via PromptSource,
                          with a local ReCross JSON fallback. 100 examples/task -> ``D_PT``
                          (36 tasks = 3600 upstream examples) and the 8 BART0 ``D_R`` tasks.
- ``mmlu_loader``       : MMLU validation loader over all 57 subjects, used to collect the
                          mispredicted-example pool ``D_R`` for the FLAN-T5 experiments.
- ``dataset_builders``  : Builds ``D_PT``, filters to ``D_PT_hat``, collects ``D_R``, performs
                          the 60/40 ``D_R^Train`` / ``D_R^Test`` split and the BART0 ID/OOD
                          task partition (Appendix B).

All heavy third-party imports (``datasets``, ``promptsource``) are performed lazily inside
the functions of the leaf modules so that this package stays importable without the full
data stack installed.
"""

from __future__ import annotations

__all__ = [
    "em_eval",
    "p3_loader",
    "mmlu_loader",
    "dataset_builders",
]
