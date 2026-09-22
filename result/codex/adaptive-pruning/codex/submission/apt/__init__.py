"""APT: Adaptive Pruning and Tuning of pretrained language models.

Reference
---------
Bowen Zhao, Hannaneh Hajishirzi, Qingqing Cao.
"APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient
Training and Inference", ICML 2024.

The package is organised as follows:

``apt.adapter``      the APT adapter (LoRA + input/output pruning masks + dynamic rank)
``apt.blocks``       prunable-block bookkeeping, salience scoring, mask bookkeeping
``apt.wrap``         attaches APT adapters / masks / salience hooks to RoBERTa & T5
``apt.tuning``       adaptive tuning (salience-based rank growth)
``apt.distill``      efficient self-knowledge distillation
``apt.schedule``     cubic sparsity schedule + mu (distillation) schedule
``apt.trainer``      the two-stage APT training loop (prune+distill, then recover)
``apt.physical``     materialise the pruned architecture for inference-time speed/memory
"""

from .adapter import APTLinear  # noqa: F401
from .blocks import Block, ParamSlice, PruningState  # noqa: F401

__all__ = ["APTLinear", "Block", "ParamSlice", "PruningState"]

__version__ = "1.0.0"
