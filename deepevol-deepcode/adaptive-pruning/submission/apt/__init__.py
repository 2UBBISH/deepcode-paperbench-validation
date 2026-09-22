"""APT: Adaptive Pruning and Tuning of Pretrained Language Models.

Reference: "APT: Adaptive Pruning and Tuning Pretrained Language Models for
Efficient Training and Inference".

This package implements:
  * APT adapter (dynamic rank LoRA with binary input/output masks)  -> apt.adapters
  * Mask state + gradual decay bookkeeping                          -> apt.masks
  * Cubic sparsity schedule / mu schedule / rank schedule           -> apt.schedulers
  * Outlier-aware salience scoring with EMA                         -> apt.salience
  * Density sort + binary-search knapsack block selection           -> apt.block_selection
  * Adaptive tuning rank controller                                 -> apt.rank_controller
  * Self-knowledge distillation (teacher duplication, Tr, phi)      -> apt.distillation
  * Model wrapper injecting adapters into HF transformer blocks     -> apt.model_wrapper
  * Two-stage prune -> recover training loop                        -> apt.training
  * Merge LoRA weights and physically remove pruned blocks          -> apt.merge
"""

__version__ = "0.1.0"

from .adapters import APTAdapter, MaskedLinear  # noqa: F401

__all__ = ["APTAdapter", "MaskedLinear", "__version__"]
