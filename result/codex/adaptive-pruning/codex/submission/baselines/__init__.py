"""Baselines compared against APT in the paper.

============================================  ==========================================
baseline                                      implementation
============================================  ==========================================
``FT``                                        :mod:`baselines.finetune`
``LoRA``                                      :mod:`baselines.lora`
``LoRA+Prune`` (Mask Tuning, Kwon et al.)     :mod:`baselines.mask_tuning`
``Prune+Distill`` (CoFi, Xia et al.)          :mod:`baselines.cofi`
``LoRA+Prune+Distill``                        :mod:`baselines.cofi` (LoRA-only tuning)
============================================  ==========================================

Per the task addendum the mask-tuning baseline follows
https://github.com/WoosukKwon/retraining-free-pruning (adapted so that it can be
applied to a LoRA-tuned model) and the CoFi baseline follows
https://github.com/princeton-nlp/CoFiPruning (adapted so that only LoRA and the
L0 modules are tuned).
"""

from .cofi import run_cofi, run_lora_prune_distill  # noqa: F401
from .finetune import run_finetune  # noqa: F401
from .lora import LoRALinear, run_lora  # noqa: F401
from .mask_tuning import (  # noqa: F401
    MaskTuningPruner,
    apply_structured_mask,
    run_lora_prune,
)

__all__ = [
    "run_finetune",
    "run_lora",
    "run_lora_prune",
    "run_cofi",
    "run_lora_prune_distill",
    "MaskTuningPruner",
    "apply_structured_mask",
    "LoRALinear",
]
