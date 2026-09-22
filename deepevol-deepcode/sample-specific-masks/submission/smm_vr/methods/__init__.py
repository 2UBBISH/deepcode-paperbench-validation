"""SMM comparison methods: shared-mask VR baselines and finetuning baselines.

This package aggregates the two method families used to compare against SMM:

* :mod:`smm_vr.methods.baselines` -- the shared-mask Visual Reprogramming
  baselines *Pad*, *Narrow*, *Medium* and *Full* (Sec. 5, Tables 1-2).
* :mod:`smm_vr.methods.finetuning` -- ``LoRA`` for ViT-L, ``Finetuning-FC`` and
  ``Finetuning-FC + SMM`` (Appendix E.1/E.2, Tables 13-14).

All imports are guarded so that ``import smm_vr.methods`` never fails during
incremental development.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []


def _extend(names: List[str]) -> None:
    """Append ``names`` to ``__all__`` without creating duplicates."""
    for name in names:
        if name not in __all__:
            __all__.append(name)


# ---------------------------------------------------------------------------
# Shared-mask VR baselines (Pad / Narrow / Medium / Full)
# ---------------------------------------------------------------------------
_BASELINES_AVAILABLE = False
try:  # pragma: no cover - guarded for incremental builds
    from .baselines import (
        BASELINE_NAMES,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_GAMMA,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_LR,
        DEFAULT_MILESTONES,
        MEDIUM_WIDTH_RATIO,
        NARROW_WIDTH_RATIO,
        SMALL_BATCH_DATASETS,
        SMALL_BATCH_SIZE,
        VIT_IMAGE_SIZE,
        BaselineEpochStats,
        BaselineHistory,
        BaselineTrainConfig,
        SharedMaskVR,
        baseline_training_config,
        build_all_baselines,
        build_baseline,
        build_pattern_optimizer,
        build_shared_mask_baseline,
        canonical_baseline_name,
        describe_baseline,
        evaluate_shared_baseline,
        list_baselines,
        make_baseline_mask,
        make_border_mask,
        make_full_mask,
        make_pad_mask,
        mask_coverage,
        resize_with_padding,
        train_baseline,
        train_baseline_one_seed,
        train_baseline_with_seeds,
        watermark_width,
    )

    _BASELINES_AVAILABLE = True
    _extend(
        [
            "BASELINE_NAMES",
            "DEFAULT_BATCH_SIZE",
            "DEFAULT_EPOCHS",
            "DEFAULT_GAMMA",
            "DEFAULT_IMAGE_SIZE",
            "DEFAULT_LR",
            "DEFAULT_MILESTONES",
            "MEDIUM_WIDTH_RATIO",
            "NARROW_WIDTH_RATIO",
            "SMALL_BATCH_DATASETS",
            "SMALL_BATCH_SIZE",
            "VIT_IMAGE_SIZE",
            "BaselineEpochStats",
            "BaselineHistory",
            "BaselineTrainConfig",
            "SharedMaskVR",
            "baseline_training_config",
            "build_all_baselines",
            "build_baseline",
            "build_pattern_optimizer",
            "build_shared_mask_baseline",
            "canonical_baseline_name",
            "describe_baseline",
            "evaluate_shared_baseline",
            "list_baselines",
            "make_baseline_mask",
            "make_border_mask",
            "make_full_mask",
            "make_pad_mask",
            "mask_coverage",
            "resize_with_padding",
            "train_baseline",
            "train_baseline_one_seed",
            "train_baseline_with_seeds",
            "watermark_width",
        ]
    )
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Finetuning baselines (LoRA / Finetuning-FC / Finetuning-FC + SMM)
# ---------------------------------------------------------------------------
_FINETUNING_AVAILABLE = False
try:  # pragma: no cover - guarded for incremental builds
    from .finetuning import (
        DEFAULT_FINETUNE_EPOCHS,
        DEFAULT_FINETUNE_LR,
        DEFAULT_GAMMA,
        DEFAULT_LORA_ALPHA,
        DEFAULT_LORA_EPOCHS,
        DEFAULT_LORA_LR,
        DEFAULT_LORA_RANK,
        DEFAULT_MILESTONES,
        FEATURE_DIMS,
        FINETUNE_METHODS,
        HIGH_RES_DATASETS,
        LORA_EXTRA_PARAMETERS_M,
        LOW_RES_DATASETS,
        MASK_LAYERS_BY_BACKBONE,
        SMM_EXTRA_PARAMETERS_M,
        FinetuneConfig,
        FinetuneEpochStats,
        FinetuneFC,
        FinetuneHistory,
        LoRALinear,
        LoRAMultiheadAttention,
        apply_lora,
        build_finetune_fc,
        build_finetune_fc_with_smm,
        build_lora_model,
        build_optimizer_and_scheduler,
        canonical_method_name,
        compare_table13,
        compare_table14,
        count_lora_parameters,
        describe_finetuning,
        evaluate_model,
        freeze_except_lora,
        list_finetuning_methods,
        lora_parameters,
        smm_trainable_parameters,
        train_epoch,
        train_finetune_fc,
        train_finetune_fc_with_smm,
        train_finetuning_method,
        train_finetuning_with_seeds,
        train_lora,
    )

    _FINETUNING_AVAILABLE = True
    _extend(
        [
            "DEFAULT_FINETUNE_EPOCHS",
            "DEFAULT_FINETUNE_LR",
            "DEFAULT_GAMMA",
            "DEFAULT_LORA_ALPHA",
            "DEFAULT_LORA_EPOCHS",
            "DEFAULT_LORA_LR",
            "DEFAULT_LORA_RANK",
            "DEFAULT_MILESTONES",
            "FEATURE_DIMS",
            "FINETUNE_METHODS",
            "HIGH_RES_DATASETS",
            "LORA_EXTRA_PARAMETERS_M",
            "LOW_RES_DATASETS",
            "MASK_LAYERS_BY_BACKBONE",
            "SMM_EXTRA_PARAMETERS_M",
            "FinetuneConfig",
            "FinetuneEpochStats",
            "FinetuneFC",
            "FinetuneHistory",
            "LoRALinear",
            "LoRAMultiheadAttention",
            "apply_lora",
            "build_finetune_fc",
            "build_finetune_fc_with_smm",
            "build_lora_model",
            "build_optimizer_and_scheduler",
            "canonical_method_name",
            "compare_table13",
            "compare_table14",
            "count_lora_parameters",
            "describe_finetuning",
            "evaluate_model",
            "freeze_except_lora",
            "list_finetuning_methods",
            "lora_parameters",
            "smm_trainable_parameters",
            "train_epoch",
            "train_finetune_fc",
            "train_finetune_fc_with_smm",
            "train_finetuning_method",
            "train_finetuning_with_seeds",
            "train_lora",
        ]
    )
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Convenience dispatchers
# ---------------------------------------------------------------------------
BASELINE_METHODS = ("pad", "narrow", "medium", "full")
"""Shared-mask VR baseline names (Sec. 5, Table 1)."""


def list_methods() -> List[str]:
    """Return every comparison method exposed by this package.

    The list starts with the shared-mask baselines (Pad, Narrow, Medium, Full),
    continues with the finetuning family (LoRA, Finetuning-FC,
    Finetuning-FC + SMM) and ends with ``"smm"`` (the paper's method, itself
    implemented in :mod:`smm_vr.modules.reprogram` + :mod:`smm_vr.engine`).
    """
    methods: List[str] = []
    if _BASELINES_AVAILABLE:
        methods.extend(list(baselines_names()))
    else:
        methods.extend(BASELINE_METHODS)
    if _FINETUNING_AVAILABLE:
        for name in _finetuning_names():
            if name not in methods:
                methods.append(name)
    for name in ("lora", "finetune_fc", "finetune_fc_smm", "smm"):
        if name not in methods:
            methods.append(name)
    return methods


def baselines_names() -> tuple:
    """Return the canonical shared-mask baseline names."""
    try:
        return tuple(list_baselines())
    except Exception:  # pragma: no cover - defensive
        return BASELINE_METHODS


def _finetuning_names() -> List[str]:
    try:
        return list(list_finetuning_methods())
    except Exception:  # pragma: no cover - defensive
        return ["lora", "finetune_fc", "finetune_fc_smm"]


def build_method(name: str, *args, **kwargs):
    """Build a comparison method by name.

    Dispatches to :func:`smm_vr.methods.baselines.build_baseline` for the
    shared-mask baselines and to the finetuning builders otherwise.
    """
    key = str(name).strip().lower().replace("-", "_")
    if key in BASELINE_METHODS:
        if not _BASELINES_AVAILABLE:  # pragma: no cover
            raise ImportError("smm_vr.methods.baselines is not importable")
        return build_baseline(key, *args, **kwargs)
    if key in ("lora", "finetune_fc", "finetune_fc_smm"):
        if not _FINETUNING_AVAILABLE:  # pragma: no cover
            raise ImportError("smm_vr.methods.finetuning is not importable")
        if key == "lora":
            return build_lora_model(*args, **kwargs)
        if key == "finetune_fc":
            return build_finetune_fc(*args, **kwargs)
        return build_finetune_fc_with_smm(*args, **kwargs)
    raise ValueError(
        f"Unknown method '{name}'. Expected one of {list_methods()!r}."
    )


_extend(["BASELINE_METHODS", "list_methods", "build_method"])
