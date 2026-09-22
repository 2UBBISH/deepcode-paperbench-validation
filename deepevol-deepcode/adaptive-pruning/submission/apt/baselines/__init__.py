"""APT baseline methods (paper Section 5.2, Addendum).

This package contains the comparison methods used in the paper's tables:

* :mod:`apt.baselines.ft`  -- full fine-tuning (FT) of the pretrained LM.
* :mod:`apt.baselines.lora` -- LoRA fine-tuning with frozen base weights.
* :mod:`apt.baselines.mask_tuning` -- Mask Tuning (``LoRA+Prune``) wrapping the
  external `retraining-free-pruning <https://github.com/WoosukKwon/retraining-free-pruning>`_
  repository, adapted so the model is additionally LoRA-tuned.
* :mod:`apt.baselines.cofi` -- CoFi pruning + self-distillation (``Prune+Distill``)
  wrapping `CoFiPruning <https://github.com/princeton-nlp/CoFiPruning>`_.
* :mod:`apt.baselines.lora_prune_distill` -- CoFi pruning/distillation with only
  LoRA + L0 modules tunable (``LoRA+Prune+Distill``).

The submodules are imported defensively: heavy or optional dependencies (and the
external baseline repositories, which are cloned separately) may be missing in a
bare environment, and importing :mod:`apt.baselines` must still succeed so that
the in-repo FT/LoRA baselines remain usable.
"""

from __future__ import annotations

import warnings
from typing import Dict, List

AVAILABLE: Dict[str, bool] = {
    "ft": False,
    "lora": False,
    "mask_tuning": False,
    "cofi": False,
    "lora_prune_distill": False,
}

__all__: List[str] = ["AVAILABLE", "available"]


def _register(name: str, symbols: List[str]) -> None:
    """Mark a submodule as available and extend ``__all__``."""
    AVAILABLE[name] = True
    for symbol in symbols:
        if symbol not in __all__:
            __all__.append(symbol)


# ---------------------------------------------------------------------------
# FT: always in-repo, dependency-light.
# ---------------------------------------------------------------------------
try:
    from . import ft  # noqa: F401
    from .ft import (  # noqa: F401
        FTConfig,
        FTTrainer,
        build_ft_optimizer,
        fine_tune,
        freeze_all,
        train_ft,
        unfreeze_all,
    )

    _register(
        "ft",
        [
            "ft",
            "FTConfig",
            "FTTrainer",
            "build_ft_optimizer",
            "fine_tune",
            "freeze_all",
            "train_ft",
            "unfreeze_all",
        ],
    )
except Exception as _exc:  # pragma: no cover - defensive
    warnings.warn(f"apt.baselines.ft unavailable: {_exc}")

# ---------------------------------------------------------------------------
# LoRA: in-repo PEFT fallback implementation.
# ---------------------------------------------------------------------------
try:
    from . import lora  # noqa: F401
    from .lora import (  # noqa: F401
        LoRAConfig,
        LoRAModel,
        LoRATrainer,
        apply_lora,
        lora_state_dict,
        merge_lora,
        train_lora,
    )

    _register(
        "lora",
        [
            "lora",
            "LoRAConfig",
            "LoRAModel",
            "LoRATrainer",
            "apply_lora",
            "lora_state_dict",
            "merge_lora",
            "train_lora",
        ],
    )
except Exception as _exc:  # pragma: no cover - defensive
    warnings.warn(f"apt.baselines.lora unavailable: {_exc}")

# ---------------------------------------------------------------------------
# Mask Tuning (LoRA+Prune): wraps the external retraining-free-pruning repo.
# ---------------------------------------------------------------------------
try:
    from . import mask_tuning  # noqa: F401
    from .mask_tuning import (  # noqa: F401
        MaskTuningConfig,
        MaskTuningMethod,
        external_repo_available as mask_tuning_available,
        prune_with_mask_tuning,
        train_mask_tuning,
    )

    _register(
        "mask_tuning",
        [
            "mask_tuning",
            "MaskTuningConfig",
            "MaskTuningMethod",
            "mask_tuning_available",
            "prune_with_mask_tuning",
            "train_mask_tuning",
        ],
    )
except Exception as _exc:  # pragma: no cover - defensive
    warnings.warn(f"apt.baselines.mask_tuning unavailable: {_exc}")

# ---------------------------------------------------------------------------
# CoFi (Prune+Distill): wraps the external CoFiPruning repo.
# ---------------------------------------------------------------------------
try:
    from . import cofi  # noqa: F401
    from .cofi import (  # noqa: F401
        CoFiConfig,
        CoFiMethod,
        external_repo_available as cofi_available,
        prune_with_cofi,
        train_cofi,
    )

    _register(
        "cofi",
        [
            "cofi",
            "CoFiConfig",
            "CoFiMethod",
            "cofi_available",
            "prune_with_cofi",
            "train_cofi",
        ],
    )
except Exception as _exc:  # pragma: no cover - defensive
    warnings.warn(f"apt.baselines.cofi unavailable: {_exc}")

# ---------------------------------------------------------------------------
# LoRA+Prune+Distill: CoFi pruning/distillation, only LoRA+L0 tunable.
# ---------------------------------------------------------------------------
try:
    from . import lora_prune_distill  # noqa: F401
    from .lora_prune_distill import (  # noqa: F401
        LoRAPruneDistillConfig,
        LoRAPruneDistillMethod,
        train_lora_prune_distill,
    )

    _register(
        "lora_prune_distill",
        [
            "lora_prune_distill",
            "LoRAPruneDistillConfig",
            "LoRAPruneDistillMethod",
            "train_lora_prune_distill",
        ],
    )
except Exception as _exc:  # pragma: no cover - defensive
    warnings.warn(f"apt.baselines.lora_prune_distill unavailable: {_exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#: Method registry mirroring the Table 2 / Table 4 / Table 7 / Table 8 rows.
BASELINE_METHODS = (
    "ft",
    "lora",
    "mask_tuning",          # paper: "LoRA+Prune"
    "cofi",                 # paper: "Prune+Distill"
    "lora_prune_distill",   # paper: "LoRA+Prune+Distill"
    "apt",
)

#: Display names used in the paper's tables.
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "ft": "FT",
    "lora": "LoRA",
    "mask_tuning": "LoRA+Prune",
    "cofi": "Prune+Distill",
    "lora_prune_distill": "LoRA+Prune+Distill",
    "apt": "APT",
}


def available() -> Dict[str, bool]:
    """Return a copy of the submodule availability map."""
    return dict(AVAILABLE)


def in_repo_methods() -> List[str]:
    """Baselines implemented inside this repository (no external clone)."""
    return [name for name in ("ft", "lora") if AVAILABLE.get(name)]


def external_methods() -> List[str]:
    """Baselines that wrap an external repository."""
    return [
        name
        for name in ("mask_tuning", "cofi", "lora_prune_distill")
        if AVAILABLE.get(name)
    ]


def get_method(name: str):
    """Return the trainer/entry point associated with ``name`` when available.

    Raises
    ------
    KeyError
        If ``name`` is not a known baseline method.
    RuntimeError
        If the method exists but its (sub)module could not be imported.
    """
    key = str(name).lower().replace("-", "_").replace("+", "_")
    key = key.replace("lora_prune", "lora_prune")  # keep canonical spelling
    aliases = {
        "finetune": "ft",
        "fine_tuning": "ft",
        "mask_tuning_lora": "mask_tuning",
        "retraining_free": "mask_tuning",
        "prune_distill": "cofi",
        "cofipruning": "cofi",
        "lora_prune_distill": "lora_prune_distill",
    }
    key = aliases.get(key, key)
    if key not in METHOD_DISPLAY_NAMES:
        raise KeyError(
            f"unknown baseline method {name!r}; expected one of "
            f"{tuple(METHOD_DISPLAY_NAMES)}"
        )
    if key not in AVAILABLE:
        raise RuntimeError(f"{key!r} is an APT method, not a baseline")
    if not AVAILABLE[key]:
        raise RuntimeError(
            f"baseline {key!r} is unavailable (its module failed to import); "
            "check the external repository / dependencies"
        )
    if key == "ft":
        return FTTrainer
    if key == "lora":
        return LoRATrainer
    if key == "mask_tuning":
        return MaskTuningMethod
    if key == "cofi":
        return CoFiMethod
    if key == "lora_prune_distill":
        return LoRAPruneDistillMethod
    raise KeyError(name)  # pragma: no cover


def method_display_name(name: str) -> str:
    """Paper table row label for a method key."""
    key = str(name).lower()
    return METHOD_DISPLAY_NAMES.get(key, name)


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    print("apt.baselines availability:", available())
    print("in-repo :", in_repo_methods())
    print("external:", external_methods())
