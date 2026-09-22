"""Training subpackage for DPMs-ANT.

Aggregates the ANT training pipeline:

* :mod:`dpm_ant.training.sg_loss`         -- similarity-guided DPM loss (Eq. 5)
* :mod:`dpm_ant.training.adv_noise`       -- adversarial noise selection (Eq. 7, Algorithm 1)
* :mod:`dpm_ant.training.ant_trainer`     -- full ANT training loop (Eq. 8, Algorithm 1)
* :mod:`dpm_ant.training.classifier_train`-- fine-tuning the binary source/target classifier p_phi

Heavy torch modules are resolved lazily via :pep:`562` ``__getattr__`` so that importing
``dpm_ant.training`` never drags in the whole model stack (useful for the toy experiment
and for partial checkouts).
"""
from __future__ import annotations

import importlib
from typing import Dict, List

__all__: List[str] = [
    # --- sg_loss (Eq. 5) ---
    "SimilarityGuidedLoss",
    "similarity_guided_loss",
    "build_sg_loss",
    # --- adv_noise (Eq. 7, Algorithm 1) ---
    "AdversarialNoiseSelector",
    "select_adversarial_noise",
    "normalize_noise",
    "build_adv_noise_selector",
    # --- ant_trainer (Eq. 8, Algorithm 1) ---
    "ANTTrainer",
    "ANTConfig",
    "train_ant",
    # --- classifier_train (Section 5.2 / addendum) ---
    "ClassifierTrainer",
    "ClassifierTrainConfig",
    "train_classifier",
    "fine_tune_classifier",
    "noised_source_target_batch",
]

# public name -> defining submodule
_EXPORTS: Dict[str, str] = {
    "SimilarityGuidedLoss": "sg_loss",
    "similarity_guided_loss": "sg_loss",
    "build_sg_loss": "sg_loss",
    "AdversarialNoiseSelector": "adv_noise",
    "select_adversarial_noise": "adv_noise",
    "normalize_noise": "adv_noise",
    "build_adv_noise_selector": "adv_noise",
    "ANTTrainer": "ant_trainer",
    "ANTConfig": "ant_trainer",
    "train_ant": "ant_trainer",
    "ClassifierTrainer": "classifier_train",
    "ClassifierTrainConfig": "classifier_train",
    "train_classifier": "classifier_train",
    "fine_tune_classifier": "classifier_train",
    "noised_source_target_batch": "classifier_train",
}


def __getattr__(name: str):
    """Lazily import and cache a public training symbol."""
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + __all__))
