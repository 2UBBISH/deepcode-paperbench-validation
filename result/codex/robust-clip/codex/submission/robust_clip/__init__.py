"""Robust CLIP -- reproduction of
"Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for
Robust Large Vision-Language Models" (Schlarmann*, Singh*, Croce, Hein, ICML 2024).

The package is organised along the three contributions of the paper:

* :mod:`robust_clip.training` -- the unsupervised adversarial fine-tuning scheme
  FARE (Sec. 3.2, Eq. (3)) together with the supervised baseline TeCoA
  (Mao et al., 2023) and the exact training hyper-parameters of App. B.1/B.3.
* :mod:`robust_clip.attacks`  -- APGD (Croce & Hein, 2020) and PGD attacks, the
  precision-aware attack *ensemble* of Sec. 4.1 / App. B.6 used for the LVLM
  evaluations, the targeted stealth attacks of Sec. 4.2 and the universal
  jailbreaking attack of Qi et al. (2023) used in Sec. 4.4.
* :mod:`robust_clip.eval` -- zero-shot classification (Sec. 4.3), LVLM
  captioning / VQA evaluation (Sec. 4.1), POPE (Sec. 4.4), SQA-I (Sec. 4.4),
  transfer attacks (Sec. 4.1) and the embedding-loss analysis of App. C.4.
* :mod:`robust_clip.models` -- OpenCLIP based CLIP encoders and the LLaVA /
  OpenFlamingo wrappers that plug a (robust) OpenCLIP vision tower into a frozen
  LVLM, as described in App. C.3 and the addendum.
"""

__version__ = "1.0.0"

from .models.clip_encoder import CLIPImageEncoder, load_clip
from .training.losses import (
    fare_loss,
    tecoa_loss,
    clean_embedding_loss,
    adversarial_embedding_loss,
)
from .attacks.apgd import APGDAttack
from .attacks.pgd import pgd_attack

__all__ = [
    "CLIPImageEncoder",
    "load_clip",
    "fare_loss",
    "tecoa_loss",
    "clean_embedding_loss",
    "adversarial_embedding_loss",
    "APGDAttack",
    "pgd_attack",
    "__version__",
]
