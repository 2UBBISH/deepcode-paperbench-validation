"""Central place for every hyper-parameter that is quoted in the FOA paper.

The default values below are exactly the ones reported in
"Test-Time Model Adaptation with Only Forward Passes" (ICML 2024),
Section 4 "Implementation Details" / Appendix B.2, unless a comment says
otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Sequence


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return False


# --------------------------------------------------------------------------------------
# CMA-ES defaults (Section 3.1 / Appendix B.2)
# --------------------------------------------------------------------------------------
def default_popsize(prompt_dim: int, formula: bool = False) -> int:
    """Population size of the CMA-ES optimiser.

    The paper writes ``K = 28 = 4 + 3 * log(prompt dim)`` and uses ``K = 28`` for
    every main experiment.  With ``N_p = 3`` prompt embeddings of dimension
    ``d = 768`` (ViT-Base) the prompt dimension is ``N_p * d = 2304`` and the
    textbook formula ``4 + floor(3 * ln(2304)) = 27``; ``ceil`` gives 28, which is
    what the paper quotes.  We therefore default to the explicitly quoted ``K = 28``
    and only use the formula when ``formula=True``.
    """
    if not formula:
        return 28
    return int(math.ceil(4 + 3 * math.log(prompt_dim)))


@dataclass
class FOAConfig:
    """Hyper-parameters of FOA (forward-only prompt adaptation + activation shifting)."""

    # ---- prompt / optimiser ---------------------------------------------------------
    num_prompts: int = 3              # N_p
    prompt_init: str = "uniform"      # "uniform initialization" (Sec. 4 / Alg. 1)
    prompt_init_bound: float = 1.0    # uniform init range [-b, b]
    popsize: int = 28                 # K
    popsize_from_formula: bool = False
    cma_sigma0: float = 1.0           # tau^(0) in Eqn. (6)
    # Algorithm 1 initialises m^(0) = 0; set this to True to centre the search on a
    # uniformly initialised prompt instead (the paper's "uniform initialization").
    cma_mean_from_prompt_init: bool = False
    cma_seed: Optional[int] = None

    # ---- fitness function (Eqn. 5) --------------------------------------------------
    lam: float = 0.4                  # trade-off lambda, scaled by BS / 64
    lam_scale_with_batch: bool = True
    use_entropy: bool = True          # left term of Eqn. (5)
    use_activation_discrepancy: bool = True  # right term of Eqn. (5)

    # ---- back-to-source activation shifting (Eqn. 7-9) -----------------------------
    use_activation_shifting: bool = True
    shift_gamma: float = 1.0          # gamma, step size of Eqn. (7)
    shift_ema_alpha: float = 0.1      # alpha, EMA factor of Eqn. (9)
    # NOTE: Algorithm 1 applies Eqn. (7) *before* the fitness value and the prediction is
    # computed, so the shift is always part of the candidate evaluation in this
    # reproduction (see foa/core/foa.py::FOA.step).

    # ---- source in-distribution statistics ------------------------------------------
    num_source_samples: int = 32      # Q, Appendix B.2 (32 samples are enough)

    # ---- evaluation -----------------------------------------------------------------
    batch_size: int = 64
    interval: int = 1                 # I; 1 == plain FOA (update every batch)
    interval_store: str = "feature"   # "feature" (FOA-I V1) or "image" (FOA-I V2)

    # ---- misc -----------------------------------------------------------------------
    device: str = "cuda" if _cuda_available() else "cpu"

    def resolved_popsize(self, prompt_dim: int) -> int:
        if self.popsize_from_formula:
            return default_popsize(prompt_dim, formula=True)
        return self.popsize

    def resolved_lambda(self, batch_size: Optional[int] = None) -> float:
        """lambda as it enters Eqn. (5)."""
        lam = self.lam
        if self.lam_scale_with_batch:
            bs = self.batch_size if batch_size is None else batch_size
            lam = lam * bs / 64.0
        return lam

    def to_dict(self):
        return asdict(self)
# --------------------------------------------------------------------------------------
# Dataset specific overrides quoted in the paper
# --------------------------------------------------------------------------------------
DATASET_LAMBDA = {
    # Appendix B.2: "The lambda in Eqn. (5) is set to 0.4 x BS/64 on
    # ImageNet-C/V2/Sketch, and 0.2 x BS/64 on ImageNet-R"
    "imagenet_c": 0.4,
    "imagenet_v2": 0.4,
    "imagenet_sketch": 0.4,
    "imagenet_r": 0.2,
    "imagenet": 0.4,
}


def foa_config_for_dataset(dataset: str, **overrides) -> FOAConfig:
    cfg = FOAConfig()
    key = dataset.lower()
    if key in DATASET_LAMBDA:
        cfg.lam = DATASET_LAMBDA[key]
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"unknown FOA option: {k}")
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------------------
# Baseline hyper-parameters (Appendix B.2 - "More Evaluation Protocols")
# --------------------------------------------------------------------------------------
@dataclass
class BaselineConfig:
    # TENT / SAR use SGD with momentum .9, lr 1e-3, batch size 64
    sgd_lr: float = 1e-3
    sgd_momentum: float = 0.9
    # CoTTA
    cotta_lr: float = 0.05
    cotta_aug_threshold: float = 0.1
    cotta_num_aug: int = 32
    cotta_restore_prob: float = 0.01
    cotta_ema_alpha: float = 0.999   # teacher EMA
    # LAME
    lame_knn_k: int = 5
    # T3A
    t3a_num_supports: int = 20


NUM_CLASSES_IMAGENET = 1000


def sar_entropy_threshold(num_classes: int = NUM_CLASSES_IMAGENET) -> float:
    """SAR: E_0 = 0.4 * ln(C) (Appendix B.2)."""
    return 0.4 * math.log(num_classes)
