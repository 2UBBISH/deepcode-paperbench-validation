"""Per-task hyper-parameters.

Section 5.2 of the paper gives the shared configuration:

    "We set c = 4 and d = 8 for DDPMs, while c = 2 and d = 8 for LDMs.  To
    ensure the adapter layer outputs are initialized to zero, we set all the
    extra layer parameters to zero.  For similarity-guided training, we set
    gamma = 5.  ... For adversarial noise selection, we set J = 10 and
    omega = 0.02.  We employ a learning rate of 5e-5 for DDPMs and 1e-5 for
    LDMs to train with approximately 300 iterations and a batch size of 40."

and the supplementary material lists the exact values used for every task of
Table 1 / Table 3 (learning rate, ``C`` = the adaptor bottleneck dimension
``d``, ``omega``, ``J``, ``gamma`` and the number of training iterations).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

from .adaptor import AdaptorConfig
from .adversarial_noise import AdversarialNoiseConfig
from .ant_trainer import ANTConfig
from .classifier import ClassifierTrainConfig


@dataclass
class TaskConfig:
    """Everything needed to reproduce one row of the paper's experiments."""

    name: str
    backbone: str            # "ddpm" | "ldm"
    source: str              # "ffhq" | "lsun_church"
    target: str              # dataset key of the 10-shot target
    learning_rate: float
    bottleneck_dim: int      # "C" in the supplementary material (= d)
    bottleneck_factor: int   # "c" of Section 5.2 (4 for DDPMs, 2 for LDMs)
    gamma: float
    iterations: int
    omega: float = 0.02
    adversarial_steps: int = 10
    batch_size: int = 40
    image_size: int = 256
    num_shots: int = 10

    def ant_config(self) -> ANTConfig:
        return ANTConfig(
            iterations=self.iterations,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            gamma=self.gamma,
            adaptor=AdaptorConfig(
                bottleneck_factor=self.bottleneck_factor,
                bottleneck_dim=self.bottleneck_dim,
                num_heads=4,
                zero_init=True,
            ),
            adversarial=AdversarialNoiseConfig(
                num_steps=self.adversarial_steps,
                step_size=self.omega,
                normalize=True,
                normalization="per_sample",
            ),
        )

    def classifier_config(self) -> ClassifierTrainConfig:
        # supplementary material: Adam, lr 1e-4, batch 64, 300 iterations
        return ClassifierTrainConfig(iterations=300, lr=1e-4, batch_size=64)


def _task(
    name: str,
    backbone: str,
    source: str,
    target: str,
    learning_rate: float,
    bottleneck_dim: int,
    gamma: float,
    iterations: int,
) -> TaskConfig:
    return TaskConfig(
        name=name,
        backbone=backbone,
        source=source,
        target=target,
        learning_rate=learning_rate,
        bottleneck_dim=bottleneck_dim,
        bottleneck_factor=4 if backbone == "ddpm" else 2,
        gamma=gamma,
        iterations=iterations,
    )


TASK_CONFIGS: Dict[str, TaskConfig] = {
    # ---------------- DDPM (Section 5.2 + supplementary) ----------------
    "ddpm_ffhq_babies": _task("ddpm_ffhq_babies", "ddpm", "ffhq", "babies", 5e-6, 8, 3.0, 160),
    "ddpm_ffhq_sunglasses": _task(
        "ddpm_ffhq_sunglasses", "ddpm", "ffhq", "sunglasses", 5e-5, 8, 15.0, 200
    ),
    "ddpm_ffhq_raphael": _task("ddpm_ffhq_raphael", "ddpm", "ffhq", "raphael", 5e-5, 8, 10.0, 500),
    "ddpm_lsun_haunted_houses": _task(
        "ddpm_lsun_haunted_houses", "ddpm", "lsun_church", "haunted_houses", 5e-5, 8, 10.0, 320
    ),
    "ddpm_lsun_landscape_drawings": _task(
        "ddpm_lsun_landscape_drawings",
        "ddpm",
        "lsun_church",
        "landscape_drawings",
        5e-5,
        16,
        10.0,
        500,
    ),
    # ----------------- LDM (Section 5.2 + supplementary) ---------------
    "ldm_ffhq_babies": _task("ldm_ffhq_babies", "ldm", "ffhq", "babies", 5e-6, 16, 5.0, 320),
    "ldm_ffhq_sunglasses": _task(
        "ldm_ffhq_sunglasses", "ldm", "ffhq", "sunglasses", 1e-5, 8, 5.0, 280
    ),
    "ldm_ffhq_raphael": _task("ldm_ffhq_raphael", "ldm", "ffhq", "raphael", 1e-5, 8, 5.0, 320),
    "ldm_lsun_haunted_houses": _task(
        "ldm_lsun_haunted_houses", "ldm", "lsun_church", "haunted_houses", 2e-5, 8, 5.0, 500
    ),
    "ldm_lsun_landscape_drawings": _task(
        "ldm_lsun_landscape_drawings",
        "ldm",
        "lsun_church",
        "landscape_drawings",
        2e-5,
        8,
        5.0,
        500,
    ),
}

#: the configuration used for the ablation/claims of Section 5.2 defaults
DEFAULT_TASK = "ddpm_ffhq_sunglasses"


def get_task(name: Optional[str] = None, **overrides) -> TaskConfig:
    """Fetch a task configuration, optionally overriding individual fields."""
    config = TASK_CONFIGS[name or DEFAULT_TASK]
    return replace(config, **overrides) if overrides else config


def sensitivity_configs() -> Dict[str, list]:
    """Grids of the sensitivity analysis of Appendix B.3 (out of scope, but
    kept so that the same trainer can be re-used for it)."""
    return {
        "gamma": [1.0, 3.0, 5.0, 7.0, 9.0],
        "omega": [0.01, 0.02, 0.03, 0.04, 0.05],
        "iterations": [0, 50, 100, 150, 200, 250, 300, 350, 400],
    }
