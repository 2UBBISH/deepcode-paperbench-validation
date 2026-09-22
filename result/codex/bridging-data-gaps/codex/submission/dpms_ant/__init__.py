"""DPMs-ANT: Bridging Data Gaps in Diffusion Models with Adversarial
Noise-Based Transfer Learning.

Reference implementation of the paper (ICML 2024):

    Xiyu Wang, Baijiong Lin, Daochang Liu, Ying-Cong Chen, Chang Xu.
    "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based
    Transfer Learning."

The package is organised around the three contributions of the paper:

* :mod:`dpms_ant.guidance`  -- similarity-guided training (Section 4.1, Eq. 5)
* :mod:`dpms_ant.adversarial_noise` -- adversarial noise selection
  (Section 4.2, Eqs. 6-7)
* :mod:`dpms_ant.adaptor` -- the parameter-efficient adaptor module and the
  *only* parameters updated during transfer (Section 4.3 / Algorithm 1)

:mod:`dpms_ant.ant_trainer` glues the three pieces together into the exact
training procedure of Algorithm 1.
"""

from .schedules import DiffusionSchedule
from .adaptor import AdaptorConfig, add_adaptors, adaptor_parameters
from .adversarial_noise import AdversarialNoiseConfig, select_adversarial_noise
from .guidance import classifier_guidance, similarity_guided_loss
from .ant_trainer import ANTConfig, ANTTrainer

__all__ = [
    "DiffusionSchedule",
    "AdaptorConfig",
    "add_adaptors",
    "adaptor_parameters",
    "AdversarialNoiseConfig",
    "select_adversarial_noise",
    "classifier_guidance",
    "similarity_guided_loss",
    "ANTConfig",
    "ANTTrainer",
]

__version__ = "0.1.0"

