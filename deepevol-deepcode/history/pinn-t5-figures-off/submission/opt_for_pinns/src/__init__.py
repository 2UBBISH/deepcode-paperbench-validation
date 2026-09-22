"""opt_for_pinns: Optimizers for Physics-Informed Neural Networks.

Reproduction of "Challenges in Training PINNs: A Loss Landscape Perspective".
"""

from . import pdes
from . import data
from . import loss
from . import metrics
from . import model

__all__ = ["pdes", "data", "loss", "metrics", "model"]
