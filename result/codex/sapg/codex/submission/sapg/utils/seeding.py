from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
