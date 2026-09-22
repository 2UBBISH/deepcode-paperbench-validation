"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all available frameworks)."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - cpu only here
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
