"""Utility functions for RICE implementation."""
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the default torch device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: str,
    state: Dict[str, Any],
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load a training checkpoint."""
    if device is None:
        device = get_device()
    return torch.load(path, map_location=device)


def compute_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """Compute discounted returns for a trajectory."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance for value function diagnostics."""
    var_y = np.var(y_true)
    if var_y == 0:
        return float("nan")
    return 1.0 - np.var(y_true - y_pred) / var_y
