"""
Utility helper functions for the FRE project.

Provides:
- Random seed setting for reproducibility
- Device selection (CPU/GPU)
- Parameter counting
- Time formatting
- Running mean/std tracking
- Cosine similarity
- Tensor/numpy conversion
- Batch iteration
- Logging configuration
"""

import os
import random
import logging
import time
from typing import Iterator, Optional, Tuple, Union, List, Dict, Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, random, and CUDA.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Return the torch device (CPU or GPU).

    Args:
        use_cuda: If True and CUDA is available, return GPU device.

    Returns:
        torch.device: 'cuda' or 'cpu'.
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model: PyTorch module.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        int: Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Convert seconds to a human-readable string (HH:MM:SS).

    Args:
        seconds: Time in seconds.

    Returns:
        str: Formatted time string.
    """
    if seconds < 0:
        return "0:00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class RunningMeanStd:
    """
    Tracks running mean and standard deviation of a data stream using Welford's
    online algorithm. Useful for state normalization when statistics are not
    known in advance.
    """

    def __init__(self, shape: Union[Tuple[int, ...], int] = ()):
        """
        Args:
            shape: Shape of the data (scalar or vector).
        """
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self._eps = 1e-8

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with a batch of data.

        Args:
            x: Data array of shape (batch_size, *shape).
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / max(total_count, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / max(total_count, 1)

        self.mean = new_mean
        self.var = M2 / max(total_count, 1)
        self.count = total_count

    @property
    def std(self) -> np.ndarray:
        """Return current standard deviation estimate."""
        return np.sqrt(self.var + self._eps)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize data using current mean and std.

        Args:
            x: Data to normalize.

        Returns:
            np.ndarray: Normalized data.
        """
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalize data using current mean and std.

        Args:
            x: Normalized data.

        Returns:
            np.ndarray: Denormalized data.
        """
        return x * self.std + self.mean


def cosine_similarity(x: Union[np.ndarray, torch.Tensor],
                      y: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        float: Cosine similarity in [-1, 1].
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    x = x.flatten().float()
    y = y.flatten().float()

    dot = torch.dot(x, y)
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)

    if norm_x < 1e-8 or norm_y < 1e-8:
        return 0.0

    return (dot / (norm_x * norm_y)).item()


def to_tensor(x: Union[np.ndarray, torch.Tensor, List],
              device: Optional[torch.device] = None,
              dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Convert numpy array or list to torch tensor.

    Args:
        x: Input data.
        device: Target device (optional).
        dtype: Target dtype (optional, defaults to float32).

    Returns:
        torch.Tensor: Converted tensor.
    """
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, list):
        t = torch.tensor(x, dtype=dtype if dtype is not None else torch.float32)
    else:
        t = torch.from_numpy(np.asarray(x))

    if dtype is not None:
        t = t.to(dtype)
    elif t.dtype == torch.float64:
        t = t.float()

    if device is not None:
        t = t.to(device)

    return t


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert torch tensor to numpy array.

    Args:
        x: Input tensor or array.

    Returns:
        np.ndarray: Converted numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def batch_iterator(data: Union[np.ndarray, Dict[str, np.ndarray]],
                   batch_size: int,
                   shuffle: bool = True,
                   rng: Optional[np.random.RandomState] = None) -> Iterator:
    """
    Yield batches from a dataset.

    Args:
        data: Either a numpy array of shape (N, ...) or a dict of such arrays.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle indices before batching.
        rng: Random state for reproducible shuffling.

    Yields:
        Batches of the same type as input (array or dict).
    """
    if rng is None:
        rng = np.random.RandomState()

    if isinstance(data, dict):
        # All arrays must have same first dimension
        keys = list(data.keys())
        n = len(data[keys[0]])
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield {k: data[k][batch_indices] for k in keys}
    else:
        n = len(data)
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, n, batch_size):
            yield data[indices[start:start + batch_size]]


def configure_logging(log_dir: Optional[str] = None,
                      level: int = logging.INFO,
                      name: str = "fre") -> logging.Logger:
    """
    Set up Python logging with console and optional file output.

    Args:
        log_dir: Directory for log file (if None, only console logging).
        level: Logging level.
        name: Logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance (R²-like metric) for regression evaluation.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        float: Explained variance score in (-inf, 1].
    """
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    var_y = np.var(y_true)
    if var_y < 1e-10:
        return 1.0 if np.mean((y_pred - y_true) ** 2) < 1e-10 else 0.0
    return 1.0 - np.var(y_true - y_pred) / var_y


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) score.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        float: R² score.
    """
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    return 1.0 - ss_res / ss_tot


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """
    Polyak (soft) update of target network parameters:
        target_param = tau * source_param + (1 - tau) * target_param

    Args:
        target: Target network.
        source: Source network.
        tau: Interpolation factor (0 < tau <= 1).
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """
    Hard update (copy) of target network parameters.

    Args:
        target: Target network.
        source: Source network.
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)


def freeze_module(module: torch.nn.Module) -> None:
    """
    Freeze all parameters of a module (set requires_grad=False).

    Args:
        module: PyTorch module to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def unfreeze_module(module: torch.nn.Module) -> None:
    """
    Unfreeze all parameters of a module (set requires_grad=True).

    Args:
        module: PyTorch module to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the total L2 norm of gradients for a model.

    Args:
        model: PyTorch module.

    Returns:
        float: Total gradient norm.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    return total_norm ** 0.5


def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """
    Clip gradients of a model by global norm and return the norm before clipping.

    Args:
        model: PyTorch module.
        max_norm: Maximum allowed gradient norm.

    Returns:
        float: Gradient norm before clipping.
    """
    grad_norm = compute_gradient_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return grad_norm


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries; later values override earlier ones.

    Args:
        *dicts: Dictionaries to merge.

    Returns:
        Dict: Merged dictionary.
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary using dot-separated keys.

    Args:
        d: Nested dictionary.
        parent_key: Key prefix for recursion.
        sep: Separator between keys.

    Returns:
        Dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)