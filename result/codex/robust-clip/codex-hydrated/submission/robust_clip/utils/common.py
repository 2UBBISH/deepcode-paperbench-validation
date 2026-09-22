"""Small helpers shared by the training / attack / evaluation entry points."""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys

import numpy as np
import torch


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("robust_clip")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOGGER = _build_logger()


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    """Pick the best available device (CUDA > MPS > CPU)."""
    if preferred is not None and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_FRACTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")


def parse_epsilon(value) -> float:
    """Accept ``"2/255"``, ``"4/255"``, ``0.007843`` or ``2`` (in 1/255 units).

    The paper always writes the radii as fractions of 255 (``eps = 2/255`` and
    ``eps = 4/255``); the command line arguments therefore accept the fraction
    notation directly.
    """
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = _FRACTION_RE.match(text)
    if match:
        return float(match.group(1)) / float(match.group(2))
    return float(text)


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"y", "yes", "t", "true", "1"}:
        return True
    if value.lower() in {"n", "no", "f", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {value!r}")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', 'cuda', 'cuda:0', 'mps', ...",
    )
    parser.add_argument("--output-dir", type=str, default="runs/default")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser


def human_readable_epsilon(eps: float) -> str:
    """``0.0078431...`` -> ``'2/255'`` (used for checkpoint naming)."""
    scaled = eps * 255.0
    if abs(scaled - round(scaled)) < 1e-6 and round(scaled) != 0:
        return f"{int(round(scaled))}/255"
    return f"{eps:g}"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
