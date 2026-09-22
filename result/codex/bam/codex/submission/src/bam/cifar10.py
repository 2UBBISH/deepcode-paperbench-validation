"""Minimal CIFAR-10 loader (Krizhevsky, 2009) used by Section 5.3.

The images are modelled as continuous and rescaled to ``[-1, 1]``, matching the
``tanh`` output layer of the decoder.  The archive is downloaded once and cached
under ``data/cifar10``.
"""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from typing import Tuple

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "data", "cifar10")


def download_cifar10(root: str = DEFAULT_ROOT) -> str:
    """Download and extract CIFAR-10 if necessary; returns the extraction directory."""
    extracted = os.path.join(root, "cifar-10-batches-py")
    if os.path.isdir(extracted):
        return extracted
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, "cifar-10-python.tar.gz")
    if not os.path.exists(archive):
        print(f"downloading {URL} -> {archive}")
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(root)
    return extracted


def _load_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fh:
        entry = pickle.load(fh, encoding="bytes")
    data = entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data.astype(np.float32), np.asarray(entry[b"labels"], dtype=np.int64)


def load_cifar10(root: str = DEFAULT_ROOT, split: str = "train", normalize: bool = True):
    """Return ``(images, labels)`` with ``images`` of shape ``(N, 32, 32, 3)``.

    With ``normalize=True`` the pixel values lie in ``[-1, 1]``; the raw data is
    in ``[0, 255]`` as ``uint8`` (cast to float32).
    """
    path = download_cifar10(root)
    if split == "train":
        xs, ys = [], []
        for i in range(1, 6):
            x, y = _load_batch(os.path.join(path, f"data_batch_{i}"))
            xs.append(x)
            ys.append(y)
        x, y = np.concatenate(xs), np.concatenate(ys)
    elif split == "test":
        x, y = _load_batch(os.path.join(path, "test_batch"))
    else:
        raise ValueError(split)
    if normalize:
        x = x / 255.0 * 2.0 - 1.0
    return x, y
