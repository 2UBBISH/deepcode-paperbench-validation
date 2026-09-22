"""Datasets used in the paper.

All datasets are loaded through ``torchvision`` (no Kaggle / API keys needed),
as requested by the addendum.  Images are kept as dense ``float32`` tensors in
``[0, 1]`` so that a coreset is a plain index tensor of the training set and
the bilevel masks can be applied with simple indexing.

Supported datasets:

* ``mnist``      -- MNIST (LeCun et al., 1998)
* ``mnist-s``    -- MNIST-S: a random subset of MNIST (Section 5.1)
* ``fmnist``     -- FashionMNIST (Xiao et al., 2017)
* ``svhn``       -- SVHN (Netzer et al., 2011)
* ``cifar10``    -- CIFAR-10 (Krizhevsky et al., 2009)

Corrupted variants used in Section 5.3:

* :func:`inject_symmetric_noise`  -- corrupt a fraction of the training labels
* :func:`make_class_imbalanced`   -- exponential class imbalance (Cao et al.,
  2019) with an imbalanced ratio of 0.01 in the paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

DEFAULT_DATA_ROOT = os.environ.get("LBCS_DATA_ROOT",
                                   os.path.join(os.path.expanduser("~"),
                                                ".lbcs_data"))

NUM_CLASSES = {
    "mnist": 10,
    "mnist-s": 10,
    "fmnist": 10,
    "svhn": 10,
    "cifar10": 10,
}


@dataclass
class DatasetBundle:
    """A train/test split kept entirely in memory."""

    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_classes: int

    @property
    def n(self) -> int:
        return int(self.train_x.size(0))

    def train_dataset(self) -> TensorDataset:
        return TensorDataset(self.train_x, self.train_y)

    def test_dataset(self) -> TensorDataset:
        return TensorDataset(self.test_x, self.test_y)

    def subset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the (images, labels) of the given training indices."""
        return self.train_x[indices], self.train_y[indices]


def _tensorise(dataset, flatten: bool = False):
    """Stack a torchvision dataset into dense ``float32`` ``NCHW`` tensors.

    ``torchvision.transforms.ToTensor`` is applied to every example, which
    both reorders the axes to ``(channels, height, width)`` -- CIFAR-10 and
    SVHN hand out ``HWC`` arrays -- and scales the pixel values to ``[0, 1]``
    so that the normalisation layers of the networks see the expected range.
    """
    xs, ys = [], []
    for x, y in dataset:
        if not torch.is_tensor(x):
            x = np.array(x)
            if x.ndim == 3 and x.shape[-1] in (1, 3, 4):    # HWC -> CHW
                x = x.transpose(2, 0, 1)
            x = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
            if x.max() > 1.0:
                x = x / 255.0
        if x.dim() == 2:                                    # H, W -> 1, H, W
            x = x.unsqueeze(0)
        xs.append(x)
        ys.append(int(y))
    x = torch.stack(xs)
    if flatten:
        x = x.view(x.size(0), -1)
    return x, torch.tensor(ys, dtype=torch.long)


def load_dataset(name: str, root: str = DEFAULT_DATA_ROOT,
                 mnist_s_size: int = 1000, mnist_s_seed: int = 0,
                 download: bool = True) -> DatasetBundle:
    """Load one of the paper's datasets."""
    name = name.lower()
    if name not in NUM_CLASSES:
        raise KeyError(f"unknown dataset '{name}', available: "
                       f"{sorted(NUM_CLASSES)}")

    if name in ("mnist", "mnist-s"):
        from torchvision.datasets import MNIST
        train = MNIST(root, train=True, download=download)
        test = MNIST(root, train=False, download=download)
        train_x, train_y = _tensorise(train)
        test_x, test_y = _tensorise(test)
        if name == "mnist-s":
            g = torch.Generator().manual_seed(mnist_s_seed)
            idx = torch.randperm(train_x.size(0), generator=g)[:mnist_s_size]
            train_x, train_y = train_x[idx], train_y[idx]
    elif name == "fmnist":
        from torchvision.datasets import FashionMNIST
        train = FashionMNIST(root, train=True, download=download)
        test = FashionMNIST(root, train=False, download=download)
        train_x, train_y = _tensorise(train)
        test_x, test_y = _tensorise(test)
    elif name == "svhn":
        from torchvision.datasets import SVHN
        train = SVHN(root, split="train", download=download)
        test = SVHN(root, split="test", download=download)
        train_x, train_y = _tensorise(train)
        test_x, test_y = _tensorise(test)
    elif name == "cifar10":
        from torchvision.datasets import CIFAR10
        train = CIFAR10(root, train=True, download=download)
        test = CIFAR10(root, train=False, download=download)
        train_x, train_y = _tensorise(train)
        test_x, test_y = _tensorise(test)
    else:  # pragma: no cover - guarded above
        raise KeyError(name)

    return DatasetBundle(name=name, train_x=train_x, train_y=train_y,
                         test_x=test_x, test_y=test_y,
                         num_classes=NUM_CLASSES[name])


# ---------------------------------------------------------------------------
# Imperfect supervision (Section 5.3)
# ---------------------------------------------------------------------------
def inject_symmetric_noise(labels: torch.Tensor, noise_rate: float,
                           num_classes: int, seed: int = 0) -> torch.Tensor:
    """Flip a ``noise_rate`` fraction of the labels uniformly at random.

    This is the symmetric (uniform) label noise of e.g. Ma et al. (2020) used
    in Section 5.3 ("the labels of 30% training data are flipped").
    """
    if noise_rate <= 0:
        return labels.clone()
    g = torch.Generator().manual_seed(seed)
    noisy = labels.clone()
    n = labels.numel()
    num_noisy = int(round(noise_rate * n))
    noisy_idx = torch.randperm(n, generator=g)[:num_noisy]
    random_labels = torch.randint(0, num_classes, (num_noisy,), generator=g)
    # make sure the label actually changes
    same = random_labels == labels[noisy_idx]
    while same.any():
        random_labels[same] = torch.randint(0, num_classes,
                                            (int(same.sum()),), generator=g)
        same = random_labels == labels[noisy_idx]
    noisy[noisy_idx] = random_labels
    return noisy


def make_class_imbalanced(x: torch.Tensor, y: torch.Tensor, num_classes: int,
                          imbalance_ratio: float = 0.01,
                          seed: int = 0):
    """Exponential class imbalance (Cao et al., 2019; Xu et al., 2021).

    The number of samples of class ``c`` is proportional to
    ``imbalance_ratio ** (c / (num_classes - 1))``, so the most frequent class
    has ``1 / imbalance_ratio`` times as many examples as the rarest one.
    The imbalance is injected in the training set only.
    """
    g = torch.Generator().manual_seed(seed)
    counts = np.array([(y == c).sum().item() for c in range(num_classes)])
    max_count = counts.max()
    num_per_class = []
    for c in range(num_classes):
        frac = imbalance_ratio ** (c / max(num_classes - 1, 1))
        num_per_class.append(max(1, int(round(max_count * frac))))
    keep = []
    for c in range(num_classes):
        idx_c = torch.nonzero(y == c, as_tuple=False).flatten()
        perm = idx_c[torch.randperm(idx_c.numel(), generator=g)]
        keep.append(perm[:min(num_per_class[c], perm.numel())])
    keep_idx = torch.cat(keep)
    keep_idx = keep_idx[torch.randperm(keep_idx.numel(), generator=g)]
    return x[keep_idx], y[keep_idx]


def make_imbalanced_bundle(bundle: DatasetBundle, imbalance_ratio: float = 0.01,
                           seed: int = 0) -> DatasetBundle:
    x, y = make_class_imbalanced(bundle.train_x, bundle.train_y,
                                 bundle.num_classes, imbalance_ratio, seed)
    return DatasetBundle(bundle.name + "-imb", x, y, bundle.test_x,
                         bundle.test_y, bundle.num_classes)


def make_noisy_bundle(bundle: DatasetBundle, noise_rate: float = 0.3,
                      seed: int = 0) -> DatasetBundle:
    y = inject_symmetric_noise(bundle.train_y, noise_rate, bundle.num_classes,
                               seed)
    return DatasetBundle(bundle.name + f"-noisy{int(noise_rate * 100)}",
                         bundle.train_x, y, bundle.test_x, bundle.test_y,
                         bundle.num_classes)
