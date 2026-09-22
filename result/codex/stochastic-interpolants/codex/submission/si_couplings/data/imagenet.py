"""ImageNet loaders (the target density rho_1 in all experiments).

The paper reports results on ImageNet-256x256 and ImageNet-512x512 for
in-painting, and on the 64x64 -> 256x256 (and 256x256 -> 512x512)
super-resolution tasks.  Images are mapped to [-1, 1], which is the range
used by the DDPM U-Net of Appendix B.

As suggested in the addendum, the dataset is loaded through HuggingFace
``datasets``:

    from datasets import load_dataset
    dataset = load_dataset("imagenet-1k", trust_remote_code=True)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_transform(resolution: int, random_flip: bool = True, random_crop: bool = True):
    """Torchvision transform mapping an ImageNet image to a [-1, 1] tensor."""
    import torchvision.transforms as T

    ops = []
    if random_crop:
        # random-resized-crop to the requested resolution (training)
        ops.append(T.RandomResizedCrop(resolution, scale=(0.8, 1.0), ratio=(0.75, 1.333)))
        if random_flip:
            ops.append(T.RandomHorizontalFlip())
    else:
        ops.append(T.Resize(resolution))
        ops.append(T.CenterCrop(resolution))
    ops.append(T.ToTensor())
    ops.append(T.Lambda(lambda x: x * 2.0 - 1.0))  # [0, 1] -> [-1, 1]
    return T.Compose(ops)


class HFImageNet(Dataset):
    """Thin wrapper around a HuggingFace image dataset returning tensors."""

    def __init__(self, hf_dataset, resolution: int = 256, transform: Optional[Callable] = None,
                 train: bool = True):
        self.ds = hf_dataset
        self.resolution = resolution
        self.transform = transform or default_transform(resolution, random_flip=train, random_crop=train)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        example = self.ds[idx]
        image = example["image"]
        label = int(example.get("label", 0))
        if image.mode != "RGB":
            image = image.convert("RGB")
        x = self.transform(image)
        return x, label


def build_imagenet(
    resolution: int = 256,
    split: str = "train",
    name: str = "imagenet-1k",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    train: Optional[bool] = None,
):
    """Load ImageNet through HuggingFace ``datasets``.

    ``trust_remote_code=True`` avoids waiting on stdin (see addendum).
    """
    from datasets import load_dataset

    if train is None:
        train = split == "train"
    ds = load_dataset(name, split=split, cache_dir=cache_dir, streaming=streaming,
                      trust_remote_code=True)
    return HFImageNet(ds, resolution=resolution, train=train)


class SyntheticImages(Dataset):
    """Cheap surrogate for ImageNet used by the smoke tests.

    Samples smooth random images with a per-class colour offset, which is
    enough to check that the training loop, the couplings and the samplers
    all run end to end without downloading ImageNet.
    """

    def __init__(self, n: int = 1024, resolution: int = 32, num_classes: int = 4, seed: int = 0):
        self.n = n
        self.resolution = resolution
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed + idx)
        label = idx % self.num_classes
        low = torch.randn(3, 4, 4, generator=g)
        img = torch.nn.functional.interpolate(low[None], size=(self.resolution, self.resolution),
                                              mode="bicubic", align_corners=False)[0]
        img = img / (img.std() + 1e-6) * 0.3
        colour = torch.tensor([(label % 3) / 3.0, ((label + 1) % 3) / 3.0, ((label + 2) % 3) / 3.0])
        img = img + (colour[:, None, None] * 2 - 1) * 0.5
        return img.clamp(-1, 1), label


def build_dataloader(
    cfg: dict,
    *,
    train: bool = True,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """Build the data loader described by the ``data`` section of a config."""
    data_cfg = cfg.get("data", {})
    kind = data_cfg.get("kind", "imagenet")
    resolution = int(data_cfg.get("image_size", 256))
    batch_size = int(batch_size or cfg.get("optim", {}).get("batch_size", 32))
    num_workers = int(num_workers if num_workers is not None else data_cfg.get("num_workers", 4))

    if kind == "imagenet":
        dataset = build_imagenet(
            resolution=resolution,
            split=data_cfg.get("split", "train" if train else "validation"),
            name=data_cfg.get("name", "imagenet-1k"),
            cache_dir=data_cfg.get("cache_dir"),
            streaming=bool(data_cfg.get("streaming", False)),
            train=train,
        )
    elif kind == "synthetic":
        dataset = SyntheticImages(
            n=int(data_cfg.get("n", 1024)),
            resolution=resolution,
            num_classes=int(data_cfg.get("num_classes", 4)),
        )
    else:
        raise ValueError(f"unknown dataset kind {kind!r}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "default_transform",
    "HFImageNet",
    "build_imagenet",
    "SyntheticImages",
    "build_dataloader",
]
