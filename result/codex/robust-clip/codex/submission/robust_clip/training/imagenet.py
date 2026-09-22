"""ImageNet-1k data pipeline used for the adversarial fine-tuning.

The robust models of the paper (TeCoA^2, FARE^2, TeCoA^4, FARE^4) are obtained by
fine-tuning the CLIP vision encoder on ImageNet-1k at resolution 224x224 for
**two epochs** (App. B.1).  The addendum specifies how the dataset is obtained:

.. code-block:: python

    from datasets import load_dataset
    dataset = load_dataset("imagenet-1k", trust_remote_code=True)

For flexibility this module also supports a plain ``ImageFolder`` directory and
pre-resized webdatasets are not needed: the standard CLIP preprocessing of
OpenCLIP (``RandomResizedCrop`` with scale ``(0.9, 1.0)`` for training and
resize + center crop for evaluation) is applied on the fly.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..data.registry import imagenet_classnames

LOGGER = logging.getLogger(__name__)


def build_transforms(
    resolution: int = 224,
    train: bool = True,
    normalize: bool = False,
    mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
    std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
) -> Callable:
    """CLIP image preprocessing (same defaults as OpenCLIP's CLIP transforms).

    ``normalize=False`` (the default here) keeps the images in ``[0, 1]``: the
    adversarial training and all attacks define the perturbation budget on the
    non-normalized pixels and the model applies the normalization internally,
    see :mod:`robust_clip.utils.transforms`.
    """
    mean = list(mean)
    std = list(std)
    normalization = [transforms.Normalize(mean=mean, std=std)] if normalize else []
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
            + normalization
        )
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
        + normalization
    )


class HFDatasetWrapper(Dataset):
    """Adapts a Hugging Face image dataset to ``(image_tensor, label)``."""

    def __init__(self, hf_dataset, transform: Callable, image_key: str = "image", label_key: str = "label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = sample[self.label_key]
        return image, int(label)


def load_imagenet(
    split: str = "train",
    data_dir: Optional[str] = None,
    dataset_name: str = "imagenet-1k",
    hf_token: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 0,
):
    """Load ImageNet-1k from the Hugging Face hub or from a local directory.

    ``data_dir`` may point to an ``ImageFolder``-style directory
    (``train/``, ``val/``); otherwise the Hugging Face ``imagenet-1k`` dataset is
    used as described in the addendum.
    """
    if data_dir is not None:
        from torchvision.datasets import ImageFolder

        return ImageFolder(os.path.join(data_dir, split))

    from datasets import load_dataset  # imported lazily, only needed for the HF path

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dataset = load_dataset(dataset_name, split=split, token=hf_token, trust_remote_code=True)
    if max_samples is not None and max_samples < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = dataset.select(indices)
    return dataset


def imagenet_class_texts(labels: Sequence[int], templates: Optional[Sequence[str]] = None) -> List[str]:
    """Class *texts* of a batch of ImageNet labels (used by the TeCoA loss)."""
    classnames = imagenet_classnames()
    templates = list(templates) if templates is not None else ["a photo of a {}."]
    texts = []
    for label in labels:
        name = classnames[int(label)]
        texts.append(templates[0].format(name))
    return texts


def build_imagenet_loaders(
    batch_size: int = 128,
    resolution: int = 224,
    num_workers: int = 8,
    data_dir: Optional[str] = None,
    dataset_name: str = "imagenet-1k",
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    normalize: bool = False,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
    mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
    std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
) -> Tuple[DataLoader, DataLoader]:
    """Create the ImageNet train / validation loaders."""
    train_transform = train_transform or build_transforms(resolution, train=True, normalize=normalize, mean=mean, std=std)
    eval_transform = eval_transform or build_transforms(resolution, train=False, normalize=normalize, mean=mean, std=std)

    train_set = load_imagenet(
        split="train",
        data_dir=data_dir,
        dataset_name=dataset_name,
        hf_token=hf_token,
        max_samples=max_train_samples,
    )
    val_set = load_imagenet(
        split="validation",
        data_dir=data_dir,
        dataset_name=dataset_name,
        hf_token=hf_token,
        max_samples=max_val_samples,
    )

    if not isinstance(train_set, Dataset) or not hasattr(train_set, "__getitem__"):
        raise TypeError("unexpected dataset type")

    # `ImageFolder` already returns tensors, HF datasets need the wrapper.
    if not hasattr(train_set, "features"):
        train_dataset = train_set
        if getattr(train_set, "transform", None) is None:
            train_dataset.transform = train_transform
        val_dataset = val_set
        if getattr(val_set, "transform", None) is None:
            val_dataset.transform = eval_transform
    else:
        train_dataset = HFDatasetWrapper(train_set, train_transform)
        val_dataset = HFDatasetWrapper(val_set, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
