"""Datasets used by the paper's few-shot image-generation experiments.

Section 5.2: "Following (Ojha et al., 2021), we use FFHQ (Karras et al.,
2020b) and LSUN Church (Yu et al., 2015) as source datasets.  For the target
datasets, we employ 10-shot Babies, Sunglasses, Raphael Peale, Sketches, and
face paintings by Amedeo Modigliani, which correspond to the source domain
FFHQ.  Additionally, we utilize 10-shot Haunted Houses and Landscape drawings
as target datasets corresponding to the LSUN Church source domain."

Only *paths* are hard-coded here: the datasets themselves are third-party
artefacts that are not redistributable.  ``scripts/prepare_datasets.py`` prints
exactly which folders are expected and validates them.

The toy experiment of Section 5.1 uses two 2-D Gaussians (source mean (1,1),
target mean (-1,-1), identity variance) and is fully synthetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class DatasetSpec:
    name: str
    source: str
    kind: str  # "source" | "target"
    origin: str
    shots: Optional[int] = None


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    # ---- source domains -------------------------------------------------
    "ffhq": DatasetSpec(
        name="ffhq",
        source="ffhq",
        kind="source",
        origin=(
            "FFHQ (Karras et al., 2020b), https://github.com/NVlabs/ffhq-dataset . "
            "Point --source-root at a directory of 256x256 FFHQ images (the "
            "ffhq256 subset used by guided-diffusion is also supported)."
        ),
    ),
    "lsun_church": DatasetSpec(
        name="lsun_church",
        source="lsun_church",
        kind="source",
        origin=(
            "LSUN Church outdoor (Yu et al., 2015), https://www.yf.io/p/lsun . "
            "Either a folder of images or a torchvision LSUN 'church_outdoor' "
            "root."
        ),
    ),
    # ---- 10-shot targets (FFHQ -> ...) ---------------------------------
    "babies": DatasetSpec("babies", "ffhq", "target", "10-shot Babies (Ojha et al., 2021 / CDC)", 10),
    "sunglasses": DatasetSpec("sunglasses", "ffhq", "target", "10-shot Sunglasses (Ojha et al., 2021 / CDC)", 10),
    "raphael": DatasetSpec(
        "raphael", "ffhq", "target", "10-shot Raphael Peale paintings (Zhao et al., 2022 / DCL)", 10
    ),
    "sketches": DatasetSpec("sketches", "ffhq", "target", "10-shot Sketches (Ojha et al., 2021 / CDC)", 10),
    "amedeo": DatasetSpec(
        "amedeo", "ffhq", "target", "10-shot Amedeo Modigliani paintings (Zhao et al., 2022 / DCL)", 10
    ),
    # ---- 10-shot targets (LSUN Church -> ...) --------------------------
    "haunted_houses": DatasetSpec(
        "haunted_houses", "lsun_church", "target", "10-shot Haunted houses (Ojha et al., 2021 / CDC)", 10
    ),
    "landscape_drawings": DatasetSpec(
        "landscape_drawings",
        "lsun_church",
        "target",
        "10-shot Landscape drawings (Ojha et al., 2021 / CDC)",
        10,
    ),
}


def list_images(root: str) -> List[str]:
    paths: List[str] = []
    for directory, _, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(directory, name))
    return sorted(paths)


def _default_transform(image_size: int) -> Callable:
    from PIL import Image

    def transform(image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        # resize the short side then centre-crop (the guided-diffusion recipe)
        width, height = image.size
        scale = image_size / min(width, height)
        resized = image.resize(
            (max(image_size, int(round(width * scale))), max(image_size, int(round(height * scale)))),
            resample=Image.BICUBIC,
        )
        width, height = resized.size
        left = (width - image_size) // 2
        top = (height - image_size) // 2
        resized = resized.crop((left, top, left + image_size, top + image_size))
        array = np.asarray(resized).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    return transform


class ImageTensorDataset(Dataset):
    """Folder of images -> ``[-1, 1]`` float tensors of shape ``[3, S, S]``."""

    def __init__(self, paths: Sequence[str], image_size: int = 256, limit: Optional[int] = None):
        self.paths = list(paths)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError("no images found")
        self.image_size = image_size
        self.transform = _default_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        from PIL import Image

        with Image.open(self.paths[index]) as handle:
            handle.load()
            return self.transform(handle)


class FewShotDataset(ImageTensorDataset):
    """A ``k``-shot target set (``k = 10`` in the paper)."""

    def __init__(self, root: str, image_size: int = 256, num_shots: int = 10, seed: int = 0):
        paths = list_images(root)
        if len(paths) > num_shots:
            rng = np.random.RandomState(seed)
            paths = list(rng.choice(sorted(paths), size=num_shots, replace=False))
        super().__init__(paths, image_size=image_size)


def target_dataset(
    name: str, root: str, image_size: int = 256, num_shots: int = 10, seed: int = 0
) -> FewShotDataset:
    """Load a 10-shot target dataset by name (``root`` = the dataset directory)."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASET_REGISTRY)}")
    spec = DATASET_REGISTRY[name]
    if spec.kind != "target":
        raise ValueError(f"{name!r} is a source dataset")
    return FewShotDataset(root, image_size=image_size, num_shots=num_shots, seed=seed)


def infinite_loader(
    dataset: Dataset, batch_size: int, num_workers: int = 0, shuffle: bool = True, drop_last: bool = False
):
    """Yield batches forever (the transfer learning loop is iteration-based)."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )
    while True:
        for batch in loader:
            yield batch


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def dataset_tensor(dataset: Dataset, index: int) -> torch.Tensor:
    """Fetch ``dataset[index]`` and unwrap ``TensorDataset``-style tuples."""
    item = dataset[index]
    if isinstance(item, (tuple, list)):
        item = item[0]
    return item


# ---------------------------------------------------------------------- #
# toy data (Section 5.1)
# ---------------------------------------------------------------------- #
def toy_gaussian_samples(
    num_samples: int, mean: Sequence[float] = (1.0, 1.0), seed: Optional[int] = None
) -> torch.Tensor:
    """``N(mean, I)`` samples of dimension 2 (Section 5.1)."""
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    return mean_tensor + torch.randn(num_samples, 2, generator=generator)


def toy_source_target(
    num_target_samples: int = 10,
    num_source_samples: int = 10000,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Source ``N((1,1), I)`` and target ``N((-1,-1), I)`` toy distributions."""
    source = toy_gaussian_samples(num_source_samples, mean=(1.0, 1.0), seed=seed)
    target = toy_gaussian_samples(num_target_samples, mean=(-1.0, -1.0), seed=seed + 1)
    return source, target


class LabelledImageDataset(Dataset):
    """Wrap two image datasets into a labelled (source=0, target=1) dataset.

    The domain classifier is trained on *noised* images across the diffusion
    timesteps, so the noising is done inside the training loop; here we only
    pair images with their domain label.
    """

    def __init__(self, source: Dataset, target: Dataset, source_weight: int = 1, target_weight: int = 1):
        self.samples: List[Tuple[Dataset, int, int]] = [
            (source, index, 0) for index in range(len(source))
        ]
        self.samples += [(target, index, 1) for index in range(len(target))]
        self.source_weight = source_weight
        self.target_weight = target_weight

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        dataset, item_index, label = self.samples[index]
        return dataset_tensor(dataset, item_index), label
