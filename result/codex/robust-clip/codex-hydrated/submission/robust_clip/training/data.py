"""ImageNet data pipeline used for the adversarial fine-tuning.

The paper fine-tunes on ImageNet for two epochs at resolution ``224 x 224``
(Sec. 4 / App. B.1).  The addendum asks for the HuggingFace copy of ImageNet::

    from datasets import load_dataset
    dataset = load_dataset("imagenet-1k", trust_remote_code=True)

which is what :func:`build_imagenet_dataset` uses by default; a plain
``ImageFolder`` layout is supported as a fallback.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.common import LOGGER


#: Prompt used to turn the 1000 ImageNet class names into text embeddings for
#: the TeCoA baseline (`"A photo of a <class>."`, Radford et al., 2021).
IMAGENET_PROMPT = "A photo of a {}."


class HuggingFaceImageNet(Dataset):
    """Thin wrapper that turns an ``image``/``label`` HF dataset into a Dataset."""

    def __init__(self, hf_dataset, transform: Optional[Callable] = None, label_key: str = "label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(item[self.label_key])


class ClassFolder(Dataset):
    """``root/<class-name>/<image>`` layout (a labelled ``ImageFolder`` clone).

    Returns both the image tensor and the *class name*, which lets us build the
    ImageNet text embeddings directly from the folder structure.
    """

    IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        self.classes_file = os.path.join(root, "classes.txt")
        dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.classes = dirs
        if os.path.exists(self.classes_file):
            # CLIP_benchmark style: the class names are listed (and ordered) in
            # ``classes.txt``.  When the sub-directories carry the same names
            # the file only fixes the *order* (and the prompt text).
            with open(self.classes_file, "r", encoding="utf-8") as handle:
                listed = [line.strip() for line in handle if line.strip()]
            if len(listed) == len(dirs) and set(listed) == set(dirs):
                self.classes = listed
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []
        for cls in dirs:
            cls_dir = os.path.join(root, cls)
            label = self.class_to_idx.get(cls, dirs.index(cls))
            for name in sorted(os.listdir(cls_dir)):
                if name.lower().endswith(self.IMG_EXT):
                    self.samples.append((os.path.join(cls_dir, name), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_imagenet_dataset(
    transform: Callable,
    root: Optional[str] = None,
    split: str = "train",
    use_hf: Optional[bool] = None,
    streaming: bool = False,
):
    """Return a dataset yielding ``(pixel_tensor, label)`` pairs."""
    if root is None:
        use_hf = True if use_hf is None else use_hf
    elif use_hf is None:
        use_hf = not os.path.isdir(root)

    if use_hf:
        from datasets import load_dataset

        LOGGER.info("loading ImageNet ('%s') from the HuggingFace hub", split)
        path = root if root else "imagenet-1k"
        dataset = load_dataset(path, split=split, trust_remote_code=True)
        return HuggingFaceImageNet(dataset, transform)

    LOGGER.info("loading ImageNet from the image-folder layout at %s", root)
    return ClassFolder(root, transform)


def imagenet_class_names(dataset) -> Optional[List[str]]:
    """Best-effort extraction of the 1000 class names (wnid order)."""
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if isinstance(dataset, ClassFolder):
        return list(dataset.classes)
    base = getattr(dataset, "dataset", dataset)
    while isinstance(base, torch.utils.data.Subset):
        base = base.dataset
    if hasattr(base, "classes"):
        return list(base.classes)
    features = getattr(base, "features", None)
    if features is not None and "label" in features:
        names = getattr(features["label"], "names", None)
        if names:
            return list(names)
    return None


def load_class_names_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


@torch.no_grad()
def imagenet_text_embeddings(
    encoder,
    tokenizer,
    class_names: Sequence[str],
    template: str = IMAGENET_PROMPT,
    batch_size: int = 256,
    normalize: bool = True,
    device=None,
) -> torch.Tensor:
    """Encode ``template.format(class)`` for every ImageNet class.

    These are the fixed text embeddings ``psi(t_k)`` that TeCoA trains against.
    """
    device = device or next(encoder.parameters()).device
    embeddings = []
    for start in range(0, len(class_names), batch_size):
        chunk = class_names[start:start + batch_size]
        tokens = tokenizer([template.format(c) for c in chunk]).to(device)
        emb = encoder.encode_text(tokens)
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.float().cpu())
    return torch.cat(embeddings, dim=0)


def build_train_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8,
    drop_last: bool = True,
    seed: int = 0,
    sampler=None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
