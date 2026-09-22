"""Zero-shot classification datasets and prompt templates (Sec. 4.3, App. B.10).

The evaluation protocol follows ``CLIP_benchmark`` (the source used by the
paper), i.e. class names are combined with a dataset-specific set of prompt
templates, the text embeddings of all templates are averaged per class and the
class with the highest cosine similarity to the image embedding wins.

* 14 datasets are evaluated: ImageNet plus 13 zero-shot datasets.
* Robustness is evaluated on 1000 samples per dataset, clean accuracy on all
  available samples.
* Everything is evaluated at ``224 x 224`` except CIFAR10, CIFAR100 and STL-10,
  which keep their native resolution.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..utils.common import LOGGER


# --------------------------------------------------------------------------- #
#                              prompt templates                                #
# --------------------------------------------------------------------------- #
#: The 80 ImageNet templates of Radford et al. (2021); also used for
#: ImageNet-R and ImageNet-Sketch.
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

SIMPLE_TEMPLATES = ["a photo of a {}."]

PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "imagenet": IMAGENET_TEMPLATES,
    "imagenet-r": IMAGENET_TEMPLATES,
    "imagenet-sketch": IMAGENET_TEMPLATES,
    "caltech101": ["a photo of a {}."],
    "stanford_cars": ["a photo of a {}."],
    "cifar10": ["a photo of a {}."],
    "cifar100": ["a photo of a {}."],
    "dtd": ["a photo of a {}."],
    "eurosat": ["a centered satellite photo of {}.", "a centered satellite photo of a {}."],
    "fgvc_aircraft": ["a photo of a {}, a type of aircraft."],
    "flowers102": ["a photo of a {}, a type of flower."],
    "oxford_pets": ["a photo of a {}, a type of pet."],
    "pcam": ["a photo of a {}."],
    "stl10": ["a photo of a {}."],
}


@dataclasses.dataclass
class ZeroShotSpec:
    name: str
    hf_path: Optional[str]
    hf_split: str = "test"
    image_key: str = "image"
    label_key: str = "label"
    resolution: int = 224
    #: the dataset's own resolution; used when ``--native-resolution`` is set
    native_resolution: Optional[int] = None
    #: directory name inside a CLIP_benchmark style data root, if available
    folder: Optional[str] = None

    @property
    def templates(self) -> List[str]:
        return PROMPT_TEMPLATES.get(self.name, SIMPLE_TEMPLATES)


ZERO_SHOT_DATASETS: Dict[str, ZeroShotSpec] = {
    "imagenet": ZeroShotSpec("imagenet", "imagenet-1k", "validation", resolution=224),
    "caltech101": ZeroShotSpec("caltech101", "tanganke/caltech101", "test", folder="caltech101"),
    "stanford_cars": ZeroShotSpec("stanford_cars", "tanganke/stanford_cars", "test", folder="stanford_cars"),
    "cifar10": ZeroShotSpec(
        "cifar10", "cifar10", "test", resolution=224, native_resolution=32, folder="cifar10"
    ),
    "cifar100": ZeroShotSpec(
        "cifar100", "cifar100", "test", resolution=224, native_resolution=32, folder="cifar100"
    ),
    "dtd": ZeroShotSpec("dtd", "tanganke/dtd", "test", folder="dtd"),
    "eurosat": ZeroShotSpec("eurosat", "tanganke/eurosat", "test", folder="eurosat"),
    "fgvc_aircraft": ZeroShotSpec("fgvc_aircraft", "tanganke/fgvc_aircraft", "test", folder="fgvc_aircraft"),
    "flowers102": ZeroShotSpec("flowers102", "tanganke/oxford_flowers102", "test", folder="flowers102"),
    "imagenet-r": ZeroShotSpec("imagenet-r", "tanganke/imagenet-r", "test", folder="imagenet-r"),
    "imagenet-sketch": ZeroShotSpec(
        "imagenet-sketch", "tanganke/imagenet_sketch", "test", folder="imagenet-sketch"
    ),
    "oxford_pets": ZeroShotSpec("oxford_pets", "tanganke/oxford_iiit_pet", "test", folder="oxford_pets"),
    "pcam": ZeroShotSpec("pcam", "1aurent/PatchCamelyon", "test", folder="pcam"),
    "stl10": ZeroShotSpec(
        "stl10", "tanganke/stl10", "test", resolution=224, native_resolution=96, folder="stl10"
    ),
}


def transform_resolution(spec: ZeroShotSpec, native_resolution: bool = False) -> int:
    """Resolution the images are resized to before they enter the model.

    App. B.10 says that everything is evaluated at 224x224 "except for CIFAR10,
    CIFAR100 and STL-10, which we evaluate at their respective original
    resolution".  Feeding a 32x32 image to a patch-14 ViT destroys the accuracy
    of CLIP itself (50% on CIFAR-10 instead of 90%), which is incompatible with
    the clean accuracies reported in Table 4, so the default here is to resize
    every dataset to 224 (what CLIP_benchmark does).  ``--native-resolution``
    switches to the literal reading; the positional embedding is interpolated
    accordingly (:meth:`robust_clip.models.CLIPImageEncoder.interpolate_pos_embed_if_needed`).
    """
    if native_resolution and spec.native_resolution is not None:
        return spec.native_resolution
    return spec.resolution


class HFDatasetWithTransform(Dataset):
    def __init__(self, hf_dataset, transform: Callable, image_key: str = "image", label_key: str = "label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item[self.image_key]
        if hasattr(image, "mode") and image.mode != "RGB":
            image = image.convert("RGB")
        label = item[self.label_key]
        if isinstance(label, str):
            label = int(label)
        return self.transform(image), int(label)


class CIFARDataset(Dataset):
    """The ``cifar10``/``cifar100`` HF datasets store images as numpy arrays."""

    def __init__(self, hf_dataset, transform: Callable):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from PIL import Image

        item = self.dataset[index]
        image = item["img"]
        if isinstance(image, dict) and "bytes" in image:       # HF ``Image`` feature
            import io

            image = Image.open(io.BytesIO(image["bytes"]))
        elif not hasattr(image, "mode"):                        # raw numpy array
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(item["fine_label"] if "fine_label" in item else item["label"])


def class_names_of(dataset) -> Optional[List[str]]:
    # unwrap ``Subset`` wrappers that the callers add for quick evaluations
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    names_file = getattr(dataset, "classes_file", None)
    if names_file and os.path.exists(names_file):
        with open(names_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    base = getattr(dataset, "dataset", dataset)
    while isinstance(base, torch.utils.data.Subset):
        base = base.dataset
    if hasattr(base, "classes"):
        return list(base.classes)
    features = getattr(base, "features", None)
    if features is None:
        return None
    for key in ("label", "fine_label"):
        if key in features:
            names = getattr(features[key], "names", None)
            if names:
                return list(names)
    return None


def load_zero_shot_dataset(
    spec: ZeroShotSpec,
    transform: Callable,
    root: Optional[str] = None,
    local_files_only: bool = False,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load one zero-shot dataset.

    ``root`` may point to a ``CLIP_benchmark`` style data root
    (``<root>/<dataset>/`` with ``classes.txt`` or ImageFolder layout); if it is
    unset, the HuggingFace hub is used.
    """
    local_dir = None
    folder = spec.folder or spec.name
    if root is not None:
        candidate = os.path.join(root, folder)
        if os.path.isdir(candidate):
            local_dir = candidate

    if local_dir is not None:
        from ..training.data import ClassFolder

        split_dir = os.path.join(local_dir, spec.hf_split)
        dataset = ClassFolder(split_dir if os.path.isdir(split_dir) else local_dir, transform)
    else:
        from datasets import load_dataset

        LOGGER.info("loading zero-shot dataset '%s' (%s)", spec.name, spec.hf_path)
        if local_files_only:
            # ``load_dataset`` forwards unknown kwargs to the builder config, so
            # the offline switch has to go through the environment.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        hf = load_dataset(spec.hf_path, split=spec.hf_split, trust_remote_code=True)
        if spec.hf_path in ("cifar10", "cifar100"):
            dataset = CIFARDataset(hf, transform)
        else:
            dataset = HFDatasetWithTransform(hf, transform, spec.image_key, spec.label_key)

    if max_samples is not None and len(dataset) > max_samples:
        dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))
    return dataset


def build_zero_shot_transform(image_size: int):
    """Resize shortest edge -> ``image_size``, centre crop, return pixel space."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


@torch.no_grad()
def build_text_embeddings(
    encoder,
    tokenizer,
    class_names: Sequence[str],
    templates: Sequence[str],
    batch_size: int = 512,
    normalize: bool = True,
    device=None,
) -> torch.Tensor:
    """Average the text embeddings of all templates for every class."""
    device = device or next(encoder.parameters()).device
    prompts = [template.format(name) for name in class_names for template in templates]
    embeddings = []
    for start in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[start:start + batch_size]).to(device)
        emb = encoder.encode_text(tokens)
        embeddings.append(emb.float())
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.view(len(class_names), len(templates), -1).mean(dim=1)
    if normalize:
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu()
