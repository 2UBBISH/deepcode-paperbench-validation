"""Zero-shot classification and its adversarial evaluation (Sec. 4.3, Table 4).

Protocol of the paper (App. B.10):

* datasets: Caltech101, StanfordCars, CIFAR10, CIFAR100, DTD, EuroSAT, FGVC
  Aircrafts, Flowers, ImageNet-R, ImageNet-Sketch, PCAM, OxfordPets, STL10 and
  the ImageNet validation set,
* *"We evaluate robustness on 1000 samples each and report clean accuracy for all
  samples of the respective datasets."*
* *"We employ the first two attacks of AutoAttack, namely APGD with cross-entropy
  loss and APGD with targeted DLR loss (100 iterations each). As the DLR loss is
  only applicable for multi-class classification, we use only the first attack on
  the binary dataset PCAM."*
* *"We consider l_inf-bounded threat models with radii eps = 2/255 and 4/255 and
  evaluate robustness on all datasets at resolution 224x224, except for CIFAR10,
  CIFAR100 and STL-10, which we evaluate at their respective original
  resolution."*
* the average of Table 4 is computed *"only over the zero-shot datasets without
  ImageNet"*.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from ..attacks.apgd import APGDAttack
from ..data.registry import imagenet_classnames, openai_templates
from ..models.clip_encoder import CLIPEncoder, build_zero_shot_classifier

LOGGER = logging.getLogger(__name__)


def select_worst_case(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Combine two attacks into the worst case for the *attacker*.

    AutoAttack runs several attacks and keeps, for every sample, the adversarial
    example that is misclassified (preferring the larger cross-entropy loss when
    both succeed).  Selecting by the loss alone would allow a successful attack
    to be overwritten by a less successful one.
    """
    correct_a = logits_a.argmax(dim=1) == labels
    correct_b = logits_b.argmax(dim=1) == labels
    loss_a = F.cross_entropy(logits_a, labels, reduction="none")
    loss_b = F.cross_entropy(logits_b, labels, reduction="none")
    take_b = ((correct_a & ~correct_b) | ((correct_a == correct_b) & (loss_b > loss_a))).view(
        -1, *([1] * (x_a.dim() - 1))
    )
    return torch.where(take_b, x_b, x_a)


@dataclass
class ZeroShotDataset:
    """Description of one zero-shot classification dataset."""

    name: str
    hf_id: str
    split: str = "test"
    image_key: str = "image"
    label_key: str = "label"
    resolution: int = 224
    binary: bool = False
    classnames: Optional[Sequence[str]] = None
    templates: Optional[Sequence[str]] = None


#: The datasets of Sec. 4.3.  Class names are read from the dataset metadata
#: (``ClassLabel`` feature) unless they are given explicitly; ImageNet-R,
#: ImageNet-Sketch and ImageNet use the 1000 ImageNet class names.
ZERO_SHOT_DATASETS: Dict[str, ZeroShotDataset] = {
    "caltech101": ZeroShotDataset("caltech101", "Bingsu/caltech101"),
    "stanford_cars": ZeroShotDataset("stanford_cars", "tanganke/stanford_cars"),
    "cifar10": ZeroShotDataset("cifar10", "uoft-cs/cifar10", split="test", resolution=32),
    "cifar100": ZeroShotDataset(
        "cifar100", "uoft-cs/cifar100", split="test", label_key="fine_label", resolution=32
    ),
    "dtd": ZeroShotDataset("dtd", "tanganke/dtd"),
    "eurosat": ZeroShotDataset("eurosat", "tanganke/eurosat"),
    "fgvc_aircraft": ZeroShotDataset("fgvc_aircraft", "tanganke/fgvc_aircraft", split="test"),
    "flowers102": ZeroShotDataset("flowers102", "tanganke/flowers102"),
    "imagenet_r": ZeroShotDataset("imagenet_r", "axiong/imagenet-r"),
    "imagenet_sketch": ZeroShotDataset("imagenet_sketch", "imagenet_sketch", split="test"),
    "pcam": ZeroShotDataset("pcam", "1aurent/PatchCamelyon", split="test", binary=True),
    "oxford_pets": ZeroShotDataset("oxford_pets", "tanganke/oxford_iiit_pet", split="test"),
    "stl10": ZeroShotDataset("stl10", "tanganke/stl10", resolution=96),
    "imagenet": ZeroShotDataset("imagenet", "imagenet-1k", split="validation"),
}

#: datasets averaged in the last column of Table 4 (everything but ImageNet)
AVERAGED_DATASETS: List[str] = [name for name in ZERO_SHOT_DATASETS if name != "imagenet"]


def _classnames_from_dataset(dataset, spec: ZeroShotDataset) -> List[str]:
    if spec.classnames is not None:
        return list(spec.classnames)
    if spec.name == "imagenet":
        return imagenet_classnames()
    if spec.name in {"imagenet_r", "imagenet_sketch"}:
        # both use (a subset of) the ImageNet class names
        try:
            return [imagenet_classnames()[i] for i in range(len(dataset.features[spec.label_key].names))]
        except Exception:  # pragma: no cover - depends on the dataset release
            return imagenet_classnames()
    try:
        return list(dataset.features[spec.label_key].names)
    except Exception as error:  # pragma: no cover
        raise ValueError(
            f"no class names found for {spec.name}; pass them explicitly via `classnames`"
        ) from error


def load_zeroshot_dataset(
    name: str,
    root: Optional[str] = None,
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
):
    """Load one dataset of Sec. 4.3 (Hugging Face hub or a local directory)."""
    if name not in ZERO_SHOT_DATASETS:
        raise KeyError(f"unknown dataset {name}; available: {sorted(ZERO_SHOT_DATASETS)}")
    spec = ZERO_SHOT_DATASETS[name]
    if root is not None and os.path.isdir(os.path.join(root, name)):
        from torchvision.datasets import ImageFolder

        dataset = ImageFolder(os.path.join(root, name))
        classnames = [c.replace("_", " ") for c in dataset.classes]
        return dataset, classnames, spec

    from datasets import load_dataset

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        dataset = load_dataset(spec.hf_id, split=spec.split, token=hf_token, trust_remote_code=True)
    except Exception as error:  # pragma: no cover - network / gated datasets
        raise RuntimeError(f"could not load {spec.hf_id} ({spec.split}): {error}") from error

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    classnames = _classnames_from_dataset(dataset, spec)
    return dataset, classnames, spec


class ZeroShotEvaluator:
    """Clean and robust (AutoAttack) zero-shot accuracy of one CLIP model."""

    def __init__(
        self,
        clip: CLIPEncoder,
        batch_size: int = 64,
        n_robust_samples: int = 1000,
        n_iter: int = 100,
        device: Optional[str] = None,
    ):
        self.clip = clip
        self.batch_size = batch_size
        self.n_robust_samples = n_robust_samples
        self.n_iter = n_iter
        self.device = torch.device(device or clip.device)
        self._classifier_cache: Dict[Tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _classifier(self, classnames: Sequence[str], templates: Sequence[str]) -> torch.Tensor:
        key = (tuple(classnames), tuple(templates))
        if key not in self._classifier_cache:
            self._classifier_cache.clear()  # datasets differ in size, keep memory low
            self._classifier_cache[key] = build_zero_shot_classifier(self.clip, classnames, templates)
        return self._classifier_cache[key]

    def _images_to_tensor(self, dataset, indices: Sequence[int], resolution: int) -> torch.Tensor:
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )
        images = []
        for index in indices:
            sample = dataset[index]
            image = sample["image"] if isinstance(sample, dict) else sample[0]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            images.append(transform(image.convert("RGB")))
        return torch.stack(images)

    def _labels(self, dataset, indices: Sequence[int], spec: ZeroShotDataset) -> torch.Tensor:
        labels = []
        for index in indices:
            sample = dataset[index]
            labels.append(int(sample[spec.label_key]))
        return torch.tensor(labels)

    def _logits(self, images: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
        features = self.clip.image_embedding(images, normalize=True)
        logit_scale = self.clip.model.logit_scale.exp()
        return logit_scale * features @ classifier.t()

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def clean_accuracy(self, dataset, classnames, spec: ZeroShotDataset, max_samples: Optional[int] = None) -> float:
        classifier = self._classifier(classnames, spec.templates or openai_templates())
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        correct = 0
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                indices = list(range(start, min(start + self.batch_size, n)))
                images = self._images_to_tensor(dataset, indices, spec.resolution)
                labels = self._labels(dataset, indices, spec)
                logits = self._logits(images, classifier)
                correct += int((logits.argmax(dim=1).cpu() == labels).sum())
        return 100.0 * correct / max(n, 1)

    def robust_accuracy(
        self,
        dataset,
        classnames,
        spec: ZeroShotDataset,
        eps: Union[str, float] = "2/255",
        n_samples: Optional[int] = None,
        use_dlr: Optional[bool] = None,
    ) -> float:
        """Robust accuracy under APGD-CE (+ APGD-DLR for multi-class datasets)."""
        classifier = self._classifier(classnames, spec.templates or openai_templates())
        n = len(dataset) if n_samples is None else min(n_samples, len(dataset))
        use_dlr = (not spec.binary) if use_dlr is None else use_dlr
        correct = 0
        for start in range(0, n, self.batch_size):
            indices = list(range(start, min(start + self.batch_size, n)))
            images = self._images_to_tensor(dataset, indices, spec.resolution)
            labels = self._labels(dataset, indices, spec)

            def forward_fn(x_adv):
                return self._logits(x_adv, classifier)

            attack_ce = APGDAttack(eps=eps, n_iter=self.n_iter, loss="ce", dtype=torch.float32)
            x_ce = attack_ce.attack(images, forward_fn, labels=labels)
            x_best = x_ce
            if use_dlr:
                attack_dlr = APGDAttack(eps=eps, n_iter=self.n_iter, loss="dlr_targeted", dtype=torch.float32)
                x_dlr = attack_dlr.attack_targeted(images, forward_fn, labels=labels, n_target_classes=9)
                with torch.no_grad():
                    logits_ce = forward_fn(x_ce)
                    logits_dlr = forward_fn(x_dlr)
                x_best = select_worst_case(x_ce, x_dlr, logits_ce, logits_dlr, labels)

            with torch.no_grad():
                predictions = forward_fn(x_best).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
        return 100.0 * correct / max(n, 1)

    def evaluate(
        self,
        names: Sequence[str] = tuple(AVERAGED_DATASETS),
        radii: Sequence[Union[str, float]] = ("2/255", "4/255"),
        root: Optional[str] = None,
        max_clean_samples: Optional[int] = None,
        hf_token: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate clean + robust accuracy of all datasets (Table 4)."""
        results: Dict[str, Dict[str, float]] = {}
        for name in names:
            dataset, classnames, spec = load_zeroshot_dataset(name, root=root, hf_token=hf_token)
            row: Dict[str, float] = {
                "clean": self.clean_accuracy(dataset, classnames, spec, max_samples=max_clean_samples)
            }
            for eps in radii:
                key = f"robust_{eps}" if isinstance(eps, str) else f"robust_{int(round(float(eps) * 255))}/255"
                row[key] = self.robust_accuracy(
                    dataset, classnames, spec, eps=eps, n_samples=self.n_robust_samples
                )
            results[name] = row
            LOGGER.info("%s: %s", name, row)

        averaged = [results[name] for name in names if name in AVERAGED_DATASETS]
        if averaged:
            results["average_zero_shot"] = {
                key: sum(row[key] for row in averaged) / len(averaged) for key in averaged[0]
            }
        return results


def evaluate_clip_autoattack(
    clip: CLIPEncoder,
    names: Sequence[str] = tuple(AVERAGED_DATASETS),
    radii: Sequence[Union[str, float]] = ("2/255", "4/255"),
    root: Optional[str] = None,
    batch_size: int = 64,
    n_robust_samples: int = 1000,
    n_iter: int = 100,
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper: AutoAttack based evaluation of one CLIP encoder."""
    evaluator = ZeroShotEvaluator(clip, batch_size=batch_size, n_robust_samples=n_robust_samples, n_iter=n_iter)
    return evaluator.evaluate(names=names, radii=radii, root=root, **kwargs)
