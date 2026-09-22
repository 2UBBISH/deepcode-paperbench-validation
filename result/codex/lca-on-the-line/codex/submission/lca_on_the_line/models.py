"""The 75 pretrained models benchmarked in the paper (Appendix A).

36 vision-only models (VMs) are loaded through ``torchvision``; 39
vision-language models (VLMs) are loaded through ``openai/CLIP``, ``OpenCLIP``
and (for ALBEF/BLIP) LAVIS, as prescribed by the paper addendum.

Every model is wrapped in a small adapter exposing the same two operations:

* ``logits(images)``   -> ``(B, 1000)`` ImageNet class scores
* ``features(images)`` -> ``(B, D)`` representation from the last hidden layer
  before the classifier, i.e. the ``M(X)`` features mentioned in the addendum.

Zero-shot VLMs build their classifier from the ImageNet prompt ensemble so the
logits live in the same 0..999 class space as the supervised models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .prompt_engineering import IMAGENET_TEMPLATES, imagenet_classnames


# --------------------------------------------------------------------------- #
# text-feature caching (zero-shot classifiers need 1000 x n_templates prompts)
# --------------------------------------------------------------------------- #
DEFAULT_CACHE_DIR = os.environ.get(
    "LCA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "lca_on_the_line")
)


def _text_cache_path(cache_dir: str, key: str) -> str:
    import hashlib

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, "text_features", "%s_%s.npy" % (key[:40], digest))


def templates_key(templates: Sequence[str]) -> str:
    """Short, content-aware identifier for a prompt-template set."""
    import hashlib

    digest = hashlib.md5("\n".join(templates).encode("utf-8")).hexdigest()[:10]
    return "%d_%s" % (len(templates), digest)


def load_or_build_text_features(cache_dir, key, builder):
    """Cache the (1000, D) text-embedding matrix on disk."""
    if cache_dir is None:
        return builder()
    path = _text_cache_path(cache_dir, key)
    if os.path.exists(path):
        return torch.from_numpy(np.load(path))
    feats = builder()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, feats.detach().float().cpu().numpy())
    return feats

# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    name: str                 # unique identifier used in result tables
    family: str               # "VM" or "VLM"
    source: str               # "torchvision" | "clip" | "open_clip" | "lavis"
    arch: str                 # architecture / checkpoint key
    pretrained: Optional[str] = None  # weights tag / pretrained tag
    input_size: int = 224
    notes: str = ""

    @property
    def key(self) -> str:
        return self.name


# --- 36 vision-only models (torchvision, ImageNet supervised) -------------- #
_TORCHVISION_ARCHS: Sequence[Tuple[str, str]] = [
    ("alexnet", "image classification"),
    ("convnext_tiny", "convnext"),
    ("densenet121", "densenet"),
    ("densenet161", "densenet"),
    ("densenet169", "densenet"),
    ("densenet201", "densenet"),
    ("efficientnet_b0", "efficientnet"),
    ("googlenet", "googlenet"),
    ("inception_v3", "inception"),
    ("mnasnet0_5", "mnasnet"),
    ("mnasnet0_75", "mnasnet"),
    ("mnasnet1_0", "mnasnet"),
    ("mnasnet1_3", "mnasnet"),
    ("mobilenet_v3_small", "mobilenetv3"),
    ("mobilenet_v3_large", "mobilenetv3"),
    ("regnet_y_1_6gf", "regnet"),
    ("wide_resnet101_2", "wide_resnet"),
    ("resnet18", "resnet"),
    ("resnet34", "resnet"),
    ("resnet50", "resnet"),
    ("resnet101", "resnet"),
    ("resnet152", "resnet"),
    ("shufflenet_v2_x2_0", "shufflenet"),
    ("squeezenet1_0", "squeezenet"),
    ("squeezenet1_1", "squeezenet"),
    ("swin_b", "swin"),
    ("vgg11", "vgg"),
    ("vgg13", "vgg"),
    ("vgg16", "vgg"),
    ("vgg19", "vgg"),
    ("vgg11_bn", "vgg"),
    ("vgg13_bn", "vgg"),
    ("vgg16_bn", "vgg"),
    ("vgg19_bn", "vgg"),
    ("vit_b_32", "vit"),
    ("vit_l_32", "vit"),
]

_VIT_INPUT_SIZE = {"vit_b_32": 224, "vit_l_32": 224}
_INCEPTION_INPUT_SIZE = {"inception_v3": 299}


def vision_model_specs() -> List[ModelSpec]:
    specs = []
    for arch, _group in _TORCHVISION_ARCHS:
        size = _INCEPTION_INPUT_SIZE.get(arch, _VIT_INPUT_SIZE.get(arch, 224))
        specs.append(
            ModelSpec(
                name=arch,
                family="VM",
                source="torchvision",
                arch=arch,
                pretrained="IMAGENET1K_V1",
                input_size=size,
            )
        )
    return specs


# --- 7 CLIP models (openai/CLIP) ------------------------------------------- #
_CLIP_ARCHS: Sequence[str] = [
    "RN50",
    "RN101",
    "RN50x4",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
]


def clip_specs() -> List[ModelSpec]:
    specs = []
    for arch in _CLIP_ARCHS:
        size = 336 if arch.endswith("336px") else 224
        specs.append(
            ModelSpec(
                name="CLIP_" + arch.replace("/", "_").replace("@", "_"),
                family="VLM",
                source="clip",
                arch=arch,
                pretrained="openai",
                input_size=size,
            )
        )
    return specs


# --- 30 OpenCLIP models ----------------------------------------------------- #
_OPENCLIP_CONFIGS: Sequence[Tuple[str, str]] = [
    ("RN101", "openai"),
    ("RN101", "yfcc15m"),
    ("RN101-quickgelu", "openai"),
    ("RN101-quickgelu", "yfcc15m"),
    ("RN50", "cc12m"),
    ("RN50", "openai"),
    ("RN50", "yfcc15m"),
    ("RN50-quickgelu", "cc12m"),
    ("RN50-quickgelu", "openai"),
    ("RN50-quickgelu", "yfcc15m"),
    ("RN50x16", "openai"),
    ("RN50x4", "openai"),
    ("RN50x64", "openai"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("ViT-B-16", "laion400m_e31"),
    ("ViT-B-16", "laion400m_e32"),
    ("ViT-B-16-plus-240", "laion400m_e31"),
    ("ViT-B-16-plus-240", "laion400m_e32"),
    ("ViT-B-32", "laion2b_e16"),
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-32", "laion400m_e31"),
    ("ViT-B-32", "laion400m_e32"),
    ("ViT-B-32", "openai"),
    ("ViT-B-32-quickgelu", "laion400m_e31"),
    ("ViT-B-32-quickgelu", "laion400m_e32"),
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-L-14", "laion400m_e31"),
    ("ViT-L-14", "laion400m_e32"),
    ("coca_ViT-B-32", "laion2b_s13b_b90k"),
    ("coca_ViT-L-14", "laion2b_s13b_b90k"),
]


def openclip_specs() -> List[ModelSpec]:
    specs = []
    for arch, pretrained in _OPENCLIP_CONFIGS:
        specs.append(
            ModelSpec(
                name="OpenCLIP_%s_%s" % (arch, pretrained),
                family="VLM",
                source="open_clip",
                arch=arch,
                pretrained=pretrained,
                input_size=224 if "240" not in arch else 240,
            )
        )
    return specs


# --- 2 LAVIS feature extractors (ALBEF / BLIP) ------------------------------ #
_LAVIS_CONFIGS: Sequence[Tuple[str, str]] = [
    ("albef_feature_extractor", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/ALBEF/albef_feature_extractor.pth"),
    ("blip_feature_extractor_base", "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"),
]


def lavis_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            name=cfg,
            family="VLM",
            source="lavis",
            arch=cfg,
            pretrained=url,
            input_size=224,
            notes="Requires the `lavis` package (salesforce/LAVIS).",
        )
        for cfg, url in _LAVIS_CONFIGS
    ]


def all_model_specs(include_lavis: bool = True) -> List[ModelSpec]:
    """The full set of 75 models (36 VMs + 39 VLMs)."""
    specs = vision_model_specs() + clip_specs() + openclip_specs()
    if include_lavis:
        specs = specs + lavis_specs()
    return specs


def model_spec_by_name(name: str) -> ModelSpec:
    for spec in all_model_specs():
        if spec.name == name:
            return spec
    raise KeyError("unknown model %r" % name)


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #


class BaseClassifier:
    """Common interface: ``logits`` and ``features`` on a batch of PIL images."""

    def __init__(self, spec: ModelSpec, device: str = "cpu", batch_size: int = 64):
        self.spec = spec
        self.device = torch.device(device)
        self.batch_size = batch_size

    # --- to be implemented by subclasses -------------------------------- #
    def _transform(self):
        raise NotImplementedError

    def _logits_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _features_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # --- public API ------------------------------------------------------ #
    @torch.no_grad()
    def logits(self, images: Sequence) -> np.ndarray:
        return self._run(images, self._logits_tensor)

    @torch.no_grad()
    def features(self, images: Sequence) -> np.ndarray:
        return self._run(images, self._features_tensor)

    def _to_tensor(self, images: Sequence) -> torch.Tensor:
        transform = self._transform()
        if isinstance(images, torch.Tensor):
            tensor = images
        else:
            tensor = torch.stack([transform(im.convert("RGB")) for im in images])
        return tensor.to(self.device)

    def _run(self, images: Sequence, fn) -> np.ndarray:
        outs: List[np.ndarray] = []
        n = len(images)
        for start in range(0, n, self.batch_size):
            batch = images[start:start + self.batch_size]
            tensor = self._to_tensor(batch)
            out = fn(tensor)
            outs.append(out.detach().float().cpu().numpy())
        if not outs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(outs, axis=0)


class TorchvisionClassifier(BaseClassifier):
    """Supervised ImageNet model from ``torchvision.models``."""

    def __init__(self, spec: ModelSpec, device: str = "cpu", batch_size: int = 64):
        super().__init__(spec, device=device, batch_size=batch_size)
        import torchvision.models as tvm

        self.tvm = tvm
        # build the model with the requested weights tag (downloads on first use)
        self.model = tvm.get_model(spec.arch, weights=spec.pretrained)
        self.model.eval().to(self.device)
        self.weights_tag = spec.pretrained
        self._transform_fn = self._make_transform()
        self._feature_hook = None
        self._register_feature_hook()

    def _make_transform(self):
        from torchvision.models._api import get_model_weights

        enum = get_model_weights(getattr(self.tvm, self.spec.arch))
        w = enum[self.spec.pretrained] if self.spec.pretrained else enum.DEFAULT
        return w.transforms()

    def _transform(self):
        return self._transform_fn

    def _register_feature_hook(self) -> None:
        """Capture the input of the *main* classifier (the ``M(X)`` features).

        The paper takes ``M(X)`` from the last hidden layer before the linear
        classifier.  Picking the last ``nn.Linear`` in module order would be
        wrong for GoogLeNet and Inception-v3 (their auxiliary classifiers are
        registered after the main one), so we resolve the main head explicitly.
        """
        target = self._feature_module()
        self._features: Optional[torch.Tensor] = None
        if target is not None:
            def hook(_module, inputs, _output):
                self._features = inputs[0]

            self._feature_hook = target.register_forward_hook(hook)

    def _feature_module(self) -> Optional[nn.Module]:
        """The classifier head of the backbone (``fc``/``classifier``/``head``)."""
        for attr in ("fc", "classifier", "head", "heads", "linear", "last_linear"):
            module = getattr(self.model, attr, None)
            if module is None:
                continue
            linear = _last_linear(module)
            return linear if linear is not None else module
        linear = _last_linear(self.model)
        if linear is not None:
            return linear
        children = list(self.model.children())
        return children[-1] if children else None

    def _logits_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def _features_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        self._features = None
        self.model(tensor)
        feats = self._features
        if feats is None:  # architectures without an nn.Linear head
            out = self.model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats = out
        return feats.flatten(1)


def _last_linear(module: nn.Module) -> Optional[nn.Linear]:
    """The last ``nn.Linear`` *within* ``module`` (excluding itself)."""
    found: Optional[nn.Linear] = None
    for child in module.modules():
        if child is module:
            continue
        if isinstance(child, nn.Linear):
            found = child
    if found is not None:
        return found
    return module if isinstance(module, nn.Linear) else None


class ClipClassifier(BaseClassifier):
    """Zero-shot classifier built on ``openai/CLIP``."""

    def __init__(self, spec: ModelSpec, device: str = "cpu", batch_size: int = 64,
                 templates: Optional[Sequence[str]] = None,
                 cache_dir: Optional[str] = DEFAULT_CACHE_DIR):
        super().__init__(spec, device=device, batch_size=batch_size)
        import clip

        self.clip = clip
        self.model, self.preprocess = clip.load(
            spec.arch, device=self.device, jit=False
        )
        self.model.eval()
        self.templates = list(templates or IMAGENET_TEMPLATES)
        self.cache_dir = cache_dir
        self.text_features = self._build_text_features()

    def _transform(self):
        return self.preprocess

    def _build_text_features(self) -> torch.Tensor:
        names = imagenet_classnames()
        key = "clip_%s_%s" % (self.spec.arch, templates_key(self.templates))

        def builder():
            class_prompts = [
                [t.format(name.replace("_", " ")) for t in self.templates]
                for name in names
            ]
            return self.encode_prompts(class_prompts)

        return load_or_build_text_features(self.cache_dir, key, builder)

    def _encode_text_batched(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.clip.tokenize(list(prompts)).to(self.device)
        return self.model.encode_text(tokens)

    def encode_prompts(self, class_prompts: Sequence[Sequence[str]]) -> torch.Tensor:
        """``class_prompts`` is ``(n_classes, n_templates)`` -> ``(n_classes, D)``."""
        flat = [p for prompts in class_prompts for p in prompts]
        n_templates = len(class_prompts[0])
        embs: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(flat), 4096):
                embs.append(self._encode_text_batched(flat[start:start + 4096]))
        embs = torch.cat(embs, dim=0)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.view(len(class_prompts), n_templates, -1).mean(dim=1)
        return embs / embs.norm(dim=-1, keepdim=True)

    def logits_with_text(self, images: Sequence,
                         text_features: torch.Tensor) -> np.ndarray:
        tensor = self._to_tensor(images)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            out = logit_scale * image_features @ text_features.t().to(image_features.dtype)
        return out.float().cpu().numpy()

    def _logits_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()

    def _features_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(tensor)
        return feats / feats.norm(dim=-1, keepdim=True)


class OpenClipClassifier(BaseClassifier):
    """Zero-shot classifier built on ``OpenCLIP``."""

    def __init__(self, spec: ModelSpec, device: str = "cpu", batch_size: int = 64,
                 templates: Optional[Sequence[str]] = None,
                 cache_dir: Optional[str] = DEFAULT_CACHE_DIR):
        super().__init__(spec, device=device, batch_size=batch_size)
        import open_clip

        self.open_clip = open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.arch, pretrained=spec.pretrained, device=self.device
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.templates = list(templates or IMAGENET_TEMPLATES)
        self.cache_dir = cache_dir
        self.tokenizer = open_clip.get_tokenizer(spec.arch)
        self.text_features = self._build_text_features()

    def _transform(self):
        return self.preprocess

    def _build_text_features(self) -> torch.Tensor:
        names = imagenet_classnames()
        key = "openclip_%s_%s_%s" % (
            self.spec.arch, self.spec.pretrained, templates_key(self.templates)
        )

        def builder():
            class_prompts = [
                [t.format(name.replace("_", " ")) for t in self.templates]
                for name in names
            ]
            return self.encode_prompts(class_prompts)

        return load_or_build_text_features(self.cache_dir, key, builder)

    def encode_prompts(self, class_prompts: Sequence[Sequence[str]]) -> torch.Tensor:
        flat = [p for prompts in class_prompts for p in prompts]
        n_templates = len(class_prompts[0])
        embs: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(flat), 4096):
                tokens = self.tokenizer(flat[start:start + 4096]).to(self.device)
                embs.append(self.model.encode_text(tokens))
        embs = torch.cat(embs, dim=0)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.view(len(class_prompts), n_templates, -1).mean(dim=1)
        return embs / embs.norm(dim=-1, keepdim=True)

    def logits_with_text(self, images: Sequence,
                         text_features: torch.Tensor) -> np.ndarray:
        tensor = self._to_tensor(images)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            out = logit_scale * image_features @ text_features.t().to(image_features.dtype)
        return out.float().cpu().numpy()

    def _logits_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()

    def _features_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(tensor)
        return feats / feats.norm(dim=-1, keepdim=True)


class LavisClassifier(BaseClassifier):
    """ALBEF / BLIP feature extractors (zero-shot via image-text contrastive)."""

    def __init__(self, spec: ModelSpec, device: str = "cpu", batch_size: int = 64,
                 templates: Optional[Sequence[str]] = None,
                 cache_dir: Optional[str] = DEFAULT_CACHE_DIR):
        super().__init__(spec, device=device, batch_size=batch_size)
        try:
            from lavis.models import load_model_and_preprocess
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The ALBEF/BLIP models require the `lavis` package "
                "(pip install salesforce-lavis)."
            ) from exc

        model_type = "albef_feature_extractor" if "albef" in spec.arch else "blip_feature_extractor"
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=model_type,
            model_type="base",
            is_eval=True,
            device=self.device,
            cache_dir=None,
        )
        self.model.eval()
        self._vis = self.vis_processors["eval"]
        self.templates = list(templates or IMAGENET_TEMPLATES)
        self.cache_dir = cache_dir
        self.text_features = self._build_text_features()

    def _transform(self):
        return self._vis

    def _build_text_features(self) -> torch.Tensor:
        names = imagenet_classnames()
        key = "lavis_%s_%s" % (self.spec.arch, templates_key(self.templates))

        def builder():
            class_prompts = [
                [t.format(name.replace("_", " ")) for t in self.templates]
                for name in names
            ]
            return self.encode_prompts(class_prompts)

        return load_or_build_text_features(self.cache_dir, key, builder)

    def encode_prompts(self, class_prompts: Sequence[Sequence[str]]) -> torch.Tensor:
        flat = [p for prompts in class_prompts for p in prompts]
        n_templates = len(class_prompts[0])
        embs: List[torch.Tensor] = []
        for start in range(0, len(flat), 1024):
            tokens = self.model.tokenizer(
                flat[start:start + 1024],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                embs.append(self.model.forward_text(tokens).text_embeds)
        embs = torch.cat(embs, dim=0)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.view(len(class_prompts), n_templates, -1).mean(dim=1)
        return embs / embs.norm(dim=-1, keepdim=True)

    def logits_with_text(self, images: Sequence,
                         text_features: torch.Tensor) -> np.ndarray:
        tensor = self._to_tensor(images)
        with torch.no_grad():
            feats = self._features_tensor(tensor)
            out = feats @ text_features.t().to(feats.dtype)
        return out.float().cpu().numpy()

    def _logits_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        image_features = self._features_tensor(tensor)
        return image_features @ self.text_features.t()

    def _features_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_image(tensor).image_embeds
        return feats / feats.norm(dim=-1, keepdim=True)


_SOURCES = {
    "torchvision": TorchvisionClassifier,
    "clip": ClipClassifier,
    "open_clip": OpenClipClassifier,
    "lavis": LavisClassifier,
}


def build_classifier(
    spec: ModelSpec,
    device: str = "cpu",
    batch_size: int = 64,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    templates: Optional[Sequence[str]] = None,
) -> BaseClassifier:
    """Instantiate the adapter for ``spec`` (weights are downloaded on demand)."""
    cls = _SOURCES[spec.source]
    if spec.source in ("clip", "open_clip", "lavis"):
        return cls(
            spec,
            device=device,
            batch_size=batch_size,
            templates=templates,
            cache_dir=cache_dir,
        )
    return cls(spec, device=device, batch_size=batch_size)


def default_image_transform(input_size: int = 224):
    """Fallback transform for datasets / models without their own preprocessing."""
    from torchvision import transforms

    resize = int(round(input_size / 0.875))
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
