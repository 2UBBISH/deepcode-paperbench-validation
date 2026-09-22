"""Vision-only Model (VM) zoo for the LCA-on-the-Line study.

Paper references
----------------
* §4 (Dataset Setup): "We leverage 75 pretrained models ... Our selection
  comprises 36 Vision Models (VMs) pretrained on ImageNet and supervised from
  class labels ... A comprehensive list of model details, ensuring
  reproducibility, is provided in Appendix A."
* Appendix A (Model Architectures): the 36 torchvision checkpoints --
  alexnet, convnext_tiny, densenet{121,161,169,201}, efficientnet_b0,
  googlenet, inception_v3, mnasnet{0.5,0.75,1.0,1.3},
  mobilenet_v3_{small,large}, regnet_y_1_6gf, wide_resnet101_2,
  resnet{18,34,50,101,152}, shufflenet_v2_x2_0, squeezenet1_{0,1}, swin_b,
  vgg{11,13,16,19}[_bn], vit_{b,l}_32.
* Addendum ("Additional information"): "All vision-only models should be
  accessed via the torchvision module."

What this module provides
-------------------------
Every VM is wrapped by :class:`VisionModelWrapper`, mirroring the interface
consumed by the rest of the code base:

* ``wrapper(x)`` / ``wrapper.logits(x)`` -> class logits ``(B, num_classes)``
* ``wrapper.features(x)`` -> last hidden layer *before* the FC layer, i.e.
  ``M(X)`` (paper §4.3.1), pooled/flattened to ``(B, D)``
* ``wrapper.forward_both(x)`` -> ``(features, logits)`` from a **single**
  forward pass (feature caching for the 75-model evaluation is expensive, so
  we never compute twice).

Feature extraction is architecture agnostic: a ``forward_pre_hook`` on the
module that produces class scores (``fc`` / ``classifier[...]`` / ``heads.head``
/ ``head`` / SqueezeNet's final 1x1 conv) captures its input, reproducing the
paper's penultimate representation for all 16 torchvision families.

``torch``/``torchvision`` are imported lazily so the registry metadata stays
usable in minimal environments.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

IMAGENET_NUM_CLASSES = 1000

# ---------------------------------------------------------------------------
# Lazy torch / torchvision access
# ---------------------------------------------------------------------------


def _torch():
    import torch  # noqa: WPS433 (lazy import by design)

    return torch


def _nn():
    import torch.nn as nn  # noqa: WPS433

    return nn


def _torchvision_models():
    import torchvision.models as models  # noqa: WPS433

    return models


# ---------------------------------------------------------------------------
# Model registry (Appendix A)
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """One entry of the Appendix A VM list."""

    name: str
    family: str
    #: torchvision factory function name (``torchvision.models.<build>``).
    build: str
    #: Preferred ``IMAGENET1K_*`` weight enum names, in order of preference.
    weight_enum: Tuple[str, ...] = ("IMAGENET1K_V1",)
    #: Official evaluation resolution (299 for Inception-v3, else 224).
    input_size: int = 224
    #: ``weights_enum`` attribute name in ``torchvision.models``
    #: (e.g. ``ResNet18_Weights``); derived when omitted.
    weights_cls: Optional[str] = None
    #: ImageNet top-1 of the checkpoint that gets downloaded (informational).
    reported_top1: Optional[float] = None
    extra_build_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def weights_class_name(self) -> str:
        """torchvision convention: ``resnet18`` -> ``ResNet18_Weights``."""
        if self.weights_cls:
            return self.weights_cls
        head = "".join(part.capitalize() for part in self.build.split("_"))
        return f"{head}_Weights"


def _specs() -> List[ModelSpec]:
    """The 36 VMs of Appendix A, in table order."""
    out: List[ModelSpec] = []

    def add(name: str, family: str, build: str,
            weight_enum: Sequence[str] = ("IMAGENET1K_V1",),
            input_size: int = 224, weights_cls: Optional[str] = None,
            reported_top1: Optional[float] = None, **kwargs: Any) -> None:
        out.append(
            ModelSpec(
                name=name,
                family=family,
                build=build,
                weight_enum=tuple(weight_enum),
                input_size=input_size,
                weights_cls=weights_cls,
                reported_top1=reported_top1,
                extra_build_kwargs=kwargs,
            )
        )

    # --- 1 AlexNet -------------------------------------------------------
    add("alexnet", "alexnet", "alexnet", reported_top1=0.5652)
    # --- 1 ConvNeXt ------------------------------------------------------
    add("convnext_tiny", "convnext", "convnext_tiny", reported_top1=0.8270)
    # --- 4 DenseNet ------------------------------------------------------
    add("densenet121", "densenet", "densenet121", reported_top1=0.7476)
    add("densenet161", "densenet", "densenet161", reported_top1=0.7774)
    add("densenet169", "densenet", "densenet169", reported_top1=0.7628)
    add("densenet201", "densenet", "densenet201", reported_top1=0.7738)
    # --- 1 EfficientNet --------------------------------------------------
    add("efficientnet_b0", "efficientnet", "efficientnet_b0",
        reported_top1=0.7755)
    # --- 1 GoogLeNet -----------------------------------------------------
    add("googlenet", "googlenet", "googlenet", reported_top1=0.6988)
    # --- 1 Inception-v3 --------------------------------------------------
    add("inception_v3", "inception", "inception_v3", input_size=299,
        reported_top1=0.7798)
    # --- 4 MnasNet -------------------------------------------------------
    add("mnasnet0_5", "mnasnet", "mnasnet0_5", reported_top1=0.6769)
    add("mnasnet0_75", "mnasnet", "mnasnet0_75", reported_top1=0.7151)
    add("mnasnet1_0", "mnasnet", "mnasnet1_0", reported_top1=0.7367)
    add("mnasnet1_3", "mnasnet", "mnasnet1_3", reported_top1=0.7663)
    # --- 2 MobileNet-V3 --------------------------------------------------
    add("mobilenet_v3_small", "mobilenetv3", "mobilenet_v3_small",
        reported_top1=0.6770)
    add("mobilenet_v3_large", "mobilenetv3", "mobilenet_v3_large",
        reported_top1=0.7407)
    # --- 1 RegNet --------------------------------------------------------
    add("regnet_y_1_6gf", "regnet", "regnet_y_1_6gf",
        weights_cls="RegNet_Y_1_6GF_Weights", reported_top1=0.7768)
    # --- 1 Wide ResNet ---------------------------------------------------
    add("wide_resnet101_2", "resnet", "wide_resnet101_2",
        reported_top1=0.7875)
    # --- 5 ResNet --------------------------------------------------------
    add("resnet18", "resnet", "resnet18", reported_top1=0.6976)
    add("resnet34", "resnet", "resnet34", reported_top1=0.7348)
    add("resnet50", "resnet", "resnet50", reported_top1=0.7613)
    add("resnet101", "resnet", "resnet101", reported_top1=0.7787)
    add("resnet152", "resnet", "resnet152", reported_top1=0.7825)
    # --- 1 ShuffleNet ----------------------------------------------------
    add("shufflenet_v2_x2_0", "shufflenet", "shufflenet_v2_x2_0",
        reported_top1=0.7692)
    # --- 2 SqueezeNet ----------------------------------------------------
    add("squeezenet1_0", "squeezenet", "squeezenet1_0", reported_top1=0.5882)
    add("squeezenet1_1", "squeezenet", "squeezenet1_1", reported_top1=0.5820)
    # --- 1 Swin Transformer ----------------------------------------------
    add("swin_b", "swin", "swin_b", reported_top1=0.8370)
    # --- 8 VGG -----------------------------------------------------------
    add("vgg11", "vgg", "vgg11", reported_top1=0.6917)
    add("vgg13", "vgg", "vgg13", reported_top1=0.6995)
    add("vgg16", "vgg", "vgg16", reported_top1=0.7162)
    add("vgg19", "vgg", "vgg19", reported_top1=0.7225)
    add("vgg11_bn", "vgg", "vgg11_bn", reported_top1=0.7048)
    add("vgg13_bn", "vgg", "vgg13_bn", reported_top1=0.7170)
    add("vgg16_bn", "vgg", "vgg16_bn", reported_top1=0.7352)
    add("vgg19_bn", "vgg", "vgg19_bn", reported_top1=0.7421)
    # --- 2 ViT -----------------------------------------------------------
    add("vit_b_32", "vit", "vit_b_32", reported_top1=0.7590)
    add("vit_l_32", "vit", "vit_l_32", reported_top1=0.7692)

    return out


VISION_MODEL_SPECS: List[ModelSpec] = _specs()
SPECS_BY_NAME: Dict[str, ModelSpec] = {s.name: s for s in VISION_MODEL_SPECS}
VM_NAMES: List[str] = [s.name for s in VISION_MODEL_SPECS]

assert len(VISION_MODEL_SPECS) == 36, (
    f"Appendix A lists 36 VMs, found {len(VISION_MODEL_SPECS)}"
)

#: Aliases accepted by :func:`get_model_spec`.
VM_ALIASES: Dict[str, str] = {
    "mnasnet_0_5": "mnasnet0_5",
    "mnasnet_0_75": "mnasnet0_75",
    "mnasnet_1_0": "mnasnet1_0",
    "mnasnet_1_3": "mnasnet1_3",
    "regnet_y_1_6_gf": "regnet_y_1_6gf",
    "wide_resnet": "wide_resnet101_2",
    "swin_b_224": "swin_b",
    "vit_b_32_224": "vit_b_32",
    "vit_l_32_224": "vit_l_32",
}


def list_vm_names() -> List[str]:
    """All 36 Appendix A VM names."""
    return list(VM_NAMES)


def specs_by_family() -> Dict[str, List[ModelSpec]]:
    out: Dict[str, List[ModelSpec]] = {}
    for spec in VISION_MODEL_SPECS:
        out.setdefault(spec.family, []).append(spec)
    return out


def get_model_spec(name: str) -> ModelSpec:
    key = str(name).strip().lower().replace("-", "_")
    key = VM_ALIASES.get(key, key)
    if key not in SPECS_BY_NAME:
        raise KeyError(f"Unknown VM '{name}'. Available: {', '.join(VM_NAMES)}")
    return SPECS_BY_NAME[key]


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------


def get_weights_enum(spec: ModelSpec) -> Optional[Any]:
    models = _torchvision_models()
    return getattr(models, spec.weights_class_name, None)


def resolve_weights(
    spec: ModelSpec,
    pretrained: bool = True,
    weights_name: Optional[str] = None,
) -> Optional[Any]:
    """Return a torchvision ``Weights`` enum member (or ``None``).

    Tries the preferred ``IMAGENET1K_*`` variants in order, falling back
    gracefully when a checkpoint is missing in the installed torchvision.
    """
    if not pretrained:
        return None
    cls = get_weights_enum(spec)
    if cls is None:
        logger.warning(
            "[vm_zoo] %s: weights class %s not found; using torchvision default",
            spec.name, spec.weights_class_name,
        )
        return "DEFAULT"
    for cand in ([weights_name] if weights_name else list(spec.weight_enum)):
        if cand is None:
            continue
        if hasattr(cls, cand):
            return getattr(cls, cand)
    try:
        return cls.DEFAULT
    except Exception:  # pragma: no cover - defensive
        logger.warning("[vm_zoo] %s: no resolvable weights enum", spec.name)
        return None


# ---------------------------------------------------------------------------
# Locating the classification head
# ---------------------------------------------------------------------------


def find_classifier_module(model: Any) -> Any:
    """Locate the module mapping features -> class scores.

    Handles all 16 torchvision families in Appendix A:

    * ResNet / WideResNet / RegNet / ShuffleNet -> ``model.fc``
    * AlexNet / VGG / SqueezeNet                -> element of ``model.classifier``
      (``nn.Linear`` or 1x1 ``nn.Conv2d``)
    * DenseNet / MnasNet / MobileNetV3 /
      EfficientNet / ConvNeXt                   -> last ``nn.Linear`` of
      ``model.classifier``
    * GoogLeNet / Inception-v3                  -> ``model.fc``
    * Swin                                      -> ``model.head``
    * ViT                                       -> ``model.heads.head``
    """
    nn = _nn()

    def _last_linear(container: Any) -> Optional[Any]:
        last = None
        for m in container.modules():
            if isinstance(m, nn.Linear):
                last = m
        return last

    # 1) explicit attributes
    for attr in ("fc", "head"):
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.Linear):
            return mod
    heads = getattr(model, "heads", None)
    if heads is not None:
        mod = getattr(heads, "head", None)
        if isinstance(mod, nn.Linear):
            return mod
    cls_attr = getattr(model, "classifier", None)
    if cls_attr is not None:
        if isinstance(cls_attr, nn.Linear):
            return cls_attr
        lin = _last_linear(cls_attr)
        if lin is not None:
            return lin

    # 2) last Linear anywhere in the ordered module tree
    lin = _last_linear(model)
    if lin is not None:
        return lin

    # 3) SqueezeNet-style: final 1x1 conv producing the class logits
    conv_candidates: List[Any] = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == IMAGENET_NUM_CLASSES:
            conv_candidates.append(m)
    if conv_candidates:
        return conv_candidates[-1]

    raise RuntimeError(
        "Could not locate a classification head for this architecture"
    )


def _as_2d_logits(out: Any) -> Any:
    if isinstance(out, (tuple, list)):  # googlenet returns (logits, aux)
        out = out[0]
    if isinstance(out, dict):
        for key in ("logits", "out", "output"):
            if key in out:
                out = out[key]
                break
        else:  # pragma: no cover - defensive
            out = next(iter(out.values()))
    if out.dim() > 2:
        out = out.flatten(1)
    if out.dim() == 2 and out.shape[1] > IMAGENET_NUM_CLASSES:
        out = out[..., :IMAGENET_NUM_CLASSES]
    return out.float()


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class VisionModelWrapper(object):
    """A pretrained VM exposing penultimate features ``M(X)`` and logits.

    Parameters
    ----------
    name:
        Registry key (one of :data:`VM_NAMES`).
    model:
        Optional pre-built ``nn.Module`` (otherwise built from torchvision).
    spec:
        Optional :class:`ModelSpec` override.
    pretrained:
        Load ImageNet weights (default ``True``).
    device:
        ``torch.device`` or string; defaults to CUDA when available.
    flatten_features:
        When ``True`` (default) ``features()`` returns ``(B, D)`` via global
        average pooling + flatten; otherwise the raw captured tensor.
    """

    def __init__(
        self,
        name: str,
        model: Any = None,
        spec: Optional[ModelSpec] = None,
        pretrained: bool = True,
        device: Optional[Any] = None,
        flatten_features: bool = True,
        weights_name: Optional[str] = None,
        eval_mode: bool = True,
    ) -> None:
        torch = _torch()
        self.name = name
        self.spec = spec if spec is not None else get_model_spec(name)
        self.flatten_features = bool(flatten_features)
        self._captured: Optional[Any] = None

        if model is None:
            model = build_torchvision_model(
                self.spec, pretrained=pretrained, weights_name=weights_name
            )
        self.model = model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        if eval_mode:
            self.model.eval()

        self.classifier = find_classifier_module(self.model)
        # This pre-hook captures the *input* of the classification head, i.e.
        # the last hidden layer M(X) of §4.3.1.
        self._hook = self.classifier.register_forward_pre_hook(
            self._capture_input
        )

    # -- hooks ------------------------------------------------------------
    def _capture_input(self, module: Any, inputs: Sequence[Any]) -> None:
        if inputs and inputs[0] is not None:
            self._captured = inputs[0]

    # -- basic properties -------------------------------------------------
    @property
    def num_classes(self) -> int:
        out = getattr(self.classifier, "out_features", None)
        if out is None:
            out = getattr(self.classifier, "out_channels",
                          IMAGENET_NUM_CLASSES)
        return int(out)

    @property
    def input_size(self) -> int:
        return int(self.spec.input_size)

    @property
    def feature_dim(self) -> Optional[int]:
        dim = getattr(self.classifier, "in_features", None)
        if dim is None:
            dim = getattr(self.classifier, "in_channels", None)
        return int(dim) if dim is not None else None

    def train(self, mode: bool = True) -> "VisionModelWrapper":
        self.model.train(mode)
        return self

    def eval(self) -> "VisionModelWrapper":
        self.model.eval()
        return self

    def to(self, device: Any) -> "VisionModelWrapper":
        torch = _torch()
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def parameters(self):  # pragma: no cover - passthrough
        return self.model.parameters()

    def state_dict(self):  # pragma: no cover - passthrough
        return self.model.state_dict()

    # -- forwards ---------------------------------------------------------
    def logits(self, x: Any) -> Any:
        """Class logits ``(B, K)``."""
        torch = _torch()
        with torch.no_grad():
            out = self.model(x)
        return _as_2d_logits(out)

    __call__ = logits

    def _pooled(self, feats: Any) -> Any:
        if feats is None:
            raise RuntimeError(
                "No features captured; did the forward pass reach the head?"
            )
        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))  # global average pooling
        elif feats.dim() == 3:
            feats = feats.mean(dim=2) if feats.shape[-1] != 1 else feats.squeeze(-1)
        elif feats.dim() > 4:
            feats = feats.flatten(2).mean(dim=2)
        out = feats.float()
        if out.dim() > 2:
            out = out.flatten(1)
        return out

    def forward_both(self, x: Any, return_logits: bool = True) -> Any:
        """Single forward pass returning ``(features, logits)``.

        ``features`` is ``(B, D)`` when ``flatten_features`` is set (the
        penultimate layer ``M(X)`` of §4.3.1); ``logits`` is ``(B, K)``.
        """
        torch = _torch()
        self._captured = None
        with torch.no_grad():
            out = self.model(x)
        feats = self._captured
        if feats is None:
            raise RuntimeError(
                f"[vm_zoo] {self.name}: classification head never reached"
            )
        feats = self._pooled(feats) if self.flatten_features else feats
        if not return_logits:
            return feats
        return feats, _as_2d_logits(out)

    def features(self, x: Any) -> Any:
        """Last hidden layer ``M(X)`` before the FC layer."""
        return self.forward_both(x, return_logits=False)

    def raw_features(self, x: Any) -> Any:
        """Un-pooled captured tensor (4D for conv heads)."""
        torch = _torch()
        with torch.no_grad():
            self.model(x)
        return self._captured

    def prediction_scores(self, x: Any) -> Tuple[Any, Any]:
        """Alias of :meth:`forward_both` -> ``(features, logits)``."""
        return self.forward_both(x)

    # -- data pipeline ----------------------------------------------------
    def eval_transform(self) -> Callable:
        return get_eval_transform(self.spec)

    def train_transform(self) -> Callable:
        return get_train_transform(self.spec)

    # -- housekeeping -----------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_hook", None) is not None:
            self._hook.remove()
            self._hook = None

    def __repr__(self) -> str:
        return (
            f"VisionModelWrapper(name={self.name!r}, "
            f"family={self.spec.family!r}, num_classes={self.num_classes}, "
            f"feature_dim={self.feature_dim}, device={self.device})"
        )


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def build_torchvision_model(
    spec: ModelSpec,
    pretrained: bool = True,
    weights_name: Optional[str] = None,
) -> Any:
    """Instantiate ``torchvision.models.<spec.build>`` with ImageNet weights."""
    models = _torchvision_models()
    factory = getattr(models, spec.build, None)
    if factory is None:
        raise AttributeError(
            f"torchvision.models has no factory '{spec.build}' "
            f"(needed for VM '{spec.name}'). Update torchvision."
        )
    weights = resolve_weights(spec, pretrained=pretrained,
                              weights_name=weights_name)
    kwargs = dict(spec.extra_build_kwargs)
    try:
        model = factory(weights=None if weights is None else weights, **kwargs)
    except TypeError:
        # older torchvision: positional ``pretrained`` flag
        model = factory(pretrained=bool(pretrained), **kwargs)
    return model


def create_vm(
    name: str,
    pretrained: bool = True,
    device: Optional[Any] = None,
    weights_name: Optional[str] = None,
    flatten_features: bool = True,
) -> VisionModelWrapper:
    """Create one wrapped VM by registry name."""
    spec = get_model_spec(name)
    return VisionModelWrapper(
        name=spec.name,
        spec=spec,
        pretrained=pretrained,
        device=device,
        flatten_features=flatten_features,
        weights_name=weights_name,
    )


def build_vm_zoo(
    names: Optional[Iterable[str]] = None,
    pretrained: bool = True,
    device: Optional[Any] = None,
    allow_failures: bool = True,
    flatten_features: bool = True,
) -> Dict[str, VisionModelWrapper]:
    """Build a ``{name: wrapper}`` dict for the requested VMs.

    Missing checkpoints / download errors are logged and skipped when
    ``allow_failures`` is set, so a partial zoo can still be evaluated.
    """
    wanted = list(names) if names is not None else list(VM_NAMES)
    zoo: Dict[str, VisionModelWrapper] = {}
    for name in wanted:
        try:
            zoo[name] = create_vm(
                name,
                pretrained=pretrained,
                device=device,
                flatten_features=flatten_features,
            )
            logger.info("[vm_zoo] loaded %s", name)
        except Exception as exc:  # pragma: no cover - environment dependent
            if not allow_failures:
                raise
            logger.warning("[vm_zoo] skipping %s: %s", name, exc)
    return zoo


def release_vm_zoo(zoo: Dict[str, VisionModelWrapper]) -> None:
    for wrapper in zoo.values():
        try:
            wrapper.close()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def get_eval_transform(spec: ModelSpec) -> Callable:
    """Deterministic evaluation transform for a VM.

    Prefers the official ``weights.transforms()`` pipeline (which reproduces the
    checkpoint's own preprocessing) and falls back to the shared ImageNet eval
    transform from :mod:`src.data.imagenet` at ``spec.input_size``.
    """
    weights_cls = get_weights_enum(spec)
    if weights_cls is not None:
        for cand in list(spec.weight_enum) + ["DEFAULT"]:
            member = getattr(weights_cls, cand, None)
            if member is None:
                continue
            try:
                return member.transforms()
            except Exception:
                continue
    from ..data.imagenet import build_transform  # local import (no cycle)

    return build_transform(resolution=spec.input_size, crop_pct=0.875)


def get_train_transform(
    spec: ModelSpec, resolution: Optional[int] = None
) -> Callable:
    weights_cls = get_weights_enum(spec)
    if weights_cls is not None:
        member = getattr(weights_cls, spec.weight_enum[0], None)
        if member is not None:
            try:
                return member.transforms()
            except Exception:
                pass
    from ..data.imagenet import build_transform

    return build_transform(
        resolution=resolution or spec.input_size,
        train=True,
        flip=True,
        color_jitter=0.4,
    )


# ---------------------------------------------------------------------------
# Feature/logit extraction + caching (used by the evaluation driver)
# ---------------------------------------------------------------------------


def extract_outputs(
    wrapper: VisionModelWrapper,
    loader: Any,
    device: Optional[Any] = None,
    max_batches: Optional[int] = None,
    desc: str = "",
) -> Dict[str, Any]:
    """Run a model over a loader, returning numpy ``features/logits/targets``.

    One forward pass per batch yields both ``M(X)`` (penultimate features) and
    the class logits, as required for LCA/ELCA and the K-means latent
    hierarchies of §4.3.1.
    """
    import numpy as np

    torch = _torch()
    dev = torch.device(device) if device is not None else wrapper.device
    wrapper.model.to(dev)
    wrapper.model.eval()

    feats_all: List[Any] = []
    logits_all: List[Any] = []
    targets_all: List[Any] = []
    try:
        from tqdm.auto import tqdm  # type: ignore

        iterator = tqdm(loader, desc=desc or wrapper.name, leave=False)
    except Exception:  # pragma: no cover
        iterator = loader

    with torch.no_grad():
        for step, batch in enumerate(iterator):
            if max_batches is not None and step >= max_batches:
                break
            images, targets = batch[0], batch[1]
            images = images.to(dev, non_blocking=True)
            feats, logits = wrapper.forward_both(images)
            feats_all.append(feats.detach().cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
            targets_all.append(
                targets.detach().cpu().numpy()
                if hasattr(targets, "detach") else np.asarray(targets)
            )

    if not feats_all:
        return {
            "features": np.zeros((0, wrapper.feature_dim or 0), dtype="float32"),
            "logits": np.zeros((0, IMAGENET_NUM_CLASSES), dtype="float32"),
            "targets": np.zeros((0,), dtype="int64"),
        }
    return {
        "features": np.concatenate(feats_all, axis=0).astype("float32"),
        "logits": np.concatenate(logits_all, axis=0).astype("float32"),
        "targets": np.concatenate(targets_all, axis=0).astype("int64"),
    }


def cache_path(cache_dir: str, model_name: str, dataset_name: str) -> str:
    safe = model_name.replace("/", "_").replace(" ", "_")
    ds = dataset_name.replace("/", "_").replace(" ", "_")
    return os.path.join(cache_dir, f"{safe}__{ds}.npz")


def save_outputs(
    path: str,
    features: Any,
    logits: Any,
    targets: Any,
    overwrite: bool = True,
) -> str:
    """Persist extracted outputs as ``.npz`` (feature caching for the zoo)."""
    import numpy as np

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path) and not overwrite:
        return path
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype="float32"),
        logits=np.asarray(logits, dtype="float32"),
        targets=np.asarray(targets, dtype="int64"),
    )
    return path


def load_outputs(path: str) -> Dict[str, Any]:
    import numpy as np

    with np.load(path) as data:
        return {
            k: data[k] for k in ("features", "logits", "targets") if k in data
        }


def extract_and_cache(
    wrapper: VisionModelWrapper,
    loader: Any,
    dataset_name: str,
    cache_dir: str,
    device: Optional[Any] = None,
    overwrite: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Cache-aware :func:`extract_outputs` used by ``evaluate_models.py``."""
    path = cache_path(cache_dir, wrapper.name, dataset_name)
    if os.path.exists(path) and not overwrite:
        logger.info("[vm_zoo] cache hit %s", path)
        return load_outputs(path)
    outputs = extract_outputs(
        wrapper, loader, device=device, max_batches=max_batches,
        desc=wrapper.name,
    )
    save_outputs(path, **outputs, overwrite=True)
    logger.info("[vm_zoo] cached %s -> %s", wrapper.name, path)
    return outputs


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


def create_dummy_vm(
    name: str = "dummy_vm",
    num_classes: int = IMAGENET_NUM_CLASSES,
    feature_dim: int = 64,
    device: Optional[Any] = None,
) -> VisionModelWrapper:
    """Small randomly-initialised CNN mimicking the wrapper interface.

    Used by smoke tests / the synthetic pipeline where the real 36 checkpoints
    are unavailable.  It keeps the usual structure (backbone -> pool ->
    ``fc`` -> head), so the same ``features``/``logits`` contract holds.
    """
    torch = _torch()
    nn = _nn()

    class _DummyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.pool = nn.Flatten(1)
            self.fc = nn.Linear(16, feature_dim)
            self.head = nn.Linear(feature_dim, num_classes)

        def forward(self, x: Any) -> Any:
            return self.head(self.fc(self.pool(self.backbone(x))))

    spec = ModelSpec(name=name, family="dummy", build="dummy_vm")
    return VisionModelWrapper(
        name=name, model=_DummyNet(), spec=spec, pretrained=False,
        device=device,
    )


__all__ = [
    "IMAGENET_NUM_CLASSES",
    "ModelSpec",
    "VISION_MODEL_SPECS",
    "SPECS_BY_NAME",
    "VM_NAMES",
    "VM_ALIASES",
    "list_vm_names",
    "specs_by_family",
    "get_model_spec",
    "resolve_weights",
    "get_weights_enum",
    "find_classifier_module",
    "VisionModelWrapper",
    "build_torchvision_model",
    "create_vm",
    "build_vm_zoo",
    "release_vm_zoo",
    "get_eval_transform",
    "get_train_transform",
    "extract_outputs",
    "extract_and_cache",
    "cache_path",
    "save_outputs",
    "load_outputs",
    "create_dummy_vm",
]
