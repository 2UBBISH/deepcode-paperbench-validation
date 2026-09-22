"""Trainable OpenCLIP vision encoder used for Robust CLIP fine-tuning / evaluation.

This module wraps `open_clip_torch` so that the *vision* tower of a CLIP model can
be used in three ways:

1. as a frozen zero-shot ImageNet / VQA feature extractor (vanilla CLIP baseline),
2. as a fine-tunable vision encoder (Robust CLIP unsupervised adversarial
   fine-tuning), optionally with only the last blocks / projection trainable,
3. as the image encoder consumed by the LLaVA-1.5 7B wrapper (see
   ``models/llava_openclip.py``), where the Addendum requires the *OpenAI CLIP
   ViT-L/14@224* weights through the *OpenCLIP* implementation rather than the
   HuggingFace CLIP implementation.

Addendum-relevant details encoded here
---------------------------------------
* Victim CLIP configuration: ``ViT-L-14`` with ``pretrained="openai"`` and
  ``resolution=224`` (LLaVA's default 336 is overridden; see §Addendum).
* Inputs are kept in **raw pixel space** ``[0, 1]`` by callers; the CLIP
  mean/std normalization lives in :func:`encode_images` / :meth:`encode_image`
  so that adversarial projections can be performed around *non-normalized*
  pixels (Addendum: the l_inf ball is computed around non-normalized inputs).
* Perturbation storage follows the int16/int32 precision policy from
  :mod:`robust_clip_repro.utils.precision`; this module only consumes raw or
  normalized pixels and never stores perturbations itself.

No attack hyper-parameters (eps / alpha / iterations / optimizer / learning rate
/ batch size / epochs) are invented here: training hyper-parameters are exposed
as explicit arguments and tagged ``UNSPECIFIED_BY_ADDENDUM`` where the paper body
does not state them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

LOGGER = logging.getLogger("robust_clip_repro.models.clip_vision_encoder")

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

# ---------------------------------------------------------------------------
# Paper / Addendum CLIP configuration
# ---------------------------------------------------------------------------

#: Vision backbone used for the paper's CLIP models and for LLaVA-1.5 7B.
DEFAULT_MODEL_NAME = "ViT-L-14"
#: OpenAI CLIP weights (the Addendum pins the OpenAI ViT-L/14@224 encoder).
DEFAULT_PRETRAINED = "openai"
#: The Addendum overrides LLaVA's default 336 resolution with 224.
DEFAULT_RESOLUTION = 224
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CONTEXT_LENGTH = 77

#: OpenAI CLIP normalization constants (mirrors ``data/imagenet.py``).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.25945126, 0.26063495, 0.2632633)

#: Values the Addendum/paper body do not state -> must be supplied externally.
EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "eps": UNSPECIFIED,
    "alpha": UNSPECIFIED,
    "iterations": UNSPECIFIED,
    "restarts": UNSPECIFIED,
    "attack_norm": UNSPECIFIED,
    "optimizer": UNSPECIFIED,
    "learning_rate": UNSPECIFIED,
    "batch_size": UNSPECIFIED,
    "epochs": UNSPECIFIED,
    "weight_decay": UNSPECIFIED,
    "warmup_epochs": UNSPECIFIED,
    "scheduler": UNSPECIFIED,
    "trainable_scope": UNSPECIFIED,
    "seed": 0,
    "num_workers": 0,
}

#: Robust-CLIP source repository (paper's own code is not part of the Addendum).
ROBUST_CLIP_REPO_URL = "https://github.com/locuslab/robust-clip"
OPENAI_CLIP_WEIGHTS_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CLIPVisionConfig:
    """Configuration for :class:`CLIPVisionEncoder`.

    All fields default to the Addendum-specified CLIP setup; attack/training
    hyper-parameters that the paper does not specify are left as ``None`` and
    flagged in :attr:`provenance`.
    """

    model_name: str = DEFAULT_MODEL_NAME
    pretrained: Optional[str] = DEFAULT_PRETRAINED
    pretrained_path: Optional[str] = None
    cache_dir: Optional[str] = None
    resolution: int = DEFAULT_RESOLUTION
    image_mean: Sequence[float] = CLIP_MEAN
    image_std: Sequence[float] = CLIP_STD
    device: Optional[str] = None
    dtype: Optional[Union[str, torch.dtype]] = None

    # Trainability (Robust CLIP: fine-tune the vision encoder).
    trainable: bool = False
    trainable_scope: str = "all"  # "all" | "last_blocks" | "projection" | "none"
    num_trainable_blocks: int = 1
    use_fp16: bool = False
    use_grad_checkpointing: bool = False
    force_precision_policy: bool = True

    # Training hyper-parameters: NOT specified by the Addendum.
    optimizer: Optional[str] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    seed: int = 0

    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.resolution) != DEFAULT_RESOLUTION:
            LOGGER.warning(
                "CLIPVisionConfig.resolution=%s differs from the Addendum's "
                "OpenAI CLIP ViT-L/14@%d setting.",
                self.resolution,
                DEFAULT_RESOLUTION,
            )
        if self.model_name not in ("ViT-L-14", "ViT-L/14"):
            LOGGER.warning(
                "CLIPVisionConfig.model_name=%s; the Addendum specifies "
                "OpenAI CLIP ViT-L/14@224.",
                self.model_name,
            )
        self._record_provenance()

    # -- provenance ---------------------------------------------------------
    def _record_provenance(self) -> None:
        prov: Dict[str, str] = dict(self.provenance or {})
        prov.setdefault(
            "model_name",
            "ADDENDUM: OpenAI CLIP ViT-L/14 (OpenCLIP implementation)",
        )
        prov.setdefault("pretrained", "ADDENDUM: OpenAI CLIP weights")
        prov.setdefault("resolution", "ADDENDUM: 224 (not LLaVA's default 336)")
        prov.setdefault("image_mean", "OpenAI CLIP normalization constants")
        prov.setdefault("image_std", "OpenAI CLIP normalization constants")
        for name in (
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "epochs",
            "trainable_scope",
        ):
            if getattr(self, name) in (None, UNSPECIFIED):
                prov.setdefault(name, UNSPECIFIED)
        self.provenance = prov

    # -- helpers ------------------------------------------------------------
    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype is None:
            return torch.float32
        if isinstance(self.dtype, torch.dtype):
            return self.dtype
        return getattr(torch, str(self.dtype))

    def as_dict(self) -> Dict[str, Any]:
        out = {}
        for f in fields(self):
            out[f.name] = getattr(self, f.name)
        return out

    @classmethod
    def from_dict(
        cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "CLIPVisionConfig":
        cfg = dict(cfg or {})
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(cfg) - known
        for key in sorted(unknown):
            LOGGER.warning("Ignoring unknown CLIPVisionConfig key: %s", key)
        kwargs = {k: v for k, v in cfg.items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class CLIPVisionEncoder(nn.Module):
    """OpenCLIP vision tower with an optional trainable scope.

    Parameters
    ----------
    config:
        :class:`CLIPVisionConfig`. When ``None`` the Addendum defaults
        (``ViT-L-14`` / ``openai`` / 224) are used.
    model:
        Optional pre-built ``open_clip`` model (used by tests / when the caller
        already owns the full CLIP model).
    tokenizer:
        Unused for pure vision usage; kept for interface parity with LLaVA.

    Notes
    -----
    ``encode_image`` expects **raw** pixels in ``[0, 1]`` and applies CLIP
    normalization internally, so adversarial perturbations can be projected in
    raw pixel space (Addendum requirement).
    """

    def __init__(
        self,
        config: Optional[CLIPVisionConfig] = None,
        model: Optional[nn.Module] = None,
        preprocess: Optional[Callable[[Any], torch.Tensor]] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = CLIPVisionConfig(**{k: v for k, v in kwargs.items() if k in {f.name for f in fields(CLIPVisionConfig)}})
        elif kwargs:
            config = CLIPVisionConfig.from_dict(config.as_dict(), **kwargs)
        self.config = config

        if model is not None:
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._open_clip_available = True
        else:
            (
                self._model,
                self._preprocess,
                self._tokenizer,
            ) = self._build_open_clip_model()

        self.visual = getattr(self._model, "visual", None)
        if self.visual is None:
            LOGGER.warning(
                "OpenCLIP model has no `.visual` attribute; vision encoding may fail."
            )

        self._configure_trainability()

        self.register_buffer(
            "mean", torch.tensor(config.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(config.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    # -- construction -------------------------------------------------------
    def _build_open_clip_model(
        self,
    ) -> Tuple[Any, Optional[Callable[[Any], torch.Tensor]], Optional[Any]]:
        """Load the OpenCLIP model required by the Addendum.

        Uses ``open_clip.create_model_and_transforms`` which returns the model,
        the eval preprocessing transform, and (for text-enabled models) the
        tokenizer. The default transform is *not* used for attack evaluation
        (it would hide the raw pixel space); it is exposed for convenience.
        """
        try:
            import open_clip  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "open_clip_torch is required to build the CLIP vision encoder. "
                "Install it with `pip install open_clip_torch`."
            ) from exc

        pretrained = self.config.pretrained
        if self.config.pretrained_path:
            pretrained = self.config.pretrained_path
        kwargs: Dict[str, Any] = {"cache_dir": self.config.cache_dir}
        model_name = self.config.model_name

        try:
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                precision="fp16" if self.config.use_fp16 else "fp32",
                device=self.config.device,
                **kwargs,
            )
        except TypeError:
            # Older open_clip signatures do not accept `precision`/`device`.
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to load %s with pretrained=%s (%s); retrying with "
                "pretrained=None and logging the externally supplied weights path.",
                model_name,
                pretrained,
                exc,
            )
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name, pretrained=None, **kwargs
            )
            self.config.provenance["pretrained_fallback"] = (
                "EXTERNAL: open_clip weights could not be downloaded; using "
                "randomly initialised vision tower (%s)" % UNSPECIFIED
            )

        # Resolution: the Addendum pins 224 for the OpenAI ViT-L/14 encoder.
        self._override_resolution(model, self.config.resolution)

        tokenizer = None
        try:
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception:  # pragma: no cover
            tokenizer = None

        if self.config.device is not None:
            model = model.to(self.config.device)
        if self.config.torch_dtype != torch.float32:
            model = model.to(self.config.torch_dtype)
        return model, preprocess_val, tokenizer

    @staticmethod
    def _override_resolution(model: nn.Module, resolution: int) -> None:
        visual = getattr(model, "visual", None)
        if visual is None:
            return
        for attr in ("image_size",):
            if hasattr(visual, attr):
                current = getattr(visual, attr)
                if isinstance(current, (tuple, list)):
                    setattr(visual, attr, (int(resolution), int(resolution)))
                else:
                    setattr(visual, attr, int(resolution))
        if hasattr(visual, "patch_size") and hasattr(visual, "grid_size"):
            try:
                grid = int(resolution) // int(visual.patch_size)
                visual.grid_size = (grid, grid)
                if hasattr(visual, "num_patches"):
                    visual.num_patches = grid * grid
            except Exception:  # pragma: no cover
                pass

    # -- trainability -------------------------------------------------------
    def _configure_trainability(self) -> None:
        cfg = self.config
        if not cfg.trainable:
            self.freeze()
            return

        scope = (cfg.trainable_scope or "all").lower()
        if scope in ("none", "frozen", "freeze"):
            self.freeze()
            return

        if scope in ("projection", "proj", "head"):
            self.freeze()
            for name, param in self.visual.named_parameters():
                if "proj" in name or "head" in name:
                    param.requires_grad = True
        elif scope in ("last_blocks", "last_block", "last_n"):
            self.freeze()
            blocks = self._transformer_blocks()
            if blocks is not None:
                n = max(1, int(cfg.num_trainable_blocks))
                for block in blocks[-n:]:
                    for param in block.parameters():
                        param.requires_grad = True
                for param in self._final_norm_parameters():
                    param.requires_grad = True
            else:  # pragma: no cover
                LOGGER.warning(
                    "Could not locate transformer blocks for trainable_scope=%s; "
                    "unfreezing the whole vision encoder.",
                    scope,
                )
                self.unfreeze()
        else:  # "all"
            self.unfreeze()

        if cfg.use_grad_checkpointing:
            self._enable_grad_checkpointing()

    def _transformer_blocks(self) -> Optional[Sequence[nn.Module]]:
        visual = self.visual
        if visual is None:
            return None
        for attr in ("transformer", "blocks", "resblocks"):
            holder = getattr(visual, attr, None)
            if holder is None:
                continue
            blocks = getattr(holder, "resblocks", None)
            if blocks is not None:
                return list(blocks)
            if isinstance(holder, (nn.ModuleList, list, tuple)):
                return list(holder)
        return None

    def _final_norm_parameters(self) -> List[nn.Parameter]:
        visual = self.visual
        if visual is None:
            return []
        params: List[nn.Parameter] = []
        for attr in ("ln_post", "norm", "ln_final"):
            module = getattr(visual, attr, None)
            if isinstance(module, nn.Module):
                params.extend(list(module.parameters()))
        return params

    def _enable_grad_checkpointing(self) -> None:
        visual = self.visual
        if visual is None:
            return
        for attr in ("set_grad_checkpointing", "grad_checkpointing"):
            fn = getattr(visual, attr, None)
            if callable(fn):
                try:
                    fn(True)
                    return
                except Exception:  # pragma: no cover
                    pass

    def freeze(self) -> "CLIPVisionEncoder":
        for param in self.parameters():
            param.requires_grad = False
        return self

    def unfreeze(self) -> "CLIPVisionEncoder":
        for param in self.parameters():
            param.requires_grad = True
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters()))

    # -- properties ---------------------------------------------------------
    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def tokenizer(self) -> Optional[Any]:
        return self._tokenizer

    @property
    def preprocess(self) -> Optional[Callable[[Any], torch.Tensor]]:
        return self._preprocess

    @property
    def output_dim(self) -> int:
        visual = self.visual
        for attr in ("output_dim", "width", "embed_dim"):
            value = getattr(visual, attr, None) if visual is not None else None
            if isinstance(value, int):
                return value
        projection = getattr(visual, "proj", None) if visual is not None else None
        if projection is not None and hasattr(projection, "weight"):
            return int(projection.weight.shape[0])
        return 768 if "B" in self.config.model_name else 1024

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    # -- normalization ------------------------------------------------------
    def normalize(self, pixels: torch.Tensor) -> torch.Tensor:
        """Apply CLIP mean/std normalization to raw pixels in ``[0, 1]``."""
        return (pixels - self.mean.to(pixels.dtype)) / self.std.to(pixels.dtype)

    def denormalize(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels * self.std.to(pixels.dtype) + self.mean.to(pixels.dtype)

    # -- forwarding ---------------------------------------------------------
    def vision_forward(self, normalized: torch.Tensor) -> torch.Tensor:
        """Run the vision tower on *already normalized* inputs."""
        visual = self.visual
        if visual is None:
            raise RuntimeError("CLIP vision tower is unavailable.")
        return visual(normalized)

    def forward(
        self,
        pixels: torch.Tensor,
        *,
        normalize: bool = True,
        return_features: bool = True,
    ) -> torch.Tensor:
        """Encode raw (default) or normalized pixels.

        Parameters
        ----------
        pixels:
            ``(B, 3, H, W)`` tensor in ``[0, 1]`` when ``normalize=True``.
        normalize:
            When ``True`` (default) pixels are treated as raw ``[0, 1]`` and CLIP
            normalization is applied here.
        return_features:
            Kept for interface parity; the vision tower always returns the
            projected image embedding.
        """
        if normalize:
            normalized = self.normalize(pixels)
        else:
            normalized = pixels
        return self.vision_forward(normalized)

    def encode_image(
        self, pixels: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """Alias of :meth:`forward` returning the projected image embedding."""
        return self.forward(pixels, normalize=normalize)

    def encode_images(
        self, pixels: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        return self.forward(pixels, normalize=normalize)

    # -- LLaVA integration --------------------------------------------------
    def forward_patch_tokens(
        self,
        pixels: torch.Tensor,
        *,
        normalize: bool = True,
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Return *patch* tokens (pre-projection) for LLaVA-style consumption.

        LLaVA-1.5 consumes a grid of vision features and projects them into the
        language model's embedding space. This helper exposes the penultimate
        grid of tokens from the OpenCLIP vision transformer so the LLaVA wrapper
        can apply its own ``mm_projector``.

        Returns a ``(B, N, D)`` tensor where ``N`` is the patch grid size.
        Raises ``RuntimeError`` if the vision tower does not expose its
        transformer internals (in that case use :meth:`forward` and the model's
        own projector).
        """
        visual = self.visual
        if visual is None:
            raise RuntimeError("CLIP vision tower is unavailable.")
        normalized = self.normalize(pixels) if normalize else pixels

        # Preferred path: open_clip vision transformer exposes forward_intermediates.
        fn = getattr(visual, "forward_intermediates", None)
        if callable(fn):
            try:
                out = fn(normalized, indices=[-2] if layer_index is None else [layer_index])
                if isinstance(out, tuple):
                    intermediates = out[1] if len(out) > 1 else out[0]
                    if isinstance(intermediates, (list, tuple)) and intermediates:
                        tokens = intermediates[-1]
                        if tokens.dim() == 4:  # (B, C, H, W)
                            b, c, h, w = tokens.shape
                            tokens = tokens.flatten(2).transpose(1, 2).contiguous()
                        return tokens
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("forward_intermediates failed (%s); falling back.", exc)

        transformer = getattr(visual, "transformer", None)
        if transformer is not None and hasattr(transformer, "forward"):
            try:
                x = visual.conv1(normalized)
                b, c, h, w = x.shape
                x = x.reshape(b, c, h * w).permute(0, 2, 1)
                cls_tokens = visual.class_embedding.to(x.dtype) + torch.zeros(
                    b, 1, x.shape[-1], dtype=x.dtype, device=x.device
                )
                x = torch.cat([cls_tokens, x], dim=1)
                x = x + visual.positional_embedding.to(x.dtype)
                x = visual.ln_pre(x)
                x = x.permute(1, 0, 2)
                x = transformer(x)
                x = x.permute(1, 0, 2)
                x = visual.ln_post(x)
                return x[:, 1:, :]
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("Manual patch-token extraction failed (%s).", exc)

        raise RuntimeError(
            "Unable to extract patch tokens from the OpenCLIP vision tower; "
            "use CLIPVisionEncoder.forward and the LLaVA mm_projector instead."
        )

    # -- zero-shot classification -------------------------------------------
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if not hasattr(self._model, "encode_text"):
            raise RuntimeError("Loaded OpenCLIP model has no text encoder.")
        return self._model.encode_text(text_tokens)

    def zero_shot_logits(
        self, pixels: torch.Tensor, text_features: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """Logits = image_features @ text_features.T with CLIP logit scale."""
        image_features = self.forward(pixels, normalize=normalize).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.t()
        return logits

    # -- summary / persistence ----------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "pretrained": self.config.pretrained,
            "pretrained_path": self.config.pretrained_path,
            "resolution": self.config.resolution,
            "output_dim": self.output_dim,
            "trainable": self.config.trainable,
            "trainable_scope": self.config.trainable_scope,
            "num_trainable_parameters": self.num_trainable_parameters(),
            "precision_policy": (
                "ADDENDUM: int16 perturbations for half-precision attacks, "
                "int32 for single-precision attacks (see utils/precision.py)"
            ),
            "provenance": dict(self.config.provenance),
            "external_defaults": dict(EXTERNAL_DEFAULTS),
        }

    def save_vision_encoder(self, path: Union[str, os.PathLike]) -> str:
        """Save the (optionally fine-tuned) vision tower state dict."""
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if self.visual is not None:
            state = self.visual.state_dict()
        else:  # pragma: no cover
            state = self.state_dict()
        torch.save(
            {
                "model_name": self.config.model_name,
                "pretrained": self.config.pretrained,
                "resolution": self.config.resolution,
                "visual": state,
                "provenance": dict(self.config.provenance),
            },
            path,
        )
        LOGGER.info("Saved CLIP vision encoder to %s", path)
        return path

    def load_vision_encoder(
        self, path: Union[str, os.PathLike], strict: bool = False
    ) -> "CLIPVisionEncoder":
        payload = torch.load(str(path), map_location="cpu")
        state = payload.get("visual", payload) if isinstance(payload, dict) else payload
        target = self.visual if self.visual is not None else self
        missing, unexpected = target.load_state_dict(state, strict=strict)
        if missing or unexpected:
            LOGGER.warning(
                "load_vision_encoder: %d missing / %d unexpected keys",
                len(missing),
                len(unexpected),
            )
        return self


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def build_clip_vision_encoder(
    config: Optional[CLIPVisionConfig] = None, **kwargs: Any
) -> CLIPVisionEncoder:
    """Convenience builder; falls back to a random-weight tower when downloads fail."""
    if config is None:
        config = CLIPVisionConfig.from_dict(kwargs)
    try:
        return CLIPVisionEncoder(config)
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning(
            "Could not instantiate the OpenCLIP vision encoder (%s); falling back "
            "to a randomly initialised vision tower for smoke tests.",
            exc,
        )
        return RandomVisionEncoder(config)


def load_vision_encoder_weights(
    encoder: CLIPVisionEncoder, path: Union[str, os.PathLike]
) -> CLIPVisionEncoder:
    return encoder.load_vision_encoder(path)


class RandomVisionEncoder(CLIPVisionEncoder):
    """Deterministic random-weight vision tower (offline smoke tests only).

    Implements the same public surface as :class:`CLIPVisionEncoder` without
    requiring ``open_clip_torch`` or any downloaded checkpoint. This is used by
    ``--smoke-test`` style checks so the pipeline can be exercised end-to-end
    without network access.
    """

    def __init__(self, config: Optional[CLIPVisionConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = CLIPVisionConfig.from_dict(kwargs)
        nn.Module.__init__(self)
        self.config = config
        self._open_clip_available = False
        dim = 1024 if config.model_name.startswith("ViT-L") else 768
        self._dim = dim
        self._grid = max(1, int(config.resolution) // 14)
        self.visual = _RandomVisual(dim=dim, grid=self._grid)
        self._model = nn.Module()
        self._model.visual = self.visual  # type: ignore[attr-defined]
        self._preprocess = None
        self._tokenizer = None
        self.register_buffer(
            "mean", torch.tensor(config.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(config.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.config.provenance["backend"] = (
            "EXTERNAL: RandomVisionEncoder used for smoke tests only "
            "(%s)" % UNSPECIFIED
        )
        if config.trainable:
            for param in self.visual.parameters():
                param.requires_grad = True
        else:
            for param in self.visual.parameters():
                param.requires_grad = False

    def forward_patch_tokens(  # type: ignore[override]
        self,
        pixels: torch.Tensor,
        *,
        normalize: bool = True,
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        normalized = self.normalize(pixels) if normalize else pixels
        return self.visual.patch_tokens(normalized)


class _RandomVisual(nn.Module):
    """Tiny deterministic stand-in for an OpenCLIP vision transformer."""

    def __init__(self, dim: int = 1024, grid: int = 16) -> None:
        super().__init__()
        self.image_size = grid * 14
        self.grid_size = (grid, grid)
        self.output_dim = dim
        self.width = dim
        self.conv1 = nn.Conv2d(3, dim, kernel_size=14, stride=14, bias=False)
        self.ln_pre = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)
        self.proj = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.class_embedding = nn.Parameter(torch.randn(dim) * 0.02)
        # Deterministic init keeps smoke tests reproducible.
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.conv1.weight.normal_(generator=gen)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.ln_pre(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_tokens(x)
        pooled = self.ln_post(tokens.mean(dim=1))
        return pooled @ self.proj


# ---------------------------------------------------------------------------
# Self test (no downloads / no CUDA required)
# ---------------------------------------------------------------------------


def _self_test(verbose: bool = True) -> Dict[str, Any]:
    torch.manual_seed(0)
    cfg = CLIPVisionConfig(trainable=True, trainable_scope="all", resolution=224)
    enc = RandomVisionEncoder(cfg)
    assert enc.config.resolution == 224

    pixels = torch.rand(2, 3, 224, 224)
    feats = enc(pixels)
    assert feats.shape[0] == 2, feats.shape
    assert feats.shape[-1] == enc.output_dim

    patches = enc.forward_patch_tokens(pixels)
    assert patches.dim() == 3 and patches.shape[0] == 2, patches.shape

    # Normalization round trip and raw-pixel semantics.
    normed = enc.normalize(pixels)
    assert torch.allclose(enc.denormalize(normed), pixels, atol=1e-5)
    assert not torch.allclose(normed, pixels)

    # Trainability scopes.
    frozen = RandomVisionEncoder(CLIPVisionConfig(trainable=False))
    assert frozen.num_trainable_parameters() == 0
    assert enc.num_trainable_parameters() > 0

    conf = CLIPVisionConfig.from_dict({"trainable_scope": "all"}, resolution=224)
    assert conf.resolution == 224
    assert conf.provenance["resolution"].startswith("ADDENDUM")

    summary = enc.summary()
    assert summary["model_name"] == "ViT-L-14"
    assert "int16" in summary["precision_policy"] and "int32" in summary["precision_policy"]

    report = {
        "ok": True,
        "output_dim": enc.output_dim,
        "patch_tokens": tuple(patches.shape),
        "num_trainable_parameters": enc.num_trainable_parameters(),
        "external_defaults": EXTERNAL_DEFAULTS,
    }
    if verbose:
        LOGGER.info("clip_vision_encoder self-test passed: %s", report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="CLIP vision encoder utility (OpenCLIP ViT-L/14@224)."
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run the offline smoke test."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the real OpenCLIP vision encoder (downloads weights).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_test or not args.build:
        report = _self_test()
        print(report)
        return 0

    cfg = CLIPVisionConfig(
        model_name=args.model_name,
        pretrained=args.pretrained,
        resolution=args.resolution,
        device=args.device,
    )
    enc = build_clip_vision_encoder(cfg)
    print(enc.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
