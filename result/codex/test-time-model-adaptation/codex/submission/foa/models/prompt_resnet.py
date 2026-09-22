"""ResNet-50 backbone with a *convolutional* prompt (Table 10 of the paper).

"For ResNet-50, we feed the original image to a learnable 7x7 Conv layer to generate
prompts with the same size as the image and then add prompts to the image as the model's
input."  The number of learnable parameters is deliberately tiny (3x3x7x7 + 3 = 1326),
which keeps the CMA-ES search space small even for a ConvNet.

Because a ConvNet has no CLS token, the "CLS features" that FOA regularises are the
globally average-pooled activations of the stem and of every residual stage, the last one
being the input of the classifier head.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptableModel

RESNET50_CHECKPOINT = "resnet50.a1_in1k"


class PromptResNet(AdaptableModel):
    """ResNet-50 (timm) whose input is the image plus a learnable 7x7-conv prompt."""

    def __init__(self, resnet: nn.Module, kernel_size: int = 7, conv_stem_scale: float = 1.0):
        super().__init__()
        self.backbone = resnet
        self.num_stages = 4
        self.num_layers = self.num_stages + 1          # stem + 4 stages
        self.embed_dim = int(getattr(resnet, "num_features", 2048))
        self.num_classes = int(getattr(resnet, "num_classes", 1000))
        self.kernel_size = kernel_size
        self.param_shape = (3, 3, kernel_size, kernel_size)
        self._prompt_dim = 3 * 3 * kernel_size * kernel_size + 3
        self.conv_stem_scale = conv_stem_scale

    # ----------------------------------------------------------------------------------
    @property
    def num_prompts(self) -> int:
        return 1

    @property
    def prompt_dim(self) -> int:
        """Dimension of the flattened 7x7 prompt convolution."""
        return self._prompt_dim

    def initial_prompt_vector(self, device=None, dtype=None) -> torch.Tensor:
        w = torch.zeros(self.param_shape, dtype=dtype or torch.float32)
        nn.init.uniform_(w, -1.0 / self.prompt_dim, 1.0 / self.prompt_dim)
        b = torch.zeros(3, dtype=dtype or torch.float32)
        v = torch.cat([w.reshape(-1), b])
        return v.to(device) if device is not None else v

    # ----------------------------------------------------------------------------------
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def _apply_prompt(self, images: torch.Tensor, prompt: Optional[torch.Tensor]) -> torch.Tensor:
        if prompt is None:
            return images
        flat = prompt.reshape(-1)
        w = flat[: 3 * 3 * self.kernel_size * self.kernel_size].reshape(self.param_shape)
        b = flat[3 * 3 * self.kernel_size * self.kernel_size :]
        pad = self.kernel_size // 2
        delta = F.conv2d(images, w.to(images.dtype), b.to(images.dtype), padding=pad)
        return images + self.conv_stem_scale * delta

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self._apply_prompt(tokens, prompt)
        bb = self.backbone
        feats: List[torch.Tensor] = []
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.act1(x)
        x = bb.maxpool(x)
        feats.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        for stage in (bb.layer1, bb.layer2, bb.layer3, bb.layer4):
            x = stage(x)
            feats.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        pooled = bb.global_pool(x).flatten(1) if hasattr(bb, "global_pool") else torch.flatten(x, 1)
        pooled = bb.fc_norm(pooled) if getattr(bb, "fc_norm", None) is not None else pooled
        e_n = pooled
        if shift is not None:
            e_n = e_n + shift.reshape(1, -1).to(e_n.dtype)
        if return_layers:
            feats[-1] = e_n
            return bb.fc(e_n) if hasattr(bb, "fc") else bb.head(e_n), e_n, feats
        head = bb.fc if hasattr(bb, "fc") else bb.head
        return head(e_n), e_n, None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_tokens(images)
        return logits


def build_prompt_resnet(checkpoint: str = RESNET50_CHECKPOINT, pretrained: bool = True) -> PromptResNet:
    import timm

    backbone = timm.create_model(checkpoint, pretrained=pretrained)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return PromptResNet(backbone)
