"""Network architectures used throughout the paper.

The architectures follow Table 7 of the paper ("The network structures of the
models used in our experiments") and the appendix notes:

* ``ConvNet`` is the CNN of Zhou et al. (2022) as pointed out by the addendum;
  it is the network used for the Figure 1 study (Section 2.1 / Appendix C.3)
  and for the Section 5.1 MNIST-S study (Table 1).
* ``LeNet`` is the network used for F-MNIST (proxy and target).
* ``SVHNCNNInner`` / ``SVHNCNNTarget`` and ``CIFAR10CNNInner`` are the simple
  CNNs of Table 7.
* ``ResNet18`` (CIFAR style) is the target network for CIFAR-10.
* ``WideResNet`` and ``ViT`` are the cross-architecture target networks for
  SVHN (Table 6).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalisation helper (mirrors the ConvNet reference implementation)
# ---------------------------------------------------------------------------
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# ConvNet of Zhou et al. (2022) -- used for MNIST / MNIST-S
# ---------------------------------------------------------------------------
class ConvNet(nn.Module):
    """Two conv/dropout/max-pool blocks followed by two dense layers.

    This is the ``ConvNet`` class from Zhou et al. (2022)
    (Probabilistic-Bilevel-Coreset-Selection), referenced by the addendum.
    """

    def __init__(self, output_dim: int = 10, maxpool: bool = True,
                 base_hid: int = 32, in_channels: int = 1,
                 normalize_mean=(0.1307,), normalize_std=(0.3081,)):
        super().__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(in_channels, base_hid, 5, 1)
        self.dp1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(base_hid, base_hid * 2, 5, 1)
        self.dp2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * base_hid * 2, base_hid * 4)
        self.dp3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(base_hid * 4, output_dim)
        self.maxpool = maxpool
        self.normalize = Normalize(normalize_mean, normalize_std)

    def embed(self, x):
        x = self.normalize(x)
        x = F.relu(self.dp1(self.conv1(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dp2(self.conv2(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.dp3(self.fc1(x)))
        return x

    def forward(self, x, return_feat: bool = False):
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out


# ---------------------------------------------------------------------------
# LeNet -- used for F-MNIST (proxy and target network)
# ---------------------------------------------------------------------------
class LeNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 normalize_mean=(0.2860,), normalize_std=(0.3530,),
                 normalize: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.ifnormalize = normalize
        self.normalize = Normalize(normalize_mean, normalize_std)

    def forward(self, x):
        if self.ifnormalize:
            x = self.normalize(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Simple CNNs of Table 7
# ---------------------------------------------------------------------------
class SVHNCNNInner(nn.Module):
    """CNN for SVHN (inner loop) -- Table 7, left column."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 normalize_mean=(0.4377, 0.4438, 0.4728),
                 normalize_std=(0.1980, 0.2010, 0.1970)):
        super().__init__()
        self.normalize = Normalize(normalize_mean, normalize_std)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SVHNCNNTarget(nn.Module):
    """CNN for SVHN (trained on coresets) -- Table 7, middle column."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 normalize_mean=(0.4377, 0.4438, 0.4728),
                 normalize_std=(0.1980, 0.2010, 0.1970)):
        super().__init__()
        self.normalize = Normalize(normalize_mean, normalize_std)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CIFAR10CNNInner(nn.Module):
    """CNN for CIFAR-10 (inner loop) -- Table 7, right column."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 normalize_mean=(0.4914, 0.4822, 0.4465),
                 normalize_std=(0.2470, 0.2435, 0.2616)):
        super().__init__()
        self.normalize = Normalize(normalize_mean, normalize_std)
        self.conv1 = nn.Conv2d(in_channels, 32, 5)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# ResNet-18 for CIFAR-10 (target network after coreset selection)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18 (3x3 stem, no max pooling)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 normalize_mean=(0.4914, 0.4822, 0.4465),
                 normalize_std=(0.2470, 0.2435, 0.2616)):
        super().__init__()
        self.normalize = Normalize(normalize_mean, normalize_std)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# WideResNet for SVHN (cross architecture evaluation, Table 6)
# ---------------------------------------------------------------------------
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.equal = in_planes == out_planes
        if not self.equal:
            self.conv_short = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                        stride=stride, bias=False)
        else:
            self.conv_short = None

    def forward(self, x):
        out = F.relu(self.bn1(x))
        short = x if self.equal else self.conv_short(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + short


class WideResNet(nn.Module):
    """WideResNet (Zagoruyko & Komodakis, 2016) used for SVHN."""

    def __init__(self, depth: int = 16, widen_factor: int = 4,
                 num_classes: int = 10, in_channels: int = 3, dropout: float = 0.0,
                 normalize_mean=(0.4377, 0.4438, 0.4728),
                 normalize_std=(0.1980, 0.2010, 0.1970)):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        self.normalize = Normalize(normalize_mean, normalize_std)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1,
                               bias=False)
        self.block1 = self._make_block(16, 16 * k, n, 1, dropout)
        self.block2 = self._make_block(16 * k, 32 * k, n, 2, dropout)
        self.block3 = self._make_block(32 * k, 64 * k, n, 2, dropout)
        self.bn1 = nn.BatchNorm2d(64 * k)
        self.fc = nn.Linear(64 * k, num_classes)

    @staticmethod
    def _make_block(in_planes, out_planes, n, stride, dropout):
        layers = [WideBasicBlock(in_planes, out_planes, stride, dropout)]
        for _ in range(n - 1):
            layers.append(WideBasicBlock(out_planes, out_planes, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize(x)
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.bn1(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Vision Transformer (ViT-small) for SVHN (cross architecture evaluation)
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """ViT-small (Dosovitskiy et al., 2021) adapted to 32x32 inputs."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=384, depth=12, num_heads=6,
                 dropout=0.0,
                 normalize_mean=(0.4377, 0.4438, 0.4728),
                 normalize_std=(0.1980, 0.2010, 0.1970)):
        super().__init__()
        self.normalize = Normalize(normalize_mean, normalize_std)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels,
                                          embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.normalize(x)
        b = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 0]
        return self.head(x)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "convnet": ConvNet,
    "lenet": LeNet,
    "svhn_cnn_inner": SVHNCNNInner,
    "svhn_cnn_target": SVHNCNNTarget,
    "cifar10_cnn_inner": CIFAR10CNNInner,
    "resnet18": ResNet18,
    "wideresnet": WideResNet,
    "vit": ViT,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}', available: "
                       f"{sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
