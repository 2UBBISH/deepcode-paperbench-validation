"""Quantization sub-package for the FOA reproduction.

Provides the PTQ4ViT adapter (Section 4.2 / Table 4 / Table 17) that produces
8-bit and 6-bit post-training-quantized ViT-Base backbones using 32 randomly
selected ImageNet-1K training samples for activation calibration, as specified
in Appendix B.2.

The FOA loop is unchanged for quantized models: it only performs forward
passes, so no backward pass or gradient bookkeeping is needed.  The official
PTQ4ViT implementation (https://github.com/hahnyuan/PTQ4ViT) is used when it is
importable, otherwise a self-contained twin-uniform min-MSE post-training
quantization emulation is used.
"""

from __future__ import annotations

__all__ = ["ptq4vit_adapter"]

__version__ = "0.1.0"
