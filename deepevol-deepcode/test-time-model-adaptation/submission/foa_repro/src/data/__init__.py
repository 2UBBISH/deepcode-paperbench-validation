"""Data loading sub-package for the FOA reproduction.

This package provides the dataset/stream plumbing used by the online
test-time-adaptation loops of the paper *"Test-Time Model Adaptation with Only
Forward Passes"* (Forward-Optimization Adaptation, FOA).

Submodules
----------
``datasets``
    Builders for the four OOD benchmarks (ImageNet-C, ImageNet-R,
    ImageNet-V2 Matched-Frequency, ImageNet-Sketch) plus ImageNet-1K via
    HuggingFace, exposing ordered single-pass :class:`OnlineStream` iterators
    with standard ViT 224x224 preprocessing (Section 4 "Datasets and Models",
    Appendix B.1).
``corruption_stream``
    Single-pass *non-i.i.d.* test streams (class-ordered label shift and
    mixed-domain corruption streams) reproducing the Section 4.4 / Table 11
    robustness protocol.

All loading here is pure data plumbing: no gradients are ever computed and no
model parameter is touched.
"""

from __future__ import annotations

__all__ = ["datasets", "corruption_stream"]

__version__ = "0.1.0"
