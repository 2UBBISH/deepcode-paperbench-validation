"""FOA (Forward-Optimization Adaptation) reproduction package.

Test-Time Model Adaptation with Only Forward Passes.

Sub-packages
------------
models       : frozen ViT-Base backbone + learnable input prompt injection
method       : FOA algorithm (fitness, CMA-ES, activation shifting, main loop)
data         : ImageNet-1K / ImageNet-C / -R / -V2 / -Sketch loaders + streams
eval         : evaluation metrics (accuracy, ECE)
baselines    : thin wrappers around official TTA baselines (LAME, T3A, TENT, ...)
quantization : PTQ4ViT adapter for 8/6-bit ViT-Base
utils        : config loading, seeding, logging, checkpoint I/O

Everything in FOA is backpropagation-free: model weights are frozen and no
gradients are ever computed inside :mod:`src.method`.
"""

__version__ = "0.1.0"

__all__ = [
    "models",
    "method",
    "data",
    "eval",
    "baselines",
    "quantization",
    "utils",
]
