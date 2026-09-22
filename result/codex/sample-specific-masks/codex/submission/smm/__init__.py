"""Sample-specific Multi-channel Masks (SMM) for visual reprogramming.

This package reproduces *Sample-specific Masks for Visual Reprogramming-based
Prompting* (Cai et al., ICML 2024).

Main components
---------------
``smm.mask_generator``  lightweight CNN mask generator ``f_mask`` + patch-wise
                        interpolation module (paper Section 3.2, 3.3).
``smm.reprogram``       input transformation ``f_in`` of SMM and of the
                        shared-mask baselines (Pad / Narrow / Medium / Full).
``smm.label_mapping``   output mapping ``f_out``: Rlm / Flm / Ilm
                        (paper Section 2.3 and Appendix A.4).
``smm.models``          frozen ImageNet pre-trained ResNet-18/50 and ViT-B/32.
``smm.datasets``        the 11 target datasets with the paper's transforms.
``smm.train``           training loop of Algorithm 1 (plus ablations).
``smm.evaluate``        evaluation of a trained reprogramming module.
``smm.theory``          Theorem 4.2 / Proposition 4.3 (+ B.1) and numerical
                        verification of the hypothesis-space inclusion.
"""

__all__ = [
    "mask_generator",
    "reprogram",
    "label_mapping",
    "models",
    "datasets",
    "train",
    "evaluate",
    "theory",
]

__version__ = "1.0.0"
