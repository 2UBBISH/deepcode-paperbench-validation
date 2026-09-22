"""Robust CLIP -- reproduction of

    Schlarmann, Singh, Croce, Hein.
    "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for
    Robust Large Vision-Language Models", ICML 2024.

The package is organised along the contributions of the paper:

``robust_clip.models``
    Loading of the OpenAI CLIP vision encoder in the two "views" that matter for
    the paper: the (projected) *class token* used by the FARE / TeCoA training
    losses and the *penultimate-layer patch tokens* consumed by LLaVA and
    OpenFlamingo.
``robust_clip.training``
    The FARE unsupervised adversarial fine-tuning scheme (Eq. (3)) and the
    supervised TeCoA baseline of Mao et al. (2023) (Eq. (2)), together with the
    PGD inner maximisation used in both.
``robust_clip.attacks``
    APGD (Croce & Hein, 2020) in the ``linf`` threat model, including the
    half-precision / single-precision ensemble attack pipeline used for the
    LVLM evaluations (Sec. 4.1) and the stealthy targeted attacks (Sec. 4.2).
``robust_clip.lvlm``
    LLaVA-1.5 7B and OpenFlamingo-9B with a swappable (Open)CLIP vision encoder.
``robust_clip.eval``
    Zero-shot classification (Sec. 4.3), captioning / VQA (Sec. 4.1), POPE
    (Sec. 4.4), SQA-I (Sec. 4.4) and jailbreaking (Sec. 4.4) evaluations.
"""
