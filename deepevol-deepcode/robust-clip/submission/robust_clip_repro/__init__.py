"""Robust CLIP reproduction package.

Unsupervised adversarial fine-tuning of vision embeddings for robust large
vision-language models (Robust CLIP), together with the robustness-evaluation
and attack procedures for CLIP-based VLMs (LLaVA-1.5 7B, OpenFlamingo).

The package is organised as follows::

    robust_clip_repro/
        main.py                    # CLI entry point: train/eval/attack dispatch
        train_robust_clip.py       # Robust CLIP unsupervised adversarial fine-tuning
        eval_imagenet.py           # zero-shot clean/robust ImageNet evaluation
        eval_vqa.py                # TextVQA / POPE / SQA-I scheduled VQA attacks
        eval_captioning.py         # OpenFlamingo captioning robustness (worst-case CIDEr)
        eval_jailbreak.py          # universal targeted jailbreak attack + grading export
        attacks/                   # PGD, APGD, jailbreak, VQA schedule, captioning suite
        models/                    # LLaVA-1.5 7B (OpenCLIP ViT-L/14@224), OpenFlamingo
        data/                      # ImageNet (HF), VQA benchmarks, COCO, jailbreak assets
        metrics/                   # classification, VQA, CIDEr, jailbreak grading
        utils/                     # precision policy, normalization, logging
        configs/                   # YAML configs (values tagged with provenance)
        prompts/                   # LLaVA / OpenFlamingo prompt templates

Provenance policy
-----------------
Every hyper-parameter that the paper/Addendum does not state is *never*
invented.  Such values are
(a) exposed through config/CLI, (b) tagged with the sentinel string
``"UNSPECIFIED_BY_ADDENDUM"`` (or ``"EXTERNAL_DEFAULT"``), and (c) logged as
externally supplied by the harnesses.

Addendum invariants implemented throughout the package
-----------------------------------------------------
* PGD: uniform random initialisation inside the ``l_inf`` ball, momentum
  ``0.9``, gradient normalisation followed by an element-wise ``sign``, and
  projection around **non-normalized** pixel inputs.
* Precision policy: half-precision attacks store perturbations as ``int16``,
  single-precision attacks as ``int32``.
* Jailbreak: 5000 iterations, ``alpha = 1/255``, no momentum, a single source
  image ``clean.jpeg``, targets from ``derogatory_corpus.csv``.
* VQA: low-precision (int16) attacks on the top-5 most frequent ground truths,
  arg-min ground-truth selection, a high-precision (int32) attack on that
  ground truth, a targeted attack on the lower-case string ``"maybe"`` with a
  clean initialisation, and a targeted attack on the capitalised string
  ``"Word"`` with a separate clean initialisation (skipped on TextVQA).
* Captioning: CIDEr is recomputed immediately after *every* attack and the
  per-sample worst case is retained.
* ImageNet: loaded via HuggingFace ``datasets`` with ``trust_remote_code=True``.
"""

from __future__ import annotations

__version__ = "0.1.0"

PACKAGE_NAME = "robust_clip_repro"

#: Sentinel used to mark hyper-parameters the paper/Addendum does not state.
UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

#: Sentinel for values supplied externally (config/CLI) rather than by the paper.
EXTERNAL_DEFAULT = "EXTERNAL_DEFAULT"

__all__ = [
    "__version__",
    "PACKAGE_NAME",
    "UNSPECIFIED",
    "EXTERNAL_DEFAULT",
]
