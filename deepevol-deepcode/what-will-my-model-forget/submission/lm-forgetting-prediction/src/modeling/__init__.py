"""``src.modeling`` -- base language models, refinement engine, encoding function ``h``, and caches.

This sub-package implements every component of the "What Will My Model Forget?"
pipeline that touches the pretrained seq2seq model itself:

* :mod:`src.modeling.base_lm`
    Thin wrapper :class:`~src.modeling.base_lm.BaseLM` around the four base PTLMs
    used in the paper (``BART0_L``, ``FLAN-T5_L``, ``FLAN-T5_3B``, ``FLAN-T5_small``).
    Exposes greedy generation (EM / ``D_PT_hat`` / ``D_R`` collection) and
    teacher-forced per-output-token logits (the logit streams ``f0(x_j)`` and
    ``f_i(x_j)`` used by the logit-change-transfer forecaster).  It is a frozen
    ``f_0`` interface -- it never trains.

* :mod:`src.modeling.refinement`
    The model-refinement engine (Sec. 2, Sec. 4.1).  Given ``f_0`` and one online
    example ``<x_i, y_i>`` it takes ``K`` gradient steps under one of three
    tuning setups and returns ``f_i``:

    ``head``    -> ``K = 100`` (LM head only, optionally untied)
    ``lora``    -> ``K = 30``  (r=16, alpha=32, dropout=0.1, bias="none", targets ['q','v'])
    ``full_ft`` -> ``K = 30``  (all parameters)

    Learning rates follow Sec. 4.1 / Appendix B (LoRA & Full FT 1e-5 for
    ``BART0_L``, 1e-4 for the FLAN-T5 models; sequential refinement uses
    1e-6 / 1e-5; head-only uses 1e-3 / 1e-4).  Replay-based refinement hooks
    (scheduled distillation against a frozen copy of the base PTLM) live here too.

* :mod:`src.modeling.encoder_h`
    The encoding function ``h`` (Sec. 3.2, Sec. 3.3, Appendix B): a trainable
    base-LM backbone (``BART0`` for the BART0 experiments, ``FLAN-T5_small`` for
    the T5 experiments) followed by a freshly initialized 2-layer MLP.  Two
    output modes are exposed -- token-level ``h(x, y) in R^{T x d}`` for the
    logit-change kernel and the mean-pooled vector ``R^d`` for the
    representation-based forecaster.  LM components use LR 1e-5, MLP LR 1e-4.

* :mod:`src.modeling.caches`
    Persistent caching layer that makes forecasting cheap: the top-``k`` logits of
    ``f_0(x_j)`` (k=100 per output token of ``y_j``), ``h(x_j, y_j)`` for every
    upstream example ``x_j in D_PT_hat``, and the frequency prior ``b_j`` are
    precomputed once and reloaded at forecast time.  Inference therefore costs
    ``O(|D_PT_hat|)`` without re-running the LM (Sec. 3.2 "Efficient Inference",
    Sec. 3.3, Appendix F Algorithms 3-4).

Heavy third-party dependencies (``torch``, ``transformers``, ``peft``) are
imported lazily inside the leaf modules so that this package remains importable
in CPU-only / library-light contexts (config parsing, table formatting and the
``--self-test`` smoke paths).
"""

from __future__ import annotations

__all__ = [
    "base_lm",
    "refinement",
    "encoder_h",
    "caches",
]
