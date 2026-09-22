"""``src.replay`` -- replay-based model-refinement protocols.

This sub-package implements the replay half of the paper *"What Will My Model
Forget? Forecasting Forgotten Examples in Language Model Refinement"*
(Sec. 4.2, Sec. 5.2 and Appendix D.2).

The protocol is: sequentially fix the errors of an instruction-tuned seq2seq LM
``f0`` on one online example ``<x_i, y_i>`` at a time (Head / LoRA / Full-FT
refinement), while every ``n`` steps replaying a small mini-batch of upstream
examples drawn from ``D_PT_hat``.  The replayed examples are selected by one of
six strategies and the refinement step additionally minimises a distillation
loss against the frozen base PTLM outputs, which is what reduces catastrophic
forgetting.

Replay strategies (``REPLAY_METHODS``):

* ``vanilla``      -- no replay at all (the upper bound on EM drop);
* ``random``       -- uniform random upstream examples;
* ``threshold``    -- frequency-threshold baseline (Eq. 1);
* ``logit``        -- trainable logit-change-transfer forecaster (Sec. 3.2);
* ``representation`` -- representation-based forecaster + frequency prior (Sec. 3.3);
* ``gt``           -- oracle selecting examples that are *actually* forgotten
  (upper bound on replay utility).

The heavy lifting lives in :mod:`src.replay.refinement_replay`, which exposes
the selector hierarchy (:class:`ReplaySelector` and friends), the
:class:`SequentialReplayRefinement` driver used by Tables 3/4, the
:class:`ReplayRefinementResult` container, and helpers to build cache-only
score providers for the forecast-driven selectors.

All third-party model/tensor dependencies (``torch``, ``transformers``) are
imported lazily inside the leaf module so that this package stays importable in
lightweight, CPU-only contexts (config parsing, self-tests).
"""

from __future__ import annotations

__all__ = ["refinement_replay"]
