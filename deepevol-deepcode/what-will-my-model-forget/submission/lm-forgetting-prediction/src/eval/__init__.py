"""Evaluation package for the *What Will My Model Forget?* reproduction.

This sub-package contains the metric library and the artifact-driven evaluation
harness used to reproduce the paper's numbers:

* :mod:`src.eval.metrics`
    Canonical, dependency-light metric implementations shared by every other
    module in the project:

    - binary forgetting-forecast precision / recall / F1 (percent scale,
      matching Table 1 e.g. ``BART0 Head Representation F1 = 79.32``),
    - SQuAD-2.0 exact match (Table 7 base-EM sanity numbers, e.g.
      ``BART0_L = 50.50``), re-exported from :mod:`src.data.em_eval`,
    - Edit Success Rate (fraction of :math:`D_R` errors that the refined model
      answers correctly after refinement),
    - EM Drop Ratio ``(EM(D_PT, f_i) - EM(D_PT, f_0)) / EM(D_PT, f_0)``
      (Tables 3 and 4),
    - per-task bucket metrics for the BART0 ID/OOD split (Table 2),
    - continual-stream running averages for Figure 3
      (:func:`average_metrics_up_to_step`).

* :mod:`src.eval.evaluate`
    The table/figure driver.  It reads the persisted artifacts produced by the
    pipeline (``forecast_summary.json``, ``pairs*.jsonl``,
    ``replay_summary.json``, ``stream_history.jsonl``, EM summaries) and
    renders Tables 1-4, Table 7 and the Figure 3 curves, also embedding the
    paper's reference values for offline comparison.

Heavy third-party dependencies (``torch``, ``matplotlib``, the model wrappers)
are imported lazily inside the leaf modules/functions so this package stays
importable in a CPU-only environment (config parsing, table formatting and the
``--self-test`` modes).
"""

from __future__ import annotations

__all__ = ["metrics", "evaluate"]
