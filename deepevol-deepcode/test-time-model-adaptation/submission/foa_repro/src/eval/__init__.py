"""Evaluation sub-package for the FOA reproduction.

Exposes the metrics module implementing the two quantities reported throughout
the paper:

* top-1 **classification accuracy** over the OOD test stream, and
* **Expected Calibration Error (ECE)** computed with equal-width probability
  bins (default 15 bins, a documented default, see Appendix D / Table 16),
  following the calibration-error definition of Naeini et al. 2015.

The metrics are computed on the logits produced by the FOA pipeline
(frozen ViT-Base + learnable input prompt + back-to-source activation shifting)
and never require gradients.
"""

from __future__ import annotations

__all__ = ["metrics"]

__version__ = "0.1.0"
