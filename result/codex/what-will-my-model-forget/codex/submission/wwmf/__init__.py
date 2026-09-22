"""Code release for

Xisen Jin, Xiang Ren. "What Will My Model Forget? Forecasting Forgotten
Examples in Language Model Refinement." ICML 2024.

The package is organised along the sections of the paper:

``wwmf.data``         construction of ``D_PT`` / ``D_R`` and their splits (Sec. 4.1)
``wwmf.models``       the base PTLMs and the head-only / LoRA / full-FT refiners (Sec. 4.1)
``wwmf.forecasting``  the three forecasting models (Sec. 3.1, 3.2, 3.3)
``wwmf.refinement``   replay based model refinement and the sequential stream (Sec. 5.2)
``wwmf.evaluation``   metrics (F1, edit success rate, EM drop ratio) and table builders
``wwmf.analysis``     computational complexity (Sec. 5.3) and logit-change analysis (Fig. 2a)
"""

__version__ = "0.1.0"
