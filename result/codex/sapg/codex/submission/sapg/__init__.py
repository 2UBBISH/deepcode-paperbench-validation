"""SAPG: Split and Aggregate Policy Gradients (Singla, Agarwal, Pathak, ICML 2024).

Clean-room re-implementation of the algorithm, its baselines and its analyses,
written from the paper (paper/paper.md) plus the clarifications in
paper/addendum.md.

Layout
------
``sapg.algorithms``  SAPG itself (leader/follower aggregation, off-policy
                     importance-sampled PPO updates, latent conditioning) plus
                     the PPO / PBT (DexPBT) / PQL baselines used in the paper.
``sapg.envs``        The 5 benchmark tasks (AllegroKuka: Regrasping, Throw,
                     Reorientation; ShadowHand / AllegroHand in-hand
                     reorientation) written against IsaacGym, and a small
                     CPU-only "multi-modal reaching" suite used for smoke tests
                     and qualitative checks of the paper's claims.
``sapg.analysis``    The diversity metrics of Sec. 6.4 (PCA reconstruction
                     error, MLP reconstruction error) and the batch-size study
                     of Figure 2.
``sapg.utils``       Shared plumbing (config, logging, plotting, state
                     recording, running statistics).
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
