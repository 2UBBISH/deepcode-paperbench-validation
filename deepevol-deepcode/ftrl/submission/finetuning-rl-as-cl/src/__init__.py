"""Fine-tuning RL models is secretly a forgetting mitigation problem.

Top-level package for the code base reproducing:

    Wołczyk, Piekos, Myers, Ostaszewski, Kicinski, Milos, Baker, Delétang,
    Chojnacki, Sarrot, Vinyals, Veness, Osindero, Shanahan, Zolna, Cao, Ortega
    "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
    Mitigation Problem" (2024).

The package is split into environment-agnostic pieces and per-environment
code:

* ``src.common``   – config / seeding / logging / checkpointing scaffolding.
* ``src.retention``– the four knowledge-retention mechanisms (EWC, BC,
  Kickstarting, Episodic Memory) plus the diagonal Fisher estimator.
* ``src.nethack``  – NetHack Human Monk fine-tuning (APPO, Table 1).
* ``src.montezuma``– Montezuma's Revenge M1 (PPO + RND) and M2 (BC) pipeline.
* ``src.robotic_sequence`` – Meta-World RoboticSequence (SAC, Table 3).
* ``src.toy``      – toy two-state MDP and AppleRetrieval sanity checks.
* ``src.analysis`` – CKA, forward transfer, log-likelihood, visualisations.

Only ``torch`` / ``numpy`` are hard requirements at import time; everything
environment specific is imported lazily so that the retention losses and the
toy examples can be exercised on a CPU-only machine.
"""

from __future__ import annotations

__all__ = [
    "common",
    "retention",
    "nethack",
    "montezuma",
    "robotic_sequence",
    "toy",
    "analysis",
]

__version__ = "1.0.0"
