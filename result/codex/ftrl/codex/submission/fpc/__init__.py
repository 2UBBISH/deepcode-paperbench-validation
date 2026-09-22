"""fpc -- Fine-tuning RL models is (secretly) a Forgetting-of-Pretrained-Capabilities
mitigation problem.

This package contains the code used to reproduce

    Wolczyk, Cupial, Ostaszewski, Bortkiewicz, Zajac, Pascanu, Kucinski, Milos,
    "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
    Mitigation Problem", ICML 2024.

The package is organised around the two contributions of the paper:

1. ``fpc.retention`` -- the knowledge-retention methods (EWC, behavioral cloning,
   kickstarting, episodic memory) that mitigate forgetting of pre-trained
   capabilities (FPC).
2. ``fpc.nethack`` / ``fpc.montezuma`` / ``fpc.metaworld`` -- the three
   experimental domains (NetHack, Montezuma's Revenge, RoboticSequence) together
   with their training loops, pre-training procedures and evaluations.

``fpc.toy`` implements the two analytic/toy environments (two-state MDPs and
AppleRetrieval) and ``fpc.analysis`` implements the forgetting diagnostics
(CKA, forward transfer, expert log-likelihoods, level-visitation densities).
"""

__version__ = "1.0.0"

__all__ = ["retention", "nethack", "montezuma", "metaworld", "toy", "analysis"]
