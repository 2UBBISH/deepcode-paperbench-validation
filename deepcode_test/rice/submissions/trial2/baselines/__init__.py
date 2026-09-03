"""
Baseline methods for RICE comparison experiments.

This package contains implementations of baseline methods compared against
RICE in the paper:

- StateMask: Original mask network training using Lagrangian method
- Jump-Start RL (JSRL): Uses a guide policy to initialize exploration
- Self-Imitation Learning (SIL): Prioritizes past successful experiences
- Random Explanation: Randomly selects states as critical; otherwise same
  refining pipeline as RICE
"""

__all__ = [
    "statemask",
    "jsrl",
    "sil",
    "random_explanation",
]