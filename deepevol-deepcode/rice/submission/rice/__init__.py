"""RICE: A Refining Scheme for Reinforcement Learning with Explanation.

Reference:
    Cheng et al. "RICE: A Refining Scheme for Reinforcement Learning with
    Explanation." ICML 2024 (PMLR 235).

The package implements the two-stage RICE pipeline:

    Stage 1 (Explanation):  train a mask network with vanilla PPO plus a
        blinding bonus ``alpha * a_t^m`` (Algorithm 1).  The mask network
        assigns a step-level importance score to every visited state, i.e. the
        probability that the mask outputs "keep" (a_t^m = 0).

    Stage 2 (Refining):  refine the frozen pre-trained policy ``pi`` with PPO
        on a mixed initial state distribution
        ``mu(s) = beta * d_rho^pihat(s) + (1 - beta) * rho(s)`` and with a
        Random Network Distillation intrinsic reward
        ``lambda * ||f(s_{t+1}) - fhat(s_{t+1})||^2`` (Algorithm 2).
"""

__version__ = "0.1.0"

__all__ = ["envs", "models", "explanation", "refining", "baselines", "utils"]
