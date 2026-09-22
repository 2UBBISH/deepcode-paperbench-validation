"""SAPG: Split and Aggregate Policy Gradients.

A new on-policy RL algorithm that scales to tens of thousands of parallel
environments by splitting them into M blocks, training M diverse policies
(1 leader + M-1 followers) sharing a backbone conditioned on per-policy
latent parameters phi_j, and aggregating follower data into the leader via
importance-sampled off-policy PPO updates with a mu correction term.
"""

__version__ = "0.1.0"
