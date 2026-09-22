"""SAPG: Split and Aggregate Policy Gradients.

An on-policy RL algorithm that scales to tens of thousands of parallel
environments by splitting them into M blocks trained by diverse
leader/follower policies (shared actor backbone B_theta and critic backbone
C_psi conditioned on per-policy latents phi_j), then aggregating follower
data into a single leader update via importance-sampled off-policy PPO.
"""

__version__ = "0.1.0"
