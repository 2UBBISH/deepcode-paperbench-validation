"""opt_for_pinns.src

Core library for reproducing "Challenges in Training PINNs: A Loss Landscape
Perspective".

Modules
-------
pdes            : Convection / Reaction / Wave PDE definitions (residual, IC, BC, exact).
model           : MLP surrogate (tanh, 3 hidden layers, Xavier-normal, zero biases).
loss            : PINN loss (residual + IC + BC) and L2RE metric.
data            : Grid / point sampling (255x100 interior, 257 IC, 101 BC).
train           : Training loops (Adam, L-BFGS, Adam+L-BFGS, NNCG, GD).
hessian         : Hessian-vector products (Pearlmutter) + SLQ spectral density.
lbfgs_precond   : L-BFGS unrolling + preconditioned matvec (Algorithms 2 & 3).
utils           : Seeding, logging, timing, checkpointing.
optimizers      : NNCG (Algorithm 4), Nystrom (5), PCG (6), Armijo (7).
"""

from __future__ import annotations

__all__ = [
    "pdes",
    "model",
    "loss",
    "data",
    "train",
    "hessian",
    "lbfgs_precond",
    "utils",
    "optimizers",
]
