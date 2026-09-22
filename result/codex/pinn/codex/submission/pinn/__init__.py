"""Reproduction of "Challenges in Training PINNs: A Loss Landscape Perspective"
(Rathore, Lei, Frangella, Lu, Udell, ICML 2024).

The package is organised as follows

``pinn.problems``   differential equations (convection, reaction, wave)
``pinn.data``       sampling of residual / initial / boundary points
``pinn.models``     the tanh MLP parametrisation used for ``u(x; w)``
``pinn.losses``     the PINN objective of Eq. (2) and its components
``pinn.metrics``    L2 relative error (L2RE)
``pinn.optim``      Adam, L-BFGS, Adam+L-BFGS, GD and NNCG
``pinn.hessian``    stochastic Lanczos quadrature & the L-BFGS preconditioner
``pinn.experiments``  drivers reproducing the individual figures/tables
"""

__version__ = "1.0.0"
