"""Core RICE algorithms.

This sub-package implements the four core pieces of the RICE method
(ICML 2024, PMLR 235) plus the supporting PPO update and the
Go-Explore-style environment state save/restore machinery:

* :mod:`rice.algorithms.mask_network` -- Algorithm 1, training the mask network.
  The mask net ``\\tilde\\pi_theta`` takes a state and emits a binary action
  ``a_t^m in {0, 1}`` which decides whether the target agent's action is
  replaced by a uniformly random action (Eq. 1):

      a_t (.) a_t^m = a_t          if a_t^m = 0
                    = a_random     if a_t^m = 1

  Because Theorem 3.3 gives ``eta(pi_bar) <= eta(pi)``, the original objective
  ``J(theta) = min |eta(pi) - eta(pi_bar)|`` collapses to ``max eta(pi_bar)``
  which can be solved by vanilla PPO.  To avoid the trivial solution
  (never blinding), the reward is augmented with ``R' = R + alpha * a_t^m``.

* :mod:`rice.algorithms.critical_state` -- state importance is the probability
  that the mask network outputs "0" ("*the probability of mask network
  outputting '0'*", Sec 3.3); the critical state of a trajectory is the
  ``argmax`` over those importances.

* :mod:`rice.algorithms.mixed_init` -- the mixed initial state distribution
  ``mu(s) = beta * d_rho^pi_hat(s) + (1 - beta) * rho(s)``, realised as a
  Bernoulli(p) roll-in at the start of every refining iteration
  (p plays the role of beta).

* :mod:`rice.algorithms.rnd` -- Random Network Distillation.  A frozen,
  randomly initialised target network ``f`` and a predictor ``f_hat`` regressed
  towards it; the normalised intrinsic bonus is
  ``R_t^RND = ||f(s_{t+1}) - f_hat(s_{t+1})||^2`` and the augmented reward is
  ``R' = R + lambda * R_t^RND``.

* :mod:`rice.algorithms.refine` -- Algorithm 2, the refining loop tying
  everything together.

* :mod:`rice.algorithms.ppo` -- the shared clipped-surrogate PPO update used by
  both the mask net training and the refining loop (Stable-Baselines3 defaults,
  since the paper does not specify PPO hyper-parameters).

* :mod:`rice.algorithms.env_reset` -- environment state save/restore, in the
  spirit of Ecoffet et al. (2019), used to reset the simulator to a previously
  identified critical state.
"""

from .critical_state import importance_scores, select_critical_state
from .env_reset import EnvStateManager
from .mask_network import MaskNetwork, MaskNetworkTrainer
from .mixed_init import MixedInitSampler
from .ppo import PPO, ActorCritic, PPOConfig
from .refine import RICERefiner
from .rnd import RND

__all__ = [
    "MaskNetwork",
    "MaskNetworkTrainer",
    "importance_scores",
    "select_critical_state",
    "MixedInitSampler",
    "RND",
    "RICERefiner",
    "PPO",
    "ActorCritic",
    "PPOConfig",
    "EnvStateManager",
]
