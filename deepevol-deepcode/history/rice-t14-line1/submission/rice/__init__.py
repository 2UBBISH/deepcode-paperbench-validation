"""RICE: A Refining scheme for ReInforcement learning with Explanation.

Reference:
    Cheng et al. "RICE: A Refining scheme for ReInforcement learning with
    Explanation." Proc. 41st ICML, PMLR 235, 2024.

Package layout
--------------
rice.algorithms   : core method (mask network, RND, mixed-init, refining, PPO)
rice.environments : in-scope application environments / wrappers
rice.explanation  : explanation methods (ours + baselines)
rice.baselines    : refining baselines (PPO-FT, StateMask-R, JSRL, SIL, SAC+GAIL)
rice.evaluation   : fidelity score, refining evaluation, hyper-param sweeps
rice.configs      : per-environment YAML hyper-parameters (Table 3)
rice.utils        : seeding, buffers, logging, plotting, normalization
"""

__version__ = "0.1.0"

from rice.algorithms.mask_network import MaskNetwork, MaskNetworkTrainer  # noqa: F401
from rice.algorithms.critical_state import (  # noqa: F401
    importance_scores,
    select_critical_state,
)
from rice.algorithms.mixed_init import MixedInitSampler  # noqa: F401
from rice.algorithms.rnd import RND  # noqa: F401
from rice.algorithms.refine import RICERefiner  # noqa: F401
from rice.algorithms.ppo import PPO, ActorCritic, PPOConfig  # noqa: F401
from rice.algorithms.env_reset import EnvStateManager  # noqa: F401
