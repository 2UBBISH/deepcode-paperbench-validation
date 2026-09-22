from __future__ import annotations

from .actor_critic import ActorCritic, ActorCriticConfig
from .storage import RolloutStorage, PolicyBuffer, compute_gae, compute_nstep_returns
from .losses import (
    OnPolicyLoss,
    OffPolicyLoss,
    compute_on_policy_loss,
    compute_off_policy_loss,
)
from .sapg import SAPGTrainer
from .ppo import PPOTrainer
from .pbt import DexPBTTrainer
from .pql import PQLTrainer

ALGORITHMS = {
    "sapg": SAPGTrainer,
    "ppo": PPOTrainer,
    "pbt": DexPBTTrainer,
    "dexpbt": DexPBTTrainer,
    "pql": PQLTrainer,
}


def make_trainer(cfg, env, device="cpu", logdir=None):
    """Instantiate the trainer named by ``cfg.algo.name``."""
    name = str(cfg.algo.name).lower()
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{name}', available: {sorted(ALGORITHMS)}")
    return ALGORITHMS[name](cfg, env, device=device, logdir=logdir)


__all__ = [
    "ActorCritic",
    "ActorCriticConfig",
    "RolloutStorage",
    "PolicyBuffer",
    "compute_gae",
    "compute_nstep_returns",
    "OnPolicyLoss",
    "OffPolicyLoss",
    "compute_on_policy_loss",
    "compute_off_policy_loss",
    "SAPGTrainer",
    "PPOTrainer",
    "DexPBTTrainer",
    "PQLTrainer",
    "ALGORITHMS",
    "make_trainer",
]
