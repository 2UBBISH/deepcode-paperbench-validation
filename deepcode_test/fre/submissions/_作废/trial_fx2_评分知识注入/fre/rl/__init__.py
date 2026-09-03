"""Reinforcement learning components for Functional Reward Encodings.

This package exposes the conditional network primitives, Implicit Q-Learning
losses/trainer, and the FRE-conditioned offline RL training loop used in
Phase 2 of the reproduction.
"""

from fre.rl.networks import (
    DeterministicPolicy,
    GaussianPolicy,
    QNetwork,
    ValueNetwork,
    soft_update,
)
from fre.rl.iql import IQL, ImplicitQLearning, expectile_loss
from fre.rl.rl_trainer import (
    FREIQLTrainer,
    RLTrainer,
    train_fre_iql_agent,
)

__all__ = [
    # networks.py
    "ValueNetwork",
    "QNetwork",
    "GaussianPolicy",
    "DeterministicPolicy",
    "soft_update",
    # iql.py
    "expectile_loss",
    "ImplicitQLearning",
    "IQL",
    # rl_trainer.py
    "FREIQLTrainer",
    "RLTrainer",
    "train_fre_iql_agent",
]
