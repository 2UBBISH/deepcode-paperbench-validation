"""RICE (Algorithm 2) and the three refining baselines of the paper.

* :func:`refine_rice`              -- mixed initial state distribution + RND
* :func:`refine_ppo_finetune`      -- PPO fine-tuning with a lowered lr
* :func:`refine_statemask_r`       -- StateMask's "reset to critical states"
* :func:`refine_jsrl`              -- Jump-Start RL curriculum
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch

from rice.envs.registry import make_env
from rice.networks import ActorCritic
from rice.policies import TorchPolicy
from rice.refining.critical_state_provider import CriticalStateProvider
from rice.refining.rnd import RND, RNDRewardShaper
from rice.refining.trainer import RefineConfig, RefineResult, RefiningTrainer


def frozen_copy(policy: ActorCritic) -> ActorCritic:
    with torch.no_grad():
        clone = copy.deepcopy(policy)
    clone.eval()
    for param in clone.parameters():
        param.requires_grad_(False)
    return clone


def make_eval_env(env_name: str, seed: int = 123):
    """A separate environment instance used for the periodic evaluation."""
    try:
        return make_env(env_name, seed=seed)
    except Exception:  # pragma: no cover - optional third party simulators
        return None


def _trainer(
    env,
    policy,
    config,
    method: str,
    eval_env=None,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> RefiningTrainer:
    return RefiningTrainer(
        env=env,
        policy=policy,
        config=config,
        method=method,
        device=device,
        eval_env=eval_env,
        verbose=verbose,
        **kwargs,
    )


def refine_rice(
    env,
    policy: ActorCritic,
    mask_net,
    config: Optional[RefineConfig] = None,
    env_name: Optional[str] = None,
    rollin_policy=None,
    rollin_policy_mode: str = "pretrained",
    device: str = "cpu",
    eval_env=None,
    verbose: bool = True,
) -> RefineResult:
    """Algorithm 2: refine ``policy`` with our explanation method."""
    config = config or RefineConfig()
    obs_dim = int(np.prod(env.observation_space.shape))

    if rollin_policy is None:
        rollin_policy = TorchPolicy(frozen_copy(policy), device=device)
    provider = CriticalStateProvider(
        env=env,
        mask_net=mask_net,
        p=config.p,
        rollin_length=config.rollin_length,
        rollin_policy=rollin_policy,
        rollin_policy_mode=rollin_policy_mode,
        seed=config.seed,
    )
    shaper = None
    if config.rnd_lambda > 0:
        rnd = RND(
            obs_dim,
            hidden=(64, 64),
            learning_rate=1e-3,
            device=device,
        )
        shaper = RNDRewardShaper(rnd, coef=config.rnd_lambda)

    trainer = _trainer(
        env,
        policy,
        config,
        method="rice",
        eval_env=eval_env,
        device=device,
        verbose=verbose,
        initial_state_provider=provider,
        reward_shaper=shaper,
    )
    result = trainer.train(config.total_steps)
    result.provider_stats = {
        "critical_resets": provider.n_resets,
        "default_resets": provider.n_default,
    }
    return result


def refine_ppo_finetune(
    env,
    policy: ActorCritic,
    config: Optional[RefineConfig] = None,
    env_name: Optional[str] = None,
    device: str = "cpu",
    eval_env=None,
    verbose: bool = True,
) -> RefineResult:
    """Baseline 1: keep training with PPO from the default initial states."""
    config = config or RefineConfig()
    trainer = _trainer(
        env,
        policy,
        config,
        method="ppo_finetune",
        eval_env=eval_env,
        device=device,
        verbose=verbose,
    )
    return trainer.train(config.total_steps)


def refine_statemask_r(
    env,
    policy: ActorCritic,
    mask_net,
    config: Optional[RefineConfig] = None,
    env_name: Optional[str] = None,
    rollin_policy=None,
    device: str = "cpu",
    eval_env=None,
    verbose: bool = True,
) -> RefineResult:
    """Baseline 2: StateMask-R, i.e. fine-tune *only* from critical states.

    ``p = 1`` in the mixed initial state distribution and no exploration bonus,
    which reproduces the overfitting behaviour described in the paper.
    """
    config = copy.deepcopy(config or RefineConfig())
    config.p = 1.0
    config.rnd_lambda = 0.0
    if rollin_policy is None:
        rollin_policy = TorchPolicy(frozen_copy(policy), device=device)
    provider = CriticalStateProvider(
        env=env,
        mask_net=mask_net,
        p=1.0,
        rollin_length=config.rollin_length,
        rollin_policy=rollin_policy,
        seed=config.seed,
    )
    trainer = _trainer(
        env,
        policy,
        config,
        method="statemask_r",
        eval_env=eval_env,
        device=device,
        verbose=verbose,
        initial_state_provider=provider,
    )
    result = trainer.train(config.total_steps)
    result.provider_stats = {
        "critical_resets": provider.n_resets,
        "default_resets": provider.n_default,
    }
    return result


def jsrl_curriculum(max_rollin: int) -> Callable[[float], int]:
    """Uniform curriculum over the number of guided steps (Uchendu et al.)."""

    def _fn(progress: float) -> int:
        horizon = int(max(1, round(max_rollin * min(1.0, max(0.0, progress)))))
        return int(np.random.randint(0, horizon + 1))

    return _fn


def refine_jsrl(
    env,
    policy: ActorCritic,
    config: Optional[RefineConfig] = None,
    env_name: Optional[str] = None,
    device: str = "cpu",
    eval_env=None,
    verbose: bool = True,
) -> RefineResult:
    """Baseline 3: Jump-Start RL turned into a refining method.

    The pre-trained policy plays the role of the guide ``pi_g``; the exploration
    policy ``pi_e`` is initialised with the same weights and is the policy that
    is returned after refining.  At the beginning of every episode a random
    number of guided steps is executed (the curriculum), and only the steps
    taken by ``pi_e`` are used for the PPO update.
    """
    config = config or RefineConfig()
    guide = TorchPolicy(frozen_copy(policy), device=device)
    trainer = _trainer(
        env,
        policy,
        config,
        method="jsrl",
        eval_env=eval_env,
        device=device,
        verbose=verbose,
        rollin_policy=guide,
        rollin_steps_fn=jsrl_curriculum(config.jsrl_max_rollin),
    )
    return trainer.train(config.total_steps)
