"""Refinement trainer for RICE.

Combines the mixed initial-state distribution (critical-state resets) with a
Random Network Distillation (RND) exploration bonus and trains a refined policy
with PPO.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from rice.agents import PPOConfig, PPOTrainer, TorchTargetPolicy
from rice.agents.target_policy import BaseTargetPolicy, MLPActorCritic, SB3TargetPolicy

from .critical_state_buffer import CriticalStateBuffer
from .mixed_reset_env import MixedResetEnv, default_restore_state, make_mixed_reset_env
from .rnd_bonus import RNDBonus, RNDRewardWrapper, make_rnd_bonus


class RefineTrainer:
    """Trainer for the RICE refinement stage.

    The refinement pipeline is:
        1. Wrap the task env with ``MixedResetEnv`` so episodes start from a
           critical state with probability ``p``.
        2. Wrap again with ``RNDRewardWrapper`` to add ``λ * b_RND(s)`` to each
           step reward.
        3. Train a refined policy ``π'`` with PPO.

    Parameters
    ----------
    env : gym.Env
        The original task environment.
    target_policy : BaseTargetPolicy
        Frozen target policy ``π``.  Used only to define the task; its weights
        may be copied to warm-start the refined policy.
    refined_policy : TorchTargetPolicy, optional
        Policy ``π'`` to refine.  If ``None``, a new actor-critic is created
        (warm-started from ``target_policy`` when possible).
    critical_buffer : CriticalStateBuffer or str, optional
        Buffer of critical states, or a path to a saved ``.npz`` buffer.  If
        ``None``, mixed resets fall back to ordinary resets.
    p : float
        Probability of sampling an initial state from the critical buffer.
    lambda_rnd : float
        Scaling coefficient for the RND bonus.
    rnd_bonus : RNDBonus, optional
        Pre-built RND bonus module.  If ``None``, one is created from the
        observation space.
    restore_fn : callable, optional
        Function ``restore_fn(env, critical_state) -> obs`` used by
        ``MixedResetEnv`` to restore simulator state.
    fallback_fn : callable, optional
        Fallback reset function used when state restoration fails.
    ppo_config : PPOConfig, optional
        Hyper-parameters for the PPO refinement loop.
    device : str or torch.device
        Device for the refined policy and RND networks.
    """

    def __init__(
        self,
        env: gym.Env,
        target_policy: BaseTargetPolicy,
        refined_policy: Optional[TorchTargetPolicy] = None,
        critical_buffer: Optional[Union[CriticalStateBuffer, str]] = None,
        p: float = 0.5,
        lambda_rnd: float = 0.01,
        rnd_bonus: Optional[RNDBonus] = None,
        restore_fn: Optional[Any] = None,
        fallback_fn: Optional[Any] = None,
        ppo_config: Optional[PPOConfig] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.env = env
        self.target_policy = target_policy
        self.p = p
        self.lambda_rnd = lambda_rnd
        self.device = self._resolve_device(device)

        # Build or load the critical-state buffer.
        if critical_buffer is None:
            self.critical_buffer = CriticalStateBuffer(capacity=1)
        elif isinstance(critical_buffer, str):
            self.critical_buffer = CriticalStateBuffer.load(critical_buffer)
        else:
            self.critical_buffer = critical_buffer

        # Build the RND bonus module if not provided.
        if rnd_bonus is None:
            self.rnd_bonus = make_rnd_bonus(
                env.observation_space,
                device=self.device,
            )
        else:
            self.rnd_bonus = rnd_bonus

        # Wrap environment: mixed reset -> RND bonus.
        mixed_env = MixedResetEnv(
            env,
            critical_buffer=self.critical_buffer,
            p=p,
            restore_fn=restore_fn or default_restore_state,
            fallback_fn=fallback_fn,
        )
        self.wrapped_env = RNDRewardWrapper(
            mixed_env,
            rnd_bonus=self.rnd_bonus,
            lambda_rnd=lambda_rnd,
        )

        # Build or reuse the refined policy.
        if refined_policy is None:
            self.refined_policy = self._build_refined_policy()
        else:
            self.refined_policy = refined_policy
            self.refined_policy.to(self.device)

        # Generic PPO trainer on the wrapped environment.
        self.ppo_config = ppo_config or PPOConfig()
        self.trainer = PPOTrainer(
            policy=self.refined_policy,
            env=self.wrapped_env,
            config=self.ppo_config,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _resolve_device(self, device: Union[str, torch.device]) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_refined_policy(self) -> TorchTargetPolicy:
        """Create a refined policy, warm-starting from the target if possible."""
        obs_space = self.target_policy.observation_space
        act_space = self.target_policy.action_space

        discrete = isinstance(act_space, gym.spaces.Discrete)
        obs_dim = int(np.prod(obs_space.shape))
        action_dim = int(act_space.n) if discrete else int(np.prod(act_space.shape))

        # Try to clone the target network if it is a PyTorch policy.
        if isinstance(self.target_policy, TorchTargetPolicy):
            model = copy.deepcopy(self.target_policy.model)
            model.to(self.device)
            return TorchTargetPolicy(
                model=model,
                observation_space=obs_space,
                action_space=act_space,
                device=self.device,
            )

        # SB3 (or other backend) target: create a fresh actor-critic with the
        # same input/output dimensions.  The refined policy is therefore trained
        # from scratch, which still satisfies the RICE pipeline when the target
        # is used only to generate the critical-state buffer.
        hidden_sizes = self._infer_hidden_sizes()
        model = MLPActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            discrete=discrete,
        ).to(self.device)
        return TorchTargetPolicy(
            model=model,
            observation_space=obs_space,
            action_space=act_space,
            device=self.device,
        )

    def _infer_hidden_sizes(self) -> tuple:
        """Best-effort inference of hidden layer sizes from the target policy."""
        if isinstance(self.target_policy, SB3TargetPolicy):
            # SB3 default MLP policy uses [64, 64].
            return (64, 64)
        return (64, 64)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        save_path: Optional[str] = None,
        save_interval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run PPO refinement and return training statistics."""
        return self.trainer.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            save_path=save_path,
            save_interval=save_interval,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the refined policy, RND module, and critical buffer."""
        os.makedirs(path, exist_ok=True)
        self.refined_policy.save(os.path.join(path, "policy.pt"))
        self.rnd_bonus.save(os.path.join(path, "rnd.pt"))
        self.critical_buffer.save(os.path.join(path, "critical_buffer.npz"))

    def load(self, path: str) -> None:
        """Load a previously saved refinement checkpoint."""
        self.refined_policy.load(os.path.join(path, "policy.pt"))
        self.rnd_bonus.load(os.path.join(path, "rnd.pt"))
        self.critical_buffer = CriticalStateBuffer.load(
            os.path.join(path, "critical_buffer.npz")
        )
        # Re-wire the wrapped env with the loaded buffer.
        self.wrapped_env.env.set_critical_buffer(self.critical_buffer)


def refine_policy(
    env: gym.Env,
    target_policy: BaseTargetPolicy,
    critical_buffer: Optional[Union[CriticalStateBuffer, str]] = None,
    total_timesteps: int = 1_000_000,
    p: float = 0.5,
    lambda_rnd: float = 0.01,
    ppo_config: Optional[PPOConfig] = None,
    device: Union[str, torch.device] = "auto",
    save_path: Optional[str] = None,
) -> RefineTrainer:
    """Convenience factory: build and run the RICE refinement pipeline.

    Returns the trained ``RefineTrainer`` so the refined policy and statistics
    can be inspected.
    """
    trainer = RefineTrainer(
        env=env,
        target_policy=target_policy,
        critical_buffer=critical_buffer,
        p=p,
        lambda_rnd=lambda_rnd,
        ppo_config=ppo_config,
        device=device,
    )
    trainer.learn(
        total_timesteps=total_timesteps,
        save_path=save_path,
    )
    if save_path is not None:
        trainer.save(save_path)
    return trainer
