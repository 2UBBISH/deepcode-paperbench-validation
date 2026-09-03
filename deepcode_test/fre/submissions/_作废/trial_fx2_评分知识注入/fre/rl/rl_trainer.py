"""FRE-conditioned offline RL trainer.

This module implements the RL training loop described in Section D of the
Functional Reward Encodings paper.  During Phase 2 the FRE encoder is kept
frozen while an Implicit Q-Learning agent is trained on rewards computed from
randomly sampled reward functions ``eta``.  Each training step:

1. Samples ``K`` encoder states from the offline dataset.
2. Samples a reward function ``eta ~ p(eta)`` from the prior mixture.
3. Encodes ``z ~ q_phi(z | {(s_k, eta(s_k))})``.
4. Samples a minibatch of transitions.
5. Computes rewards ``r = eta(s)`` for those transitions.
6. Updates V, Q, and policy using the IQL losses conditioned on ``z``.

The encoder is placed in evaluation mode and its parameters are frozen, which
keeps the latent reward representation stationary during temporal-difference
learning.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from fre.config import Config, IQLConfig, RewardSamplerConfig
from fre.data.dataset import OfflineDataset, TransitionBatch
from fre.data.reward_sampler import RewardFunction, sample_reward
from fre.modeling.fre_vae import FREVAE
from fre.rl.iql import IQL, ImplicitQLearning

logger = logging.getLogger(__name__)

__all__ = ["FREIQLTrainer", "RLTrainer", "train_fre_iql_agent"]


def _cfg_value(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first non-``None`` attribute among ``names``.

    This helper is intentionally defensive so the trainer works both with the
    full :class:`~fre.config.Config` hierarchy and with standalone dataclass
    instances whose fields may use slightly different names.
    """
    for name in names:
        if obj is None:
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


class FREIQLTrainer:
    """Trainer for FRE-conditioned Implicit Q-Learning.

    Parameters
    ----------
    dataset:
        The offline dataset used to sample encoder states and RL transitions.
    model:
        A pretrained (or otherwise frozen) :class:`FREVAE`.  The trainer always
        disables gradient computation for the model to keep ``z`` stationary.
    agent:
        An :class:`~fre.rl.iql.ImplicitQLearning` agent whose networks are
        conditioned on ``z``.
    cfg:
        Top-level :class:`~fre.config.Config` or an IQL-specific config.
    reward_sampler_cfg:
        Configuration for the prior reward-function mixture.  Defaults to
        ``cfg.reward_sampler`` when omitted.
    device:
        Torch device used for training.
    num_encoder_states:
        Number of context states ``K``.  Defaults to the reward-sampler config
        or 32.
    batch_size:
        RL minibatch size.  Defaults to the IQL config or 256.
    seed:
        Integer seed for the NumPy random generator used by reward sampling.
    """

    def __init__(
        self,
        dataset: OfflineDataset,
        model: FREVAE,
        agent: ImplicitQLearning,
        cfg: Optional[Config] = None,
        reward_sampler_cfg: Optional[RewardSamplerConfig] = None,
        device: str = "cpu",
        num_encoder_states: Optional[int] = None,
        batch_size: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.agent = agent
        self.cfg = cfg
        self.device = torch.device(device)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.reward_sampler_cfg = reward_sampler_cfg or _cfg_value(
            cfg, "reward_sampler", "reward_sampler_config", default=None
        )
        if self.reward_sampler_cfg is None:
            self.reward_sampler_cfg = RewardSamplerConfig()
            logger.debug("No reward-sampler config supplied; using defaults.")

        iql_cfg = _cfg_value(cfg, "iql", "iql_config", default=None)

        if num_encoder_states is None:
            num_encoder_states = _cfg_value(
                self.reward_sampler_cfg,
                "num_encoder_states",
                "encoder_states",
                default=32,
            )
        if batch_size is None:
            batch_size = _cfg_value(iql_cfg, "batch_size", default=256)
        self.num_encoder_states = int(num_encoder_states)
        self.batch_size = int(batch_size)

        self.log_every = int(_cfg_value(iql_cfg, "log_every", default=1000))
        self.checkpoint_every = int(
            _cfg_value(iql_cfg, "checkpoint_every", default=25000)
        )

        # Phase 2 requires a frozen encoder.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.global_step = 0

    # ------------------------------------------------------------------ #
    # Reward encoding
    # ------------------------------------------------------------------ #
    def encode_reward(
        self,
        reward_fn: RewardFunction,
        states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a reward function from ``K`` labelled context states.

        Returns
        -------
        z:
            Detached latent reward vector with shape ``(1, z_dim)``.
        rewards:
            Scalar rewards ``eta(states)`` used to form the context.
        states:
            The context states on the training device.
        """
        if states is None:
            states = self.dataset.sample_states(
                self.num_encoder_states, device=self.device
            )
        else:
            states = states.to(self.device)

        if hasattr(reward_fn, "to"):
            reward_fn = reward_fn.to(self.device)

        with torch.no_grad():
            rewards = reward_fn(states)
            # FREVAE.encode returns (mu, log_sigma, z) when return_z is True.
            result = self.model.encode(states, rewards, return_z=True)
            mu, log_sigma, z = result[0], result[1], result[2]
            if z is None:
                # Fallback for encode implementations that only return moments.
                z = self.model.reparameterize(mu, log_sigma)
            z = z.detach()

        # Keep a consistent leading batch dimension for agent conditioning.
        while z.dim() < 2:
            z = z.unsqueeze(0)
        return z, rewards.detach(), states.detach()

    # ------------------------------------------------------------------ #
    # Training steps
    # ------------------------------------------------------------------ #
    def train_step(
        self,
        reward_fn: Optional[RewardFunction] = None,
        transitions: Optional[TransitionBatch] = None,
    ) -> Dict[str, float]:
        """Perform one FRE-conditioned IQL update.

        Parameters
        ----------
        reward_fn:
            Optional pre-sampled reward function.  If omitted, one is sampled
            from the prior mixture.
        transitions:
            Optional transition minibatch.  If omitted, one is sampled from
            the dataset.

        Returns
        -------
        Dictionary of scalar training metrics for the step.
        """
        context_states = self.dataset.sample_states(
            self.num_encoder_states, device=self.device
        )

        if reward_fn is None:
            reward_fn = sample_reward(
                context_states, self.reward_sampler_cfg, self.rng
            )
        if hasattr(reward_fn, "to"):
            reward_fn = reward_fn.to(self.device)

        z, context_rewards, _ = self.encode_reward(reward_fn, context_states)

        if transitions is None:
            transitions = self.dataset.sample_transitions(
                self.batch_size, device=self.device
            )

        rewards = reward_fn(transitions.states)
        # Reward functions are required to return shape (batch,) or broadcastable.
        if rewards.dim() == 2 and rewards.size(-1) == 1:
            rewards = rewards.squeeze(-1)

        info = self.agent.train_step(transitions, condition=z, rewards=rewards)

        info["context_reward_mean"] = float(context_rewards.mean().item())
        info["context_reward_std"] = float(context_rewards.std().item())
        info["z_norm"] = float(z.norm(dim=-1).mean().item())
        info["global_step"] = float(self.global_step)
        self.global_step += 1
        return info

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(
        self,
        num_steps: int,
        log_every: Optional[int] = None,
        checkpoint_every: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        start_step: int = 0,
    ) -> Dict[str, Any]:
        """Run a full FRE-conditioned IQL training loop.

        Parameters
        ----------
        num_steps:
            Number of RL gradient steps to perform.
        log_every:
            Logging interval; defaults to the IQL config value.
        checkpoint_every:
            Checkpoint interval; defaults to the IQL config value.
        checkpoint_dir:
            Directory in which to save agent checkpoints.
        start_step:
            Global step offset (useful for resuming).

        Returns
        -------
        Dictionary with a list of per-step metrics and final summary values.
        """
        log_every = self.log_every if log_every is None else int(log_every)
        checkpoint_every = (
            self.checkpoint_every
            if checkpoint_every is None
            else int(checkpoint_every)
        )

        history: Dict[str, list] = {
            "value_loss": [],
            "q_loss": [],
            "policy_loss": [],
            "context_reward_mean": [],
            "context_reward_std": [],
            "z_norm": [],
        }

        start_time = time.time()
        for step in range(start_step, start_step + int(num_steps)):
            info = self.train_step()
            for key in history:
                if key in info:
                    history[key].append(info[key])

            if (step + 1) % log_every == 0 or step == start_step:
                elapsed = time.time() - start_time
                parts = " ".join(
                    f"{key}={info.get(key, float('nan')):.4f}"
                    for key in ("value_loss", "q_loss", "policy_loss")
                )
                logger.info(
                    "RL step %d/%d [%.1fs] %s", step + 1, num_steps, elapsed, parts
                )

            if checkpoint_dir is not None and (step + 1) % checkpoint_every == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"iql_step_{step + 1}.pt")
                self.agent.save(path)
                logger.info("Saved IQL checkpoint to %s", path)

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, "iql_final.pt")
            self.agent.save(path)
            logger.info("Saved final IQL checkpoint to %s", path)

        summary: Dict[str, Any] = {"history": history}
        for key, values in history.items():
            if values:
                summary[f"final_{key}"] = float(np.mean(values[-100:]))
        summary["total_steps"] = int(num_steps)
        summary["elapsed_seconds"] = float(time.time() - start_time)
        return summary

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save the RL agent (FRE encoder is assumed already checkpointed)."""
        self.agent.save(path)

    def load(self, path: str) -> None:
        """Load the RL agent weights."""
        self.agent.load(path)


# Public alias matching the naming used elsewhere in the repository.
RLTrainer = FREIQLTrainer


def train_fre_iql_agent(
    cfg: Config,
    dataset: OfflineDataset,
    model: FREVAE,
    agent: ImplicitQLearning,
    device: Optional[str] = None,
    num_steps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    log_every: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Convenience wrapper for training an FRE-conditioned IQL agent.

    This mirrors :meth:`FREIQLTrainer.train` while reading hyperparameters from
    a top-level :class:`~fre.config.Config`.
    """
    device = device or getattr(cfg, "device", "cpu")
    if num_steps is None:
        num_steps = int(_cfg_value(cfg.iql, "num_steps", default=1_000_000))
    if checkpoint_dir is None:
        checkpoint_dir = getattr(cfg, "checkpoint_dir", None)

    trainer = FREIQLTrainer(
        dataset=dataset,
        model=model,
        agent=agent,
        cfg=cfg,
        device=device,
        seed=seed,
    )
    return trainer.train(
        num_steps=num_steps,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
    )
