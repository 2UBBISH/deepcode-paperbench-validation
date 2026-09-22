"""Unsupervised pre-training of FRE (Algorithm 1).

Algorithm 1 uses a **strided** training scheme:

  1. train the FRE encoder/decoder (Equation 6) for ``encoder_steps`` while the
     RL components are untouched;
  2. freeze the encoder and train the IQL policy / value / Q networks for
     ``policy_steps``, using the frozen encoder's latents as task conditioning.

Section 4.1 explains why: "In this way, we can make the mapping from eta to z
stationary during policy learning, which we found to be important to correctly
estimate multitask Q values using TD learning."
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from fre.configs import TrainConfig
from fre.datasets import OfflineDataset
from fre.fre import FRE, FREConfig
from fre.iql import IQLAgent, IQLConfig
from fre.reward_functions import RewardPrior


def make_preprocess_fn(config: TrainConfig, obs_dim: int) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return the state preprocessing used by the model (AntMaze XY binning)."""
    if config.domain == "antmaze" and config.discretize_xy:
        from fre.tasks.antmaze import AntMazeStatePreprocessor

        return AntMazeStatePreprocessor(num_bins=config.num_xy_bins)
    return None


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


class FRETrainer:
    """Owns the FRE module, the IQL agent, and the strided training loop."""

    def __init__(
        self,
        config: TrainConfig,
        dataset: OfflineDataset,
        prior: RewardPrior,
        preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.prior = prior
        self.preprocess_fn = preprocess_fn
        self.device = resolve_device(config.device)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        # Standardise the model-facing states using statistics of the
        # *preprocessed* encoder observations.  The paper's four agents share a
        # single codebase, and observations in AntMaze / ExORL have very
        # different per-dimension scales, so standardising is required for the
        # MLP decoder and the RL networks to train stably.
        model_states = (
            preprocess_fn(dataset.encoder_observations) if preprocess_fn is not None
            else dataset.encoder_observations
        )
        self.state_mean = np.asarray(model_states.mean(axis=0), dtype=np.float32)
        self.state_std = np.asarray(model_states.std(axis=0), dtype=np.float32)
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std).astype(np.float32)

        self.model = FRE(
            FREConfig(
                state_dim=dataset.encoder_obs_dim,
                z_dim=config.z_dim,
                state_embed_dim=config.state_embed_dim,
                reward_embed_dim=config.reward_embed_dim,
                d_model=config.d_model,
                n_layers=config.encoder_layers,
                n_heads=config.encoder_heads,
                encoder_mlp_dim=config.encoder_mlp_dim,
                num_reward_bins=config.num_reward_bins,
                decoder_hidden_dims=config.decoder_hidden_dims,
                beta=config.beta,
                num_encode_pairs=config.num_encode_pairs,
                num_decode_pairs=config.num_decode_pairs,
            )
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.agent = IQLAgent(
            IQLConfig(
                # The value / Q / policy networks operate on the environment's
                # underlying observation space (Appendix C.2: performance was
                # not greatly affected by giving them the auxiliary physics
                # information, so they are trained on the plain observation).
                obs_dim=dataset.obs_dim,
                action_dim=dataset.action_dim,
                z_dim=config.z_dim,
                hidden_dims=config.rl_hidden_dims,
                discount=config.discount,
                expectile=config.expectile,
                awr_temperature=config.awr_temperature,
                target_update_rate=config.target_update_rate,
                learning_rate=config.learning_rate,
            ),
            device=self.device,
        )
        self.step = 0
        self.history: Dict[str, list] = {"encoder_loss": [], "recon": [], "kl": [], "q_loss": [], "v_loss": [], "policy_loss": []}

    # -- helpers -------------------------------------------------------------------
    def _to_model_states(self, states: np.ndarray) -> torch.Tensor:
        if self.preprocess_fn is not None:
            states = self.preprocess_fn(states)
        states = (np.asarray(states, dtype=np.float32) - self.state_mean) / self.state_std
        return torch.as_tensor(states, device=self.device)

    def sample_reward_functions(self, batch_size: int):
        """Sample a batch of reward functions and their analytic reward ranges."""
        return self.prior.sample(batch_size, self.device)

    def encode_context(self, states: np.ndarray, eta) -> torch.Tensor:
        """Encode a context set of ``(s, eta(s))`` pairs into latents ``z``."""
        rewards = eta.reward(torch.as_tensor(states, device=self.device, dtype=torch.float32))
        bins = self._discretise_per_element(rewards, eta)
        return self.model.encode(self._to_model_states(states), bins, sample=False)

    # -- phase 1: encoder ----------------------------------------------------------
    def train_encoder_step(self) -> Dict[str, float]:
        b = self.config.batch_size
        k, kd = self.config.num_encode_pairs, self.config.num_decode_pairs
        eta = self.sample_reward_functions(b)

        enc_states = self.dataset.random_states(b * k, rng=self.rng).reshape(b, k, -1).numpy()
        enc_states = eta.maybe_insert_goal_state(
            torch.as_tensor(enc_states, device=self.device)
        ).cpu().numpy()
        dec_states = self.dataset.random_states(b * kd, rng=self.rng).reshape(b, kd, -1).numpy()

        enc_raw = torch.as_tensor(enc_states, device=self.device, dtype=torch.float32)
        dec_raw = torch.as_tensor(dec_states, device=self.device, dtype=torch.float32)
        enc_rewards = eta.reward(enc_raw)
        dec_rewards = eta.reward(dec_raw)

        # Each reward family has its own analytic range; discretise with the
        # per-element range so bins line up with the embedding table.
        bins = self._discretise_per_element(enc_rewards, eta)
        loss, recon, kl = self.model.loss(
            self._to_model_states(enc_states),
            bins,
            self._to_model_states(dec_states),
            dec_rewards,
            beta=self.config.beta,
            return_parts=True,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return {"encoder_loss": float(loss.item()), "recon": float(recon.item()), "kl": float(kl.item())}

    def _discretise_per_element(self, rewards: torch.Tensor, eta) -> torch.Tensor:
        r_min, r_max = eta.r_min, eta.r_max
        while r_min.dim() < rewards.dim():
            r_min = r_min.unsqueeze(-1)
            r_max = r_max.unsqueeze(-1)
        scaled = ((rewards - r_min) / (r_max - r_min).clamp_min(1e-6)).clamp(0.0, 1.0)
        bins = torch.floor(scaled * self.config.num_reward_bins)
        return bins.clamp_(0, self.config.num_reward_bins - 1).long()

    def train_encoder(self, num_steps: Optional[int] = None, log_fn: Optional[Callable] = None) -> None:
        steps = num_steps or self.config.encoder_steps
        t0 = time.time()
        for i in range(steps):
            stats = self.train_encoder_step()
            self.step += 1
            self.history["encoder_loss"].append(stats["encoder_loss"])
            self.history["recon"].append(stats["recon"])
            self.history["kl"].append(stats["kl"])
            if log_fn and (i + 1) % self.config.log_interval == 0:
                log_fn(
                    {
                        "phase": "encoder",
                        "step": i + 1,
                        "total_steps": steps,
                        "encoder_loss": stats["encoder_loss"],
                        "recon": stats["recon"],
                        "kl": stats["kl"],
                        "elapsed": time.time() - t0,
                    }
                )
            if (i + 1) % self.config.checkpoint_interval == 0:
                self.save_encoder_checkpoint()
        self.save_encoder_checkpoint()
        # Freeze the encoder for policy learning (Algorithm 1).
        self.model.encoder.eval()
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)

    # -- phase 2: policy -----------------------------------------------------------
    def train_policy_step(self) -> Dict[str, float]:
        b = self.config.batch_size
        k = self.config.num_encode_pairs
        eta = self.sample_reward_functions(b)
        enc_states = self.dataset.random_states(b * k, rng=self.rng).reshape(b, k, -1).numpy()
        enc_states = eta.maybe_insert_goal_state(
            torch.as_tensor(enc_states, device=self.device)
        ).cpu().numpy()
        with torch.no_grad():
            z = self.encode_context(enc_states, eta)

        batch = self.dataset.sample_transitions(b, rng=self.rng)
        action = batch["actions"].to(self.device)
        done = batch["terminals"].to(self.device)
        # Reward functions are pure functions of the (encoder) environment
        # state, so they are evaluated on the encoder observation.
        encoder_obs = batch["encoder_observations"].to(self.device)
        reward = eta.reward(encoder_obs)  # Algorithm 1: r = eta(s)
        # Appendix B: goal-reaching reward functions terminate on goal
        # achievement; combine that with the dataset's own terminal flags.
        done = torch.clamp(done + eta.done(encoder_obs), max=1.0)

        obs_m = self._to_model_states(batch["observations"].numpy())
        next_obs_m = self._to_model_states(batch["next_observations"].numpy())

        stats = {}
        stats.update(self.agent.update_value(obs_m, action, z))
        stats.update(self.agent.update_q(obs_m, action, reward, next_obs_m, done, z))
        stats.update(self.agent.update_policy(obs_m, action, z))
        self.agent.update_target()
        return stats

    def train_policy(self, num_steps: Optional[int] = None, log_fn: Optional[Callable] = None) -> None:
        steps = num_steps or self.config.policy_steps
        t0 = time.time()
        for i in range(steps):
            stats = self.train_policy_step()
            self.step += 1
            for key in ("q_loss", "v_loss", "policy_loss"):
                if key in stats:
                    self.history[key].append(stats[key])
            if log_fn and (i + 1) % self.config.log_interval == 0:
                entry = {"phase": "policy", "step": i + 1, "total_steps": steps, "elapsed": time.time() - t0}
                entry.update({k: v for k, v in stats.items() if isinstance(v, float)})
                log_fn(entry)
            if (i + 1) % self.config.checkpoint_interval == 0:
                self.save_policy_checkpoint()
        self.save_policy_checkpoint()

    # -- full run ------------------------------------------------------------------
    def fit(self, log_fn: Optional[Callable] = None) -> None:
        self.train_encoder(log_fn=log_fn)
        self.train_policy(log_fn=log_fn)

    # -- checkpointing -------------------------------------------------------------
    def _output_dir(self) -> str:
        name = self.config.run_name or f"{self.config.domain}-{self.config.prior_name}-s{self.config.seed}"
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    def save_encoder_checkpoint(self) -> str:
        path = os.path.join(self._output_dir(), "encoder.pt")
        torch.save(
            {
                "encoder": self.model.encoder.state_dict(),
                "decoder": self.model.decoder.state_dict(),
                "config": asdict(self.config),
                "state_mean": self.state_mean,
                "state_std": self.state_std,
            },
            path,
        )
        return path

    def save_policy_checkpoint(self) -> str:
        path = os.path.join(self._output_dir(), "policy.pt")
        torch.save(
            {
                "encoder": self.model.encoder.state_dict(),
                "decoder": self.model.decoder.state_dict(),
                "agent": self.agent.state_dict(),
                "config": asdict(self.config),
                "state_mean": self.state_mean,
                "state_std": self.state_std,
            },
            path,
        )
        return path

    def save_history(self) -> str:
        path = os.path.join(self._output_dir(), "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f)
        return path
