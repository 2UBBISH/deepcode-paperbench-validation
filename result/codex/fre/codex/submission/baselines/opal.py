"""OPAL: offline skill discovery by auto-encoding trajectories (Ajay et al., 2020).

Section 5.2 describes OPAL as "a representative offline unsupervised skill
discovery method where latent skills are learned by auto-encoding
trajectories", and the addendum adds:

  * no manually designed rewards are used;
  * "for the OPAL encoder, the same transformer architecture is used as in
    FRE";
  * OPAL "does not solve the problem of understanding a reward function
    zero-shot", so it is compared in a *privileged* setting: "10 random skills
    are sampled from a unit Gaussian, for each skill z the policy is conditioned
    on it and evaluated for the entire episode, and the best performing rollout
    is taken."

The implementation therefore has three pieces:

  1. a trajectory VAE over chunks of ``(s, a)`` pairs whose encoder reuses the
     FRE transformer architecture;
  2. a latent-conditioned skill policy trained by behavioural cloning on the
     dataset with the inferred skills as labels (OPAL trains the policy on the
     skills discovered from the data, not on rewards);
  3. the privileged evaluation driver, which samples and selects skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fre.datasets import OfflineDataset
from fre.decoder import mlp
from fre.encoder import SetTransformerBlock, gaussian_kl

from baselines.gc_bc import GaussianActor


@dataclass
class OPALConfig:
    """Configuration for the OPAL baseline."""

    obs_dim: int
    action_dim: int
    skill_dim: int = 128
    context_length: int = 32
    state_embed_dim: int = 64
    action_embed_dim: int = 64
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_dim: int = 256
    decoder_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    policy_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    beta: float = 0.01
    learning_rate: float = 1e-4

    @property
    def token_dim(self) -> int:
        return self.state_embed_dim + self.action_embed_dim


class TrajectoryVAE(nn.Module):
    """Auto-encodes chunks of ``(s, a)`` pairs into a continuous skill ``z``.

    The encoder is the same permutation-invariant transformer used by FRE,
    applied to tokens formed by concatenating a state embedding with an action
    embedding (analogously to FRE's state + reward-embedding tokens).
    """

    def __init__(self, config: OPALConfig) -> None:
        super().__init__()
        self.config = config
        self.state_proj = nn.Linear(config.obs_dim, config.state_embed_dim)
        self.action_proj = nn.Linear(config.action_dim, config.action_embed_dim)
        self.blocks = nn.ModuleList(
            [SetTransformerBlock(config.d_model, config.n_heads, config.mlp_dim) for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.fc_mean = nn.Linear(config.d_model, config.skill_dim)
        self.fc_log_std = nn.Linear(config.d_model, config.skill_dim)
        # Dynamics decoder: p(s_{t+1} | s_t, a_t, z)
        self.decoder = mlp(
            config.obs_dim + config.action_dim + config.skill_dim,
            config.decoder_hidden_dims,
            config.obs_dim,
        )

    def encode(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.state_proj(states), self.action_proj(actions)], dim=-1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.fc_mean(x), self.fc_log_std(x).clamp(-10.0, 2.0)

    def forward(self, states, actions, next_states):
        mean, log_std = self.encode(states, actions)
        z = mean + log_std.exp() * torch.randn_like(log_std)
        z_exp = z.unsqueeze(1).expand(-1, states.shape[1], -1)
        pred_next = self.decoder(torch.cat([states, actions, z_exp], dim=-1))
        recon = torch.nn.functional.mse_loss(pred_next, next_states)
        kl = gaussian_kl(mean, log_std)
        return recon + self.config.beta * kl, recon.detach(), kl.detach()


class OPALAgent:
    """Skill VAE plus a latent-conditioned behavioural-cloning skill policy."""

    def __init__(self, config: OPALConfig, device: torch.device = torch.device("cpu")) -> None:
        self.config = config
        self.device = device
        self.vae = TrajectoryVAE(config).to(device)
        self.policy = GaussianActor(
            config.obs_dim + config.skill_dim,
            config.action_dim,
            config.policy_hidden_dims,
            log_std_min=-5.0,
        ).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=config.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    # -- training ------------------------------------------------------------------
    def update_vae(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> Dict[str, float]:
        loss, recon, kl = self.vae(states, actions, next_states)
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()
        return {"vae_loss": float(loss.item()), "recon": float(recon.item()), "kl": float(kl.item())}

    @torch.no_grad()
    def infer_skills(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, _ = self.vae.encode(states, actions)
        return mean

    def update_policy(self, states: torch.Tensor, actions: torch.Tensor, skills: torch.Tensor) -> Dict[str, float]:
        log_prob = self.policy.log_prob(states, skills, actions)
        loss = -log_prob.mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return {"skill_policy_loss": float(loss.item())}

    # -- inference -----------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, skill: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        skill_t = torch.as_tensor(np.asarray(skill, dtype=np.float32), device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if skill_t.dim() == 1:
            skill_t = skill_t.unsqueeze(0)
        return self.policy.act(obs_t, skill_t).cpu().numpy()

    def sample_skills(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample ``n`` skills from the unit Gaussian prior."""
        rng = rng or np.random.default_rng(0)
        return rng.normal(size=(n, self.config.skill_dim)).astype(np.float32)

    def state_dict(self):
        return {"vae": self.vae.state_dict(), "policy": self.policy.state_dict(), "config": self.config}

    def load_state_dict(self, state) -> None:
        self.vae.load_state_dict(state["vae"])
        self.policy.load_state_dict(state["policy"])


# --------------------------------------------------------------------------------------
# Privileged evaluation
# --------------------------------------------------------------------------------------
def privileged_evaluate(
    env,
    task,
    agent: OPALAgent,
    num_skills: int = 10,
    num_episodes: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    """Privileged OPAL evaluation (the ``OPAL-10`` column of Table 1).

    "10 random skills are sampled from a unit Gaussian, for each skill z the
    policy is conditioned on it and evaluated for the entire episode, and the
    best performing rollout is taken."
    """
    rng = np.random.default_rng(seed)
    skills = agent.sample_skills(num_skills, rng)
    best_return = -np.inf
    best_skill = None
    for skill in skills:
        total = 0.0
        obs = np.asarray(task.reset_env(env, seed), dtype=np.float32).reshape(-1)
        for _ in range(task.max_episode_steps):
            action = agent.act(obs, skill).reshape(-1)
            step_out = env.step(action)
            obs = np.asarray(step_out[0], dtype=np.float32).reshape(-1)
            total += float(np.asarray(task.reward(obs.reshape(1, -1))).reshape(-1)[0])
            if len(step_out) > 2 and bool(np.asarray(step_out[2]).reshape(-1)[0]):
                break
        if total > best_return:
            best_return = total
            best_skill = skill
    raw = np.asarray([best_return], dtype=np.float64)
    return {
        "task": task.name,
        "raw_return_mean": float(raw.mean()),
        "normalized_mean": float(task.normalize(raw).mean()),
        "best_skill": best_skill,
        "num_skills": num_skills,
        "num_episodes": num_episodes,
    }


def sample_trajectory_chunks(
    dataset: OfflineDataset,
    batch_size: int,
    context_length: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Sample contiguous chunks of ``(s, a, s')`` used to train the OPAL VAE."""
    rng = rng or np.random.default_rng(0)
    starts = rng.integers(0, len(dataset), size=batch_size)
    states = np.empty((batch_size, context_length, dataset.encoder_obs_dim), dtype=np.float32)
    actions = np.empty((batch_size, context_length, dataset.action_dim), dtype=np.float32)
    next_states = np.empty_like(states)
    for i, start in enumerate(starts):
        end = min(start + context_length, len(dataset))
        idx = np.arange(start, end)
        if len(idx) < context_length:
            idx = np.concatenate([idx, np.full(context_length - len(idx), idx[-1])])
        states[i] = dataset.encoder_observations[idx]
        actions[i] = dataset.actions[idx]
        next_states[i] = dataset.next_encoder_observations[idx]
    return {
        "states": torch.as_tensor(states),
        "actions": torch.as_tensor(actions),
        "next_states": torch.as_tensor(next_states),
    }
