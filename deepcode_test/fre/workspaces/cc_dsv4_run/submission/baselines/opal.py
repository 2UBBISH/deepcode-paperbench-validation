"""
OPAL (Offline Primitive Discovery for Accelerating Offline RL) baseline.

Implements the OPAL unsupervised skill discovery method from Ajay et al. (2020):
- Encoder: transformer-based (same architecture as FRE) produces latent skill z
- Decoder: action reconstruction conditioned on latent skill
- Latent skills discovered by auto-encoding trajectory chunks (behavioral cloning)
- At evaluation: 10 random skills sampled from unit Gaussian, best performing selected

Per addendum:
- Same transformer architecture as FRE for the encoder
- OPAL's task policy is NOT used during evaluation
- 10 random skills sampled from N(0,I), best rollout selected (privileged execution)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Dict, Callable
import copy


class OPALEncoder(nn.Module):
    """
    OPAL encoder: encodes a trajectory chunk (state-action sequence) into latent z.

    Uses the same transformer architecture as FRE:
    - State projected to 64-dim, action projected to 64-dim
    - Concatenated to 128-dim, passed through 4-layer transformer
    - Output pooled to parametrize μ and log σ of latent z
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        state_embed_dim: int = 64,
        action_embed_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
    ):
        super().__init__()
        embed_dim = state_embed_dim + action_embed_dim

        self.state_embed = nn.Linear(state_dim, state_embed_dim)
        self.action_embed = nn.Linear(action_dim, action_embed_dim)

        from fre.encoder import TransformerEncoderBlock
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, mlp_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = states.shape

        state_emb = self.state_embed(states)    # (B, T, 64)
        action_emb = self.action_embed(actions)  # (B, T, 64)
        x = torch.cat([state_emb, action_emb], dim=-1)  # (B, T, 128)

        for block in self.encoder_blocks:
            x = block(x)
        x = self.ln_final(x)
        x_pooled = x.mean(dim=1)  # (B, 128)

        mu = self.fc_mu(x_pooled)
        logvar = self.fc_logvar(x_pooled)
        return mu, logvar

    def encode(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.forward(states, actions)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class OPALDecoder(nn.Module):
    """
    OPAL decoder: reconstructs actions given a state and latent skill z.

    Network: MLP [512, 512, 512] predicting Gaussian action distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        input_dim = state_dim + latent_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, z], dim=-1)
        x = self.backbone(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-5.0)
        return mean, log_std

    def sample_action(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, z)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        action = mean + eps * std
        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-6)).pow(2)
            + 2 * log_std
            + math.log(2 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob


class OPALModel(nn.Module):
    """
    Full OPAL autoencoder: encodes trajectory chunks, decodes actions.
    Trained via behavioral cloning (MLE) on the actions from the trajectory chunk.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
    ):
        super().__init__()
        self.encoder = OPALEncoder(state_dim, action_dim, latent_dim)
        self.decoder = OPALDecoder(state_dim, action_dim, latent_dim, hidden_dims)
        self.latent_dim = latent_dim

    def forward(
        self,
        chunk_states: torch.Tensor,   # (B, T, state_dim)
        chunk_actions: torch.Tensor,  # (B, T, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: total_loss, recon_loss, kl_loss
        """
        B, T, D = chunk_states.shape

        mu, logvar = self.encoder.forward(chunk_states, chunk_actions)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (B, latent_dim)

        # Decode actions for each timestep
        z_expanded = z.unsqueeze(1).expand(-1, T, -1)  # (B, T, latent_dim)
        states_flat = chunk_states.view(B * T, D)
        z_flat = z_expanded.reshape(B * T, -1)

        mean, log_std = self.decoder.forward(states_flat, z_flat)
        mean = mean.view(B, T, -1)
        log_std = log_std.view(B, T, -1)

        # NLL of reconstructed actions
        std_act = torch.exp(log_std) + 1e-6
        var = std_act.pow(2)
        nll = 0.5 * (
            ((chunk_actions - mean).pow(2) / var)
            + 2 * log_std
            + math.log(2 * math.pi)
        )
        nll = nll.sum(dim=-1).mean()  # mean over B, T, action_dim -> scalar

        # KL against unit Gaussian
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        beta = 0.01  # same as FRE
        total_loss = nll + beta * kl_loss

        return total_loss, nll, kl_loss


class OPAL:
    """
    OPAL training and evaluation pipeline.

    Training: auto-encode trajectory chunks via behavioral cloning.
    Evaluation (privileged): sample 10 random z ~ N(0,I), evaluate each,
    return the best-performing rollout.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        lr: float = 1e-4,
        chunk_length: int = 10,
        device: str = "cpu",
    ):
        self.model = OPALModel(state_dim, action_dim, latent_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device(device)
        self.chunk_length = chunk_length
        self.latent_dim = latent_dim
        self.state_dim = state_dim

    def train_step(
        self,
        chunk_states: torch.Tensor,
        chunk_actions: torch.Tensor,
    ) -> dict:
        self.optimizer.zero_grad()
        total_loss, nll, kl_loss = self.model(chunk_states, chunk_actions)
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "nll": nll.item(),
            "kl_loss": kl_loss.item(),
        }

    def get_action(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Get action for a given latent skill z."""
        s = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
        z = z.unsqueeze(0).to(self.device) if z.dim() == 1 else z.to(self.device)
        mean, _ = self.model.decoder.forward(s, z)
        return mean.squeeze(0).cpu()

    def evaluate_privileged(
        self,
        env_step_fn: Callable,
        initial_state_fn: Callable,
        reward_fn: Callable,
        num_skills: int = 10,
        num_episodes: int = 20,
        max_steps: int = 500,
    ) -> Tuple[float, float]:
        """
        Privileged OPAL evaluation: sample 10 random skills, evaluate each
        for the entire episode, take the best performing.

        Returns: (best_mean_return, best_std_return)
        """
        best_mean = -float('inf')
        best_results = None

        for skill_idx in range(num_skills):
            z = torch.randn(self.latent_dim, device=self.device)
            returns = []

            for ep in range(num_episodes):
                state = initial_state_fn()
                ep_return = 0.0
                done = False
                step = 0

                while not done and step < max_steps:
                    action = self.get_action(
                        torch.tensor(state, dtype=torch.float32), z
                    )
                    next_state, done = env_step_fn(state, action.numpy())
                    reward = reward_fn(torch.tensor(next_state, dtype=torch.float32))
                    if hasattr(reward, 'item'):
                        reward = reward.item()
                    ep_return += reward
                    state = next_state
                    step += 1

                returns.append(ep_return)

            mean_ret = np.mean(returns)
            if mean_ret > best_mean:
                best_mean = mean_ret
                best_results = returns

        return best_mean, np.std(best_results)

    def save(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])