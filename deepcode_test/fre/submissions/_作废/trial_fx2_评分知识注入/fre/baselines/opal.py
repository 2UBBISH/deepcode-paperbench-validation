"""OPAL offline skill-discovery baseline.

OPAL learns a state-conditioned action-sequence variational autoencoder.
A latent skill ``z`` is encoded from a contiguous action chunk, and the decoder
maps ``(state, z)`` back to the corresponding action sequence.  At test time the
policy is the first action predicted for a chosen skill.

This implementation is intentionally self-contained and mirrors the public
interface of the other baselines (``ForwardBackward`` and ``SuccessorFeatures``)
so :mod:`fre.baselines.baseline_eval` can use the same privileged-evaluation
protocol.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["OPAL", "train_opal_agent"]


def _cfg_value(cfg: Any, *names: str, default: Any = None) -> Any:
    """Read a configuration value using several aliases/fallbacks."""
    for name in names:
        if cfg is None:
            break
        if isinstance(cfg, dict):
            if name in cfg:
                return cfg[name]
        else:
            if hasattr(cfg, name):
                return getattr(cfg, name)
    return default


def _make_mlp(input_dim: int, output_dim: int, hidden_dim: int = 256,
              num_hidden: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_hidden):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class OPAL(nn.Module):
    """Offline Primitive and Abstracted Latent (OPAL) skill VAE.

    Parameters
    ----------
    state_dim:
        Dimension of environment states.
    action_dim:
        Dimension of environment actions.
    skill_dim:
        Dimension of the latent skill vector.
    hidden_dim:
        Hidden width of all MLPs.
    num_hidden:
        Number of hidden layers in the encoder/decoder/prior MLPs.
    sequence_length:
        Number of actions in an action chunk.
    beta:
        KL weight in the VAE objective.
    lr:
        Optimizer learning rate.
    use_learned_prior:
        If ``True`` use a state-conditioned prior ``p(z|s)``; otherwise use
        a fixed standard Gaussian prior.
    device:
        Torch device.
    """

    def __init__(
        self,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        cfg: Optional[Any] = None,
        skill_dim: int = 16,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        sequence_length: int = 10,
        beta: float = 1.0,
        lr: float = 3e-4,
        use_learned_prior: bool = True,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Allow state/action dims to come from the baseline configuration.
        if state_dim is None:
            state_dim = int(_cfg_value(cfg, "state_dim", "state_size", default=0) or 0)
        if action_dim is None:
            action_dim = int(_cfg_value(cfg, "action_dim", "action_size", default=0) or 0)
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError(
                "OPAL requires positive state_dim and action_dim; got "
                f"state_dim={state_dim}, action_dim={action_dim}."
            )

        skill_dim = int(_cfg_value(cfg, "opal_skill_dim", "skill_dim", default=skill_dim) or skill_dim)
        hidden_dim = int(_cfg_value(cfg, "opal_hidden_dim", "hidden_dim", default=hidden_dim) or hidden_dim)
        num_hidden = int(_cfg_value(cfg, "opal_num_hidden", "num_hidden", default=num_hidden) or num_hidden)
        sequence_length = int(
            _cfg_value(cfg, "opal_sequence_length", "sequence_length", default=sequence_length)
            or sequence_length
        )
        beta = float(_cfg_value(cfg, "opal_beta", "beta", default=beta) or beta)
        lr = float(_cfg_value(cfg, "opal_lr", "lr", default=lr) or lr)
        use_learned_prior = bool(
            _cfg_value(cfg, "opal_learned_prior", "use_learned_prior", default=use_learned_prior)
        )

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.skill_dim = int(skill_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden = int(num_hidden)
        self.sequence_length = int(sequence_length)
        self.beta = float(beta)
        self.lr = float(lr)
        self.use_learned_prior = bool(use_learned_prior)
        self.device = torch.device(device)

        self.action_chunk_dim = self.sequence_length * self.action_dim

        # Encoder q(z | s, a_1:T).
        self.encoder = _make_mlp(
            self.state_dim + self.action_chunk_dim,
            2 * self.skill_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=self.num_hidden,
        )

        # Decoder p(a_1:T | s, z). Tanh keeps actions in [-1, 1], which is the
        # usual convention for D4RL/ExORL action spaces.
        self.decoder = _make_mlp(
            self.state_dim + self.skill_dim,
            self.action_chunk_dim,
            hidden_dim=self.hidden_dim,
            num_hidden=self.num_hidden,
        )
        self.decoder.add_module("tanh", nn.Tanh())

        # Optional state-conditioned prior p(z|s).
        if self.use_learned_prior:
            self.prior = _make_mlp(
                self.state_dim,
                2 * self.skill_dim,
                hidden_dim=self.hidden_dim,
                num_hidden=self.num_hidden,
            )
        else:
            self.prior = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

    # ------------------------------------------------------------------
    # Sampling and inference helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def _kl_normal(mu: torch.Tensor, logvar: torch.Tensor,
                   mu_p: Optional[torch.Tensor] = None,
                   logvar_p: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mu_p is None:
            mu_p = torch.zeros_like(mu)
        if logvar_p is None:
            logvar_p = torch.zeros_like(logvar)
        var = torch.exp(logvar)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (
            logvar_p - logvar
            + (var + (mu - mu_p) ** 2) / (var_p + 1e-6)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def _prior_params(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_learned_prior and self.prior is not None:
            out = self.prior(states)
            mu, logvar = out.chunk(2, dim=-1)
            return mu, logvar
        mu = torch.zeros(states.shape[0], self.skill_dim, device=states.device)
        logvar = torch.zeros_like(mu)
        return mu, logvar

    def sample_skills(
        self,
        num_skills: int,
        states: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """Sample skill vectors from the prior.

        When ``states`` is provided and a learned prior is active, samples are
        state-conditioned.  Otherwise the fixed Gaussian prior is used.
        """
        device = torch.device(device or self.device)
        if states is not None and self.use_learned_prior and self.prior is not None:
            states = states.to(device)
            mu, logvar = self._prior_params(states)
            if deterministic:
                return mu
            return self._reparameterize(mu, logvar)
        z = torch.randn(num_skills, self.skill_dim, device=device)
        return z

    def encode_chunk(self, states: torch.Tensor, action_chunks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return posterior ``(mu, logvar)`` for a state/action chunk."""
        x = torch.cat([states, action_chunks], dim=-1)
        out = self.encoder(x)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar

    def decode_chunk(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode an action chunk ``[B, T * action_dim]`` from states and skills."""
        z = z.to(states.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] == 1 and states.shape[0] != 1:
            z = z.expand(states.shape[0], -1)
        if z.shape[0] != states.shape[0]:
            raise ValueError(
                f"Skill batch size {z.shape[0]} does not match state batch size {states.shape[0]}."
            )
        x = torch.cat([states, z], dim=-1)
        return self.decoder(x)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(
        self,
        state: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return the first action of the decoded chunk for ``state``.

        ``condition`` is a skill vector ``z``.  If it is not supplied, a skill
        is sampled from the prior.
        """
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if condition is None:
            condition = self.sample_skills(
                num_skills=state.shape[0],
                states=state,
                deterministic=deterministic,
                device=self.device,
            )
        else:
            condition = condition.to(self.device)

        chunk = self.decode_chunk(state, condition)
        chunk = chunk.reshape(state.shape[0], self.sequence_length, self.action_dim)
        action = chunk[:, 0]
        return action.squeeze(0) if state.shape[0] == 1 else action

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _dataset_array(dataset: Any, name: str) -> Optional[torch.Tensor]:
        """Best-effort extraction of a cached dataset array."""
        candidates = [
            getattr(dataset, name, None),
            getattr(dataset, f"_{name}", None),
        ]
        for cand in candidates:
            if isinstance(cand, np.ndarray):
                return torch.from_numpy(cand)
            if isinstance(cand, torch.Tensor):
                return cand
        data = getattr(dataset, "data", None)
        if isinstance(data, dict):
            raw = data.get(name)
            if isinstance(raw, np.ndarray):
                return torch.from_numpy(raw)
            if isinstance(raw, torch.Tensor):
                return raw
        return None

    def _sample_action_chunks(
        self,
        dataset: Any,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self._dataset_array(dataset, "states")
        actions = self._dataset_array(dataset, "actions")

        if states is None or actions is None:
            # Fall back to single-transition sampling.  This still keeps OPAL
            # trainable, although the skill sequence length is effectively one.
            batch = dataset.sample_transitions(batch_size)
            states = batch.states
            actions = batch.actions
            n = states.shape[0]
            idx = np.arange(n)
            action_chunks = actions.reshape(n, 1, self.action_dim).reshape(n, -1)
            if self.sequence_length > 1:
                action_chunks = action_chunks.repeat_interleave(self.sequence_length, dim=1)
            return states.to(self.device), action_chunks.to(self.device)

        states = states.to(self.device)
        actions = actions.to(self.device)
        n = states.shape[0]
        max_start = max(1, n - self.sequence_length + 1)
        idx = torch.randint(0, max_start, (batch_size,), device=self.device)
        idx = torch.clamp(idx, 0, max_start - 1)
        action_chunks = torch.stack(
            [actions[idx + t] for t in range(self.sequence_length)], dim=1
        )
        action_chunks = action_chunks.reshape(batch_size, -1)
        sampled_states = states[idx]
        return sampled_states, action_chunks

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(
        self,
        states: torch.Tensor,
        action_chunks: torch.Tensor,
    ) -> Dict[str, float]:
        self.train()
        states = states.to(self.device)
        action_chunks = action_chunks.to(self.device)

        mu, logvar = self.encode_chunk(states, action_chunks)
        z = self._reparameterize(mu, logvar)

        pred_chunks = self.decode_chunk(states, z)
        recon_loss = F.mse_loss(pred_chunks, action_chunks, reduction="none").sum(dim=-1).mean()

        mu_p, logvar_p = self._prior_params(states)
        kl = self._kl_normal(mu, logvar, mu_p, logvar_p)

        loss = recon_loss + self.beta * kl

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "recon_loss": float(recon_loss.detach().cpu().item()),
            "kl": float(kl.detach().cpu().item()),
        }

    def train(
        self,
        dataset: Any,
        num_steps: int = 100_000,
        batch_size: int = 256,
        log_every: int = 1000,
    ) -> Dict[str, Any]:
        self.train()
        all_metrics: Dict[str, list[float]] = {"loss": [], "recon_loss": [], "kl": []}
        last_metrics: Dict[str, float] = {}

        for step in range(1, int(num_steps) + 1):
            states, action_chunks = self._sample_action_chunks(dataset, int(batch_size))
            metrics = self.train_step(states, action_chunks)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)
            last_metrics = metrics

            if step % int(log_every) == 0:
                msg = (
                    f"OPAL step {step}/{int(num_steps)} "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )
                logger.info(msg)

        mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
        return {"mean_metrics": mean_metrics, "last_metrics": last_metrics}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "skill_dim": self.skill_dim,
                "sequence_length": self.sequence_length,
                "use_learned_prior": self.use_learned_prior,
            },
            path,
        )
        logger.info("Saved OPAL checkpoint to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if "state_dict" in ckpt:
            self.load_state_dict(ckpt["state_dict"])
        else:
            self.load_state_dict(ckpt)
        self.to(self.device)
        logger.info("Loaded OPAL checkpoint from %s", path)

    def to(self, device: Any) -> "OPAL":
        self.device = torch.device(device)
        return super().to(self.device)


def train_opal_agent(
    dataset: Any,
    cfg: Optional[Any] = None,
    device: str = "cpu",
    num_steps: int = 100_000,
    batch_size: int = 256,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    skill_dim: int = 16,
    hidden_dim: int = 256,
    num_hidden: int = 2,
    sequence_length: int = 10,
    beta: float = 1.0,
    lr: float = 3e-4,
    use_learned_prior: bool = True,
) -> OPAL:
    """Create and train an OPAL agent, returning the trained model."""
    if state_dim is None:
        state_dim = _cfg_value(cfg, "state_dim", "state_size", default=None)
        if state_dim is None:
            states = getattr(dataset, "states", None)
            if states is None:
                sample = dataset.sample_transitions(1)
                states = sample.states
            state_dim = int(states.shape[-1])
    if action_dim is None:
        action_dim = _cfg_value(cfg, "action_dim", "action_size", default=None)
        if action_dim is None:
            actions = getattr(dataset, "actions", None)
            if actions is None:
                sample = dataset.sample_transitions(1)
                actions = sample.actions
            action_dim = int(actions.shape[-1])

    agent = OPAL(
        state_dim=state_dim,
        action_dim=action_dim,
        cfg=cfg,
        skill_dim=skill_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
        sequence_length=sequence_length,
        beta=beta,
        lr=lr,
        use_learned_prior=use_learned_prior,
        device=device,
    )
    agent.train(dataset, num_steps=num_steps, batch_size=batch_size)
    return agent
