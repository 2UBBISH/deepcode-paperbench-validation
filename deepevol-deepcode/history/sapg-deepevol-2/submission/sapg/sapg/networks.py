"""Neural network modules for SAPG.

Implements the shared actor ``B_theta`` and critic ``C_psi`` networks that are
conditioned on per-worker parameters ``phi_j`` (one embedding per follower /
leader).  The same shared network weights are used for every worker; only the
conditioning vector ``phi_j`` differs, which is what allows SAPG to run many
"follower" policies in parallel while keeping the parameter count constant.

Two actor variants are supported:

* ``mlp``  -- plain feed-forward MLP (used by ShadowHand / AllegroHand).
* ``lstm`` -- MLP feature extractor followed by an LSTM (used by AllegroKuka).

The critic is always an MLP value head conditioned on ``phi_j``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
def get_activation(name: str) -> nn.Module:
    """Return an activation module by name (defaults to ELU)."""
    name = (name or "elu").lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "elu",
    out_dim: Optional[int] = None,
) -> Tuple[nn.Sequential, int]:
    """Build an MLP and return ``(module, output_dim)``."""
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(get_activation(activation))
        last = h
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))
        last = out_dim
    return nn.Sequential(*layers), last


# ---------------------------------------------------------------------------
# Per-worker conditioning
# ---------------------------------------------------------------------------
class WorkerEmbedding(nn.Module):
    """Learnable per-worker parameter vector ``phi_j``.

    Each follower (and the leader) is identified by an integer index; this
    module maps that index to a learnable embedding vector that is fed into the
    shared actor / critic networks.
    """

    def __init__(self, num_workers: int, embed_dim: int):
        super().__init__()
        self.num_workers = num_workers
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_workers, embed_dim)
        # Small init so that all workers start from (nearly) the same policy.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, worker_ids: torch.Tensor) -> torch.Tensor:
        """Return ``phi_j`` for the given worker ids.

        Args:
            worker_ids: LongTensor of shape ``(N,)`` or ``(N, T)``.
        Returns:
            Tensor of shape ``(*worker_ids.shape, embed_dim)``.
        """
        return self.embedding(worker_ids)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------
class ActorNetwork(nn.Module):
    """Shared actor ``B_theta`` conditioned on per-worker parameters ``phi_j``.

    The network outputs the mean of a Gaussian action distribution.  The
    standard deviation is handled by :class:`~sapg.policy.GaussianPolicy`.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer sizes of the MLP trunk.
        embed_dim: Dimension of the per-worker embedding ``phi_j``.
        num_workers: Number of workers (followers + leader).
        activation: Activation function name.
        use_lstm: If True, append an LSTM after the MLP trunk.
        lstm_hidden: LSTM hidden size (only used when ``use_lstm``).
        lstm_layers: Number of LSTM layers.
        condition_mode: How ``phi_j`` is injected -- ``"concat"`` (concatenate to
            the observation) or ``"add"`` (project and add to the first hidden
            activation).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        embed_dim: int = 16,
        num_workers: int = 1,
        activation: str = "elu",
        use_lstm: bool = False,
        lstm_hidden: int = 768,
        lstm_layers: int = 1,
        condition_mode: str = "concat",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_workers = num_workers
        self.use_lstm = use_lstm
        self.lstm_hidden = lstm_hidden
        self.condition_mode = condition_mode

        self.worker_embedding = WorkerEmbedding(num_workers, embed_dim)

        trunk_in = obs_dim + (embed_dim if condition_mode == "concat" else 0)
        self.trunk, trunk_out = _build_mlp(
            trunk_in, hidden_dims, activation=activation
        )

        if condition_mode == "add":
            # Project phi_j into the trunk output space and add it.
            self.phi_proj = nn.Linear(embed_dim, trunk_out)
        else:
            self.phi_proj = None

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=trunk_out,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            head_in = lstm_hidden
        else:
            self.lstm = None
            head_in = trunk_out

        self.mean_head = nn.Linear(head_in, action_dim)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    # -- helpers -----------------------------------------------------------
    def _condition(
        self, obs: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """Inject the worker embedding into the observation."""
        if self.condition_mode == "concat":
            return torch.cat([obs, phi], dim=-1)
        return obs

    def forward(
        self,
        obs: torch.Tensor,
        worker_ids: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute the Gaussian mean.

        Args:
            obs: ``(N, obs_dim)`` or ``(N, T, obs_dim)`` observations.
            worker_ids: ``(N,)`` or ``(N, T)`` worker indices.
            hidden_state: Optional LSTM ``(h, c)`` state.
            masks: Optional ``(N, T)`` done masks for LSTM state resets.
        Returns:
            ``(mean, new_hidden_state)``.
        """
        phi = self.worker_embedding(worker_ids)
        x = self._condition(obs, phi)
        x = self.trunk(x)

        if self.phi_proj is not None:
            x = x + self.phi_proj(phi)

        if self.use_lstm:
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (N, 1, F)
                squeeze = True
            else:
                squeeze = False
            if hidden_state is None:
                hidden_state = self.init_hidden(x.shape[0], x.device)
            if masks is not None:
                # Reset hidden state where an episode terminated.
                if masks.dim() == 1:
                    masks = masks.unsqueeze(1)
                hidden_state = (
                    hidden_state[0] * masks.unsqueeze(0),
                    hidden_state[1] * masks.unsqueeze(0),
                )
            x, hidden_state = self.lstm(x, hidden_state)
            if squeeze:
                x = x.squeeze(1)
        else:
            hidden_state = None

        mean = self.mean_head(x)
        return mean, hidden_state

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a zero-initialised LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return h, c


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
class CriticNetwork(nn.Module):
    """Shared critic ``C_psi`` conditioned on per-worker parameters ``phi_j``.

    Outputs a scalar state-value estimate.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        embed_dim: int = 16,
        num_workers: int = 1,
        activation: str = "elu",
        condition_mode: str = "concat",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.num_workers = num_workers
        self.condition_mode = condition_mode

        self.worker_embedding = WorkerEmbedding(num_workers, embed_dim)

        trunk_in = obs_dim + (embed_dim if condition_mode == "concat" else 0)
        self.trunk, trunk_out = _build_mlp(
            trunk_in, hidden_dims, activation=activation
        )
        if condition_mode == "add":
            self.phi_proj = nn.Linear(embed_dim, trunk_out)
        else:
            self.phi_proj = None

        self.value_head = nn.Linear(trunk_out, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, obs: torch.Tensor, worker_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return the value estimate of shape ``(*batch, 1)``."""
        phi = self.worker_embedding(worker_ids)
        if self.condition_mode == "concat":
            x = torch.cat([obs, phi], dim=-1)
        else:
            x = obs
        x = self.trunk(x)
        if self.phi_proj is not None:
            x = x + self.phi_proj(phi)
        return self.value_head(x)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_actor(cfg, obs_dim: int, action_dim: int, num_workers: int) -> ActorNetwork:
    """Build an :class:`ActorNetwork` from a config object / dict.

    Recognised keys: ``hidden_dims``, ``embed_dim``, ``activation``,
    ``use_lstm``, ``lstm_hidden``, ``lstm_layers``, ``condition_mode``.
    """
    get = _cfg_getter(cfg)
    return ActorNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=get("hidden_dims", (512, 256, 128)),
        embed_dim=get("embed_dim", 16),
        num_workers=num_workers,
        activation=get("activation", "elu"),
        use_lstm=get("use_lstm", False),
        lstm_hidden=get("lstm_hidden", 768),
        lstm_layers=get("lstm_layers", 1),
        condition_mode=get("condition_mode", "concat"),
    )


def build_critic(cfg, obs_dim: int, num_workers: int) -> CriticNetwork:
    """Build a :class:`CriticNetwork` from a config object / dict."""
    get = _cfg_getter(cfg)
    return CriticNetwork(
        obs_dim=obs_dim,
        hidden_dims=get("critic_hidden_dims", get("hidden_dims", (512, 256, 128))),
        embed_dim=get("embed_dim", 16),
        num_workers=num_workers,
        activation=get("activation", "elu"),
        condition_mode=get("condition_mode", "concat"),
    )


def _cfg_getter(cfg):
    """Return a ``get(key, default)`` callable for dicts or attribute objects."""
    if isinstance(cfg, dict):
        return lambda k, d=None: cfg.get(k, d)

    def _get(k, d=None):
        return getattr(cfg, k, d)

    return _get
