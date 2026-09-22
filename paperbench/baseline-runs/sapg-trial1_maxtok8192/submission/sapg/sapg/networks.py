"""Shared policy and value networks for SAPG.

The core idea of SAPG is that a *single* shared network ``B_theta`` (actor mean)
and ``C_psi`` (critic) are conditioned on a per-follower / per-leader parameter
vector ``phi_j``.  Each follower ``j`` therefore corresponds to a different
"slice" of the same network, which lets us aggregate all off-policy data from
all followers to train a leader policy.

Architectures (from the paper):

* AllegroKuka:
    - obs -> MLP [768, 512, 256] with ELU
    - -> LSTM (1 layer, 768 hidden)
    - -> actor mean (action dim)
    - sigma is a *fixed learnable vector* (input independent)
* Shadow Hand:
    - MLP [512, 512, 256, 128], ELU
* Allegro Hand:
    - MLP [512, 256, 128], ELU

The ``phi_j`` conditioning is implemented by concatenating ``phi_j`` to the
observation before the first layer (a simple, general and effective scheme).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: str = "elu",
    last_activation: bool = False,
) -> nn.Sequential:
    """Build a plain MLP with the given hidden sizes."""
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if last_activation:
        layers.append(act_cls())
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Actor network ``B_theta`` conditioned on ``phi_j``.

    Parameters
    ----------
    obs_dim:
        Dimension of the environment observation.
    action_dim:
        Dimension of the action.
    phi_dim:
        Dimension of the per-follower conditioning vector ``phi_j``.
    hidden_sizes:
        Hidden layer sizes of the MLP trunk.
    use_lstm:
        If True, an LSTM is placed after the MLP trunk (AllegroKuka).
    lstm_hidden:
        Hidden size of the LSTM.
    learnable_sigma:
        If True, ``sigma`` is a fixed learnable vector (input independent).
        If False, the network outputs both mean and log-std (state dependent).
    per_block_sigma:
        If True, each block has its own learnable sigma vector (entropy
        exploration variant from the ablation).
    num_blocks:
        Number of blocks (only used when ``per_block_sigma`` is True).
    activation:
        Activation function name.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        phi_dim: int = 0,
        hidden_sizes: Sequence[int] = (512, 512, 256, 128),
        use_lstm: bool = False,
        lstm_hidden: int = 768,
        learnable_sigma: bool = True,
        per_block_sigma: bool = False,
        num_blocks: int = 1,
        activation: str = "elu",
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.phi_dim = phi_dim
        self.use_lstm = use_lstm
        self.lstm_hidden = lstm_hidden
        self.learnable_sigma = learnable_sigma
        self.per_block_sigma = per_block_sigma
        self.num_blocks = num_blocks

        in_dim = obs_dim + phi_dim
        self.trunk = _mlp(in_dim, hidden_sizes, hidden_sizes[-1], activation=activation)

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_sizes[-1],
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
            )
            head_in = lstm_hidden
        else:
            self.lstm = None
            head_in = hidden_sizes[-1]

        self.mean_head = nn.Linear(head_in, action_dim)

        if learnable_sigma:
            if per_block_sigma:
                # one sigma vector per block
                self.log_std = nn.Parameter(
                    torch.full((num_blocks, action_dim), init_log_std)
                )
            else:
                self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        else:
            self.log_std_head = nn.Linear(head_in, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # small gain on the mean head for stable initial policy
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _condition(self, obs: torch.Tensor, phi: Optional[torch.Tensor]) -> torch.Tensor:
        if self.phi_dim > 0:
            if phi is None:
                raise ValueError("phi must be provided when phi_dim > 0")
            if phi.dim() == 1:
                phi = phi.unsqueeze(0)
            if phi.shape[0] == 1 and obs.shape[0] != 1:
                phi = phi.expand(obs.shape[0], -1)
            obs = torch.cat([obs, phi], dim=-1)
        return obs

    def _get_log_std(
        self, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.per_block_sigma:
            if block_ids is None:
                block_ids = torch.zeros(1, dtype=torch.long, device=self.log_std.device)
            return self.log_std[block_ids]
        return self.log_std

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_len: int = 1,
        dones: Optional[torch.Tensor] = None,
    ):
        """Return ``(mean, log_std, lstm_state)``.

        When ``use_lstm`` is True the input is expected to be shaped
        ``(batch, seq_len, obs_dim)`` (or ``(batch, obs_dim)`` which is
        reshaped to ``(batch, 1, obs_dim)``).
        """
        if self.use_lstm:
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            b, s, _ = obs.shape
            x = obs.reshape(b * s, -1)
            if self.phi_dim > 0:
                if phi is not None and phi.dim() == 3:
                    phi = phi.reshape(b * s, -1)
                x = self._condition(x, phi)
            x = self.trunk(x)
            x = x.reshape(b, s, -1)
            if lstm_state is None:
                out, new_state = self.lstm(x)
            else:
                out, new_state = self.lstm(x, lstm_state)
            mean = self.mean_head(out)
            if self.learnable_sigma:
                log_std = self._get_log_std(block_ids)
                log_std = log_std.expand_as(mean)
            else:
                log_std = self.log_std_head(out)
            return mean, log_std, new_state

        x = self._condition(obs, phi)
        x = self.trunk(x)
        mean = self.mean_head(x)
        if self.learnable_sigma:
            log_std = self._get_log_std(block_ids)
            log_std = log_std.expand_as(mean)
        else:
            log_std = self.log_std_head(x)
        return mean, log_std, None

    # ------------------------------------------------------------------
    # distribution helpers
    # ------------------------------------------------------------------
    def distribution(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.distributions.Normal, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        mean, log_std, new_state = self.forward(
            obs, phi=phi, block_ids=block_ids, lstm_state=lstm_state
        )
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std), new_state

    def act(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ):
        dist, new_state = self.distribution(
            obs, phi=phi, block_ids=block_ids, lstm_state=lstm_state
        )
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, new_state

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        dist, _ = self.distribution(
            obs, phi=phi, block_ids=block_ids, lstm_state=lstm_state
        )
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network ``C_psi`` conditioned on ``phi_j``."""

    def __init__(
        self,
        obs_dim: int,
        phi_dim: int = 0,
        hidden_sizes: Sequence[int] = (512, 512, 256, 128),
        use_lstm: bool = False,
        lstm_hidden: int = 768,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.phi_dim = phi_dim
        self.use_lstm = use_lstm

        in_dim = obs_dim + phi_dim
        self.trunk = _mlp(in_dim, hidden_sizes, hidden_sizes[-1], activation=activation)

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_sizes[-1],
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
            )
            head_in = lstm_hidden
        else:
            self.lstm = None
            head_in = hidden_sizes[-1]

        self.value_head = nn.Linear(head_in, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _condition(self, obs: torch.Tensor, phi: Optional[torch.Tensor]) -> torch.Tensor:
        if self.phi_dim > 0:
            if phi is None:
                raise ValueError("phi must be provided when phi_dim > 0")
            if phi.dim() == 1:
                phi = phi.unsqueeze(0)
            if phi.shape[0] == 1 and obs.shape[0] != 1:
                phi = phi.expand(obs.shape[0], -1)
            obs = torch.cat([obs, phi], dim=-1)
        return obs

    def forward(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.use_lstm:
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            b, s, _ = obs.shape
            x = obs.reshape(b * s, -1)
            if self.phi_dim > 0:
                if phi is not None and phi.dim() == 3:
                    phi = phi.reshape(b * s, -1)
                x = self._condition(x, phi)
            x = self.trunk(x)
            x = x.reshape(b, s, -1)
            if lstm_state is None:
                out, new_state = self.lstm(x)
            else:
                out, new_state = self.lstm(x, lstm_state)
            value = self.value_head(out).squeeze(-1)
            return value, new_state

        x = self._condition(obs, phi)
        x = self.trunk(x)
        value = self.value_head(x).squeeze(-1)
        return value, None


def build_networks(cfg, obs_dim: int, action_dim: int, num_blocks: int = 1):
    """Factory that builds ``(policy, value)`` from a config dict/namespace.

    ``cfg`` is expected to contain (with sensible defaults):
        hidden_sizes, use_lstm, lstm_hidden, phi_dim, learnable_sigma,
        per_block_sigma, activation.
    """
    def _get(key, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    hidden_sizes = _get("hidden_sizes", (512, 512, 256, 128))
    use_lstm = _get("use_lstm", False)
    lstm_hidden = _get("lstm_hidden", 768)
    phi_dim = _get("phi_dim", 0)
    learnable_sigma = _get("learnable_sigma", True)
    per_block_sigma = _get("per_block_sigma", False)
    activation = _get("activation", "elu")

    policy = GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        phi_dim=phi_dim,
        hidden_sizes=hidden_sizes,
        use_lstm=use_lstm,
        lstm_hidden=lstm_hidden,
        learnable_sigma=learnable_sigma,
        per_block_sigma=per_block_sigma,
        num_blocks=num_blocks,
        activation=activation,
    )
    value = ValueNetwork(
        obs_dim=obs_dim,
        phi_dim=phi_dim,
        hidden_sizes=hidden_sizes,
        use_lstm=use_lstm,
        lstm_hidden=lstm_hidden,
        activation=activation,
    )
    return policy, value
