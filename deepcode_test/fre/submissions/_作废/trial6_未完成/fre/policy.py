"""
Policy network for FRE-conditioned offline RL.

Implements a Gaussian policy π(a|s,z) for continuous action spaces,
and a discrete policy for discrete action spaces, both conditioned on
the latent vector z produced by the FRE encoder.

Architecture:
    Input: concatenate(state, z) -> MLP -> action distribution parameters.
    For continuous actions: outputs mean and log_std of a Gaussian.
    For discrete actions: outputs logits for a categorical distribution.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helper: weight initialization
# ---------------------------------------------------------------------------

def _init_weights(m: nn.Module, gain: float = 1.0) -> None:
    """Initialize linear layers with orthogonal / near-orthogonal weights."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# MLP backbone shared by all policy variants
# ---------------------------------------------------------------------------

class MLPBackbone(nn.Module):
    """Simple MLP with ReLU activations and optional layer norm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: nn.Module = nn.ReLU,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Gaussian Policy (continuous actions)
# ---------------------------------------------------------------------------

class GaussianPolicy(nn.Module):
    """
    Gaussian policy π(a|s,z) for continuous action spaces.

    Input: concatenation of state s and latent z.
    Output: mean μ(s,z) and log standard deviation log σ.

    The log_std can be state-independent (learned parameter) or
    state-dependent (output of the network). We default to a learned
    parameter for simplicity and stability, following common practice
    in offline RL (e.g., IQL, CQL).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_latent: int = 64,
        hidden_dims: Optional[list] = None,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        use_state_dependent_std: bool = False,
    ):
        """
        Args:
            state_dim: Dimension of state observations.
            action_dim: Dimension of action space.
            d_latent: Dimension of latent vector z.
            hidden_dims: List of hidden layer sizes (default: [256, 256]).
            activation: Activation function class.
            dropout: Dropout probability.
            log_std_min: Minimum log standard deviation.
            log_std_max: Maximum log standard deviation.
            use_state_dependent_std: If True, log_std is output by the network;
                otherwise a learned parameter independent of state.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_latent = d_latent
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_state_dependent_std = use_state_dependent_std

        input_dim = state_dim + d_latent

        # Shared trunk
        self.trunk = MLPBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],
            activation=activation,
            dropout=dropout,
        )

        # Mean head
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)

        # Log std head or learned parameter
        if use_state_dependent_std:
            self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.apply(_init_weights)

    def forward(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action distribution parameters.

        Args:
            state: (batch, state_dim) or (batch, num_states, state_dim)
            z: (batch, d_latent)

        Returns:
            mean: (batch, action_dim) or (batch, num_states, action_dim)
            log_std: same shape as mean
        """
        # Handle multi-state input (e.g., for evaluation over multiple states)
        if state.dim() == 3:
            # (batch, num_states, state_dim)
            batch_size, num_states, _ = state.shape
            # Expand z to match: (batch, num_states, d_latent)
            z_expanded = z.unsqueeze(1).expand(-1, num_states, -1)
            x = torch.cat([state, z_expanded], dim=-1)
            # Flatten for MLP
            x_flat = x.reshape(batch_size * num_states, -1)
            h = self.trunk(x_flat)
            mean = self.mean_head(h)
            if self.use_state_dependent_std:
                log_std = self.log_std_head(h)
            else:
                log_std = self.log_std.expand(batch_size * num_states, -1)
            # Reshape back
            mean = mean.reshape(batch_size, num_states, self.action_dim)
            log_std = log_std.reshape(batch_size, num_states, self.action_dim)
        else:
            # (batch, state_dim)
            x = torch.cat([state, z], dim=-1)
            h = self.trunk(x)
            mean = self.mean_head(h)
            if self.use_state_dependent_std:
                log_std = self.log_std_head(h)
            else:
                log_std = self.log_std.expand_as(mean)

        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: (batch, state_dim)
            z: (batch, d_latent)
            deterministic: If True, return mean action (no noise).

        Returns:
            action: (batch, action_dim)
            log_prob: (batch,) log probability of the sampled action
        """
        mean, log_std = self.forward(state, z)
        std = torch.exp(log_std)

        if deterministic:
            action = mean
        else:
            # Reparameterization trick
            noise = torch.randn_like(mean)
            action = mean + std * noise

        # Compute log probability
        # log π(a|s,z) = -0.5 * ( (a-μ)/σ )^2 - log(σ) - 0.5*log(2π)
        # Sum over action dimensions
        log_prob = -0.5 * (((action - mean) / (std + 1e-6)) ** 2).sum(dim=-1)
        log_prob = log_prob - log_std.sum(dim=-1)
        log_prob = log_prob - 0.5 * self.action_dim * np.log(2 * np.pi)

        return action, log_prob

    def get_log_prob(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of a given action under the policy.

        Args:
            state: (batch, state_dim)
            z: (batch, d_latent)
            action: (batch, action_dim)

        Returns:
            log_prob: (batch,) log probability
        """
        mean, log_std = self.forward(state, z)
        std = torch.exp(log_std)

        log_prob = -0.5 * (((action - mean) / (std + 1e-6)) ** 2).sum(dim=-1)
        log_prob = log_prob - log_std.sum(dim=-1)
        log_prob = log_prob - 0.5 * self.action_dim * np.log(2 * np.pi)

        return log_prob

    def get_action_mean(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Return the mean action (deterministic policy)."""
        mean, _ = self.forward(state, z)
        return mean


# ---------------------------------------------------------------------------
# Discrete Policy (for discrete action spaces)
# ---------------------------------------------------------------------------

class DiscretePolicy(nn.Module):
    """
    Discrete (categorical) policy π(a|s,z) for discrete action spaces.

    Input: concatenation of state s and latent z.
    Output: logits over discrete actions.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        d_latent: int = 64,
        hidden_dims: Optional[list] = None,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ):
        """
        Args:
            state_dim: Dimension of state observations.
            num_actions: Number of discrete actions.
            d_latent: Dimension of latent vector z.
            hidden_dims: List of hidden layer sizes (default: [256, 256]).
            activation: Activation function class.
            dropout: Dropout probability.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.d_latent = d_latent

        input_dim = state_dim + d_latent

        self.net = MLPBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_actions,
            activation=activation,
            dropout=dropout,
        )

        self.apply(_init_weights)

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits.

        Args:
            state: (batch, state_dim)
            z: (batch, d_latent)

        Returns:
            logits: (batch, num_actions)
        """
        x = torch.cat([state, z], dim=-1)
        return self.net(x)

    def sample(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the categorical distribution.

        Args:
            state: (batch, state_dim)
            z: (batch, d_latent)
            deterministic: If True, return argmax action.

        Returns:
            action: (batch,) integer actions
            log_prob: (batch,) log probability of sampled actions
        """
        logits = self.forward(state, z)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Log probability of the chosen action
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob

    def get_log_prob(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of given actions.

        Args:
            state: (batch, state_dim)
            z: (batch, d_latent)
            action: (batch,) integer actions

        Returns:
            log_prob: (batch,)
        """
        logits = self.forward(state, z)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

    def get_probs(
        self, state: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Return action probabilities."""
        logits = self.forward(state, z)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_policy(
    state_dim: int,
    action_dim: int,
    d_latent: int = 64,
    hidden_dims: Optional[list] = None,
    discrete: bool = False,
    num_actions: Optional[int] = None,
    **kwargs,
) -> Union[GaussianPolicy, DiscretePolicy]:
    """
    Create a policy network conditioned on latent z.

    Args:
        state_dim: State dimension.
        action_dim: Action dimension (ignored if discrete=True).
        d_latent: Latent dimension.
        hidden_dims: Hidden layer sizes.
        discrete: Whether to create a discrete policy.
        num_actions: Number of discrete actions (required if discrete=True).
        **kwargs: Additional arguments passed to the policy constructor.

    Returns:
        GaussianPolicy or DiscretePolicy instance.
    """
    if discrete:
        if num_actions is None:
            raise ValueError("num_actions must be provided for discrete policy.")
        return DiscretePolicy(
            state_dim=state_dim,
            num_actions=num_actions,
            d_latent=d_latent,
            hidden_dims=hidden_dims,
            **kwargs,
        )
    else:
        return GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            d_latent=d_latent,
            hidden_dims=hidden_dims,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test Gaussian policy
    state_dim = 29  # AntMaze state dim
    action_dim = 8
    d_latent = 64
    batch_size = 256

    policy = GaussianPolicy(state_dim, action_dim, d_latent)

    state = torch.randn(batch_size, state_dim)
    z = torch.randn(batch_size, d_latent)

    mean, log_std = policy.forward(state, z)
    print(f"Gaussian policy - mean shape: {mean.shape}, log_std shape: {log_std.shape}")

    action, log_prob = policy.sample(state, z)
    print(f"Sampled action shape: {action.shape}, log_prob shape: {log_prob.shape}")

    # Test with multi-state input
    num_states = 32
    state_multi = torch.randn(batch_size, num_states, state_dim)
    mean_multi, log_std_multi = policy.forward(state_multi, z)
    print(f"Multi-state mean shape: {mean_multi.shape}")

    # Test discrete policy
    num_actions = 4
    disc_policy = DiscretePolicy(state_dim, num_actions, d_latent)
    logits = disc_policy.forward(state, z)
    print(f"Discrete policy logits shape: {logits.shape}")

    action_d, log_prob_d = disc_policy.sample(state, z)
    print(f"Discrete action shape: {action_d.shape}, log_prob shape: {log_prob_d.shape}")

    print("All policy tests passed!")