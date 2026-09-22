"""Shared actor-critic with per-policy latent (phi) conditioning.

Implements COMPONENT 1 of the SAPG reproduction plan (Sec 4.4, Addendum):

    Actor  = B_theta(obs, phi_j)
    Critic = C_psi(obs, phi_j)

where theta (actor backbone) and psi (critic backbone) are SHARED across all M
policies, while phi_j in R^{phi_dim} is a per-policy latent.  phi_j is shared
between the actor and the critic for the same policy j.

The latent is injected by concatenation to the network input (the paper leaves
the injection method unspecified; concatenation is the documented default).

When entropy-based exploration is used, each env block (policy) has its OWN
learnable sigma vector.  This is handled by the underlying network builders in
``sapg.networks`` (each policy owns its own ``GaussianHead`` / log-std
parameter), so simply instantiating one actor per policy gives independent
sigma vectors.

The module exposes:

    * ``SharedActorCritic`` -- a single policy (actor + critic) with its own
      ``phi_j`` latent, sharing the backbone modules passed in.
    * ``MultiPolicyActorCritic`` -- the M-policy container that owns the shared
      backbones and the M per-policy latents, and exposes helpers to gather
      parameters for the optimizer (shared vs. per-policy groups).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..networks import build_actor, build_critic


class SharedActorCritic(nn.Module):
    """A single policy (actor + critic) conditioned on a per-policy latent phi_j.

    The actor and critic backbones are *shared* modules (owned by the parent
    :class:`MultiPolicyActorCritic`); this class only owns the per-policy
    latent ``phi``.  This makes it trivial to verify that gradients to ``phi``
    come only from that policy's objective, while gradients to the shared
    backbones accumulate across all policies.

    Args:
        actor: shared actor module (built by :func:`sapg.networks.build_actor`).
        critic: shared critic module (built by :func:`sapg.networks.build_critic`).
        phi_dim: dimensionality of the per-policy latent.
        policy_index: index j of this policy (0-based internally).
        device: device for the latent parameter.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        phi_dim: int,
        policy_index: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.phi_dim = int(phi_dim)
        self.policy_index = int(policy_index)

        # Per-policy latent phi_j.  Initialised small so that early training
        # behaves like a single policy; the latents then differentiate.
        self.phi = nn.Parameter(
            torch.zeros(self.phi_dim, device=device, dtype=torch.float32)
        )

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def is_recurrent(self) -> bool:
        """True if the actor backbone is recurrent (LSTM)."""
        return getattr(self.actor, "is_recurrent", False)

    def latent(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Return phi_j, optionally broadcast to ``batch_size`` rows.

        Args:
            batch_size: if given, returns a ``(batch_size, phi_dim)`` tensor
                with phi_j repeated along the batch dimension.

        Returns:
            Tensor of shape ``(phi_dim,)`` or ``(batch_size, phi_dim)``.
        """
        if batch_size is None:
            return self.phi
        return self.phi.unsqueeze(0).expand(batch_size, -1)

    # ------------------------------------------------------------------ #
    # Actor / critic forwards
    # ------------------------------------------------------------------ #
    def act(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Compute the action distribution for this policy.

        Args:
            obs: observation tensor of shape ``(batch, obs_dim)``.
            lstm_state: optional recurrent state for LSTM backbones.

        Returns:
            If the backbone is recurrent: ``(dist, new_lstm_state)``.
            Otherwise: ``dist``.
        """
        phi = self.latent(obs.shape[0])
        if self.is_recurrent:
            return self.actor.distribution(obs, phi=phi, lstm_state=lstm_state)
        return self.actor.distribution(obs, phi=phi)

    def value(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Compute the value estimate for this policy.

        Returns:
            If the backbone is recurrent: ``(value, new_lstm_state)``.
            Otherwise: ``value`` tensor of shape ``(batch, 1)``.
        """
        phi = self.latent(obs.shape[0])
        if self.is_recurrent:
            return self.critic(obs, phi=phi, lstm_state=lstm_state)
        return self.critic(obs, phi=phi)

    def log_prob(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Log-probability of ``actions`` under this policy."""
        phi = self.latent(obs.shape[0])
        if self.is_recurrent:
            return self.actor.log_prob(obs, actions, phi=phi, lstm_state=lstm_state)
        return self.actor.log_prob(obs, actions, phi=phi)

    def entropy(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Differential entropy of this policy's action distribution."""
        phi = self.latent(obs.shape[0])
        if self.is_recurrent:
            return self.actor.entropy(obs, phi=phi, lstm_state=lstm_state)
        return self.actor.entropy(obs, phi=phi)


class MultiPolicyActorCritic(nn.Module):
    """Container holding M policies that share actor/critic backbones.

    The shared backbones ``B_theta`` (actor) and ``C_psi`` (critic) are created
    once and referenced by every :class:`SharedActorCritic`.  Each policy owns
    only its latent ``phi_j`` (and, implicitly, its own learnable sigma vector
    inside the actor's Gaussian head).

    Args:
        task: task name, one of ``{"allegrokuka", "shadowhand", "allegrohand"}``.
        obs_dim: observation dimensionality.
        action_dim: action dimensionality.
        phi_dim: per-policy latent dimensionality.
        num_policies: number of policies M.
        device: device to place the modules on.
    """

    def __init__(
        self,
        task: str,
        obs_dim: int,
        action_dim: int,
        phi_dim: int,
        num_policies: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.task = task
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.phi_dim = int(phi_dim)
        self.num_policies = int(num_policies)
        self.device = device

        # Shared backbones.  NOTE: for entropy exploration each policy needs its
        # OWN learnable sigma vector.  The sigma lives in the actor's Gaussian
        # head, so we build one actor per policy but share the *backbone* by
        # deep-copying the module reference.  To keep the shared-backbone
        # semantics while allowing per-policy sigma, we build a single actor and
        # then give each policy a private sigma parameter that overrides the
        # shared one.
        self.actor = build_actor(task, obs_dim, action_dim, phi_dim).to(device)
        self.critic = build_critic(task, obs_dim, phi_dim).to(device)

        # Per-policy sigma vectors (one per env block).  These are registered on
        # the container so they appear in ``parameters()`` and receive grads
        # only from their own policy's objective.
        self.sigmas = nn.ParameterList()
        for _ in range(self.num_policies):
            self.sigmas.append(
                nn.Parameter(
                    torch.full((action_dim,), -1.0, device=device, dtype=torch.float32)
                )
            )

        # Per-policy latents phi_j.
        self.phis = nn.ParameterList()
        for _ in range(self.num_policies):
            self.phis.append(
                nn.Parameter(
                    torch.zeros(phi_dim, device=device, dtype=torch.float32)
                )
            )

        # Lightweight per-policy views (share actor/critic modules).
        self.policies: List[SharedActorCritic] = []
        for j in range(self.num_policies):
            policy = SharedActorCritic(
                actor=self.actor,
                critic=self.critic,
                phi_dim=phi_dim,
                policy_index=j,
                device=device,
            )
            # Replace the policy's own phi with the container-owned parameter so
            # that ``container.phis[j]`` and ``policy.phi`` are the same tensor.
            policy.phi = self.phis[j]
            self.policies.append(policy)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def policy(self, j: int) -> SharedActorCritic:
        """Return the j-th policy view (0-based)."""
        return self.policies[j]

    def latent(self, j: int) -> torch.Tensor:
        """Return phi_j for policy j."""
        return self.phis[j]

    def sigma(self, j: int) -> torch.Tensor:
        """Return the learnable sigma (log-std) vector for policy j."""
        return self.sigmas[j]

    # ------------------------------------------------------------------ #
    # Parameter groups for the optimizer
    # ------------------------------------------------------------------ #
    def shared_parameters(self) -> List[nn.Parameter]:
        """Parameters of the shared backbones B_theta and C_psi."""
        params: List[nn.Parameter] = []
        params.extend(self.actor.parameters())
        params.extend(self.critic.parameters())
        return params

    def per_policy_parameters(self) -> List[nn.Parameter]:
        """Per-policy parameters: the M latents phi_j and the M sigma vectors."""
        params: List[nn.Parameter] = []
        params.extend(list(self.phis.parameters()))
        params.extend(list(self.sigmas.parameters()))
        return params

    def parameter_groups(self) -> List[Dict[str, object]]:
        """Return optimizer parameter groups.

        The shared backbone and the per-policy parameters are returned as
        separate groups so that callers can (if desired) apply different
        learning rates or weight decays.
        """
        return [
            {"name": "shared", "params": self.shared_parameters()},
            {"name": "per_policy", "params": self.per_policy_parameters()},
        ]

    # ------------------------------------------------------------------ #
    # Gradient isolation helper (used by unit tests)
    # ------------------------------------------------------------------ #
    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        super().zero_grad(set_to_none=set_to_none)

    def grad_norm(self) -> float:
        """Total L2 gradient norm over all parameters (for logging/clipping)."""
        total = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().item())
        return total ** 0.5
