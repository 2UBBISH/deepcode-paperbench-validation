"""Centralised value function with a shared backbone conditioned on per-policy latents.

Paper reference
---------------
Section 4.4 ("Encouraging diversity via latent conditioning")::

    "We mitigate this by having a shared backbone B_theta for each policy
     conditioned on hanging parameters phi_j local to each policy. Similarly,
     the critic consists of a shared backbone C_psi conditioned on parameters
     phi_j. The parameters psi, theta are shared across the leader and all
     followers and updated with gradients from each objective, while the
     parameters phi_j are only updated with the objective for that particular
     policy."

Section 5.2: phi_j in R^32 for the AllegroKuka tasks, R^16 for ShadowHand and
AllegroHand.  Appendix B.1-B.3 give the backbone widths:

* AllegroKuka (B.1): observation -> MLP 768 x 512 x 256 (ELU) -> LSTM(1 layer,
  768 hidden units).  Recurrent policy.
* ShadowHand (B.2): MLP 512 x 512 x 256 x 128, ELU, MLP policy.
* AllegroHand (B.3): MLP 512 x 256 x 128, ELU, MLP policy.

As with the actor, the conditioning mechanism itself is not spelled out in the
paper; following the strategy recorded in the reproduction plan we concatenate
phi_j to the observation embedding at the backbone input and initialise
phi_j ~ N(0, I).  The critic *shares* the same phi_j parameter tensor with the
actor of that policy (see :class:`sapg.models.actor.ActorCritic`): the critic is
constructed with ``phi_shared`` pointing at the actor's ``phi`` parameter so
that a single object owns phi_j and the "only updated with the objective for
that particular policy" rule is implementable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .networks import LearnableSigma, MLP, init_weights, make_backbone, mlp_units_for_task

__all__ = ["Critic"]


class Critic(nn.Module):
    """Shared value backbone ``C_psi(o, phi_j) -> V(s)``.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector ``o_t``.
    phi_dim:
        Dimensionality of the per-policy latent ``phi_j`` (0 disables
        conditioning, i.e. a single-policy critic).
    num_policies:
        Number of latent vectors to allocate when ``phi_shared`` is not given.
    mlp_units:
        Hidden widths of the MLP trunk, e.g. ``(768, 512, 256)`` for
        AllegroKuka, ``(512, 512, 256, 128)`` for ShadowHand and
        ``(512, 256, 128)`` for AllegroHand (Appendix B.1-B.3).
    use_lstm:
        Whether to append an LSTM head (AllegroKuka uses a recurrent policy).
    phi_shared:
        Optional existing ``nn.Parameter`` of shape ``[num_policies, phi_dim]``
        owned by the actor.  When supplied it is reused instead of a fresh
        tensor is allocated, so that actor and critic consume exactly the same
        ``phi_j`` (Section 4.4).
    """

    def __init__(
        self,
        obs_dim: int,
        phi_dim: int = 0,
        num_policies: int = 1,
        mlp_units: Sequence[int] = (768, 512, 256),
        activation: str = "elu",
        use_lstm: bool = False,
        lstm_hidden_size: int = 768,
        lstm_num_layers: int = 1,
        use_layer_norm: bool = False,
        phi_init_scale: float = 1.0,
        learnable_phi: bool = True,
        phi_init: str = "normal",
        random_phi: bool = False,
        phi_shared: Optional[nn.Parameter] = None,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Configuration-driven overrides (Appendix B tables)
        # ------------------------------------------------------------------
        if config is not None:
            obs_dim = int(getattr(config, "obs_dim", obs_dim))
            phi_dim = int(getattr(config, "phi_dim", phi_dim))
            num_policies = int(getattr(config, "num_policies", num_policies))
            task = getattr(config, "task_group", None) or getattr(config, "task", None)
            units = getattr(config, "critic_mlp_units", None)
            if units is None and task is not None:
                units = mlp_units_for_task(task)
            if units is not None:
                mlp_units = tuple(int(u) for u in units)
            activation = getattr(config, "actor_activation", activation)
            use_lstm = bool(getattr(config, "use_lstm", use_lstm))
            lstm_hidden_size = int(getattr(config, "lstm_hidden_size", lstm_hidden_size))
            lstm_num_layers = int(getattr(config, "lstm_num_layers", lstm_num_layers))
            learnable_phi = bool(getattr(config, "learnable_phi", learnable_phi))
            random_phi = bool(getattr(config, "random_phi", random_phi))

        self.obs_dim = int(obs_dim)
        self.phi_dim = int(phi_dim)
        self.num_policies = max(1, int(num_policies))
        self.mlp_units: Tuple[int, ...] = tuple(int(u) for u in mlp_units)
        self.activation = activation
        self.use_lstm = bool(use_lstm)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_num_layers = int(lstm_num_layers)
        self.learnable_phi = bool(learnable_phi) and not bool(random_phi)

        # phi_j is concatenated to the *observation* before the trunk.
        input_dim = self.obs_dim + self.phi_dim

        self.backbone = make_backbone(
            input_dim=input_dim,
            mlp_units=self.mlp_units,
            activation=self.activation,
            use_lstm=self.use_lstm,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            use_layer_norm=use_layer_norm,
            output_dim=None,
        )
        trunk_out = int(getattr(self.backbone, "out_features", self.mlp_units[-1]))
        self.trunk_out = trunk_out

        self.value = nn.Linear(trunk_out, 1)
        init_weights(self.value, gain=1.0)
        nn.init.zeros_(self.value.bias)

        # ------------------------------------------------------------------
        # Per-policy latent phi_j (shared with the actor when provided)
        # ------------------------------------------------------------------
        if self.phi_dim > 0:
            if phi_shared is not None:
                if tuple(phi_shared.shape) != (self.num_policies, self.phi_dim):
                    raise ValueError(
                        "phi_shared must have shape "
                        f"({self.num_policies}, {self.phi_dim}), got {tuple(phi_shared.shape)}"
                    )
                self.phi = phi_shared
                self._owns_phi = False
            else:
                self._owns_phi = True
                self.phi = self._make_phi(num_policies, phi_dim, phi_init, phi_init_scale)
        else:
            self._owns_phi = False
            self.register_parameter("phi", None)

        self._last_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # phi helpers
    # ------------------------------------------------------------------
    def _make_phi(
        self, num_policies: int, phi_dim: int, phi_init: str, phi_init_scale: float
    ) -> nn.Parameter:
        if phi_init == "normal":
            values = torch.randn(num_policies, phi_dim) * float(phi_init_scale)
        elif phi_init == "uniform":
            values = (torch.rand(num_policies, phi_dim) * 2.0 - 1.0) * float(phi_init_scale)
        elif phi_init == "zeros":
            values = torch.zeros(num_policies, phi_dim)
        else:
            raise ValueError(f"unknown phi_init '{phi_init}'")
        param = nn.Parameter(values)
        if not self.learnable_phi:
            param.requires_grad_(False)
        return param

    def phi_for(self, policy_index: Optional[int] = None) -> Optional[torch.Tensor]:
        """Return ``phi_j`` (shape ``[phi_dim]``) or ``None`` when unconditioned."""
        if self.phi is None:
            return None
        if policy_index is None:
            return self.phi
        return self.phi[int(policy_index) % self.phi.shape[0]]

    def all_phi(self) -> Optional[torch.Tensor]:
        return self.phi

    def latent_parameters(self):
        """Yield the *owned* phi parameter (empty when shared with the actor)."""
        if self.phi is not None and self._owns_phi:
            yield self.phi

    def critic_parameters(self):
        """Backbone + value head parameters (excludes the shared latent)."""
        for name, param in self.named_parameters():
            if name == "phi":
                continue
            yield param

    def init_hidden(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.use_lstm:
            return None
        device = device if device is not None else self.value.weight.device
        return self.backbone.init_hidden(batch_size, device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute ``V(s)``.

        ``obs`` may be shaped ``[..., obs_dim]``; ``phi`` may be a single vector,
        a ``[batch, phi_dim]`` tensor, or ``None`` (in which case the latent of
        ``policy_index`` is used).
        """
        phi_vec = self._resolve_phi(phi, obs, policy_index)
        x = obs if phi_vec is None else torch.cat([obs, phi_vec], dim=-1)

        out = self.backbone(x, hidden_state, masks)
        if isinstance(out, tuple):
            features, hidden_state = out
            self._last_hidden = hidden_state
        else:
            features = out

        return self.value(features).squeeze(-1)

    def get_value(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(obs, phi, hidden_state, masks, policy_index)

    @property
    def last_hidden(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._last_hidden

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_phi(
        self,
        phi: Optional[torch.Tensor],
        obs: torch.Tensor,
        policy_index: Optional[int],
    ) -> Optional[torch.Tensor]:
        if self.phi_dim <= 0:
            return None

        if phi is None:
            phi = self.phi_for(policy_index)

        if phi is None:
            return None

        phi = torch.as_tensor(phi, device=obs.device, dtype=obs.dtype)
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        if phi.shape[0] == 1 and obs.shape[0] != 1:
            phi = phi.expand(obs.shape[0], -1)
        return phi

    @staticmethod
    def mask_hidden(
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        masks: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Zero the recurrent state of finished environments (auto-reset)."""
        if hidden_state is None or masks is None:
            return hidden_state
        h, c = hidden_state
        m = masks.to(h.dtype).reshape(-1, 1)
        return h * m.unsqueeze(0), c * m.unsqueeze(0)

    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, phi_dim={self.phi_dim}, "
            f"mlp_units={self.mlp_units}, activation={self.activation}, "
            f"use_lstm={self.use_lstm}"
        )
