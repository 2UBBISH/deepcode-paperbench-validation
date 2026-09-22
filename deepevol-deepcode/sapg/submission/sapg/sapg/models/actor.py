"""Actor networks for SAPG.

Paper reference (SAPG, "Split and Aggregate Policy Gradients"):

* Sec. 4.4 -- *Encouraging diversity via latent conditioning*:

    "We mitigate this by having a shared backbone :math:`B_\\theta` for each policy
     conditioned on [latent] parameters :math:`\\phi_j` local to each policy. Similarly, the
     critic consists of a shared backbone :math:`C_\\psi` conditioned on parameters
     :math:`\\phi_j`. The parameters :math:`\\psi, \\theta` are shared across the leader and
     all followers and updated with gradients from each objective, while the parameters
     :math:`\\phi_j` are only updated with the objective for that particular policy.
     We choose :math:`\\phi_j \\in \\mathbb{R}^{32}` for complex environments while
     :math:`\\phi_j \\in \\mathbb{R}^{16}` for the relatively simpler ones."

* Appendix B.1 -- AllegroKuka tasks: "We use a Gaussian policy where the mean network is
  an LSTM with 1 layer containing 768 hidden units. The observation is also passed through
  an MLP of with hidden layer dimensions 768 x 512 x 256 and an ELU activation before being
  input to the LSTM. The sigma for the Gaussian is a fixed learnable vector independent of
  input observation."

* Appendix B.3 (note) -- for the entropy-exploration experiments each block owns its own
  learnable sigma vector (see :class:`~sapg.models.networks.LearnableSigma` with
  ``num_policies > 1``).

The paper does not specify *how* :math:`\\phi_j` is injected into the backbone; following the
reproduction plan we concatenate :math:`\\phi_j` to the observation (backbone input) and
initialise :math:`\\phi_j \\sim \\mathcal{N}(0, I)`.

This module also provides :class:`ActorCritic`, the policy object consumed by
``sapg.algorithms.rollout`` / ``sapg.algorithms.ppo``: it bundles the shared actor backbone
with the critic backbone (``C_\\psi``, see :mod:`sapg.models.critic`) so that a single object
can ``act`` (actions/logprobs/values) and be ``evaluate_actions``-ed during the update.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Normal

from .networks import LearnableSigma, MLP, make_backbone, mlp_units_for_task

__all__ = [
    "Actor",
    "ActorCritic",
    "Policy",
    "squash_action",
    "tanh_log_prob",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _as_phi_batch(
    phi: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Normalise a per-policy latent ``phi_j`` to shape ``[batch_size, phi_dim]``.

    ``phi`` may be a single ``[phi_dim]`` vector (broadcast over the batch) or already a
    ``[batch, phi_dim]`` tensor.
    """

    if phi is None:
        return None
    if not torch.is_tensor(phi):
        phi = torch.as_tensor(phi, device=device, dtype=dtype)
    phi = phi.to(device=device, dtype=dtype)
    if phi.dim() == 1:
        phi = phi.unsqueeze(0).expand(batch_size, -1).contiguous()
    elif phi.dim() == 2 and phi.shape[0] == 1:
        phi = phi.expand(batch_size, -1).contiguous()
    elif phi.dim() == 2 and phi.shape[0] != batch_size:
        raise ValueError(
            f"phi has batch dimension {phi.shape[0]}, expected 1 or {batch_size}"
        )
    return phi


def squash_action(raw_action: torch.Tensor, action_scale: float = 1.0) -> torch.Tensor:
    """Map an unbounded sample to the bounded action range via ``scale * tanh(x)``."""

    return action_scale * torch.tanh(raw_action)


def tanh_log_prob(
    dist: Normal,
    raw_action: torch.Tensor,
    action_scale: float = 1.0,
) -> torch.Tensor:
    """Log-density of ``tanh``-squashed Gaussian samples (change-of-variables).

    ``log p(a) = log N(x) - sum_i log(1 - tanh(x_i)^2) - action_dim * log(action_scale)``
    where ``a = action_scale * tanh(x)``.
    """

    log_prob = dist.log_prob(raw_action).sum(-1)
    log_det = torch.log1p(-torch.tanh(raw_action).pow(2) + 1e-6).sum(-1)
    log_prob = log_prob - log_det
    if action_scale != 1.0:
        log_prob = log_prob - raw_action.shape[-1] * math.log(abs(action_scale))
    return log_prob


# --------------------------------------------------------------------------------------
# Actor
# --------------------------------------------------------------------------------------
class Actor(nn.Module):
    """Gaussian actor ``pi_theta(a | o, phi_j)`` with a shared backbone ``B_theta``.

    Architecture follows Appendix B.1-B.3: an MLP trunk followed by an optional LSTM head,
    and an input-independent learnable sigma vector for the Gaussian policy.  The per-policy
    latent ``phi_j`` is concatenated to the observation before the trunk, and the learnable
    phi vectors live in ``self.phi_parameters`` (shape ``[num_policies, phi_dim]``) so that
    gradients from policy ``j``'s objective flow only into ``phi_j`` (Sec. 4.4).

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality (excluding ``phi_j``).
    action_dim : int
        Action dimensionality.
    phi_dim : int
        Latent conditioning dimension: 32 for the complex AllegroKuka tasks, 16 for the
        simpler ShadowHand / AllegroHand tasks (Sec. 4.4).
    num_policies : int
        Number of policies ``M`` sharing the backbone (one ``phi_j`` each).
    mlp_units : sequence of int
        Trunk hidden widths (default Table 2: ``768 x 512 x 256``).
    activation : str
        Activation used by the trunk (``"elu"`` per Appendix B).
    use_lstm, lstm_hidden_size, lstm_num_layers : bool, int, int
        Recurrent head (AllegroKuka uses a single 768-unit LSTM layer).
    per_block_sigma : bool
        When ``True`` (entropy-exploration experiments) each block keeps its own learnable
        sigma vector; otherwise a single vector is shared (Sec. 4.5 / Appendix B.3 note).
    learnable_sigma : bool
        If ``False`` the sigma is a fixed (non-optimised) vector.
    init_sigma : float
        Initial value of the (input independent) sigma vector.
    use_tanh : bool
        Squash the mean through ``action_scale * tanh(.)`` (bounded actions).  The pre-tanh
        mean is exposed as ``pre_tanh_mean`` for the action-bounds regulariser.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        phi_dim: int = 0,
        num_policies: int = 1,
        mlp_units: Sequence[int] = (768, 512, 256),
        activation: str = "elu",
        use_lstm: bool = False,
        lstm_hidden_size: int = 768,
        lstm_num_layers: int = 1,
        learnable_sigma: bool = True,
        init_sigma: float = 1.0,
        per_block_sigma: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        use_tanh: bool = True,
        action_scale: float = 1.0,
        phi_init_scale: float = 1.0,
        learnable_phi: bool = True,
        phi_init: str = "normal",
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if config is not None:
            obs_dim = int(getattr(config, "obs_dim", obs_dim))
            action_dim = int(getattr(config, "action_dim", action_dim))
            phi_dim = int(getattr(config, "phi_dim", phi_dim))
            num_policies = int(getattr(config, "num_policies", num_policies))
            units = tuple(getattr(config, "actor_mlp_units", mlp_units))
            mlp_units = units
            activation = str(getattr(config, "actor_activation", activation))
            use_lstm = bool(getattr(config, "use_lstm", use_lstm))
            lstm_hidden_size = int(getattr(config, "lstm_hidden_size", lstm_hidden_size))
            lstm_num_layers = int(getattr(config, "lstm_num_layers", lstm_num_layers))
            per_block_sigma = bool(
                getattr(config, "per_block_sigma", per_block_sigma)
            )
            action_scale = float(getattr(config, "action_scale", action_scale))
            learnable_phi = bool(getattr(config, "learnable_phi", learnable_phi))
            if bool(getattr(config, "random_phi", False)):
                learnable_phi = False

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.phi_dim = int(phi_dim)
        self.num_policies = max(1, int(num_policies))
        self.use_tanh = bool(use_tanh)
        self.action_scale = float(action_scale)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # ---- shared backbone B_theta conditioned on phi_j -----------------------------
        backbone_input_dim = self.obs_dim + self.phi_dim
        self.backbone = make_backbone(
            backbone_input_dim,
            mlp_units=tuple(int(u) for u in mlp_units),
            activation=activation,
            use_lstm=bool(use_lstm),
            lstm_hidden_size=int(lstm_hidden_size),
            lstm_num_layers=int(lstm_num_layers),
            output_dim=None,
        )
        trunk_out = int(getattr(self.backbone, "out_features", mlp_units[-1]))
        self.trunk_out_features = trunk_out
        self.is_recurrent = bool(getattr(self.backbone, "is_recurrent", use_lstm))

        # ---- Gaussian mean head (linear on the shared trunk output) --------------------
        self.mu = nn.Linear(trunk_out, self.action_dim)
        # learnable, input-independent sigma vector (Appendix B.1)
        self.sigma_module = LearnableSigma(
            self.action_dim,
            init_sigma=float(init_sigma),
            num_policies=self.num_policies if per_block_sigma else 1,
            learnable=bool(learnable_sigma),
            log_std_min=float(log_std_min),
            log_std_max=float(log_std_max),
        )
        self.per_block_sigma = bool(per_block_sigma)

        # ---- per-policy latent parameters phi_j (Sec. 4.4) -----------------------------
        if self.phi_dim > 0:
            if str(phi_init).lower() == "zeros":
                init = torch.zeros(self.num_policies, self.phi_dim)
            else:  # N(0, I) as in the reproduction plan
                init = torch.randn(self.num_policies, self.phi_dim) * float(phi_init_scale)
            self.phi_parameters = nn.Parameter(init)
            self.phi_parameters.requires_grad_(bool(learnable_phi))
        else:
            self.register_parameter("phi_parameters", None)
        self.learnable_phi = bool(learnable_phi) and self.phi_dim > 0

        # orthogonal init for the mean head (small gain keeps the initial policy mild)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)

    # ------------------------------------------------------------------ phi accessors
    def phi(self, policy_index: Optional[int] = None) -> Optional[torch.Tensor]:
        """Return the latent ``phi_j`` of policy ``policy_index`` (0-based).

        The returned tensor is a *view* of the underlying parameter, so gradients computed
        from policy ``j``'s objective flow into ``phi_j`` only.
        """

        if self.phi_parameters is None:
            return None
        if policy_index is None:
            return self.phi_parameters
        return self.phi_parameters[int(policy_index) % self.num_policies]

    def phi_for(self, policy_index: Optional[int] = None) -> Optional[torch.Tensor]:
        """Alias of :meth:`phi` (kept for readability at call sites)."""

        return self.phi(policy_index)

    def all_phi(self) -> Optional[torch.Tensor]:
        """Return the full ``[num_policies, phi_dim]`` parameter matrix."""

        return self.phi_parameters

    @torch.no_grad()
    def reset_phi(self, policy_index: Optional[int] = None, scale: float = 1.0) -> None:
        """Re-initialise ``phi_j ~ N(0, I)`` (used by DexPBT-style mutations)."""

        if self.phi_parameters is None:
            return
        if policy_index is None:
            self.phi_parameters.normal_(mean=0.0, std=scale)
        else:
            idx = int(policy_index) % self.num_policies
            self.phi_parameters[idx].normal_(mean=0.0, std=scale)

    def actor_parameters(self) -> List[nn.Parameter]:
        """Parameters of the shared actor backbone ``theta`` (excluding ``phi_j``)."""

        names = {"phi_parameters"}
        return [p for n, p in self.named_parameters() if n not in names and p.requires_grad]

    def phi_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        if self.phi_parameters is None:
            return []
        return [("phi_parameters", self.phi_parameters)]

    # ------------------------------------------------------------------------ forward
    def _backbone_features(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.float()
        if self.phi_dim > 0:
            phi_b = _as_phi_batch(phi, obs.shape[0], obs.device, obs.dtype)
            if phi_b is None:
                phi_b = torch.zeros(
                    obs.shape[0], self.phi_dim, device=obs.device, dtype=obs.dtype
                )
            x = torch.cat([obs, phi_b], dim=-1)
        else:
            x = obs

        if self.is_recurrent:
            x, hidden_state = self.backbone(x, hidden_state, masks)
        else:
            x = self.backbone(x)
        return x, hidden_state

    def distribution(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
    ) -> Tuple[Normal, torch.Tensor, torch.Tensor, Optional[Any]]:
        """Return ``(dist_over_raw_actions, pre_tanh_mean, sigma, hidden_state)``."""

        features, hidden_state = self._backbone_features(obs, phi, hidden_state, masks)
        pre_tanh_mean = self.mu(features)
        sigma = self.sigma_module.sigma_for(policy_index)
        sigma = sigma.to(pre_tanh_mean.device, pre_tanh_mean.dtype)
        dist = Normal(pre_tanh_mean, sigma)
        return dist, pre_tanh_mean, sigma, hidden_state

    def forward(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample (or take the mode of) ``pi_theta(. | o, phi_j)``.

        Returns a dict with ``actions``, ``logprobs``, ``mu`` (bounded mean), ``pre_tanh_mean``,
        ``sigma``, ``entropy`` and ``hidden_state``.
        """

        dist, pre_tanh_mean, sigma, hidden_state = self.distribution(
            obs, phi, hidden_state, masks, policy_index
        )
        if deterministic:
            raw_action = pre_tanh_mean
        else:
            raw_action = dist.rsample()
        log_prob = tanh_log_prob(dist, raw_action, self.action_scale) if self.use_tanh \
            else dist.log_prob(raw_action).sum(-1)
        if self.use_tanh:
            action = squash_action(raw_action, self.action_scale)
        else:
            action = raw_action
        entropy = dist.entropy().sum(-1)
        return {
            "actions": action,
            "logprobs": log_prob,
            "log_prob": log_prob,
            "mu": action,
            "pre_tanh_mean": pre_tanh_mean,
            "raw_actions": raw_action,
            "sigma": sigma,
            "entropy": entropy,
            "hidden_state": hidden_state,
        }

    # ``act`` is the duck-typed API used by sapg.algorithms.rollout
    def act(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(
            obs,
            phi=phi,
            hidden_state=hidden_state,
            masks=masks,
            policy_index=policy_index,
            deterministic=deterministic,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Re-evaluate ``log pi_theta(a | o, phi_j)`` and the entropy of the current policy."""

        dist, pre_tanh_mean, sigma, hidden_state = self.distribution(
            obs, phi, hidden_state, masks, policy_index
        )
        if self.use_tanh and self.action_scale != 1.0:
            raw_action = torch.atanh(
                torch.clamp(actions / self.action_scale, -1.0 + 1e-6, 1.0 - 1e-6)
            )
        else:
            raw_action = actions
        if self.use_tanh:
            log_prob = tanh_log_prob(dist, raw_action, self.action_scale)
        else:
            log_prob = dist.log_prob(raw_action).sum(-1)
        return {
            "logprobs": log_prob,
            "log_prob": log_prob,
            "entropy": dist.entropy().sum(-1),
            "mu": squash_action(pre_tanh_mean, self.action_scale) if self.use_tanh
            else pre_tanh_mean,
            "pre_tanh_mean": pre_tanh_mean,
            "sigma": sigma,
            "hidden_state": hidden_state,
        }

    # ------------------------------------------------------------------ init / hidden
    def init_hidden(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Optional[Any]:
        if not self.is_recurrent:
            return None
        device = device if device is not None else next(self.parameters()).device
        layer = getattr(self.backbone, "recurrent", None)
        if layer is None:
            return None
        return layer.init_hidden(batch_size, device)

    @staticmethod
    def mask_hidden(hidden_state: Optional[Any], masks: Optional[torch.Tensor]) -> Optional[Any]:
        """Zero the recurrent state of environments that just finished (auto-reset)."""

        if hidden_state is None or masks is None:
            return hidden_state
        return type(hidden_state).__mro__ and _mask_hidden_impl(hidden_state, masks)


def _mask_hidden_impl(hidden_state: Any, masks: torch.Tensor) -> Any:
    if isinstance(hidden_state, (tuple, list)):
        return tuple(_mask_hidden_impl(h, masks) for h in hidden_state)
    mask = masks.to(hidden_state.device, hidden_state.dtype)
    while mask.dim() < hidden_state.dim():
        mask = mask.unsqueeze(-1)
    return hidden_state * mask


# --------------------------------------------------------------------------------------
# Combined actor-critic policy
# --------------------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Shared-backbone actor + critic conditioned on per-policy latents ``phi_j``.

    This is the object passed to :mod:`sapg.algorithms.rollout` and
    :mod:`sapg.algorithms.ppo`: ``act`` returns actions/log-probs/values in one call, while
    ``actor_parameters()`` / ``critic_parameters()`` let the trainer keep the two optimisers
    (and their learning rates) separate.  ``phi_parameters`` are exposed separately so the
    SAPG loop can update every ``phi_j`` from that policy's own objective only (Sec. 4.4).
    """

    def __init__(
        self,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        phi_dim: int = 0,
        num_policies: int = 1,
        actor: Optional[Actor] = None,
        critic: Optional[nn.Module] = None,
        config: Any = None,
        **actor_kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            obs_dim = obs_dim if obs_dim is not None else getattr(config, "obs_dim", None)
            action_dim = (
                action_dim if action_dim is not None else getattr(config, "action_dim", None)
            )
            phi_dim = int(getattr(config, "phi_dim", phi_dim))
            num_policies = int(getattr(config, "num_policies", num_policies))

        if actor is None:
            actor = Actor(
                obs_dim=int(obs_dim),
                action_dim=int(action_dim),
                phi_dim=int(phi_dim),
                num_policies=int(num_policies),
                config=config,
                **actor_kwargs,
            )
        self.actor = actor

        if critic is None:
            from .critic import Critic

            critic = Critic(
                obs_dim=int(obs_dim),
                phi_dim=int(phi_dim),
                num_policies=int(num_policies),
                config=config,
                **{
                    k: v
                    for k, v in actor_kwargs.items()
                    if k
                    in (
                        "critic_mlp_units",
                        "activation",
                        "use_lstm",
                        "lstm_hidden_size",
                        "lstm_num_layers",
                        "learnable_phi",
                        "phi_init",
                        "phi_init_scale",
                    )
                },
            )
        self.critic = critic

        self.obs_dim = int(getattr(self.actor, "obs_dim", obs_dim or 0))
        self.action_dim = int(getattr(self.actor, "action_dim", action_dim or 0))
        self.phi_dim = int(getattr(self.actor, "phi_dim", phi_dim))
        self.num_policies = int(getattr(self.actor, "num_policies", num_policies))
        self.is_recurrent = bool(getattr(self.actor, "is_recurrent", False))
        self.recurrent = self.is_recurrent

    # ------------------------------------------------------------------ phi bridging
    def phi(self, policy_index: Optional[int] = None) -> Optional[torch.Tensor]:
        if policy_index is None:
            return self.actor.phi(None)
        # share phi_j between the actor and the critic backbones (Sec. 4.4)
        phi = self.actor.phi(policy_index)
        critic_phi = getattr(self.critic, "phi", None)
        if critic_phi is not None and getattr(self.critic, "phi_parameters", None) is not None:
            critic_phi(policy_index)
        return phi

    def phi_for(self, policy_index: Optional[int] = None) -> Optional[torch.Tensor]:
        return self.phi(policy_index)

    def all_phi(self) -> Optional[torch.Tensor]:
        return self.actor.all_phi()

    @property
    def phi_parameters(self) -> Optional[nn.Parameter]:
        return self.actor.phi_parameters

    @property
    def phi_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        params: List[Tuple[str, nn.Parameter]] = []
        if self.actor.phi_parameters is not None:
            params.append(("actor.phi_parameters", self.actor.phi_parameters))
        critic_phi = getattr(self.critic, "phi_parameters", None)
        if critic_phi is not None:
            params.append(("critic.phi_parameters", critic_phi))
        return params

    # ------------------------------------------------------------------ policy API
    def act(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
        deterministic: bool = False,
        value_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """One environment step: returns ``actions``, ``logprobs``, ``values`` (+ extras)."""

        if policy_index is not None and phi is None:
            phi = self.phi(policy_index)

        if hidden_state is not None:
            actor_hidden, critic_hidden = self._split_hidden(hidden_state)
        else:
            actor_hidden = critic_hidden = None

        out: Dict[str, torch.Tensor] = {}
        if not value_only:
            actor_out = self.actor(
                obs,
                phi=phi,
                hidden_state=actor_hidden,
                masks=masks,
                policy_index=policy_index,
                deterministic=deterministic,
            )
            out.update(actor_out)
        else:
            actor_out = {"hidden_state": actor_hidden}

        critic_out = self.critic(
            obs,
            phi=phi,
            hidden_state=critic_hidden,
            masks=masks,
            policy_index=policy_index,
        )
        values = critic_out["values"] if isinstance(critic_out, dict) else critic_out
        out["values"] = values
        out["value"] = values
        out["hidden_state"] = (actor_out.get("hidden_state"), critic_out.get("hidden_state")
                               if isinstance(critic_out, dict) else None)
        return out

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.act(*args, **kwargs)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Recompute ``log pi_theta(a|o,phi_j)``, the entropy, and ``V_psi(o, phi_j)``."""

        if policy_index is not None and phi is None:
            phi = self.phi(policy_index)
        actor_hidden, critic_hidden = self._split_hidden(hidden_state)

        out: Dict[str, torch.Tensor] = {}
        if actions is not None:
            out.update(
                self.actor.evaluate_actions(
                    obs,
                    actions,
                    phi=phi,
                    hidden_state=actor_hidden,
                    masks=masks,
                    policy_index=policy_index,
                )
            )
        critic_out = self.critic(
            obs, phi=phi, hidden_state=critic_hidden, masks=masks,
            policy_index=policy_index,
        )
        values = critic_out["values"] if isinstance(critic_out, dict) else critic_out
        out["values"] = values
        out["value"] = values
        out["hidden_state"] = (out.get("hidden_state"), critic_out.get("hidden_state")
                               if isinstance(critic_out, dict) else None)
        return out

    def get_value(
        self,
        obs: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        hidden_state: Optional[Any] = None,
        masks: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
    ) -> torch.Tensor:
        critic_out = self.critic(
            obs, phi=phi, hidden_state=hidden_state, masks=masks, policy_index=policy_index
        )
        return critic_out["values"] if isinstance(critic_out, dict) else critic_out

    # ------------------------------------------------------------------ helpers
    def _split_hidden(self, hidden_state: Optional[Any]) -> Tuple[Optional[Any], Optional[Any]]:
        """Split a combined ``(actor_hidden, critic_hidden)`` state.

        Accepts ``None``, a 2-tuple produced by :meth:`act`, or a bare recurrent state that is
        reused for both networks.
        """

        if hidden_state is None:
            return None, None
        if (
            isinstance(hidden_state, (tuple, list))
            and len(hidden_state) == 2
            and not torch.is_tensor(hidden_state[0])
        ):
            return hidden_state[0], hidden_state[1]
        if isinstance(hidden_state, (tuple, list)) and len(hidden_state) == 2:
            # bare (h, c) LSTM state -> share it between actor and critic
            return hidden_state, hidden_state
        return hidden_state, hidden_state

    def init_hidden(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[Optional[Any], Optional[Any]]:
        device = device if device is not None else next(self.parameters()).device
        actor_hidden = self.actor.init_hidden(batch_size, device)
        critic_hidden = None
        critic_init = getattr(self.critic, "init_hidden", None)
        if critic_init is not None:
            critic_hidden = critic_init(batch_size, device)
        return actor_hidden, critic_hidden

    def actor_parameters(self) -> List[nn.Parameter]:
        """Shared actor parameters ``theta`` -- each policy contributes a gradient."""

        return list(self.actor.actor_parameters())

    def critic_parameters(self) -> List[nn.Parameter]:
        """Shared critic parameters ``psi``."""

        getter = getattr(self.critic, "critic_parameters", None)
        if getter is not None:
            return list(getter())
        return [p for p in self.critic.parameters() if p.requires_grad]

    def latent_parameters(self) -> List[nn.Parameter]:
        """Latent conditioning parameters ``phi_1 .. phi_M`` (Sec. 4.4)."""

        params: List[nn.Parameter] = []
        if self.actor.phi_parameters is not None:
            params.append(self.actor.phi_parameters)
        critic_phi = getattr(self.critic, "phi_parameters", None)
        if critic_phi is not None:
            params.append(critic_phi)
        return params

    def policy_optimizer_groups(self) -> List[Dict[str, Any]]:
        """Parameter groups for policy optimisation (backbone + all ``phi_j``)."""

        groups: List[Dict[str, Any]] = [{"params": self.actor_parameters(), "name": "actor"}]
        groups.extend(
            {"params": [p], "name": name} for name, p in self.phi_named_parameters
        )
        return groups

    def critic_optimizer_groups(self) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = [{"params": self.critic_parameters(), "name": "critic"}]
        return groups


# convenience alias (used across the code base as "the policy")
Policy = ActorCritic
