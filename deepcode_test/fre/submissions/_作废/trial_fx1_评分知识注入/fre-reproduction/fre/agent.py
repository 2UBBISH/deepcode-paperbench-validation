"""FRE-conditioned offline RL agent.

This module ties together the frozen Functional Reward Encoding (FRE) VAE,
the reward-function prior, and Implicit Q-Learning (IQL) networks. It is used
in phase 2 of the strided training procedure: the VAE encoder/decoder are
trained first and then frozen, after which this agent samples reward
functions, encodes them into latent vectors ``z``, and trains IQL on the
resulting task-conditioned transitions.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .fre_vae import FREVAE
from .iql import IQLNetworks, SquashedGaussianPolicy
from .reward_prior import RewardFunction, RewardPrior, make_default_reward_prior


def _to_tensor(
    x: Union[np.ndarray, torch.Tensor, Sequence[float]],
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert array-like data to a torch tensor on the requested device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


class FREAgent:
    """FRE-conditioned IQL agent.

    Parameters
    ----------
    state_dim:
        Dimensionality of the environment state space.
    action_dim:
        Dimensionality of the action space.
    latent_dim:
        Dimensionality of the reward-function latent code ``z``.
    vae:
        Optional pre-trained :class:`FREVAE`. If ``None``, a new VAE is
        constructed using the remaining VAE hyperparameters.
    reward_prior:
        Optional :class:`RewardPrior`. If ``None``, the default uniform
        mixture over goal/linear/MLP reward families is constructed.
    state_pool:
        Optional pool of raw states used for sampling encoder-context states
        when ``reward_prior`` or ``vae`` is not provided. May be a NumPy array
        or torch tensor of shape ``(N, state_dim)``.
    dataset:
        Optional offline dataset with ``sample_states(n)`` and ``sample`` or
        ``sample_batch`` methods. Used by :meth:`train_on_dataset`.
    encoder_states:
        Number of state-reward context pairs used by the FRE encoder.
    freeze_vae:
        Whether to freeze the FRE VAE. Defaults to ``True``; phase 1 training
        should use the raw :class:`FREVAE` training loop directly.
    q_hidden / v_hidden / policy_hidden:
        Hidden-layer widths for IQL Q, V, and policy networks.
    gamma, expectile, awr_temperature, target_tau, advantage_clip:
        IQL hyperparameters.
    lr:
        Adam learning rate for all IQL networks.
    device:
        Torch device to use.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        vae: Optional[FREVAE] = None,
        reward_prior: Optional[RewardPrior] = None,
        state_pool: Optional[Union[np.ndarray, torch.Tensor]] = None,
        dataset: Any = None,
        encoder_states: int = 32,
        freeze_vae: bool = True,
        vae_kwargs: Optional[Dict[str, Any]] = None,
        q_hidden: Sequence[int] = (256, 256),
        v_hidden: Sequence[int] = (256, 256),
        policy_hidden: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        expectile: float = 0.9,
        awr_temperature: float = 3.0,
        target_tau: float = 0.005,
        advantage_clip: Tuple[float, float] = (-5.0, 2.0),
        lr: float = 3e-4,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder_states = encoder_states
        self.gamma = gamma
        self.expectile = expectile
        self.awr_temperature = awr_temperature
        self.target_tau = target_tau
        self.advantage_clip = tuple(advantage_clip)

        # Convert the state pool once so that both reward-prior construction
        # and encoder-context sampling share the same source.
        if state_pool is not None:
            state_pool = _to_tensor(state_pool, self.device, torch.float32)
        self.state_pool = state_pool
        self.dataset = dataset

        # Build or reuse the FRE VAE.
        vae_kwargs = dict(vae_kwargs or {})
        if vae is None:
            vae = FREVAE(
                state_dim=state_dim,
                latent_dim=latent_dim,
                device=self.device,
                **vae_kwargs,
            )
        self.vae = vae
        self.vae.to(self.device)
        if freeze_vae:
            self.freeze_vae()

        # Build or reuse the reward-function prior.
        if reward_prior is None:
            reward_prior = make_default_reward_prior(
                state_dim=state_dim,
                state_pool=self.state_pool,
                device=self.device,
            )
        self.reward_prior = reward_prior

        # IQL networks and optimizers.
        self.networks = IQLNetworks(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dims=tuple(q_hidden),
            gamma=gamma,
            expectile=expectile,
            awr_temperature=awr_temperature,
            target_tau=target_tau,
            advantage_clip=self.advantage_clip,
        )
        self.networks.to(self.device)

        self.q_optimizer = torch.optim.Adam(
            self.networks.q1.parameters(), lr=lr
        )
        self.q_optimizer.add_param_group(
            {"params": self.networks.q2.parameters()}
        )
        self.v_optimizer = torch.optim.Adam(
            self.networks.v.parameters(), lr=lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.networks.policy.parameters(), lr=lr
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def freeze_vae(self) -> None:
        """Freeze all FRE VAE parameters and switch it to eval mode."""
        for param in self.vae.parameters():
            param.requires_grad_(False)
        self.vae.eval()

    def to(self, device: Union[str, torch.device]) -> "FREAgent":
        self.device = torch.device(device)
        self.vae.to(self.device)
        self.networks.to(self.device)
        if self.state_pool is not None:
            self.state_pool = self.state_pool.to(self.device)
        return self

    # ------------------------------------------------------------------
    # State and reward sampling
    # ------------------------------------------------------------------
    def _sample_states(self, num_states: int) -> torch.Tensor:
        """Sample encoder-context states from the offline state pool."""
        if self.dataset is not None and hasattr(self.dataset, "sample_states"):
            states = self.dataset.sample_states(num_states)
            return _to_tensor(states, self.device, torch.float32)

        if self.state_pool is not None:
            n = self.state_pool.shape[0]
            idx = torch.randint(0, n, (num_states,), device=self.device)
            return self.state_pool[idx]

        # No pool: fall back to uniform states in a normalized state box.
        return torch.empty(
            (num_states, self.state_dim), device=self.device
        ).uniform_(-1.0, 1.0)

    def _sample_dataset_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Sample ``(states, actions, next_states, dones)`` from the dataset.

        Accepts datasets implementing either ``sample`` or ``sample_batch``.
        """
        if self.dataset is None:
            raise ValueError("An offline dataset must be provided for dataset sampling.")

        if hasattr(self.dataset, "sample_batch"):
            batch = self.dataset.sample_batch(batch_size)
        elif hasattr(self.dataset, "sample"):
            batch = self.dataset.sample(batch_size)
        else:
            raise AttributeError(
                "Dataset must implement 'sample' or 'sample_batch'."
            )

        # Common dataset batch formats:
        #   (states, actions, rewards, next_states, dones)
        #   (states, actions, next_states, dones)
        if isinstance(batch, dict):
            states = batch["states"]
            actions = batch["actions"]
            next_states = batch["next_states"]
            dones = batch.get("dones", batch.get("terminals"))
        else:
            states, actions = batch[0], batch[1]
            next_states = batch[2] if len(batch) > 2 else batch[1]
            dones = batch[3] if len(batch) > 3 else None
            # If the dataset includes rewards at index 2, skip them.
            if (
                isinstance(next_states, (np.ndarray, torch.Tensor))
                and next_states.ndim == 1
                and len(batch) > 3
            ):
                # Ambiguous layout; prefer next_states at index 2 when it is
                # a vector whose shape matches states.
                if next_states.shape[0] == states.shape[0] and len(batch) > 2:
                    # Actually index 2 is already assigned to next_states.
                    pass

        return (
            _to_tensor(states, self.device, torch.float32),
            _to_tensor(actions, self.device, torch.float32),
            _to_tensor(next_states, self.device, torch.float32),
            _to_tensor(dones, self.device, torch.float32)
            if dones is not None
            else None,
        )

    def sample_reward_functions(self, batch_size: int) -> List[RewardFunction]:
        """Sample a batch of random reward functions from the prior."""
        return list(self.reward_prior.sample_reward_fns(batch_size))

    def _evaluate_reward_functions(
        self, reward_fns: Sequence[RewardFunction], states: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate a distinct reward function for each row of ``states``."""
        rewards: List[torch.Tensor] = []
        for i, reward_fn in enumerate(reward_fns):
            # Evaluate one transition's state under its corresponding reward
            # function. Unsqueeze to keep RewardFunction.call batch-agnostic.
            value = reward_fn(states[i].unsqueeze(0)).reshape(-1)[0]
            rewards.append(value)
        return torch.stack(rewards).to(self.device)

    # ------------------------------------------------------------------
    # Latent encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_reward_functions(
        self,
        reward_fns: Sequence[RewardFunction],
        encoder_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of reward functions into latent codes ``z``.

        Each reward function is encoded from ``encoder_states`` state-reward
        context examples. The VAE is assumed to be frozen (the default agent
        configuration), so this method is wrapped in ``torch.no_grad``.
        """
        batch_size = len(reward_fns)
        if encoder_states is None:
            flat_states = self._sample_states(
                batch_size * self.encoder_states
            )
            encoder_states = flat_states.view(
                batch_size, self.encoder_states, self.state_dim
            )
        else:
            encoder_states = _to_tensor(
                encoder_states, self.device, torch.float32
            )

        zs: List[torch.Tensor] = []
        for i, reward_fn in enumerate(reward_fns):
            context = encoder_states[i]
            # ``encode_reward_fn`` returns (mu, logvar, z); we only need z.
            out = self.vae.encode_reward_fn(reward_fn, context)
            if isinstance(out, (tuple, list)):
                z = out[-1]
            elif isinstance(out, dict):
                z = out["z"]
            else:
                z = out
            zs.append(z.reshape(self.latent_dim))
        return torch.stack(zs).to(self.device)

    def encode_task(
        self,
        reward_fn: RewardFunction,
        num_examples: int = 32,
        states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a downstream reward function from exactly ``num_examples``
        state-reward examples (default 32, as used in zero-shot evaluation).
        """
        if states is None:
            states = self._sample_states(num_examples)
        else:
            states = _to_tensor(states, self.device, torch.float32)
        with torch.no_grad():
            out = self.vae.encode_reward_fn(reward_fn, states)
            if isinstance(out, (tuple, list)):
                z = out[-1]
            elif isinstance(out, dict):
                z = out["z"]
            else:
                z = out
        return z.reshape(self.latent_dim)

    # ------------------------------------------------------------------
    # IQL training
    # ------------------------------------------------------------------
    def train_step(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        next_states: Union[np.ndarray, torch.Tensor],
        dones: Optional[Union[np.ndarray, torch.Tensor]] = None,
        reward_fns: Optional[Sequence[RewardFunction]] = None,
        encoder_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Perform one IQL update.

        Parameters
        ----------
        states, actions, next_states, dones:
            A batch of offline transitions. Rewards are *not* taken from the
            dataset; they are produced by the sampled reward functions.
        reward_fns:
            Optional pre-sampled reward functions. If omitted, the agent
            samples one reward function per transition from its prior.
        encoder_states:
            Optional ``(B, K, state_dim)`` context states for encoding. If
            omitted, the agent samples ``B * K`` states from the state pool.
        """
        states = _to_tensor(states, self.device, torch.float32)
        actions = _to_tensor(actions, self.device, torch.float32)
        next_states = _to_tensor(next_states, self.device, torch.float32)
        if dones is not None:
            dones = _to_tensor(dones, self.device, torch.float32)

        batch_size = states.shape[0]
        if reward_fns is None:
            reward_fns = self.sample_reward_functions(batch_size)

        with torch.no_grad():
            z = self.encode_reward_functions(reward_fns, encoder_states)
            rewards = self._evaluate_reward_functions(reward_fns, states)

        losses = self.networks.compute_losses(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            z=z,
            dones=dones,
        )

        # Normalize the returned-loss format to a simple dictionary.
        if isinstance(losses, dict):
            loss_dict = losses
        elif isinstance(losses, (tuple, list)):
            loss_dict = {
                "q_loss": losses[0],
                "v_loss": losses[1],
                "policy_loss": losses[2],
            }
        else:
            raise TypeError("IQL compute_losses returned an unsupported type.")

        # Optimize all networks from the shared computation graph.
        self.q_optimizer.zero_grad()
        self.v_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        total_loss = (
            loss_dict.get("q_loss", 0.0)
            + loss_dict.get("v_loss", 0.0)
            + loss_dict.get("policy_loss", 0.0)
        )
        if torch.is_tensor(total_loss):
            total_loss.backward()

        self.q_optimizer.step()
        self.v_optimizer.step()
        self.policy_optimizer.step()
        self.networks.update_target_networks()

        metrics: Dict[str, float] = {}
        for key, value in loss_dict.items():
            metrics[key] = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
        metrics["reward_mean"] = float(rewards.mean().detach().cpu().item())
        metrics["z_mean_abs"] = float(z.abs().mean().detach().cpu().item())
        return metrics

    def train_on_dataset(
        self,
        batch_size: int,
        reward_fns: Optional[Sequence[RewardFunction]] = None,
        encoder_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Sample a batch from the dataset and run :meth:`train_step`."""
        states, actions, next_states, dones = self._sample_dataset_batch(batch_size)
        return self.train_step(
            states=states,
            actions=actions,
            next_states=next_states,
            dones=dones,
            reward_fns=reward_fns,
            encoder_states=encoder_states,
        )

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def get_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        z: Optional[Union[np.ndarray, torch.Tensor]] = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Return an action for ``state`` conditioned on latent code ``z``."""
        state_t = _to_tensor(state, self.device, torch.float32)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        if z is not None:
            z_t = _to_tensor(z, self.device, torch.float32)
            if z_t.ndim == 1:
                z_t = z_t.unsqueeze(0)
        else:
            z_t = None
        with torch.no_grad():
            action = self.networks.policy.get_action(
                state_t, z_t, deterministic=deterministic
            )
        return action.squeeze(0).cpu().numpy() if action.ndim == 2 else action.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "vae": self.vae.state_dict(),
            "networks": self.networks.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.vae.load_state_dict(state_dict["vae"])
        self.networks.load_state_dict(state_dict["networks"])
        if "q_optimizer" in state_dict:
            self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        if "v_optimizer" in state_dict:
            self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
        if "policy_optimizer" in state_dict:
            self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
