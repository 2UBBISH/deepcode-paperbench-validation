"""Rollout collection for SAPG.

This module implements per-policy data collection (Section 4, 4.6 of the paper).
The N parallel environments are split into M blocks of N/M environments each.
Block ``j`` is rolled out by policy ``j`` (the shared actor backbone ``B_theta``
conditioned on the per-policy latent ``phi_j``), producing a buffer ``D_j``.

Each buffer stores, for every environment instance in the block, a horizon of
``H`` transitions.  The layout is time-major ``[T, B, ...]`` where ``T`` is the
horizon and ``B`` is the number of environments per block.  This matches the
convention used by :mod:`sapg.returns` and :mod:`sapg.losses`.

The module is deliberately simulator-agnostic: it only requires an environment
object exposing the standard vectorized Gym-like API::

    obs = env.reset()                       # -> [N, obs_dim]
    obs, rew, done, info = env.step(actions)  # -> [N, ...]

The :class:`EnvBlockSplitter` helper assigns contiguous environment indices to
blocks and provides per-block views of the global observation/action tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Environment block splitting
# ---------------------------------------------------------------------------
class EnvBlockSplitter:
    """Splits ``num_envs`` parallel environments into ``num_blocks`` blocks.

    Block ``j`` (0-indexed) owns the contiguous slice
    ``[j * envs_per_block, (j + 1) * envs_per_block)`` of the global environment
    index space.  This mirrors the paper's description (Sec. 4.6) where
    ``N = 24576`` environments are split into ``M = 6`` blocks of
    ``N / M = 4096`` environments each.
    """

    def __init__(self, num_envs: int, num_blocks: int):
        if num_envs % num_blocks != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_blocks "
                f"({num_blocks})"
            )
        self.num_envs = int(num_envs)
        self.num_blocks = int(num_blocks)
        self.envs_per_block = self.num_envs // self.num_blocks

    def block_indices(self, block_idx: int) -> torch.Tensor:
        """Return the global environment indices owned by ``block_idx``."""
        if not 0 <= block_idx < self.num_blocks:
            raise IndexError(
                f"block_idx {block_idx} out of range [0, {self.num_blocks})"
            )
        start = block_idx * self.envs_per_block
        return torch.arange(start, start + self.envs_per_block, dtype=torch.long)

    def split(self, tensor: torch.Tensor, dim: int = 0) -> List[torch.Tensor]:
        """Split a global tensor into ``num_blocks`` contiguous chunks."""
        return list(torch.chunk(tensor, self.num_blocks, dim=dim))

    def gather_block(self, tensor: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Select the rows of ``tensor`` belonging to ``block_idx`` (dim 0)."""
        start = block_idx * self.envs_per_block
        return tensor[start : start + self.envs_per_block]

    def scatter_block(
        self, global_tensor: torch.Tensor, block_idx: int, block_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Write ``block_tensor`` into the ``block_idx`` slice of ``global_tensor``."""
        start = block_idx * self.envs_per_block
        global_tensor[start : start + self.envs_per_block] = block_tensor
        return global_tensor


# ---------------------------------------------------------------------------
# Per-policy rollout buffer
# ---------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    """Storage for a single policy's rollout of horizon ``T`` over ``B`` envs.

    All tensors are time-major with shape ``[T, B, ...]`` unless noted.
    """

    horizon: int
    num_envs: int
    obs_dim: int
    action_dim: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Populated by :meth:`add` / :meth:`finalize`.
    obs: torch.Tensor = field(init=False)
    actions: torch.Tensor = field(init=False)
    log_probs: torch.Tensor = field(init=False)
    rewards: torch.Tensor = field(init=False)
    dones: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    # Optional LSTM hidden states (for recurrent policies).
    lstm_hidden: Optional[torch.Tensor] = field(init=False, default=None)
    lstm_cell: Optional[torch.Tensor] = field(init=False, default=None)

    # Computed after collection.
    advantages: Optional[torch.Tensor] = field(init=False, default=None)
    value_targets: Optional[torch.Tensor] = field(init=False, default=None)
    valid_mask: Optional[torch.Tensor] = field(init=False, default=None)

    _t: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        T, B = self.horizon, self.num_envs
        dev = self.device
        self.obs = torch.zeros(T, B, self.obs_dim, device=dev)
        self.actions = torch.zeros(T, B, self.action_dim, device=dev)
        self.log_probs = torch.zeros(T, B, device=dev)
        self.rewards = torch.zeros(T, B, device=dev)
        self.dones = torch.zeros(T, B, device=dev)
        self.values = torch.zeros(T, B, device=dev)
        self._t = 0

    # -- collection ---------------------------------------------------------
    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """Append one timestep of transitions for all ``B`` environments."""
        t = self._t
        if t >= self.horizon:
            raise RuntimeError(
                f"RolloutBuffer already full (horizon={self.horizon})"
            )
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        if lstm_state is not None:
            h, c = lstm_state
            if self.lstm_hidden is None:
                self.lstm_hidden = torch.zeros(
                    self.horizon, *h.shape, device=self.device, dtype=h.dtype
                )
                self.lstm_cell = torch.zeros(
                    self.horizon, *c.shape, device=self.device, dtype=c.dtype
                )
            self.lstm_hidden[t] = h
            self.lstm_cell[t] = c
        self._t += 1

    @property
    def is_full(self) -> bool:
        return self._t >= self.horizon

    def reset(self) -> None:
        """Zero the buffer for reuse (keeps allocated storage)."""
        self.obs.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()
        if self.lstm_hidden is not None:
            self.lstm_hidden.zero_()
            self.lstm_cell.zero_()
        self.advantages = None
        self.value_targets = None
        self.valid_mask = None
        self._t = 0

    # -- post-processing ----------------------------------------------------
    def compute_returns(
        self,
        next_value: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        on_policy_target_steps: int = 3,
    ) -> None:
        """Compute GAE advantages and n-step value targets (Eq. 5)."""
        from .returns import compute_advantages_and_targets

        adv, targets, valid = compute_advantages_and_targets(
            rewards=self.rewards,
            values=self.values,
            dones=self.dones,
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
            on_policy_target_steps=on_policy_target_steps,
        )
        self.advantages = adv
        self.value_targets = targets
        self.valid_mask = valid

    # -- accessors ----------------------------------------------------------
    def flat(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flatten a ``[T, B, ...]`` tensor to ``[T*B, ...]``."""
        return tensor.reshape(-1, *tensor.shape[2:])

    def size(self) -> int:
        return self.horizon * self.num_envs

    def __len__(self) -> int:
        return self.size()


# ---------------------------------------------------------------------------
# Rollout driver
# ---------------------------------------------------------------------------
class RolloutCollector:
    """Collects rollouts for all ``M`` policies from their environment blocks.

    Parameters
    ----------
    env:
        Vectorized environment exposing ``reset()`` and ``step(actions)``.
    networks:
        :class:`sapg.networks.SAPGNetworks` container (shared backbones + phi).
    splitter:
        :class:`EnvBlockSplitter` describing the block assignment.
    horizon:
        Number of environment steps collected per update (16 for AllegroKuka,
        8 for ShadowHand / AllegroHand).
    device:
        Torch device for the buffers.
    """

    def __init__(
        self,
        env: Any,
        networks: Any,
        splitter: EnvBlockSplitter,
        horizon: int,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
    ):
        self.env = env
        self.networks = networks
        self.splitter = splitter
        self.horizon = int(horizon)
        self.device = device
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        # One buffer per policy / block.
        self.buffers: List[RolloutBuffer] = [
            RolloutBuffer(
                horizon=self.horizon,
                num_envs=splitter.envs_per_block,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=self.device,
            )
            for _ in range(splitter.num_blocks)
        ]

        # Per-block LSTM hidden state (recurrent policies only).
        self._lstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(splitter.num_blocks)
        ]

        self._global_obs: Optional[torch.Tensor] = None

    # -- helpers ------------------------------------------------------------
    def _reset_lstm_states(self) -> None:
        self._lstm_states = [None for _ in range(self.splitter.num_blocks)]

    def _init_lstm_state(self, block_idx: int, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero LSTM hidden/cell states for a block."""
        actor = self.networks.actor
        num_layers = getattr(actor, "lstm_num_layers", 1)
        hidden = getattr(actor, "lstm_hidden_size", 768)
        h = torch.zeros(num_layers, batch, hidden, device=self.device)
        c = torch.zeros(num_layers, batch, hidden, device=self.device)
        return h, c

    # -- main API -----------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset the environment and all buffers; returns global observations."""
        obs = self.env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        self._global_obs = obs
        for buf in self.buffers:
            buf.reset()
        self._reset_lstm_states()
        return obs

    @torch.no_grad()
    def collect(self) -> List[RolloutBuffer]:
        """Collect ``horizon`` steps for every block.

        Returns the list of per-policy buffers ``[D_1, ..., D_M]`` with
        advantages / value targets already computed.
        """
        if self._global_obs is None:
            self.reset()

        for _ in range(self.horizon):
            global_obs = self._global_obs
            global_actions = torch.zeros(
                self.splitter.num_envs, self.action_dim, device=self.device
            )
            global_log_probs = torch.zeros(self.splitter.num_envs, device=self.device)
            global_values = torch.zeros(self.splitter.num_envs, device=self.device)

            for j in range(self.splitter.num_blocks):
                block_obs = self.splitter.gather_block(global_obs, j)
                latent = self.networks.latent(j)

                # Initialise LSTM state lazily on first step of the rollout.
                if self.networks.actor.use_lstm:
                    if self._lstm_states[j] is None:
                        self._lstm_states[j] = self._init_lstm_state(
                            j, block_obs.shape[0]
                        )
                    lstm_state = self._lstm_states[j]
                else:
                    lstm_state = None

                action, log_prob, new_state = self.networks.actor_act(
                    block_obs, latent, lstm_state=lstm_state, deterministic=False
                )
                value, _ = self.networks.critic_value(
                    block_obs, latent, lstm_state=lstm_state
                )

                if self.networks.actor.use_lstm:
                    self._lstm_states[j] = new_state

                start = j * self.splitter.envs_per_block
                end = start + self.splitter.envs_per_block
                global_actions[start:end] = action
                global_log_probs[start:end] = log_prob
                global_values[start:end] = value

            # Step the environment with the aggregated actions.
            step_out = self.env.step(global_actions)
            next_obs, rewards, dones, _info = _unpack_step(step_out)
            next_obs = next_obs.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            # Store per-block transitions.
            for j in range(self.splitter.num_blocks):
                buf = self.buffers[j]
                start = j * self.splitter.envs_per_block
                end = start + self.splitter.envs_per_block
                lstm_state = self._lstm_states[j] if self.networks.actor.use_lstm else None
                buf.add(
                    obs=global_obs[start:end],
                    actions=global_actions[start:end],
                    log_probs=global_log_probs[start:end],
                    rewards=rewards[start:end],
                    dones=dones[start:end],
                    values=global_values[start:end],
                    lstm_state=lstm_state,
                )

            # Reset LSTM states for environments that finished an episode.
            if self.networks.actor.use_lstm:
                for j in range(self.splitter.num_blocks):
                    start = j * self.splitter.envs_per_block
                    end = start + self.splitter.envs_per_block
                    block_dones = dones[start:end]
                    if block_dones.any() and self._lstm_states[j] is not None:
                        h, c = self._lstm_states[j]
                        mask = (1.0 - block_dones).view(1, -1, 1)
                        self._lstm_states[j] = (h * mask, c * mask)

            self._global_obs = next_obs

        # Compute advantages / targets for each buffer.
        for j, buf in enumerate(self.buffers):
            block_obs = self.splitter.gather_block(self._global_obs, j)
            latent = self.networks.latent(j)
            lstm_state = self._lstm_states[j] if self.networks.actor.use_lstm else None
            next_value, _ = self.networks.critic_value(
                block_obs, latent, lstm_state=lstm_state
            )
            buf.compute_returns(
                next_value=next_value,
                gamma=self.networks.config.gamma,
                gae_lambda=self.networks.config.gae_lambda,
                on_policy_target_steps=self.networks.config.on_policy_target_steps,
            )

        return self.buffers


def _unpack_step(step_out: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Normalise the many vectorized-env step return conventions."""
    if isinstance(step_out, tuple):
        if len(step_out) == 4:
            obs, rew, done, info = step_out
        elif len(step_out) == 5:
            # Gymnasium-style: obs, rew, terminated, truncated, info
            obs, rew, terminated, truncated, info = step_out
            done = torch.as_tensor(terminated) | torch.as_tensor(truncated)
        else:
            raise ValueError(f"Unexpected step tuple length: {len(step_out)}")
    else:
        raise ValueError(f"Unexpected step output type: {type(step_out)}")

    obs = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
    rew = rew if isinstance(rew, torch.Tensor) else torch.as_tensor(rew, dtype=torch.float32)
    done = done if isinstance(done, torch.Tensor) else torch.as_tensor(done, dtype=torch.float32)
    rew = rew.reshape(-1).float()
    done = done.reshape(-1).float()
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    return obs, rew, done, info if isinstance(info, dict) else {}


def build_rollout_collector(
    env: Any,
    networks: Any,
    config: Any,
    obs_dim: int,
    action_dim: int,
    device: Optional[torch.device] = None,
) -> RolloutCollector:
    """Factory building a :class:`RolloutCollector` from a config."""
    device = device if device is not None else torch.device(config.device)
    splitter = EnvBlockSplitter(config.num_envs, config.num_blocks)
    return RolloutCollector(
        env=env,
        networks=networks,
        splitter=splitter,
        horizon=config.horizon,
        device=device,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
