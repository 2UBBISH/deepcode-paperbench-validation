"""Rollout collection for SAPG over many massively-parallel environments.

Source: §4 (Split and Aggregate Policy Gradients -- divide-and-conquer over the
``N`` environments: train ``M`` policies ``pi_1, ..., pi_M`` instead of one),
§4.6 (Algorithm 1 -- "We roll out ``M`` different policies and collect data
``D_1, ..., D_M`` for each"), §5.2 ("For each task, we use ``M=6`` policies for
our method ... in a total of ``N=24576`` environments ... We collect 16 steps of
experience per instance of the environment before every PPO update step").

The environment vector is split into ``M`` contiguous blocks of ``N / M``
environments.  Block ``j`` is driven by policy ``pi_j`` (the leader is block
index 1, see §4.3) and its transitions are accumulated into the per-policy
buffer ``D_j``.

The collector is deliberately written against a minimal environment interface so
that it works with either the IsaacGym vectorised task wrapper
(:mod:`sapg.envs.isaac_env`) or a lightweight stand-in used for unit tests:

    reset() -> obs
    step(actions) -> (obs, rewards, dones, infos)

where ``obs`` has shape ``(N, obs_dim)``, ``rewards``/``dones`` have shape
``(N,)`` and ``actions`` has shape ``(N, action_dim)``.  ``dones`` mark
transitions that terminated/were truncated; the environment auto-resets those
instances and returns the *reset* observations, exactly like IsaacGym's
``VecTask`` (and like ``rl_games``).

Policies must expose::

    out = policy.act(obs, phi=phi_j, hidden_state=h, masks=m, deterministic=False)

with ``out`` a mapping containing ``actions`` ``(n, action_dim)``, ``logprobs``
``(n,)``, ``values`` ``(n,)`` and (for recurrent policies) ``hidden_state``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ..buffers.rollout_buffer import RolloutBuffer
from ..utils.config import NUM_POLICIES, TOTAL_ENVS, SAPGConfig

__all__ = [
    "BlockManager",
    "RolloutCollector",
    "collect_data",
    "collect_on_policy",
    "split_blocks",
]


# --------------------------------------------------------------------------- #
# Block management
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlockManager:
    """Splits ``N`` environments into ``M`` contiguous blocks.

    The split is contiguous (Algorithm 1 / §4.6): block ``j`` owns the
    environment indices ``[j * N/M, (j+1) * N/M)``.  The leader is block index
    ``1`` (§4.3), i.e. ``blocks[0]`` in zero-based indexing.
    """

    num_envs: int = TOTAL_ENVS
    num_policies: int = NUM_POLICIES
    leader_index: int = 1  # 1-based, per §4.3 / Algorithm 1

    def __post_init__(self) -> None:  # pragma: no cover - trivial validation
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.num_policies <= 0:
            raise ValueError(
                f"num_policies must be positive, got {self.num_policies}"
            )
        if self.num_envs % self.num_policies != 0:
            raise ValueError(
                "num_envs must be divisible by num_policies so that blocks are "
                f"contiguous and equal-sized (got {self.num_envs} and "
                f"{self.num_policies})"
            )
        if not 1 <= self.leader_index <= self.num_policies:
            raise ValueError(
                f"leader_index {self.leader_index} out of range "
                f"[1, {self.num_policies}]"
            )

    # -- geometry ----------------------------------------------------------- #
    @property
    def block_size(self) -> int:
        """Number of environments per block, ``N / M``."""
        return self.num_envs // self.num_policies

    @property
    def leader(self) -> int:
        """Zero-based index of the leader policy (``0`` by default, §4.3)."""
        return self.leader_index - 1

    @property
    def followers(self) -> List[int]:
        """Zero-based indices of the follower policies."""
        return [j for j in range(self.num_policies) if j != self.leader]

    def slice(self, j: int) -> slice:
        """Contiguous environment slice handled by policy ``j``."""
        if not 0 <= j < self.num_policies:
            raise IndexError(f"policy index {j} out of range [0, {self.num_policies})")
        start = j * self.block_size
        return slice(start, start + self.block_size)

    def indices(self, j: int) -> torch.Tensor:
        """Environment indices handled by policy ``j`` (long tensor)."""
        s = self.slice(j)
        return torch.arange(s.start, s.stop, dtype=torch.long)

    def block_of_env(self) -> torch.Tensor:
        """Map each environment index to its policy index."""
        return torch.arange(self.num_envs, dtype=torch.long) // self.block_size

    def is_leader(self, j: int) -> bool:
        return j == self.leader

    def __len__(self) -> int:
        return self.num_policies


def split_blocks(
    num_envs: int = TOTAL_ENVS, num_policies: int = NUM_POLICIES
) -> List[slice]:
    """Return the list of contiguous environment slices, one per policy."""
    return [BlockManager(num_envs, num_policies).slice(j) for j in range(num_policies)]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _policy_act(
    policy: Any,
    obs: torch.Tensor,
    phi: Optional[torch.Tensor],
    hidden_state: Optional[Any],
    masks: Optional[torch.Tensor],
    deterministic: bool,
) -> Dict[str, torch.Tensor]:
    """Call ``policy.act`` and normalise the returned keys."""
    out = policy.act(
        obs,
        phi=phi,
        hidden_state=hidden_state,
        masks=masks,
        deterministic=deterministic,
    )
    if not isinstance(out, Mapping):  # pragma: no cover - defensive
        raise TypeError(
            "policy.act must return a mapping with 'actions', 'logprobs' and "
            f"'values', got {type(out)}"
        )
    out = dict(out)
    if "logprobs" not in out and "logprob" in out:
        out["logprobs"] = out["logprob"]
    if "values" not in out and "value" in out:
        out["values"] = out["value"]
    for key in ("actions", "logprobs", "values"):
        if key not in out:
            raise KeyError(f"policy.act output is missing required key '{key}'")
    if out["logprobs"].dim() > 1:
        out["logprobs"] = out["logprobs"].reshape(obs.shape[0])
    if out["values"].dim() > 1:
        out["values"] = out["values"].reshape(obs.shape[0])
    return out


def _phi_for(phis: Optional[Sequence[torch.Tensor]], j: int) -> Optional[torch.Tensor]:
    if phis is None:
        return None
    return phis[j]


def _reset_hidden(hidden_state: Any, dones: torch.Tensor) -> Any:
    """Zero the recurrent state of environments that just finished an episode."""
    if hidden_state is None:
        return None
    mask = (~dones.bool()).to(dtype=torch.float32)
    if isinstance(hidden_state, torch.Tensor):
        view = (hidden_state.shape[0],) + (1,) * (hidden_state.dim() - 1)
        return hidden_state * mask.view(view)
    # (h, c) tuple, as returned by nn.LSTM
    reset = []
    for term in hidden_state:
        view = (term.shape[1],) + (1,) * (term.dim() - 1)
        reset.append(term * mask.view((1,) + view))
    return type(hidden_state)(*reset) if not isinstance(hidden_state, list) else reset


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #
class RolloutCollector:
    """Collects ``horizon_length`` steps for every block of environments.

    Parameters
    ----------
    env:
        Vectorised environment exposing ``step``/``reset`` (optionally
        ``get_obs``) over all ``N`` environments.
    config:
        :class:`~sapg.utils.config.SAPGConfig`; ``horizon_length`` is the number
        of steps collected per environment instance before a PPO update (§5.2:
        16 for the AllegroKuka tasks).
    block_manager:
        Optional block geometry; defaults to ``num_policies`` blocks of equal
        size over ``config.num_envs``.
    device:
        Device for the created rollout buffers (defaults to ``config.device``).
    """

    def __init__(
        self,
        env: Any,
        config: SAPGConfig,
        block_manager: Optional[BlockManager] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device or config.device)
        self.blocks = block_manager or BlockManager(
            num_envs=config.num_envs,
            num_policies=config.num_policies,
            leader_index=config.leader_index,
        )
        if self.blocks.num_envs != config.num_envs:
            raise ValueError(
                f"block manager holds {self.blocks.num_envs} environments but the "
                f"config specifies {config.num_envs}"
            )
        self._obs: Optional[torch.Tensor] = None
        self._masks: Optional[torch.Tensor] = None
        self._episode_returns: Optional[torch.Tensor] = None
        self._episode_lengths: Optional[torch.Tensor] = None
        self._finished_returns: List[float] = []
        self._finished_lengths: List[float] = []

    # -- state -------------------------------------------------------------- #
    @property
    def obs(self) -> torch.Tensor:
        if self._obs is None:
            raise RuntimeError("RolloutCollector.reset() must be called first")
        return self._obs

    def reset(self) -> torch.Tensor:
        """Reset the environment and the episode bookkeeping."""
        self._obs = self.env.reset()
        if isinstance(self._obs, tuple):  # (obs, info)
            self._obs = self._obs[0]
        self._obs = self._obs.to(self.device)
        n = self.blocks.num_envs
        self._masks = torch.ones(n, dtype=torch.bool, device=self.device)
        self._episode_returns = torch.zeros(n, device=self.device)
        self._episode_lengths = torch.zeros(n, device=self.device)
        self._finished_returns = []
        self._finished_lengths = []
        return self._obs

    # -- buffers ------------------------------------------------------------ #
    def create_buffers(
        self, phi_dim: Optional[int] = None, per_block_sigma: bool = False
    ) -> List[RolloutBuffer]:
        """Allocate one :class:`RolloutBuffer` per policy (``D_1 .. D_M``)."""
        hidden = None
        if self.config.use_lstm:
            hidden = (self.config.lstm_hidden_size, self.config.lstm_num_layers)
        buffers = []
        for j in range(self.blocks.num_policies):
            buffers.append(
                RolloutBuffer(
                    num_envs=self.blocks.block_size,
                    horizon_length=self.config.horizon_length,
                    obs_dim=self.config.obs_dim,
                    action_dim=self.config.action_dim,
                    device=self.device,
                    recurrent=self.config.recurrent and self.config.use_lstm,
                    lstm_sequence_length=self.config.lstm_sequence_length,
                    hidden_shape=hidden,
                    phi_dim=self.config.phi_dim if phi_dim is None else phi_dim,
                    policy_index=j,
                    is_leader=self.blocks.is_leader(j),
                )
            )
        return buffers

    # -- collection --------------------------------------------------------- #
    def collect(
        self,
        policies: Sequence[Any],
        phis: Optional[Sequence[torch.Tensor]] = None,
        hidden_states: Optional[Sequence[Any]] = None,
        buffers: Optional[Sequence[RolloutBuffer]] = None,
        deterministic: bool = False,
        obs: Optional[torch.Tensor] = None,
    ) -> Tuple[List[RolloutBuffer], torch.Tensor, List[Any], Dict[str, float]]:
        """Roll out every policy on its own block for ``horizon_length`` steps.

        Returns ``(buffers, obs, hidden_states, metrics)``.  ``obs`` is the
        observation *after* the last step and ``hidden_states`` the matching
        per-policy recurrent state, both to be fed into the next call.
        """
        if len(policies) != self.blocks.num_policies:
            raise ValueError(
                f"expected {self.blocks.num_policies} policies, got {len(policies)}"
            )
        if self._obs is None or obs is not None:
            if obs is None:
                self.reset()
            else:
                self._obs = obs.to(self.device)

        if buffers is None:
            buffers = self.create_buffers()
        if len(buffers) != self.blocks.num_policies:
            raise ValueError(
                f"expected {self.blocks.num_policies} buffers, got {len(buffers)}"
            )

        if hidden_states is None:
            hidden_states = [None] * self.blocks.num_policies
        hidden_states = list(hidden_states)

        action_dim = buffers[0].action_dim
        metrics = self._empty_metrics()
        for _ in range(self.config.horizon_length):
            actions = torch.zeros(
                self.blocks.num_envs, action_dim, device=self.device
            )
            step_records: List[Dict[str, torch.Tensor]] = []
            for j in range(self.blocks.num_policies):
                sl = self.blocks.slice(j)
                out = _policy_act(
                    policies[j],
                    self._obs[sl],
                    _phi_for(phis, j),
                    hidden_states[j],
                    None if self._masks is None else self._masks[sl],
                    deterministic,
                )
                actions[sl] = out["actions"]
                if "hidden_state" in out:
                    hidden_states[j] = out["hidden_state"]
                step_records.append(out)

            next_obs, rewards, dones, infos = self._env_step(actions)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            for j in range(self.blocks.num_policies):
                sl = self.blocks.slice(j)
                out = step_records[j]
                buffers[j].add(
                    obs=self._obs[sl],
                    actions=out["actions"],
                    logprobs=out["logprobs"],
                    values=out["values"],
                    rewards=rewards[sl],
                    dones=dones[sl],
                    phi=_phi_for(phis, j),
                    hidden_state=hidden_states[j],
                )
                hidden_states[j] = _reset_hidden(hidden_states[j], dones[sl])

            self._accumulate_episode_stats(rewards, dones, infos)
            self._obs = next_obs.to(self.device)
            self._masks = dones.bool()

        # Bootstrap values for the (auto-reset) observations that end each rollout.
        for j in range(self.blocks.num_policies):
            sl = self.blocks.slice(j)
            out = _policy_act(
                policies[j],
                self._obs[sl],
                _phi_for(phis, j),
                hidden_states[j],
                None,
                deterministic=True,
            )
            done_mask = (~self._masks[sl].bool()).to(out["values"].dtype)
            buffers[j].set_last_values(out["values"] * done_mask)
            buffers[j].finalise()

        metrics.update(self._episode_metrics())
        return buffers, self._obs, hidden_states, metrics

    # -- environment / bookkeeping ------------------------------------------ #
    def _env_step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        out = self.env.step(actions)
        if not isinstance(out, (tuple, list)) or len(out) < 4:
            raise TypeError(
                "env.step must return (obs, rewards, dones, infos), got "
                f"{type(out)}"
            )
        obs, rewards, dones, infos = out[0], out[1], out[2], out[3]
        if isinstance(obs, tuple):  # (obs, info) style API
            obs = obs[0]
        return obs, rewards, dones, infos

    def _accumulate_episode_stats(
        self, rewards: torch.Tensor, dones: torch.Tensor, infos: Any
    ) -> None:
        if self._episode_returns is None or self._episode_lengths is None:
            return
        self._episode_returns = self._episode_returns + rewards
        self._episode_lengths = self._episode_lengths + 1.0
        done = dones.bool()
        if done.any():
            self._finished_returns.extend(
                self._episode_returns[done].detach().cpu().tolist()
            )
            self._finished_lengths.extend(
                self._episode_lengths[done].detach().cpu().tolist()
            )
            self._episode_returns = self._episode_returns * (~done).to(rewards.dtype)
            self._episode_lengths = self._episode_lengths * (~done).to(rewards.dtype)
        # Task-specific success information (dict of per-env arrays) taken from
        # the environment when available, e.g. {'successes': ...}.
        if self._extra_metrics and isinstance(infos, Mapping):
            for key in self._extra_metrics:
                value = infos.get(key)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    self._metric_sums[key] += float(value.float().sum().item())
                    self._metric_counts[key] += float(done.sum().item()) or float(
                        value.numel()
                    )

    def _empty_metrics(self) -> Dict[str, float]:
        self._extra_metrics = list(getattr(self.config, "logged_metrics", []) or [])
        self._metric_sums: Dict[str, float] = {k: 0.0 for k in self._extra_metrics}
        self._metric_counts: Dict[str, float] = {k: 0.0 for k in self._extra_metrics}
        return {}

    def _episode_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self._finished_returns:
            metrics["reward/episode_return"] = float(
                sum(self._finished_returns) / len(self._finished_returns)
            )
            metrics["reward/episode_length"] = float(
                sum(self._finished_lengths) / len(self._finished_lengths)
            )
            metrics["reward/num_episodes"] = float(len(self._finished_returns))
        else:
            metrics["reward/mean_step_reward"] = float(
                self._episode_returns.mean().item()
                if self._episode_returns is not None
                else 0.0
            )
        for key, total in self._metric_sums.items():
            count = self._metric_counts.get(key, 0.0)
            if count:
                metrics[f"task/{key}"] = total / count
        return metrics


# --------------------------------------------------------------------------- #
# Functional API (used by the trainers)
# --------------------------------------------------------------------------- #
def collect_data(
    policies: Sequence[Any],
    env: Any,
    config: SAPGConfig,
    obs: Optional[torch.Tensor] = None,
    phis: Optional[Sequence[torch.Tensor]] = None,
    hidden_states: Optional[Sequence[Any]] = None,
    block_manager: Optional[BlockManager] = None,
    deterministic: bool = False,
) -> Tuple[List[RolloutBuffer], torch.Tensor, List[Any], Dict[str, float]]:
    """Convenience wrapper: one-shot collection over all ``M`` blocks (§4.6)."""
    collector = RolloutCollector(env, config, block_manager=block_manager)
    if obs is not None:
        collector._obs = obs.to(collector.device)
        n = collector.blocks.num_envs
        if collector._masks is None:
            collector._masks = torch.zeros(n, dtype=torch.bool, device=collector.device)
        if collector._episode_returns is None:
            collector._episode_returns = torch.zeros(n, device=collector.device)
            collector._episode_lengths = torch.zeros(n, device=collector.device)
    else:
        collector.reset()
    return collector.collect(
        policies,
        phis=phis,
        hidden_states=hidden_states,
        deterministic=deterministic,
    )


def collect_on_policy(
    policy: Any,
    env: Any,
    config: SAPGConfig,
    obs: Optional[torch.Tensor] = None,
    hidden_state: Optional[Any] = None,
    deterministic: bool = False,
) -> Tuple[RolloutBuffer, torch.Tensor, Any, Dict[str, float]]:
    """Collect a single-policy rollout (vanilla PPO baseline, §5.2)."""
    single = SAPGConfig.from_dict(
        {**config.to_dict(), "num_policies": 1, "num_envs": config.num_envs}
    )
    buffers, new_obs, hidden_states, metrics = collect_data(
        [policy],
        env,
        single,
        obs=obs,
        phis=None,
        hidden_states=[hidden_state],
        deterministic=deterministic,
    )
    return buffers[0], new_obs, hidden_states[0], metrics
