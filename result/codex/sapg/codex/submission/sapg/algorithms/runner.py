"""Rollout collection for ``M`` policies sharing one vectorised environment.

The design follows Sec. 4.6 / Algorithm 1: the ``N`` environments of the
simulator are split into ``M`` blocks of ``N / M`` environments; every block is
stepped by its own policy (all policies share ``B_theta`` / ``C_psi`` and are
conditioned on their own ``phi_j``), and the collected transitions land in the
per-policy buffers ``D_1 ... D_M``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .actor_critic import ActorCritic
from .storage import RolloutStorage


class MPolicyRunner:
    def __init__(
        self,
        env,
        model: ActorCritic,
        num_policies: int,
        device: torch.device,
        recurrent: bool = False,
    ) -> None:
        self.env = env
        self.model = model
        self.num_policies = num_policies
        self.device = device
        self.recurrent = recurrent
        if env.num_envs % num_policies != 0:
            raise ValueError(
                f"num_envs ({env.num_envs}) must be divisible by num_policies ({num_policies})"
            )
        self.num_envs_per_policy = env.num_envs // num_policies
        self.obs: Optional[torch.Tensor] = None
        self.recurrent_states: Optional[List] = None
        self._pending_rewards: Optional[List[torch.Tensor]] = None
        self._pending_dones: Optional[List[torch.Tensor]] = None
        self._pending_timeouts: Optional[List[torch.Tensor]] = None

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.obs = self.env.reset()
        self.recurrent_states = [None] * self.num_policies
        zeros = torch.zeros(self.num_envs_per_policy, device=self.device)
        self._pending_rewards = [zeros.clone() for _ in range(self.num_policies)]
        self._pending_dones = [zeros.clone() for _ in range(self.num_policies)]
        self._pending_timeouts = [zeros.clone() for _ in range(self.num_policies)]

    def _block(self, policy_id: int) -> Tuple[int, int]:
        start = policy_id * self.num_envs_per_policy
        return start, start + self.num_envs_per_policy

    # ------------------------------------------------------------------ #
    def collect(self, storage: RolloutStorage, horizon: int) -> Dict[int, torch.Tensor]:
        """Fill ``storage`` with ``horizon`` steps for every policy."""
        if self.obs is None:
            self.reset()

        for t in range(horizon):
            actions_all = torch.zeros(
                (self.env.num_envs, self.env.action_dim), device=self.device
            )
            for policy_id in range(self.num_policies):
                start, end = self._block(policy_id)
                obs_block = self.obs[start:end]
                policy_ids = torch.full(
                    (obs_block.shape[0],), policy_id, dtype=torch.long, device=self.device
                )
                actions, logprobs, values, states = self.model.act(
                    obs_block,
                    policy_ids=policy_ids,
                    recurrent_states=self.recurrent_states[policy_id],
                    dones=self._pending_dones[policy_id] if self.recurrent else None,
                )
                self.recurrent_states[policy_id] = states
                storage.buffer(policy_id).insert(
                    t,
                    obs_block,
                    actions,
                    logprobs,
                    values,
                    self._pending_rewards[policy_id],
                    self._pending_dones[policy_id],
                    self._pending_timeouts[policy_id],
                    obs_block,  # next_obs placeholder, patched below
                )
                actions_all[start:end] = actions

            step = self.env.step(actions_all)
            self.obs = step.obs
            for policy_id in range(self.num_policies):
                start, end = self._block(policy_id)
                self._pending_rewards[policy_id] = step.rewards[start:end]
                self._pending_dones[policy_id] = step.dones[start:end]
                self._pending_timeouts[policy_id] = step.timeouts[start:end]
                storage.buffer(policy_id).next_obs[t] = self.obs[start:end]

        # bootstrap values V_old(s_T) for every policy
        last_values: Dict[int, torch.Tensor] = {}
        for policy_id in range(self.num_policies):
            start, end = self._block(policy_id)
            obs_block = self.obs[start:end]
            policy_ids = torch.full(
                (obs_block.shape[0],), policy_id, dtype=torch.long, device=self.device
            )
            value, states = self.model.get_value(
                obs_block,
                policy_ids=policy_ids,
                recurrent_states=self.recurrent_states[policy_id],
                dones=self._pending_dones[policy_id] if self.recurrent else None,
            )
            self.recurrent_states[policy_id] = states
            last_values[policy_id] = value
        return last_values
