"""SAPG: Split and Aggregate Policy Gradients (implementation of Algorithm 1).

One training iteration
----------------------
1. **Split.** ``N`` environments are divided into ``M`` blocks of ``N/M``
   environments.  Policy ``j`` (shared backbone ``B_theta`` / ``C_psi``
   conditioned on its own local ``phi_j``) rolls out its block and fills
   ``D_j``.
2. **Aggregate.** The leader (``i = 1``) is updated with its own on-policy data
   *plus* importance-sampled off-policy data from the followers, with the
   off-policy data subsampled so that a minibatch holds as many off-policy as
   on-policy transitions (Sec. 4.3, ``lambda = 1``).  Followers are updated with
   ordinary PPO on their own data; the ``j``-th follower additionally receives
   an entropy bonus ``sigma * j * H(pi_j(a|s))`` (Sec. 4.5, the leader gets
   none).
3. **Update.** A single backward pass over the summed objective of all ``M``
   policies updates ``theta`` and ``psi`` with the gradients of *every* policy
   while ``phi_j`` only receives the gradient of policy ``j``'s own objective
   (Sec. 4.4) -- ``phi_j`` appears in no other term of the sum.

Ablations from Sec. 6.3 are switches on this trainer:

============================  =================================================
``aggregation: none``         SAPG without the off-policy combination
                              (= ``lambda = 0``).
``aggregation: symmetric``    Sec. 4.2: every policy uses off-policy data from
                              all others (subsampled to match on-policy size).
``offpolicy_subsample: false``   "high off-policy ratio": the complete
                              off-policy dataset is used instead.
``entropy_coef: sigma``       entropy coefficient ``sigma in {0, 0.003, 0.005}``
                              applied as ``sigma * j`` to follower ``j``.
============================  =================================================
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch

from .base import TrainerBase
from .losses import OffPolicyLoss, OnPolicyLoss
from .storage import OffPolicyData


class SAPGTrainer(TrainerBase):
    name = "sapg"

    def __init__(self, cfg, env, device: str = "cpu", logdir: Optional[str] = None) -> None:
        super().__init__(cfg, env, device=device, logdir=logdir)
        self.aggregation = str(self.a_get("aggregation", "leader_follower")).lower()
        if self.aggregation not in ("leader_follower", "symmetric", "none"):
            raise ValueError(f"Unknown aggregation scheme '{self.aggregation}'")
        self.use_offpolicy = bool(self.a_get("use_offpolicy", True)) and self.aggregation != "none"
        self.offpolicy_lambda = float(self.a_get("offpolicy_lambda", 1.0))
        self.subsample_offpolicy = bool(self.a_get("offpolicy_subsample", True))
        self.entropy_coef = float(self.a_get("entropy_coef", 0.0))
        self.on_policy_loss = OnPolicyLoss(cfg.get("algo", {}))
        self.off_policy_loss = OffPolicyLoss(cfg.get("algo", {}))

    # ------------------------------------------------------------------ #
    # Sec. 4.5: entropy coefficient sigma * (i - 1) for the (i-1)-th follower
    # ------------------------------------------------------------------ #
    def entropy_coef_for(self, policy_id: int) -> float:
        return 0.0 if policy_id == 0 else self.entropy_coef * policy_id

    def offpolicy_sources(self, policy_id: int) -> List[int]:
        """Return X_i, the set of policies whose data updates policy ``i``."""
        if not self.use_offpolicy:
            return []
        if self.aggregation == "leader_follower":
            return list(range(1, self.num_policies)) if policy_id == 0 else []
        if self.aggregation == "symmetric":
            return [j for j in range(self.num_policies) if j != policy_id]
        return []

    # ------------------------------------------------------------------ #
    # Sec. 4.1: build the off-policy mini-dataset for every receiving policy
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prepare_offpolicy(self) -> Dict[int, OffPolicyData]:
        data: Dict[int, OffPolicyData] = {}
        for policy_id in range(self.num_policies):
            sources = self.offpolicy_sources(policy_id)
            if not sources:
                continue
            data[policy_id] = self._build_offpolicy_dataset(policy_id, sources)
        return data

    @torch.no_grad()
    def _build_offpolicy_dataset(self, policy_id: int, sources: List[int]) -> OffPolicyData:
        horizon = self.horizon
        if self.subsample_offpolicy:
            # "we subsample the off-policy data such that we use equal amounts of
            # on-policy and off-policy data in a mini-batch update" (Sec. 4.3):
            # |D'_i| = |D_i| transitions, sampled uniformly from the union over
            # all source policies.  Whole sequences are sampled so that the
            # recurrent minibatching (seq_len = horizon) stays valid.
            num_sequences = self.num_envs_per_policy
            source_ids = torch.randint(0, len(sources), (num_sequences,), generator=self.generator)
            seq_ids = torch.randint(
                0, self.num_envs_per_policy, (num_sequences,), generator=self.generator
            )
        else:
            # "high off-policy ratio" ablation: use the full off-policy dataset
            num_sequences = len(sources) * self.num_envs_per_policy
            source_ids = torch.arange(len(sources)).repeat_interleave(self.num_envs_per_policy)
            seq_ids = torch.arange(self.num_envs_per_policy).repeat(len(sources))

        obs = torch.zeros((horizon, num_sequences, self.env.obs_dim), device=self.device)
        next_obs = torch.zeros_like(obs)
        actions = torch.zeros((horizon, num_sequences, self.env.action_dim), device=self.device)
        behavior_logprobs = torch.zeros((horizon, num_sequences), device=self.device)
        rewards = torch.zeros((horizon, num_sequences), device=self.device)
        dones = torch.zeros((horizon, num_sequences), device=self.device)
        for source_idx, source_policy in enumerate(sources):
            mask = source_ids == source_idx
            if not bool(mask.any()):
                continue
            buffer = self.storage.buffer(source_policy)
            sel = seq_ids[mask]
            obs[:, mask] = buffer.obs[:, sel]
            next_obs[:, mask] = buffer.next_obs[:, sel]
            actions[:, mask] = buffer.actions[:, sel]
            behavior_logprobs[:, mask] = buffer.logprobs[:, sel]
            rewards[:, mask] = buffer.rewards[:, sel]
            dones[:, mask] = buffer.dones[:, sel]

        leader_ids = torch.full((num_sequences,), policy_id, dtype=torch.long, device=self.device)
        # pi_{i,old}(a|s) and V_{i,old} are evaluated *before* any gradient step,
        # i.e. with the parameters that are "old" for this update.
        old_logprobs, _, value_s, _, _ = self.model.evaluate_actions(
            obs, actions, policy_ids=leader_ids, dones=dones
        )
        value_next, _ = self.model.get_value(next_obs, policy_ids=leader_ids, dones=dones)

        nonterminal = 1.0 - dones
        # "we assume that an off-policy transition can be used to approximate a
        #  1-step return":  V_off^target(s_t) = r_t + gamma V_old(s_{t+1})
        value_targets = rewards + self.gamma * nonterminal * value_next
        advantages = value_targets - value_s
        mu = torch.exp(old_logprobs - behavior_logprobs)
        mu = torch.clamp(mu, max=float(self.a_get("max_importance_weight", 10.0)))

        dataset = OffPolicyData(
            obs=obs,
            actions=actions,
            behavior_logprobs=behavior_logprobs,
            old_logprobs=old_logprobs,
            mu=mu,
            advantages=advantages,
            value_targets=value_targets,
            dones=dones,
        )
        # which follower produced each sampled sequence (book-keeping only)
        dataset.source_policies = torch.tensor(
            [sources[int(s)] for s in source_ids], dtype=torch.long, device=self.device
        )
        return dataset

    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        offpolicy_data = self.prepare_offpolicy()
        metrics = defaultdict(list)

        for _epoch in range(self.mini_epochs):
            on_policy_chunks = {
                j: self.storage.on_policy_chunks(
                    j, self.num_minibatches, self.minibatch_size, self.generator
                )
                for j in range(self.num_policies)
            }
            off_policy_chunks = {
                i: dataset.chunks(self.num_minibatches, self.generator)
                for i, dataset in offpolicy_data.items()
            }

            for chunk_idx in range(self.num_minibatches):
                total_loss = torch.zeros((), device=self.device)

                # ---- on-policy losses of the leader and of every follower ----
                for j in range(self.num_policies):
                    losses = self.on_policy_loss(
                        self.model,
                        on_policy_chunks[j][chunk_idx],
                        entropy_coef=self.entropy_coef_for(j),
                    )
                    total_loss = total_loss + losses["loss"]
                    for key in (
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "approx_kl",
                        "clip_fraction",
                        "ratio_std",
                    ):
                        metrics[f"on/{key}"].append(float(losses[key]))
                    if self.entropy_coef_for(j) > 0:
                        metrics["on/follower_entropy"].append(float(losses["entropy"]))

                # ---- off-policy losses (leader, or every policy if symmetric) --
                for i, chunks in off_policy_chunks.items():
                    losses = self.off_policy_loss(self.model, chunks[chunk_idx])
                    total_loss = total_loss + self.offpolicy_lambda * losses["loss"]
                    for key in ("policy_loss", "value_loss", "approx_kl", "ess", "ratio_mean", "mu_mean"):
                        metrics[f"off/{key}"].append(float(losses[key]))

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                metrics["grad_norm"].append(float(grad_norm))
                self.optimizer.step()

        approx_kl = float(sum(metrics["on/approx_kl"]) / max(1, len(metrics["on/approx_kl"])))
        new_lr = self.lr_scheduler.update(approx_kl)

        result = {key: float(sum(values) / max(1, len(values))) for key, values in metrics.items()}
        result["learning_rate"] = float(new_lr)
        result["approx_kl"] = approx_kl
        if offpolicy_data:
            sizes = [d.num_transitions() for d in offpolicy_data.values()]
            result["off/num_transitions"] = float(sum(sizes) / len(sizes))
        result["collect_time"] = float(getattr(self, "collect_time", 0.0))
        return result
