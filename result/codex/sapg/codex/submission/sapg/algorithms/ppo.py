"""Vanilla PPO baseline (Schulman et al., 2017) in the large-batch setting.

PPO is SAPG with a single policy (``M = 1``), no splitting and no off-policy
aggregation; its batch size grows proportionally with the number of parallel
environments, which is the "naive scaling" studied in Figure 2 of the paper.
"""

from __future__ import annotations

from typing import Optional

from .sapg import SAPGTrainer


class PPOTrainer(SAPGTrainer):
    name = "ppo"

    def __init__(self, cfg, env, device: str = "cpu", logdir: Optional[str] = None) -> None:
        super().__init__(cfg, env, device=device, logdir=logdir)
        if self.num_policies != 1:
            raise ValueError(
                "PPOTrainer uses a single policy; use SAPGTrainer for M > 1 "
                "(set algo.num_policies=1 for the PPO baseline)"
            )
        self.use_offpolicy = False
        self.aggregation = "none"
