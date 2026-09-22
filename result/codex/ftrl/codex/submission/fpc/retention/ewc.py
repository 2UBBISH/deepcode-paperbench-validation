"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

The EWC auxiliary loss penalises changes of the parameters that were important
for the pre-trained solution::

    L_EWC(theta) = sum_i F^i (theta_pre^i - theta^i)^2

Only the *actor* is regularised, following Wolczyk et al. (2022) and
Appendix C.5.

Hyperparameters used in the paper
---------------------------------
* NetHack: regularization coefficient ``2e6`` (Appendix B.1).
* Montezuma's Revenge: same regularization form, Fisher estimated from the
  pre-training data.
* Meta-World: actor regression coefficient ``100`` (Table 3).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import RetentionConfig, RetentionMethod
from .fisher import DiagonalFisher


class EWC(RetentionMethod):
    """Elastic Weight Consolidation applied to a subset of parameters."""

    def __init__(self, config: Optional[RetentionConfig] = None) -> None:
        super().__init__(config)
        self.fisher: Optional[Dict[str, Tensor]] = None
        self.anchor: Dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    def set_fisher(self, fisher: Mapping[str, Tensor], normalize: bool = False) -> None:
        """Register the diagonal Fisher computed on the pre-training data."""

        self.fisher = {k: v.clone() for k, v in fisher.items()}
        if normalize:
            max_val = max((float(v.max()) for v in self.fisher.values()), default=1.0)
            if max_val > 0:
                self.fisher = {k: v / max_val for k, v in self.fisher.items()}

    def register_anchor(self, module: nn.Module, param_filter=None) -> None:
        """Snapshot the pre-trained parameters ``theta_pre``."""

        param_filter = param_filter or (lambda name: True)
        self.anchor = {
            name: p.detach().clone()
            for name, p in module.named_parameters()
            if param_filter(name)
        }

    def compute_fisher_from(
        self,
        module: nn.Module,
        log_prob_fn,
        batches,
        num_batches: int = 10000,
        param_filter=None,
    ) -> Dict[str, Tensor]:
        """Convenience wrapper around :class:`DiagonalFisher`."""

        fisher = DiagonalFisher.estimate(
            module, log_prob_fn, batches, num_batches=num_batches, param_filter=param_filter
        )
        self.set_fisher(fisher)
        return fisher

    # ------------------------------------------------------------------
    def aux_loss(self, module: Optional[nn.Module] = None, **_: object) -> Tensor:
        """Return ``sum_i F^i (theta_pre^i - theta^i)^2``.

        ``module`` must be provided (or the anchor/fisher registered through
        :meth:`register_anchor`/:meth:`set_fisher`).  Parameters missing from the
        Fisher are skipped, which makes it possible to regularise only the actor
        of an actor-critic network.
        """

        if self.fisher is None:
            raise RuntimeError("EWC.fisher is not set; call set_fisher() first.")
        device = next(iter(self.fisher.values())).device
        total = torch.zeros((), device=device)
        named = dict(module.named_parameters()) if module is not None else {}
        for name, f in self.fisher.items():
            param = named.get(name)
            if param is None:
                continue
            anchor = self.anchor.get(name)
            if anchor is None:
                anchor = param.detach().clone()
                self.anchor[name] = anchor
            total = total + (f.to(param.device) * (anchor.to(param.device) - param) ** 2).sum()
        return total

    def state_dict(self) -> Dict[str, object]:
        state = super().state_dict()
        state["fisher_keys"] = list(self.fisher.keys()) if self.fisher else []
        return state
