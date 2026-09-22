"""The two objectives of Refined Coreset Selection.

Objective (O1) -- primary:

    f_1(m) = (1/n) sum_{i=1..n} l(h(x_i; theta(m)), y_i)
    s.t. theta(m) in arg min_theta L(m, theta)

Objective (O2) -- secondary:

    f_2(m) = ||m||_0

Evaluating ``f_1`` requires solving the inner loop for the mask, so the
evaluator caches results for already queried masks.  The cache is keyed by the
*discretised* mask, because the continuous search variable only influences the
objectives through its projection onto ``{0, 1}`` (Appendix A).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .data import DatasetBundle
from .inner_loop import InnerLoopTrainer
from .utils import discretize, evaluate_loss, make_loader


@dataclass
class ObjectiveConfig:
    """Configuration of the f1 / f2 evaluator."""

    eval_batch_size: int = 512
    f1_eval_size: Optional[int] = None   # None -> evaluate on the full dataset
    f1_eval_seed: int = 0


class BilevelObjective:
    """Callable ``F(m) = [f_1(m), f_2(m)]`` for bilevel coreset selection."""

    def __init__(self, bundle: DatasetBundle, trainer: InnerLoopTrainer,
                 device: torch.device, cfg: Optional[ObjectiveConfig] = None,
                 group_size: int = 1):
        self.bundle = bundle
        self.trainer = trainer
        self.device = device
        self.cfg = cfg or ObjectiveConfig()
        self.group_size = int(group_size)

        # f_1 is the loss on the *full* training set; for very large datasets a
        # fixed random subset can be used instead (f1_eval_size).
        if self.cfg.f1_eval_size is None or self.cfg.f1_eval_size >= bundle.n:
            self.eval_x, self.eval_y = bundle.train_x, bundle.train_y
        else:
            g = torch.Generator().manual_seed(self.cfg.f1_eval_seed)
            idx = torch.randperm(bundle.n, generator=g)[:self.cfg.f1_eval_size]
            self.eval_x, self.eval_y = bundle.train_x[idx], bundle.train_y[idx]
        self._eval_loader = make_loader(self.eval_x, self.eval_y,
                                        self.cfg.eval_batch_size,
                                        shuffle=False)

        self.cache: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        self.history: list = []          # sequence of queried discrete masks
        self.num_trainings = 0

    # -- helpers -----------------------------------------------------------
    def group_indices(self, mask: torch.Tensor) -> torch.Tensor:
        """Map a fine-grained mask onto group representatives.

        The paper treats several examples as a group when the search space is
        too large for the outer loop; all examples of a group share one mask
        entry.  Grouping is applied on the *continuous* variable before
        discretisation.
        """
        if self.group_size <= 1:
            return mask
        mask = mask.view(-1)
        n = mask.numel()
        pad = (-n) % self.group_size
        if pad:
            mask = torch.cat([mask, mask.new_zeros(pad)])
        grouped = mask.view(-1, self.group_size).mean(dim=1)[:, None]
        grouped = grouped.expand(-1, self.group_size).reshape(-1)
        return grouped[:n]

    # -- evaluation --------------------------------------------------------
    def f1_from_model(self, model: nn.Module) -> float:
        return evaluate_loss(model, self._eval_loader, self.device)

    def evaluate(self, continuous_mask: torch.Tensor,
                 warm_start: Optional[bool] = None) -> torch.Tensor:
        """Return ``[f_1(m), f_2(m)]`` for a (possibly continuous) mask."""
        continuous_mask = self.group_indices(
            continuous_mask.detach().reshape(-1).to(self.device))
        discrete = discretize(continuous_mask)
        key = tuple(int(v) for v in discrete.cpu().tolist())
        if key in self.cache:
            f1, f2 = self.cache[key]
            return torch.tensor([f1, f2], dtype=torch.float64)

        # the dataset lives on the CPU, so indices must come back to the CPU
        # even when the mask is optimised on an accelerator
        indices = torch.nonzero(discrete > 0.5,
                                as_tuple=False).flatten().cpu()
        if indices.numel() == 0:            # degenerate mask: empty coreset
            f1 = float("inf")
            f2 = 0.0
        else:
            model = self.trainer(self.bundle, indices, warm_start=warm_start)
            self.num_trainings += 1
            f1 = self.f1_from_model(model)
            f2 = float(indices.numel())
        self.cache[key] = (f1, f2)
        self.history.append((key, f1, f2))
        return torch.tensor([f1, f2], dtype=torch.float64)

    # -- bookkeeping -------------------------------------------------------
    def known_thresholds(self) -> torch.Tensor:
        """``[min f1, min f2]`` over everything evaluated so far."""
        if not self.cache:
            return torch.tensor([float("inf"), float("inf")], dtype=torch.float64)
        f1s = [v[0] for v in self.cache.values()]
        f2s = [v[1] for v in self.cache.values()]
        return torch.tensor([min(f1s), min(f2s)], dtype=torch.float64)
