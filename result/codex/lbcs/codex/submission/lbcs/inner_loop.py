"""Inner loop of the bilevel coreset selection problem.

The inner loop solves

    theta(m) in arg min_theta  L(m, theta),
    L(m, theta) = (1 / ||m||_0) sum_i m_i l(h(x_i; theta), y_i),

that is, it trains the network on the currently selected coreset.

Two acceleration tricks of Section 3.2 are implemented here:

* warm starting: "we can first train a model with random masks and then
  finetune it with other different masks in Step 3";
* grouping: several examples can share one mask entry, which shrinks the
  search space of the outer loop (used for large datasets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DatasetBundle
from .utils import (CosineScheduler, make_loader, make_optimizer, set_seed,
                    train_one_epoch)


@dataclass
class InnerLoopConfig:
    epochs: int = 100
    batch_size: int = 128
    optimizer: str = "adam"
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: Optional[str] = None        # None | "cosine" | "step"
    step_size: int = 30
    gamma: float = 0.1
    warm_start: bool = True                # finetune instead of re-initialising
    max_steps_per_epoch: Optional[int] = None
    grad_clip: Optional[float] = None      # optional stability safeguard
    # Full-batch mode: one gradient step per "epoch" over the whole coreset,
    # which is how the reference implementation of Zhou et al. (2022) trains
    # the inner loop (``train_to_converge`` uses the entire coreset as a single
    # batch and anneals the learning rate cosinely from ``lr`` to 0).
    full_batch: bool = False


def train_on_coreset(bundle: DatasetBundle, indices: torch.Tensor,
                     model: nn.Module, cfg: InnerLoopConfig,
                     device: torch.device, seed: Optional[int] = None
                     ) -> nn.Module:
    """Train ``model`` on the coreset ``indices`` for ``cfg.epochs`` epochs."""
    if seed is not None:
        set_seed(seed)
    model = model.to(device)
    # the dataset is kept on the CPU; only batches are moved to the device
    indices = torch.as_tensor(indices).detach().cpu().long().view(-1)
    batch_size = (max(int(indices.numel()), 1) if cfg.full_batch
                  else cfg.batch_size)
    loader = make_loader(bundle.train_x[indices], bundle.train_y[indices],
                         batch_size, shuffle=True)
    optimizer = make_optimizer(model, cfg.optimizer, cfg.lr, cfg.momentum,
                               cfg.weight_decay)
    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.epochs, 1))
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    for _ in range(cfg.epochs):
        train_one_epoch(model, loader, optimizer, device,
                        max_steps=cfg.max_steps_per_epoch,
                        grad_clip=cfg.grad_clip)
        if scheduler is not None:
            scheduler.step()
    return model


class InnerLoopTrainer:
    """Stateful inner-loop solver with warm-start support.

    The trainer keeps the last trained parameter vector so that subsequent
    mask evaluations can finetune it (Section 3.2 acceleration trick), while
    still being able to re-initialise from scratch on demand.
    """

    def __init__(self, model_factory: Callable[[], nn.Module],
                 cfg: InnerLoopConfig, device: torch.device):
        self.model_factory = model_factory
        self.cfg = cfg
        self.device = device
        self._state: Optional[dict] = None

    # -- state handling ----------------------------------------------------
    def reset(self) -> None:
        self._state = None

    def _fresh_model(self) -> nn.Module:
        model = self.model_factory().to(self.device)
        if self._state is not None and self.cfg.warm_start:
            model.load_state_dict(self._state)
        return model

    def _store(self, model: nn.Module) -> None:
        self._state = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    # -- main entry point --------------------------------------------------
    def __call__(self, bundle: DatasetBundle, indices: torch.Tensor,
                 warm_start: Optional[bool] = None,
                 epochs: Optional[int] = None) -> nn.Module:
        warm = self.cfg.warm_start if warm_start is None else warm_start
        saved = self.cfg.warm_start
        if not warm:
            self.cfg.warm_start = False
            self._state = None
        try:
            model = self._fresh_model()
            cfg = self.cfg
            if epochs is not None and epochs != cfg.epochs:
                from dataclasses import replace
                cfg = replace(cfg, epochs=epochs)
            model = train_on_coreset(bundle, indices, model, cfg, self.device)
            self._store(model)
            return model
        finally:
            self.cfg.warm_start = saved

    def pretrain(self, bundle: DatasetBundle, epochs: Optional[int] = None
                 ) -> nn.Module:
        """Train on a random coreset to obtain a warm-start initialisation."""
        n = bundle.n
        k = max(1, n // 2)
        perm = torch.randperm(n)[:k]
        return self(bundle, perm, warm_start=False, epochs=epochs)
