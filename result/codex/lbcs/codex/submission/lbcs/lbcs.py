"""The proposed method: Lexicographic Bilevel Coreset Selection (Algorithm 1).

    1: Require: a network, a dataset D, a predefined size k, and voluntary
       performance compromise eps
    2: Initialise masks m randomly with |m|_0 = k
    3: for training iteration t = 1, 2, ..., T do
    4:     Train the inner loop with D to converge:
               theta(m) <- arg min_theta L(m, theta)
    5:     Update masks m with theta(m) by lexicographic optimisation
    6: Output: masks m after all training iterations

The lexicographic optimisation of step 5 is the black-box optimiser of
:mod:`lbcs.lexiflow`; every mask query inside it solves the inner loop of
step 4 through :class:`lbcs.objectives.BilevelObjective`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .data import DatasetBundle
from .inner_loop import InnerLoopConfig, InnerLoopTrainer
from .lexiflow import LexiFlow, LexiFlowConfig
from .objectives import BilevelObjective, ObjectiveConfig
from .utils import (accuracy_on_tensor, discretize, evaluate_accuracy,
                    evaluate_loss, make_loader, set_seed)


def random_mask(n: int, k: int, seed: Optional[int] = None,
                device=None, generator: Optional[torch.Generator] = None
                ) -> torch.Tensor:
    """A random ``{0, 1}`` mask of ``n`` entries with exactly ``k`` ones."""
    if generator is None:
        generator = torch.Generator().manual_seed(0 if seed is None else seed)
    if k > n:
        raise ValueError(f"k={k} exceeds the number of examples n={n}")
    perm = torch.randperm(n, generator=generator)[:k]
    mask = torch.zeros(n, dtype=torch.float32)
    mask[perm] = 1.0
    return mask.to(device) if device is not None else mask


def mask_to_continuous(mask: torch.Tensor) -> torch.Tensor:
    """``{0, 1}`` -> ``{-1, +1}`` as used by the continuous search variable."""
    return 2.0 * (mask > 0.5).to(torch.float32) - 1.0


@dataclass
class LBCSConfig:
    """Everything needed to run Algorithm 1."""

    k: int = 1000
    epsilon: float = 0.2
    T: int = 500
    group_size: int = 1
    seed: int = 0
    inner: InnerLoopConfig = field(default_factory=InnerLoopConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    lexiflow: LexiFlowConfig = field(default_factory=LexiFlowConfig)
    # optional mask initialisation, e.g. by "Moderate" (Table 5)
    init_indices: Optional[torch.Tensor] = None
    init_label: str = "random"
    verbose: bool = False


class LBCS:
    """Lexicographic bilevel coreset selection."""

    def __init__(self, bundle: DatasetBundle,
                 model_factory: Callable[[], nn.Module],
                 device: torch.device, config: Optional[LBCSConfig] = None):
        self.bundle = bundle
        self.model_factory = model_factory
        self.device = device
        self.cfg = config or LBCSConfig()
        self._reset()

    def _reset(self) -> None:
        self.trainer = InnerLoopTrainer(self.model_factory, self.cfg.inner,
                                        self.device)
        self.objective = BilevelObjective(self.bundle, self.trainer,
                                          self.device, self.cfg.objective,
                                          group_size=self.cfg.group_size)
        self._lexiflow_cfg = copy.deepcopy(self.cfg.lexiflow)
        self._lexiflow_cfg.epsilon = self.cfg.epsilon

    # ------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------
    def initial_mask(self) -> torch.Tensor:
        """Line 2 of Algorithm 1: a random mask with exactly ``k`` ones."""
        n = self.bundle.n
        if self.cfg.init_indices is not None:
            idx = torch.as_tensor(self.cfg.init_indices, dtype=torch.long)
            m = torch.zeros(n, dtype=torch.float32)
            m[idx] = 1.0
            return m
        return random_mask(n, self.cfg.k, seed=self.cfg.seed)

    # ------------------------------------------------------------------
    # Algorithm 1
    # ------------------------------------------------------------------
    def select(self, report_every: int = 0) -> Dict[str, object]:
        cfg = self.cfg
        set_seed(cfg.seed)

        # ---- line 2: initialisation ---------------------------------
        init_mask = self.initial_mask()
        continuous0 = mask_to_continuous(init_mask).to(self.device)
        init_value = self.objective.evaluate(continuous0, warm_start=False)

        # ---- lines 3-5: inner loop + lexicographic outer loop --------
        search = LexiFlow(self.objective.evaluate, self._lexiflow_cfg)
        best_continuous = search.optimize(continuous0, max_steps=cfg.T)

        best_mask = discretize(best_continuous)
        best_value = self.objective.evaluate(best_continuous)

        info = {
            "init_mask": init_mask,
            "init_objectives": init_value,
            "mask": best_mask,
            "objectives": best_value,
            "f1": float(best_value[0]),
            "f2": float(best_value[1]),
            "coreset_size": int(best_mask.sum().item()),
            "num_queries": len(self.objective.history),
            "num_trainings": self.objective.num_trainings,
            "history": self.objective.history,
            "init_label": cfg.init_label,
            "epsilon": cfg.epsilon,
            "k": cfg.k,
        }
        if report_every:
            print(f"[LBCS] eps={cfg.epsilon} k={cfg.k} "
                  f"f1={info['f1']:.4f} f2={info['f2']:.0f} "
                  f"(init f1={float(init_value[0]):.4f} "
                  f"f2={float(init_value[1]):.0f})")
        return info


# ----------------------------------------------------------------------
# Post-selection evaluation: train a target model on the constructed coreset
# ----------------------------------------------------------------------
@dataclass
class TargetTrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    optimizer: str = "adam"
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: Optional[str] = None      # None | "cosine"
    warmup_epochs: int = 0


def train_target_model(bundle: DatasetBundle, indices: torch.Tensor,
                       model_factory: Callable[[], nn.Module],
                       device: torch.device, cfg: TargetTrainingConfig,
                       seed: int = 0) -> nn.Module:
    """Train the (target) network on the constructed coreset."""
    from .inner_loop import train_on_coreset, InnerLoopConfig

    inner = InnerLoopConfig(epochs=cfg.epochs, batch_size=cfg.batch_size,
                            optimizer=cfg.optimizer, lr=cfg.lr,
                            momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay,
                            scheduler=cfg.scheduler,
                            warm_start=False)
    set_seed(seed)
    model = model_factory()
    return train_on_coreset(bundle, indices, model, inner, device, seed=seed)


def evaluate_coreset(bundle: DatasetBundle, mask: torch.Tensor,
                     model_factory: Callable[[], nn.Module],
                     device: torch.device,
                     cfg: Optional[TargetTrainingConfig] = None,
                     seed: int = 0, batch_size: int = 512
                     ) -> Dict[str, float]:
    """Train on the coreset given by ``mask`` and report test accuracy.

    ``mask`` is a *discrete* ``{0, 1}`` mask (as returned by
    :meth:`LBCS.select`), i.e. ``1`` marks a selected example.  Note that this
    is different from the continuous search variable used inside LexiFlow,
    which lives in ``[-1, 1]`` and is turned into a ``{0, 1}`` mask by
    :func:`lbcs.utils.discretize`.
    """
    cfg = cfg or TargetTrainingConfig()
    indices = torch.nonzero(mask.reshape(-1) > 0.5,
                            as_tuple=False).flatten()
    model = train_target_model(bundle, indices, model_factory, device, cfg,
                               seed=seed)
    acc = accuracy_on_tensor(model, bundle.test_x, bundle.test_y, device,
                             batch_size=batch_size)
    return {
        "test_accuracy": acc,
        "coreset_size": int(indices.numel()),
        "accuracy_per_datapoint": acc / max(int(indices.numel()), 1) * 1000.0,
    }
