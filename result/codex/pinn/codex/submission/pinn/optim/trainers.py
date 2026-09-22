"""Training loops for Adam, L-BFGS and Adam+L-BFGS (Sections 2.2, 6 and 7)."""

from __future__ import annotations

import time
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..data import DataSet
from ..hessian.lbfgs_precond import LBFGSHistory
from ..metrics import l2_relative_error
from ..problems import Problem
from .lbfgs import LBFGSOptimizer
from .objective import Objective


@dataclass
class TrainConfig:
    """Everything needed to reproduce one run of Section 6."""

    optimizer: str = "adam_lbfgs"  # adam | lbfgs | adam_lbfgs
    lr: float = 1e-3
    switch_iter: int = 11000  # Adam -> L-BFGS switch (1000, 11000, 31000)
    iters: int = 41000  # total number of iterations
    lbfgs_history_size: int = 100
    lbfgs_lr: float = 1.0
    log_every: int = 100
    aggregation: str = "combined"
    # iterations at which a copy of the parameters is kept (Figure 5 needs the
    # solution right after the Adam phase of Adam+L-BFGS)
    snapshot_iters: Sequence[int] = ()


@dataclass
class TrainResult:
    config: TrainConfig
    seed: int
    width: int
    final_loss: float
    final_l2re: float
    final_grad_norm: float
    iterations: int
    wall_clock: float
    trace: Dict[str, List[float]] = field(default_factory=dict)
    lbfgs_history: Optional[LBFGSHistory] = None
    lbfgs_steps: int = 0
    adam_steps: int = 0
    lbfgs_lr_history: Optional[List[float]] = None
    snapshots: Dict[int, dict] = field(default_factory=dict)

    @property
    def seconds_per_iteration(self) -> float:
        return self.wall_clock / max(self.iterations, 1)


def _make_objective(problem, net, ds, aggregation) -> Objective:
    return Objective(problem, net, ds, aggregation=aggregation)


def train(
    problem: Problem,
    net: nn.Module,
    ds: DataSet,
    config: TrainConfig,
    seed: int = 0,
    l2re_every: Optional[int] = None,
    progress: bool = False,
) -> TrainResult:
    """Train ``net`` with the requested optimizer.

    The three optimizers of Section 6 are

    ``adam``         Adam with the tuned learning rate for ``iters`` steps,
    ``lbfgs``        L-BFGS (lr 1.0, memory 100, strong Wolfe) for ``iters`` steps,
    ``adam_lbfgs``   Adam for ``switch_iter`` steps, then L-BFGS until ``iters``.
    """
    obj = _make_objective(problem, net, ds, config.aggregation)
    l2re_every = config.log_every if l2re_every is None else l2re_every

    trace: Dict[str, List[float]] = {
        "iteration": [],
        "loss": [],
        "grad_norm": [],
        "l2re": [],
        "l2re_iteration": [],
        "adam_loss": [],
        "adam_iteration": [],
        "adam_grad_norm": [],
    }
    t0 = time.time()

    def record(it: int, loss: float, gnorm: float) -> None:
        trace["iteration"].append(it)
        trace["loss"].append(loss)
        trace["grad_norm"].append(gnorm)
        if it % l2re_every == 0 or it == config.iters:
            trace["l2re_iteration"].append(it)
            trace["l2re"].append(l2_relative_error(net, problem, ds.X_eval))

    adam_steps = 0
    lbfgs_steps = 0
    lr_history: List[float] = []
    lbfgs_history: Optional[LBFGSHistory] = None
    snapshots: Dict[int, dict] = {}

    def maybe_snapshot(it: int) -> None:
        if it + 1 in set(config.snapshot_iters):
            snapshots[it + 1] = copy.deepcopy({k: v.clone() for k, v in net.state_dict().items()})

    if config.optimizer in ("adam", "adam_lbfgs"):
        adam = torch.optim.Adam(net.parameters(), lr=config.lr)
        n_adam = config.iters if config.optimizer == "adam" else config.switch_iter
        for it in range(n_adam):
            adam.zero_grad(set_to_none=True)
            loss = obj.loss()
            loss.backward()
            adam.step()
            adam_steps += 1
            with torch.no_grad():
                gnorm = float(
                    torch.linalg.norm(
                        torch.cat([p.grad.reshape(-1) for p in obj.params if p.grad is not None])
                    )
                )
            trace["adam_iteration"].append(it)
            trace["adam_loss"].append(float(loss.detach()))
            trace["adam_grad_norm"].append(gnorm)
            record(it, float(loss.detach()), gnorm)
            maybe_snapshot(it)
            if progress and (it + 1) % max(l2re_every * 10, 1) == 0:
                print(f"    adam {it + 1}/{n_adam} loss={float(loss.detach()):.3e}")

    if config.optimizer in ("lbfgs", "adam_lbfgs"):
        if config.optimizer == "lbfgs":
            n_lbfgs = config.iters
        else:
            n_lbfgs = max(config.iters - config.switch_iter, 0)
        if n_lbfgs > 0:
            base = 0 if config.optimizer == "lbfgs" else config.switch_iter
            opt = LBFGSOptimizer(
                obj, lr=config.lbfgs_lr, history_size=config.lbfgs_history_size
            )
            for it in range(n_lbfgs):
                loss = opt.step()
                lbfgs_steps += 1
                gnorm = float(torch.linalg.norm(obj.grad()))
                lr_history.append(opt.last_step_size())
                record(base + it, loss, gnorm)
                maybe_snapshot(base + it)
                if progress and (it + 1) % max(l2re_every * 10, 1) == 0:
                    print(f"    lbfgs {it + 1}/{n_lbfgs} loss={loss:.3e}")
            lbl = opt.state
            if len(lbl.get("old_dirs", [])) > 0:
                lbfgs_history = opt.history()

    wall = time.time() - t0
    res = TrainResult(
        config=config,
        seed=seed,
        width=net.width,
        final_loss=obj.evaluate(),
        final_l2re=l2_relative_error(net, problem, ds.X_eval),
        final_grad_norm=float(torch.linalg.norm(obj.grad())),
        iterations=adam_steps + lbfgs_steps,
        wall_clock=wall,
        trace=trace,
        lbfgs_history=lbfgs_history,
        lbfgs_steps=lbfgs_steps,
        adam_steps=adam_steps,
        lbfgs_lr_history=lr_history,
        snapshots=snapshots,
    )
    return res


def adam_lbfgs_train(problem, net, ds, config: TrainConfig, seed: int = 0, **kwargs) -> TrainResult:
    config = TrainConfig(**{**config.__dict__, "optimizer": "adam_lbfgs"})
    return train(problem, net, ds, config, seed=seed, **kwargs)
