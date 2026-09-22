"""Shared plumbing for the experiment drivers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from ..data import build_dataset
from ..metrics import l2_relative_error
from ..models import build_model
from ..optim.objective import Objective
from ..optim.trainers import TrainConfig, train
from ..problems import build_problem

# ---------------------------------------------------------------------- #
# default search spaces (Section 2.2)
# ---------------------------------------------------------------------- #
WIDTHS: Sequence[int] = (50, 100, 200, 400)
SEEDS: Sequence[int] = (123, 234, 345, 456, 567)
ADAM_LRS: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SWITCHES: Sequence[int] = (1000, 11000, 31000)
TOTAL_ITERS: int = 41000
PDES: Sequence[str] = ("convection", "reaction", "wave")

# reduced settings used by ``--quick`` so that the pipeline can be smoke tested
QUICK = dict(widths=(20,), seeds=(0,), lrs=(1e-2,), switches=(50,), iters=200)


def optimizer_label(optimizer: str, switch: int = 0) -> str:
    if optimizer == "adam":
        return "Adam"
    if optimizer == "lbfgs":
        return "L-BFGS"
    if optimizer == "adam_lbfgs":
        return f"Adam+L-BFGS ({switch // 1000}k)"
    return optimizer


@dataclass(frozen=True)
class RunSpec:
    """One training run of the hyper-parameter grid of Section 6."""

    pde: str
    width: int
    seed: int
    optimizer: str  # adam | lbfgs | adam_lbfgs
    lr: float
    switch: int = 11000
    iters: int = TOTAL_ITERS

    @property
    def run_id(self) -> str:
        sw = f"_sw{self.switch}" if self.optimizer == "adam_lbfgs" else ""
        return f"{self.pde}_w{self.width}_s{self.seed}_{self.optimizer}{sw}_lr{self.lr:g}_it{self.iters}"

    @property
    def label(self) -> str:
        return optimizer_label(self.optimizer, self.switch)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id
        d["label"] = self.label
        return d


@dataclass
class Paths:
    root: str = "runs"

    @property
    def checkpoints(self) -> str:
        return os.path.join(self.root, "checkpoints")

    @property
    def traces(self) -> str:
        return os.path.join(self.root, "traces")

    @property
    def figures(self) -> str:
        return os.path.join(self.root, "figures")

    @property
    def tables(self) -> str:
        return os.path.join(self.root, "tables")

    @property
    def runs_jsonl(self) -> str:
        return os.path.join(self.root, "runs.jsonl")

    def makedirs(self) -> None:
        for d in (self.root, self.checkpoints, self.traces, self.figures, self.tables):
            os.makedirs(d, exist_ok=True)

    def checkpoint(self, run_id: str) -> str:
        return os.path.join(self.checkpoints, f"{run_id}.pt")

    def trace(self, run_id: str) -> str:
        return os.path.join(self.traces, f"{run_id}.json")

    def figure(self, name: str) -> str:
        return os.path.join(self.figures, name)

    def table(self, name: str) -> str:
        return os.path.join(self.tables, name)


def make_problem_and_data(spec: RunSpec):
    problem = build_problem(spec.pde)
    ds = build_dataset(problem, seed=spec.seed)
    return problem, ds


def make_model(spec: RunSpec):
    return build_model(width=spec.width, n_layers=3, seed=spec.seed)


def run_training(
    spec: RunSpec,
    outdir: Paths,
    save: bool = True,
    log_every: int = 100,
    progress: bool = False,
    aggregation: str = "combined",
    skip_existing: bool = True,
) -> dict:
    """Train one model and (optionally) persist the result.

    With ``skip_existing`` a run whose checkpoint and record already exist is
    not repeated, so an interrupted grid search can simply be restarted.
    """
    if save and skip_existing:
        for record in read_records(outdir):
            if record.get("run_id") == spec.run_id and os.path.exists(outdir.checkpoint(spec.run_id)):
                if progress:
                    print(f"    cached {spec.run_id}")
                return record
    problem, ds = make_problem_and_data(spec)
    net = make_model(spec)
    cfg = TrainConfig(
        optimizer=spec.optimizer,
        lr=spec.lr,
        switch_iter=spec.switch,
        iters=spec.iters,
        log_every=log_every,
        aggregation=aggregation,
    )
    t0 = time.time()
    res = train(problem, net, ds, cfg, seed=spec.seed, progress=progress)
    record = spec.to_dict()
    record.update(
        {
            "final_loss": res.final_loss,
            "final_l2re": res.final_l2re,
            "final_grad_norm": res.final_grad_norm,
            "wall_clock": res.wall_clock,
            "seconds_per_iteration": res.seconds_per_iteration,
            "adam_steps": res.adam_steps,
            "lbfgs_steps": res.lbfgs_steps,
            "n_parameters": net.n_parameters(),
            "aggregation": aggregation,
        }
    )
    if save:
        outdir.makedirs()
        torch.save(net.state_dict(), outdir.checkpoint(spec.run_id))
        with open(outdir.trace(spec.run_id), "w") as fh:
            json.dump(res.trace, fh)
        with open(outdir.runs_jsonl, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return record


def load_trained(spec: RunSpec, outdir: Paths):
    """Rebuild the problem/data/model of a stored run."""
    problem, ds = make_problem_and_data(spec)
    net = make_model(spec)
    path = outdir.checkpoint(spec.run_id)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(True)
    return problem, ds, net


def read_records(outdir: Paths) -> List[dict]:
    path = outdir.runs_jsonl
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
