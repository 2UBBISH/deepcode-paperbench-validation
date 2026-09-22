"""Shared configuration and reporting helpers for the experiments."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from lbcs.data import DatasetBundle, load_dataset
from lbcs.inner_loop import InnerLoopConfig
from lbcs.models import build_model
from lbcs.objectives import ObjectiveConfig
from lbcs.utils import resolve_device, set_seed

DEFAULT_RESULTS_DIR = os.environ.get("LBCS_RESULTS_DIR",
                                     os.path.join(os.path.dirname(
                                         os.path.dirname(os.path.abspath(__file__))),
                                         "results"))


# ---------------------------------------------------------------------------
# Dataset / model specification
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    """Which networks the paper uses for a dataset (Section 5.2 / Table 7)."""

    proxy_model: str               # network used inside coreset selection
    target_model: str              # network trained on the constructed coreset
    inner_epochs: int = 100
    inner_optimizer: str = "adam"
    inner_lr: float = 1e-3
    target_epochs: int = 100
    target_optimizer: str = "adam"
    target_lr: float = 1e-3
    target_scheduler: Optional[str] = None
    target_weight_decay: float = 0.0
    batch_size: int = 128
    inner_full_batch: bool = False
    inner_scheduler: Optional[str] = None


SPECS: Dict[str, ExperimentSpec] = {
    # Section 5.1 / Figure 1: ConvNet (Zhou et al., 2022) on MNIST-S.
    # Appendix C.3 states "for the inner loop, the model is trained for 100
    # epochs using SGD with a learning rate of 0.1 and momentum of 0.9"; the
    # reference implementation of Zhou et al. (2022) realises this with
    # full-batch steps and a cosine-annealed learning rate (100 -> 0), which is
    # what ``full_batch`` + ``scheduler='cosine'`` reproduce here.  The
    # learning rate is annealed to zero, so the run is stable.
    "mnist-s": ExperimentSpec(proxy_model="convnet", target_model="convnet",
                              inner_epochs=100, inner_optimizer="sgd",
                              inner_lr=0.1, target_epochs=100,
                              target_optimizer="sgd", target_lr=0.05,
                              target_scheduler="cosine",
                              inner_full_batch=True, inner_scheduler="cosine"),
    "mnist": ExperimentSpec(proxy_model="convnet", target_model="convnet",
                            inner_epochs=100, inner_optimizer="sgd",
                            inner_lr=0.1, target_epochs=100,
                            target_optimizer="sgd", target_lr=0.05,
                            target_scheduler="cosine",
                            inner_full_batch=True, inner_scheduler="cosine"),
    # Section 5.2: LeNet proxy and LeNet target for F-MNIST
    "fmnist": ExperimentSpec(proxy_model="lenet", target_model="lenet",
                             inner_epochs=100, inner_optimizer="adam",
                             inner_lr=1e-3, target_epochs=100,
                             target_optimizer="adam", target_lr=1e-3),
    # Section 5.2: simple CNN proxy, CNN target for SVHN
    "svhn": ExperimentSpec(proxy_model="svhn_cnn_inner",
                           target_model="svhn_cnn_target", inner_epochs=100,
                           inner_optimizer="adam", inner_lr=1e-3,
                           target_epochs=100, target_optimizer="adam",
                           target_lr=1e-3),
    # Section 5.2: simple CNN proxy, ResNet-18 target for CIFAR-10
    "cifar10": ExperimentSpec(proxy_model="cifar10_cnn_inner",
                              target_model="resnet18", inner_epochs=100,
                              inner_optimizer="adam", inner_lr=1e-3,
                              target_epochs=200, target_optimizer="sgd",
                              target_lr=0.1, target_scheduler="cosine",
                              target_weight_decay=5e-4),
}


def model_factory(name: str, **overrides) -> Callable[[], torch.nn.Module]:
    def factory():
        return build_model(name, **overrides)
    return factory


def get_spec(dataset: str, target_model: Optional[str] = None,
             proxy_model: Optional[str] = None) -> ExperimentSpec:
    spec = SPECS[dataset]
    spec = ExperimentSpec(**asdict(spec))
    if target_model:
        spec.target_model = target_model
    if proxy_model:
        spec.proxy_model = proxy_model
    return spec


def load_bundle(dataset: str, addendum_style: bool = True, **kwargs
                ) -> DatasetBundle:
    kwargs.setdefault("download", True)
    return load_dataset(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)
                                                   if arr.size > 1 else 0.0),
            "n": int(arr.size)}


def format_mean_std(stats: Dict[str, float], digits: int = 1) -> str:
    if stats["n"] == 0:
        return "n/a"
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def write_rows_csv(path: str, rows: List[dict]) -> None:
    """Write a list of flat dicts to CSV without pandas."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(_csv_cell(row.get(k, "")) for k in keys) + "\n")


def _csv_cell(value) -> str:
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    if "," in text or '"' in text:
        text = '"' + text.replace('"', '""') + '"'
    return text


def markdown_table(rows: List[dict], columns: Sequence[str],
                   headers: Optional[Sequence[str]] = None) -> str:
    headers = list(headers or columns)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns)
                   + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def base_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / mps (default: auto)")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seeds", type=int, default=None,
                        help="override the number of repeats")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the number of training epochs")
    parser.add_argument("--inner-optimizer", dest="inner_optimizer",
                        default=None, help="adam / sgd (default: per dataset)")
    parser.add_argument("--inner-lr", dest="inner_lr", type=float,
                        default=None,
                        help="override the inner-loop learning rate")
    parser.add_argument("--grad-clip", dest="grad_clip", type=float,
                        default=None,
                        help="optional gradient-norm clipping in the inner loop")
    parser.add_argument("--f1-eval-size", dest="f1_eval_size", type=int,
                        default=None,
                        help="evaluate f1 on a fixed random subset of this "
                             "size instead of the full training set")
    parser.add_argument("--target-epochs", dest="target_epochs", type=int,
                        default=None,
                        help="override the number of target-training epochs")
    parser.add_argument("--dry-run", action="store_true",
                        help="tiny configuration to check the pipeline")
    return parser


def objective_cfg(args, spec, **overrides) -> ObjectiveConfig:
    """Objective configuration honouring ``--f1-eval-size``."""
    size = getattr(args, "f1_eval_size", None)
    # f_1 is defined on the *full* data set; for the large benchmarks the
    # experiments may use a fixed random subset to keep the outer loop cheap.
    cfg = ObjectiveConfig(eval_batch_size=512, f1_eval_size=size)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def target_epochs(args, spec) -> int:
    if getattr(args, "dry_run", False):
        return min(2, spec.target_epochs)
    if getattr(args, "target_epochs", None):
        return args.target_epochs
    return spec.target_epochs


def inner_cfg(spec, epochs: int, args, **overrides) -> InnerLoopConfig:
    """Inner-loop configuration, honouring the CLI overrides."""
    cfg = InnerLoopConfig(
        epochs=epochs,
        batch_size=spec.batch_size,
        optimizer=getattr(args, "inner_optimizer", None) or spec.inner_optimizer,
        lr=(getattr(args, "inner_lr", None)
            if getattr(args, "inner_lr", None) is not None else spec.inner_lr),
        momentum=0.9,
        warm_start=True,
        grad_clip=getattr(args, "grad_clip", None),
    )
    cfg.full_batch = getattr(spec, "inner_full_batch", False)
    cfg.scheduler = getattr(spec, "inner_scheduler", None)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def resolve_cli(args) -> dict:
    """Turn the parsed CLI arguments into a plain dictionary."""
    ctx: Dict[str, object] = {"results_dir": args.results_dir,
                              "device": resolve_device(args.device)}
    if args.data_root:
        ctx["data_root"] = args.data_root
    if getattr(args, "dry_run", False) and \
            getattr(args, "f1_eval_size", None) is None:
        # keep the smoke test cheap: f_1 on a 1k subset instead of the full set
        args.f1_eval_size = 1000
    return ctx


def make_bundle(dataset: str, data_root: Optional[str] = None,
                mnist_s_size: int = 1000, mnist_s_seed: int = 0
                ) -> DatasetBundle:
    kwargs = {}
    if data_root:
        kwargs["root"] = data_root
    return load_bundle(dataset, mnist_s_size=mnist_s_size,
                       mnist_s_seed=mnist_s_seed, **kwargs)


def seeds_for(repeats: int, base: int = 0) -> List[int]:
    return [base + i for i in range(repeats)]
