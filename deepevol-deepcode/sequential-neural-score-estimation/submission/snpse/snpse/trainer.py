"""Shared training loop for SNPSE / NPSE / NLSE score networks.

Implements the training protocol of the paper (Section 5.1 and Appendix E.3.2):

* Adam optimiser (Kingma & Ba, 2015) with learning rate ``1e-4`` (PyTorch Adam
  defaults for the remaining hyper-parameters, as the paper leaves them
  unspecified).
* A maximum of ``3000`` training iterations, for both sequential and
  non-sequential experiments.
* ``15 %`` of the data is held back as a validation set; the loss is evaluated
  on those samples after every training step.  If the validation loss does not
  decrease for ``1000`` steps, training stops and the network that achieved the
  lowest validation loss is returned.
* Batch size: ``50`` (non-sequential) / ``200`` (sequential) for simulation
  budgets of 1,000 or 10,000; ``500`` for both when the budget is 100,000.

The loop is deliberately agnostic with respect to the training objective: the
caller supplies a *loss closure* ``loss_fn(batch) -> scalar tensor`` where
``batch`` is a dict of tensors (``{"theta": (n, d), "x": (n, p)}`` plus any
extra keys such as per-sample ``weight`` used by SNPSE-B).  This keeps the
trainer usable for every objective in ``losses.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "TrainConfig",
    "TrainHistory",
    "EarlyStopping",
    "select_batch_size",
    "split_dataset",
    "Trainer",
    "train_network",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyper-parameters of the shared training loop (Section 5.1, E.3.2).

    Defaults follow the paper; the values the paper leaves unspecified fall
    back to the PyTorch defaults (Adam betas, no gradient clipping, ...).
    """

    lr: float = 1e-4
    max_iters: int = 3000
    batch_size: int = 50
    val_fraction: float = 0.15
    patience: int = 1000
    #: relative/absolute improvement required to reset the patience counter
    min_delta: float = 0.0
    #: Adam hyper-parameters (paper only specifies the learning rate)
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    #: optional gradient clipping (None = disabled, PyTorch default)
    grad_clip: Optional[float] = None
    #: number of datapoints used to evaluate the validation loss (None = all)
    val_batch_size: Optional[int] = 512
    #: logging / bookkeeping
    log_every: int = 100
    verbose: bool = False
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TrainHistory:
    """Bookkeeping of a :meth:`Trainer.fit` run."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_iter: int = -1
    n_iters: int = 0
    stopped_early: bool = False
    val_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    train_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "best_val_loss": float(self.best_val_loss),
            "best_iter": int(self.best_iter),
            "n_iters": int(self.n_iters),
            "stopped_early": bool(self.stopped_early),
            "n_train": int(self.train_indices.numel()),
            "n_val": int(self.val_indices.numel()),
        }


# ---------------------------------------------------------------------------
# Batch-size schedule of the paper
# ---------------------------------------------------------------------------
_DEFAULT_BATCH_TABLE = {
    # (budget, sequential) -> batch size
    (1000, False): 50,
    (1000, True): 200,
    (10000, False): 50,
    (10000, True): 200,
    (100000, False): 500,
    (100000, True): 500,
}


def select_batch_size(budget: int, sequential: bool = False) -> int:
    """Return the batch size prescribed by the paper for a simulation budget.

    Budgets of 1000/10000 use batch 50 (non-sequential) or 200 (sequential);
    a budget of 100000 uses 500 for both.  Unknown budgets are mapped onto the
    nearest tabulated budget.
    """
    key = (int(budget), bool(sequential))
    if key in _DEFAULT_BATCH_TABLE:
        return _DEFAULT_BATCH_TABLE[key]
    known = sorted({b for b, _ in _DEFAULT_BATCH_TABLE})
    nearest = min(known, key=lambda b: abs(b - int(budget)))
    return _DEFAULT_BATCH_TABLE[(nearest, bool(sequential))]


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stop when the validation loss has not improved for ``patience`` steps.

    Mirrors Appendix E.3.2: "if this loss does not decrease for 1000 steps then
    we stop training and return the network which gave the lowest validation
    loss".
    """

    def __init__(self, patience: int = 1000, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_iter = -1
        self.bad_steps = 0

    def step(self, value: float, iteration: int = 0) -> bool:
        """Update the state; return ``True`` when training should stop."""
        value = float(value)
        if value < self.best - self.min_delta:
            self.best = value
            self.best_iter = int(iteration)
            self.bad_steps = 0
            return False
        self.bad_steps += 1
        return self.bad_steps >= self.patience


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
def split_dataset(
    theta: torch.Tensor,
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    val_fraction: float = 0.15,
    generator: Optional[torch.Generator] = None,
    shuffle: bool = True,
) -> Dict[str, torch.Tensor]:
    """Hold back ``val_fraction`` of the data as a validation set.

    Returns a dict with the keys ``"theta"``, ``"x"`` (and ``"weight"`` when
    provided) for both the ``"train"`` and ``"val"`` splits, plus the
    corresponding ``"train_idx"`` / ``"val_idx"`` index tensors.
    """
    n = int(theta.shape[0])
    if weights is not None and int(weights.shape[0]) != n:
        raise ValueError("theta and weight must have matching leading dimension")
    idx = torch.arange(n)
    if shuffle:
        idx = idx[torch.randperm(n, generator=generator)]
    n_val = int(round(val_fraction * n))
    n_val = max(min(n_val, n - 1), 0) if n > 1 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    out: Dict[str, torch.Tensor] = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "theta_train": theta[train_idx],
        "x_train": x[train_idx],
        "theta_val": theta[val_idx],
        "x_val": x[val_idx],
    }
    if weights is not None:
        out["weight_train"] = weights[train_idx]
        out["weight_val"] = weights[val_idx]
    return out


def _take(tensor: Optional[torch.Tensor], index: torch.Tensor) -> Optional[torch.Tensor]:
    return None if tensor is None else tensor[index]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Minimal, objective-agnostic trainer implementing the paper protocol.

    Parameters
    ----------
    network:
        The ``nn.Module`` to optimise (the score network ``s_psi``).
    loss_fn:
        Callable mapping a batch dict to a scalar loss tensor.  The batch dict
        contains at least ``"theta"`` (``(n, d)``) and ``"x"`` (``(n, p)``);
        extra tensors are propagated (e.g. per-sample ``"weight"``).
    config:
        :class:`TrainConfig` with the optimiser / early-stopping settings.
    device:
        Torch device; defaults to the device of ``network``'s parameters.
    """

    def __init__(
        self,
        network: nn.Module,
        loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        config: Optional[TrainConfig] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.config = config if config is not None else TrainConfig()
        self.network = network
        if device is None:
            try:
                device = next(network.parameters()).device
            except StopIteration:  # pragma: no cover - network without params
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.network.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        self.early_stopping = EarlyStopping(
            patience=self.config.patience, min_delta=self.config.min_delta
        )
        self.history = TrainHistory()

    # -- helpers ----------------------------------------------------------
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            elif isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
                moved[key] = [v.to(self.device) for v in value]
            else:
                moved[key] = value
        return moved

    def _batches(self, n: int, batch_size: int, generator: Optional[torch.Generator]):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            yield perm[start : start + batch_size]

    def _eval_batch(
        self, theta: torch.Tensor, x: torch.Tensor, weight: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {"theta": theta, "x": x}
        if weight is not None:
            batch["weight"] = weight
        return batch

    # -- main loop --------------------------------------------------------
    def fit(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        val_fraction: Optional[float] = None,
        split: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> TrainHistory:
        """Train ``self.network`` on ``(theta, x)`` and return the history.

        ``split`` may be supplied (as produced by :func:`split_dataset`) to keep
        the same train/validation partition across sequential rounds.
        """
        cfg = self.config
        if generator is None and cfg.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        val_fraction = cfg.val_fraction if val_fraction is None else val_fraction

        if split is None:
            split = split_dataset(
                theta, x, weights=weights, val_fraction=val_fraction, generator=generator
            )

        theta_tr = split["theta_train"].to(self.device)
        x_tr = split["x_train"].to(self.device)
        w_tr = split.get("weight_train")
        w_tr = None if w_tr is None else w_tr.to(self.device)
        theta_val = split["theta_val"].to(self.device)
        x_val = split["x_val"].to(self.device)
        w_val = split.get("weight_val")
        w_val = None if w_val is None else w_val.to(self.device)

        self.history.train_indices = split["train_idx"].detach().cpu()
        self.history.val_indices = split["val_idx"].detach().cpu()

        n_train = int(theta_tr.shape[0])
        if n_train == 0:
            raise ValueError("empty training split")

        # Fixed validation subset (deterministic comparison across steps).
        if theta_val.shape[0] > 0:
            vb = cfg.val_batch_size
            if vb is not None and theta_val.shape[0] > vb:
                vperm = torch.randperm(theta_val.shape[0], generator=generator)[:vb]
                theta_val = theta_val[vperm]
                x_val = x_val[vperm]
                if w_val is not None:
                    w_val = w_val[vperm]

        batch_size = max(1, min(int(cfg.batch_size), n_train))
        best_state = copy.deepcopy(self.network.state_dict())
        best_val = float("inf")

        for it in range(int(cfg.max_iters)):
            self.network.train()
            for batch_idx in self._batches(n_train, batch_size, generator):
                batch = self._eval_batch(
                    theta_tr[batch_idx],
                    x_tr[batch_idx],
                    None if w_tr is None else w_tr[batch_idx],
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(self._to_device(batch))
                loss = loss if torch.is_tensor(loss) else torch.as_tensor(loss)
                loss = loss.to(self.device)
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.grad_clip)
                self.optimizer.step()

            # --- validation loss (Appendix E.3.2) ---
            if theta_val.shape[0] > 0:
                self.network.eval()
                with torch.no_grad():
                    val_loss = self.loss_fn(
                        self._to_device(self._eval_batch(theta_val, x_val, w_val))
                    )
                    val_loss = float(
                        val_loss.detach().mean().cpu()
                        if torch.is_tensor(val_loss)
                        else val_loss
                    )
            else:  # no validation data: fall back to the training loss
                val_loss = float(loss.detach().mean().cpu())

            self.history.train_loss.append(float(loss.detach().mean().cpu()))
            self.history.val_loss.append(val_loss)
            self.history.n_iters = it + 1

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.network.state_dict())
                self.history.best_val_loss = best_val
                self.history.best_iter = it

            if cfg.verbose and (cfg.log_every > 0) and (
                it % cfg.log_every == 0 or it == int(cfg.max_iters) - 1
            ):
                print(
                    f"[trainer] iter {it + 1:5d}  train {self.history.train_loss[-1]:.6f}"
                    f"  val {val_loss:.6f}"
                )

            if self.early_stopping.step(val_loss, it):
                self.history.stopped_early = True
                if cfg.verbose:
                    print(
                        f"[trainer] early stopping at iter {it + 1} "
                        f"(no val improvement for {cfg.patience} steps)"
                    )
                break

        # restore the network with the lowest validation loss
        self.network.load_state_dict(best_state)
        self.network.eval()
        return self.history


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def train_network(
    network: nn.Module,
    loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    theta: torch.Tensor,
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    config: Optional[TrainConfig] = None,
    device: Optional[torch.device | str] = None,
    split: Optional[Dict[str, torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    return_history: bool = False,
):
    """Train ``network`` with the shared loop and return *it* (and the history).

    Parameters mirror :meth:`Trainer.fit`.  When ``return_history`` is true a
    ``(network, history)`` tuple is returned.
    """
    trainer = Trainer(network, loss_fn, config=config, device=device)
    history = trainer.fit(
        theta, x, weights=weights, split=split, generator=generator
    )
    if return_history:
        return trainer.network, history
    return trainer.network
