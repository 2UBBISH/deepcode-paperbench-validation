"""Mask Tuning baseline -- the ``LoRA+Prune`` row of Table 2.

Follows Kwon et al., *A Fast Post-Training Pruning Framework for Transformers*
(https://github.com/WoosukKwon/retraining-free-pruning), adapted -- as the task
addendum requires -- so that it can be applied to a **LoRA-tuned** model.

Three stages, mirroring the original framework:

1. **Importance estimation.**  Per-weight Fisher information
   ``F_ij = E[(dL/dW_ij)^2]`` accumulated over a calibration set.  Unit
   importance (an MHA head, an FFN neuron) is the sum over the unit's weights
   and the score used for allocation is the importance *per parameter*.
2. **Mask search.**  Iteratively remove the unit with the smallest
   importance-per-parameter until the parameter budget implied by the target
   sparsity is met (the paper's "mask search" step).
3. **Mask tuning.**  Add continuous mask variables to every unit and tune
   *only those* for a few epochs on the task data (weights frozen), then
   hard-threshold and physically prune the model.

The APT block table (:mod:`apt.wrap`) is re-used to enumerate the units, which
is exactly what makes the baseline applicable to a LoRA-tuned model: the
importance is measured on the *merged* weight ``W + s W_B W_A``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from apt.blocks import HEAD, NEURON
from apt.trainer import resolve_device
from apt.wrap import WrapConfig, salience_plan, wrap_model

from .common import LogWriter, batches, measure_inference, train_supervised


# --------------------------------------------------------------------------- #
# Importance estimation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _unit_parameter_indices(topo) -> Dict[int, List[Tuple[str, str, int, int]]]:
    """``block index -> list of (linear, dim, start, size)`` physical slices."""
    out: Dict[int, List[Tuple[str, str, int, int]]] = {}
    for bi, block in enumerate(topo.blocks):
        if block.kind not in (HEAD, NEURON):
            continue
        out[bi] = [(sl.linear, sl.dim, sl.start, sl.size) for sl in block.slices]
    return out


def estimate_fisher(
    model,
    topo,
    task,
    features,
    device,
    batch_size: int = 16,
    max_batches: int = 20,
) -> torch.Tensor:
    """Accumulate ``E[(dL/dW)^2]`` for every wrapped linear.

    Returns a flat tensor with one entry per *unit* (head / neuron), holding the
    summed Fisher information over the unit's weights.
    """
    fisher: Dict[str, torch.Tensor] = {}
    for name, lin in topo.linears.items():
        fisher[name] = torch.zeros(lin.out_features, lin.in_features, dtype=torch.float64)
        lin.base.weight.requires_grad_(True)

    model.eval()
    seen = 0
    for i, batch in enumerate(batches(features, batch_size, False, device, task.collate)):
        if i >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        out = task.forward(model, batch, output_hidden_states=False)
        out["loss"].backward()
        with torch.no_grad():
            for name, lin in topo.linears.items():
                g = lin.base.weight.grad
                if g is not None:
                    fisher[name] += g.detach().double().pow(2).cpu()
        seen += 1
    for lin in topo.linears.values():
        lin.base.weight.grad = None
        lin.base.weight.requires_grad_(False)

    unit_idx = _unit_parameter_indices(topo)
    scores = torch.zeros(len(topo.blocks), dtype=torch.float64)
    for bi, slices in unit_idx.items():
        total = 0.0
        for name, dim, start, size in slices:
            f = fisher[name]
            sub = f[start : start + size] if dim == "out" else f[:, start : start + size]
            total += float(sub.sum().item())
        scores[bi] = total / max(1, seen)
    return scores


# --------------------------------------------------------------------------- #
# Mask search
# --------------------------------------------------------------------------- #
def mask_search(topo, importance: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """Iteratively drop the unit with the smallest salience *per parameter*.

    This is the iterative mask-search of the Mask Tuning framework: at every
    step the least valuable remaining unit is removed until the target sparsity
    (defined on the retained sub-network, exactly like APT's) is reached.
    """
    units = [b for b in topo.blocks if b.kind in (HEAD, NEURON)]
    index_of = {id(b): i for i, b in enumerate(topo.blocks)}
    costs = {b.bid: b.param_count for b in units}
    scores = {b.bid: float(importance[index_of[id(b)]].item()) for b in units}

    order = sorted(units, key=lambda b: scores[b.bid] / max(1, costs[b.bid]))
    state = topo.build_pruning_state()
    retained = {b.bid: True for b in units}

    def retained_params() -> int:
        # keep every dimension; count the retained heads / neurons
        n_head = sum(1 for b in units if b.kind == HEAD and retained[b.bid])
        n_neuron = sum(1 for b in units if b.kind == NEURON and retained[b.bid])
        n_dim = sum(1 for b in topo.blocks if b.kind == 2)
        k_ff = 3 if any(b.kind == NEURON and len(b.slices) == 3 for b in topo.blocks) else 2
        return n_dim * (4 * topo.d_h * n_head + k_ff * n_neuron)

    total = state.total_parameters
    budget = (1.0 - target_sparsity) * total
    for b in order:
        if retained_params() <= budget:
            break
        retained[b.bid] = False

    mask = torch.zeros(len(topo.blocks), dtype=torch.bool)
    for bi, b in enumerate(topo.blocks):
        if b.kind == 2:
            mask[bi] = True                      # dimensions are kept by the baseline
        else:
            mask[bi] = retained[b.bid]
    return mask


# --------------------------------------------------------------------------- #
# Mask tuning
# --------------------------------------------------------------------------- #
class MaskTuningPruner:
    """End-to-end Mask Tuning applied to a (possibly LoRA-tuned) model."""

    def __init__(
        self,
        model: nn.Module,
        task,
        config,
        target_sparsity: float = 0.6,
        calibration_features=None,
        device=None,
    ) -> None:
        self.device = device or resolve_device(config.device)
        self.model = model.to(self.device)
        self.task = task
        self.config = config
        self.target_sparsity = target_sparsity
        self.topo = wrap_model(self.model, WrapConfig(adapt_ffn=config.adapt_ffn))
        self.topo.freeze_backbone()
        self.calibration_features = calibration_features
        self.mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ main
    def fit(self, calibration_features=None) -> torch.Tensor:
        feats = calibration_features if calibration_features is not None else self.calibration_features
        importance = estimate_fisher(
            self.model, self.topo, self.task, feats, self.device,
            batch_size=max(1, self.config.batch_size // 2),
            max_batches=20,
        )
        self.importance = importance
        self.mask = mask_search(self.topo, importance, self.target_sparsity)
        return self.mask

    def tune_masks(self, features, epochs: int = 2, lr: float = 1e-2, batch_size: int = 16) -> None:
        """Tune continuous per-unit mask variables with the weights frozen.

        The masks are applied through forward hooks on the wrapped linears, so
        the rest of the network is untouched -- the "mask tuning" stage of
        Kwon et al.
        """
        soft = nn.Parameter(
            torch.where(
                self.mask,
                torch.full((len(self.mask),), 3.0),
                torch.full((len(self.mask),), -3.0),
            )
        )
        opt = torch.optim.Adam([soft], lr=lr)
        self.model.train()

        plan = salience_plan(self.topo)
        names = list(plan["names"])
        sizes_in = [self.topo.linears[n].in_features for n in names]
        sizes_out = [self.topo.linears[n].out_features for n in names]

        def apply_soft() -> Dict[str, tuple]:
            """Rebind every mask buffer to a *differentiable* function of ``soft``."""
            saved: Dict[str, tuple] = {}
            vals = torch.sigmoid(soft)
            masks_in = vals[plan["idx_in"]]
            masks_out = vals[plan["idx_out"]]
            for name, mi, mo in zip(
                names, torch.split(masks_in, sizes_in), torch.split(masks_out, sizes_out)
            ):
                lin = self.topo.linears[name]
                saved[name] = (lin.mask_in, lin.mask_out)
                lin.mask_in = mi
                lin.mask_out = mo
            return saved

        def restore(saved: Dict[str, tuple]) -> None:
            for name, (mi, mo) in saved.items():
                lin = self.topo.linears[name]
                lin.mask_in = mi
                lin.mask_out = mo

        for epoch in range(epochs):
            for batch in batches(features, batch_size, True, self.device, self.task.collate, seed=epoch):
                saved = apply_soft()
                out = self.task.forward(self.model, batch, output_hidden_states=False)
                loss = out["loss"]
                # keep the mask close to the searched (hard) decision
                target = self.mask.float()
                loss = loss + 1e-4 * (torch.sigmoid(soft) - target).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                restore(saved)
        with torch.no_grad():
            self.mask = torch.sigmoid(soft).detach() > 0.5

    def apply_mask(self) -> None:
        state = self.topo.build_pruning_state()
        state.retain = self.mask.clone()
        state.harden_masks()
        state.write_masks_to_linears(self.topo.linears)
        self.state = state

    def physical(self):
        from apt.physical import materialize

        return materialize(self.topo, self.state)


def apply_structured_mask(topo, mask: torch.Tensor) -> None:
    state = topo.build_pruning_state()
    state.retain = mask.clone()
    state.harden_masks()
    state.write_masks_to_linears(topo.linears)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_lora_prune(
    model,
    tokenizer,
    task,
    config,
    train_features,
    eval_features=None,
    raw_eval=None,
    output_dir: Optional[str] = None,
    measure: bool = True,
) -> Dict[str, Any]:
    """``LoRA + Prune``: LoRA-tune, Mask-Tune, retrain the pruned LM."""
    from .lora import attach_lora, merge_and_unwrap

    device = resolve_device(config.device)
    model = model.to(device)
    out_dir = output_dir or config.output_dir
    log = LogWriter(f"{out_dir}/lora_prune_log.jsonl")

    # 1. LoRA fine-tuning of the dense model
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = attach_lora(model, r=config.initial_rank, scaling=config.scaling)
    lora_params = [p for _, _, lin in wrapped for p in (lin.lora_A, lin.lora_B)]
    stage1 = train_supervised(
        model, task, train_features, device,
        epochs=config.distill_epochs, batch_size=config.batch_size, lr=config.learning_rate,
        trainable=lora_params, eval_features=eval_features, raw_eval=raw_eval,
        eval_interval=config.eval_interval, tta_target=config.tta_target, seed=config.seed,
    )

    # 2. merge LoRA, then run Mask Tuning on the merged model
    merge_and_unwrap(wrapped)
    pruner = MaskTuningPruner(
        model, task, config,
        target_sparsity=config.target_sparsity,
        calibration_features=train_features,
        device=device,
    )
    pruner.fit(train_features)
    pruner.tune_masks(train_features, epochs=max(1, config.epochs // 20))
    pruner.apply_mask()
    pruned_model = pruner.physical()

    # 3. retrain the pruned LM with LoRA to recover accuracy (the baseline's
    #    training memory in Table 2 equals the LoRA-only memory footprint)
    for p in pruned_model.parameters():
        p.requires_grad_(False)
    rewrapped = attach_lora(pruned_model, r=config.initial_rank, scaling=config.scaling)
    retrain_params = [p for _, _, lin in rewrapped for p in (lin.lora_A, lin.lora_B)]
    stage3 = train_supervised(
        pruned_model, task, train_features, device,
        epochs=config.distill_epochs, batch_size=config.batch_size, lr=config.learning_rate,
        trainable=retrain_params,
        eval_features=eval_features, raw_eval=raw_eval, eval_interval=config.eval_interval,
        tta_target=config.tta_target, seed=config.seed, log=log,
    )
    merge_and_unwrap(rewrapped)
    achieved = float(pruner.state.current_sparsity())
    result = {
        "method": "LoRA+Prune",
        "target_sparsity": config.target_sparsity,
        "achieved_sparsity": achieved,
        "stage1": {k: v for k, v in stage1.items() if k != "history"},
        "stage3": {k: v for k, v in stage3.items() if k != "history"},
        "final_metrics": stage3.get("final_metrics", {}),
        "wall_time_s": stage1["wall_time_s"] + stage3["wall_time_s"],
        "tta_s": stage3.get("tta_s"),
        "n_parameters": sum(p.numel() for p in pruned_model.parameters()),
    }
    if measure and eval_features is not None:
        result["inference"] = measure_inference(
            pruned_model, task, eval_features, device, batch_size=config.eval_batch_size
        )
    return result


__all__ = [
    "MaskTuningPruner",
    "estimate_fisher",
    "mask_search",
    "apply_structured_mask",
    "run_lora_prune",
]
