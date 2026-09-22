"""CoFi pruning + distillation baseline (``Prune+Distill`` /
``LoRA+Prune+Distill`` rows of Table 2).

Follows Xia et al., *Structured Pruning Learns Compact and Accurate Models*
(https://github.com/princeton-nlp/CoFiPruning), adapted -- as the task addendum
requires -- so that in the ``LoRA+Prune+Distill`` variant *only the L0 modules
and the LoRA parameters are tunable*.

Objective (CoFi Eq. 1 / 5)::

    L = lambda_ce  * CE(student, labels)
      + lambda_hid * sum_l MSE(H_l^s, T(H_l^t))
      + lambda_att * sum_l MSE(A_l^s, A_l^t)
      + lambda_l0  * L0(gates)

The structured sparsity is produced by hard-concrete L0 gates (Louizos et al.
2018) placed on the same structural units APT prunes -- MHA heads, FFN neurons
and the model hidden dimension -- which are read off from the shared block
table built in :mod:`apt.wrap`.

CoFi keeps a *separate, full-size teacher copy of the LM in memory*, which is
exactly the cost that motivates APT's self-distillation (Section 4.4); the paper
therefore only compares CoFi on RoBERTa.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from apt.distill import layer_mapping, track_off
from apt.trainer import resolve_device, set_seed
from apt.wrap import WrapConfig, salience_plan, wrap_model

from .common import LogWriter, batches, count_parameters, evaluate_task, measure_inference


@dataclass
class CoFiConfig:
    lambda_ce: float = 1.0
    lambda_hidden: float = 1.0
    lambda_attention: float = 1.0
    lambda_l0: float = 6e-4
    temperature: float = 1.0
    beta: float = 2.0 / 3.0          # hard-concrete temperature
    gamma: float = -0.1
    zeta: float = 1.1
    learn_layer_mapping: bool = True


class HardConcreteGates(nn.Module):
    """Per-unit hard-concrete L0 gates (Louizos et al. 2018)."""

    def __init__(self, n_units: int, init_log_alpha: float = 2.0, cfg: Optional[CoFiConfig] = None):
        super().__init__()
        self.cfg = cfg or CoFiConfig()
        self.log_alpha = nn.Parameter(torch.full((n_units,), float(init_log_alpha)))
        self.n_units = n_units

    def _cdf(self, x: float) -> torch.Tensor:
        c = self.cfg
        v = math.log(-c.gamma / c.zeta)
        return torch.sigmoid(self.log_alpha - c.beta * v) - c.beta * torch.log(-c.gamma / c.zeta)

    def l0_penalty(self) -> torch.Tensor:
        c = self.cfg
        v = math.log(-c.gamma / c.zeta)
        return torch.sigmoid(self.log_alpha - c.beta * v).sum()

    def forward(self, training: bool = True) -> torch.Tensor:
        c = self.cfg
        if training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / c.beta)
            s = s * (c.zeta - c.gamma) + c.gamma
            z = s.clamp(0.0, 1.0)
        else:
            z = (torch.sigmoid(self.log_alpha) * (c.zeta - c.gamma) + c.gamma).clamp(0.0, 1.0)
        return z


def _write_gates(topo, gate_values: torch.Tensor) -> None:
    """Broadcast one gate value per block onto every wrapped linear.

    Implemented with a single gather per direction so that (a) it is cheap and
    (b) the result stays attached to the autograd graph of the gates -- the L0
    parameters must receive gradients.  Buffers are *rebound* rather than
    modified in place so that no live graph is invalidated.
    """
    plan = salience_plan(topo)
    names = list(plan["names"])
    masks_in = gate_values[plan["idx_in"]]
    masks_out = gate_values[plan["idx_out"]]
    sizes_in = [topo.linears[n].in_features for n in names]
    sizes_out = [topo.linears[n].out_features for n in names]
    for name, mi, mo in zip(names, torch.split(masks_in, sizes_in), torch.split(masks_out, sizes_out)):
        lin = topo.linears[name]
        lin.mask_in = mi
        lin.mask_out = mo


class CoFiPruner:
    """L0 pruning + layer-wise distillation, then a fine-tuning stage."""

    def __init__(
        self,
        model: nn.Module,
        task,
        config,
        cofi: Optional[CoFiConfig] = None,
        target_sparsity: float = 0.6,
        lora_only: bool = False,
        device=None,
    ) -> None:
        self.device = device or resolve_device(config.device)
        self.config = config
        self.cofi = cofi or CoFiConfig()
        self.task = task
        self.target_sparsity = target_sparsity
        self.lora_only = lora_only

        # CoFi keeps a *separate full-size teacher model* in memory, which is
        # exactly the cost that motivates APT's self-distillation (Section 4.4).
        # The copy is taken *before* the student is wrapped so that the teacher
        # stays a plain (dense) model.
        self.teacher = copy.deepcopy(model).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student = model.to(self.device)
        self.student_topo = wrap_model(
            self.student, WrapConfig(initial_rank=config.initial_rank, scaling=config.scaling)
        )
        self.gates = HardConcreteGates(len(self.student_topo.blocks), cfg=self.cofi).to(self.device)

        self.teacher_topo = wrap_model(self.teacher, WrapConfig(adapt_ffn=config.adapt_ffn))
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- params
    def tunable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = [self.gates.log_alpha]
        if self.lora_only:
            for _, lin in self.student_topo.linears.items():
                if lin.use_lora:
                    params += [lin.lora_A, lin.lora_B]
        else:
            for p in self.student.parameters():
                p.requires_grad_(True)
                params.append(p)
        return params

    # -------------------------------------------------------------- training
    def train(
        self,
        train_features,
        eval_features=None,
        raw_eval=None,
        distill_epochs: int = 20,
        finetune_epochs: int = 20,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        set_seed(cfg.seed)
        log = LogWriter(log_path) if log_path else (lambda rec: None)

        if self.lora_only:
            for p in self.student.parameters():
                p.requires_grad_(False)
            self.student_topo.freeze_backbone()
        params = self.tunable_parameters()
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        steps_per_epoch = max(1, math.ceil(len(train_features) / cfg.batch_size))
        total_steps = steps_per_epoch * (distill_epochs + finetune_epochs)
        warmup = max(1, int(total_steps * cfg.warmup_fraction))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
        )

        step = 0
        # CoFi's progressive-pruning controller: the L0 coefficient is adapted
        # online so that the expected sparsity tracks a cubic ramp towards the
        # target (this is what makes the method reach a *requested* sparsity).
        total_prune_steps = max(1, steps_per_epoch * distill_epochs)
        self.lambda_l0 = self.cofi.lambda_l0
        for epoch in range(distill_epochs + finetune_epochs):
            distill = epoch < distill_epochs
            running, n = 0.0, 0
            for batch in batches(train_features, cfg.batch_size, True, self.device, self.task.collate, seed=cfg.seed + epoch):
                opt.zero_grad(set_to_none=True)
                z = self.gates(training=True)
                _write_gates(self.student_topo, z)
                out = self.task.forward(
                    self.student, batch, output_hidden_states=distill
                )
                loss = self.cofi.lambda_ce * out["loss"]
                if distill:
                    loss = loss + self._distillation_loss(batch, out)
                    u = min(1.0, (step + 1) / total_prune_steps)
                    target_t = self.target_sparsity * (1.0 - (1.0 - u) ** 3)
                    cur = self._measure_sparsity()
                    if cur < target_t:
                        self.lambda_l0 = min(1.0, self.lambda_l0 * 1.05)
                    elif cur > target_t + 0.01:
                        self.lambda_l0 = max(1e-6, self.lambda_l0 * 0.95)
                    loss = loss + self.lambda_l0 * self.gates.l0_penalty()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                sched.step()
                running += float(loss.detach().item())
                n += 1
                step += 1
                if eval_features is not None and step % cfg.eval_interval == 0:
                    with torch.no_grad():
                        _write_gates(self.student_topo, self.gates(training=False))
                    metrics = evaluate_task(self.student, self.task, eval_features, raw_eval, cfg.eval_batch_size, self.device)
                    log({"step": step, "stage": "distill" if distill else "finetune", **metrics})
            log({"epoch": epoch, "loss": running / max(1, n)})

        with torch.no_grad():
            self.finalize(self.target_sparsity)
        log({"event": "final_sparsity", "sparsity": self.sparsity})
        return {"final_sparsity": self.sparsity, "steps": step}

    def finalize(self, target_sparsity: float) -> None:
        """Threshold the learned gates so that the requested sparsity is met.

        The controller can lag behind the target on short schedules, so the
        distillation stage ends with the same budgeted selection APT uses: the
        blocks are ranked by their (learned) gate value and the top ones that fit
        into ``(1 - target_sparsity) * C`` are kept.  This keeps the sparsity
        accounting identical between CoFi and APT.
        """
        state = self.student_topo.build_pruning_state()
        state.salience_ema = self.gates(training=False).detach().double().cpu()
        state.select_for_budget(keep_ratio=1.0 - target_sparsity)
        state.harden_masks()
        state.write_masks_to_linears(self.student_topo.linears)
        self.state = state
        self.retain = state.retain
        self.sparsity = state.current_sparsity()

    # ---------------------------------------------------------- distillation
    def _student_hidden(self, out):
        return self._flatten(out.get("hidden_states"))

    @staticmethod
    def _flatten(hidden) -> List[torch.Tensor]:
        if hidden is None:
            return []
        if isinstance(hidden, (tuple, list)) and len(hidden) and isinstance(hidden[0], (tuple, list)):
            flat: List[torch.Tensor] = []
            for part in hidden:
                flat += [h for h in part if torch.is_tensor(h)]
            return flat
        return [h for h in hidden if torch.is_tensor(h)]

    def _distillation_loss(self, batch, student_out) -> torch.Tensor:
        """Layer-wise hidden-state distillation with dynamic layer mapping."""
        with torch.no_grad(), track_off(self.teacher_topo):
            teacher_out = self.task.forward(self.teacher, batch, output_hidden_states=True)
        s_hidden = self._student_hidden(student_out)
        t_hidden = self._flatten(teacher_out.get("hidden_states"))
        if not s_hidden or not t_hidden:
            return torch.zeros((), device=self.device)
        n = min(len(s_hidden), len(t_hidden))
        mapping = layer_mapping(list(range(n)), list(range(n)))
        total = torch.zeros((), device=self.device)
        for t_idx, s_idx in mapping.items():
            total = total + F.mse_loss(s_hidden[s_idx], t_hidden[t_idx])
        return self.cofi.lambda_hidden * total / max(1, len(mapping))

    # -------------------------------------------------------------- measure
    def _measure_sparsity(self, z: Optional[torch.Tensor] = None) -> float:
        if z is None:
            z = self.gates(training=False)
        keep = z > 0.5
        kinds = torch.tensor([b.kind for b in self.student_topo.blocks])
        d_h = self.student_topo.d_h
        k_ff = 3 if any(b.kind == 1 and len(b.slices) == 3 for b in self.student_topo.blocks) else 2
        n_head = int((keep & (kinds == 0)).sum().item())
        n_neuron = int((keep & (kinds == 1)).sum().item())
        n_dim = int((keep & (kinds == 2)).sum().item())
        total = self.student_topo.build_pruning_state().total_parameters
        kept = n_dim * (4 * d_h * n_head + k_ff * n_neuron)
        return 1.0 - kept / max(1e-9, total)

    def physical(self):
        from apt.physical import materialize

        if not hasattr(self, "state"):
            self.finalize(self.target_sparsity)
        return materialize(self.student_topo, self.state)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run_cofi(
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
    """``Prune + Distill``: CoFi with fully tunable parameters."""
    return _run_cofi_common(
        model, task, config, train_features, eval_features, raw_eval,
        output_dir=output_dir, measure=measure, lora_only=False,
    )


def run_lora_prune_distill(
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
    """``LoRA + Prune + Distill``: CoFi with only the L0 gates + LoRA tunable."""
    return _run_cofi_common(
        model, task, config, train_features, eval_features, raw_eval,
        output_dir=output_dir, measure=measure, lora_only=True,
    )


def _run_cofi_common(
    model,
    task,
    config,
    train_features,
    eval_features,
    raw_eval,
    output_dir: Optional[str],
    measure: bool,
    lora_only: bool,
) -> Dict[str, Any]:
    device = resolve_device(config.device)
    out_dir = output_dir or config.output_dir
    pruner = CoFiPruner(
        model, task, config,
        target_sparsity=config.target_sparsity,
        lora_only=lora_only,
        device=device,
    )
    pruner.train(
        train_features,
        eval_features=eval_features,
        raw_eval=raw_eval,
        distill_epochs=config.distill_epochs,
        finetune_epochs=max(0, config.epochs - config.distill_epochs),
        log_path=f"{out_dir}/{'lora_prune_distill' if lora_only else 'cofi'}_log.jsonl",
    )
    pruned = pruner.physical()
    result = {
        "method": "LoRA+Prune+Distill" if lora_only else "Prune+Distill",
        "target_sparsity": config.target_sparsity,
        "achieved_sparsity": pruner.sparsity,
        "n_parameters": count_parameters(pruned),
    }
    if eval_features is not None:
        result["final_metrics"] = evaluate_task(pruned, task, eval_features, raw_eval, config.eval_batch_size, device)
    if measure and eval_features is not None:
        result["inference"] = measure_inference(pruned, task, eval_features, device, batch_size=config.eval_batch_size)
    return result


__all__ = ["CoFiConfig", "CoFiPruner", "HardConcreteGates", "run_cofi", "run_lora_prune_distill"]
