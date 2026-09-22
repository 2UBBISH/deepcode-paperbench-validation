"""The two-stage APT training loop.

Stage 1 -- *adaptive pruning and tuning with self-distillation*
    The LM is fine-tuned while the pruning masks are annealed from "keep
    everything" to the target sparsity following the cubic schedule of
    Appendix A.  At every adjustment step the block salience (Eq. 5) is turned
    into a new mask set by the binary-search knapsack of Appendix C, the ranks
    of the top-half salient APT adapters are linearly increased (Section 4.3)
    and the optimizer is reset because the parameter shapes changed.
    The objective is ``L = mu L_distill + (1 - mu) L_ft`` (Eq. 7).

Stage 2 -- *performance recovery*
    The masks are hardened (and the model may be physically shrunk) and the
    pruned LM is fine-tuned with the supervised objective only, as described in
    Appendix A ("we first prune and train the LM with the self-distillation
    objective, and then fine-tune the pruned LM to recover its end-task
    performance").

Efficiency bookkeeping: ``Trainer`` records the wall-clock time to accuracy
(TTA, Appendix/Table 11) and the peak training memory
(``torch.cuda.max_memory_allocated`` -- see the addendum).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import schedule as sched
from .blocks import HEAD, NEURON, DIM
from .distill import (
    DistillWeights,
    TeacherCache,
    distill_loss,
    layer_mapping,
    layerwise_distillation_loss,
    sample_teacher_layers,
    teacher_masks,
    total_loss,
    track_off,
)
from .tuning import grow_salient_adapters, grow_uniform_adapters
from .wrap import Topology, WrapConfig, compute_block_salience, wrap_model


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class APTConfig:
    """All hyper-parameters of the reproduction (defaults: paper Table 6)."""

    model_name: str = "roberta-base"
    task: str = "sst2"
    output_dir: str = "runs/apt"

    # --- method -----------------------------------------------------------
    target_sparsity: float = 0.60
    initial_sparsity: float = 0.0
    initial_rank: int = 8
    target_rank: int = 64
    top_adapter_fraction: float = 0.5
    scaling: float = 2.0
    ema_beta: float = 0.85           # addendum: 0.85 EMA of the block salience
    mask_alpha: float = 0.01         # Appendix C gradual mask decay
    adjust_interval: int = 1         # steps between mask / rank adjustments
    """Algorithm 1 re-selects the blocks and re-allocates the tuning ranks at
    *every* step, which is the default here.  Larger values trade a little
    fidelity for speed."""
    use_adaptive_pruning: bool = True
    use_adaptive_tuning: bool = True
    use_distillation: bool = True
    use_kurtosis: bool = True
    salience_based_allocation: bool = True
    adapt_ffn: bool = True
    tr_rank: int = 8
    teacher_layer_fraction: float = 0.5

    # --- optimisation (Table 6) -------------------------------------------
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_eps: float = 1e-8
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 40
    distill_epochs: int = 20
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.06
    max_input_length: Optional[int] = None

    # --- bookkeeping -------------------------------------------------------
    seed: int = 42
    device: str = "auto"
    eval_interval: int = 200
    log_interval: int = 25
    tta_target: Optional[float] = None
    physically_prune: bool = True
    num_workers: int = 2

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_memory_mb() -> float:
    """Peak memory in MB.

    The addendum specifies ``torch.cuda.max_memory_allocated()``; on CPU we fall
    back to the process high-water mark so that the same code path can be
    exercised without a GPU.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":       # macOS reports bytes
        return peak / (1024 ** 2)
    return peak / 1024.0               # Linux reports KiB


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Time-to-accuracy
# --------------------------------------------------------------------------- #
class TimeToAccuracy:
    """Records (wall-clock seconds, dev metric) pairs and resolves the TTA.

    The addendum defines the training-time metric as "the time-to-accuracy of
    reaching 97% of the dev (/test) set performance of the finetuning
    baseline".
    """

    def __init__(self, target: Optional[float], ratio: float = 0.97, higher_is_better: bool = True):
        self.target = target
        self.ratio = ratio
        self.higher_is_better = higher_is_better
        self.history: List[Tuple[float, float]] = []
        self.tta: Optional[float] = None

    def update(self, seconds: float, metric: float) -> None:
        self.history.append((seconds, metric))
        if self.tta is not None:
            return
        if self.target is None:
            return
        threshold = self.ratio * self.target if self.higher_is_better else (
            (2.0 - self.ratio) * self.target
        )
        reached = metric >= threshold if self.higher_is_better else metric <= threshold
        if reached:
            self.tta = seconds

    def resolve_with_final(self) -> Optional[float]:
        """Fallback used when no external FT baseline is supplied."""
        if self.tta is not None or not self.history:
            return self.tta
        best = max(m for _, m in self.history)
        for sec, m in self.history:
            if m >= self.ratio * best:
                self.tta = sec
                break
        return self.tta


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class APTTrainer:
    """Runs APT (or one of its ablations) on a single task."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        task,
        config: APTConfig,
        train_features,
        eval_features=None,
        raw_eval=None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.task = task
        self.tokenizer = tokenizer
        self.train_features = train_features
        self.eval_features = eval_features
        self.raw_eval = raw_eval

        self.model = model.to(self.device)
        wrap_cfg = WrapConfig(
            initial_rank=config.initial_rank,
            scaling=config.scaling,
            track=True,
            adapt_ffn=config.adapt_ffn,
        )
        self.topo: Topology = wrap_model(self.model, wrap_cfg)
        self.topo.freeze_backbone()

        self.state = self.topo.build_pruning_state(
            ema_beta=config.ema_beta, alpha=config.mask_alpha
        )
        self.state.check_coverage(self.topo.linears)

        self.distill_weights = DistillWeights.for_task(task.name)
        self.teacher = TeacherCache(
            self.topo, hidden_size=self._hidden_size(), tr_rank=config.tr_rank, device=self.device
        )
        self.tta = TimeToAccuracy(config.tta_target, higher_is_better=task.higher_is_better())
        self.logs: List[Dict[str, Any]] = []
        self.global_step = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        #: model/topology currently being trained (stage 1 -> the masked model,
        #: stage 2 -> the physically pruned model when pruning is enabled)
        self.active_model = self.model
        self.active_topo = self.topo

    # ------------------------------------------------------------- plumbing
    def _hidden_size(self) -> int:
        for name, lin in self.topo.linears.items():
            if name.endswith(".q"):
                return lin.in_features
        return self.model.config.hidden_size

    def _trainable_parameters(self) -> List[nn.Parameter]:
        params = [p for p in self.active_topo.trainable_parameters()]
        if self.config.use_distillation and self.teacher is not None:
            params += self.teacher.trainable_parameters()
        return params

    def _build_optimizer(self, total_steps: int) -> None:
        params = self._trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )
        warmup = max(1, int(total_steps * self.config.warmup_fraction))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _reset_optimizer_after_resize(self, total_steps: int) -> None:
        """The paper resets the optimizer whenever the parameter sizes change."""
        self._build_optimizer(total_steps)

    # --------------------------------------------------------------- batches
    def _loader(self, features, shuffle: bool):
        indices = list(range(len(features)))
        if shuffle:
            random.shuffle(indices)
        bs = self.config.batch_size
        for i in range(0, len(indices), bs):
            chunk = [features[j] for j in indices[i : i + bs]]
            batch = self.task.collate(chunk)
            yield {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # ------------------------------------------------------------- training
    def train(self) -> Dict[str, Any]:
        cfg = self.config
        set_seed(cfg.seed)
        steps_per_epoch = max(1, math.ceil(len(self.train_features) / cfg.batch_size))
        n_prune_steps = steps_per_epoch * cfg.distill_epochs
        n_recover_steps = steps_per_epoch * max(0, cfg.epochs - cfg.distill_epochs)
        total_steps = n_prune_steps + n_recover_steps
        self._build_optimizer(total_steps)

        sparsity = sched.SparsitySchedule(
            target_sparsity=cfg.target_sparsity if cfg.use_adaptive_pruning else 0.0,
            total_steps=max(1, n_prune_steps),
            initial_sparsity=cfg.initial_sparsity,
        )
        adjustments = set(sched.adjustment_steps(n_prune_steps, cfg.adjust_interval))

        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "apt_config.json"), "w") as fh:
            fh.write(cfg.to_json())

        reset_peak_memory()
        t0 = time.time()
        loss_acc, loss_n = 0.0, 0

        # ------------------------------ stage 1: prune + distill ------------ #
        for epoch in range(cfg.distill_epochs):
            for batch in self._loader(self.train_features, shuffle=True):
                if self.global_step >= n_prune_steps:
                    break
                loss_val = self._prune_step(batch, sparsity, adjustments, n_prune_steps, total_steps)
                loss_acc += loss_val
                loss_n += 1
                self._maybe_eval(t0, n_prune_steps)
                self._maybe_log(epoch, loss_acc / max(1, loss_n))
                loss_acc, loss_n = 0.0, 0
                self.global_step += 1

        # The cubic schedule only reaches gamma_T at t == T while the loop stops
        # at T-1, so re-select the blocks once more for the *target* sparsity
        # before hardening: the returned model is then guaranteed to satisfy the
        # paper's constraint 1 - C(Theta_T, M_T)/C(Theta_0, M_0) >= gamma_T.
        if cfg.use_adaptive_pruning:
            self.state.select_for_budget(keep_ratio=1.0 - cfg.target_sparsity)
        # harden the masks once pruning is over
        self.state.harden_masks()
        self.state.write_masks_to_linears(self.topo.linears)
        self._log({"event": "pruning_finished", "sparsity": self.state.current_sparsity()})

        self.pruned_model = None
        if cfg.physically_prune and cfg.use_adaptive_pruning:
            try:
                from .physical import materialize

                self.pruned_model = materialize(self.topo, self.state)
                self.pruned_n_params = int(sum(p.numel() for p in self.pruned_model.parameters()))
                self._log({"event": "physical_prune", "n_params": self.pruned_n_params})
            except Exception as exc:  # pragma: no cover - reported in the logs
                self._log({"event": "physical_prune_failed", "error": repr(exc)})

        # ------------------------------ stage 2: recover -------------------- #
        if self.pruned_model is not None:
            # "we first prune and train the LM with the self-distillation
            # objective, and then fine-tune the pruned LM" (Appendix A) -- the
            # recovery stage therefore runs on the physically pruned model, which
            # is also what gives APT its training-memory advantage.
            self.pruned_model = self.pruned_model.to(self.device)
            recovery_cfg = WrapConfig(
                initial_rank=cfg.initial_rank,
                scaling=cfg.scaling,
                track=False,
                adapt_ffn=cfg.adapt_ffn,
            )
            self.active_topo = wrap_model(self.pruned_model, recovery_cfg)
            self.active_topo.freeze_backbone()
            self.active_model = self.pruned_model
        else:
            self.active_topo = self.topo
            self.active_model = self.model
            for p in self.model.parameters():
                p.requires_grad_(False)
            for lin in self.active_topo.linears.values():
                if lin.use_lora:
                    lin.lora_A.requires_grad_(True)
                    lin.lora_B.requires_grad_(True)
        self._build_optimizer(max(1, n_recover_steps))
        self.teacher = None
        self.state.track_enabled = False
        for lin in self.topo.linears.values():
            lin.track_kurtosis = False

        for epoch in range(max(0, cfg.epochs - cfg.distill_epochs)):
            for batch in self._loader(self.train_features, shuffle=True):
                self.optimizer.zero_grad(set_to_none=True)
                out = self.task.forward(self.active_model, batch, output_hidden_states=False)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                loss_acc += float(loss.detach().item())
                loss_n += 1
                self._maybe_eval(t0, total_steps)
                self.global_step += 1
            self._log({"epoch": epoch, "stage": "recover", "loss": loss_acc / max(1, loss_n)})
            loss_acc, loss_n = 0.0, 0

        self.tta.resolve_with_final()
        wall = time.time() - t0
        inference = self._measure_inference()
        result = {
            "config": asdict(cfg),
            "target_sparsity": cfg.target_sparsity,
            "achieved_sparsity": self.state.current_sparsity(),
            "wall_time_s": wall,
            "tta_s": self.tta.tta,
            "tta_history": self.tta.history,
            "peak_train_memory_mb": peak_memory_mb(),
            "final_metrics": self._final_metrics(),
            "inference": inference,
            "n_parameters_dense": int(sum(p.numel() for p in self.model.parameters())),
            "n_parameters_pruned": getattr(self, "pruned_n_params", None),
            "logs": self.logs,
        }
        with open(os.path.join(cfg.output_dir, "result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    def _measure_inference(self) -> Dict[str, Any]:
        """Inference throughput / memory of the dense and the pruned LM."""
        pruned = getattr(self, "pruned_model", None)
        if self.eval_features is None:
            return {}
        from .measure import measure_inference, relative_efficiency

        bs = self.config.eval_batch_size
        out: Dict[str, Any] = {}
        with teacher_masks(self.topo), track_off(self.topo):
            out["dense"] = measure_inference(
                self.model, self.task, self.eval_features, self.device, batch_size=bs, max_batches=10
            )
        if pruned is not None:
            pruned = pruned.to(self.device)
            out["pruned"] = measure_inference(
                pruned, self.task, self.eval_features, self.device, batch_size=bs, max_batches=10
            )
            out["relative"] = relative_efficiency(out["dense"], out["pruned"])
        return out

    # ------------------------------------------------------------- one step
    def _prune_step(self, batch, sparsity, adjustments, n_prune_steps, total_steps) -> float:
        cfg = self.config
        debug = os.environ.get("APT_DEBUG_STEPS") == "1"
        if debug:
            self._log({"step": self.global_step, "phase": "begin"})
        self.topo.reset_statistics()
        self.optimizer.zero_grad(set_to_none=True)
        if debug:
            self._log({"step": self.global_step, "phase": "forward"})

        # The teacher forward runs *before* the student forward so that no
        # parameter/buffer is mutated between the student's forward and its
        # backward pass (which would corrupt the autograd graph).
        t_stacks: List[List[torch.Tensor]] = []
        mu = 0.0
        if cfg.use_distillation and self.teacher is not None:
            mu = sched.mu_schedule(self.global_step, 0, max(1, n_prune_steps))
            self.teacher.sync_from_student()
            with torch.no_grad(), track_off(self.topo), teacher_masks(self.topo):
                with self.teacher.load_teacher_weights():
                    t_out = self.task.forward(self.model, batch, output_hidden_states=True)
            t_stacks = self._hidden_stacks(t_out.get("hidden_states"))

        out = self.task.forward(self.model, batch, output_hidden_states=cfg.use_distillation)
        l_ft = out["loss"]
        loss = l_ft

        if t_stacks:
            s_stacks = self._hidden_stacks(out.get("hidden_states"))
            l_layer = None
            for si, (s_stack, t_stack) in enumerate(zip(s_stacks, t_stacks)):
                n_layers = len(t_stack) - 1
                sampled = sample_teacher_layers(
                    max(1, n_layers), max(1, int(n_layers * cfg.teacher_layer_fraction))
                )
                mapping = layer_mapping([s + 1 for s in sampled], list(range(1, len(s_stack))))
                term = layerwise_distillation_loss(self.teacher.tr_for(si), s_stack, t_stack, mapping)
                l_layer = term if l_layer is None else l_layer + term
            if l_layer is None:
                l_layer = torch.zeros((), device=self.device)
            l_dist = distill_loss(l_ft, l_layer, self.distill_weights)
            loss = total_loss(l_dist, l_ft, mu)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), cfg.max_grad_norm)

        # ---- block salience -> mask search -------------------------------- #
        if cfg.use_adaptive_pruning:
            scores = compute_block_salience(self.topo, use_kurtosis=cfg.use_kurtosis)
            self.state.update_salience(scores)
            if self.global_step in adjustments or self.global_step == n_prune_steps - 1:
                self.state.select_for_budget(keep_ratio=sparsity.density(self.global_step))
                self.state.anneal_masks()
                grown = self._grow_ranks()
                if grown:
                    self._reset_optimizer_after_resize(total_steps)
                self._log(
                    {
                        "event": "adjust",
                        "step": self.global_step,
                        "target_sparsity": sparsity.sparsity(self.global_step),
                        "achieved_sparsity": self.state.current_sparsity(),
                        "grown_adapters": len(grown),
                    }
                )
            elif self.state.mask.numel():
                self.state.anneal_masks()
            self.state.write_masks_to_linears(self.topo.linears)

        if debug:
            self._log({"step": self.global_step, "phase": "step"})
        self.optimizer.step()
        self.scheduler.step()
        return float(loss.detach().item())

    def _grow_ranks(self) -> List[str]:
        cfg = self.config
        if not cfg.use_adaptive_tuning:
            return []
        target = sched.linear_rank(
            cfg.target_rank, cfg.initial_rank, self.global_step,
            max(1, cfg.distill_epochs * max(1, len(self.train_features) // cfg.batch_size)),
        )
        if target <= cfg.initial_rank:
            return []
        if cfg.salience_based_allocation:
            return grow_salient_adapters(self.topo, target, cfg.top_adapter_fraction)
        grown = grow_uniform_adapters(self.topo, target)
        return grown

    @staticmethod
    def _hidden_stacks(hidden) -> List[List[torch.Tensor]]:
        """Group hidden states into stacks: one for RoBERTa, two for T5.

        Encoder and decoder hidden states have different sequence lengths, so
        they must be distilled separately (a teacher encoder layer can only be
        matched with a student encoder layer, and likewise for the decoder).
        """
        if hidden is None:
            return []
        if isinstance(hidden, (tuple, list)) and len(hidden) and isinstance(hidden[0], (tuple, list)):
            return [[h for h in part if torch.is_tensor(h)] for part in hidden]
        return [[h for h in hidden if torch.is_tensor(h)]]

    # ------------------------------------------------------------ evaluation
    def _maybe_eval(self, t0: float, total_steps: int) -> None:
        if self.eval_features is None:
            return
        if self.global_step % self.config.eval_interval != 0:
            return
        metrics = self.evaluate()
        value = self.task.primary_metric(metrics)
        self.tta.update(time.time() - t0, value)
        self._log({"event": "eval", "step": self.global_step, "seconds": time.time() - t0, **metrics})

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if hasattr(self.task, "evaluate") and self.raw_eval is not None:
            return self.task.evaluate(
                self.active_model,
                self.eval_features,
                self.raw_eval,
                batch_size=self.config.eval_batch_size,
                device=str(self.device),
            )
        self.active_model.eval()
        preds, labels = [], []
        for batch in self._loader(self.eval_features, shuffle=False):
            out = self.task.forward(self.active_model, batch, output_hidden_states=False)
            logits = out["logits"]
            if isinstance(logits, tuple):
                logits = logits[0]
            preds.append(logits.detach().cpu())
            labels.append(batch["labels"].detach().cpu())
        self.active_model.train()
        if not preds:
            return {self.task.metric_name: 0.0}
        logits = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        return self.task.metrics(logits, {"labels": labels})

    # --------------------------------------------------------------- logging
    def _log(self, record: Dict[str, Any]) -> None:
        rec = dict(record)
        rec.setdefault("step", self.global_step)
        self.logs.append(rec)
        path = os.path.join(self.config.output_dir, "train_log.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _maybe_log(self, epoch: int, loss: float) -> None:
        if self.global_step % self.config.log_interval == 0:
            self._log({"epoch": epoch, "stage": "prune", "loss": loss, "sparsity": self.state.current_sparsity()})

    def _final_metrics(self) -> Dict[str, float]:
        if self.eval_features is None:
            return {}
        return self.evaluate()


# --------------------------------------------------------------------------- #
# Convenience entry points
# --------------------------------------------------------------------------- #
def train_apt(model, tokenizer, task, config: APTConfig, train_features, eval_features=None, raw_eval=None):
    trainer = APTTrainer(model, tokenizer, task, config, train_features, eval_features, raw_eval)
    return trainer, trainer.train()


__all__ = [
    "APTConfig",
    "APTTrainer",
    "TimeToAccuracy",
    "set_seed",
    "resolve_device",
    "peak_memory_mb",
    "train_apt",
]
