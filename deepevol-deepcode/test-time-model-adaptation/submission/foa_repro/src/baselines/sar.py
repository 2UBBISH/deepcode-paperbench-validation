"""SAR baseline (Niu et al., 2023) -- Sharpness-Aware and Reliable entropy
minimization for fully test-time adaptation.

Paper (FOA) Appendix B.2, "SAR" paragraph, specifies the hyper-parameters used
for the comparison:

    "We follow all hyper-parameters that are set in SAR unless it does not
     provide. Specifically, we use SGD as the update rule, with a momentum of
     0.9, batch size of 64 and the learning rate of 0.001. The entropy
     threshold E_0 is set to 0.4 x ln C, where C is the number of task classes.
     The trainable parameters are the affine parameters of the layer
     normalization layers from blocks1 to blocks8 for ViT-Base."

Additional SAR hyper-parameters taken from the original paper (Niu et al., 2023)
because the FOA paper does not restate them:

    * SAM neighbourhood size ``rho``                      = 0.05
    * model-restoration / reset threshold ``eta``          = 0.01
    * per-sample reliable-entropy threshold                = 0.4 * ln(C)
    * de-emphasise (rather than discard) unreliable samples via
      ``w_t = exp(-(H_t - E_0))`` weighting.

This module is a self-contained PyTorch adapter: no gradients are required by
FOA, but SAR *does* use gradients, so this wrapper is only imported by the
baseline runner.  It supports both a plain ``nn.Module`` / timm ViT and the FOA
:class:`~src.models.vit_loader.ViTWithCLSFeatures` wrapper (which exposes the
inner timm model as ``.model`` and returns a ``dict`` from ``forward``).
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "SAR",
    "SARConfig",
    "build_sar",
    "build_baseline",
    "reliable_entropy",
    "sample_weights",
    "softmax_entropy",
    "entropy_loss",
    "select_norm_affine_params",
    "select_block_norm_affine_params",
    "DEFAULT_LR",
    "DEFAULT_MOMENTUM",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_RHO",
    "DEFAULT_ETA",
    "DEFAULT_ENTROPY_FACTOR",
]

# --------------------------------------------------------------------------- #
# Defaults (FOA Appendix B.2 + SAR paper)
# --------------------------------------------------------------------------- #
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.9
DEFAULT_BATCH_SIZE = 64
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_RHO = 0.05            # SAM perturbation radius (SAR paper default)
DEFAULT_ETA = 0.01            # model-restoration fraction threshold (SAR default)
DEFAULT_ENTROPY_FACTOR = 0.4  # E_0 = 0.4 * ln C  (FOA Appendix B.2)
DEFAULT_NUM_BLOCKS = 8        # ViT-Base blocks 1..8 per FOA Appendix B.2
DEFAULT_ENTROPY_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def softmax_entropy(logits: torch.Tensor, eps: float = DEFAULT_ENTROPY_EPS) -> torch.Tensor:
    """Per-sample Shannon entropy of ``softmax(logits)`` -> shape ``[B]``."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def entropy_loss(
    logits: torch.Tensor,
    eps: float = DEFAULT_ENTROPY_EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """Entropy objective used by TENT/SAR-style adaptation."""
    ent = softmax_entropy(logits, eps=eps)
    if reduction == "mean":
        return ent.mean()
    if reduction == "sum":
        return ent.sum()
    if reduction in ("none", "elementwise"):
        return ent
    raise ValueError(f"unknown reduction '{reduction}'")


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying classifier module (handles the FOA wrapper)."""
    if hasattr(model, "model") and isinstance(getattr(model, "model"), nn.Module):
        return model.model
    return model


def _logits_from_output(output: Any) -> torch.Tensor:
    """Normalise the many possible forward outputs to a logits tensor."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "output", "out", "pred", "predictions"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() >= 2:
                return item
    raise TypeError(f"cannot extract logits from object of type {type(output)!r}")


def _iter_named_modules(root: nn.Module, prefix: str = ""):
    for name, module in root.named_children():
        full = f"{prefix}.{name}" if prefix else name
        yield full, module
        yield from _iter_named_modules(module, full)


def _find_block_list(root: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the ``ModuleList`` holding the transformer blocks."""
    candidates = []
    for name, module in _iter_named_modules(root):
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            leaf = name.split(".")[-1]
            if leaf in ("blocks", "layers", "encoderlayer", "resblocks", "stages"):
                # prefer the longest list (the transformer stack)
                candidates.append((len(module), module))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _norm_affine_params(module: nn.Module) -> List[nn.Parameter]:
    """All affine parameters of normalisation layers inside ``module``."""
    params: List[nn.Parameter] = []
    for sub in module.modules():
        if isinstance(sub, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            if getattr(sub, "weight", None) is not None:
                params.append(sub.weight)
            if getattr(sub, "bias", None) is not None:
                params.append(sub.bias)
    return params


def select_block_norm_affine_params(
    model: nn.Module,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    block_start: int = 1,
    include_final_norm: bool = False,
) -> List[nn.Parameter]:
    """LN affine parameters of blocks ``block_start .. block_start+num_blocks-1``.

    Reproduces FOA Appendix B.2: "the affine parameters of the layer
    normalization layers from blocks1 to blocks8 for ViT-Base".  ``block_start``
    and ``block_start + num_blocks - 1`` are 1-based block indices.
    """
    root = _unwrap(model)
    blocks = _find_block_list(root)
    if blocks is None:
        logger.warning("no transformer block list found; falling back to all norm affine params")
        return select_norm_affine_params(root)

    params: List[nn.Parameter] = []
    seen = set()
    start = max(0, block_start - 1)
    end = min(len(blocks), start + num_blocks)
    for block in list(blocks)[start:end]:
        for p in _norm_affine_params(block):
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)

    if include_final_norm:
        for name, module in root.named_modules():
            if name.split(".")[-1] in ("norm", "fc_norm", "final_norm") and isinstance(
                module, (nn.LayerNorm, nn.GroupNorm)
            ):
                if getattr(module, "weight", None) is not None and id(module.weight) not in seen:
                    seen.add(id(module.weight))
                    params.append(module.weight)
                if getattr(module, "bias", None) is not None and id(module.bias) not in seen:
                    seen.add(id(module.bias))
                    params.append(module.bias)

    if not params:
        logger.warning("block-limited selection produced no parameters; using all norm affine params")
        return select_norm_affine_params(root)
    return params


def select_norm_affine_params(model: nn.Module) -> List[nn.Parameter]:
    """All normalisation affine parameters (TENT-style trainable set)."""
    return _norm_affine_params(_unwrap(model))


def reliable_entropy(
    logits: torch.Tensor,
    entropy_threshold: float,
    eps: float = DEFAULT_ENTROPY_EPS,
) -> Dict[str, torch.Tensor]:
    """SAR "reliable entropy" mask.

    Returns a dict with the per-sample entropy ``entropy`` (``[B]``), the
    boolean ``mask`` of samples whose entropy is *below* ``E_0`` (``[B]``), and
    ``unreliable_fraction`` (scalar tensor) used for model restoration.
    """
    ent = softmax_entropy(logits, eps=eps)
    mask = ent < entropy_threshold
    unreliable_fraction = (~mask).float().mean()
    return {"entropy": ent, "mask": mask, "unreliable_fraction": unreliable_fraction}


def sample_weights(
    entropy: torch.Tensor,
    entropy_threshold: float,
    mode: str = "exp",
) -> torch.Tensor:
    """Per-sample weights that de-emphasise unreliable (high-entropy) samples.

    ``mode='exp'``  -> ``exp(-(H - E_0))`` clipped at 1.0 (SAR paper).
    ``mode='mask'`` -> hard 1/0 gate (strict reliable-entropy filtering).
    """
    if mode == "exp":
        w = torch.exp(-(entropy - entropy_threshold)).clamp(max=1.0)
        return w
    if mode == "mask":
        return (entropy < entropy_threshold).float()
    raise ValueError(f"unknown weighting mode '{mode}'")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SARConfig:
    """Hyper-parameters for the SAR baseline (FOA Appendix B.2)."""

    lr: float = DEFAULT_LR
    momentum: float = DEFAULT_MOMENTUM
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    rho: float = DEFAULT_RHO
    eta: float = DEFAULT_ETA
    entropy_factor: float = DEFAULT_ENTROPY_FACTOR
    entropy_threshold: Optional[float] = None
    num_classes: int = 1000
    num_blocks: int = DEFAULT_NUM_BLOCKS
    block_start: int = 1
    include_final_norm: bool = False
    weighting: str = "exp"
    restore: bool = True
    episodic: bool = False
    entropy_eps: float = DEFAULT_ENTROPY_EPS
    grad_clip: Optional[float] = None
    adapt_after_loss: bool = True

    def resolved_entropy_threshold(self) -> float:
        if self.entropy_threshold is not None:
            return float(self.entropy_threshold)
        return float(self.entropy_factor) * math.log(max(2, int(self.num_classes)))

    @classmethod
    def from_config(cls, cfg: Any) -> "SARConfig":
        """Build from a (nested) FOA config, reading a ``baselines.sar`` block."""
        kwargs: Dict[str, Any] = {}
        block: Any = None
        if cfg is not None:
            baselines = None
            if isinstance(cfg, dict):
                baselines = cfg.get("baselines")
                if isinstance(baselines, dict):
                    block = baselines.get("sar")
            else:  # attribute style
                baselines = getattr(cfg, "baselines", None)
                if baselines is not None:
                    block = getattr(baselines, "sar", None)

            # fall back to legacy top-level 'sar' key
            if block is None:
                if isinstance(cfg, dict):
                    block = cfg.get("sar")
                else:
                    block = getattr(cfg, "sar", None)

            # dataset-specific class count
            data = cfg.get("data") if isinstance(cfg, dict) else getattr(cfg, "data", None)
            if data is not None:
                n_eval = (
                    (data.get("num_classes_eval") if isinstance(data, dict) else getattr(data, "num_classes_eval", None))
                )
                if n_eval:
                    kwargs["num_classes"] = int(n_eval)

        aliases = {
            "learning_rate": "lr",
            "th": "entropy_threshold",
            "e0": "entropy_threshold",
            "trainable_blocks": "num_blocks",
        }
        if block is not None:
            items = block.items() if isinstance(block, dict) else vars(block).items()
            for key, value in items:
                key = aliases.get(str(key), str(key))
                if key in cls.__dataclass_fields__ and value is not None:
                    kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["entropy_threshold_resolved"] = self.resolved_entropy_threshold()
        return d


# --------------------------------------------------------------------------- #
# SAR
# --------------------------------------------------------------------------- #
class SAR:
    """Sharpness-Aware and Reliable entropy minimisation (SAR) TTA wrapper.

    Protocol per test batch (``step``):

      1. Forward the current batch; compute per-sample entropy.
      2. Reliable entropy: weight ``w_t = exp(-(H_t - E_0))`` (or a hard mask),
         used to re-weight the entropy objective.
      3. Sharpness-aware update: ascend to the SAM neighbour
         ``w + rho * g / ||g||``, recompute the weighted entropy there, take the
         gradient of that perturbed loss, and descend from the original point.
      4. Model restoration: if the fraction of unreliable samples exceeds
         ``eta``, reset the adaptable parameters to their source values (and
         clear the optimiser momentum), protecting against collapse.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SARConfig] = None,
        *,
        lr: Optional[float] = None,
        momentum: Optional[float] = None,
        batch_size: Optional[int] = None,
        weight_decay: Optional[float] = None,
        rho: Optional[float] = None,
        eta: Optional[float] = None,
        num_classes: Optional[int] = None,
        entropy_threshold: Optional[float] = None,
        num_blocks: Optional[int] = None,
        block_start: Optional[int] = None,
        include_final_norm: Optional[bool] = None,
        weighting: Optional[str] = None,
        restore: Optional[bool] = None,
        episodic: Optional[bool] = None,
        device: Optional[Union[str, torch.device]] = None,
        grad_clip: Optional[float] = None,
        adapt_after_loss: Optional[bool] = None,
    ) -> None:
        base = config or SARConfig()
        overrides = {
            "lr": lr,
            "momentum": momentum,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "rho": rho,
            "eta": eta,
            "num_classes": num_classes,
            "entropy_threshold": entropy_threshold,
            "num_blocks": num_blocks,
            "block_start": block_start,
            "include_final_norm": include_final_norm,
            "weighting": weighting,
            "restore": restore,
            "episodic": episodic,
            "grad_clip": grad_clip,
            "adapt_after_loss": adapt_after_loss,
        }
        for key, value in overrides.items():
            if value is not None:
                setattr(base, key, value)
        self.cfg = base
        if base.entropy_threshold is None:
            base.entropy_threshold = base.resolved_entropy_threshold()

        self.model = model
        self.device = torch.device(device) if device is not None else _infer_device(model)
        self.model.to(self.device)
        self.model.eval()

        # Never adapt the classifier head / backbone weights other than LN affine.
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.params: List[nn.Parameter] = select_block_norm_affine_params(
            model,
            num_blocks=base.num_blocks,
            block_start=base.block_start,
            include_final_norm=base.include_final_norm,
        )
        if not self.params:
            raise RuntimeError("SAR: no adaptable normalisation affine parameters found")
        for param in self.params:
            param.requires_grad_(True)

        self._param_names: List[str] = []
        name_map = {id(p): n for n, p in _unwrap(model).named_parameters()}
        for idx, param in enumerate(self.params):
            self._param_names.append(name_map.get(id(param), f"adaptable_param_{idx}"))

        # Keep an immutable copy of the source (pre-adaptation) weights.
        self._source_state: List[torch.Tensor] = [p.detach().clone() for p in self.params]

        self.optimizer = torch.optim.SGD(
            self.params,
            lr=float(base.lr),
            momentum=float(base.momentum),
            weight_decay=float(base.weight_decay),
        )

        self._num_restores = 0
        self._num_steps = 0

    # ------------------------------------------------------------------ #
    # State handling
    # ------------------------------------------------------------------ #
    def trainable_parameter_names(self) -> List[str]:
        return list(self._param_names)

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.params))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "params": [p.detach().clone() for p in self.params],
            "source_params": [s.clone() for s in self._source_state],
            "optimizer": self.optimizer.state_dict(),
            "num_steps": self._num_steps,
            "num_restores": self._num_restores,
            "config": self.cfg.to_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        params = state.get("params")
        if params is not None:
            for p, v in zip(self.params, params):
                with torch.no_grad():
                    p.copy_(v.to(p.device, p.dtype))
        source = state.get("source_params")
        if source is not None:
            self._source_state = [s.detach().clone() for s in source]
        opt = state.get("optimizer")
        if opt is not None and opt:  # avoid empty dicts from fresh optimizers
            try:
                self.optimizer.load_state_dict(opt)
            except (ValueError, KeyError) as exc:  # pragma: no cover - defensive
                logger.warning("could not restore SAR optimizer state: %s", exc)
        self._num_steps = int(state.get("num_steps", self._num_steps))
        self._num_restores = int(state.get("num_restores", self._num_restores))

    def load_source_state(self, state_dict: Optional[Mapping_like] = None) -> None:  # type: ignore[valid-type]
        """Restore the frozen source weights (optionally from a saved mapping)."""
        if state_dict is None:
            for p, src in zip(self.params, self._source_state):
                with torch.no_grad():
                    p.copy_(src)
        else:
            self.load_state_dict({"params": state_dict})  # type: ignore[arg-type]
        self._clear_optimizer_state()

    def _clear_optimizer_state(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.optimizer.state.pop(p, None)

    def reset(self) -> None:
        """Episodic reset: rewind to source weights and clear optimiser state."""
        self.load_source_state()
        self._num_restores += 1

    def freeze_source(self) -> None:
        """Re-snapshot the current adaptable weights as the new source state."""
        self._source_state = [p.detach().clone() for p in self.params]

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        return _logits_from_output(output)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        self.model.eval()
        return self._forward(images)

    def loss(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reliable-entropy objective (weighted) on the current weights."""
        logits = self._forward(images)
        info = reliable_entropy(logits, self.cfg.entropy_threshold, eps=self.cfg.entropy_eps)
        weights = sample_weights(info["entropy"], self.cfg.entropy_threshold, mode=self.cfg.weighting)
        ent = softmax_entropy(logits, eps=self.cfg.entropy_eps)
        if float(weights.sum()) <= 0.0:
            loss = ent.mean() * 0.0
        else:
            loss = (ent * weights).sum() / weights.sum().clamp_min(1e-12)
        info["weights"] = weights
        info["num_reliable"] = int(info["mask"].sum().item())
        return loss, info

    # ------------------------------------------------------------------ #
    # Adaptation
    # ------------------------------------------------------------------ #
    def _sam_perturb(self, images: torch.Tensor) -> Tuple[Dict[str, Any], float]:
        """First SAM step: gradient ascent to the neighbourhood point."""
        self.optimizer.zero_grad(set_to_none=True)
        loss, info = self.loss(images)
        if loss.requires_grad:
            loss.backward()
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg.grad_clip))
        grad_norm = self._grad_norm()
        scale = float(self.cfg.rho) / (grad_norm + 1e-12)
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                p.add_(p.grad * scale)
        return info, grad_norm

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().item())
        return math.sqrt(total)

    def _restore_perturbation(self, saved_grads: Sequence[Optional[torch.Tensor]]) -> None:
        with torch.no_grad():
            for p, g in zip(self.params, saved_grads):
                if g is None:
                    p.grad = None
                    continue
                p.add_(-g * (float(self.cfg.rho) / (math.sqrt(float(g.pow(2).sum().item())) + 1e-12)))
                p.grad = g.clone()

    def adapt(self, images: torch.Tensor) -> Dict[str, Any]:
        """One SAR optimisation step on a test batch."""
        images = images.to(self.device)
        self.model.train()
        self.model.eval()  # keep dropout/BN statistics frozen; only affine params adapt

        # --- step 1: perturb towards the SAM neighbour ------------------- #
        info, grad_norm = self._sam_perturb(images)
        saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in self.params]
        saved_perturb = [p.grad.detach().clone() if p.grad is not None else None for p in self.params]

        # --- step 2: forward at the perturbed point ---------------------- #
        self.optimizer.zero_grad(set_to_none=True)
        perturbed_loss, perturbed_info = self.loss(images)
        if perturbed_loss.requires_grad:
            perturbed_loss.backward()

        # --- step 3: return to the original point, keep perturbed grads -- #
        with torch.no_grad():
            for p, g in zip(self.params, saved_perturb):
                if g is None:
                    continue
                p.add_(-g * (float(self.cfg.rho) / (math.sqrt(float(g.pow(2).sum().item())) + 1e-12)))

        self.optimizer.step()
        self._num_steps += 1

        # --- step 4: model restoration on excessive unreliable entropy --- #
        restored = False
        if self.cfg.restore and float(info["unreliable_fraction"]) > float(self.cfg.eta):
            self.load_source_state()
            restored = True

        if self.cfg.episodic:
            self.reset()

        return {
            "loss": float(perturbed_loss.detach().item()),
            "sam_loss": float(perturbed_info["weights"].sum().item()),
            "entropy": float(info["entropy"].mean().item()),
            "grad_norm": float(grad_norm),
            "unreliable_fraction": float(info["unreliable_fraction"].item()),
            "num_reliable": info["num_reliable"],
            "restored": restored,
            "entropy_threshold": float(self.cfg.entropy_threshold),
        }

    # ------------------------------------------------------------------ #
    # Public batch protocol
    # ------------------------------------------------------------------ #
    def step(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adapt on ``images`` (unsupervised) and return the updated logits."""
        info = self.adapt(images)
        with torch.no_grad():
            self.model.eval()
            logits = self._forward(images)
        self.last_info = info
        return logits

    __call__ = step

    def extra_repr(self) -> str:
        return (
            f"lr={self.cfg.lr}, momentum={self.cfg.momentum}, rho={self.cfg.rho}, "
            f"eta={self.cfg.eta}, E0={self.cfg.entropy_threshold:.4f}, "
            f"blocks={self.cfg.block_start}..{self.cfg.block_start + self.cfg.num_blocks - 1}"
        )


Mapping_like = Union[Dict[str, Any], List[torch.Tensor]]


def _infer_device(model: nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_sar(model: Optional[nn.Module] = None, cfg: Any = None, **kwargs: Any) -> SAR:
    """Config-driven factory matching the runner's ``build_<name>`` convention."""
    if model is None:
        model = kwargs.pop("model", None)
    if model is None:
        raise ValueError("build_sar requires a model instance")
    config = kwargs.pop("config", None)
    if config is None:
        config = SARConfig.from_config(cfg)
    device = kwargs.pop("device", None)
    if device is not None:
        kwargs.setdefault("device", device)
    return SAR(model, config, **kwargs)


build_baseline = build_sar
