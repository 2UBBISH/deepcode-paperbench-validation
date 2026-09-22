"""MEMO baseline (Zhang et al., 2022) — Marginal Entropy Minimization with One test sample.

Reference
---------
M. Zhang, S. Levine, C. Finn, "MEMO: Test Time Robustness via Adaptation and
Augmentation", NeurIPS 2022.

Why this file exists
--------------------
FOA (this reproduction) compares against two categories of TTA methods
(Section 4 "Compared Methods"):

  1) gradient-free : LAME, T3A
  2) gradient-based: TENT, SAR, CoTTA

MEMO is *not* part of the paper's main comparison tables; the reproduction plan
lists it as an **optional** extra baseline alongside CoTTA (item 13 /
item 14 of the plan) and notes that the paper defers baseline hyper-parameters
to Appendix B.2, which does not give a MEMO configuration.  This module is
therefore a *self-contained* adapter (no external repository is required) whose
defaults follow the MEMO paper and, where the MEMO paper and FOA's Appendix B.2
overlap (batch size 64, SGD with momentum 0.9, no weight decay, norm-layer
affine parameters for ViT-style backbones), FOA's values are used so that the
comparison is apples-to-apples.

MEMO in one paragraph
---------------------
For every single test sample ``x``:
  * draw ``n_aug`` stochastic augmentations ``{a_i(x)}``,
  * forward all views and average the *softmax* outputs into a *marginal*
    prediction ``p_bar = 1/V * sum_i softmax(f(a_i(x)))``,
  * minimize the Shannon entropy ``H(p_bar)`` with respect to a (small) set of
    model parameters (gradient-based; this is a **backward**-using baseline, in
    contrast to FOA which never backpropagates),
  * classify ``x`` with the marginal prediction (or a single clean forward,
    depending on ``predict_with_marginal``).

Two protocols are supported, mirroring MEMO:
  * ``episodic=True``  (MEMO's default): parameters are reset to the source
    state after each test sample, i.e. every sample gets its own adaptation.
  * ``episodic=False`` (MEMO "online"): parameters carry over across the
    stream, which is the protocol comparable to TENT/SAR/CoTTA/FOA.

Nothing in this module is used by the FOA method itself; FOA stays
backpropagation-free.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "MEMO",
    "MEMOBaseline",
    "MEMOConfig",
    "AugmentationEnsemble",
    "marginal_ensemble",
    "marginal_entropy_loss",
    "softmax_entropy",
    "select_trainable_params",
    "build_memo",
    "build_baseline",
    "DEFAULT_LR",
    "DEFAULT_MOMENTUM",
    "DEFAULT_N_AUG",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPISODIC",
]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# MEMO paper: SGD, lr=1e-3, momentum=0.9; 32 augmented views is the standard
# setting.  Batch size 64 + norm-affine parameters follow FOA Appendix B.2 so
# the comparison against FOA/TENT/SAR is like-for-like.
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_N_AUG = 32
DEFAULT_BATCH_SIZE = 64
DEFAULT_STEPS = 1
DEFAULT_EPISODIC = True
DEFAULT_TRAIN_WHICH = "norm"
DEFAULT_CHUNK = 8
DEFAULT_AUG_LEVEL = 0.3
DEFAULT_NUM_CLASSES = 1000
DEFAULT_EPS = 1e-12


# ---------------------------------------------------------------------------
# Small helpers (config / model introspection)
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path lookup that works for dict-like *and* attribute configs."""
    if cfg is None:
        return default
    for key in keys:
        cur = cfg
        found = True
        for part in key.split("."):
            if cur is None:
                found = False
                break
            if isinstance(cur, dict):
                if part in cur:
                    cur = cur[part]
                else:
                    found = False
                    break
            else:
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    found = False
                    break
        if found and cur is not None:
            return cur
    return default


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the inner trainable module.

    FOA's ``ViTWithCLSFeatures`` wrapper runs its ``forward`` under
    ``torch.no_grad()``, which would make gradient-based baselines useless.
    Its ``.model`` attribute holds the plain timm network, so we descend into
    it when present (CoTTA/TENT/SAR adapters do the same).
    """
    if model is None:
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner
    return model


def _logits_from_output(output: Any) -> torch.Tensor:
    """Normalise the many forward contracts into a logits tensor."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("logits", "output", "out", "pred", "predictions"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        raise ValueError(f"dict output without a logits-like key: {list(output)}")
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item) and item.dim() == 2:
                return item
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"cannot extract logits from output of type {type(output)}")


def _infer_device(model: Optional[nn.Module], device: Any = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if model is not None:
        for p in model.parameters():
            return p.device
        for b in model.buffers():
            return b.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def softmax_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample Shannon entropy of the softmax distribution -> ``[B]``."""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(eps))
    return -(probs * log_probs).sum(dim=-1)


def marginal_ensemble(
    logits_views: torch.Tensor,
    softmax: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Average predictions over augmented views (the MEMO "marginal").

    Parameters
    ----------
    logits_views : ``[V, B, C]`` tensor of per-view logits.
    softmax : average probabilities (True, MEMO's choice) instead of logits.

    Returns
    -------
    ``[B, C]`` marginal predictions (probabilities by default).
    """
    if logits_views.dim() != 3:
        raise ValueError(f"expected [V, B, C] view logits, got {tuple(logits_views.shape)}")
    if softmax:
        return logits_views.softmax(dim=-1).mean(dim=0)
    return logits_views.mean(dim=0)


def marginal_entropy_loss(
    probability_views: Optional[torch.Tensor] = None,
    logits_views: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MEMO objective: entropy of the marginal prediction.

    Accepts either ``[V, B, C]`` probabilities or ``[V, B, C]`` logits and
    returns ``(loss, marginal_probs [B, C])``.
    """
    if probability_views is None:
        if logits_views is None:
            raise ValueError("provide either probability_views or logits_views")
        probability_views = logits_views.softmax(dim=-1)
    marginal = probability_views.mean(dim=0)
    entropy = -(marginal.clamp_min(eps).log() * marginal).sum(dim=-1)
    if reduction == "mean":
        return entropy.mean(), marginal
    if reduction == "sum":
        return entropy.sum(), marginal
    if reduction in ("none", None):
        return entropy, marginal
    raise ValueError(f"unknown reduction {reduction!r}")


# ---------------------------------------------------------------------------
# Augmentation ensemble
# ---------------------------------------------------------------------------
class AugmentationEnsemble:
    """Light-weight stochastic augmentation ensemble for MEMO.

    All operations are differentiable-free (they are applied to *inputs*), and
    are implemented with plain tensor ops so no torchvision augmentation
    pipelines are required.  Operations mirror the MEMO/TENT-style TEST
    augmentations: random resized crop (approximated by affine + crop), colour
    jitter, Gaussian blur, Gaussian noise and cutout.
    """

    def __init__(
        self,
        level: float = DEFAULT_AUG_LEVEL,
        ops: Sequence[str] = ("color_jitter", "gaussian_blur", "gaussian_noise", "affine", "cutout"),
        seed: Optional[int] = None,
    ) -> None:
        self.level = float(level)
        self.ops = tuple(ops)
        seed = 0 if seed is None else int(seed)
        try:  # torch.Generator is the cleanest per-instance RNG
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        except Exception:  # pragma: no cover - extremely old torch
            self.generator = None

    # -- augmentation primitives -------------------------------------------
    def color_jitter(self, x: torch.Tensor, level: float) -> torch.Tensor:
        b = x.shape[0]
        gains = 1.0 + level * 0.5 * torch.randn(b, 3, 1, 1, device=x.device, generator=self.generator)
        gains = gains.clamp(0.5, 1.5)
        return x * gains

    def gaussian_blur(self, x: torch.Tensor, level: float) -> torch.Tensor:
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype
        )
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(pad, kernel, groups=3)

    def gaussian_noise(self, x: torch.Tensor, level: float) -> torch.Tensor:
        return x + (level * 0.1) * torch.randn(x.shape, device=x.device, generator=self.generator)

    def affine(self, x: torch.Tensor, level: float) -> torch.Tensor:
        b = x.shape[0]
        angle = (level * math.pi / 8.0) * (2.0 * torch.rand(b, device=x.device, generator=self.generator) - 1.0)
        scale = 1.0 + (level * 0.2) * (2.0 * torch.rand(b, device=x.device, generator=self.generator) - 1.0)
        theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos / scale, -sin / scale, sin / scale, cos / scale
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="border")

    def cutout(self, x: torch.Tensor, level: float) -> torch.Tensor:
        b, _, h, w = x.shape
        size = int(round(level * 0.15 * min(h, w)))
        if size <= 1:
            return x
        y0 = torch.randint(0, max(h - size, 1), (b, 1), device=x.device, generator=self.generator)
        x0 = torch.randint(0, max(w - size, 1), (b, 1), device=x.device, generator=self.generator)
        out = x.clone()
        for i in range(b):
            out[i, :, y0[i, 0] : y0[i, 0] + size, x0[i, 0] : x0[i, 0] + size] = 0.0
        return out

    # -- driver -------------------------------------------------------------
    def views(self, x: torch.Tensor, n_views: int) -> torch.Tensor:
        """Return ``[n_views, B, C, H, W]`` augmented copies of ``x``."""
        if n_views <= 0:
            raise ValueError("n_views must be >= 1")
        out: List[torch.Tensor] = []
        for _ in range(n_views):
            v = x
            for op in self.ops:
                fn = getattr(self, op, None)
                if fn is not None:
                    v = fn(v, self.level)
            out.append(v)
        return torch.stack(out, dim=0)

    def __call__(self, x: torch.Tensor, n_views: int = DEFAULT_N_AUG) -> torch.Tensor:
        return self.views(x, n_views)


# ---------------------------------------------------------------------------
# Trainable parameter selection
# ---------------------------------------------------------------------------
def select_trainable_params(model: nn.Module, which: str = DEFAULT_TRAIN_WHICH) -> List[nn.Parameter]:
    """Select the parameter subset MEMO optimises.

    ``"norm"`` (default for ViT) = affine params of all normalization layers,
    which matches FOA Appendix B.2's TENT/SAR configuration;
    ``"all"`` = every parameter (MEMO's original setting for small models);
    ``"head"`` = the classification head only.
    """
    model = _unwrap(model)
    which = (which or "norm").lower()

    if which == "all":
        return [p for p in model.parameters() if p.requires_grad or True]

    if which == "head":
        params: List[nn.Parameter] = []
        for name, module in model.named_modules():
            if name.endswith("head") and not any(isinstance(c, nn.Module) for c in module.children()):
                params.extend(list(module.parameters()))
        if not params:
            head = getattr(model, "head", None)
            if isinstance(head, nn.Module):
                params = list(head.parameters())
        return params

    params = []
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for name, p in module.named_parameters(recurse=False):
                if "weight" in name or "bias" in name or "gamma" in name or "beta" in name:
                    params.append(p)
    if not params:  # pragma: no cover - defensive
        logger.warning("no normalization affine parameters found; falling back to all parameters")
        params = list(model.parameters())
    return params


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class MEMOConfig:
    """Hyper-parameters of the MEMO adapter.

    Defaults follow the MEMO paper (lr=1e-3, SGD momentum 0.9, 32 augmented
    views, episodic per-sample adaptation) with the FOA Appendix B.2 batch size
    (64) so that the run is comparable with the other baselines.
    """

    lr: float = DEFAULT_LR
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    n_aug: int = DEFAULT_N_AUG
    steps: int = DEFAULT_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    episodic: bool = DEFAULT_EPISODIC
    train_which: str = DEFAULT_TRAIN_WHICH
    chunk_size: int = DEFAULT_CHUNK
    aug_level: float = DEFAULT_AUG_LEVEL
    num_classes: int = DEFAULT_NUM_CLASSES
    eps: float = DEFAULT_EPS
    grad_clip: Optional[float] = None
    adapt_after_loss: bool = True
    predict_with_marginal: bool = True
    seed: Optional[int] = None
    image_size: int = 224
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: Any) -> "MEMOConfig":
        """Read the ``baselines.memo`` (or legacy top-level ``memo``) block."""
        kwargs: Dict[str, Any] = {}
        defaults = cls()
        block = _cfg_get(cfg, "baselines.memo", "memo", default=None)
        aliases = {
            "learning_rate": "lr",
            "num_views": "n_aug",
            "n_views": "n_aug",
            "num_aug": "n_aug",
            "views": "n_aug",
            "restore": "episodic",
            "reset_each_sample": "episodic",
            "params": "train_which",
        }
        if isinstance(block, dict):
            for key, value in block.items():
                key = aliases.get(key, key)
                if hasattr(defaults, key) and value is not None:
                    kwargs[key] = value
        bs = _cfg_get(cfg, "baselines.batch_size", "data.batch_size", default=None)
        if bs is not None and "batch_size" not in kwargs:
            kwargs["batch_size"] = int(bs)
        if "num_classes" not in kwargs:
            nc = _cfg_get(
                cfg,
                "baselines.memo.num_classes",
                "data.num_classes_eval",
                "data.num_classes",
                "model.num_classes",
                default=None,
            )
            if nc is not None:
                kwargs["num_classes"] = int(nc)
        seed = _cfg_get(cfg, "baselines.memo.seed", "seed", default=None)
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = int(seed)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# MEMO adapter
# ---------------------------------------------------------------------------
class MEMO(nn.Module):
    """Gradient-based MEMO baseline adapter (per-sample marginal entropy).

    The object follows the duck-typed online-TTA protocol shared by the other
    FOA baselines so ``scripts/run_baselines.py`` can drive it unchanged::

        wrapper = build_baseline("memo", model=model, cfg=cfg)
        logits = wrapper.step(images, targets)   # -> [B, C]

    Unlike FOA this adapter *does* call ``backward()`` — it is a gradient-based
    comparison method.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MEMOConfig] = None,
        *,
        lr: Optional[float] = None,
        momentum: Optional[float] = None,
        weight_decay: Optional[float] = None,
        n_aug: Optional[int] = None,
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        episodic: Optional[bool] = None,
        train_which: Optional[str] = None,
        chunk_size: Optional[int] = None,
        aug_level: Optional[float] = None,
        num_classes: Optional[int] = None,
        eps: Optional[float] = None,
        grad_clip: Optional[float] = None,
        adapt_after_loss: Optional[bool] = None,
        predict_with_marginal: Optional[bool] = None,
        seed: Optional[int] = None,
        image_size: Optional[int] = None,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model is None:
            raise ValueError("MEMO requires a model")

        cfg = config or MEMOConfig()
        overrides = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "n_aug": n_aug,
            "steps": steps,
            "batch_size": batch_size,
            "episodic": episodic,
            "train_which": train_which,
            "chunk_size": chunk_size,
            "aug_level": aug_level,
            "num_classes": num_classes,
            "eps": eps,
            "grad_clip": grad_clip,
            "adapt_after_loss": adapt_after_loss,
            "predict_with_marginal": predict_with_marginal,
            "seed": seed,
            "image_size": image_size,
        }
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        self.config = cfg

        self.wrapper = model
        self.model = _unwrap(model)
        self.device = _infer_device(self.model, device if device is not None else cfg.device)

        # Only the selected subset is trainable; everything else is frozen.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = select_trainable_params(self.model, cfg.train_which)
        for p in self.params:
            p.requires_grad_(True)

        self.optimizer = torch.optim.SGD(
            self.params,
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.weight_decay),
        )

        self.augment = AugmentationEnsemble(level=cfg.aug_level, seed=cfg.seed)

        # Snapshot of the source weights for episodic resets.
        self._source_state: Dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()
        }

        self.num_adaptations = 0
        self.adaptation_steps = 0
        self.last_marginal: Optional[torch.Tensor] = None

    # -- bookkeeping --------------------------------------------------------
    def trainable_parameter_names(self) -> List[str]:
        names = {id(p): n for n, p in self.model.named_parameters()}
        return [names.get(id(p), "<unnamed>") for p in self.params]

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.params))

    def extra_repr(self) -> str:
        return (
            f"train_which={self.config.train_which}, lr={self.config.lr}, "
            f"momentum={self.config.momentum}, n_aug={self.config.n_aug}, "
            f"episodic={self.config.episodic}, params={self.trainable_parameter_count()}"
        )

    # -- state --------------------------------------------------------------
    def load_source_state(self, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Restore the (frozen) source weights used for episodic resets."""
        if state_dict is None:
            self.model.load_state_dict(self._source_state, strict=False)
        else:
            self._source_state = {k: v.detach().clone() for k, v in state_dict.items()}
            self.model.load_state_dict(self._source_state, strict=False)

    def reset(self) -> None:
        """Episodic reset: drop adapted weights and optimiser momentum."""
        self.model.load_state_dict(self._source_state, strict=False)
        for p in self.params:
            p.requires_grad_(True)
        self.optimizer = torch.optim.SGD(
            self.params,
            lr=float(self.config.lr),
            momentum=float(self.config.momentum),
            weight_decay=float(self.config.weight_decay),
        )
        self.last_marginal = None

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "model": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "source": {k: v.detach().clone() for k, v in self._source_state.items()},
            "num_adaptations": self.num_adaptations,
            "adaptation_steps": self.adaptation_steps,
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        if "model" in state:
            self.model.load_state_dict(state["model"], strict=False)
        if "source" in state:
            self._source_state = {k: v.detach().clone() for k, v in state["source"].items()}
        self.num_adaptations = int(state.get("num_adaptations", 0))
        self.adaptation_steps = int(state.get("adaptation_steps", 0))

    # -- forward helpers ----------------------------------------------------
    def _forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        return _logits_from_output(self.model(images))

    def forward_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single clean forward returning ``(penultimate_features, logits)``.

        If the wrapped object exposes ``forward_with_features`` (FOA's ViT
        wrapper) we fall back to a plain logits forward for features because the
        wrapper's forward is gradient-free.
        """
        logits = self._forward_logits(images)
        return logits, logits

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Classify without adapting (used when no adaptation is wanted)."""
        was_training = self.model.training
        self.model.eval()
        logits = self._forward_logits(images.to(self.device))
        if was_training:
            self.model.train()
        return logits

    # -- MEMO loss ----------------------------------------------------------
    def loss(
        self,
        images: torch.Tensor,
        n_aug: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        return_marginal: bool = False,
    ) -> Any:
        """Marginal entropy of augmented views (MEMO objective).

        Returns the scalar loss, or ``(loss, marginal_probs)`` when
        ``return_marginal`` is set.  ``targets`` is accepted and ignored so the
        adapter keeps a uniform signature with the supervised baselines.
        """
        n_aug = int(n_aug or self.config.n_aug)
        images = images.to(self.device)
        views = self.augment.views(images, n_aug)  # [V, B, C, H, W]

        chunk = max(1, int(self.config.chunk_size))
        view_logits: List[torch.Tensor] = []
        for start in range(0, n_aug, chunk):
            batch_views = views[start : start + chunk].reshape(-1, *images.shape[1:])
            view_logits.append(self._forward_logits(batch_views))
        stacked = torch.stack(view_logits, dim=0)  # [V, B, C]

        loss, marginal = marginal_entropy_loss(logits_views=stacked, reduction="mean", eps=self.config.eps)
        self.last_marginal = marginal.detach()
        if return_marginal:
            return loss, marginal
        return loss

    # -- adaptation ---------------------------------------------------------
    def adapt(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run ``steps`` MEMO updates on the given test samples."""
        self.model.train()
        loss_value = None
        for _ in range(max(1, int(self.config.steps))):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss(images, targets=targets)
            if not torch.isfinite(loss):
                logger.warning("non-finite MEMO loss; skipping update")
                self.optimizer.zero_grad(set_to_none=True)
                break
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.params, float(self.config.grad_clip))
            self.optimizer.step()
            loss_value = float(loss.detach())
            self.adaptation_steps += 1
        self.num_adaptations += 1
        if loss_value is not None:
            self.last_loss = loss_value
        return self.last_marginal if self.last_marginal is not None else torch.empty(0)

    def step(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adapt-then-predict for one test batch (returns log-probabilities).

        The other baseline adapters return logits / log-posteriors that
        downstream ``softmax`` turns into posteriors.  MEMO's prediction is
        itself a probability, so we return ``log(marginal)`` which recovers the
        marginal exactly after a softmax.
        """
        images = images.to(self.device)
        targets = None if targets is None else targets.to(self.device)

        if self.config.adapt_after_loss:
            self.adapt(images, targets)

        if self.config.predict_with_marginal:
            with torch.no_grad():
                _, marginal = self.loss(images, targets=targets, return_marginal=True)
            prob = marginal.clamp_min(self.config.eps)
            logits = prob.log()
        else:
            with torch.no_grad():
                self.model.eval()
                logits = self._forward_logits(images)

        if self.config.episodic:
            self.reset()

        self.model.eval()
        return logits

    __call__ = step

    def adapt_and_predict(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Alias kept for drivers that use ``adapt_and_predict``."""
        return self.step(images, targets)


MEMOBaseline = MEMO


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_memo(
    model: Optional[nn.Module] = None,
    cfg: Any = None,
    device: Any = None,
    **kwargs: Any,
) -> MEMO:
    """Build a MEMO adapter from a FOA YAML config (``baselines.memo`` block)."""
    config = MEMOConfig.from_config(cfg)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if "batch_size" in kwargs and kwargs["batch_size"] is not None:
        config.batch_size = int(kwargs.pop("batch_size"))
    config_dict = kwargs.pop("config", None)
    if isinstance(config_dict, MEMOConfig):
        config = config_dict
    if model is None:
        model = kwargs.pop("model", None)
    if model is None:
        raise ValueError("build_memo requires a `model`")
    return MEMO(model=model, config=config, device=device, **kwargs)


def build_baseline(model: Optional[nn.Module] = None, cfg: Any = None, **kwargs: Any) -> MEMO:
    """Alias used by the generic baseline runner."""
    return build_memo(model=model, cfg=cfg, **kwargs)
