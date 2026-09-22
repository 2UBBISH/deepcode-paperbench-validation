"""ANT (Adapted Noise-guided Transfer) trainer for DPMs-ANT.

Implements Algorithm 1 and Equation (8) of "Adapting Pretrained Diffusion Models
for Few-Shot Image Generation" (DPMs-ANT):

    L(psi) = E_{t, x_0}[ || eps* - eps_{theta,psi}(x_t*, t)
                          - sigma_hat_t^2 * gamma * grad_{x_t*} log p_phi(y=T | x_t*) ||^2 ]
    s.t.  eps* = argmax_eps || eps - eps_theta(sqrt(alpha_bar_t) x_0
                                              + sqrt(1-alpha_bar_t) eps, t) ||^2
          with Norm(eps*) ~ (0, I)

The pretrained backbone ``theta`` and the classifier ``phi`` are frozen; only the
adaptor parameters ``psi`` are updated:

    x_t^l = theta^l(x_t^{l-1}) + psi^l(x_t^{l-1})

Everything here is backbone agnostic: any object exposing ``epsilon_theta(x, t)``
(the :class:`~dpm_ant.models.unet_loader.FrozenDDPMUNet`,
:class:`~dpm_ant.models.ldm_loader.FrozenLDMUNet` or a plain ``nn.Module``) can be
trained.

Design notes / resolved ambiguities (see reproduction plan):
  * Optimizer is Adam (paper does not name it) with the reported learning rates
    (DDPM 5e-5, LDM 1e-5) unless overridden per task.
  * Batch size 40, ~300 outer iterations by default (160-500 per task).
  * ``use_adv_noise=False`` reproduces the "DPMs-ANT w/o AN" ablation (plain
    Gaussian noise + similarity-guided loss only).
  * ``only_adaptor=False`` / ``freeze_backbone=False`` reproduces "direct
    full-model fine-tuning" and "adaptor-only" ablations.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion.schedule import NoiseSchedule, build_schedule
from ..diffusion.gaussian_diffusion import GaussianDiffusion, build_diffusion
from .adv_noise import (
    AdversarialNoiseSelector,
    build_adv_noise_selector,
    build_xt_from_noise,
)

try:  # pragma: no cover - optional, implemented in the same package
    from .sg_loss import SimilarityGuidedLoss, build_sg_loss  # type: ignore
except Exception:  # pragma: no cover
    SimilarityGuidedLoss = None  # type: ignore
    build_sg_loss = None  # type: ignore


__all__ = [
    "ANTConfig",
    "ANTTrainer",
    "train_ant",
    "build_ant_trainer",
    "adaptor_trainable_parameters",
    "parameter_rate",
]

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def adaptor_trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Return the adaptor-only trainable parameters of ``module``.

    Parameters whose (dotted) name contains ``.adaptor.`` are considered part of
    the adaptor module (matching the insertion scheme used by the U-Net/LDM
    loaders).  If the model exposes an explicit ``adaptor_parameters()`` method
    that is used instead.
    """
    if module is None:
        return []
    fn = getattr(module, "adaptor_parameters", None)
    if callable(fn):
        try:
            params = list(fn())
            if params:
                return [p for p in params if p.requires_grad]
        except Exception:  # pragma: no cover - defensive
            pass
    params = [p for n, p in module.named_parameters() if ".adaptor." in n and p.requires_grad]
    if not params:  # also accept modules stored directly as `.adaptor*`
        params = [p for n, p in module.named_parameters() if "adaptor" in n and p.requires_grad]
    return params


def count_parameters(module: Optional[nn.Module], only_trainable: bool = False) -> int:
    """Count parameters of ``module`` (optionally trainable only)."""
    if module is None:
        return 0
    if only_trainable:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in module.parameters()))


def parameter_rate(module: Optional[nn.Module], trainable: Optional[nn.Module] = None) -> float:
    """Fraction of parameters that are fine-tuned (adaptor) w.r.t. the total.

    Expected reference values (Table 1): DDPM-ANT 1.3%, LDM-ANT 1.6%.
    """
    if module is None:
        return 0.0
    total = count_parameters(module, only_trainable=False)
    if total == 0:
        return 0.0
    trainable = module if trainable is None else trainable
    n_train = count_parameters(trainable, only_trainable=True)
    return float(n_train) / float(total)


def _unwrap_prediction(out: Any) -> torch.Tensor:
    """Extract the predicted noise from a model output (tensor/tuple/dict)."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for key in ("eps", "epsilon", "pred", "output", "model_output", "noise"):
            if key in out:
                return _unwrap_prediction(out[key])
        raise KeyError(f"Cannot find noise prediction in dict keys {list(out.keys())}")
    if isinstance(out, (tuple, list)):
        # guided-diffusion returns (eps, sigma) when learn_sigma=True
        eps = out[0]
        if eps.shape[1] % 2 == 0 and len(out) == 2 and isinstance(out[1], torch.Tensor):
            pass  # keep eps as-is
        return _unwrap_prediction(eps)
    raise TypeError(f"Unsupported model output type: {type(out)}")


def _resolve_timestep_input(t: Union[torch.Tensor, int], batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        if t.ndim == 0:
            return t.to(device).expand(batch_size)
        return t.to(device)
    return torch.full((batch_size,), int(t), device=device, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ANTConfig:
    """Hyper-parameters of the ANT adaptation stage (Algorithm 1)."""

    # --- outer optimisation -------------------------------------------------
    iterations: int = 300
    batch_size: int = 40
    lr: float = 5e-5
    lr_ddpm: float = 5e-5
    lr_ldm: float = 1e-5
    optimizer: str = "adam"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    grad_accum: int = 1

    # --- objective ----------------------------------------------------------
    gamma: float = 5.0
    omega: float = 0.02
    J: int = 10
    norm: str = "per_sample"
    use_adv_noise: bool = True
    detach_classifier_grad: bool = True
    classifier_target_index: int = 1
    loss_reduction: str = "mean"

    # --- diffusion ----------------------------------------------------------
    num_timesteps: int = 1000
    schedule: str = "linear"
    eta: float = 0.0

    # --- trainable surface / freezing ---------------------------------------
    freeze_backbone: bool = True
    only_adaptor: bool = True
    train_classifier: bool = False

    # --- bookkeeping --------------------------------------------------------
    log_interval: int = 50
    save_interval: int = 100
    seed: Optional[int] = None
    device: Optional[str] = None
    backbone: str = "ddpm"

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, backbone: str = "ddpm", task: Optional[str] = None,
                  **overrides: Any) -> "ANTConfig":
        """Build a config from (possibly nested) dict blocks and overrides.

        Recognised blocks: ``ant``/``defaults`` (flat), ``diffusion``,
        ``tasks.<task>`` (per-task overrides).  Explicit keyword overrides win.
        """
        cfg = dict(cfg or {})
        merged: Dict[str, Any] = {}

        # nested blocks -> flat
        for block in ("ant", "defaults", "training"):
            sub = cfg.get(block)
            if isinstance(sub, dict):
                merged.update({k: v for k, v in sub.items() if not isinstance(v, dict)})
        if "diffusion" in cfg and isinstance(cfg["diffusion"], dict):
            d = cfg["diffusion"]
            merged.setdefault("num_timesteps", d.get("num_timesteps", d.get("T", 1000)))
            merged.setdefault("schedule", d.get("schedule", d.get("beta_schedule", "linear")))
            merged.setdefault("eta", d.get("eta", 0.0))
        if task is not None and isinstance(cfg.get("tasks"), dict) and task in cfg["tasks"]:
            tcfg = cfg["tasks"][task]
            if isinstance(tcfg, dict):
                for k, v in tcfg.items():
                    if not isinstance(v, dict) and v is not None:
                        merged[k] = v
        # already-flat keys
        for k, v in cfg.items():
            if not isinstance(v, (dict, list)) or k in ("adam_betas",):
                merged.setdefault(k, v)
        merged.update({k: v for k, v in overrides.items() if v is not None})

        # learning-rate resolution: per-task lr > backbone lr > default lr
        lr = merged.get("lr")
        if lr is None:
            lr = merged.get("lr_ddpm", 5e-5) if backbone == "ddpm" else merged.get("lr_ldm", 1e-5)
        merged["lr"] = float(lr)
        merged["backbone"] = backbone

        # keep only declared fields
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {}
        for k, v in merged.items():
            if k in fields:
                if k == "adam_betas" and isinstance(v, (list, tuple)):
                    v = tuple(float(x) for x in v)
                kwargs[k] = v
        return cls(**kwargs)

    def for_backbone(self, backbone: str) -> "ANTConfig":
        d = asdict(self)
        d["backbone"] = backbone
        d["lr"] = self.lr_ddpm if backbone == "ddpm" else self.lr_ldm
        return ANTConfig(**d)

    def replace(self, **overrides: Any) -> "ANTConfig":
        d = asdict(self)
        d.update(overrides)
        return ANTConfig(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class ANTTrainer:
    """Adaptor-only ANT trainer implementing Algorithm 1 / Equation (8).

    Parameters
    ----------
    model:
        Frozen (or partially frozen) pretrained diffusion backbone exposing
        ``epsilon_theta(x_t, t, y=None)`` / ``forward(x_t, t)``.
    classifier:
        Frozen binary source/target classifier exposing
        ``grad_log_target(x_t, t)``.  ``None`` disables the similarity-guided
        term (plain DDPM loss).
    config:
        :class:`ANTConfig` or a plain dict.
    diffusion / schedule:
        Optional pre-built :class:`GaussianDiffusion` / :class:`NoiseSchedule`.
    """

    def __init__(
        self,
        model: nn.Module,
        classifier: Optional[nn.Module] = None,
        config: Optional[Union[ANTConfig, Dict[str, Any]]] = None,
        diffusion: Optional[GaussianDiffusion] = None,
        schedule: Optional[NoiseSchedule] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_data: Optional[Any] = None,
        backbone: Optional[str] = None,
        **overrides: Any,
    ) -> None:
        if isinstance(config, ANTConfig):
            self.config = config if not overrides else config.replace(**overrides)
        else:
            backbone = backbone or (config or {}).get("backbone", "ddpm") if isinstance(config, dict) else (backbone or "ddpm")
            self.config = ANTConfig.from_dict(config, backbone=backbone, **overrides)

        self.cfg = self.config
        self.backbone = self.cfg.backbone

        # ---- device -----------------------------------------------------
        if device is None:
            device = self.cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = model
        if self.model is not None:
            self.model.to(self.device)

        # ---- diffusion --------------------------------------------------
        if diffusion is not None:
            self.diffusion = diffusion
        elif schedule is not None:
            self.diffusion = GaussianDiffusion(schedule=schedule, eta=self.cfg.eta)
        else:
            self.diffusion = build_diffusion(
                num_timesteps=self.cfg.num_timesteps,
                schedule=self.cfg.schedule,
                eta=self.cfg.eta,
                device=self.device,
            )
        self.schedule: NoiseSchedule = self.diffusion.schedule
        self.schedule.to(self.device)

        # ---- classifier --------------------------------------------------
        self.classifier = classifier
        if self.classifier is not None:
            self.classifier.to(self.device)
            for p in self.classifier.parameters():
                p.requires_grad_(bool(self.cfg.train_classifier))
            self.classifier.eval()

        # ---- adversarial noise selector ---------------------------------
        self.selector = AdversarialNoiseSelector(
            model=self.model,
            schedule=self.schedule,
            J=self.cfg.J,
            omega=self.cfg.omega,
            norm=self.cfg.norm,
            loss_reduction="sum",
            device=self.device,
        )

        # ---- freezing / trainable surface -------------------------------
        self._trainable_params: List[nn.Parameter] = []
        self._prepare_trainable()

        # ---- optimiser ----------------------------------------------------
        self.optimizer = self._build_optimizer()

        # ---- state --------------------------------------------------------
        self.step = 0
        self.history: List[Dict[str, float]] = []
        self._target_data = target_data
        self._rng = None
        if self.cfg.seed is not None:
            self._rng = torch.Generator(device="cpu")
            self._rng.manual_seed(int(self.cfg.seed))

        LOGGER.info(
            "ANTTrainer(backbone=%s, trainable=%d, rate=%.4f%%, iters=%d, bs=%d, lr=%g, gamma=%g, omega=%g, J=%d, AN=%s)",
            self.backbone,
            sum(p.numel() for p in self._trainable_params),
            100.0 * self.parameter_rate(),
            self.cfg.iterations,
            self.cfg.batch_size,
            self.cfg.lr,
            self.cfg.gamma,
            self.cfg.omega,
            self.cfg.J,
            self.cfg.use_adv_noise,
        )

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    def _prepare_trainable(self) -> None:
        """Freeze the backbone and collect the parameters to optimise."""
        cfg = self.cfg
        if cfg.freeze_backbone:
            # keep adaptor parameters trainable
            for name, p in self.model.named_parameters():
                is_adaptor = ".adaptor." in name or name.startswith("adaptor")
                p.requires_grad_(bool(is_adaptor))
        if cfg.only_adaptor:
            params = adaptor_trainable_parameters(self.model)
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
        # deduplicate while preserving order
        seen = set()
        uniq: List[nn.Parameter] = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                uniq.append(p)
        self._trainable_params = uniq
        if not self._trainable_params:
            LOGGER.warning(
                "No trainable parameters found. Did you insert zero-initialised adaptors "
                "(insert_adaptors / insert_adaptors_ldm) before creating the trainer?"
            )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg
        name = str(cfg.optimizer).lower()
        params = self._trainable_params
        if name in ("adam", "adamw"):
            cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
            return cls(
                params,
                lr=cfg.lr,
                betas=tuple(cfg.adam_betas),
                eps=cfg.adam_eps,
                weight_decay=cfg.weight_decay,
            )
        if name == "sgd":
            return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def trainable_parameters(self) -> List[nn.Parameter]:
        return list(self._trainable_params)

    def parameter_rate(self) -> float:
        total = count_parameters(self.model, only_trainable=False)
        if total == 0:
            return 0.0
        return sum(p.numel() for p in self._trainable_params) / float(total)

    def verify_only_adaptor_updated(self) -> bool:
        """True if only adaptor parameters currently require grad."""
        for name, p in self.model.named_parameters():
            is_adaptor = ".adaptor." in name or name.startswith("adaptor")
            if p.requires_grad and not is_adaptor:
                return False
        return True

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #
    def _get_batch(self, batch: Optional[Any] = None, batch_size: Optional[int] = None) -> torch.Tensor:
        """Fetch a batch of target images ``x_0`` as float tensors."""
        batch_size = batch_size or self.cfg.batch_size
        src = batch if batch is not None else self._target_data
        if src is None:
            raise ValueError("No target data provided: pass `target_data=` to the trainer or a batch to train_step().")

        if isinstance(src, torch.Tensor):
            data = src
            idx = torch.randint(0, data.shape[0], (batch_size,), generator=self._rng)
            x0 = data[idx]
        elif hasattr(src, "__iter__") and not hasattr(src, "shape"):
            # DataLoader / iterable of batches
            if not hasattr(self, "_data_iter") or self._data_iter is None:
                self._data_iter = iter(src)
            try:
                x0 = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(src)
                x0 = next(self._data_iter)
            if isinstance(x0, (tuple, list)):
                x0 = x0[0]
        else:  # Dataset-like
            idx = torch.randint(0, len(src), (batch_size,), generator=self._rng)
            items = [src[int(i)] for i in idx]
            x0 = torch.stack([it[0] if isinstance(it, (tuple, list)) else it for it in items])

        x0 = x0.to(self.device).float()
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)
        return x0

    def sample_timesteps(self, batch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """t ~ Uniform({1, ..., T}) as in Algorithm 1."""
        return torch.randint(1, self.cfg.num_timesteps + 1, (batch_size,), device=self.device, generator=generator)

    # ------------------------------------------------------------------ #
    # forward / losses
    # ------------------------------------------------------------------ #
    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """eps_{theta,psi}(x_t, t) using the adapted model."""
        fn = getattr(self.model, "epsilon_theta", None)
        if callable(fn):
            out = fn(x_t, t)
        else:
            out = self.model(x_t, t)
        return _unwrap_prediction(out)

    def classifier_gradient(self, x_t: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        """Detached ``grad_{x_t} log p_phi(y=T | x_t)`` (or ``None``)."""
        if self.classifier is None:
            return None
        fn = getattr(self.classifier, "grad_log_target", None)
        if callable(fn):
            kwargs = dict(detach=self.cfg.detach_classifier_grad)
            try:
                return fn(x_t, t, target_index=self.cfg.classifier_target_index, **kwargs)
            except TypeError:  # pragma: no cover - simpler signature
                return fn(x_t, t)
        raise AttributeError("classifier must expose `grad_log_target(x_t, t)`")

    def similarity_guided_loss(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        sigma_hat: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Equation (8) / (5).

        ``|| eps* - eps_{theta,psi}(x_t*, t) - sigma_hat_t^2 gamma grad log p_phi ||^2``
        implemented as an MSE against the corrected target
        ``eps* - sigma_hat_t^2 gamma grad log p_phi`` (the gradient is detached).
        """
        info: Dict[str, float] = {}
        grad = self.classifier_gradient(x_t, t) if self.cfg.gamma != 0.0 else None
        correction = None
        if grad is not None:
            correction = (sigma_hat ** 2) * float(self.cfg.gamma) * grad
            # Eq. (5): || eps - eps_theta - sigma_hat^2 gamma grad ||^2
            #  => target for eps_theta is (eps - correction)
            info["classifier_grad_norm"] = float(grad.detach().float().norm())
        target = eps_target if correction is None else (eps_target - correction)

        if self.cfg.loss_reduction == "sum":
            diff = (eps_pred - target).reshape(eps_pred.shape[0], -1)
            loss = diff.pow(2).sum(dim=1).mean()
        else:
            loss = F.mse_loss(eps_pred, target)
        info["loss"] = float(loss.detach())
        return loss, info

    # ------------------------------------------------------------------ #
    # one optimisation step
    # ------------------------------------------------------------------ #
    def train_step(self, batch: Optional[Any] = None, iteration: Optional[int] = None) -> Dict[str, float]:
        """One ANT outer iteration (Algorithm 1 lines 3-8)."""
        cfg = self.cfg
        self.model.train()
        x0 = self._get_batch(batch)
        b = x0.shape[0]
        t = self.sample_timesteps(b)

        info: Dict[str, float] = {}

        # ---- inner maximisation: worst-case noise eps* (Eq. 7) ----------
        if cfg.use_adv_noise:
            eps_star, x_t_star, info_an = self.selector.select(
                x0, t, model=self.model, record_history=False, return_info=True
            )
            if isinstance(info_an, dict):
                info.update({f"an_{k}": v for k, v in info_an.items() if isinstance(v, (int, float))})
        else:  # ablation: plain Gaussian noise (DPMs-ANT w/o AN)
            eps_star = torch.randn_like(x0)
            x_t_star = self.diffusion.q_sample(x0, t, noise=eps_star)

        eps_star = eps_star.detach()
        x_t_star = x_t_star.detach()

        sigma_hat = self.diffusion.sch_sigma_hat(t, x_t_star.ndim).to(x_t_star.dtype)

        # ---- outer loss + adaptor update (Eq. 8) ------------------------
        self.optimizer.zero_grad(set_to_none=True)
        for _ in range(max(1, int(cfg.grad_accum))):
            eps_pred = self.predict_noise(x_t_star, t)
            loss, loss_info = self.similarity_guided_loss(eps_pred, eps_star, sigma_hat, x_t_star, t)
            (loss / max(1, int(cfg.grad_accum))).backward()
        info.update(loss_info)

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable_params, float(cfg.grad_clip))
            info["grad_norm"] = float(grad_norm)
            if torch.is_tensor(grad_norm):
                info["grad_norm"] = float(grad_norm.detach())

        if self._trainable_params:
            self.optimizer.step()
        else:
            LOGGER.warning("No trainable adaptor parameters; skipping optimizer step.")

        self.step = (iteration if iteration is not None else self.step) + 1
        info["step"] = float(self.step)
        return info

    # ------------------------------------------------------------------ #
    # training loop
    # ------------------------------------------------------------------ #
    def train(
        self,
        target_data: Optional[Any] = None,
        iterations: Optional[int] = None,
        batch: Optional[Any] = None,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Run Algorithm 1 for ``iterations`` outer steps.

        Returns the per-iteration info dictionary list (also stored on
        ``self.history``).
        """
        if target_data is not None:
            self._target_data = target_data
        iterations = int(iterations if iterations is not None else self.cfg.iterations)

        history: List[Dict[str, float]] = []
        t0 = time.time()
        for it in range(iterations):
            info = self.train_step(batch=batch, iteration=it)
            history.append(info)
            if verbose and (it % max(1, int(self.cfg.log_interval)) == 0 or it == iterations - 1):
                msg = " | ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(info.items())
                    if k in ("step", "loss", "classifier_grad_norm", "grad_norm", "an_inner_loss")
                )
                LOGGER.info("ANT iter %d/%d | %s", it + 1, iterations, msg)
            if callback is not None:
                callback(it, info)

        elapsed = time.time() - t0
        self.history.extend(history)

        # ---- final introspection ---------------------------------------
        with torch.no_grad():
            params = [p.detach().float().norm().item() for p in self._trainable_params]
        LOGGER.info(
            "ANT finished: %d iters in %.1fs (%.2fs/iter) | adaptor param norm=%.4f",
            iterations,
            elapsed,
            elapsed / max(1, iterations),
            float(sum(params)),
        )
        return history

    # ------------------------------------------------------------------ #
    def evaluate_loss(self, target_data: Optional[Any] = None, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Average ANT loss over a handful of batches (no parameter update)."""
        was_training = self.model.training
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for _ in range(4):
                x0 = self._get_batch(target_data, batch_size=batch_size)
                b = x0.shape[0]
                t = self.sample_timesteps(b)
                if self.cfg.use_adv_noise:
                    eps_star, x_t_star = self.selector.select(x0, t, model=self.model, record_history=False)
                else:
                    eps_star = torch.randn_like(x0)
                    x_t_star = self.diffusion.q_sample(x0, t, noise=eps_star)
                eps_pred = self.predict_noise(x_t_star.detach(), t)
                sigma_hat = self.diffusion.sch_sigma_hat(t, x_t_star.ndim)
                grad = self.classifier_gradient(x_t_star.detach(), t)
                target = eps_star if grad is None else eps_star - sigma_hat ** 2 * self.cfg.gamma * grad
                losses.append(float(F.mse_loss(eps_pred, target.detach())))
        if was_training:
            self.model.train()
        return {"loss": float(sum(losses) / max(1, len(losses)))}

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def adaptor_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict of the adaptor tensors only (small checkpoint)."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
            if ".adaptor." in k or k.startswith("adaptor")
        }

    def load_adaptor_state_dict(self, state: Dict[str, torch.Tensor], strict: bool = False) -> None:
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Adaptor state dict mismatch: missing={missing}, unexpected={unexpected}")
        if unexpected:
            LOGGER.warning("Unexpected keys while loading adaptors: %s", unexpected)

    def save(self, path: str, include_optimizer: bool = False) -> str:
        payload: Dict[str, Any] = {
            "adaptor": self.adaptor_state_dict(),
            "config": self.cfg.to_dict(),
            "step": self.step,
        }
        if include_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, path)
        LOGGER.info("Saved ANT adaptor checkpoint to %s", path)
        return path

    def load(self, path: str, device: Optional[str] = None) -> "ANTTrainer":
        payload = torch.load(path, map_location=device or str(self.device))
        state = payload.get("adaptor", payload)
        self.load_adaptor_state_dict(state)
        self.step = int(payload.get("step", 0))
        return self


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def build_ant_trainer(
    cfg: Optional[Dict[str, Any]] = None,
    model: Optional[nn.Module] = None,
    classifier: Optional[nn.Module] = None,
    target_data: Optional[Any] = None,
    backbone: str = "ddpm",
    task: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **overrides: Any,
) -> ANTTrainer:
    """Config-driven factory returning an :class:`ANTTrainer`."""
    config = ANTConfig.from_dict(cfg, backbone=backbone, task=task, **overrides)
    return ANTTrainer(
        model=model,
        classifier=classifier,
        config=config,
        target_data=target_data,
        device=device,
    )


def train_ant(
    model: nn.Module,
    target_data: Any,
    classifier: Optional[nn.Module] = None,
    cfg: Optional[Dict[str, Any]] = None,
    backbone: str = "ddpm",
    task: Optional[str] = None,
    iterations: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
    **overrides: Any,
) -> Tuple[ANTTrainer, List[Dict[str, float]]]:
    """End-to-end ANT adaptation (Algorithm 1).

    Parameters
    ----------
    model:
        Frozen pretrained backbone with adaptors already inserted.
    target_data:
        Tensor / DataLoader / Dataset of the (10-shot) target images.
    classifier:
        Frozen source-vs-target classifier providing the guidance gradient.
    cfg:
        Config dict (e.g. ``defaults`` + ``tasks[task]`` from ``configs/*.yaml``).
    checkpoint_path:
        Optional path to save the adaptor-only checkpoint.

    Returns
    -------
    (trainer, history)
    """
    trainer = build_ant_trainer(
        cfg=cfg,
        model=model,
        classifier=classifier,
        target_data=target_data,
        backbone=backbone,
        task=task,
        device=device,
        **overrides,
    )
    history = trainer.train(target_data=target_data, iterations=iterations, verbose=verbose)
    if checkpoint_path:
        trainer.save(checkpoint_path)
    return trainer, history
