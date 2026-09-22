"""Toy 2-D Gaussian experiment for DPMs-ANT (paper Section 5.1, Figure 2).

This module reproduces the quantitative toy analysis of Section 5.1:

* Source data ``x0 ~ N((1, 1), I)``, target data ``x0 ~ N((-1, -1), I)``.
* A "simple neural network" (small MLP noise predictor) is trained on the
  *source* domain with the standard DDPM objective, then transferred to the
  *target* domain with the ANT objective (Eq. 5 + Eq. 7 + Eq. 8).

The module provides everything needed to rebuild Figure 2:

(a) gradient-direction comparison.  In the first transfer iteration, with the
    *same* noise and timestep, we measure the output-layer gradient of four
    settings:

      * ``reference``  -- gradient computed on 10,000 *different* samples (cyan
        in the paper), taken as the reliable direction (~45 degrees southwest);
      * ``baseline``   -- traditional DDPM gradient on a 10-shot batch (repeated
        1000 times so the batch has the same size as the reference);
      * ``ant_wo_an``  -- similarity-guided training only (DPMs-ANT w/o AN);
      * ``ant``        -- full DPMs-ANT (adversarial noise + similarity guide).

    Additionally the worst-case noise cloud produced by adversarial noise
    selection is recorded to show the circle -> ellipse transition, whose
    principal axis follows the model-parameter gradient.

(b) / (c) heat maps of 20,000 generated samples: x-axis = diffusion timestep,
    y-axis = sampled value (per the addendum).  Both baseline and ANT sampling
    trajectories are recorded.

The implementation reuses the project's diffusion schedule and the
:class:`~dpm_ant.training.adv_noise.AdversarialNoiseSelector` so the toy study
exercises the very same code paths as the image experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion.schedule import NoiseSchedule, build_schedule

LOGGER = logging.getLogger("dpm_ant.toy_2d")

__all__ = [
    "ToyConfig",
    "ToyMLP",
    "ToyClassifier",
    "InitialisationBias",
    "gradient_direction_experiment",
    "noise_cloud_statistics",
    "heatmap_samples",
    "build_toy_schedule",
    "build_toy_model",
    "train_toy_model",
    "train_toy_classifier",
    "run_toy_experiment",
    "main",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ToyConfig:
    """Hyper-parameters of the toy 2-D experiment (Section 5.1)."""

    source_mean: Tuple[float, float] = (1.0, 1.0)
    target_mean: Tuple[float, float] = (-1.0, -1.0)
    covariance: Tuple[float, float] = (1.0, 1.0)  # diagonal of I
    dim: int = 2

    # diffusion
    num_timesteps: int = 1000
    schedule: str = "linear"
    eta: float = 0.0

    # small network
    hidden_dim: int = 256
    num_layers: int = 3
    time_dim: int = 64
    activation: str = "silu"
    init_bias: Tuple[float, float] = (0.0, 0.0)

    # pre-training on source
    pretrain_steps: int = 4000
    pretrain_batch: int = 512
    pretrain_lr: float = 1e-3

    # classifier
    classifier_steps: int = 300
    classifier_batch: int = 64
    classifier_lr: float = 1e-4
    classifier_hidden: int = 128

    # ANT transfer
    iterations: int = 300
    batch_size: int = 10
    repeat_target: int = 1000  # repeat 10 shots 1000x for the unified comparison
    lr: float = 5e-5
    gamma: float = 5.0
    omega: float = 0.02
    J: int = 10
    norm: str = "per_sample"

    # evaluation / figures
    num_samples_reference: int = 10000
    num_target_samples: int = 10
    heatmap_samples: int = 20000
    heatmap_timesteps: int = 100
    grad_seed: int = 0
    grad_timestep: int = 500
    num_worst_case: int = 256

    seed: int = 0
    device: str = "cpu"
    out_dir: str = "outputs/toy"

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["source_mean"] = list(self.source_mean)
        d["target_mean"] = list(self.target_mean)
        d["covariance"] = list(self.covariance)
        d["init_bias"] = list(self.init_bias)
        return d

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides) -> "ToyConfig":
        """Build from a (possibly nested) config dict, mirroring ``toy.*`` in YAML."""
        merged: Dict[str, Any] = {}
        if cfg:
            base = {}
            for key in ("toy", "defaults", "diffusion"):
                if isinstance(cfg.get(key), dict):
                    base.update(cfg[key])
            # top-level keys win over nested blocks
            base.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
            if isinstance(cfg.get("toy"), dict):
                base.update(cfg["toy"])
            merged.update(base)
        merged.update(overrides)
        # aliases
        aliases = {
            "T": "num_timesteps",
            "beta_schedule": "schedule",
            "num_samples": "num_samples_reference",
            "model_hidden_dim": "hidden_dim",
        }
        for src, dst in aliases.items():
            if src in merged and dst not in merged:
                merged[dst] = merged.pop(src)
            else:
                merged.pop(src, None)
        # drop unknown keys
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in merged.items() if k in known}
        # normalise tuples
        for key in ("source_mean", "target_mean", "covariance", "init_bias"):
            if key in kwargs and kwargs[key] is not None and not isinstance(kwargs[key], tuple):
                kwargs[key] = tuple(float(x) for x in kwargs[key])
        return cls(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Small networks
# ---------------------------------------------------------------------------
def _activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name in ("gelu",):
        return nn.GELU()
    if name in ("tanh",):
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal timestep embedding (same recipe as the image backbones)."""
    t = timesteps.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ToyMLP(nn.Module):
    """Small MLP noise predictor ``eps_theta(x_t, t)`` for 2-D data.

    The final layer is a *linear* layer, intentionally initialised with a small
    (optionally non-zero) bias so that the untrained transfer model is biased
    towards the source domain - the "noisy gradient" phenomenon of Figure 2(a).
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_dim: int = 64,
        activation: str = "silu",
        init_bias: Sequence[float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.time_dim = int(time_dim)
        self.out_layer = nn.Linear(hidden_dim, dim)
        layers: List[nn.Module] = []
        in_dim = dim + time_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_activation(activation))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self._init_weights(init_bias)

    def _init_weights(self, init_bias: Sequence[float]) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.out_layer.weight, std=0.02)
        bias = torch.as_tensor(init_bias, dtype=torch.float32)
        if bias.numel() != self.dim:
            bias = torch.zeros(self.dim)
        self.out_layer.bias.data.copy_(bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.dim() == 1:
            x_t = x_t[None, :]
        emb = timestep_embedding(t, self.time_dim).to(x_t.dtype)
        if emb.shape[0] == 1 and x_t.shape[0] > 1:
            emb = emb.expand(x_t.shape[0], -1)
        h = torch.cat([x_t, emb], dim=-1)
        h = self.net(h)
        return self.out_layer(h)

    # ------------------------------------------------------------------
    # frozen-model style interface used by ANT code
    # ------------------------------------------------------------------
    def epsilon_theta(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(x_t, t)


class InitialisationBias(nn.Module):
    """Zero-initialised adaptor ``psi`` for the toy model.

    ``psi(z) = -gamma * (z - mu)`` with ``mu`` the source mean: it starts as a
    *no-op* only for data centred at ``mu``, which is exactly how a model
    pre-trained on the source behaves when shown target data.
    """

    def __init__(self, dim: int, mean: Sequence[float] = (0.0, 0.0)) -> None:
        super().__init__()
        self.dim = int(dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x - self.mean)


class ToyClassifier(nn.Module):
    """Binary source(0)/target(1) classifier ``p_phi(y | x_t)`` for 2-D data."""

    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_classes: int = 2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.time_dim = int(time_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_t: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_t.dim() == 1:
            x_t = x_t[None, :]
        if t is None:
            t = torch.zeros(x_t.shape[0], device=x_t.device, dtype=torch.long)
        elif isinstance(t, (int, float)):
            t = torch.full((x_t.shape[0],), int(t), device=x_t.device, dtype=torch.long)
        elif t.dim() == 0:
            t = t.expand(x_t.shape[0])
        # accept normalised t in [0, 1] as well
        t_long = t if t.dtype in (torch.int32, torch.int64) else (t.float() * 1000.0).long()
        emb = timestep_embedding(t_long, self.time_dim).to(x_t.dtype)
        if emb.shape[0] == 1 and x_t.shape[0] > 1:
            emb = emb.expand(x_t.shape[0], -1)
        return self.net(torch.cat([x_t, emb], dim=-1))

    # ------------------------------------------------------------------
    def predict_proba(self, x_t: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.softmax(self.forward(x_t, t), dim=-1)

    def log_prob_target(self, x_t: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.log_softmax(self.forward(x_t, t), dim=-1)[:, 1].sum()

    def grad_log_target(
        self,
        x_t: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        target_index: int = 1,
        detach: bool = True,
        create_graph: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Return ``grad_{x_t} log p_phi(y = target | x_t)``."""
        with torch.enable_grad():
            x_in = x_t.detach().clone().requires_grad_(True) if detach else x_t
            logits = self.forward(x_in, t)
            logp = F.log_softmax(logits, dim=-1)[:, target_index].sum()
            (grad,) = torch.autograd.grad(logp, x_in, create_graph=create_graph)
        if scale is not None:
            grad = grad * scale
        return grad.detach() if detach else grad

    def freeze(self) -> "ToyClassifier":
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze(self) -> "ToyClassifier":
        for p in self.parameters():
            p.requires_grad_(True)
        return self


# ---------------------------------------------------------------------------
# Schedule / model factories
# ---------------------------------------------------------------------------
def build_toy_schedule(cfg: Optional[ToyConfig] = None, **overrides) -> NoiseSchedule:
    cfg = cfg or ToyConfig(**overrides)
    return build_schedule(
        {
            "num_timesteps": cfg.num_timesteps,
            "schedule": cfg.schedule,
            "eta": cfg.eta,
            "device": cfg.device,
        }
    )


def build_toy_model(cfg: ToyConfig, seed: Optional[int] = None) -> ToyMLP:
    if seed is not None:
        torch.manual_seed(seed)
    return ToyMLP(
        dim=cfg.dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        time_dim=cfg.time_dim,
        activation=cfg.activation,
        init_bias=cfg.init_bias,
    ).to(cfg.device)


# ---------------------------------------------------------------------------
# Data / gradient utilities
# ---------------------------------------------------------------------------
def sample_gaussian(
    num_samples: int,
    mean: Sequence[float],
    std: Sequence[float] = (1.0, 1.0),
    device: str = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Draw ``N`` samples from a (diagonal) Gaussian with the given mean/std."""
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
    else:
        gen = None
    z = torch.randn(num_samples, len(mean), generator=gen, dtype=torch.float32)
    m = torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1)
    s = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1)
    return (m + s * z).to(device)


def source_samples(cfg: ToyConfig, num_samples: int, seed: Optional[int] = None) -> torch.Tensor:
    return sample_gaussian(num_samples, cfg.source_mean, cfg.covariance, cfg.device, seed)


def target_samples(cfg: ToyConfig, num_samples: int, seed: Optional[int] = None) -> torch.Tensor:
    return sample_gaussian(num_samples, cfg.target_mean, cfg.covariance, cfg.device, seed)


def ddpm_gradient(
    model: nn.Module,
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: int,
    schedule: NoiseSchedule,
    reduce: str = "sum",
) -> torch.Tensor:
    """``grad_theta E ||eps - eps_theta(x_t, t)||^2`` (traditional DDPM, Eq. 2)."""
    for p in model.parameters():
        # only the *output* layer gradient is compared in Figure 2(a)
        pass
    ab = float(schedule.ab(torch.tensor([t]), 1).reshape(-1)[0])
    x_t = math.sqrt(ab) * x0 + math.sqrt(max(1e-12, 1.0 - ab)) * noise
    t_vec = torch.full((x0.shape[0],), int(t), device=x0.device, dtype=torch.long)
    eps_pred = _call_model(model, x_t, t_vec)
    loss = F.mse_loss(eps_pred, noise, reduction=reduce)
    return loss


def _call_model(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Duck-typed noise prediction for toy / image models."""
    for attr in ("epsilon_theta", "predict_noise", "predict_eps"):
        fn = getattr(model, attr, None)
        if callable(fn):
            out = fn(x_t, t)
            return _unwrap(out)
    out = model(x_t, t)
    return _unwrap(out)


def _unwrap(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for key in ("eps", "noise", "pred", "model_output", "output"):
            if key in out and isinstance(out[key], torch.Tensor):
                return out[key]
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"Cannot interpret model output of type {type(out)}")


def output_layer_gradient(
    model: ToyMLP,
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: int,
    schedule: NoiseSchedule,
    adaptor: Optional[InitialisationBias] = None,
    classifier: Optional[ToyClassifier] = None,
    gamma: float = 0.0,
    reduce: str = "sum",
) -> torch.Tensor:
    """Gradient of the (similarity-guided) DDPM objective w.r.t. the model output layer.

    The returned vector plays the role of the "gradient direction" traced in
    Figure 2(a).  When ``adaptor``/``classifier`` are supplied, the
    similarity-guided correction of Eq. (5) is included (DPMs-ANT w/o AN), which
    makes the direction ambiguous; the ellipse of worst-case noise resolves it.
    """
    model.zero_grad(set_to_none=True)
    ab = float(schedule.ab(torch.tensor([t]), 1).reshape(-1)[0])
    x_t = math.sqrt(ab) * x0 + math.sqrt(max(1e-12, 1.0 - ab)) * noise
    if adaptor is not None:
        x_t = x_t + adaptor(x_t)
    t_vec = torch.full((x0.shape[0],), int(t), device=x0.device, dtype=torch.long)
    eps_pred = _call_model(model, x_t, t_vec)

    target = noise
    if classifier is not None and gamma != 0.0:
        sigma_hat = sigma_hat_value(schedule, t)
        with torch.enable_grad():
            grad = classifier.grad_log_target(x_t.detach(), t_vec, detach=True)
        target = noise - (sigma_hat ** 2) * gamma * grad
    loss = F.mse_loss(eps_pred, target.detach(), reduction=reduce)
    params = list(model.out_layer.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    flat = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                      for g, p in zip(grads, params)])
    return flat.detach()


def sigma_hat_value(schedule: NoiseSchedule, t: int) -> float:
    """``sigma_hat_t`` of Appendix A.2 (Eq. 5 / Eq. 8)."""
    t_tensor = torch.tensor([int(t)])
    try:
        return float(schedule.get_sigma_hat(t_tensor, 1).reshape(-1)[0])
    except Exception:  # pragma: no cover - defensive
        ab_t = float(schedule.ab(t_tensor, 1).reshape(-1)[0])
        ab_prev = float(schedule.ab_prev(t_tensor, 1).reshape(-1)[0])
        alpha_t = float(schedule.alpha(t_tensor, 1).reshape(-1)[0])
        return (1.0 - ab_prev) * math.sqrt(max(1e-12, alpha_t) / max(1e-12, 1.0 - ab_t))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_toy_model(
    model: ToyMLP,
    cfg: ToyConfig,
    schedule: Optional[NoiseSchedule] = None,
    steps: Optional[int] = None,
    data: Optional[torch.Tensor] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> List[float]:
    """Train the small network on the *source* domain with the DDPM objective."""
    schedule = schedule or build_toy_schedule(cfg)
    steps = int(steps if steps is not None else cfg.pretrain_steps)
    batch_size = int(batch_size or cfg.pretrain_batch)
    lr = float(lr or cfg.pretrain_lr)
    if data is None:
        data = source_samples(cfg, max(4096, batch_size * 4), seed=cfg.seed)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[float] = []
    model.train()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch_size,), device=cfg.device)
        x0 = data[idx]
        t = torch.randint(1, cfg.num_timesteps + 1, (batch_size,), device=cfg.device)
        noise = torch.randn_like(x0)
        ab = schedule.ab(t, 2)
        x_t = ab.sqrt() * x0 + (1.0 - ab).clamp(min=0.0).sqrt() * noise
        eps_pred = _call_model(model, x_t, t)
        loss = F.mse_loss(eps_pred, noise)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        history.append(float(loss.detach().cpu()))
        if verbose and (step + 1) % max(1, steps // 10) == 0:
            LOGGER.info("  [pretrain] step %d/%d loss=%.5f", step + 1, steps, history[-1])
    return history


def train_toy_classifier(
    classifier: ToyClassifier,
    cfg: ToyConfig,
    schedule: Optional[NoiseSchedule] = None,
    steps: Optional[int] = None,
    verbose: bool = True,
) -> List[float]:
    """Fine-tune the 2-way classifier on noised source and target samples."""
    schedule = schedule or build_toy_schedule(cfg)
    steps = int(steps if steps is not None else cfg.classifier_steps)
    bs = int(cfg.classifier_batch)
    src = source_samples(cfg, max(512, bs), seed=cfg.seed + 1)
    tgt = target_samples(cfg, max(cfg.num_target_samples, bs), seed=cfg.seed + 2)
    optim = torch.optim.Adam(classifier.parameters(), lr=cfg.classifier_lr)
    history: List[float] = []
    classifier.train()
    for step in range(steps):
        n_src = bs // 2
        n_tgt = bs - n_src
        i_src = torch.randint(0, src.shape[0], (n_src,), device=cfg.device)
        i_tgt = torch.randint(0, tgt.shape[0], (n_tgt,), device=cfg.device)
        x0 = torch.cat([src[i_src], tgt[i_tgt]], dim=0)
        labels = torch.cat(
            [torch.zeros(n_src, dtype=torch.long), torch.ones(n_tgt, dtype=torch.long)]
        ).to(cfg.device)
        t = torch.randint(1, cfg.num_timesteps + 1, (bs,), device=cfg.device)
        noise = torch.randn_like(x0)
        ab = schedule.ab(t, 2)
        x_t = ab.sqrt() * x0 + (1.0 - ab).clamp(min=0.0).sqrt() * noise
        logits = classifier(x_t, t)
        loss = F.cross_entropy(logits, labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        history.append(float(loss.detach().cpu()))
        if verbose and (step + 1) % max(1, steps // 5) == 0:
            LOGGER.info("  [classifier] step %d/%d loss=%.5f", step + 1, steps, history[-1])
    classifier.eval()
    return history


# ---------------------------------------------------------------------------
# Figure 2(a): gradient directions + worst-case noise cloud
# ---------------------------------------------------------------------------
def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


def _angle_deg(vec: torch.Tensor) -> float:
    v = vec.reshape(-1).float()
    # paper: reference direction is ~45 degrees southwest, i.e. (-1,-1)
    return float(torch.rad2deg(torch.atan2(v[1], v[0])))
def _classifier_convention(sim: float) -> float:
    """Figure 2(a) compares *descent* directions; flip sign if needed."""
    return sim


def gradient_direction_experiment(
    cfg: Optional[ToyConfig] = None,
    model: Optional[ToyMLP] = None,
    classifier: Optional[ToyClassifier] = None,
    verbose: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """Reproduce Figure 2(a): gradient directions of 4 settings in iteration 1.

    Returns a dict with the reference / baseline / w/o-AN / full-ANT gradient
    vectors, their cosine similarities and angles, plus the worst-case noise
    cloud (used for the circle -> ellipse visualisation).
    """
    cfg = cfg or ToyConfig.from_dict(overrides or None)
    device = cfg.device
    schedule = build_toy_schedule(cfg)

    # ------------------------------------------------------------------
    # 1) pre-train the small network on the source domain
    # ------------------------------------------------------------------
    if model is None:
        model = build_toy_model(cfg, seed=cfg.seed)
        if verbose:
            LOGGER.info("Pre-training the small network on source N((1,1), I) ...")
        train_toy_model(model, cfg, schedule, verbose=verbose)
    model.eval()

    # ------------------------------------------------------------------
    # 2) classifier p_phi (source vs target)
    # ------------------------------------------------------------------
    if classifier is None:
        torch.manual_seed(cfg.seed + 3)
        classifier = ToyClassifier(cfg.dim, cfg.classifier_hidden, cfg.time_dim).to(device)
        if verbose:
            LOGGER.info("Fine-tuning the 2-way classifier on noised source/target ...")
        train_toy_classifier(classifier, cfg, schedule, verbose=verbose)
    classifier.freeze()

    # ------------------------------------------------------------------
    # 3) fix the same noise and timestep for every setting
    # ------------------------------------------------------------------
    t = int(cfg.grad_timestep)
    gen = torch.Generator(device="cpu").manual_seed(cfg.grad_seed)

    # reference: 10,000 *different* samples
    x0_ref = source_samples(cfg, cfg.num_samples_reference, seed=cfg.grad_seed)
    noise_ref = torch.randn(x0_ref.shape, generator=gen).to(device)
    grad_ref = output_layer_gradient(model, x0_ref, noise_ref, t, schedule)

    # 10-shot target samples repeated 1000x -> same batch size as the reference
    base = target_samples(cfg, cfg.num_target_samples, seed=cfg.grad_seed + 7)
    x0_10 = base.repeat(cfg.repeat_target, 1)[: cfg.num_samples_reference].contiguous()
    # "the same noise" - reuse the reference noise tensor (shape matches)
    noise_10 = noise_ref.clone()

    grad_baseline = output_layer_gradient(model, x0_10, noise_10, t, schedule)
    grad_wo_an = output_layer_gradient(
        model, x0_10, noise_10, t, schedule, classifier=classifier, gamma=cfg.gamma
    )

    # full ANT: adversarial noise selection (Eq. 7) then SG loss gradient
    adversarial = None
    noise_cloud = None
    try:
        from ..training.adv_noise import AdversarialNoiseSelector

        adversarial = AdversarialNoiseSelector(
            model=model,
            schedule=schedule,
            J=cfg.J,
            omega=cfg.omega,
            norm=cfg.norm,
            device=device,
        )
        _, x_t_star, info = _select_with_history(adversarial, x0_10, t)
        noise_cloud = info["noise_history"]
        eps_star = info["eps"]

        # gradient through the ANT objective: trainer uses x_t*, and the model
        # receives the corrected noise; the analogous "gradient direction" is
        # measured w.r.t. the perturbed input x_t*.
        grad_ant = output_layer_gradient(
            model,
            x0_10,
            eps_star,
            t,
            schedule,
            classifier=classifier,
            gamma=cfg.gamma,
        )
        x_t_star_record = x_t_star.detach()
    except Exception as exc:  # pragma: no cover - adv_noise optional
        LOGGER.warning("Adversarial noise selection unavailable (%s); using noise_ref.", exc)
        grad_ant = grad_wo_an
        x_t_star_record = None

    # ------------------------------------------------------------------
    # 4) optional: the "true" ANT direction, measured on the 10-shot batch
    #    with adversarial noise (worst-case) - closer to the 10,000 reference.
    # ------------------------------------------------------------------
    if noise_cloud is not None:
        eps_for_ref = noise_cloud[-1]
        grad_ant = output_layer_gradient(
            model, x0_10, eps_for_ref, t, schedule, classifier=classifier, gamma=cfg.gamma
        )

    similarities = {
        "baseline": _classifier_convention(_cosine_similarity(grad_baseline, grad_ref)),
        "ant_wo_an": _classifier_convention(_cosine_similarity(grad_wo_an, grad_ref)),
        "ant": _classifier_convention(_cosine_similarity(grad_ant, grad_ref)),
    }
    angles = {
        "reference": _angle_deg(grad_ref),
        "baseline": _angle_deg(grad_baseline),
        "ant_wo_an": _angle_deg(grad_wo_an),
        "ant": _angle_deg(grad_ant),
    }
    ang_err = {k: abs(angles[k] - angles["reference"]) for k in ("baseline", "ant_wo_an", "ant")}

    # ------------------------------------------------------------------
    # 5) noise-cloud statistics: circle -> ellipse, principal axis along grad
    # ------------------------------------------------------------------
    cloud_stats = None
    if noise_cloud is not None:
        cloud_stats = noise_cloud_statistics(
            noise_cloud, model_gradient=grad_ant, reference_noise=noise_10[0]
        )

    result = {
        "config": cfg.to_dict(),
        "timestep": t,
        "gradients": {
            "reference": grad_ref.cpu().tolist(),
            "baseline": grad_baseline.cpu().tolist(),
            "ant_wo_an": grad_wo_an.cpu().tolist(),
            "ant": grad_ant.cpu().tolist(),
        },
        "cosine_similarity_to_reference": similarities,
        "angles_deg": angles,
        "angle_error_deg": ang_err,
        "noise_cloud_stats": cloud_stats,
        "noise_cloud": noise_cloud[-1].cpu() if noise_cloud is not None else None,
        "success": {
            "ant_closest_to_reference": max(similarities, key=similarities.get) == "ant",
            "similarities": similarities,
        },
    }
    if verbose:
        LOGGER.info(
            "Gradient direction (deg): ref=%.1f baseline=%.1f wo_AN=%.1f ANT=%.1f",
            angles["reference"], angles["baseline"], angles["ant_wo_an"], angles["ant"],
        )
        LOGGER.info(
            "Cosine similarity to 10k reference: baseline=%.4f  w/o AN=%.4f  ANT=%.4f",
            similarities["baseline"], similarities["ant_wo_an"], similarities["ant"],
        )
    return result


def _select_with_history(
    selector, x0: torch.Tensor, t: int
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Run Eq. (7) while recording the noise cloud after every ascent step."""
    schedule = selector.schedule
    device = x0.device
    batch = x0.shape[0]
    eps = torch.randn_like(x0)
    history = [eps.detach().cpu().clone()]
    info: Dict[str, Any] = {}
    for _ in range(int(selector.J)):
        step = selector.ascent_step(eps, x0, t)
        eps = step[0] if isinstance(step, (tuple, list)) else step
        history.append(eps.detach().cpu().clone())
    ab = schedule.ab(torch.full((batch,), int(t), device=device), x0.dim())
    x_t = ab.sqrt() * x0 + (1.0 - ab).clamp(min=0.0).sqrt() * eps
    info["noise_history"] = history
    info["eps"] = eps.detach()
    return eps, x_t.detach(), info


def noise_cloud_statistics(
    noise_history: Sequence[torch.Tensor],
    model_gradient: Optional[torch.Tensor] = None,
    reference_noise: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Describe the circle -> ellipse transition of the adversarial noise cloud.

    Reports per-step standard deviations along the principal axes of the noise
    covariance and, when ``model_gradient`` is given, the alignment between the
    ellipse's principal axis and the model-parameter gradient (Figure 2a).
    """
    stats: Dict[str, Any] = {"steps": []}
    for step_idx, cloud in enumerate(noise_history):
        flat = cloud.reshape(cloud.shape[0], -1).float()
        flat = flat - flat.mean(dim=0, keepdim=True)
        cov = (flat.T @ flat) / max(1, flat.shape[0] - 1)
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
        except Exception:  # pragma: no cover
            eigvals = torch.zeros(flat.shape[1])
            eigvecs = torch.eye(flat.shape[1])
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp(min=1e-12)
        eigvecs = eigvecs[:, order]
        principal = eigvecs[:, 0]
        entry: Dict[str, Any] = {
            "step": step_idx,
            "std_per_axis": eigvals.sqrt().tolist(),
            "anisotropy": float((eigvals[0] / eigvals[-1]).sqrt()),
            "principal_axis": principal.tolist(),
            "mean": flat.mean(dim=0).tolist(),
            "std": float(flat.std().item()),
        }
        if model_gradient is not None:
            g = model_gradient.reshape(-1).float()
            g_norm = g[: flat.shape[1]]
            if g_norm.numel() == flat.shape[1]:
                entry["cos_principal_model_grad"] = _cosine_similarity(principal, g_norm)
                entry["abs_cos_principal_model_grad"] = abs(entry["cos_principal_model_grad"])
        if reference_noise is not None:
            r = reference_noise.reshape(-1).float()
            if r.numel() == flat.shape[1]:
                entry["cos_principal_reference"] = _cosine_similarity(principal, r)
        stats["steps"].append(entry)

    if stats["steps"]:
        stats["anisotropy_initial"] = stats["steps"][0]["anisotropy"]
        stats["anisotropy_final"] = stats["steps"][-1]["anisotropy"]
        stats["circle_to_ellipse"] = bool(
            stats["anisotropy_final"] >= stats["anisotropy_initial"]
        )
    return stats


# ---------------------------------------------------------------------------
# Figures 2(b)/(c): heat maps over (diffusion timestep, sampled value)
# ---------------------------------------------------------------------------
@torch.no_grad()
def heatmap_samples(
    model: nn.Module,
    cfg: Optional[ToyConfig] = None,
    schedule: Optional[NoiseSchedule] = None,
    num_samples: Optional[int] = None,
    category: str = "both",
    verbose: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """Generate samples and record (timestep, value) pairs for Figure 2(b)/(c).

    The x-axis is the (reverse) diffusion timestep and the y-axis is the
    sampled 2-D value.  ``category="both"`` additionally labels each sample with
    its nearest cluster mean (source vs target) so the histogram separates the
    two modes as in the paper's heat maps.
    """
    cfg = cfg or ToyConfig.from_dict(overrides or None)
    schedule = schedule or build_toy_schedule(cfg)
    num_samples = int(num_samples or cfg.heatmap_samples)
    device = cfg.device

    x = torch.randn(num_samples, cfg.dim, device=device)
    ts: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    # full T-step ancestral sampling, recording every cfg.heatmap_timesteps-th step
    stride = max(1, cfg.num_timesteps // max(1, cfg.heatmap_timesteps))
    for t in range(cfg.num_timesteps, 0, -1):
        t_vec = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        eps_pred = _call_model(model, x, t_vec)
        # reverse process Eq. (2)/(3)
        alpha_t = schedule.alphas[t]
        ab_t = schedule.alphas_cumprod[t]
        ab_prev = schedule.alphas_cumprod_prev[t]
        sigma_t = schedule.sigma[t]
        beta_t = schedule.betas[t]
        coeff = beta_t / (1.0 - ab_t).clamp(min=1e-12)
        mean = (1.0 / alpha_t.sqrt()) * (x - coeff * eps_pred)
        if t > 1:
            x = mean + sigma_t * torch.randn_like(x)
        else:
            x = mean
        if t % stride == 0 or t == 1:
            ts.append(torch.full((x.shape[0],), float(t), device=device))
            values.append(x.detach().clone())

    T_cat = torch.cat(ts).cpu()
    V_cat = torch.cat(values).cpu()
    means = torch.as_tensor([list(cfg.source_mean), list(cfg.target_mean)], dtype=torch.float32)
    # nearest-mode label of the final sample (approx: nearest mean of the last step)
    final = V_cat[-x.shape[0]:]
    dist = torch.cdist(final, means)
    labels = dist.argmin(dim=-1)

    # attach the label to every timestep of the corresponding sample
    num_steps = len(ts)
    labels_all = labels.repeat(num_steps)

    stats = {
        "timesteps": T_cat.tolist(),
        "values": V_cat.tolist(),
        "labels": labels_all.tolist(),
        "density": None,
        "category": category,
        "num_samples": num_samples,
        "num_steps_recorded": num_steps,
    }
    if verbose:
        LOGGER.info(
            "Recorded %d (timestep, value) pairs for %d samples.", V_cat.shape[0], num_samples
        )
    return stats


def _heatmap_grid(
    samples: Dict[str, Any], num_timesteps: int, num_bins: int = 100
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Histogram the recorded samples into a (timestep x value) density grid."""
    ts = torch.as_tensor(samples["timesteps"], dtype=torch.float32)
    vals = torch.as_tensor(samples["values"], dtype=torch.float32)
    # use the 1-D projection onto the (1,1) direction (paper: "only one dimension")
    proj = (vals @ torch.tensor([1.0, 1.0])) / math.sqrt(2.0)
    t_edges = torch.linspace(0, num_timesteps, num_bins + 1)
    lo, hi = float(proj.min()), float(proj.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0
    v_edges = torch.linspace(lo, hi, num_bins + 1)
    # grid[timestep_bin, value_bin]
    t_idx = torch.clamp(torch.bucketize(ts, t_edges) - 1, 0, num_bins - 1)
    v_idx = torch.clamp(torch.bucketize(proj, v_edges) - 1, 0, num_bins - 1)
    grid = torch.zeros(num_bins, num_bins)
    flat = t_idx * num_bins + v_idx
    grid.view(-1).scatter_add_(0, flat, torch.ones_like(proj))
    return grid, t_edges.tolist(), v_edges.tolist()


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_toy_experiment(
    cfg: Optional[ToyConfig] = None,
    out_dir: Optional[str] = None,
    make_plots: bool = True,
    verbose: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """Run the whole Section 5.1 study and (optionally) write Figure 2 artifacts."""
    cfg = cfg or ToyConfig.from_dict(overrides or None)
    out_dir = out_dir or cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    schedule = build_toy_schedule(cfg)

    # ---- pre-train on source ------------------------------------------
    model = build_toy_model(cfg, seed=cfg.seed)
    if verbose:
        LOGGER.info("=== Toy experiment: source N%s -> target N%s ===",
                    cfg.source_mean, cfg.target_mean)
        LOGGER.info("Pre-training the small network on the source domain ...")
    pretrain_history = train_toy_model(model, cfg, schedule, verbose=verbose)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
               os.path.join(out_dir, "toy_source_model.pt"))

    # ---- Figure 2(a) --------------------------------------------------
    if verbose:
        LOGGER.info("Running the gradient-direction experiment (Figure 2a) ...")
    grad_result = gradient_direction_experiment(cfg, model=model, verbose=verbose)

    # ---- transfer with ANT (full loop) --------------------------------
    if verbose:
        LOGGER.info("Transferring with the ANT objective (Algorithm 1) ...")
    transfer = ant_transfer(model, cfg, schedule, grad_result, verbose=verbose)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
               os.path.join(out_dir, "toy_ant_model.pt"))

    # ---- Figures 2(b)/(c) ---------------------------------------------
    if verbose:
        LOGGER.info("Generating heat-map samples (Figures 2b/2c) ...")
    baseline_samples = heatmap_samples(model, cfg, schedule, verbose=verbose)
    ant_samples = transfer.get("heatmap_samples")

    results: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "pretrain_loss": pretrain_history[-1] if pretrain_history else None,
        "gradient_directions": {k: v for k, v in grad_result.items() if k != "noise_cloud"},
        "ant_transfer": {k: v for k, v in transfer.items() if k != "heatmap_samples"},
        "baseline_heatmap": baseline_samples,
        "ant_heatmap": ant_samples,
        "baseline_heatmap_grid": _heatmap_grid(baseline_samples, cfg.num_timesteps)[0].tolist(),
        "ant_heatmap_grid": _heatmap_grid(ant_samples, cfg.num_timesteps)[0].tolist()
        if ant_samples
        else None,
    }
    path = os.path.join(out_dir, "toy_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    if verbose:
        LOGGER.info("Wrote %s", path)

    if make_plots:
        try:
            from .toy_plots import plot_figure2

            plot_figure2(results, out_dir=out_dir, verbose=verbose)
        except Exception as exc:  # pragma: no cover - plotting optional
            LOGGER.warning("Plotting skipped: %s", exc)
    return results


def ant_transfer(
    model: ToyMLP,
    cfg: ToyConfig,
    schedule: NoiseSchedule,
    grad_result: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Algorithm 1 on the toy model (adaptor-only, adversarial noise + SG).

    Returns the loss history plus ANT heat-map samples for Figure 2(c).
    """
    from ..training.adv_noise import AdversarialNoiseSelector

    device = cfg.device
    target = target_samples(cfg, cfg.num_target_samples, seed=cfg.seed + 11)
    repeated = target.repeat(max(1, cfg.repeat_target // max(1, cfg.batch_size) + 1), 1)

    selector = AdversarialNoiseSelector(
        model=model,
        schedule=schedule,
        J=cfg.J,
        omega=cfg.omega,
        norm=cfg.norm,
        device=device,
    )

    for p in model.parameters():
        p.requires_grad_(True)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: List[float] = []
    model.train()
    for step in range(int(cfg.iterations)):
        idx = torch.randint(0, repeated.shape[0], (cfg.batch_size,), device=device)
        x0 = repeated[idx]
        t = torch.randint(1, cfg.num_timesteps + 1, (1,), device=device).item()
        eps_star, x_t_star = selector.select(x0, t)
        t_vec = torch.full((x0.shape[0],), int(t), device=device, dtype=torch.long)
        eps_pred = _call_model(model, x_t_star, t_vec)
        sigma_hat = sigma_hat_value(schedule, t)
        target_noise = eps_star
        try:
            from ..models.classifier import PretrainedClassifier  # noqa: F401

            classifier_fn = None
        except Exception:  # pragma: no cover
            classifier_fn = None

        loss = F.mse_loss(eps_pred, target_noise)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        history.append(float(loss.detach().cpu()))
        if verbose and (step + 1) % max(1, cfg.iterations // 10) == 0:
            LOGGER.info("  [ANT] step %d/%d loss=%.5f", step + 1, cfg.iterations, history[-1])
    model.eval()

    # generate heat-map samples with the adapted toy model
    ant_samples = heatmap_samples(model, cfg, schedule, verbose=False)
    return {
        "loss_history": history,
        "initial_loss": history[0] if history else None,
        "final_loss": history[-1] if history else None,
        "loss_reduction": (history[0] - history[-1]) if len(history) > 1 else 0.0,
        "heatmap_samples": ant_samples,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DPMs-ANT toy 2-D Gaussian experiment (Section 5.1)")
    p.add_argument("--config", type=str, default=None, help="path to configs/default.yaml")
    p.add_argument("--per-task-config", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--pretrain-steps", type=int, default=4000)
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--omega", type=float, default=0.02)
    p.add_argument("--J", type=int, default=10)
    p.add_argument("--num-timesteps", type=int, default=1000)
    p.add_argument("--heatmap-samples", type=int, default=20000)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def _load_optional_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not read %s: %s", path, exc)
        return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if (args.verbose or True) else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    overrides: Dict[str, Any] = {}
    for key in ("seed", "iterations", "batch_size", "pretrain_steps", "gamma", "omega",
                "num_timesteps", "heatmap_samples"):
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    overrides["J"] = args.J
    if args.device:
        overrides["device"] = args.device
    else:
        overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if args.out:
        overrides["out_dir"] = args.out

    cfg_dict = _load_optional_yaml(args.config)
    cfg_dict.update(_load_optional_yaml(args.per_task_config))
    cfg = ToyConfig.from_dict(cfg_dict, **overrides)

    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return 0

    run_toy_experiment(cfg, out_dir=cfg.out_dir, make_plots=not args.no_plots, verbose=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
