"""Configuration loading and object construction for coupled stochastic interpolants.

This module is pure *engineering glue*: the paper does not specify a config format
(see the reproduction plan item 15, "not specified in the paper").  It provides:

* a nested :class:`Config` dataclass tree (``data`` / ``coupling`` / ``model`` /
  ``loss`` / ``optim`` / ``train`` / ``sampler`` / ``eval``) with paper defaults,
* YAML loading with recursive merging and ``key=value`` CLI-style overrides,
* builders that instantiate the objects implemented in the rest of the package
  (coupling, interpolant, U-Net velocity/score model, loss, optimizer, scheduler,
  sampler, dataloader).

The default values reproduce the hyperparameters the paper *does* specify:
Algorithm 1 / Appendix B / Addendum -> batch size 32, 200000 gradient steps,
Adam lr 2e-4, StepLR gamma 0.99 every 1000 steps, no weight decay, gradient-norm
clip 10000, U-Net dim_mults (1,1,2,3,4), channels 256, ResNet-block groups 8,
learned-sinusoidal cond with dim 32, attention dim-head 64 / heads 4,
random Fourier features False.  Section 4.1 -> in-painting coupling with 64 tiles
at missing probability 0.3 and the ``alpha_t=t, beta_t=1-t, gamma_t=0`` preset.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "DataConfig",
    "CouplingConfig",
    "ModelConfig",
    "LossConfig",
    "OptimConfig",
    "TrainConfig",
    "SamplerConfig",
    "EvalConfig",
    "Config",
    "DEFAULTS",
    "load_config",
    "config_from_dict",
    "save_config",
    "merge_dicts",
    "apply_overrides",
    "parse_override",
    "build_coupling",
    "build_interpolant",
    "build_model",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "build_sampler",
    "build_dataloader",
    "build_all",
    "TASK_ALIASES",
    "main",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Task naming
# --------------------------------------------------------------------------------------

#: Accepted spellings for the two paper tasks (Section 4.1 in-painting, Section 4.2 SR).
TASK_ALIASES: Dict[str, str] = {
    "inpainting": "inpainting",
    "in-painting": "inpainting",
    "in_painting": "inpaint",
    "inpaint": "inpainting",
    "superres": "superres",
    "super-resolution": "superres",
    "super_resolution": "superres",
    "superresolution": "superres",
    "sr": "superres",
}


def normalize_task(task: str) -> str:
    """Map a free-form task spelling onto ``"inpainting"`` or ``"superres"``."""
    key = str(task).strip().lower()
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    if "paint" in key:
        return "inpainting"
    if "res" in key or key == "sr":
        return "superres"
    raise ValueError(
        f"unknown task {task!r}; expected one of {sorted(set(TASK_ALIASES.values()))}"
    )


# --------------------------------------------------------------------------------------
# Default configuration (paper values where specified, sensible defaults otherwise)
# --------------------------------------------------------------------------------------

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "data": {
        "name": "imagenet",
        "dataset": "imagenet-1k",
        "split": "train",
        #: ImageNet-1k directory (parquet/HF cache) -- see si/data/imagenet.py
        "root": None,
        "resolution": 256,
        "crop_size": None,
        "batch_size": 32,
        "num_workers": 8,
        "num_classes": 1000,
        "low_res": None,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "cache_dir": None,
        "trust_remote_code": True,
        "max_samples": None,
    },
    "coupling": {
        # "inpainting" | "superres" (see si.couplings.get_coupling)
        "name": "inpainting",
        # noise scale of x0 = m(x1) + sigma * zeta (paper leaves it open; tuned)
        "sigma": 1.0,
        # interpolant preset from si.interpolants.coefficients
        "coefficients": "inpainting",
        # in-painting (Section 4.1): 64 tiles, each missing with probability 0.3
        "num_tiles": 64,
        "missing_prob": 0.3,
        "expand_channels": False,
        # super-resolution (Section 4.2)
        "low_res": None,
        "down_mode": "area",
        "up_mode": "bilinear",
        "antialias": True,
        "requires_conditioning": True,
    },
    "model": {
        "name": "unet",
        # ---- Appendix B hyperparameters ----
        "channels": 256,
        "dim_mults": (1, 1, 2, 3, 4),
        "resnet_block_groups": 8,
        "learned_sinusoidal_cond": True,
        "learned_sinusoidal_dim": 32,
        "attention_dim_head": 64,
        "attention_heads": 4,
        "random_fourier_features": False,
        # ---- conditioning ----
        "num_classes": 1000,
        "class_dropout_prob": 0.1,
        "project_conditioning": False,
        "attn_resolution": 16,
        "mid_attention": True,
        "in_channels": 3,
        # ---- in-painting structural velocity masking ----
        "mask_observed": False,
        "observed_value": 1.0,
        "out_activation": None,
    },
    "loss": {
        "name": "velocity",
        "reduction": "mean",
        "return_dict": False,
        "t_eps": 0.0,
    },
    "optim": {
        "optimizer": "adam",
        "lr": 2.0e-4,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "scheduler": "steplr",
        "step_size": 1000,
        "gamma": 0.99,
        "warmup_steps": 0,
        # whole-parameter-vector gradient-norm clip (PyTorch default norm type)
        "grad_clip": 10000.0,
        "grad_clip_norm_type": 2.0,
    },
    "train": {
        # Addendum: 200,000 gradient steps at batch size 32
        "steps": 200000,
        "batch_size": 32,
        "log_every": 100,
        "save_every": 10000,
        "eval_every": 0,
        "checkpoint_dir": "runs",
        "run_name": None,
        "seed": 0,
        "device": "auto",
        "amp": False,
        "ema": False,
        "ema_decay": 0.9999,
        "resume": None,
        "fabric": True,
        "t_eps": 0.0,
    },
    "sampler": {
        "name": "dopri5",
        "method": "dopri5",
        "steps": 50,
        "atol": 1e-5,
        "rtol": 1e-5,
        "epsilon": 0.0,
        "project_observed": True,
        "return_trajectory": False,
        #: number of ODE slices to record for Fig. 5 style probability-flow figures
        "trajectory_steps": 50,
    },
    "eval": {
        "num_samples": 50000,
        "batch_size": 32,
        "reference_stats": None,
        "features_dir": None,
        "save_dir": "samples",
        "compute_fid": True,
        "save_images": False,
        "num_figures": 8,
        "task": None,
        "paper_fid": None,
    },
}


def merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = copy.deepcopy(base)
    if not override:
        return out
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# --------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------


@dataclass
class DataConfig:
    name: str = "imagenet"
    dataset: str = "imagenet-1k"
    split: str = "train"
    root: Optional[str] = None
    resolution: int = 256
    crop_size: Optional[int] = None
    batch_size: int = 32
    num_workers: int = 8
    num_classes: int = 1000
    low_res: Optional[int] = None
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    max_samples: Optional[int] = None

    @property
    def crop(self) -> int:
        return int(self.crop_size or self.resolution)

    @property
    def in_channels(self) -> int:
        return 3


@dataclass
class CouplingConfig:
    name: str = "inpainting"
    sigma: float = 1.0
    coefficients: str = "inpainting"
    num_tiles: int = 64
    missing_prob: float = 0.3
    expand_channels: bool = False
    low_res: Optional[int] = None
    down_mode: str = "area"
    up_mode: str = "bilinear"
    antialias: bool = True
    requires_conditioning: bool = True

    @property
    def task(self) -> str:
        return normalize_task(self.name)

    def kwargs(self) -> Dict[str, Any]:
        """Return the keyword arguments consumed by :func:`si.couplings.get_coupling`."""
        if self.task == "inpainting":
            return dict(
                sigma=float(self.sigma),
                num_tiles=int(self.num_tiles),
                missing_prob=float(self.missing_prob),
                expand_channels=bool(self.expand_channels),
                coefficients=self.coefficients,
                requires_conditioning=bool(self.requires_conditioning),
            )
        kwargs: Dict[str, Any] = dict(
            sigma=float(self.sigma),
            down_mode=self.down_mode,
            up_mode=self.up_mode,
            antialias=bool(self.antialias),
            coefficients=self.coefficients,
            requires_conditioning=bool(self.requires_conditioning),
        )
        if self.low_res is not None:
            kwargs["low_res"] = int(self.low_res)
        return kwargs


@dataclass
class ModelConfig:
    name: str = "unet"
    channels: int = 256
    dim_mults: Sequence[int] = (1, 1, 2, 3, 4)
    resnet_block_groups: int = 8
    learned_sinusoidal_cond: bool = True
    learned_sinusoidal_dim: int = 32
    attention_dim_head: int = 64
    attention_heads: int = 4
    random_fourier_features: bool = False
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    project_conditioning: bool = False
    attn_resolution: int = 16
    mid_attention: bool = True
    in_channels: int = 3
    mask_observed: bool = False
    observed_value: float = 1.0
    out_activation: Optional[str] = None

    def kwargs(self, conditioning_channels: int = 0, resolution: int = 256) -> Dict[str, Any]:
        """Flat kwargs accepted by ``si.models.unet_from_config``."""
        return dict(
            in_channels=int(self.in_channels),
            channels=int(self.channels),
            dim_mults=tuple(int(m) for m in self.dim_mults),
            resnet_block_groups=int(self.resnet_block_groups),
            num_classes=int(self.num_classes) if self.num_classes else None,
            class_dropout_prob=float(self.class_dropout_prob),
            conditioning_channels=int(conditioning_channels),
            project_conditioning=bool(self.project_conditioning),
            learned_sinusoidal_cond=bool(self.learned_sinusoidal_cond),
            learned_sinusoidal_dim=int(self.learned_sinusoidal_dim),
            attention_dim_head=int(self.attention_dim_head),
            attention_heads=int(self.attention_heads),
            random_fourier_features=bool(self.random_fourier_features),
            attn_resolution=int(self.attn_resolution),
            mid_attention=bool(self.mid_attention),
            mask_observed=bool(self.mask_observed),
            observed_value=float(self.observed_value),
            image_size=int(resolution),
            out_activation=self.out_activation,
        )


@dataclass
class LossConfig:
    name: str = "velocity"
    reduction: str = "mean"
    return_dict: bool = False
    t_eps: float = 0.0

    def kwargs(self, coefficients: str = "linear", mask_fn: bool = False) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            coefficients=coefficients,
            reduction=self.reduction,
            return_dict=bool(self.return_dict),
            t_eps=float(self.t_eps),
        )
        if mask_fn:
            kwargs["mask_fn"] = True
        return kwargs


@dataclass
class OptimConfig:
    optimizer: str = "adam"
    lr: float = 2.0e-4
    betas: Sequence[float] = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.0
    scheduler: Optional[str] = "steplr"
    step_size: int = 1000
    gamma: float = 0.99
    warmup_steps: int = 0
    grad_clip: Optional[float] = 10000.0
    grad_clip_norm_type: float = 2.0


@dataclass
class TrainConfig:
    steps: int = 200000
    batch_size: int = 32
    log_every: int = 100
    save_every: int = 10000
    eval_every: int = 0
    checkpoint_dir: str = "runs"
    run_name: Optional[str] = None
    seed: int = 0
    device: str = "auto"
    amp: bool = False
    ema: bool = False
    ema_decay: float = 0.9999
    resume: Optional[str] = None
    fabric: bool = True
    t_eps: float = 0.0


@dataclass
class SamplerConfig:
    name: str = "dopri5"
    method: str = "dopri5"
    steps: int = 50
    atol: float = 1e-5
    rtol: float = 1e-5
    epsilon: float = 0.0
    project_observed: bool = True
    return_trajectory: bool = False
    trajectory_steps: int = 50

    @property
    def is_sde(self) -> bool:
        return bool(self.epsilon) and float(self.epsilon) > 0.0


@dataclass
class EvalConfig:
    num_samples: int = 50000
    batch_size: int = 32
    reference_stats: Optional[str] = None
    features_dir: Optional[str] = None
    save_dir: str = "samples"
    compute_fid: bool = True
    save_images: bool = False
    num_figures: int = 8
    task: Optional[str] = None
    paper_fid: Optional[float] = None


@dataclass
class Config:
    """Nested configuration for one task/scale run."""

    name: str = "inpainting_256"
    task: str = "inpainting"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------------- derived properties -------------------------------------------
    @property
    def resolution(self) -> int:
        return int(self.data.resolution)

    @property
    def low_res(self) -> Optional[int]:
        if self.data.low_res is not None:
            return int(self.data.low_res)
        if self.coupling.low_res is not None:
            return int(self.coupling.low_res)
        return None

    @property
    def batch_size(self) -> int:
        return int(self.train.batch_size or self.data.batch_size)

    @property
    def conditioning_channels(self) -> int:
        """Channels of the image-shaped conditioning appended to the model input.

        In-painting appends the missingness mask (1 channel); super-resolution
        appends the upsampled low-resolution image (``in_channels`` channels).
        """
        if not self.coupling.requires_conditioning:
            return 0
        if self.coupling.task == "inpainting":
            if self.coupling.expand_channels:
                return int(self.data.in_channels)
            return 1
        return int(self.data.in_channels)

    @property
    def model_in_channels(self) -> int:
        return int(self.data.in_channels) + self.conditioning_channels

    @property
    def coefficients_name(self) -> str:
        return self.coupling.coefficients

    @property
    def sigma(self) -> float:
        return float(self.coupling.sigma)

    # ---------------- serialization -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value):
                out[f.name] = {k: _to_serializable(v) for k, v in dataclasses.asdict(value).items()}
            else:
                out[f.name] = _to_serializable(value)
        return out

    def save(self, path: str) -> str:
        return save_config(self, path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return config_from_dict(data)

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        return load_config(path, overrides=overrides)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config(name={self.name!r}, task={self.task!r}, resolution={self.resolution})"


def _to_serializable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------------------
# Loading / construction
# --------------------------------------------------------------------------------------


def _coerce_section(section_cls: type, values: Optional[Dict[str, Any]]) -> Any:
    """Instantiate a section dataclass from a (possibly partial) dict of values."""
    if not values:
        return section_cls()
    valid = {f.name for f in fields(section_cls)}
    known: Dict[str, Any] = {}
    unknown: Dict[str, Any] = {}
    for key, value in values.items():
        if key in valid:
            known[key] = value
        else:
            unknown[key] = value
    obj = section_cls(**known)
    if unknown:
        # keep unknown keys around on the section object for forward-compatibility
        for key, value in unknown.items():
            setattr(obj, key, value)
    return obj


def config_from_dict(data: Optional[Dict[str, Any]] = None) -> Config:
    """Build a :class:`Config` from a (possibly partial) nested dict."""
    merged = merge_dicts(DEFAULTS, data or {})
    name = merged.pop("name", "run")
    task = merged.pop("task", None)
    seed = merged.pop("seed", merged.get("train", {}).get("seed", 0))
    extra = merged.pop("extra", {}) or {}

    # data.batch_size falls back to train.batch_size
    data_cfg = _coerce_section(DataConfig, merged.get("data"))
    train_cfg = _coerce_section(TrainConfig, {**merged.get("train", {})})
    if not data_cfg.batch_size or (
        "batch_size" not in (data or {}).get("data", {})
        and merged.get("train", {}).get("batch_size")
    ):
        data_cfg.batch_size = int(train_cfg.batch_size)

    coupling_cfg = _coerce_section(CouplingConfig, merged.get("coupling"))
    model_cfg = _coerce_section(ModelConfig, merged.get("model"))
    loss_cfg = _coerce_section(LossConfig, merged.get("loss"))
    optim_cfg = _coerce_section(OptimConfig, merged.get("optim"))
    sampler_cfg = _coerce_section(SamplerConfig,
                                  _sync_sampler(merged.get("sampler")))
    eval_cfg = _coerce_section(EvalConfig, merged.get("eval"))

    model_cfg.dim_mults = tuple(int(m) for m in model_cfg.dim_mults)
    model_cfg.betas = None  # unused guard (keeps dataclass clean)

    resolved_task = normalize_task(task or coupling_cfg.name)
    coupling_cfg.name = resolved_task
    # keep the coupling coefficients consistent with the task defaults
    if "coefficients" not in ((data or {}).get("coupling", {}) or {}):
        coupling_cfg.coefficients = _default_coefficients(resolved_task)
    if resolved_task == "inpainting":
        model_cfg.mask_observed = bool(model_cfg.mask_observed or _mask_default(data))
    else:
        model_cfg.mask_observed = False

    if not data_cfg.low_res and coupling_cfg.low_res:
        data_cfg.low_res = int(coupling_cfg.low_res)
    if resolved_task == "superres" and not coupling_cfg.low_res and data_cfg.low_res:
        coupling_cfg.low_res = int(data_cfg.low_res)

    cfg = Config(
        name=str(name),
        task=resolved_task,
        seed=int(seed),
        data=data_cfg,
        coupling=coupling_cfg,
        model=model_cfg,
        loss=loss_cfg,
        optim=optim_cfg,
        train=train_cfg,
        sampler=sampler_cfg,
        eval=eval_cfg,
        extra=dict(extra),
    )
    if sampler_cfg.is_sde:
        cfg.loss.name = "velocity"
    return cfg


def _sync_sampler(sampler: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Mirror ``name`` -> ``method`` so either key can be provided in YAML."""
    out = dict(sampler or {})
    if "name" in out and "method" not in out:
        out["method"] = out["name"]
    if "method" in out and "name" not in out:
        out["name"] = out["method"]
    return out


def _default_coefficients(task: str) -> str:
    return "inpainting" if task == "inpainting" else "gamma0"


def _mask_default(data: Optional[Dict[str, Any]]) -> bool:
    return True


def load_config(
    path: Optional[Union[str, Sequence[str]]] = None,
    overrides: Optional[Union[Dict[str, Any], Sequence[str]]] = None,
) -> Config:
    """Load a YAML config (or several, merged left-to-right) plus overrides.

    ``overrides`` may be a nested dict or a sequence of ``"a.b=value"`` strings as
    produced by the command line (see :func:`apply_overrides`).
    """
    data: Dict[str, Any] = {}
    if path is not None:
        paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
        for one in paths:
            data = merge_dicts(data, _read_yaml(str(one)))
    if overrides:
        if isinstance(overrides, dict):
            data = merge_dicts(data, overrides)
        else:
            data = apply_overrides(data, overrides)
    return config_from_dict(data)


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - pyyaml is in requirements
        raise ImportError("PyYAML is required to load config files") from exc
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {path} must contain a mapping at the top level")
    return loaded


def save_config(config: Union[Config, Dict[str, Any]], path: str) -> str:
    """Write a config to YAML (falling back to JSON if PyYAML is unavailable)."""
    data = config.to_dict() if isinstance(config, Config) else config
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(_to_serializable(data), handle, sort_keys=False)
    except ImportError:  # pragma: no cover
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_to_serializable(data), handle, indent=2)
    return path


# --------------------------------------------------------------------------------------
# CLI-style overrides
# --------------------------------------------------------------------------------------


def parse_override(spec: str) -> Tuple[List[str], Any]:
    """Parse ``"a.b=value"`` into ``(["a", "b"], value)`` with YAML-ish coercion."""
    if "=" not in spec:
        raise ValueError(f"override {spec!r} must be of the form key.path=value")
    key, _, raw = spec.partition("=")
    keys = [part for part in key.strip().split(".") if part]
    if not keys:
        raise ValueError(f"override {spec!r} has an empty key")
    return keys, _coerce_scalar(raw.strip())


def _coerce_scalar(raw: str) -> Any:
    if raw.lower() in {"none", "null"}:
        return None
    if raw.lower() in {"true", "yes"}:
        return True
    if raw.lower() in {"false", "no"}:
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            import yaml  # type: ignore

            return yaml.safe_load(raw)
        except Exception:  # pragma: no cover - best effort
            return raw
    return raw


def apply_overrides(
    data: Optional[Dict[str, Any]],
    overrides: Sequence[str],
) -> Dict[str, Any]:
    """Apply ``["train.steps=10", "coupling.sigma=0.05"]`` to a nested dict."""
    out = copy.deepcopy(data or {})
    for spec in overrides or []:
        if not spec:
            continue
        keys, value = parse_override(spec)
        cursor = out
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = value
    return out


# --------------------------------------------------------------------------------------
# Builders for the rest of the package
# --------------------------------------------------------------------------------------


def build_coupling(config: Union[Config, Dict[str, Any]]):
    """Instantiate the task coupling (``si.couplings.get_coupling``)."""
    cfg = _as_config(config)
    from ..couplings import get_coupling

    return get_coupling(cfg.coupling.name, **cfg.coupling.kwargs())


def build_interpolant(config: Union[Config, Dict[str, Any]]):
    """Instantiate the :class:`~si.interpolants.interpolant.Interpolant`."""
    cfg = _as_config(config)
    from ..interpolants.interpolant import Interpolant
    from ..interpolants.coefficients import get_coefficients

    return Interpolant(get_coefficients(cfg.coupling.coefficients))


def build_model(
    config: Union[Config, Dict[str, Any]],
    name: Optional[str] = None,
):
    """Instantiate the velocity (or score) U-Net with Appendix-B hyperparameters."""
    cfg = _as_config(config)
    model_name = str(name or cfg.model.name or "unet")
    kwargs = cfg.model.kwargs(
        conditioning_channels=cfg.conditioning_channels,
        resolution=cfg.resolution,
    )
    if "score" in model_name.lower() or model_name.lower() in {"g", "hat_g"}:
        from ..models.score_net import score_net_from_config

        kwargs.pop("mask_observed", None)
        kwargs.pop("observed_value", None)
        return score_net_from_config(kwargs)
    from ..models.unet import unet_from_config

    return unet_from_config(kwargs)


def build_loss(config: Union[Config, Dict[str, Any]]):
    """Instantiate the training objective (velocity loss by default)."""
    cfg = _as_config(config)
    from ..losses import get_loss

    mask_fn = cfg.coupling.task == "inpainting"
    kwargs: Dict[str, Any] = dict(
        coefficients=cfg.coupling.coefficients,
        reduction=cfg.loss.reduction,
        return_dict=bool(cfg.loss.return_dict),
        t_eps=float(cfg.loss.t_eps or cfg.train.t_eps),
    )
    if mask_fn:
        kwargs["mask_fn"] = True
    name = cfg.loss.name or "velocity"
    try:
        return get_loss(name, **kwargs)
    except (ImportError, ValueError):  # pragma: no cover - defensive
        from ..losses.velocity_loss import VelocityLoss

        return VelocityLoss(**kwargs)


def build_optimizer(config: Union[Config, Dict[str, Any]], parameters):
    """Build the Adam optimizer with the paper's hyperparameters."""
    cfg = _as_config(config)
    import torch

    params = list(parameters)
    name = str(cfg.optim.optimizer or "adam").lower()
    lr = float(cfg.optim.lr)
    weight_decay = float(cfg.optim.weight_decay)
    if name in {"adam", "adamw"}:
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(
            params,
            lr=lr,
            betas=tuple(float(b) for b in cfg.optim.betas),
            eps=float(cfg.optim.eps),
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optim.optimizer!r}")


def build_scheduler(config: Union[Config, Dict[str, Any]], optimizer):
    """Build the StepLR scheduler (gamma 0.99 every 1000 steps by default)."""
    cfg = _as_config(config)
    import torch

    name = (cfg.optim.scheduler or "").lower()
    if name in {"", "none", "null"}:
        return None
    if "step" in name:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.optim.step_size),
            gamma=float(cfg.optim.gamma),
        )
    if "cosine" in name:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.train.steps)
        )
    if "exp" in name:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.optim.gamma))
    if "constant" in name:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    raise ValueError(f"unknown scheduler {cfg.optim.scheduler!r}")


def build_sampler(config: Union[Config, Dict[str, Any]]):
    """Instantiate the probability-flow ODE (or SDE) sampler."""
    cfg = _as_config(config)
    if cfg.sampler.is_sde:
        from ..samplers.sde import SDESampler

        return SDESampler(
            method="euler_maruyama",
            steps=int(cfg.sampler.steps),
            epsilon=float(cfg.sampler.epsilon),
            interpolant=cfg.coupling.coefficients,
            coefficients=cfg.coupling.coefficients,
        )
    from ..samplers.ode import ODESampler

    return ODESampler(
        method=str(cfg.sampler.method or cfg.sampler.name or "dopri5"),
        steps=int(cfg.sampler.steps),
        atol=float(cfg.sampler.atol),
        rtol=float(cfg.sampler.rtol),
        project_observed=bool(cfg.sampler.project_observed),
        return_trajectory=bool(cfg.sampler.return_trajectory),
    )


def build_dataloader(
    config: Union[Config, Dict[str, Any]],
    split: Optional[str] = None,
    batch_size: Optional[int] = None,
    **kwargs: Any,
):
    """Build the ImageNet dataloader described in ``si/data/imagenet.py``."""
    cfg = _as_config(config)
    from ..data.imagenet import build_imagenet_dataloader

    return build_imagenet_dataloader(
        resolution=cfg.data.resolution,
        batch_size=int(batch_size or cfg.batch_size),
        split=split or cfg.data.split,
        low_res=cfg.low_res,
        num_workers=cfg.data.num_workers,
        root=cfg.data.root,
        cache_dir=cfg.data.cache_dir,
        shuffle=cfg.data.shuffle,
        drop_last=cfg.data.drop_last,
        max_samples=cfg.data.max_samples,
        **kwargs,
    )


def build_all(config: Union[Config, Dict[str, Any]], device: Any = None, **model_kwargs: Any):
    """Convenience bundle: ``(coupling, interpolant, model, loss, sampler)``."""
    cfg = _as_config(config)
    coupling = build_coupling(cfg)
    interpolant = build_interpolant(cfg)
    model = build_model(cfg, **model_kwargs)
    if device is not None:
        model = model.to(device)
    loss = build_loss(cfg)
    sampler = build_sampler(cfg)
    return coupling, interpolant, model, loss, sampler


def _as_config(config: Union[Config, Dict[str, Any], None]) -> Config:
    if config is None:
        return config_from_dict({})
    if isinstance(config, Config):
        return config
    return config_from_dict(dict(config))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Inspect / dump a run configuration")
    parser.add_argument("--config", "-c", default=None, help="path to a YAML config")
    parser.add_argument("--set", "-s", dest="overrides", action="append", default=[],
                        help="override, e.g. -s train.steps=100")
    parser.add_argument("--out", default=None, help="write the resolved config here")
    parser.add_argument("--json", action="store_true", help="print as JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config, overrides=args.overrides)
    payload = cfg.to_dict()
    if args.json:
        print(json.dumps(_to_serializable(payload), indent=2))
    else:
        try:
            import yaml  # type: ignore

            print(yaml.safe_dump(_to_serializable(payload), sort_keys=False))
        except ImportError:
            print(json.dumps(_to_serializable(payload), indent=2))
    if args.out:
        save_config(cfg, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
